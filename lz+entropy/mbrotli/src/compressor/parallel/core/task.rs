//! Single terminal completion, including unwind and abandonment paths.
use super::{
    artifact::{Artifact, Descriptor},
    *,
};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::SyncSender,
};

pub(in crate::compressor::parallel) struct Completion {
    pub(in crate::compressor::parallel) result: Result<Artifact, ParallelEncodeError>,
    pub(in crate::compressor::parallel) worker: Option<Worker>,
}
pub(in crate::compressor::parallel) type Slot = Arc<Mutex<Option<Completion>>>;
pub(in crate::compressor::parallel) struct Task<'a> {
    pub(in crate::compressor::parallel) id: TaskId,
    pub(in crate::compressor::parallel) range: Range<u64>,
    pub(in crate::compressor::parallel) input: Input<'a>,
    pub(in crate::compressor::parallel) plan: Arc<Plan>,
    pub(in crate::compressor::parallel) worker: Option<Worker>,
    pub(in crate::compressor::parallel) artifact: Option<Artifact>,
    pub(in crate::compressor::parallel) slot: Slot,
    pub(in crate::compressor::parallel) sender: Option<SyncSender<usize>>,
    pub(in crate::compressor::parallel) cancelled: Arc<AtomicBool>,
}
impl Task<'_> {
    pub(in crate::compressor::parallel) fn run(mut self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute()));
        let result = match result {
            Ok(result) => result,
            Err(_) => {
                self.worker = None;
                Err(ParallelEncodeError::TaskPanicked { task: self.id })
            }
        };
        if result.is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        let result = result.and_then(|()| {
            self.artifact
                .take()
                .ok_or(ParallelEncodeError::FragmentInvariant(
                    "missing task artifact",
                ))
        });
        self.publish(result);
    }
    fn execute(&mut self) -> Result<(), ParallelEncodeError> {
        let worker = self
            .worker
            .as_mut()
            .ok_or(ParallelEncodeError::FragmentInvariant("missing worker"))?;
        let artifact = self
            .artifact
            .as_mut()
            .ok_or(ParallelEncodeError::FragmentInvariant("missing artifact"))?;
        for segment in self.range.clone() {
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(ParallelEncodeError::Cancelled);
            }
            let range = self.plan.source_range(segment)?;
            let len = usize::try_from(range.end - range.start)
                .map_err(|_| ParallelEncodeError::SizeOverflow)?;
            let src = match &self.input {
                Input::Slice(bytes) => bytes
                    .get(range.start as usize..range.end as usize)
                    .ok_or(ParallelEncodeError::FragmentInvariant("slice range"))?,
                Input::Source(source) => {
                    if worker.input.len() < len {
                        worker.input.try_reserve_exact(len - worker.input.len())?;
                        worker.input.resize(len, 0);
                    }
                    source
                        .read_exact_at(range.start, &mut worker.input[..len])
                        .map_err(|source| ParallelEncodeError::SourceRead {
                            task: self.id,
                            segment: SegmentId(segment),
                            source,
                        })?;
                    &worker.input[..len]
                }
            };
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(ParallelEncodeError::Cancelled);
            }
            let offset = artifact.len;
            let write_result = if self.plan.serial {
                match worker.serial.compress(src) {
                    Ok(bytes) => artifact.append(&bytes),
                    Err(source) => {
                        return Err(ParallelEncodeError::Encode {
                            task: self.id,
                            segment: SegmentId(segment),
                            source,
                        });
                    }
                }
            } else {
                match worker.fragment.encode(src, segment == 0) {
                    Ok(fragment) => artifact.append(fragment.bytes),
                    Err(source) => {
                        return Err(ParallelEncodeError::Encode {
                            task: self.id,
                            segment: SegmentId(segment),
                            source: crate::compressor::EncodeError::from_core(source, 0),
                        });
                    }
                }
            };
            if let Err(source) = write_result {
                return Err(ParallelEncodeError::Staging {
                    task: self.id,
                    source,
                });
            }
            artifact.descriptors.push(Descriptor {
                segment,
                source: range,
                offset,
                len: artifact.len - offset,
            });
        }
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(ParallelEncodeError::Cancelled);
        }
        Ok(())
    }
    fn publish(&mut self, result: Result<Artifact, ParallelEncodeError>) {
        if let Some(sender) = self.sender.take() {
            // Only this task writes its slot, exactly once. The mutex protects
            // ownership transfer only; codec work runs without a held lock.
            *self
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Completion {
                result,
                worker: self.worker.take(),
            });
            // Capacity equals the task count and each task sends once, so this
            // never needs a coordinator to drain payload or make progress.
            let _ = sender.send(self.id.0 as usize);
        }
    }
}
impl Drop for Task<'_> {
    fn drop(&mut self) {
        if self.sender.is_some() {
            self.publish(Err(ParallelEncodeError::TaskAbandoned { task: self.id }));
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }
}
