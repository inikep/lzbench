//! Single terminal completion, including unwind and abandonment paths.
use super::{
    artifact::{Artifact, Descriptor},
    *,
};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
};

pub(in crate::compressor::parallel) struct Completion {
    pub(in crate::compressor::parallel) result: Result<Artifact, ParallelEncodeError>,
    pub(in crate::compressor::parallel) worker: Option<Worker>,
}
pub(in crate::compressor::parallel) type Slot = Arc<Mutex<Option<Completion>>>;
/// The segments one task encodes, and where their bytes come from.
pub(in crate::compressor::parallel) enum Job<'a> {
    /// Contiguous segments of a planned batch, read from its input.
    Planned {
        range: Range<u64>,
        input: Input<'a>,
        plan: Arc<Plan>,
    },
    /// One streamed segment already copied into the worker's input buffer,
    /// after the `history` bytes of input that precede it.
    Streamed {
        segment: u64,
        source: Range<u64>,
        first: bool,
        serial: bool,
        history: usize,
    },
}
impl Job<'_> {
    pub(in crate::compressor::parallel) fn segments(&self) -> Range<u64> {
        match self {
            Self::Planned { range, .. } => range.clone(),
            Self::Streamed { segment, .. } => *segment..segment + 1,
        }
    }
}
/// One segment's bytes and how to encode them.
struct Unit<'s> {
    segment: u64,
    source: Range<u64>,
    /// Input right before `src`, which copies may reach back into.
    history: &'s [u8],
    src: &'s [u8],
    first: bool,
    serial: bool,
}
pub(in crate::compressor::parallel) struct Task<'a> {
    pub(in crate::compressor::parallel) id: TaskId,
    pub(in crate::compressor::parallel) job: Job<'a>,
    pub(in crate::compressor::parallel) worker: Option<Worker>,
    pub(in crate::compressor::parallel) artifact: Option<Artifact>,
    pub(in crate::compressor::parallel) slot: Slot,
    pub(in crate::compressor::parallel) sender: Option<Sender<usize>>,
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
        match &self.job {
            Job::Planned { range, input, plan } => {
                for segment in range.clone() {
                    if self.cancelled.load(Ordering::Relaxed) {
                        return Err(ParallelEncodeError::Cancelled);
                    }
                    let source = plan.source_range(segment)?;
                    // The history is read together with the segment.
                    let from = plan.history_start(source.start);
                    let history = usize::try_from(source.start - from)
                        .map_err(|_| ParallelEncodeError::SizeOverflow)?;
                    let len = usize::try_from(source.end - from)
                        .map_err(|_| ParallelEncodeError::SizeOverflow)?;
                    let bytes = match input {
                        Input::Slice(bytes) => bytes
                            .get(from as usize..source.end as usize)
                            .ok_or(ParallelEncodeError::FragmentInvariant("slice range"))?,
                        Input::Source(shared) => {
                            if worker.input.len() < len {
                                worker.input.try_reserve_exact(len - worker.input.len())?;
                                worker.input.resize(len, 0);
                            }
                            shared
                                .read_exact_at(from, &mut worker.input[..len])
                                .map_err(|source| ParallelEncodeError::SourceRead {
                                    task: self.id,
                                    segment: SegmentId(segment),
                                    source,
                                })?;
                            &worker.input[..len]
                        }
                    };
                    let (history, src) = bytes.split_at(history);
                    let unit = Unit {
                        segment,
                        source,
                        history,
                        src,
                        first: segment == 0,
                        serial: plan.serial,
                    };
                    encode(
                        self.id,
                        &self.cancelled,
                        &mut worker.fragment,
                        &mut worker.serial,
                        artifact,
                        unit,
                    )?;
                }
            }
            Job::Streamed {
                segment,
                source,
                first,
                serial,
                history,
            } => {
                let len = usize::try_from(source.end - source.start)
                    .map_err(|_| ParallelEncodeError::SizeOverflow)?;
                let end = history
                    .checked_add(len)
                    .ok_or(ParallelEncodeError::SizeOverflow)?;
                let bytes = worker
                    .input
                    .get(..end)
                    .ok_or(ParallelEncodeError::FragmentInvariant("streamed segment"))?;
                let (history, src) = bytes.split_at(*history);
                if self.cancelled.load(Ordering::Relaxed) {
                    return Err(ParallelEncodeError::Cancelled);
                }
                let unit = Unit {
                    segment: *segment,
                    source: source.clone(),
                    history,
                    src,
                    first: *first,
                    serial: *serial,
                };
                encode(
                    self.id,
                    &self.cancelled,
                    &mut worker.fragment,
                    &mut worker.serial,
                    artifact,
                    unit,
                )?;
            }
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
            // The channel is unbounded and each task sends once, so this never
            // needs a coordinator to drain payload or make progress.
            let _ = sender.send(self.id.0 as usize);
        }
    }
}
/// Encodes one segment into the artifact and records its descriptor.
fn encode(
    task: TaskId,
    cancelled: &AtomicBool,
    fragment: &mut FragmentEncoder,
    serial: &mut SerialCompressor,
    artifact: &mut Artifact,
    unit: Unit<'_>,
) -> Result<(), ParallelEncodeError> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(ParallelEncodeError::Cancelled);
    }
    let segment = SegmentId(unit.segment);
    let offset = artifact.len;
    let write_result = if unit.serial {
        match serial.compress(unit.src) {
            Ok(bytes) => artifact.append(&bytes),
            Err(source) => {
                return Err(ParallelEncodeError::Encode {
                    task,
                    segment,
                    source,
                });
            }
        }
    } else {
        match fragment.encode(unit.history, unit.src, unit.first) {
            Ok(fragment) => artifact.append(fragment.bytes),
            Err(source) => {
                return Err(ParallelEncodeError::Encode {
                    task,
                    segment,
                    source: crate::compressor::EncodeError::from_core(source, 0),
                });
            }
        }
    };
    if let Err(source) = write_result {
        return Err(ParallelEncodeError::Staging { task, source });
    }
    artifact.descriptors.push(Descriptor {
        segment: unit.segment,
        source: unit.source,
        offset,
        len: artifact.len - offset,
    });
    Ok(())
}
impl Drop for Task<'_> {
    fn drop(&mut self) {
        if self.sender.is_some() {
            self.publish(Err(ParallelEncodeError::TaskAbandoned { task: self.id }));
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }
}
