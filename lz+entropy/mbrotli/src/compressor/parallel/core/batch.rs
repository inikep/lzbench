//! Nonblocking drop, deterministic failure selection, and transactional delivery.
use super::{
    artifact::{self, Artifact},
    task::{Completion, Slot},
    *,
};
use std::{
    io,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, TryRecvError},
    },
    time::Instant,
};

pub(in crate::compressor::parallel) struct Batch<'e, 'i> {
    parent: &'e mut Compressor,
    pub(in crate::compressor::parallel) plan: Arc<Plan>,
    input: Input<'i>,
    identity: Option<SourceIdentity>,
    tasks: Option<Vec<Task<'i>>>,
    receiver: Receiver<usize>,
    slots: Vec<Slot>,
    received: Vec<bool>,
    artifacts: Vec<Option<Artifact>>,
    failures: Vec<(usize, ParallelEncodeError)>,
    failure: Option<Arc<ParallelEncodeError>>,
    completed: usize,
    cancelled: Arc<AtomicBool>,
    stats: ParallelStats,
    empty: Vec<u8>,
}
impl<'e, 'i> Batch<'e, 'i> {
    pub(in crate::compressor::parallel) fn prepare(
        parent: &'e mut Compressor,
        input: Input<'i>,
        config: BatchConfig,
    ) -> Result<Self, ParallelEncodeError> {
        let len = input.len()?;
        let identity = input.identity();
        let plan = Arc::new(parent.plan(len, &config)?);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(plan.tasks.max(1));
        let mut tasks = Vec::new();
        tasks.try_reserve_exact(plan.tasks)?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(plan.tasks)?;
        let mut artifacts = Vec::new();
        artifacts.try_reserve_exact(plan.tasks)?;
        artifacts.resize_with(plan.tasks, || None);
        let mut received = Vec::new();
        received.try_reserve_exact(plan.tasks)?;
        received.resize(plan.tasks, false);
        let mut failures = Vec::new();
        failures.try_reserve_exact(plan.tasks)?;
        let reused = parent.workers.len().min(plan.tasks);
        let mut segment = 0u64;
        // Complete staging creation before lending out any existing worker. A
        // creation failure cannot strand task resources outside the coordinator.
        for id in 0..plan.tasks {
            let count = plan.estimate.segment_count / plan.tasks as u64
                + u64::from((id as u64) < plan.estimate.segment_count % plan.tasks as u64);
            let end = segment
                .checked_add(count)
                .ok_or(ParallelEncodeError::SizeOverflow)?;
            let start_byte = plan.source_range(segment)?.start;
            let end_byte = plan.source_range(end - 1)?.end;
            let bound = (end_byte - start_byte)
                .checked_mul(2)
                .and_then(|n| count.checked_mul(1024).and_then(|c| n.checked_add(c)))
                .ok_or(ParallelEncodeError::SizeOverflow)?;
            let artifact = Artifact::new(
                &config.staging,
                usize::try_from(count).map_err(|_| ParallelEncodeError::SizeOverflow)?,
                bound,
            )
            .map_err(|source| ParallelEncodeError::Staging {
                task: TaskId(id as u32),
                source,
            })?;
            let slot = Arc::new(Mutex::new(None));
            slots.push(Arc::clone(&slot));
            tasks.push(Task {
                id: TaskId(id as u32),
                range: segment..end,
                input: input.clone(),
                plan: Arc::clone(&plan),
                worker: None,
                artifact: Some(artifact),
                slot,
                sender: Some(sender.clone()),
                cancelled: Arc::clone(&cancelled),
            });
            segment = end;
        }
        for task in &mut tasks {
            task.worker = Some(match parent.workers.pop() {
                Some(w) => w,
                None => Worker::new(parent.encoder, parent.backend)?,
            });
        }
        let empty = if len == 0 {
            let mut serial = SerialCompressor::builder(parent.encoder)
                .with_backend(parent.backend)
                .build()
                .map_err(ParallelConfigError::from)?;
            match serial.compress(&[]) {
                Ok(bytes) => bytes,
                Err(source) => {
                    return Err(ParallelEncodeError::Encode {
                        task: TaskId(0),
                        segment: SegmentId(0),
                        source,
                    });
                }
            }
        } else {
            Vec::new()
        };
        let stats = ParallelStats {
            input_bytes: len,
            output_bytes: 0,
            segment_size: plan.segment_size,
            segment_count: plan.estimate.segment_count,
            requested_tasks: config.task_count.get(),
            effective_tasks: plan.tasks,
            serial_fallback: plan.serial,
            staging_kind: match config.staging {
                Staging::Memory(_) => StagingKind::Memory,
                Staging::Directory(_) => StagingKind::Directory,
            },
            maximum_staged_bytes: plan.estimate.maximum_staged_bytes,
            context_prefix_bytes: if plan.serial {
                0
            } else {
                (len / plan.segment_size as u64) * 2 + (len % plan.segment_size as u64).min(2)
            },
            workers_reused: reused,
            workers_created: plan.tasks - reused,
            retained_worker_bytes: 0,
        };
        Ok(Self {
            parent,
            plan,
            input,
            identity,
            tasks: Some(tasks),
            receiver,
            slots,
            received,
            artifacts,
            failures,
            failure: None,
            completed: 0,
            cancelled,
            stats,
            empty,
        })
    }
    pub(in crate::compressor::parallel) fn take_tasks(
        &mut self,
    ) -> Result<Vec<Task<'i>>, ParallelEncodeError> {
        self.tasks
            .take()
            .ok_or(ParallelEncodeError::TasksAlreadyTaken)
    }
    pub(in crate::compressor::parallel) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    fn receive(&mut self, id: usize) -> Result<(), ParallelEncodeError> {
        let seen = self
            .received
            .get_mut(id)
            .ok_or(ParallelEncodeError::FragmentInvariant("unknown completion"))?;
        if *seen {
            return Err(ParallelEncodeError::FragmentInvariant(
                "duplicate completion",
            ));
        }
        let Completion { result, worker } = self.slots[id]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .ok_or(ParallelEncodeError::FragmentInvariant(
                "missing completion slot",
            ))?;
        *seen = true;
        self.completed += 1;
        if let Some(worker) = worker {
            self.parent.workers.push(worker);
        }
        match result {
            Ok(artifact) => self.artifacts[id] = Some(artifact),
            Err(error) => {
                self.failures.push((id, error));
                self.cancel();
            }
        }
        Ok(())
    }
    fn ready(&mut self) -> Result<BatchPoll, ParallelEncodeError> {
        if self.completed < self.plan.tasks {
            return Ok(BatchPoll::Pending {
                completed: self.completed,
                total: self.plan.tasks,
            });
        }
        if self.failure.is_none() && !self.failures.is_empty() {
            // Cancellation fallout never hides the originating task error.
            self.failures
                .sort_by_key(|(id, e)| (matches!(e, ParallelEncodeError::Cancelled), *id));
            self.failure = Some(Arc::new(self.failures.remove(0).1));
        }
        if let Some(error) = &self.failure {
            return Err(ParallelEncodeError::BatchFailed(Arc::clone(error)));
        }
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(ParallelEncodeError::Cancelled);
        }
        Ok(BatchPoll::Ready)
    }
    pub(in crate::compressor::parallel) fn poll(
        &mut self,
    ) -> Result<BatchPoll, ParallelEncodeError> {
        loop {
            match self.receiver.try_recv() {
                Ok(id) => self.receive(id)?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if self.completed == self.plan.tasks => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(ParallelEncodeError::FragmentInvariant(
                        "completion channel closed early",
                    ));
                }
            }
        }
        self.ready()
    }
    pub(in crate::compressor::parallel) fn wait(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<WaitStatus, ParallelEncodeError> {
        if self.tasks.is_some() && self.plan.tasks > 0 {
            return Err(ParallelEncodeError::NotReady);
        }
        let start = Instant::now();
        while self.poll()? != BatchPoll::Ready {
            let id = if let Some(timeout) = timeout {
                let Some(remaining) = timeout.checked_sub(start.elapsed()) else {
                    return Ok(WaitStatus::TimedOut);
                };
                match self.receiver.recv_timeout(remaining) {
                    Ok(id) => id,
                    Err(RecvTimeoutError::Timeout) => return Ok(WaitStatus::TimedOut),
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err(ParallelEncodeError::FragmentInvariant(
                            "completion channel closed early",
                        ));
                    }
                }
            } else {
                match self.receiver.recv() {
                    Ok(id) => id,
                    Err(_) => {
                        return Err(ParallelEncodeError::FragmentInvariant(
                            "completion channel closed early",
                        ));
                    }
                }
            };
            self.receive(id)?;
        }
        Ok(WaitStatus::Ready)
    }
    fn validate(&mut self) -> Result<u64, ParallelEncodeError> {
        self.wait(None)?;
        if let Input::Source(source) = &self.input {
            match self.parent.parallel.source_consistency {
                SourceConsistency::AssumeImmutable => (),
                policy => {
                    if source.len().map_err(ParallelEncodeError::SourceMetadata)?
                        != self.plan.estimate.input_bytes
                        || (policy == SourceConsistency::VerifyLengthAndIdentity
                            && source.identity() != self.identity)
                    {
                        return Err(ParallelEncodeError::SourceChanged);
                    }
                }
            }
        }
        if self.plan.tasks == 0 {
            return Ok(self.empty.len() as u64);
        }
        let mut segment = 0;
        let mut total = 0u64;
        for (id, artifact) in self.artifacts.iter().enumerate() {
            let artifact = artifact
                .as_ref()
                .ok_or(ParallelEncodeError::FragmentInvariant(
                    "missing success artifact",
                ))?;
            let valid_len = match artifact.validate_len() {
                Ok(valid) => valid,
                Err(source) => {
                    return Err(ParallelEncodeError::Staging {
                        task: TaskId(id as u32),
                        source,
                    });
                }
            };
            if !valid_len {
                return Err(ParallelEncodeError::FragmentInvariant(
                    "artifact length changed",
                ));
            }
            let mut offset = 0;
            for descriptor in &artifact.descriptors {
                if descriptor.segment != segment
                    || descriptor.source != self.plan.source_range(segment)?
                    || descriptor.offset != offset
                    || descriptor.len == 0
                {
                    return Err(ParallelEncodeError::FragmentInvariant(
                        "descriptor differs from plan",
                    ));
                }
                offset = offset
                    .checked_add(descriptor.len)
                    .ok_or(ParallelEncodeError::SizeOverflow)?;
                segment += 1;
            }
            if offset != artifact.len {
                return Err(ParallelEncodeError::FragmentInvariant("descriptor bounds"));
            }
            total = total
                .checked_add(artifact.len)
                .ok_or(ParallelEncodeError::SizeOverflow)?;
        }
        if segment != self.plan.estimate.segment_count {
            return Err(ParallelEncodeError::FragmentInvariant("segment count"));
        }
        total
            .checked_add(u64::from(!self.plan.serial))
            .ok_or(ParallelEncodeError::SizeOverflow)
    }
    fn assemble<W: Write>(&mut self, writer: &mut W, written: &mut u64) -> io::Result<()> {
        let mut scratch = [0u8; 128 << 10];
        if self.plan.tasks == 0 {
            return artifact::write_counted(writer, &self.empty, written);
        }
        for artifact in &mut self.artifacts {
            let artifact = artifact
                .as_mut()
                .ok_or_else(|| io::Error::other("missing validated artifact"))?;
            artifact.copy(writer, written, &mut scratch)?;
        }
        if !self.plan.serial {
            // Artifacts can only be appended from sealed AlignedFragment values.
            // Hence ISLAST=1, ISLASTEMPTY=1 is at bit offset zero here.
            artifact::write_counted(writer, &[3], written)?;
        }
        Ok(())
    }
    fn stats(&mut self, written: u64) -> ParallelStats {
        self.parent.trim(ParallelRetentionPolicy::CurrentPlan);
        self.stats.output_bytes = written;
        self.stats.retained_worker_bytes = self.parent.retained_bytes();
        self.stats.clone()
    }
    pub(in crate::compressor::parallel) fn finish_into(
        &mut self,
        dst: &mut Vec<u8>,
    ) -> Result<ParallelOutput, ParallelEncodeError> {
        let len =
            usize::try_from(self.validate()?).map_err(|_| ParallelEncodeError::SizeOverflow)?;
        let start = dst.len();
        let end = start
            .checked_add(len)
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        dst.try_reserve_exact(len)?;
        let mut written = 0;
        if let Err(source) = self.assemble(dst, &mut written) {
            dst.truncate(start);
            return Err(ParallelEncodeError::AssemblyIo {
                bytes_written: written,
                source,
            });
        }
        if written != len as u64 {
            dst.truncate(start);
            return Err(ParallelEncodeError::FragmentInvariant("assembled length"));
        }
        Ok(ParallelOutput {
            range: start..end,
            stats: self.stats(written),
        })
    }
    pub(in crate::compressor::parallel) fn finish_to_writer<W: Write>(
        &mut self,
        mut writer: W,
    ) -> Result<(W, ParallelStats), ParallelFinishError<W>> {
        let result = self.validate();
        let len = match result {
            Ok(len) => len,
            Err(error) => {
                return Err(ParallelFinishError {
                    writer,
                    error,
                    bytes_written: 0,
                });
            }
        };
        let mut written = 0;
        if let Err(source) = self.assemble(&mut writer, &mut written) {
            return Err(ParallelFinishError {
                writer,
                error: ParallelEncodeError::AssemblyIo {
                    bytes_written: written,
                    source,
                },
                bytes_written: written,
            });
        }
        if written != len {
            return Err(ParallelFinishError {
                writer,
                error: ParallelEncodeError::FragmentInvariant("assembled length"),
                bytes_written: written,
            });
        }
        Ok((writer, self.stats(written)))
    }
}
impl Drop for Batch<'_, '_> {
    fn drop(&mut self) {
        self.cancel();
        // Drop untaken tasks first: their guards publish completion without
        // blocking. Drain only records already available; never await late work.
        self.tasks.take();
        while let Ok(id) = self.receiver.try_recv() {
            let _ = self.receive(id);
        }
        self.parent.trim(ParallelRetentionPolicy::CurrentPlan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assembly_rejects_missing_duplicate_and_corrupted_artifacts() {
        let mut c = Compressor::new(
            EncoderConfig::default(),
            ParallelConfig::default().with_minimum_parallel_size(0),
            Backend::SCALAR,
        )
        .unwrap();
        let mut b = Batch::prepare(
            &mut c,
            Input::Slice(b"abc"),
            BatchConfig::memory(TaskCount::ONE, 4096),
        )
        .unwrap();
        for task in b.take_tasks().unwrap() {
            task.run();
        }
        b.wait(None).unwrap();
        assert!(b.receive(0).is_err());
        assert!(b.receive(99).is_err());
        b.artifacts[0].as_mut().unwrap().descriptors[0].segment = 1;
        assert!(b.validate().is_err());
        b.artifacts[0].as_mut().unwrap().descriptors[0].segment = 0;
        b.artifacts[0].as_mut().unwrap().len += 1;
        assert!(b.validate().is_err());
        b.artifacts[0] = None;
        assert!(b.assemble(&mut Vec::new(), &mut 0).is_err());
        assert!(b.validate().is_err());
    }
}
