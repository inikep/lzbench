//! Incremental input, a bounded window of queued segments, and ordered output.
//!
//! The stream copies caller input into one filling buffer. A full segment moves
//! into a worker, whose task goes to a shared queue that caller-owned executors
//! drain. Completions come back through the same single-assignment slots and
//! ID channel as batch tasks, and finished segments are delivered strictly in
//! input order. Segment boundaries, the first-part flag and the serial fallback
//! rule are those of a batch, so a stream without flushes produces the bytes
//! `prepare_slice` produces for the same input.
use super::{
    artifact::Artifact,
    task::{Completion, Job, Slot},
    *,
};
use crate::{compressor::Operation, shared::uninit::copy_to_uninit};
use std::{
    collections::VecDeque,
    mem::{self, MaybeUninit},
    sync::{
        Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
};

/// Tasks waiting for an executor, shared by every queue handle and the stream.
struct Queue {
    state: Mutex<QueueState>,
    ready: Condvar,
}
struct QueueState {
    tasks: VecDeque<Task<'static>>,
    /// Set when the stream ends or the last handle is dropped; never cleared.
    closed: bool,
    /// Live [`QueueHandle`] values.
    handles: usize,
}
impl Queue {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        // Only queue bookkeeping runs under this lock, never a task.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
    /// Queues `task`, or drops it when no executor can ever take it.
    ///
    /// A dropped task publishes its abandonment, which returns its worker.
    fn push(&self, task: Task<'static>) -> bool {
        let mut state = self.lock();
        if state.closed {
            drop(state);
            drop(task);
            return false;
        }
        state.tasks.push_back(task);
        drop(state);
        self.ready.notify_one();
        true
    }
    /// Closes the queue and returns the tasks no executor took.
    ///
    /// The caller drops them outside the lock: each publishes its abandonment.
    fn close(&self) -> VecDeque<Task<'static>> {
        let mut state = self.lock();
        state.closed = true;
        let tasks = mem::take(&mut state.tasks);
        drop(state);
        self.ready.notify_all();
        tasks
    }
}

/// A receiving handle; the stream is closed for executors when the last drops.
pub(in crate::compressor::parallel) struct QueueHandle(Arc<Queue>);
impl QueueHandle {
    /// Blocks until a task is queued, or returns `None` once the queue is closed.
    pub(in crate::compressor::parallel) fn recv(&self) -> Option<Task<'static>> {
        let mut state = self.0.lock();
        loop {
            if let Some(task) = state.tasks.pop_front() {
                return Some(task);
            }
            if state.closed {
                return None;
            }
            state = self
                .0
                .ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
    /// Takes a queued task without blocking.
    pub(in crate::compressor::parallel) fn try_recv(&self) -> Option<Task<'static>> {
        self.0.lock().tasks.pop_front()
    }
}
impl Clone for QueueHandle {
    fn clone(&self) -> Self {
        self.0.lock().handles += 1;
        Self(Arc::clone(&self.0))
    }
}
impl Drop for QueueHandle {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.handles -= 1;
        if state.handles != 0 {
            return;
        }
        // No executor can take what is queued now: abandon it, which the
        // stream observes as completions instead of waiting forever.
        state.closed = true;
        let tasks = mem::take(&mut state.tasks);
        drop(state);
        drop(tasks);
    }
}

/// A caller destination the stream copies delivered bytes into.
///
/// Delivery only ever writes, so an uninitialized slice serves as well as an
/// initialized one, and each gets its own instantiation of `process`.
pub(in crate::compressor::parallel) trait Sink {
    fn len(&self) -> usize;
    /// Writes `bytes` at `at`; `at + bytes.len()` never exceeds [`Self::len`].
    fn put(&mut self, at: usize, bytes: &[u8]);
}
impl Sink for [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }
    fn put(&mut self, at: usize, bytes: &[u8]) {
        self[at..at + bytes.len()].copy_from_slice(bytes);
    }
}
impl Sink for [MaybeUninit<u8>] {
    fn len(&self) -> usize {
        <[MaybeUninit<u8>]>::len(self)
    }
    fn put(&mut self, at: usize, bytes: &[u8]) {
        copy_to_uninit(&mut self[at..at + bytes.len()], bytes);
    }
}

enum Phase {
    /// Accepting input.
    Open,
    /// The last part is queued; delivering what remains.
    Finishing,
    /// Every byte has been delivered.
    Finished,
    /// Every later call reports this cause.
    Failed(Arc<ParallelEncodeError>),
}
enum State {
    /// Queued or running.
    Running,
    /// Encoded; `cursor` bytes have been delivered.
    Done { bytes: Vec<u8>, cursor: usize },
    /// Reported a failure that the stream has recorded.
    Failed,
}
/// One part of the output, in input order.
struct Entry {
    segment: u64,
    /// The completion slot a queued task owns; `None` for parts made inline.
    slot: Option<usize>,
    state: State,
}

/// Stream state shared by the borrowed and the owned session.
///
/// Everything that needs the parent compressor takes it as an argument, so the
/// same state serves a session over `&mut Compressor` and one that owns it.
pub(in crate::compressor::parallel) struct Stream {
    queue: Arc<Queue>,
    sender: Sender<usize>,
    receiver: Receiver<usize>,
    /// Completion slots, created on demand up to `window`.
    slots: Vec<Slot>,
    free: Vec<usize>,
    window: usize,
    /// Output parts in input order: running, encoded, or failed.
    flight: VecDeque<Entry>,
    /// Input of the segment being gathered.
    filling: Vec<u8>,
    /// The last `history_len` bytes of input before `filling`, which the
    /// next part may copy from.
    history: Vec<u8>,
    history_len: usize,
    segment_size: usize,
    minimum_parallel_size: u64,
    next_segment: u64,
    /// Stream offset of the first byte in `filling`.
    offset: u64,
    /// Whether a fragment has been queued; the serial fallback is then gone.
    fragments: bool,
    cancelled: Arc<AtomicBool>,
    phase: Phase,
    failures: Vec<(u64, ParallelEncodeError)>,
}
impl Stream {
    /// Starts a stream with at most `window` parts queued, running or undelivered.
    pub(in crate::compressor::parallel) fn new(
        parent: &Compressor,
        window: TaskCount,
    ) -> Result<(Self, QueueHandle), ParallelEncodeError> {
        let window = window.get();
        let segment_size = parent.parallel.segment_size.get();
        let history_len = parent.history_len();
        if let Some(limit) = parent.parallel.aggregate_memory_limit {
            // Per part: a worker, its staged output and input with history,
            // plus the filling and history buffers and what the parent
            // already retains.
            let staged = segment_size
                .checked_mul(3)
                .and_then(|n| n.checked_add(history_len))
                .and_then(|n| n.checked_add(1024 + 4096));
            let required = parent
                .worker_estimate()?
                .checked_add(staged.ok_or(ParallelEncodeError::SizeOverflow)?)
                .and_then(|n| n.checked_mul(window))
                .and_then(|n| n.checked_add(segment_size))
                .and_then(|n| n.checked_add(history_len))
                .and_then(|n| n.checked_add(parent.retained_bytes()))
                .ok_or(ParallelEncodeError::SizeOverflow)?;
            if required > limit {
                return Err(ParallelEncodeError::WorkerMemoryLimit);
            }
        }
        let queue = Arc::new(Queue {
            state: Mutex::new(QueueState {
                tasks: VecDeque::new(),
                closed: false,
                handles: 1,
            }),
            ready: Condvar::new(),
        });
        let (sender, receiver) = mpsc::channel();
        let stream = Self {
            queue: Arc::clone(&queue),
            sender,
            receiver,
            slots: Vec::new(),
            free: Vec::new(),
            window,
            flight: VecDeque::new(),
            filling: Vec::new(),
            history: Vec::new(),
            history_len,
            segment_size,
            minimum_parallel_size: parent.parallel.minimum_parallel_size,
            next_segment: 0,
            offset: 0,
            fragments: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            phase: Phase::Open,
            failures: Vec::new(),
        };
        Ok((stream, QueueHandle(queue)))
    }

    pub(in crate::compressor::parallel) const fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Finished)
    }

    /// Accepts input, queues full segments, and delivers finished parts in order.
    ///
    /// Exactly `output[..produced]` is written, whatever the sink.
    pub(in crate::compressor::parallel) fn process<O: Sink + ?Sized>(
        &mut self,
        parent: &mut Compressor,
        input: &[u8],
        output: &mut O,
        operation: Operation,
    ) -> Result<ParallelProgress, ParallelEncodeError> {
        self.check()?;
        if self.is_finished() {
            return Ok(ParallelProgress {
                consumed: 0,
                produced: 0,
                status: ParallelEncoderStatus::Finished,
            });
        }
        self.collect(parent)?;
        let mut produced = self.drain(output, 0);
        let mut consumed = 0;
        if matches!(self.phase, Phase::Open) {
            consumed = self.accept(parent, input)?;
            if consumed == input.len() {
                match operation {
                    Operation::Process => {}
                    Operation::Flush => {
                        if !self.filling.is_empty() {
                            self.dispatch(parent, false)?;
                        }
                    }
                    Operation::Finish => self.begin_finish(parent)?,
                }
            }
            produced = self.drain(output, produced);
        }
        let status = self.status(parent, consumed < input.len(), operation);
        Ok(ParallelProgress {
            consumed,
            produced,
            status,
        })
    }

    /// Blocks until a queued task completes; returns at once when none runs.
    pub(in crate::compressor::parallel) fn wait(
        &mut self,
        parent: &mut Compressor,
        timeout: Option<Duration>,
    ) -> Result<WaitStatus, ParallelEncodeError> {
        self.check()?;
        self.collect(parent)?;
        if !self.running() {
            return Ok(WaitStatus::Ready);
        }
        // The stream keeps its own sender, so the channel cannot disconnect.
        let id = match timeout {
            None => self.receiver.recv().ok(),
            Some(timeout) => match self.receiver.recv_timeout(timeout) {
                Ok(id) => Some(id),
                Err(RecvTimeoutError::Timeout) => return Ok(WaitStatus::TimedOut),
                Err(RecvTimeoutError::Disconnected) => None,
            },
        };
        let Some(id) = id else {
            let error = ParallelEncodeError::FragmentInvariant("completion channel closed");
            return Err(self.fail(error));
        };
        if let Err(error) = self.receive(parent, id) {
            return Err(self.fail(error));
        }
        self.collect(parent)?;
        Ok(WaitStatus::Ready)
    }

    /// Cancels queued work, closes the queue and returns every worker that
    /// has already published, without waiting for running tasks.
    pub(in crate::compressor::parallel) fn release(&mut self, parent: &mut Compressor) {
        self.cancelled.store(true, Ordering::Relaxed);
        drop(self.queue.close());
        while let Ok(id) = self.receiver.try_recv() {
            // A malformed completion only loses its worker here.
            let _ = self.receive(parent, id);
        }
        parent.trim(ParallelRetentionPolicy::CurrentPlan);
    }

    fn check(&self) -> Result<(), ParallelEncodeError> {
        match &self.phase {
            Phase::Failed(error) => Err(ParallelEncodeError::StreamFailed(Arc::clone(error))),
            _ => Ok(()),
        }
    }

    fn running(&self) -> bool {
        self.flight
            .iter()
            .any(|entry| matches!(entry.state, State::Running))
    }

    /// Records the stream's failure and stops every task that has not started.
    fn fail(&mut self, error: ParallelEncodeError) -> ParallelEncodeError {
        let error = Arc::new(error);
        self.phase = Phase::Failed(Arc::clone(&error));
        self.cancelled.store(true, Ordering::Relaxed);
        drop(self.queue.close());
        ParallelEncodeError::StreamFailed(error)
    }

    /// Takes every published completion without blocking.
    fn collect(&mut self, parent: &mut Compressor) -> Result<(), ParallelEncodeError> {
        while let Ok(id) = self.receiver.try_recv() {
            if let Err(error) = self.receive(parent, id) {
                return Err(self.fail(error));
            }
        }
        self.settle()
    }

    fn receive(&mut self, parent: &mut Compressor, id: usize) -> Result<(), ParallelEncodeError> {
        let Completion { result, worker } = self
            .slots
            .get(id)
            .ok_or(ParallelEncodeError::FragmentInvariant("unknown completion"))?
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or(ParallelEncodeError::FragmentInvariant(
                "missing completion slot",
            ))?;
        if let Some(worker) = worker {
            parent.workers.push(worker);
        }
        let entry = self
            .flight
            .iter_mut()
            .find(|entry| entry.slot == Some(id) && matches!(entry.state, State::Running))
            .ok_or(ParallelEncodeError::FragmentInvariant(
                "duplicate completion",
            ))?;
        let result = result.and_then(|artifact| {
            artifact
                .into_segment()
                .ok_or(ParallelEncodeError::FragmentInvariant("streamed artifact"))
        });
        match result {
            Ok(bytes) => entry.state = State::Done { bytes, cursor: 0 },
            Err(error) => {
                entry.state = State::Failed;
                self.failures.push((entry.segment, error));
                self.cancelled.store(true, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Fails the stream once its originating task error is known.
    ///
    /// A task that failed cancels the rest, and one of those may publish its
    /// `Cancelled` first; it waits for the cause while other parts still run.
    fn settle(&mut self) -> Result<(), ParallelEncodeError> {
        if self.failures.is_empty()
            || self.running()
                && self
                    .failures
                    .iter()
                    .all(|(_, error)| matches!(error, ParallelEncodeError::Cancelled))
        {
            return Ok(());
        }
        self.failures.sort_by_key(|(segment, error)| {
            (matches!(error, ParallelEncodeError::Cancelled), *segment)
        });
        let error = self.failures.swap_remove(0);
        self.failures.clear();
        Err(self.fail(error.1))
    }

    /// Copies delivered parts into `output` from `produced` on, in input
    /// order, and returns where the delivered bytes end.
    fn drain<O: Sink + ?Sized>(&mut self, output: &mut O, mut produced: usize) -> usize {
        while let Some(entry) = self.flight.front_mut() {
            let State::Done { bytes, cursor } = &mut entry.state else {
                break;
            };
            let take = (bytes.len() - *cursor).min(output.len() - produced);
            output.put(produced, &bytes[*cursor..*cursor + take]);
            produced += take;
            *cursor += take;
            if *cursor < bytes.len() {
                break;
            }
            if let Some(slot) = entry.slot {
                self.free.push(slot);
            }
            self.flight.pop_front();
        }
        produced
    }

    /// Copies input into segments and queues every full one that may go.
    fn accept(
        &mut self,
        parent: &mut Compressor,
        input: &[u8],
    ) -> Result<usize, ParallelEncodeError> {
        let mut consumed = 0;
        loop {
            if self.filling.len() == self.segment_size {
                // A first segment that may be the whole input stays here until
                // a following byte or Finish decides the serial fallback.
                let undecided = !self.fragments
                    && (self.segment_size as u64) < self.minimum_parallel_size
                    && consumed == input.len();
                if undecided || !self.dispatch(parent, false)? {
                    break;
                }
            }
            let take = (self.segment_size - self.filling.len()).min(input.len() - consumed);
            if take == 0 {
                break;
            }
            if let Err(error) = self
                .filling
                .try_reserve_exact(self.segment_size - self.filling.len())
            {
                return Err(self.fail(error.into()));
            }
            self.filling
                .extend_from_slice(&input[consumed..consumed + take]);
            consumed += take;
        }
        Ok(consumed)
    }

    /// Queues the filling buffer as one part, or returns `false` when the
    /// window is full.
    fn dispatch(
        &mut self,
        parent: &mut Compressor,
        serial: bool,
    ) -> Result<bool, ParallelEncodeError> {
        match self.prepare(parent, serial) {
            Ok(Some(task)) => {
                if self.queue.push(task) {
                    Ok(true)
                } else {
                    Err(self.fail(ParallelEncodeError::TaskQueueClosed))
                }
            }
            Ok(None) => Ok(false),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn prepare(
        &mut self,
        parent: &mut Compressor,
        serial: bool,
    ) -> Result<Option<Task<'static>>, ParallelEncodeError> {
        if self.free.is_empty() {
            if self.slots.len() == self.window {
                return Ok(None);
            }
            self.slots.try_reserve(1)?;
            self.free.try_reserve(1)?;
            self.slots.push(Arc::new(Mutex::new(None)));
            self.free.push(self.slots.len() - 1);
        }
        self.flight.try_reserve(1)?;
        let len = self.filling.len() as u64;
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        // Twice the raw bytes plus 1 KiB, as the batch plan allows per part.
        let bound = len
            .checked_mul(2)
            .and_then(|n| n.checked_add(1024))
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let staging = Staging::Memory(MemoryStaging::from(usize::MAX));
        let Some(&slot) = self.free.last() else {
            return Err(ParallelEncodeError::FragmentInvariant("free slot"));
        };
        // Slots stay below the window, and a task count fits `u32`.
        let id = TaskId(slot as u32);
        let artifact = match Artifact::new(&staging, 1, bound) {
            Ok(artifact) => artifact,
            Err(source) => return Err(ParallelEncodeError::Staging { task: id, source }),
        };
        let mut worker = match parent.workers.pop() {
            Some(worker) => worker,
            None => Worker::new(parent.encoder, parent.backend)?,
        };
        // The worker gets the history, then the segment. A serial part is
        // the whole input, so nothing precedes it.
        let history = if serial { 0 } else { self.history.len() };
        worker.input.clear();
        worker
            .input
            .try_reserve_exact(history + self.filling.len())?;
        self.history
            .try_reserve_exact(self.history_len.saturating_sub(self.history.len()))?;
        self.free.pop();
        worker.input.extend_from_slice(&self.history[..history]);
        worker.input.extend_from_slice(&self.filling);
        self.advance_history();
        self.filling.clear();
        let segment = self.next_segment;
        let first = !self.fragments;
        self.flight.push_back(Entry {
            segment,
            slot: Some(slot),
            state: State::Running,
        });
        self.next_segment += 1;
        self.fragments |= !serial;
        let source = self.offset..end;
        self.offset = end;
        Ok(Some(Task {
            id,
            job: Job::Streamed {
                segment,
                source,
                first,
                serial,
                history,
            },
            worker: Some(worker),
            artifact: Some(artifact),
            slot: Arc::clone(&self.slots[slot]),
            sender: Some(self.sender.clone()),
            cancelled: Arc::clone(&self.cancelled),
        }))
    }

    /// Moves the history on past `filling`: it keeps the last `history_len`
    /// bytes of the input so far. Its capacity is already reserved.
    fn advance_history(&mut self) {
        let keep = self.history_len;
        let filling = self.filling.len();
        if filling >= keep {
            self.history.clear();
            self.history
                .extend_from_slice(&self.filling[filling - keep..]);
        } else {
            let excess = (self.history.len() + filling).saturating_sub(keep);
            self.history.drain(..excess);
            self.history.extend_from_slice(&self.filling);
        }
    }

    /// Queues the last part and the stream end, choosing the batch's serial
    /// fallback when no fragment has been queued yet.
    fn begin_finish(&mut self, parent: &mut Compressor) -> Result<(), ParallelEncodeError> {
        let total = self.filling.len() as u64;
        if !self.fragments
            && (total == 0
                || total < self.minimum_parallel_size && total <= self.segment_size as u64)
        {
            if total != 0 {
                if self.dispatch(parent, true)? {
                    self.phase = Phase::Finishing;
                }
                return Ok(());
            }
            let bytes = match Self::empty(parent) {
                Ok(bytes) => bytes,
                Err(error) => return Err(self.fail(error)),
            };
            self.push_inline(bytes)?;
            self.phase = Phase::Finishing;
            return Ok(());
        }
        if !self.filling.is_empty() && !self.dispatch(parent, false)? {
            return Ok(());
        }
        // Every queued part ends aligned, so ISLAST=1, ISLASTEMPTY=1 is the
        // whole final byte.
        self.push_inline(vec![3])?;
        self.phase = Phase::Finishing;
        Ok(())
    }

    /// The canonical serial encoding of empty input, as a batch emits it.
    fn empty(parent: &mut Compressor) -> Result<Vec<u8>, ParallelEncodeError> {
        let mut worker = match parent.workers.pop() {
            Some(worker) => worker,
            None => Worker::new(parent.encoder, parent.backend)?,
        };
        let result = worker.serial.compress(&[]);
        parent.workers.push(worker);
        match result {
            Ok(bytes) => Ok(bytes),
            Err(source) => Err(ParallelEncodeError::Encode {
                task: TaskId(0),
                segment: SegmentId(0),
                source,
            }),
        }
    }

    fn push_inline(&mut self, bytes: Vec<u8>) -> Result<(), ParallelEncodeError> {
        if let Err(error) = self.flight.try_reserve(1) {
            return Err(self.fail(error.into()));
        }
        self.flight.push_back(Entry {
            segment: self.next_segment,
            slot: None,
            state: State::Done { bytes, cursor: 0 },
        });
        Ok(())
    }

    fn status(
        &mut self,
        parent: &mut Compressor,
        unconsumed: bool,
        operation: Operation,
    ) -> ParallelEncoderStatus {
        if self
            .flight
            .front()
            .is_some_and(|entry| matches!(entry.state, State::Done { .. }))
        {
            // Drained until the output filled up.
            return ParallelEncoderStatus::NeedsOutput;
        }
        match self.phase {
            Phase::Finishing if self.flight.is_empty() => {
                self.phase = Phase::Finished;
                drop(self.queue.close());
                parent.trim(ParallelRetentionPolicy::CurrentPlan);
                ParallelEncoderStatus::Finished
            }
            Phase::Finished => ParallelEncoderStatus::Finished,
            Phase::Open
                if !unconsumed
                    && match operation {
                        Operation::Process => true,
                        Operation::Flush => self.filling.is_empty() && self.flight.is_empty(),
                        Operation::Finish => false,
                    } =>
            {
                ParallelEncoderStatus::NeedsInput
            }
            _ => ParallelEncoderStatus::Pending,
        }
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        // An owned session dropped without returning its compressor still
        // stops executors; `release` has already done this otherwise.
        self.cancelled.store(true, Ordering::Relaxed);
        drop(self.queue.close());
    }
}

/// A stream over an exclusively borrowed parent.
pub(in crate::compressor::parallel) struct BorrowedStream<'c> {
    pub(in crate::compressor::parallel) parent: &'c mut Compressor,
    pub(in crate::compressor::parallel) stream: Stream,
}
impl Drop for BorrowedStream<'_> {
    fn drop(&mut self) {
        self.stream.release(self.parent);
    }
}

/// A stream that owns its parent.
pub(in crate::compressor::parallel) struct OwnedStream {
    pub(in crate::compressor::parallel) parent: Compressor,
    pub(in crate::compressor::parallel) stream: Stream,
}
impl OwnedStream {
    pub(in crate::compressor::parallel) fn into_parent(mut self) -> Compressor {
        self.stream.release(&mut self.parent);
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent() -> Compressor {
        Compressor::new(
            EncoderConfig::default(),
            ParallelConfig::from(SegmentSize::try_from(64 << 10).unwrap())
                .with_minimum_parallel_size(0),
            Backend::SCALAR,
        )
        .unwrap()
    }

    #[test]
    fn malformed_completions_fail_the_stream() {
        let mut parent = parent();
        let (mut stream, queue) = Stream::new(&parent, TaskCount::ONE).unwrap();
        assert!(stream.receive(&mut parent, 7).is_err());
        stream
            .process(&mut parent, b"abc", &mut [0u8; 0][..], Operation::Flush)
            .unwrap();
        queue.try_recv().unwrap().run();
        let id = stream.receiver.recv().unwrap();
        stream.receive(&mut parent, id).unwrap();
        // The slot is empty and the part is no longer running.
        assert!(stream.receive(&mut parent, id).is_err());
        *stream.slots[id].lock().unwrap() = Some(Completion {
            result: Err(ParallelEncodeError::Cancelled),
            worker: None,
        });
        assert!(stream.receive(&mut parent, id).is_err());
        stream.sender.send(99).unwrap();
        assert!(matches!(
            stream.wait(&mut parent, None),
            Err(ParallelEncodeError::StreamFailed(cause))
                if matches!(*cause, ParallelEncodeError::FragmentInvariant(_))
        ));
        stream.release(&mut parent);
    }

    #[test]
    fn a_cancellation_waits_for_its_cause_which_wins_by_segment() {
        let mut parent = parent();
        let (mut stream, _queue) = Stream::new(&parent, TaskCount::ONE).unwrap();
        stream.flight.push_back(Entry {
            segment: 0,
            slot: None,
            state: State::Running,
        });
        stream.failures.push((1, ParallelEncodeError::Cancelled));
        // Only fallout is known while a part still runs.
        stream.settle().unwrap();
        stream.failures.push((3, ParallelEncodeError::SizeOverflow));
        stream
            .failures
            .push((2, ParallelEncodeError::TaskAbandoned { task: TaskId(0) }));
        assert!(matches!(
            stream.settle(),
            Err(ParallelEncodeError::StreamFailed(cause))
                if matches!(*cause, ParallelEncodeError::TaskAbandoned { .. })
        ));
        assert!(stream.failures.is_empty());
        stream.release(&mut parent);
    }

    #[test]
    fn the_queue_closes_when_its_last_handle_drops() {
        let parent = parent();
        let (stream, queue) = Stream::new(&parent, TaskCount::ONE).unwrap();
        let other = queue.clone();
        drop(queue);
        assert!(!stream.queue.lock().closed);
        drop(other);
        assert!(stream.queue.lock().closed);
        assert_eq!(stream.queue.lock().handles, 0);
    }
}
