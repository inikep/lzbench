//! Incremental input, a bounded window of queued segments, and ordered output.
//!
//! The stream copies caller input into one filling buffer, which starts with
//! the overlap the next segment may reach back into. A full segment moves
//! into a task on a shared queue that the caller's executors drain; tasks
//! publish their blocks into the same shared state, and finished parts are
//! delivered strictly in input order: the frame header, every segment's
//! blocks, then the epilogue. Segment boundaries and the serial fallback are
//! those of a batch, so a stream without flushes and with a declared size
//! produces the bytes `prepare_slice` produces for the same input.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{self, MaybeUninit};
use std::sync::{Condvar, Mutex, PoisonError};

use super::{EnginePool, FramePlan, compress_segment, compress_serial, lock};
use crate::compressor::core::cctx::RequestedParams;
use crate::compressor::error::EncodeError;
use crate::compressor::parallel::{
    ParallelEncodeError, ParallelEncoderStatus, ParallelProgress, TaskCount,
};
use crate::compressor::session::Operation;
use crate::shared::xxhash::Xxh64;

/// What the stream and its executors share.
#[derive(Debug)]
struct Shared {
    state: Mutex<QueueState>,
    /// Signalled when a task is queued or the queue closes.
    ready: Condvar,
    /// Signalled when a task publishes its result.
    completed: Condvar,
}

/// One task's result: its segment and its blocks.
type Done = (u64, Result<Vec<u8>, ParallelEncodeError>);

#[derive(Debug)]
struct QueueState {
    tasks: VecDeque<StreamTask>,
    /// Set when the stream ends or the last handle drops; never cleared.
    closed: bool,
    /// Set when the stream finishes, fails or is dropped: tasks that have
    /// not started skip their work.
    cancelled: bool,
    /// Live [`QueueHandle`] values.
    handles: usize,
    /// Results not yet collected by the stream.
    done: Vec<Done>,
}

impl Shared {
    /// Queues `task`, or drops it when no executor can ever take it.
    ///
    /// A dropped task publishes its abandonment.
    fn push(&self, task: StreamTask) -> bool {
        let mut state = lock(&self.state);
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

    /// Closes the queue and drops the tasks no executor took, outside the
    /// lock: each publishes its abandonment, which takes the lock again.
    fn close(&self, cancel: bool) {
        let mut state = lock(&self.state);
        state.closed = true;
        state.cancelled |= cancel;
        let tasks = mem::take(&mut state.tasks);
        drop(state);
        self.ready.notify_all();
        drop(tasks);
    }
}

/// An executor's handle; the queue closes when the last one drops.
#[derive(Debug)]
pub(crate) struct QueueHandle(Arc<Shared>);

impl QueueHandle {
    /// Blocks until a task is queued, or returns `None` once the queue is
    /// closed.
    pub(crate) fn recv(&self) -> Option<StreamTask> {
        let mut state = lock(&self.0.state);
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
    pub(crate) fn try_recv(&self) -> Option<StreamTask> {
        lock(&self.0.state).tasks.pop_front()
    }

    /// Number of tasks queued and not yet taken.
    pub(crate) fn queued(&self) -> usize {
        lock(&self.0.state).tasks.len()
    }
}

impl Clone for QueueHandle {
    fn clone(&self) -> Self {
        lock(&self.0.state).handles += 1;
        Self(Arc::clone(&self.0))
    }
}

impl Drop for QueueHandle {
    fn drop(&mut self) {
        let mut state = lock(&self.0.state);
        state.handles -= 1;
        if state.handles != 0 {
            return;
        }
        drop(state);
        // No executor can take what is queued now: abandon it, which the
        // stream observes as completions instead of waiting forever.
        self.0.close(false);
    }
}

/// What a queued task compresses.
#[derive(Debug)]
enum Work {
    /// Blocks of one segment; the data starts with `overlap` bytes of history.
    Fragment { plan: FramePlan, overlap: usize },
    /// The whole input as one serial frame.
    Serial { requested: RequestedParams },
}

/// One queued segment compression.
#[derive(Debug)]
pub(crate) struct StreamTask {
    work: Work,
    data: Vec<u8>,
    segment: u64,
    shared: Arc<Shared>,
    pool: Arc<EnginePool>,
    /// Whether compression has started, so an unwinding drop reports a panic.
    running: bool,
    published: bool,
}

impl StreamTask {
    /// Compresses the segment and publishes its blocks.
    pub(crate) fn run(mut self) {
        if lock(&self.shared.state).cancelled {
            self.publish(Err(ParallelEncodeError::TaskAbandoned {
                segment: self.segment,
            }));
            return;
        }
        self.running = true;
        let mut engine = self.pool.take();
        let result = match &self.work {
            // Segments size their own buffers, so an error is a core bug
            // reported rather than a panic.
            Work::Fragment { plan, overlap } => {
                let mut blocks = Vec::new();
                match compress_segment(&mut engine, plan, &self.data, *overlap, &mut blocks) {
                    Ok(()) => Ok(blocks),
                    Err(error) => Err(EncodeError::from_core(error, 0, None)),
                }
            }
            Work::Serial { requested } => compress_serial(&mut engine, requested, &self.data),
        };
        // An engine that unwound is dropped instead of being returned here.
        self.pool.give(engine);
        self.publish(result.map_err(ParallelEncodeError::from));
    }

    /// Records the result exactly once and wakes the stream.
    fn publish(&mut self, result: Result<Vec<u8>, ParallelEncodeError>) {
        self.published = true;
        lock(&self.shared.state).done.push((self.segment, result));
        self.shared.completed.notify_all();
    }
}

impl Drop for StreamTask {
    fn drop(&mut self) {
        if !self.published {
            let segment = self.segment;
            self.publish(Err(if self.running {
                ParallelEncodeError::TaskPanicked { segment }
            } else {
                ParallelEncodeError::TaskAbandoned { segment }
            }));
        }
    }
}

/// A caller destination the stream copies delivered bytes into.
///
/// Delivery only ever writes, so an uninitialized slice serves as well as an
/// initialized one, and each gets its own instantiation of `process`.
pub(crate) trait Sink {
    /// Capacity of the destination.
    fn capacity(&self) -> usize;
    /// Writes `bytes` at `at`; `at + bytes.len()` never exceeds the capacity.
    fn put(&mut self, at: usize, bytes: &[u8]);
}

impl Sink for [u8] {
    fn capacity(&self) -> usize {
        self.len()
    }

    fn put(&mut self, at: usize, bytes: &[u8]) {
        self[at..at + bytes.len()].copy_from_slice(bytes);
    }
}

impl Sink for [MaybeUninit<u8>] {
    fn capacity(&self) -> usize {
        self.len()
    }

    fn put(&mut self, at: usize, bytes: &[u8]) {
        for (slot, &byte) in self[at..at + bytes.len()].iter_mut().zip(bytes) {
            slot.write(byte);
        }
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
enum State {
    /// Queued or running.
    Running,
    /// Compressed; `cursor` bytes have been delivered.
    Done { bytes: Vec<u8>, cursor: usize },
    /// The task failed with this cause.
    Failed(ParallelEncodeError),
}

/// One part of the output, in order.
#[derive(Debug)]
struct Entry {
    /// The segment of a queued task; `None` for parts made inline (the
    /// header, the epilogue and the frame of an empty input).
    segment: Option<u64>,
    state: State,
}

/// The state machine shared by the borrowed and the owned session.
#[derive(Debug)]
pub(crate) struct Stream {
    shared: Arc<Shared>,
    pool: Arc<EnginePool>,
    requested: RequestedParams,
    plan: FramePlan,
    declared: Option<u64>,
    segment_size: usize,
    window: usize,
    /// Parts with a task that are queued, running or not fully delivered.
    in_flight: usize,
    /// Output parts in order.
    flight: VecDeque<Entry>,
    /// The history the next segment may reach into, then its input.
    filling: Vec<u8>,
    /// Leading bytes of `filling` that belong to earlier segments.
    history: usize,
    received: u64,
    hasher: Xxh64,
    next_segment: u64,
    /// Whether a fragment has been queued; the serial fallback is then gone.
    fragments: bool,
    phase: Phase,
}

impl Stream {
    /// Starts a stream with at most `window` parts queued, running or
    /// undelivered, and the handle its executors take tasks from.
    pub(crate) fn new(
        requested: RequestedParams,
        declared: Option<u64>,
        segment_size: usize,
        window: TaskCount,
        pool: Arc<EnginePool>,
    ) -> (Self, QueueHandle) {
        let shared = Arc::new(Shared {
            state: Mutex::new(QueueState {
                tasks: VecDeque::new(),
                closed: false,
                cancelled: false,
                handles: 1,
                done: Vec::new(),
            }),
            ready: Condvar::new(),
            completed: Condvar::new(),
        });
        let stream = Self {
            shared: Arc::clone(&shared),
            pool,
            requested,
            plan: FramePlan::new(&requested, declared),
            declared,
            segment_size,
            window: window.get(),
            in_flight: 0,
            flight: VecDeque::new(),
            filling: Vec::new(),
            history: 0,
            received: 0,
            hasher: Xxh64::new(0),
            next_segment: 0,
            fragments: false,
            phase: Phase::Open,
        };
        (stream, QueueHandle(shared))
    }

    pub(crate) const fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Finished)
    }

    /// Accepts input, queues full segments, and delivers finished parts in
    /// order. Never blocks.
    ///
    /// Exactly `output[..produced]` is written, whatever the sink.
    pub(crate) fn process<O: Sink + ?Sized>(
        &mut self,
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
        self.collect()?;
        let mut produced = self.drain(output, 0);
        let mut consumed = 0;
        if matches!(self.phase, Phase::Open) {
            consumed = self.accept(input)?;
            if consumed == input.len() {
                match operation {
                    Operation::Continue => {}
                    Operation::Flush => {
                        if self.pending() != 0 {
                            self.dispatch_fragment()?;
                        }
                    }
                    Operation::End => self.begin_finish()?,
                }
            }
            produced = self.drain(output, produced);
        }
        let status = self.status(consumed < input.len(), operation);
        Ok(ParallelProgress {
            consumed,
            produced,
            status,
        })
    }

    /// Blocks until a queued task completes; returns at once when one has
    /// completed since the last call, or when none is queued or running.
    pub(crate) fn wait(&mut self) -> Result<(), ParallelEncodeError> {
        self.check()?;
        if self.collect()? != 0
            || !self
                .flight
                .iter()
                .any(|entry| matches!(entry.state, State::Running))
        {
            return Ok(());
        }
        {
            // Every task publishes exactly once, even when it is dropped or
            // unwinds, so a running part always ends this wait.
            let mut state = lock(&self.shared.state);
            while state.done.is_empty() {
                state = self
                    .shared
                    .completed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        self.collect().map(drop)
    }

    /// Bytes of the segment being gathered.
    const fn pending(&self) -> usize {
        self.filling.len() - self.history
    }

    fn check(&self) -> Result<(), ParallelEncodeError> {
        match &self.phase {
            Phase::Failed(cause) => Err(ParallelEncodeError::StreamFailed(Arc::clone(cause))),
            _ => Ok(()),
        }
    }

    /// Records the stream's failure and stops every task that has not
    /// started.
    fn fail(&mut self, error: ParallelEncodeError) -> ParallelEncodeError {
        let cause = Arc::new(error);
        self.phase = Phase::Failed(Arc::clone(&cause));
        self.shared.close(true);
        ParallelEncodeError::StreamFailed(cause)
    }

    /// Takes every published result without blocking and returns how many
    /// it took; fails the stream once the first part not yet compressed is a
    /// failure.
    ///
    /// Waiting for earlier parts makes the reported cause the first failing
    /// segment in input order, whatever order tasks finished in.
    fn collect(&mut self) -> Result<usize, ParallelEncodeError> {
        let done = mem::take(&mut lock(&self.shared.state).done);
        let count = done.len();
        for (segment, result) in done {
            if let Some(entry) = self
                .flight
                .iter_mut()
                .find(|entry| entry.segment == Some(segment))
            {
                entry.state = match result {
                    Ok(bytes) => State::Done { bytes, cursor: 0 },
                    Err(error) => State::Failed(error),
                };
            }
        }
        let first = self
            .flight
            .iter()
            .find(|entry| !matches!(entry.state, State::Done { .. }));
        if let Some(Entry {
            state: State::Failed(error),
            ..
        }) = first
        {
            let error = error.clone();
            return Err(self.fail(error));
        }
        Ok(count)
    }

    /// Copies compressed parts into `output` from `produced` on, in order,
    /// and returns where the delivered bytes end.
    fn drain<O: Sink + ?Sized>(&mut self, output: &mut O, mut produced: usize) -> usize {
        while let Some(entry) = self.flight.front_mut() {
            let State::Done { bytes, cursor } = &mut entry.state else {
                break;
            };
            let take = (bytes.len() - *cursor).min(output.capacity() - produced);
            output.put(produced, &bytes[*cursor..*cursor + take]);
            produced += take;
            *cursor += take;
            if *cursor < bytes.len() {
                break;
            }
            if entry.segment.is_some() {
                self.in_flight -= 1;
            }
            self.flight.pop_front();
        }
        produced
    }

    /// Copies input into segments and queues every full one the window
    /// admits; returns how much was taken.
    fn accept(&mut self, input: &[u8]) -> Result<usize, ParallelEncodeError> {
        if let Some(declared) = self.declared {
            let actual = self.received + input.len() as u64;
            if actual > declared {
                return Err(self.fail(ParallelEncodeError::InputSizeMismatch { declared, actual }));
            }
        }
        let mut consumed = 0;
        loop {
            if self.pending() == self.segment_size {
                // A first segment that may be the whole input stays here
                // until a following byte or `End` decides the serial
                // fallback.
                let undecided = !self.fragments && consumed == input.len();
                if undecided || !self.dispatch_fragment()? {
                    break;
                }
            }
            let take = (self.segment_size - self.pending()).min(input.len() - consumed);
            if take == 0 {
                break;
            }
            let chunk = &input[consumed..consumed + take];
            if self.filling.capacity() < self.history + self.segment_size {
                self.filling
                    .reserve_exact(self.history + self.segment_size - self.filling.len());
            }
            self.filling.extend_from_slice(chunk);
            if self.plan.applied.fparams.checksum {
                self.hasher.update(chunk);
            }
            consumed += take;
        }
        self.received += consumed as u64;
        Ok(consumed)
    }

    /// Queues the gathered segment as one fragment, or returns `false` when
    /// the window is full. The first fragment is preceded by the header.
    fn dispatch_fragment(&mut self) -> Result<bool, ParallelEncodeError> {
        if self.in_flight == self.window {
            return Ok(false);
        }
        if !self.fragments {
            self.fragments = true;
            self.push_inline(self.plan.header());
        }
        let keep = self.plan.overlap().min(self.filling.len());
        // The next segment's buffer is allocated once, at its full size.
        let mut next = Vec::with_capacity(keep + self.segment_size);
        next.extend_from_slice(&self.filling[self.filling.len() - keep..]);
        let data = mem::replace(&mut self.filling, next);
        let overlap = mem::replace(&mut self.history, keep);
        self.queue(
            Work::Fragment {
                plan: self.plan,
                overlap,
            },
            data,
        )?;
        Ok(true)
    }

    /// Hands `data` to a new task at the back of the output order.
    fn queue(&mut self, work: Work, data: Vec<u8>) -> Result<(), ParallelEncodeError> {
        let segment = self.next_segment;
        self.next_segment += 1;
        self.in_flight += 1;
        self.flight.push_back(Entry {
            segment: Some(segment),
            state: State::Running,
        });
        let task = StreamTask {
            work,
            data,
            segment,
            shared: Arc::clone(&self.shared),
            pool: Arc::clone(&self.pool),
            running: false,
            published: false,
        };
        if self.shared.push(task) {
            Ok(())
        } else {
            Err(self.fail(ParallelEncodeError::TaskQueueClosed))
        }
    }

    /// Queues the last part and the epilogue, choosing the batch's serial
    /// fallback when no fragment has been queued yet.
    fn begin_finish(&mut self) -> Result<(), ParallelEncodeError> {
        if let Some(declared) = self.declared
            && self.received != declared
        {
            return Err(self.fail(ParallelEncodeError::InputSizeMismatch {
                declared,
                actual: self.received,
            }));
        }
        if !self.fragments {
            // `accept` keeps a full first segment back until more input
            // arrives, so everything gathered fits in one segment, and
            // nothing is in flight to fill the window.
            let data = mem::take(&mut self.filling);
            if data.is_empty() {
                // An empty input needs no executor.
                let mut engine = self.pool.take();
                let frame = compress_serial(&mut engine, &self.requested, &data);
                self.pool.give(engine);
                match frame {
                    Ok(frame) => self.push_inline(frame),
                    Err(error) => return Err(self.fail(error.into())),
                }
            } else {
                self.queue(
                    Work::Serial {
                        requested: self.requested,
                    },
                    data,
                )?;
            }
            self.phase = Phase::Finishing;
            return Ok(());
        }
        if self.pending() != 0 && !self.dispatch_fragment()? {
            return Ok(());
        }
        self.push_inline(self.plan.epilogue(&self.hasher));
        self.phase = Phase::Finishing;
        Ok(())
    }

    fn push_inline(&mut self, bytes: Vec<u8>) {
        self.flight.push_back(Entry {
            segment: None,
            state: State::Done { bytes, cursor: 0 },
        });
    }

    fn status(&mut self, unconsumed: bool, operation: Operation) -> ParallelEncoderStatus {
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
                self.shared.close(true);
                ParallelEncoderStatus::Finished
            }
            Phase::Open
                if !unconsumed
                    && match operation {
                        Operation::Continue => true,
                        Operation::Flush => self.pending() == 0 && self.flight.is_empty(),
                        Operation::End => false,
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
        // Ends every executor loop and skips tasks that have not started.
        self.shared.close(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::parallel::SegmentSize;

    fn stream(window: usize) -> (Stream, QueueHandle) {
        Stream::new(
            RequestedParams::default(),
            None,
            SegmentSize::MIN.get(),
            TaskCount::try_from(window).unwrap(),
            Arc::new(EnginePool::default()),
        )
    }

    #[test]
    fn results_for_unknown_segments_are_ignored() {
        let (mut stream, _queue) = stream(1);
        lock(&stream.shared.state).done.push((9, Ok(Vec::new())));
        assert_eq!(stream.collect().unwrap(), 1);
        assert!(stream.flight.is_empty());
    }

    #[test]
    fn the_first_failure_in_input_order_fails_the_stream() {
        let (mut stream, _queue) = stream(4);
        for segment in 0..3 {
            stream.flight.push_back(Entry {
                segment: Some(segment),
                state: State::Running,
            });
        }
        // A later segment's failure waits while an earlier one runs.
        lock(&stream.shared.state)
            .done
            .push((2, Err(ParallelEncodeError::TaskAbandoned { segment: 2 })));
        stream.collect().unwrap();
        lock(&stream.shared.state).done.extend([
            (0, Ok(vec![1])),
            (1, Err(ParallelEncodeError::TaskPanicked { segment: 1 })),
        ]);
        let error = stream.collect().unwrap_err();
        assert_eq!(
            error,
            ParallelEncodeError::StreamFailed(Arc::new(ParallelEncodeError::TaskPanicked {
                segment: 1
            }))
        );
        assert_eq!(stream.wait(), Err(error));
    }

    #[test]
    fn cancelled_tasks_report_abandonment_without_compressing() {
        let (mut stream, queue) = stream(1);
        stream.filling.extend_from_slice(b"abc");
        assert!(stream.dispatch_fragment().unwrap());
        let task = queue.try_recv().unwrap();
        lock(&stream.shared.state).cancelled = true;
        task.run();
        let done = mem::take(&mut lock(&stream.shared.state).done);
        assert_eq!(
            done,
            [(0, Err(ParallelEncodeError::TaskAbandoned { segment: 0 }))]
        );
    }

    #[test]
    fn a_task_that_unwinds_reports_a_panic() {
        let (mut stream, queue) = stream(1);
        stream.filling.extend_from_slice(b"abc");
        assert!(stream.dispatch_fragment().unwrap());
        let mut task = queue.try_recv().unwrap();
        // What an unwinding `run` leaves behind: started, never published.
        task.running = true;
        drop(task);
        assert_eq!(
            stream.wait(),
            Err(ParallelEncodeError::StreamFailed(Arc::new(
                ParallelEncodeError::TaskPanicked { segment: 0 }
            )))
        );
    }

    #[test]
    fn the_queue_closes_when_its_last_handle_drops() {
        let (stream, queue) = stream(1);
        let other = queue.clone();
        drop(queue);
        assert!(!lock(&stream.shared.state).closed);
        drop(other);
        let state = lock(&stream.shared.state);
        assert!(state.closed);
        assert!(!state.cancelled);
        assert_eq!(state.handles, 0);
    }

    #[test]
    fn uninitialized_sinks_write_only_the_delivered_range() {
        let mut output = [MaybeUninit::new(0xAAu8); 6];
        let sink: &mut [MaybeUninit<u8>] = &mut output;
        assert_eq!(sink.capacity(), 6);
        sink.put(1, b"abcd");
        // SAFETY: the array literal initialized every element.
        let bytes = output.map(|slot| unsafe { slot.assume_init() });
        assert_eq!(&bytes, b"\xAAabcd\xAA");
    }
}
