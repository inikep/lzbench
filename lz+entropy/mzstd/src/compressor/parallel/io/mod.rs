//! `std::io` adapters over a parallel session.
//!
//! Both adapters own only buffering and the conventions `Read` and `Write`
//! impose; every segmentation, scheduling and delivery decision belongs to
//! the [`ParallelEncoderSession`](super::ParallelEncoderSession) underneath,
//! so they produce exactly the bytes the session does. Unlike the session's
//! `process`, they block in `wait` while the window is full or the next part
//! is pending, so the queue must run on other threads.

mod reader;
mod writer;

pub use reader::ParallelEncoderReader;
pub use writer::ParallelEncoderWriter;

use std::io::{Read, Write};

use super::{ParallelCompressor, ParallelTaskQueue, TaskCount};
use crate::compressor::session::StreamConfig;

impl ParallelCompressor {
    /// Compresses everything written to the returned adapter into `sink` in
    /// parallel, and returns the queue its tasks go to.
    ///
    /// `stream` and `window` are those of [`Self::start`]. No thread is
    /// created: run the queue's tasks on executors other than the writing
    /// thread, since a write blocks while the window is full. See
    /// [`ParallelEncoderWriter`] for the protocol.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use mzstd::io::FinishError;
    /// use std::io::Write;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let frame = std::thread::scope(|scope| {
    ///     let (mut sink, queue) = compressor.writer(Vec::new(), Default::default(), TaskCount::try_from(2)?);
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     sink.write_all(b"one executor, two parts in flight")?;
    ///     Ok::<_, Box<dyn std::error::Error>>(sink.finish().map_err(FinishError::into_error)?)
    /// })?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(
        &mut self,
        sink: W,
        stream: StreamConfig,
        window: TaskCount,
    ) -> (ParallelEncoderWriter<'_, W>, ParallelTaskQueue) {
        let (session, queue) = self.start(stream, window);
        (ParallelEncoderWriter::new(session, sink), queue)
    }

    /// Reads `source` to its end and yields its parallel-compressed frame,
    /// and returns the queue its tasks go to.
    ///
    /// `stream` and `window` are those of [`Self::start`]. No thread is
    /// created: run the queue's tasks on executors other than the reading
    /// thread, since a read blocks until the next part is ready. See
    /// [`ParallelEncoderReader`] for the protocol.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use std::io::Read;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let frame = std::thread::scope(|scope| {
    ///     let (mut source, queue) =
    ///         compressor.reader(&b"read in parallel"[..], Default::default(), TaskCount::ONE);
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     let mut frame = Vec::new();
    ///     source.read_to_end(&mut frame)?;
    ///     Ok::<_, std::io::Error>(frame)
    /// })?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(
        &mut self,
        source: R,
        stream: StreamConfig,
        window: TaskCount,
    ) -> (ParallelEncoderReader<'_, R>, ParallelTaskQueue) {
        let (session, queue) = self.start(stream, window);
        (ParallelEncoderReader::new(session, source), queue)
    }
}
