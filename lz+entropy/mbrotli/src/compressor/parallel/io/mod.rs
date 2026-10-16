//! `std::io` adapters over a parallel session.
//!
//! Both adapters only own buffering and the conventions `Read` and `Write`
//! impose; every segmentation, scheduling and delivery decision belongs to the
//! [`ParallelEncoderSession`](super::ParallelEncoderSession) underneath, so
//! they produce exactly the bytes the session does.

mod reader;
mod writer;

pub use reader::ParallelEncoderReader;
pub use writer::ParallelEncoderWriter;

use super::{ParallelCompressor, ParallelEncodeError, ParallelTaskQueue, TaskCount};
use std::io::{Read, Write};

impl ParallelCompressor {
    /// Wraps `sink` in an adapter that compresses everything written to it in
    /// parallel, and returns the queue its tasks go to.
    ///
    /// `window` bounds the segments queued, running or awaiting delivery, as
    /// for [`Self::start`]. No thread is created: run the queue's tasks on
    /// executors other than the writing thread, since a write blocks while
    /// the window is full. See [`ParallelEncoderWriter`] for the protocol.
    ///
    /// # Errors
    ///
    /// As [`Self::start`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use mbrotli::io::FinishError;
    /// use std::io::Write;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let compressed = std::thread::scope(|scope| {
    ///     let (mut sink, queue) = compressor.writer(Vec::new(), TaskCount::try_from(2)?)?;
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     sink.write_all(b"one executor, two parts in flight")?;
    ///     Ok::<_, Box<dyn std::error::Error>>(sink.finish().map_err(FinishError::into_error)?)
    /// })?;
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(
        &mut self,
        sink: W,
        window: TaskCount,
    ) -> Result<(ParallelEncoderWriter<'_, W>, ParallelTaskQueue), ParallelEncodeError> {
        let (session, queue) = self.start(window)?;
        Ok((ParallelEncoderWriter::new(session, sink), queue))
    }

    /// Wraps `source` in an adapter that yields its parallel-compressed form,
    /// and returns the queue its tasks go to.
    ///
    /// `window` bounds the segments queued, running or awaiting delivery, as
    /// for [`Self::start`]. No thread is created: run the queue's tasks on
    /// executors other than the reading thread, since a read blocks until the
    /// next part is ready. See [`ParallelEncoderReader`] for the protocol.
    ///
    /// # Errors
    ///
    /// As [`Self::start`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use std::io::Read;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let compressed = std::thread::scope(|scope| {
    ///     let (mut source, queue) = compressor.reader(&b"read in parallel"[..], TaskCount::ONE)?;
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     let mut compressed = Vec::new();
    ///     source.read_to_end(&mut compressed)?;
    ///     Ok::<_, Box<dyn std::error::Error>>(compressed)
    /// })?;
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(
        &mut self,
        source: R,
        window: TaskCount,
    ) -> Result<(ParallelEncoderReader<'_, R>, ParallelTaskQueue), ParallelEncodeError> {
        let (session, queue) = self.start(window)?;
        Ok((ParallelEncoderReader::new(session, source), queue))
    }
}
