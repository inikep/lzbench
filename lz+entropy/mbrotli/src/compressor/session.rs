//! The incremental encoder: one stream, driven a chunk at a time.
//!
//! [`EncoderSession`] is the low-level state machine every streaming path in
//! this crate is built on. `EncoderReader` and
//! `EncoderWriter` are adapters over it, and they
//! add buffering and `std::io` conventions rather than a second encoder.
//!
//! A session borrows its compressor exclusively for as long as it lives, and
//! borrows at most one dictionary immutably. It never keeps the caller's input
//! or output slices: everything it needs between calls lives in the
//! compressor's own retained buffers.
//!
//! # Byte identity across API shapes
//!
//! A zero-offset session with [`InputSize::Exact`] and no explicit flush produces
//! exactly the same bytes as [`Compressor::compress`](super::Compressor::compress),
//! including empty and incompressible inputs. Vector, slice, reader and writer
//! destinations do not select a different encoding. Short slices return an error
//! rather than a smaller alternative stream.
//!
//! Declared input size, dictionary and flush boundaries are stream settings:
//! changing them can change output. C's native one-shot empty-input and whole-stream
//! uncompressed rewrites are deliberately not used, since an incremental stream
//! cannot rewind output already delivered to its caller.
//!
//! One-shot and incremental encoding share a private block scheduler. Sessions
//! stage undecided input tails so caller chunk boundaries do not affect output.
//! Native C quality-zero/one streaming emits fragments at PROCESS boundaries,
//! so arbitrary C chunk schedules are not a byte-identity oracle for those
//! qualities.

use super::dictionary::PreparedDictionary;
use super::encoder::Compressor;
use super::error::EncodeError;

/// How much input a stream will carry, when that is known in advance.
///
/// Qualities four and five choose a different match finder for inputs of a
/// mebibyte or more, so telling the encoder how much is coming changes the
/// bytes it emits. [`InputSize::Exact`] is what makes a streamed stream match
/// the same bytes compressed in one shot.
///
/// `Exact(0)` declares a stream that is known to be empty, which is a different
/// statement from `Unknown` even though the reference resolves the same match
/// finder for both.
///
/// # Examples
///
/// ```
/// use mbrotli::InputSize;
///
/// assert_eq!(InputSize::default(), InputSize::Unknown);
/// assert_eq!(InputSize::from(4096u64), InputSize::Exact(4096));
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum InputSize {
    /// How much input is coming is not known.
    #[default]
    Unknown,
    /// The stream will carry exactly this many bytes.
    Exact(u64),
}

impl InputSize {
    /// Returns the size hint the encoders resolve their match finder from.
    ///
    /// An unknown size is zero, which is what the reference's streaming entry
    /// point leaves `BROTLI_PARAM_SIZE_HINT` at.
    pub(crate) const fn hint(self) -> usize {
        match self {
            Self::Unknown => 0,
            // A hint wider than the address space cannot select a different
            // match finder than the widest one that fits, so saturating here
            // changes no decision.
            Self::Exact(size) if size > usize::MAX as u64 => usize::MAX,
            Self::Exact(size) => size as usize,
        }
    }
}

impl From<u64> for InputSize {
    /// Declares an exactly known input size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::InputSize;
    ///
    /// assert_eq!(InputSize::from(0u64), InputSize::Exact(0));
    /// ```
    fn from(value: u64) -> Self {
        Self::Exact(value)
    }
}

/// What a single stream knows about itself.
///
/// Everything here belongs to one stream rather than to the encoder: how much
/// input is coming, and where the stream sits logically. The encoder's own
/// settings are in [`EncoderConfig`](super::EncoderConfig).
///
/// # Examples
///
/// ```
/// use mbrotli::{InputSize, StreamConfig};
///
/// let stream = StreamConfig::from(InputSize::Exact(4096));
///
/// assert_eq!(stream.input_size(), InputSize::Exact(4096));
/// assert_eq!(stream.stream_offset(), 0);
/// assert_eq!(StreamConfig::default().input_size(), InputSize::Unknown);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct StreamConfig {
    /// How much input the stream will carry.
    input_size: InputSize,
    /// Where the stream begins, logically.
    stream_offset: u64,
}

impl StreamConfig {
    /// Sets how much input the stream will carry.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{InputSize, StreamConfig};
    ///
    /// let stream = StreamConfig::default().with_input_size(InputSize::Exact(10));
    ///
    /// assert_eq!(stream.input_size(), InputSize::Exact(10));
    /// ```
    #[must_use]
    pub const fn with_input_size(mut self, input_size: InputSize) -> Self {
        self.input_size = input_size;
        self
    }

    /// Returns how much input the stream will carry.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{InputSize, StreamConfig};
    ///
    /// assert_eq!(StreamConfig::default().input_size(), InputSize::Unknown);
    /// ```
    #[must_use]
    pub const fn input_size(&self) -> InputSize {
        self.input_size
    }

    /// Sets where the stream begins, logically.
    ///
    /// A non-zero offset requires the `experimental` feature and quality 2 or
    /// higher. It emits a headerless continuation after a byte-aligned flush,
    /// with no references to unavailable prior history. The caller must join
    /// it to a compatible stream; it is not independently decodable. Logical
    /// positions, including the input, must fit in 63 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::StreamConfig;
    ///
    /// assert_eq!(StreamConfig::default().with_stream_offset(64).stream_offset(), 64);
    /// ```
    #[must_use]
    pub const fn with_stream_offset(mut self, stream_offset: u64) -> Self {
        self.stream_offset = stream_offset;
        self
    }

    /// Returns where the stream begins, logically.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::StreamConfig;
    ///
    /// assert_eq!(StreamConfig::default().stream_offset(), 0);
    /// ```
    #[must_use]
    pub const fn stream_offset(&self) -> u64 {
        self.stream_offset
    }
}

impl From<InputSize> for StreamConfig {
    /// Builds a stream configuration from its size alone, at offset zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{InputSize, StreamConfig};
    ///
    /// assert_eq!(
    ///     StreamConfig::from(InputSize::Unknown),
    ///     StreamConfig::default()
    /// );
    /// ```
    fn from(value: InputSize) -> Self {
        Self {
            input_size: value,
            stream_offset: 0,
        }
    }
}

/// What a call to [`EncoderSession::process`] should do with the stream.
///
/// # Examples
///
/// ```
/// use mbrotli::Operation;
///
/// assert_eq!(Operation::default(), Operation::Process);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Operation {
    /// Take input and emit whatever completes; keep gathering otherwise.
    #[default]
    Process,
    /// Make everything accepted so far decodable, without ending the stream.
    ///
    /// Costs ratio: the meta-block ends early, so its entropy codes are built
    /// from less data, and an empty metadata block is added to realign the
    /// stream to a byte boundary. Flushing per small write can make the output
    /// larger than the input; flush on the boundaries the protocol has.
    Flush,
    /// Emit everything left and terminate the stream.
    Finish,
}

/// What a session needs next.
///
/// # Examples
///
/// ```
/// use mbrotli::EncoderStatus;
///
/// assert_ne!(EncoderStatus::NeedsInput, EncoderStatus::Finished);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EncoderStatus {
    /// More source bytes are needed before anything else can happen.
    NeedsInput,
    /// Encoded bytes are waiting; call again with room to put them.
    NeedsOutput,
    /// The stream is complete and the final bytes have been delivered.
    Finished,
}

/// What one [`EncoderSession::process`] call did.
///
/// `consumed` and `produced` are exact: the session never takes a byte it did
/// not stage, and never claims a byte it did not write.
///
/// # Examples
///
/// ```
/// use mbrotli::{EncoderStatus, Progress};
///
/// let progress = Progress { consumed: 4, produced: 0, status: EncoderStatus::NeedsInput };
///
/// assert_eq!(progress.consumed, 4);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Progress {
    /// How many bytes were taken from the caller's input.
    pub consumed: usize,
    /// How many bytes were written into the caller's output.
    pub produced: usize,
    /// What the session needs next.
    pub status: EncoderStatus,
}

/// One incremental Brotli stream.
///
/// Created by [`Compressor::start`](super::Compressor::start) or
/// [`Compressor::start_with_dictionary`](super::Compressor::start_with_dictionary),
/// and driven by [`EncoderSession::process`] until it reports
/// [`EncoderStatus::Finished`].
///
/// Dropping a session before it finishes abandons the stream: the bytes emitted
/// so far are not a complete Brotli stream and no decoder will accept them. The
/// compressor is left ready for the next stream either way.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, InputSize, Operation, Quality};
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
/// let payload = b"a payload compressed one chunk at a time".repeat(10);
///
/// let mut compressed = Vec::new();
/// let mut buffer = [0u8; 64];
/// let mut input = payload.as_slice();
/// {
///     let mut session = encoder.start(InputSize::Exact(payload.len() as u64).into())?;
///     loop {
///         let progress = session.process(input, &mut buffer, Operation::Finish)?;
///         input = &input[progress.consumed..];
///         compressed.extend_from_slice(&buffer[..progress.produced]);
///         if progress.status == EncoderStatus::Finished {
///             break;
///         }
///     }
/// }
///
/// assert!(!compressed.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct EncoderSession<'c, 'd> {
    core: super::core::session::SessionCore<'c, 'd>,
}

impl<'c, 'd> EncoderSession<'c, 'd> {
    /// Starts a stream on `compressor`.
    ///
    /// The caller has already validated the stream configuration and acquired
    /// the encoder, which is what fixes `limit`.
    pub(crate) fn new(
        compressor: &'c mut Compressor,
        dictionary: Option<&'d PreparedDictionary>,
        limit: usize,
        stream: StreamConfig,
    ) -> Self {
        Self {
            core: super::core::session::SessionCore::new(compressor, dictionary, limit, stream),
        }
    }

    /// Moves the stream forward by one step.
    ///
    /// Takes what it can from `input`, writes what it can into `output`, and
    /// reports exactly how much of each it moved along with what it needs next.
    /// Both slices may be empty, and either may be a single byte; the session
    /// never spins on a call that made no progress, it reports why instead.
    ///
    /// A call returns [`EncoderStatus::NeedsOutput`] while encoded bytes are
    /// still waiting, [`EncoderStatus::NeedsInput`] when the operation it was
    /// given has done all it can, and [`EncoderStatus::Finished`] once a
    /// [`Operation::Finish`] has been completed and delivered. After that it is
    /// idempotent: further calls consume nothing, produce nothing and report
    /// `Finished`.
    ///
    /// The operation may change between calls. A `Finish` that returns
    /// `NeedsOutput` must be repeated — with the same operation — until it
    /// reports `Finished`; the final meta-block is encoded once however many
    /// calls it takes to deliver.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidState`] when the stream has already failed,
    /// and propagates whatever the encoder reports. A failed session encodes
    /// nothing further.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut session = encoder.start(Default::default())?;
    /// let mut output = [0u8; 256];
    ///
    /// // An empty input still ends in a complete stream.
    /// let progress = session.process(b"", &mut output, Operation::Finish)?;
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// assert!(progress.produced > 0);
    ///
    /// // And the finished session stays finished.
    /// let again = session.process(b"", &mut output, Operation::Finish)?;
    /// assert_eq!(again, mbrotli::Progress {
    ///     consumed: 0,
    ///     produced: 0,
    ///     status: EncoderStatus::Finished,
    /// });
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.core.process(input, output, operation)
    }

    /// Makes everything accepted so far decodable, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Flush`]. Repeat it while it reports
    /// [`EncoderStatus::NeedsOutput`]; `NeedsInput` means the flush has been
    /// delivered and the stream stays open.
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut encoder = Compressor::new(Default::default())?;
    /// let mut session = encoder.start(Default::default())?;
    /// let mut output = [0u8; 256];
    /// session.process(b"flushed payload", &mut output, Operation::Process)?;
    /// let progress = session.flush(&mut output)?;
    /// assert_eq!(progress.status, EncoderStatus::NeedsInput);
    /// assert!(progress.produced > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
        self.process(&[], output, Operation::Flush)
    }

    /// Terminates the stream, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Finish`], for when every input byte has already been
    /// passed to `process`. Repeat it while it reports
    /// [`EncoderStatus::NeedsOutput`], until [`EncoderStatus::Finished`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut encoder = Compressor::new(Default::default())?;
    /// let mut session = encoder.start(Default::default())?;
    /// let mut compressed = Vec::new();
    /// let mut output = [0u8; 4];
    /// let mut input = &b"finished without input"[..];
    /// while !input.is_empty() {
    ///     let progress = session.process(input, &mut output, Operation::Process)?;
    ///     input = &input[progress.consumed..];
    ///     compressed.extend_from_slice(&output[..progress.produced]);
    /// }
    /// loop {
    ///     let progress = session.finish(&mut output)?;
    ///     compressed.extend_from_slice(&output[..progress.produced]);
    ///     if progress.status == EncoderStatus::Finished {
    ///         break;
    ///     }
    /// }
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
        self.process(&[], output, Operation::Finish)
    }

    /// Returns whether the stream has been terminated and delivered.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Operation, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut session = encoder.start(Default::default())?;
    /// let mut output = [0u8; 256];
    ///
    /// assert!(!session.is_finished());
    /// session.process(b"payload", &mut output, Operation::Finish)?;
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.core.is_finished()
    }
}

/// One incremental Brotli stream that owns its compressor.
///
/// [`EncoderSession`] borrows its [`Compressor`] with `&mut` for as long as it
/// lives. `EncoderSessionOwned` consumes the compressor instead, so the stream
/// state is one ordinary owned value with no lifetime tied to the compressor.
/// That suits wrappers which have to store the codec state themselves, such as
/// adapters that keep it in a struct between calls. It is still the same
/// synchronous, caller-driven API: it is not an async interface, and it keeps
/// no references into its own fields.
///
/// Created by [`Compressor::into_session`](super::Compressor::into_session) or
/// [`Compressor::into_session_with_dictionary`](super::Compressor::into_session_with_dictionary).
/// [`Self::process`] runs the same state machine as
/// [`EncoderSession::process`], so the same calls produce the same bytes,
/// counts, statuses and errors. [`Self::into_compressor`] ends the stream the
/// way dropping a borrowed session does and hands the compressor back ready for
/// the next operation. Dropping the owned session instead drops the compressor
/// with it.
///
/// The session has no lifetime. A dictionary, when one is attached, is owned
/// too, as any `D: AsRef<PreparedDictionary> + 'static`: an
/// `Arc<PreparedDictionary>` to share one without copying it, a
/// `&'static PreparedDictionary`, or the dictionary itself. Without one, `D`
/// stays at its default and is never constructed.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncoderStatus, Operation};
///
/// let compressor = Compressor::new(Default::default())?;
/// let mut session = compressor.into_session(Default::default())?;
/// let mut output = [0u8; 256];
///
/// let progress = session.process(b"owned stream", &mut output, Operation::Finish)?;
/// assert_eq!(progress.status, EncoderStatus::Finished);
///
/// let mut compressor = session.into_compressor();
/// assert!(!compressor.compress(b"reused")?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct EncoderSessionOwned<D = PreparedDictionary> {
    core: super::core::session::OwnedSessionCore<D>,
}

impl<D: AsRef<PreparedDictionary> + 'static> EncoderSessionOwned<D> {
    /// Starts a stream that owns `compressor`.
    ///
    /// The caller has already validated the stream configuration and acquired
    /// the encoder, which is what fixes `limit`.
    pub(crate) fn new(
        compressor: Compressor,
        dictionary: Option<D>,
        limit: usize,
        stream: StreamConfig,
    ) -> Self {
        Self {
            core: super::core::session::OwnedSessionCore::new(
                compressor, dictionary, limit, stream,
            ),
        }
    }

    /// Moves the stream forward by one step.
    ///
    /// Behaves exactly as [`EncoderSession::process`]: the same operations,
    /// the same exact `consumed` and `produced` counts, the same statuses, and
    /// a `Finish` that returns `NeedsOutput` must be repeated until it reports
    /// `Finished`, after which further calls are idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidState`] when the stream has already failed,
    /// and propagates whatever the encoder reports. A failed session encodes
    /// nothing further; [`Self::into_compressor`] still returns a usable
    /// compressor.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let compressor = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let mut session = compressor.into_session(Default::default())?;
    /// let mut compressed = Vec::new();
    /// let mut buffer = [0u8; 8];
    /// let mut input = &b"a payload compressed through a tiny buffer"[..];
    /// loop {
    ///     let progress = session.process(input, &mut buffer, Operation::Finish)?;
    ///     input = &input[progress.consumed..];
    ///     compressed.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status == EncoderStatus::Finished {
    ///         break;
    ///     }
    /// }
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.core.process(input, output, operation)
    }

    /// Makes everything accepted so far decodable, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Flush`]. Repeat it while it reports
    /// [`EncoderStatus::NeedsOutput`]; `NeedsInput` means the flush has been
    /// delivered and the stream stays open.
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0u8; 256];
    /// session.process(b"flushed payload", &mut output, Operation::Process)?;
    /// let progress = session.flush(&mut output)?;
    /// assert_eq!(progress.status, EncoderStatus::NeedsInput);
    /// assert!(progress.produced > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
        self.process(&[], output, Operation::Flush)
    }

    /// Terminates the stream, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Finish`], for when every input byte has already been
    /// passed to `process`. Repeat it while it reports
    /// [`EncoderStatus::NeedsOutput`], until [`EncoderStatus::Finished`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut compressed = Vec::new();
    /// let mut output = [0u8; 4];
    /// let mut input = &b"finished without input"[..];
    /// while !input.is_empty() {
    ///     let progress = session.process(input, &mut output, Operation::Process)?;
    ///     input = &input[progress.consumed..];
    ///     compressed.extend_from_slice(&output[..progress.produced]);
    /// }
    /// loop {
    ///     let progress = session.finish(&mut output)?;
    ///     compressed.extend_from_slice(&output[..progress.produced]);
    ///     if progress.status == EncoderStatus::Finished {
    ///         break;
    ///     }
    /// }
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
        self.process(&[], output, Operation::Finish)
    }

    /// Returns whether the stream has been terminated and delivered.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0u8; 256];
    ///
    /// assert!(!session.is_finished());
    /// session.process(b"payload", &mut output, Operation::Finish)?;
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.core.is_finished()
    }

    /// Ends the current stream and starts a new, independent one in place.
    ///
    /// Releases the current stream exactly as [`Self::into_compressor`] does
    /// — finished, unfinished, flushed or failed — and starts the next one
    /// with the same dictionary through the same path as
    /// [`Compressor::into_session`](super::Compressor::into_session). The new
    /// stream's bytes equal a fresh borrowed session's. To change the
    /// dictionary, go through [`Self::into_compressor`] and
    /// [`Compressor::into_session_with_dictionary`](super::Compressor::into_session_with_dictionary).
    ///
    /// # Errors
    ///
    /// As [`Compressor::start`](super::Compressor::start). After an error the
    /// session is failed: [`Self::process`] returns
    /// [`EncodeError::InvalidState`] until a later `reinit` succeeds, and
    /// [`Self::into_compressor`] still returns a usable compressor.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0u8; 256];
    /// session.process(b"abandoned", &mut output, Operation::Process)?;
    ///
    /// session.reinit(Default::default())?;
    /// let progress = session.process(b"fresh", &mut output, Operation::Finish)?;
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reinit(&mut self, stream: StreamConfig) -> Result<(), EncodeError> {
        self.core.reinit(stream)
    }

    /// Ends the stream and returns the compressor, ready for the next one.
    ///
    /// Releases the operation exactly as dropping an [`EncoderSession`] does,
    /// whether the stream finished, was left mid-way, was flushed, or failed:
    /// an unfinished stream is abandoned, and the compressor's retention policy
    /// is applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0u8; 256];
    /// session.process(b"never finished", &mut output, Operation::Process)?;
    ///
    /// // The abandoned stream does not block the next one.
    /// let mut compressor = session.into_compressor();
    /// assert!(!compressor.compress(b"payload")?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_compressor(self) -> Compressor {
        self.core.into_compressor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_size_hints_zero_the_way_the_reference_does() {
        assert_eq!(InputSize::Unknown.hint(), 0);
        assert_eq!(InputSize::Exact(0).hint(), 0);
        assert_eq!(InputSize::Exact(4096).hint(), 4096);
        assert_eq!(InputSize::Exact(u64::MAX).hint(), usize::MAX);
        assert_eq!(InputSize::default(), InputSize::Unknown);
        assert_eq!(InputSize::from(7u64), InputSize::Exact(7));
    }

    #[test]
    fn a_stream_configuration_carries_only_what_one_stream_knows() {
        let stream = StreamConfig::default()
            .with_input_size(InputSize::Exact(10))
            .with_stream_offset(64);
        assert_eq!(stream.input_size(), InputSize::Exact(10));
        assert_eq!(stream.stream_offset(), 64);

        let plain = StreamConfig::from(InputSize::Exact(10));
        assert_eq!(plain.input_size(), InputSize::Exact(10));
        assert_eq!(plain.stream_offset(), 0);
        assert_eq!(
            StreamConfig::default(),
            StreamConfig::from(InputSize::Unknown)
        );
    }

    #[test]
    fn the_operation_and_status_values_are_distinct() {
        assert_eq!(Operation::default(), Operation::Process);
        assert_ne!(Operation::Flush, Operation::Finish);
        assert_ne!(EncoderStatus::NeedsInput, EncoderStatus::NeedsOutput);
        assert_ne!(EncoderStatus::NeedsOutput, EncoderStatus::Finished);
    }
}
