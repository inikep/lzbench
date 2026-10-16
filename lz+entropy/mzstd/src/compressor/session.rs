//! The incremental encoder: one frame, driven a chunk at a time.
//!
//! [`EncoderSession::process`] is the reference's `ZSTD_compressStream2`:
//! the same sequence of inputs, output capacities and operations produces
//! the same bytes. Two things make the bytes depend on that sequence:
//! [`Operation::Flush`] ends the current block early, and an
//! [`Operation::End`] whose remaining input is small enough for the offered
//! output to hold its worst case is compressed straight from the caller's
//! memory.

use core::mem::MaybeUninit;

use super::core::error::CoreError;
use super::core::output::Destination;
use super::core::stream::EndOp;
use super::dictionary::PreparedDictionary;
use super::encoder::Compressor;
use super::error::EncodeError;

/// How much input a frame will carry, when that is known in advance.
///
/// A declared size selects the parameters for that size and is written into
/// the frame header; the frame fails if the input differs, including when the
/// whole input arrives with a first [`Operation::End`].
///
/// # Examples
///
/// ```
/// use mzstd::InputSize;
///
/// assert_eq!(InputSize::default(), InputSize::Unknown);
/// assert_eq!(InputSize::from(4096u64), InputSize::Exact(4096));
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum InputSize {
    /// Not known.
    #[default]
    Unknown,
    /// Exactly this many bytes.
    Exact(u64),
}

impl From<u64> for InputSize {
    fn from(value: u64) -> Self {
        Self::Exact(value)
    }
}

/// What a single frame knows about itself.
///
/// # Examples
///
/// ```
/// use mzstd::{InputSize, StreamConfig};
///
/// let stream = StreamConfig::from(InputSize::Exact(10));
/// assert_eq!(stream.input_size(), InputSize::Exact(10));
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct StreamConfig {
    input_size: InputSize,
}

impl StreamConfig {
    /// Sets how much input the frame will carry.
    #[must_use]
    pub const fn with_input_size(mut self, input_size: InputSize) -> Self {
        self.input_size = input_size;
        self
    }

    /// How much input the frame will carry.
    #[must_use]
    pub const fn input_size(&self) -> InputSize {
        self.input_size
    }
}

impl From<InputSize> for StreamConfig {
    fn from(input_size: InputSize) -> Self {
        Self { input_size }
    }
}

/// What a call to [`EncoderSession::process`] should do (`ZSTD_EndDirective`).
///
/// # Examples
///
/// ```
/// assert_eq!(mzstd::Operation::default(), mzstd::Operation::Continue);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Operation {
    /// Take input; compress whole blocks as they fill.
    #[default]
    Continue,
    /// Make everything accepted so far decodable without ending the frame.
    /// The current block ends early, which costs ratio.
    Flush,
    /// Compress everything left and end the frame.
    End,
}

impl Operation {
    const fn to_core(self) -> EndOp {
        match self {
            Self::Continue => EndOp::Continue,
            Self::Flush => EndOp::Flush,
            Self::End => EndOp::End,
        }
    }
}

/// What a session needs next.
///
/// # Examples
///
/// ```
/// use mzstd::EncoderStatus;
///
/// assert_ne!(EncoderStatus::NeedsInput, EncoderStatus::Finished);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EncoderStatus {
    /// The operation has done all it can; more input or a new operation is
    /// needed.
    NeedsInput,
    /// Compressed bytes are waiting; call again with room for them.
    NeedsOutput,
    /// The frame is complete and delivered.
    Finished,
}

/// What one [`EncoderSession::process`] call did.
///
/// # Examples
///
/// ```
/// use mzstd::{EncoderStatus, Progress};
///
/// let progress = Progress { consumed: 4, produced: 0, status: EncoderStatus::NeedsInput };
/// assert_eq!(progress.consumed, 4);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Progress {
    /// Bytes taken from the input.
    pub consumed: usize,
    /// Bytes written to the output.
    pub produced: usize,
    /// What the session needs next.
    pub status: EncoderStatus,
}

/// Lifecycle of a session.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum State {
    Open,
    Finished,
    Failed,
}

/// The shared state machine of borrowed and owned sessions.
#[derive(Debug)]
struct Core {
    state: State,
    declared: Option<u64>,
}

impl Core {
    fn new(stream: StreamConfig) -> Self {
        Self {
            state: State::Open,
            declared: match stream.input_size() {
                InputSize::Exact(size) => Some(size),
                InputSize::Unknown => None,
            },
        }
    }

    fn process<D: Destination + ?Sized>(
        &mut self,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        input: &[u8],
        output: &mut D,
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        match self.state {
            State::Failed => return Err(EncodeError::InvalidState),
            State::Finished => {
                return Ok(Progress {
                    consumed: 0,
                    produced: 0,
                    status: EncoderStatus::Finished,
                });
            }
            State::Open => {}
        }
        // On a first `End` call the reference replaces the pledged size
        // with the input's length (`ZSTD_CCtx_init_compressStream2`) and
        // writes a frame for that; a session holds the input to its
        // declaration instead, as later calls already are.
        if operation == Operation::End
            && compressor.stream.is_idle()
            && let Some(declared) = self.declared
            && declared != input.len() as u64
        {
            self.state = State::Failed;
            compressor.stream.reset_session();
            return Err(EncodeError::InputSizeMismatch { declared });
        }
        let output_len = output.capacity();
        let result = compressor.stream.compress(
            &mut compressor.engine,
            &compressor.requested,
            dictionary.map(|dictionary| dictionary.cdict()),
            output,
            input,
            operation.to_core(),
        );
        let step = match result {
            Ok(step) => step,
            Err(error) => {
                self.state = State::Failed;
                return Err(match error {
                    CoreError::DstSizeTooSmall => EncodeError::Internal,
                    other => EncodeError::from_core(other, output_len, self.declared),
                });
            }
        };
        let status =
            if operation == Operation::End && step.remaining == 0 && compressor.stream.is_idle() {
                self.state = State::Finished;
                compressor.apply_retention();
                EncoderStatus::Finished
            } else if step.remaining > 0 {
                EncoderStatus::NeedsOutput
            } else {
                EncoderStatus::NeedsInput
            };
        Ok(Progress {
            consumed: step.consumed,
            produced: step.produced,
            status,
        })
    }
}

/// The dictionary a borrowed session compresses with.
const fn borrowed_dictionary<'d>(
    dictionary: &Option<&'d PreparedDictionary>,
) -> Option<&'d PreparedDictionary> {
    *dictionary
}

/// The dictionary an owned session compresses with.
fn owned_dictionary<D: AsRef<PreparedDictionary>>(
    dictionary: &Option<D>,
) -> Option<&PreparedDictionary> {
    dictionary.as_ref().map(AsRef::as_ref)
}

macro_rules! session_methods {
    ($compressor:ident, $dictionary:expr) => {
        /// Moves the frame forward by one step: takes what it can from
        /// `input`, writes what it can into `output`, and reports exactly how
        /// much of each it moved and what it needs next.
        ///
        /// An [`Operation::End`] that returns
        /// [`EncoderStatus::NeedsOutput`] must be repeated until it reports
        /// [`EncoderStatus::Finished`]; afterwards calls are idempotent.
        ///
        /// # Errors
        ///
        /// [`EncodeError::InputSizeMismatch`] when the input contradicts a
        /// declared size: when more arrives than was declared, when the
        /// frame ends short of it, or when a first [`Operation::End`] call
        /// carries any other length. [`EncodeError::InvalidState`] after a
        /// failure.
        /// A failed session encodes nothing further.
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::{Compressor, EncoderStatus, Operation};
        ///
        /// let mut encoder = Compressor::new(Default::default())?;
        /// let mut session = encoder.start(Default::default());
        /// let mut out = [0u8; 64];
        /// let progress = session.process(b"", &mut out, Operation::End)?;
        /// assert_eq!(progress.status, EncoderStatus::Finished);
        /// assert!(progress.produced > 0);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn process(
            &mut self,
            input: &[u8],
            output: &mut [u8],
            operation: Operation,
        ) -> Result<Progress, EncodeError> {
            let compressor = &mut *self.$compressor;
            let dictionary = $dictionary(&self.dictionary);
            self.core
                .process(compressor, dictionary, input, output, operation)
        }

        /// [`Self::process`] into uninitialized memory, such as
        /// [`Vec::spare_capacity_mut`](alloc::vec::Vec::spare_capacity_mut),
        /// without a temporary buffer.
        ///
        /// It produces the same bytes, `consumed` and `produced` counts,
        /// statuses and errors as [`Self::process`], and calls of the two
        /// may be mixed on one frame. On success exactly
        /// `output[..produced]` holds compressed bytes. As with
        /// [`Self::process`], the encoder may also have written scratch
        /// bytes after them, so treat only `output[..produced]` as
        /// initialized, and none of `output` after an error.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::{Compressor, EncoderStatus, Operation};
        ///
        /// let mut encoder = Compressor::new(Default::default())?;
        /// let mut session = encoder.start(Default::default());
        /// let mut out: Vec<u8> = Vec::with_capacity(64);
        /// let progress = session.process_uninit(b"data", out.spare_capacity_mut(), Operation::End)?;
        /// // SAFETY: `process_uninit` initialized the first `produced` bytes.
        /// unsafe { out.set_len(progress.produced) };
        /// assert_eq!(progress.status, EncoderStatus::Finished);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn process_uninit(
            &mut self,
            input: &[u8],
            output: &mut [MaybeUninit<u8>],
            operation: Operation,
        ) -> Result<Progress, EncodeError> {
            let compressor = &mut *self.$compressor;
            let dictionary = $dictionary(&self.dictionary);
            self.core
                .process(compressor, dictionary, input, output, operation)
        }

        /// Makes everything accepted so far decodable, without taking input.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::{Compressor, EncoderStatus, Operation};
        ///
        /// let mut encoder = Compressor::new(Default::default())?;
        /// let mut session = encoder.start(Default::default());
        /// let mut out = [0u8; 256];
        /// session.process(b"flushed payload", &mut out, Operation::Continue)?;
        /// let progress = session.flush(&mut out)?;
        /// assert_eq!(progress.status, EncoderStatus::NeedsInput);
        /// assert!(progress.produced > 0);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn flush(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
            self.process(&[], output, Operation::Flush)
        }

        /// Ends the frame, without taking input.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::{Compressor, EncoderStatus, Operation};
        ///
        /// let mut encoder = Compressor::new(Default::default())?;
        /// let mut session = encoder.start(Default::default());
        /// let mut out = [0u8; 256];
        /// session.process(b"payload", &mut out, Operation::Continue)?;
        /// assert_eq!(session.finish(&mut out)?.status, EncoderStatus::Finished);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn finish(&mut self, output: &mut [u8]) -> Result<Progress, EncodeError> {
            self.process(&[], output, Operation::End)
        }

        /// Whether the frame has been completed and delivered.
        #[must_use]
        pub fn is_finished(&self) -> bool {
            self.core.state == State::Finished
        }
    };
}

/// One incremental frame borrowing its compressor.
///
/// Dropping a session before it finishes abandons the frame; the compressor
/// is left ready for the next one either way.
///
/// # Examples
///
/// ```
/// use mzstd::{Compressor, EncoderStatus, InputSize, Operation};
///
/// let mut encoder = Compressor::new(Default::default())?;
/// let payload = b"a payload compressed one chunk at a time".repeat(10);
/// let mut compressed = Vec::new();
/// let mut buffer = [0u8; 64];
/// let mut input = payload.as_slice();
/// {
///     let mut session = encoder.start(InputSize::Exact(payload.len() as u64).into());
///     loop {
///         let progress = session.process(input, &mut buffer, Operation::End)?;
///         input = &input[progress.consumed..];
///         compressed.extend_from_slice(&buffer[..progress.produced]);
///         if progress.status == EncoderStatus::Finished {
///             break;
///         }
///     }
/// }
/// assert!(!compressed.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct EncoderSession<'c> {
    compressor: &'c mut Compressor,
    dictionary: Option<&'c PreparedDictionary>,
    core: Core,
}

impl<'c> EncoderSession<'c> {
    pub(crate) fn new(compressor: &'c mut Compressor, stream: StreamConfig) -> Self {
        Self::with_dictionary(compressor, None, stream)
    }

    /// A session compressing with `dictionary`, which the caller checked
    /// serves the compressor.
    pub(crate) fn with_dictionary(
        compressor: &'c mut Compressor,
        dictionary: Option<&'c PreparedDictionary>,
        stream: StreamConfig,
    ) -> Self {
        Self {
            compressor,
            dictionary,
            core: Core::new(stream),
        }
    }

    session_methods!(compressor, borrowed_dictionary);
}

/// One incremental frame that owns its compressor.
///
/// # Examples
///
/// ```
/// use mzstd::{Compressor, EncoderStatus, Operation};
///
/// let mut session = Compressor::new(Default::default())?.into_session(Default::default());
/// let mut out = [0u8; 64];
/// assert_eq!(session.process(b"x", &mut out, Operation::End)?.status, EncoderStatus::Finished);
/// let compressor = session.into_compressor();
/// assert!(compressor.retained_bytes() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A session started with
/// [`Compressor::into_session_with_dictionary`] also owns its dictionary,
/// of any type `D` that borrows as a [`PreparedDictionary`] (the dictionary
/// itself, a `Box` or an `Arc` of one).
#[derive(Debug)]
pub struct EncoderSessionOwned<D = PreparedDictionary> {
    compressor: alloc::boxed::Box<Compressor>,
    dictionary: Option<D>,
    core: Core,
}

impl EncoderSessionOwned {
    pub(crate) fn new(compressor: Compressor, stream: StreamConfig) -> Self {
        Self::with_dictionary(compressor, None, stream)
    }
}

impl<D: AsRef<PreparedDictionary>> EncoderSessionOwned<D> {
    /// A session compressing with `dictionary`, which the caller checked
    /// serves the compressor.
    pub(crate) fn with_dictionary(
        compressor: Compressor,
        dictionary: Option<D>,
        stream: StreamConfig,
    ) -> Self {
        Self {
            compressor: alloc::boxed::Box::new(compressor),
            dictionary,
            core: Core::new(stream),
        }
    }

    session_methods!(compressor, owned_dictionary);

    /// Starts a new frame on the same compressor.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Compressor, EncoderStatus, Operation};
    ///
    /// let mut session = Compressor::new(Default::default())?.into_session(Default::default());
    /// let mut out = [0u8; 64];
    /// session.process(b"one", &mut out, Operation::End)?;
    /// session.reinit(Default::default());
    /// assert!(!session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reinit(&mut self, stream: StreamConfig) {
        self.compressor.begin_stream(stream);
        self.core = Core::new(stream);
    }

    /// Returns the compressor and the dictionary, if any, abandoning an
    /// unfinished frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::Compressor;
    ///
    /// let dictionary = DictionaryBuilder::new(b"history").build()?;
    /// let session = Compressor::default().into_session_with_dictionary(dictionary, Default::default())?;
    /// let (_compressor, dictionary) = session.into_parts();
    /// assert_eq!(dictionary.map(|dictionary| dictionary.as_bytes().len()), Some(7));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (Compressor, Option<D>) {
        let mut compressor = *self.compressor;
        compressor.stream.reset_session();
        (compressor, self.dictionary)
    }

    /// Returns the compressor, abandoning an unfinished frame.
    #[must_use]
    pub fn into_compressor(self) -> Compressor {
        let mut compressor = *self.compressor;
        compressor.stream.reset_session();
        compressor
    }
}
