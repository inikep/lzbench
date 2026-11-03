use ::core::mem::MaybeUninit;

use super::core::{DictView, Stream};
use super::{DecodeError, Decompressor};
use crate::dictionary::{DecodeDictionary, DictionaryRef};

/// Whether more input may follow a call.
///
/// Use [`Self::Process`] while more input may arrive. Once the input is
/// known to end, use [`Self::Finish`] on every remaining call, each time
/// passing exactly the input not yet consumed. See
/// [`DecoderSession::process`] for a complete loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DecodeOperation {
    /// More input may follow; an empty slice does not end the input.
    #[default]
    Process,
    /// This is all the remaining input.
    Finish,
}

/// Why an incremental call stopped.
///
/// Handle the counts in [`DecodeProgress`] before acting on the status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderStatus {
    /// All offered input was consumed and more is needed.
    NeedsInput,
    /// Decoded output is waiting for output space.
    NeedsOutput,
    /// The operation is complete: every frame was decoded and validated,
    /// and all output delivered.
    Finished,
}

/// Bytes accepted and delivered by one successful call.
///
/// Advance the input by `consumed` and use `output[..produced]`. The counts
/// are per call, not cumulative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeProgress {
    /// Length of the accepted input prefix.
    pub consumed: usize,
    /// Length of the written output prefix.
    pub produced: usize,
    /// Why the call stopped.
    pub status: DecoderStatus,
}

/// A failed incremental call, with that call's exact progress.
///
/// Output counted by `produced` was written before the failure and is not
/// rolled back; it is a partial, unvalidated result. The session stays
/// failed: later calls report [`DecodeError::InvalidState`] until the
/// decoder starts a new operation.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeError, DecodeOperation, Decompressor};
/// let mut decoder = Decompressor::default();
/// let mut session = decoder.start();
/// // The input ends before any frame.
/// let failure = session.process(&[], &mut [], DecodeOperation::Finish).unwrap_err();
/// assert_eq!((failure.consumed, failure.produced), (0, 0));
/// assert!(matches!(failure.into_error(), DecodeError::UnexpectedEndOfInput));
/// ```
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct DecodeFailure {
    /// The underlying error.
    #[source]
    pub error: DecodeError,
    /// Length of the input prefix the call accepted.
    pub consumed: usize,
    /// Length of the output prefix the call wrote.
    pub produced: usize,
}

impl DecodeFailure {
    /// The error, once the progress has been handled.
    #[must_use]
    pub fn into_error(self) -> DecodeError {
        self.error
    }
}

impl From<DecodeFailure> for DecodeError {
    fn from(failure: DecodeFailure) -> Self {
        failure.error
    }
}

/// An incremental decoding operation borrowing its [`Decompressor`].
///
/// Created by [`Decompressor::start`], or by
/// [`Decompressor::start_with_dictionary`], which also borrows the
/// dictionary every frame starts from. Input and output move through
/// caller-owned slices of any size; the session keeps its own buffers for
/// the headers and block being assembled and for regenerated output not yet
/// delivered. The decoder's [`RetentionPolicy`](crate::RetentionPolicy)
/// is applied when the operation finishes or fails, and again when the next
/// operation starts. Dropping the session abandons the operation.
#[derive(Debug)]
pub struct DecoderSession<'d> {
    decoder: &'d mut Decompressor,
    dictionary: Option<DictView<'d>>,
}

impl<'d> DecoderSession<'d> {
    pub(super) fn new(decoder: &'d mut Decompressor, dictionary: Option<DictView<'d>>) -> Self {
        decoder.apply_retention();
        let config = decoder.config;
        decoder.stream.reset(config);
        Self {
            decoder,
            dictionary,
        }
    }
}

/// The borrowed view of an owned session's dictionary.
fn dictionary_view<D: AsRef<DecodeDictionary>>(dictionary: Option<&D>) -> Option<DictView<'_>> {
    dictionary.map(|dictionary| DictionaryRef::from(dictionary.as_ref()).view())
}

/// Runs one call on `decoder`'s stream and applies its retention policy
/// once the operation has ended, successfully or not.
fn process_on(
    decoder: &mut Decompressor,
    call: impl FnOnce(&mut Stream) -> Result<DecodeProgress, DecodeFailure>,
) -> Result<DecodeProgress, DecodeFailure> {
    let result = call(&mut decoder.stream);
    if result
        .as_ref()
        .map_or(true, |progress| progress.status == DecoderStatus::Finished)
    {
        decoder.apply_retention();
    }
    result
}

impl DecoderSession<'_> {
    /// Decodes as much of `input` into `output` as it can.
    ///
    /// Deliver `output[..progress.produced]` and advance `input` by
    /// `progress.consumed` after every call. With
    /// [`DecodeOperation::Process`] an empty `input` only drains pending
    /// output. After the first [`DecodeOperation::Finish`], keep using
    /// `Finish` with exactly the unconsumed suffix until the status is
    /// [`DecoderStatus::Finished`].
    ///
    /// In [`super::FrameMode::Single`] the session finishes right after its
    /// first frame, Zstandard or skippable, and leaves any later input
    /// unconsumed. In
    /// [`super::FrameMode::Concatenated`] a frame boundary reports
    /// `NeedsInput` until `Finish` confirms that the input ends there. A
    /// finished session reports zero progress and `Finished` again.
    ///
    /// # Errors
    ///
    /// Returns a [`DecodeFailure`] with the call's progress for invalid
    /// data, exceeded limits, failed checks, and input that ends inside a
    /// frame ([`DecodeError::UnexpectedEndOfInput`]). Changing the operation
    /// or the final input after `Finish`, or calling a failed session,
    /// reports [`DecodeError::InvalidState`]. Failures are terminal.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
    /// // "hello" as one raw block in a single-segment frame.
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
    /// let mut decoder = Decompressor::default();
    /// let mut session = decoder.start();
    /// let mut remaining = &frame[..];
    /// let mut decoded = Vec::new();
    /// loop {
    ///     let mut buffer = [0u8; 2];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Finish)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status == DecoderStatus::Finished {
    ///         break;
    ///     }
    /// }
    /// assert_eq!(decoded, b"hello");
    /// assert_eq!(session.total_in(), frame.len() as u64);
    /// assert_eq!(session.total_out(), 5);
    /// assert_eq!(session.frames_decoded(), 1);
    /// assert_eq!(session.window_size(), Some(5));
    /// # Ok::<(), mzstd::DecodeFailure>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let dictionary = self.dictionary;
        process_on(self.decoder, |stream| {
            stream.process(input, output, operation, dictionary)
        })
    }

    /// [`Self::process`] into uninitialized memory, such as
    /// [`Vec::spare_capacity_mut`](alloc::vec::Vec::spare_capacity_mut),
    /// without zeroing it first.
    ///
    /// It produces the same bytes, `consumed` and `produced` counts,
    /// statuses and errors as [`Self::process`], and calls of the two may
    /// be mixed on one operation. Exactly `output[..produced]` is written,
    /// on success and in a [`DecodeFailure`] alike; no byte after it is
    /// touched.
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
    /// let mut decoder = Decompressor::default();
    /// let mut session = decoder.start();
    /// let mut decoded: Vec<u8> = Vec::with_capacity(16);
    /// let progress =
    ///     session.process_uninit(&frame, decoded.spare_capacity_mut(), DecodeOperation::Finish)?;
    /// // SAFETY: `process_uninit` initialized the first `produced` bytes.
    /// unsafe { decoded.set_len(progress.produced) };
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// assert_eq!(decoded, b"hello");
    /// # Ok::<(), mzstd::DecodeFailure>(())
    /// ```
    pub fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let dictionary = self.dictionary;
        process_on(self.decoder, |stream| {
            stream.process_uninit(input, output, operation, dictionary)
        })
    }

    /// Delivers pending output without offering input or ending it.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Process`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`]; after a `Finish` this reports
    /// [`DecodeError::InvalidState`].
    pub fn flush(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }

    /// Ends the input with nothing left to offer, delivering pending output.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Finish`]; repeat it while it reports
    /// [`DecoderStatus::NeedsOutput`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`]: stopping inside a frame reports
    /// [`DecodeError::UnexpectedEndOfInput`].
    pub fn finish(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }

    /// Whether the operation completed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.decoder.stream.is_finished()
    }

    /// Compressed bytes accepted, including by failing calls.
    #[must_use]
    pub fn total_in(&self) -> u64 {
        self.decoder.stream.total_in()
    }

    /// Decompressed bytes delivered, including by failing calls.
    #[must_use]
    pub fn total_out(&self) -> u64 {
        self.decoder.stream.total_out()
    }

    /// Zstandard frames completed and validated; skippable frames are not
    /// counted.
    #[must_use]
    pub fn frames_decoded(&self) -> u64 {
        self.decoder.stream.frames_decoded()
    }

    /// The window size of the most recent frame header, if any was read.
    #[must_use]
    pub fn window_size(&self) -> Option<u64> {
        self.decoder.stream.window_size()
    }
}

/// An incremental decoding operation that owns its [`Decompressor`].
///
/// The same state machine as [`DecoderSession`], without a borrow: it suits
/// wrappers that store the codec state in a struct between calls. Created by
/// [`Decompressor::into_session`]; [`Self::into_decompressor`] hands the
/// decoder back.
///
/// [`Decompressor::into_session_with_dictionary`] also hands it a
/// dictionary, as any `D: AsRef<DecodeDictionary>`: the dictionary itself,
/// or an `Arc` of it to share one between sessions. Without one, `D` keeps
/// its default and no dictionary is ever constructed.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
/// let mut session = Decompressor::default().into_session();
/// let mut output = [0u8; 8];
/// let progress = session.process(&frame, &mut output, DecodeOperation::Finish)?;
/// assert_eq!(progress.status, DecoderStatus::Finished);
/// assert_eq!(&output[..progress.produced], b"ok");
///
/// session.reinit();
/// let progress = session.process(&frame, &mut output, DecodeOperation::Finish)?;
/// assert_eq!(&output[..progress.produced], b"ok");
/// let mut decoder = session.into_decompressor();
/// assert_eq!(decoder.decompress(&frame)?, b"ok");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DecoderSessionOwned<D = DecodeDictionary> {
    decoder: Decompressor,
    dictionary: Option<D>,
}

impl<D: AsRef<DecodeDictionary>> DecoderSessionOwned<D> {
    pub(super) fn new(mut decoder: Decompressor, dictionary: Option<D>) -> Self {
        decoder.apply_retention();
        let config = decoder.config;
        decoder.stream.reset(config);
        Self {
            decoder,
            dictionary,
        }
    }

    /// Decodes as much of `input` into `output` as it can, exactly as
    /// [`DecoderSession::process`].
    ///
    /// # Errors
    ///
    /// As [`DecoderSession::process`].
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let dictionary = dictionary_view(self.dictionary.as_ref());
        process_on(&mut self.decoder, |stream| {
            stream.process(input, output, operation, dictionary)
        })
    }

    /// Decodes into uninitialized memory, exactly as
    /// [`DecoderSession::process_uninit`].
    ///
    /// # Errors
    ///
    /// As [`DecoderSession::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut session = Decompressor::default().into_session();
    /// let mut output = [MaybeUninit::<u8>::uninit(); 8];
    /// let progress = session.process_uninit(&frame, &mut output, DecodeOperation::Finish)?;
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// // SAFETY: `process_uninit` initialized the first `produced` bytes.
    /// let decoded = output[..progress.produced].iter().map(|byte| unsafe { byte.assume_init() });
    /// assert!(decoded.eq(*b"ok"));
    /// # Ok::<(), mzstd::DecodeFailure>(())
    /// ```
    pub fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let dictionary = dictionary_view(self.dictionary.as_ref());
        process_on(&mut self.decoder, |stream| {
            stream.process_uninit(input, output, operation, dictionary)
        })
    }

    /// Delivers pending output, as [`DecoderSession::flush`].
    ///
    /// # Errors
    ///
    /// As [`DecoderSession::process`].
    pub fn flush(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }

    /// Ends the input, as [`DecoderSession::finish`].
    ///
    /// # Errors
    ///
    /// As [`DecoderSession::process`].
    pub fn finish(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }

    /// Whether the operation completed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.decoder.stream.is_finished()
    }

    /// Compressed bytes accepted, including by failing calls.
    #[must_use]
    pub fn total_in(&self) -> u64 {
        self.decoder.stream.total_in()
    }

    /// Decompressed bytes delivered, including by failing calls.
    #[must_use]
    pub fn total_out(&self) -> u64 {
        self.decoder.stream.total_out()
    }

    /// Zstandard frames completed and validated.
    #[must_use]
    pub fn frames_decoded(&self) -> u64 {
        self.decoder.stream.frames_decoded()
    }

    /// The window size of the most recent frame header, if any was read.
    #[must_use]
    pub fn window_size(&self) -> Option<u64> {
        self.decoder.stream.window_size()
    }

    /// Abandons the current operation, whatever its state, and starts a new
    /// one in place, applying the decoder's retention policy in between.
    pub fn reinit(&mut self) {
        self.decoder.apply_retention();
        let config = self.decoder.config;
        self.decoder.stream.reset(config);
    }

    /// Ends the operation and returns the decoder, ready for reuse, after
    /// applying its retention policy.
    #[must_use]
    pub fn into_decompressor(mut self) -> Decompressor {
        self.decoder.apply_retention();
        self.decoder
    }
}
