use super::{
    DecodeError, DecodeStreamConfig, Decompressor,
    core::{Delivery, OperationState},
};
use crate::Window;
use crate::dictionary::{DecodeDictionary, DictionaryRef};

/// Whether more input may follow this call.
///
/// Start with [`Self::Process`] while more chunks may arrive. Once EOF is known,
/// use [`Self::Finish`] on every remaining call, advancing input by the reported
/// consumed count. See [`DecoderSession::process`] for a complete loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DecodeOperation {
    /// More input may follow; an empty slice is not EOF.
    #[default]
    Process,
    /// This is the final input; retries must pass the exact remaining suffix.
    Finish,
}
/// Reason an incremental decoding call stopped.
///
/// Always handle the counts in [`DecodeProgress`] before acting on the status.
/// `NeedsInput` requires more input or an EOF declaration; `NeedsOutput`
/// requires fresh output space, even if all offered input was consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderStatus {
    /// All offered input was consumed and additional input is required.
    NeedsInput,
    /// The next payload byte needs output space.
    NeedsOutput,
    /// The operation is complete and its output size has been validated.
    Finished,
}
/// Bytes accepted and delivered during one successful call.
///
/// Advance input by `consumed` and use only `output[..produced]`. These counts
/// are per call, not cumulative. See [`DecoderSession::process`] for an example.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeProgress {
    /// Accepted prefix length of this call's input.
    pub consumed: usize,
    /// Written prefix length of this call's output.
    pub produced: usize,
    /// Reason decoding stopped.
    pub status: DecoderStatus,
}
/// Terminal codec failure with exact progress for the failing call.
///
/// Output reported by `produced` has already been written and is not rolled
/// back. It is only a partial result; the operation has failed validation.
/// Drop the session before starting another operation on the same decoder.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeError, DecodeOperation, DecoderConfig, Decompressor};
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// let mut session = decoder.start(Default::default())?;
/// // Physical EOF without even one member is truncated input.
/// let failure = session.process(&[], &mut [], DecodeOperation::Finish).unwrap_err();
/// assert_eq!((failure.consumed, failure.produced), (0, 0));
/// assert!(matches!(failure.into_error(), DecodeError::UnexpectedEndOfInput));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct DecodeFailure {
    /// Underlying typed failure.
    #[source]
    pub error: DecodeError,
    /// Accepted input prefix, excluding all unaccepted suffix bytes.
    pub consumed: usize,
    /// Payload prefix delivered before the failure.
    pub produced: usize,
}
impl DecodeFailure {
    /// Extracts the codec error when progress has already been handled.
    pub fn into_error(self) -> DecodeError {
        self.error
    }
}

/// Exclusive incremental decoding operation. Drop clears per-stream state.
///
/// The decoder and any external dictionary remain borrowed for the session's
/// lifetime. Dropping the session permits decoder reuse and applies its
/// retention policy, even after failure or incomplete input. Forgetting it with
/// [`core::mem::forget`] requires [`Decompressor::recover`] before reuse.
/// See [`Self::process`] for a loop with a small output buffer, and
/// [`DecoderSessionOwned`] for a session that owns its decoder instead.
#[derive(Debug)]
pub struct DecoderSession<'d, 'dict> {
    decoder: &'d mut Decompressor,
    dictionary: Option<DictionaryRef<'dict>>,
    operation: OperationState,
}

impl<'d, 'dict> DecoderSession<'d, 'dict> {
    /// Output the current meta-block still declares, for growing a destination.
    pub(super) fn declared_remaining(&self) -> usize {
        usize::try_from(self.decoder.workspace.declared_remaining()).unwrap_or(usize::MAX)
    }

    pub(super) fn start(
        decoder: &'d mut Decompressor,
        stream: DecodeStreamConfig,
        dictionary: Option<DictionaryRef<'dict>>,
    ) -> Result<Self, DecodeError> {
        let operation = OperationState::start(decoder, stream)?;
        Ok(Self {
            decoder,
            dictionary,
            operation,
        })
    }
}

impl DecoderSession<'_, '_> {
    /// Decodes available bytes without retaining caller slices.
    ///
    /// Deliver `output[..progress.produced]` and advance `input` by
    /// `progress.consumed` after each call. An empty input with `Process` does
    /// not declare EOF. After the first `Finish`, keep using `Finish` with
    /// exactly the unconsumed suffix, including an empty suffix when only output
    /// remains. Input bytes themselves must remain unchanged between retries.
    ///
    /// In single-member mode, `Finished` can leave a protocol suffix unconsumed.
    /// Concatenated mode needs `Finish` to confirm the final member boundary.
    /// A completed session returns zero counts and `Finished` on further calls.
    ///
    /// # Errors
    /// Returns terminal format/resource errors and exact call progress in
    /// [`DecodeFailure`]. Truncated final input yields
    /// [`DecodeError::UnexpectedEndOfInput`]. Changing the operation or final
    /// suffix length after `Finish`, or using a failed session, yields
    /// [`DecodeError::InvalidState`]. A failure cannot be retried in this session.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderConfig, DecoderStatus, Decompressor, OutputSize};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut session = decoder.start(OutputSize::Exact(5).into())?;
    /// assert_eq!(session.window(), None); // No header accepted yet.
    /// let mut remaining = compressed.as_slice();
    /// let mut decoded = Vec::new();
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Finish)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status == DecoderStatus::Finished {
    ///         break;
    ///     }
    ///     assert_eq!(progress.status, DecoderStatus::NeedsOutput);
    /// }
    /// assert_eq!(decoded, b"hello");
    /// assert!(remaining.is_empty());
    /// assert!(session.is_finished());
    /// assert_eq!(session.total_in(), compressed.len() as u64);
    /// assert_eq!(session.total_out(), 5);
    /// assert_eq!(session.members_decoded(), 1);
    /// assert!(session.window().is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        self.operation.process(
            self.decoder,
            self.dictionary,
            input,
            output,
            operation,
            Delivery::Slice,
        )
    }

    /// Delivers output for input already accepted, without declaring EOF.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Process`]. Use it to drain a
    /// [`DecoderStatus::NeedsOutput`] into fresh space; it never declares the
    /// input complete. After a `Finish`, only `Finish` calls are valid.
    ///
    /// # Errors
    /// As [`Self::process`]; after a `Finish` this reports
    /// [`DecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(Default::default())?;
    /// let mut session = decoder.start(Default::default())?;
    /// let mut decoded = Vec::new();
    /// let mut buffer = [0; 2];
    /// let mut remaining = &compressed[..];
    /// while !remaining.is_empty() {
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    /// }
    /// // Every input byte is accepted; drain whatever it still produces.
    /// loop {
    ///     let progress = session.flush(&mut buffer)?;
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status != DecoderStatus::NeedsOutput {
    ///         break;
    ///     }
    /// }
    /// assert_eq!(decoded, b"hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }

    /// Declares EOF with no input left, and delivers the remaining output.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Finish`], for when every input byte has already been
    /// accepted. Repeat it while it reports [`DecoderStatus::NeedsOutput`],
    /// until [`DecoderStatus::Finished`].
    ///
    /// # Errors
    /// As [`Self::process`]. Stopping mid-member reports
    /// [`DecodeError::UnexpectedEndOfInput`], and an unconsumed suffix left
    /// by an earlier `Finish` reports [`DecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(Default::default())?;
    /// let mut session = decoder.start(Default::default())?;
    /// let progress = session.process(&compressed, &mut [0; 16], DecodeOperation::Process)?;
    /// assert_eq!(progress.consumed, compressed.len());
    /// assert_eq!(session.finish(&mut [])?.status, DecoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }

    /// Collects at most one history window without a second output allocation.
    pub(super) fn collect(&mut self, input: &[u8]) -> Result<DecodeProgress, DecodeFailure> {
        let capacity = self.operation.window().map_or(0, |window| {
            usize::try_from(1u64 << window.bits()).unwrap_or(usize::MAX)
        });
        self.operation.process(
            self.decoder,
            self.dictionary,
            input,
            &mut [],
            DecodeOperation::Finish,
            Delivery::Collect(capacity),
        )
    }

    /// One `Finish` call that decodes the operation's first member straight
    /// into `output`, which also serves as its history. Progress, statuses and
    /// errors are those of [`Self::process`]; only the one-shot slice API uses
    /// it, because the history of a paused member is not kept for a later call.
    pub(super) fn finish_linear(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<DecodeProgress, DecodeFailure> {
        self.operation.process(
            self.decoder,
            self.dictionary,
            input,
            output,
            DecodeOperation::Finish,
            Delivery::Linear,
        )
    }

    pub(super) fn take_collected(&mut self) -> alloc::vec::Vec<u8> {
        self.decoder.workspace.take_collected()
    }

    pub(super) fn collected(&self) -> &[u8] {
        self.decoder.workspace.collected()
    }

    /// Whether all members and the exact-size contract were validated.
    pub const fn is_finished(&self) -> bool {
        self.operation.is_finished()
    }
    /// Total accepted compressed bytes, including failing calls.
    pub const fn total_in(&self) -> u64 {
        self.operation.total_in()
    }
    /// Total payload bytes delivered, including failing calls.
    pub const fn total_out(&self) -> u64 {
        self.operation.total_out()
    }
    /// Number of validated members whose output has been delivered.
    pub const fn members_decoded(&self) -> u64 {
        self.operation.members_decoded()
    }
    /// Most recently accepted window header.
    ///
    /// Returns `None` before the first header is accepted. Between concatenated
    /// members, retains the preceding member's window until a new one is accepted.
    pub const fn window(&self) -> Option<Window> {
        self.operation.window()
    }
}

impl Drop for DecoderSession<'_, '_> {
    fn drop(&mut self) {
        self.operation.release(self.decoder);
    }
}

/// Incremental decoding operation that owns its decoder.
///
/// [`DecoderSession`] borrows its [`Decompressor`] with `&mut` for as long as
/// it lives. `DecoderSessionOwned` consumes the decoder instead, so the stream
/// state is one ordinary owned value with no lifetime tied to the decoder. That
/// suits wrappers which have to store the codec state themselves, such as
/// adapters that keep it in a struct between calls. It is still the same
/// synchronous, caller-driven API: it is not an async interface, and it keeps
/// no references into its own fields.
///
/// The session has no lifetime. An external dictionary, when one is attached,
/// is owned too, as any `D: AsRef<DecodeDictionary> + 'static`: an
/// `Arc<DecodeDictionary>` to share one without copying it, a
/// `&'static DecodeDictionary`, or the dictionary itself. Without one, `D`
/// stays at its default and is never constructed.
///
/// Created by [`Decompressor::into_session`] or
/// [`Decompressor::into_session_with_dictionary`]. [`Self::process`] runs the
/// same state machine as [`DecoderSession::process`], so the same calls decode
/// the same bytes with the same counts, statuses and errors.
/// [`Self::into_decompressor`] ends the operation the way dropping a borrowed
/// session does and hands the decoder back ready for reuse. Dropping the owned
/// session instead drops the decoder with it.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
/// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
///
/// let decoder = Decompressor::new(Default::default())?;
/// let mut session = decoder.into_session(Default::default())?;
/// let mut output = [0u8; 16];
///
/// let progress = session.process(&compressed, &mut output, DecodeOperation::Finish)?;
/// assert_eq!(progress.status, DecoderStatus::Finished);
/// assert_eq!(&output[..progress.produced], b"hello");
///
/// let mut decoder = session.into_decompressor();
/// assert_eq!(decoder.decompress(&compressed)?, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DecoderSessionOwned<D = DecodeDictionary> {
    decoder: Decompressor,
    dictionary: Option<D>,
    operation: OperationState,
}

impl<D: AsRef<DecodeDictionary> + 'static> DecoderSessionOwned<D> {
    pub(super) fn start(
        mut decoder: Decompressor,
        stream: DecodeStreamConfig,
        dictionary: Option<D>,
    ) -> Result<Self, DecodeError> {
        let operation = OperationState::start(&mut decoder, stream)?;
        Ok(Self {
            decoder,
            dictionary,
            operation,
        })
    }

    /// Decodes available bytes without retaining caller slices.
    ///
    /// Behaves exactly as [`DecoderSession::process`], including the
    /// `Finish` suffix contract, concatenated members, exact output sizes,
    /// limits and the exact progress carried by a failure.
    ///
    /// # Errors
    /// As [`DecoderSession::process`]. A failure is terminal for this session;
    /// [`Self::into_decompressor`] still returns a reusable decoder.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor, OutputSize};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut session = Decompressor::new(Default::default())?
    ///     .into_session(OutputSize::Exact(5).into())?;
    /// let mut remaining = compressed.as_slice();
    /// let mut decoded = Vec::new();
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Finish)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status == DecoderStatus::Finished {
    ///         break;
    ///     }
    /// }
    /// assert_eq!(decoded, b"hello");
    /// assert_eq!(session.total_in(), compressed.len() as u64);
    /// assert_eq!(session.total_out(), 5);
    /// assert_eq!(session.members_decoded(), 1);
    /// assert!(session.window().is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let dictionary = self
            .dictionary
            .as_ref()
            .map(|dictionary| DictionaryRef::from(dictionary.as_ref()));
        self.operation.process(
            &mut self.decoder,
            dictionary,
            input,
            output,
            operation,
            Delivery::Slice,
        )
    }

    /// Delivers output for input already accepted, without declaring EOF.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Process`]. Use it to drain a
    /// [`DecoderStatus::NeedsOutput`] into fresh space; it never declares the
    /// input complete. After a `Finish`, only `Finish` calls are valid.
    ///
    /// # Errors
    /// As [`Self::process`]; after a `Finish` this reports
    /// [`DecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut session = Decompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut decoded = Vec::new();
    /// let mut buffer = [0; 2];
    /// let mut remaining = &compressed[..];
    /// while !remaining.is_empty() {
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    /// }
    /// // Every input byte is accepted; drain whatever it still produces.
    /// loop {
    ///     let progress = session.flush(&mut buffer)?;
    ///     decoded.extend_from_slice(&buffer[..progress.produced]);
    ///     if progress.status != DecoderStatus::NeedsOutput {
    ///         break;
    ///     }
    /// }
    /// assert_eq!(decoded, b"hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }

    /// Declares EOF with no input left, and delivers the remaining output.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Finish`], for when every input byte has already been
    /// accepted. Repeat it while it reports [`DecoderStatus::NeedsOutput`],
    /// until [`DecoderStatus::Finished`].
    ///
    /// # Errors
    /// As [`Self::process`]. Stopping mid-member reports
    /// [`DecodeError::UnexpectedEndOfInput`], and an unconsumed suffix left
    /// by an earlier `Finish` reports [`DecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut session = Decompressor::new(Default::default())?.into_session(Default::default())?;
    /// let progress = session.process(&compressed, &mut [0; 16], DecodeOperation::Process)?;
    /// assert_eq!(progress.consumed, compressed.len());
    /// assert_eq!(session.finish(&mut [])?.status, DecoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(&mut self, output: &mut [u8]) -> Result<DecodeProgress, DecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }

    /// Whether all members and the exact-size contract were validated.
    pub const fn is_finished(&self) -> bool {
        self.operation.is_finished()
    }
    /// Total accepted compressed bytes, including failing calls.
    pub const fn total_in(&self) -> u64 {
        self.operation.total_in()
    }
    /// Total payload bytes delivered, including failing calls.
    pub const fn total_out(&self) -> u64 {
        self.operation.total_out()
    }
    /// Number of validated members whose output has been delivered.
    pub const fn members_decoded(&self) -> u64 {
        self.operation.members_decoded()
    }
    /// Most recently accepted window header, as [`DecoderSession::window`].
    pub const fn window(&self) -> Option<Window> {
        self.operation.window()
    }

    /// Ends the current operation and starts a new, independent one in place.
    ///
    /// Releases the current operation exactly as [`Self::into_decompressor`]
    /// does — finished, waiting for input or output, or failed — and starts
    /// the next one with the same dictionary through the same path as
    /// [`Decompressor::into_session`]. To change the dictionary, go through
    /// [`Self::into_decompressor`] and
    /// [`Decompressor::into_session_with_dictionary`].
    ///
    /// # Errors
    /// As [`Decompressor::start`]. After an error the session is failed:
    /// [`Self::process`] returns [`DecodeError::InvalidState`] until a later
    /// `reinit` succeeds, and [`Self::into_decompressor`] still returns a
    /// reusable decoder.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut session = Decompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0; 16];
    /// assert!(session.process(&[0xff, 0xff], &mut output, DecodeOperation::Finish).is_err());
    ///
    /// session.reinit(Default::default())?;
    /// let progress = session.process(&compressed, &mut output, DecodeOperation::Finish)?;
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// assert_eq!(&output[..progress.produced], b"hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reinit(&mut self, stream: DecodeStreamConfig) -> Result<(), DecodeError> {
        self.operation.release(&mut self.decoder);
        match OperationState::start(&mut self.decoder, stream) {
            Ok(operation) => {
                self.operation = operation;
                Ok(())
            }
            Err(error) => {
                self.operation.poison();
                Err(error)
            }
        }
    }

    /// Ends the operation and returns the decoder, ready for the next one.
    ///
    /// Releases the operation exactly as dropping a [`DecoderSession`] does,
    /// whether it finished, stopped mid-stream or failed, and applies the
    /// decoder's retention policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut session = Decompressor::new(Default::default())?.into_session(Default::default())?;
    /// // Stop part-way through the member.
    /// session.process(&compressed[..4], &mut [0; 16], DecodeOperation::Process)?;
    ///
    /// let mut decoder = session.into_decompressor();
    /// assert_eq!(decoder.decompress(&compressed)?, b"hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_decompressor(self) -> Decompressor {
        let Self {
            mut decoder,
            operation,
            ..
        } = self;
        operation.release(&mut decoder);
        decoder
    }
}
