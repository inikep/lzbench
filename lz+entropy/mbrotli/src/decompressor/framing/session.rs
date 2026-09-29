use super::{core::Tag, *};
use crate::DecodeOperation;

/// Exact progress and at most one semantic event from the current call.
#[derive(Debug)]
pub struct FramedDecodeProgress<'a> {
    /// Accepted input prefix.
    pub consumed: usize,
    /// Delivered resource payload prefix.
    pub produced: usize,
    /// Event or backpressure reason.
    pub status: FramedDecoderStatus<'a>,
}
/// Incremental progress reason.
#[derive(Debug)]
pub enum FramedDecoderStatus<'a> {
    /// All offered input was accepted; supply more or declare EOF.
    NeedsInput,
    /// The next payload byte requires output space.
    NeedsOutput,
    /// Exactly one semantic event.
    Event(FramedEvent<'a>),
    /// Complete validation, with zero progress; subsequent calls remain finished.
    Finished,
}
/// Exclusive operation. External borrows live only here; Drop cancels without I/O.
///
/// The session borrows its [`FramedDecompressor`]; [`FramedDecoderSessionOwned`]
/// is the same operation owning it instead.
///
/// Forgetting a session protects the owner until `recover` or `reconfigure`.
/// Borrowed events prevent another mutating call until their last use.
///
/// ```compile_fail
/// use mbrotli::framing::{FramedDecompressor, FramedDecodeConfig, InputMode};
/// use mbrotli::DecodeOperation;
/// let mut decoder = FramedDecompressor::new(
///     FramedDecodeConfig::default().with_input_mode(InputMode::Auto)).unwrap();
/// let mut session = decoder.start(Default::default()).unwrap();
/// let mut output = [0; 1];
/// let event = session.process(&[0x3b], &mut output, DecodeOperation::Finish).unwrap();
/// session.process(&[0x3b], &mut [], DecodeOperation::Finish).unwrap();
/// println!("{event:?}"); // The first borrow is still live.
/// ```
pub struct FramedDecoderSession<'d, 'dict> {
    pub(super) owner: &'d mut FramedDecompressor,
    pub(super) resolver: Option<DictionaryResolverRef<'dict>>,
}
impl FramedDecoderSession<'_, '_> {
    /// Advances one object without retaining input or output references.
    /// After the first `Finish`, reoffer precisely the remaining suffix with
    /// `Finish` on every call. Output events borrow `output[..produced]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let mut session = decoder.start(Default::default())?;
    /// let mut remaining = bytes.as_slice();
    /// let mut payload = Vec::new();
    /// loop {
    ///     let mut buffer = [0; 1];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Finish)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// Terminal validation, policy, codec, or allocation failure with exact progress.
    /// Reusing a failed session or changing the final boundary yields `InvalidState`.
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn process<'a>(
        &'a mut self,
        input: &[u8],
        output: &'a mut [u8],
        operation: DecodeOperation,
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        self.owner
            .session_process(self.resolver, input, output, operation)
    }

    /// Delivers output for input already accepted, without declaring EOF.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Process`]; events still borrow `output` and the
    /// session. After a `Finish`, only `Finish` calls are valid.
    ///
    /// # Errors
    /// As [`Self::process`]; after a `Finish` this reports
    /// [`FramedDecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let mut session = decoder.start(Default::default())?;
    /// let mut payload = Vec::new();
    /// let mut buffer = [0; 2];
    /// let mut remaining = &bytes[..];
    /// while !remaining.is_empty() {
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     if let FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) = progress.status {
    ///         payload.extend_from_slice(data.bytes);
    ///     }
    /// }
    /// // Every input byte is accepted; drain the events it still produces.
    /// loop {
    ///     let progress = session.flush(&mut buffer)?;
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::NeedsInput | FramedDecoderStatus::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn flush<'a>(
        &'a mut self,
        output: &'a mut [u8],
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }
    /// Declares EOF with no input left, and delivers what remains.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Finish`], for when every input byte has already been
    /// accepted. Call it until it reports [`FramedDecoderStatus::Finished`],
    /// handling each event in between.
    ///
    /// # Errors
    /// As [`Self::process`]. An incomplete object reports its truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let mut session = decoder.start(Default::default())?;
    /// let mut payload = Vec::new();
    /// let mut remaining = &bytes[..];
    /// // Offer all input, handling events, until it has been accepted.
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::NeedsInput => break,
    ///         _ => {}
    ///     }
    /// }
    /// // Then declare EOF and take the rest.
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.finish(&mut buffer)?;
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn finish<'a>(
        &'a mut self,
        output: &'a mut [u8],
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub(super) fn step(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<(usize, usize, Tag), FramedDecodeFailure> {
        self.owner
            .session_step(self.resolver, input, output, operation)
    }
    /// Accepted wire bytes, including failing calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.total_in(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn total_in(&self) -> u64 {
        self.owner.engine.total_in
    }
    /// Delivered resource bytes, including failing calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.total_out(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn total_out(&self) -> u64 {
        self.owner.engine.total_out
    }
    /// Regenerated resource and metadata bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.total_decoded(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn total_decoded(&self) -> u64 {
        self.owner.engine.total_decoded
    }
    /// Locally completed resource payloads; not an authentication count.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.resources_decoded(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn resources_decoded(&self) -> u64 {
        self.owner.engine.completed
    }
    /// Whether all object validation succeeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.is_finished(), false);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        self.owner.engine.finished
    }
    /// Selected input format, or `None` before detection.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.input_format(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn input_format(&self) -> Option<StreamInfo> {
        self.owner.engine.format
    }
}
impl Drop for FramedDecoderSession<'_, '_> {
    fn drop(&mut self) {
        self.owner.cancel();
    }
}
impl ::core::fmt::Debug for FramedDecoderSession<'_, '_> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.debug_struct("FramedDecoderSession")
            .field("total_in", &self.total_in())
            .field("total_out", &self.total_out())
            .field("format", &self.input_format())
            .finish_non_exhaustive()
    }
}

// The call bodies shared by the borrowed and owned session facades.
impl FramedDecompressor {
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    fn session_step(
        &mut self,
        resolver: Option<DictionaryResolverRef<'_>>,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
    ) -> Result<(usize, usize, Tag), FramedDecodeFailure> {
        self.engine
            .process(input, output, operation, self.backend, resolver)
    }
    /// Runs one step and lends its event from `output` and the engine for `'a`.
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    fn session_process<'a>(
        &'a mut self,
        resolver: Option<DictionaryResolverRef<'_>>,
        input: &[u8],
        output: &'a mut [u8],
        operation: DecodeOperation,
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        let (consumed, produced, tag) = self.session_step(resolver, input, output, operation)?;
        let status = match tag {
            Tag::Input => FramedDecoderStatus::NeedsInput,
            Tag::Output => FramedDecoderStatus::NeedsOutput,
            Tag::Finished => FramedDecoderStatus::Finished,
            _ => FramedDecoderStatus::Event(self.engine.event(tag, &output[..produced]).ok_or(
                FramedDecodeFailure {
                    error: FramedDecodeError::InvalidState,
                    consumed,
                    produced,
                    last_output: None,
                },
            )?),
        };
        Ok(FramedDecodeProgress {
            consumed,
            produced,
            status,
        })
    }
}

/// Exclusive operation that owns its [`FramedDecompressor`].
///
/// [`FramedDecoderSession`] borrows its decoder with `&mut` and its resolver
/// for `'dict`. This session owns both, so a composition layer can hold the
/// whole streaming operation as one value with no lifetime. It keeps no
/// references into its own fields, and it is the same synchronous,
/// caller-driven API: calls run the borrowed session's code on the same
/// framing engine, so events, progress and errors are identical.
///
/// `R` is the owned resolver of external dictionaries. A session started by
/// [`FramedDecompressor::into_session`] has none and leaves `R` at its default,
/// [`NoDictionaries`]. The session is `Send` exactly when `R` is.
///
/// Events still borrow the session and `output`, so the next call cannot run
/// while an event is alive:
/// ```compile_fail
/// use mbrotli::framing::{FramedDecompressor, FramedDecodeConfig, InputMode};
/// use mbrotli::DecodeOperation;
/// let decoder = FramedDecompressor::new(
///     FramedDecodeConfig::default().with_input_mode(InputMode::Auto)).unwrap();
/// let mut session = decoder.into_session(Default::default()).unwrap();
/// let mut output = [0; 1];
/// let event = session.process(&[0x3b], &mut output, DecodeOperation::Finish).unwrap();
/// session.process(&[0x3b], &mut [], DecodeOperation::Finish).unwrap();
/// println!("{event:?}"); // The first borrow is still live.
/// ```
///
/// [`Self::into_framed_decompressor`] cancels the object exactly as dropping a
/// borrowed session does and hands the decoder back for reuse. Dropping the
/// owned session instead drops the decoder with it.
///
/// # Examples
///
/// ```
/// use mbrotli::framing::*;
/// use mbrotli::DecodeOperation;
/// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
/// let decoder = FramedDecompressor::new(Default::default())?;
/// let mut session = decoder.into_session(Default::default())?;
/// let mut remaining = bytes.as_slice();
/// let mut payload = Vec::new();
/// loop {
///     let mut buffer = [0; 1];
///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Finish)?;
///     remaining = &remaining[progress.consumed..];
///     match progress.status {
///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
///             payload.extend_from_slice(data.bytes),
///         FramedDecoderStatus::Finished => break,
///         _ => {}
///     }
/// }
/// assert_eq!(payload, b"abc");
/// let decoder = session.into_framed_decompressor();
/// let next = decoder.into_session(Default::default())?;
/// assert_eq!(next.total_in(), 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FramedDecoderSessionOwned<R = NoDictionaries> {
    pub(super) owner: FramedDecompressor,
    pub(super) resolver: Option<R>,
}
impl<R: DictionaryResolver + 'static> FramedDecoderSessionOwned<R> {
    /// Advances one object, as [`FramedDecoderSession::process`].
    ///
    /// Output events borrow `output[..produced]` and the session for `'a`.
    ///
    /// # Errors
    /// As [`FramedDecoderSession::process`].
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn process<'a>(
        &'a mut self,
        input: &[u8],
        output: &'a mut [u8],
        operation: DecodeOperation,
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        let resolver = self.resolver.as_ref().map(DictionaryResolverRef::from);
        self.owner
            .session_process(resolver, input, output, operation)
    }

    /// Delivers output for input already accepted, without declaring EOF.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Process`]; events still borrow `output` and the
    /// session. After a `Finish`, only `Finish` calls are valid.
    ///
    /// # Errors
    /// As [`Self::process`]; after a `Finish` this reports
    /// [`FramedDecodeError::InvalidState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut session = FramedDecompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut payload = Vec::new();
    /// let mut buffer = [0; 2];
    /// let mut remaining = &bytes[..];
    /// while !remaining.is_empty() {
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     if let FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) = progress.status {
    ///         payload.extend_from_slice(data.bytes);
    ///     }
    /// }
    /// // Every input byte is accepted; drain the events it still produces.
    /// loop {
    ///     let progress = session.flush(&mut buffer)?;
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::NeedsInput | FramedDecoderStatus::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn flush<'a>(
        &'a mut self,
        output: &'a mut [u8],
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        self.process(&[], output, DecodeOperation::Process)
    }
    /// Declares EOF with no input left, and delivers what remains.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`DecodeOperation::Finish`], for when every input byte has already been
    /// accepted. Call it until it reports [`FramedDecoderStatus::Finished`],
    /// handling each event in between.
    ///
    /// # Errors
    /// As [`Self::process`]. An incomplete object reports its truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let bytes = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut session = FramedDecompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut payload = Vec::new();
    /// let mut remaining = &bytes[..];
    /// // Offer all input, handling events, until it has been accepted.
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.process(remaining, &mut buffer, DecodeOperation::Process)?;
    ///     remaining = &remaining[progress.consumed..];
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::NeedsInput => break,
    ///         _ => {}
    ///     }
    /// }
    /// // Then declare EOF and take the rest.
    /// loop {
    ///     let mut buffer = [0; 2];
    ///     let progress = session.finish(&mut buffer)?;
    ///     match progress.status {
    ///         FramedDecoderStatus::Event(FramedEvent::ResourceData(data)) =>
    ///             payload.extend_from_slice(data.bytes),
    ///         FramedDecoderStatus::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn finish<'a>(
        &'a mut self,
        output: &'a mut [u8],
    ) -> Result<FramedDecodeProgress<'a>, FramedDecodeFailure> {
        self.process(&[], output, DecodeOperation::Finish)
    }
    /// Accepted wire bytes, including failing calls.
    pub const fn total_in(&self) -> u64 {
        self.owner.engine.total_in
    }
    /// Delivered resource bytes, including failing calls.
    pub const fn total_out(&self) -> u64 {
        self.owner.engine.total_out
    }
    /// Regenerated resource and metadata bytes.
    pub const fn total_decoded(&self) -> u64 {
        self.owner.engine.total_decoded
    }
    /// Locally completed resource payloads; not an authentication count.
    pub const fn resources_decoded(&self) -> u64 {
        self.owner.engine.completed
    }
    /// Whether all object validation succeeded.
    pub const fn is_finished(&self) -> bool {
        self.owner.engine.finished
    }
    /// Selected input format, or `None` before detection.
    pub const fn input_format(&self) -> Option<StreamInfo> {
        self.owner.engine.format
    }
    /// Ends the current object and starts a new, independent one in place.
    ///
    /// Cancels the current object exactly as
    /// [`Self::into_framed_decompressor`] does, then starts the next one with
    /// the same resolver through the same path as
    /// [`FramedDecompressor::into_session`]. To change the resolver, go through
    /// [`Self::into_framed_decompressor`] and
    /// [`FramedDecompressor::into_session_with_dictionaries`].
    /// # Errors
    /// As [`FramedDecompressor::start`]. After an error the session is failed:
    /// [`Self::process`] returns [`FramedDecodeError::InvalidState`] until a
    /// later `reinit` succeeds.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let valid = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut session = FramedDecompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0; 16];
    /// assert!(session.process(&[0x91, 10, 66, 0xff], &mut output, DecodeOperation::Finish).is_err());
    ///
    /// session.reinit(Default::default())?;
    /// let progress = session.process(&valid, &mut output, DecodeOperation::Finish)?;
    /// assert!(matches!(progress.status, FramedDecoderStatus::Event(FramedEvent::StreamStart(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reinit(&mut self, stream: FramedDecodeStreamConfig) -> Result<(), FramedDecodeError> {
        self.owner.cancel();
        let started = self.owner.begin_session(stream);
        if started.is_err() {
            self.owner.engine.poison();
        }
        started
    }
    /// Ends the object and returns the decoder, ready for the next one.
    ///
    /// Cancels exactly as dropping a [`FramedDecoderSession`] does, after a
    /// finished, incomplete, backpressured or failed operation. The resolver
    /// is dropped.
    #[must_use]
    pub fn into_framed_decompressor(self) -> FramedDecompressor {
        let Self { mut owner, .. } = self;
        owner.cancel();
        owner
    }
}
impl<R> ::core::fmt::Debug for FramedDecoderSessionOwned<R> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.debug_struct("FramedDecoderSessionOwned")
            .field("total_in", &self.owner.engine.total_in)
            .field("total_out", &self.owner.engine.total_out)
            .field("format", &self.owner.engine.format)
            .finish_non_exhaustive()
    }
}
