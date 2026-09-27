use super::*;
use crate::dictionary::PreparedDictionary;
use crate::{Operation, StreamConfig};

/// Container output operation; payload is supplied through a resource guard.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FramedEncodeOperation {
    /// Drain queued output and accept further structural commands.
    #[default]
    Process,
    /// Generate and deliver repeats, directory and footer exactly once.
    Finish,
}
/// Required next action after a native encoding call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramedEncoderStatus {
    /// Queue is empty; more input or another command may be accepted.
    NeedsInput,
    /// Pending wire bytes need destination space.
    NeedsOutput,
    /// This resource or container is fully encoded and delivered.
    Finished,
}
/// Exact per-call payload acceptance and wire output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedEncodeProgress {
    /// Payload accepted; always zero for container `process`.
    pub consumed: usize,
    /// Initialized destination prefix length.
    pub produced: usize,
    /// Next action required.
    pub status: FramedEncoderStatus,
}
/// Exclusive container operation. Drop cancels without I/O and releases the owner.
///
/// The session borrows its [`FramedCompressor`]; [`FramedEncoderSessionOwned`]
/// is the same operation owning it instead.
/// A forgotten session requires explicit owner recovery.
/// ```compile_fail
/// use mbrotli::framing::*;
/// let mut owner = FramedCompressor::new(Default::default()).unwrap();
/// let first = owner.start(Default::default()).unwrap();
/// let second = owner.start(Default::default()).unwrap();
/// drop(first);
/// ```
#[derive(Debug)]
pub struct FramedEncoderSession<'c> {
    pub(super) owner: &'c mut FramedCompressor,
}
impl FramedEncoderSession<'_> {
    /// Drains the header/pending command, or incrementally finalizes the container.
    /// # Errors
    /// Process errors are terminal and carry exact per-call progress. Command errors
    /// before commit, including `OutputPending`, leave that command unaccepted.
    pub fn process(
        &mut self,
        output: &mut [u8],
        operation: FramedEncodeOperation,
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.owner.engine.process(output, operation)
    }

    /// Delivers queued wire bytes without finishing the container.
    ///
    /// Shorthand for [`Self::process`] with [`FramedEncodeOperation::Process`].
    /// Repeat it while it reports [`FramedEncoderStatus::NeedsOutput`];
    /// commands are accepted once it reports `NeedsInput`.
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut session = owner.start(Default::default())?;
    /// let mut output = [0; 128];
    /// let progress = session.flush(&mut output)?;
    /// assert_eq!(&output[..progress.produced], &[0x91, 10, 66, 82, 4]);
    /// assert_eq!(progress.status, FramedEncoderStatus::NeedsInput);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(output, FramedEncodeOperation::Process)
    }
    /// Generates and delivers the container suffix.
    ///
    /// Shorthand for [`Self::process`] with [`FramedEncodeOperation::Finish`].
    /// Repeat it while it reports [`FramedEncoderStatus::NeedsOutput`], until
    /// [`FramedEncoderStatus::Finished`].
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut session = owner.start(Default::default())?;
    /// let mut output = [0; 128];
    /// let progress = session.finish(&mut output)?;
    /// assert_eq!(progress.status, FramedEncoderStatus::Finished);
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(output, FramedEncodeOperation::Finish)
    }
    /// Queues uncompressed ordered metadata.
    /// # Errors
    /// Rejects pending output, ordering, invalid fields and exhausted budgets before commit.
    pub fn metadata(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
    ) -> Result<(), FramedEncodeError> {
        self.metadata_with_options(kind, fields, Default::default())
    }
    /// Queues independently encoded original and repeated metadata.
    /// # Errors
    /// Rejects invalid commands before commit; dictionary is borrowed only during this call.
    pub fn metadata_with_options(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), FramedEncodeError> {
        self.owner.session_metadata(kind, fields, options)
    }
    /// Selects repeated fields before the first metadata command.
    /// # Errors
    /// Rejects pending output, disabled repetition, invalid codes or a late selection.
    pub fn repeat_metadata_fields(&mut self, codes: &[[u8; 2]]) -> Result<(), FramedEncodeError> {
        self.owner.engine.repeat(codes)
    }
    /// Queues one padding command without accepting payload.
    /// # Errors
    /// Rejects pending output, invalid state or budgets before commit.
    pub fn padding(&mut self, bytes: usize) -> Result<(), FramedEncodeError> {
        self.owner.engine.padding(bytes)
    }
    /// Starts a compressed resource; Finish and drop it before the next command.
    /// # Errors
    /// Rejects pending output, invalid ordering/settings or exhausted budgets.
    pub fn resource(
        &mut self,
        options: ResourceOptions,
        stream: StreamConfig,
    ) -> Result<FramedResourceSession<'_, 'static>, FramedEncodeError> {
        self.owner
            .open_resource(options, stream, ResourceEncoding::Brotli, None)
    }
    /// Starts a resource borrowing an explicit prepared dictionary.
    /// # Errors
    /// As `resource`, plus invalid references or unsupported dictionary settings.
    /// # Examples
    /// The dictionary lives only for one resource, and dies before the container finishes.
    /// ```
    /// use mbrotli::{Operation, dictionary::DictionaryBuilder, framing::*};
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut session = owner.start(Default::default())?;
    /// let mut output = [0; 32];
    /// let mut wire = Vec::new();
    /// loop {
    ///     let p = session.process(&mut output, FramedEncodeOperation::Process)?;
    ///     wire.extend_from_slice(&output[..p.produced]);
    ///     if p.status == FramedEncoderStatus::NeedsInput { break; }
    /// }
    /// {
    ///     let dictionary = DictionaryBuilder::new().add_prefix(&b"shared words"[..]).build()?;
    ///     let references = [DictionaryReference::PrefixId(DictionaryId([7; 32]))];
    ///     let mut resource = session.resource_with_dictionary(
    ///         Default::default(), Default::default(), &dictionary, &references,
    ///     )?;
    ///     let mut input = &b"shared words shared words"[..];
    ///     loop {
    ///         let p = resource.process(input, &mut output, Operation::Finish)?;
    ///         input = &input[p.consumed..];
    ///         wire.extend_from_slice(&output[..p.produced]);
    ///         if p.status == FramedEncoderStatus::Finished { break; }
    ///     }
    /// } // Finished resource guard, then dictionary, are destroyed here.
    /// loop {
    ///     let p = session.process(&mut output, FramedEncodeOperation::Finish)?;
    ///     wire.extend_from_slice(&output[..p.produced]);
    ///     if p.status == FramedEncoderStatus::Finished { break; }
    /// }
    /// assert_eq!(session.resources_encoded(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resource_with_dictionary<'s, 'dict>(
        &'s mut self,
        options: ResourceOptions,
        stream: StreamConfig,
        dictionary: &'dict PreparedDictionary,
        references: &[DictionaryReference],
    ) -> Result<FramedResourceSession<'s, 'dict>, FramedEncodeError> {
        self.owner
            .open_shared_resource(options, stream, dictionary, references)
    }
    /// Starts a verbatim resource using the same chunk and lifecycle rules.
    /// # Errors
    /// Rejects pending output, invalid ordering or exhausted budgets.
    pub fn uncompressed_resource(
        &mut self,
        options: ResourceOptions,
    ) -> Result<FramedResourceSession<'_, 'static>, FramedEncodeError> {
        self.owner.open_resource(
            options,
            Default::default(),
            ResourceEncoding::Uncompressed,
            None,
        )
    }
    /// Accepted payload across resources, including hidden resources.
    pub const fn total_in(&self) -> u64 {
        self.owner.engine.total_in
    }
    /// Wire bytes delivered to callers, relative to this container's start.
    pub const fn total_out(&self) -> u64 {
        self.owner.engine.total_out
    }
    /// Resources whose final chunks have been completely delivered.
    pub const fn resources_encoded(&self) -> u64 {
        self.owner.engine.resources()
    }
    /// Whether the suffix was generated and fully delivered.
    pub const fn is_finished(&self) -> bool {
        self.owner.engine.finished()
    }
    /// Queued offset of the next chunk, independent of destination transport.
    pub const fn next_chunk_offset(&self) -> u64 {
        self.owner.engine.offset()
    }
}
impl Drop for FramedEncoderSession<'_> {
    fn drop(&mut self) {
        self.owner.cancel();
    }
}

// Command bodies shared by the borrowed and owned session facades.
impl FramedCompressor {
    fn session_metadata(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), FramedEncodeError> {
        self.engine.metadata(&mut self.raw, kind, fields, options)
    }
    fn open_resource<'s, 'dict>(
        &'s mut self,
        options: ResourceOptions,
        stream: StreamConfig,
        encoding: ResourceEncoding<'_>,
        dictionary: Option<&'dict PreparedDictionary>,
    ) -> Result<FramedResourceSession<'s, 'dict>, FramedEncodeError> {
        self.engine
            .begin_resource(&mut self.raw, options, stream, encoding)?;
        Ok(FramedResourceSession {
            owner: self,
            dictionary,
        })
    }
    fn open_shared_resource<'s, 'dict>(
        &'s mut self,
        options: ResourceOptions,
        stream: StreamConfig,
        dictionary: &'dict PreparedDictionary,
        references: &[DictionaryReference],
    ) -> Result<FramedResourceSession<'s, 'dict>, FramedEncodeError> {
        let encoding = ResourceEncoding::Shared {
            dictionary,
            references,
        };
        self.open_resource(options, stream, encoding, Some(dictionary))
    }
}

/// Exclusive container operation that owns its [`FramedCompressor`].
///
/// [`FramedEncoderSession`] borrows its encoder with `&mut`; this session
/// consumes it, so a composition layer can hold the whole streaming operation
/// as one owned value with no lifetime. It keeps no references into its own
/// fields, and it is the same synchronous, caller-driven API: every method
/// runs the borrowed session's code on the same framing engine, so the wire
/// bytes, progress and errors are identical.
///
/// Resources still open through the borrowing [`FramedResourceSession`]
/// guard. While one is live the session is mutably borrowed, so neither a
/// container command nor [`Self::into_framed_compressor`] can run:
/// ```compile_fail
/// use mbrotli::framing::*;
/// let owner = FramedCompressor::new(Default::default()).unwrap();
/// let mut session = owner.into_session(Default::default()).unwrap();
/// session.process(&mut [0; 16], FramedEncodeOperation::Process).unwrap();
/// let resource = session.resource(Default::default(), Default::default()).unwrap();
/// session.padding(1).unwrap(); // The resource still borrows the session.
/// drop(resource);
/// ```
///
/// [`Self::into_framed_compressor`] cancels the container exactly as dropping
/// a borrowed session does and hands the encoder back for reuse. Dropping the
/// owned session instead drops the encoder with it.
///
/// # Examples
/// ```
/// use mbrotli::{Operation, framing::*};
/// let encoder = FramedCompressor::new(Default::default())?;
/// let mut session = encoder.into_session(Default::default())?;
/// let mut output = [0; 128];
/// let mut wire = Vec::new();
/// let p = session.process(&mut output, FramedEncodeOperation::Process)?;
/// wire.extend_from_slice(&output[..p.produced]);
/// {
///     let mut resource = session.resource(Default::default(), Default::default())?;
///     let p = resource.process(b"owned", &mut output, Operation::Finish)?;
///     assert_eq!(p.status, FramedEncoderStatus::Finished);
///     wire.extend_from_slice(&output[..p.produced]);
/// }
/// let p = session.process(&mut output, FramedEncodeOperation::Finish)?;
/// assert_eq!(p.status, FramedEncoderStatus::Finished);
/// wire.extend_from_slice(&output[..p.produced]);
///
/// let mut encoder = session.into_framed_compressor();
/// let items = [FramedItem::Resource(FramedResource::from(&b"owned"[..]))];
/// assert_eq!(encoder.compress(items.as_slice().into())?, wire);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FramedEncoderSessionOwned {
    pub(super) owner: FramedCompressor,
}
impl FramedEncoderSessionOwned {
    /// Drains queued output or finalizes, as [`FramedEncoderSession::process`].
    /// # Errors
    /// As [`FramedEncoderSession::process`].
    pub fn process(
        &mut self,
        output: &mut [u8],
        operation: FramedEncodeOperation,
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.owner.engine.process(output, operation)
    }

    /// Delivers queued wire bytes without finishing the container.
    ///
    /// Shorthand for [`Self::process`] with [`FramedEncodeOperation::Process`].
    /// Repeat it while it reports [`FramedEncoderStatus::NeedsOutput`];
    /// commands are accepted once it reports `NeedsInput`.
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut session = FramedCompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0; 128];
    /// let progress = session.flush(&mut output)?;
    /// assert_eq!(&output[..progress.produced], &[0x91, 10, 66, 82, 4]);
    /// assert_eq!(progress.status, FramedEncoderStatus::NeedsInput);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(output, FramedEncodeOperation::Process)
    }
    /// Generates and delivers the container suffix.
    ///
    /// Shorthand for [`Self::process`] with [`FramedEncodeOperation::Finish`].
    /// Repeat it while it reports [`FramedEncoderStatus::NeedsOutput`], until
    /// [`FramedEncoderStatus::Finished`].
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut session = FramedCompressor::new(Default::default())?.into_session(Default::default())?;
    /// let mut output = [0; 128];
    /// let progress = session.finish(&mut output)?;
    /// assert_eq!(progress.status, FramedEncoderStatus::Finished);
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(output, FramedEncodeOperation::Finish)
    }
    /// Queues uncompressed metadata, as [`FramedEncoderSession::metadata`].
    /// # Errors
    /// As [`FramedEncoderSession::metadata`].
    pub fn metadata(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
    ) -> Result<(), FramedEncodeError> {
        self.owner
            .session_metadata(kind, fields, Default::default())
    }
    /// Queues metadata with options, as [`FramedEncoderSession::metadata_with_options`].
    /// # Errors
    /// As [`FramedEncoderSession::metadata_with_options`].
    pub fn metadata_with_options(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), FramedEncodeError> {
        self.owner.session_metadata(kind, fields, options)
    }
    /// Selects repeated fields, as [`FramedEncoderSession::repeat_metadata_fields`].
    /// # Errors
    /// As [`FramedEncoderSession::repeat_metadata_fields`].
    pub fn repeat_metadata_fields(&mut self, codes: &[[u8; 2]]) -> Result<(), FramedEncodeError> {
        self.owner.engine.repeat(codes)
    }
    /// Queues padding, as [`FramedEncoderSession::padding`].
    /// # Errors
    /// As [`FramedEncoderSession::padding`].
    pub fn padding(&mut self, bytes: usize) -> Result<(), FramedEncodeError> {
        self.owner.engine.padding(bytes)
    }
    /// Starts a compressed resource borrowing this session, as
    /// [`FramedEncoderSession::resource`].
    /// # Errors
    /// As [`FramedEncoderSession::resource`].
    pub fn resource(
        &mut self,
        options: ResourceOptions,
        stream: StreamConfig,
    ) -> Result<FramedResourceSession<'_, 'static>, FramedEncodeError> {
        self.owner
            .open_resource(options, stream, ResourceEncoding::Brotli, None)
    }
    /// Starts a resource borrowing a prepared dictionary, as
    /// [`FramedEncoderSession::resource_with_dictionary`].
    /// # Errors
    /// As [`FramedEncoderSession::resource_with_dictionary`].
    pub fn resource_with_dictionary<'s, 'dict>(
        &'s mut self,
        options: ResourceOptions,
        stream: StreamConfig,
        dictionary: &'dict PreparedDictionary,
        references: &[DictionaryReference],
    ) -> Result<FramedResourceSession<'s, 'dict>, FramedEncodeError> {
        self.owner
            .open_shared_resource(options, stream, dictionary, references)
    }
    /// Starts a verbatim resource, as [`FramedEncoderSession::uncompressed_resource`].
    /// # Errors
    /// As [`FramedEncoderSession::uncompressed_resource`].
    pub fn uncompressed_resource(
        &mut self,
        options: ResourceOptions,
    ) -> Result<FramedResourceSession<'_, 'static>, FramedEncodeError> {
        self.owner.open_resource(
            options,
            Default::default(),
            ResourceEncoding::Uncompressed,
            None,
        )
    }
    /// Accepted payload across resources, including hidden resources.
    pub const fn total_in(&self) -> u64 {
        self.owner.engine.total_in
    }
    /// Wire bytes delivered to callers, relative to this container's start.
    pub const fn total_out(&self) -> u64 {
        self.owner.engine.total_out
    }
    /// Resources whose final chunks have been completely delivered.
    pub const fn resources_encoded(&self) -> u64 {
        self.owner.engine.resources()
    }
    /// Whether the suffix was generated and fully delivered.
    pub const fn is_finished(&self) -> bool {
        self.owner.engine.finished()
    }
    /// Queued offset of the next chunk, independent of destination transport.
    pub const fn next_chunk_offset(&self) -> u64 {
        self.owner.engine.offset()
    }
    /// Ends the current container and starts a new, independent one in place.
    ///
    /// Cancels the current container exactly as
    /// [`Self::into_framed_compressor`] does, then starts the next one through
    /// the same path as [`FramedCompressor::into_session`], keeping the
    /// encoder and its retained storage.
    /// # Errors
    /// As [`FramedCompressor::start`]. After an error the session is failed:
    /// calls and commands return [`FramedEncodeError::InvalidState`] until a
    /// later `reinit` succeeds.
    /// # Examples
    /// ```
    /// use mbrotli::{Operation, framing::*};
    /// let mut session = FramedCompressor::new(Default::default())?.into_session(Default::default())?;
    /// session.process(&mut [0; 128], FramedEncodeOperation::Process)?;
    /// session.reinit(Default::default())?;
    /// // The new container starts with a fresh header.
    /// let mut output = [0; 128];
    /// let p = session.process(&mut output, FramedEncodeOperation::Process)?;
    /// assert_eq!(&output[..4], &[0x91, 10, 66, 82]);
    /// assert_eq!(session.total_out(), p.produced as u64);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reinit(&mut self, stream: FramedEncodeStreamConfig) -> Result<(), FramedEncodeError> {
        self.owner.cancel();
        let started = self.owner.begin_session(stream);
        if started.is_err() {
            self.owner.engine.poison();
        }
        started
    }
    /// Ends the container and returns the encoder, ready for the next one.
    ///
    /// Cancels exactly as dropping a [`FramedEncoderSession`] does, whether the
    /// container finished, is mid-way, still has output queued, or failed.
    #[must_use]
    pub fn into_framed_compressor(self) -> FramedCompressor {
        let Self { mut owner } = self;
        owner.cancel();
        owner
    }
}
/// Exclusive byte-input resource guard. An unfinished drop abandons the container.
///
/// The dictionary cannot be destroyed while its resource guard is live:
/// ```compile_fail
/// use mbrotli::framing::*;
/// let mut owner = FramedCompressor::new(Default::default()).unwrap();
/// let mut session = owner.start(Default::default()).unwrap();
/// session.process(&mut [0; 5], FramedEncodeOperation::Process).unwrap();
/// let dictionary = mbrotli::dictionary::DictionaryBuilder::new().add_prefix(&b"prefix"[..]).build().unwrap();
/// let resource = session.resource_with_dictionary(Default::default(), Default::default(), &dictionary,
///     &[DictionaryReference::PrefixId(DictionaryId([0; 32]))]).unwrap();
/// drop(dictionary);
/// drop(resource);
/// ```
#[derive(Debug)]
pub struct FramedResourceSession<'session, 'dict> {
    pub(super) owner: &'session mut FramedCompressor,
    dictionary: Option<&'dict PreparedDictionary>,
}
impl FramedResourceSession<'_, '_> {
    /// Accepts payload and emits bounded framing chunks. Flush/Finish calls must
    /// retain their operation and remaining input suffix until completed.
    /// # Errors
    /// Terminal failures retain exact per-call progress; later calls return InvalidState.
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.owner.engine.process_resource(
            &mut self.owner.raw,
            self.dictionary,
            input,
            output,
            operation,
        )
    }

    /// Emits everything accepted so far as complete chunks, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Flush`]. Repeat it while it reports
    /// [`FramedEncoderStatus::NeedsOutput`], until `NeedsInput`.
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::{Operation, framing::*};
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut session = owner.start(Default::default())?;
    /// session.flush(&mut [0; 128])?;
    /// let mut resource = session.resource(Default::default(), Default::default())?;
    /// resource.process(b"flushed", &mut [0; 128], Operation::Process)?;
    /// let progress = resource.flush(&mut [0; 128])?;
    /// assert_eq!(progress.status, FramedEncoderStatus::NeedsInput);
    /// assert!(progress.produced > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(&[], output, Operation::Flush)
    }
    /// Ends the resource, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Finish`], for when the whole payload has already been
    /// passed to `process`. Repeat it while it reports
    /// [`FramedEncoderStatus::NeedsOutput`], until
    /// [`FramedEncoderStatus::Finished`].
    /// # Errors
    /// As [`Self::process`].
    /// # Examples
    /// ```
    /// use mbrotli::{Operation, framing::*};
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut session = owner.start(Default::default())?;
    /// session.flush(&mut [0; 128])?;
    /// let mut resource = session.resource(Default::default(), Default::default())?;
    /// resource.process(b"payload", &mut [0; 128], Operation::Process)?;
    /// assert_eq!(resource.finish(&mut [0; 128])?.status, FramedEncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        self.process(&[], output, Operation::Finish)
    }
    /// Whether the final resource chunk has been completely delivered.
    pub const fn is_finished(&self) -> bool {
        self.owner.engine.resource_finished()
    }
    /// Accepted resource payload bytes.
    pub const fn total_in(&self) -> u64 {
        self.owner.engine.resource_in()
    }
    /// Delivered wire bytes for this resource, including its chunk headers.
    pub const fn total_out(&self) -> u64 {
        self.owner.engine.resource_out()
    }
}

impl Drop for FramedResourceSession<'_, '_> {
    fn drop(&mut self) {
        self.owner.engine.release_resource(&mut self.owner.raw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rejected `reinit` poisons the engine; only allocation failure can
    /// reject a framed start, so the state is set directly here.
    #[test]
    fn a_poisoned_owned_session_refuses_work_until_a_reinit_succeeds() {
        let mut session = FramedCompressor::new(Default::default())
            .expect("config")
            .into_session(Default::default())
            .expect("start");
        session.owner.cancel();
        session.owner.engine.poison();
        let failure = session
            .process(&mut [0; 64], FramedEncodeOperation::Process)
            .unwrap_err();
        assert!(matches!(failure.error, FramedEncodeError::InvalidState));
        assert!(matches!(
            session.padding(1),
            Err(FramedEncodeError::InvalidState)
        ));
        assert!(matches!(
            session.resource(Default::default(), Default::default()),
            Err(FramedEncodeError::InvalidState)
        ));

        session.reinit(Default::default()).expect("reinit");
        let mut output = [0; 64];
        let progress = session
            .process(&mut output, FramedEncodeOperation::Process)
            .expect("header");
        assert_eq!(&output[..progress.produced], &[0x91, 10, 66, 82, 4]);
    }
}
