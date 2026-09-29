use super::{
    core::{Engine, Tag},
    *,
};
use crate::decompressor::core::Stream;
use crate::{Backend, DecodeOperation, RetentionPolicy};
use ::core::ops::Range;
use FramedDecodeError as E;
use alloc::vec::Vec;

/// Reusable structured decoder, independent of the raw decoder's API.
#[derive(Debug)]
pub struct FramedDecompressor {
    pub(super) engine: Engine,
    pub(super) backend: Backend,
    retention: RetentionPolicy,
    pub(super) active: bool,
}
/// Construction policy with an optional explicitly selected host backend.
#[derive(Clone, Copy, Debug)]
pub struct FramedDecompressorBuilder {
    config: FramedDecodeConfig,
    backend: Option<Backend>,
    retention: RetentionPolicy,
}
impl FramedDecompressorBuilder {
    /// Selects a host-validated backend outside inner loops.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let decoder = FramedDecompressor::builder(Default::default())
    ///     .with_backend(mbrotli::Backend::default())
    ///     .with_retention(mbrotli::RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(decoder.retention(), mbrotli::RetentionPolicy::ReleaseAll);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }
    /// Selects storage retention after session cancellation or completion.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let decoder = FramedDecompressor::builder(Default::default())
    ///     .with_backend(mbrotli::Backend::default())
    ///     .with_retention(mbrotli::RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(decoder.retention(), mbrotli::RetentionPolicy::ReleaseAll);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_retention(mut self, retention: RetentionPolicy) -> Self {
        self.retention = retention;
        self
    }
    /// Constructs an empty reusable owner.
    /// # Errors
    /// Typed configuration currently requires no additional validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let decoder = FramedDecompressor::builder(Default::default())
    ///     .with_backend(mbrotli::Backend::default())
    ///     .with_retention(mbrotli::RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(decoder.retention(), mbrotli::RetentionPolicy::ReleaseAll);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build(self) -> Result<FramedDecompressor, E> {
        Ok(FramedDecompressor {
            engine: Engine::new(self.config, Default::default(), Stream::default()),
            backend: self.backend.unwrap_or_default(),
            retention: self.retention,
            active: false,
        })
    }
}
impl FramedDecompressor {
    /// Constructs an empty reusable owner with default backend and retention.
    /// # Errors
    /// As `FramedDecompressorBuilder::build`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// assert_eq!(decoder.decompress(&input)?.resources[0].data, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: FramedDecodeConfig) -> Result<Self, E> {
        Self::builder(config).build()
    }
    /// Begins configuration without allocating or detecting a backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let decoder = FramedDecompressor::builder(Default::default())
    ///     .with_backend(mbrotli::Backend::default())
    ///     .with_retention(mbrotli::RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(decoder.retention(), mbrotli::RetentionPolicy::ReleaseAll);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn builder(config: FramedDecodeConfig) -> FramedDecompressorBuilder {
        FramedDecompressorBuilder {
            config,
            backend: None,
            retention: RetentionPolicy::Aggressive,
        }
    }
    /// Reusable decode policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// assert_eq!(decoder.config().input_mode(), InputMode::FramedOnly);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn config(&self) -> &FramedDecodeConfig {
        &self.engine.config
    }
    /// Retention policy applied on session Drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let decoder = FramedDecompressor::builder(Default::default())
    ///     .with_backend(mbrotli::Backend::default())
    ///     .with_retention(mbrotli::RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(decoder.retention(), mbrotli::RetentionPolicy::ReleaseAll);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn retention(&self) -> RetentionPolicy {
        self.retention
    }
    /// Reconfigures and clears abandoned-operation protection.
    /// # Errors
    /// Typed configuration currently requires no additional validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// decoder.reconfigure(FramedDecodeConfig::default().with_input_mode(InputMode::Auto))?;
    /// assert!(decoder.decompress(&[0x3b])?.resources[0].data.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reconfigure(&mut self, config: FramedDecodeConfig) -> Result<(), E> {
        if self.active
            || (self.retention == RetentionPolicy::CurrentConfig && config != self.engine.config)
        {
            self.recover();
        }
        self.engine.config = config;
        self.cancel();
        Ok(())
    }
    /// Currently retained requested heap bytes, excluding returned output.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// drop(decoder.decompress(&input)?);
    /// assert!(decoder.retained_bytes() > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn retained_bytes(&self) -> usize {
        self.engine.retained_bytes()
    }
    /// Applies a one-time storage policy without clearing abandoned protection.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// drop(decoder.decompress(&input)?);
    /// decoder.trim(mbrotli::RetentionPolicy::ReleaseAll);
    /// assert_eq!(decoder.retained_bytes(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn trim(&mut self, policy: RetentionPolicy) {
        if policy == RetentionPolicy::ReleaseAll
            || matches!(policy, RetentionPolicy::Bounded { max_bytes } if self.retained_bytes() > max_bytes)
        {
            self.engine = Engine::new(self.engine.config, Default::default(), Stream::default());
        }
    }
    /// Releases all storage and clears abandoned protection, preserving policies.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let session = decoder.start(Default::default())?;
    /// std::mem::forget(session);
    /// assert!(matches!(decoder.start(Default::default()), Err(FramedDecodeError::AbandonedSession)));
    /// decoder.recover();
    /// let session = decoder.start(Default::default())?;
    /// assert_eq!(session.total_in(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recover(&mut self) {
        self.trim(RetentionPolicy::ReleaseAll);
        self.active = false;
    }
    /// Copies configuration and backend into an independent empty owner.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let independent = decoder.fork_empty();
    /// assert_eq!(independent.config(), decoder.config());
    /// assert_eq!(independent.retained_bytes(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fork_empty(&self) -> Self {
        Self {
            engine: Engine::new(self.engine.config, Default::default(), Stream::default()),
            backend: self.backend,
            retention: self.retention,
            active: false,
        }
    }
    /// Releases the active object; the one path both session shapes end on.
    pub(super) fn cancel(&mut self) {
        self.engine.clear();
        self.active = false;
        self.trim(self.retention);
    }
    fn begin<'d, 'dict>(
        &'d mut self,
        resolver: Option<DictionaryResolverRef<'dict>>,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedDecoderSession<'d, 'dict>, E> {
        self.begin_session(stream)?;
        Ok(FramedDecoderSession {
            owner: self,
            resolver,
        })
    }
    /// Claims the owner for one object after lifecycle and policy checks.
    ///
    /// The single start path behind borrowed and owned sessions.
    pub(super) fn begin_session(&mut self, stream: FramedDecodeStreamConfig) -> Result<(), E> {
        if self.active {
            return Err(E::AbandonedSession);
        }
        if let crate::OutputSize::Exact(expected) = stream.output_size() {
            super::core::check(
                expected,
                self.config().limits().max_output_bytes,
                FramedLimitKind::OutputBytes,
            )?;
        }
        if !self.engine.fits_policy() {
            self.recover();
        }
        self.engine.stream = stream;
        self.active = true;
        Ok(())
    }
    /// Starts one exclusive input object without external dictionaries.
    /// # Errors
    /// Rejects an abandoned session or an exact size exceeding output policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// {
    ///     let session = decoder.start(mbrotli::OutputSize::Exact(3).into())?;
    ///     assert_eq!(session.total_out(), 0);
    /// } // Drop cancels this operation and releases the owner borrow.
    /// let next = decoder.start(Default::default())?;
    /// assert_eq!(next.total_in(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        &mut self,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedDecoderSession<'_, 'static>, E> {
        self.begin(None, stream)
    }
    /// Starts one object with explicit framing dictionary lookup.
    /// Raw Auto input never invokes this resolver.
    /// # Errors
    /// As `start`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let mut session = decoder.start_with_dictionaries(&dictionaries, Default::default())?;
    /// let progress = session.process(&input, &mut [], mbrotli::DecodeOperation::Finish)?;
    /// assert!(matches!(progress.status, FramedDecoderStatus::Event(FramedEvent::StreamStart(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start_with_dictionaries<'d, 'dict>(
        &'d mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedDecoderSession<'d, 'dict>, E> {
        self.begin(Some(dictionaries.into()), stream)
    }
    /// Starts one object that takes ownership of this decoder.
    ///
    /// Validates and starts exactly as [`Self::start`] does, but the returned
    /// [`FramedDecoderSessionOwned`] owns the decoder instead of borrowing it.
    /// Recover the decoder with
    /// [`FramedDecoderSessionOwned::into_framed_decompressor`].
    /// # Errors
    /// As [`Self::start`]. The decoder is dropped with the error; use
    /// [`Self::start`] when it must survive a rejected start.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use mbrotli::DecodeOperation;
    /// let decoder = FramedDecompressor::new(Default::default())?;
    /// let mut session = decoder.into_session(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut output = [0; 16];
    /// let progress = session.process(&input, &mut output, DecodeOperation::Process)?;
    /// assert!(matches!(progress.status, FramedDecoderStatus::Event(FramedEvent::StreamStart(_))));
    /// let mut decoder = session.into_framed_decompressor();
    /// assert_eq!(decoder.decompress(&input)?.resources[0].data, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session(
        mut self,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedDecoderSessionOwned, E> {
        self.begin_session(stream)?;
        Ok(FramedDecoderSessionOwned {
            owner: self,
            resolver: None,
        })
    }
    /// Starts one owned object resolving external dictionaries through `dictionaries`.
    ///
    /// As [`Self::start_with_dictionaries`], but the session owns both the
    /// decoder and the resolver, so it carries no lifetime. Pass an
    /// `Arc` of a resolver to share it between sessions, or a `&'static` one.
    /// Raw Auto input never invokes the resolver.
    /// # Errors
    /// As [`Self::start`]. The decoder and resolver are dropped with the error.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::framing::*;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Arc::new(Dictionaries);
    /// let decoder = FramedDecompressor::new(Default::default())?;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let mut session =
    ///     decoder.into_session_with_dictionaries(Arc::clone(&dictionaries), Default::default())?;
    /// let progress = session.process(&input, &mut [], mbrotli::DecodeOperation::Finish)?;
    /// assert!(matches!(progress.status, FramedDecoderStatus::Event(FramedEvent::StreamStart(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session_with_dictionaries<R: DictionaryResolver + 'static>(
        mut self,
        dictionaries: R,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedDecoderSessionOwned<R>, E> {
        self.begin_session(stream)?;
        Ok(FramedDecoderSessionOwned {
            owner: self,
            resolver: Some(dictionaries),
        })
    }
    /// Returns independently owned resources after complete object validation.
    /// # Errors
    /// Rejects malformed input, trailing bytes, exhausted budgets, and failed allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let output = decoder.decompress(&input)?;
    /// assert_eq!(output.resources[0].data, b"abc");
    /// assert!(matches!(output.structure, OutputStructure::Framed { .. }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress(&mut self, src: &[u8]) -> Result<FramedOutput<Vec<u8>>, E> {
        self.owned(None, src)
    }
    /// Decodes with explicit external dictionary lookup in wire attachment order.
    /// # Errors
    /// As `decompress`, plus missing or malformed dictionary attachments.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let output = decoder.decompress_with_dictionaries(&dictionaries, &input)?;
    /// assert!(output.resources[0].data.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_with_dictionaries<'dict>(
        &mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        src: &[u8],
    ) -> Result<FramedOutput<Vec<u8>>, E> {
        self.owned(Some(dictionaries.into()), src)
    }
    fn owned(
        &mut self,
        dictionaries: Option<DictionaryResolverRef<'_>>,
        src: &[u8],
    ) -> Result<FramedOutput<Vec<u8>>, E> {
        let mut session = self.begin(dictionaries, Default::default())?;
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut p = 0;
        let mut output = [0; 8192];
        loop {
            let (consumed, produced, tag) = session
                .step(&src[p..], &mut output, DecodeOperation::Finish)
                .map_err(|e| e.error)?;
            p += consumed;
            match tag {
                Tag::Start => {
                    session.owner.engine.reserve_collection(&mut data)?;
                    data.push(Vec::new());
                }
                Tag::Data(_) => {
                    let bytes = data.last_mut().ok_or(E::InvalidState)?;
                    bytes
                        .try_reserve(produced)
                        .map_err(|_| E::AllocationFailed)?;
                    bytes.extend_from_slice(&output[..produced]);
                }
                Tag::Finished => {
                    if p != src.len() {
                        return Err(E::TrailingData);
                    }
                    return session.owner.engine.result(data);
                }
                _ => {}
            }
        }
    }
    /// Appends resource payload and returns absolute ranges. Every failure rolls
    /// back the original Vec length and preserves its prefix; capacity may grow.
    /// # Errors
    /// As `decompress`, including late directory, footer, and trailing-data errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut bytes = b"prefix".to_vec();
    /// let output = decoder.decompress_into(&input, &mut bytes)?;
    /// assert_eq!(output.resources[0].data, 6..9);
    /// assert_eq!(&bytes[output.resources[0].data.clone()], b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_into(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<FramedOutput<Range<usize>>, E> {
        self.append(None, src, dst)
    }
    /// Appends with explicit dictionary lookup and whole-operation rollback.
    /// # Errors
    /// As `decompress_with_dictionaries`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let mut bytes = b"prefix".to_vec();
    /// let output = decoder.decompress_with_dictionaries_into(&dictionaries, &input, &mut bytes)?;
    /// assert_eq!(output.resources[0].data, 6..6);
    /// assert_eq!(bytes, b"prefix");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_with_dictionaries_into<'dict>(
        &mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<FramedOutput<Range<usize>>, E> {
        self.append(Some(dictionaries.into()), src, dst)
    }
    fn append(
        &mut self,
        dictionaries: Option<DictionaryResolverRef<'_>>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<FramedOutput<Range<usize>>, E> {
        let original = dst.len();
        let result = (|| {
            let mut session = self.begin(dictionaries, Default::default())?;
            let mut data: Vec<Range<usize>> = Vec::new();
            let mut p = 0;
            let mut output = [0; 8192];
            loop {
                let (consumed, produced, tag) = session
                    .step(&src[p..], &mut output, DecodeOperation::Finish)
                    .map_err(|e| e.error)?;
                p += consumed;
                match tag {
                    Tag::Start => {
                        session.owner.engine.reserve_collection(&mut data)?;
                        data.push(dst.len()..dst.len());
                    }
                    Tag::Data(_) => {
                        dst.try_reserve(produced).map_err(|_| E::AllocationFailed)?;
                        dst.extend_from_slice(&output[..produced]);
                        data.last_mut().ok_or(E::InvalidState)?.end = dst.len();
                    }
                    Tag::Finished => {
                        if p != src.len() {
                            return Err(E::TrailingData);
                        }
                        return session.owner.engine.result(data);
                    }
                    _ => {}
                }
            }
        })();
        if result.is_err() {
            dst.truncate(original);
        }
        result
    }
    /// Writes from the start of `dst`, returning ranges relative to it.
    /// Unused suffix bytes are unchanged; errors preserve the written prefix.
    /// # Errors
    /// Returns cumulative progress, including the last fragment's precise identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut bytes = [0; 3];
    /// let output = decoder.decompress_to_slice(&input, &mut bytes)?;
    /// assert_eq!(output.resources[0].data, 0..3);
    /// assert_eq!(&bytes, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn decompress_to_slice(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<FramedOutput<Range<usize>>, FramedDecodeFailure> {
        self.slice(None, src, dst)
    }
    /// Writes to a slice with explicit external dictionaries.
    /// # Errors
    /// As `decompress_to_slice`, plus dictionary lookup and preparation failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let output = decoder.decompress_with_dictionaries_to_slice(&dictionaries, &input, &mut [])?;
    /// assert_eq!(output.resources[0].data, 0..0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn decompress_with_dictionaries_to_slice<'dict>(
        &mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<FramedOutput<Range<usize>>, FramedDecodeFailure> {
        self.slice(Some(dictionaries.into()), src, dst)
    }
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    fn slice(
        &mut self,
        dictionaries: Option<DictionaryResolverRef<'_>>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<FramedOutput<Range<usize>>, FramedDecodeFailure> {
        let mut p = 0;
        let mut written = 0;
        let mut last = None;
        let result = (|| {
            let mut session = self.begin(dictionaries, Default::default())?;
            let mut data: Vec<Range<usize>> = Vec::new();
            loop {
                let (consumed, produced, tag) =
                    match session.step(&src[p..], &mut dst[written..], DecodeOperation::Finish) {
                        Ok(progress) => progress,
                        Err(failure) => {
                            p += failure.consumed;
                            if let Some(mut fragment) = failure.last_output {
                                fragment.range.start += written;
                                fragment.range.end += written;
                                last = Some(fragment);
                            }
                            written += failure.produced;
                            return Err(failure.error);
                        }
                    };
                p += consumed;
                if let Tag::Data(position) = tag {
                    last = Some(ProducedFragment {
                        range: written..written + produced,
                        position,
                    });
                }
                written += produced;
                match tag {
                    Tag::Start => {
                        session.owner.engine.reserve_collection(&mut data)?;
                        data.push(written..written);
                    }
                    Tag::Data(_) => data.last_mut().ok_or(E::InvalidState)?.end = written,
                    Tag::Output => return Err(E::OutputTooSmall),
                    Tag::Finished => {
                        if p != src.len() {
                            return Err(E::TrailingData);
                        }
                        return session.owner.engine.result(data);
                    }
                    _ => {}
                }
            }
        })();
        result.map_err(|error| FramedDecodeFailure {
            error,
            consumed: p,
            produced: written,
            last_output: last,
        })
    }
}
