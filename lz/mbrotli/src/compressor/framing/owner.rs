use super::{
    core::{Container, driver::Driver, engine::Engine},
    *,
};
use crate::{Backend, Compressor, RetentionPolicy};
use ::core::ops::Range;
use FramedEncodeError as E;
use alloc::vec::Vec;

/// Reusable owner of raw compression and bounded framing storage.
#[derive(Debug)]
pub struct FramedCompressor {
    pub(super) raw: Compressor,
    pub(super) engine: Engine,
    config: FramedEncodeConfig,
    retention: RetentionPolicy,
    active: bool,
}
/// Construction settings; backend selection occurs once when built.
#[derive(Clone, Copy, Debug)]
pub struct FramedCompressorBuilder {
    config: FramedEncodeConfig,
    backend: Option<Backend>,
    retention: RetentionPolicy,
}
impl FramedCompressorBuilder {
    /// Selects a host-validated SIMD backend.
    pub const fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }
    /// Selects retention at the container boundary, rather than each resource.
    pub const fn with_retention(mut self, retention: RetentionPolicy) -> Self {
        self.retention = retention;
        self
    }
    /// Validates both policies and creates an empty owner without I/O.
    /// # Errors
    /// Returns invalid encoder settings or inconsistent framing profiles/budgets.
    pub fn build(self) -> Result<FramedCompressor, E> {
        Container::validate(*self.config.framing_config())?;
        let mut builder = Compressor::builder(*self.config.encoder_config());
        if let Some(backend) = self.backend {
            builder = builder.with_backend(backend);
        }
        Ok(FramedCompressor {
            raw: builder.build()?,
            engine: Engine::new(*self.config.framing_config()),
            config: self.config,
            retention: self.retention,
            active: false,
        })
    }
}
impl FramedCompressor {
    /// Creates an empty reusable framed encoder.
    /// # Errors
    /// Rejects invalid raw settings and framing profiles or budgets.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let bytes = encoder.compress(items.as_slice().into())?;
    /// assert_eq!(&bytes[..4], &[0x91, 10, 66, 82]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: FramedEncodeConfig) -> Result<Self, E> {
        Self::builder(config).build()
    }
    /// Begins configuration without allocating or selecting a backend.
    /// # Examples
    /// ```
    /// use mbrotli::{Backend, RetentionPolicy, framing::*};
    /// let encoder = FramedCompressor::builder(Default::default())
    ///     .with_backend(Backend::default())
    ///     .with_retention(RetentionPolicy::ReleaseAll)
    ///     .build()?;
    /// assert_eq!(encoder.retained_bytes(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn builder(config: FramedEncodeConfig) -> FramedCompressorBuilder {
        FramedCompressorBuilder {
            config,
            backend: None,
            retention: RetentionPolicy::Aggressive,
        }
    }
    /// Current resource encoder and framing configuration.
    pub const fn config(&self) -> &FramedEncodeConfig {
        &self.config
    }
    /// Retention policy applied at completion or cancellation.
    pub const fn retention(&self) -> RetentionPolicy {
        self.retention
    }
    /// Owned heap capacities, excluding destinations and borrowed dictionaries.
    pub fn retained_bytes(&self) -> usize {
        self.raw.retained_bytes() + self.engine.retained_bytes()
    }
    /// Validates before replacing policy and clearing abandoned protection.
    /// # Errors
    /// Invalid configuration leaves both configuration and protection unchanged.
    /// # Examples
    /// ```
    /// use mbrotli::{EncoderConfig, Quality, RetentionPolicy, framing::*};
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let items = [FramedItem::Resource(FramedResource::from(&b"payload"[..]))];
    /// let first = encoder.compress(items.as_slice().into())?;
    /// assert_eq!(first, encoder.compress(items.as_slice().into())?);
    /// encoder.reconfigure(encoder.config().with_encoder_config(
    ///     EncoderConfig::default().with_quality(Quality::Q1)))?;
    /// let independent = encoder.fork_empty();
    /// assert_eq!(independent.config(), encoder.config());
    /// encoder.trim(RetentionPolicy::ReleaseAll);
    /// assert_eq!(encoder.retained_bytes(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reconfigure(&mut self, config: FramedEncodeConfig) -> Result<(), E> {
        Container::validate(*config.framing_config())?;
        config.encoder_config().validate()?;
        let release = self.active
            || (config != self.config && self.retention == RetentionPolicy::CurrentConfig);
        if release {
            self.recover();
        } else {
            self.cancel();
        }
        self.raw.reconfigure(*config.encoder_config())?;
        self.config = config;
        self.engine.reconfigure(*config.framing_config());
        self.trim(self.retention);
        Ok(())
    }
    /// Applies storage policy without clearing forgotten-session protection.
    pub fn trim(&mut self, policy: RetentionPolicy) {
        if policy == RetentionPolicy::ReleaseAll
            || matches!(policy, RetentionPolicy::Bounded { max_bytes } if self.retained_bytes() > max_bytes)
        {
            self.engine.clear(&mut self.raw);
            self.raw.trim(RetentionPolicy::ReleaseAll);
            self.engine = Engine::new(*self.config.framing_config());
        }
    }
    /// Releases storage and cancels a forgotten operation, preserving policies.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// core::mem::forget(encoder.start(Default::default())?);
    /// assert!(matches!(encoder.start(Default::default()), Err(FramedEncodeError::AbandonedSession)));
    /// encoder.recover();
    /// let session = encoder.start(Default::default())?;
    /// assert_eq!(session.total_in(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recover(&mut self) {
        self.engine.clear(&mut self.raw);
        self.raw.recover();
        self.raw.trim(RetentionPolicy::ReleaseAll);
        self.engine = Engine::new(*self.config.framing_config());
        self.active = false;
    }
    /// Copies configuration and backend into an independent owner without buffers.
    pub fn fork_empty(&self) -> Self {
        Self {
            raw: self.raw.fork_empty(),
            engine: Engine::new(*self.config.framing_config()),
            config: self.config,
            retention: self.retention,
            active: false,
        }
    }
    /// Releases the active container; the one path both session shapes end on.
    pub(super) fn cancel(&mut self) {
        self.engine.clear(&mut self.raw);
        self.active = false;
        self.trim(self.retention);
    }
    /// Claims the owner for one container and queues its header.
    ///
    /// The single start path behind [`Self::start`] and [`Self::into_session`].
    pub(super) fn begin_session(&mut self, stream: FramedEncodeStreamConfig) -> Result<(), E> {
        if self.active {
            return Err(E::AbandonedSession);
        }
        if self.engine.retained_bytes() > self.config.framing_config().max_buffer_bytes {
            self.recover();
        }
        self.engine.start(stream)?;
        self.active = true;
        Ok(())
    }
    /// Starts one exclusive container and queues its header without I/O.
    /// # Errors
    /// Returns `AbandonedSession` after a forgotten guard, or an allocation failure.
    /// # Examples
    /// ```
    /// use mbrotli::{InputSize, Operation, framing::*};
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let mut session = encoder.start(InputSize::Exact(5).into())?;
    /// let mut wire = Vec::new();
    /// let mut output = [0; 1];
    /// loop {
    ///     let p = session.process(&mut output, FramedEncodeOperation::Process)?;
    ///     wire.extend_from_slice(&output[..p.produced]);
    ///     if p.status == FramedEncoderStatus::NeedsInput { break; }
    /// }
    /// {
    ///     let mut resource = session.resource(Default::default(), InputSize::Exact(5).into())?;
    ///     let mut input = &b"hel"[..];
    ///     loop {
    ///         let p = resource.process(input, &mut output, Operation::Process)?;
    ///         input = &input[p.consumed..];
    ///         wire.extend_from_slice(&output[..p.produced]);
    ///         if p.status == FramedEncoderStatus::NeedsInput { break; }
    ///     }
    ///     let mut input = &b"lo"[..];
    ///     loop {
    ///         let p = resource.process(input, &mut output, Operation::Finish)?;
    ///         input = &input[p.consumed..];
    ///         wire.extend_from_slice(&output[..p.produced]);
    ///         if p.status == FramedEncoderStatus::Finished { break; }
    ///     }
    /// }
    /// session.metadata(MetadataKind::Footer, &[
    ///     MetadataField { code: *b"AB", value: b"footer annotation" },
    /// ])?;
    /// loop {
    ///     let p = session.process(&mut output, FramedEncodeOperation::Finish)?;
    ///     wire.extend_from_slice(&output[..p.produced]);
    ///     if p.status == FramedEncoderStatus::Finished { break; }
    /// }
    /// assert_eq!(session.resources_encoded(), 1);
    /// assert_eq!(session.total_out(), wire.len() as u64);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        &mut self,
        stream: FramedEncodeStreamConfig,
    ) -> Result<FramedEncoderSession<'_>, E> {
        self.begin_session(stream)?;
        Ok(FramedEncoderSession { owner: self })
    }
    /// Starts one container that takes ownership of this encoder.
    ///
    /// Validates and starts exactly as [`Self::start`] does, but the returned
    /// [`FramedEncoderSessionOwned`] owns the encoder instead of borrowing it.
    /// Recover the encoder with [`FramedEncoderSessionOwned::into_framed_compressor`].
    /// # Errors
    /// As [`Self::start`]. The encoder is dropped with the error; use
    /// [`Self::start`] when it must survive a rejected start.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let encoder = FramedCompressor::new(Default::default())?;
    /// let mut session = encoder.into_session(Default::default())?;
    /// let mut output = [0; 128];
    /// let progress = session.process(&mut output, FramedEncodeOperation::Process)?;
    /// assert_eq!(&output[..4], &[0x91, 10, 66, 82]);
    /// assert_eq!(progress.status, FramedEncoderStatus::NeedsInput);
    /// // The unfinished container is cancelled; the encoder starts afresh.
    /// let mut encoder = session.into_framed_compressor();
    /// assert_eq!(encoder.start(Default::default())?.total_out(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session(
        mut self,
        stream: FramedEncodeStreamConfig,
    ) -> Result<FramedEncoderSessionOwned, E> {
        self.begin_session(stream)?;
        Ok(FramedEncoderSessionOwned { owner: self })
    }
    /// Encodes borrowed resources and metadata as one finalized container.
    /// # Errors
    /// Reports malformed commands, length mismatches, limits or encoding failures.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let fields = [MetadataField { code: *b"id", value: b"dictionary.bin" }];
    /// let items = [
    ///     FramedItem::Metadata { kind: MetadataKind::Resource, fields: &fields, options: Default::default() },
    ///     FramedItem::Resource(FramedResource {
    ///         data: b"hidden prefix bytes",
    ///         options: ResourceOptions { hidden: true, ..Default::default() },
    ///         stream: Default::default(), encoding: ResourceEncoding::Uncompressed,
    ///     }),
    ///     FramedItem::Resource(FramedResource::from(&b"visible resource"[..])),
    /// ];
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let bytes = encoder.compress(items.as_slice().into())?;
    /// assert_eq!(&bytes[..4], &[0x91, 10, 66, 82]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress(&mut self, input: FramedInput<'_>) -> Result<Vec<u8>, E> {
        let mut output = Vec::new();
        self.compress_into(input, &mut output)?;
        Ok(output)
    }
    /// Appends a canonical container. Every error restores the original length.
    /// # Errors
    /// As `compress`, including allocation or late suffix errors; prefix is preserved.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
    /// let mut output = b"caller prefix".to_vec();
    /// let range = encoder.compress_into(items.as_slice().into(), &mut output)?;
    /// assert_eq!(&output[..range.start], b"caller prefix");
    /// assert_eq!(&output[range.start..range.start + 4], &[0x91, 10, 66, 82]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_into(
        &mut self,
        input: FramedInput<'_>,
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, E> {
        let start = dst.len();
        let result = (|| {
            let stream = Driver::aggregate(input)?.into();
            let mut driver = Driver::new(self.start(stream)?, input);
            let mut buffer = [0; 8192];
            loop {
                let p = driver
                    .process(&mut buffer)
                    .map_err(FramedEncodeFailure::into_error)?;
                dst.try_reserve(p.produced)
                    .map_err(|_| E::AllocationFailed)?;
                dst.extend_from_slice(&buffer[..p.produced]);
                if p.status == FramedEncoderStatus::Finished {
                    return Ok(start..dst.len());
                }
            }
        })();
        if result.is_err() {
            dst.truncate(start);
        }
        result
    }
    /// Writes the canonical stream into the slice prefix without whole-container staging.
    /// # Errors
    /// Returns cumulative accepted input and written prefix; untouched tail stays intact.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
    /// let mut output = [0xa5; 512];
    /// let written = encoder.compress_to_slice(items.as_slice().into(), &mut output)?;
    /// assert!(output[written..].iter().all(|byte| *byte == 0xa5));
    /// let failure = encoder.compress_to_slice(items.as_slice().into(), &mut output[..2]).unwrap_err();
    /// assert_eq!(failure.produced, 2);
    /// assert!(matches!(failure.into_error(), FramedEncodeError::OutputTooSmall));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_to_slice(
        &mut self,
        input: FramedInput<'_>,
        dst: &mut [u8],
    ) -> Result<usize, FramedEncodeFailure> {
        let stream = match Driver::aggregate(input) {
            Ok(size) => size.into(),
            Err(error) => return Err(self.engine.failure(error, 0, 0)),
        };
        let session = self.start(stream).map_err(|error| FramedEncodeFailure {
            error,
            consumed: 0,
            produced: 0,
            location: Default::default(),
        })?;
        let mut driver = Driver::new(session, input);
        let mut consumed = 0;
        let mut produced = 0;
        loop {
            let p = driver.process(&mut dst[produced..]).map_err(|mut e| {
                e.consumed += consumed;
                e.produced += produced;
                e
            })?;
            consumed += p.consumed;
            produced += p.produced;
            if p.status == FramedEncoderStatus::Finished {
                return Ok(produced);
            }
            if produced == dst.len() {
                return Err(driver.failure(E::OutputTooSmall, consumed, produced));
            }
        }
    }
}
