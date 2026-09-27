//! The stateful, reusable encoder.

use alloc::vec::Vec;

use super::config::{ConfigError, EncoderConfig, SizeOverflow};
use super::core::bound::{append_reserve, bound};
use super::core::driver::{
    EncoderCache, compress_to_slice_attached, compress_to_vec_attached, quality_reads_a_prefix,
};
use super::dictionary::PreparedDictionary;
use super::error::EncodeError;
use super::internal::{CompressParams, QualityLevel, WindowBits};
use super::session::{EncoderSession, EncoderSessionOwned, StreamConfig};
use ::core::ops::Range;
use fearless_simd::Level;

/// The configuration whose compressed-size bound is the loosest.
///
/// The bound is driven by how often the per-meta-block reservation is paid,
/// which is largest when the blocks are smallest: a ten-bit window at a quality
/// that cuts its input at the window. Every other configuration fits inside
/// this one, which is what lets the bound be an associated function.
const WIDEST_BOUND: CompressParams = CompressParams::new(QualityLevel::Q0, WindowBits::MIN);

use crate::RetentionPolicy;

/// A reusable Brotli encoder.
///
/// A compressor owns its configuration, the instruction set it resolved once at
/// construction, and every buffer the encoders need. All of that is reused:
/// a call at a given shape allocates nothing an earlier one did not already
/// pay for, except that a match finder may take its full table on the
/// second stream once the compressor has shown it is reused.
///
/// Every encoding method takes `&mut self`, because every one of them advances
/// state the compressor owns. That is deliberate: one compressor belongs to one
/// worker, and the borrow checker is what says so. For parallel compression,
/// build one compressor per worker — a lock around a single one would serialise
/// the compression itself, not merely the access. An immutable
/// [`PreparedDictionary`] can be shared by all of them.
///
/// There is no lock, no atomic and no interior mutability inside a compressor.
///
/// # Examples
///
/// Repeated compression into a reused destination, which is the shape the whole
/// type is built around:
///
/// ```
/// use mbrotli::{Compressor, EncoderConfig, Quality};
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
/// let mut output = Vec::new();
///
/// for payload in [&b"first payload"[..], b"second payload", b"third payload"] {
///     output.clear();
///     let range = encoder.compress_into(payload, &mut output)?;
///     assert_eq!(range, 0..output.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Compressor {
    /// The instruction set, resolved once.
    level: Level,
    /// The validated configuration every operation runs under.
    config: EncoderConfig,
    /// What to keep between operations.
    retention: RetentionPolicy,
    /// The retained encoder.
    pub(crate) workspace: EncoderCache,
    /// Input accepted by a session but not yet handed to the encoder.
    pub(crate) staging: Vec<u8>,
    /// Encoded bytes not yet delivered to the caller.
    pub(crate) pending: Vec<u8>,
    /// How much of [`Compressor::pending`] has already been delivered.
    pub(crate) served: usize,
    /// Whether a session is, or was left, in flight.
    pub(crate) active: bool,
}

impl Compressor {
    /// Creates a compressor for `config`.
    ///
    /// Validates the whole configuration, resolves the instruction set once,
    /// and creates an empty workspace. Nothing large is allocated: the match
    /// finder and the window are built when the first operation shows how much
    /// input there is.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] for a combination that is individually legal but
    /// jointly meaningless — today, a Large Window at quality zero, one or two,
    /// whose distance model cannot carry one.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, ConfigError, EncoderConfig, Quality, Window};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// assert!(!encoder.compress(b"payload payload")?.is_empty());
    ///
    /// let refused = EncoderConfig::default()
    ///     .with_quality(Quality::Q1)
    ///     .with_window(Window::large(30)?);
    /// assert!(matches!(
    ///     Compressor::new(refused),
    ///     Err(ConfigError::LargeWindowUnsupportedForQuality { quality: Quality::Q1 })
    /// ));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn new(config: EncoderConfig) -> Result<Self, ConfigError> {
        Self::builder(config).build()
    }

    /// Starts building a compressor whose retention policy is chosen too.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, RetentionPolicy};
    ///
    /// let encoder = Compressor::builder(EncoderConfig::default())
    ///     .with_retention(RetentionPolicy::ReleaseAll)
    ///     .build()?;
    ///
    /// assert_eq!(encoder.retention(), RetentionPolicy::ReleaseAll);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn builder(config: EncoderConfig) -> CompressorBuilder {
        CompressorBuilder {
            config,
            retention: RetentionPolicy::Aggressive,
            level: None,
        }
    }

    /// Returns the configuration this compressor encodes under.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let config = EncoderConfig::default().with_quality(Quality::Q5);
    /// let encoder = Compressor::new(config)?;
    ///
    /// assert_eq!(*encoder.config(), config);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Returns what this compressor keeps between operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, RetentionPolicy};
    ///
    /// let encoder = Compressor::new(EncoderConfig::default())?;
    ///
    /// assert_eq!(encoder.retention(), RetentionPolicy::Aggressive);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn retention(&self) -> RetentionPolicy {
        self.retention
    }

    /// Replaces the configuration, keeping whatever buffers still apply.
    ///
    /// Transactional: the new configuration is validated first, and the old one
    /// is left untouched if it is rejected. On success every trace of the
    /// previous stream is gone, and no buffer built for the old shape is reused
    /// as if it held valid data.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] for a configuration [`Compressor::new`] would
    /// also refuse. The compressor is unchanged in that case.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality, Window};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let first = encoder.compress(b"payload payload payload")?;
    ///
    /// encoder.reconfigure(EncoderConfig::default().with_quality(Quality::Q9))?;
    /// assert_eq!(encoder.config().quality(), Quality::Q9);
    ///
    /// // A rejected configuration changes nothing.
    /// let refused = EncoderConfig::default()
    ///     .with_quality(Quality::Q0)
    ///     .with_window(Window::large(30)?);
    /// assert!(encoder.reconfigure(refused).is_err());
    /// assert_eq!(encoder.config().quality(), Quality::Q9);
    ///
    /// // And going back reproduces the original stream exactly.
    /// encoder.reconfigure(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// assert_eq!(encoder.compress(b"payload payload payload")?, first);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reconfigure(&mut self, config: EncoderConfig) -> Result<(), ConfigError> {
        config.validate()?;
        let changed = self.config != config;
        self.config = config;
        self.reset_stream_state();
        if changed && matches!(self.retention, RetentionPolicy::CurrentConfig) {
            self.workspace.invalidate();
        }
        Ok(())
    }

    /// Returns a compressed size no stream of `input_size` bytes can exceed.
    ///
    /// The bound holds for every configuration, so a buffer sized by it fits
    /// whatever the compressor is set to — which is why it needs no compressor
    /// to compute. It is therefore looser than the stream any one configuration
    /// actually produces; a caller who wants to spend less memory should use
    /// [`Compressor::compress_into`] and let the destination grow.
    ///
    /// # Errors
    ///
    /// Returns [`SizeOverflow`] when the bound does not fit in a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let payload = b"a payload to be compressed into a caller-owned buffer".repeat(20);
    ///
    /// let mut buffer = vec![0u8; Compressor::max_compressed_size(payload.len())?];
    /// let written = encoder.compress_to_slice(&payload, &mut buffer)?;
    ///
    /// assert_eq!(&buffer[..written], encoder.compress(&payload)?.as_slice());
    /// assert!(Compressor::max_compressed_size(usize::MAX).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_compressed_size(input_size: usize) -> Result<usize, SizeOverflow> {
        match bound(&WIDEST_BOUND, input_size) {
            Ok(bound) => Ok(bound),
            Err(_) => Err(SizeOverflow),
        }
    }

    /// Compresses `src` into a freshly allocated stream.
    ///
    /// A convenience over [`Compressor::compress_into`], which is the method to
    /// reach for when the destination can be reused.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::AllocationFailed`] when the destination cannot be
    /// reserved, and propagates any encoder failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let payload = "brotli ".repeat(1000);
    ///
    /// let compressed = encoder.compress(payload.as_bytes())?;
    ///
    /// assert_eq!(compressed.len(), 41);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn compress(&mut self, src: &[u8]) -> Result<Vec<u8>, EncodeError> {
        let mut output = Vec::new();
        self.compress_into(src, &mut output)?;
        Ok(output)
    }

    /// Appends a complete stream to `dst` and returns the range it occupies.
    ///
    /// The primary one-shot entry point. It reuses the compressor's workspace
    /// and the destination's capacity, so a warm compressor writing into a
    /// destination that is already big enough allocates nothing at all.
    ///
    /// Transactional: whatever `dst` held before the call is still there
    /// afterwards, byte for byte, whether the call succeeded or failed. On
    /// failure `dst` is truncated back to the length it had.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::AllocationFailed`] when `dst` cannot be grown,
    /// [`EncodeError::Bound`] when the reservation overflows, and propagates
    /// any encoder failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut output = b"a prefix the caller already had".to_vec();
    /// let start = output.len();
    ///
    /// let range = encoder.compress_into(b"payload payload payload", &mut output)?;
    ///
    /// assert_eq!(range.start, start);
    /// assert_eq!(range.end, output.len());
    /// assert_eq!(&output[..start], b"a prefix the caller already had");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn compress_into(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, EncodeError> {
        self.compress_attached_into(None, src, dst)
    }

    /// Compresses `src` into `dst`, returning how many bytes were written.
    ///
    /// Writes from `dst[0]`. Size `dst` with
    /// [`Compressor::max_compressed_size`] to be sure it fits.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::OutputTooSmall`] when `dst` cannot hold the whole
    /// stream. The contents of `dst` are then unspecified — the encoder writes
    /// as it goes and does not buffer a whole stream to be able to undo it —
    /// but the compressor itself is left ready for the next operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncodeError, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut buffer = vec![0u8; Compressor::max_compressed_size(5)?];
    ///
    /// let written = encoder.compress_to_slice(b"aaaaa", &mut buffer)?;
    /// assert_eq!(&buffer[..written], encoder.compress(b"aaaaa")?.as_slice());
    ///
    /// let mut cramped = [0u8; 1];
    /// assert!(matches!(
    ///     encoder.compress_to_slice(b"aaaaa", &mut cramped),
    ///     Err(EncodeError::OutputTooSmall { provided: 1 })
    /// ));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn compress_to_slice(&mut self, src: &[u8], dst: &mut [u8]) -> Result<usize, EncodeError> {
        self.compress_attached_to_slice(None, src, dst)
    }

    /// Compresses `src` against `dictionary` into a freshly allocated stream.
    ///
    /// Nothing about the dictionary is written into the stream: a decoder has to
    /// be given the same bytes, in the same order, out of band.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::DictionaryUnsupportedForQuality`] below quality
    /// five, where no match finder can consult a dictionary — refused rather
    /// than silently ignored. Otherwise as [`Compressor::compress`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"the quick brown fox jumps over the lazy dog"[..])
    ///     .build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    ///
    /// let payload = b"the quick brown fox jumps over the lazy dog";
    /// assert!(
    ///     encoder.compress_with_dictionary(&dictionary, payload)?.len()
    ///         < encoder.compress(payload)?.len()
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_with_dictionary(
        &mut self,
        dictionary: &PreparedDictionary,
        src: &[u8],
    ) -> Result<Vec<u8>, EncodeError> {
        let mut output = Vec::new();
        self.compress_with_dictionary_into(dictionary, src, &mut output)?;
        Ok(output)
    }

    /// Appends a stream compressed against `dictionary` to `dst`.
    ///
    /// [`Compressor::compress_into`] with a dictionary attached, and
    /// transactional in exactly the same way.
    ///
    /// # Errors
    ///
    /// As [`Compressor::compress_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let mut output = Vec::new();
    ///
    /// let range = encoder.compress_with_dictionary_into(&dictionary, b"a common prefix", &mut output)?;
    ///
    /// assert_eq!(range, 0..output.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_with_dictionary_into(
        &mut self,
        dictionary: &PreparedDictionary,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, EncodeError> {
        self.compress_attached_into(Some(dictionary), src, dst)
    }

    /// Compresses `src` against `dictionary` into `dst`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::compress_to_slice`] and
    /// [`Compressor::compress_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let mut buffer = vec![0u8; Compressor::max_compressed_size(15)?];
    ///
    /// let written =
    ///     encoder.compress_with_dictionary_to_slice(&dictionary, b"a common prefix", &mut buffer)?;
    ///
    /// assert_eq!(
    ///     &buffer[..written],
    ///     encoder.compress_with_dictionary(&dictionary, b"a common prefix")?.as_slice()
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_with_dictionary_to_slice(
        &mut self,
        dictionary: &PreparedDictionary,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, EncodeError> {
        self.compress_attached_to_slice(Some(dictionary), src, dst)
    }

    /// Starts an incremental stream.
    ///
    /// The session borrows the compressor for as long as it lives. See
    /// [`EncoderSession`] for how the two differ from the one-shot entry
    /// points.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::UnsupportedStreamOffset`] for a non-zero stream
    /// offset, [`EncodeError::AbandonedSession`] when a previous session was
    /// leaked rather than dropped, and propagates encoder construction
    /// failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut session = encoder.start(Default::default())?;
    /// let mut output = [0u8; 512];
    ///
    /// let progress = session.process(b"streamed payload", &mut output, Operation::Finish)?;
    ///
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        &mut self,
        stream: StreamConfig,
    ) -> Result<EncoderSession<'_, 'static>, EncodeError> {
        let limit = self.begin(None, stream)?;
        Ok(EncoderSession::new(self, None, limit, stream))
    }

    /// Starts an incremental stream compressed against `dictionary`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start`], plus
    /// [`EncodeError::DictionaryUnsupportedForQuality`] below quality five.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let mut session = encoder.start_with_dictionary(&dictionary, Default::default())?;
    /// let mut output = [0u8; 512];
    ///
    /// let progress = session.process(b"a common prefix", &mut output, Operation::Finish)?;
    ///
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start_with_dictionary<'c, 'd>(
        &'c mut self,
        dictionary: &'d PreparedDictionary,
        stream: StreamConfig,
    ) -> Result<EncoderSession<'c, 'd>, EncodeError> {
        let limit = self.begin(Some(dictionary), stream)?;
        Ok(EncoderSession::new(self, Some(dictionary), limit, stream))
    }

    /// Starts an incremental stream that takes ownership of this compressor.
    ///
    /// Validates and starts the stream exactly as [`Compressor::start`] does,
    /// but the returned [`EncoderSessionOwned`] owns the compressor rather than
    /// borrowing it. Recover the compressor with
    /// [`EncoderSessionOwned::into_compressor`].
    ///
    /// # Errors
    ///
    /// As [`Compressor::start`]. The compressor is dropped with the error; use
    /// [`Compressor::start`] when it must survive a failed start.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let compressor = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut session = compressor.into_session(Default::default())?;
    /// let mut output = [0u8; 512];
    ///
    /// let progress = session.process(b"streamed payload", &mut output, Operation::Finish)?;
    ///
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// let _compressor = session.into_compressor();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session(
        mut self,
        stream: StreamConfig,
    ) -> Result<EncoderSessionOwned, EncodeError> {
        let limit = self.begin(None, stream)?;
        Ok(EncoderSessionOwned::new(self, None, limit, stream))
    }

    /// Starts an owned incremental stream compressed against `dictionary`.
    ///
    /// The session takes `dictionary` by value, so it carries no lifetime.
    /// Pass an `Arc<PreparedDictionary>` to share one dictionary between
    /// sessions without copying its payload.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start_with_dictionary`]. The compressor is dropped with
    /// the error.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, EncoderStatus, Operation, Quality};
    ///
    /// let dictionary = Arc::new(DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?);
    /// let compressor = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let mut session =
    ///     compressor.into_session_with_dictionary(Arc::clone(&dictionary), Default::default())?;
    /// let mut output = [0u8; 512];
    ///
    /// let progress = session.process(b"a common prefix", &mut output, Operation::Finish)?;
    ///
    /// assert_eq!(progress.status, EncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session_with_dictionary<D: AsRef<PreparedDictionary> + 'static>(
        mut self,
        dictionary: D,
        stream: StreamConfig,
    ) -> Result<EncoderSessionOwned<D>, EncodeError> {
        let limit = self.begin(Some(dictionary.as_ref()), stream)?;
        Ok(EncoderSessionOwned::new(
            self,
            Some(dictionary),
            limit,
            stream,
        ))
    }

    /// Returns how many bytes this compressor is keeping allocated.
    ///
    /// Counts the retained encoder — its window, match finder, command and
    /// output buffers — together with the streaming staging buffers. It does
    /// not count a dictionary, which the compressor does not own.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// assert_eq!(encoder.retained_bytes(), 0);
    ///
    /// encoder.compress(b"payload payload payload")?;
    /// assert!(encoder.retained_bytes() > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.workspace.retained_bytes() + self.staging.capacity() + self.pending.capacity()
    }

    /// Applies `policy` once, without changing the compressor's own policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality, RetentionPolicy};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q9))?;
    /// encoder.compress(b"payload payload payload")?;
    ///
    /// encoder.trim(RetentionPolicy::Bounded { max_bytes: 0 });
    /// assert_eq!(encoder.retained_bytes(), 0);
    /// // The compressor still works, and still produces the same bytes.
    /// assert!(!encoder.compress(b"payload payload payload")?.is_empty());
    /// assert_eq!(encoder.retention(), RetentionPolicy::Aggressive);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn trim(&mut self, policy: RetentionPolicy) {
        let release = match policy {
            RetentionPolicy::Aggressive | RetentionPolicy::CurrentConfig => false,
            RetentionPolicy::Bounded { max_bytes } => self.retained_bytes() > max_bytes,
            RetentionPolicy::ReleaseAll => true,
        };
        if release {
            self.workspace.invalidate();
            self.staging = Vec::new();
            self.pending = Vec::new();
            self.served = 0;
        }
    }

    /// Puts the compressor back into a known state after a leaked session.
    ///
    /// Only needed when a session was passed to [`::core::mem::forget`], which
    /// skips the cleanup dropping it would have done. Ordinary use never needs
    /// this.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncodeError, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// ::core::mem::forget(encoder.start(Default::default())?);
    ///
    /// assert!(matches!(encoder.compress(b"payload"), Err(EncodeError::AbandonedSession)));
    ///
    /// encoder.recover();
    /// assert!(!encoder.compress(b"payload")?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recover(&mut self) {
        self.workspace.invalidate();
        self.reset_stream_state();
    }

    /// Returns another compressor with this one's settings and no buffers.
    ///
    /// The way to give a worker its own compressor without repeating the
    /// configuration. Nothing large is copied: the new compressor starts as
    /// cold as [`Compressor::new`] would leave it.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// encoder.compress(b"payload payload payload")?;
    ///
    /// let mut worker = encoder.fork_empty();
    /// assert_eq!(worker.config(), encoder.config());
    /// assert_eq!(worker.retained_bytes(), 0);
    /// assert_eq!(
    ///     worker.compress(b"payload payload payload")?,
    ///     encoder.compress(b"payload payload payload")?
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn fork_empty(&self) -> Self {
        Self {
            level: self.level,
            config: self.config,
            retention: self.retention,
            workspace: EncoderCache::default(),
            staging: Vec::new(),
            pending: Vec::new(),
            served: 0,
            active: false,
        }
    }

    /// Runs the one-shot vector path with an optional dictionary.
    fn compress_attached_into(
        &mut self,
        dictionary: Option<&PreparedDictionary>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, EncodeError> {
        self.ensure_available()?;
        if dictionary.is_some() {
            self.check_dictionary()?;
        }
        let start = dst.len();
        let params = self.config.lower(Some(src.len()));
        let reserve = append_reserve(&params, src.len()).map_err(|_| SizeOverflow)?;
        dst.try_reserve(reserve)
            .map_err(|_| EncodeError::AllocationFailed { requested: reserve })?;

        let attached = dictionary.map(PreparedDictionary::inner);
        let outcome =
            compress_to_vec_attached(&mut self.workspace, self.level, &params, attached, src, dst);
        self.finish_operation();
        match outcome {
            Ok(()) => Ok(start..dst.len()),
            Err(error) => {
                dst.truncate(start);
                Err(EncodeError::from_core(error, 0))
            }
        }
    }

    /// Runs the one-shot slice path with an optional dictionary.
    fn compress_attached_to_slice(
        &mut self,
        dictionary: Option<&PreparedDictionary>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, EncodeError> {
        self.ensure_available()?;
        if dictionary.is_some() {
            self.check_dictionary()?;
        }
        let params = self.config.lower(Some(src.len()));
        let attached = dictionary.map(PreparedDictionary::inner);
        let provided = dst.len();
        let outcome = compress_to_slice_attached(
            &mut self.workspace,
            self.level,
            &params,
            attached,
            src,
            dst,
        );
        self.finish_operation();
        outcome.map_err(|error| EncodeError::from_core(error, provided))
    }

    /// Prepares the compressor for a new session and returns its block size.
    pub(crate) fn begin(
        &mut self,
        dictionary: Option<&PreparedDictionary>,
        stream: StreamConfig,
    ) -> Result<usize, EncodeError> {
        self.ensure_available()?;
        if stream.stream_offset() != 0
            && (!cfg!(feature = "experimental") || self.config.quality().get() < 2)
        {
            return Err(EncodeError::UnsupportedStreamOffset {
                offset: stream.stream_offset(),
            });
        }
        if dictionary.is_some() {
            self.check_dictionary()?;
        }

        let params = self.config.lower(Some(stream.input_size().hint()));
        #[cfg(feature = "experimental")]
        let params = {
            let input_bytes = match stream.input_size() {
                super::InputSize::Exact(size) => size,
                super::InputSize::Unknown => 0,
            };
            if stream
                .stream_offset()
                .checked_add(input_bytes)
                .is_none_or(|end| end > (1u64 << 63) - 1)
            {
                return Err(EncodeError::StreamPositionOverflow {
                    position: stream.stream_offset(),
                    input_bytes,
                });
            }
            let mut params = params;
            let bits = self.config.window().bits().min(30);
            params.stream_offset = stream.stream_offset().min((1u64 << bits) - 16) as usize;
            params
        };
        let limit = match self.workspace.acquire(self.level, &params, 0) {
            Ok(encoder) => encoder.block_size_limit(),
            Err(error) => {
                self.workspace.invalidate();
                return Err(EncodeError::from_core(error, 0));
            }
        };

        self.staging.clear();
        self.pending.clear();
        self.served = 0;
        if self.staging.capacity() < limit {
            self.staging
                .try_reserve(limit)
                .map_err(|_| EncodeError::AllocationFailed { requested: limit })?;
        }
        self.active = true;
        Ok(limit)
    }

    /// Refuses to start anything while an abandoned session is unaccounted for.
    const fn ensure_available(&self) -> Result<(), EncodeError> {
        if self.active {
            return Err(EncodeError::AbandonedSession);
        }
        Ok(())
    }

    /// Refuses a dictionary at a quality whose match finder cannot read one.
    const fn check_dictionary(&self) -> Result<(), EncodeError> {
        if quality_reads_a_prefix(self.config.quality().level()) {
            return Ok(());
        }
        Err(EncodeError::DictionaryUnsupportedForQuality {
            quality: self.config.quality(),
        })
    }

    /// Applies the retention policy once an operation has finished.
    pub(crate) fn finish_operation(&mut self) {
        self.trim(self.retention);
    }

    /// Drops every trace of a stream, keeping capacity.
    fn reset_stream_state(&mut self) {
        self.staging.clear();
        self.pending.clear();
        self.served = 0;
        self.active = false;
    }

    /// Returns whether encoded bytes are still waiting to be delivered.
    pub(crate) const fn has_pending(&self) -> bool {
        self.served < self.pending.len()
    }
}

/// Builds a [`Compressor`], choosing what it keeps and which backend it uses.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncoderConfig, Quality, RetentionPolicy};
///
/// let encoder = Compressor::builder(EncoderConfig::default().with_quality(Quality::Q5))
///     .with_retention(RetentionPolicy::Bounded { max_bytes: 8 << 20 })
///     .build()?;
///
/// assert_eq!(encoder.config().quality(), Quality::Q5);
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[derive(Debug)]
pub struct CompressorBuilder {
    /// The configuration the compressor will encode under.
    config: EncoderConfig,
    /// What the compressor will keep between operations.
    retention: RetentionPolicy,
    /// The backend to pin, when the caller chose one.
    level: Option<Level>,
}

impl CompressorBuilder {
    /// Sets what the compressor keeps between operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, RetentionPolicy};
    ///
    /// let encoder = Compressor::builder(EncoderConfig::default())
    ///     .with_retention(RetentionPolicy::CurrentConfig)
    ///     .build()?;
    ///
    /// assert_eq!(encoder.retention(), RetentionPolicy::CurrentConfig);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn with_retention(mut self, retention: RetentionPolicy) -> Self {
        self.retention = retention;
        self
    }

    /// Pins the instruction set instead of detecting it.
    ///
    /// Every backend produces identical bytes, so this changes speed and
    /// nothing else. It exists so that a test can exercise a backend the host
    /// would not have chosen, and so that a caller who has already detected one
    /// need not detect it again.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Backend;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let config = EncoderConfig::default().with_quality(Quality::Q1);
    /// let mut detected = Compressor::new(config)?;
    /// let mut selected = Compressor::builder(config).with_backend(Backend::default()).build()?;
    ///
    /// assert_eq!(selected.compress(b"identical bytes")?, detected.compress(b"identical bytes")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_backend(mut self, backend: super::Backend) -> Self {
        self.level = Some(backend.0);
        self
    }

    /// Validates the configuration and creates the compressor.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] for a configuration [`Compressor::new`] would
    /// also refuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig};
    ///
    /// let encoder = Compressor::builder(EncoderConfig::default()).build()?;
    ///
    /// assert_eq!(encoder.retained_bytes(), 0);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[inline]
    pub fn build(self) -> Result<Compressor, ConfigError> {
        self.config.validate()?;
        Ok(Compressor {
            level: self.level.unwrap_or_else(|| super::Backend::default().0),
            config: self.config,
            retention: self.retention,
            workspace: EncoderCache::default(),
            staging: Vec::new(),
            pending: Vec::new(),
            served: 0,
            active: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::config::{Quality, Window};

    fn compressor(quality: Quality) -> Compressor {
        Compressor::new(EncoderConfig::default().with_quality(quality)).expect("a legal config")
    }

    #[cfg(feature = "no_std")]
    #[test]
    fn default_construction_uses_the_compile_time_backend() {
        let encoder = compressor(Quality::Q5);
        assert_eq!(
            super::super::Backend(encoder.level),
            super::super::Backend(Level::baseline())
        );
    }

    #[test]
    fn a_compressor_is_send_and_movable() {
        const fn assert_send<T: Send>() {}
        assert_send::<Compressor>();
        assert_send::<CompressorBuilder>();

        let mut moved = std::thread::spawn(|| {
            let mut encoder = compressor(Quality::Q1);
            encoder.compress(b"payload payload").expect("compressed")
        })
        .join()
        .expect("the worker finished");
        assert!(!moved.is_empty());
        moved.clear();
    }

    #[test]
    fn the_widest_bound_covers_every_configuration() {
        // The associated bound has to hold for the configuration with the most
        // meta-blocks per byte, which is what `WIDEST_BOUND` names.
        for quality in 0u8..=11 {
            for bits in [10u8, 16, 22, 24] {
                let config = EncoderConfig::default()
                    .with_quality(Quality::try_from(quality).expect("legal"))
                    .with_window(Window::standard(bits).expect("legal"));
                for size in [0usize, 1, 1024, 1 << 16] {
                    let specific =
                        bound(&config.lower(Some(size)), size).expect("a representable bound");
                    let widest = Compressor::max_compressed_size(size).expect("representable");
                    assert!(
                        widest >= specific,
                        "q{quality} w{bits} at {size} bytes: {widest} < {specific}"
                    );
                }
            }
        }
        assert!(Compressor::max_compressed_size(usize::MAX).is_err());
    }

    #[test]
    fn a_retention_policy_releases_what_it_says_it_will() {
        let mut encoder = compressor(Quality::Q5);
        encoder.compress(b"payload payload payload").expect("ok");
        let warm = encoder.retained_bytes();
        assert!(warm > 0);

        encoder.trim(RetentionPolicy::Aggressive);
        assert_eq!(encoder.retained_bytes(), warm);
        encoder.trim(RetentionPolicy::CurrentConfig);
        assert_eq!(encoder.retained_bytes(), warm);
        encoder.trim(RetentionPolicy::Bounded {
            max_bytes: usize::MAX,
        });
        assert_eq!(encoder.retained_bytes(), warm);
        encoder.trim(RetentionPolicy::Bounded { max_bytes: 0 });
        assert_eq!(encoder.retained_bytes(), 0);

        encoder.compress(b"payload payload payload").expect("ok");
        assert!(encoder.retained_bytes() > 0);
        encoder.trim(RetentionPolicy::ReleaseAll);
        assert_eq!(encoder.retained_bytes(), 0);
    }

    #[test]
    fn the_release_all_policy_retains_nothing_across_calls() {
        let mut encoder = Compressor::builder(EncoderConfig::default().with_quality(Quality::Q5))
            .with_retention(RetentionPolicy::ReleaseAll)
            .build()
            .expect("legal");
        let first = encoder.compress(b"payload payload payload").expect("ok");
        assert_eq!(encoder.retained_bytes(), 0);
        assert_eq!(
            encoder.compress(b"payload payload payload").expect("ok"),
            first
        );
    }

    #[test]
    fn reconfiguring_to_the_same_shape_keeps_the_workspace() {
        let mut encoder = compressor(Quality::Q5);
        encoder.compress(b"payload payload payload").expect("ok");
        let warm = encoder.retained_bytes();

        encoder
            .reconfigure(EncoderConfig::default().with_quality(Quality::Q5))
            .expect("legal");
        assert_eq!(encoder.retained_bytes(), warm);
    }

    #[test]
    fn the_current_config_policy_releases_on_a_real_change() {
        let mut encoder = Compressor::builder(EncoderConfig::default().with_quality(Quality::Q5))
            .with_retention(RetentionPolicy::CurrentConfig)
            .build()
            .expect("legal");
        encoder.compress(b"payload payload payload").expect("ok");
        assert!(encoder.retained_bytes() > 0);

        encoder
            .reconfigure(EncoderConfig::default().with_quality(Quality::Q5))
            .expect("legal");
        assert!(encoder.retained_bytes() > 0, "an identical config released");

        encoder
            .reconfigure(EncoderConfig::default().with_quality(Quality::Q1))
            .expect("legal");
        assert_eq!(encoder.retained_bytes(), 0);
    }

    #[test]
    fn a_dictionary_is_refused_below_the_quality_that_can_read_one() {
        for quality in 0u8..=11 {
            let encoder = compressor(Quality::try_from(quality).expect("legal"));
            let outcome = encoder.check_dictionary();
            if quality >= 5 {
                assert!(outcome.is_ok(), "q{quality} refused a dictionary");
            } else {
                assert!(
                    matches!(
                        outcome,
                        Err(EncodeError::DictionaryUnsupportedForQuality { .. })
                    ),
                    "q{quality} accepted a dictionary"
                );
            }
        }
    }

    #[test]
    fn a_forked_compressor_copies_the_settings_and_none_of_the_buffers() {
        let mut encoder = Compressor::builder(EncoderConfig::default().with_quality(Quality::Q5))
            .with_retention(RetentionPolicy::ReleaseAll)
            .build()
            .expect("legal");
        encoder.staging.extend_from_slice(&[0u8; 64]);

        let forked = encoder.fork_empty();
        assert_eq!(forked.config(), encoder.config());
        assert_eq!(forked.retention(), RetentionPolicy::ReleaseAll);
        assert_eq!(forked.retained_bytes(), 0);
    }
}
