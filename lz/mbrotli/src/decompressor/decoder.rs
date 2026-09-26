use super::{
    DecodeConfigError, DecodeError, DecodeOperation, DecodeStreamConfig, DecoderConfig,
    DecoderSession, DecoderSessionOwned, DecoderStatus, core::Stream,
};
use crate::Backend;
use crate::RetentionPolicy;
use crate::dictionary::{DecodeDictionary, DictionaryRef};
use ::core::ops::Range;
use alloc::vec::Vec;

/// Reusable Brotli decoder with exclusive per-operation access.
///
/// Keep this object alive to reuse workspace between payloads. One-shot methods
/// require complete input; use [`Self::start`] for incremental decoding. The
/// default retention policy keeps allocations after each operation.
/// A fresh owned decode can transfer output storage into its returned vector;
/// that storage then belongs to the caller. Use [`Self::decompress_to_slice`]
/// or a preallocated [`Self::decompress_into`] destination to retain decoder
/// storage from the first operation.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecoderConfig, Decompressor};
/// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// assert_eq!(decoder.decompress(&compressed)?, b"hello");
/// assert_eq!(decoder.decompress(&compressed)?, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Decompressor {
    pub(super) config: DecoderConfig,
    pub(super) backend: Backend,
    retention: RetentionPolicy,
    pub(super) workspace: Stream,
    pub(super) active: bool,
}

/// Decoder construction with an optional backend and retention policy.
///
/// Obtain this from [`Decompressor::builder`]. The backend specializes the
/// command loop and its SIMD history copies; parsing and errors stay identical.
///
/// # Examples
///
/// ```
/// use mbrotli::{Backend, DecoderConfig, Decompressor, RetentionPolicy};
/// let mut decoder = Decompressor::builder(DecoderConfig::default())
///     .with_backend(Backend::default())
///     .with_retention(RetentionPolicy::ReleaseAll)
///     .build()?;
/// assert!(decoder.decompress(&[0x3b])?.is_empty());
/// assert_eq!(decoder.retention(), RetentionPolicy::ReleaseAll);
/// assert_eq!(decoder.retained_bytes(), 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DecompressorBuilder {
    config: DecoderConfig,
    backend: Option<Backend>,
    retention: RetentionPolicy,
}

impl DecompressorBuilder {
    /// Sets the policy applied after each session is dropped.
    pub const fn with_retention(mut self, value: RetentionPolicy) -> Self {
        self.retention = value;
        self
    }
    /// Selects an already host-validated execution backend.
    pub const fn with_backend(mut self, value: Backend) -> Self {
        self.backend = Some(value);
        self
    }
    /// Constructs an empty decoder without allocating a history window.
    ///
    /// # Errors
    /// Configuration values are validated at construction; current typed
    /// configurations have no additional cross-field restrictions.
    #[inline]
    pub fn build(self) -> Result<Decompressor, DecodeConfigError> {
        Ok(Decompressor {
            config: self.config,
            backend: self.backend.unwrap_or_default(),
            retention: self.retention,
            workspace: Stream::default(),
            active: false,
        })
    }
}

impl Decompressor {
    /// Constructs a reusable decoder with default retention and host backend.
    /// See [`Decompressor`] for a decoding example, or [`DecompressorBuilder`]
    /// to select a retention policy.
    ///
    /// # Errors
    /// Returns invalid configuration errors, as [`DecompressorBuilder::build`].
    #[inline]
    pub fn new(config: DecoderConfig) -> Result<Self, DecodeConfigError> {
        Self::builder(config).build()
    }
    /// Starts configuring a decoder without detecting CPU capabilities yet.
    pub const fn builder(config: DecoderConfig) -> DecompressorBuilder {
        DecompressorBuilder {
            config,
            backend: None,
            retention: RetentionPolicy::Aggressive,
        }
    }
    /// Returns the reusable policy.
    pub const fn config(&self) -> &DecoderConfig {
        &self.config
    }
    /// Returns the configured retention policy.
    pub const fn retention(&self) -> RetentionPolicy {
        self.retention
    }
    /// Changes policy and clears abandoned or previous stream state.
    ///
    /// Retains compatible workspace according to the configured retention
    /// policy. Current typed configurations have no cross-field restrictions.
    ///
    /// # Errors
    /// Currently always succeeds, as [`DecompressorBuilder::build`] does.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor, MemberMode};
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// decoder.reconfigure(DecoderConfig::default().with_member_mode(MemberMode::Concatenated))?;
    /// assert_eq!(decoder.config().member_mode(), MemberMode::Concatenated);
    /// assert!(decoder.decompress(&[0x3b, 0x3b])?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reconfigure(&mut self, config: DecoderConfig) -> Result<(), DecodeConfigError> {
        if self.active
            || self.retention == RetentionPolicy::ReleaseAll
            || (config != self.config && self.retention == RetentionPolicy::CurrentConfig)
        {
            self.recover();
        }
        self.config = config;
        self.workspace.reset(config);
        self.active = false;
        Ok(())
    }
    /// Returns live retained decoder heap storage, excluding caller output.
    pub const fn retained_bytes(&self) -> usize {
        self.workspace.retained_bytes()
    }
    /// Applies a retention policy once, preserving abandoned-session protection.
    ///
    /// This does not change [`Self::retention`]. `Bounded` releases all workspace
    /// when its limit is exceeded; `ReleaseAll` always releases it. `Aggressive`
    /// and `CurrentConfig` keep current storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor, RetentionPolicy};
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// decoder.decompress(&compressed)?;
    /// decoder.trim(RetentionPolicy::ReleaseAll);
    /// assert_eq!(decoder.retained_bytes(), 0);
    /// assert_eq!(decoder.retention(), RetentionPolicy::Aggressive);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn trim(&mut self, policy: RetentionPolicy) {
        if policy == RetentionPolicy::ReleaseAll
            || matches!(policy, RetentionPolicy::Bounded { max_bytes } if self.retained_bytes() > max_bytes)
        {
            self.workspace = Stream::default();
        }
    }
    /// Clears abandoned sessions and releases all owned workspace.
    ///
    /// Configuration and retention policy are preserved. Ordinary session drop
    /// already permits reuse; recovery is needed after a forgotten session.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeError, DecoderConfig, Decompressor};
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// core::mem::forget(decoder.start(Default::default())?);
    /// assert!(matches!(decoder.decompress(&[0x3b]), Err(DecodeError::AbandonedSession)));
    /// decoder.recover();
    /// assert_eq!(decoder.retained_bytes(), 0);
    /// assert!(decoder.decompress(&[0x3b])?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recover(&mut self) {
        self.workspace = Stream::default();
        self.active = false;
    }
    /// Copies configuration into an independent object without copying storage.
    ///
    /// Copies the backend and retention policy too, but no stream progress.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// let decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut other = decoder.fork_empty();
    /// assert_eq!(other.config(), decoder.config());
    /// assert_eq!(other.retention(), decoder.retention());
    /// assert_eq!(other.retained_bytes(), 0);
    /// assert!(other.decompress(&[0x3b])?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fork_empty(&self) -> Self {
        Self {
            config: self.config,
            backend: self.backend,
            retention: self.retention,
            workspace: Stream::default(),
            active: false,
        }
    }
    /// Borrows this decoder for an incremental operation.
    /// See [`DecoderSession::process`] for a complete incremental example.
    ///
    /// # Errors
    /// Rejects abandoned sessions and exact sizes exceeding the output budget.
    pub fn start(
        &mut self,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderSession<'_, 'static>, DecodeError> {
        DecoderSession::start(self, stream, None)
    }
    /// Starts an incremental operation that takes ownership of this decoder.
    ///
    /// Validates and starts exactly as [`Self::start`] does, but the returned
    /// [`DecoderSessionOwned`] owns the decoder rather than borrowing it.
    /// Recover the decoder with [`DecoderSessionOwned::into_decompressor`].
    ///
    /// # Errors
    /// As [`Self::start`]. The decoder is dropped with the error; use
    /// [`Self::start`] when it must survive a rejected start.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let decoder = Decompressor::new(Default::default())?;
    /// let mut session = decoder.into_session(Default::default())?;
    /// let progress = session.process(&[0x3b], &mut [], DecodeOperation::Finish)?;
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// let _decoder = session.into_decompressor();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session(
        self,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderSessionOwned, DecodeError> {
        DecoderSessionOwned::start(self, stream, None)
    }
    /// Decodes all input into a new vector. Single mode rejects trailing data.
    ///
    /// Empty input is truncated, not an empty Brotli member. See [`Decompressor`]
    /// for a successful decode and [`super::MemberMode`] for concatenated input.
    ///
    /// # Errors
    /// Returns codec, resource, lifecycle, or trailing-data errors.
    pub fn decompress(&mut self, src: &[u8]) -> Result<Vec<u8>, DecodeError> {
        // A complete stored member needs neither a session nor history. Keep
        // resource-policy and abandoned-session error ordering in the driver.
        if !self.active
            && self.config.member_mode() == super::MemberMode::Single
            && self.config.limits() == super::DecodeLimits::default()
            && let Some(payload) = super::core::stored_payload(src, self.config.window_limit())
        {
            let mut dst = Vec::new();
            dst.try_reserve_exact(payload.len())
                .map_err(|_| DecodeError::AllocationFailed)?;
            dst.extend_from_slice(payload);
            self.trim(self.retention);
            return Ok(dst);
        }
        let mut dst = Vec::new();
        self.decompress_into(src, &mut dst)?;
        Ok(dst)
    }
    /// Appends a decoded operation, rolling back its entire append on error.
    ///
    /// Returns the range of newly appended bytes, preserving the original
    /// prefix. Rollback restores the length and contents, but may retain an
    /// allocation grown during the failed operation.
    ///
    /// # Errors
    /// Returns codec, resource, lifecycle, allocation, or trailing-data errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut output = b"prefix: ".to_vec();
    /// let range = decoder.decompress_into(&compressed, &mut output)?;
    /// assert_eq!(&output[range], b"hello");
    /// assert!(decoder.decompress_into(&compressed[..4], &mut output).is_err());
    /// assert_eq!(output, b"prefix: hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_into(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, DecodeError> {
        self.decode_into(None, src, dst)
    }

    fn decode_into(
        &mut self,
        dictionary: Option<DictionaryRef<'_>>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, DecodeError> {
        let start = dst.len();
        // A fresh owned destination can take the history allocation. Retained
        // workspaces, appends, dictionaries and concatenation use normal delivery.
        let collect = start == 0
            && dst.capacity() == 0
            && self.retained_bytes() == 0
            && dictionary.is_none()
            && self.config.member_mode() == super::MemberMode::Single;
        let result = (|| {
            let mut session =
                DecoderSession::start(self, DecodeStreamConfig::default(), dictionary)?;
            let mut consumed = 0;
            let mut written = 0;
            // Parse up to the first byte of output with no destination at all,
            // so the first reservation can use the size the meta-block
            // declares instead of guessing; the tail is trimmed at the end.
            let probe = session
                .process(src, &mut [], DecodeOperation::Finish)
                .map_err(super::DecodeFailure::into_error)?;
            consumed += probe.consumed;
            if probe.status == DecoderStatus::Finished {
                if consumed != src.len() {
                    return Err(DecodeError::TrailingData {
                        offset: consumed as u64,
                    });
                }
                return Ok(start..start);
            }
            if collect {
                let progress = session
                    .collect(&src[consumed..])
                    .map_err(super::DecodeFailure::into_error)?;
                consumed += progress.consumed;
                if progress.status == DecoderStatus::Finished {
                    if consumed != src.len() {
                        return Err(DecodeError::TrailingData {
                            offset: consumed as u64,
                        });
                    }
                    *dst = session.take_collected();
                    return Ok(0..dst.len());
                }
                // A member larger than its window needs ordinary streaming
                // delivery from here; retain history for future references.
                dst.try_reserve(session.collected().len())
                    .map_err(|_| DecodeError::AllocationFailed)?;
                dst.extend_from_slice(session.collected());
                written = dst.len();
            }
            let mut chunk = src
                .len()
                .saturating_mul(4)
                .clamp(256, 1 << 16)
                .max(session.declared_remaining())
                .min(1 << 24);
            loop {
                dst.try_reserve(written + chunk)
                    .map_err(|_| DecodeError::AllocationFailed)?;
                dst.resize(start + written + chunk, 0);
                let progress = session
                    .process(
                        &src[consumed..],
                        &mut dst[start + written..],
                        DecodeOperation::Finish,
                    )
                    .map_err(super::DecodeFailure::into_error)?;
                consumed += progress.consumed;
                written += progress.produced;
                dst.truncate(start + written);
                if progress.status == DecoderStatus::Finished {
                    if consumed != src.len() {
                        return Err(DecodeError::TrailingData {
                            offset: consumed as u64,
                        });
                    }
                    return Ok(start..dst.len());
                }
                // Grow to what the meta-block declares when that is known,
                // otherwise geometrically, so one reservation usually covers
                // the rest of a block instead of a chain of copies.
                chunk = chunk
                    .saturating_mul(2)
                    .max(session.declared_remaining())
                    .min(1 << 24);
            }
        })();
        if result.is_err() {
            dst.truncate(start);
        }
        result
    }
    /// Decodes into a fixed slice, leaving its unused suffix unchanged.
    ///
    /// The first member decodes straight into `dst`, which also serves as its
    /// history, so no output passes through the decoder's window.
    ///
    /// # Errors
    /// Returns codec/resource errors, trailing data, or `OutputTooSmall`.
    /// Bytes written before an error are not rolled back. `OutputTooSmall`
    /// fills the whole slice; after another error the bytes past the decoded
    /// prefix may hold part of the failing meta-block.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecodeError, DecoderConfig, Decompressor};
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut output = [0xaa; 8];
    /// let written = decoder.decompress_to_slice(&compressed, &mut output)?;
    /// assert_eq!(&output[..written], b"hello");
    /// assert_eq!(&output[written..], &[0xaa; 3]);
    /// let mut small = [0; 2];
    /// assert!(matches!(decoder.decompress_to_slice(&compressed, &mut small),
    ///     Err(DecodeError::OutputTooSmall { written: 2 })));
    /// assert_eq!(&small, b"he");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_to_slice(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, DecodeError> {
        self.decode_to_slice(None, src, dst)
    }

    fn decode_to_slice(
        &mut self,
        dictionary: Option<DictionaryRef<'_>>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let mut session = DecoderSession::start(self, DecodeStreamConfig::default(), dictionary)?;
        // The session ends with this call, so its first member may use `dst`
        // itself as history instead of a ring it would copy out of.
        let progress = session
            .finish_linear(src, dst)
            .map_err(super::DecodeFailure::into_error)?;
        if progress.status == DecoderStatus::NeedsOutput {
            return Err(DecodeError::OutputTooSmall {
                written: progress.produced,
            });
        }
        if progress.consumed != src.len() {
            return Err(DecodeError::TrailingData {
                offset: progress.consumed as u64,
            });
        }
        Ok(progress.produced)
    }
}

impl Decompressor {
    /// Starts a session borrowing its dictionary independently of the decoder.
    ///
    /// The dictionary must match the encoder's effective attachments and remain
    /// alive until the session is dropped. See [`crate::dictionary::DictionaryRef`]
    /// for constructing the borrowed view and [`DecoderSession::process`] for
    /// driving the session.
    ///
    /// # Errors
    /// Rejects an abandoned session or an exact size exceeding the output budget.
    pub fn start_with_dictionary<'d, 'dict>(
        &'d mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderSession<'d, 'dict>, DecodeError> {
        DecoderSession::start(self, stream, Some(dictionary.into()))
    }

    /// Starts an owned operation decoding against an external dictionary.
    ///
    /// As [`Self::start_with_dictionary`], but the returned
    /// [`DecoderSessionOwned`] owns both the decoder and `dictionary`, so it
    /// carries no lifetime. Pass an `Arc<DecodeDictionary>` to share one
    /// dictionary between sessions without copying its payload.
    ///
    /// # Errors
    /// As [`Self::start_with_dictionary`]. The decoder is dropped with the
    /// error.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::dictionary::{DecodeDictionary, DictionaryAttachment};
    /// use mbrotli::{DecodeOperation, DecoderStatus, Decompressor};
    /// let dictionary = Arc::new(DecodeDictionary::new(
    ///     &[DictionaryAttachment::Raw(b"prefix")], Default::default())?);
    /// let decoder = Decompressor::new(Default::default())?;
    /// let mut session =
    ///     decoder.into_session_with_dictionary(Arc::clone(&dictionary), Default::default())?;
    /// let progress = session.process(&[0x3b], &mut [], DecodeOperation::Finish)?;
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session_with_dictionary<D: AsRef<DecodeDictionary> + 'static>(
        self,
        dictionary: D,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderSessionOwned<D>, DecodeError> {
        DecoderSessionOwned::start(self, stream, Some(dictionary))
    }

    /// Decodes a stream using the supplied effective external dictionary.
    ///
    /// Accepts a borrowed [`crate::dictionary::DecodeDictionary`] or, with
    /// `compression`, a borrowed `PreparedDictionary`. No dictionary payload
    /// is copied by this operation. See [`crate::dictionary::DecodeDictionary`]
    /// for a round-trip example using an external prefix.
    ///
    /// # Errors
    /// Returns codec, resource, allocation, lifecycle or trailing-data errors.
    /// Raw Brotli cannot always detect an incorrect external dictionary.
    pub fn decompress_with_dictionary<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
    ) -> Result<Vec<u8>, DecodeError> {
        let mut output = Vec::new();
        self.decode_into(Some(dictionary.into()), src, &mut output)?;
        Ok(output)
    }

    /// Appends dictionary-decoded bytes and rolls back the append on any error.
    ///
    /// # Errors
    /// Returns the same failures as [`Self::decompress_with_dictionary`].
    pub fn decompress_with_dictionary_into<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, DecodeError> {
        self.decode_into(Some(dictionary.into()), src, dst)
    }

    /// Decodes with an external dictionary into an exact or larger slice.
    ///
    /// As [`Self::decompress_to_slice`], the first member uses `dst` as its
    /// history and a successful decode leaves the unused suffix unchanged.
    ///
    /// # Errors
    /// Returns codec/resource failures, trailing data or `OutputTooSmall`.
    /// Any written prefix remains available on error; after an error other
    /// than `OutputTooSmall` the bytes past it may hold part of the failing
    /// meta-block.
    pub fn decompress_with_dictionary_to_slice<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, DecodeError> {
        self.decode_to_slice(Some(dictionary.into()), src, dst)
    }
}
