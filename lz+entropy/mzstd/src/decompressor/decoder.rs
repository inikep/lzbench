use ::core::mem::MaybeUninit;
use ::core::ops::Range;
use alloc::vec::Vec;

use super::core::{DictView, Stream};
use super::{
    DecodeError, DecodeFailure, DecodeOperation, DecodeProgress, DecoderConfig, DecoderSession,
    DecoderSessionOwned, DecoderStatus,
};
use crate::dictionary::{DecodeDictionary, DictionaryRef};
use crate::{Backend, RetentionPolicy};

/// A reusable Zstandard decoder.
///
/// Keep it alive to reuse its buffers between payloads; its
/// [`RetentionPolicy`] decides what stays allocated after each operation.
/// One-shot methods take complete input; [`Self::start`] and
/// [`Self::into_session`] decode incrementally, and with `std` its
/// `reader` and `writer` methods (in the `decompressor::io` module) adapt
/// it to `std::io`.
///
/// # Examples
///
/// ```
/// use mzstd::Decompressor;
/// // "hello" as one raw block in a single-segment frame.
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
/// let mut decoder = Decompressor::default();
/// assert_eq!(decoder.decompress(&frame)?, b"hello");
/// assert_eq!(decoder.decompress(&[frame, frame].concat())?, b"hellohello");
/// # Ok::<(), mzstd::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct Decompressor {
    pub(super) config: DecoderConfig,
    retention: RetentionPolicy,
    pub(super) stream: Stream,
}

/// Builds a [`Decompressor`] with non-default options.
///
/// Obtained from [`Decompressor::builder`].
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeLimits, DecoderConfig, Decompressor, FrameMode, RetentionPolicy};
/// let config = DecoderConfig::default()
///     .with_frame_mode(FrameMode::Single)
///     .with_limits(DecodeLimits::default().with_max_window_log(23)?);
/// let decoder = Decompressor::builder(config)
///     .with_retention(RetentionPolicy::ReleaseAll)
///     .build();
/// assert_eq!(decoder.config().frame_mode(), FrameMode::Single);
/// assert_eq!(decoder.retention(), RetentionPolicy::ReleaseAll);
/// # Ok::<(), mzstd::DecodeConfigError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecompressorBuilder {
    config: DecoderConfig,
    retention: RetentionPolicy,
    backend: Option<Backend>,
}

impl DecompressorBuilder {
    /// Sets what is kept allocated between operations.
    #[must_use]
    pub const fn with_retention(mut self, retention: RetentionPolicy) -> Self {
        self.retention = retention;
        self
    }

    /// Selects the SIMD backend instead of the best one the host supports.
    ///
    /// Every backend produces identical output and accepts exactly the
    /// same frames; this exists for reproducible measurements and
    /// differential tests.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Backend, Decompressor};
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// for backend in Backend::available() {
    ///     let mut decoder = Decompressor::builder(Default::default())
    ///         .with_backend(backend)
    ///         .build();
    ///     assert_eq!(decoder.backend(), backend);
    ///     assert_eq!(decoder.decompress(&frame)?, b"ok");
    /// }
    /// # Ok::<(), mzstd::DecodeError>(())
    /// ```
    #[must_use]
    pub const fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Builds the decoder; nothing is allocated until it decodes.
    #[must_use]
    pub fn build(self) -> Decompressor {
        Decompressor {
            config: self.config,
            retention: self.retention,
            stream: Stream::new(self.config, self.backend.unwrap_or_default()),
        }
    }
}

impl Default for Decompressor {
    /// A decoder with [`DecoderConfig::DEFAULT`].
    fn default() -> Self {
        Self::new(DecoderConfig::DEFAULT)
    }
}

impl From<DecoderConfig> for Decompressor {
    fn from(config: DecoderConfig) -> Self {
        Self::new(config)
    }
}

impl Decompressor {
    /// A decoder with `config` that keeps its buffers between operations;
    /// nothing is allocated until it decodes.
    #[must_use]
    pub fn new(config: DecoderConfig) -> Self {
        Self::builder(config).build()
    }

    /// Starts building a decoder with non-default options.
    #[must_use]
    pub const fn builder(config: DecoderConfig) -> DecompressorBuilder {
        DecompressorBuilder {
            config,
            retention: RetentionPolicy::Aggressive,
            backend: None,
        }
    }

    /// The SIMD backend the decoder runs on.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Backend, Decompressor};
    /// assert_eq!(Decompressor::default().backend(), Backend::default());
    /// ```
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.stream.backend()
    }

    /// The decoder's policy.
    #[must_use]
    pub const fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// The retention policy applied after each operation.
    #[must_use]
    pub const fn retention(&self) -> RetentionPolicy {
        self.retention
    }

    /// Replaces the policy for later operations, keeping allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Decompressor, FrameMode};
    /// let mut decoder = Decompressor::default();
    /// decoder.reconfigure(FrameMode::Single.into());
    /// assert_eq!(decoder.config().frame_mode(), FrameMode::Single);
    /// ```
    pub fn reconfigure(&mut self, config: DecoderConfig) {
        self.config = config;
        self.stream.reset(config);
    }

    /// Heap bytes the decoder keeps between operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Decompressor;
    /// let mut decoder = Decompressor::default();
    /// assert_eq!(decoder.retained_bytes(), 0);
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00, 0x09, 0x00, 0x00, b'o'];
    /// decoder.start().process(&frame, &mut [0; 4], Default::default())?;
    /// assert!(decoder.retained_bytes() > 0);
    /// # Ok::<(), mzstd::DecodeFailure>(())
    /// ```
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.stream.retained_bytes()
    }

    /// Releases buffers that `policy` does not allow to keep.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Decompressor, RetentionPolicy};
    /// let mut decoder = Decompressor::default();
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00, 0x09, 0x00, 0x00, b'o'];
    /// decoder.start().process(&frame, &mut [0; 4], Default::default())?;
    /// decoder.trim(RetentionPolicy::Bounded { max_bytes: 1 << 30 });
    /// assert!(decoder.retained_bytes() > 0);
    /// decoder.trim(RetentionPolicy::ReleaseAll);
    /// assert_eq!(decoder.retained_bytes(), 0);
    /// # Ok::<(), mzstd::DecodeFailure>(())
    /// ```
    pub fn trim(&mut self, policy: RetentionPolicy) {
        if !policy.allows(self.retained_bytes()) {
            self.stream.release();
        }
    }

    /// Applies the configured retention policy between operations.
    pub(super) fn apply_retention(&mut self) {
        self.trim(self.retention);
    }

    /// Starts an incremental operation borrowing this decoder.
    ///
    /// See [`DecoderSession::process`] for a complete loop.
    pub fn start(&mut self) -> DecoderSession<'_> {
        DecoderSession::new(self, None)
    }

    /// Starts an incremental operation borrowing this decoder and
    /// `dictionary`, which every frame of the operation starts from.
    ///
    /// The dictionary must be the one the frames were compressed with; see
    /// [`DecodeDictionary`] for how IDs are checked. It stays borrowed until
    /// the session is dropped. See [`DecoderSession::process`] for a
    /// complete loop and [`DictionaryRef`] for an example.
    pub fn start_with_dictionary<'d>(
        &'d mut self,
        dictionary: impl Into<DictionaryRef<'d>>,
    ) -> DecoderSession<'d> {
        DecoderSession::new(self, Some(dictionary.into().view()))
    }

    /// Starts an incremental operation that owns this decoder.
    ///
    /// See [`DecoderSessionOwned`] for an example.
    #[must_use]
    pub fn into_session(self) -> DecoderSessionOwned {
        DecoderSessionOwned::new(self, None)
    }

    /// Starts an incremental operation that owns this decoder and
    /// `dictionary`, which every frame of the operation starts from.
    ///
    /// Pass an `Arc<DecodeDictionary>` to share one dictionary between
    /// sessions without copying it; the session then carries no lifetime.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
    ///
    /// let dictionary = Arc::new(DecodeDictionary::try_from(&b"history"[..])?);
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut session = Decompressor::default().into_session_with_dictionary(Arc::clone(&dictionary));
    /// let mut output = [0u8; 8];
    /// let progress = session.process(&frame, &mut output, DecodeOperation::Finish)?;
    /// assert_eq!(progress.status, DecoderStatus::Finished);
    /// assert_eq!(&output[..progress.produced], b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_session_with_dictionary<D: AsRef<DecodeDictionary> + 'static>(
        self,
        dictionary: D,
    ) -> DecoderSessionOwned<D> {
        DecoderSessionOwned::new(self, Some(dictionary))
    }

    /// Decodes complete input into a new vector.
    ///
    /// In [`FrameMode::Concatenated`](super::FrameMode::Concatenated) the
    /// input must consist of whole frames, at least one; in
    /// [`FrameMode::Single`](super::FrameMode::Single) of exactly one
    /// frame, which may be a skippable one (decoding to nothing), and
    /// nothing else.
    ///
    /// # Errors
    ///
    /// Returns format, checksum, content-size, limit and allocation errors,
    /// [`DecodeError::UnexpectedEndOfInput`] for empty or truncated input,
    /// and [`DecodeError::TrailingData`] for bytes after a single frame.
    ///
    /// # Examples
    ///
    /// See [`Decompressor`].
    pub fn decompress(&mut self, src: &[u8]) -> Result<Vec<u8>, DecodeError> {
        let mut dst = Vec::new();
        self.decompress_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Decodes complete input onto the end of `dst`, returning the range of
    /// the appended bytes.
    ///
    /// The vector itself serves as the frames' history, so output is never
    /// copied. On error `dst` is truncated back to its original length,
    /// though it may keep capacity it grew.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Decompressor;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut output = b"status: ".to_vec();
    /// let range = decoder.decompress_into(&frame, &mut output)?;
    /// assert_eq!(&output[range], b"ok");
    /// assert!(decoder.decompress_into(&frame[..10], &mut output).is_err());
    /// assert_eq!(output, b"status: ok");
    /// # Ok::<(), mzstd::DecodeError>(())
    /// ```
    pub fn decompress_into(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, DecodeError> {
        self.decode_into(src, dst, None)
    }

    /// [`Self::decompress_into`] with an optional dictionary.
    fn decode_into(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
        dictionary: Option<DictView<'_>>,
    ) -> Result<Range<usize>, DecodeError> {
        let start = dst.len();
        self.stream.reset(self.config);
        let result = self.stream.decode_all(src, dst, dictionary);
        self.apply_retention();
        match result {
            Ok(()) => Ok(start..dst.len()),
            Err(error) => {
                dst.truncate(start);
                Err(error)
            }
        }
    }

    /// Decodes complete input into `dst`, returning the number of bytes
    /// written; the rest of `dst` is left unchanged.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress`], plus [`DecodeError::OutputTooSmall`] when
    /// the output does not fit; `dst` is then completely filled with the
    /// output's prefix. After other errors `dst` may hold a partial result.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{DecodeError, Decompressor};
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut output = [0xaa; 4];
    /// let written = decoder.decompress_to_slice(&frame, &mut output)?;
    /// assert_eq!(&output[..written], b"ok");
    /// assert_eq!(&output[written..], &[0xaa, 0xaa]);
    /// assert!(matches!(
    ///     decoder.decompress_to_slice(&frame, &mut [0; 1]),
    ///     Err(DecodeError::OutputTooSmall { written: 1 })
    /// ));
    /// # Ok::<(), mzstd::DecodeError>(())
    /// ```
    pub fn decompress_to_slice(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, DecodeError> {
        self.decode_to_slice(src, dst, None)
    }

    /// [`Self::decompress_to_slice`] with an optional dictionary.
    fn decode_to_slice(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        dictionary: Option<DictView<'_>>,
    ) -> Result<usize, DecodeError> {
        // Frames that declare sizes that fit decode straight into `dst`;
        // anything else, errors included, takes the incremental path, which
        // defines this method's results.
        self.stream.reset(self.config);
        let direct = self.stream.decode_to_slice(src, dst, dictionary);
        if let Some(written) = direct {
            self.apply_retention();
            return Ok(written);
        }
        let mut session = DecoderSession::new(self, dictionary);
        one_shot_outcome(
            session.process(src, dst, DecodeOperation::Finish),
            src.len(),
        )
    }

    /// [`Self::decompress_to_slice`] into uninitialized memory, such as
    /// [`Vec::spare_capacity_mut`], without zeroing it first.
    ///
    /// It returns the same counts and errors as
    /// [`Self::decompress_to_slice`]. On success exactly `dst[..written]`
    /// is initialized and no byte after it has been written. After
    /// [`DecodeError::OutputTooSmall`] all of `dst` holds the output's
    /// prefix; after other errors treat `dst` as uninitialized.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress_to_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Decompressor;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut output: Vec<u8> = Vec::with_capacity(16);
    /// let written = decoder.decompress_to_uninit(&frame, output.spare_capacity_mut())?;
    /// // SAFETY: `decompress_to_uninit` initialized the first `written` bytes.
    /// unsafe { output.set_len(written) };
    /// assert_eq!(output, b"ok");
    /// # Ok::<(), mzstd::DecodeError>(())
    /// ```
    pub fn decompress_to_uninit(
        &mut self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        self.decode_to_uninit(src, dst, None)
    }

    /// [`Self::decompress_to_uninit`] with an optional dictionary.
    fn decode_to_uninit(
        &mut self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
        dictionary: Option<DictView<'_>>,
    ) -> Result<usize, DecodeError> {
        // As `decode_to_slice`. A failed direct attempt may have written
        // inside the frames it reached; the incremental path rewrites at
        // least those on success, and its results stand.
        self.stream.reset(self.config);
        let direct = self.stream.decode_to_uninit(src, dst, dictionary);
        if let Some(written) = direct {
            self.apply_retention();
            return Ok(written);
        }
        let mut session = DecoderSession::new(self, dictionary);
        one_shot_outcome(
            session.process_uninit(src, dst, DecodeOperation::Finish),
            src.len(),
        )
    }

    /// Decodes complete input that was compressed with `dictionary` into a
    /// new vector; every frame starts from the dictionary.
    ///
    /// The dictionary is borrowed for this call only and not copied.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress`], plus [`DecodeError::DictionaryMismatch`]
    /// for a frame that names another dictionary ID.
    ///
    /// # Examples
    ///
    /// See [`DecodeDictionary`].
    pub fn decompress_with_dictionary<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
    ) -> Result<Vec<u8>, DecodeError> {
        let mut dst = Vec::new();
        self.decode_into(src, &mut dst, Some(dictionary.into().view()))?;
        Ok(dst)
    }

    /// Decodes complete input that was compressed with `dictionary` onto the
    /// end of `dst`, returning the range of the appended bytes, as
    /// [`Self::decompress_into`] does.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress_with_dictionary`]; `dst` is truncated back to
    /// its original length.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::Decompressor;
    /// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut output = b"status: ".to_vec();
    /// let range = Decompressor::default().decompress_with_dictionary_into(&dictionary, &frame, &mut output)?;
    /// assert_eq!(&output[range], b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_with_dictionary_into<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
        dst: &mut Vec<u8>,
    ) -> Result<Range<usize>, DecodeError> {
        self.decode_into(src, dst, Some(dictionary.into().view()))
    }

    /// Decodes complete input that was compressed with `dictionary` into
    /// `dst`, returning the number of bytes written, as
    /// [`Self::decompress_to_slice`] does.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress_with_dictionary`], plus
    /// [`DecodeError::OutputTooSmall`] when the output does not fit.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::Decompressor;
    /// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut output = [0u8; 4];
    /// let written = Decompressor::default().decompress_with_dictionary_to_slice(&dictionary, &frame, &mut output)?;
    /// assert_eq!(&output[..written], b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_with_dictionary_to_slice<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<usize, DecodeError> {
        self.decode_to_slice(src, dst, Some(dictionary.into().view()))
    }

    /// Decodes complete input that was compressed with `dictionary` into
    /// uninitialized memory, as [`Self::decompress_to_uninit`] does.
    ///
    /// # Errors
    ///
    /// As [`Self::decompress_with_dictionary_to_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::Decompressor;
    /// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut output: Vec<u8> = Vec::with_capacity(4);
    /// let written = Decompressor::default().decompress_with_dictionary_to_uninit(
    ///     &dictionary,
    ///     &frame,
    ///     output.spare_capacity_mut(),
    /// )?;
    /// // SAFETY: the first `written` bytes were initialized.
    /// unsafe { output.set_len(written) };
    /// assert_eq!(output, b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress_with_dictionary_to_uninit<'dict>(
        &mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        self.decode_to_uninit(src, dst, Some(dictionary.into().view()))
    }
}

/// The result of a one-shot call from its single `Finish` session call
/// over all `input_len` bytes.
fn one_shot_outcome(
    result: Result<DecodeProgress, DecodeFailure>,
    input_len: usize,
) -> Result<usize, DecodeError> {
    let progress = result?;
    match progress.status {
        DecoderStatus::NeedsOutput => Err(DecodeError::OutputTooSmall {
            written: progress.produced,
        }),
        DecoderStatus::NeedsInput => Err(DecodeError::UnexpectedEndOfInput),
        DecoderStatus::Finished if progress.consumed != input_len => {
            Err(DecodeError::TrailingData {
                offset: progress.consumed as u64,
            })
        }
        DecoderStatus::Finished => Ok(progress.produced),
    }
}
