//! Dictionaries prepared for compression.
//!
//! Preparing a dictionary indexes its content with the match finder tables
//! of one encoder configuration and reads its entropy tables; compressing
//! with it then costs no more than compressing without one. The prepared
//! dictionary is immutable and owns everything it needs, so any number of
//! compressors may use it at once.

use alloc::boxed::Box;

use super::config::{CompressionParams, EncoderConfig, Level};
use super::core::cctx::{FrameParams, RequestedParams};
use super::core::cdict::CDict;
use super::core::params::DEFAULT_CLEVEL;
use super::internal::requested_params;
use crate::dictionary::{DictionaryContent, DictionaryError, DictionaryId};

/// Prepares a [`PreparedDictionary`] from dictionary bytes.
///
/// The bytes are copied; the builder only borrows them. By default they are
/// read as [`DictionaryContent::Auto`] and prepared for
/// [`EncoderConfig::default`].
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::{DictionaryBuilder, DictionaryContent};
/// use mzstd::{EncoderConfig, Level};
///
/// let config = EncoderConfig::default().with_level(Level::try_from(9)?);
/// let dictionary = DictionaryBuilder::new(b"shared history for small payloads")
///     .with_content(DictionaryContent::Raw)
///     .with_config(config)
///     .build()?;
/// assert_eq!(dictionary.config(), &config);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug)]
pub struct DictionaryBuilder<'a> {
    bytes: &'a [u8],
    content: DictionaryContent,
    config: EncoderConfig,
    level_parameters: Option<Level>,
}

impl<'a> DictionaryBuilder<'a> {
    /// A builder for `bytes`, read as [`DictionaryContent::Auto`] and
    /// prepared for the default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new(b"raw dictionary content").build()?;
    /// assert_eq!(dictionary.id(), None);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            content: DictionaryContent::Auto,
            config: EncoderConfig::default(),
            level_parameters: None,
        }
    }

    /// Reads the bytes as `content` says.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::{DictionaryBuilder, DictionaryContent, DictionaryError};
    ///
    /// let error = DictionaryBuilder::new(b"not a zstd dictionary")
    ///     .with_content(DictionaryContent::Zstd)
    ///     .build()
    ///     .unwrap_err();
    /// assert_eq!(error, DictionaryError::NotZstdFormat);
    /// ```
    #[must_use]
    pub const fn with_content(mut self, content: DictionaryContent) -> Self {
        self.content = content;
        self
    }

    /// Prepares the dictionary for compressors built with `config`.
    ///
    /// Its frame options (checksum, content size) do not matter: the
    /// dictionary serves every configuration that differs only in them.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::{EncoderConfig, Level};
    ///
    /// let config = EncoderConfig::default().with_level(Level::try_from(19)?);
    /// let dictionary = DictionaryBuilder::new(b"history").with_config(config).build()?;
    /// assert_eq!(dictionary.config().level(), Level::try_from(19)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_config(mut self, config: EncoderConfig) -> Self {
        self.config = config;
        self
    }

    /// Prepares the dictionary like `ZSTD_createCDict_advanced` with
    /// `ZSTD_getCParams(level, 0, size)`, for use by any compressor.
    ///
    /// The dictionary's tables are built with the parameters of `level` for a
    /// dictionary of this size, and every compressor accepts it whatever its
    /// own configuration, as `ZSTD_CCtx_refCDict` lets any context use a
    /// `ZSTD_CDict`. Overrides [`Self::with_config`].
    ///
    /// Compressing with such a dictionary follows `ZSTD_CCtx_refCDict`
    /// rather than `ZSTD_CCtx_loadDictionary`, which differ in zstd 1.5.7:
    ///
    /// - the compressor's own level is ignored, and the default level's
    ///   parameters stand in for it when sizing the window; the other
    ///   explicit parameters of the compressor (window log and so on) still
    ///   apply;
    /// - the frame attaches or copies the dictionary's tables by input size
    ///   (see [`DictionaryAttach`](crate::DictionaryAttach)) and never
    ///   re-indexes the dictionary with the compressor's parameters, except
    ///   under [`DictionaryAttach::Load`](crate::DictionaryAttach::Load).
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::{Compressor, EncoderConfig, Level};
    ///
    /// let dictionary = DictionaryBuilder::new(b"the quick brown fox jumps over the lazy dog")
    ///     .with_level_parameters(Level::try_from(9)?)
    ///     .build()?;
    /// // A compressor at another level can use it.
    /// let config = EncoderConfig::default().with_level(Level::try_from(1)?);
    /// let frame = Compressor::new(config)?
    ///     .compress_with_dictionary(&dictionary, b"the quick brown fox")?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_level_parameters(mut self, level: Level) -> Self {
        self.level_parameters = Some(level);
        self
    }

    /// Copies and prepares the dictionary.
    ///
    /// # Errors
    ///
    /// [`DictionaryError::Config`] when the configuration is invalid,
    /// [`DictionaryError::NotZstdFormat`] when [`DictionaryContent::Zstd`]
    /// was requested for bytes in another format, and
    /// [`DictionaryError::Corrupted`] when a Zstandard-format dictionary's
    /// entropy tables or repeat offsets are malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::{DictionaryBuilder, DictionaryError};
    ///
    /// // The Zstandard magic number followed by a broken entropy section.
    /// let mut bytes = 0xEC30_A437u32.to_le_bytes().to_vec();
    /// bytes.extend_from_slice(&[1, 0, 0, 0, 0xFF, 0xFF]);
    /// assert_eq!(DictionaryBuilder::new(&bytes).build().unwrap_err(), DictionaryError::Corrupted);
    /// ```
    pub fn build(self) -> Result<PreparedDictionary, DictionaryError> {
        let config = match self.level_parameters {
            // `ZSTD_createCDict_advanced`: the default level with the
            // level's parameters for this dictionary size as overrides.
            Some(level) => EncoderConfig::default().with_params(CompressionParams::for_level(
                level,
                0,
                self.bytes.len(),
            )),
            None => self.config,
        };
        config.validate()?;
        let requested = requested_params(&config);
        let cdict = CDict::new(Box::from(self.bytes), self.content, &requested)?;
        Ok(PreparedDictionary {
            inner: Box::new(Prepared {
                #[cfg(feature = "decompression")]
                decode: crate::dictionary::decode::parse_decode_parts(self.bytes, self.content)?,
                cdict,
                config,
                key: dictionary_key(&requested),
                standalone: self.level_parameters.is_some(),
            }),
        })
    }
}

/// The parts of a configuration a prepared dictionary depends on: all of
/// it but the frame options.
pub(crate) fn dictionary_key(requested: &RequestedParams) -> RequestedParams {
    RequestedParams {
        fparams: FrameParams {
            no_dict_id: requested.fparams.no_dict_id,
            ..FrameParams::default()
        },
        ..*requested
    }
}

/// A dictionary indexed for one encoder configuration, ready to compress
/// with (`ZSTD_CDict`).
///
/// Built by [`DictionaryBuilder`]. Immutable: every method takes `&self`, so
/// one dictionary can serve any number of compressors on any number of
/// threads; wrap it in an `Arc` to share ownership. A compressor accepts it
/// when its configuration equals the dictionary's apart from the frame
/// options (checksum, content size).
///
/// Frames are byte-identical to zstd 1.5.7's `ZSTD_compress2` and
/// `ZSTD_compressStream2` after `ZSTD_CCtx_loadDictionary_advanced` with the
/// same bytes and content type.
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::DictionaryBuilder;
/// use mzstd::Compressor;
///
/// let dictionary = DictionaryBuilder::new(b"{\"user\":\"alice\",\"city\":\"Oslo\"}").build()?;
/// let mut encoder = Compressor::default();
/// let record = b"{\"user\":\"alice\",\"city\":\"Oslo\"}";
/// let with = encoder.compress_with_dictionary(&dictionary, record)?;
/// let without = encoder.compress(record)?;
/// assert!(with.len() < without.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PreparedDictionary {
    /// Boxed so the handle is one pointer wide: a session that may hold a
    /// dictionary (`Option<PreparedDictionary>`) stays small without one.
    inner: Box<Prepared>,
}

/// What a [`PreparedDictionary`] owns.
struct Prepared {
    cdict: CDict,
    config: EncoderConfig,
    key: RequestedParams,
    /// Whether any compressor may use the dictionary, as `ZSTD_CCtx_refCDict`
    /// allows (see [`DictionaryBuilder::with_level_parameters`]).
    standalone: bool,
    /// The decoder's tables, so the dictionary also decodes its frames.
    #[cfg(feature = "decompression")]
    decode: crate::dictionary::decode::DecodeParts,
}

impl PreparedDictionary {
    /// The ID frames compressed with this dictionary name, `None` for
    /// raw-content dictionaries (and Zstandard ones declaring ID zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    ///
    /// assert_eq!(DictionaryBuilder::new(b"raw content").build()?.id(), None);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn id(&self) -> Option<DictionaryId> {
        DictionaryId::new(self.inner.cdict.dict_id)
    }

    /// The configuration the dictionary was prepared for.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::EncoderConfig;
    ///
    /// let dictionary = DictionaryBuilder::new(b"history").build()?;
    /// assert_eq!(dictionary.config(), &EncoderConfig::default());
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn config(&self) -> &EncoderConfig {
        &self.inner.config
    }

    /// The dictionary as it was given to the builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    ///
    /// assert_eq!(DictionaryBuilder::new(b"history").build()?.as_bytes(), b"history");
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.cdict.bytes()
    }

    /// How the bytes were read.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::{DictionaryBuilder, DictionaryContent};
    ///
    /// let dictionary = DictionaryBuilder::new(b"history").build()?;
    /// assert_eq!(dictionary.content(), DictionaryContent::Auto);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn content(&self) -> DictionaryContent {
        self.inner.cdict.content()
    }

    /// Bytes the dictionary keeps allocated: its copy of the bytes and its
    /// match finder tables.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new(b"history").build()?;
    /// assert!(dictionary.retained_bytes() > dictionary.as_bytes().len());
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        let bytes = self.inner.cdict.retained_bytes();
        #[cfg(feature = "decompression")]
        let bytes = bytes + self.inner.decode.retained_bytes();
        bytes
    }

    /// The compression tables.
    pub(crate) fn cdict(&self) -> &CDict {
        &self.inner.cdict
    }

    /// The decoder's view of this dictionary.
    #[cfg(feature = "decompression")]
    pub(crate) fn decode_view(&self) -> crate::decompressor::DictView<'_> {
        self.inner.decode.view(self.inner.cdict.bytes())
    }

    /// Whether compressors with `requested` parameters may use this
    /// dictionary.
    pub(crate) fn serves(&self, requested: &RequestedParams) -> bool {
        self.inner.standalone || dictionary_key(requested) == self.inner.key
    }

    /// The parameters a compressor with `requested` resolves frames with
    /// when it uses this dictionary.
    ///
    /// `ZSTD_CCtx_init_compressStream2` replaces the context's level with the
    /// level of a referenced `ZSTD_CDict` (`0`, the default level, for one
    /// created with `ZSTD_createCDict_advanced`), but not with that of the
    /// dictionary `ZSTD_CCtx_loadDictionary` makes.
    pub(crate) fn requested_for(&self, requested: &RequestedParams) -> RequestedParams {
        if self.inner.standalone {
            RequestedParams {
                compression_level: DEFAULT_CLEVEL,
                ..*requested
            }
        } else {
            *requested
        }
    }
}

impl core::fmt::Debug for PreparedDictionary {
    /// Shows the dictionary's identity and size, not its tables.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new(b"history").build()?;
    /// assert!(format!("{dictionary:?}").contains("len: 7"));
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedDictionary")
            .field("id", &self.id())
            .field("content", &self.content())
            .field("len", &self.as_bytes().len())
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl AsRef<Self> for PreparedDictionary {
    /// Borrows the dictionary itself, so an owned dictionary, a reference
    /// and an `Arc` of one all satisfy `AsRef<PreparedDictionary>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mzstd::dictionary::{DictionaryBuilder, PreparedDictionary};
    ///
    /// let shared = Arc::new(DictionaryBuilder::new(b"history").build()?);
    /// let view: &PreparedDictionary = shared.as_ref().as_ref();
    /// assert!(std::ptr::eq(view, &*shared));
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    fn as_ref(&self) -> &Self {
        self
    }
}
