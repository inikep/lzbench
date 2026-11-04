use super::{DictionaryId, DictionaryReference};
use crate::dictionary::PreparedDictionary;
use crate::{EncoderConfig, InputSize};

/// Container policy and explicit resource ceilings.
///
/// Defaults enable a container and central directory, without repeated
/// metadata. Validation occurs in [`super::FramedCompressor::new`]. Use struct
/// update syntax to override only selected fields.
///
/// # Examples
///
/// ```
/// use mzstd::framing::FramingConfig;
/// let config = FramingConfig { chunk_bytes: 4096, ..Default::default() };
/// assert!(config.container && config.central_directory);
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FramingConfig {
    /// Include a final footer and permit multiple resources and metadata.
    pub container: bool,
    /// Emit a central directory containing every data and metadata header.
    pub central_directory: bool,
    /// Repeat all resource metadata before the central directory.
    pub repeat_metadata: bool,
    /// Maximum uncompressed input of one chunk, and so of one Zstandard
    /// frame (default 64 KiB, at most 16 MiB).
    pub chunk_bytes: usize,
    /// Maximum aggregate metadata content (default 1 MiB).
    pub max_metadata_bytes: usize,
    /// Maximum framing storage (default 8 MiB), excluding the sink, compressor
    /// workspace, and separately prepared dictionaries.
    pub max_buffer_bytes: usize,
    /// Maximum number of resources (default 10,000).
    pub max_resources: u64,
    /// Maximum number of chunks, including generated directory/footer chunks.
    pub max_chunks: u64,
}

impl FramingConfig {
    /// The default policy, usable in `const` contexts.
    const DEFAULT: Self = Self {
        container: true,
        central_directory: true,
        repeat_metadata: false,
        chunk_bytes: 65536,
        max_metadata_bytes: 1 << 20,
        max_buffer_bytes: 8 << 20,
        max_resources: 10000,
        max_chunks: 1000000,
    };
}

impl Default for FramingConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Resource visibility and optional caller-supplied checksum.
///
/// Defaults to visible with no checksum. Setting `id` only records the supplied
/// value; the writer does not compute or verify it against the resource bytes.
///
/// A visible resource is stored as regular Zstandard frames, which every
/// Zstandard decoder extracts; a hidden one, for instance a dictionary
/// other resources refer to, is stored inside skippable frames.
///
/// # Examples
///
/// ```
/// use mzstd::framing::ResourceOptions;
/// let options = ResourceOptions { hidden: true, ..Default::default() };
/// assert!(options.hidden && options.id.is_none());
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct ResourceOptions {
    /// Suppress implicit extraction, for example for dictionary resources.
    pub hidden: bool,
    /// Checksum of the whole uncompressed resource, emitted on its last chunk.
    pub id: Option<DictionaryId>,
}

/// Compression for a self-contained metadata chunk.
///
/// `Zstd` uses the borrowed compressor's configuration. Shared references
/// must match the supplied dictionary and attachment order. Repeated
/// metadata permits only external references through this API.
///
/// # Examples
///
/// ```
/// use mzstd::framing::{MetadataEncoding, MetadataOptions};
/// let options = MetadataOptions { encoding: MetadataEncoding::Zstd, ..Default::default() };
/// assert!(matches!(options.repeated_encoding, MetadataEncoding::Uncompressed));
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub enum MetadataEncoding<'a> {
    /// Store the serialized fields without compression.
    #[default]
    Uncompressed,
    /// Compress into one Zstandard frame without a dictionary.
    Zstd,
    /// Compress into one Zstandard frame with explicit dictionary references.
    Shared {
        /// Immutable dictionary borrowed only while this metadata is queued.
        dictionary: &'a PreparedDictionary,
        /// Caller-supplied references in decoder attachment order.
        references: &'a [DictionaryReference],
    },
}

/// Independent encodings for original and repeated metadata.
///
/// # Examples
///
/// ```
/// use mzstd::framing::{MetadataEncoding, MetadataOptions};
/// let options = MetadataOptions {
///     encoding: MetadataEncoding::Uncompressed,
///     repeated_encoding: MetadataEncoding::Zstd,
/// };
/// assert!(matches!(options.repeated_encoding, MetadataEncoding::Zstd));
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct MetadataOptions<'a> {
    /// Encoding of the metadata adjacent to its resource, or global metadata.
    pub encoding: MetadataEncoding<'a>,
    /// Encoding of its repeated copy, when repetition is enabled.
    pub repeated_encoding: MetadataEncoding<'a>,
}

/// Reusable raw encoder settings and framing policy.
///
/// # Examples
///
/// ```
/// use mzstd::{EncoderConfig, Level, framing::*};
/// let level = Level::try_from(5)?;
/// let config = FramedEncodeConfig::default()
///     .with_encoder_config(EncoderConfig::default().with_level(level))
///     .with_framing_config(FramingConfig { chunk_bytes: 4096, ..Default::default() });
/// assert_eq!(config.encoder_config().level(), level);
/// assert_eq!(config.framing_config().chunk_bytes, 4096);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedEncodeConfig {
    encoder: EncoderConfig,
    framing: FramingConfig,
}

impl Default for FramedEncodeConfig {
    /// Returns [`FramedEncodeConfig::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl FramedEncodeConfig {
    /// Returns the default configuration in a `const` context.
    ///
    /// Equal to [`FramedEncodeConfig::default`]: [`EncoderConfig::new`] and
    /// the default [`FramingConfig`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{EncoderConfig, framing::*};
    /// const CHECKED: FramedEncodeConfig =
    ///     FramedEncodeConfig::new().with_encoder_config(EncoderConfig::new().with_checksum(true));
    /// assert_eq!(FramedEncodeConfig::new(), FramedEncodeConfig::default());
    /// assert!(CHECKED.encoder_config().checksum());
    /// assert_eq!(*CHECKED.framing_config(), FramingConfig::default());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            encoder: EncoderConfig::new(),
            framing: FramingConfig::DEFAULT,
        }
    }

    /// Raw resource and metadata compression settings.
    #[must_use]
    pub const fn encoder_config(&self) -> &EncoderConfig {
        &self.encoder
    }

    /// Container profile and budgets.
    #[must_use]
    pub const fn framing_config(&self) -> &FramingConfig {
        &self.framing
    }

    /// Replaces raw encoder settings.
    #[must_use]
    pub const fn with_encoder_config(mut self, value: EncoderConfig) -> Self {
        self.encoder = value;
        self
    }

    /// Replaces container policy.
    #[must_use]
    pub const fn with_framing_config(mut self, value: FramingConfig) -> Self {
        self.framing = value;
        self
    }
}

/// Per-container aggregate resource payload contract; metadata is excluded.
///
/// # Examples
///
/// ```
/// use mzstd::{InputSize, framing::FramedEncodeStreamConfig};
/// let stream = FramedEncodeStreamConfig::default().with_input_size(InputSize::Exact(5));
/// assert_eq!(stream.input_size(), InputSize::Exact(5));
/// assert_eq!(stream, FramedEncodeStreamConfig::from(InputSize::Exact(5)));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FramedEncodeStreamConfig {
    input_size: InputSize,
}

impl FramedEncodeStreamConfig {
    /// Aggregate input size, including hidden resources.
    #[must_use]
    pub const fn input_size(&self) -> InputSize {
        self.input_size
    }

    /// Sets the aggregate payload size without reserving that much memory.
    #[must_use]
    pub const fn with_input_size(mut self, value: InputSize) -> Self {
        self.input_size = value;
        self
    }
}

impl From<InputSize> for FramedEncodeStreamConfig {
    fn from(value: InputSize) -> Self {
        Self { input_size: value }
    }
}
