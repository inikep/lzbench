use crate::{DecodeConfigError, DecodeLimits};

/// Which top-level inputs the framed decoder accepts.
///
/// # Examples
///
/// ```
/// use mzstd::framing::InputMode;
/// assert_eq!(InputMode::default(), InputMode::FramedOnly);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum InputMode {
    /// Only a framed container.
    #[default]
    FramedOnly,
    /// A framed container, or raw Zstandard frames as one resource.
    Auto,
}

/// Whether chunks may reference earlier content of the same container.
///
/// # Examples
///
/// ```
/// use mzstd::framing::InternalDictionaryPolicy;
/// assert_eq!(InternalDictionaryPolicy::default(), InternalDictionaryPolicy::Retain);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InternalDictionaryPolicy {
    /// Reject internal references and retain no decoded content for them.
    Reject,
    /// Retain decoded content so later chunks may reference it.
    #[default]
    Retain,
}

/// Explicit numeric ceilings; `None` disables a ceiling.
///
/// The workspace, framing and dictionary ceilings count requested heap
/// bytes owned by the decoder, not caller output or borrowed dictionaries.
///
/// # Examples
///
/// ```
/// use mzstd::framing::FramedDecodeLimits;
/// let limits = FramedDecodeLimits::default().with_max_resources(Some(4));
/// assert_eq!(limits.max_resources(), Some(4));
/// assert_eq!(limits.max_chunks(), Some(1_000_000));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedDecodeLimits {
    /// Accepted wire bytes.
    pub max_input_bytes: Option<u64>,
    /// Delivered resource payload bytes.
    pub max_output_bytes: Option<u64>,
    /// Regenerated resource and metadata bytes.
    pub max_decoded_bytes: Option<u64>,
    /// Bytes of one resource.
    pub max_resource_bytes: Option<u64>,
    /// Every owned allocation: framing, dictionaries and the codec.
    pub max_workspace_bytes: Option<usize>,
    /// Owned framing records and staging.
    pub max_framing_bytes: Option<usize>,
    /// Owned dictionaries and retained dictionary content.
    pub max_dictionary_bytes: Option<usize>,
    /// Regenerated metadata bytes.
    pub max_metadata_bytes: Option<u64>,
    /// Metadata fields.
    pub max_metadata_fields: Option<u64>,
    /// Resources, including hidden ones.
    pub max_resources: Option<u64>,
    /// Chunks, including padding, directory and footer.
    pub max_chunks: Option<u64>,
}

impl Default for FramedDecodeLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

macro_rules! limit_accessors {
    ($($with:ident, $get:ident, $field:ident: $ty:ty;)*) => {$(
        #[doc = concat!("Replaces `", stringify!($field), "`.")]
        #[must_use]
        pub const fn $with(mut self, value: Option<$ty>) -> Self {
            self.$field = value;
            self
        }

        #[doc = concat!("The `", stringify!($field), "` ceiling.")]
        #[must_use]
        pub const fn $get(self) -> Option<$ty> {
            self.$field
        }
    )*};
}

impl FramedDecodeLimits {
    /// The default ceilings, usable in `const` contexts.
    const DEFAULT: Self = Self {
        max_input_bytes: None,
        max_output_bytes: None,
        max_decoded_bytes: None,
        max_resource_bytes: None,
        max_workspace_bytes: Some(64 << 20),
        max_framing_bytes: Some(8 << 20),
        max_dictionary_bytes: Some(32 << 20),
        max_metadata_bytes: Some(1 << 20),
        max_metadata_fields: Some(65536),
        max_resources: Some(10000),
        max_chunks: Some(1000000),
    };

    limit_accessors! {
        with_max_input_bytes, max_input_bytes, max_input_bytes: u64;
        with_max_output_bytes, max_output_bytes, max_output_bytes: u64;
        with_max_decoded_bytes, max_decoded_bytes, max_decoded_bytes: u64;
        with_max_resource_bytes, max_resource_bytes, max_resource_bytes: u64;
        with_max_workspace_bytes, max_workspace_bytes, max_workspace_bytes: usize;
        with_max_framing_bytes, max_framing_bytes, max_framing_bytes: usize;
        with_max_dictionary_bytes, max_dictionary_bytes, max_dictionary_bytes: usize;
        with_max_metadata_bytes, max_metadata_bytes, max_metadata_bytes: u64;
        with_max_metadata_fields, max_metadata_fields, max_metadata_fields: u64;
        with_max_resources, max_resources, max_resources: u64;
        with_max_chunks, max_chunks, max_chunks: u64;
    }
}

/// Reusable framed decoding policy.
///
/// # Examples
///
/// ```
/// use mzstd::framing::*;
/// let config = FramedDecodeConfig::default()
///     .with_input_mode(InputMode::Auto)
///     .with_max_window_log(23)?;
/// assert_eq!(config.input_mode(), InputMode::Auto);
/// assert_eq!(config.max_window_log(), 23);
/// assert!(FramedDecodeConfig::default().with_max_window_log(9).is_err());
/// # Ok::<(), mzstd::DecodeConfigError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedDecodeConfig {
    input_mode: InputMode,
    window_log: u8,
    limits: FramedDecodeLimits,
    internal_dictionaries: InternalDictionaryPolicy,
}

impl Default for FramedDecodeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl FramedDecodeConfig {
    /// The default policy in a `const` context.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::framing::*;
    /// const CONFIG: FramedDecodeConfig = FramedDecodeConfig::new();
    /// assert_eq!(CONFIG, FramedDecodeConfig::default());
    /// assert_eq!(CONFIG.max_window_log(), mzstd::DecodeLimits::DEFAULT_WINDOW_LOG);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            input_mode: InputMode::FramedOnly,
            window_log: DecodeLimits::DEFAULT_WINDOW_LOG,
            limits: FramedDecodeLimits::DEFAULT,
            internal_dictionaries: InternalDictionaryPolicy::Retain,
        }
    }

    /// Replaces the accepted top-level formats.
    #[must_use]
    pub const fn with_input_mode(mut self, value: InputMode) -> Self {
        self.input_mode = value;
        self
    }

    /// The accepted top-level formats.
    #[must_use]
    pub const fn input_mode(&self) -> InputMode {
        self.input_mode
    }

    /// Replaces the largest accepted frame window, as a power of two.
    ///
    /// # Errors
    ///
    /// [`DecodeConfigError::WindowLog`] outside
    /// [`DecodeLimits::MIN_WINDOW_LOG`]`..=`[`DecodeLimits::MAX_WINDOW_LOG`].
    pub const fn with_max_window_log(mut self, log: u8) -> Result<Self, DecodeConfigError> {
        match DecodeLimits::DEFAULT.with_max_window_log(log) {
            Ok(_) => {
                self.window_log = log;
                Ok(self)
            }
            Err(error) => Err(error),
        }
    }

    /// The largest accepted frame window, as a power of two.
    #[must_use]
    pub const fn max_window_log(&self) -> u8 {
        self.window_log
    }

    /// Replaces all numeric ceilings.
    #[must_use]
    pub const fn with_limits(mut self, value: FramedDecodeLimits) -> Self {
        self.limits = value;
        self
    }

    /// The numeric ceilings.
    #[must_use]
    pub const fn limits(&self) -> FramedDecodeLimits {
        self.limits
    }

    /// Replaces the internal dictionary policy.
    #[must_use]
    pub const fn with_internal_dictionaries(mut self, value: InternalDictionaryPolicy) -> Self {
        self.internal_dictionaries = value;
        self
    }

    /// The internal dictionary policy.
    #[must_use]
    pub const fn internal_dictionaries(&self) -> InternalDictionaryPolicy {
        self.internal_dictionaries
    }
}

/// Declared total resource payload of one object.
///
/// # Examples
///
/// ```
/// use mzstd::framing::OutputSize;
/// assert_eq!(OutputSize::from(5), OutputSize::Exact(5));
/// assert_eq!(OutputSize::default(), OutputSize::Unknown);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum OutputSize {
    /// No contract.
    #[default]
    Unknown,
    /// Exactly this many resource payload bytes.
    Exact(u64),
}

impl From<u64> for OutputSize {
    fn from(value: u64) -> Self {
        Self::Exact(value)
    }
}

/// Per-object output contract.
///
/// # Examples
///
/// ```
/// use mzstd::framing::*;
/// let stream = FramedDecodeStreamConfig::default().with_output_size(OutputSize::Exact(3));
/// assert_eq!(stream.output_size(), OutputSize::Exact(3));
/// assert_eq!(stream, FramedDecodeStreamConfig::from(OutputSize::Exact(3)));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FramedDecodeStreamConfig {
    size: OutputSize,
}

impl FramedDecodeStreamConfig {
    /// Replaces the payload contract.
    #[must_use]
    pub const fn with_output_size(mut self, value: OutputSize) -> Self {
        self.size = value;
        self
    }

    /// The payload contract.
    #[must_use]
    pub const fn output_size(&self) -> OutputSize {
        self.size
    }
}

impl From<OutputSize> for FramedDecodeStreamConfig {
    fn from(size: OutputSize) -> Self {
        Self { size }
    }
}
