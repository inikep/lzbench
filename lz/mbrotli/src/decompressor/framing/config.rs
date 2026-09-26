use crate::{DecoderConfig, OutputSize, WindowLimit};
/// Top-level input selection; raw decoding requires explicit opt-in.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum InputMode {
    /// Require a framing signature.
    #[default]
    FramedOnly,
    /// Accept framing or one raw member.
    Auto,
}
/// Storage policy for future references to earlier content.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InternalDictionaryPolicy {
    /// Reject internal references without caching resource payload.
    Reject,
    /// Retain addressable content until completion under dictionary budgets.
    /// Large inputs can exhaust the budget before any internal reference occurs.
    #[default]
    Retain,
}
/// Optional ceilings. `None` disables a ceiling; `Some(0)` permits none.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedDecodeLimits {
    /// Accepted wire bytes.
    pub max_input_bytes: Option<u64>,
    /// Delivered resource bytes, including hidden resources.
    pub max_output_bytes: Option<u64>,
    /// Regenerated resource and metadata bytes.
    pub max_decoded_bytes: Option<u64>,
    /// Bytes in one resource.
    pub max_resource_bytes: Option<u64>,
    /// Aggregate decoder-owned requested heap.
    pub max_workspace_bytes: Option<usize>,
    /// Framing records and metadata storage.
    pub max_framing_bytes: Option<usize>,
    /// Retained content and dictionary preparation.
    pub max_dictionary_bytes: Option<usize>,
    /// Aggregate regenerated metadata bytes.
    pub max_metadata_bytes: Option<u64>,
    /// Aggregate metadata field count.
    pub max_metadata_fields: Option<u64>,
    /// Resource count including empty resources.
    pub max_resources: Option<u64>,
    /// All wire chunks.
    pub max_chunks: Option<u64>,
}
impl Default for FramedDecodeLimits {
    fn default() -> Self {
        Self {
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
        }
    }
}
impl FramedDecodeLimits {
    /// Sets the ceiling for accepted wire bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_input_bytes(Some(1024));
    /// assert_eq!(limits.max_input_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_input_bytes(None).max_input_bytes(), None);
    /// ```
    pub const fn with_max_input_bytes(mut self, value: Option<u64>) -> Self {
        self.max_input_bytes = value;
        self
    }
    /// Returns the ceiling for accepted wire bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_input_bytes(Some(1024));
    /// assert_eq!(limits.max_input_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_input_bytes(None).max_input_bytes(), None);
    /// ```
    pub const fn max_input_bytes(self) -> Option<u64> {
        self.max_input_bytes
    }
    /// Sets the ceiling for delivered resource bytes, including hidden resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_output_bytes(Some(1024));
    /// assert_eq!(limits.max_output_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_output_bytes(None).max_output_bytes(), None);
    /// ```
    pub const fn with_max_output_bytes(mut self, value: Option<u64>) -> Self {
        self.max_output_bytes = value;
        self
    }
    /// Returns the ceiling for delivered resource bytes, including hidden resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_output_bytes(Some(1024));
    /// assert_eq!(limits.max_output_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_output_bytes(None).max_output_bytes(), None);
    /// ```
    pub const fn max_output_bytes(self) -> Option<u64> {
        self.max_output_bytes
    }
    /// Sets the ceiling for regenerated resource and metadata bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_decoded_bytes(Some(1024));
    /// assert_eq!(limits.max_decoded_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_decoded_bytes(None).max_decoded_bytes(), None);
    /// ```
    pub const fn with_max_decoded_bytes(mut self, value: Option<u64>) -> Self {
        self.max_decoded_bytes = value;
        self
    }
    /// Returns the ceiling for regenerated resource and metadata bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_decoded_bytes(Some(1024));
    /// assert_eq!(limits.max_decoded_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_decoded_bytes(None).max_decoded_bytes(), None);
    /// ```
    pub const fn max_decoded_bytes(self) -> Option<u64> {
        self.max_decoded_bytes
    }
    /// Sets the ceiling for bytes in one resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_resource_bytes(Some(1024));
    /// assert_eq!(limits.max_resource_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_resource_bytes(None).max_resource_bytes(), None);
    /// ```
    pub const fn with_max_resource_bytes(mut self, value: Option<u64>) -> Self {
        self.max_resource_bytes = value;
        self
    }
    /// Returns the ceiling for bytes in one resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_resource_bytes(Some(1024));
    /// assert_eq!(limits.max_resource_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_resource_bytes(None).max_resource_bytes(), None);
    /// ```
    pub const fn max_resource_bytes(self) -> Option<u64> {
        self.max_resource_bytes
    }
    /// Sets the ceiling for aggregate decoder-owned requested heap.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_workspace_bytes(Some(1024));
    /// assert_eq!(limits.max_workspace_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_workspace_bytes(None).max_workspace_bytes(), None);
    /// ```
    pub const fn with_max_workspace_bytes(mut self, value: Option<usize>) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the ceiling for aggregate decoder-owned requested heap.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_workspace_bytes(Some(1024));
    /// assert_eq!(limits.max_workspace_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_workspace_bytes(None).max_workspace_bytes(), None);
    /// ```
    pub const fn max_workspace_bytes(self) -> Option<usize> {
        self.max_workspace_bytes
    }
    /// Sets the ceiling for framing records and metadata storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_framing_bytes(Some(1024));
    /// assert_eq!(limits.max_framing_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_framing_bytes(None).max_framing_bytes(), None);
    /// ```
    pub const fn with_max_framing_bytes(mut self, value: Option<usize>) -> Self {
        self.max_framing_bytes = value;
        self
    }
    /// Returns the ceiling for framing records and metadata storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_framing_bytes(Some(1024));
    /// assert_eq!(limits.max_framing_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_framing_bytes(None).max_framing_bytes(), None);
    /// ```
    pub const fn max_framing_bytes(self) -> Option<usize> {
        self.max_framing_bytes
    }
    /// Sets the ceiling for retained content and dictionary preparation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_dictionary_bytes(Some(1024));
    /// assert_eq!(limits.max_dictionary_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_dictionary_bytes(None).max_dictionary_bytes(), None);
    /// ```
    pub const fn with_max_dictionary_bytes(mut self, value: Option<usize>) -> Self {
        self.max_dictionary_bytes = value;
        self
    }
    /// Returns the ceiling for retained content and dictionary preparation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_dictionary_bytes(Some(1024));
    /// assert_eq!(limits.max_dictionary_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_dictionary_bytes(None).max_dictionary_bytes(), None);
    /// ```
    pub const fn max_dictionary_bytes(self) -> Option<usize> {
        self.max_dictionary_bytes
    }
    /// Sets the ceiling for aggregate regenerated metadata bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_metadata_bytes(Some(1024));
    /// assert_eq!(limits.max_metadata_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_metadata_bytes(None).max_metadata_bytes(), None);
    /// ```
    pub const fn with_max_metadata_bytes(mut self, value: Option<u64>) -> Self {
        self.max_metadata_bytes = value;
        self
    }
    /// Returns the ceiling for aggregate regenerated metadata bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_metadata_bytes(Some(1024));
    /// assert_eq!(limits.max_metadata_bytes(), Some(1024));
    /// assert_eq!(limits.with_max_metadata_bytes(None).max_metadata_bytes(), None);
    /// ```
    pub const fn max_metadata_bytes(self) -> Option<u64> {
        self.max_metadata_bytes
    }
    /// Sets the ceiling for aggregate metadata field count.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_metadata_fields(Some(1024));
    /// assert_eq!(limits.max_metadata_fields(), Some(1024));
    /// assert_eq!(limits.with_max_metadata_fields(None).max_metadata_fields(), None);
    /// ```
    pub const fn with_max_metadata_fields(mut self, value: Option<u64>) -> Self {
        self.max_metadata_fields = value;
        self
    }
    /// Returns the ceiling for aggregate metadata field count.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_metadata_fields(Some(1024));
    /// assert_eq!(limits.max_metadata_fields(), Some(1024));
    /// assert_eq!(limits.with_max_metadata_fields(None).max_metadata_fields(), None);
    /// ```
    pub const fn max_metadata_fields(self) -> Option<u64> {
        self.max_metadata_fields
    }
    /// Sets the ceiling for resource count including empty resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_resources(Some(1024));
    /// assert_eq!(limits.max_resources(), Some(1024));
    /// assert_eq!(limits.with_max_resources(None).max_resources(), None);
    /// ```
    pub const fn with_max_resources(mut self, value: Option<u64>) -> Self {
        self.max_resources = value;
        self
    }
    /// Returns the ceiling for resource count including empty resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_resources(Some(1024));
    /// assert_eq!(limits.max_resources(), Some(1024));
    /// assert_eq!(limits.with_max_resources(None).max_resources(), None);
    /// ```
    pub const fn max_resources(self) -> Option<u64> {
        self.max_resources
    }
    /// Sets the ceiling for all wire chunks.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_chunks(Some(1024));
    /// assert_eq!(limits.max_chunks(), Some(1024));
    /// assert_eq!(limits.with_max_chunks(None).max_chunks(), None);
    /// ```
    pub const fn with_max_chunks(mut self, value: Option<u64>) -> Self {
        self.max_chunks = value;
        self
    }
    /// Returns the ceiling for all wire chunks.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let limits = FramedDecodeLimits::default().with_max_chunks(Some(1024));
    /// assert_eq!(limits.max_chunks(), Some(1024));
    /// assert_eq!(limits.with_max_chunks(None).max_chunks(), None);
    /// ```
    pub const fn max_chunks(self) -> Option<u64> {
        self.max_chunks
    }
}
/// Reusable policy, containing no resolver borrows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramedDecodeConfig {
    input_mode: InputMode,
    window_limit: WindowLimit,
    limits: FramedDecodeLimits,
    internal_dictionaries: InternalDictionaryPolicy,
}
impl Default for FramedDecodeConfig {
    fn default() -> Self {
        Self {
            input_mode: InputMode::default(),
            window_limit: DecoderConfig::default().window_limit(),
            limits: FramedDecodeLimits::default(),
            internal_dictionaries: InternalDictionaryPolicy::default(),
        }
    }
}
impl FramedDecodeConfig {
    /// Sets the input mode policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = InputMode::Auto;
    /// let config = FramedDecodeConfig::default().with_input_mode(policy);
    /// assert_eq!(config.input_mode(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_input_mode(mut self, value: InputMode) -> Self {
        self.input_mode = value;
        self
    }
    /// Returns the input mode policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = InputMode::Auto;
    /// let config = FramedDecodeConfig::default().with_input_mode(policy);
    /// assert_eq!(config.input_mode(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn input_mode(&self) -> InputMode {
        self.input_mode
    }
    /// Sets the window limit policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = mbrotli::WindowLimit::standard(22)?;
    /// let config = FramedDecodeConfig::default().with_window_limit(policy);
    /// assert_eq!(config.window_limit(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_window_limit(mut self, value: WindowLimit) -> Self {
        self.window_limit = value;
        self
    }
    /// Returns the window limit policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = mbrotli::WindowLimit::standard(22)?;
    /// let config = FramedDecodeConfig::default().with_window_limit(policy);
    /// assert_eq!(config.window_limit(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn window_limit(&self) -> WindowLimit {
        self.window_limit
    }
    /// Sets the limits policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = FramedDecodeLimits::default().with_max_resources(Some(8));
    /// let config = FramedDecodeConfig::default().with_limits(policy);
    /// assert_eq!(config.limits(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_limits(mut self, value: FramedDecodeLimits) -> Self {
        self.limits = value;
        self
    }
    /// Returns the limits policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = FramedDecodeLimits::default().with_max_resources(Some(8));
    /// let config = FramedDecodeConfig::default().with_limits(policy);
    /// assert_eq!(config.limits(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn limits(&self) -> FramedDecodeLimits {
        self.limits
    }
    /// Sets the internal dictionaries policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = InternalDictionaryPolicy::Reject;
    /// let config = FramedDecodeConfig::default().with_internal_dictionaries(policy);
    /// assert_eq!(config.internal_dictionaries(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_internal_dictionaries(mut self, value: InternalDictionaryPolicy) -> Self {
        self.internal_dictionaries = value;
        self
    }
    /// Returns the internal dictionaries policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let policy = InternalDictionaryPolicy::Reject;
    /// let config = FramedDecodeConfig::default().with_internal_dictionaries(policy);
    /// assert_eq!(config.internal_dictionaries(), policy);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn internal_dictionaries(&self) -> InternalDictionaryPolicy {
        self.internal_dictionaries
    }
}
/// Per-operation contract on the sum of resource payload, excluding metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FramedDecodeStreamConfig {
    size: OutputSize,
}
impl FramedDecodeStreamConfig {
    /// Sets the aggregate output contract without reserving output storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let stream = FramedDecodeStreamConfig::default()
    ///     .with_output_size(mbrotli::OutputSize::Exact(3));
    /// assert_eq!(stream.output_size(), mbrotli::OutputSize::Exact(3));
    /// ```
    pub const fn with_output_size(mut self, value: OutputSize) -> Self {
        self.size = value;
        self
    }
    /// Returns the output contract.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let stream = FramedDecodeStreamConfig::default()
    ///     .with_output_size(mbrotli::OutputSize::Exact(3));
    /// assert_eq!(stream.output_size(), mbrotli::OutputSize::Exact(3));
    /// ```
    pub const fn output_size(&self) -> OutputSize {
        self.size
    }
}
impl From<OutputSize> for FramedDecodeStreamConfig {
    fn from(size: OutputSize) -> Self {
        Self { size }
    }
}
