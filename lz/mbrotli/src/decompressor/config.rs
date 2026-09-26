use super::DecodeConfigError;

/// Number of complete raw streams accepted by an operation.
///
/// [`Self::Single`] sessions leave bytes after the first member unconsumed;
/// one-shot decoding rejects those bytes. [`Self::Concatenated`] treats them
/// as another member and needs final input to confirm the last boundary.
/// At least one complete member is required in either mode.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecoderConfig, Decompressor, MemberMode};
/// let config = DecoderConfig::default().with_member_mode(MemberMode::Concatenated);
/// let mut decoder = Decompressor::new(config)?;
/// // Two complete empty members.
/// assert!(decoder.decompress(&[0x3b, 0x3b])?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MemberMode {
    /// Stop at the first complete stream.
    #[default]
    Single,
    /// Decode successive streams until the caller declares final input.
    Concatenated,
}

/// Accepted window headers and their maximum bit count.
///
/// This limits headers, not total output or allocated workspace. Use
/// [`DecodeLimits`] to set those budgets. An extended header may declare a
/// small window too; [`Self::standard`] rejects every extended header.
///
/// # Examples
///
/// ```
/// use mbrotli::WindowLimit;
/// let standard = WindowLimit::standard(24)?;
/// assert_eq!(standard.max_bits(), 24);
/// assert!(!standard.allows_large());
/// let extended = WindowLimit::large(24)?;
/// assert!(extended.allows_large());
/// assert!(WindowLimit::standard(25).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowLimit {
    bits: u8,
    large: bool,
}

impl WindowLimit {
    /// Accepts only RFC 7932 headers up to this bit count.
    ///
    /// # Errors
    /// Returns an error unless `max_bits` is in `10..=24`.
    pub const fn standard(max_bits: u8) -> Result<Self, DecodeConfigError> {
        if max_bits < 10 || max_bits > 24 {
            return Err(DecodeConfigError::StandardWindow { max_bits });
        }
        Ok(Self {
            bits: max_bits,
            large: false,
        })
    }

    /// Accepts standard and extended headers up to this bit count.
    ///
    /// # Errors
    /// Returns an error unless `max_bits` is in `10..=62`.
    pub const fn large(max_bits: u8) -> Result<Self, DecodeConfigError> {
        if max_bits < 10 || max_bits > 62 {
            return Err(DecodeConfigError::LargeWindow { max_bits });
        }
        Ok(Self {
            bits: max_bits,
            large: true,
        })
    }

    /// Largest accepted window bit count.
    pub const fn max_bits(self) -> u8 {
        self.bits
    }
    /// Whether extended headers are accepted, including those below 25 bits.
    pub const fn allows_large(self) -> bool {
        self.large
    }
}

/// Optional cumulative input/output and live workspace budgets.
///
/// Defaults impose no numeric budgets. Dictionary storage and caller-owned
/// output do not count towards the workspace budget, nor do I/O adapter buffers.
/// `None` disables a budget; `Some(0)` permits none of that resource. Input and
/// output budgets count accepted bytes across all members of one operation and
/// reset when a new session starts. These are not total process-memory limits.
///
/// # Examples
///
/// ```
/// use mbrotli::DecodeLimits;
/// let limits = DecodeLimits::default()
///     .with_max_input_bytes(Some(1 << 20))
///     .with_max_output_bytes(Some(8 << 20))
///     .with_max_workspace_bytes(Some(32 << 20));
/// assert_eq!(limits.max_input_bytes(), Some(1 << 20));
/// assert_eq!(limits.max_output_bytes(), Some(8 << 20));
/// assert_eq!(limits.max_workspace_bytes(), Some(32 << 20));
/// assert_eq!(limits.with_max_input_bytes(None).max_input_bytes(), None);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeLimits {
    input: Option<u64>,
    output: Option<u64>,
    workspace: Option<usize>,
}

impl DecodeLimits {
    /// Sets the operation's compressed input budget, including metadata.
    pub const fn with_max_input_bytes(mut self, value: Option<u64>) -> Self {
        self.input = value;
        self
    }
    /// Sets the operation's regenerated output budget across all members.
    pub const fn with_max_output_bytes(mut self, value: Option<u64>) -> Self {
        self.output = value;
        self
    }
    /// Sets the live requested heap budget owned by the decoder.
    pub const fn with_max_workspace_bytes(mut self, value: Option<usize>) -> Self {
        self.workspace = value;
        self
    }
    /// Returns the compressed input budget.
    pub const fn max_input_bytes(self) -> Option<u64> {
        self.input
    }
    /// Returns the regenerated output budget.
    pub const fn max_output_bytes(self) -> Option<u64> {
        self.output
    }
    /// Returns the decoder workspace budget.
    pub const fn max_workspace_bytes(self) -> Option<usize> {
        self.workspace
    }
}

/// Reusable decoder policy. Defaults accept extended windows and one member.
///
/// The default window limit is 62 bits and numeric resource budgets are
/// unlimited. Builder methods return an updated copy; they do not change a
/// decoder already constructed from this value.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeLimits, DecoderConfig, MemberMode, WindowLimit};
/// let config = DecoderConfig::default()
///     .with_window_limit(WindowLimit::standard(22)?)
///     .with_member_mode(MemberMode::Single)
///     .with_limits(DecodeLimits::default().with_max_output_bytes(Some(4096)));
/// assert_eq!(config.window_limit().max_bits(), 22);
/// assert_eq!(config.member_mode(), MemberMode::Single);
/// assert_eq!(config.limits().max_output_bytes(), Some(4096));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderConfig {
    window: WindowLimit,
    members: MemberMode,
    limits: DecodeLimits,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            window: WindowLimit {
                bits: 62,
                large: true,
            },
            members: MemberMode::Single,
            limits: DecodeLimits::default(),
        }
    }
}

impl DecoderConfig {
    /// Sets the accepted window headers.
    pub const fn with_window_limit(mut self, value: WindowLimit) -> Self {
        self.window = value;
        self
    }
    /// Sets the operation's member policy.
    pub const fn with_member_mode(mut self, value: MemberMode) -> Self {
        self.members = value;
        self
    }
    /// Sets explicit resource budgets.
    pub const fn with_limits(mut self, value: DecodeLimits) -> Self {
        self.limits = value;
        self
    }
    /// Returns the accepted window headers.
    pub const fn window_limit(&self) -> WindowLimit {
        self.window
    }
    /// Returns the member policy.
    pub const fn member_mode(&self) -> MemberMode {
        self.members
    }
    /// Returns the resource budgets.
    pub const fn limits(&self) -> DecodeLimits {
        self.limits
    }
}

/// Expected total output, validated without trusting it as an allocation size.
///
/// Convert this to [`DecodeStreamConfig`] for use with sessions and adapters.
/// An exact size applies to all members combined; it neither reserves output
/// storage nor replaces a workspace budget. See [`DecodeStreamConfig`] for an
/// executable example.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputSize {
    /// No externally declared size.
    #[default]
    Unknown,
    /// Exact total payload bytes across the operation's members.
    Exact(u64),
}

/// Per-operation output validation.
///
/// Defaults to [`OutputSize::Unknown`]. A mismatch fails the operation even
/// when the compressed input is otherwise valid.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeOperation, DecodeStreamConfig, DecoderConfig, Decompressor, OutputSize};
/// let stream = DecodeStreamConfig::default().with_output_size(OutputSize::Exact(0));
/// assert_eq!(stream.output_size(), OutputSize::Exact(0));
/// assert_eq!(stream, DecodeStreamConfig::from(OutputSize::Exact(0)));
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// let mut session = decoder.start(stream)?;
/// session.process(&[0x3b], &mut [], DecodeOperation::Finish)?;
/// assert!(session.is_finished());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeStreamConfig {
    size: OutputSize,
}

impl DecodeStreamConfig {
    /// Sets the exact size contract, or disables it with `Unknown`.
    pub const fn with_output_size(mut self, value: OutputSize) -> Self {
        self.size = value;
        self
    }
    /// Returns the expected output size.
    pub const fn output_size(&self) -> OutputSize {
        self.size
    }
}

impl From<OutputSize> for DecodeStreamConfig {
    fn from(size: OutputSize) -> Self {
        Self { size }
    }
}
