use super::DecodeConfigError;

/// Which frames one operation decodes.
///
/// Zstandard streams are commonly several frames back to back, possibly
/// with skippable frames (user metadata) in between, and their content is
/// the concatenation of the frames' contents. [`Self::Concatenated`], the
/// default, decodes all of them until the input ends. [`Self::Single`]
/// decodes one Zstandard frame and stops right after it, so a protocol can
/// carry other data behind the frame. Skippable frames are always skipped.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeError, DecoderConfig, Decompressor, FrameMode};
/// // Two empty frames (single segment, content size 0, one empty raw block).
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x00, 0x01, 0x00, 0x00];
/// let input = [frame, frame].concat();
///
/// let mut decoder = Decompressor::default();
/// assert!(decoder.decompress(&input)?.is_empty());
///
/// let single = DecoderConfig::default().with_frame_mode(FrameMode::Single);
/// let mut decoder = Decompressor::new(single);
/// assert!(matches!(
///     decoder.decompress(&input),
///     Err(DecodeError::TrailingData { offset: 9 })
/// ));
/// # Ok::<(), mzstd::DecodeError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FrameMode {
    /// Decode every frame until the input ends.
    #[default]
    Concatenated,
    /// Decode exactly one Zstandard frame, after any skippable frames.
    Single,
}

/// Resource limits for decoding untrusted input.
///
/// The window limit bounds the history a frame may demand, and with it the
/// decoder's memory: a frame whose window is larger is rejected before any
/// of its blocks are read. The default is 2^27 bytes (128 MiB), the
/// reference decoder's streaming default (`ZSTD_WINDOWLOG_LIMIT_DEFAULT`);
/// frames written by the reference encoder at standard levels need at most
/// 2^23 bytes, and `--ultra` or long-distance mode up to 2^27. A
/// single-segment frame's window is its content size.
///
/// Input and output budgets are off by default. They count compressed bytes
/// accepted and decompressed bytes regenerated over one whole operation,
/// across all its frames.
///
/// # Examples
///
/// ```
/// use mzstd::DecodeLimits;
/// let limits = DecodeLimits::default()
///     .with_max_window_log(23)?
///     .with_max_input_bytes(Some(1 << 20))
///     .with_max_output_bytes(Some(64 << 20));
/// assert_eq!(limits.max_window_log(), 23);
/// assert_eq!(limits.max_window_size(), 8 << 20);
/// assert_eq!(limits.max_input_bytes(), Some(1 << 20));
/// assert_eq!(limits.max_output_bytes(), Some(64 << 20));
/// assert!(DecodeLimits::default().with_max_window_log(32).is_err());
/// # Ok::<(), mzstd::DecodeConfigError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    window_log: u8,
    input: Option<u64>,
    output: Option<u64>,
}

impl DecodeLimits {
    /// Smallest accepted window log: 1 KiB, the smallest window a frame can
    /// declare.
    pub const MIN_WINDOW_LOG: u8 = 10;
    /// Largest accepted window log: 2 GiB, the reference decoder's limit.
    pub const MAX_WINDOW_LOG: u8 = 31;
    /// Default window log: 128 MiB.
    pub const DEFAULT_WINDOW_LOG: u8 = 27;
    /// The default limits: a 2^27-byte window and no byte budgets.
    pub const DEFAULT: Self = Self {
        window_log: Self::DEFAULT_WINDOW_LOG,
        input: None,
        output: None,
    };

    /// Accepts frames whose window is at most `2^log` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeConfigError::WindowLog`] unless `log` is in
    /// [`Self::MIN_WINDOW_LOG`]`..=`[`Self::MAX_WINDOW_LOG`].
    pub const fn with_max_window_log(mut self, log: u8) -> Result<Self, DecodeConfigError> {
        if log < Self::MIN_WINDOW_LOG || log > Self::MAX_WINDOW_LOG {
            return Err(DecodeConfigError::WindowLog { log });
        }
        self.window_log = log;
        Ok(self)
    }

    /// Sets the compressed input budget; `None` disables it.
    #[must_use]
    pub const fn with_max_input_bytes(mut self, limit: Option<u64>) -> Self {
        self.input = limit;
        self
    }

    /// Sets the decompressed output budget; `None` disables it.
    #[must_use]
    pub const fn with_max_output_bytes(mut self, limit: Option<u64>) -> Self {
        self.output = limit;
        self
    }

    /// The largest accepted window, as a power of two.
    #[must_use]
    pub const fn max_window_log(&self) -> u8 {
        self.window_log
    }

    /// The largest accepted window in bytes.
    #[must_use]
    pub const fn max_window_size(&self) -> u64 {
        1 << self.window_log
    }

    /// The compressed input budget.
    #[must_use]
    pub const fn max_input_bytes(&self) -> Option<u64> {
        self.input
    }

    /// The decompressed output budget.
    #[must_use]
    pub const fn max_output_bytes(&self) -> Option<u64> {
        self.output
    }
}

impl Default for DecodeLimits {
    /// Returns [`DecodeLimits::DEFAULT`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Reusable decoder policy: the frame mode and resource limits.
///
/// Builder methods return an updated copy.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeLimits, DecoderConfig, FrameMode};
/// const STRICT: DecoderConfig = DecoderConfig::DEFAULT.with_frame_mode(FrameMode::Single);
/// let config = STRICT.with_limits(DecodeLimits::default().with_max_output_bytes(Some(4096)));
/// assert_eq!(config.frame_mode(), FrameMode::Single);
/// assert_eq!(config.limits().max_output_bytes(), Some(4096));
/// assert_eq!(DecoderConfig::default(), DecoderConfig::DEFAULT);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderConfig {
    frame_mode: FrameMode,
    limits: DecodeLimits,
}

impl DecoderConfig {
    /// Concatenated frames and [`DecodeLimits::DEFAULT`].
    pub const DEFAULT: Self = Self {
        frame_mode: FrameMode::Concatenated,
        limits: DecodeLimits::DEFAULT,
    };

    /// Sets which frames an operation decodes.
    #[must_use]
    pub const fn with_frame_mode(mut self, mode: FrameMode) -> Self {
        self.frame_mode = mode;
        self
    }

    /// Sets the resource limits.
    #[must_use]
    pub const fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Which frames an operation decodes.
    #[must_use]
    pub const fn frame_mode(&self) -> FrameMode {
        self.frame_mode
    }

    /// The resource limits.
    #[must_use]
    pub const fn limits(&self) -> DecodeLimits {
        self.limits
    }
}

impl Default for DecoderConfig {
    /// Returns [`DecoderConfig::DEFAULT`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<DecodeLimits> for DecoderConfig {
    /// The default policy with `limits`.
    fn from(limits: DecodeLimits) -> Self {
        Self::DEFAULT.with_limits(limits)
    }
}

impl From<FrameMode> for DecoderConfig {
    /// The default policy with `mode`.
    fn from(mode: FrameMode) -> Self {
        Self::DEFAULT.with_frame_mode(mode)
    }
}
