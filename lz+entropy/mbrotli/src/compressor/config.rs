//! Validated encoder configuration.
//!
//! [`EncoderConfig`] is everything about a stream that does not change from one
//! call to the next: how hard to search, how wide the window is, which header
//! it carries, how literals are modelled. It holds no input length, no stream
//! offset, no dictionary, no buffer and no workspace — those belong to the
//! operation, not to the encoder.
//!
//! Each value in it is a type that cannot hold a number the format could not
//! express: [`Quality`] is `0..=11`, [`BlockBits`] is `16..=24`, and a
//! [`Window`] is built by naming the header it carries. Combinations that are
//! individually legal but jointly meaningless — a Large Window at a quality
//! whose distance model cannot carry one — are rejected by
//! [`Compressor::new`](crate::Compressor::new), which is the one place that
//! sees the whole configuration at once.

use super::internal::WindowBits;
use super::internal::{CompressMode, DistanceCodes, QualityLevel};
pub use crate::{ConfigError, Window, WindowEncoding};
use thiserror::Error;

/// Compression quality: how much work the encoder spends per byte.
///
/// The Brotli format defines twelve, `0` through `11`. Zero is one pass with
/// static entropy codes; eleven runs a Zopfli dynamic program over every match
/// a binary tree can find. The type is ordered by effort, so two qualities
/// compare the way their numbers do.
///
/// # Examples
///
/// ```
/// use mbrotli::Quality;
///
/// assert!(Quality::Q1 < Quality::Q5);
/// assert_eq!(u8::from(Quality::Q5), 5);
/// assert_eq!(Quality::try_from(10u8)?, Quality::Q10);
/// assert!(Quality::try_from(12u8).is_err());
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Quality(u8);

impl Quality {
    /// One pass over each fragment with static entropy codes: fastest, largest.
    pub const Q0: Self = Self(0);
    /// Two passes over each fragment, with entropy codes built per block.
    pub const Q1: Self = Self(1);
    /// Greedy matching, with the format's fixed command and distance codes.
    pub const Q2: Self = Self(2);
    /// Greedy matching with one prefix code for the whole stream.
    pub const Q3: Self = Self(3);
    /// Adds block splitting, histogram optimisation and distance parameters.
    pub const Q4: Self = Self(4);
    /// Adds a delayed search and literal context modelling.
    pub const Q5: Self = Self(5);
    /// Deepens the search to thirty-two bucket candidates.
    pub const Q6: Self = Self(6);
    /// Deepens it again and adds the three-context literal model.
    pub const Q7: Self = Self(7);
    /// Deeper still, with more cached distances checked per position.
    pub const Q8: Self = Self(8);
    /// The deepest greedy search: 256 bucket candidates, 16 cached distances.
    pub const Q9: Self = Self(9);
    /// A Zopfli search over every match the binary tree finds.
    pub const Q10: Self = Self(10);
    /// The same search run harder and re-priced: slowest, smallest.
    pub const Q11: Self = Self(11);

    /// The lowest quality the format defines.
    pub const MIN: Self = Self::Q0;

    /// The highest quality the format defines.
    pub const MAX: Self = Self::Q11;

    /// Returns the numeric quality the Brotli format uses.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Quality;
    ///
    /// assert_eq!(Quality::Q9.get(), 9);
    /// ```
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Returns the quality as the closed enum the encoders are written against.
    pub(crate) const fn level(self) -> QualityLevel {
        match self.0 {
            0 => QualityLevel::Q0,
            1 => QualityLevel::Q1,
            2 => QualityLevel::Q2,
            3 => QualityLevel::Q3,
            4 => QualityLevel::Q4,
            5 => QualityLevel::Q5,
            6 => QualityLevel::Q6,
            7 => QualityLevel::Q7,
            8 => QualityLevel::Q8,
            9 => QualityLevel::Q9,
            10 => QualityLevel::Q10,
            // Unreachable: no `Quality` outside `0..=11` can be constructed.
            _ => QualityLevel::Q11,
        }
    }
}

impl Default for Quality {
    /// Returns [`Quality::Q11`], the reference encoder's default.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Quality;
    ///
    /// assert_eq!(Quality::default(), Quality::Q11);
    /// ```
    fn default() -> Self {
        Self::Q11
    }
}

impl TryFrom<u8> for Quality {
    type Error = ConfigError;

    /// Creates a quality from its numeric value.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Quality`] above eleven.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{ConfigError, Quality};
    ///
    /// assert_eq!(Quality::try_from(0u8)?, Quality::Q0);
    /// assert!(matches!(
    ///     Quality::try_from(12u8),
    ///     Err(ConfigError::Quality { requested: 12 })
    /// ));
    /// # Ok::<(), ConfigError>(())
    /// ```
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value > Self::MAX.0 {
            return Err(ConfigError::Quality { requested: value });
        }
        Ok(Self(value))
    }
}

impl From<Quality> for u8 {
    /// Returns the numeric quality the Brotli format uses.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Quality;
    ///
    /// assert_eq!(u8::from(Quality::Q0), 0);
    /// assert_eq!(u8::from(Quality::Q11), 11);
    /// ```
    fn from(value: Quality) -> Self {
        value.0
    }
}

/// How the encoder cuts its input into meta-blocks.
///
/// This is a work-partitioning choice, not a window choice: the two are
/// independent, and a Large Window is never inferred from a block size.
///
/// # Examples
///
/// ```
/// use mbrotli::{BlockBits, BlockSize};
///
/// assert_eq!(BlockSize::default(), BlockSize::Auto);
/// assert_eq!(BlockSize::from(BlockBits::MAX), BlockSize::Bits(BlockBits::MAX));
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum BlockSize {
    /// Let the encoder choose, as the reference does for the quality.
    #[default]
    Auto,
    /// Ask for an explicit block size.
    ///
    /// Qualities below four ignore this and always work in blocks of `1 << 14`
    /// bytes, exactly as the reference does.
    Bits(BlockBits),
}

impl From<BlockBits> for BlockSize {
    /// Wraps an explicit block size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{BlockBits, BlockSize};
    ///
    /// assert_eq!(BlockSize::from(BlockBits::MIN), BlockSize::Bits(BlockBits::MIN));
    /// ```
    fn from(value: BlockBits) -> Self {
        Self::Bits(value)
    }
}

/// Base-2 logarithm of the encoder's input block size.
///
/// The Brotli encoder restricts an explicitly requested block size to the
/// inclusive range `16..=24`; every way of building a `BlockBits` enforces that
/// range.
///
/// # Examples
///
/// ```
/// use mbrotli::BlockBits;
///
/// assert_eq!(usize::from(BlockBits::try_from(18u8)?), 18);
/// assert!(BlockBits::try_from(15u8).is_err());
/// assert!(BlockBits::try_from(25u8).is_err());
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BlockBits(pub(crate) usize);

impl BlockBits {
    /// Smallest block size the encoder accepts: 2^16 bytes.
    pub const MIN: Self = Self(16);

    /// Largest block size the encoder accepts: 2^24 bytes.
    pub const MAX: Self = Self(24);

    /// Returns the base-2 logarithm of the block size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::BlockBits;
    ///
    /// assert_eq!(BlockBits::MIN.get(), 16);
    /// ```
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0 as u8
    }
}

impl TryFrom<u8> for BlockBits {
    type Error = ConfigError;

    /// Creates a block size from its base-2 logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::BlockBits`] outside `16..=24`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{BlockBits, ConfigError};
    ///
    /// assert_eq!(BlockBits::try_from(16u8)?, BlockBits::MIN);
    /// assert!(matches!(
    ///     BlockBits::try_from(15u8),
    ///     Err(ConfigError::BlockBits { requested: 15 })
    /// ));
    /// # Ok::<(), ConfigError>(())
    /// ```
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if usize::from(value) < Self::MIN.0 || usize::from(value) > Self::MAX.0 {
            return Err(ConfigError::BlockBits { requested: value });
        }
        Ok(Self(usize::from(value)))
    }
}

impl From<BlockBits> for usize {
    /// Returns the base-2 logarithm of the block size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::BlockBits;
    ///
    /// assert_eq!(usize::from(BlockBits::MAX), 24);
    /// ```
    fn from(value: BlockBits) -> Self {
        value.0
    }
}

/// The kind of data a stream carries.
///
/// A hint only: every mode produces a valid stream that any decoder reads back
/// identically.
///
/// # Examples
///
/// ```
/// use mbrotli::CompressionMode;
///
/// assert_eq!(CompressionMode::default(), CompressionMode::Generic);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum CompressionMode {
    /// No assumption about the data.
    #[default]
    Generic,
    /// UTF-8 text.
    Text,
    /// Font data, in the WOFF 2.0 sense.
    Font,
}

impl CompressionMode {
    /// Returns the mode in the form the encoders are written against.
    pub(crate) const fn resolve(self) -> CompressMode {
        match self {
            Self::Generic => CompressMode::Generic,
            Self::Text => CompressMode::Text,
            Self::Font => CompressMode::Font,
        }
    }
}

/// Layout of the distance alphabet.
///
/// `Auto` resolves exactly as the reference does for the chosen quality, mode
/// and window — which for font mode means one postfix bit and twelve direct
/// codes, and otherwise none of either. An explicit layout is validated when it
/// is built, so a value of this type always describes an alphabet RFC 7932 can
/// express.
///
/// Qualities below four always use the default layout, whatever is asked for,
/// because the reference does.
///
/// # Examples
///
/// ```
/// use mbrotli::DistanceParams;
///
/// assert_eq!(DistanceParams::default(), DistanceParams::Auto);
///
/// let explicit = DistanceParams::explicit(1, 12)?;
/// assert_eq!(explicit, DistanceParams::Explicit { postfix_bits: 1, direct_codes: 12 });
///
/// // Six is not a whole number of `1 << 2` groups.
/// assert!(DistanceParams::explicit(2, 6).is_err());
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum DistanceParams {
    /// Let the encoder choose, as the reference does.
    #[default]
    Auto,
    /// Ask for an explicit postfix-bit and direct-code pair.
    Explicit {
        /// Number of postfix bits, at most three.
        postfix_bits: u8,
        /// Number of direct distance codes, at most one hundred and twenty.
        direct_codes: u16,
    },
}

impl DistanceParams {
    /// Largest number of postfix bits RFC 7932 allows.
    pub const MAX_POSTFIX_BITS: u8 = 3;

    /// Largest number of direct distance codes RFC 7932 allows.
    pub const MAX_DIRECT_CODES: u16 = 120;

    /// Creates a validated explicit distance layout.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::DistancePostfixBits`] above three postfix bits,
    /// [`ConfigError::DirectDistanceCodes`] above one hundred and twenty direct
    /// codes, and [`ConfigError::MisalignedDistanceCodes`] when the direct
    /// codes are not a whole number of `1 << postfix_bits` groups that the
    /// header's four-bit field can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{ConfigError, DistanceParams};
    ///
    /// assert!(DistanceParams::explicit(0, 0).is_ok());
    /// assert!(matches!(
    ///     DistanceParams::explicit(4, 0),
    ///     Err(ConfigError::DistancePostfixBits { requested: 4 })
    /// ));
    /// assert!(matches!(
    ///     DistanceParams::explicit(0, 121),
    ///     Err(ConfigError::DirectDistanceCodes { requested: 121 })
    /// ));
    /// ```
    pub const fn explicit(postfix_bits: u8, direct_codes: u16) -> Result<Self, ConfigError> {
        if postfix_bits > Self::MAX_POSTFIX_BITS {
            return Err(ConfigError::DistancePostfixBits {
                requested: postfix_bits,
            });
        }
        if direct_codes > Self::MAX_DIRECT_CODES {
            return Err(ConfigError::DirectDistanceCodes {
                requested: direct_codes,
            });
        }
        let groups = (direct_codes >> postfix_bits) & 0x0F;
        if (groups << postfix_bits) != direct_codes {
            return Err(ConfigError::MisalignedDistanceCodes {
                postfix_bits,
                direct_codes,
            });
        }
        Ok(Self::Explicit {
            postfix_bits,
            direct_codes,
        })
    }

    /// Returns the layout in the form the encoders are written against.
    pub(crate) const fn resolve(self) -> DistanceCodes {
        match self {
            Self::Auto => DistanceCodes::DEFAULT,
            Self::Explicit {
                postfix_bits,
                direct_codes,
            } => DistanceCodes::from_raw(postfix_bits as u32, direct_codes as u32),
        }
    }
}

/// Whether literals are modelled by their preceding bytes.
///
/// Only quality five and above model literal contexts at all; below that the
/// setting has nothing to apply to and is ignored, exactly as in the reference.
///
/// `Auto` follows the reference, which models contexts wherever the quality
/// allows it — so `Auto` and `Enabled` currently resolve alike, and the pair
/// exists so that a caller can say which of the two they meant.
///
/// # Examples
///
/// ```
/// use mbrotli::LiteralContextMode;
///
/// assert_eq!(LiteralContextMode::default(), LiteralContextMode::Auto);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum LiteralContextMode {
    /// Follow the reference encoder for the chosen quality.
    #[default]
    Auto,
    /// Model literal contexts wherever the quality allows it.
    Enabled,
    /// Never model literal contexts, trading ratio for decoding speed.
    Disabled,
}

impl LiteralContextMode {
    /// Returns whether the encoders should model literal contexts.
    pub(crate) const fn resolve(self) -> bool {
        match self {
            Self::Auto | Self::Enabled => true,
            Self::Disabled => false,
        }
    }
}

/// Everything about a stream that does not change between operations.
///
/// A `Compressor` is built from one of these and keeps it; the values that
/// belong to a single operation — how many bytes are coming, where the stream
/// starts, which dictionary is attached, where the output goes — are passed to
/// the operation instead.
///
/// The default mirrors the reference encoder's: quality 11, an ordinary 22-bit
/// window, automatic block size, generic mode, automatic distance parameters
/// and automatic literal context modelling. Quality 11 is the densest and by
/// far the slowest; for online compression [`Quality::Q5`] is the usual choice.
///
/// # Examples
///
/// ```
/// use mbrotli::{CompressionMode, EncoderConfig, Quality, Window};
///
/// let config = EncoderConfig::default()
///     .with_quality(Quality::Q5)
///     .with_window(Window::standard(22)?)
///     .with_mode(CompressionMode::Text);
///
/// assert_eq!(config.quality(), Quality::Q5);
/// assert_eq!(config.mode(), CompressionMode::Text);
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct EncoderConfig {
    quality: Quality,
    window: Window,
    block_size: BlockSize,
    mode: CompressionMode,
    distance: DistanceParams,
    literal_context: LiteralContextMode,
}

impl EncoderConfig {
    /// Sets the compression quality.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, Quality};
    ///
    /// assert_eq!(
    ///     EncoderConfig::default().with_quality(Quality::Q0).quality(),
    ///     Quality::Q0
    /// );
    /// ```
    #[must_use]
    pub const fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Returns the configured quality.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, Quality};
    ///
    /// assert_eq!(EncoderConfig::default().quality(), Quality::Q11);
    /// ```
    #[must_use]
    pub const fn quality(&self) -> Quality {
        self.quality
    }

    /// Sets the sliding window and the header that declares it.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, Window};
    ///
    /// let config = EncoderConfig::default().with_window(Window::large(30)?);
    ///
    /// assert_eq!(config.window().bits(), 30);
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Returns the configured window.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, Window};
    ///
    /// assert_eq!(EncoderConfig::default().window(), Window::DEFAULT);
    /// ```
    #[must_use]
    pub const fn window(&self) -> Window {
        self.window
    }

    /// Sets how the encoder cuts its input into meta-blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{BlockBits, BlockSize, EncoderConfig};
    ///
    /// let config = EncoderConfig::default().with_block_size(BlockSize::Bits(BlockBits::MAX));
    ///
    /// assert_eq!(config.block_size(), BlockSize::Bits(BlockBits::MAX));
    /// ```
    #[must_use]
    pub const fn with_block_size(mut self, block_size: BlockSize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Returns the configured block size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{BlockSize, EncoderConfig};
    ///
    /// assert_eq!(EncoderConfig::default().block_size(), BlockSize::Auto);
    /// ```
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Sets the kind of data being compressed.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{CompressionMode, EncoderConfig};
    ///
    /// let config = EncoderConfig::default().with_mode(CompressionMode::Font);
    ///
    /// assert_eq!(config.mode(), CompressionMode::Font);
    /// ```
    #[must_use]
    pub const fn with_mode(mut self, mode: CompressionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the configured mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{CompressionMode, EncoderConfig};
    ///
    /// assert_eq!(EncoderConfig::default().mode(), CompressionMode::Generic);
    /// ```
    #[must_use]
    pub const fn mode(&self) -> CompressionMode {
        self.mode
    }

    /// Sets the distance alphabet layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DistanceParams, EncoderConfig};
    ///
    /// let config = EncoderConfig::default().with_distance(DistanceParams::explicit(1, 4)?);
    ///
    /// assert_eq!(
    ///     config.distance(),
    ///     DistanceParams::Explicit { postfix_bits: 1, direct_codes: 4 }
    /// );
    /// # Ok::<(), mbrotli::ConfigError>(())
    /// ```
    #[must_use]
    pub const fn with_distance(mut self, distance: DistanceParams) -> Self {
        self.distance = distance;
        self
    }

    /// Returns the configured distance alphabet layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DistanceParams, EncoderConfig};
    ///
    /// assert_eq!(EncoderConfig::default().distance(), DistanceParams::Auto);
    /// ```
    #[must_use]
    pub const fn distance(&self) -> DistanceParams {
        self.distance
    }

    /// Sets whether literals are modelled by their preceding bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, LiteralContextMode};
    ///
    /// let config = EncoderConfig::default().with_literal_context(LiteralContextMode::Disabled);
    ///
    /// assert_eq!(config.literal_context(), LiteralContextMode::Disabled);
    /// ```
    #[must_use]
    pub const fn with_literal_context(mut self, literal_context: LiteralContextMode) -> Self {
        self.literal_context = literal_context;
        self
    }

    /// Returns the configured literal context policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{EncoderConfig, LiteralContextMode};
    ///
    /// assert_eq!(
    ///     EncoderConfig::default().literal_context(),
    ///     LiteralContextMode::Auto
    /// );
    /// ```
    #[must_use]
    pub const fn literal_context(&self) -> LiteralContextMode {
        self.literal_context
    }

    /// Checks the cross-field constraints one value cannot check alone.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::LargeWindowUnsupportedForQuality`] for a Large
    /// Window at a quality that writes distances through a model built for the
    /// RFC 7932 alphabet, which is qualities zero, one and two.
    pub(crate) const fn validate(&self) -> Result<(), ConfigError> {
        if matches!(self.window.encoding(), WindowEncoding::Large) && self.quality.0 <= 2 {
            return Err(ConfigError::LargeWindowUnsupportedForQuality {
                quality: self.quality,
            });
        }
        Ok(())
    }

    /// Lowers the configuration into the shape the encoders take.
    ///
    /// `size_hint` is the operation's total-input knowledge, which is not part
    /// of the configuration: qualities four and five choose their match finder
    /// from it, so it arrives per stream rather than per encoder.
    pub(crate) const fn lower(&self, size_hint: Option<usize>) -> super::internal::CompressParams {
        super::internal::CompressParams {
            #[cfg(feature = "experimental")]
            stream_offset: 0,
            quality: self.quality.level(),
            lgwin: self.window.resolve(),
            lgblock: match self.block_size {
                BlockSize::Auto => None,
                BlockSize::Bits(bits) => Some(bits),
            },
            mode: self.mode.resolve(),
            size_hint,
            distance_codes: self.distance.resolve(),
            literal_context_modeling: self.literal_context.resolve(),
        }
    }
}

/// Error returned when a compressed-size bound does not fit in a `usize`.
///
/// # Examples
///
/// ```
/// use mbrotli::Compressor;
///
/// assert!(Compressor::max_compressed_size(4096).is_ok());
/// assert!(Compressor::max_compressed_size(usize::MAX).is_err());
/// ```
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[error("the compressed-size bound overflows the address space")]
pub struct SizeOverflow;

impl Window {
    /// Returns the window in the form the encoders are written against.
    pub(crate) const fn resolve(self) -> WindowBits {
        // Both constructors have already checked the range their header
        // allows, so neither arm can fail; the fallback keeps this a
        // `const fn` without an unreachable panic.
        let outcome = match self.encoding() {
            WindowEncoding::Standard => WindowBits::standard(self.bits()),
            WindowEncoding::Large => WindowBits::large(self.bits()),
        };
        match outcome {
            Ok(bits) => bits,
            Err(_) => WindowBits::DEFAULT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn every_quality_the_format_defines_round_trips() {
        for value in 0u8..=11 {
            let quality = Quality::try_from(value).expect("a legal quality");
            assert_eq!(quality.get(), value);
            assert_eq!(u8::from(quality), value);
            assert_eq!(usize::from(quality.level()), usize::from(value));
        }
        for value in [12u8, 13, 255] {
            assert_eq!(
                Quality::try_from(value),
                Err(ConfigError::Quality { requested: value })
            );
        }
    }

    #[test]
    fn the_quality_constants_are_their_numbers() {
        let constants = [
            Quality::Q0,
            Quality::Q1,
            Quality::Q2,
            Quality::Q3,
            Quality::Q4,
            Quality::Q5,
            Quality::Q6,
            Quality::Q7,
            Quality::Q8,
            Quality::Q9,
            Quality::Q10,
            Quality::Q11,
        ];
        for (index, quality) in constants.into_iter().enumerate() {
            assert_eq!(usize::from(quality.get()), index);
        }
        assert_eq!(Quality::MIN, Quality::Q0);
        assert_eq!(Quality::MAX, Quality::Q11);
        assert_eq!(Quality::default(), Quality::Q11);
    }

    #[test]
    fn a_window_carries_its_header_as_well_as_its_size() {
        for bits in 10u8..=24 {
            let ordinary = Window::standard(bits).expect("a legal ordinary window");
            let large = Window::large(bits).expect("a legal large window");
            assert_eq!(ordinary.bits(), large.bits());
            assert_ne!(ordinary, large);
            assert_eq!(ordinary.encoding(), WindowEncoding::Standard);
            assert_eq!(large.encoding(), WindowEncoding::Large);
            assert_ne!(ordinary.resolve(), large.resolve());
        }
        for bits in 25u8..=62 {
            assert!(Window::standard(bits).is_err());
            assert!(Window::large(bits).is_ok());
        }
    }

    #[test]
    fn a_window_outside_its_header_is_refused() {
        for bits in 0u8..10 {
            assert_eq!(
                Window::standard(bits),
                Err(ConfigError::StandardWindow { requested: bits })
            );
            assert_eq!(
                Window::large(bits),
                Err(ConfigError::LargeWindow { requested: bits })
            );
        }
        assert_eq!(
            Window::standard(25),
            Err(ConfigError::StandardWindow { requested: 25 })
        );
        assert_eq!(
            Window::large(63),
            Err(ConfigError::LargeWindow { requested: 63 })
        );
        assert_eq!(Window::default(), Window::DEFAULT);
        assert_eq!(Window::DEFAULT.bits(), 22);
    }

    #[test]
    fn block_bits_accept_exactly_the_encoders_range() {
        for bits in 16u8..=24 {
            let block = BlockBits::try_from(bits).expect("a legal block size");
            assert_eq!(block.get(), bits);
            assert_eq!(usize::from(block), usize::from(bits));
            assert_eq!(BlockSize::from(block), BlockSize::Bits(block));
        }
        for bits in [0u8, 15, 25, 255] {
            assert_eq!(
                BlockBits::try_from(bits),
                Err(ConfigError::BlockBits { requested: bits })
            );
        }
        assert_eq!(BlockBits::MIN.get(), 16);
        assert_eq!(BlockBits::MAX.get(), 24);
        assert!(BlockBits::MIN < BlockBits::MAX);
        assert_eq!(BlockSize::default(), BlockSize::Auto);
    }

    #[test]
    fn an_explicit_distance_layout_is_validated_when_it_is_built() {
        for postfix in 0u8..=3 {
            for groups in 0u16..16 {
                let direct = groups << postfix;
                if direct > DistanceParams::MAX_DIRECT_CODES {
                    continue;
                }
                assert_eq!(
                    DistanceParams::explicit(postfix, direct),
                    Ok(DistanceParams::Explicit {
                        postfix_bits: postfix,
                        direct_codes: direct,
                    })
                );
            }
        }
        assert_eq!(
            DistanceParams::explicit(4, 0),
            Err(ConfigError::DistancePostfixBits { requested: 4 })
        );
        assert_eq!(
            DistanceParams::explicit(0, 121),
            Err(ConfigError::DirectDistanceCodes { requested: 121 })
        );
        assert_eq!(
            DistanceParams::explicit(2, 6),
            Err(ConfigError::MisalignedDistanceCodes {
                postfix_bits: 2,
                direct_codes: 6,
            })
        );
        // Sixteen groups is one too many for the four-bit field.
        assert_eq!(
            DistanceParams::explicit(0, 16),
            Err(ConfigError::MisalignedDistanceCodes {
                postfix_bits: 0,
                direct_codes: 16,
            })
        );
        assert_eq!(DistanceParams::default(), DistanceParams::Auto);
    }

    #[test]
    fn the_literal_context_policy_resolves_the_way_the_reference_does() {
        assert!(LiteralContextMode::Auto.resolve());
        assert!(LiteralContextMode::Enabled.resolve());
        assert!(!LiteralContextMode::Disabled.resolve());
        assert_eq!(LiteralContextMode::default(), LiteralContextMode::Auto);
    }

    #[test]
    fn every_mode_lowers_to_its_own_encoder_mode() {
        assert_eq!(CompressionMode::Generic.resolve(), CompressMode::Generic);
        assert_eq!(CompressionMode::Text.resolve(), CompressMode::Text);
        assert_eq!(CompressionMode::Font.resolve(), CompressMode::Font);
        assert_eq!(CompressionMode::default(), CompressionMode::Generic);
    }

    #[test]
    fn the_default_configuration_mirrors_the_reference() {
        let config = EncoderConfig::default();
        assert_eq!(config.quality(), Quality::Q11);
        assert_eq!(config.window(), Window::DEFAULT);
        assert_eq!(config.block_size(), BlockSize::Auto);
        assert_eq!(config.mode(), CompressionMode::Generic);
        assert_eq!(config.distance(), DistanceParams::Auto);
        assert_eq!(config.literal_context(), LiteralContextMode::Auto);
    }

    #[test]
    fn every_setter_changes_only_its_own_field() {
        let base = EncoderConfig::default();
        let quality = base.with_quality(Quality::Q1);
        assert_eq!(quality.quality(), Quality::Q1);
        assert_eq!(quality.window(), base.window());
        assert_eq!(quality.mode(), base.mode());

        let window = base.with_window(Window::large(30).expect("legal"));
        assert_eq!(window.window().bits(), 30);
        assert_eq!(window.quality(), base.quality());

        let block = base.with_block_size(BlockSize::Bits(BlockBits::MAX));
        assert_eq!(block.block_size(), BlockSize::Bits(BlockBits::MAX));
        assert_eq!(block.distance(), base.distance());

        let mode = base.with_mode(CompressionMode::Font);
        assert_eq!(mode.mode(), CompressionMode::Font);

        let distance = base.with_distance(DistanceParams::explicit(1, 12).expect("legal"));
        assert_eq!(
            distance.distance(),
            DistanceParams::Explicit {
                postfix_bits: 1,
                direct_codes: 12,
            }
        );

        let literals = base.with_literal_context(LiteralContextMode::Disabled);
        assert_eq!(literals.literal_context(), LiteralContextMode::Disabled);
        assert_eq!(literals.quality(), base.quality());
    }

    #[test]
    fn a_large_window_is_refused_only_at_the_qualities_that_cannot_carry_one() {
        let large = Window::large(30).expect("legal");
        for value in 0u8..=11 {
            let quality = Quality::try_from(value).expect("legal");
            let config = EncoderConfig::default()
                .with_quality(quality)
                .with_window(large);
            if value <= 2 {
                assert_eq!(
                    config.validate(),
                    Err(ConfigError::LargeWindowUnsupportedForQuality { quality })
                );
            } else {
                assert_eq!(config.validate(), Ok(()));
            }
            // An ordinary window is never refused.
            assert_eq!(
                EncoderConfig::default().with_quality(quality).validate(),
                Ok(())
            );
        }
    }

    #[test]
    fn lowering_carries_every_field_and_the_operations_size_hint() {
        let config = EncoderConfig::default()
            .with_quality(Quality::Q5)
            .with_window(Window::large(30).expect("legal"))
            .with_block_size(BlockSize::Bits(BlockBits::MAX))
            .with_mode(CompressionMode::Text)
            .with_distance(DistanceParams::explicit(1, 12).expect("legal"))
            .with_literal_context(LiteralContextMode::Disabled);
        let lowered = config.lower(Some(4096));

        assert_eq!(lowered.quality, QualityLevel::Q5);
        assert_eq!(lowered.lgwin, WindowBits::large(30).expect("legal"));
        assert_eq!(lowered.lgblock, Some(BlockBits::MAX));
        assert_eq!(lowered.mode, CompressMode::Text);
        assert_eq!(lowered.size_hint, Some(4096));
        assert_eq!(lowered.distance_codes.postfix_bits(), 1);
        assert_eq!(lowered.distance_codes.direct_codes(), 12);
        assert!(!lowered.literal_context_modeling);

        // The size hint is the only thing an operation contributes.
        assert_eq!(config.lower(None).size_hint, None);
    }

    #[test]
    fn a_configuration_error_says_what_was_asked_for() {
        let messages = [
            ConfigError::Quality { requested: 12 }.to_string(),
            ConfigError::StandardWindow { requested: 25 }.to_string(),
            ConfigError::LargeWindow { requested: 63 }.to_string(),
            ConfigError::BlockBits { requested: 15 }.to_string(),
            ConfigError::DistancePostfixBits { requested: 4 }.to_string(),
            ConfigError::DirectDistanceCodes { requested: 121 }.to_string(),
            ConfigError::MisalignedDistanceCodes {
                postfix_bits: 2,
                direct_codes: 6,
            }
            .to_string(),
            ConfigError::LargeWindowUnsupportedForQuality {
                quality: Quality::Q1,
            }
            .to_string(),
        ];
        assert!(messages[0].contains("12"));
        assert!(messages[1].contains("25"));
        assert!(messages[2].contains("63"));
        assert!(messages[3].contains("15"));
        assert!(messages[4].contains('4'));
        assert!(messages[5].contains("121"));
        assert!(messages[6].contains('6'));
        assert!(messages[7].contains('1'));
        assert_eq!(
            SizeOverflow.to_string(),
            "the compressed-size bound overflows the address space"
        );
    }
}
