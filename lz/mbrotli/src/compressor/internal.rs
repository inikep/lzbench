//! The encoder parameters and errors the `core` tree is written against.
//!
//! These are the shapes the ported encoders take, kept exactly as the port
//! produced them: a flat per-call `CompressParams`, a closed `QualityLevel`,
//! and one error enum covering every low-level failure. They are private.
//!
//! The public API in [`super::config`] and [`super::error`] is a different
//! shape — validated once, split by domain, and free of per-call values — and
//! converts into these on the way down. Keeping the two apart is what lets the
//! public surface be redesigned without touching a byte of the encoders, and
//! therefore without moving the bitstream.

use super::config::BlockBits;
use super::shared::SharedBrotliError;
use thiserror::Error;

/// Every knob the encoder exposes, resolved per compression call.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompressParams {
    #[cfg(feature = "experimental")]
    pub(crate) stream_offset: usize,
    pub(crate) quality: QualityLevel,
    pub(crate) lgwin: WindowBits,
    pub(crate) lgblock: Option<BlockBits>,
    pub(crate) mode: CompressMode,
    pub(crate) size_hint: Option<usize>,
    pub(crate) distance_codes: DistanceCodes,
    pub(crate) literal_context_modeling: bool,
}

impl CompressParams {
    /// Creates compression parameters from a quality level and a window size.
    ///
    /// Everything else starts at the encoder's own default: a generic mode, an
    /// automatically chosen block size, no size hint, no direct distance codes
    /// and literal context modelling left on.
    pub(crate) const fn new(quality: QualityLevel, lgwin: WindowBits) -> Self {
        Self {
            #[cfg(feature = "experimental")]
            stream_offset: 0,
            quality,
            lgwin,
            lgblock: None,
            mode: CompressMode::Generic,
            size_hint: None,
            distance_codes: DistanceCodes::DEFAULT,
            literal_context_modeling: true,
        }
    }

    /// Returns the configured quality level.
    pub(crate) const fn quality(&self) -> QualityLevel {
        self.quality
    }

    /// Returns the configured sliding window size.
    pub(crate) const fn lgwin(&self) -> WindowBits {
        self.lgwin
    }

    /// Sets the input block size, or restores the encoder's own choice.
    ///
    /// Qualities below four ignore this: they always work in blocks of
    /// `1 << 14` bytes.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn with_block_bits(mut self, lgblock: Option<BlockBits>) -> Self {
        self.lgblock = lgblock;
        self
    }

    /// Returns the configured input block size, if one was requested.
    pub(crate) const fn lgblock(&self) -> Option<BlockBits> {
        self.lgblock
    }

    /// Sets the kind of data being compressed.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn with_mode(mut self, mode: CompressMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the configured mode.
    pub(crate) const fn mode(&self) -> CompressMode {
        self.mode
    }

    /// Sets the expected total input size.
    ///
    /// Qualities four and five pick a different match finder for inputs of a
    /// mebibyte or more, so the hint changes the compressed bytes. The one-shot
    /// entry points substitute the real input length when no hint is given,
    /// which is what makes them match the reference encoder's one-shot API; the
    /// streaming adapters have no such length to substitute and treat a missing
    /// hint as zero. Set it explicitly to make both agree.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn with_size_hint(mut self, size_hint: Option<usize>) -> Self {
        self.size_hint = size_hint;
        self
    }

    /// Returns the configured size hint, if one was given.
    pub(crate) const fn size_hint(&self) -> Option<usize> {
        self.size_hint
    }

    /// Returns the configured distance code layout.
    pub(crate) const fn distance_codes(&self) -> DistanceCodes {
        self.distance_codes
    }

    /// Enables or disables literal context modelling.
    ///
    /// Only quality five and above model literal contexts; switching it off
    /// trades compression ratio for decoding speed.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn with_literal_context_modeling(mut self, enabled: bool) -> Self {
        self.literal_context_modeling = enabled;
        self
    }

    /// Returns whether literal context modelling is enabled.
    pub(crate) const fn literal_context_modeling(&self) -> bool {
        self.literal_context_modeling
    }
}

/// The kind of data a stream carries.
///
/// The encoder uses this as a hint only: every mode produces a valid stream
/// that any decoder reads back identically.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) enum CompressMode {
    /// No assumption about the data.
    #[default]
    Generic,
    /// UTF-8 text.
    Text,
    /// Font data, in the WOFF 2.0 sense.
    Font,
}

/// Layout of the distance alphabet: postfix bits and direct distance codes.
///
/// The two numbers are not independent. RFC 7932 allows at most three postfix
/// bits and one hundred and twenty direct codes, and the number of direct codes
/// has to be a multiple of `1 << postfix_bits` whose quotient still fits in four
/// bits. Every way of building a `DistanceCodes` enforces all three rules, so a
/// value of this type always describes an alphabet the format can express.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DistanceCodes {
    postfix_bits: u32,
    direct_codes: u32,
}

impl DistanceCodes {
    /// The alphabet with neither postfix bits nor direct distance codes.
    pub(crate) const DEFAULT: Self = Self {
        postfix_bits: 0,
        direct_codes: 0,
    };

    /// Returns the number of postfix bits.
    pub(crate) const fn postfix_bits(&self) -> u32 {
        self.postfix_bits
    }

    /// Returns the number of direct distance codes.
    pub(crate) const fn direct_codes(&self) -> u32 {
        self.direct_codes
    }

    /// Builds a pair without re-validating it.
    ///
    /// The public `DistanceParams` validated the pair when it was built, and
    /// the encoders' own sanitiser re-checks anything it did not, so this is
    /// the lowering step rather than a way in.
    pub(crate) const fn from_raw(postfix_bits: u32, direct_codes: u32) -> Self {
        Self {
            postfix_bits,
            direct_codes,
        }
    }
}

/// Base-2 logarithm of the Brotli sliding window size, and the syntax it uses.
///
/// Brotli has two stream headers for the window, and which one a stream carries
/// is a choice rather than a consequence of the size:
///
/// - [`WindowBits::standard`] writes the RFC 7932 header and allows `10..=24`.
/// - [`WindowBits::large`] writes the fourteen-bit [RFC 9841] Large Window
///   header and allows `10..=62`.
///
/// The two ranges overlap on purpose. `WindowBits::large(22)` and
/// `WindowBits::standard(22)` describe the same window size but produce
/// different streams, because the header and the distance alphabet differ, so a
/// large window is never inferred from a size — it is asked for by name.
///
/// Those two constructors are the only way to build a value, and each validates
/// its own range, so a `WindowBits` always describes a window some header can
/// express. Read it back with [`WindowBits::bits`] and
/// [`WindowBits::is_large`].
///
/// A large window above 30 bits is written to the header faithfully but costs
/// nothing: the encoder never keeps more than 30 bits of history, which is also
/// where the reference C encoder stops.
///
/// [RFC 9841]: https://www.rfc-editor.org/rfc/rfc9841.html
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct WindowBits(WindowKind);

/// Which header a window is written with, and how wide it is.
///
/// Private so that the only way to build a [`WindowBits`] is through a
/// constructor that has checked the range for the header it picked.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum WindowKind {
    /// An RFC 7932 window: `10..=24` bits, the ordinary stream header.
    Standard(u8),
    /// An RFC 9841 Large Window: `10..=62` bits, the fourteen-bit header.
    Large(u8),
}

impl WindowBits {
    /// Smallest window either header allows: 2^10 bytes.
    pub(crate) const MIN: Self = Self(WindowKind::Standard(10));

    /// Largest window the RFC 7932 header allows: 2^24 bytes.
    #[cfg(test)]
    pub(crate) const MAX: Self = Self(WindowKind::Standard(24));

    /// Window size used when no other is requested: 2^22 bytes.
    pub(crate) const DEFAULT: Self = Self(WindowKind::Standard(22));

    /// Smallest base-2 logarithm either header allows.
    const MIN_BITS: u8 = 10;

    /// Largest base-2 logarithm the RFC 7932 header allows.
    const MAX_STANDARD_BITS: u8 = 24;

    /// Largest base-2 logarithm the RFC 9841 Large Window header allows.
    const MAX_LARGE_BITS: u8 = 62;

    /// Creates an ordinary RFC 7932 window from its base-2 logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`ParseWindowBitsError::LowerBound`] below ten and
    /// [`ParseWindowBitsError::UpperBound`] above twenty-four. A wider window
    /// needs [`WindowBits::large`], which changes the stream header.
    pub(crate) const fn standard(bits: u8) -> Result<Self, WindowOutOfRange> {
        if bits < Self::MIN_BITS || bits > Self::MAX_STANDARD_BITS {
            return Err(WindowOutOfRange);
        }
        Ok(Self(WindowKind::Standard(bits)))
    }

    /// Creates an RFC 9841 Large Window from its base-2 logarithm.
    ///
    /// Selecting this is always explicit, including for a size an ordinary
    /// window could have expressed: it changes the stream header and the
    /// distance alphabet, so it is never inferred from the size, the input, the
    /// quality or the target.
    ///
    /// # Errors
    ///
    /// Returns [`ParseWindowBitsError::LowerBound`] below ten and
    /// [`ParseWindowBitsError::LargeUpperBound`] above sixty-two.
    pub(crate) const fn large(bits: u8) -> Result<Self, WindowOutOfRange> {
        if bits < Self::MIN_BITS || bits > Self::MAX_LARGE_BITS {
            return Err(WindowOutOfRange);
        }
        Ok(Self(WindowKind::Large(bits)))
    }

    /// Returns the base-2 logarithm of the window size.
    pub(crate) const fn bits(self) -> u8 {
        match self.0 {
            WindowKind::Standard(bits) | WindowKind::Large(bits) => bits,
        }
    }

    /// Returns whether this window uses the RFC 9841 Large Window header.
    pub(crate) const fn is_large(self) -> bool {
        matches!(self.0, WindowKind::Large(_))
    }
}

/// Error returned when a window size falls outside the range its header allows.
///
/// The public [`ConfigError`](crate::ConfigError) names which header refused it
/// and what was asked for; down here the only thing anyone does with this is
/// stop, so it carries nothing.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[error("the window size is outside the range its header can express")]
pub(crate) struct WindowOutOfRange;

/// Compression quality: how much work the encoder spends per byte.
///
/// The variants are ordered the way the format numbers them, so they compare
/// and sort by effort.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum QualityLevel {
    /// One pass over each fragment with static entropy codes: fastest, largest.
    Q0,
    /// Two passes over each fragment, with entropy codes built per block.
    Q1,
    /// Two passes over each fragment with static entropy codes.
    Q2,
    /// Greedy matching with one prefix code for the whole stream.
    Q3,
    /// Adds block splitting, histogram optimisation and distance parameters.
    Q4,
    /// Adds a delayed search and literal context modelling.
    Q5,
    /// Deepens the search to sixty-four bucket candidates.
    Q6,
    /// Deepens the search again and adds the three-context literal model.
    Q7,
    /// Deeper still, with more cached distances checked per position.
    Q8,
    /// The deepest greedy search: 256 bucket candidates, 16 cached distances.
    Q9,
    /// Zopfli search over every match the binary tree finds, with real context
    /// maps built by histogram clustering.
    Q10,
    /// The same search run harder, re-priced from the commands its first pass
    /// produced: slowest, smallest.
    Q11,
}

impl From<QualityLevel> for usize {
    /// Returns the numeric quality understood by the Brotli format.
    fn from(value: QualityLevel) -> Self {
        match value {
            QualityLevel::Q0 => 0,
            QualityLevel::Q1 => 1,
            QualityLevel::Q2 => 2,
            QualityLevel::Q3 => 3,
            QualityLevel::Q4 => 4,
            QualityLevel::Q5 => 5,
            QualityLevel::Q6 => 6,
            QualityLevel::Q7 => 7,
            QualityLevel::Q8 => 8,
            QualityLevel::Q9 => 9,
            QualityLevel::Q10 => 10,
            QualityLevel::Q11 => 11,
        }
    }
}

/// Error returned by the compression entry points.
#[derive(Error, Debug)]
#[non_exhaustive]
pub(crate) enum BrotliCompressError {
    /// The inner reader or writer of a streaming adapter failed.
    #[error("IO error: {0}")]
    #[cfg(not(feature = "no_std"))]
    IOError(#[from] std::io::Error),
    /// The requested quality has no implementation yet.
    #[error("Quality level {0} is not implemented")]
    UnsupportedQuality(usize),
    /// The caller-provided output buffer cannot hold the stream.
    #[error("The output buffer is too small for the compressed stream")]
    OutputTooSmall,
    /// The internal scratch buffer proved too small; this indicates a bug.
    #[error("The encoder output buffer overflowed")]
    BufferOverflow,
    /// The compressed-size bound does not fit in a `usize`.
    #[error("The compressed-size bound overflows the address space")]
    BoundOverflow,
    /// An RFC 9841 shared-Brotli feature reported a failure.
    #[error(transparent)]
    Shared(#[from] SharedBrotliError),
}

/// Result alias used throughout the compressor API.
pub(crate) type BrotliResult<T> = Result<T, BrotliCompressError>;
