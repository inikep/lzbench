//! Shared window descriptions and validated configuration errors.

#[cfg(feature = "compression")]
use crate::compressor::Quality;
use thiserror::Error;

/// Which header a [`Window`] is written with.
///
/// The two are separate syntaxes for the same idea, and a stream carries one or
/// the other. They overlap in size on purpose: a Large Window is asked for by
/// name, never reached by widening a number.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum WindowEncoding {
    /// The RFC 7932 header, which expresses `10..=24` bits.
    Standard,
    /// The RFC 9841 Large Window header, which expresses `10..=62` bits.
    Large,
}

/// The sliding window: how wide it is, and which header declares it.
///
/// Both halves live in one value, because they are one decision.
/// `Window::large(22)` and `Window::standard(22)` describe the same size and
/// produce different streams: the header differs, and so does the distance
/// alphabet. There is no separate `large_window` flag to disagree with the
/// size.
///
/// A declaration wider than the encoder retains costs nothing. The encoder
/// keeps at most 30 bits of history whatever the header says — which is where
/// the reference encoder stops too — so a 62-bit window allocates no more than
/// a 30-bit one and emits the same payload behind a different header.
///
/// # Examples
///
/// ```
/// use mbrotli::{Window, WindowEncoding};
///
/// let ordinary = Window::standard(22)?;
/// let large = Window::large(22)?;
///
/// assert_eq!(ordinary.bits(), large.bits());
/// assert_ne!(ordinary, large);
/// assert_eq!(large.encoding(), WindowEncoding::Large);
///
/// assert!(Window::standard(25).is_err());
/// assert!(Window::large(63).is_err());
/// # Ok::<(), mbrotli::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Window {
    /// Base-2 logarithm of the window size.
    bits: u8,
    /// The header that declares it.
    encoding: WindowEncoding,
}

impl Window {
    /// Smallest window either header expresses: 2^10 bytes.
    pub const MIN_BITS: u8 = 10;

    /// Largest window the RFC 7932 header expresses: 2^24 bytes.
    pub const MAX_STANDARD_BITS: u8 = 24;

    /// Largest window the RFC 9841 header expresses: 2^62 bytes.
    pub const MAX_LARGE_BITS: u8 = 62;

    /// The window used when none is asked for: an ordinary 2^22 bytes.
    pub const DEFAULT: Self = Self {
        bits: 22,
        encoding: WindowEncoding::Standard,
    };

    /// Creates an ordinary RFC 7932 window from its base-2 logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::StandardWindow`] outside `10..=24`. A wider
    /// window needs [`Window::large`], which changes the stream header.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{ConfigError, Window};
    ///
    /// assert_eq!(Window::standard(22)?, Window::DEFAULT);
    /// assert!(matches!(
    ///     Window::standard(9),
    ///     Err(ConfigError::StandardWindow { requested: 9 })
    /// ));
    /// # Ok::<(), ConfigError>(())
    /// ```
    pub const fn standard(bits: u8) -> Result<Self, ConfigError> {
        if bits < Self::MIN_BITS || bits > Self::MAX_STANDARD_BITS {
            return Err(ConfigError::StandardWindow { requested: bits });
        }
        Ok(Self {
            bits,
            encoding: WindowEncoding::Standard,
        })
    }

    /// Creates an RFC 9841 Large Window from its base-2 logarithm.
    ///
    /// Selecting this is always explicit, including for a size the ordinary
    /// header could have expressed: it changes the header and the distance
    /// alphabet, so it is never inferred from the size, the input, the quality
    /// or the target.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::LargeWindow`] outside `10..=62`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{ConfigError, Window, WindowEncoding};
    ///
    /// assert_eq!(Window::large(30)?.encoding(), WindowEncoding::Large);
    /// assert!(matches!(
    ///     Window::large(63),
    ///     Err(ConfigError::LargeWindow { requested: 63 })
    /// ));
    /// # Ok::<(), ConfigError>(())
    /// ```
    pub const fn large(bits: u8) -> Result<Self, ConfigError> {
        if bits < Self::MIN_BITS || bits > Self::MAX_LARGE_BITS {
            return Err(ConfigError::LargeWindow { requested: bits });
        }
        Ok(Self {
            bits,
            encoding: WindowEncoding::Large,
        })
    }

    /// Returns the base-2 logarithm of the window size.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Window;
    ///
    /// assert_eq!(Window::DEFAULT.bits(), 22);
    /// ```
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// Returns the header this window is written with.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Window, WindowEncoding};
    ///
    /// assert_eq!(Window::DEFAULT.encoding(), WindowEncoding::Standard);
    /// ```
    #[must_use]
    pub const fn encoding(self) -> WindowEncoding {
        self.encoding
    }
}

impl Default for Window {
    /// Returns [`Window::DEFAULT`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::Window;
    ///
    /// assert_eq!(Window::default(), Window::DEFAULT);
    /// ```
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Error returned when a configuration cannot be expressed or cannot be used.
///
/// Every variant is a decision the caller made, reported before any input is
/// touched: an individually illegal value from the type that would have held
/// it, and a jointly meaningless combination from
/// `Compressor::new` when compression is enabled.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mbrotli::{ConfigError, Quality};
///
/// let error = Quality::try_from(12u8).unwrap_err();
///
/// assert_eq!(error, ConfigError::Quality { requested: 12 });
/// assert!(error.to_string().contains("11"));
/// # }
/// ```
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// A quality outside the `0..=11` the format defines.
    #[error("quality {requested} is outside the 0..=11 the format defines")]
    #[cfg(feature = "compression")]
    Quality {
        /// The quality that was asked for.
        requested: u8,
    },
    /// A window outside the `10..=24` the RFC 7932 header expresses.
    #[error("an ordinary window of {requested} bits is outside the 10..=24 RFC 7932 expresses")]
    StandardWindow {
        /// The window size that was asked for, in bits.
        requested: u8,
    },
    /// A window outside the `10..=62` the RFC 9841 header expresses.
    #[error("a large window of {requested} bits is outside the 10..=62 RFC 9841 expresses")]
    LargeWindow {
        /// The window size that was asked for, in bits.
        requested: u8,
    },
    /// A block size outside the `16..=24` the encoder accepts.
    #[error("a block size of {requested} bits is outside the 16..=24 the encoder accepts")]
    #[cfg(feature = "compression")]
    BlockBits {
        /// The block size that was asked for, in bits.
        requested: u8,
    },
    /// More than three distance postfix bits.
    #[error("{requested} distance postfix bits is more than the 3 RFC 7932 allows")]
    #[cfg(feature = "compression")]
    DistancePostfixBits {
        /// The number of postfix bits that was asked for.
        requested: u8,
    },
    /// More than one hundred and twenty direct distance codes.
    #[error("{requested} direct distance codes is more than the 120 RFC 7932 allows")]
    #[cfg(feature = "compression")]
    DirectDistanceCodes {
        /// The number of direct codes that was asked for.
        requested: u16,
    },
    /// Direct codes that are not a whole number of postfix groups.
    #[error(
        "{direct_codes} direct distance codes is not a whole number of \
         1 << {postfix_bits} groups the header can hold"
    )]
    #[cfg(feature = "compression")]
    MisalignedDistanceCodes {
        /// The number of postfix bits that was asked for.
        postfix_bits: u8,
        /// The number of direct codes that was asked for.
        direct_codes: u16,
    },
    /// A Large Window at a quality whose distance model cannot carry one.
    ///
    /// Qualities zero, one and two may write distances through a code built for
    /// the RFC 7932 alphabet. The reference silently drops the request; this
    /// crate refuses it, because a stream that quietly stopped being a Large
    /// Window stream is invisible until a decoder disagrees.
    #[error("quality {} cannot carry a large window", quality.get())]
    #[cfg(feature = "compression")]
    LargeWindowUnsupportedForQuality {
        /// The quality that was asked for.
        quality: Quality,
    },
}
