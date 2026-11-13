//! The container format of a Zstandard frame.

use core::fmt;

/// Whether frames begin with the Zstandard magic number (`ZSTD_format_e`).
///
/// [`FrameFormat::Magicless`] frames omit the four magic bytes
/// `28 B5 2F FD` and have no skippable frames. They are for containers that
/// already know a stream is Zstandard, as OpenZL does. The encoder writes the
/// bytes of the standard frame without its first four, and the decoder reads
/// the same, from the first frame header byte. Both sides must agree: a
/// decoder cannot tell the formats apart.
///
/// # Examples
///
/// ```
/// use mzstd::FrameFormat;
///
/// assert_eq!(FrameFormat::default(), FrameFormat::Standard);
/// assert_eq!(i32::from(FrameFormat::Magicless), 1);
/// assert_eq!(FrameFormat::try_from(0)?, FrameFormat::Standard);
/// assert!(FrameFormat::try_from(2).is_err());
/// assert_eq!(FrameFormat::Magicless.to_string(), "magicless");
/// # Ok::<(), mzstd::FrameFormatError>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum FrameFormat {
    /// Frames start with the magic number (`ZSTD_f_zstd1`).
    #[default]
    Standard,
    /// Frames start at the frame header (`ZSTD_f_zstd1_magicless`).
    Magicless,
}

/// A number that is not a [`FrameFormat`].
///
/// # Examples
///
/// ```
/// let error = mzstd::FrameFormat::try_from(7).unwrap_err();
/// assert_eq!(error.to_string(), "7 is not a frame format (0 or 1)");
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[error("{0} is not a frame format (0 or 1)")]
pub struct FrameFormatError(i32);

impl TryFrom<i32> for FrameFormat {
    type Error = FrameFormatError;

    /// The format with the reference's numeric value: `0` for
    /// `ZSTD_f_zstd1`, `1` for `ZSTD_f_zstd1_magicless`.
    ///
    /// # Errors
    ///
    /// [`FrameFormatError`] for any other value.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Magicless),
            other => Err(FrameFormatError(other)),
        }
    }
}

impl From<FrameFormat> for i32 {
    fn from(format: FrameFormat) -> Self {
        match format {
            FrameFormat::Standard => 0,
            FrameFormat::Magicless => 1,
        }
    }
}

impl fmt::Display for FrameFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Standard => "standard",
            Self::Magicless => "magicless",
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn numeric_values_round_trip() {
        for format in [FrameFormat::Standard, FrameFormat::Magicless] {
            assert_eq!(FrameFormat::try_from(i32::from(format)), Ok(format));
        }
        assert_eq!(FrameFormat::try_from(-1), Err(FrameFormatError(-1)));
    }

    #[test]
    fn displays_its_name() {
        assert_eq!(FrameFormat::Standard.to_string(), "standard");
        assert_eq!(FrameFormat::Magicless.to_string(), "magicless");
    }
}
