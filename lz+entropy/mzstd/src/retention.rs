//! What a codec keeps allocated between operations.

/// What a codec keeps allocated between operations.
///
/// Reuse is the point of a long-lived codec, so the default keeps every
/// buffer. A caller compressing occasionally, or juggling many codecs, can
/// trade that back for memory. The policy never changes the bytes produced.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mzstd::{Compressor, RetentionPolicy};
///
/// let mut encoder = Compressor::new(Default::default())?;
/// encoder.compress(b"warm the tables up")?;
/// assert!(encoder.retained_bytes() > 0);
/// encoder.trim(RetentionPolicy::ReleaseAll);
/// assert_eq!(encoder.retained_bytes(), 0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RetentionPolicy {
    /// Keep every buffer.
    #[default]
    Aggressive,
    /// Keep buffers while they fit inside a ceiling, and release them when not.
    Bounded {
        /// The most the codec may retain, in bytes.
        max_bytes: usize,
    },
    /// Keep nothing between operations.
    ReleaseAll,
}

impl RetentionPolicy {
    /// Whether `retained` bytes may be kept.
    pub(crate) const fn allows(self, retained: usize) -> bool {
        match self {
            Self::Aggressive => true,
            Self::Bounded { max_bytes } => retained <= max_bytes,
            Self::ReleaseAll => retained == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RetentionPolicy;

    #[test]
    fn policies_bound_retained_memory() {
        assert!(RetentionPolicy::Aggressive.allows(usize::MAX));
        assert!(RetentionPolicy::Bounded { max_bytes: 10 }.allows(10));
        assert!(!RetentionPolicy::Bounded { max_bytes: 10 }.allows(11));
        assert!(RetentionPolicy::ReleaseAll.allows(0));
        assert!(!RetentionPolicy::ReleaseAll.allows(1));
    }
}
