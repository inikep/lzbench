/// What a codec keeps allocated between operations.
///
/// Reuse is the point of a `Compressor`, so the default is to keep everything.
/// A caller compressing occasionally, or juggling many compressors, can trade
/// that back for memory.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mbrotli::{Compressor, EncoderConfig, Quality, RetentionPolicy};
///
/// let config = EncoderConfig::default().with_quality(Quality::Q5);
/// let mut encoder = Compressor::new(config)?;
/// encoder.compress(b"warm the workspace up")?;
/// assert!(encoder.retained_bytes() > 0);
///
/// encoder.trim(RetentionPolicy::ReleaseAll);
/// assert_eq!(encoder.retained_bytes(), 0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum RetentionPolicy {
    /// Keep every buffer, so repeated operations allocate only what the
    /// first streams grow into.
    #[default]
    Aggressive,
    /// Keep only what the current configuration needs.
    ///
    /// Differs from [`RetentionPolicy::Aggressive`] when the configuration
    /// changes: the encoder built for the old one is released at that moment
    /// rather than when the next operation replaces it.
    CurrentConfig,
    /// Keep buffers while they fit inside a ceiling, and release them when not.
    Bounded {
        /// The most the compressor may retain, in bytes.
        max_bytes: usize,
    },
    /// Keep nothing: release the encoder after every operation.
    ReleaseAll,
}
