//! Sanitised parameters for qualities ten and eleven.
//!
//! Ports the quality-ten and quality-eleven arms of `SanitizeParams`,
//! `ComputeLgBlock`, `ComputeRbBits`, `MaxMetablockSize`, `MaxZopfliLen` and
//! `MaxZopfliCandidates` from `c/enc/quality.h` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Both qualities use the same binary-tree matcher, so there is no hasher plan
//! to choose: what separates them is how hard the dynamic program looks. Every
//! one of those limits is resolved here, once, from the caller's parameters
//! alone — nothing about the running machine takes part.

use crate::compressor::core::rfc9841::window::ResolvedWindow;
use crate::compressor::{BrotliCompressError, CompressMode, CompressParams, QualityLevel};
use crate::shared::constants::WINDOW_GAP;
use crate::shared::distance::DistanceParams;
use crate::shared::format::ContextMode;

/// Longest copy quality ten gives distinct lengths (`MAX_ZOPFLI_LEN_QUALITY_10`).
const MAX_ZOPFLI_LEN_Q10: usize = 150;

/// Longest copy quality eleven gives distinct lengths
/// (`MAX_ZOPFLI_LEN_QUALITY_11`).
const MAX_ZOPFLI_LEN_Q11: usize = 325;

/// Largest input block size, in bits (`BROTLI_MAX_INPUT_BLOCK_BITS`).
const MAX_INPUT_BLOCK_BITS: usize = 24;

/// Smallest explicit input block size, in bits.
const MIN_INPUT_BLOCK_BITS: usize = 16;

/// The two qualities this encoder implements.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum HqQuality {
    /// One cost model, one pass, one expanded start position.
    Q10,
    /// Precomputed matches, two passes, five expanded start positions.
    Q11,
}

impl TryFrom<QualityLevel> for HqQuality {
    type Error = BrotliCompressError;

    /// Routes qualities ten and eleven to the high-quality path.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] for every other
    /// quality, which belongs to a different encoder.
    fn try_from(value: QualityLevel) -> Result<Self, Self::Error> {
        match value {
            QualityLevel::Q10 => Ok(Self::Q10),
            QualityLevel::Q11 => Ok(Self::Q11),
            other => Err(BrotliCompressError::UnsupportedQuality(usize::from(other))),
        }
    }
}

/// Every parameter the high-quality encoder needs, already sanitised.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct HqParams {
    #[cfg(feature = "experimental")]
    pub(crate) stream_offset: usize,
    #[cfg(feature = "experimental")]
    pub(crate) dictionary_context_mode: ContextMode,
    /// Quality this encoder runs at.
    pub(crate) quality: HqQuality,
    /// Window this stream declares, and whether it is a large one.
    pub(crate) window: ResolvedWindow,
    /// Base-2 logarithm of the history the encoder actually keeps.
    ///
    /// For an ordinary stream this is the declared window. For a large window
    /// it is capped at `MAX_ENCODER_WINDOW_BITS`, so a stream may declare more
    /// than the encoder ever indexes.
    pub(crate) lgwin: usize,
    /// Base-2 logarithm of the input block size.
    pub(crate) lgblock: usize,
    /// Whether literal context modelling is switched off by the caller.
    pub(crate) disable_literal_context_modeling: bool,
    /// Resolved distance alphabet.
    ///
    /// The meta-block builder re-tunes this per block, so what this carries is
    /// only the alphabet the commands are first encoded with.
    pub(crate) dist: DistanceParams,
}

impl HqParams {
    /// Logical placement shifts dictionary addresses, never ring-buffer probes.
    pub(crate) const fn logical_position(&self, position: usize) -> usize {
        #[cfg(feature = "experimental")]
        {
            position + self.stream_offset
        }
        #[cfg(not(feature = "experimental"))]
        {
            position
        }
    }
    /// Resolves `params` for a stream.
    ///
    /// Unlike the lower qualities, nothing here depends on the size hint: both
    /// qualities use the same matcher whatever the input size.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] when the quality is
    /// outside the range this encoder implements.
    pub(crate) fn new(params: &CompressParams) -> Result<Self, BrotliCompressError> {
        let quality = HqQuality::try_from(params.quality())?;
        let window = ResolvedWindow::new(params);
        let lgwin = window.encoder_bits();
        let lgblock = compute_lgblock(params.lgblock().map(usize::from), lgwin);
        Ok(Self {
            #[cfg(feature = "experimental")]
            stream_offset: params.stream_offset,
            #[cfg(feature = "experimental")]
            dictionary_context_mode: ContextMode::Utf8,
            quality,
            window,
            lgwin,
            lgblock,
            disable_literal_context_modeling: !params.literal_context_modeling(),
            dist: choose_distance_params(window.is_large(), params.mode(), params.distance_codes()),
        })
    }

    /// Returns the longest copy given a distinct length (`MaxZopfliLen`).
    pub(crate) const fn max_zopfli_len(&self) -> usize {
        match self.quality {
            HqQuality::Q10 => MAX_ZOPFLI_LEN_Q10,
            HqQuality::Q11 => MAX_ZOPFLI_LEN_Q11,
        }
    }

    /// Returns how many queued start positions are expanded
    /// (`MaxZopfliCandidates`).
    pub(crate) const fn max_zopfli_candidates(&self) -> usize {
        match self.quality {
            HqQuality::Q10 => 1,
            HqQuality::Q11 => 5,
        }
    }

    /// Returns how many recent positions the short backward scan examines.
    pub(crate) const fn short_scan(&self) -> usize {
        match self.quality {
            HqQuality::Q10 => 16,
            HqQuality::Q11 => 64,
        }
    }

    /// Returns how many times the block splitter refines its partition.
    pub(crate) const fn split_iterations(&self) -> usize {
        match self.quality {
            HqQuality::Q10 => 3,
            HqQuality::Q11 => 10,
        }
    }

    /// Returns the number of bytes one `process` call may consume.
    pub(crate) const fn input_block_size(&self) -> usize {
        1usize << self.lgblock
    }

    /// Returns the ring buffer's window size in bits (`ComputeRbBits`).
    pub(crate) const fn rb_bits(&self) -> usize {
        1 + if self.lgwin > self.lgblock {
            self.lgwin
        } else {
            self.lgblock
        }
    }

    /// Returns the largest meta-block this encoder emits.
    pub(crate) const fn max_metablock_size(&self) -> usize {
        let bits = if self.rb_bits() < MAX_INPUT_BLOCK_BITS {
            self.rb_bits()
        } else {
            MAX_INPUT_BLOCK_BITS
        };
        1usize << bits
    }

    /// Returns the largest backward distance (`BROTLI_MAX_BACKWARD_LIMIT`).
    pub(crate) const fn max_backward_limit(&self) -> usize {
        (1usize << self.lgwin) - WINDOW_GAP
    }

    /// Returns the literal context model for a block (`ChooseContextMode`).
    ///
    /// Quality ten is the first that considers anything but the UTF-8 model;
    /// data that does not look like text is modelled as signed integers
    /// instead.
    pub(crate) fn choose_context_mode(
        &self,
        data: &[u8],
        pos: usize,
        mask: usize,
        length: usize,
    ) -> ContextMode {
        if super::utf8::is_mostly_utf8(data, pos, mask, length) {
            ContextMode::Utf8
        } else {
            ContextMode::Signed
        }
    }
}

/// Returns the block size these qualities use (`ComputeLgBlock`).
///
/// The default is sixteen, raised to `min(18, lgwin)` when the window is wider
/// — the same rule quality nine follows; an explicit request is clamped into
/// the range the format allows.
const fn compute_lgblock(requested: Option<usize>, lgwin: usize) -> usize {
    match requested {
        None => {
            if lgwin > 16 {
                if lgwin < 18 { lgwin } else { 18 }
            } else {
                16
            }
        }
        Some(lgblock) => {
            if lgblock < MIN_INPUT_BLOCK_BITS {
                MIN_INPUT_BLOCK_BITS
            } else if lgblock > MAX_INPUT_BLOCK_BITS {
                MAX_INPUT_BLOCK_BITS
            } else {
                lgblock
            }
        }
    }
}

/// Chooses the starting distance alphabet (`ChooseDistanceParams`).
///
/// The meta-block builder re-tunes this per block, so what matters here is only
/// that the commands are first encoded with the same alphabet the reference
/// starts from.
const fn choose_distance_params(
    large_window: bool,
    mode: CompressMode,
    codes: crate::compressor::DistanceCodes,
) -> DistanceParams {
    let mut postfix_bits;
    let mut num_direct;
    if matches!(mode, CompressMode::Font) {
        postfix_bits = 1;
        num_direct = 12;
    } else {
        postfix_bits = codes.postfix_bits();
        num_direct = codes.direct_codes();
    }
    let ndirect_msb = (num_direct >> postfix_bits) & 0x0F;
    if postfix_bits > crate::shared::distance::MAX_NPOSTFIX
        || num_direct > crate::shared::distance::MAX_NDIRECT
        || (ndirect_msb << postfix_bits) != num_direct
    {
        postfix_bits = 0;
        num_direct = 0;
    }
    DistanceParams::for_window(large_window, postfix_bits, num_direct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{BlockBits, DistanceCodes, WindowBits};
    use alloc::vec::Vec;

    /// Resolves the parameters for one quality and window size.
    fn params(quality: QualityLevel, lgwin: u8) -> HqParams {
        let lgwin = WindowBits::standard(lgwin).unwrap_or(WindowBits::DEFAULT);
        HqParams::new(&CompressParams::new(quality, lgwin)).expect("supported quality")
    }

    #[test]
    fn quality_routing_accepts_only_the_high_quality_range() {
        assert_eq!(
            HqQuality::try_from(QualityLevel::Q10).ok(),
            Some(HqQuality::Q10)
        );
        assert_eq!(
            HqQuality::try_from(QualityLevel::Q11).ok(),
            Some(HqQuality::Q11)
        );
        assert!(matches!(
            HqQuality::try_from(QualityLevel::Q9),
            Err(BrotliCompressError::UnsupportedQuality(9))
        ));
        assert!(matches!(
            HqQuality::try_from(QualityLevel::Q0),
            Err(BrotliCompressError::UnsupportedQuality(0))
        ));
    }

    #[test]
    fn the_search_limits_match_the_reference() {
        let q10 = params(QualityLevel::Q10, 22);
        assert_eq!(q10.max_zopfli_len(), 150);
        assert_eq!(q10.max_zopfli_candidates(), 1);
        assert_eq!(q10.short_scan(), 16);
        assert_eq!(q10.split_iterations(), 3);

        let q11 = params(QualityLevel::Q11, 22);
        assert_eq!(q11.max_zopfli_len(), 325);
        assert_eq!(q11.max_zopfli_candidates(), 5);
        assert_eq!(q11.short_scan(), 64);
        assert_eq!(q11.split_iterations(), 10);
    }

    #[test]
    fn the_default_block_grows_with_the_window() {
        assert_eq!(compute_lgblock(None, 10), 16);
        assert_eq!(compute_lgblock(None, 16), 16);
        assert_eq!(compute_lgblock(None, 17), 17);
        assert_eq!(compute_lgblock(None, 18), 18);
        assert_eq!(compute_lgblock(None, 24), 18);
    }

    #[test]
    fn an_explicit_block_size_is_clamped_into_range() {
        assert_eq!(compute_lgblock(Some(10), 22), MIN_INPUT_BLOCK_BITS);
        assert_eq!(compute_lgblock(Some(20), 22), 20);
        assert_eq!(compute_lgblock(Some(30), 22), MAX_INPUT_BLOCK_BITS);
    }

    #[test]
    fn derived_sizes_follow_the_window_and_block() {
        let resolved = params(QualityLevel::Q11, 22);
        assert_eq!(resolved.lgblock, 18);
        assert_eq!(resolved.input_block_size(), 1 << 18);
        assert_eq!(resolved.rb_bits(), 23);
        assert_eq!(resolved.max_metablock_size(), 1 << 23);
        assert_eq!(resolved.max_backward_limit(), (1 << 22) - 16);
    }

    #[test]
    fn a_huge_ring_buffer_is_capped_at_the_input_block_limit() {
        let public = CompressParams::new(QualityLevel::Q11, WindowBits::MAX)
            .with_block_bits(Some(BlockBits::MAX));
        let resolved = HqParams::new(&public).expect("supported quality");
        assert_eq!(resolved.rb_bits(), 25);
        assert_eq!(resolved.max_metablock_size(), 1 << MAX_INPUT_BLOCK_BITS);
    }

    #[test]
    fn font_mode_asks_for_the_reference_distance_parameters() {
        let font = choose_distance_params(false, CompressMode::Font, DistanceCodes::DEFAULT);
        assert_eq!((font.postfix_bits, font.num_direct), (1, 12));

        let generic = choose_distance_params(false, CompressMode::Generic, DistanceCodes::DEFAULT);
        assert_eq!((generic.postfix_bits, generic.num_direct), (0, 0));
    }

    #[test]
    fn an_unrepresentable_distance_layout_falls_back_to_zero() {
        let bad =
            choose_distance_params(false, CompressMode::Generic, DistanceCodes::from_raw(2, 6));
        assert_eq!((bad.postfix_bits, bad.num_direct), (0, 0));

        let good =
            choose_distance_params(false, CompressMode::Generic, DistanceCodes::from_raw(2, 8));
        assert_eq!((good.postfix_bits, good.num_direct), (2, 8));
    }

    #[test]
    fn the_context_mode_follows_whether_the_block_looks_like_text() {
        let resolved = params(QualityLevel::Q11, 22);
        let text = b"The quick brown fox jumps over the lazy dog, repeatedly.";
        assert_eq!(
            resolved.choose_context_mode(text, 0, usize::MAX, text.len()),
            ContextMode::Utf8
        );

        let binary: Vec<u8> = (0..256u32).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(
            resolved.choose_context_mode(&binary, 0, usize::MAX, binary.len()),
            ContextMode::Signed
        );
    }
}
