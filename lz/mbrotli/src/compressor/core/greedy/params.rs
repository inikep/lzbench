//! Sanitised encoder parameters and the deterministic hasher plan.
//!
//! Ports `SanitizeParams`, `ComputeLgBlock`, `ComputeRbBits`,
//! `MaxMetablockSize` and `ChooseHasher` from `c/enc/quality.h`, together with
//! `ChooseDistanceParams` from `c/enc/encode.c` and
//! `BrotliInitDistanceParams` from `c/enc/metablock.c` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Everything here is resolved once, before any hot loop runs: the quality
//! routing, the block sizes, the distance alphabet and the matcher choice are
//! all functions of the caller's parameters alone. In particular the matcher
//! never depends on the instruction set, which is what keeps the output
//! identical across SIMD backends.

use crate::compressor::core::rfc9841::window::ResolvedWindow;
use crate::compressor::shared::SharedBrotliError;
use crate::compressor::{
    BrotliCompressError, CompressMode, CompressParams, DistanceCodes, QualityLevel,
};
use crate::shared::constants::WINDOW_GAP;
use crate::shared::distance::{DistanceParams, MAX_NDIRECT, MAX_NPOSTFIX};

/// Smallest quality that splits meta-blocks into blocks.
pub(crate) const MIN_QUALITY_FOR_BLOCK_SPLIT: usize = 4;

/// Smallest quality that searches every candidate at a delayed position.
pub(crate) const MIN_QUALITY_FOR_EXTENSIVE_REFERENCE_SEARCH: usize = 5;

/// Smallest quality that models literal contexts.
pub(crate) const MIN_QUALITY_FOR_CONTEXT_MODELING: usize = 5;

/// Smallest quality that may choose the three-context literal model.
pub(crate) const MIN_QUALITY_FOR_HQ_CONTEXT_MODELING: usize = 7;

/// Symbols buffered before a quality below four has to flush.
pub(crate) const MAX_NUM_DELAYED_SYMBOLS: usize = 0x2FFF;

/// Highest quality that may store a meta-block with static entropy codes.
///
/// Mirrors `MAX_QUALITY_FOR_STATIC_ENTROPY_CODES`.
pub(crate) const MAX_QUALITY_FOR_STATIC_ENTROPY_CODES: usize = 2;

/// The eight qualities this encoder implements.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum GreedyQuality {
    /// Quick matcher, and meta-blocks stored with the format's fixed command
    /// and distance codes while the command count stays small.
    Q2,
    /// Quick matcher, trivial meta-block storage.
    Q3,
    /// Block splitting, histogram optimisation, distance parameters.
    Q4,
    /// Extensive search and literal context modelling.
    Q5,
    /// Thirty-two bucket candidates.
    Q6,
    /// Sixty-four bucket candidates, ten cached distances, three contexts.
    Q7,
    /// A hundred and twenty-eight bucket candidates.
    Q8,
    /// Two hundred and fifty-six candidates, sixteen cached distances,
    /// a larger default block and a later sparse search.
    Q9,
}

impl GreedyQuality {
    /// Returns the numeric quality the reference formulas are written in.
    pub(crate) const fn number(self) -> usize {
        match self {
            Self::Q2 => 2,
            Self::Q3 => 3,
            Self::Q4 => 4,
            Self::Q5 => 5,
            Self::Q6 => 6,
            Self::Q7 => 7,
            Self::Q8 => 8,
            Self::Q9 => 9,
        }
    }

    /// Returns whether this quality splits a meta-block into blocks.
    pub(crate) const fn splits_blocks(self) -> bool {
        self.number() >= MIN_QUALITY_FOR_BLOCK_SPLIT
    }

    /// Returns whether a delayed search re-examines every candidate.
    pub(crate) const fn extensive_reference_search(self) -> bool {
        self.number() >= MIN_QUALITY_FOR_EXTENSIVE_REFERENCE_SEARCH
    }

    /// Returns whether this quality may model literal contexts at all.
    pub(crate) const fn models_literal_contexts(self) -> bool {
        self.number() >= MIN_QUALITY_FOR_CONTEXT_MODELING
    }

    /// Returns whether the three-context literal model is eligible.
    ///
    /// The reference prices it out of reach below quality seven, because three
    /// context models cost the decoder more than they save.
    pub(crate) const fn hq_context_modeling(self) -> bool {
        self.number() >= MIN_QUALITY_FOR_HQ_CONTEXT_MODELING
    }

    /// Returns whether the meta-block may be stored with static entropy codes.
    ///
    /// Mirrors the `quality <= MAX_QUALITY_FOR_STATIC_ENTROPY_CODES` test in
    /// `WriteMetaBlockInternal`.
    pub(crate) const fn uses_static_entropy_codes(self) -> bool {
        self.number() <= MAX_QUALITY_FOR_STATIC_ENTROPY_CODES
    }

    /// Returns how many cached distances the matcher probes.
    ///
    /// `ChooseHasher` uses four below quality seven, ten below quality nine
    /// and all sixteen at quality nine.
    pub(crate) const fn last_distances_to_check(self) -> usize {
        if self.number() < 7 {
            4
        } else if self.number() < 9 {
            10
        } else {
            16
        }
    }
}

impl TryFrom<QualityLevel> for GreedyQuality {
    type Error = BrotliCompressError;

    /// Routes qualities two to nine to the greedy path.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] for every other
    /// quality, which belongs to a different encoder.
    fn try_from(value: QualityLevel) -> Result<Self, Self::Error> {
        match value {
            QualityLevel::Q2 => Ok(Self::Q2),
            QualityLevel::Q3 => Ok(Self::Q3),
            QualityLevel::Q4 => Ok(Self::Q4),
            QualityLevel::Q5 => Ok(Self::Q5),
            QualityLevel::Q6 => Ok(Self::Q6),
            QualityLevel::Q7 => Ok(Self::Q7),
            QualityLevel::Q8 => Ok(Self::Q8),
            QualityLevel::Q9 => Ok(Self::Q9),
            other => Err(BrotliCompressError::UnsupportedQuality(usize::from(other))),
        }
    }
}

/// Chooses the distance alphabet for a quality, mode and caller request.
///
/// Mirrors `ChooseDistanceParams`: font mode asks for one postfix bit and
/// twelve direct codes, every other mode uses whatever the caller configured,
/// and a combination the format cannot express falls back to zero. Qualities
/// below four always use zero.
pub(crate) const fn choose_distance_params(
    quality: GreedyQuality,
    large_window: bool,
    mode: CompressMode,
    codes: DistanceCodes,
) -> DistanceParams {
    let mut postfix_bits = 0u32;
    let mut num_direct = 0u32;

    if quality.number() >= MIN_QUALITY_FOR_BLOCK_SPLIT {
        if matches!(mode, CompressMode::Font) {
            postfix_bits = 1;
            num_direct = 12;
        } else {
            postfix_bits = codes.postfix_bits();
            num_direct = codes.direct_codes();
        }
        let ndirect_msb = (num_direct >> postfix_bits) & 0x0F;
        if postfix_bits > MAX_NPOSTFIX
            || num_direct > MAX_NDIRECT
            || (ndirect_msb << postfix_bits) != num_direct
        {
            postfix_bits = 0;
            num_direct = 0;
        }
    }

    DistanceParams::for_window(large_window, postfix_bits, num_direct)
}

/// The matcher a set of parameters selects, with its shape baked in.
///
/// `ChooseHasher` picks this from quality, window size and size hint only.
/// Nothing about the running machine takes part, so two runs with the same
/// parameters always search the same candidates in the same order.
///
/// The reference build selects the tagged `H58`/`H68` variants in place of
/// `H5`/`H6` for qualities at or below `BROTLI_MAX_SIMD_QUALITY`. Those
/// variants evaluate exactly the same candidates in exactly the same order —
/// see the equivalence argument on [`BucketMatcher`] — so this plan names the
/// portable pair for both, and the differential tests against the C library
/// check that the streams agree byte for byte.
///
/// [`BucketMatcher`]: super::hashers::BucketMatcher
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum HasherPlan {
    /// Quality 2: quick matcher, sixteen bucket bits, a single candidate slot
    /// and a static-dictionary probe.
    H2,
    /// Quality 3: quick matcher, sixteen bucket bits, one candidate slot.
    H3,
    /// Quality 4 below the size-hint threshold: quick matcher with a
    /// shallow static-dictionary probe.
    H4,
    /// Quality 4 at or above the size-hint threshold: wider quick matcher,
    /// seven-byte hash, no dictionary probe.
    H54,
    /// Qualities 5 to 9 with a small window: forgetful chains
    /// (`H40`, `H41`, `H42`).
    Chain(ChainShape),
    /// Bucketed matcher over a four-byte hash (`H5`, tagged `H58`).
    H5(BucketShape),
    /// Bucketed matcher over an eight-byte hash (`H6`, tagged `H68`).
    H6(BucketShape),
}

/// Geometry of an `H5` or `H6` bucket table.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct BucketShape {
    /// Base-2 logarithm of the number of buckets.
    pub(crate) bucket_bits: u32,
    /// Base-2 logarithm of the number of slots per bucket.
    pub(crate) block_bits: u32,
    /// How many cached distances are probed before the bucket.
    pub(crate) last_distances: usize,
}

/// Geometry of an `H40`, `H41` or `H42` forgetful-chain table.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChainShape {
    /// Number of storage banks the chains share.
    pub(crate) num_banks: usize,
    /// Base-2 logarithm of the number of slots in one bank.
    pub(crate) bank_bits: u32,
    /// How many cached distances are probed before the chain.
    pub(crate) last_distances: usize,
    /// How many chain links one search follows (`max_hops`).
    pub(crate) max_hops: usize,
}

/// Size hint at which quality four and above switch to their large matchers.
pub(crate) const LARGE_INPUT_SIZE_HINT: usize = 1 << 20;

/// Returns the chain depth `quality` searches (`self->max_hops`).
///
/// `(quality > 6 ? 7 : 8) << (quality - 4)`, which is 16, 32, 56, 112 and 224
/// for qualities five to nine.
const fn max_hops(quality: GreedyQuality) -> usize {
    let number = quality.number();
    (if number > 6 { 7usize } else { 8usize }) << (number - 4)
}

/// Selects the matcher for `quality`, `lgwin` and `size_hint`.
///
/// Mirrors `ChooseHasher` restricted to qualities three to nine. The
/// large-window matchers `H35`, `H55` and `H65` are unreachable because
/// retained history is capped at thirty bits by `ResolvedWindow::encoder_bits`,
/// however wide a window the stream header declares.
pub(crate) const fn choose_hasher(
    quality: GreedyQuality,
    lgwin: usize,
    size_hint: usize,
) -> HasherPlan {
    match quality {
        // `ChooseHasher` sets the type to the quality itself below five.
        GreedyQuality::Q2 => HasherPlan::H2,
        GreedyQuality::Q3 => HasherPlan::H3,
        GreedyQuality::Q4 => {
            if size_hint >= LARGE_INPUT_SIZE_HINT {
                HasherPlan::H54
            } else {
                HasherPlan::H4
            }
        }
        _ => {
            let number = quality.number();
            let last_distances = quality.last_distances_to_check();
            if lgwin <= 16 {
                // H42 spreads its chains over five hundred and twelve small
                // banks; H40 and H41 share one large one.
                let (num_banks, bank_bits) = if number >= 9 { (512, 9) } else { (1, 16) };
                HasherPlan::Chain(ChainShape {
                    num_banks,
                    bank_bits,
                    last_distances,
                    max_hops: max_hops(quality),
                })
            } else if size_hint >= LARGE_INPUT_SIZE_HINT && lgwin >= 19 {
                HasherPlan::H6(BucketShape {
                    bucket_bits: 15,
                    block_bits: (number - 1) as u32,
                    last_distances,
                })
            } else {
                HasherPlan::H5(BucketShape {
                    bucket_bits: if number < 7 { 14 } else { 15 },
                    block_bits: (number - 1) as u32,
                    last_distances,
                })
            }
        }
    }
}

/// Every parameter the greedy encoder needs, already sanitised.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct GreedyParams {
    #[cfg(feature = "experimental")]
    pub(crate) stream_offset: usize,
    /// Quality this encoder runs at.
    pub(crate) quality: GreedyQuality,
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
    /// Expected total input size, used only to select the matcher.
    pub(crate) size_hint: usize,
    /// Whether literal context modelling is switched off by the caller.
    pub(crate) disable_literal_context_modeling: bool,
    /// Resolved distance alphabet.
    pub(crate) dist: DistanceParams,
    /// Resolved matcher.
    pub(crate) hasher: HasherPlan,
}

impl GreedyParams {
    /// Resolves `params` for a stream whose size hint is `size_hint`.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] when the quality is
    /// outside the range this encoder implements.
    pub(crate) fn new(
        params: &CompressParams,
        size_hint: usize,
    ) -> Result<Self, BrotliCompressError> {
        let quality = GreedyQuality::try_from(params.quality())?;
        if quality.uses_static_entropy_codes() && params.lgwin().is_large() {
            // The fixed distance code this quality may fall back to is built
            // for the RFC 7932 alphabet, so it cannot carry the wider one.
            // `SanitizeParams` drops the request silently; refusing is this
            // crate's contract, and it has to happen here rather than only in
            // the one-shot entry points so the streaming adapters agree.
            return Err(SharedBrotliError::UnsupportedLargeWindow {
                quality: usize::from(params.quality()),
            }
            .into());
        }
        let window = ResolvedWindow::new(params);
        let lgwin = window.encoder_bits();
        let lgblock = compute_lgblock(quality, params.lgblock().map(usize::from), lgwin);
        Ok(Self {
            #[cfg(feature = "experimental")]
            stream_offset: params.stream_offset,
            quality,
            window,
            lgwin,
            lgblock,
            size_hint,
            disable_literal_context_modeling: !params.literal_context_modeling(),
            dist: choose_distance_params(
                quality,
                window.is_large(),
                params.mode(),
                params.distance_codes(),
            ),
            hasher: choose_hasher(quality, lgwin, size_hint),
        })
    }

    /// Returns whether `other` resolves to the same encoder shape, whatever
    /// its size hint.
    ///
    /// The hint reaches the output only through the context-modelling
    /// decision and the matcher's layout choice, both of which take it from
    /// the parameters at every stream, so an encoder built for one hint can
    /// serve another once retargeted.
    pub(crate) fn same_shape(&self, other: &Self) -> bool {
        Self {
            size_hint: other.size_hint,
            ..*self
        } == *other
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

    /// Returns the literal spree length that triggers the sparse search.
    ///
    /// `LiteralSpreeLengthForSparseSearch` uses sixty-four below quality nine
    /// and five hundred and twelve at quality nine.
    pub(crate) const fn random_heuristics_window_size(&self) -> usize {
        if self.quality.number() < 9 { 64 } else { 512 }
    }
}

/// Smallest explicit input block size, in bits.
pub(crate) const MIN_INPUT_BLOCK_BITS: usize = 16;

/// Largest input block size, in bits.
pub(crate) const MAX_INPUT_BLOCK_BITS: usize = 24;

/// Returns the block size `quality` uses (`ComputeLgBlock`).
///
/// Qualities below four always use fourteen bits, whatever the caller asked
/// for; the others default to sixteen, which quality nine raises to
/// `min(18, lgwin)` when the window is wider, and clamp an explicit request
/// into the range the format allows.
pub(crate) const fn compute_lgblock(
    quality: GreedyQuality,
    requested: Option<usize>,
    lgwin: usize,
) -> usize {
    if quality.number() < MIN_QUALITY_FOR_BLOCK_SPLIT {
        return 14;
    }
    match requested {
        None => {
            if quality.number() >= 9 && lgwin > 16 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{BlockBits, WindowBits};

    #[test]
    fn quality_routing_accepts_only_the_greedy_range() {
        for (public, expected) in [
            (QualityLevel::Q2, GreedyQuality::Q2),
            (QualityLevel::Q3, GreedyQuality::Q3),
            (QualityLevel::Q4, GreedyQuality::Q4),
            (QualityLevel::Q5, GreedyQuality::Q5),
            (QualityLevel::Q6, GreedyQuality::Q6),
            (QualityLevel::Q7, GreedyQuality::Q7),
            (QualityLevel::Q8, GreedyQuality::Q8),
            (QualityLevel::Q9, GreedyQuality::Q9),
        ] {
            assert_eq!(GreedyQuality::try_from(public).ok(), Some(expected));
        }
        assert!(matches!(
            GreedyQuality::try_from(QualityLevel::Q1),
            Err(BrotliCompressError::UnsupportedQuality(1))
        ));
        assert!(matches!(
            GreedyQuality::try_from(QualityLevel::Q11),
            Err(BrotliCompressError::UnsupportedQuality(11))
        ));
    }

    #[test]
    fn only_quality_seven_and_above_may_use_three_contexts() {
        assert!(!GreedyQuality::Q5.hq_context_modeling());
        assert!(!GreedyQuality::Q6.hq_context_modeling());
        assert!(GreedyQuality::Q7.hq_context_modeling());
        assert!(GreedyQuality::Q9.hq_context_modeling());
    }

    #[test]
    fn cached_distance_counts_match_the_reference() {
        let expected = [
            (GreedyQuality::Q3, 4),
            (GreedyQuality::Q5, 4),
            (GreedyQuality::Q6, 4),
            (GreedyQuality::Q7, 10),
            (GreedyQuality::Q8, 10),
            (GreedyQuality::Q9, 16),
        ];
        for (quality, count) in expected {
            assert_eq!(quality.last_distances_to_check(), count, "{quality:?}");
        }
    }

    #[test]
    fn chain_hop_limits_match_the_reference() {
        let expected = [
            (GreedyQuality::Q5, 16),
            (GreedyQuality::Q6, 32),
            (GreedyQuality::Q7, 56),
            (GreedyQuality::Q8, 112),
            (GreedyQuality::Q9, 224),
        ];
        for (quality, hops) in expected {
            assert_eq!(max_hops(quality), hops, "{quality:?}");
        }
    }

    #[test]
    fn quality_three_never_splits_and_never_searches_extensively() {
        assert!(!GreedyQuality::Q3.splits_blocks());
        assert!(GreedyQuality::Q4.splits_blocks());
        assert!(!GreedyQuality::Q4.extensive_reference_search());
        assert!(GreedyQuality::Q5.extensive_reference_search());
    }

    #[test]
    fn only_quality_five_models_literal_contexts() {
        assert!(!GreedyQuality::Q3.models_literal_contexts());
        assert!(!GreedyQuality::Q4.models_literal_contexts());
        assert!(GreedyQuality::Q5.models_literal_contexts());
    }

    #[test]
    fn lgblock_is_fourteen_below_the_block_split_quality() {
        assert_eq!(compute_lgblock(GreedyQuality::Q3, None, 22), 14);
        assert_eq!(compute_lgblock(GreedyQuality::Q3, Some(24), 22), 14);
        assert_eq!(compute_lgblock(GreedyQuality::Q4, None, 22), 16);
        assert_eq!(compute_lgblock(GreedyQuality::Q5, None, 22), 16);
        assert_eq!(compute_lgblock(GreedyQuality::Q5, Some(18), 22), 18);
        assert_eq!(compute_lgblock(GreedyQuality::Q5, Some(10), 22), 16);
        assert_eq!(compute_lgblock(GreedyQuality::Q5, Some(30), 22), 24);
    }

    #[test]
    fn quality_nine_raises_the_default_block_with_the_window() {
        // The default only grows once the window is wider than sixteen bits,
        // and stops at eighteen.
        assert_eq!(compute_lgblock(GreedyQuality::Q9, None, 16), 16);
        assert_eq!(compute_lgblock(GreedyQuality::Q9, None, 17), 17);
        assert_eq!(compute_lgblock(GreedyQuality::Q9, None, 18), 18);
        assert_eq!(compute_lgblock(GreedyQuality::Q9, None, 24), 18);
        // An explicit request still wins.
        assert_eq!(compute_lgblock(GreedyQuality::Q9, Some(20), 24), 20);
        // Quality eight keeps the flat default.
        assert_eq!(compute_lgblock(GreedyQuality::Q8, None, 24), 16);
    }

    #[test]
    fn the_sparse_search_threshold_rises_at_quality_nine() {
        for (quality, window) in [
            (QualityLevel::Q5, 64),
            (QualityLevel::Q8, 64),
            (QualityLevel::Q9, 512),
        ] {
            let public = CompressParams::new(quality, WindowBits::DEFAULT);
            let resolved = GreedyParams::new(&public, 0).expect("supported quality");
            assert_eq!(resolved.random_heuristics_window_size(), window);
        }
    }

    #[test]
    fn hasher_selection_follows_the_size_hint_and_window() {
        assert_eq!(choose_hasher(GreedyQuality::Q3, 22, 0), HasherPlan::H3);
        assert_eq!(
            choose_hasher(GreedyQuality::Q3, 22, LARGE_INPUT_SIZE_HINT),
            HasherPlan::H3
        );

        assert_eq!(
            choose_hasher(GreedyQuality::Q4, 22, LARGE_INPUT_SIZE_HINT - 1),
            HasherPlan::H4
        );
        assert_eq!(
            choose_hasher(GreedyQuality::Q4, 22, LARGE_INPUT_SIZE_HINT),
            HasherPlan::H54
        );
        assert_eq!(
            choose_hasher(GreedyQuality::Q4, 22, LARGE_INPUT_SIZE_HINT + 1),
            HasherPlan::H54
        );

        assert!(matches!(
            choose_hasher(GreedyQuality::Q5, 16, 0),
            HasherPlan::Chain(ChainShape {
                num_banks: 1,
                bank_bits: 16,
                last_distances: 4,
                max_hops: 16,
            })
        ));
        assert!(matches!(
            choose_hasher(GreedyQuality::Q5, 18, LARGE_INPUT_SIZE_HINT),
            HasherPlan::H5 { .. }
        ));
        assert!(matches!(
            choose_hasher(GreedyQuality::Q5, 19, LARGE_INPUT_SIZE_HINT - 1),
            HasherPlan::H5 { .. }
        ));
    }

    /// The `ChooseHasher` table of `c/enc/quality.h`, quality by quality.
    #[test]
    fn the_hasher_table_matches_the_reference_comment() {
        let small_window = [
            (GreedyQuality::Q5, 1usize, 16u32, 4usize, 16usize),
            (GreedyQuality::Q6, 1, 16, 4, 32),
            (GreedyQuality::Q7, 1, 16, 10, 56),
            (GreedyQuality::Q8, 1, 16, 10, 112),
            (GreedyQuality::Q9, 512, 9, 16, 224),
        ];
        for (quality, num_banks, bank_bits, last_distances, max_hops) in small_window {
            assert_eq!(
                choose_hasher(quality, 16, LARGE_INPUT_SIZE_HINT),
                HasherPlan::Chain(ChainShape {
                    num_banks,
                    bank_bits,
                    last_distances,
                    max_hops,
                }),
                "{quality:?} with a small window"
            );
        }

        let normal = [
            (GreedyQuality::Q5, 14u32, 4u32, 4usize),
            (GreedyQuality::Q6, 14, 5, 4),
            (GreedyQuality::Q7, 15, 6, 10),
            (GreedyQuality::Q8, 15, 7, 10),
            (GreedyQuality::Q9, 15, 8, 16),
        ];
        for (quality, bucket_bits, block_bits, last_distances) in normal {
            assert_eq!(
                choose_hasher(quality, 22, 0),
                HasherPlan::H5(BucketShape {
                    bucket_bits,
                    block_bits,
                    last_distances,
                }),
                "{quality:?} with an ordinary input"
            );
            assert_eq!(
                choose_hasher(quality, 19, LARGE_INPUT_SIZE_HINT),
                HasherPlan::H6(BucketShape {
                    bucket_bits: 15,
                    block_bits,
                    last_distances,
                }),
                "{quality:?} with a large input"
            );
        }
    }

    #[test]
    fn distance_alphabet_matches_the_reference_formulas() {
        let default = DistanceParams::default();
        assert_eq!(default.alphabet_size_max, 64);
        assert_eq!(default.alphabet_size_limit, 64);
        assert_eq!(default.max_distance, 0x3FF_FFFC);

        let font = DistanceParams::new(1, 12);
        assert_eq!(font.alphabet_size_max, 16 + 12 + (24 << 2));
        assert_eq!(font.postfix_bits, 1);
    }

    #[test]
    fn font_mode_asks_for_the_reference_distance_parameters() {
        let font = choose_distance_params(
            GreedyQuality::Q4,
            false,
            CompressMode::Font,
            DistanceCodes::DEFAULT,
        );
        assert_eq!((font.postfix_bits, font.num_direct), (1, 12));

        let generic = choose_distance_params(
            GreedyQuality::Q4,
            false,
            CompressMode::Generic,
            DistanceCodes::DEFAULT,
        );
        assert_eq!((generic.postfix_bits, generic.num_direct), (0, 0));

        // Quality three never uses non-zero distance parameters.
        let low = choose_distance_params(
            GreedyQuality::Q3,
            false,
            CompressMode::Font,
            DistanceCodes::DEFAULT,
        );
        assert_eq!((low.postfix_bits, low.num_direct), (0, 0));
    }

    #[test]
    fn invalid_distance_parameters_fall_back_to_zero() {
        // `ndirect` must be a multiple of `1 << npostfix` and its high nibble
        // must fit in four bits; the reference drops both otherwise.
        let bad = choose_distance_params(
            GreedyQuality::Q5,
            false,
            CompressMode::Generic,
            DistanceCodes::from_raw(2, 6),
        );
        assert_eq!((bad.postfix_bits, bad.num_direct), (0, 0));

        let good = choose_distance_params(
            GreedyQuality::Q5,
            false,
            CompressMode::Generic,
            DistanceCodes::from_raw(2, 8),
        );
        assert_eq!((good.postfix_bits, good.num_direct), (2, 8));

        let too_many = choose_distance_params(
            GreedyQuality::Q5,
            false,
            CompressMode::Generic,
            DistanceCodes::from_raw(0, 121),
        );
        assert_eq!((too_many.postfix_bits, too_many.num_direct), (0, 0));
    }

    #[test]
    fn derived_sizes_follow_the_window_and_block() -> Result<(), BrotliCompressError> {
        let params = CompressParams::new(QualityLevel::Q5, WindowBits::DEFAULT);
        let resolved = GreedyParams::new(&params, 0)?;
        assert_eq!(resolved.lgblock, 16);
        assert_eq!(resolved.input_block_size(), 1 << 16);
        assert_eq!(resolved.rb_bits(), 23);
        assert_eq!(resolved.max_metablock_size(), 1 << 23);
        assert_eq!(resolved.max_backward_limit(), (1 << 22) - 16);
        assert_eq!(resolved.random_heuristics_window_size(), 64);
        Ok(())
    }

    #[test]
    fn a_huge_ring_buffer_is_capped_at_the_input_block_limit() -> Result<(), BrotliCompressError> {
        let params = CompressParams::new(QualityLevel::Q5, WindowBits::MAX)
            .with_block_bits(Some(BlockBits::MAX));
        let resolved = GreedyParams::new(&params, 0)?;
        assert_eq!(resolved.rb_bits(), 25);
        assert_eq!(resolved.max_metablock_size(), 1 << MAX_INPUT_BLOCK_BITS);
        Ok(())
    }
}
