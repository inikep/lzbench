//! Deciding how many literal contexts a meta-block should use.
//!
//! Ports `EstimateEntropy`, `ChooseContextMap`, `ShouldUseComplexStaticContextMap`
//! and `DecideOverLiteralContextModeling` from `c/enc/encode.c` of the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Context modelling costs the decoder time, so the encoder only turns it on
//! when sampling the data says it will pay for itself. Quality five is the
//! first quality that considers it at all; qualities five and six are barred
//! from the three-context model, which the reference reserves for quality
//! seven and above.

use crate::shared::fast_log::fast_log2;
use crate::shared::format::{
    CONTEXT_LUT_UTF8, MAX_STATIC_CONTEXTS, STATIC_CONTEXT_MAP_COMPLEX_UTF8,
    STATIC_CONTEXT_MAP_CONTINUATION, STATIC_CONTEXT_MAP_SIMPLE_UTF8,
};

/// Shortest meta-block that is worth analysing at all.
const MIN_ANALYSED_LENGTH: usize = 64;

/// Bytes examined at each sampling point.
const STRIDE_LENGTH: usize = 64;

/// Distance between sampling points.
const STRIDE_STEP: usize = 4096;

/// Size hint below which the thirteen-context map is not even considered.
const COMPLEX_MAP_SIZE_HINT: usize = 1 << 20;

/// Bits per symbol a context model has to save to be worth using.
const MIN_SAVINGS: f64 = 0.2;

/// Entropy above which data is judged too incompressible for the complex map.
const COMPLEX_MAP_MAX_ENTROPY: f64 = 3.0;

/// Returns the context of a literal given the two bytes before it.
#[inline(always)]
pub(crate) fn context(prev1: u8, prev2: u8) -> usize {
    usize::from(CONTEXT_LUT_UTF8[usize::from(prev1)] | CONTEXT_LUT_UTF8[256 + usize::from(prev2)])
}

/// Returns the Shannon entropy of `population`, in bits times the total count.
///
/// Mirrors `EstimateEntropy`. This is the plain Shannon measure rather than
/// [`crate::shared::histogram::bits_entropy`]: the prefix that a context predicts is
/// coded together with the rest of the byte, so the "at least one bit per
/// symbol" floor does not apply.
fn estimate_entropy(population: &[u32]) -> f64 {
    let mut total = 0usize;
    let mut result = 0f64;
    for &value in population {
        let value = value as usize;
        total += value;
        result += value as f64 * fast_log2(value);
    }
    total as f64 * fast_log2(total) - result
}

/// The literal context model a meta-block will use.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContextModel {
    /// Number of distinct literal contexts.
    pub(crate) num_contexts: usize,
    /// Map from the sixty-four raw contexts to those, if there is more than one.
    pub(crate) map: Option<&'static [u32; 64]>,
}

impl ContextModel {
    /// The model that gives every literal the same prefix code.
    pub(crate) const SINGLE: Self = Self {
        num_contexts: 1,
        map: None,
    };
}

/// Chooses between one, two and three contexts from a bigram histogram.
///
/// Mirrors `ChooseContextMap`. Below `MIN_QUALITY_FOR_HQ_CONTEXT_MODELING` the
/// caller passes `hq == false`, and the three-context option is priced out of
/// reach rather than compared honestly — the reference does this because three
/// context models cost more to decode than they are worth at those qualities.
fn choose_context_map(bigram_histo: &[u32; 9], hq: bool) -> ContextModel {
    let mut monogram_histo = [0u32; 3];
    let mut two_prefix_histo = [0u32; 6];
    for (index, &count) in bigram_histo.iter().enumerate() {
        monogram_histo[index % 3] += count;
        two_prefix_histo[index % 6] += count;
    }

    let mut entropy = [0f64; 4];
    entropy[1] = estimate_entropy(&monogram_histo);
    entropy[2] =
        estimate_entropy(&two_prefix_histo[..3]) + estimate_entropy(&two_prefix_histo[3..]);
    entropy[3] = (0..3)
        .map(|index| estimate_entropy(&bigram_histo[3 * index..3 * index + 3]))
        .sum();

    let total =
        monogram_histo[0] as usize + monogram_histo[1] as usize + monogram_histo[2] as usize;
    if total == 0 {
        return ContextModel::SINGLE;
    }
    entropy[0] = 1.0 / total as f64;
    entropy[1] *= entropy[0];
    entropy[2] *= entropy[0];
    entropy[3] *= entropy[0];

    if !hq {
        // Three context models are slower to decode; below quality seven they
        // are deliberately made ineligible.
        entropy[3] = entropy[1] * 10.0;
    }

    if entropy[1] - entropy[2] < MIN_SAVINGS && entropy[1] - entropy[3] < MIN_SAVINGS {
        ContextModel::SINGLE
    } else if entropy[2] - entropy[3] < 0.02 {
        ContextModel {
            num_contexts: 2,
            map: Some(&STATIC_CONTEXT_MAP_SIMPLE_UTF8),
        }
    } else {
        ContextModel {
            num_contexts: 3,
            map: Some(&STATIC_CONTEXT_MAP_CONTINUATION),
        }
    }
}

/// Considers the thirteen-context map for long inputs.
///
/// Mirrors `ShouldUseComplexStaticContextMap`. The thresholds were tuned by
/// the reference against the Silesia corpus; they are strict, so the map is
/// only chosen when it reliably helps.
fn should_use_complex_static_context_map(
    input: &[u8],
    start_pos: usize,
    length: usize,
    mask: usize,
    size_hint: usize,
) -> Option<ContextModel> {
    if size_hint < COMPLEX_MAP_SIZE_HINT {
        return None;
    }
    let end_pos = start_pos + length;
    let mut combined_histo = [0u32; 32];
    let mut context_histo = [[0u32; 32]; MAX_STATIC_CONTEXTS];
    let mut total = 0usize;

    // Sixty-four byte strides every four kibibytes, over the five most
    // significant bits of each literal.
    let mut start_pos = start_pos;
    while start_pos + STRIDE_LENGTH <= end_pos {
        let stride_end_pos = start_pos + STRIDE_LENGTH;
        let mut prev2 = input[start_pos & mask];
        let mut prev1 = input[(start_pos + 1) & mask];
        for pos in start_pos + 2..stride_end_pos {
            let literal = input[pos & mask];
            let context = STATIC_CONTEXT_MAP_COMPLEX_UTF8[context(prev1, prev2)] as usize;
            total += 1;
            combined_histo[usize::from(literal >> 3)] += 1;
            if let Some(bucket) = context_histo.get_mut(context) {
                bucket[usize::from(literal >> 3)] += 1;
            }
            prev2 = prev1;
            prev1 = literal;
        }
        start_pos += STRIDE_STEP;
    }
    if total == 0 {
        return None;
    }

    let inverse_total = 1.0 / total as f64;
    let combined = estimate_entropy(&combined_histo) * inverse_total;
    let contextual = context_histo
        .iter()
        .map(|bucket| estimate_entropy(bucket))
        .sum::<f64>()
        * inverse_total;

    if contextual > COMPLEX_MAP_MAX_ENTROPY || combined - contextual < MIN_SAVINGS {
        return None;
    }
    Some(ContextModel {
        num_contexts: MAX_STATIC_CONTEXTS,
        map: Some(&STATIC_CONTEXT_MAP_COMPLEX_UTF8),
    })
}

/// Chooses the literal context model for one meta-block.
///
/// Mirrors `DecideOverLiteralContextModeling`. Returns
/// [`ContextModel::SINGLE`] whenever modelling is switched off, the quality is
/// too low, or the block is too short to sample.
pub(crate) fn decide_over_literal_context_modeling(
    input: &[u8],
    start_pos: usize,
    length: usize,
    mask: usize,
    models_contexts: bool,
    hq_contexts: bool,
    size_hint: usize,
) -> ContextModel {
    if !models_contexts || length < MIN_ANALYSED_LENGTH {
        return ContextModel::SINGLE;
    }
    if let Some(model) =
        should_use_complex_static_context_map(input, start_pos, length, mask, size_hint)
    {
        return model;
    }

    // Bigram statistics over the UTF-8 byte prefixes, sampled the same way.
    let end_pos = start_pos + length;
    let lut = [0usize, 0, 1, 2];
    let mut bigram_prefix_histo = [0u32; 9];
    let mut start_pos = start_pos;
    while start_pos + STRIDE_LENGTH <= end_pos {
        let stride_end_pos = start_pos + STRIDE_LENGTH;
        let mut prev = lut[usize::from(input[start_pos & mask] >> 6)] * 3;
        for pos in start_pos + 1..stride_end_pos {
            let literal = input[pos & mask];
            bigram_prefix_histo[prev + lut[usize::from(literal >> 6)]] += 1;
            prev = lut[usize::from(literal >> 6)] * 3;
        }
        start_pos += STRIDE_STEP;
    }
    choose_context_map(&bigram_prefix_histo, hq_contexts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn contexts_stay_inside_the_six_bit_range() {
        for prev1 in 0..=255u8 {
            for prev2 in [0u8, 65, 128, 200, 255] {
                assert!(context(prev1, prev2) < 64);
            }
        }
    }

    #[test]
    fn ascii_letters_land_in_the_letter_contexts() {
        // Lower-case consonant after a lower-case letter.
        assert_eq!(context(b'b', b'a'), 60 | 3);
        // A space resets to its own class.
        assert_eq!(context(b' ', b'a'), 8 | 3);
    }

    #[test]
    fn modelling_is_off_below_the_minimum_length() {
        let data = vec![b'a'; 63];
        assert_eq!(
            decide_over_literal_context_modeling(&data, 0, 63, usize::MAX, true, false, 0),
            ContextModel::SINGLE
        );
    }

    #[test]
    fn modelling_is_off_when_the_caller_disabled_it() {
        let data: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(
            decide_over_literal_context_modeling(
                &data,
                0,
                data.len(),
                usize::MAX,
                false,
                false,
                1 << 21
            ),
            ContextModel::SINGLE
        );
    }

    #[test]
    fn uniform_bytes_do_not_earn_a_context_model() {
        let data = vec![b'a'; 40_000];
        let model =
            decide_over_literal_context_modeling(&data, 0, data.len(), usize::MAX, true, false, 0);
        assert_eq!(model, ContextModel::SINGLE);
    }

    #[test]
    fn mixed_utf8_prefixes_earn_a_context_model() {
        // Alternating one-byte and two-byte UTF-8 sequences make the previous
        // byte highly predictive of the next one.
        let mut data = Vec::new();
        while data.len() < 40_000 {
            data.extend_from_slice("añbñcñdñ".as_bytes());
        }
        let model =
            decide_over_literal_context_modeling(&data, 0, data.len(), usize::MAX, true, false, 0);
        assert!(model.num_contexts > 1, "model was {model:?}");
        assert!(model.map.is_some());
    }

    #[test]
    fn the_three_context_model_is_never_chosen_below_quality_seven() {
        // `entropy[3]` is priced out, so the continuation map can only be
        // selected through the `entropy[2] - entropy[3] >= 0.02` branch, which
        // needs the two-context estimate to be worse than ten times the
        // one-context one. Sampling a wide range of shapes must never produce
        // it.
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..64 {
            let mut data = Vec::with_capacity(40_000);
            while data.len() < 40_000 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                data.push((rng >> 24) as u8);
            }
            let model = decide_over_literal_context_modeling(
                &data,
                0,
                data.len(),
                usize::MAX,
                true,
                false,
                0,
            );
            assert_ne!(model.num_contexts, 3, "the continuation map was selected");
        }
    }

    #[test]
    fn the_complex_map_needs_a_large_size_hint() {
        let mut data = Vec::new();
        while data.len() < 200_000 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        let without =
            decide_over_literal_context_modeling(&data, 0, data.len(), usize::MAX, true, false, 0);
        assert_ne!(without.num_contexts, MAX_STATIC_CONTEXTS);

        let with = decide_over_literal_context_modeling(
            &data,
            0,
            data.len(),
            usize::MAX,
            true,
            false,
            COMPLEX_MAP_SIZE_HINT,
        );
        assert_eq!(with.num_contexts, MAX_STATIC_CONTEXTS);
        assert_eq!(with.map, Some(&STATIC_CONTEXT_MAP_COMPLEX_UTF8));
    }

    #[test]
    fn incompressible_data_is_refused_the_complex_map() {
        let mut rng = 0x0BAD_C0DEu64;
        let mut data = Vec::with_capacity(200_000);
        while data.len() < 200_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            data.push((rng >> 24) as u8);
        }
        let model = decide_over_literal_context_modeling(
            &data,
            0,
            data.len(),
            usize::MAX,
            true,
            false,
            COMPLEX_MAP_SIZE_HINT,
        );
        assert_ne!(model.num_contexts, MAX_STATIC_CONTEXTS);
    }

    #[test]
    fn quality_seven_prices_three_contexts_honestly() {
        // A histogram whose three trailing-byte classes are each strongly
        // predicted by the leading one: three contexts are a real win, and are
        // only reachable once the high-quality comparison is enabled.
        let bigram = [900u32, 10, 10, 10, 900, 10, 10, 10, 900];
        assert_eq!(choose_context_map(&bigram, false).num_contexts, 2);
        assert_eq!(choose_context_map(&bigram, true).num_contexts, 3);
        assert_eq!(
            choose_context_map(&bigram, true).map,
            Some(&STATIC_CONTEXT_MAP_CONTINUATION)
        );
    }

    #[test]
    fn an_empty_sample_never_earns_a_context_model() {
        assert_eq!(choose_context_map(&[0; 9], true), ContextModel::SINGLE);
        assert_eq!(choose_context_map(&[0; 9], false), ContextModel::SINGLE);
    }

    #[test]
    fn entropy_of_a_single_symbol_population_is_zero() {
        assert_eq!(estimate_entropy(&[10, 0, 0]), 0.0);
        assert_eq!(estimate_entropy(&[0, 0, 0]), 0.0);
        assert!((estimate_entropy(&[1, 1]) - 2.0).abs() < 1e-9);
    }
}
