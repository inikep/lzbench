//! Estimating what each literal of a block would cost to code.
//!
//! Ports `c/enc/literal_cost.c` from the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`).
//!
//! The Zopfli search needs a price for every literal before it knows what the
//! prefix codes will be, so it estimates one from a sliding window of the
//! surrounding bytes. The estimate is deliberately not the true entropy: the
//! constants below — the window widths, the two additive nudges, the halving
//! below one bit, the prologue surcharge — were tuned by the reference against
//! its corpora, and the dynamic program compares the results directly, so each
//! is part of the output contract.

use alloc::vec::Vec;

use super::utf8::is_mostly_utf8;
use crate::shared::fast_log::fast_log2;

/// Half-width of the sliding window used for UTF-8 text.
const UTF8_WINDOW_HALF: usize = 495;

/// Half-width of the sliding window used for everything else.
const BINARY_WINDOW_HALF: usize = 2000;

/// Bytes over which the first-byte surcharge is applied.
const PROLOGUE_LENGTH: usize = 2000;

/// Surcharge per byte across the prologue.
const PROLOGUE_MULTIPLIER: f64 = 0.35 / 2000.0;

/// Flat surcharge at the very first byte of the prologue.
const PROLOGUE_BASE: f64 = 0.35;

/// Nudge added to every UTF-8 literal cost.
const UTF8_NUDGE: f64 = 0.02905;

/// Nudge added to every non-UTF-8 literal cost.
const BINARY_NUDGE: f64 = 0.029;

/// Number of position classes the UTF-8 model keys its histograms on.
const UTF8_POSITIONS: usize = 3;

/// Scratch the estimator needs, allocated once per stream.
///
/// Three histograms, because the UTF-8 model keys on position within a
/// sequence; the binary model uses only the first. The block copy is used
/// only when a block wraps the ring buffer, so the pricing loops always read
/// through one contiguous slice.
pub(crate) struct LiteralCostArena {
    histogram: Vec<u32>,
    block: Vec<u8>,
}

impl LiteralCostArena {
    /// Counts the literal-model histogram and block-copy allocations.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.histogram.capacity() * size_of::<u32>() + self.block.capacity()
    }
}

impl Default for LiteralCostArena {
    /// Returns zeroed histograms and an empty block copy.
    fn default() -> Self {
        Self {
            histogram: vec![0u32; UTF8_POSITIONS * 256],
            block: Vec::new(),
        }
    }
}

/// Returns `pos..pos + len` of the ring buffer as one contiguous slice.
///
/// The bytes are borrowed from `data` when the block does not wrap, which is
/// every block of a one-shot stream, and copied into `scratch` otherwise.
/// Every read the estimators make lies inside this range, so this is the only
/// place the ring-buffer mask is applied.
pub(crate) fn contiguous_block<'a>(
    data: &'a [u8],
    pos: usize,
    mask: usize,
    len: usize,
    scratch: &'a mut Vec<u8>,
) -> &'a [u8] {
    let end = pos + len;
    let direct = end <= data.len() && (end.saturating_sub(1) <= mask || len == 0);
    if let (true, Some(block)) = (direct, data.get(pos..end)) {
        return block;
    }
    scratch.clear();
    scratch.extend((pos..end).map(|index| data.get(index & mask).copied().unwrap_or(0)));
    scratch
}

/// A one-entry cache in front of [`fast_log2`].
///
/// The sliding-window counts change rarely and by one, and the library
/// logarithm they need past the table's end costs more than the rest of a
/// position, so the last argument's value is kept. The cached value is the
/// same function's result, so nothing is rounded differently.
#[derive(Clone, Copy)]
struct Log2Memo {
    value: usize,
    log: f64,
}

impl Log2Memo {
    const EMPTY: Self = Self { value: 0, log: 0.0 };

    #[inline(always)]
    fn get(&mut self, value: usize) -> f64 {
        if value != self.value {
            self.value = value;
            self.log = fast_log2(value);
        }
        self.log
    }
}

/// Returns which byte of a UTF-8 sequence comes next (`UTF8Position`).
///
/// `clamp` is the widest sequence the model distinguishes, so a run of
/// three-byte characters can be modelled as two-byte ones when that compresses
/// better.
#[inline]
const fn utf8_position(last: usize, c: usize, clamp: usize) -> usize {
    if c < 128 {
        // The next byte starts a fresh sequence.
        0
    } else if c >= 192 {
        // A lead byte: the next one is a continuation.
        if clamp < 1 { clamp } else { 1 }
    } else if last < 0xE0 {
        // A continuation that completed a two- or three-byte sequence.
        0
    } else if clamp < 2 {
        clamp
    } else {
        2
    }
}

/// Chooses how many UTF-8 position classes to model (`DecideMultiByteStatsLevel`).
///
/// The reference notes that one is better than two even for three-byte text,
/// and drops to zero when there is barely any multi-byte content at all.
fn decide_multi_byte_stats_level(block: &[u8]) -> usize {
    let mut counts = [0usize; UTF8_POSITIONS];
    let mut last_c = 0usize;
    for &byte in block {
        let c = usize::from(byte);
        counts[utf8_position(last_c, c, 2)] += 1;
        last_c = c;
    }
    // `max_utf8` starts at one and only ever falls, which is what the
    // reference's comment about two compressing worse than one describes.
    if counts[1] + counts[2] < 25 { 0 } else { 1 }
}

/// Prices every literal of `pos..pos + len` into `cost`.
///
/// Mirrors `BrotliEstimateBitCostsForLiterals`, choosing between its UTF-8 and
/// its single-histogram model. `cost` must be at least `len` long.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn estimate_bit_costs_for_literals(
    pos: usize,
    len: usize,
    mask: usize,
    data: &[u8],
    arena: &mut LiteralCostArena,
    cost: &mut [f32],
) {
    let LiteralCostArena { histogram, block } = arena;
    let block = contiguous_block(data, pos, mask, len, block);
    let Some(cost) = cost.get_mut(..len) else {
        return;
    };
    if is_mostly_utf8(block, 0, usize::MAX, len) {
        estimate_utf8(block, histogram, cost);
    } else {
        estimate_binary(block, histogram, cost);
    }
}

/// The three-histogram model for text (`EstimateBitCostsForLiteralsUTF8`).
///
/// `block` and `cost` are the same length.
fn estimate_utf8(block: &[u8], histogram: &mut [u32], cost: &mut [f32]) {
    let len = block.len().min(cost.len());
    let block = &block[..len];
    let cost = &mut cost[..len];
    let Some(histogram) = histogram.get_mut(..UTF8_POSITIONS * 256) else {
        return;
    };
    let max_utf8 = decide_multi_byte_stats_level(block);
    let window_half = UTF8_WINDOW_HALF;
    let in_window = window_half.min(len);
    let mut in_window_utf8 = [0usize; UTF8_POSITIONS];
    let mut window_log = [Log2Memo::EMPTY; UTF8_POSITIONS];
    let mut histo_log = Log2Memo::EMPTY;
    histogram.fill(0);
    let at = |index: usize| usize::from(block[index]);

    {
        // Bootstrap the histograms over the first window.
        let mut last_c = 0usize;
        let mut utf8_pos = 0usize;
        for &byte in &block[..in_window] {
            let c = usize::from(byte);
            histogram[256 * utf8_pos + c] += 1;
            in_window_utf8[utf8_pos] += 1;
            utf8_pos = utf8_position(last_c, c, max_utf8);
            last_c = c;
        }
    }

    for index in 0..len {
        if index >= window_half {
            // Drop the byte leaving the window behind.
            let c = if index < window_half + 1 {
                0
            } else {
                at(index - window_half - 1)
            };
            let last_c = if index < window_half + 2 {
                0
            } else {
                at(index - window_half - 2)
            };
            let utf8_pos = utf8_position(last_c, c, max_utf8);
            histogram[256 * utf8_pos + at(index - window_half)] -= 1;
            in_window_utf8[utf8_pos] -= 1;
        }
        if index + window_half < len {
            // Take in the byte entering the window ahead.
            let c = at(index + window_half - 1);
            let last_c = at(index + window_half - 2);
            let utf8_pos = utf8_position(last_c, c, max_utf8);
            histogram[256 * utf8_pos + at(index + window_half)] += 1;
            in_window_utf8[utf8_pos] += 1;
        }
        {
            let c = if index < 1 { 0 } else { at(index - 1) };
            let last_c = if index < 2 { 0 } else { at(index - 2) };
            let utf8_pos = utf8_position(last_c, c, max_utf8);
            let histo = histogram[256 * utf8_pos + at(index)].max(1) as usize;
            let mut lit_cost =
                window_log[utf8_pos].get(in_window_utf8[utf8_pos]) - histo_log.get(histo);
            lit_cost += UTF8_NUDGE;
            if lit_cost < 1.0 {
                lit_cost *= 0.5;
                lit_cost += 0.5;
            }
            // The reference makes the first bytes dearer; its comment offers
            // the beginning of a file being a statistical anomaly as the
            // reason, and leaves it at that.
            if index < PROLOGUE_LENGTH {
                lit_cost += PROLOGUE_BASE + PROLOGUE_MULTIPLIER * index as f64;
            }
            cost[index] = lit_cost as f32;
        }
    }
}

/// The single-histogram model for everything else.
///
/// `block` and `cost` are the same length.
fn estimate_binary(block: &[u8], histogram: &mut [u32], cost: &mut [f32]) {
    let len = block.len().min(cost.len());
    let block = &block[..len];
    let cost = &mut cost[..len];
    let Some(histogram) = histogram.get_mut(..256) else {
        return;
    };
    let window_half = BINARY_WINDOW_HALF;
    let mut in_window = window_half.min(len);
    let mut window_log = Log2Memo::EMPTY;
    let mut histo_log = Log2Memo::EMPTY;
    histogram.fill(0);
    let at = |index: usize| usize::from(block[index]);

    for &byte in &block[..in_window] {
        histogram[usize::from(byte)] += 1;
    }

    for index in 0..len {
        if index >= window_half {
            histogram[at(index - window_half)] -= 1;
            in_window -= 1;
        }
        if index + window_half < len {
            histogram[at(index + window_half)] += 1;
            in_window += 1;
        }
        let histo = histogram[at(index)].max(1) as usize;
        let mut lit_cost = window_log.get(in_window) - histo_log.get(histo);
        lit_cost += BINARY_NUDGE;
        if lit_cost < 1.0 {
            lit_cost *= 0.5;
            lit_cost += 0.5;
        }
        cost[index] = lit_cost as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prices `data` from its start, returning one cost per byte.
    fn costs(data: &[u8]) -> Vec<f32> {
        let mut cost = vec![0f32; data.len()];
        let mut arena = LiteralCostArena::default();
        estimate_bit_costs_for_literals(0, data.len(), usize::MAX, data, &mut arena, &mut cost);
        cost
    }

    #[test]
    fn the_position_classes_follow_the_reference() {
        // ASCII resets to the first class.
        assert_eq!(utf8_position(0, usize::from(b'a'), 2), 0);
        // A lead byte promises a continuation.
        assert_eq!(utf8_position(0, 0xC3, 2), 1);
        // A continuation after a two-byte lead completes the sequence.
        assert_eq!(utf8_position(0xC3, 0xA9, 2), 0);
        // A continuation after a three-byte lead promises another.
        assert_eq!(utf8_position(0xE2, 0x82, 2), 2);
        // The clamp caps the class the model will use.
        assert_eq!(utf8_position(0xE2, 0x82, 1), 1);
        assert_eq!(utf8_position(0, 0xC3, 0), 0);
    }

    #[test]
    fn barely_any_multi_byte_content_drops_to_one_class() {
        let ascii = vec![b'a'; 4096];
        assert_eq!(decide_multi_byte_stats_level(&ascii), 0);

        let mut text = Vec::new();
        while text.len() < 4096 {
            text.extend_from_slice("héllo wörld ".as_bytes());
        }
        assert_eq!(decide_multi_byte_stats_level(&text), 1);
    }

    #[test]
    fn a_predictable_byte_costs_less_than_a_surprising_one() {
        // A long run of one byte with a single intruder: the run is nearly
        // free and the intruder is not.
        let mut data = vec![b'a'; 3000];
        data[2500] = b'Z';
        let cost = costs(&data);
        assert!(
            cost[2500] > cost[2400] * 4.0,
            "{} vs {}",
            cost[2500],
            cost[2400]
        );
    }

    #[test]
    fn no_cost_falls_below_the_half_bit_floor() {
        // Below one bit the reference halves and re-centres, so nothing can be
        // cheaper than half a bit.
        let data = vec![b'q'; 8000];
        for (index, &bits) in costs(&data).iter().enumerate() {
            assert!(bits >= 0.5, "byte {index} cost {bits}");
        }
    }

    #[test]
    fn the_prologue_surcharge_grows_and_then_vanishes() {
        // Over a uniform block every literal has the same underlying cost, so
        // what is left is the surcharge alone: it starts at `PROLOGUE_BASE`,
        // rises across the prologue and stops dead at its end.
        let data = vec![b'x'; 8000];
        let cost = costs(&data);
        let base = cost[PROLOGUE_LENGTH];

        assert!((f64::from(cost[0]) - f64::from(base) - PROLOGUE_BASE).abs() < 1e-6);
        assert!(cost[0] < cost[100]);
        assert!(cost[100] < cost[1900]);
        assert!(cost[1999] > cost[PROLOGUE_LENGTH]);
        // Past the prologue the surcharge is gone entirely.
        assert_eq!(cost[PROLOGUE_LENGTH], cost[PROLOGUE_LENGTH + 500]);
        // The last prologue byte carries very nearly the full surcharge.
        let last = f64::from(cost[PROLOGUE_LENGTH - 1]) - f64::from(base);
        assert!((last - (PROLOGUE_BASE + PROLOGUE_MULTIPLIER * 1999.0)).abs() < 1e-6);
    }

    #[test]
    fn text_and_binary_take_different_paths() {
        // The same length of input, priced by the two models: the UTF-8 one
        // uses a narrower window, so its costs react to local structure.
        let mut text = Vec::new();
        while text.len() < 4000 {
            text.extend_from_slice("naïve café ".as_bytes());
        }
        text.truncate(4000);
        assert!(is_mostly_utf8(&text, 0, usize::MAX, text.len()));

        let binary: Vec<u8> = (0..4000u32).map(|i| (i * 37 % 256) as u8).collect();
        assert!(!is_mostly_utf8(&binary, 0, usize::MAX, binary.len()));

        // Both price every byte, and neither produces a NaN or a negative.
        for data in [text, binary] {
            for &bits in &costs(&data) {
                assert!(bits.is_finite() && bits > 0.0, "cost was {bits}");
            }
        }
    }

    #[test]
    fn an_empty_block_prices_nothing() {
        assert!(costs(b"").is_empty());
    }

    #[test]
    fn a_wrapping_block_prices_the_same_bytes() {
        let text = b"the quick brown fox jumps over the lazy dog, twice over";
        let mut ring = vec![0u8; 128];
        let mask = ring.len() - 1;
        let start = ring.len() - 20;
        for (offset, &byte) in text.iter().enumerate() {
            ring[(start + offset) & mask] = byte;
        }
        let mut wrapped = vec![0f32; text.len()];
        let mut arena = LiteralCostArena::default();
        estimate_bit_costs_for_literals(start, text.len(), mask, &ring, &mut arena, &mut wrapped);
        assert_eq!(wrapped, costs(text));
    }

    #[test]
    fn the_arena_can_be_reused_without_changing_the_result() {
        let first = b"the first block of literals, long enough to matter a bit";
        let second: Vec<u8> = (0..3000u32).map(|i| (i * 11 % 256) as u8).collect();
        let mut arena = LiteralCostArena::default();

        let mut once = vec![0f32; second.len()];
        estimate_bit_costs_for_literals(
            0,
            second.len(),
            usize::MAX,
            &second,
            &mut arena,
            &mut once,
        );

        let mut scratch = vec![0f32; first.len()];
        let mut reused = LiteralCostArena::default();
        estimate_bit_costs_for_literals(
            0,
            first.len(),
            usize::MAX,
            first,
            &mut reused,
            &mut scratch,
        );
        let mut twice = vec![0f32; second.len()];
        estimate_bit_costs_for_literals(
            0,
            second.len(),
            usize::MAX,
            &second,
            &mut reused,
            &mut twice,
        );
        assert_eq!(once, twice);
    }
}
