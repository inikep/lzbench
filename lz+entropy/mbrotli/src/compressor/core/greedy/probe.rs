//! A glance at a stream's first block that predicts how many positions the
//! match finder will store, which decides whether clearing a dense table pays.
//!
//! The tagged bucket shapes (qualities five and six) choose between a dense
//! table, cleared once at a cost of one or two mebibytes, and on-demand
//! layouts that pay a dependent load and block bookkeeping per store. Which is
//! cheaper depends on how many stores the stream makes, and that depends on
//! the data far more than on its length: text stores at almost every
//! position, incompressible input about a third as often once the search's
//! random-data heuristic starts skipping, and repetitive input almost never,
//! because long copies cover it.

use crate::shared::histogram::bits_entropy;

/// Bytes in one sampled window.
const WINDOW: usize = 8;

/// Windows sampled across the block.
const SAMPLES: usize = 256;

/// Share of sampled windows that repeat an earlier one, in 256ths, from which
/// a block counts as repetitive.
///
/// Text, markup, JSON and binary formats repeat at most an eighth of their
/// windows at this sampling; inputs made of long copies repeat four fifths
/// and more.
const REPETITIVE_REPEATS: usize = SAMPLES / 2;

/// Base-2 logarithm of the repeat filter's bits.
const SEEN_BITS: u32 = 12;

/// Words of the repeat filter.
const SEEN_WORDS: usize = (1 << SEEN_BITS) / 64;

/// Odd multiplier spreading a window over the repeat filter.
const SEEN_HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bits per sampled byte from which a block counts as incompressible.
///
/// Random and already-compressed bytes sample at 7.9 to 8 bits; the densest
/// ordinary data measured, map tiles, at 7.2.
const INCOMPRESSIBLE_BITS: f64 = 7.6;

/// What a block's sample predicts about the stream's store rate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum StoreOutlook {
    /// Most windows recur: long copies, few stores.
    Repetitive,
    /// Near-uniform bytes: the random-data heuristic skips most stores.
    Incompressible,
    /// A store at most positions.
    Ordinary,
}

/// Samples up to [`SAMPLES`] windows evenly across `block`.
///
/// Returns `None` for a block shorter than one window, which says nothing.
///
/// Kept out of line: it runs once per stream, and inlined into the matcher's
/// preparation it cost quality 6 eight per cent on 32-48 KiB inputs whose
/// layout it did not change.
#[inline(never)]
pub(crate) fn sample_store_outlook(block: &[u8]) -> Option<StoreOutlook> {
    let starts = block
        .len()
        .checked_sub(WINDOW - 1)
        .filter(|&starts| starts != 0)?;
    let samples = starts.min(SAMPLES);
    // An odd stride keeps the sample from locking onto an even period.
    let stride = (starts / samples).max(1) | 1;
    let mut windows = [0u64; SAMPLES];
    let mut start = 0;
    for window in windows.iter_mut().take(samples) {
        let mut bytes = [0u8; WINDOW];
        bytes.copy_from_slice(&block[start..start + WINDOW]);
        *window = u64::from_le_bytes(bytes);
        // `stride` is at most `starts / samples + 1`, so one subtraction
        // keeps every start in range without a division per sample.
        start += stride;
        if start >= starts {
            start -= starts;
        }
    }
    let windows = &windows[..samples];
    // Repeats are counted against a 4096-bit filter rather than by sorting:
    // an exact repeat always finds its bit set, and a stray collision adds
    // about one window in thirty to the count, far below the gap between
    // ordinary and repetitive blocks.
    let mut seen = [0u64; SEEN_WORDS];
    let repetitive = REPETITIVE_REPEATS * samples;
    let mut repeats = 0;
    for &window in windows {
        let hash = (window.wrapping_mul(SEEN_HASH_MUL) >> (64 - SEEN_BITS)) as usize;
        let (word, bit) = (hash / 64, 1u64 << (hash % 64));
        repeats += usize::from(seen[word] & bit != 0);
        seen[word] |= bit;
        if repeats * SAMPLES >= repetitive {
            return Some(StoreOutlook::Repetitive);
        }
    }
    // Counted only for a block that is not repetitive: a run of one byte
    // would chain every increment through the same counter.
    let mut histogram = [0u32; 256];
    for window in windows.iter() {
        for byte in window.to_le_bytes() {
            histogram[usize::from(byte)] += 1;
        }
    }
    Some(
        if bits_entropy(&histogram) >= INCOMPRESSIBLE_BITS * (samples * WINDOW) as f64 {
            StoreOutlook::Incompressible
        } else {
            StoreOutlook::Ordinary
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn pseudorandom(len: usize) -> Vec<u8> {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    fn text(len: usize) -> Vec<u8> {
        let words = [
            "alice ",
            "was ",
            "beginning ",
            "to ",
            "get ",
            "very ",
            "tired ",
            "of ",
            "sitting ",
            "by ",
            "her ",
            "sister ",
            "on ",
            "the ",
            "bank, ",
            "and ",
            "having ",
            "nothing ",
        ];
        let mut state = 7usize;
        let mut out = Vec::with_capacity(len + 16);
        while out.len() < len {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            out.extend_from_slice(words[(state >> 16) % words.len()].as_bytes());
        }
        out.truncate(len);
        out
    }

    #[test]
    fn long_runs_and_short_periods_are_repetitive() {
        assert_eq!(
            sample_store_outlook(&[b'a'; 32 << 10]),
            Some(StoreOutlook::Repetitive)
        );
        let period: Vec<u8> = b"The quick brown fox jumps over the lazy dog. "
            .iter()
            .copied()
            .cycle()
            .take(32 << 10)
            .collect();
        assert_eq!(
            sample_store_outlook(&period),
            Some(StoreOutlook::Repetitive)
        );
    }

    #[test]
    fn uniform_bytes_are_incompressible() {
        assert_eq!(
            sample_store_outlook(&pseudorandom(64 << 10)),
            Some(StoreOutlook::Incompressible)
        );
    }

    #[test]
    fn text_is_ordinary() {
        assert_eq!(
            sample_store_outlook(&text(32 << 10)),
            Some(StoreOutlook::Ordinary)
        );
    }

    #[test]
    fn blocks_shorter_than_the_sample_take_every_window() {
        assert_eq!(
            sample_store_outlook(&[b'a'; 100]),
            Some(StoreOutlook::Repetitive)
        );
        assert_eq!(
            sample_store_outlook(&text(100)),
            Some(StoreOutlook::Ordinary)
        );
    }

    #[test]
    fn blocks_shorter_than_a_window_say_nothing() {
        assert_eq!(sample_store_outlook(&[]), None);
        assert_eq!(sample_store_outlook(&[0; WINDOW - 1]), None);
        // One window is a sample, too small to repeat or to reach eight bits.
        assert_eq!(
            sample_store_outlook(&[0; WINDOW]),
            Some(StoreOutlook::Ordinary)
        );
    }
}
