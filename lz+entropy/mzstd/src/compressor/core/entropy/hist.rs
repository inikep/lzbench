//! Symbol histograms (`hist.c`).

use crate::compressor::core::error::CoreError as EntropyError;

/// Counts the symbols of `src` into `count[..=*max_symbol_value]`.
///
/// On return `*max_symbol_value` is the largest symbol present (`0` for an
/// empty input) and the result is the largest count. With `check`, a symbol
/// above the incoming bound is an error, as in `HIST_count_wksp`; without it
/// the caller guarantees none exists, as in `HIST_countFast_wksp`, and one
/// that does is reported the same way rather than counted out of bounds.
///
/// # Errors
///
/// [`EntropyError::MaxSymbolValueTooSmall`] when `src` holds a symbol above
/// the incoming bound.
pub(crate) fn count(
    count: &mut [u32],
    max_symbol_value: &mut u32,
    src: &[u8],
    check: bool,
) -> Result<u32, EntropyError> {
    let bound = *max_symbol_value as usize;
    let count = &mut count[..=bound];
    if src.len() < PARALLEL_THRESHOLD {
        let result = count_simple(count, max_symbol_value, src);
        debug_assert!(check || result.is_ok(), "trusted input above its bound");
        return result;
    }
    let full = count_parallel(src);
    let mut max = 255usize;
    while full[max] == 0 {
        max -= 1;
    }
    if max > bound {
        debug_assert!(check, "trusted input above its bound");
        return Err(EntropyError::MaxSymbolValueTooSmall);
    }
    count.copy_from_slice(&full[..=bound]);
    *max_symbol_value = max as u32;
    Ok(full.iter().copied().max().unwrap_or(0))
}

/// Inputs shorter than this are counted with one table
/// (`HIST_countFast_wksp`'s heuristic threshold).
const PARALLEL_THRESHOLD: usize = 1500;

/// `HIST_count_simple`: counts into `count` directly, so only the
/// alphabet's own entries are cleared and scanned.
///
/// A symbol above the bound is reported as an error; the reference trusts
/// its callers here.
fn count_simple(
    count: &mut [u32],
    max_symbol_value: &mut u32,
    src: &[u8],
) -> Result<u32, EntropyError> {
    count.fill(0);
    if let Ok(full) = <&mut [u32; 256]>::try_from(&mut *count) {
        // Every byte has a counter: no per-byte bound test.
        for &byte in src {
            full[usize::from(byte)] += 1;
        }
    } else {
        for &byte in src {
            let Some(cell) = count.get_mut(usize::from(byte)) else {
                return Err(EntropyError::MaxSymbolValueTooSmall);
            };
            *cell += 1;
        }
    }
    let Some(max) = count.iter().rposition(|&c| c != 0) else {
        *max_symbol_value = 0;
        return Ok(0);
    };
    *max_symbol_value = max as u32;
    Ok(count[..=max].iter().copied().max().unwrap_or(0))
}

/// `HIST_count_parallel_wksp`: four interleaved tables, so consecutive
/// bytes increment different counters and repeated symbols do not
/// serialize on one memory location.
fn count_parallel(src: &[u8]) -> [u32; 256] {
    let mut tables = [[0u32; 256]; 4];
    let (words, tail) = src.as_chunks::<4>();
    for word in words {
        tables[0][usize::from(word[0])] += 1;
        tables[1][usize::from(word[1])] += 1;
        tables[2][usize::from(word[2])] += 1;
        tables[3][usize::from(word[3])] += 1;
    }
    for &byte in tail {
        tables[0][usize::from(byte)] += 1;
    }
    let [mut total, second, third, fourth] = tables;
    for (symbol, cell) in total.iter_mut().enumerate() {
        *cell += second[symbol] + third[symbol] + fourth[symbol];
    }
    total
}

/// Adds the symbols of `src` to `count` (`HIST_add`).
pub(crate) fn add(count: &mut [u32; 256], src: &[u8]) {
    for &byte in src {
        count[usize::from(byte)] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_report_the_largest_symbol_and_count() {
        let mut counts = [0u32; 256];
        let mut max = 255;
        let largest = count(&mut counts, &mut max, b"aabbbz", true).expect("count");
        assert_eq!(largest, 3);
        assert_eq!(max, u32::from(b'z'));
        assert_eq!(counts[usize::from(b'b')], 3);
    }

    #[test]
    fn empty_input_resets_the_bound() {
        let mut counts = [7u32; 8];
        let mut max = 7;
        assert_eq!(count(&mut counts, &mut max, &[], true), Ok(0));
        assert_eq!(max, 0);
        assert_eq!(counts, [0; 8]);
    }

    #[test]
    fn a_symbol_above_a_checked_bound_is_an_error() {
        let mut counts = [0u32; 4];
        let mut max = 3;
        assert_eq!(
            count(&mut counts, &mut max, &[1, 9], true),
            Err(EntropyError::MaxSymbolValueTooSmall)
        );
    }

    /// One straightforward count per symbol, the oracle for both paths.
    fn reference(src: &[u8]) -> ([u32; 256], u32, u32) {
        let mut full = [0u32; 256];
        for &b in src {
            full[usize::from(b)] += 1;
        }
        let max = full.iter().rposition(|&c| c != 0).unwrap_or(0) as u32;
        (full, max, full.iter().copied().max().unwrap_or(0))
    }

    #[test]
    fn simple_and_interleaved_counts_match_a_plain_count() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        // Sizes on both sides of the interleaved path's threshold, with
        // tails that are not a whole number of words.
        for len in [1usize, 7, 1499, 1500, 1501, 1503, 4096, 10_007] {
            for bound in [255u32, 52, 3] {
                let src: alloc::vec::Vec<u8> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed % u64::from(bound + 1)) as u8
                    })
                    .collect();
                let (full, max, largest) = reference(&src);
                let mut counts = alloc::vec![9u32; bound as usize + 1];
                let mut got_max = bound;
                assert_eq!(count(&mut counts, &mut got_max, &src, true), Ok(largest));
                assert_eq!(got_max, max, "{len} {bound}");
                assert_eq!(counts[..], full[..=bound as usize], "{len} {bound}");
            }
        }
    }

    #[test]
    fn a_symbol_above_the_bound_is_an_error_on_both_paths() {
        for len in [10usize, 2000] {
            let mut src = alloc::vec![1u8; len];
            src[len / 2] = 9;
            let mut counts = [0u32; 4];
            let mut max = 3;
            assert_eq!(
                count(&mut counts, &mut max, &src, true),
                Err(EntropyError::MaxSymbolValueTooSmall)
            );
        }
    }

    #[test]
    fn add_accumulates() {
        let mut counts = [0u32; 256];
        add(&mut counts, b"aa");
        add(&mut counts, b"a");
        assert_eq!(counts[usize::from(b'a')], 3);
    }
}
