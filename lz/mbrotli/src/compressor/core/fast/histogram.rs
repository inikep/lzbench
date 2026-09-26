//! Byte histograms for the literal prefix codes.
//!
//! Counting bytes is a scatter update, so it does not vectorise directly. What
//! it does suffer from is the store-to-load dependency a single counter array
//! creates whenever the same byte repeats. Splitting the input into fixed-size
//! chunks lets several independent sub-histograms advance in parallel, and
//! `as_chunks` keeps the loop free of bounds checks with a short scalar tail.

use super::constants::NUM_LITERAL_SYMBOLS;

/// Number of independent sub-histograms the chunked path keeps.
const LANES: usize = 4;

/// Shortest input for which splitting into lanes pays for the merge.
///
/// Below this the merge of `LANES * 256` counters costs more than the
/// dependency chain it removes.
const LANE_THRESHOLD: usize = 4 * 1024;

/// Counts the bytes of `input` into `histogram`, which must start cleared.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn accumulate(input: &[u8], histogram: &mut [u32; NUM_LITERAL_SYMBOLS]) {
    if input.len() < LANE_THRESHOLD {
        for &byte in input {
            histogram[usize::from(byte)] += 1;
        }
        return;
    }

    let mut lanes = [[0u32; NUM_LITERAL_SYMBOLS]; LANES];
    let (chunks, tail) = input.as_chunks::<LANES>();
    for chunk in chunks {
        for (lane, &byte) in lanes.iter_mut().zip(chunk) {
            lane[usize::from(byte)] += 1;
        }
    }
    for &byte in tail {
        histogram[usize::from(byte)] += 1;
    }
    for lane in &lanes {
        for (slot, &count) in histogram.iter_mut().zip(lane) {
            *slot += count;
        }
    }
}

/// Counts every `stride`-th byte of `input` into `histogram`.
///
/// Sampling loops stay scalar: they touch too little data for a lane split to
/// pay for itself, and the reference uses the same strides.
pub(crate) fn accumulate_sampled(
    input: &[u8],
    stride: usize,
    histogram: &mut [u32; NUM_LITERAL_SYMBOLS],
) {
    for &byte in input.iter().step_by(stride) {
        histogram[usize::from(byte)] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn reference(input: &[u8]) -> [u32; NUM_LITERAL_SYMBOLS] {
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        for &byte in input {
            histogram[usize::from(byte)] += 1;
        }
        histogram
    }

    #[test]
    fn chunked_and_scalar_counting_agree_at_every_length() {
        let mut state = 0x1234_5678u32;
        let data: Vec<u8> = (0..LANE_THRESHOLD + 64)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect();

        for length in [
            0usize,
            1,
            3,
            4,
            5,
            LANE_THRESHOLD - 1,
            LANE_THRESHOLD,
            LANE_THRESHOLD + 3,
        ] {
            let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
            accumulate(&data[..length], &mut histogram);
            assert_eq!(histogram, reference(&data[..length]), "length {length}");
            assert_eq!(histogram.iter().sum::<u32>(), length as u32);
        }
    }

    #[test]
    fn chunked_counting_handles_a_single_repeated_byte() {
        let data = vec![7u8; LANE_THRESHOLD * 2 + 3];
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        accumulate(&data, &mut histogram);
        assert_eq!(histogram[7], data.len() as u32);
        assert_eq!(histogram.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn sampled_counting_visits_every_stride_th_byte() {
        let data: Vec<u8> = (0..100u8).collect();
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        accumulate_sampled(&data, 29, &mut histogram);
        assert_eq!(histogram[0], 1);
        assert_eq!(histogram[29], 1);
        assert_eq!(histogram[58], 1);
        assert_eq!(histogram[87], 1);
        assert_eq!(histogram.iter().sum::<u32>(), 4);
    }
}
