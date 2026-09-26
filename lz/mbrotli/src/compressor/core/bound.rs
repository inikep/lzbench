//! Upper bound on the size of a compressed stream.

use crate::compressor::core::rfc9841::window::MAX_ENCODER_WINDOW_BITS;
use crate::compressor::{BrotliCompressError, BrotliResult, CompressParams, QualityLevel};
use crate::shared::constants::{OUTPUT_RESERVE_CONST, OUTPUT_SLACK};

/// Returns an upper bound on the compressed size of `input_size` bytes.
///
/// Both encoders reserve `2 * fragment + 503` bytes per meta-block, exactly
/// like the reference encoder, plus the bit writer's whole-word headroom and
/// two bytes for the stream header. Counting the headroom per fragment is what
/// lets the fast path write straight into a buffer sized by this bound instead
/// of copying through its own scratch space.
///
/// # Errors
///
/// Returns [`BrotliCompressError::BoundOverflow`] when that arithmetic does not
/// fit in a `usize`, rather than wrapping or saturating into a bound that no
/// longer bounds anything.
pub(crate) const fn bound(params: &CompressParams, input_size: usize) -> BrotliResult<usize> {
    let fragment = 1usize << fragment_bits(params);
    let fragments = if input_size == 0 {
        1
    } else {
        (input_size - 1) / fragment + 1
    };

    let Some(overhead) = fragments.checked_mul(OUTPUT_RESERVE_CONST + OUTPUT_SLACK) else {
        return Err(BrotliCompressError::BoundOverflow);
    };
    let Some(payload) = input_size.checked_mul(2) else {
        return Err(BrotliCompressError::BoundOverflow);
    };
    let Some(total) = payload.checked_add(overhead) else {
        return Err(BrotliCompressError::BoundOverflow);
    };
    match total.checked_add(2) {
        Some(total) => Ok(total),
        None => Err(BrotliCompressError::BoundOverflow),
    }
}

/// Returns what a one-shot call appending to a vector reserves up front.
///
/// The fast qualities write their bits straight into the vector's spare
/// capacity, so they reserve the full [`bound`]. Every other quality builds
/// each meta-block in its own scratch buffer and appends the finished bytes,
/// so the vector only has to hold the stream itself: those reserve the
/// reference's `BrotliEncoderMaxCompressedSize` — the input plus four bytes per
/// 16 KiB and six bytes of framing, which is what an uncompressed stream takes
/// — and let the vector grow in the rare case a stream is longer. Reserving
/// twice the input instead pushes a cold one-shot call's peak footprint past
/// glibc's adaptive trim threshold, so the allocator hands the pages back
/// after every call and the next call faults them in again.
///
/// # Errors
///
/// Returns [`BrotliCompressError::BoundOverflow`] exactly when [`bound`] does.
///
/// Not `const`: in some feature sets the error has a destructor, which a
/// constant function cannot run on its early return.
pub(crate) fn append_reserve(params: &CompressParams, input_size: usize) -> BrotliResult<usize> {
    let full = bound(params, input_size)?;
    match params.quality {
        QualityLevel::Q0 | QualityLevel::Q1 => Ok(full),
        _ => {
            let stream = reference_max_compressed_size(input_size);
            Ok(if stream < full { stream } else { full })
        }
    }
}

/// The reference's `BrotliEncoderMaxCompressedSize`, saturating.
///
/// Window bits and an empty metadata block, four header bytes per 16 KiB
/// uncompressed meta-block, and the final empty meta-block.
const fn reference_max_compressed_size(input_size: usize) -> usize {
    if input_size == 0 {
        return 2;
    }
    input_size
        .saturating_add(4 * (input_size >> 14))
        .saturating_add(6)
}

/// Returns the base-2 logarithm of the input the encoder consumes per step.
///
/// Both encoder families emit at most one meta-block per step, so this is what
/// bounds how often the per-meta-block overhead is paid. The fast qualities
/// cut the input at the window size; the greedy ones use the block size, which
/// is fourteen bits below quality four and the requested or default sixteen
/// above it.
///
/// A window is counted at the history the encoder actually keeps rather than
/// at what its header declares, because a declared sixty-two-bit window would
/// otherwise claim the whole input as one fragment and stop reserving
/// per-meta-block overhead the encoder still pays.
const fn fragment_bits(params: &CompressParams) -> usize {
    match params.quality {
        QualityLevel::Q3 => 14,
        QualityLevel::Q4 | QualityLevel::Q5 => match params.lgblock {
            Some(lgblock) => lgblock.0,
            None => 16,
        },
        // Quality 0 and 1 cut at the window size; an unimplemented quality
        // never reaches an encoder, but still has to return some bound.
        _ => {
            let bits = params.lgwin.bits() as usize;
            if bits > MAX_ENCODER_WINDOW_BITS {
                MAX_ENCODER_WINDOW_BITS
            } else {
                bits
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{QualityLevel, WindowBits, WindowOutOfRange};

    fn params(lgwin: u8) -> Result<CompressParams, WindowOutOfRange> {
        Ok(CompressParams::new(
            QualityLevel::Q0,
            WindowBits::standard(lgwin)?,
        ))
    }

    #[test]
    fn empty_input_still_reserves_one_fragment() -> Result<(), WindowOutOfRange> {
        assert!(matches!(bound(&params(22)?, 0), Ok(513)));
        Ok(())
    }

    #[test]
    fn bound_grows_with_the_number_of_fragments() -> Result<(), WindowOutOfRange> {
        let small_window = bound(&params(10)?, 1 << 20).ok();
        let large_window = bound(&params(22)?, 1 << 20).ok();
        assert!(small_window > large_window);
        Ok(())
    }

    #[test]
    fn bound_covers_at_least_the_input() -> Result<(), WindowOutOfRange> {
        let params = params(22)?;
        for size in [0usize, 1, 1024, 1 << 20] {
            assert!(bound(&params, size).is_ok_and(|value| value >= size));
        }
        Ok(())
    }

    #[test]
    fn fast_qualities_reserve_the_full_bound_for_appending() -> Result<(), WindowOutOfRange> {
        let params = params(22)?;
        for size in [0usize, 1, 1024, 1 << 20] {
            assert_eq!(
                append_reserve(&params, size).ok(),
                bound(&params, size).ok()
            );
        }
        Ok(())
    }

    #[test]
    fn other_qualities_reserve_the_reference_stream_bound() -> Result<(), WindowOutOfRange> {
        for quality in [
            QualityLevel::Q2,
            QualityLevel::Q4,
            QualityLevel::Q9,
            QualityLevel::Q11,
        ] {
            let params = CompressParams::new(quality, WindowBits::standard(22)?);
            assert_eq!(append_reserve(&params, 0).ok(), Some(2));
            assert_eq!(append_reserve(&params, 44).ok(), Some(50));
            assert_eq!(
                append_reserve(&params, 1 << 20).ok(),
                Some((1 << 20) + 256 + 6)
            );
            for size in [1usize, 1024, 1 << 20] {
                let reserve = append_reserve(&params, size).unwrap_or_default();
                assert!(reserve >= size && bound(&params, size).is_ok_and(|full| reserve <= full));
            }
        }
        Ok(())
    }

    #[test]
    fn the_append_reserve_overflows_exactly_when_the_bound_does() -> Result<(), WindowOutOfRange> {
        let params = CompressParams::new(QualityLevel::Q5, WindowBits::standard(22)?);
        assert!(matches!(
            append_reserve(&params, usize::MAX),
            Err(BrotliCompressError::BoundOverflow)
        ));
        assert_eq!(reference_max_compressed_size(usize::MAX), usize::MAX);
        Ok(())
    }

    #[test]
    fn bound_reports_an_overflow_instead_of_wrapping() -> Result<(), WindowOutOfRange> {
        assert!(matches!(
            bound(&params(22)?, usize::MAX),
            Err(BrotliCompressError::BoundOverflow)
        ));
        assert!(matches!(
            bound(&params(10)?, usize::MAX / 2),
            Err(BrotliCompressError::BoundOverflow)
        ));
        Ok(())
    }
}
