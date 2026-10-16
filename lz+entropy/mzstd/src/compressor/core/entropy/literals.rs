//! The literals section of a compressed block (`zstd_compress_literals.c`).

use super::HufTables;
use super::huf::{self, Flags, Options, Streams};
use crate::Backend;
use crate::compressor::core::error::CoreError;
use crate::compressor::core::params::Strategy;
use crate::shared::format::{EncodingType, LIT_HUF_LOG};

/// `ZSTD_minGain`: what a compressed section must save to be kept.
pub(crate) const fn min_gain(src_size: usize, strategy: Strategy) -> usize {
    let min_log = if strategy as u32 >= Strategy::BtUltra as u32 {
        strategy as u32 - 1
    } else {
        6
    };
    (src_size >> min_log) + 2
}

/// Header size of an uncompressed or RLE literals section.
const fn raw_header_size(src_size: usize) -> usize {
    1 + (src_size > 31) as usize + (src_size > 4095) as usize
}

/// Writes the header of a raw or RLE literals section.
fn write_raw_header(dst: &mut [u8], kind: EncodingType, src_size: usize, header_size: usize) {
    let kind = kind as u32;
    let size = src_size as u32;
    match header_size {
        1 => dst[0] = (kind + (size << 3)) as u8,
        2 => dst[..2].copy_from_slice(&((kind + (1 << 2) + (size << 4)) as u16).to_le_bytes()),
        _ => dst[..3].copy_from_slice(&(kind + (3 << 2) + (size << 4)).to_le_bytes()[..3]),
    }
}

/// Stores the literals uncompressed (`ZSTD_noCompressLiterals`).
///
/// # Errors
///
/// [`CoreError::DstSizeTooSmall`] when the section does not fit.
pub(crate) fn no_compress(dst: &mut [u8], src: &[u8]) -> Result<usize, CoreError> {
    let header_size = raw_header_size(src.len());
    if src.len() + header_size > dst.len() {
        return Err(CoreError::DstSizeTooSmall);
    }
    write_raw_header(dst, EncodingType::Basic, src.len(), header_size);
    dst[header_size..header_size + src.len()].copy_from_slice(src);
    Ok(src.len() + header_size)
}

/// Stores one repeated literal (`ZSTD_compressRleLiteralsBlock`).
pub(crate) fn compress_rle(dst: &mut [u8], src: &[u8]) -> usize {
    let header_size = raw_header_size(src.len());
    write_raw_header(dst, EncodingType::Rle, src.len(), header_size);
    dst[header_size] = src[0];
    header_size + 1
}

/// Fewest literals worth a Huffman attempt (`ZSTD_minLiteralsToCompress`).
fn min_literals_to_compress(strategy: Strategy, repeat: huf::Repeat) -> usize {
    let shift = (9 - strategy as u32).min(3);
    if repeat == huf::Repeat::Valid {
        6
    } else {
        8 << shift
    }
}

/// Settings of one literals compression.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct LiteralsSettings {
    pub(crate) strategy: Strategy,
    /// Store literals raw regardless of their statistics.
    pub(crate) disable_compression: bool,
    /// The block looks incompressible; sample before counting.
    pub(crate) suspect_uncompressible: bool,
    /// The SIMD level the Huffman stream encoder runs with.
    pub(crate) backend: Backend,
}

/// Attempts a compressed or RLE literals section (`ZSTD_compressLiterals`).
/// Returns zero when the caller should store the literals uncompressed.
///
/// `next` starts as a copy of `prev` and ends as the Huffman state the next
/// block inherits.
///
/// # Errors
///
/// [`CoreError::DstSizeTooSmall`] when the compressed header does not fit.
pub(crate) fn try_compress(
    dst: &mut [u8],
    src: &[u8],
    prev: &HufTables,
    next: &mut HufTables,
    settings: LiteralsSettings,
) -> Result<usize, CoreError> {
    const KB: usize = 1 << 10;
    let src_size = src.len();
    let lh_size = 3 + usize::from(src_size >= KB) + usize::from(src_size >= 16 * KB);
    let mut single_stream = src_size < 256;
    let mut kind = EncodingType::Compressed;
    let strategy = settings.strategy;

    next.clone_from(prev);

    if settings.disable_compression {
        return Ok(0);
    }
    if src_size < min_literals_to_compress(strategy, prev.repeat) {
        return Ok(0);
    }
    if dst.len() < lh_size + 1 {
        return Err(CoreError::DstSizeTooSmall);
    }

    let mut repeat = prev.repeat;
    let flags = Flags {
        prefer_repeat: strategy < Strategy::Lazy && src_size <= 1024,
        optimal_depth: strategy >= Strategy::BtUltra,
        suspect_uncompressible: settings.suspect_uncompressible,
    };
    if repeat == huf::Repeat::Valid && lh_size == 3 {
        single_stream = true;
    }
    let options = Options {
        max_symbol_value: huf::SYMBOL_VALUE_MAX,
        huff_log: LIT_HUF_LOG,
        streams: if single_stream {
            Streams::Single
        } else {
            Streams::Four
        },
        flags,
        backend: settings.backend,
    };
    let result = huf::compress(
        &mut dst[lh_size..],
        src,
        options,
        &mut next.table,
        &mut repeat,
    );
    if repeat != huf::Repeat::None {
        kind = EncodingType::Repeat;
    }

    let c_lit_size = match result {
        Ok(size) if size != 0 && size < src_size - min_gain(src_size, strategy) => size,
        _ => {
            next.clone_from(prev);
            return Ok(0);
        }
    };
    if c_lit_size == 1 && (src_size >= 8 || src.iter().all(|&b| b == src[0])) {
        next.clone_from(prev);
        return Ok(compress_rle(dst, src));
    }

    if kind == EncodingType::Compressed {
        next.repeat = huf::Repeat::Check;
    }

    let kind = kind as u32;
    let size = src_size as u32;
    let c_size = c_lit_size as u32;
    match lh_size {
        3 => {
            let lhc = kind + (u32::from(!single_stream) << 2) + (size << 4) + (c_size << 14);
            dst[..3].copy_from_slice(&lhc.to_le_bytes()[..3]);
        }
        4 => {
            let lhc = kind + (2 << 2) + (size << 4) + (c_size << 18);
            dst[..4].copy_from_slice(&lhc.to_le_bytes());
        }
        _ => {
            let lhc = kind + (3 << 2) + (size << 4) + (c_size << 22);
            dst[..4].copy_from_slice(&lhc.to_le_bytes());
            dst[4] = (c_lit_size >> 10) as u8;
        }
    }
    Ok(lh_size + c_lit_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(strategy: Strategy) -> LiteralsSettings {
        LiteralsSettings {
            strategy,
            disable_compression: false,
            suspect_uncompressible: false,
            backend: Backend::SCALAR,
        }
    }

    #[test]
    fn raw_headers_grow_with_the_literal_count() {
        let mut dst = [0u8; 5000];
        assert_eq!(no_compress(&mut dst, &[1; 31]), Ok(32));
        assert_eq!(dst[0], 31 << 3);
        assert_eq!(no_compress(&mut dst, &[1; 32]), Ok(34));
        assert_eq!(no_compress(&mut dst, &[1; 4096]), Ok(4099));
        assert_eq!(
            no_compress(&mut dst[..3], &[1; 3]),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    #[test]
    fn repeated_literals_become_an_rle_section() {
        let prev = HufTables::default();
        let mut next = HufTables::default();
        let mut dst = [0u8; 256];
        let size = try_compress(
            &mut dst,
            &[7; 100],
            &prev,
            &mut next,
            settings(Strategy::Fast),
        )
        .expect("literals");
        assert_eq!(size, 3);
        assert_eq!(dst[2], 7);
    }

    #[test]
    fn a_valid_previous_table_admits_short_rle_sections() {
        // Only a dictionary marks a table valid in the reference; below
        // eight literals the section must then be identical bytes to be RLE.
        let prev = HufTables {
            repeat: huf::Repeat::Valid,
            ..HufTables::default()
        };
        let mut next = HufTables::default();
        let mut dst = [0u8; 64];
        let size = try_compress(
            &mut dst,
            &[9; 6],
            &prev,
            &mut next,
            settings(Strategy::Lazy2),
        );
        assert_eq!(size, Ok(2));
        assert_eq!(dst[..2], [1 | (6 << 3), 9]);
    }

    #[test]
    fn short_or_disabled_literals_stay_raw() {
        let prev = HufTables::default();
        let mut next = HufTables::default();
        let mut dst = [0u8; 256];
        let size = try_compress(
            &mut dst,
            b"short",
            &prev,
            &mut next,
            settings(Strategy::Fast),
        );
        assert_eq!(size, Ok(0));
        let disabled = LiteralsSettings {
            disable_compression: true,
            ..settings(Strategy::BtUltra2)
        };
        let text = b"abababababababababababababababababababababababababababababababababababab";
        assert_eq!(
            try_compress(&mut dst, text, &prev, &mut next, disabled),
            Ok(0)
        );
        assert!(
            try_compress(
                &mut dst[..2],
                text,
                &prev,
                &mut next,
                settings(Strategy::BtUltra2)
            )
            .is_err()
        );
    }

    #[test]
    fn min_gain_tightens_for_ultra_strategies() {
        assert_eq!(min_gain(1 << 12, Strategy::Fast), (1 << 6) + 2);
        assert_eq!(min_gain(1 << 12, Strategy::BtUltra2), (1 << 4) + 2);
    }
}
