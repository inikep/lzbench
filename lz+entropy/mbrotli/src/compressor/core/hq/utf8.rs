//! Deciding whether a stretch of input is text.
//!
//! Ports `c/enc/utf8_util.c` from the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`).
//!
//! Two decisions turn on this. The literal-cost estimator models UTF-8 text
//! with three histograms keyed by position within a sequence, and one
//! histogram otherwise; and at quality ten and above the meta-block picks the
//! signed context model over the UTF-8 one when the answer is no.

/// Fraction of the input that has to parse as UTF-8 (`kMinUTF8Ratio`).
const MIN_UTF8_RATIO: f64 = 0.75;

/// Code point marking a byte that is not part of a valid sequence.
const NOT_UTF8: u32 = 0x0011_0000;

/// Decodes one sequence at the start of `input`, returning its width and value.
///
/// Mirrors `BrotliParseAsUTF8`. A byte that starts nothing valid is reported as
/// one byte wide with a symbol above the Unicode range, which is what makes the
/// caller's ratio meaningful. Overlong encodings and surrogate-range values
/// fall into the same bucket, because each length check demands a value the
/// shorter forms could not have produced.
#[inline]
fn parse_as_utf8(input: &[u8]) -> (usize, u32) {
    let size = input.len();
    let byte = |index: usize| input.get(index).copied().unwrap_or(0);

    // ASCII, excluding NUL, which the reference deliberately treats as binary.
    if byte(0) & 0x80 == 0 {
        let symbol = u32::from(byte(0));
        if symbol > 0 {
            return (1, symbol);
        }
    }
    if size > 1 && byte(0) & 0xE0 == 0xC0 && byte(1) & 0xC0 == 0x80 {
        let symbol = (u32::from(byte(0) & 0x1F) << 6) | u32::from(byte(1) & 0x3F);
        if symbol > 0x7F {
            return (2, symbol);
        }
    }
    if size > 2 && byte(0) & 0xF0 == 0xE0 && byte(1) & 0xC0 == 0x80 && byte(2) & 0xC0 == 0x80 {
        let symbol = (u32::from(byte(0) & 0x0F) << 12)
            | (u32::from(byte(1) & 0x3F) << 6)
            | u32::from(byte(2) & 0x3F);
        if symbol > 0x7FF {
            return (3, symbol);
        }
    }
    if size > 3
        && byte(0) & 0xF8 == 0xF0
        && byte(1) & 0xC0 == 0x80
        && byte(2) & 0xC0 == 0x80
        && byte(3) & 0xC0 == 0x80
    {
        let symbol = (u32::from(byte(0) & 0x07) << 18)
            | (u32::from(byte(1) & 0x3F) << 12)
            | (u32::from(byte(2) & 0x3F) << 6)
            | u32::from(byte(3) & 0x3F);
        if symbol > 0xFFFF && symbol <= 0x0010_FFFF {
            return (4, symbol);
        }
    }
    (1, NOT_UTF8 | u32::from(byte(0)))
}

/// Returns whether more than three quarters of `length` bytes parse as UTF-8.
///
/// Mirrors `BrotliIsMostlyUTF8`. `pos` and `mask` address the ring buffer, so
/// the run may wrap; a run that does not is scanned in place, and one that
/// does hands the sequence decoder a contiguous copy of what it needs.
///
/// The scan stops as soon as the answer is settled: once enough bytes have
/// parsed, or once too few remain for the rest to make up the difference.
pub(crate) fn is_mostly_utf8(data: &[u8], pos: usize, mask: usize, length: usize) -> bool {
    let needed = utf8_bytes_needed(length);
    let end = pos + length;
    match data.get(pos..end) {
        Some(run) if end.saturating_sub(1) <= mask || length == 0 => reaches_in(run, needed),
        _ => reaches_wrapping(data, pos, mask, length, needed),
    }
}

/// The fewest valid-sequence bytes that pass the reference's
/// `size_utf8 > kMinUTF8Ratio * length`.
///
/// `0.75 * length` is exact in an `f64` for every addressable length, so the
/// count passes exactly when it reaches the next integer above the product.
fn utf8_bytes_needed(length: usize) -> usize {
    (MIN_UTF8_RATIO * length as f64) as usize + 1
}

/// Returns whether at least `needed` bytes of `run` belong to valid sequences.
///
/// Eight bytes that are all ASCII other than NUL each parse as a one-byte
/// sequence, so such a word is counted whole without decoding it.
fn reaches_in(run: &[u8], needed: usize) -> bool {
    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
    const LOW_BITS: u64 = 0x0101_0101_0101_0101;
    let mut size_utf8 = 0usize;
    let mut index = 0usize;
    while index < run.len() {
        if let Some(word) = run.get(index..index + 8).and_then(<[u8]>::first_chunk::<8>) {
            let word = u64::from_le_bytes(*word);
            let has_zero = word.wrapping_sub(LOW_BITS) & !word & HIGH_BITS;
            if word & HIGH_BITS == 0 && has_zero == 0 {
                size_utf8 += 8;
                index += 8;
                if size_utf8 >= needed {
                    return true;
                }
                continue;
            }
        }
        let (bytes_read, symbol) = parse_as_utf8(&run[index..]);
        index += bytes_read;
        if symbol < NOT_UTF8 {
            size_utf8 += bytes_read;
            if size_utf8 >= needed {
                return true;
            }
        } else if size_utf8 + (run.len() - index) < needed {
            // Even if every remaining byte parsed, the run would fall short.
            return false;
        }
    }
    size_utf8 >= needed
}

/// [`reaches_in`] for a run that wraps the ring buffer.
fn reaches_wrapping(data: &[u8], pos: usize, mask: usize, length: usize, needed: usize) -> bool {
    let mut size_utf8 = 0usize;
    let mut index = 0usize;
    // Four bytes is the widest sequence, so a small window is enough to hand
    // the decoder a contiguous view of a wrapping run.
    let mut window = [0u8; 4];
    while index < length {
        let remaining = length - index;
        let width = remaining.min(4);
        for (offset, slot) in window.iter_mut().enumerate().take(width) {
            *slot = data
                .get((pos + index + offset) & mask)
                .copied()
                .unwrap_or(0);
        }
        let (bytes_read, symbol) = parse_as_utf8(&window[..width]);
        index += bytes_read;
        if symbol < NOT_UTF8 {
            size_utf8 += bytes_read;
            if size_utf8 >= needed {
                return true;
            }
        } else if size_utf8 + (length - index) < needed {
            return false;
        }
    }
    size_utf8 >= needed
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Runs the ratio test over a contiguous slice.
    fn mostly_utf8(data: &[u8]) -> bool {
        is_mostly_utf8(data, 0, usize::MAX, data.len())
    }

    #[test]
    fn ascii_parses_one_byte_at_a_time() {
        assert_eq!(parse_as_utf8(b"a"), (1, u32::from(b'a')));
        assert_eq!(parse_as_utf8(b"~xyz"), (1, u32::from(b'~')));
    }

    #[test]
    fn a_nul_byte_is_not_treated_as_text() {
        // The reference falls through the ASCII case for NUL, so it lands in
        // the "not UTF-8" bucket like a stray continuation byte would.
        let (width, symbol) = parse_as_utf8(&[0x00]);
        assert_eq!(width, 1);
        assert!(symbol >= NOT_UTF8);
    }

    #[test]
    fn every_sequence_width_is_decoded() {
        assert_eq!(parse_as_utf8("é".as_bytes()), (2, 0xE9));
        assert_eq!(parse_as_utf8("€".as_bytes()), (3, 0x20AC));
        assert_eq!(parse_as_utf8("𝄞".as_bytes()), (4, 0x0001_D11E));
    }

    #[test]
    fn a_truncated_sequence_is_not_text() {
        // A lead byte with its continuation cut off by the end of the input.
        let (width, symbol) = parse_as_utf8(&[0xE2, 0x82]);
        assert_eq!(width, 1);
        assert!(symbol >= NOT_UTF8);
    }

    #[test]
    fn an_overlong_encoding_is_rejected() {
        // Two-byte encoding of `/`, which must be one byte.
        let (width, symbol) = parse_as_utf8(&[0xC0, 0xAF]);
        assert_eq!(width, 1);
        assert!(symbol >= NOT_UTF8);

        // Three-byte encoding of a value that fits in two.
        let (width, symbol) = parse_as_utf8(&[0xE0, 0x80, 0xAF]);
        assert_eq!(width, 1);
        assert!(symbol >= NOT_UTF8);
    }

    #[test]
    fn a_value_past_the_unicode_range_is_rejected() {
        // U+110000, one past the last code point.
        let (width, symbol) = parse_as_utf8(&[0xF4, 0x90, 0x80, 0x80]);
        assert_eq!(width, 1);
        assert!(symbol >= NOT_UTF8);
    }

    #[test]
    fn text_is_mostly_utf8_and_binary_is_not() {
        assert!(mostly_utf8(b"The quick brown fox jumps over the lazy dog."));
        assert!(mostly_utf8("Å tenke på ζωή og 日本語".as_bytes()));

        let binary: Vec<u8> = (0..256u32).map(|i| (i * 7 % 256) as u8).collect();
        assert!(!mostly_utf8(&binary));
        assert!(!mostly_utf8(&[0xFFu8; 64]));
    }

    #[test]
    fn the_ratio_boundary_is_three_quarters() {
        // Seventy-five per cent exactly fails the strict comparison; anything
        // above it passes.
        let mut at = vec![b'a'; 3];
        at.push(0xFF);
        assert_eq!(at.len(), 4);
        assert!(!mostly_utf8(&at));

        let mut above = vec![b'a'; 7];
        above.push(0xFF);
        assert!(mostly_utf8(&above));
    }

    #[test]
    fn an_empty_run_is_not_text() {
        // Zero is not greater than zero, so the reference reports false.
        assert!(!mostly_utf8(b""));
    }

    #[test]
    fn a_wrapping_run_reads_the_same_bytes() {
        // The same text laid out so the run wraps the ring buffer must give
        // the same answer as the contiguous copy.
        let text = "naïve café £5".as_bytes();
        let mut ring = vec![0u8; 32];
        let mask = ring.len() - 1;
        let start = ring.len() - 5;
        for (offset, &byte) in text.iter().enumerate() {
            ring[(start + offset) & mask] = byte;
        }
        assert_eq!(
            is_mostly_utf8(&ring, start, mask, text.len()),
            mostly_utf8(text)
        );
    }

    /// The reference's full count, without stopping early.
    fn reference_answer(data: &[u8]) -> bool {
        let mut size_utf8 = 0usize;
        let mut index = 0usize;
        while index < data.len() {
            let (bytes_read, symbol) = parse_as_utf8(&data[index..]);
            index += bytes_read;
            if symbol < NOT_UTF8 {
                size_utf8 += bytes_read;
            }
        }
        size_utf8 as f64 > MIN_UTF8_RATIO * data.len() as f64
    }

    #[test]
    fn the_needed_count_is_the_first_integer_past_three_quarters() {
        assert_eq!(utf8_bytes_needed(0), 1);
        assert_eq!(utf8_bytes_needed(4), 4);
        assert_eq!(utf8_bytes_needed(5), 4);
        assert_eq!(utf8_bytes_needed(8), 7);
        for length in 0..10_000usize {
            let needed = utf8_bytes_needed(length);
            assert!(needed as f64 > MIN_UTF8_RATIO * length as f64);
            assert!((needed - 1) as f64 <= MIN_UTF8_RATIO * length as f64);
        }
    }

    #[test]
    fn stopping_early_gives_the_full_counts_answer() {
        let mut rng = 0x5851_F42D_4C95_7F2Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let pieces: [&[u8]; 6] = [
            b"text ",
            b"\xC3\xA9",
            b"\xE2\x82\xAC",
            b"\xF0\x9F\x98\x80",
            b"\xFF",
            b"\0",
        ];
        for round in 0..3000 {
            // The share of invalid pieces sweeps through the ratio boundary.
            let invalid_share = round % 11;
            let mut data = Vec::new();
            let target = (next() % 96) as usize;
            while data.len() < target {
                let piece = if (next() % 10) < invalid_share as u64 {
                    pieces[4 + (next() % 2) as usize]
                } else {
                    pieces[(next() % 4) as usize]
                };
                data.extend_from_slice(piece);
            }
            data.truncate(target);
            assert_eq!(
                mostly_utf8(&data),
                reference_answer(&data),
                "round {round}: {data:?}"
            );
            let mut ring = vec![0u8; 128];
            let mask = ring.len() - 1;
            let start = (next() % 128) as usize;
            for (offset, &byte) in data.iter().enumerate() {
                ring[(start + offset) & mask] = byte;
            }
            assert_eq!(
                is_mostly_utf8(&ring, start, mask, data.len()),
                reference_answer(&data),
                "round {round}, start {start}"
            );
        }
    }
}
