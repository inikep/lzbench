//! text_detect.rs - minimal measured text detector for 1.19.0 final ship
//! Context: 1.18.3 lock 243,675 slice MTBT 73 tests, 2,765,585 full at 2M, 900K->2,891,109 -125K, 4M vs 2M -3,086 NS. Pack cap at 2M.

/// Return ratio of printable ASCII + \n \r \t over total len. 0.0 if empty.
#[inline]
pub fn text_ratio(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut good = 0usize;
    for &b in data {
        if (32..=126).contains(&b) || b == b'\n' || b == b'\r' || b == b'\t' {
            good += 1;
        }
    }
    good as f32 / data.len() as f32
}

/// True if >85% of bytes are text-like (32..=126, \n \r \t)
#[inline]
pub fn is_text(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    text_ratio(data) > 0.85
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_text_returns_true() {
        let s = b"The quick brown fox jumps over the lazy dog.\nThis is English text with numbers 12345 and punctuation! \t\r\n";
        assert!(is_text(s));
        assert!(text_ratio(s) > 0.85);
    }

    #[test]
    fn binary_zero_to_ten_returns_false() {
        // 0..10 repeated - contains \t \n but mostly non-text
        let mut v = Vec::new();
        for _ in 0..100 {
            for b in 0u8..=10u8 {
                v.push(b);
            }
        }
        assert!(!is_text(&v));
        assert!(text_ratio(&v) < 0.85);
    }

    #[test]
    fn empty_false() {
        assert!(!is_text(b""));
        assert_eq!(text_ratio(b""), 0.0);
    }
}
