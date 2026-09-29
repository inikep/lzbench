//! Borrow an entire stored member when no history will ever be referenced.
//! Unsupported shapes and malformed headers return to the full state machine.

use super::{
    bits::{Bits, Input},
    header::{self, MetaBlock},
};
use crate::decompressor::WindowLimit;

pub(crate) fn payload(src: &[u8], limit: WindowLimit) -> Option<&[u8]> {
    // A four-bit standard window plus a non-final, four-nibble raw block
    // occupies exactly three bytes: WBITS(4), ISLAST(1), MNIBBLES(2),
    // MLEN(16), ISUNCOMPRESSED(1). Check the whole shape before borrowing.
    if let Some(&[a, b, c]) = src.first_chunk::<3>() {
        let header = u32::from_le_bytes([a, b, c, 0]);
        if header & 0x80_0071 == 0x80_0001 && header & 0x0e != 0 {
            let window = 17 + ((a >> 1) & 7);
            let length = ((header >> 7) & 0xffff) as usize + 1;
            if window <= limit.max_bits() && src.len() == length + 4 && src.last() == Some(&3) {
                return Some(&src[3..3 + length]);
            }
            return None;
        }
    }
    let mut bits = Bits::default();
    let mut input = Input::new(src, 0, None);
    header::window(&mut bits, &mut input, limit).ok()??;
    let payload = match header::metablock(&mut bits, &mut input).ok()?? {
        MetaBlock::End => &src[..0],
        MetaBlock::Uncompressed { length } => {
            // Header parsing is demand-driven and raw payload begins aligned.
            // No payload byte has been speculatively accepted into `bits`.
            let end = input.consumed.checked_add(usize::try_from(length).ok()?)?;
            let payload = src.get(input.consumed..end)?;
            input.consumed = end;
            if !matches!(
                header::metablock(&mut bits, &mut input).ok()??,
                MetaBlock::End
            ) {
                return None;
            }
            payload
        }
        _ => return None,
    };
    bits.align().ok()?;
    (input.consumed == src.len()).then_some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_stored_member_borrows_payload_and_rejects_padding_and_tails() {
        let src = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
        assert_eq!(
            payload(
                &src,
                crate::decompressor::DecoderConfig::default().window_limit()
            ),
            Some(&b"hello"[..])
        );
        assert_eq!(
            payload(
                &[0x3b],
                crate::decompressor::DecoderConfig::default().window_limit()
            ),
            Some(&b""[..])
        );
        for end in 0..src.len() {
            assert!(
                payload(
                    &src[..end],
                    crate::decompressor::DecoderConfig::default().window_limit()
                )
                .is_none()
            );
        }
        let mut bad = src.to_vec();
        bad.push(0);
        assert!(
            payload(
                &bad,
                crate::decompressor::DecoderConfig::default().window_limit()
            )
            .is_none()
        );
        bad.pop();
        bad[8] |= 0x80;
        assert!(
            payload(
                &bad,
                crate::decompressor::DecoderConfig::default().window_limit()
            )
            .is_none()
        );
        assert!(payload(&src, WindowLimit::standard(10).unwrap()).is_none());
    }
}
