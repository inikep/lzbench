//! Private decoder algorithms and resumable parsing state.

mod bits;
mod block;
mod context_map;
mod dictionary;
mod distance;
mod header;
mod huffman;
mod memory;
mod session;
mod stored;
mod stream;

pub(crate) use bits::Input;
pub(crate) use session::{Delivery, OperationState};
pub(crate) use stored::payload as stored_payload;
pub(crate) use stream::{Output, Stop, Stream};

#[cfg(test)]
mod fixtures {
    use super::Input;
    use alloc::vec::Vec;

    /// Test-only RFC fields, written least-significant bit first.
    pub(super) fn fields(fields: &[(u8, u64)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut position = 0usize;
        for &(width, value) in fields {
            assert!(width < 64 && value >> width == 0);
            for bit in 0..width {
                if position.is_multiple_of(8) {
                    bytes.push(0);
                }
                bytes[position / 8] |= (((value >> bit) & 1) as u8) << (position % 8);
                position += 1;
            }
        }
        bytes
    }
    pub(super) fn input(bytes: &[u8]) -> Input<'_> {
        Input::new(bytes, 0, None)
    }
}
