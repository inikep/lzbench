//! Safe writes into caller-provided uninitialized output.

use core::mem::MaybeUninit;

#[cfg(feature = "decompression")]
use borrowed_buf::BorrowedBuf;

/// Initializes `destination` from `source`, which must have the same length.
///
/// `<[MaybeUninit<u8>]>::write_copy_of_slice` needs Rust 1.93; this loop keeps
/// the 1.89 MSRV and compiles to a `memcpy`. Unequal lengths copy the shorter
/// prefix, which no caller relies on.
#[inline]
pub(crate) fn copy_to_uninit(destination: &mut [MaybeUninit<u8>], source: &[u8]) {
    debug_assert_eq!(destination.len(), source.len());
    for (slot, &byte) in destination.iter_mut().zip(source) {
        slot.write(byte);
    }
}

/// Extends the filled prefix of `buffer` by `len` zero bytes, or by what is
/// left of its capacity when that is less.
///
/// Zeroing turns uninitialized space into bytes safe code may read back, so
/// the filled prefix can serve as history.
#[cfg(feature = "decompression")]
pub(crate) fn zero_extend(buffer: &mut BorrowedBuf<'_, u8>, len: usize) {
    let mut cursor = buffer.unfilled();
    let len = len.min(cursor.capacity());
    cursor.append_fill(len, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_every_byte_and_leaves_nothing_else_written() {
        let mut destination = [MaybeUninit::new(0xAA); 6];
        copy_to_uninit(&mut destination[1..5], b"abcd");
        // SAFETY: every element was initialized by the array literal and the copy.
        let bytes = destination.map(|slot| unsafe { slot.assume_init() });
        assert_eq!(&bytes, b"\xAAabcd\xAA");
    }

    #[test]
    fn an_empty_copy_is_a_no_op() {
        copy_to_uninit(&mut [], &[]);
    }

    #[cfg(feature = "decompression")]
    #[test]
    fn zero_extend_appends_zeros_after_the_filled_prefix() {
        let mut storage = [MaybeUninit::new(0xAA); 10_000];
        let mut buffer = BorrowedBuf::<u8>::from(&mut storage[..]);
        buffer.unfilled().append(b"ab");
        zero_extend(&mut buffer, 5000);
        assert_eq!(buffer.len(), 5002);
        assert_eq!(&buffer.filled()[..2], b"ab");
        assert!(buffer.filled()[2..].iter().all(|&byte| byte == 0));
        // SAFETY: every element was initialized by the array literal.
        let rest = storage[5002..]
            .iter()
            .map(|slot| unsafe { slot.assume_init() });
        assert!(rest.into_iter().all(|byte| byte == 0xAA));
    }

    #[cfg(feature = "decompression")]
    #[test]
    fn zero_extend_stops_at_the_capacity() {
        let mut storage = [MaybeUninit::uninit(); 6];
        let mut buffer = BorrowedBuf::<u8>::from(&mut storage[..]);
        zero_extend(&mut buffer, 4);
        zero_extend(&mut buffer, 4);
        assert_eq!(buffer.filled(), &[0; 6]);
        zero_extend(&mut buffer, 1);
        assert_eq!(buffer.len(), 6);
    }

    #[cfg(feature = "decompression")]
    #[test]
    fn zero_extend_by_nothing_leaves_the_buffer_empty() {
        let mut storage = [MaybeUninit::uninit(); 3];
        let mut buffer = BorrowedBuf::<u8>::from(&mut storage[..]);
        zero_extend(&mut buffer, 0);
        assert_eq!(buffer.len(), 0);
    }
}
