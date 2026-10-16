//! Integration with [`std::io`]. Requires the `std` feature.
//!
//! These are stable stand-ins for the nightly-only `Read::read_buf` and `Read::read_buf_exact`,
//! with the same behavior as their default implementations.

use crate::BorrowedCursor;
use std::io::{self, ErrorKind, IoSlice, Read, Write};

/// Reads from `reader` into `cursor` with a single [`Read::read`] call.
///
/// Mirrors the default implementation of the nightly `Read::read_buf`. The unfilled part of the
/// underlying [`BorrowedBuf`](crate::BorrowedBuf) is zeroed first, but only once over its lifetime,
/// so reading in a loop stays cheap.
///
/// # Panics
///
/// Panics if `reader` reports more bytes than the slice it was given.
///
/// ```
/// use borrowed_buf::{BorrowedBuf, io};
/// use core::mem::MaybeUninit;
///
/// let mut reader: &[u8] = b"hello world";
/// let mut storage = [MaybeUninit::uninit(); 64];
/// let mut buf = BorrowedBuf::from(&mut storage[..]);
///
/// // Read until EOF, zeroing the buffer only once.
/// loop {
///     let before = buf.len();
///     io::read_buf(&mut reader, buf.unfilled())?;
///     if buf.len() == before {
///         break;
///     }
/// }
/// assert_eq!(buf.filled(), b"hello world");
/// # Ok::<(), std::io::Error>(())
/// ```
#[inline]
pub fn read_buf<R: Read + ?Sized>(
    reader: &mut R,
    mut cursor: BorrowedCursor<'_, u8>,
) -> io::Result<()> {
    let n = reader.read(cursor.ensure_init())?;
    cursor.advance_checked(n);
    Ok(())
}

/// Reads from `reader` until `cursor` is full.
///
/// Mirrors the default implementation of the nightly `Read::read_buf_exact`. Retries on
/// [`ErrorKind::Interrupted`]. Returns [`ErrorKind::UnexpectedEof`] if the reader ends first;
/// whatever was read up to that point stays in the buffer.
///
/// ```
/// use borrowed_buf::{BorrowedBuf, io};
/// use core::mem::MaybeUninit;
///
/// let mut reader: &[u8] = b"\x00\x05rest";
/// let mut storage = [MaybeUninit::uninit(); 2];
/// let mut header = BorrowedBuf::from(&mut storage[..]);
/// io::read_buf_exact(&mut reader, header.unfilled())?;
/// assert_eq!(u16::from_be_bytes([header.filled()[0], header.filled()[1]]), 5);
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn read_buf_exact<R: Read + ?Sized>(
    reader: &mut R,
    mut cursor: BorrowedCursor<'_, u8>,
) -> io::Result<()> {
    while cursor.capacity() > 0 {
        let before = cursor.written();
        match read_buf(reader, cursor.reborrow()) {
            Ok(()) if cursor.written() == before => {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl Write for BorrowedCursor<'_, u8> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = buf.len().min(self.capacity());
        self.append(&buf[..n]);
        Ok(n)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut written = 0;
        for buf in bufs {
            let n = self.write(buf)?;
            written += n;
            if n < buf.len() {
                break;
            }
        }
        Ok(written)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.write(buf)? < buf.len() {
            Err(io::Error::new(
                ErrorKind::WriteZero,
                "failed to write whole buffer",
            ))
        } else {
            Ok(())
        }
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
