#![cfg(feature = "std")]

use borrowed_buf::{BorrowedBuf, io};
use core::mem::MaybeUninit;
use std::io::{ErrorKind, IoSlice, Read, Write};

/// Yields at most `chunk` bytes per call, interleaved with `Interrupted` errors.
struct Choppy<'a> {
    data: &'a [u8],
    chunk: usize,
    interrupt: bool,
}

impl Read for Choppy<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        self.interrupt = !self.interrupt;
        if self.interrupt {
            return Err(ErrorKind::Interrupted.into());
        }
        let n = out.len().min(self.chunk).min(self.data.len());
        out[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

#[test]
fn read_buf_single_read() {
    let mut reader: &[u8] = b"hello world";
    let mut storage = [MaybeUninit::uninit(); 5];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    io::read_buf(&mut reader, buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"hello");
    assert_eq!(reader, b" world");
}

#[test]
fn read_buf_zeroes_once() {
    let mut storage = [MaybeUninit::uninit(); 8];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut reader: &[u8] = b"ab";
    io::read_buf(&mut reader, buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"ab");
    assert!(buf.is_init());
    // EOF leaves the buffer untouched.
    io::read_buf(&mut reader, buf.unfilled()).unwrap();
    assert_eq!(buf.len(), 2);
}

#[test]
fn read_buf_exact_retries_interrupts_and_short_reads() {
    let mut reader = Choppy {
        data: b"0123456789",
        chunk: 3,
        interrupt: false,
    };
    let mut storage = [MaybeUninit::uninit(); 8];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    io::read_buf_exact(&mut reader, buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"01234567");
}

#[test]
fn read_buf_exact_eof_keeps_partial_data() {
    let mut reader: &[u8] = b"abc";
    let mut storage = [MaybeUninit::uninit(); 8];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let err = io::read_buf_exact(&mut reader, buf.unfilled()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    assert_eq!(buf.filled(), b"abc");
}

#[test]
fn read_buf_propagates_errors() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(ErrorKind::BrokenPipe.into())
        }
    }
    let mut storage = [MaybeUninit::uninit(); 4];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let err = io::read_buf_exact(&mut Broken, buf.unfilled()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    assert_eq!(buf.len(), 0);
}

#[test]
#[should_panic = "advanced past the initialized part"]
fn read_buf_rejects_lying_reader() {
    struct Liar;
    impl Read for Liar {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            Ok(out.len() + 1)
        }
    }
    let mut storage = [MaybeUninit::uninit(); 4];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let _ = io::read_buf(&mut Liar, buf.unfilled());
}

#[test]
fn cursor_implements_write() {
    let mut storage = [MaybeUninit::uninit(); 8];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    let (num, word) = (12, "ab");
    write!(cursor, "{num}-{word}").unwrap();
    // Short write once full.
    assert_eq!(cursor.write(b"xyzw").unwrap(), 3);
    assert_eq!(cursor.write(b"!").unwrap(), 0);
    cursor.flush().unwrap();
    assert_eq!(buf.filled(), b"12-abxyz");
}

#[test]
fn read_buf_into_partially_filled_buffer() {
    let mut storage = [MaybeUninit::uninit(); 6];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    let mut reader: &[u8] = b"cdefgh";
    io::read_buf(&mut reader, buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"abcdef");
}

#[test]
fn read_buf_inside_with_unfilled_buf() {
    let mut storage = [MaybeUninit::uninit(); 6];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut reader: &[u8] = b"xyz";
    let n = buf.unfilled().with_unfilled_buf(|sub| {
        io::read_buf(&mut reader, sub.unfilled()).unwrap();
        sub.len()
    });
    assert_eq!(n, 3);
    assert_eq!(buf.filled(), b"xyz");
    assert!(buf.is_init());
}

#[test]
fn cursor_write_vectored_and_write_all() {
    let mut storage = [MaybeUninit::uninit(); 6];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    let bufs = [
        IoSlice::new(b"ab"),
        IoSlice::new(b"cd"),
        IoSlice::new(b"ef"),
    ];
    assert_eq!(cursor.write_vectored(&bufs[..2]).unwrap(), 4);
    // Stops at the first short write.
    assert_eq!(
        cursor
            .write_vectored(&[IoSlice::new(b"xyz"), IoSlice::new(b"w")])
            .unwrap(),
        2
    );
    let err = cursor.write_all(b"!").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WriteZero);
    assert_eq!(buf.filled(), b"abcdxy");
}
