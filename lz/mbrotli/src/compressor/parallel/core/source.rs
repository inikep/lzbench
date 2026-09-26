//! Source type erasure and atomic seek/read transactions.
use crate::compressor::parallel::{RandomAccessSource, SourceIdentity};
use std::{
    io::{self, Read, Seek, SeekFrom},
    sync::{Arc, Mutex},
};

// A sized bridge permits Arc<S> with S: ?Sized to enter the existing erased
// source path. Only the shared handle is wrapped; payload bytes are not copied.
pub(in crate::compressor::parallel) struct SharedSource<S: ?Sized>(pub Arc<S>);
impl<S: RandomAccessSource + ?Sized> RandomAccessSource for SharedSource<S> {
    fn len(&self) -> io::Result<u64> {
        self.0.len()
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        self.0.read_exact_at(offset, dst)
    }
    fn identity(&self) -> Option<SourceIdentity> {
        self.0.identity()
    }
}

#[derive(Debug)]
pub(in crate::compressor::parallel) struct SeekReader<R>(Mutex<R>);
impl<R> From<R> for SeekReader<R> {
    fn from(reader: R) -> Self {
        Self(Mutex::new(reader))
    }
}
impl<R: Read + Seek> SeekReader<R> {
    pub(in crate::compressor::parallel) fn len(&self) -> io::Result<u64> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("seek source lock poisoned"))?
            .seek(SeekFrom::End(0))
    }

    pub(in crate::compressor::parallel) fn read_exact_at(
        &self,
        offset: u64,
        dst: &mut [u8],
    ) -> io::Result<()> {
        let mut reader = self
            .0
            .lock()
            .map_err(|_| io::Error::other("seek source lock poisoned"))?;
        // Keep the guard until read_exact finishes: another task must never
        // move this cursor between a seek and its corresponding read.
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct FaultReader {
        inner: Cursor<Vec<u8>>,
        seek_error: bool,
        read_error: bool,
        interrupt: bool,
        panic: bool,
    }
    impl Read for FaultReader {
        fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
            assert!(!self.panic, "injected reader panic");
            if std::mem::take(&mut self.interrupt) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            if self.read_error {
                return Err(io::ErrorKind::PermissionDenied.into());
            }
            let len = dst.len().min(1);
            self.inner.read(&mut dst[..len])
        }
    }
    impl Seek for FaultReader {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            if self.seek_error {
                return Err(io::ErrorKind::Unsupported.into());
            }
            self.inner.seek(pos)
        }
    }

    #[test]
    fn exact_reads_retry_interruptions_and_short_reads_and_propagate_failures() {
        let source = SeekReader::from(FaultReader {
            inner: Cursor::new(b"abcdef".to_vec()),
            seek_error: false,
            read_error: false,
            interrupt: true,
            panic: false,
        });
        let mut out = [0; 3];
        assert_eq!(source.len().unwrap(), 6);
        source.read_exact_at(2, &mut out).unwrap();
        assert_eq!(&out, b"cde");
        assert_eq!(
            source.read_exact_at(5, &mut out).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        source.0.lock().unwrap().read_error = true;
        assert_eq!(
            source.read_exact_at(0, &mut out).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        source.0.lock().unwrap().seek_error = true;
        assert_eq!(source.len().unwrap_err().kind(), io::ErrorKind::Unsupported);
        assert_eq!(
            source.read_exact_at(0, &mut out).unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
    }

    #[test]
    fn a_panicking_reader_is_not_reused_after_mutex_poisoning() {
        let source = SeekReader::from(FaultReader {
            inner: Cursor::new(vec![1]),
            seek_error: false,
            read_error: false,
            interrupt: false,
            panic: true,
        });
        assert!(std::panic::catch_unwind(|| source.read_exact_at(0, &mut [0])).is_err());
        assert_eq!(
            source.len().unwrap_err().to_string(),
            "seek source lock poisoned"
        );
        assert_eq!(
            source.read_exact_at(0, &mut [0]).unwrap_err().to_string(),
            "seek source lock poisoned"
        );
    }
}
