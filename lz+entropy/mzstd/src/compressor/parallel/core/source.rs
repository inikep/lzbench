//! What a batch reads: a borrowed slice or a shared random-access source.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Mutex;

use crate::compressor::parallel::{
    ParallelEncodeError, RandomAccessSource, SourceIdentity, SourceIoError,
};
use crate::shared::xxhash::Xxh64;

/// Lets an `Arc<S>` with `S: ?Sized` become an `Arc<dyn
/// RandomAccessSource>`: only the handle is wrapped, the bytes are not
/// copied.
pub(crate) struct SharedSource<S: ?Sized>(pub(crate) Arc<S>);

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

/// A reader whose seeks and reads happen as one transaction under a mutex.
#[derive(Debug)]
pub(crate) struct SeekReader<R>(Mutex<R>);

impl<R> From<R> for SeekReader<R> {
    fn from(reader: R) -> Self {
        Self(Mutex::new(reader))
    }
}

impl<R: Read + Seek> SeekReader<R> {
    /// Locks the reader; a reader that panicked is not used again, since its
    /// state is unknown.
    fn reader(&self) -> io::Result<std::sync::MutexGuard<'_, R>> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("seek source lock poisoned"))
    }

    /// The length, found by seeking to the end.
    pub(crate) fn len(&self) -> io::Result<u64> {
        self.reader()?.seek(SeekFrom::End(0))
    }

    /// Fills `dst` from the absolute `offset`.
    pub(crate) fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        // The guard lives until `read_exact` returns: no other task may move
        // the cursor between this seek and its read.
        let mut reader = self.reader()?;
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(dst)
    }
}

/// The input of a batch.
#[derive(Clone)]
pub(crate) enum Input<'a> {
    /// Borrowed bytes.
    Slice(&'a [u8]),
    /// A source tasks read their ranges from.
    Source(Arc<dyn RandomAccessSource>),
}

impl fmt::Debug for Input<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slice(bytes) => f.debug_tuple("Slice").field(&bytes.len()).finish(),
            Self::Source(_) => f.debug_tuple("Source").finish_non_exhaustive(),
        }
    }
}

impl Input<'_> {
    /// The input's length.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::SourceMetadata`] when the source cannot tell.
    pub(crate) fn len(&self) -> Result<u64, ParallelEncodeError> {
        match self {
            Self::Slice(bytes) => Ok(bytes.len() as u64),
            Self::Source(source) => source
                .len()
                .map_err(|error| ParallelEncodeError::SourceMetadata(error.into())),
        }
    }

    /// The source's version token; a slice has none.
    pub(crate) fn identity(&self) -> Option<SourceIdentity> {
        match self {
            Self::Slice(_) => None,
            Self::Source(source) => source.identity(),
        }
    }

    /// Bytes `range` of the input, which must lie within the planned
    /// length. A source is read into `buffer`, resized to the range.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::SourceRead`] naming `segment` when the source
    /// fails.
    pub(crate) fn read<'b>(
        &'b self,
        range: Range<u64>,
        segment: u64,
        buffer: &'b mut Vec<u8>,
    ) -> Result<&'b [u8], ParallelEncodeError> {
        match self {
            Self::Slice(bytes) => Ok(&bytes[range.start as usize..range.end as usize]),
            Self::Source(source) => {
                // Every byte is overwritten by the read; only a grown tail is
                // zeroed first.
                buffer.resize((range.end - range.start) as usize, 0);
                source.read_exact_at(range.start, buffer).map_err(|error| {
                    ParallelEncodeError::SourceRead {
                        segment,
                        source: SourceIoError::from(error),
                    }
                })?;
                Ok(buffer)
            }
        }
    }

    /// Feeds the first `len` bytes to `hasher` in order, reading a source
    /// one `segment`-sized chunk at a time.
    ///
    /// # Errors
    ///
    /// As [`Self::read`], naming the segment whose chunk failed.
    pub(crate) fn hash(
        &self,
        len: u64,
        segment: usize,
        hasher: &mut Xxh64,
    ) -> Result<(), ParallelEncodeError> {
        let mut buffer = Vec::new();
        let segment = segment as u64;
        for index in 0..len.div_ceil(segment) {
            let start = index * segment;
            let end = (start + segment).min(len);
            hasher.update(self.read(start..end, index, &mut buffer)?);
        }
        Ok(())
    }

    /// Checks that the input still has the length and identity it was
    /// planned with; a slice always does.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::SourceMetadata`] when its length cannot be
    /// read, and [`ParallelEncodeError::SourceChanged`] when either differs.
    pub(crate) fn verify(
        &self,
        len: u64,
        identity: Option<&SourceIdentity>,
    ) -> Result<(), ParallelEncodeError> {
        if self.len()? != len || self.identity().as_ref() != identity {
            return Err(ParallelEncodeError::SourceChanged);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::parallel::SeekSource;
    use std::io::Cursor;

    struct FaultReader {
        inner: Cursor<Vec<u8>>,
        seek_error: bool,
        read_error: bool,
        interrupt: bool,
        panic: bool,
    }

    impl FaultReader {
        fn new(bytes: &[u8]) -> Self {
            Self {
                inner: Cursor::new(bytes.to_vec()),
                seek_error: false,
                read_error: false,
                interrupt: false,
                panic: false,
            }
        }
    }

    impl Read for FaultReader {
        fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
            assert!(!self.panic, "injected reader panic");
            if core::mem::take(&mut self.interrupt) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            if self.read_error {
                return Err(io::ErrorKind::PermissionDenied.into());
            }
            // Short reads: one byte at a time.
            let len = dst.len().min(1);
            self.inner.read(&mut dst[..len])
        }
    }

    impl Seek for FaultReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            if self.seek_error {
                return Err(io::ErrorKind::Unsupported.into());
            }
            self.inner.seek(position)
        }
    }

    #[test]
    fn exact_reads_retry_interruptions_and_short_reads_and_propagate_failures() {
        let source = SeekReader::from(FaultReader {
            interrupt: true,
            ..FaultReader::new(b"abcdef")
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
    fn a_panicking_reader_is_not_reused() {
        let source = SeekReader::from(FaultReader {
            panic: true,
            ..FaultReader::new(&[1])
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

    #[test]
    fn slices_and_sources_read_and_hash_the_same_bytes() {
        let bytes: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let source: Arc<dyn RandomAccessSource> = Arc::new(SharedSource(Arc::new(
            SeekSource::from(Cursor::new(bytes.clone())),
        )));
        let inputs = [Input::Slice(&bytes), Input::Source(source)];
        let mut buffer = vec![9; 2000];
        for input in &inputs {
            assert_eq!(input.len().unwrap(), 1000);
            assert_eq!(input.identity(), None);
            assert_eq!(input.read(10..20, 0, &mut buffer).unwrap(), &bytes[10..20]);
            input.verify(1000, None).unwrap();
            assert_eq!(
                input.verify(999, None),
                Err(ParallelEncodeError::SourceChanged)
            );
            let identity = SourceIdentity::from(vec![1]);
            assert_eq!(
                input.verify(1000, Some(&identity)),
                Err(ParallelEncodeError::SourceChanged)
            );
            // Chunks of 64 bytes, the last one short.
            let mut chunked = Xxh64::new(0);
            input.hash(1000, 64, &mut chunked).unwrap();
            let mut whole = Xxh64::new(0);
            whole.update(&bytes);
            assert_eq!(chunked.digest(), whole.digest());
        }
        assert_eq!(format!("{:?}", inputs[0]), "Slice(1000)");
        assert_eq!(format!("{:?}", inputs[1]), "Source(..)");
    }

    #[test]
    fn source_failures_name_the_segment_they_hit() {
        let source: Arc<dyn RandomAccessSource> =
            Arc::new(SeekSource::from(FaultReader::new(&[0; 100])));
        let input = Input::Source(source);
        let mut hasher = Xxh64::new(0);
        // Planned as 130 bytes: the third 64-byte chunk runs past the end.
        let error = input.hash(130, 64, &mut hasher).unwrap_err();
        let ParallelEncodeError::SourceRead { segment, source } = error else {
            panic!("{error:?}");
        };
        assert_eq!(segment, 1);
        assert_eq!(source.as_ref().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn a_source_without_a_length_reports_its_metadata_failure() {
        let mut reader = FaultReader::new(&[]);
        reader.seek_error = true;
        let input = Input::Source(Arc::new(SeekSource::from(reader)));
        let Err(ParallelEncodeError::SourceMetadata(error)) = input.len() else {
            panic!("expected a metadata failure");
        };
        assert_eq!(error.as_ref().kind(), io::ErrorKind::Unsupported);
        assert!(matches!(
            input.verify(0, None),
            Err(ParallelEncodeError::SourceMetadata(_))
        ));
    }
}
