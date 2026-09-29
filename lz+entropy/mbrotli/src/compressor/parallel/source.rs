//! Immutable bytes, synchronized seekable readers, and positional file access.
use std::{
    fs::{File, Metadata},
    io,
    path::Path,
    sync::Arc,
    time::SystemTime,
};

/// Caller-defined metadata token. Equality is checked before output mutation.
///
/// Construct with `SourceIdentity::from(Vec<u8>)`. A custom source can encode an
/// immutable object version here; use the same token whenever its content is
/// unchanged. The library compares bytes and does not compute a content hash.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SourceIdentity(Vec<u8>);
impl From<Vec<u8>> for SourceIdentity {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// Owned random-access input shared by detached tasks.
/// Implementations must fill each requested range or return an error, and keep
/// bytes immutable for the batch lifetime. A blocking implementation can block
/// its executor thread; the library cannot forcibly interrupt that call.
///
/// Offsets are absolute byte positions, independent of any cursor. Concurrent
/// calls may overlap. On error, `dst` may contain a partial read; the batch fails
/// and does not assemble that segment. [`ArcBytesSource`], [`SeekSource`], and
/// [`FileSource`] implement this trait for common input types.
///
/// # Examples
///
/// A custom source can delegate immutable storage while supplying a version:
///
/// ```
/// use mbrotli::compressor::parallel::{ArcBytesSource, RandomAccessSource, SourceIdentity};
/// use std::{io, sync::Arc};
/// struct Snapshot { bytes: ArcBytesSource, version: SourceIdentity }
/// impl RandomAccessSource for Snapshot {
///     fn len(&self) -> io::Result<u64> { self.bytes.len() }
///     fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
///         self.bytes.read_exact_at(offset, dst)
///     }
///     fn identity(&self) -> Option<SourceIdentity> { Some(self.version.clone()) }
/// }
/// let source = Snapshot {
///     bytes: ArcBytesSource::from(Arc::<[u8]>::from(&b"payload"[..])),
///     version: SourceIdentity::from(b"version-1".to_vec()),
/// };
/// assert!(!source.is_empty()?);
/// let mut bytes = [0; 4];
/// source.read_exact_at(3, &mut bytes)?;
/// assert_eq!(&bytes, b"load");
/// assert!(source.identity().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait RandomAccessSource: Send + Sync + 'static {
    /// Current byte length.
    /// # Errors
    /// Returns an I/O error when metadata is unavailable.
    fn len(&self) -> io::Result<u64>;
    /// Whether the current length is zero.
    /// # Errors
    /// Propagates `len` failures.
    fn is_empty(&self) -> io::Result<bool> {
        self.len().map(|n| n == 0)
    }
    /// Fills exactly `dst` from an absolute offset, safely under concurrent calls.
    /// # Errors
    /// Returns an error for an unavailable or truncated range.
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()>;
    /// Current identity/version token, if available. Metadata alone cannot prove
    /// immutability against every in-place mutation on every filesystem.
    fn identity(&self) -> Option<SourceIdentity> {
        None
    }
}
/// Immutable reference-counted bytes for detached tasks.
///
/// Construction and cloning share the allocation without copying payload bytes.
/// See [`RandomAccessSource`] for a read example and
/// [`super::ParallelCompressor::prepare_source`] for task scheduling.
#[derive(Clone, Debug)]
pub struct ArcBytesSource(Arc<[u8]>);
impl From<Arc<[u8]>> for ArcBytesSource {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self(bytes)
    }
}
impl AsRef<[u8]> for ArcBytesSource {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl RandomAccessSource for ArcBytesSource {
    fn len(&self) -> io::Result<u64> {
        Ok(self.0.len() as u64)
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        let offset = usize::try_from(offset).map_err(|_| io::ErrorKind::UnexpectedEof)?;
        let end = offset
            .checked_add(dst.len())
            .ok_or(io::ErrorKind::UnexpectedEof)?;
        let bytes = self
            .0
            .get(offset..end)
            .ok_or(io::ErrorKind::UnexpectedEof)?;
        dst.copy_from_slice(bytes);
        Ok(())
    }
}
/// Adapts an owned `Read + Seek + Send` reader to shared random-access input.
/// Each seek-and-read holds one mutex; reads are serialized, while compression
/// runs outside the lock. The reader need not implement `Sync`.
///
/// Length is queried by seeking to the end. Every read seeks to an absolute
/// offset, so the original cursor position is ignored. The underlying bytes
/// must remain immutable; this adapter provides length checks, but no identity
/// token. A panic while accessing the reader poisons it and later I/O fails.
/// Use [`FileSource`] for concurrent positional reads of regular files.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{RandomAccessSource, SeekSource};
/// use std::io::Cursor;
/// let source = SeekSource::from(Cursor::new(b"abcdef".to_vec()));
/// assert_eq!(source.len()?, 6);
/// let mut bytes = [0; 3];
/// source.read_exact_at(2, &mut bytes)?;
/// assert_eq!(&bytes, b"cde");
/// assert!(source.read_exact_at(5, &mut bytes).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SeekSource<R> {
    inner: super::core::source::SeekReader<R>,
}
impl<R> From<R> for SeekSource<R> {
    /// Takes exclusive ownership without reading or seeking.
    fn from(reader: R) -> Self {
        Self {
            inner: super::core::source::SeekReader::from(reader),
        }
    }
}
impl<R: io::Read + io::Seek + Send + 'static> RandomAccessSource for SeekSource<R> {
    fn len(&self) -> io::Result<u64> {
        self.inner.len()
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact_at(offset, dst)
    }
}

/// Stable open regular-file handle; positional reads never modify its cursor.
///
/// The handle does not freeze file contents. Keep the file unchanged for the
/// entire batch; metadata verification cannot detect every concurrent write.
/// On platforms other than Unix and Windows, positional reads return
/// [`io::ErrorKind::Unsupported`].
#[derive(Debug)]
pub struct FileSource {
    file: File,
}
impl FileSource {
    /// Opens a regular file for concurrent read-only positional access.
    /// # Errors
    /// Propagates open/metadata errors and rejects non-regular files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mbrotli::compressor::parallel::{FileSource, RandomAccessSource};
    /// let source = FileSource::open("input.bin")?;
    /// let mut header = [0; 4];
    /// source.read_exact_at(0, &mut header)?;
    /// assert!(source.len()? >= 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::try_from(File::open(path)?)
    }
}
impl TryFrom<File> for FileSource {
    type Error = io::Error;
    /// Takes ownership of a regular file handle.
    /// # Errors
    /// Rejects non-regular files and unavailable metadata.
    fn try_from(file: File) -> io::Result<Self> {
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "parallel input must be a regular file",
            ));
        }
        Ok(Self { file })
    }
}
impl RandomAccessSource for FileSource {
    fn len(&self) -> io::Result<u64> {
        self.file.metadata().map(|m| m.len())
    }
    fn read_exact_at(&self, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
        while !dst.is_empty() {
            #[cfg(unix)]
            let result = {
                use std::os::unix::fs::FileExt;
                self.file.read_at(dst, offset)
            };
            #[cfg(windows)]
            let result = {
                use std::os::windows::fs::FileExt;
                self.file.seek_read(dst, offset)
            };
            #[cfg(not(any(unix, windows)))]
            let result: io::Result<usize> = Err(io::ErrorKind::Unsupported.into());
            match result {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    offset = offset
                        .checked_add(n as u64)
                        .ok_or(io::ErrorKind::InvalidInput)?;
                    dst = &mut dst[n..];
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => (),
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    fn identity(&self) -> Option<SourceIdentity> {
        self.file.metadata().ok().map(metadata_identity)
    }
}
fn metadata_identity(metadata: Metadata) -> SourceIdentity {
    let mut bytes = metadata.len().to_le_bytes().to_vec();
    if let Ok(time) = metadata.modified().and_then(|t| {
        t.duration_since(SystemTime::UNIX_EPOCH)
            .map_err(io::Error::other)
    }) {
        bytes.extend_from_slice(&time.as_nanos().to_le_bytes());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        bytes.extend_from_slice(&metadata.dev().to_le_bytes());
        bytes.extend_from_slice(&metadata.ino().to_le_bytes());
        bytes.extend_from_slice(&metadata.ctime().to_le_bytes());
        bytes.extend_from_slice(&metadata.ctime_nsec().to_le_bytes());
    }
    SourceIdentity(bytes)
}
