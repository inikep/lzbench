//! Owned inputs of a batch: immutable bytes, synchronized seekable readers
//! and positional file access.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::fs::{File, Metadata};
use std::io;
use std::path::Path;
use std::time::SystemTime;

use super::core::SeekReader;

/// Caller-defined version token of a [`RandomAccessSource`].
///
/// A batch records the token when it is prepared and compares it before it
/// writes the frame; a different token fails the batch with
/// [`ParallelEncodeError::SourceChanged`](super::ParallelEncodeError::SourceChanged).
/// The library compares bytes and does not hash content: use the same token
/// whenever the content is unchanged.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::SourceIdentity;
///
/// assert_eq!(SourceIdentity::from(b"v1".to_vec()), SourceIdentity::from(b"v1".to_vec()));
/// assert_ne!(SourceIdentity::from(b"v1".to_vec()), SourceIdentity::from(b"v2".to_vec()));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SourceIdentity(Vec<u8>);

impl From<Vec<u8>> for SourceIdentity {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// Owned random-access input shared by the tasks of a batch.
///
/// Offsets are absolute byte positions, independent of any cursor, and
/// concurrent calls may overlap. Implementations fill each requested range
/// or return an error, and keep the bytes unchanged until the batch is
/// finished: a task reads its segments and their overlap, and a frame with a
/// checksum reads the whole source once more when it is finished. A blocking
/// implementation blocks the executor thread that runs the task.
///
/// [`ArcBytesSource`], [`SeekSource`] and [`FileSource`] implement it for
/// common inputs.
///
/// # Examples
///
/// A custom source can delegate storage while supplying a version:
///
/// ```
/// use mzstd::compressor::parallel::{ArcBytesSource, RandomAccessSource, SourceIdentity};
/// use std::{io, sync::Arc};
///
/// struct Snapshot {
///     bytes: ArcBytesSource,
///     version: SourceIdentity,
/// }
///
/// impl RandomAccessSource for Snapshot {
///     fn len(&self) -> io::Result<u64> {
///         self.bytes.len()
///     }
///     fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
///         self.bytes.read_exact_at(offset, dst)
///     }
///     fn identity(&self) -> Option<SourceIdentity> {
///         Some(self.version.clone())
///     }
/// }
///
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
    /// The current length in bytes.
    ///
    /// # Errors
    ///
    /// An I/O error when the length is unavailable.
    fn len(&self) -> io::Result<u64>;

    /// Whether the current length is zero.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::len`] returns.
    fn is_empty(&self) -> io::Result<bool> {
        self.len().map(|len| len == 0)
    }

    /// Fills all of `dst` from the absolute `offset`; safe to call
    /// concurrently.
    ///
    /// # Errors
    ///
    /// An I/O error for an unavailable or truncated range; `dst` may then
    /// hold a partial read, which the batch discards.
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()>;

    /// The current version token, if the source has one. Metadata cannot
    /// detect every in-place write.
    fn identity(&self) -> Option<SourceIdentity> {
        None
    }
}

/// Immutable reference-counted bytes.
///
/// Construction and cloning share the allocation without copying it. See
/// [`RandomAccessSource`] for a read and
/// [`ParallelCompressor::prepare_source`](super::ParallelCompressor::prepare_source)
/// for a batch.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ArcBytesSource, RandomAccessSource};
/// use std::sync::Arc;
///
/// let source = ArcBytesSource::from(Arc::<[u8]>::from(&b"shared"[..]));
/// assert_eq!(source.as_ref(), b"shared");
/// assert!(source.read_exact_at(4, &mut [0; 3]).is_err());
/// ```
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
        let bytes = usize::try_from(offset)
            .ok()
            .and_then(|start| self.0.get(start..start.checked_add(dst.len())?))
            .ok_or(io::ErrorKind::UnexpectedEof)?;
        dst.copy_from_slice(bytes);
        Ok(())
    }
}

/// Adapts an owned `Read + Seek + Send` reader to random access.
///
/// Every read seeks to its absolute offset and reads under one mutex, so
/// reads are serialized while compression runs outside the lock, and the
/// reader need not be `Sync`. The original cursor position is ignored and
/// the length is queried by seeking to the end. The adapter checks the
/// length but has no identity token, so keep the bytes unchanged. A panic
/// inside the reader poisons it, and later reads fail. Use [`FileSource`]
/// for concurrent positional reads of regular files.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{RandomAccessSource, SeekSource};
/// use std::io::Cursor;
///
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
    inner: SeekReader<R>,
}

impl<R> From<R> for SeekSource<R> {
    /// Takes the reader without reading or seeking.
    fn from(reader: R) -> Self {
        Self {
            inner: SeekReader::from(reader),
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

/// An open regular file read with positional reads, which never move its
/// cursor, so tasks read it concurrently.
///
/// The handle does not freeze the file: keep it unchanged until the batch is
/// finished. Its identity (length, modification time and, on Unix, device,
/// inode and change time) catches most writes, but not every one. On
/// platforms other than Unix and Windows reads fail with
/// [`io::ErrorKind::Unsupported`].
#[derive(Debug)]
pub struct FileSource {
    file: File,
}

impl FileSource {
    /// Opens a regular file.
    ///
    /// # Errors
    ///
    /// Whatever opening the file or reading its metadata returns, and
    /// [`io::ErrorKind::InvalidInput`] when it is not a regular file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mzstd::compressor::parallel::{FileSource, RandomAccessSource};
    ///
    /// let source = FileSource::open("input.bin")?;
    /// let mut magic = [0; 4];
    /// source.read_exact_at(0, &mut magic)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::try_from(File::open(path)?)
    }
}

impl TryFrom<File> for FileSource {
    type Error = io::Error;

    /// Takes a regular file's handle.
    ///
    /// # Errors
    ///
    /// Whatever reading the metadata returns, and
    /// [`io::ErrorKind::InvalidInput`] when it is not a regular file.
    fn try_from(file: File) -> io::Result<Self> {
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a parallel source must be a regular file",
            ));
        }
        Ok(Self { file })
    }
}

impl RandomAccessSource for FileSource {
    fn len(&self) -> io::Result<u64> {
        self.file.metadata().map(|metadata| metadata.len())
    }

    fn read_exact_at(&self, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
        while !dst.is_empty() {
            #[cfg(unix)]
            let result = std::os::unix::fs::FileExt::read_at(&self.file, dst, offset);
            #[cfg(windows)]
            let result = std::os::windows::fs::FileExt::seek_read(&self.file, dst, offset);
            #[cfg(not(any(unix, windows)))]
            let result: io::Result<usize> = Err(io::ErrorKind::Unsupported.into());
            match result {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(read) => {
                    offset = offset
                        .checked_add(read as u64)
                        .ok_or(io::ErrorKind::InvalidInput)?;
                    dst = &mut dst[read..];
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn identity(&self) -> Option<SourceIdentity> {
        self.file.metadata().ok().map(metadata_identity)
    }
}

/// The length, modification time and, on Unix, the device, inode and change
/// time of a file.
fn metadata_identity(metadata: Metadata) -> SourceIdentity {
    let mut bytes = metadata.len().to_le_bytes().to_vec();
    if let Ok(time) = metadata.modified().and_then(|time| {
        time.duration_since(SystemTime::UNIX_EPOCH)
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
