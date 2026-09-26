//! Exclusively created task files, closed before best-effort path cleanup.
use std::collections::hash_map::RandomState;
use std::fs::{self, File, OpenOptions};
use std::hash::BuildHasher;
use std::io;
use std::path::{Path, PathBuf};

pub(in crate::compressor::parallel) struct Spool {
    // Fields drop in declaration order: Windows requires closing before removal.
    pub(super) file: File,
    _path: SpoolPath,
}

struct SpoolPath(PathBuf);

impl Drop for SpoolPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

impl Spool {
    pub(super) fn new(directory: &Path) -> io::Result<Self> {
        // Keep cleanup independent of subsequent changes to the working directory.
        let directory = directory.canonicalize()?;
        let names = RandomState::new();
        for attempt in 0..128 {
            let name = format!("mbrotli-task-{:016x}", names.hash_one(attempt));
            match Self::create(directory.join(name)) {
                Ok(spool) => return Ok(spool),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create a unique task spool after 128 attempts",
        ))
    }

    fn create(path: PathBuf) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok(Self {
            file,
            _path: SpoolPath(path),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    #[test]
    fn spools_are_independent_seekable_and_removed_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let mut first = Spool::new(directory.path()).unwrap();
        let second = Spool::new(directory.path()).unwrap();
        assert_ne!(first._path.0, second._path.0);
        assert!(first._path.0.is_absolute());
        assert_eq!(second.file.metadata().unwrap().len(), 0);
        first.file.write_all(b"staged bytes").unwrap();
        first.file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        first.file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"staged bytes");
        drop(first);
        assert!(second._path.0.exists());
        drop(second);
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn existing_files_are_neither_overwritten_nor_removed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("occupied");
        fs::write(&path, b"keep").unwrap();
        let error = Spool::create(path.clone()).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(path).unwrap(), b"keep");
    }

    #[test]
    fn invalid_directories_return_io_errors() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");
        let error = Spool::new(&missing).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        let file = directory.path().join("file");
        fs::write(&file, []).unwrap();
        assert!(Spool::new(&file).is_err());
    }

    #[test]
    fn cleanup_failure_does_not_panic() {
        let directory = tempfile::tempdir().unwrap();
        let spool = Spool::new(directory.path()).unwrap();
        // Move the cleanup path away to exercise a missing path on every platform.
        let moved = directory.path().join("moved");
        fs::rename(&spool._path.0, &moved).unwrap();
        drop(spool);
        fs::remove_file(moved).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn spools_are_owner_only_and_do_not_follow_existing_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let spool = Spool::new(directory.path()).unwrap();
        let permissions = spool.file.metadata().unwrap().permissions().mode();
        assert_eq!(permissions & 0o077, 0);
        let link = directory.path().join("link");
        let target = directory.path().join("target");
        symlink(&target, &link).unwrap();
        let error = Spool::create(link.clone()).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(link.is_symlink());
        assert!(!target.exists());
    }
}
