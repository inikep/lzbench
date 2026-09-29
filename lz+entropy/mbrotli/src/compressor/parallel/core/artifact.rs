//! Single-owner private spools and exact-capacity memory artifacts.
use super::spool::Spool;
use super::*;
use std::io::{self, Read, Seek, SeekFrom, Write};

pub(in crate::compressor::parallel) struct Descriptor {
    pub(in crate::compressor::parallel) segment: u64,
    pub(in crate::compressor::parallel) source: Range<u64>,
    pub(in crate::compressor::parallel) offset: u64,
    pub(in crate::compressor::parallel) len: u64,
}
pub(in crate::compressor::parallel) enum Storage {
    Memory(Vec<u8>),
    File(Spool),
}
pub(in crate::compressor::parallel) struct Artifact {
    pub(in crate::compressor::parallel) storage: Storage,
    pub(in crate::compressor::parallel) descriptors: Vec<Descriptor>,
    pub(in crate::compressor::parallel) len: u64,
    bound: u64,
}
impl Artifact {
    pub(in crate::compressor::parallel) fn new(
        staging: &Staging,
        count: usize,
        bound: u64,
    ) -> io::Result<Self> {
        let storage = match staging {
            Staging::Memory(_) => Storage::Memory(Vec::new()),
            Staging::Directory(d) => Storage::File(Spool::new(&d.directory)?),
        };
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(count)
            .map_err(io::Error::other)?;
        Ok(Self {
            storage,
            descriptors,
            len: 0,
            bound,
        })
    }
    pub(in crate::compressor::parallel) fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        let len = self
            .len
            .checked_add(bytes.len() as u64)
            .filter(|&n| n <= self.bound)
            .ok_or_else(|| io::Error::other("fragment exceeded staging bound"))?;
        match &mut self.storage {
            Storage::Memory(v) => {
                v.try_reserve_exact(bytes.len()).map_err(io::Error::other)?;
                v.extend_from_slice(bytes);
            }
            Storage::File(f) => f.file.write_all(bytes)?,
        }
        self.len = len;
        Ok(())
    }
    pub(in crate::compressor::parallel) fn validate_len(&self) -> io::Result<bool> {
        let actual = match &self.storage {
            Storage::Memory(v) => v.len() as u64,
            Storage::File(f) => f.file.metadata()?.len(),
        };
        Ok(self.len == actual)
    }
    pub(in crate::compressor::parallel) fn copy<W: Write>(
        &mut self,
        writer: &mut W,
        written: &mut u64,
        scratch: &mut [u8],
    ) -> io::Result<()> {
        match &mut self.storage {
            Storage::Memory(v) => write_counted(writer, v, written),
            Storage::File(f) => {
                f.file.seek(SeekFrom::Start(0))?;
                let mut remaining = self.len;
                while remaining != 0 {
                    let take = remaining.min(scratch.len() as u64) as usize;
                    f.file.read_exact(&mut scratch[..take])?;
                    write_counted(writer, &scratch[..take], written)?;
                    remaining -= take as u64;
                }
                Ok(())
            }
        }
    }
}
pub(in crate::compressor::parallel) fn write_counted<W: Write>(
    writer: &mut W,
    mut bytes: &[u8],
    written: &mut u64,
) -> io::Result<()> {
    while !bytes.is_empty() {
        match writer.write(bytes) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) if n <= bytes.len() => {
                *written = written
                    .checked_add(n as u64)
                    .ok_or_else(|| io::Error::other("output length overflow"))?;
                bytes = &bytes[n..];
            }
            Ok(_) => return Err(io::Error::other("writer returned an invalid byte count")),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_allocation_failure_removes_the_created_spool() {
        let directory = tempfile::tempdir().unwrap();
        let staging = Staging::Directory(DirectoryStaging::from(directory.path().to_path_buf()));
        assert!(Artifact::new(&staging, usize::MAX, 0).is_err());
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn staging_bound_and_count_overflow_return_errors() {
        let staging = Staging::Memory(MemoryStaging::from(8));
        let mut artifact = Artifact::new(&staging, 1, 2).unwrap();
        artifact.append(b"ab").unwrap();
        assert!(artifact.append(b"c").is_err());
        assert_eq!(artifact.len, 2);
        let mut count = u64::MAX;
        assert!(write_counted(&mut Vec::new(), b"a", &mut count).is_err());
    }
}
