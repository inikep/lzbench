//! Single-owner private spools and bounded memory artifacts.
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
                // Doubling, capped by the staging bound the plan accounted
                // for: an exact reservation per segment reallocated on every
                // append. A first append is exact, so a single segment
                // carries no spare capacity into `into_segment`.
                if v.capacity() - v.len() < bytes.len() {
                    let bound = usize::try_from(self.bound).unwrap_or(usize::MAX);
                    let needed = v.len().saturating_add(bytes.len());
                    let target = v.len().saturating_mul(2).min(bound).max(needed);
                    v.try_reserve_exact(target - v.len())
                        .map_err(io::Error::other)?;
                }
                v.extend_from_slice(bytes);
            }
            Storage::File(f) => f.file.write_all(bytes)?,
        }
        self.len = len;
        Ok(())
    }
    /// Returns the payload of a memory artifact holding exactly one segment.
    pub(in crate::compressor::parallel) fn into_segment(self) -> Option<Vec<u8>> {
        match (self.storage, self.descriptors.as_slice()) {
            (Storage::Memory(bytes), [descriptor])
                if descriptor.offset == 0
                    && descriptor.len != 0
                    && descriptor.len == self.len
                    && bytes.len() as u64 == self.len =>
            {
                Some(bytes)
            }
            _ => None,
        }
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
        scratch: &mut Vec<u8>,
    ) -> io::Result<()> {
        match &mut self.storage {
            Storage::Memory(v) => write_counted(writer, v, written),
            Storage::File(f) => {
                // Only spooled artifacts need a bounce buffer; memory staging
                // never pays for zeroing one.
                if scratch.is_empty() {
                    scratch.resize(SPOOL_CHUNK, 0);
                }
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
/// Bytes read from a spooled artifact per copy step.
const SPOOL_CHUNK: usize = 128 << 10;

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

    fn memory(bytes: &[u8], segments: usize) -> Artifact {
        let staging = Staging::Memory(MemoryStaging::from(usize::MAX));
        let mut artifact = Artifact::new(&staging, segments, 1 << 20).unwrap();
        for segment in 0..segments {
            let offset = artifact.len;
            artifact.append(bytes).unwrap();
            artifact.descriptors.push(Descriptor {
                segment: segment as u64,
                source: 0..1,
                offset,
                len: bytes.len() as u64,
            });
        }
        artifact
    }

    #[test]
    fn memory_staging_doubles_within_its_bound() {
        let staging = Staging::Memory(MemoryStaging::from(usize::MAX));
        let mut artifact = Artifact::new(&staging, 10, 1000).unwrap();
        let mut capacities = Vec::new();
        for segment in 0..10u8 {
            artifact.append(&[segment; 100]).unwrap();
            let Storage::Memory(v) = &artifact.storage else {
                panic!("memory storage")
            };
            assert!(v.capacity() <= 1000);
            if capacities.last() != Some(&v.capacity()) {
                capacities.push(v.capacity());
            }
        }
        // The first append is exact; later growth doubles until the bound.
        assert_eq!(capacities, [100, 200, 400, 800, 1000]);
        let Storage::Memory(v) = &artifact.storage else {
            panic!("memory storage")
        };
        assert!(v.chunks(100).enumerate().all(|(i, c)| c == [i as u8; 100]));
        assert!(artifact.append(b"x").is_err());
    }

    #[test]
    fn only_a_single_nonempty_memory_segment_is_extracted() {
        assert_eq!(memory(b"part", 1).into_segment().unwrap(), b"part");
        assert!(memory(b"part", 2).into_segment().is_none());
        assert!(memory(b"", 1).into_segment().is_none());
        let mut grown = memory(b"part", 1);
        grown.len += 1;
        assert!(grown.into_segment().is_none());
    }
}
