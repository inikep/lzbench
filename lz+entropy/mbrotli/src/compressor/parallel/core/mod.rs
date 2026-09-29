//! Checked planning, exclusive workers, completion slots, and ordered assembly.
mod artifact;
mod batch;
pub(in crate::compressor::parallel) mod source;
mod spool;
mod task;
use super::*;
use crate::compressor::core::fragment::FragmentEncoder;
use crate::compressor::{Compressor as SerialCompressor, WindowEncoding};
pub(in crate::compressor::parallel) use batch::Batch;
pub(in crate::compressor::parallel) use task::Task;

pub(in crate::compressor::parallel) const fn staging_memory_bound(
    payload: u64,
    segments: u64,
) -> Result<u64, ParallelEncodeError> {
    let Some(descriptors) = segments.checked_mul(size_of::<artifact::Descriptor>() as u64) else {
        return Err(ParallelEncodeError::SizeOverflow);
    };
    match payload.checked_add(descriptors) {
        Some(bytes) => Ok(bytes),
        None => Err(ParallelEncodeError::SizeOverflow),
    }
}

#[derive(Clone)]
pub(in crate::compressor::parallel) enum Input<'a> {
    Slice(&'a [u8]),
    Source(Arc<dyn RandomAccessSource>),
}
impl Input<'_> {
    fn len(&self) -> Result<u64, ParallelEncodeError> {
        match self {
            Self::Slice(s) => Ok(s.len() as u64),
            Self::Source(s) => s.len().map_err(ParallelEncodeError::SourceMetadata),
        }
    }
    fn identity(&self) -> Option<SourceIdentity> {
        match self {
            Self::Slice(_) => None,
            Self::Source(s) => s.identity(),
        }
    }
}

pub(in crate::compressor::parallel) struct Worker {
    fragment: FragmentEncoder,
    serial: SerialCompressor,
    input: Vec<u8>,
}
impl Worker {
    fn new(config: EncoderConfig, backend: Backend) -> Result<Self, ParallelConfigError> {
        Ok(Self {
            fragment: FragmentEncoder::new(config, backend),
            serial: SerialCompressor::builder(config)
                .with_backend(backend)
                .build()?,
            input: Vec::new(),
        })
    }
    fn retained_bytes(&self) -> usize {
        self.fragment.retained_bytes() + self.serial.retained_bytes() + self.input.capacity()
    }
}
pub(in crate::compressor::parallel) struct Compressor {
    pub(in crate::compressor::parallel) encoder: EncoderConfig,
    pub(in crate::compressor::parallel) parallel: ParallelConfig,
    backend: Backend,
    pub(in crate::compressor::parallel) workers: Vec<Worker>,
}
impl Compressor {
    pub(in crate::compressor::parallel) fn new(
        encoder: EncoderConfig,
        parallel: ParallelConfig,
        backend: Backend,
    ) -> Result<Self, ParallelConfigError> {
        let mut compressor = Self {
            encoder,
            parallel,
            backend,
            workers: Vec::new(),
        };
        compressor.reconfigure(encoder)?;
        Ok(compressor)
    }
    pub(in crate::compressor::parallel) fn reconfigure(
        &mut self,
        encoder: EncoderConfig,
    ) -> Result<(), ParallelConfigError> {
        encoder.validate()?;
        if encoder.window().encoding() != WindowEncoding::Standard {
            return Err(ParallelConfigError::UnsupportedParallelWindow);
        }
        if self.encoder != encoder {
            self.workers = Vec::new();
            self.encoder = encoder;
        }
        Ok(())
    }
    pub(in crate::compressor::parallel) fn retained_bytes(&self) -> usize {
        self.workers
            .iter()
            .map(Worker::retained_bytes)
            .sum::<usize>()
            + self.workers.capacity() * size_of::<Worker>()
    }
    pub(in crate::compressor::parallel) fn trim(&mut self, policy: ParallelRetentionPolicy) {
        match policy {
            ParallelRetentionPolicy::ReleaseAll => self.workers = Vec::new(),
            ParallelRetentionPolicy::Aggressive => (),
            ParallelRetentionPolicy::CurrentPlan => {
                self.workers.truncate(self.parallel.max_retained_workers);
                self.workers.shrink_to_fit();
            }
            ParallelRetentionPolicy::Bounded { max_bytes } => {
                self.workers.truncate(self.parallel.max_retained_workers);
                while self.retained_bytes() > max_bytes && !self.workers.is_empty() {
                    self.workers.pop();
                }
                self.workers.shrink_to_fit();
            }
        }
    }
    pub(in crate::compressor::parallel) fn plan(
        &self,
        len: u64,
        config: &BatchConfig,
    ) -> Result<Plan, ParallelEncodeError> {
        let z = self.parallel.segment_size.get() as u64;
        let segments = len / z + u64::from(!len.is_multiple_of(z));
        let tasks = usize::try_from(segments.min(config.task_count.get() as u64))
            .map_err(|_| ParallelEncodeError::SizeOverflow)?;
        let serial = len < self.parallel.minimum_parallel_size && len <= z || len == 0;
        // Twice the raw bytes plus 1 KiB per part covers prefix, block headers,
        // flush alignment and the serial encoder's uncompressed fallback bound.
        let staged = len
            .checked_mul(2)
            .and_then(|n| {
                segments
                    .max(1)
                    .checked_mul(1024)
                    .and_then(|s| n.checked_add(s))
            })
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let descriptors = segments
            .checked_mul(size_of::<artifact::Descriptor>() as u64)
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let segment = self.parallel.segment_size.get();
        // Deliberately conservative, including the q9 bucket payload and HQ
        // candidate/DP arenas. These are ceilings, not expected resident usage.
        let (base, scale) = match self.encoder.quality().get() {
            0..=1 => (4 << 20, 16),
            2..=4 => (64 << 20, 128),
            _ => (256 << 20, 256),
        };
        let per_worker = segment
            .checked_mul(scale)
            .and_then(|n| n.checked_add(base))
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let metadata =
            usize::try_from(descriptors).map_err(|_| ParallelEncodeError::SizeOverflow)?;
        let active = per_worker
            .checked_mul(tasks)
            .and_then(|n| n.checked_add(metadata))
            .and_then(|n| tasks.checked_mul(4096).and_then(|t| n.checked_add(t)))
            .and_then(|n| n.checked_add(128 << 10))
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let staging_memory = match &config.staging {
            Staging::Memory(m) => {
                let bytes =
                    usize::try_from(staged).map_err(|_| ParallelEncodeError::SizeOverflow)?;
                let required = usize::try_from(staging_memory_bound(staged, segments)?)
                    .map_err(|_| ParallelEncodeError::SizeOverflow)?;
                if required > m.max_total_bytes {
                    return Err(ParallelEncodeError::MemoryStagingLimit);
                }
                bytes
            }
            Staging::Directory(_) => 0,
        };
        let aggregate = active
            .checked_add(staging_memory)
            .and_then(|n| n.checked_add(self.retained_bytes()))
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        if self
            .parallel
            .aggregate_memory_limit
            .is_some_and(|limit| aggregate > limit)
        {
            return Err(ParallelEncodeError::WorkerMemoryLimit);
        }
        Ok(Plan {
            tasks,
            serial,
            segment_size: segment,
            estimate: ParallelSizeEstimate {
                input_bytes: len,
                segment_count: segments,
                maximum_staged_bytes: staged,
                maximum_final_bytes: staged
                    .checked_add(1)
                    .ok_or(ParallelEncodeError::SizeOverflow)?,
                estimated_active_workspace_bytes: active,
            },
        })
    }
}
pub(in crate::compressor::parallel) struct Plan {
    pub(in crate::compressor::parallel) tasks: usize,
    serial: bool,
    segment_size: usize,
    pub(in crate::compressor::parallel) estimate: ParallelSizeEstimate,
}
impl Plan {
    fn source_range(&self, segment: u64) -> Result<Range<u64>, ParallelEncodeError> {
        let start = segment
            .checked_mul(self.segment_size as u64)
            .ok_or(ParallelEncodeError::SizeOverflow)?;
        let end = start
            .saturating_add(self.segment_size as u64)
            .min(self.estimate.input_bytes);
        if start >= end {
            return Err(ParallelEncodeError::FragmentInvariant(
                "invalid source range",
            ));
        }
        Ok(start..end)
    }
}
