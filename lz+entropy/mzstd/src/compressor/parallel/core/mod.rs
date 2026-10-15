//! Segment jobs and frame assembly of parallel compression.
//!
//! A frame is cut into fixed segments. Each segment is compressed by its own
//! engine into non-final blocks: the engine first runs over up to an
//! overlap's worth of the preceding input, whose blocks are discarded, so the
//! segment's matches may reach back into it; then its entropy tables and
//! repeat offsets are reset, so the first kept block refers to nothing the
//! decoder has not seen. Assembly writes the frame header, the segments'
//! blocks in order, an empty last block and the checksum.
//!
//! A segment's bytes depend only on the parameters, the overlap and the
//! segment itself, so the frame is the same however many tasks produced it
//! and in whatever order they ran.
//!
//! [`stream`] is the incremental form: input arrives in chunks, full
//! segments become queued tasks, and finished parts are delivered in order.

mod stream;

use alloc::vec;
use alloc::vec::Vec;
use std::sync::{Mutex, MutexGuard, PoisonError};

pub(crate) use stream::{QueueHandle, Stream, StreamTask};

use crate::compressor::core::cctx::{AppliedParams, Engine, RequestedParams, compress_bound};
use crate::compressor::core::error::CoreError;
use crate::compressor::core::params::{CONTENT_SIZE_UNKNOWN, Strategy};
use crate::compressor::core::window::{Addr, Segment, Segments};
use crate::compressor::error::EncodeError;
use crate::shared::format::{FRAME_HEADER_SIZE_MAX, MAGIC_NUMBER, WINDOW_LOG_ABSOLUTE_MIN};
use crate::shared::xxhash::Xxh64;

/// Locks `mutex`, ignoring poisoning: only bookkeeping runs under these
/// locks, never codec work, so a panic elsewhere cannot leave them torn.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Engines shared by the tasks of one compressor.
#[derive(Debug, Default)]
pub(crate) struct EnginePool {
    /// Idle engines, held inline: an engine's large state (tables, block
    /// states) is already on the heap, so moving one in or out is cheap.
    engines: Mutex<Vec<Engine>>,
}

impl EnginePool {
    /// An idle engine, or a new one on the detected backend.
    pub(crate) fn take(&self) -> Engine {
        lock(&self.engines).pop().unwrap_or_else(Engine::new)
    }

    /// Returns an engine whose task completed.
    pub(crate) fn give(&self, engine: Engine) {
        lock(&self.engines).push(engine);
    }

    /// Bytes retained by idle engines.
    pub(crate) fn retained_bytes(&self) -> usize {
        lock(&self.engines)
            .iter()
            .map(|engine| engine.retained_bytes())
            .sum()
    }

    /// Drops every idle engine.
    pub(crate) fn release(&self) {
        lock(&self.engines).clear();
    }
}

/// Compresses `input` as one serial frame, the bytes
/// [`Compressor::compress`](crate::Compressor::compress) writes.
///
/// # Errors
///
/// None for valid input: the destination holds any frame, so a failure is a
/// core bug reported rather than a panic.
pub(crate) fn compress_serial(
    engine: &mut Engine,
    requested: &RequestedParams,
    input: &[u8],
) -> Result<Vec<u8>, EncodeError> {
    let bound = compress_bound(input.len()).unwrap_or(0);
    let mut out = vec![0u8; bound];
    match crate::compressor::core::compress_once_into(engine, requested, None, input, 1, &mut out) {
        Ok(size) => {
            out.truncate(size);
            Ok(out)
        }
        Err(error) => Err(EncodeError::from_core(
            error,
            bound,
            Some(input.len() as u64),
        )),
    }
}

/// Parameters every segment of one frame shares.
#[derive(Copy, Clone, Debug)]
pub(crate) struct FramePlan {
    pub(crate) applied: AppliedParams,
    /// Declared content size, when known.
    pub(crate) content_size: Option<u64>,
}

impl FramePlan {
    /// Resolves the frame's parameters for `content_size` (unknown when
    /// `None`).
    pub(crate) fn new(requested: &RequestedParams, content_size: Option<u64>) -> Self {
        let mut applied = requested.resolve(content_size.unwrap_or(CONTENT_SIZE_UNKNOWN));
        if content_size.is_none() {
            applied.fparams.content_size = false;
        }
        Self {
            applied,
            content_size,
        }
    }

    /// History a segment may reach into: an eighth of the window, or the
    /// whole window for the ultra strategies, whose larger windows pay for
    /// themselves in ratio.
    pub(crate) fn overlap(&self) -> usize {
        let window = 1usize << self.applied.cparams.window_log;
        if self.applied.cparams.strategy >= Strategy::BtUltra {
            window
        } else {
            window >> 3
        }
    }

    /// Whether segments compress their overlap and discard the result
    /// instead of loading it as a prefix, where that measured faster
    /// (`benches/parallel.rs` `parallel-corpora`). Lazy and lazy2 skip ahead
    /// through input that does not match while compressing, but a load
    /// inserts every position into their row tables (random input took up to
    /// three times as long); btlazy2 sorts its tree lazily while compressing,
    /// but a load sorts all of it (text took 9% longer).
    const fn compresses_overlap(&self) -> bool {
        matches!(
            self.applied.cparams.strategy,
            Strategy::Lazy | Strategy::Lazy2 | Strategy::BtLazy2
        )
    }

    /// The frame header (`ZSTD_writeFrameHeader` without dictionaries).
    pub(crate) fn header(&self) -> Vec<u8> {
        let fparams = self.applied.fparams;
        let window_log = self.applied.cparams.window_log;
        let window_size = 1u64 << window_log;
        let pledged = self.content_size.unwrap_or(0);
        let content_size = fparams.content_size && self.content_size.is_some();
        let single_segment = content_size && window_size >= pledged;
        let fcs_code: u8 = if content_size {
            u8::from(pledged >= 256)
                + u8::from(pledged >= 65536 + 256)
                + u8::from(pledged >= 0xFFFF_FFFF)
        } else {
            0
        };
        let descriptor =
            (u8::from(fparams.checksum) << 2) + (u8::from(single_segment) << 5) + (fcs_code << 6);
        let mut out = Vec::with_capacity(18);
        out.extend_from_slice(&MAGIC_NUMBER.to_le_bytes());
        out.push(descriptor);
        if !single_segment {
            out.push(((window_log - WINDOW_LOG_ABSOLUTE_MIN) << 3) as u8);
        }
        match fcs_code {
            0 if single_segment => out.push(pledged as u8),
            0 => {}
            1 => out.extend_from_slice(&((pledged - 256) as u16).to_le_bytes()),
            2 => out.extend_from_slice(&(pledged as u32).to_le_bytes()),
            _ => out.extend_from_slice(&pledged.to_le_bytes()),
        }
        out
    }

    /// The frame's end: an empty last block and, when enabled, the
    /// checksum of `hasher`'s content.
    pub(crate) fn epilogue(&self, hasher: &Xxh64) -> Vec<u8> {
        let mut out = vec![1, 0, 0];
        if self.applied.fparams.checksum {
            out.extend_from_slice(&(hasher.digest() as u32).to_le_bytes());
        }
        out
    }
}

/// Compresses one segment and appends its blocks, none of them last, to
/// `out`.
///
/// `data` holds the overlap followed by the segment; `overlap` is the
/// overlap's length. The overlap is loaded as a prefix: it fills the window
/// and tables without being compressed, as ZSTDMT primes its jobs; for the
/// strategies [`FramePlan::compresses_overlap`] names it is compressed into
/// `out`'s spare room and dropped instead. The
/// segment's first block refers to nothing the decoder has not seen: its
/// entropy tables start empty and its repeat offsets match nothing
/// (`ZSTD_invalidateRepCodes`), while matches into the overlap stay valid
/// because the decoder holds it as the previous segment's output.
///
/// # Errors
///
/// Whatever the engine reports; with room for `compressBound` bytes it
/// reports nothing for valid input. `out` keeps its previous content then.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn compress_segment(
    engine: &mut Engine,
    plan: &FramePlan,
    data: &[u8],
    overlap: usize,
    out: &mut Vec<u8>,
) -> Result<(), CoreError> {
    let mut applied = plan.applied;
    applied.fparams.checksum = false;
    engine.begin(applied, CONTENT_SIZE_UNKNOWN);
    let segments = Segments::input(data, 1, None);
    let src = Addr::new(Segment::Input(1), overlap as i64);
    let start = out.len();
    if plan.compresses_overlap() {
        // Writes the frame header and the overlap's blocks, then drops them.
        out.resize(start + compress_bound(overlap).unwrap_or(0) + 32, 0);
        let result = engine.compress_continue(
            &segments,
            &mut out[start..],
            Addr::new(Segment::Input(1), 0),
            overlap,
            false,
        );
        out.truncate(start);
        result?;
        engine.prev.entropy.reset();
    } else {
        engine.load_prefix(&segments, Addr::new(Segment::Input(1), 0), overlap);
        // Writes the frame header, which is discarded.
        let mut header = [0u8; FRAME_HEADER_SIZE_MAX];
        engine.compress_continue(&segments, &mut header, src, 0, false)?;
    }
    engine.prev.rep = [0; 3];
    let segment_len = data.len() - overlap;
    out.resize(start + compress_bound(segment_len).unwrap_or(0) + 32, 0);
    let result = engine.compress_continue(&segments, &mut out[start..], src, segment_len, false);
    out.truncate(start + *result.as_ref().unwrap_or(&0));
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zstd_ffi::corpus;

    fn frame(plan: &FramePlan, input: &[u8], segment: usize) -> Vec<u8> {
        let mut out = plan.header();
        let mut engine = Engine::new();
        let mut start = 0;
        while start < input.len() {
            let end = (start + segment).min(input.len());
            let from = start.saturating_sub(plan.overlap());
            compress_segment(&mut engine, plan, &input[from..end], start - from, &mut out).unwrap();
            start = end;
        }
        let mut hasher = Xxh64::new(0);
        hasher.update(input);
        out.extend(plan.epilogue(&hasher));
        out
    }

    #[test]
    fn assembled_frames_decode_with_the_reference() {
        let input = corpus::mixed(1_200_000, 9);
        for level in [-5, 1, 3, 7, 13, 17, 19] {
            for checksum in [false, true] {
                for size in [None, Some(input.len() as u64)] {
                    let mut requested = RequestedParams {
                        compression_level: level,
                        ..RequestedParams::default()
                    };
                    requested.fparams.checksum = checksum;
                    let plan = FramePlan::new(&requested, size);
                    let compressed = frame(&plan, &input, 300_000);
                    let decoded =
                        zstd_ffi::oracle::decompress_stream(&compressed, 1 << 16, Some(31))
                            .expect("decodes");
                    assert_eq!(decoded, input, "level {level}");
                }
            }
        }
    }

    /// `Engine::load_prefix` is the reference's raw-prefix load: a frame
    /// whose prefix sits right before its input in memory, loaded with
    /// `forceWindow`, is byte-identical to `ZSTD_CCtx_refPrefix` with
    /// `ZSTD_c_forceMaxWindow`, at every level.
    #[test]
    fn prefix_loads_match_the_reference() {
        use crate::compressor::core::cctx::compress_bound;
        use crate::compressor::core::params::{CParamMode, ParamSwitch};
        let whole = [
            corpus::text(150_000, 21),
            corpus::mixed(250_000, 22),
            corpus::text(40_000, 23),
        ]
        .concat();
        for level in -5..=22 {
            for (prefix_len, ldm) in [(150_000, false), (7, false), (390_000, true)] {
                let mut requested = RequestedParams {
                    compression_level: level,
                    ..RequestedParams::default()
                };
                if ldm {
                    requested.ldm.enable = ParamSwitch::Enable;
                    requested.cparams.window_log = 20;
                }
                let input = &whole[prefix_len..];
                let pledged = input.len() as u64;
                let mut engine = Engine::new();
                engine.begin(
                    requested.resolve_for(pledged, prefix_len as u64, CParamMode::NoAttachDict),
                    pledged,
                );
                let segments = Segments::input(&whole, 3, None);
                engine.load_prefix(&segments, Addr::new(Segment::Input(3), 0), prefix_len);
                let mut ours = vec![0u8; compress_bound(input.len()).unwrap()];
                let size = engine
                    .compress_end_to(
                        &segments,
                        &mut crate::compressor::core::output::SliceOutput(&mut ours),
                        Addr::new(Segment::Input(3), prefix_len as i64),
                        input.len(),
                    )
                    .expect("compresses");
                let params = crate::compressor::core::advanced_tests::oracle_params(&requested)
                    .with(zstd_ffi::ZSTD_c_forceMaxWindow, 1);
                let theirs = zstd_ffi::oracle::compress2_with_prefix(&whole, prefix_len, &params)
                    .expect("oracle");
                assert!(
                    ours[..size] == theirs[..],
                    "level {level} prefix {prefix_len} ldm {ldm}: {size} vs {}",
                    theirs.len()
                );
            }
        }
    }

    #[test]
    fn the_pool_reuses_and_releases_engines() {
        let pool = EnginePool::default();
        let mut engine = pool.take();
        let frame = compress_serial(&mut engine, &RequestedParams::default(), b"pooled").unwrap();
        assert_eq!(
            zstd_ffi::oracle::decompress(&frame, 6).unwrap(),
            b"pooled".to_vec()
        );
        pool.give(engine);
        assert!(pool.retained_bytes() > 0);
        pool.release();
        assert_eq!(pool.retained_bytes(), 0);
    }
}
