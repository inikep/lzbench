//! The streaming quality 3, 4 and 5 encoder.
//!
//! Ports `EncodeData`, `WriteMetaBlockInternal`, `ShouldCompress` and
//! `CopyInputToRingBuffer` from `c/enc/encode.c` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! A retained kernel selected by `core::dispatch` hands a SIMD token to the
//! match scan, and everything below it is monomorphised on that token. The matcher,
//! the block sizes and the distance alphabet were all resolved before the
//! encoder was built, so nothing about the machine can reach a decision that
//! shows up in the output.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::compressor::core::dispatch::{self, GreedyInput, Kernels};
use fearless_simd::Level;

use super::backward_references::ReferenceState;
use super::context_model::decide_over_literal_context_modeling;
use super::hashers::{DistanceCache, MatchFinder, NUM_REMEMBERED_DISTANCES, Sweep};
use super::metablock::build_meta_block_greedy_into;
use super::params::{GreedyParams, MAX_NUM_DELAYED_SYMBOLS};
use crate::compressor::core::rfc9841::context::SharedContextInner;
use crate::compressor::{BrotliCompressError, BrotliResult, CompressParams};
use crate::shared::bits::{BYTE_PADDING_SLACK, BitWriter, inject_byte_padding};
use crate::shared::bitstream::{MetaBlockWriter, store_uncompressed_meta_block};
use crate::shared::command::Command;
use crate::shared::command::CommandExtension;
use crate::shared::constants::{OUTPUT_RESERVE_CONST, OUTPUT_SLACK};
use crate::shared::format::ContextMode;
use crate::shared::histogram::{HistogramLiteral, bits_entropy};
use crate::shared::metablock::{MetaBlockSplit, optimize_histograms};
use crate::shared::ringbuffer::{BlockSpan, Window};
use crate::shared::ringbuffer::{RingBuffer, wrap_position};

/// Stride the compressibility check samples literals at (`kSampleRate`).
const SAMPLE_RATE: u32 = 13;

/// Bits per byte above which sampled data is stored uncompressed.
const MIN_ENTROPY: f64 = 7.92;

/// Fraction of a block that has to be literals before it is even sampled.
const LITERAL_FRACTION: f64 = 0.99;

/// How a reset leaves the match finder clean for the next stream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Cleanup {
    /// Replay the partial sweep of an input this long: it clears exactly the
    /// slots the stream could have dirtied.
    Replay(usize),
    /// The stream dirtied the table in places no cheap sweep could find; the
    /// next prepare pays for the wipe.
    Wipe,
    /// The finder empties itself on every prepare; nothing to do.
    Nothing,
}

/// Streaming encoder for qualities three to five.
pub(crate) struct GreedyEncoder {
    kernels: Box<dyn Kernels>,
    params: GreedyParams,
    ringbuffer: RingBuffer,
    matcher: MatchFinder,
    is_prepared: bool,
    /// Whether the match finder holds entries from a stream already written.
    ///
    /// A fresh table lets [`MatchFinder::prepare`] clear only the slots a
    /// short one-shot input reaches. That shortcut is wrong on a table another
    /// stream has stored into, so a reused encoder that could not clean up
    /// after itself asks for the full sweep instead.
    matcher_dirty: bool,
    /// What the last prepare's sweep left for a reset to do.
    cleanup: Cleanup,
    input_pos: u64,
    last_processed_pos: u64,
    last_flush_pos: u64,
    commands: Vec<Command>,
    references: ReferenceState,
    saved_dist_cache: [i32; NUM_REMEMBERED_DISTANCES],
    prev_byte: u8,
    prev_byte2: u8,
    last_bytes: u16,
    last_bytes_bits: u32,
    is_last_block_emitted: bool,
    finished: bool,
    storage: Vec<u8>,
    writer: MetaBlockWriter,
    metablock: MetaBlockSplit,
    output_len: usize,
}

impl GreedyEncoder {
    /// Installs fragment-only kernels when constructing an independent worker.
    #[cfg(not(feature = "no_std"))]
    pub(crate) fn select_fragment_kernels(&mut self, level: Level) {
        self.kernels = dispatch::select_independent(level);
    }

    /// Resets the header and seeds only this segment's literal context/history.
    #[cfg(not(feature = "no_std"))]
    pub(crate) fn begin_fragment(&mut self, prefix: &[u8]) -> BrotliResult<()> {
        self.last_bytes = 0;
        self.last_bytes_bits = 0;
        self.references.dictionary = crate::shared::dictionary::DictionaryStats::DISABLED;
        // Two bytes cannot make a copy. Flush seeds the ring and prior-byte
        // context; the shared fragment writer owns the actual prefix output.
        self.flush_block(prefix, None)?;
        Ok(())
    }

    /// Confirms the flush left no pending partial byte.
    #[cfg(not(feature = "no_std"))]
    pub(crate) const fn fragment_aligned(&self) -> bool {
        self.last_bytes_bits == 0
    }

    /// Creates an encoder for `params`, expecting `size_hint` bytes in total.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] when the quality is
    /// outside the range this encoder implements.
    pub(crate) fn new(
        level: Level,
        params: &CompressParams,
        size_hint: usize,
    ) -> BrotliResult<Self> {
        let resolved = GreedyParams::new(params, size_hint)?;
        let (last_bytes, last_bytes_bits) = resolved.window.header();
        #[cfg(feature = "experimental")]
        let (last_bytes, last_bytes_bits) = if resolved.stream_offset != 0 {
            (0, 0)
        } else {
            (last_bytes, last_bytes_bits)
        };
        let mut ringbuffer = RingBuffer::new(resolved.rb_bits(), resolved.lgblock);
        ringbuffer.expect_input(size_hint);
        let references = ReferenceState::default();
        #[cfg(feature = "experimental")]
        let references = {
            let mut references = references;
            if resolved.stream_offset != 0 {
                references.dist_cache[..4].fill(-16);
            }
            references
        };
        Ok(Self {
            kernels: dispatch::select(level),
            params: resolved,
            ringbuffer,
            matcher: MatchFinder::for_input(resolved.hasher, size_hint),
            is_prepared: false,
            matcher_dirty: false,
            cleanup: Cleanup::Nothing,
            input_pos: 0,
            last_processed_pos: 0,
            last_flush_pos: 0,
            commands: Vec::new(),
            saved_dist_cache: remembered(&references.dist_cache),
            references,
            prev_byte: 0,
            prev_byte2: 0,
            last_bytes,
            last_bytes_bits,
            is_last_block_emitted: false,
            finished: false,
            storage: Vec::new(),
            writer: MetaBlockWriter::default(),
            metablock: MetaBlockSplit::default(),
            output_len: 0,
        })
    }

    /// Returns the largest input this encoder accepts in one call.
    pub(crate) const fn block_size_limit(&self) -> usize {
        self.params.input_block_size()
    }

    /// Returns whether the final meta-block has already been written.
    #[cfg(test)]
    pub(crate) const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Returns the parameters this encoder was resolved for.
    ///
    /// A workspace compares these against what a new call would resolve to:
    /// equal parameters mean an equally shaped encoder, so resetting this one
    /// gives the same stream a fresh one would.
    /// Re-aims the encoder at the stream `fresh` describes.
    ///
    /// Returns `false`, changing nothing, when `fresh` is another shape or
    /// its size hint would have selected another match finder, in which
    /// case the caller has to build a new encoder. Otherwise the size hint
    /// is taken over — it decides literal context modelling and the
    /// matcher's next layout — and the encoder is as good as one built for
    /// `fresh`, once reset.
    pub(crate) fn retarget(&mut self, fresh: GreedyParams) -> bool {
        if !self.params.same_shape(&fresh) {
            return false;
        }
        if !self.matcher.retarget(fresh.hasher, fresh.size_hint) {
            return false;
        }
        self.ringbuffer.expect_input(fresh.size_hint);
        self.params = fresh;
        true
    }

    /// Restores the encoder to the state its constructor left it in.
    ///
    /// Every allocation survives. The window, the match finder and the
    /// per-stream position counters go back to zero; the scratch buffer, the
    /// meta-block writer and the command vector are per-meta-block state that
    /// is already rebuilt on every use, so only their contents are dropped.
    ///
    /// The match finder is cleared in full rather than by
    /// [`MatchFinder::prepare`]'s partial sweep, which is only correct on a
    /// table that was never used.
    pub(crate) fn reset(&mut self) {
        let (last_bytes, last_bytes_bits) = self.params.window.header();
        #[cfg(feature = "experimental")]
        let (last_bytes, last_bytes_bits) = if self.params.stream_offset != 0 {
            (0, 0)
        } else {
            (last_bytes, last_bytes_bits)
        };
        // Clean the match finder before the window it read from is dropped:
        // the sweep hashes the very bytes the previous stream stored, and they
        // are still where that stream left them.
        match ::core::mem::replace(&mut self.cleanup, Cleanup::Nothing) {
            Cleanup::Replay(input_size) => {
                self.matcher
                    .prepare(true, input_size, self.ringbuffer.buffer(), true);
                self.matcher_dirty = false;
            }
            Cleanup::Wipe => self.matcher_dirty = true,
            Cleanup::Nothing => self.matcher_dirty = false,
        }
        self.ringbuffer.reset();
        self.is_prepared = false;
        self.input_pos = 0;
        self.last_processed_pos = 0;
        self.last_flush_pos = 0;
        self.commands.clear();
        self.references = ReferenceState::default();
        #[cfg(feature = "experimental")]
        if self.params.stream_offset != 0 {
            self.references.dist_cache[..4].fill(-16);
        }
        self.saved_dist_cache = remembered(&self.references.dist_cache);
        self.prev_byte = 0;
        self.prev_byte2 = 0;
        self.last_bytes = last_bytes;
        self.last_bytes_bits = last_bytes_bits;
        self.is_last_block_emitted = false;
        self.finished = false;
        self.output_len = 0;
    }

    /// Compresses one input block and returns the bytes it completed.
    ///
    /// A block shorter than [`GreedyEncoder::block_size_limit`] is allowed only
    /// when `is_last` is set. The result may be empty: the encoder buffers
    /// input until a meta-block is worth emitting.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] when the scratch buffer
    /// proved too small, which would indicate a bug in the size bound.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    #[cfg(test)]
    pub(crate) fn encode_block(&mut self, input: &[u8], is_last: bool) -> BrotliResult<&[u8]> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit());

        self.encode_block_with(input, is_last, None)
    }

    /// Compresses one input block, consulting `attached` for matches.
    ///
    /// The attached context is passed per call rather than held, because it is
    /// the caller's and is only borrowed for the length of one compression.
    /// `None` is exactly [`GreedyEncoder::encode_block`].
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] when the scratch buffer
    /// proved too small, which would indicate a bug in the size bound.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn encode_block_with(
        &mut self,
        input: &[u8],
        is_last: bool,
        attached: Option<&SharedContextInner>,
    ) -> BrotliResult<&[u8]> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit());

        self.copy_input_to_ring_buffer(input);
        self.encode_data(is_last, false, attached)?;
        match self.storage.get(..self.output_len) {
            Some(output) => Ok(output),
            None => Err(BrotliCompressError::BufferOverflow),
        }
    }

    /// Compresses `input` and closes the meta-block, leaving the stream open.
    ///
    /// Mirrors `BROTLI_OPERATION_FLUSH`: the buffered input is written out as
    /// a meta-block even when the encoder would rather keep gathering, and the
    /// stream is then realigned to a byte boundary with an empty metadata
    /// block. Everything returned so far therefore decodes to everything fed
    /// in so far, which is what a caller draining into a socket needs.
    ///
    /// Unlike [`GreedyEncoder::encode_block`] this may return an empty slice
    /// only when there was nothing buffered *and* the stream was already
    /// aligned; otherwise it always produces bytes.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] when the scratch buffer
    /// proved too small, which would indicate a bug in the size bound.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn flush_block(
        &mut self,
        input: &[u8],
        attached: Option<&SharedContextInner>,
    ) -> BrotliResult<&[u8]> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit());

        self.copy_input_to_ring_buffer(input);
        self.encode_data(false, true, attached)?;

        // `encode_data` may have returned without touching the scratch buffer
        // — a flush with nothing buffered still has to emit the padding, so
        // the buffer has to be able to hold it either way.
        self.reserve_storage(0)?;
        let padded = match self.storage.get_mut(self.output_len..) {
            Some(tail) if tail.len() >= BYTE_PADDING_SLACK => {
                inject_byte_padding(&mut self.last_bytes, &mut self.last_bytes_bits, tail)
            }
            _ => return Err(BrotliCompressError::BufferOverflow),
        };
        self.output_len += padded;
        match self.storage.get(..self.output_len) {
            Some(output) => Ok(output),
            None => Err(BrotliCompressError::BufferOverflow),
        }
    }

    /// Returns the bytes this encoder keeps allocated between blocks.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.ringbuffer.retained_bytes()
            + size_of_val(&*self.kernels)
            + self.matcher.retained_bytes()
            + self.commands.capacity() * size_of::<Command>()
            + self.storage.capacity()
            + self.writer.retained_bytes()
            + self.metablock.retained_bytes()
    }

    /// Appends `input` to the window (`CopyInputToRingBuffer`).
    fn copy_input_to_ring_buffer(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }
        self.ringbuffer.write(input);
        self.input_pos += input.len() as u64;
        self.ringbuffer.clear_margin();
    }

    /// Marks the input as processed, reporting whether positions wrapped.
    ///
    /// Mirrors `UpdateLastProcessedPos`.
    fn update_last_processed_pos(&mut self) -> bool {
        let wrapped_last = wrap_position(self.last_processed_pos);
        let wrapped_input = wrap_position(self.input_pos);
        self.last_processed_pos = self.input_pos;
        wrapped_input < wrapped_last
    }

    /// Makes sure the scratch buffer can hold a meta-block of `size` bytes.
    fn reserve_storage(&mut self, size: usize) -> BrotliResult<()> {
        let Some(reserve) = size
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_add(OUTPUT_RESERVE_CONST))
            .and_then(|reserve| reserve.checked_add(OUTPUT_SLACK))
        else {
            return Err(BrotliCompressError::BufferOverflow);
        };
        if self.storage.len() < reserve {
            self.storage = vec![0u8; reserve];
        }
        Ok(())
    }

    /// Seeds the scratch buffer with the bits carried from the last meta-block.
    ///
    /// The bit writer resumes inside that byte, so it has to be in place before
    /// anything is written. [`GreedyEncoder::reserve_storage`] always leaves
    /// room for it.
    fn seed_storage_with_the_partial_byte(&mut self) {
        if let Some(head) = self.storage.get_mut(..2) {
            head[0] = self.last_bytes as u8;
            head[1] = (self.last_bytes >> 8) as u8;
        }
    }

    /// Emits the two bits that close a stream that never had any input.
    fn emit_empty_stream(&mut self) -> BrotliResult<()> {
        self.reserve_storage(0)?;
        self.last_bytes |= 3u16 << self.last_bytes_bits;
        self.last_bytes_bits += 2;
        self.seed_storage_with_the_partial_byte();
        self.output_len = ((self.last_bytes_bits + 7) >> 3) as usize;
        self.finished = true;
        Ok(())
    }

    /// Runs the match scan over the unprocessed input.
    ///
    /// The retained kernel passes its already-selected token into the scan.
    fn create_references(&mut self, span: BlockSpan, attached: Option<&SharedContextInner>) {
        let Self {
            kernels,
            params,
            ringbuffer,
            matcher,
            references,
            commands,
            ..
        } = self;
        let window = Window {
            data: ringbuffer.buffer(),
            mask: ringbuffer.mask(),
        };
        kernels.greedy(GreedyInput {
            matcher,
            params,
            window,
            span,
            attached,
            references,
            commands,
        });
    }

    /// Processes the accumulated input, emitting a meta-block if one is due.
    ///
    /// Mirrors `EncodeData` for the non-fast qualities.
    fn encode_data(
        &mut self,
        is_last: bool,
        force_flush: bool,
        attached: Option<&SharedContextInner>,
    ) -> BrotliResult<()> {
        self.output_len = 0;
        let delta = self.input_pos - self.last_processed_pos;
        let mut span = BlockSpan {
            position: wrap_position(self.last_processed_pos),
            bytes: delta as u32,
        };

        if delta == 0 {
            if !self.ringbuffer.is_allocated() {
                if is_last {
                    return self.emit_empty_stream();
                }
                return Ok(());
            }
            if !is_last && !force_flush {
                return Ok(());
            }
        }
        if self.is_last_block_emitted {
            return Err(BrotliCompressError::BufferOverflow);
        }
        if is_last {
            self.is_last_block_emitted = true;
        }

        // Theoretical maximum of one command per two bytes, plus room to merge
        // the next block in without reallocating.
        let needed = self.commands.len() + span.bytes as usize / 2 + 1;
        if self.commands.capacity() < needed {
            self.commands
                .reserve(needed + span.bytes as usize / 4 + 16 - self.commands.len());
        }

        self.prepare_matcher(span.position as usize, span.bytes as usize, is_last);

        if !self.commands.is_empty() && self.references.last_insert_len == 0 {
            let Self {
                kernels,
                params,
                ringbuffer,
                references,
                commands,
                last_processed_pos,
                ..
            } = self;
            if let Some(command) = commands.last_mut() {
                kernels.extend(CommandExtension {
                    command,
                    lgwin: params.lgwin,
                    dist: &params.dist,
                    last_distance: references.dist_cache[0],
                    window: Window {
                        data: ringbuffer.buffer(),
                        mask: ringbuffer.mask(),
                    },
                    last_processed_pos: *last_processed_pos,
                    attached,
                    span: &mut span,
                });
            }
        }

        self.create_references(span, attached);

        {
            let max_length = self.params.max_metablock_size();
            let max_literals = max_length / 8;
            let max_commands = max_length / 8;
            let processed_bytes = (self.input_pos - self.last_flush_pos) as usize;
            let next_input_fits_metablock =
                processed_bytes + self.params.input_block_size() <= max_length;
            // Without block splitting there is no point in gathering more than
            // a bounded number of symbols, so a low quality flushes early.
            let should_flush = !self.params.quality.splits_blocks()
                && self.references.num_literals + self.commands.len() >= MAX_NUM_DELAYED_SYMBOLS;
            if !is_last
                && !force_flush
                && !should_flush
                && next_input_fits_metablock
                && self.references.num_literals < max_literals
                && self.commands.len() < max_commands
            {
                // Merge with the next input block instead.
                if self.update_last_processed_pos() {
                    self.is_prepared = false;
                }
                return Ok(());
            }
        }

        if self.references.last_insert_len > 0 {
            self.commands
                .push(Command::insert_only(self.references.last_insert_len));
            self.references.num_literals += self.references.last_insert_len;
            self.references.last_insert_len = 0;
        }

        if !is_last && self.input_pos == self.last_flush_pos {
            return Ok(());
        }
        debug_assert!(self.input_pos >= self.last_flush_pos);
        debug_assert!(self.input_pos - self.last_flush_pos <= 1 << 24);

        let metablock_size = (self.input_pos - self.last_flush_pos) as usize;
        self.reserve_storage(metablock_size)?;
        self.seed_storage_with_the_partial_byte();

        let position = self.write_meta_block(metablock_size, is_last)?;

        let complete = position >> 3;
        self.last_bytes = u16::from(self.storage.get(complete).copied().unwrap_or(0));
        self.last_bytes_bits = (position & 7) as u32;
        self.last_flush_pos = self.input_pos;
        if self.update_last_processed_pos() {
            self.is_prepared = false;
        }
        let mask = self.ringbuffer.mask();
        if self.last_flush_pos > 0 {
            self.prev_byte = self
                .ringbuffer
                .buffer()
                .get(((self.last_flush_pos as u32).wrapping_sub(1) as usize) & mask)
                .copied()
                .unwrap_or(0);
        }
        if self.last_flush_pos > 1 {
            self.prev_byte2 = self
                .ringbuffer
                .buffer()
                .get(((self.last_flush_pos as u32).wrapping_sub(2) as usize) & mask)
                .copied()
                .unwrap_or(0);
        }
        self.commands.clear();
        self.references.num_literals = 0;
        self.saved_dist_cache = remembered(&self.references.dist_cache);
        self.output_len = complete;
        self.finished = is_last;
        Ok(())
    }

    /// Sets the matcher up and hands it the block boundary positions.
    ///
    /// Mirrors `InitOrStitchToPreviousBlock`.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn prepare_matcher(&mut self, position: usize, input_size: usize, is_last: bool) {
        let data = self.ringbuffer.buffer();
        let mask = self.ringbuffer.mask();
        if !self.is_prepared {
            let one_shot = position == 0 && is_last && !self.matcher_dirty;
            self.cleanup =
                match self
                    .matcher
                    .prepare(one_shot, input_size, data, self.matcher_dirty)
                {
                    Sweep::Partial => Cleanup::Replay(input_size),
                    Sweep::Full => Cleanup::Wipe,
                    Sweep::SelfCleaning => Cleanup::Nothing,
                };
            self.matcher_dirty = true;
            self.is_prepared = true;
        }
        self.matcher
            .stitch_to_previous_block(input_size, position, data, mask);
    }

    /// Writes one meta-block, returning the bit position after it.
    ///
    /// Mirrors `WriteMetaBlockInternal`.
    fn write_meta_block(&mut self, bytes: usize, is_last: bool) -> BrotliResult<usize> {
        let wrapped_last_flush_pos = wrap_position(self.last_flush_pos) as usize;
        let Self {
            params,
            ringbuffer,
            commands,
            references,
            saved_dist_cache,
            prev_byte,
            prev_byte2,
            last_bytes_bits,
            storage,
            writer,
            ..
        } = self;
        let data = ringbuffer.buffer();
        let mask = ringbuffer.mask();

        let mut w = BitWriter::new(storage, *last_bytes_bits as usize);
        if bytes == 0 {
            // Write the ISLAST and ISEMPTY bits and stop.
            w.write(2, 3);
            w.align();
            let position = w.position();
            if w.overflowed() {
                return Err(BrotliCompressError::BufferOverflow);
            }
            return Ok(position);
        }

        if !should_compress(
            data,
            mask,
            self.last_flush_pos,
            bytes,
            references.num_literals,
            commands.len(),
        ) {
            // The distance cache was updated for commands that are now
            // discarded, so it has to be restored. Only the four remembered
            // entries are saved: the derived ones are rebuilt from them before
            // the next search reads them.
            references.dist_cache[..NUM_REMEMBERED_DISTANCES].copy_from_slice(saved_dist_cache);
            store_uncompressed_meta_block(
                is_last,
                data,
                wrapped_last_flush_pos,
                mask,
                bytes,
                &mut w,
            );
            let position = w.position();
            if w.overflowed() {
                return Err(BrotliCompressError::BufferOverflow);
            }
            return Ok(position);
        }

        let saved_last_bytes = u16::from(w.byte(1)) << 8 | u16::from(w.byte(0));
        let saved_last_bytes_bits = *last_bytes_bits as usize;

        if params.quality.splits_blocks() {
            let model = decide_over_literal_context_modeling(
                data,
                wrapped_last_flush_pos,
                bytes,
                mask,
                params.quality.models_literal_contexts()
                    && !params.disable_literal_context_modeling,
                params.quality.hq_context_modeling(),
                params.size_hint,
            );
            let mb = &mut self.metablock;
            build_meta_block_greedy_into(
                data,
                wrapped_last_flush_pos,
                mask,
                *prev_byte,
                *prev_byte2,
                model,
                commands,
                mb,
            );
            optimize_histograms(params.dist.alphabet_size_limit as usize, mb);
            writer.store_meta_block(
                data,
                wrapped_last_flush_pos,
                bytes,
                mask,
                *prev_byte,
                *prev_byte2,
                is_last,
                // Qualities below ten always model literal contexts as UTF-8;
                // `ChooseContextMode` cannot reach `CONTEXT_SIGNED` here.
                ContextMode::Utf8,
                &params.dist,
                commands,
                mb,
                &mut w,
            );
        } else if params.quality.uses_static_entropy_codes() {
            writer.store_meta_block_fast(
                data,
                wrapped_last_flush_pos,
                bytes,
                mask,
                is_last,
                &params.dist,
                commands,
                &mut w,
            );
        } else {
            writer.store_meta_block_trivial(
                data,
                wrapped_last_flush_pos,
                bytes,
                mask,
                is_last,
                &params.dist,
                commands,
                &mut w,
            );
        }

        if bytes + 4 < (w.position() >> 3) {
            // Compressing made it bigger; store the bytes as they are.
            references.dist_cache[..NUM_REMEMBERED_DISTANCES].copy_from_slice(saved_dist_cache);
            w.rewind(saved_last_bytes_bits);
            w.set_byte(0, saved_last_bytes as u8);
            w.set_byte(1, (saved_last_bytes >> 8) as u8);
            store_uncompressed_meta_block(
                is_last,
                data,
                wrapped_last_flush_pos,
                mask,
                bytes,
                &mut w,
            );
        }
        let position = w.position();
        if w.overflowed() {
            return Err(BrotliCompressError::BufferOverflow);
        }
        Ok(position)
    }
}

/// Returns the four distance-cache entries that survive a meta-block.
///
/// Mirrors `saved_dist_cache_`, which the reference declares four wide even
/// though the live cache is sixteen: the rest are derived again by
/// `PrepareDistanceCache` before any search reads them.
const fn remembered(cache: &DistanceCache) -> [i32; NUM_REMEMBERED_DISTANCES] {
    [cache[0], cache[1], cache[2], cache[3]]
}

/// Decides whether a meta-block is worth compressing at all.
///
/// Mirrors `ShouldCompress`. Tiny blocks cannot win, and a block that is almost
/// all literals is sampled: if the sample looks like noise, storing it verbatim
/// is both smaller and much faster.
fn should_compress(
    data: &[u8],
    mask: usize,
    last_flush_pos: u64,
    bytes: usize,
    num_literals: usize,
    num_commands: usize,
) -> bool {
    if bytes <= 2 {
        return false;
    }
    if num_commands >= (bytes >> 8) + 2 {
        return true;
    }
    if num_literals as f64 <= LITERAL_FRACTION * bytes as f64 {
        return true;
    }
    let mut literal_histo = HistogramLiteral::default();
    let bit_cost_threshold = bytes as f64 * MIN_ENTROPY / f64::from(SAMPLE_RATE);
    let samples = bytes.div_ceil(SAMPLE_RATE as usize);
    let mut pos = last_flush_pos as u32;
    for _ in 0..samples {
        literal_histo.add(usize::from(
            data.get(pos as usize & mask).copied().unwrap_or(0),
        ));
        pos = pos.wrapping_add(SAMPLE_RATE);
    }
    bits_entropy(&literal_histo.data) <= bit_cost_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{QualityLevel, WindowBits};

    fn encoder(quality: QualityLevel, size_hint: usize) -> GreedyEncoder {
        let params = CompressParams::new(quality, WindowBits::DEFAULT);
        GreedyEncoder::new(
            Level::try_detect().unwrap_or_else(Level::baseline),
            &params,
            size_hint,
        )
        .expect("supported quality")
    }

    /// Compresses `data` in blocks and returns the whole stream.
    fn compress(quality: QualityLevel, data: &[u8]) -> Vec<u8> {
        let mut encoder = encoder(quality, data.len());
        let limit = encoder.block_size_limit();
        let mut out = Vec::new();
        let mut offset = 0usize;
        loop {
            let take = (data.len() - offset).min(limit);
            let is_last = offset + take == data.len();
            out.extend_from_slice(
                encoder
                    .encode_block(&data[offset..offset + take], is_last)
                    .expect("encoding failed"),
            );
            offset += take;
            if is_last {
                break;
            }
        }
        out
    }

    #[test]
    fn the_window_header_matches_the_reference_encoding() {
        let header = |lgwin| {
            GreedyParams::new(
                &CompressParams::new(
                    QualityLevel::Q5,
                    WindowBits::standard(lgwin).expect("a legal window"),
                ),
                0,
            )
            .expect("a supported quality")
            .window
            .header()
        };
        assert_eq!(header(16), (0, 1));
        assert_eq!(header(17), (1, 7));
        assert_eq!(header(18), (3, 4));
        assert_eq!(header(22), (11, 4));
        assert_eq!(header(24), (15, 4));
        assert_eq!(header(10), (0x21, 7));
    }

    #[test]
    fn a_tiny_block_is_never_compressed() {
        assert!(!should_compress(&[1, 2], usize::MAX, 0, 2, 2, 0));
        assert!(!should_compress(&[], usize::MAX, 0, 0, 0, 0));
    }

    #[test]
    fn a_block_with_enough_commands_is_always_compressed() {
        let data = vec![0u8; 1024];
        assert!(should_compress(&data, usize::MAX, 0, 1024, 1024, 6));
    }

    #[test]
    fn random_literals_are_stored_uncompressed() {
        let mut rng = 0x1234_5678u64;
        let data: Vec<u8> = (0..200_000)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 24) as u8
            })
            .collect();
        assert!(!should_compress(
            &data,
            usize::MAX,
            0,
            data.len(),
            data.len(),
            1
        ));
    }

    #[test]
    fn repetitive_literals_are_worth_compressing() {
        let data = vec![b'a'; 4096];
        assert!(should_compress(
            &data,
            usize::MAX,
            0,
            data.len(),
            data.len(),
            1
        ));
    }

    #[test]
    fn every_quality_produces_a_non_empty_stream() {
        for quality in [QualityLevel::Q3, QualityLevel::Q4, QualityLevel::Q5] {
            let stream = compress(quality, b"hello hello hello hello hello");
            assert!(!stream.is_empty(), "quality {quality:?} produced nothing");
        }
    }

    #[test]
    fn an_empty_stream_is_two_bits() {
        for quality in [QualityLevel::Q3, QualityLevel::Q4, QualityLevel::Q5] {
            let mut encoder = encoder(quality, 0);
            let stream = encoder.encode_block(&[], true).expect("encoding failed");
            assert!(!stream.is_empty());
            assert!(encoder.is_finished());
        }
    }

    #[test]
    fn the_block_size_limit_follows_the_quality() {
        assert_eq!(encoder(QualityLevel::Q3, 0).block_size_limit(), 1 << 14);
        assert_eq!(encoder(QualityLevel::Q4, 0).block_size_limit(), 1 << 16);
        assert_eq!(encoder(QualityLevel::Q5, 0).block_size_limit(), 1 << 16);
    }

    #[test]
    fn every_backend_produces_the_same_stream() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i * 7 % 253) as u8).collect();
        for quality in [QualityLevel::Q3, QualityLevel::Q4, QualityLevel::Q5] {
            let params = CompressParams::new(quality, WindowBits::DEFAULT);
            let mut streams = Vec::new();
            for level in [
                Level::try_detect().unwrap_or_else(Level::baseline),
                Level::baseline(),
                Level::fallback(),
            ] {
                let mut encoder =
                    GreedyEncoder::new(level, &params, data.len()).expect("supported quality");
                let limit = encoder.block_size_limit();
                let mut out = Vec::new();
                let mut offset = 0usize;
                loop {
                    let take = (data.len() - offset).min(limit);
                    let is_last = offset + take == data.len();
                    out.extend_from_slice(
                        encoder
                            .encode_block(&data[offset..offset + take], is_last)
                            .expect("encoding failed"),
                    );
                    offset += take;
                    if is_last {
                        break;
                    }
                }
                streams.push(out);
            }
            assert!(
                streams.windows(2).all(|pair| pair[0] == pair[1]),
                "quality {quality:?} differed between backends"
            );
        }
    }
}
