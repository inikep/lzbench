//! Quality 0 and quality 1 encoders and their runtime SIMD dispatch.
//!
//! A [`FastEncoder`] retains a kernel selected once by `core::dispatch`.
//! Block calls pass its concrete token down into the match scan. Nothing below
//! re-detects features, and every backend produces byte-identical output.
//!
//! The block loop reproduces `BrotliEncoderCompressStreamFast` from
//! `c/enc/encode.c` of the pinned reference (`google/brotli` v1.2.0, commit
//! `028fb5a`): the input is cut into `1 << lgwin` fragments, each fragment gets
//! a freshly cleared hash table, and the trailing partial byte is carried into
//! the next fragment.

use alloc::boxed::Box;
use alloc::vec::Vec;

pub(crate) mod commands;
pub(crate) mod constants;
pub(crate) mod histogram;
pub(crate) mod q0;
pub(crate) mod q1;
pub(crate) mod tables;
pub(crate) mod workspace;

pub(crate) use crate::shared::{bits, huffman, match_len};

use super::dispatch::{self, Kernels};
use fearless_simd::{Level, Simd};

use self::bits::{BYTE_PADDING_SLACK, BitWriter, ByteBuffer, inject_byte_padding};
use self::constants::{OUTPUT_RESERVE_CONST, OUTPUT_SLACK, WINDOW_BITS_FAST};
use self::q1::TwoPassState;
use self::tables::DEFAULT_COMMAND_CODE_NUM_BITS;
use self::workspace::OnePassArena;
use crate::compressor::core::rfc9841::window::ResolvedWindow;
use crate::compressor::shared::SharedBrotliError;
use crate::compressor::{BrotliCompressError, BrotliResult, CompressParams, QualityLevel};

/// The two qualities this encoder implements.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FastQuality {
    /// One-pass encoding.
    Q0,
    /// Two-pass encoding.
    Q1,
}

impl TryFrom<QualityLevel> for FastQuality {
    type Error = BrotliCompressError;

    /// Routes quality 0 and 1 to the fast path.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] for every other
    /// quality, which has no implementation yet.
    fn try_from(value: QualityLevel) -> Result<Self, Self::Error> {
        match value {
            QualityLevel::Q0 => Ok(Self::Q0),
            QualityLevel::Q1 => Ok(Self::Q1),
            other => Err(BrotliCompressError::UnsupportedQuality(usize::from(other))),
        }
    }
}

/// Quality-specific scratch state.
pub(crate) enum FastCore {
    /// Quality 0 state. The arena is built by the first fragment that scans
    /// for matches; a stream whose only fragment is stored verbatim never
    /// allocates one.
    OnePass { arena: Option<Box<OnePassArena>> },
    /// Quality 1 state, including the pass-one command and literal buffers.
    TwoPass { state: Box<TwoPassState> },
}

impl FastCore {
    /// Counts boxed state and every allocation owned by that state.
    fn retained_bytes(&self) -> usize {
        match self {
            Self::OnePass { arena } => arena.as_ref().map_or(0, |arena| {
                size_of::<OnePassArena>()
                    + arena.tree.capacity() * size_of::<huffman::HuffmanNode>()
            }),
            Self::TwoPass { state } => {
                size_of::<TwoPassState>()
                    + size_of::<workspace::TwoPassArena>()
                    + state.arena.tmp_tree.capacity() * size_of::<huffman::HuffmanNode>()
                    + state.commands.capacity() * size_of::<u32>()
                    + state.literals.capacity()
            }
        }
    }

    /// Creates the state for `quality`.
    fn new(quality: FastQuality) -> Self {
        match quality {
            FastQuality::Q0 => Self::OnePass { arena: None },
            FastQuality::Q1 => Self::TwoPass {
                state: Box::default(),
            },
        }
    }

    /// Returns the quality this core implements.
    const fn quality(&self) -> FastQuality {
        match self {
            Self::OnePass { .. } => FastQuality::Q0,
            Self::TwoPass { .. } => FastQuality::Q1,
        }
    }

    /// Restores the arena to the state [`FastCore::new`] would produce.
    ///
    /// Assigns through the `Box` rather than replacing it, so the allocation
    /// is reused. Quality 0 carries its pre-compressed command code from one
    /// fragment to the next, and quality 1 its histograms, so neither is
    /// scratch that the next fragment would have overwritten regardless.
    fn reset(&mut self) {
        match self {
            Self::OnePass { arena } => {
                if let Some(arena) = arena {
                    arena.reset();
                }
            }
            Self::TwoPass { state } => state.reset(),
        }
    }

    /// Returns whether a final fragment of `len` bytes is stored verbatim.
    ///
    /// Such a fragment needs neither the arena nor a hash table, so the
    /// encoder skips preparing both. Quality 1 always scans.
    fn stores_verbatim(&self, independent: bool, len: usize, is_last: bool) -> bool {
        match self {
            Self::OnePass { arena } => {
                let cmd_code_numbits = arena
                    .as_ref()
                    .map_or(DEFAULT_COMMAND_CODE_NUM_BITS, |arena| {
                        arena.cmd_code_numbits
                    });
                q0::stores_verbatim(cmd_code_numbits, independent, len, is_last)
            }
            Self::TwoPass { .. } => false,
        }
    }

    /// Returns the hash table size this core needs for a fragment.
    fn table_entries(&self, input_size: usize) -> usize {
        match self {
            Self::OnePass { .. } => q0::TableBits::for_input(input_size).entries(),
            Self::TwoPass { .. } => q1::TableBits::for_input(input_size).entries(),
        }
    }
}

/// Compresses one fragment with a resolved SIMD token.
///
/// This is the only place the fast path branches on the instruction set; the
/// token is passed by value into every leaf that uses it.
#[inline(always)]
pub(crate) fn encode_fragment<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    core: &mut FastCore,
    input: &[u8],
    is_last: bool,
    table: &mut [i32],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    if core.stores_verbatim(INDEPENDENT, input.len(), is_last) {
        q0::store_final_verbatim(input, w);
        return;
    }
    match core {
        FastCore::OnePass { arena } => {
            q0::compress_fragment::<_, INDEPENDENT>(
                simd,
                arena.get_or_insert_with(Box::default),
                input,
                is_last,
                q0::TableBits::for_input(input.len()),
                table,
                w,
            );
        }
        FastCore::TwoPass { state } => q1::compress_fragment::<_, INDEPENDENT>(
            simd,
            state,
            input,
            is_last,
            q1::TableBits::for_input(input.len()),
            table,
            w,
        ),
    }
}

/// Streaming quality 0 / quality 1 encoder.
///
/// One instance owns every buffer the encoder needs and reuses them across
/// blocks, so after construction no allocation happens in the match scan or the
/// command replay.
pub(crate) struct FastEncoder {
    kernels: Box<dyn Kernels>,
    /// Whether `kernels` are the fragment-only ones of an independent worker.
    independent: bool,
    core: FastCore,
    block_size_limit: usize,
    /// The stream header, kept so a reused encoder can start over from it.
    header: (u16, u32),
    last_bytes: u16,
    last_bytes_bits: u32,
    table: Vec<i32>,
    storage: Vec<u8>,
    finished: bool,
}

impl FastEncoder {
    /// Installs fragment-only kernels when constructing an independent worker.
    #[cfg(any(test, not(feature = "no_std")))]
    pub(crate) fn select_fragment_kernels(&mut self, level: Level) {
        self.kernels = dispatch::select_independent(level);
        self.independent = true;
    }

    /// Resets the header and seeds only this segment's literal context/history.
    #[cfg(not(feature = "no_std"))]
    pub(crate) fn begin_fragment(&mut self, prefix: &[u8]) -> BrotliResult<()> {
        self.last_bytes = 0;
        self.last_bytes_bits = 0;
        let _ = prefix;
        Ok(())
    }

    /// Confirms the flush left no pending partial byte.
    #[cfg(not(feature = "no_std"))]
    pub(crate) const fn fragment_aligned(&self) -> bool {
        self.last_bytes_bits == 0
    }

    /// Creates an encoder for `params` running at SIMD `level`.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::UnsupportedQuality`] when the quality is
    /// outside the range this encoder implements.
    pub(crate) fn new(level: Level, params: &CompressParams) -> BrotliResult<Self> {
        let quality = FastQuality::try_from(params.quality())?;
        if params.lgwin().is_large() {
            // These qualities write distances through a static entropy model
            // built for the RFC 7932 alphabet, so they cannot carry the wider
            // one. Refuse rather than quietly emitting an ordinary stream.
            return Err(SharedBrotliError::UnsupportedLargeWindow {
                quality: usize::from(params.quality()),
            }
            .into());
        }
        let window = ResolvedWindow::new(params);
        let lgwin = window.encoder_bits();
        // The reference fast path always advertises at least eighteen window
        // bits, while still cutting the input at the requested window size.
        let (last_bytes, last_bytes_bits) = window.at_least(WINDOW_BITS_FAST).header();
        Ok(Self {
            kernels: dispatch::select(level),
            independent: false,
            core: FastCore::new(quality),
            block_size_limit: 1usize << lgwin,
            header: (last_bytes, last_bytes_bits),
            last_bytes,
            last_bytes_bits,
            table: Vec::new(),
            storage: Vec::new(),
            finished: false,
        })
    }

    /// Returns the largest fragment this encoder compresses in one go.
    pub(crate) const fn block_size_limit(&self) -> usize {
        self.block_size_limit
    }

    /// Returns whether `params` would build an encoder of exactly this shape.
    ///
    /// A workspace uses this to decide whether resetting is equivalent to
    /// rebuilding. Nothing is allocated: the three things that shape a fast
    /// encoder — the quality, the fragment limit and the stream header — are
    /// all recomputed from `params` directly.
    pub(crate) fn matches(&self, params: &CompressParams) -> bool {
        let Ok(quality) = FastQuality::try_from(params.quality()) else {
            return false;
        };
        if params.lgwin().is_large() {
            return false;
        }
        let window = ResolvedWindow::new(params);
        self.core.quality() == quality
            && self.block_size_limit == 1usize << window.encoder_bits()
            && self.header == window.at_least(WINDOW_BITS_FAST).header()
    }

    /// Restores the encoder to the state its constructor left it in.
    ///
    /// Every allocation is kept: the hash table and the scratch buffer are
    /// resized and cleared per fragment anyway, and the arena is rebuilt in
    /// place so its `Box` survives. What has to go back is the stream state —
    /// the carried bits, the pre-compressed command code the next fragment
    /// would have reused, and the finished flag.
    pub(crate) fn reset(&mut self) {
        self.core.reset();
        (self.last_bytes, self.last_bytes_bits) = self.header;
        self.finished = false;
    }

    /// Returns the bytes this encoder keeps allocated between fragments.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.core.retained_bytes()
            + size_of_val(&*self.kernels)
            + self.table.capacity() * size_of::<i32>()
            + self.storage.capacity()
    }

    /// Returns the scratch capacity one fragment of `input_len` bytes needs.
    ///
    /// This is the reference's `2 * block_size + 503`, plus the headroom the
    /// bit writer's whole-word stores reach past the last bit.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] when the arithmetic
    /// does not fit in a `usize`.
    pub(crate) const fn fragment_reserve(input_len: usize) -> BrotliResult<usize> {
        let Some(doubled) = input_len.checked_mul(2) else {
            return Err(BrotliCompressError::BufferOverflow);
        };
        let Some(reserve) = doubled.checked_add(OUTPUT_RESERVE_CONST) else {
            return Err(BrotliCompressError::BufferOverflow);
        };
        match reserve.checked_add(OUTPUT_SLACK) {
            Some(reserve) => Ok(reserve),
            None => Err(BrotliCompressError::BufferOverflow),
        }
    }

    /// Clears the hash table for a fragment of `input_len` bytes.
    ///
    /// A final fragment the core stores verbatim gets no table at all: the
    /// kernel never reads one, and clearing it would be the whole cost of
    /// such a call.
    fn prepare_table(&mut self, input_len: usize, is_last: bool) -> usize {
        if self
            .core
            .stores_verbatim(self.independent, input_len, is_last)
        {
            return 0;
        }
        let entries = self.core.table_entries(input_len);
        if self.table.len() < entries {
            // A fresh zeroed allocation, rather than growing in place: the
            // allocator hands out zero pages, while `resize` would memset a
            // buffer the encoder is about to overwrite anyway. It also arrives
            // already cleared.
            self.table = vec![0i32; entries];
        } else {
            // Only the active range is cleared; unused capacity stays untouched.
            self.table[..entries].fill(0);
        }
        entries
    }

    /// Compresses one fragment into `storage`, returning the completed bytes.
    ///
    /// `storage` must hold at least [`FastEncoder::fragment_reserve`] bytes.
    fn run_fragment(
        &mut self,
        input: &[u8],
        is_last: bool,
        entries: usize,
        storage: &mut [u8],
    ) -> BrotliResult<usize> {
        storage[0] = self.last_bytes as u8;
        storage[1] = (self.last_bytes >> 8) as u8;

        let Self {
            kernels,
            core,
            table,
            last_bytes_bits,
            ..
        } = self;
        let mut w = BitWriter::new(storage, *last_bytes_bits as usize);
        let table = &mut table[..entries];
        kernels.fast(core, input, is_last, table, &mut w);

        if w.overflowed() {
            return Err(BrotliCompressError::BufferOverflow);
        }
        let position = w.position();
        let complete = position >> 3;
        self.last_bytes = u16::from(w.byte(complete));
        self.last_bytes_bits = (position & 7) as u32;
        self.finished = is_last;
        Ok(complete)
    }

    /// Encodes directly into an append destination, initializing only output.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn encode_block_append(
        &mut self,
        input: &[u8],
        is_last: bool,
        output: &mut Vec<u8>,
    ) -> BrotliResult<usize> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit);
        let start = output.len();
        let position = start
            .checked_mul(8)
            .and_then(|bits| bits.checked_add(self.last_bytes_bits as usize))
            .ok_or(BrotliCompressError::BufferOverflow)?;
        output.extend_from_slice(&self.last_bytes.to_le_bytes());
        let entries = self.prepare_table(input.len(), is_last);
        let mut writer = BitWriter::append(output, position);
        self.kernels.fast_append(
            &mut self.core,
            input,
            is_last,
            &mut self.table[..entries],
            &mut writer,
        );
        let position = writer.position();
        let complete = position >> 3;
        self.last_bytes = u16::from(writer.byte(complete));
        self.last_bytes_bits = (position & 7) as u32;
        self.finished = is_last;
        output.truncate(complete);
        Ok(complete - start)
    }

    /// Compresses one fragment and returns the bytes it completed.
    ///
    /// A fragment shorter than [`FastEncoder::block_size_limit`] is allowed
    /// only when `is_last` is set. The trailing partial byte is retained and
    /// emitted with the next fragment, or by the final one.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] if the internal scratch
    /// buffer proved too small, which would indicate a bug in the size bound.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn encode_block(&mut self, input: &[u8], is_last: bool) -> BrotliResult<&[u8]> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit);

        let reserve = Self::fragment_reserve(input.len())?;
        if self.storage.len() < reserve {
            self.storage = vec![0u8; reserve];
        }
        let entries = self.prepare_table(input.len(), is_last);

        let mut storage = core::mem::take(&mut self.storage);
        let outcome = self.run_fragment(input, is_last, entries, &mut storage);
        self.storage = storage;
        let complete = outcome?;
        Ok(&self.storage[..complete])
    }

    /// Compresses `input` as a non-final fragment and realigns the stream.
    ///
    /// Mirrors `BROTLI_OPERATION_FLUSH` on the reference's fast path. These
    /// qualities already close a meta-block on every call, so the flush adds
    /// only the empty metadata block that pushes the stream back onto a byte
    /// boundary — after which everything returned so far decodes to everything
    /// fed in so far.
    ///
    /// An empty `input` skips the fragment entirely, exactly as the reference
    /// does when a flush arrives with nothing buffered. The result is then
    /// empty too whenever the stream was already aligned.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::BufferOverflow`] if the internal scratch
    /// buffer proved too small, which would indicate a bug in the size bound.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn flush_block(&mut self, input: &[u8]) -> BrotliResult<&[u8]> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit);

        let reserve = Self::fragment_reserve(input.len())?;
        if self.storage.len() < reserve {
            self.storage = vec![0u8; reserve];
        }

        let mut storage = core::mem::take(&mut self.storage);
        let outcome = self.run_flush(input, &mut storage);
        self.storage = storage;
        let complete = outcome?;
        match self.storage.get(..complete) {
            Some(output) => Ok(output),
            None => Err(BrotliCompressError::BufferOverflow),
        }
    }

    /// Compresses `input` as a non-final fragment, then pads to a byte.
    ///
    /// `storage` must hold at least [`FastEncoder::fragment_reserve`] bytes,
    /// whose slack covers the padding block on top of the fragment.
    fn run_flush(&mut self, input: &[u8], storage: &mut [u8]) -> BrotliResult<usize> {
        let complete = if input.is_empty() {
            0
        } else {
            let entries = self.prepare_table(input.len(), false);
            self.run_fragment(input, false, entries, storage)?
        };

        // The padding starts inside the byte the fragment left partly written,
        // which is `storage[complete]` — the same bits `last_bytes` carries —
        // so the seal overwrites it rather than following it.
        let padded = match storage.get_mut(complete..) {
            Some(tail) if tail.len() >= BYTE_PADDING_SLACK => {
                inject_byte_padding(&mut self.last_bytes, &mut self.last_bytes_bits, tail)
            }
            _ => return Err(BrotliCompressError::BufferOverflow),
        };
        Ok(complete + padded)
    }

    /// Compresses one fragment straight into `dst`, returning its length.
    ///
    /// This is the in-place path: nothing is copied afterwards. `dst` has to
    /// hold at least [`FastEncoder::fragment_reserve`] bytes, which a buffer
    /// sized by the public compressed-size bound always does.
    ///
    /// # Errors
    ///
    /// Returns [`BrotliCompressError::OutputTooSmall`] when `dst` is shorter
    /// than the reservation, and [`BrotliCompressError::BufferOverflow`] if
    /// the encoder still ran out of room.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn encode_block_into(
        &mut self,
        input: &[u8],
        is_last: bool,
        dst: &mut [u8],
    ) -> BrotliResult<usize> {
        debug_assert!(!self.finished);
        debug_assert!(input.len() <= self.block_size_limit);

        if dst.len() < Self::fragment_reserve(input.len())? {
            return Err(BrotliCompressError::OutputTooSmall);
        }
        let entries = self.prepare_table(input.len(), is_last);
        self.run_fragment(input, is_last, entries, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::WindowBits;

    #[test]
    fn direct_appends_preserve_prefixes_and_partial_bytes_on_every_backend() {
        for backend in crate::compressor::Backend::available() {
            for quality in [QualityLevel::Q0, QualityLevel::Q1] {
                for independent in [false, true] {
                    let params =
                        CompressParams::new(quality, WindowBits::standard(10).expect("window"));
                    let mut fixed = FastEncoder::new(backend.0, &params).expect("encoder");
                    let mut append = FastEncoder::new(backend.0, &params).expect("encoder");
                    if independent {
                        fixed.select_fragment_kernels(backend.0);
                        append.select_fragment_kernels(backend.0);
                    }
                    let mut expected = b"existing prefix".to_vec();
                    let mut actual = expected.clone();
                    for (data, last) in [(&[b'a'; 1024][..], false), (&[b'b'; 17][..], true)] {
                        let encoded = fixed.encode_block(data, last).expect("encode");
                        expected.extend_from_slice(encoded);
                        let written = append
                            .encode_block_append(data, last, &mut actual)
                            .expect("append");
                        assert_eq!(written, encoded.len());
                        assert_eq!(actual, expected, "{backend:?}, {quality:?}, {independent}");
                    }
                }
            }
        }
    }

    #[test]
    fn window_header_matches_the_reference_encoding() {
        let header = |lgwin| {
            ResolvedWindow::new(&CompressParams::new(
                QualityLevel::Q0,
                WindowBits::standard(lgwin).expect("a legal window"),
            ))
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
    fn quality_routing_accepts_only_the_fast_path() {
        assert_eq!(
            FastQuality::try_from(QualityLevel::Q0).ok(),
            Some(FastQuality::Q0)
        );
        assert_eq!(
            FastQuality::try_from(QualityLevel::Q1).ok(),
            Some(FastQuality::Q1)
        );
        assert!(matches!(
            FastQuality::try_from(QualityLevel::Q5),
            Err(BrotliCompressError::UnsupportedQuality(5))
        ));
    }

    #[test]
    fn block_size_limit_follows_the_requested_window() -> Result<(), BrotliCompressError> {
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        for lgwin in [10u8, 16, 18, 22, 24] {
            let lgwin_bits = WindowBits::standard(lgwin).unwrap_or(WindowBits::DEFAULT);
            let params = CompressParams::new(QualityLevel::Q0, lgwin_bits);
            let encoder = FastEncoder::new(level, &params)?;
            assert_eq!(
                encoder.block_size_limit(),
                1 << usize::from(lgwin_bits.bits())
            );
        }
        Ok(())
    }

    #[test]
    fn table_sizing_depends_on_the_quality() {
        assert_eq!(FastCore::new(FastQuality::Q0).table_entries(10), 512);
        assert_eq!(FastCore::new(FastQuality::Q1).table_entries(10), 256);
        assert_eq!(
            FastCore::new(FastQuality::Q0).table_entries(1 << 20),
            32_768
        );
        assert_eq!(
            FastCore::new(FastQuality::Q1).table_entries(1 << 20),
            131_072
        );
    }
}
