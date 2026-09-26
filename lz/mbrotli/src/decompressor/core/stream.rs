//! Meta-block regeneration: a byte-exact resumable state machine plus a fast
//! path that decodes whole commands while a word-sized refill is possible.
//!
//! The fast path keeps the same state fields as the resumable stages and
//! returns to them at every point where input or output may run short, so
//! any chunking of input and output produces identical results and errors.

use super::super::{DecodeError, DecoderConfig, InvalidDataKind, OutputSize};
use super::{
    bits::{Bits, Input},
    block::Block,
    context_map::ContextMap,
    dictionary,
    distance::{Cache, DistanceLayout, Partial},
    header::{self, MetaBlock},
    huffman::{Builder, Group},
    memory::Memory,
};
use crate::shared::dictionary::transform::SCRATCH_BYTES;
use crate::{
    Window, WindowEncoding,
    dictionary::DictionaryRef,
    shared::format::{
        CONTEXT_LUT_SIGNED, CONTEXT_LUT_UTF8, COPY_BASE, COPY_EXTRA, INS_BASE, INS_EXTRA,
    },
};
use alloc::vec::Vec;
use fearless_simd::{Simd, SimdBase, dispatch, u8x16, u8x32};

/// Smallest ring allocation; growth doubles up to the window size.
const MIN_RING: u64 = 64;

const fn lsb6_lut() -> [u8; 512] {
    let mut lut = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        lut[i] = (i & 63) as u8;
        i += 1;
    }
    lut
}

const fn msb6_lut() -> [u8; 512] {
    let mut lut = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        lut[i] = (i >> 2) as u8;
        i += 1;
    }
    lut
}

const CONTEXT_LUT_LSB6: [u8; 512] = lsb6_lut();
const CONTEXT_LUT_MSB6: [u8; 512] = msb6_lut();

/// Context lookup for a literal block's context mode. Each mode combines
/// the two previous bytes as `lut[p1] | lut[256 + p2]`.
const fn context_lut(mode: u8) -> &'static [u8; 512] {
    match mode {
        0 => &CONTEXT_LUT_LSB6,
        1 => &CONTEXT_LUT_MSB6,
        2 => &CONTEXT_LUT_UTF8,
        _ => &CONTEXT_LUT_SIGNED,
    }
}

const CELLS: [usize; 11] = [0, 1, 0, 1, 8, 9, 2, 16, 10, 17, 18];

#[derive(Debug, Clone, Copy, Default)]
enum Stage {
    #[default]
    Window,
    Meta,
    Metadata,
    Raw,
    Blocks(usize),
    DistanceParams,
    Modes(usize),
    Maps(usize),
    Trees(usize, usize),
    Command,
    InsertExtra(usize, usize),
    CopyExtra(usize),
    Literals,
    Distance,
    DistanceExtra(usize),
    DistanceExtraHigh(usize, u64),
    Resolve,
    Copy,
    Dictionary,
    Prefix {
        offset: u64,
        start: u64,
    },
    PrefixHistory {
        offset: usize,
        start: u64,
    },
    EndBlock,
    End,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum Stop {
    Input,
    Output,
    Member,
}

/// Why the fast literal loop stopped before the insert run ended.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Pause {
    Done,
    Input,
    Output,
}

pub(crate) struct Output<'a> {
    pub(crate) bytes: &'a mut [u8],
    /// Collect a member in history, stopping before any ring byte is overwritten.
    pub(crate) collect: Option<usize>,
    /// The member began at `bytes[0]` in this call and `bytes` is its
    /// history: the ring is neither written nor read, and delivering history
    /// only advances `produced`. Set only for a whole member in one call.
    pub(crate) linear: bool,
    pub(crate) produced: usize,
    pub(crate) total_before: u64,
    pub(crate) limit: Option<u64>,
    pub(crate) exact: OutputSize,
}

impl Output<'_> {
    /// Whether delivery copies history bytes into `bytes`.
    const fn delivers(&self) -> bool {
        self.collect.is_none() && !self.linear
    }

    /// Length of the linear history visible to bulk work starting at
    /// `position` with `remaining` meta-block bytes, within the call's fast
    /// end (`out_end`, at most the slice length): bounding it keeps the
    /// speculative tail of a 16- or 32-byte copy inside bytes this
    /// meta-block writes, so the caller's unused suffix stays untouched.
    fn history_end(&self, position: u64, remaining: u64, out_end: usize) -> usize {
        usize::try_from(position.saturating_add(remaining)).map_or(out_end, |end| end.min(out_end))
    }

    fn ready(&self) -> Result<bool, DecodeError> {
        let next = self
            .total_before
            .checked_add(self.produced as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or(DecodeError::SizeOverflow)?;
        if let Some(limit) = self.limit
            && next > limit
        {
            return Err(DecodeError::OutputLimitExceeded { limit });
        }
        if let OutputSize::Exact(expected) = self.exact
            && next > expected
        {
            return Err(DecodeError::OutputSizeMismatch {
                expected,
                actual: next,
            });
        }
        Ok(self.produced < self.collect.unwrap_or(self.bytes.len()))
    }

    /// Largest `produced` this call may reach without per-byte policy checks.
    fn fast_end(&self) -> usize {
        let mut budget = u64::MAX - self.total_before;
        if let Some(limit) = self.limit {
            budget = budget.min(limit.saturating_sub(self.total_before));
        }
        if let OutputSize::Exact(expected) = self.exact {
            budget = budget.min(expected.saturating_sub(self.total_before));
        }
        let capacity = self.collect.unwrap_or(self.bytes.len());
        usize::try_from(budget).map_or(capacity, |budget| budget.min(capacity))
    }
}

/// Per-symbol command decoding, in the shape of C's `kCmdLut`: one table
/// lookup yields both length bases, both extra-bit widths, the implicit
/// distance flag and the distance context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Command {
    insert_extra: u8,
    copy_extra: u8,
    /// Copy length code, for resuming at `Stage::CopyExtra`.
    copy_code: u8,
    /// Distance context from the copy length: `min(copy - 2, 3)`. Copy codes
    /// with extra bits all start above four, so the base decides it.
    distance_context: u8,
    /// The distance is the most recent one, without a distance symbol.
    implicit: bool,
    insert_base: u16,
    copy_base: u16,
}

/// The command alphabet padded to a power of two, so a masked index replaces
/// a bounds check; a decoded symbol is always below 704.
const COMMAND_SLOTS: usize = 1024;

const fn commands() -> [Command; COMMAND_SLOTS] {
    let mut table = [Command {
        insert_extra: 0,
        copy_extra: 0,
        copy_code: 0,
        distance_context: 0,
        implicit: false,
        insert_base: 0,
        copy_base: 0,
    }; COMMAND_SLOTS];
    let mut symbol = 0;
    while symbol < 704 {
        let cell = CELLS[symbol >> 6];
        let insert = (cell & 24) + ((symbol >> 3) & 7);
        let copy = ((cell << 3) & 24) + (symbol & 7);
        let copy_base = COPY_BASE[copy];
        table[symbol] = Command {
            insert_extra: INS_EXTRA[insert] as u8,
            copy_extra: COPY_EXTRA[copy] as u8,
            copy_code: copy as u8,
            distance_context: if copy_base > 4 {
                3
            } else {
                (copy_base - 2) as u8
            },
            implicit: symbol < 128,
            insert_base: INS_BASE[insert] as u16,
            copy_base: copy_base as u16,
        };
        symbol += 1;
    }
    table
}

const COMMANDS: [Command; COMMAND_SLOTS] = commands();

/// Index mask of a history buffer: a ring is a power of two and wraps; a
/// linear history is the caller's slice, which never wraps because every
/// position it holds is below its length. A power-of-two linear slice gets
/// the same mask, which is the identity on those positions.
#[inline(always)]
const fn history_mask(len: usize) -> u64 {
    if len.is_power_of_two() {
        len as u64 - 1
    } else {
        u64::MAX
    }
}

/// Delivers ring bytes decoded since `flushed`. The pending region never
/// crosses the ring end: writers flush whenever a write reaches it.
fn flush_ring(ring: &[u8], position: u64, flushed: &mut u64, output: &mut Output<'_>) {
    let pending = (position - *flushed) as usize;
    if pending != 0 {
        if output.delivers() {
            let start = (*flushed & history_mask(ring.len())) as usize;
            output.bytes[output.produced..output.produced + pending]
                .copy_from_slice(&ring[start..start + pending]);
        }
        output.produced += pending;
        *flushed = position;
    }
}

/// Copies `len` bytes from `distance` back where the two regions overlap
/// and neither wraps: `src + len > dst`. A unit distance is a fill; longer
/// runs double the replicated prefix; short runs go byte by byte.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn copy_overlapping(ring: &mut [u8], dst: usize, src: usize, len: usize, distance: usize) {
    if distance == 1 {
        let byte = ring[src];
        ring[dst..dst + len].fill(byte);
    } else if len <= 16 {
        for i in 0..len {
            ring[dst + i] = ring[src + i];
        }
    } else {
        let mut done = 0;
        let mut chunk = distance;
        while done < len {
            let n = chunk.min(len - done);
            ring.copy_within(src..src + n, dst + done);
            done += n;
            chunk <<= 1;
        }
    }
}

/// Copies sixteen bytes when both windows are inside the ring; the slots
/// past the real copy are the format's unreachable bytes ahead of the
/// window, or still unwritten ring space. The vector stays in registers
/// between the source and destination bounds checks.
#[inline(always)]
fn copy16<S: Simd>(simd: S, ring: &mut [u8], src: usize, dst: usize) -> bool {
    if let Some(word) = ring.get(src..).and_then(|s| s.first_chunk::<16>())
        && let word = u8x16::load_array_ref(simd, word)
        && let Some(target) = ring.get_mut(dst..).and_then(|t| t.first_chunk_mut::<16>())
    {
        word.store_array(target);
        return true;
    }
    false
}

/// Copies a snapshot of thirty-two bytes for a non-overlapping copy of 17
/// to 32 bytes. At most fifteen unreachable bytes ahead are overwritten.
/// Loading the whole source first also permits physically overlapping slots.
#[inline(always)]
fn copy32<S: Simd>(simd: S, ring: &mut [u8], src: usize, dst: usize) -> bool {
    if let Some(word) = ring.get(src..).and_then(|s| s.first_chunk::<32>())
        && let word = u8x32::load_array_ref(simd, word)
        && let Some(target) = ring.get_mut(dst..).and_then(|t| t.first_chunk_mut::<32>())
    {
        word.store_array(target);
        return true;
    }
    false
}

/// Copies `length` bytes from `distance` back, in ring pieces that neither
/// wrap nor need per-byte handling. The ring already holds `position + length`.
#[inline(always)]
fn copy_ring<S: Simd>(
    simd: S,
    ring: &mut [u8],
    position: &mut u64,
    distance: u64,
    mut length: usize,
    flushed: &mut u64,
    output: &mut Output<'_>,
) {
    let size = ring.len();
    let mask = history_mask(size);
    while length != 0 {
        let dst = (*position & mask) as usize;
        let src = ((*position - distance) & mask) as usize;
        let piece = length.min(size - dst).min(size - src);
        if src < dst && (distance as usize) < piece {
            copy_overlapping(ring, dst, src, piece, distance as usize);
        } else if !(piece <= 16 && copy16(simd, ring, src, dst)) {
            ring.copy_within(src..src + piece, dst);
        }
        *position += piece as u64;
        length -= piece;
        if dst + piece == size {
            flush_ring(ring, *position, flushed, output);
        }
    }
}

/// Writes `bytes` at `position` in ring pieces. The ring already holds them,
/// which is why the emptiness check comes first: a transformed dictionary word
/// can decode to no bytes at all, and as a member's first command it reaches
/// this with nothing written and the ring still unallocated.
fn write_ring(
    ring: &mut [u8],
    position: &mut u64,
    mut bytes: &[u8],
    mut flushed: Option<(&mut u64, &mut Output<'_>)>,
) {
    if bytes.is_empty() {
        return;
    }
    let size = ring.len();
    let mask = history_mask(size);
    while !bytes.is_empty() {
        let dst = (*position & mask) as usize;
        let piece = bytes.len().min(size - dst);
        ring[dst..dst + piece].copy_from_slice(&bytes[..piece]);
        bytes = &bytes[piece..];
        *position += piece as u64;
        if dst + piece == size
            && let Some((flushed, output)) = flushed.as_mut()
        {
            flush_ring(ring, *position, flushed, output);
        }
    }
}

/// Grows the ring so positions below `end` are addressable, doubling up to
/// the window size. A full ring wraps instead.
#[cold]
#[inline(never)]
fn grow_ring(
    memory: &mut Memory,
    ring: &mut Vec<u8>,
    window_size: u64,
    end: u64,
) -> Result<(), DecodeError> {
    let len = ring.len() as u64;
    if end <= len || len >= window_size {
        return Ok(());
    }
    let desired = ring_size(len, window_size, end)?;
    memory.resize(ring, desired)
}

/// Power-of-two allocation length, shared by ordinary and initialized growth.
fn ring_size(len: u64, window_size: u64, end: u64) -> Result<usize, DecodeError> {
    let desired = end
        .checked_next_power_of_two()
        .unwrap_or(u64::MAX)
        .max(len.saturating_mul(2))
        .max(MIN_RING)
        .min(window_size);
    usize::try_from(desired).map_err(|_| DecodeError::SizeOverflow)
}

/// Grows the ring for a unit-distance run, initializing the new allocation and
/// the space up to `end` with the repeated `byte`. A distance-one copy is a
/// fill, so producing the bytes as the growth's initial value avoids a zero
/// fill followed by an overwrite. Callers ensure `end` needs growth and stays
/// within the window (`ring.len() < end <= window_size`). Returns the new length.
fn repeat_grow(
    memory: &mut Memory,
    ring: &mut Vec<u8>,
    window_size: u64,
    position: u64,
    byte: u8,
    end: u64,
) -> Result<usize, DecodeError> {
    let old_len = ring.len();
    let desired = ring_size(old_len as u64, window_size, end)?;
    memory.reserve(ring, desired)?;
    ring[position as usize..].fill(byte);
    ring.resize(desired, byte);
    Ok(desired)
}

/// The two most recent bytes before `position`, zero before any output.
#[inline(always)]
fn previous_bytes(ring: &[u8], position: u64) -> (u8, u8) {
    if position >= 2 {
        let mask = history_mask(ring.len());
        (
            ring[((position - 1) & mask) as usize],
            ring[((position - 2) & mask) as usize],
        )
    } else {
        first_bytes(ring, position)
    }
}

/// The bytes history positions index this call: the caller's slice for a
/// linear member, the ring otherwise.
fn history<'b>(ring: &'b [u8], output: &'b Output<'_>) -> &'b [u8] {
    if output.linear { output.bytes } else { ring }
}

#[cold]
fn first_bytes(ring: &[u8], position: u64) -> (u8, u8) {
    if position == 0 { (0, 0) } else { (ring[0], 0) }
}

/// Everything only a compressed meta-block needs: the code reader, block
/// switch state, context maps, prefix-code groups, the distance layout and
/// the dictionary transform scratch. Kept on the heap and created by the
/// first compressed meta-block, so a decoder that only ever sees stored
/// members, metadata or an empty stream never initializes or copies it.
#[derive(Debug)]
struct Tables {
    builder: Builder,
    blocks: [Block; 3],
    modes: [u8; 256],
    maps: [ContextMap; 2],
    trees: [Group; 3],
    distances: DistanceLayout,
    /// Per-symbol split of the current distance alphabet, and the layout it
    /// was filled for so an unchanged layout reuses it.
    distance_table: Vec<Partial>,
    distance_table_layout: Option<DistanceLayout>,
    scratch: [u8; SCRATCH_BYTES],
    scratch_pos: usize,
    scratch_len: usize,
}

impl Default for Tables {
    fn default() -> Self {
        Self {
            builder: Builder::default(),
            blocks: Default::default(),
            modes: [0; 256],
            maps: Default::default(),
            trees: Default::default(),
            distances: DistanceLayout::default(),
            distance_table: Vec::new(),
            distance_table_layout: None,
            scratch: [0; SCRATCH_BYTES],
            scratch_pos: 0,
            scratch_len: 0,
        }
    }
}

/// Literal context of the next byte: the block's context mode applied to
/// the two most recent output bytes.
fn context(tables: &Tables, ring: &[u8], position: u64) -> usize {
    let (p1, p2) = previous_bytes(ring, position);
    let lut = context_lut(tables.modes[tables.blocks[0].current]);
    usize::from(lut[usize::from(p1)] | lut[256 + usize::from(p2)])
}

#[derive(Debug, Default)]
pub(crate) struct Stream {
    bits: Bits,
    stage: Stage,
    memory: Memory,
    /// Power-of-two history ring, grown on output up to the window size.
    /// Its length never shrinks between operations; stale bytes are never
    /// addressable because references stay within the current position.
    ring: Vec<u8>,
    // A prefix-crossing reference can exceed the sliding window. Preserve only
    // the original history bytes that would be overwritten before being read.
    prefix_history: Vec<u8>,
    position: u64,
    window_size: u64,
    max_backward: u64,
    pub(crate) window: Option<Window>,
    large: bool,
    last: bool,
    remaining: u64,
    /// At most one element: the compressed meta-block workspace, created by
    /// the first compressed meta-block and retained with the rest.
    tables: Vec<Tables>,
    literals: u64,
    copy: u64,
    implicit: bool,
    distance: u64,
    distance_code: usize,
    cache: Cache,
}

impl Stream {
    /// The framing layer shares its outer live budget with this workspace.
    #[cfg(feature = "experimental")]
    pub(crate) fn set_framed_workspace_limit(&mut self, limit: Option<usize>) {
        self.memory.limit = limit;
    }

    pub(crate) const fn retained_bytes(&self) -> usize {
        self.memory.live
    }

    /// Transfers a completely collected member; its output has never wrapped.
    pub(crate) fn take_collected(&mut self) -> Vec<u8> {
        debug_assert!(self.position <= self.ring.len() as u64);
        let mut output = core::mem::take(&mut self.ring);
        self.memory.live -= output.capacity();
        output.truncate(self.position as usize);
        output
    }

    /// Copies the collected prefix before resuming with ordinary caller output.
    pub(crate) fn collected(&self) -> &[u8] {
        &self.ring[..self.position as usize]
    }

    /// Bytes the current meta-block still declares, as a reservation hint
    /// for callers growing a destination; zero outside a data meta-block.
    pub(crate) const fn declared_remaining(&self) -> u64 {
        match self.stage {
            Stage::Window | Stage::Meta | Stage::Metadata | Stage::EndBlock | Stage::End => 0,
            _ => self.remaining,
        }
    }

    pub(crate) fn reset(&mut self, config: DecoderConfig) {
        self.bits = Bits::default();
        self.stage = Stage::Window;
        if let Some(tables) = self.tables.first_mut() {
            tables.builder.reset();
        }
        self.prefix_history.clear();
        self.position = 0;
        self.window = None;
        self.cache = Cache::default();
        self.memory.limit = config.limits().max_workspace_bytes();
    }

    /// Grows the ring so positions below `end` are addressable, doubling up
    /// to the window size. A full ring wraps instead.
    fn ensure_ring(&mut self, end: u64) -> Result<(), DecodeError> {
        if end <= self.ring.len() as u64 {
            return Ok(());
        }
        grow_ring(&mut self.memory, &mut self.ring, self.window_size, end)
    }

    /// Grow with already known raw bytes instead of zeroing and overwriting.
    fn write_raw(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        let end = self.position + bytes.len() as u64;
        let old_len = self.ring.len();
        if end > old_len as u64 && end <= self.window_size {
            let desired = ring_size(old_len as u64, self.window_size, end)?;
            self.memory.reserve(&mut self.ring, desired)?;
            let existing = old_len - self.position as usize;
            self.ring[self.position as usize..].copy_from_slice(&bytes[..existing]);
            self.ring.extend_from_slice(&bytes[existing..]);
            self.ring.resize(desired, 0);
            self.position = end;
        } else {
            self.ensure_ring(end)?;
            write_ring(&mut self.ring, &mut self.position, bytes, None);
        }
        Ok(())
    }

    /// A unit-distance copy can initialize the new allocation with its final
    /// byte. Returns false when ordinary ring copying is needed (including wrap).
    fn repeat_growing(&mut self, end: u64) -> Result<bool, DecodeError> {
        if self.distance != 1 || end <= self.ring.len() as u64 || end > self.window_size {
            return Ok(false);
        }
        let byte = self.ring[self.position as usize - 1];
        repeat_grow(
            &mut self.memory,
            &mut self.ring,
            self.window_size,
            self.position,
            byte,
            end,
        )?;
        self.position = end;
        Ok(true)
    }

    fn ring_mask(&self) -> u64 {
        self.ring.len() as u64 - 1
    }

    fn emit(&mut self, byte: u8, output: &mut Output<'_>) -> Result<(), DecodeError> {
        let next = self
            .position
            .checked_add(1)
            .ok_or(DecodeError::SizeOverflow)?;
        if !output.linear {
            self.ensure_ring(next)?;
            let index = (self.position & self.ring_mask()) as usize;
            self.ring[index] = byte;
        }
        self.position = next;
        if output.collect.is_none() {
            output.bytes[output.produced] = byte;
        }
        output.produced += 1;
        self.remaining -= 1;
        Ok(())
    }

    /// The compressed meta-block workspace, created on first use. Its
    /// storage counts against the workspace budget like every other buffer.
    fn ensure_tables(&mut self) -> Result<&mut Tables, DecodeError> {
        if self.tables.is_empty() {
            self.memory.reserve(&mut self.tables, 1)?;
            self.tables.push(Tables::default());
        }
        self.tables
            .first_mut()
            .ok_or(DecodeError::InternalInvariant)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn run(
        &mut self,
        backend: crate::Backend,
        input: &mut Input<'_>,
        output: &mut Output<'_>,
        config: DecoderConfig,
        dictionary: Option<DictionaryRef<'_>>,
    ) -> Result<Stop, DecodeError> {
        let result = self.run_stages(backend, input, output, config, dictionary);
        // The fast path may pull whole speculative bytes into the reservoir.
        // Return read-ahead at output pauses too: a long pending copy can end
        // the member on a later call without accepting more input. Keeping its
        // read-ahead would then strand the next member in an older call.
        // Incomplete fields at input pauses still retain all accepted bytes.
        if matches!(result, Ok(Stop::Member | Stop::Output)) {
            self.bits.unread(input);
        } else {
            self.bits.settle();
        }
        result
    }

    /// Decodes whole commands while whole-word refills and output space
    /// allow, then leaves the resumable stage the byte-exact path continues from.
    ///
    /// Every hot quantity lives in a local for the duration of the loop and
    /// is written back only when the loop pauses. Decoded bytes accumulate in
    /// the ring and are delivered when a write reaches the ring end, when the
    /// loop pauses, or when an error is returned; the output space still free
    /// is `out_end` less what was delivered and what is pending, so no
    /// per-command delivery is needed to know whether a copy fits. State that
    /// depends only on the current block types is refreshed at block switches
    /// rather than per command.
    ///
    /// The outer state machine dispatches a feature-enabled function so these
    /// locals have their own register allocation. SIMD helpers inline here;
    /// the command loop never detects features or dispatches a copy.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    #[inline(always)]
    fn fast<S: Simd>(
        &mut self,
        simd: S,
        input: &mut Input<'_>,
        output: &mut Output<'_>,
        out_end: usize,
        dictionary: Option<DictionaryRef<'_>>,
    ) -> Result<(), DecodeError> {
        let Self {
            bits: saved_bits,
            stage,
            memory,
            ring: storage,
            position: saved_position,
            window_size,
            max_backward,
            remaining: saved_remaining,
            tables,
            literals: saved_literals,
            copy: saved_copy,
            implicit: saved_implicit,
            distance: saved_distance,
            distance_code: saved_distance_code,
            cache: saved_cache,
            ..
        } = self;
        let Some(Tables {
            blocks,
            modes,
            maps,
            trees,
            distances,
            distance_table,
            scratch,
            scratch_pos,
            scratch_len,
            ..
        }) = tables.first_mut()
        else {
            return Err(DecodeError::InternalInvariant);
        };
        let window_size = *window_size;
        let max_backward = *max_backward;
        let postfix = distances.postfix();
        // Bits a distance needs before its symbol: 15 for the symbol plus a
        // standard-window extra field of at most 25, or the 32-bit low half
        // that a large-window field is split at.
        let distance_need = if distances.is_large() { 47 } else { 40 };
        let distance_table = distances.table(distance_table);
        let mut bits = *saved_bits;
        let mut position = *saved_position;
        let mut remaining = *saved_remaining;
        let mut flushed = position;
        // The input cursor and the recent-distance cache live in locals too.
        let fast_input = &input.bytes[..input.fast_end()];
        let mut consumed = input.consumed;
        let mut cache = *saved_cache;
        // Command state the resumable stages read; written back on a pause.
        let mut literals = 0u64;
        let mut copy = 0u64;
        let mut implicit = false;
        let mut distance = 0u64;
        let mut distance_code = 0usize;
        // Output space still free, counting pending ring bytes as used; it
        // only ever shrinks, since delivering pending bytes does not change it.
        let mut space = out_end - output.produced;
        // A linear member decodes straight into the caller's slice, which is
        // taken out of `output` until the loop leaves; it never grows, since
        // `space` keeps every write below its bounded length.
        let linear = output.linear;
        let history_end = output.history_end(position, remaining, out_end);
        let linear_bytes: &mut [u8] = if linear {
            core::mem::take(&mut output.bytes)
        } else {
            &mut []
        };
        let mut ring: &mut [u8] = if linear {
            &mut linear_bytes[..history_end]
        } else {
            storage.as_mut_slice()
        };
        let mut ring_len = ring.len();
        let mut mask = history_mask(ring_len);
        // Positions below this need no growth: the ring length, or unbounded
        // once the ring has reached the window and wraps instead.
        let mut ring_limit = if linear || ring_len as u64 >= window_size {
            u64::MAX
        } else {
            ring_len as u64
        };
        let literal_map = maps[0].values.as_slice();
        let literal_tables = trees[0].tables();
        let command_tables = trees[1].tables();
        let distance_tables = trees[2].tables();
        let distance_map = maps[1].values.as_slice();
        let [literal_block, command_block, distance_block] = blocks;
        // Per-block-type command and distance tables, refreshed at switches.
        let mut command_type = command_block.current;
        let mut command_table = command_tables.table(command_type);
        let mut distance_type = usize::MAX;
        // Per-block-type literal state, refreshed at every literal block switch.
        let mut literal_type = usize::MAX;
        let mut lut = context_lut(0);
        let mut contexts: &[u8; 64] = &[0; 64];
        let mut trivial = false;
        let mut trivial_table = literal_tables.table(0);
        // Per-block-type distance context map slice.
        let mut distance_contexts = [0u8; 4];
        macro_rules! leave {
            () => {{
                flush_ring(ring, position, &mut flushed, output);
                if linear {
                    output.bytes = linear_bytes;
                }
                input.consumed = consumed;
                *saved_bits = bits;
                *saved_position = position;
                *saved_remaining = remaining;
                *saved_cache = cache;
                *saved_literals = literals;
                *saved_copy = copy;
                *saved_implicit = implicit;
                *saved_distance = distance;
                *saved_distance_code = distance_code;
            }};
        }
        macro_rules! pause {
            ($stage:expr) => {{
                leave!();
                *stage = $stage;
                return Ok(());
            }};
        }
        macro_rules! check {
            ($result:expr) => {
                match $result {
                    Ok(value) => value,
                    Err(error) => {
                        leave!();
                        return Err(error);
                    }
                }
            };
        }
        // Grows the ring for output up to `end`, and while at it for every
        // byte this call could still produce, so growth happens once per call
        // rather than once per doubling. The pending bytes keep their indices
        // because a ring only grows before it has wrapped.
        macro_rules! reach {
            ($end:expr) => {{
                let end: u64 = $end;
                if end > ring_limit {
                    let bound = position
                        .saturating_add(remaining.min(space as u64))
                        .max(end);
                    // A failed growth leaves the storage, and so the pending
                    // bytes, untouched; the slice is re-borrowed either way.
                    let grown = grow_ring(memory, storage, window_size, bound);
                    ring = storage.as_mut_slice();
                    ring_len = ring.len();
                    mask = history_mask(ring_len);
                    ring_limit = if ring_len as u64 >= window_size {
                        u64::MAX
                    } else {
                        ring_len as u64
                    };
                    check!(grown);
                }
            }};
        }
        // Entered at `Stage::Literals`, the loop resumes the pending insert
        // run of the command whose fields the stages saved, then continues.
        let mut resume = matches!(*stage, Stage::Literals);
        loop {
            let (insert, distance_context) = if resume {
                resume = false;
                literals = *saved_literals;
                copy = *saved_copy;
                implicit = *saved_implicit;
                (literals, copy.saturating_sub(2).min(3) as u8)
            } else {
                if remaining == 0 || !bits.refill_from(fast_input, &mut consumed) {
                    pause!(Stage::Command);
                }
                if command_block.remaining == 0 {
                    if !command_block.switch_ready() {
                        pause!(Stage::Command);
                    }
                    command_block.switch_fast(&mut bits);
                    if command_block.current != command_type {
                        command_type = command_block.current;
                        command_table = command_tables.table(command_type);
                    }
                    if !bits.refill_from(fast_input, &mut consumed) {
                        pause!(Stage::Command);
                    }
                }
                let symbol = command_table.decode_fast(&mut bits);
                command_block.remaining -= 1;
                let command = COMMANDS[symbol & (COMMAND_SLOTS - 1)];
                let insert =
                    u64::from(command.insert_base) + bits.take(u32::from(command.insert_extra));
                if insert > remaining {
                    check!(Err(InvalidDataKind::MetaBlock.into()));
                }
                implicit = command.implicit;
                literals = insert;
                if bits.count() < 24 && !bits.refill_from(fast_input, &mut consumed) {
                    pause!(Stage::CopyExtra(usize::from(command.copy_code)));
                }
                copy = u64::from(command.copy_base) + bits.take(u32::from(command.copy_extra));
                (insert, command.distance_context)
            };
            if insert != 0 {
                // `position + remaining` was validated at the meta-block header.
                reach!(position + insert);
                let outcome = loop {
                    if literals == 0 {
                        break Pause::Done;
                    }
                    if literal_block.remaining == 0 {
                        if !literal_block.switch_ready()
                            || (bits.count() < 54 && !bits.refill_from(fast_input, &mut consumed))
                        {
                            break Pause::Input;
                        }
                        literal_block.switch_fast(&mut bits);
                        continue;
                    }
                    if literal_block.current != literal_type {
                        literal_type = literal_block.current;
                        lut = context_lut(modes[literal_type]);
                        contexts = literal_map[literal_type * 64..]
                            .first_chunk::<64>()
                            .unwrap_or(&[0; 64]);
                        trivial = maps[0].trivial(literal_type);
                        trivial_table = literal_tables.table(usize::from(contexts[0]));
                    }
                    let index = (position & mask) as usize;
                    // A run bounded by every loop-invariant limit, so the run
                    // itself checks only the bit reservoir.
                    let run = literals
                        .min(literal_block.remaining)
                        .min(space as u64)
                        .min((ring_len - index) as u64) as usize;
                    if run == 0 {
                        break Pause::Output;
                    }
                    let mut done = 0;
                    if trivial {
                        // Three symbols need at most 45 bits. Amortize the
                        // reservoir check across a batch; short input and the
                        // final one or two literals use the scalar loop below.
                        for slots in ring[index..index + run].as_chunks_mut::<3>().0 {
                            if bits.count() < 45 && !bits.refill_from(fast_input, &mut consumed) {
                                break;
                            }
                            slots[0] = trivial_table.decode_fast(&mut bits) as u8;
                            slots[1] = trivial_table.decode_fast(&mut bits) as u8;
                            slots[2] = trivial_table.decode_fast(&mut bits) as u8;
                            done += 3;
                        }
                        for slot in ring[index + done..index + run].iter_mut() {
                            if bits.count() < 15 && !bits.refill_from(fast_input, &mut consumed) {
                                break;
                            }
                            *slot = trivial_table.decode_fast(&mut bits) as u8;
                            done += 1;
                        }
                    } else {
                        let (mut p1, mut p2) = previous_bytes(ring, position);
                        for slot in ring[index..index + run].iter_mut() {
                            if bits.count() < 15 && !bits.refill_from(fast_input, &mut consumed) {
                                break;
                            }
                            let context =
                                usize::from(lut[usize::from(p1)] | lut[256 + usize::from(p2)]);
                            let tree = usize::from(contexts[context & 63]);
                            let byte = literal_tables.table(tree).decode_fast(&mut bits) as u8;
                            *slot = byte;
                            p2 = p1;
                            p1 = byte;
                            done += 1;
                        }
                    }
                    position += done as u64;
                    literals -= done as u64;
                    remaining -= done as u64;
                    space -= done;
                    literal_block.remaining -= done as u64;
                    if index + done == ring_len {
                        flush_ring(ring, position, &mut flushed, output);
                    }
                    if done < run {
                        break Pause::Input;
                    }
                };
                if outcome != Pause::Done {
                    pause!(Stage::Literals);
                }
            }
            if remaining == 0 {
                pause!(Stage::Command);
            }
            distance = if implicit {
                distance_code = 0;
                // Re-pushed below, which keeps the cache unchanged.
                cache.pop()
            } else {
                if bits.count() < distance_need && !bits.refill_from(fast_input, &mut consumed) {
                    pause!(Stage::Distance);
                }
                if distance_block.remaining == 0 {
                    if !distance_block.switch_ready() {
                        pause!(Stage::Distance);
                    }
                    distance_block.switch_fast(&mut bits);
                    if !bits.refill_from(fast_input, &mut consumed) {
                        pause!(Stage::Distance);
                    }
                }
                if distance_block.current != distance_type {
                    distance_type = distance_block.current;
                    distance_contexts = distance_map[distance_type * 4..]
                        .first_chunk::<4>()
                        .copied()
                        .unwrap_or([0; 4]);
                }
                let tree = usize::from(distance_contexts[usize::from(distance_context) & 3]);
                let symbol = distance_tables.table(tree).decode_fast(&mut bits);
                distance_block.remaining -= 1;
                distance_code = symbol;
                if symbol == 0 {
                    // Explicit "last distance": no push, like the implicit one.
                    cache.pop()
                } else if symbol < 16 {
                    check!(DistanceLayout::short(symbol, &cache))
                } else {
                    let entry = distance_table[symbol];
                    let width = entry.width as u32;
                    let extra = if width > 32 {
                        if consumed + 16 > fast_input.len()
                            || !bits.refill_from(fast_input, &mut consumed)
                        {
                            pause!(Stage::DistanceExtra(symbol));
                        }
                        let low = bits.take(32);
                        if !bits.refill_from(fast_input, &mut consumed) {
                            check!(Err(DecodeError::InternalInvariant));
                        }
                        low | (bits.take(width - 32) << 32)
                    } else {
                        bits.take(width)
                    };
                    entry.base + (extra << postfix)
                }
            };
            let available = position.min(max_backward);
            if distance > available {
                if distance_code == 0 {
                    // Neither a prefix nor a dictionary reference enters the
                    // cache, and the byte-exact stages expect it whole.
                    cache.push(distance);
                }
                let prefix_len = dictionary.map_or(0, DictionaryRef::prefix_len);
                if distance - available <= prefix_len {
                    pause!(Stage::Resolve);
                }
                let (p1, p2) = previous_bytes(ring, position);
                let lut = context_lut(modes[literal_block.current]);
                let context = usize::from(lut[usize::from(p1)] | lut[256 + usize::from(p2)]);
                let length = check!(dictionary::resolve(
                    dictionary,
                    distance - available - prefix_len - 1,
                    copy as usize,
                    context,
                    scratch,
                ));
                if length as u64 > remaining {
                    check!(Err(InvalidDataKind::MetaBlock.into()));
                }
                if length == 0 && distance <= 120 {
                    check!(Err(InvalidDataKind::DictionaryReference.into()));
                }
                *scratch_len = length;
                *scratch_pos = 0;
                if length > space {
                    pause!(Stage::Dictionary);
                }
                reach!(position + length as u64);
                write_ring(
                    ring,
                    &mut position,
                    &scratch[..length],
                    Some((&mut flushed, &mut *output)),
                );
                remaining -= length as u64;
                space -= length;
            } else {
                if copy > remaining {
                    check!(Err(InvalidDataKind::MetaBlock.into()));
                }
                // An implicit distance re-enters the slot it was popped from;
                // an explicit one takes a new slot, both as one write. This
                // precedes the output check because `Stage::Copy` resumes the
                // copy without touching the cache.
                cache.push(distance);
                if copy > space as u64 {
                    pause!(Stage::Copy);
                }
                // A unit-distance run that must grow the ring is a fill: grow
                // with the repeated byte in one pass instead of zeroing the new
                // region and overwriting it. Runs that fit, or that wrap past
                // the window, take the general copy below.
                let end = position + copy;
                if distance == 1 && end > ring_limit && end <= window_size {
                    let byte = ring[((position - 1) & mask) as usize];
                    // Bind the result and re-borrow before `check!`, whose
                    // failure path flushes through `ring`.
                    let grown = repeat_grow(memory, storage, window_size, position, byte, end);
                    ring = storage.as_mut_slice();
                    ring_len = check!(grown);
                    mask = history_mask(ring_len);
                    ring_limit = if ring_len as u64 >= window_size {
                        u64::MAX
                    } else {
                        ring_len as u64
                    };
                    position = end;
                    remaining -= copy;
                    space -= copy as usize;
                    if position == ring_len as u64 {
                        flush_ring(ring, position, &mut flushed, output);
                    }
                    continue;
                }
                reach!(position + copy);
                let len = copy as usize;
                let dst = (position & mask) as usize;
                let src = ((position - distance) & mask) as usize;
                if dst.max(src) + len <= ring_len {
                    if distance >= copy {
                        if len <= 16 {
                            if !copy16(simd, ring, src, dst) {
                                ring.copy_within(src..src + len, dst);
                            }
                        } else if len > 32 || !copy32(simd, ring, src, dst) {
                            ring.copy_within(src..src + len, dst);
                        }
                    } else {
                        copy_overlapping(ring, dst, src, len, distance as usize);
                    }
                    position += copy;
                    if dst + len == ring_len {
                        flush_ring(ring, position, &mut flushed, output);
                    }
                } else {
                    copy_ring(
                        simd,
                        ring,
                        &mut position,
                        distance,
                        len,
                        &mut flushed,
                        output,
                    );
                }
                remaining -= copy;
                space -= len;
            }
        }
    }

    fn run_stages(
        &mut self,
        backend: crate::Backend,
        input: &mut Input<'_>,
        output: &mut Output<'_>,
        config: DecoderConfig,
        dictionary: Option<DictionaryRef<'_>>,
    ) -> Result<Stop, DecodeError> {
        let fast_end = input.fast_end();
        let out_end = output.fast_end();
        macro_rules! read {
            ($n:expr) => {
                match self.bits.read($n, input)? {
                    Some(v) => v,
                    None => return Ok(Stop::Input),
                }
            };
        }
        // Every stage inside a compressed meta-block runs after `Stage::Meta`
        // created the workspace; the stage machine is private, so a missing
        // one is an internal invariant failure rather than a format error.
        macro_rules! tables {
            () => {
                match self.tables.first_mut() {
                    Some(tables) => tables,
                    None => return Err(DecodeError::InternalInvariant),
                }
            };
        }
        loop {
            match self.stage {
                Stage::Window => {
                    let Some(window) =
                        header::window(&mut self.bits, input, config.window_limit())?
                    else {
                        return Ok(Stop::Input);
                    };
                    self.large = window.encoding() == WindowEncoding::Large;
                    self.window_size = 1u64 << window.bits();
                    self.max_backward = self.window_size - 16;
                    self.window = Some(window);
                    self.stage = Stage::Meta;
                }
                Stage::Meta => {
                    let Some(header) = header::metablock(&mut self.bits, input)? else {
                        return Ok(Stop::Input);
                    };
                    // Every position reached inside the meta-block stays
                    // below this sum, so the hot path adds without checks.
                    if let MetaBlock::Uncompressed { length } | MetaBlock::Compressed { length, .. } =
                        header
                        && self.position.checked_add(length).is_none()
                    {
                        return Err(DecodeError::SizeOverflow);
                    }
                    match header {
                        MetaBlock::End => self.stage = Stage::End,
                        MetaBlock::Uncompressed { length } => {
                            self.remaining = length;
                            self.last = false;
                            self.stage = Stage::Raw;
                        }
                        MetaBlock::Metadata { length, last } => {
                            self.remaining = length;
                            self.last = last;
                            self.stage = Stage::Metadata;
                        }
                        MetaBlock::Compressed { length, last } => {
                            self.remaining = length;
                            self.last = last;
                            let tables = self.ensure_tables()?;
                            for block in &mut tables.blocks {
                                block.reset();
                            }
                            for map in &mut tables.maps {
                                map.reset();
                            }
                            self.stage = Stage::Blocks(0);
                        }
                    }
                }
                Stage::Metadata => {
                    if self.remaining == 0 {
                        self.stage = Stage::EndBlock;
                        continue;
                    }
                    if self.bits.count() >= 8 {
                        self.bits.take(8);
                        self.remaining -= 1;
                        continue;
                    }
                    let skip = self.remaining.min((fast_end - input.consumed) as u64) as usize;
                    if skip == 0 {
                        read!(8);
                        self.remaining -= 1;
                        continue;
                    }
                    // The bytes bypass the reservoir, so it must stop
                    // holding them as bits a whole-word refill read ahead.
                    self.bits.settle();
                    input.consumed += skip;
                    self.remaining -= skip as u64;
                }
                Stage::Raw => {
                    if self.remaining == 0 {
                        self.stage = Stage::EndBlock;
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    if self.bits.count() >= 8 {
                        let byte = self.bits.take(8) as u8;
                        self.emit(byte, output)?;
                        continue;
                    }
                    let count = self
                        .remaining
                        .min((fast_end - input.consumed) as u64)
                        .min((out_end - output.produced) as u64)
                        as usize;
                    if count == 0 {
                        let byte = read!(8) as u8;
                        self.emit(byte, output)?;
                        continue;
                    }
                    // As for metadata, the copied bytes bypass the reservoir.
                    self.bits.settle();
                    let bytes = &input.bytes[input.consumed..input.consumed + count];
                    if output.linear {
                        self.position += count as u64;
                    } else {
                        self.write_raw(bytes)?;
                    }
                    if output.collect.is_none() {
                        output.bytes[output.produced..output.produced + count]
                            .copy_from_slice(bytes);
                    }
                    input.consumed += count;
                    output.produced += count;
                    self.remaining -= count as u64;
                }
                Stage::Blocks(i) => {
                    let tables = tables!();
                    if !tables.blocks[i].header(
                        &mut self.bits,
                        input,
                        &mut tables.builder,
                        &mut self.memory,
                    )? {
                        return Ok(Stop::Input);
                    }
                    self.stage = if i == 2 {
                        Stage::DistanceParams
                    } else {
                        Stage::Blocks(i + 1)
                    };
                }
                Stage::DistanceParams => {
                    let params = read!(6);
                    let tables = tables!();
                    tables.distances = DistanceLayout::from_header(params as u8, self.large);
                    if tables.distance_table_layout != Some(tables.distances)
                        && !tables.distances.is_standard()
                    {
                        tables.distance_table_layout = None;
                        tables
                            .distances
                            .fill_table(&mut tables.distance_table, &mut self.memory)?;
                        tables.distance_table_layout = Some(tables.distances);
                    }
                    self.stage = Stage::Modes(0);
                }
                Stage::Modes(i) => {
                    let value = read!(2) as u8;
                    let tables = tables!();
                    tables.modes[i] = value;
                    self.stage = if i + 1 == tables.blocks[0].count {
                        Stage::Maps(0)
                    } else {
                        Stage::Modes(i + 1)
                    };
                }
                Stage::Maps(i) => {
                    let tables = tables!();
                    let size = tables.blocks[if i == 0 { 0 } else { 2 }].count
                        << if i == 0 { 6 } else { 2 };
                    if !tables.maps[i].read(
                        size,
                        &mut self.bits,
                        input,
                        &mut tables.builder,
                        &mut self.memory,
                    )? {
                        return Ok(Stop::Input);
                    }
                    if i == 0 {
                        tables.maps[0].detect_trivial();
                        self.stage = Stage::Maps(1);
                    } else {
                        let counts = [
                            tables.maps[0].trees,
                            tables.blocks[1].count,
                            tables.maps[1].trees,
                        ];
                        let alphabets = [256, 704, tables.distances.alphabet()];
                        for (group, (count, alphabet)) in tables
                            .trees
                            .iter_mut()
                            .zip(counts.into_iter().zip(alphabets))
                        {
                            group.prepare(count, alphabet, &mut self.memory)?;
                        }
                        self.stage = Stage::Trees(0, 0);
                    }
                }
                Stage::Trees(group, index) => {
                    let tables = tables!();
                    let alphabet = match group {
                        0 => 256,
                        1 => 704,
                        _ => tables.distances.alphabet(),
                    };
                    if !tables.builder.read(alphabet, &mut self.bits, input)? {
                        return Ok(Stop::Input);
                    }
                    let max_symbol = tables.builder.build_slot(
                        alphabet,
                        &mut tables.trees[group],
                        index,
                        &mut self.memory,
                    )?;
                    if group == 2 {
                        tables.distances.validate_symbol(max_symbol)?;
                    }
                    self.stage = if index + 1 < tables.trees[group].count() {
                        Stage::Trees(group, index + 1)
                    } else if group < 2 {
                        Stage::Trees(group + 1, 0)
                    } else {
                        Stage::Command
                    };
                }
                Stage::Command => {
                    if self.remaining == 0 {
                        self.stage = Stage::EndBlock;
                        continue;
                    }
                    dispatch!(backend.0, simd => self.fast(simd, input, output, out_end, dictionary))?;
                    if !matches!(self.stage, Stage::Command) {
                        continue;
                    }
                    if self.remaining == 0 {
                        self.stage = Stage::EndBlock;
                        continue;
                    }
                    let tables = tables!();
                    if !tables.blocks[1].prepare(&mut self.bits, input)? {
                        return Ok(Stop::Input);
                    }
                    let Some(symbol) = tables.trees[1]
                        .table(tables.blocks[1].current)
                        .decode(&mut self.bits, input)?
                    else {
                        return Ok(Stop::Input);
                    };
                    tables.blocks[1].advance();
                    let cell = CELLS[symbol >> 6];
                    let insert = (cell & 24) + ((symbol >> 3) & 7);
                    let copy = ((cell << 3) & 24) + (symbol & 7);
                    self.implicit = symbol < 128;
                    self.stage = Stage::InsertExtra(insert, copy);
                }
                Stage::InsertExtra(insert, copy) => {
                    self.literals = u64::from(INS_BASE[insert]) + read!(INS_EXTRA[insert]);
                    if self.literals > self.remaining {
                        return Err(InvalidDataKind::MetaBlock.into());
                    }
                    self.stage = Stage::CopyExtra(copy);
                }
                Stage::CopyExtra(copy) => {
                    self.copy = u64::from(COPY_BASE[copy]) + read!(COPY_EXTRA[copy]);
                    self.stage = Stage::Literals;
                }
                Stage::Literals => {
                    if self.literals == 0 {
                        self.stage = if self.remaining == 0 {
                            Stage::EndBlock
                        } else {
                            Stage::Distance
                        };
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    // Resume the run in bulk while whole-word refills and
                    // output room allow; only its byte-exact remainder is
                    // decoded one symbol at a time below. Entering the bulk
                    // loop is worth it only if it can refill at all.
                    if input.consumed + 8 <= fast_end {
                        dispatch!(backend.0, simd => self.fast(simd, input, output, out_end, dictionary))?;
                        if !matches!(self.stage, Stage::Literals) || self.literals == 0 {
                            continue;
                        }
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    let tables = tables!();
                    if !tables.blocks[0].prepare(&mut self.bits, input)? {
                        return Ok(Stop::Input);
                    }
                    if input.consumed + 8 <= fast_end {
                        // The bulk loop stopped at a boundary it does not
                        // cross; one byte here lets it resume after it.
                        let context = context(tables, history(&self.ring, output), self.position);
                        let tree = usize::from(
                            tables.maps[0].values[tables.blocks[0].current * 64 + context],
                        );
                        let Some(byte) =
                            tables.trees[0].table(tree).decode(&mut self.bits, input)?
                        else {
                            return Ok(Stop::Input);
                        };
                        tables.blocks[0].advance();
                        self.emit(byte as u8, output)?;
                        self.literals -= 1;
                        continue;
                    }
                    // The input tail: decode byte-exactly, but settle the
                    // block, context mode and output room once per run. The
                    // `ready` check above means the room is at least one.
                    let room = out_end.saturating_sub(output.produced) as u64;
                    let run = self
                        .literals
                        .min(tables.blocks[0].remaining)
                        .min(room.max(1));
                    let lut = context_lut(tables.modes[tables.blocks[0].current]);
                    let row = tables.blocks[0].current * 64;
                    let map = tables.maps[0]
                        .values
                        .get(row..row + 64)
                        .ok_or(DecodeError::InternalInvariant)?;
                    let trees = tables.trees[0].tables();
                    let mut done = 0;
                    // Every emitted byte is counted below before any pause or
                    // error leaves, exactly as the per-symbol path would.
                    let outcome = loop {
                        if done == run {
                            break Ok(true);
                        }
                        let (p1, p2) = previous_bytes(history(&self.ring, output), self.position);
                        let context =
                            usize::from(lut[usize::from(p1)] | lut[256 + usize::from(p2)]);
                        let byte = match trees
                            .table(usize::from(map[context]))
                            .decode(&mut self.bits, input)
                        {
                            Ok(Some(symbol)) => symbol as u8,
                            Ok(None) => break Ok(false),
                            Err(error) => break Err(error),
                        };
                        let Some(next) = self.position.checked_add(1) else {
                            break Err(DecodeError::SizeOverflow);
                        };
                        if !output.linear {
                            if next > self.ring.len() as u64
                                && let Err(error) = grow_ring(
                                    &mut self.memory,
                                    &mut self.ring,
                                    self.window_size,
                                    next,
                                )
                            {
                                break Err(error);
                            }
                            let mask = self.ring.len() as u64 - 1;
                            self.ring[(self.position & mask) as usize] = byte;
                        }
                        self.position = next;
                        if output.collect.is_none() {
                            output.bytes[output.produced] = byte;
                        }
                        output.produced += 1;
                        done += 1;
                    };
                    tables.blocks[0].remaining -= done;
                    self.remaining -= done;
                    self.literals -= done;
                    if !outcome? {
                        return Ok(Stop::Input);
                    }
                }
                Stage::Distance => {
                    if self.implicit {
                        self.distance_code = 0;
                        self.distance = self.cache.recent(0);
                        self.stage = Stage::Resolve;
                        continue;
                    }
                    let tables = tables!();
                    if !tables.blocks[2].prepare(&mut self.bits, input)? {
                        return Ok(Stop::Input);
                    }
                    let context = self.copy.saturating_sub(2).min(3) as usize;
                    let tree =
                        usize::from(tables.maps[1].values[tables.blocks[2].current * 4 + context]);
                    let Some(symbol) = tables.trees[2].table(tree).decode(&mut self.bits, input)?
                    else {
                        return Ok(Stop::Input);
                    };
                    tables.blocks[2].advance();
                    self.distance_code = symbol;
                    self.stage = Stage::DistanceExtra(symbol);
                }
                Stage::DistanceExtra(symbol) => {
                    let distances = tables!().distances;
                    let width = distances.extra_bits(symbol);
                    if width > 32 {
                        let low = read!(32);
                        self.stage = Stage::DistanceExtraHigh(symbol, low);
                        continue;
                    }
                    let extra = read!(width);
                    self.distance = distances.resolve(symbol, extra, &self.cache)?;
                    self.stage = Stage::Resolve;
                }
                Stage::DistanceExtraHigh(symbol, low) => {
                    let distances = tables!().distances;
                    let high = read!(distances.extra_bits(symbol) - 32);
                    let extra = low | (high << 32);
                    self.distance = distances.resolve(symbol, extra, &self.cache)?;
                    self.stage = Stage::Resolve;
                }
                Stage::Resolve => {
                    let available = self.position.min(self.max_backward);
                    let prefix_len = dictionary.map_or(0, DictionaryRef::prefix_len);
                    if self.distance > available && self.distance - available <= prefix_len {
                        let offset = prefix_len - (self.distance - available);
                        if self.copy > self.remaining {
                            return Err(InvalidDataKind::DictionaryReference.into());
                        }
                        if self.distance > self.max_backward {
                            let crossing = self.copy.saturating_sub(prefix_len - offset);
                            let length = usize::try_from(crossing.min(available))
                                .map_err(|_| DecodeError::SizeOverflow)?;
                            self.memory.resize(&mut self.prefix_history, length)?;
                            if length != 0 {
                                let history = history(&self.ring, output);
                                let mask = history_mask(history.len());
                                let start = self.position - available;
                                for (i, byte) in self.prefix_history.iter_mut().enumerate() {
                                    *byte = history[((start + i as u64) & mask) as usize];
                                }
                            }
                        }
                        if self.distance_code != 0 {
                            self.cache.push(self.distance);
                        }
                        self.stage = Stage::Prefix {
                            offset,
                            start: offset,
                        };
                    } else if self.distance > available {
                        let tables = tables!();
                        let context = context(tables, history(&self.ring, output), self.position);
                        tables.scratch_len = dictionary::resolve(
                            dictionary,
                            self.distance - available - prefix_len - 1,
                            self.copy as usize,
                            context,
                            &mut tables.scratch,
                        )?;
                        if tables.scratch_len as u64 > self.remaining {
                            return Err(InvalidDataKind::MetaBlock.into());
                        }
                        // RFC/C reject these zero-output references: they can
                        // otherwise repeat using exclusively zero-bit trees.
                        if tables.scratch_len == 0 && self.distance <= 120 {
                            return Err(InvalidDataKind::DictionaryReference.into());
                        }
                        tables.scratch_pos = 0;
                        self.stage = Stage::Dictionary;
                    } else {
                        if self.copy > self.remaining {
                            return Err(InvalidDataKind::MetaBlock.into());
                        }
                        if self.distance_code != 0 {
                            self.cache.push(self.distance);
                        }
                        self.stage = Stage::Copy;
                    }
                }
                Stage::Copy => {
                    if self.copy == 0 {
                        self.stage = Stage::Command;
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    // Bulk-copy the run even when the fast path could not run
                    // (for example near the end of a small, highly expanding
                    // input). `out_end` already folds in slice length, output
                    // limits and the exact-size contract.
                    let n = self.copy.min((out_end - output.produced) as u64);
                    let end = self
                        .position
                        .checked_add(n)
                        .ok_or(DecodeError::SizeOverflow)?;
                    let mut flushed = self.position;
                    if output.linear {
                        let history_end =
                            output.history_end(self.position, self.remaining, out_end);
                        let bytes = core::mem::take(&mut output.bytes);
                        dispatch!(backend.0, simd => copy_ring(
                            simd,
                            &mut bytes[..history_end],
                            &mut self.position,
                            self.distance,
                            n as usize,
                            &mut flushed,
                            output,
                        ));
                        output.bytes = bytes;
                    } else if !self.repeat_growing(end)? {
                        self.ensure_ring(end)?;
                        dispatch!(backend.0, simd => copy_ring(
                            simd,
                            &mut self.ring,
                            &mut self.position,
                            self.distance,
                            n as usize,
                            &mut flushed,
                            output,
                        ));
                    }
                    flush_ring(&self.ring, self.position, &mut flushed, output);
                    self.copy -= n;
                    self.remaining -= n;
                    if self.copy != 0 {
                        output.ready()?;
                        return Ok(Stop::Output);
                    }
                    self.stage = Stage::Command;
                }
                Stage::Prefix { offset, start } => {
                    if self.copy == 0 {
                        self.stage = Stage::Command;
                        continue;
                    }
                    if offset == dictionary.map_or(0, DictionaryRef::prefix_len) {
                        self.stage = if self.distance <= self.max_backward {
                            Stage::Copy
                        } else {
                            Stage::PrefixHistory { offset: 0, start }
                        };
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    // Emit a contiguous prefix run rather than one byte at a
                    // time. The run stops at its segment end, the copy length,
                    // the prefix end and the current output room.
                    let run = dictionary.map_or(&[][..], |value| value.prefix_run(offset));
                    if run.is_empty() {
                        return Err(InvalidDataKind::DictionaryReference.into());
                    }
                    let n = (run.len() as u64)
                        .min(self.copy)
                        .min((out_end - output.produced) as u64)
                        as usize;
                    let end = self
                        .position
                        .checked_add(n as u64)
                        .ok_or(DecodeError::SizeOverflow)?;
                    let mut flushed = self.position;
                    if output.linear {
                        let start = output.produced;
                        output.bytes[start..start + n].copy_from_slice(&run[..n]);
                        self.position = end;
                    } else {
                        self.ensure_ring(end)?;
                        write_ring(
                            &mut self.ring,
                            &mut self.position,
                            &run[..n],
                            Some((&mut flushed, &mut *output)),
                        );
                    }
                    flush_ring(&self.ring, self.position, &mut flushed, output);
                    self.copy -= n as u64;
                    self.remaining -= n as u64;
                    self.stage = Stage::Prefix {
                        offset: offset + n as u64,
                        start,
                    };
                }
                Stage::PrefixHistory { offset, start } => {
                    if self.copy == 0 {
                        self.stage = Stage::Command;
                        continue;
                    }
                    if offset == self.prefix_history.len() {
                        self.stage = Stage::Prefix {
                            offset: start,
                            start,
                        };
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    self.emit(self.prefix_history[offset], output)?;
                    self.copy -= 1;
                    self.stage = Stage::PrefixHistory {
                        offset: offset + 1,
                        start,
                    };
                }
                Stage::Dictionary => {
                    let tables = tables!();
                    if tables.scratch_pos == tables.scratch_len {
                        self.stage = Stage::Command;
                        continue;
                    }
                    if !output.ready()? {
                        return Ok(Stop::Output);
                    }
                    let byte = tables.scratch[tables.scratch_pos];
                    tables.scratch_pos += 1;
                    self.emit(byte, output)?;
                }
                Stage::EndBlock => {
                    self.stage = if self.last { Stage::End } else { Stage::Meta };
                }
                Stage::End => {
                    self.bits.align()?;
                    return Ok(Stop::Member);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::{DecodeDictionary, DecodeDictionaryLimits, DictionaryAttachment};

    #[test]
    fn raw_growth_initializes_only_padding_and_preserves_wrapped_history() {
        let mut stream = Stream {
            window_size: 128,
            ..Stream::default()
        };
        stream.write_raw(b"abc").unwrap();
        assert_eq!(&stream.ring[..3], b"abc");
        assert_eq!(stream.ring.len(), 64);
        stream.write_raw(&[7; 65]).unwrap();
        assert_eq!(&stream.ring[3..68], &[7; 65]);
        assert_eq!(&stream.ring[68..], &[0; 60]);
        stream.write_raw(&[9; 65]).unwrap();
        assert_eq!(stream.position, 133);
        assert_eq!(&stream.ring[..5], &[9; 5]);
        assert_eq!(stream.memory.live, stream.ring.capacity());
        let output = stream.ring.clone();
        stream.window_size = 256;
        stream.memory.limit = Some(stream.memory.live);
        // Simulate the pre-wrap growth boundary, with an exhausted budget.
        stream.position = 128;
        assert!(matches!(
            stream.write_raw(&[1; 10]),
            Err(DecodeError::MemoryLimitExceeded { .. })
        ));
        assert_eq!(stream.ring, output);
        assert_eq!(stream.position, 128);
    }

    #[test]
    fn repeat_growth_fills_new_storage_once_and_keeps_failure_atomic() {
        let mut stream = Stream {
            window_size: 256,
            distance: 1,
            ..Stream::default()
        };
        stream.write_raw(b"x").unwrap();
        assert!(!stream.repeat_growing(63).unwrap());
        stream.memory.limit = Some(stream.memory.live);
        assert!(stream.repeat_growing(129).is_err());
        assert_eq!(stream.position, 1);
        assert_eq!(&stream.ring[1..], &[0; 63]);
        stream.memory.limit = None;
        assert!(stream.repeat_growing(129).unwrap());
        assert_eq!(&stream.ring[..], &[b'x'; 256]);
        assert_eq!(stream.position, 129);
        assert!(!stream.repeat_growing(257).unwrap());
        stream.distance = 2;
        assert!(!stream.repeat_growing(256).unwrap());
    }

    #[test]
    fn ring_sizing_caps_doubling_at_the_window() {
        assert_eq!(ring_size(0, 1024, 1).unwrap(), 64);
        assert_eq!(ring_size(64, 1024, 65).unwrap(), 128);
        assert_eq!(ring_size(64, 1024, 900).unwrap(), 1024);
        assert_eq!(ring_size(64, 1024, u64::MAX).unwrap(), 1024);
    }

    #[test]
    fn prefix_references_continue_through_history_and_overlap() {
        for (window_size, maximum) in [(8u64, 4u64), (1024, 1008)] {
            let dictionary = DecodeDictionary::new(
                &[DictionaryAttachment::Raw(b"xy")],
                DecodeDictionaryLimits::default(),
            )
            .unwrap();
            let mut ring = b"abcd".to_vec();
            ring.resize(8, 0);
            let mut stream = Stream {
                stage: Stage::Resolve,
                window_size,
                max_backward: maximum,
                position: 4,
                ring,
                distance: 6,
                copy: 15,
                remaining: 15,
                last: true,
                ..Stream::default()
            };
            stream.memory.live = stream.ring.capacity();
            let mut input = Input::new(&[], 0, None);
            let mut decoded = Vec::new();
            loop {
                let mut bytes = [0; 1];
                let mut output = Output {
                    collect: None,
                    linear: false,
                    bytes: &mut bytes,
                    produced: 0,
                    total_before: decoded.len() as u64,
                    limit: None,
                    exact: OutputSize::Unknown,
                };
                let result = stream
                    .run(
                        crate::Backend::SCALAR,
                        &mut input,
                        &mut output,
                        DecoderConfig::default(),
                        Some((&dictionary).into()),
                    )
                    .unwrap();
                decoded.extend_from_slice(&output.bytes[..output.produced]);
                if result == Stop::Member {
                    break;
                }
                assert_eq!(result, Stop::Output);
            }
            assert_eq!(decoded, b"xyabcdxyabcdxya");
        }
    }

    #[test]
    fn ring_copies_replicate_patterns_and_wrap_without_per_byte_work() {
        // Reference: byte-at-a-time copy through a masked ring.
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut random = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for backend in crate::Backend::available() {
            for size in [64usize, 256] {
                for _ in 0..400 {
                    let mut expected = alloc::vec![0u8; size];
                    for byte in &mut expected {
                        *byte = random() as u8;
                    }
                    let mut ring = expected.clone();
                    let position = 3 * size as u64 + (random() % size as u64);
                    let distance = 1 + random() % (size as u64 - 16);
                    let length = 1 + (random() as usize) % 100;
                    let mut reference_position = position;
                    let mut produced = alloc::vec![0u8; length];
                    for byte in &mut produced {
                        let source = expected
                            [((reference_position - distance) & (size as u64 - 1)) as usize];
                        expected[(reference_position & (size as u64 - 1)) as usize] = source;
                        // The decoder emits each byte before its ring slot is reused.
                        *byte = source;
                        reference_position += 1;
                    }
                    let mut sink = alloc::vec![0u8; length];
                    let mut output = Output {
                        collect: None,
                        linear: false,
                        bytes: &mut sink,
                        produced: 0,
                        total_before: 0,
                        limit: None,
                        exact: OutputSize::Unknown,
                    };
                    let mut fast_position = position;
                    let mut flushed = position;
                    dispatch!(backend.0, simd => copy_ring(
                        simd,
                        &mut ring,
                        &mut fast_position,
                        distance,
                        length,
                        &mut flushed,
                        &mut output,
                    ));
                    flush_ring(&ring, fast_position, &mut flushed, &mut output);
                    assert_eq!(fast_position, reference_position);
                    assert_eq!(output.produced, length);
                    // Only the copied bytes and the 16 unreachable slots ahead may differ.
                    for (i, (a, b)) in ring.iter().zip(&expected).enumerate() {
                        let ahead = (i as u64).wrapping_sub(fast_position) & (size as u64 - 1);
                        assert!(a == b || ahead < 16, "slot {i} differs");
                    }
                    assert_eq!(sink, produced);
                }
            }
        }
    }

    #[test]
    fn scalar_and_host_backends_decode_goldens_with_output_backpressure() {
        use crate::{Backend, DecodeOperation, DecodeStreamConfig, DecoderStatus, Decompressor};
        let cases: &[(&[u8], &[u8])] = &[
            (
                include_bytes!(
                    "../../../brotli-ffi/vendor/brotli/tests/testdata/alice29.txt.compressed"
                ),
                include_bytes!("../../../brotli-ffi/vendor/brotli/tests/testdata/alice29.txt"),
            ),
            (
                include_bytes!(
                    "../../../brotli-ffi/vendor/brotli/tests/testdata/quickfox_repeated.compressed"
                ),
                include_bytes!(
                    "../../../brotli-ffi/vendor/brotli/tests/testdata/quickfox_repeated"
                ),
            ),
        ];
        for backend in Backend::available() {
            let mut decoder = Decompressor::builder(DecoderConfig::default())
                .with_backend(backend)
                .build()
                .unwrap();
            for &(encoded, expected) in cases {
                assert_eq!(decoder.decompress(encoded).unwrap(), expected);
                for capacity in [1, 15, 16, 17, 31, 32, 33, 127, 65536] {
                    let mut session = decoder.start(DecodeStreamConfig::default()).unwrap();
                    let mut output = alloc::vec![0; capacity];
                    let mut result = Vec::new();
                    let mut cursor = 0;
                    loop {
                        let progress = session
                            .process(&encoded[cursor..], &mut output, DecodeOperation::Finish)
                            .unwrap();
                        cursor += progress.consumed;
                        result.extend_from_slice(&output[..progress.produced]);
                        if progress.status == DecoderStatus::Finished {
                            break;
                        }
                        assert!(progress.consumed != 0 || progress.produced != 0);
                    }
                    assert_eq!(cursor, encoded.len());
                    assert_eq!(result, expected, "{backend}, capacity {capacity}");
                }
            }
        }
    }

    #[test]
    fn vector_copies_match_snapshots_for_every_host_backend_and_ring_boundary() {
        for backend in crate::Backend::available() {
            for size in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97] {
                let original: Vec<u8> = (0..size).map(|i| (i * 37) as u8).collect();
                for src in 0..=size + 1 {
                    for dst in 0..=size + 1 {
                        for width in [16, 32] {
                            let mut actual = original.clone();
                            let mut expected = original.clone();
                            let fits = src + width <= size && dst + width <= size;
                            if fits {
                                expected[dst..dst + width]
                                    .copy_from_slice(&original[src..src + width]);
                            }
                            let copied = dispatch!(backend.0, simd => {
                                if width == 16 { copy16(simd, &mut actual, src, dst) }
                                else { copy32(simd, &mut actual, src, dst) }
                            });
                            assert_eq!(copied, fits, "{backend}, {size}, {src}, {dst}, {width}");
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn command_table_matches_the_cell_formula_for_every_symbol() {
        assert_eq!(commands()[..], COMMANDS[..]);
        for symbol in 0..704 {
            let cell = CELLS[symbol >> 6];
            let insert = (cell & 24) + ((symbol >> 3) & 7);
            let copy = ((cell << 3) & 24) + (symbol & 7);
            let command = COMMANDS[symbol];
            assert_eq!(u32::from(command.insert_base), INS_BASE[insert]);
            assert_eq!(u32::from(command.insert_extra), INS_EXTRA[insert]);
            assert_eq!(u32::from(command.copy_base), COPY_BASE[copy]);
            assert_eq!(u32::from(command.copy_extra), COPY_EXTRA[copy]);
            assert_eq!(usize::from(command.copy_code), copy);
            assert_eq!(command.implicit, symbol < 128);
            let context = (u64::from(COPY_BASE[copy]).saturating_sub(2)).min(3);
            assert_eq!(u64::from(command.distance_context), context);
        }
        // Padding slots decode as an empty command and are never selected.
        assert_eq!(COMMANDS[704].copy_base, 0);
        assert_eq!(COMMANDS[COMMAND_SLOTS - 1].insert_base, 0);
    }

    #[test]
    fn context_lookup_tables_match_the_format_modes() {
        assert_eq!(lsb6_lut(), CONTEXT_LUT_LSB6);
        assert_eq!(msb6_lut(), CONTEXT_LUT_MSB6);
        // Modes 0 and 1 are the low and high six bits of the previous byte.
        assert_eq!(context_lut(0)[0xff], 63);
        assert_eq!(context_lut(0)[256 + 0xff], 0);
        assert_eq!(context_lut(1)[0xff], 63);
        assert_eq!(context_lut(1)[0x04], 1);
        // Modes 2 and 3 are the shared UTF8 and signed tables verbatim.
        assert_eq!(context_lut(2) as &[u8], CONTEXT_LUT_UTF8.as_slice());
        assert_eq!(context_lut(3) as &[u8], CONTEXT_LUT_SIGNED.as_slice());
        // A stream with no output yet has zero context in every mode.
        let stream = Stream::default();
        assert_eq!(previous_bytes(&stream.ring, stream.position), (0, 0));
        let mut tables = Tables::default();
        for mode in 0..4 {
            tables.modes[0] = mode;
            assert_eq!(context(&tables, &stream.ring, stream.position), 0);
        }
    }
}
