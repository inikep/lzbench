//! Canonical prefix codes as two-level lookup tables, plus the resumable
//! code description reader.
//!
//! Every table has a 256-entry root indexed by the next eight stream bits.
//! Codes longer than eight bits go through a second-level table whose root
//! entry carries the combined width and the absolute offset of that table in
//! the group's second-level storage. Complete codes fill every entry, so a
//! lookup never needs a validity check; a single-symbol code fills the root
//! with zero-width entries.
//!
//! The reader keeps the symbols of each code length in an intrusive linked
//! list as it reads them, which is already canonical order, so building a
//! table walks only the symbols that have a code rather than the alphabet.

use super::super::{DecodeError, InvalidDataKind};
use super::bits::{Bits, Input, mask};
use super::memory::Memory;
use alloc::vec::Vec;

// RFC 9841: 16 short codes + 120 direct codes + (124 << 3).
pub(super) const MAX_ALPHABET: usize = 1128;
const MAX_LENGTH: usize = 15;
const ROOT_BITS: u32 = 8;
const ROOT_SIZE: usize = 1 << ROOT_BITS;
/// Linked-list terminator; every symbol index is below [`MAX_ALPHABET`].
const NONE: u16 = u16::MAX;
const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// One table entry packed in a word without padding, so table storage can
/// be zeroed as plain memory: the low byte is the code width (or, for a root
/// entry pointing at a second-level table, the combined width), the high
/// bits are the symbol or the absolute second-level offset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Code(u32);

impl Code {
    const fn new(bits: u32, value: usize) -> Self {
        Self((value as u32) << 8 | bits)
    }

    #[inline(always)]
    const fn bits(self) -> u32 {
        self.0 & 0xff
    }

    #[inline(always)]
    const fn value(self) -> usize {
        (self.0 >> 8) as usize
    }
}

/// Borrowed lookup tables of one complete code.
#[derive(Debug, Clone, Copy)]
pub(super) struct Table<'a> {
    root: &'a [Code; ROOT_SIZE],
    /// The owning group's whole storage; root entries hold absolute offsets
    /// of the second-level tables in it.
    second: &'a [Code],
}

impl Table<'_> {
    /// Decodes with at least fifteen buffered bits.
    #[inline(always)]
    pub(super) fn decode_fast(self, bits: &mut Bits) -> usize {
        let peek = bits.value();
        let mut entry = self.root[(peek & (ROOT_SIZE as u64 - 1)) as usize];
        if entry.bits() > ROOT_BITS {
            let index =
                entry.value() + ((peek >> ROOT_BITS) & mask(entry.bits() - ROOT_BITS)) as usize;
            bits.drop(ROOT_BITS);
            entry = self.second[index];
        }
        bits.drop(entry.bits());
        entry.value()
    }

    /// Decodes with whatever is buffered; `None` needs at least one more byte.
    fn try_decode(self, bits: &mut Bits) -> Option<usize> {
        let available = bits.count();
        let peek = bits.value();
        let mut entry = self.root[(peek & (ROOT_SIZE as u64 - 1)) as usize];
        let mut width = entry.bits();
        if width > ROOT_BITS {
            if available < ROOT_BITS {
                return None;
            }
            let index = entry.value() + ((peek >> ROOT_BITS) & mask(width - ROOT_BITS)) as usize;
            entry = self.second[index];
            width = ROOT_BITS + entry.bits();
        }
        if width > available {
            return None;
        }
        bits.drop(width);
        Some(entry.value())
    }

    /// Decodes one symbol, accepting bytes only as the code needs them.
    pub(super) fn decode(
        self,
        bits: &mut Bits,
        input: &mut Input<'_>,
    ) -> Result<Option<usize>, DecodeError> {
        loop {
            if let Some(symbol) = self.try_decode(bits) {
                return Ok(Some(symbol));
            }
            if !bits.load_byte(input)? {
                return Ok(None);
            }
        }
    }

    /// Decodes one symbol, loading a whole word when the input allows and
    /// otherwise byte by byte.
    #[inline]
    pub(super) fn decode_refilling(
        self,
        bits: &mut Bits,
        input: &mut Input<'_>,
    ) -> Result<Option<usize>, DecodeError> {
        if bits.count() >= MAX_LENGTH as u32 || bits.refill(input) {
            return Ok(Some(self.decode_fast(bits)));
        }
        self.decode(bits, input)
    }
}

/// Every prefix code of one kind for a meta-block in one vector: the roots
/// first, one per code, then the second-level tables appended as codes are
/// built, so only tables that exist are ever written. The vector's length is
/// a high-water mark that only grows (and is zeroed only when it grows);
/// `used` is the logical end, so warm meta-blocks write no fill at all.
#[derive(Debug, Default)]
pub(super) struct Group {
    codes: Vec<Code>,
    count: usize,
    used: usize,
}

impl Group {
    /// Reserves `count` roots for `alphabet`, discarding earlier tables.
    pub(super) fn prepare(
        &mut self,
        count: usize,
        alphabet: usize,
        memory: &mut Memory,
    ) -> Result<(), DecodeError> {
        if alphabet == 0 || alphabet > MAX_ALPHABET {
            return Err(InvalidDataKind::Huffman.into());
        }
        let roots = count
            .checked_mul(ROOT_SIZE)
            .ok_or(DecodeError::SizeOverflow)?;
        if self.codes.len() < roots {
            memory.resize(&mut self.codes, roots)?;
        }
        self.count = count;
        self.used = roots;
        Ok(())
    }

    pub(super) const fn count(&self) -> usize {
        self.count
    }

    /// The tables of code `tree`. Only the root lookup is bounds checked.
    #[inline(always)]
    pub(super) fn table(&self, tree: usize) -> Table<'_> {
        self.tables().table(tree)
    }

    /// Borrows every table of the group at once, so a loop that selects a
    /// different code per symbol resolves the group's storage only once.
    #[inline(always)]
    pub(super) fn tables(&self) -> Tables<'_> {
        Tables {
            roots: self.codes[..self.count * ROOT_SIZE]
                .as_chunks::<ROOT_SIZE>()
                .0,
            second: &self.codes,
        }
    }
}

/// Every table of a group, borrowed for repeated per-symbol selection.
#[derive(Debug, Clone, Copy)]
pub(super) struct Tables<'a> {
    roots: &'a [[Code; ROOT_SIZE]],
    /// The whole group storage, addressed absolutely by root entries.
    second: &'a [Code],
}

impl<'a> Tables<'a> {
    /// The tables of code `tree`. Only the root lookup is bounds checked.
    #[inline(always)]
    pub(super) fn table(self, tree: usize) -> Table<'a> {
        Table {
            root: &self.roots[tree],
            second: self.second,
        }
    }
}

/// One owned prefix code, a group with a single slot.
pub(super) type Huffman = Group;

/// Width of the code-length code's table: its lengths are at most five bits.
const LENGTH_BITS: u32 = 5;
const LENGTH_SIZE: usize = 1 << LENGTH_BITS;

/// The code-length code of a complex description: eighteen symbols of at
/// most five bits, so one inline 32-entry table decodes every code without
/// a second level or any heap storage.
#[derive(Debug, Clone, Copy, Default)]
struct LengthCode([Code; LENGTH_SIZE]);

impl LengthCode {
    /// Builds the table from per-symbol lengths the caller has already
    /// checked to form a complete code, or to hold exactly one symbol, which
    /// decodes with zero bits.
    fn build(lengths: &[u8; 18], count: usize) -> Self {
        let mut table = [Code::default(); LENGTH_SIZE];
        if count == 1 {
            let symbol = lengths.iter().position(|&length| length != 0).unwrap_or(0);
            table.fill(Code::new(0, symbol));
            return Self(table);
        }
        let mut code = 0usize;
        for length in 1..=LENGTH_BITS as usize {
            for (symbol, _) in lengths
                .iter()
                .enumerate()
                .filter(|&(_, &width)| usize::from(width) == length)
            {
                replicate(
                    &mut table,
                    key_of(code, length),
                    1 << length,
                    Code::new(length as u32, symbol),
                );
                code += 1;
            }
            code <<= 1;
        }
        Self(table)
    }

    /// Decodes one symbol, loading a whole word when the input allows and
    /// otherwise accepting bytes only as the code needs them.
    #[inline(always)]
    fn decode(&self, bits: &mut Bits, input: &mut Input<'_>) -> Result<Option<usize>, DecodeError> {
        if bits.count() < LENGTH_BITS && !bits.refill(input) {
            loop {
                let entry = self.0[(bits.value() & (LENGTH_SIZE as u64 - 1)) as usize];
                if entry.bits() <= bits.count() {
                    break;
                }
                if !bits.load_byte(input)? {
                    return Ok(None);
                }
            }
        }
        let entry = self.0[(bits.value() & (LENGTH_SIZE as u64 - 1)) as usize];
        bits.drop(entry.bits());
        Ok(Some(entry.value()))
    }
}

impl Huffman {
    #[inline(always)]
    pub(super) fn codes(&self) -> Table<'_> {
        self.table(0)
    }
}

/// Appends `symbol` to the list of `length`. Symbols must arrive in
/// increasing order so every list stays canonical and `last` is the maximum.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "split borrows of the builder's list fields inside the symbol loop"
)]
fn push(
    next: &mut [u16; MAX_ALPHABET],
    head: &mut [u16; MAX_LENGTH + 1],
    tail: &mut [u16; MAX_LENGTH + 1],
    counts: &mut [u16; MAX_LENGTH + 1],
    total: &mut usize,
    last: &mut usize,
    symbol: usize,
    length: usize,
) {
    let previous = tail[length];
    if previous == NONE {
        head[length] = symbol as u16;
    } else {
        next[usize::from(previous)] = symbol as u16;
    }
    // Terminate here: the slot may still hold a link from an earlier code.
    next[symbol] = NONE;
    tail[length] = symbol as u16;
    counts[length] += 1;
    *total += 1;
    *last = symbol;
}

/// Appends the consecutive symbols `start..end` to the list of `length`.
/// The run links itself in one pass; only its ends touch the list state.
/// Callers never pass an empty run.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "split borrows of the builder's list fields inside the symbol loop"
)]
fn push_run(
    next: &mut [u16; MAX_ALPHABET],
    head: &mut [u16; MAX_LENGTH + 1],
    tail: &mut [u16; MAX_LENGTH + 1],
    counts: &mut [u16; MAX_LENGTH + 1],
    total: &mut usize,
    last: &mut usize,
    start: usize,
    end: usize,
    length: usize,
) {
    debug_assert!(start < end);
    for (index, link) in next[start..end - 1].iter_mut().enumerate() {
        *link = (start + index + 1) as u16;
    }
    next[end - 1] = NONE;
    let previous = tail[length];
    if previous == NONE {
        head[length] = start as u16;
    } else {
        next[usize::from(previous)] = start as u16;
    }
    tail[length] = (end - 1) as u16;
    counts[length] += (end - start) as u16;
    *total += end - start;
    *last = end - 1;
}

const fn reverse_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut value = 0usize;
    while value < 256 {
        table[value] = (value as u8).reverse_bits();
        value += 1;
    }
    table
}

/// Bit reversal of one byte, for turning canonical codes into table keys.
const REVERSE: [u8; 256] = reverse_table();

/// The table key of canonical `code` of `length` bits: the code with its
/// bits reversed, since the stream delivers the first code bit lowest.
#[inline(always)]
const fn key_of(code: usize, length: usize) -> usize {
    let reversed = ((REVERSE[code & 0xff] as usize) << 8) | REVERSE[(code >> 8) & 0xff] as usize;
    reversed >> (16 - length)
}

/// Width of the second-level table starting at `length`, given the counts
/// still unplaced at each width.
const fn next_table_bits(remaining: &[u16; MAX_LENGTH + 1], mut length: usize) -> u32 {
    let mut left = 1i32 << (length - ROOT_BITS as usize);
    while length < MAX_LENGTH {
        left -= remaining[length] as i32;
        if left <= 0 {
            break;
        }
        length += 1;
        left <<= 1;
    }
    (length - ROOT_BITS as usize) as u32
}

/// Exact number of second-level entries a complete code with these length
/// counts appends. Replays the canonical code walk of [`Builder::fill`] over
/// the counts alone: it advances the same `code` through the root lengths,
/// then creates a sub-table at each low-eight-bit key change with the same
/// [`next_table_bits`] width. `fill` therefore allocates exactly what it
/// writes, so a cold build zero-fills only entries it overwrites.
fn second_size(counts: &[u16; MAX_LENGTH + 1]) -> usize {
    let mut max_length = 0;
    for (length, &count) in counts.iter().enumerate() {
        if count != 0 {
            max_length = length;
        }
    }
    if max_length <= ROOT_BITS as usize {
        return 0;
    }
    // Advance `code` through the root lengths exactly as the root loop does.
    let mut code = 0usize;
    for &count in &counts[1..=ROOT_BITS as usize] {
        code = (code + usize::from(count)) << 1;
    }
    let mut remaining = *counts;
    let mut appended = 0usize;
    let mut low = usize::MAX;
    for length in ROOT_BITS as usize + 1..=MAX_LENGTH {
        let mut placed = 0u16;
        while placed < counts[length] {
            let key = key_of(code, length);
            if key & (ROOT_SIZE - 1) != low {
                remaining[length] = counts[length] - placed;
                appended += 1 << next_table_bits(&remaining, length);
                low = key & (ROOT_SIZE - 1);
            }
            placed += 1;
            code += 1;
        }
        code <<= 1;
    }
    appended
}

fn replicate(table: &mut [Code], start: usize, step: usize, code: Code) {
    let mut index = start;
    while index < table.len() {
        table[index] = code;
        index += step;
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum Stage {
    #[default]
    Start,
    SimpleCount,
    SimpleSymbols,
    SimpleShape,
    CodeLengths,
    Symbols,
    Repeat(u8),
}

/// Resumable reader of one prefix-code description. After `read` reports a
/// complete description, `build` or `build_slot` turns it into a table.
#[derive(Debug)]
pub(super) struct Builder {
    stage: Stage,
    /// Next symbol with the same code length, in increasing symbol order.
    next: [u16; MAX_ALPHABET],
    head: [u16; MAX_LENGTH + 1],
    tail: [u16; MAX_LENGTH + 1],
    counts: [u16; MAX_LENGTH + 1],
    total: usize,
    last: usize,
    small: [u8; 18],
    code: LengthCode,
    symbols: [usize; 4],
    count: usize,
    index: usize,
    space: i32,
    previous: u8,
    repeat_length: u8,
    repeat: usize,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            stage: Stage::Start,
            next: [NONE; MAX_ALPHABET],
            head: [NONE; MAX_LENGTH + 1],
            tail: [NONE; MAX_LENGTH + 1],
            counts: [0; MAX_LENGTH + 1],
            total: 0,
            last: 0,
            small: [0; 18],
            code: LengthCode::default(),
            symbols: [0; 4],
            count: 0,
            index: 0,
            space: 0,
            previous: 8,
            repeat_length: 0,
            repeat: 0,
        }
    }
}

impl Builder {
    /// Abandons a partial description. The next start clears its lists.
    pub(super) const fn reset(&mut self) {
        self.stage = Stage::Start;
    }

    const fn clear(&mut self) {
        self.head = [NONE; MAX_LENGTH + 1];
        self.tail = [NONE; MAX_LENGTH + 1];
        self.counts = [0; MAX_LENGTH + 1];
        self.total = 0;
        self.last = 0;
    }

    /// Appends `symbol` to the list of `length`; callers append in increasing
    /// symbol order so every list stays canonical.
    #[inline(always)]
    fn push(&mut self, symbol: usize, length: usize) {
        let Self {
            next,
            head,
            tail,
            counts,
            total,
            last,
            ..
        } = self;
        push(next, head, tail, counts, total, last, symbol, length);
    }

    /// Replaces the lists with an explicit length per symbol.
    #[cfg(test)]
    fn assign(&mut self, lengths: &[u8]) -> Result<(), DecodeError> {
        self.clear();
        for (symbol, &length) in lengths.iter().enumerate() {
            if usize::from(length) > MAX_LENGTH {
                return Err(InvalidDataKind::Huffman.into());
            }
            if length != 0 {
                self.push(symbol, usize::from(length));
            }
        }
        Ok(())
    }

    /// Fills the root at `start` of `codes` from the lists, appending any
    /// second-level tables at `used` (the absolute offset new tables get) and
    /// advancing it past them. The vector only grows to the most this code
    /// could append, so a warm workspace is never refilled. Returns the
    /// largest symbol.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn fill(
        &self,
        codes: &mut Vec<Code>,
        start: usize,
        used: &mut usize,
        memory: &mut Memory,
    ) -> Result<usize, DecodeError> {
        if self.total == 0 {
            return Err(InvalidDataKind::Huffman.into());
        }
        let second_base = *used;
        if start + ROOT_SIZE > second_base {
            return Err(DecodeError::InternalInvariant);
        }
        if self.total == 1 {
            if codes.len() < second_base {
                memory.resize(codes, second_base)?;
            }
            let root = codes
                .get_mut(start..start + ROOT_SIZE)
                .ok_or(DecodeError::InternalInvariant)?;
            root.fill(Code::new(0, self.last));
            return Ok(self.last);
        }
        let mut space = 1i32;
        let mut max_length = 0;
        for length in 1..=MAX_LENGTH {
            space = space * 2 - i32::from(self.counts[length]);
            if space < 0 {
                return Err(InvalidDataKind::Huffman.into());
            }
            if self.counts[length] != 0 {
                max_length = length;
            }
        }
        if space != 0 {
            return Err(InvalidDataKind::Huffman.into());
        }
        // Allocate exactly the second-level entries this code appends, so the
        // zero-fill covers only slots the fill below overwrites.
        let bound = second_base
            .checked_add(second_size(&self.counts))
            .ok_or(DecodeError::SizeOverflow)?;
        if codes.len() < bound {
            memory.resize(codes, bound)?;
        }
        let (front, second) = codes.split_at_mut(second_base);
        let root = front
            .get_mut(start..start + ROOT_SIZE)
            .ok_or(DecodeError::InternalInvariant)?;
        let table_bits = max_length.min(ROOT_BITS as usize);
        let mut current = 1usize << table_bits;
        // Canonical codes count up within a length and double between lengths.
        let mut code = 0usize;
        let mut step = 2usize;
        for length in 1..=table_bits {
            let mut symbol = self.head[length];
            while symbol != NONE {
                replicate(
                    &mut root[..current],
                    key_of(code, length),
                    step,
                    Code::new(length as u32, usize::from(symbol)),
                );
                code += 1;
                symbol = self.next[usize::from(symbol)];
            }
            code <<= 1;
            step <<= 1;
        }
        while current != ROOT_SIZE {
            root.copy_within(..current, current);
            current <<= 1;
        }
        let mut remaining = self.counts;
        let mut appended = 0usize;
        let mut low = usize::MAX;
        let mut table: &mut [Code] = &mut [];
        step = 2;
        for length in ROOT_BITS as usize + 1..=MAX_LENGTH {
            let mut symbol = self.head[length];
            let mut placed = 0u16;
            while symbol != NONE {
                let key = key_of(code, length);
                if key & (ROOT_SIZE - 1) != low {
                    let table_start = appended;
                    remaining[length] = self.counts[length] - placed;
                    let bits = next_table_bits(&remaining, length);
                    current = 1 << bits;
                    appended += current;
                    low = key & (ROOT_SIZE - 1);
                    root[low] = Code::new(bits + ROOT_BITS, second_base + table_start);
                    table = second
                        .get_mut(table_start..appended)
                        .ok_or(DecodeError::InternalInvariant)?;
                }
                replicate(
                    table,
                    key >> ROOT_BITS,
                    step,
                    Code::new((length - ROOT_BITS as usize) as u32, usize::from(symbol)),
                );
                placed += 1;
                code += 1;
                symbol = self.next[usize::from(symbol)];
            }
            code <<= 1;
            step <<= 1;
        }
        *used = second_base + appended;
        Ok(self.last)
    }

    /// Builds the completed description into an owned single-slot table.
    /// Returns the largest symbol with a code.
    pub(super) fn build(
        &mut self,
        alphabet: usize,
        memory: &mut Memory,
        target: &mut Huffman,
    ) -> Result<usize, DecodeError> {
        target.prepare(1, alphabet, memory)?;
        self.build_slot(alphabet, target, 0, memory)
    }

    /// Builds the completed description into one of a group's slots.
    pub(super) fn build_slot(
        &mut self,
        alphabet: usize,
        group: &mut Group,
        tree: usize,
        memory: &mut Memory,
    ) -> Result<usize, DecodeError> {
        if self.last >= alphabet || tree >= group.count {
            return Err(DecodeError::InternalInvariant);
        }
        self.fill(&mut group.codes, tree * ROOT_SIZE, &mut group.used, memory)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(super) fn read(
        &mut self,
        alphabet: usize,
        bits: &mut Bits,
        input: &mut Input<'_>,
    ) -> Result<bool, DecodeError> {
        if alphabet == 0 || alphabet > MAX_ALPHABET {
            return Err(InvalidDataKind::Huffman.into());
        }
        loop {
            match self.stage {
                Stage::Start => {
                    let Some(skip) = bits.read(2, input)? else {
                        return Ok(false);
                    };
                    self.clear();
                    self.small.fill(0);
                    self.index = skip as usize;
                    self.space = 32;
                    self.count = 0;
                    self.stage = if skip == 1 {
                        Stage::SimpleCount
                    } else {
                        Stage::CodeLengths
                    };
                }
                Stage::SimpleCount => {
                    let Some(count) = bits.read(2, input)? else {
                        return Ok(false);
                    };
                    self.count = count as usize + 1;
                    self.index = 0;
                    self.stage = Stage::SimpleSymbols;
                }
                Stage::SimpleSymbols => {
                    let width = usize::BITS - (alphabet - 1).leading_zeros();
                    while self.index < self.count {
                        let Some(symbol) = bits.read(width, input)? else {
                            return Ok(false);
                        };
                        let symbol = symbol as usize;
                        if symbol >= alphabet || self.symbols[..self.index].contains(&symbol) {
                            return Err(InvalidDataKind::Huffman.into());
                        }
                        self.symbols[self.index] = symbol;
                        self.index += 1;
                    }
                    self.stage = Stage::SimpleShape;
                }
                Stage::SimpleShape => {
                    let shape = if self.count == 4 {
                        let Some(shape) = bits.read(1, input)? else {
                            return Ok(false);
                        };
                        shape
                    } else {
                        0
                    };
                    let lengths = match (self.count, shape) {
                        (1, _) => [1, 0, 0, 0],
                        (2, _) => [1, 1, 0, 0],
                        (3, _) => [1, 2, 2, 0],
                        (_, 0) => [2, 2, 2, 2],
                        _ => [1, 2, 3, 3],
                    };
                    // Lists must be canonical: sorted by length, then symbol.
                    let mut pairs = [(0u8, 0usize); 4];
                    for (pair, (&length, &symbol)) in
                        pairs.iter_mut().zip(lengths.iter().zip(&self.symbols))
                    {
                        *pair = (length, symbol);
                    }
                    let pairs = &mut pairs[..self.count];
                    // At most four entries: a fixed insertion sort beats the
                    // generic sort's setup here.
                    for i in 1..pairs.len() {
                        let mut j = i;
                        while j > 0 && pairs[j] < pairs[j - 1] {
                            pairs.swap(j, j - 1);
                            j -= 1;
                        }
                    }
                    self.clear();
                    for &(length, symbol) in pairs.iter() {
                        self.push(symbol, usize::from(length));
                    }
                    self.last = pairs.iter().map(|&(_, symbol)| symbol).max().unwrap_or(0);
                    self.stage = Stage::Start;
                    return Ok(true);
                }
                Stage::CodeLengths => {
                    while self.index < 18 && self.space > 0 {
                        let Some(first) = bits.peek(2, input)? else {
                            return Ok(false);
                        };
                        let (width, value) = match first {
                            0 => (2, 0),
                            1 => (2, 4),
                            2 => (2, 3),
                            _ => {
                                let Some(next) = bits.peek(3, input)? else {
                                    return Ok(false);
                                };
                                if next == 3 {
                                    (3, 2)
                                } else {
                                    let Some(last) = bits.peek(4, input)? else {
                                        return Ok(false);
                                    };
                                    (4, if last == 7 { 1 } else { 5 })
                                }
                            }
                        };
                        bits.drop(width);
                        self.small[ORDER[self.index]] = value;
                        self.index += 1;
                        if value != 0 {
                            self.count += 1;
                            self.space -= 32 >> value;
                        }
                    }
                    if self.space < 0 || (self.count != 1 && self.space != 0) {
                        return Err(InvalidDataKind::Huffman.into());
                    }
                    self.code = LengthCode::build(&self.small, self.count);
                    self.clear();
                    self.index = 0;
                    self.space = 32768;
                    self.previous = 8;
                    self.repeat = 0;
                    self.repeat_length = 0;
                    self.stage = Stage::Symbols;
                }
                Stage::Symbols | Stage::Repeat(_) => {
                    let Self {
                        stage,
                        next,
                        head,
                        tail,
                        counts,
                        total,
                        last,
                        code,
                        index,
                        space,
                        previous,
                        repeat_length,
                        repeat,
                        ..
                    } = self;
                    let table = &*code;
                    let outcome = loop {
                        let symbol = if let Stage::Repeat(symbol) = *stage {
                            usize::from(symbol)
                        } else {
                            if *space == 0 {
                                break Ok(true);
                            }
                            if *space < 0 || *index == alphabet {
                                break Err(InvalidDataKind::Huffman.into());
                            }
                            match table.decode(bits, input) {
                                Ok(Some(symbol)) => symbol,
                                Ok(None) => break Ok(false),
                                Err(error) => break Err(error),
                            }
                        };
                        if symbol < 16 {
                            if symbol != 0 {
                                push(next, head, tail, counts, total, last, *index, symbol);
                                *previous = symbol as u8;
                                *space -= 32768 >> symbol;
                            }
                            *index += 1;
                            *repeat = 0;
                            continue;
                        }
                        // The symbol is consumed; only its extra field may still wait.
                        *stage = Stage::Repeat(symbol as u8);
                        let width = symbol as u32 - 14;
                        let extra = match bits.read(width, input) {
                            Ok(Some(extra)) => extra,
                            Ok(None) => break Ok(false),
                            Err(error) => break Err(error),
                        };
                        *stage = Stage::Symbols;
                        let length = if symbol == 16 { *previous } else { 0 };
                        if *repeat_length != length {
                            *repeat = 0;
                            *repeat_length = length;
                        }
                        let before = *repeat;
                        *repeat = if before == 0 {
                            0
                        } else {
                            (before - 2) << width
                        };
                        *repeat += extra as usize + 3;
                        let delta = *repeat - before;
                        let end = *index + delta;
                        if end > alphabet {
                            break Err(InvalidDataKind::Huffman.into());
                        }
                        if length != 0 {
                            push_run(
                                next,
                                head,
                                tail,
                                counts,
                                total,
                                last,
                                *index,
                                end,
                                usize::from(length),
                            );
                            *space -= (delta as i32) << (15 - length);
                        }
                        *index = end;
                    };
                    match outcome {
                        Ok(true) => {
                            *stage = Stage::Start;
                            return Ok(true);
                        }
                        Ok(false) => return Ok(false),
                        Err(error) => return Err(error),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;

    fn build(lengths: &[u8]) -> Result<(Huffman, usize), DecodeError> {
        let mut builder = Builder::default();
        builder.assign(lengths)?;
        let mut code = Huffman::default();
        let max_symbol = builder.build(lengths.len(), &mut Memory::default(), &mut code)?;
        Ok((code, max_symbol))
    }

    fn entry(code: Code) -> (u32, usize) {
        (code.bits(), code.value())
    }

    /// Reference decoder: walks the canonical code bit by bit.
    fn canonical(lengths: &[u8], mut next_bit: impl FnMut() -> u64) -> usize {
        let mut code = 0usize;
        let mut first = 0usize;
        let mut index = 0usize;
        let mut sorted: Vec<(u8, usize)> = lengths
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l != 0)
            .map(|(s, &l)| (l, s))
            .collect();
        sorted.sort_unstable();
        for length in 1..=MAX_LENGTH as u8 {
            code |= next_bit() as usize;
            let count = sorted.iter().filter(|(l, _)| *l == length).count();
            if code - first < count {
                return sorted[index + code - first].1;
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        panic!("incomplete code");
    }

    #[test]
    fn length_codes_decode_canonically_and_a_single_symbol_takes_no_bits() {
        // A complete five-bit code over symbols spread across the alphabet.
        let mut lengths = [0u8; 18];
        for (symbol, length) in [
            (0, 2),
            (5, 2),
            (17, 3),
            (1, 3),
            (12, 3),
            (16, 4),
            (9, 5),
            (3, 5),
        ] {
            lengths[symbol] = length;
        }
        let code = LengthCode::build(&lengths, 8);
        for byte in 0..=u8::MAX {
            let mut bits = Bits::default();
            let bytes = [byte];
            let mut input = fixtures::input(&bytes);
            let mut taken = 0;
            let expected = canonical(&lengths, || {
                taken += 1;
                u64::from(byte >> (taken - 1) & 1)
            });
            assert_eq!(code.decode(&mut bits, &mut input).unwrap(), Some(expected));
            assert_eq!(bits.count(), 8 - taken);
        }
        // An exhausted input waits for another byte.
        let mut bits = Bits::default();
        assert_eq!(
            code.decode(&mut bits, &mut fixtures::input(&[])).unwrap(),
            None
        );
        // With one nonzero length the only symbol decodes from no bits.
        let mut single = [0u8; 18];
        single[16] = 3;
        let code = LengthCode::build(&single, 1);
        assert_eq!(
            code.decode(&mut bits, &mut fixtures::input(&[])).unwrap(),
            Some(16)
        );
        assert_eq!(bits.count(), 0);
    }

    #[test]
    fn tables_agree_with_bitwise_canonical_decoding_for_random_complete_codes() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for alphabet in [2usize, 3, 4, 18, 26, 256, 258, 272, 704, 1128] {
            for _ in 0..40 {
                // Build a complete code by splitting leaves at random.
                let mut lengths = alloc::vec![0u8; alphabet];
                let symbols = 2 + (random() as usize) % (alphabet - 1);
                let mut lengths_pool: Vec<u8> = alloc::vec![1, 1];
                while lengths_pool.len() < symbols {
                    let pick = (random() as usize) % lengths_pool.len();
                    let length = lengths_pool[pick];
                    if length >= MAX_LENGTH as u8 {
                        continue;
                    }
                    lengths_pool.swap_remove(pick);
                    lengths_pool.push(length + 1);
                    lengths_pool.push(length + 1);
                }
                // Assign distinct symbols in random order.
                let mut candidates: Vec<usize> = (0..alphabet).collect();
                for i in (1..candidates.len()).rev() {
                    let j = (random() as usize) % (i + 1);
                    candidates.swap(i, j);
                }
                for (symbol, &length) in candidates.iter().zip(&lengths_pool) {
                    lengths[*symbol] = length;
                }
                let (code, max_symbol) = build(&lengths).unwrap();
                assert_eq!(max_symbol, lengths.iter().rposition(|&l| l != 0).unwrap());
                let table = code.codes();
                for _ in 0..64 {
                    let word = random();
                    let word_bytes = word.to_le_bytes();
                    let mut bits = Bits::default();
                    let mut input = fixtures::input(&word_bytes);
                    bits.peek(56, &mut input).unwrap();
                    let before = bits.count();
                    let fast = table.decode_fast(&mut bits);
                    let used = before - bits.count();
                    let mut position = 0;
                    let expected = canonical(&lengths, || {
                        let bit = (word >> position) & 1;
                        position += 1;
                        bit
                    });
                    assert_eq!((fast, used as usize), (expected, position));
                    assert_eq!(used, u32::from(lengths[fast]));
                    // Byte-exact decoding reads only the bytes the code needs.
                    let mut slow = Bits::default();
                    let mut input = fixtures::input(&word_bytes);
                    assert_eq!(table.decode(&mut slow, &mut input).unwrap(), Some(expected));
                    assert_eq!(input.consumed, position.div_ceil(8));
                    // The refilling reader agrees whether or not a word is available.
                    let mut refilling = Bits::default();
                    let mut input = fixtures::input(&word_bytes);
                    assert_eq!(
                        table.decode_refilling(&mut refilling, &mut input).unwrap(),
                        Some(expected)
                    );
                    let mut short = Bits::default();
                    let mut input = fixtures::input(&word_bytes[..7]);
                    let via_bytes = table.decode_refilling(&mut short, &mut input).unwrap();
                    assert!(via_bytes.is_none() || via_bytes == Some(expected));
                }
            }
        }
    }

    #[test]
    fn single_symbol_codes_consume_no_bits_and_bad_shapes_are_rejected() {
        let (code, max_symbol) = build(&[0, 0, 7, 0]).unwrap();
        assert_eq!((code.count(), max_symbol), (1, 2));
        let mut bits = Bits::default();
        let mut input = fixtures::input(&[]);
        assert_eq!(code.codes().decode(&mut bits, &mut input).unwrap(), Some(2));
        assert!(build(&[0, 0]).is_err());
        assert!(build(&[1, 1, 1]).is_err());
        assert!(build(&[1, 2, 0]).is_err());
        assert!(build(&[16, 1]).is_err());
        // A slot outside the group, or a symbol outside the alphabet, is a
        // library defect rather than a format error.
        let mut memory = Memory::default();
        let mut group = Group::default();
        group.prepare(1, 4, &mut memory).unwrap();
        let mut builder = Builder::default();
        builder.assign(&[1, 1]).unwrap();
        assert!(matches!(
            builder.build_slot(2, &mut group, 1, &mut memory),
            Err(DecodeError::InternalInvariant)
        ));
        assert!(matches!(
            builder.build_slot(1, &mut group, 0, &mut memory),
            Err(DecodeError::InternalInvariant)
        ));
        assert!(matches!(
            group.prepare(1, 0, &mut memory),
            Err(DecodeError::InvalidData { .. })
        ));
        // Codes within the root append nothing; a nine-bit code appends a
        // two-entry table at the absolute offset its root entry names.
        let mut lengths = alloc::vec![0u8; 16];
        lengths[0] = 1;
        for (i, length) in (2..=8).zip(&mut lengths[1..]) {
            *length = i;
        }
        lengths[8] = 8;
        builder.assign(&lengths).unwrap();
        let mut codes = alloc::vec![Code::default(); ROOT_SIZE + 3];
        let mut used = ROOT_SIZE + 3;
        assert_eq!(
            builder.fill(&mut codes, 0, &mut used, &mut memory).unwrap(),
            8
        );
        assert_eq!(used, ROOT_SIZE + 3);
        lengths[8] = 9;
        lengths[9] = 9;
        builder.assign(&lengths).unwrap();
        assert_eq!(
            builder.fill(&mut codes, 0, &mut used, &mut memory).unwrap(),
            9
        );
        assert_eq!(used, ROOT_SIZE + 5);
        assert!(codes.len() >= ROOT_SIZE + 5);
        assert_eq!(entry(codes[0xff]), (9, ROOT_SIZE + 3));
        assert_eq!(entry(codes[ROOT_SIZE + 3]), (1, 8));
        assert_eq!(entry(codes[ROOT_SIZE + 4]), (1, 9));
        // A root past the logical end is a defect; a workspace budget bounds
        // second-level growth like any storage.
        let mut short = ROOT_SIZE;
        assert!(matches!(
            builder.fill(&mut codes, ROOT_SIZE, &mut short, &mut memory),
            Err(DecodeError::InternalInvariant)
        ));
        let mut tight = Memory {
            live: 0,
            limit: Some(4),
        };
        let mut small = alloc::vec![Code::default(); ROOT_SIZE];
        let mut small_used = ROOT_SIZE;
        assert!(matches!(
            builder.fill(&mut small, 0, &mut small_used, &mut tight),
            Err(DecodeError::MemoryLimitExceeded { .. })
        ));
        // Exact second-level sizing matches the loose reference bound's shape:
        // a two-nine-bit split needs one sub-table of two entries.
        let mut split = alloc::vec![0u8; 16];
        split[0] = 1;
        for (i, length) in (2..=8).zip(&mut split[1..]) {
            *length = i;
        }
        split[7] = 8;
        let mut builder = Builder::default();
        builder.assign(&split).unwrap();
        assert_eq!(second_size(&builder.counts), 0);
        split[7] = 9;
        split[8] = 9;
        builder.assign(&split).unwrap();
        assert_eq!(second_size(&builder.counts), 2);
    }

    #[test]
    fn partial_input_decoding_waits_for_exactly_the_needed_bytes() {
        // Lengths 1..=8 fill the root exactly; symbol 8 is 1111_1111.
        let mut lengths = alloc::vec![0u8; 16];
        lengths[0] = 1;
        for (i, length) in (2..=8).zip(&mut lengths[1..]) {
            *length = i;
        }
        lengths[8] = 8;
        let (code, _) = build(&lengths).unwrap();
        let mut bits = Bits::default();
        let mut input = fixtures::input(&[0x7f]);
        assert_eq!(code.codes().decode(&mut bits, &mut input).unwrap(), Some(7));
        assert_eq!(input.consumed, 1);
        let mut bits = Bits::default();
        assert_eq!(
            code.codes()
                .decode(&mut bits, &mut fixtures::input(&[]))
                .unwrap(),
            None
        );
        // Replacing the eight-bit leaf with two nine-bit leaves adds a
        // second-level table; those codes wait for their ninth bit.
        lengths[8] = 9;
        lengths[9] = 9;
        let (code, _) = build(&lengths).unwrap();
        let mut bits = Bits::default();
        assert_eq!(
            code.codes()
                .decode(&mut bits, &mut fixtures::input(&[0xff]))
                .unwrap(),
            None
        );
        let mut bits = Bits::default();
        let mut input = fixtures::input(&[0xff, 0x00]);
        assert_eq!(code.codes().decode(&mut bits, &mut input).unwrap(), Some(8));
        assert_eq!((input.consumed, bits.count()), (2, 7));
        let mut bits = Bits::default();
        assert_eq!(
            code.codes()
                .decode(&mut bits, &mut fixtures::input(&[0xff, 0x01]))
                .unwrap(),
            Some(9)
        );
    }

    #[test]
    fn groups_reserve_fixed_strides_and_reject_impossible_alphabets() {
        let mut memory = Memory::default();
        let mut group = Group::default();
        assert_eq!(group.count(), 0);
        group.prepare(3, 256, &mut memory).unwrap();
        assert_eq!((group.count(), group.codes.len()), (3, 3 * ROOT_SIZE));
        assert!(group.prepare(1, 4000, &mut memory).is_err());
        let mut builder = Builder::default();
        builder.assign(&[2, 2, 2, 2]).unwrap();
        assert_eq!(
            builder.build_slot(4, &mut group, 1, &mut memory).unwrap(),
            3
        );
        assert!(matches!(
            builder.build_slot(4, &mut group, 3, &mut memory),
            Err(DecodeError::InternalInvariant)
        ));
        let mut owned = Huffman::default();
        assert_eq!(builder.build(4, &mut memory, &mut owned).unwrap(), 3);
        assert_eq!(owned.count(), 1);
        // Preparing again keeps the storage and restarts the logical end.
        let (len, capacity) = (group.codes.len(), group.codes.capacity());
        group.prepare(1, 256, &mut memory).unwrap();
        assert_eq!(
            (group.used, group.codes.len(), group.codes.capacity()),
            (ROOT_SIZE, len, capacity)
        );
    }

    #[test]
    fn table_keys_are_bit_reversed_canonical_codes() {
        assert_eq!(reverse_table(), REVERSE);
        assert_eq!(REVERSE[0b0000_0001], 0b1000_0000);
        assert_eq!(REVERSE[0b1011_0000], 0b0000_1101);
        // A one-bit code 1 keys slot 1; a nine-bit code 0b1_0000_0000 keys 1.
        assert_eq!(key_of(1, 1), 1);
        assert_eq!(key_of(0b110, 3), 0b011);
        assert_eq!(key_of(0b1_0000_0000, 9), 1);
        assert_eq!(key_of(0b111_1111_1111_1111, 15), 0x7fff);
        // A run links its symbols in order and terminates at its end.
        let mut builder = Builder::default();
        builder.clear();
        push_run(
            &mut builder.next,
            &mut builder.head,
            &mut builder.tail,
            &mut builder.counts,
            &mut builder.total,
            &mut builder.last,
            5,
            8,
            3,
        );
        assert_eq!((builder.total, builder.head[3], builder.tail[3]), (3, 5, 7));
        assert_eq!(
            (builder.next[5], builder.next[6], builder.next[7]),
            (6, 7, NONE)
        );
    }

    #[test]
    fn simple_codes_order_symbols_canonically_regardless_of_stream_order() {
        // Two one-bit symbols written as 5 then 3: symbol 3 must take code 0.
        let bytes = fixtures::fields(&[(2, 1), (2, 1), (8, 5), (8, 3), (1, 0)]);
        let mut bits = Bits::default();
        let mut input = fixtures::input(&bytes);
        let mut memory = Memory::default();
        let mut builder = Builder::default();
        assert!(builder.read(256, &mut bits, &mut input).unwrap());
        let mut tree = Huffman::default();
        assert_eq!(builder.build(256, &mut memory, &mut tree).unwrap(), 5);
        assert_eq!(tree.codes().decode(&mut bits, &mut input).unwrap(), Some(3));
        // Four symbols with the 1,2,3,3 shape: the two three-bit symbols are
        // ordered by value even when written descending.
        let bytes = fixtures::fields(&[
            (2, 1),
            (2, 3),
            (8, 9),
            (8, 8),
            (8, 7),
            (8, 6),
            (1, 1),
            (3, 0b111),
            (3, 0b011),
        ]);
        let mut bits = Bits::default();
        let mut input = fixtures::input(&bytes);
        assert!(builder.read(256, &mut bits, &mut input).unwrap());
        assert_eq!(builder.build(256, &mut memory, &mut tree).unwrap(), 9);
        assert_eq!(tree.codes().decode(&mut bits, &mut input).unwrap(), Some(7));
        assert_eq!(tree.codes().decode(&mut bits, &mut input).unwrap(), Some(6));
    }

    #[test]
    fn complex_descriptions_repeat_previous_lengths_and_zero_lengths() {
        // Code-length trees name exactly {2,16} or {2,17}; both codes have
        // length one. The target tree has four length-two leaves.
        for repeat in [16, 17] {
            let mut fields = alloc::vec![(2, 0)];
            for &symbol in &ORDER {
                fields.push(if symbol == 2 || symbol == repeat {
                    (4, 7)
                } else {
                    (2, 0)
                });
                if symbol == repeat {
                    break;
                }
            }
            if repeat == 16 {
                fields.extend([(1, 0), (1, 1), (2, 0)]); // length 2, then repeat it 3 times
            } else {
                fields.extend([(1, 1), (3, 1), (4, 0)]); // four zeros, then four length 2s
            }
            fields.extend([(2, 0), (2, 2), (2, 1), (2, 3)]);
            let bytes = fixtures::fields(&fields);
            for pad in [0usize, 16] {
                // Padding lets the reader refill whole words; without it every
                // symbol arrives byte by byte. Both must agree.
                let mut padded = bytes.clone();
                padded.resize(bytes.len() + pad, 0);
                let mut bits = Bits::default();
                let mut input = fixtures::input(&padded);
                let mut memory = Memory::default();
                let mut builder = Builder::default();
                let alphabet = if repeat == 16 { 4 } else { 8 };
                assert!(builder.read(alphabet, &mut bits, &mut input).unwrap());
                let mut tree = Huffman::default();
                assert_eq!(
                    builder.build(alphabet, &mut memory, &mut tree).unwrap(),
                    alphabet - 1
                );
                for symbol in alphabet - 4..alphabet {
                    assert_eq!(
                        tree.codes().decode(&mut bits, &mut input).unwrap(),
                        Some(symbol)
                    );
                }
            }
        }
    }
}
