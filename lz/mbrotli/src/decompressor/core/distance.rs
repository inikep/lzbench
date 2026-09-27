//! Wire distance arithmetic, independent of host address size and history.

use super::super::{DecodeError, InvalidDataKind};
use super::memory::Memory;
use alloc::vec::Vec;

const SHORT_CODES: usize = 16;
const MAX_DISTANCE: u128 = (1u128 << 63) - 4;
/// Same bound as [`MAX_DISTANCE`], but the cache path resolves in `u64` and
/// never needs the wider type.
const MAX_DISTANCE_U64: u64 = (1u64 << 63) - 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DistanceLayout {
    postfix: u8,
    direct: usize,
    large: bool,
}

/// A distance symbol split into the extra-bit width and the distance it
/// yields once the extra field is known, so the field can be read between.
/// Both fields are `u64` so the table has no padding and its zero fill is a
/// plain memory set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Partial {
    pub(super) width: u64,
    /// Distance before the extra field; add `extra << postfix`.
    pub(super) base: u64,
}

/// The four most recent distances as a ring, in the shape of C's `dist_rb`:
/// `index` counts pushes, so slot `(index - 1) & 3` is the most recent one.
/// Reading the most recent distance and pushing it back is then the same
/// slot write as pushing a new one, which keeps the implicit-distance
/// command free of a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cache {
    slots: [u64; 4],
    index: usize,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            slots: [16, 15, 11, 4],
            index: 4,
        }
    }
}

impl Cache {
    /// The `back`-th most recent distance, zero being the most recent.
    #[inline(always)]
    pub(super) const fn recent(&self, back: usize) -> u64 {
        self.slots[self.index.wrapping_sub(1 + back) & 3]
    }

    /// Takes the most recent distance out so that [`Self::push`] returns it.
    #[inline(always)]
    pub(super) const fn pop(&mut self) -> u64 {
        self.index = self.index.wrapping_sub(1);
        self.slots[self.index & 3]
    }

    #[inline(always)]
    pub(super) const fn push(&mut self, distance: u64) {
        self.slots[self.index & 3] = distance;
        self.index = self.index.wrapping_add(1);
    }
}

impl DistanceLayout {
    pub(super) const fn from_header(header: u8, large: bool) -> Self {
        let postfix = header & 3;
        Self {
            postfix,
            direct: ((header >> 2) as usize) << postfix,
            large,
        }
    }

    pub(super) const fn postfix(self) -> u32 {
        self.postfix as u32
    }

    /// Whether the layout belongs to a large-window stream, whose extra
    /// fields may exceed 32 bits.
    pub(super) const fn is_large(self) -> bool {
        self.large
    }

    pub(super) const fn alphabet(self) -> usize {
        SHORT_CODES + self.direct + ((if self.large { 124 } else { 48 }) << self.postfix)
    }

    pub(super) const fn extra_bits(self, symbol: usize) -> u32 {
        if symbol < SHORT_CODES + self.direct {
            0
        } else {
            (1 + ((symbol - SHORT_CODES - self.direct) >> (self.postfix + 1))) as u32
        }
    }

    /// RFC 9841 constrains reachable symbols by their maximum possible value,
    /// even when the stream happens to select smaller extra bits.
    pub(super) fn validate_symbol(self, symbol: usize) -> Result<(), DecodeError> {
        if symbol >= self.alphabet() {
            return Err(InvalidDataKind::Distance.into());
        }
        if symbol >= SHORT_CODES
            && self.long_distance(symbol, (1u64 << self.extra_bits(symbol)) - 1) > MAX_DISTANCE
        {
            return Err(InvalidDataKind::Distance.into());
        }
        Ok(())
    }

    const fn long_distance(self, symbol: usize, extra: u64) -> u128 {
        if symbol < SHORT_CODES + self.direct {
            return (symbol - SHORT_CODES + 1) as u128;
        }
        let code = symbol - SHORT_CODES - self.direct;
        let high = code >> self.postfix;
        let low = code & ((1 << self.postfix) - 1);
        let offset = (((2 + (high & 1)) as u128) << self.extra_bits(symbol)) - 4;
        ((offset + extra as u128) << self.postfix) + low as u128 + self.direct as u128 + 1
    }

    /// Standard-window distance in native `u64`. Every intermediate fits: the
    /// largest standard alphabet keeps `extra_bits <= 25` and the final value
    /// well below `2^32`, so no `u128` widening is needed on the hot path.
    #[inline]
    fn short_window_distance(self, symbol: usize, extra: u64) -> u64 {
        if symbol < SHORT_CODES + self.direct {
            return (symbol - SHORT_CODES + 1) as u64;
        }
        let code = symbol - SHORT_CODES - self.direct;
        let high = code >> self.postfix;
        let low = code & ((1 << self.postfix) - 1);
        let offset = (((2 + (high & 1)) as u64) << self.extra_bits(symbol)) - 4;
        ((offset + extra) << self.postfix) + (low + self.direct + 1) as u64
    }

    /// Large-window distance through the widening path, kept out of line so the
    /// common standard-window resolve stays small enough to inline.
    #[cold]
    #[inline(never)]
    fn resolve_large(self, symbol: usize, extra: u64) -> Result<u64, DecodeError> {
        u64::try_from(self.long_distance(symbol, extra))
            .map_err(|_| InvalidDataKind::Distance.into())
    }

    #[inline]
    pub(super) fn resolve(
        self,
        symbol: usize,
        extra: u64,
        cache: &Cache,
    ) -> Result<u64, DecodeError> {
        if symbol >= SHORT_CODES {
            if self.large {
                return self.resolve_large(symbol, extra);
            }
            // The header validated every reachable symbol, so a standard-window
            // distance is always in range and needs no further check.
            return Ok(self.short_window_distance(symbol, extra));
        }
        Self::short(symbol, cache)
    }

    /// Resolves one of the sixteen short codes against the recent distances.
    #[inline(always)]
    pub(super) fn short(symbol: usize, cache: &Cache) -> Result<u64, DecodeError> {
        const INDEX: [usize; 16] = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        const OFFSET: [i8; 16] = [0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3];
        cache
            .recent(INDEX[symbol & 15])
            .checked_add_signed(i64::from(OFFSET[symbol & 15]))
            .filter(|&v| v != 0 && v <= MAX_DISTANCE_U64)
            .ok_or_else(|| InvalidDataKind::Distance.into())
    }

    /// The split of one symbol: short codes get an empty entry and resolve
    /// through the cache. A large-window symbol whose maximum distance the
    /// header rejects is never decoded, so its entry may wrap.
    const fn entry(self, symbol: usize) -> Partial {
        if symbol < SHORT_CODES {
            Partial { width: 0, base: 0 }
        } else if symbol < SHORT_CODES + self.direct {
            Partial {
                width: 0,
                base: (symbol - SHORT_CODES + 1) as u64,
            }
        } else {
            let code = symbol - SHORT_CODES - self.direct;
            let width = 1 + (code >> (self.postfix + 1)) as u32;
            let high = code >> self.postfix;
            let low = code & ((1 << self.postfix) - 1);
            let offset = (((2 + (high & 1)) as u64) << width).wrapping_sub(4);
            Partial {
                width: width as u64,
                base: (offset << self.postfix).wrapping_add((low + self.direct + 1) as u64),
            }
        }
    }

    /// Whether this is the standard-window layout without postfix or direct
    /// codes, whose split is the constant [`STANDARD_TABLE`].
    pub(super) const fn is_standard(self) -> bool {
        self.postfix == 0 && self.direct == 0 && !self.large
    }

    /// The per-symbol split of this layout: the constant table for the
    /// standard layout, otherwise the table [`Self::fill_table`] filled.
    #[inline(always)]
    pub(super) fn table(self, filled: &[Partial]) -> &[Partial] {
        if self.is_standard() {
            &STANDARD_TABLE
        } else {
            filled
        }
    }

    /// Fills `table` with the split of every symbol of the alphabet, so the
    /// command loop resolves a distance with one lookup. The standard layout
    /// never needs this: its split is constant.
    pub(super) fn fill_table(
        self,
        table: &mut Vec<Partial>,
        memory: &mut Memory,
    ) -> Result<(), DecodeError> {
        let alphabet = self.alphabet();
        if table.len() < alphabet {
            memory.resize(table, alphabet)?;
        }
        for (symbol, entry) in table.iter_mut().enumerate().take(alphabet) {
            *entry = self.entry(symbol);
        }
        Ok(())
    }
}

/// Number of symbols in the standard layout without postfix or direct codes.
const STANDARD_ALPHABET: usize = SHORT_CODES + 48;

const fn standard_table() -> [Partial; STANDARD_ALPHABET] {
    let layout = DistanceLayout {
        postfix: 0,
        direct: 0,
        large: false,
    };
    let mut table = [Partial { width: 0, base: 0 }; STANDARD_ALPHABET];
    let mut symbol = 0;
    while symbol < STANDARD_ALPHABET {
        table[symbol] = layout.entry(symbol);
        symbol += 1;
    }
    table
}

/// The split of every symbol of the most common layout, so a fresh decoder
/// does not compute or allocate it.
static STANDARD_TABLE: [Partial; STANDARD_ALPHABET] = standard_table();

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_layout_agrees_with_an_independent_wide_integer_model() {
        for header in 0..64 {
            let layout = DistanceLayout::from_header(header, true);
            assert!(layout.is_large());
            assert!(!DistanceLayout::from_header(header, false).is_large());
            let postfix = u32::from(header & 3);
            let direct = u128::from(header >> 2) << postfix;
            for symbol in 16..layout.alphabet() {
                let width = layout.extra_bits(symbol);
                for extra in [0, (1u64 << width) - 1] {
                    let expected = if (symbol as u128) < 16 + direct {
                        symbol as u128 - 15
                    } else {
                        let code = symbol as u128 - 16 - direct;
                        let bucket = code / (1 << postfix);
                        let low = code % (1 << postfix);
                        let n = 1 + bucket / 2;
                        ((2 + bucket % 2) * (1 << n) - 4 + u128::from(extra)) * (1 << postfix)
                            + low
                            + direct
                            + 1
                    };
                    assert_eq!(layout.long_distance(symbol, extra), expected);
                }
                let maximum = layout.long_distance(symbol, (1u64 << width) - 1);
                assert_eq!(
                    layout.validate_symbol(symbol).is_ok(),
                    maximum <= MAX_DISTANCE
                );
            }
            assert!(layout.validate_symbol(layout.alphabet()).is_err());
        }
    }
    #[test]
    fn cache_offsets_cannot_make_zero_or_overflowed_distances() {
        let layout = DistanceLayout::default();
        let mut ones = Cache::default();
        for _ in 0..4 {
            ones.push(1);
        }
        for symbol in [4, 6, 8, 10, 12, 14] {
            assert!(layout.resolve(symbol, 0, &ones).is_err());
        }
        let mut huge = Cache::default();
        for _ in 0..4 {
            huge.push(u64::MAX);
        }
        assert!(layout.resolve(5, 0, &huge).is_err());
        assert_eq!(layout.resolve(3, 0, &Cache::default()).unwrap(), 16);
        assert!(
            DistanceLayout::from_header(63, true)
                .resolve(1127, u64::MAX, &Cache::default())
                .is_err()
        );
    }

    #[test]
    fn cache_orders_recent_distances_and_pop_push_keeps_the_most_recent() {
        let mut cache = Cache::default();
        assert_eq!(
            [
                cache.recent(0),
                cache.recent(1),
                cache.recent(2),
                cache.recent(3)
            ],
            [4, 11, 15, 16]
        );
        cache.push(100);
        assert_eq!(
            [
                cache.recent(0),
                cache.recent(1),
                cache.recent(2),
                cache.recent(3)
            ],
            [100, 4, 11, 15]
        );
        let last = cache.pop();
        assert_eq!(last, 100);
        cache.push(last);
        assert_eq!(
            [
                cache.recent(0),
                cache.recent(1),
                cache.recent(2),
                cache.recent(3)
            ],
            [100, 4, 11, 15]
        );
        for extra in 0..20 {
            cache.push(1000 + extra);
        }
        assert_eq!(cache.recent(0), 1019);
        assert_eq!(cache.recent(3), 1016);
    }

    #[test]
    fn symbol_table_matches_full_resolution_for_every_layout() {
        for large in [false, true] {
            for header in 0..64 {
                let layout = DistanceLayout::from_header(header, large);
                let mut table = Vec::new();
                layout
                    .fill_table(&mut table, &mut Memory::default())
                    .unwrap();
                assert_eq!(table.len(), layout.alphabet());
                assert_eq!(table[3], Partial::default());
                for (symbol, &entry) in table.iter().enumerate().skip(16) {
                    if layout.validate_symbol(symbol).is_err() {
                        continue;
                    }
                    assert_eq!(entry.width, u64::from(layout.extra_bits(symbol)));
                    for extra in [0, (1u64 << entry.width) - 1] {
                        assert_eq!(
                            entry.base + (extra << layout.postfix()),
                            layout.resolve(symbol, extra, &Cache::default()).unwrap(),
                            "header {header} large {large} symbol {symbol}"
                        );
                    }
                }
            }
        }
        // A smaller alphabet reuses the table without shrinking it.
        let mut table = Vec::new();
        DistanceLayout::from_header(63, true)
            .fill_table(&mut table, &mut Memory::default())
            .unwrap();
        let len = table.len();
        DistanceLayout::default()
            .fill_table(&mut table, &mut Memory::default())
            .unwrap();
        assert_eq!(table.len(), len);
    }

    #[test]
    fn the_constant_standard_table_is_what_the_layout_computes() {
        // `STANDARD_TABLE` is built at compile time, so nothing executes
        // `standard_table` at run time and the decoder never recomputes what
        // it holds. Calling it here pins the constant to the layout it claims
        // to be a copy of.
        let computed = standard_table();
        assert_eq!(computed.len(), STANDARD_ALPHABET);
        assert_eq!(computed, STANDARD_TABLE);
        let layout = DistanceLayout::default();
        assert!(layout.is_standard());
        for (symbol, entry) in STANDARD_TABLE.iter().enumerate() {
            assert_eq!(*entry, layout.entry(symbol), "symbol {symbol}");
        }
        assert_eq!(layout.table(&[]), &STANDARD_TABLE);
    }
}
