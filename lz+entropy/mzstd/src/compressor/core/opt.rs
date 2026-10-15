//! The optimal parser: `btopt`, `btultra` and `btultra2` (`zstd_opt.c`).
//!
//! Prices are fixed-point bit costs (1/256 bit) derived from running symbol
//! statistics, exactly as the reference computes them, so the chosen parse
//! — and therefore every emitted byte — matches it.

use alloc::vec::Vec;

use fearless_simd::{Simd, dispatch};
use fearless_simd_macros::simd;

use super::entropy::{Tables, fse, hist, huf};
use super::ldm::RawSeqStore;
use super::match_state::{BlockInput, DictMatchState, MatchState};
use super::mem;
use super::opt_count;
use super::params::ParamSwitch;
use super::seq_store::{
    SeqStore, ll_code, ml_code, new_rep, offset_to_offbase, repcode_to_offbase,
};
use crate::shared::format::{
    LL_BITS, MAX_LIT, MAX_LL, MAX_ML, MAX_OFF, MIN_MATCH, ML_BITS, REP_NUM, highbit32,
};

/// `ZSTD_OPT_NUM`.
pub(crate) const OPT_NUM: usize = 1 << 12;
/// `ZSTD_OPT_SIZE`.
pub(crate) const OPT_SIZE: usize = OPT_NUM + 3;

const LITFREQ_ADD: u32 = 2;
/// Literal runs from this length on refresh the whole literal weight table
/// ([`weights`]) rather than each literal's entry as it is counted.
const BULK_WEIGHT_LITERALS: usize = 64;
const MAX_PRICE: i32 = 1 << 30;
const PREDEF_THRESHOLD: usize = 8;
const BITCOST_ACCURACY: u32 = 8;
const BITCOST_MULTIPLIER: u32 = 1 << BITCOST_ACCURACY;
/// `prime4bytes`, `prime5bytes` and `prime6bytes` of `ZSTD_hashPtr`.
const PRIME_4_BYTES: u32 = 2_654_435_761;
const PRIME_5_BYTES: u64 = 889_523_592_379;
const PRIME_6_BYTES: u64 = 227_718_039_650_203;
/// `prime7bytes`: dictionary loads hash with an unclamped minimum match.
const PRIME_7_BYTES: u64 = 58_295_818_150_454_627;

/// How prices are derived (`ZSTD_OptPrice_e`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum PriceType {
    #[default]
    Dynamic,
    Predef,
}

/// One position of the forward pass (`ZSTD_optimal_t`).
#[derive(Copy, Clone, Debug, Default)]
struct Optimal {
    price: i32,
    off: u32,
    mlen: u32,
    litlen: u32,
    rep: [u32; REP_NUM],
}

/// One candidate match (`ZSTD_match_t`).
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Match {
    pub(crate) off: u32,
    pub(crate) len: u32,
}

/// Pricing statistics of the optimal parser (`optState_t`).
#[derive(Clone, Debug, Default)]
pub(crate) struct OptState {
    pub(crate) lit_freq: Vec<u32>,
    pub(crate) lit_length_freq: Vec<u32>,
    pub(crate) match_length_freq: Vec<u32>,
    pub(crate) off_code_freq: Vec<u32>,
    match_table: Vec<Match>,
    price_table: Vec<Optimal>,
    pub(crate) lit_sum: u32,
    pub(crate) lit_length_sum: u32,
    pub(crate) match_length_sum: u32,
    pub(crate) off_code_sum: u32,
    pub(crate) lit_sum_base_price: u32,
    pub(crate) lit_length_sum_base_price: u32,
    pub(crate) match_length_sum_base_price: u32,
    pub(crate) off_code_sum_base_price: u32,
    pub(crate) price_type: PriceType,
    /// The frame's literal compression switch.
    pub(crate) literal_compression: ParamSwitch,
    /// The frequencies' weights, kept current with them.
    tables: PriceTables,
}

/// The symbol statistics' `WEIGHT`s, folded with each symbol's extra bits,
/// so that a dynamic price is one table read plus the sum's base price.
///
/// [`OptState::rescale_freqs`] rebuilds them for the block's optimization
/// level and [`OptState::update_stats`] patches the entries it changes,
/// so every price equals the reference's `ZSTD_litLengthPrice`,
/// `ZSTD_getMatchPrice` and `ZSTD_rawLiteralsCost`, computed from the
/// frequencies on each call. Sums wrap as the reference's `U32` sums do.
/// Each table is a power of two longer than any code its index function
/// returns, so a masked index needs no bounds check.
#[derive(Clone, Debug)]
struct PriceTables {
    /// `WEIGHT(litFreq[byte])`.
    literal: [u32; 256],
    /// `LL_bits[code] * BITCOST_MULTIPLIER - WEIGHT(litLengthFreq[code])`.
    lit_length: [u32; 64],
    /// `ML_bits[code] * BITCOST_MULTIPLIER - WEIGHT(matchLengthFreq[code])`
    /// plus the per-sequence surcharge.
    match_length: [u32; 128],
    /// `code * BITCOST_MULTIPLIER - WEIGHT(offCodeFreq[code])` plus the
    /// long-distance handicap below level 2.
    off_code: [u32; 32],
    /// The optimization level the weights were taken at.
    opt_level: u32,
}

impl Default for PriceTables {
    fn default() -> Self {
        Self {
            literal: [0; 256],
            lit_length: [0; 64],
            match_length: [0; 128],
            off_code: [0; 32],
            opt_level: 0,
        }
    }
}

impl PriceTables {
    /// The literal length entry for `code`.
    #[inline(always)]
    fn set_lit_length(&mut self, code: usize, freq: u32) {
        self.lit_length[code & 63] = (u32::from(LL_BITS[code]) * BITCOST_MULTIPLIER)
            .wrapping_sub(weight(freq, self.opt_level));
    }

    /// The match length entry for `code`.
    #[inline(always)]
    fn set_match_length(&mut self, code: usize, freq: u32) {
        self.match_length[code & 127] = (u32::from(ML_BITS[code]) * BITCOST_MULTIPLIER)
            .wrapping_sub(weight(freq, self.opt_level))
            .wrapping_add(BITCOST_MULTIPLIER / 5);
    }

    /// The offset code entry for `code`.
    #[inline(always)]
    fn set_off_code(&mut self, code: usize, freq: u32) {
        let code32 = code as u32;
        let mut price = (code32 * BITCOST_MULTIPLIER).wrapping_sub(weight(freq, self.opt_level));
        if self.opt_level < 2 && code32 >= 20 {
            price = price.wrapping_add((code32 - 19) * 2 * BITCOST_MULTIPLIER);
        }
        self.off_code[code & 31] = price;
    }
}

impl OptState {
    /// Sizes the frequency tables (`ZSTD_reset_matchState`'s opt space).
    ///
    /// The position and match tables are left to [`Self::reserve_matches`]
    /// and [`grow_positions`]: the reference carves them uninitialized
    /// from its workspace, and filling all [`OPT_SIZE`] entries here
    /// dominated frames that never reach the parser (empty input) or use a
    /// few of them (tiny or incompressible input).
    pub(crate) fn allocate(&mut self) {
        self.lit_freq.resize(MAX_LIT as usize + 1, 0);
        self.lit_length_freq.resize(MAX_LL as usize + 1, 0);
        self.match_length_freq.resize(MAX_ML as usize + 1, 0);
        self.off_code_freq.resize(MAX_OFF as usize + 1, 0);
    }

    /// Grows the match table to hold every match one search can report in
    /// a block of `block_len` bytes with `nb_compares` tree comparisons.
    ///
    /// A search reports matches of strictly increasing lengths: up to
    /// three repeat offsets or one 3-byte match, one per tree comparison,
    /// and one LDM candidate. Lengths lie within the block and the tree
    /// walk stops past [`OPT_NUM`], so the list is also shorter than
    /// `block_len + 3` and than [`OPT_SIZE`]. The table only grows, so each
    /// entry is written once per engine.
    fn reserve_matches(&mut self, block_len: usize, nb_compares: u32) {
        let len = OPT_SIZE.min(block_len + 3).min(nb_compares as usize + 5);
        if self.match_table.len() < len {
            self.match_table.resize(len, Match::default());
        }
    }

    /// Bytes held by the parser's position and match tables.
    pub(crate) fn table_bytes(&self) -> usize {
        self.price_table.capacity() * core::mem::size_of::<Optimal>()
            + self.match_table.capacity() * core::mem::size_of::<Match>()
    }

    fn compressed_literals(&self) -> bool {
        self.literal_compression != ParamSwitch::Disable
    }

    fn set_base_prices(&mut self, opt_level: u32) {
        if self.compressed_literals() {
            self.lit_sum_base_price = weight(self.lit_sum, opt_level);
        }
        self.lit_length_sum_base_price = weight(self.lit_length_sum, opt_level);
        self.match_length_sum_base_price = weight(self.match_length_sum, opt_level);
        self.off_code_sum_base_price = weight(self.off_code_sum, opt_level);
    }

    /// `ZSTD_rescaleFreqs`: the first block of a frame starts from the
    /// inherited entropy tables when a dictionary validated them
    /// (`symbol_costs`, `ms->opt.symbolCosts`), else from `src`'s literals
    /// and fixed sequence statistics; later blocks scale the running
    /// statistics down.
    fn rescale_freqs(&mut self, src: &[u8], opt_level: u32, symbol_costs: Option<&Tables>) {
        let compressed_literals = self.compressed_literals();
        self.price_type = PriceType::Dynamic;
        if self.lit_length_sum == 0 {
            if src.len() <= PREDEF_THRESHOLD {
                self.price_type = PriceType::Predef;
            }
            if let Some(tables) = symbol_costs
                && tables.huf.repeat == huf::Repeat::Valid
            {
                // Huffman statistics over every byte: tables presumed
                // generated by a dictionary.
                self.price_type = PriceType::Dynamic;
                self.seed_from_dictionary(tables);
            } else {
                self.seed_without_dictionary(src);
            }
        } else {
            if compressed_literals {
                self.lit_sum = scale_stats(&mut self.lit_freq, MAX_LIT, 12);
            }
            self.lit_length_sum = scale_stats(&mut self.lit_length_freq, MAX_LL, 11);
            self.match_length_sum = scale_stats(&mut self.match_length_freq, MAX_ML, 11);
            self.off_code_sum = scale_stats(&mut self.off_code_freq, MAX_OFF, 11);
        }
        self.set_base_prices(opt_level);
        self.rebuild_tables(opt_level);
    }

    /// `ZSTD_rescaleFreqs`'s first-block statistics from a dictionary's
    /// entropy tables: each symbol's frequency is `2^(scale - bits)` for
    /// the bits its code costs (literals scaled to 2K, sequence symbols to
    /// 1K), and `1` for a symbol without a code.
    ///
    /// The reference asserts that no code is longer than the scale; a
    /// malformed table past that shifts as its unchecked `1 << n` does on
    /// common targets (the count taken modulo 32) instead of failing.
    fn seed_from_dictionary(&mut self, tables: &Tables) {
        let frequency = |bit_cost: u32, scale_log: u32| {
            if bit_cost != 0 {
                1u32.wrapping_shl(scale_log.wrapping_sub(bit_cost))
            } else {
                1
            }
        };
        if self.compressed_literals() {
            let mut sum = 0u32;
            for (literal, freq) in (0..=MAX_LIT).zip(self.lit_freq.iter_mut()) {
                *freq = frequency(tables.huf.table.symbol_nb_bits(literal), 11);
                sum = sum.wrapping_add(*freq);
            }
            self.lit_sum = sum;
        }
        let seed = |freqs: &mut [u32], table: &fse::CTable, last: u32| {
            let mut sum = 0u32;
            for (symbol, freq) in (0..=last).zip(freqs.iter_mut()) {
                *freq = frequency(table.max_nb_bits(symbol), 10);
                sum = sum.wrapping_add(*freq);
            }
            sum
        };
        self.lit_length_sum = seed(&mut self.lit_length_freq, &tables.fse.litlength, MAX_LL);
        self.match_length_sum = seed(&mut self.match_length_freq, &tables.fse.matchlength, MAX_ML);
        self.off_code_sum = seed(&mut self.off_code_freq, &tables.fse.offcode, MAX_OFF);
    }

    /// `ZSTD_rescaleFreqs`'s first-block statistics without a dictionary:
    /// literal frequencies from `src`, fixed ones for the sequence symbols.
    fn seed_without_dictionary(&mut self, src: &[u8]) {
        const BASE_LL_FREQS: [u32; MAX_LL as usize + 1] = [
            4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
        ];
        const BASE_OFC_FREQS: [u32; MAX_OFF as usize + 1] = [
            6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
        ];
        if self.compressed_literals() {
            let mut lit = MAX_LIT;
            let _ = hist::count(&mut self.lit_freq, &mut lit, src, false);
            self.lit_sum = downscale_stats(&mut self.lit_freq, MAX_LIT, 8, false);
        }
        self.lit_length_freq.copy_from_slice(&BASE_LL_FREQS);
        self.lit_length_sum = BASE_LL_FREQS.iter().sum();
        self.match_length_freq.fill(1);
        self.match_length_sum = MAX_ML + 1;
        self.off_code_freq.copy_from_slice(&BASE_OFC_FREQS);
        self.off_code_sum = BASE_OFC_FREQS.iter().sum();
    }

    /// Recomputes every [`PriceTables`] entry from the frequencies.
    ///
    /// Equal to calling each `set_*` in turn, but the weights of a whole
    /// table are taken at once ([`weights`]) and folded with straight-line
    /// loops the compiler vectorizes: a block of a few bytes pays for this
    /// rebuild on every call.
    fn rebuild_tables(&mut self, opt_level: u32) {
        let tables = &mut self.tables;
        tables.opt_level = opt_level;
        if self.literal_compression != ParamSwitch::Disable {
            weights(&self.lit_freq, opt_level, &mut tables.literal);
        }
        let mut weight_of = [0u32; 64];
        weights(&self.lit_length_freq, opt_level, &mut weight_of);
        for ((entry, &bits), &weight) in tables.lit_length.iter_mut().zip(&LL_BITS).zip(&weight_of)
        {
            *entry = (u32::from(bits) * BITCOST_MULTIPLIER).wrapping_sub(weight);
        }
        weights(&self.match_length_freq, opt_level, &mut weight_of);
        for ((entry, &bits), &weight) in
            tables.match_length.iter_mut().zip(&ML_BITS).zip(&weight_of)
        {
            *entry = (u32::from(bits) * BITCOST_MULTIPLIER)
                .wrapping_sub(weight)
                .wrapping_add(BITCOST_MULTIPLIER / 5);
        }
        weights(&self.off_code_freq, opt_level, &mut weight_of);
        for (code, (entry, &weight)) in tables.off_code.iter_mut().zip(&weight_of).enumerate() {
            let code = code as u32;
            let handicap = if opt_level < 2 && code >= 20 {
                (code - 19) * 2 * BITCOST_MULTIPLIER
            } else {
                0
            };
            *entry = (code * BITCOST_MULTIPLIER)
                .wrapping_sub(weight)
                .wrapping_add(handicap);
        }
    }

    /// `ZSTD_rawLiteralsCost` for one literal.
    #[inline(always)]
    fn literal_price(&self, literal: u8) -> i32 {
        if !self.compressed_literals() {
            return ((1 << 3) * BITCOST_MULTIPLIER) as i32;
        }
        if self.price_type == PriceType::Predef {
            return (6 * BITCOST_MULTIPLIER) as i32;
        }
        let lit_price_max = self.lit_sum_base_price - BITCOST_MULTIPLIER;
        let lit_price = self.tables.literal[usize::from(literal)].min(lit_price_max);
        self.lit_sum_base_price.wrapping_sub(lit_price) as i32
    }

    /// `ZSTD_litLengthPrice`.
    #[inline(always)]
    fn lit_length_price(&self, lit_length: u32) -> i32 {
        if self.price_type == PriceType::Predef {
            return weight(lit_length, self.tables.opt_level) as i32;
        }
        if lit_length as usize == crate::shared::format::BLOCK_SIZE_MAX {
            return BITCOST_MULTIPLIER as i32 + self.lit_length_price(lit_length - 1);
        }
        let code = usize::from(ll_code(lit_length));
        self.tables.lit_length[code & 63].wrapping_add(self.lit_length_sum_base_price) as i32
    }

    /// The offset part of `ZSTD_getMatchPrice`; [`Self::match_length_price`]
    /// adds the rest. The sum wraps as the reference's `U32` sum does.
    #[inline(always)]
    fn offset_price(&self, off_base: u32) -> u32 {
        let off_code = highbit32(off_base);
        if self.price_type == PriceType::Predef {
            return (16 + off_code) * BITCOST_MULTIPLIER;
        }
        self.tables.off_code[off_code as usize & 31].wrapping_add(self.off_code_sum_base_price)
    }

    /// The match length part of `ZSTD_getMatchPrice`, including its
    /// heuristic surcharge per sequence.
    #[inline(always)]
    fn match_length_price(&self, match_length: u32) -> u32 {
        let ml_base = match_length - MIN_MATCH as u32;
        if self.price_type == PriceType::Predef {
            return weight(ml_base, self.tables.opt_level);
        }
        let code = usize::from(ml_code(ml_base));
        self.tables.match_length[code & 127].wrapping_add(self.match_length_sum_base_price)
    }

    /// `ZSTD_getMatchPrice`.
    #[cfg(test)]
    fn match_price(&self, off_base: u32, match_length: u32) -> u32 {
        self.offset_price(off_base)
            .wrapping_add(self.match_length_price(match_length))
    }

    /// `ZSTD_updateStats`.
    fn update_stats(&mut self, literals: &[u8], off_base: u32, match_length: u32) {
        if self.compressed_literals() {
            // A byte indexes the 256 frequencies without a bounds check, and
            // the level's weight form is chosen once for the run.
            let level = self.tables.opt_level;
            let literal_weights = &mut self.tables.literal;
            if let Some(freqs) = self.lit_freq.first_chunk_mut::<256>() {
                if literals.len() >= BULK_WEIGHT_LITERALS {
                    // A long run touches most of the table: count it as
                    // the reference does, then take every weight at once.
                    for &lit in literals {
                        freqs[usize::from(lit)] += LITFREQ_ADD;
                    }
                    weights(freqs, level, literal_weights);
                } else if level != 0 {
                    for &lit in literals {
                        let freq = &mut freqs[usize::from(lit)];
                        *freq += LITFREQ_ADD;
                        literal_weights[usize::from(lit)] = frac_weight(*freq);
                    }
                } else {
                    for &lit in literals {
                        let freq = &mut freqs[usize::from(lit)];
                        *freq += LITFREQ_ADD;
                        literal_weights[usize::from(lit)] = bit_weight(*freq);
                    }
                }
            } else {
                for &lit in literals {
                    let freq = &mut self.lit_freq[usize::from(lit)];
                    *freq += LITFREQ_ADD;
                    literal_weights[usize::from(lit)] = weight(*freq, level);
                }
            }
            self.lit_sum += literals.len() as u32 * LITFREQ_ADD;
        }
        let ll = usize::from(ll_code(literals.len() as u32));
        self.lit_length_freq[ll] += 1;
        self.tables.set_lit_length(ll, self.lit_length_freq[ll]);
        self.lit_length_sum += 1;
        let off_code = highbit32(off_base) as usize;
        self.off_code_freq[off_code] += 1;
        self.tables
            .set_off_code(off_code, self.off_code_freq[off_code]);
        self.off_code_sum += 1;
        let ml = usize::from(ml_code(match_length - MIN_MATCH as u32));
        self.match_length_freq[ml] += 1;
        self.tables.set_match_length(ml, self.match_length_freq[ml]);
        self.match_length_sum += 1;
    }
}

/// Grows the parser's position table to `end` positions and returns it.
///
/// The parser reads and writes positions up to two past the furthest
/// match end it prices, which lies below [`OPT_NUM`] and before the end
/// of the block. It grows the table when it first finds a match short of
/// that, instead of covering [`OPT_SIZE`] positions up front, which an
/// incompressible block never reaches. A table only grows, so each entry
/// is written once per engine.
///
/// The parser indexes the returned slice, and calls this only when that
/// slice is too short: the table itself is touched only when it grows, so
/// the slice stays in registers across the parser's calls.
#[cold]
#[inline(never)]
fn grow_positions(table: &mut Vec<Optimal>, end: usize) -> &mut [Optimal] {
    if table.len() < end {
        table.resize(end, Optimal::default());
    }
    table
}

/// `ZSTD_bitWeight`.
const fn bit_weight(stat: u32) -> u32 {
    highbit32(stat + 1) * BITCOST_MULTIPLIER
}

/// `ZSTD_fracWeight`.
const fn frac_weight(raw_stat: u32) -> u32 {
    let stat = raw_stat + 1;
    let hb = highbit32(stat);
    let b_weight = hb * BITCOST_MULTIPLIER;
    let f_weight = (stat << BITCOST_ACCURACY) >> hb;
    b_weight + f_weight
}

/// `WEIGHT(stat, optLevel)`.
#[inline(always)]
const fn weight(stat: u32, opt_level: u32) -> u32 {
    if opt_level != 0 {
        frac_weight(stat)
    } else {
        bit_weight(stat)
    }
}

/// `WEIGHT(stats[i], opt_level)` into `out[i]`, for as many entries as both
/// hold.
///
/// For `stat + 1 < 2^24`, `stat + 1` converts to `f32` exactly, its
/// exponent is `highbit32(stat + 1) + 127`, and its top eight mantissa bits
/// are `((stat + 1) << 8 >> highbit) - 256`. So `bits >> 23` gives
/// `ZSTD_bitWeight` and `bits >> 15` gives `ZSTD_fracWeight`, each up to a
/// constant, in a loop the compiler vectorizes. From `2^24` on, the
/// reference's `stat << 8` wraps, so a table holding such a statistic is
/// redone with the integer form, which wraps the same way.
fn weights(stats: &[u32], opt_level: u32, out: &mut [u32]) {
    let bits = |stat: u32| ((stat + 1) as i32 as f32).to_bits();
    let mut seen = 0u32;
    if opt_level != 0 {
        for (weight_out, &stat) in out.iter_mut().zip(stats) {
            seen |= stat + 1;
            *weight_out = (bits(stat) >> 15).wrapping_sub(126 * BITCOST_MULTIPLIER);
        }
    } else {
        for (weight_out, &stat) in out.iter_mut().zip(stats) {
            seen |= stat + 1;
            *weight_out = ((bits(stat) >> 23).wrapping_sub(127)).wrapping_mul(BITCOST_MULTIPLIER);
        }
    }
    if seen >> 24 != 0 {
        for (weight_out, &stat) in out.iter_mut().zip(stats) {
            *weight_out = weight(stat, opt_level);
        }
    }
}

/// `ZSTD_downscaleStats`.
fn downscale_stats(table: &mut [u32], last: u32, shift: u32, base1: bool) -> u32 {
    let mut sum = 0u32;
    for value in &mut table[..=last as usize] {
        let base = if base1 { 1 } else { u32::from(*value > 0) };
        let new_stat = base + (*value >> shift);
        sum += new_stat;
        *value = new_stat;
    }
    sum
}

/// `ZSTD_scaleStats`.
fn scale_stats(table: &mut [u32], last: u32, log_target: u32) -> u32 {
    let prev_sum: u32 = table[..=last as usize].iter().sum();
    let factor = prev_sum >> log_target;
    if factor <= 1 {
        return prev_sum;
    }
    downscale_stats(table, last, highbit32(factor), true)
}

// ---------------------------------------------------------------------------
// Binary tree.

/// Per-block constants of the tree searches.
///
/// The reference walks raw pointers: `base + index` in the prefix and
/// `dictBase + index` in the external dictionary. Here both segments are
/// plain slices, so an index below `dict_limit` is an offset from the end of
/// `ext` and one at or above it an offset from the start of `prefix`.
#[derive(Copy, Clone, Debug)]
struct Ctx<'a, S> {
    simd: S,
    window: mem::Window<'a>,
    /// Bytes from index `dict_limit` to the end of the block.
    prefix: &'a [u8],
    /// Bytes from index `low_limit` to `dict_limit`.
    ext: &'a [u8],
    dict_limit: u32,
    iend: u32,
    mls: u32,
    /// The attached dictionary; [`DictSearch::EMPTY`], and never read,
    /// outside the dictMatchState variants. Behind a reference, so that
    /// the other variants' context keeps its size and field offsets.
    dms: &'a DictSearch<'a>,
}

/// An attached dictionary's tables and bytes for one block: the `dms*`
/// constants of `ZSTD_insertBtAndGetAllMatches`.
///
/// Indices are the dictionary's own. `bytes` runs from index `start` to
/// `high_limit` (`dmsEnd`); a run reaching its end continues at the start
/// of the block's prefix, as the reference's `ZSTD_count_2segments` does.
#[derive(Copy, Clone, Debug)]
struct DictSearch<'a> {
    hash_table: &'a [u32],
    /// The dictionary's tree, cut to its `2 << dmsBtLog` cells.
    bt: &'a [u32],
    bytes: &'a [u8],
    /// Index of `bytes[0]`.
    start: u32,
    /// `dmsLowLimit`: candidates at or below it end the walk.
    low_limit: u32,
    /// `dmsHighLimit`: the index one past the dictionary's content.
    high_limit: u32,
    /// `dmsHashLog`.
    hash_log: u32,
    /// `dmsBtMask`.
    bt_mask: u32,
    /// `dmsBtLow`: the walk stops after a candidate at or below it.
    bt_low: u32,
}

impl DictSearch<'static> {
    /// No dictionary: empty tables and limits that end every walk.
    const EMPTY: Self = Self {
        hash_table: &[],
        bt: &[],
        bytes: &[],
        start: 0,
        low_limit: 0,
        high_limit: 0,
        hash_log: 0,
        bt_mask: 0,
        bt_low: 0,
    };
}

impl<'a> DictSearch<'a> {
    /// The constants of `dms` (`ms->dictMatchState`).
    fn new(dms: &DictMatchState<'a>) -> Self {
        let cparams = dms.ms.cparams;
        let bt_mask = (1u32 << (cparams.chain_log - 1)) - 1;
        let prefix = dms.window.prefix;
        let high_limit = prefix.end();
        let low_limit = dms.ms.window.low_limit;
        let bt_low = if bt_mask < high_limit.wrapping_sub(low_limit) {
            high_limit - bt_mask
        } else {
            low_limit
        };
        Self {
            hash_table: &dms.ms.hash_table,
            bt: &dms.ms.chain_table[..2 * (bt_mask as usize + 1)],
            bytes: prefix.slice(prefix.start(), high_limit),
            start: prefix.start(),
            low_limit,
            high_limit,
            hash_log: cparams.hash_log,
            bt_mask,
            bt_low,
        }
    }

    /// `ZSTD_readMINMATCH`'s word at dictionary index `index`.
    fn read32(&self, index: u32) -> u32 {
        let at = index.wrapping_sub(self.start) as usize;
        let mut word = [0u8; 4];
        word.copy_from_slice(&self.bytes[at..at + 4]);
        u32::from_le_bytes(word)
    }
}

impl<'a, S: Simd> Ctx<'a, S> {
    fn new(simd: S, block: &BlockInput<'a>, mls: u32) -> Self {
        let window = block.window;
        let prefix = window.prefix;
        let ext = window.ext;
        Self {
            simd,
            window,
            // Cut at the block's end, so that the end is the slice's: a
            // run then needs no limit of its own.
            prefix: prefix.slice(prefix.start(), block.end),
            ext: ext.slice(ext.start(), ext.end()),
            dict_limit: prefix.start(),
            iend: block.end,
            mls,
            dms: &DictSearch::EMPTY,
        }
    }

    /// [`Self::new`] with the attached dictionary `dms` searched too.
    fn with_dictionary(simd: S, block: &BlockInput<'a>, mls: u32, dms: &'a DictSearch<'a>) -> Self {
        Self {
            dms,
            ..Self::new(simd, block, mls)
        }
    }

    /// `ZSTD_count_2segments(ip, dmsBase + candidate, iend, dmsEnd,
    /// prefixStart)` for dictionary index `candidate`. A candidate past the
    /// dictionary's end counts nothing, as the reference's does (its
    /// segment end then lies before `ip`).
    #[inline(always)]
    fn count_dictionary(&self, ip: u32, candidate: u32) -> usize {
        let earlier = candidate.wrapping_sub(self.dms.start) as usize;
        if earlier > self.dms.bytes.len() {
            return 0;
        }
        opt_count::count_2segments(
            self.simd,
            self.prefix,
            self.offset(ip),
            self.dms.bytes,
            earlier,
            self.prefix.len(),
        )
    }

    /// Offset of prefix index `index` in [`Self::prefix`].
    #[inline(always)]
    const fn offset(&self, index: u32) -> usize {
        (index - self.dict_limit) as usize
    }

    /// Offset of external index `index` in [`Self::ext`].
    #[inline(always)]
    const fn ext_offset(&self, index: u32) -> usize {
        self.ext.len() - (self.dict_limit - index) as usize
    }

    /// The byte at window index `index`.
    #[inline(always)]
    fn byte<const EXT: bool>(&self, index: usize) -> u8 {
        let dict_limit = self.dict_limit as usize;
        if !EXT || index >= dict_limit {
            self.prefix[index - dict_limit]
        } else {
            self.ext[self.ext.len() - (dict_limit - index)]
        }
    }

    /// Little-endian `u32` at prefix index `index`.
    #[inline(always)]
    fn read32(&self, index: u32) -> u32 {
        let at = self.offset(index);
        let mut word = [0u8; 4];
        word.copy_from_slice(&self.prefix[at..at + 4]);
        u32::from_le_bytes(word)
    }

    /// `ZSTD_hashPtr(base + index, hash_log, MLS)`: the 4-byte hash for
    /// `MLS` 3 and 4, the 5-, 6- and 7-byte ones above (7 only in
    /// dictionary loads, which do not clamp the minimum match).
    #[inline(always)]
    fn hash<const MLS: u32>(&self, index: u32, hash_log: u32) -> usize {
        if MLS <= 4 {
            return (self.read32(index).wrapping_mul(PRIME_4_BYTES) >> (32 - hash_log)) as usize;
        }
        let at = self.offset(index);
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.prefix[at..at + 8]);
        let (shift, prime) = match MLS {
            5 => (24, PRIME_5_BYTES),
            6 => (16, PRIME_6_BYTES),
            _ => (8, PRIME_7_BYTES),
        };
        ((u64::from_le_bytes(word) << shift).wrapping_mul(prime) >> (64 - hash_log)) as usize
    }

    /// Hints that the hash cells of `ip + 4` will be read soon: the parser
    /// and the tree update mostly go position after position, and a search
    /// is much shorter than a load from memory, so the hint runs a few
    /// searches ahead. With `HASH3` and `MLS` 3 the 3-byte table's cell is
    /// hinted too. A position whose bytes do not lie in the block is
    /// skipped.
    #[inline(always)]
    fn prefetch_ahead<const MLS: u32, const HASH3: bool>(&self, finder: &Finder<'_>, ip: u32) {
        let ahead = ip + 4;
        if MLS <= 4 {
            // The 4 bytes at `ip + 4` end the word at `ip`, which the
            // search reads too.
            let at = self.offset(ip);
            let Some(bytes) = self.prefix.get(at..at + 8) else {
                return;
            };
            let mut word = [0u8; 8];
            word.copy_from_slice(bytes);
            let word = (u64::from_le_bytes(word) >> 32) as u32;
            let h = word.wrapping_mul(PRIME_4_BYTES) >> (32 - finder.hash_log);
            prefetch_l1(finder.hash_table, h as usize);
            if HASH3 && MLS == 3 {
                let h3 = mem::hash3(word, finder.hash_log3, 0);
                prefetch_l1(finder.hash_table3, h3);
            }
        } else if self.offset(ahead) + 8 <= self.prefix.len() {
            prefetch_l1(finder.hash_table, self.hash::<MLS>(ahead, finder.hash_log));
        }
    }

    /// `ZSTD_count(ip, match, iend)` inside the prefix.
    #[inline(always)]
    fn count(&self, ip: u32, candidate: u32) -> usize {
        opt_count::count(
            self.simd,
            self.prefix,
            self.offset(ip),
            self.offset(candidate),
            self.prefix.len(),
        )
    }

    /// `ZSTD_count_2segments(ip, dictBase + candidate, iend, dictEnd,
    /// prefixStart)` for a candidate in the external dictionary.
    #[inline(always)]
    fn count_2segments(&self, ip: u32, candidate: u32) -> usize {
        opt_count::count_2segments(
            self.simd,
            self.prefix,
            self.offset(ip),
            self.ext,
            self.ext_offset(candidate),
            self.prefix.len(),
        )
    }

    /// The bytes from prefix index `curr` to the end of the block.
    #[inline(always)]
    fn current(&self, curr: u32) -> &'a [u8] {
        &self.prefix[self.offset(curr)..]
    }

    /// The tree walk's comparison: `known` bytes are already equal; returns
    /// the full common length of `curr` (whose bytes to the end of the
    /// block are `current`) and `candidate`, the latter read through both
    /// segments with `EXT`, and whether the candidate's first differing
    /// byte is the smaller (`false` when the run reaches the end of the
    /// block, where the walk stops without looking).
    #[inline(always)]
    fn extend<const EXT: bool>(
        &self,
        current: &[u8],
        curr: u32,
        candidate: u32,
        known: usize,
    ) -> (usize, bool) {
        let dict_limit = self.dict_limit as usize;
        if !EXT || candidate as usize + known >= dict_limit {
            let (length, smaller) = opt_count::count_ordered(
                self.simd,
                current,
                self.prefix,
                // The candidate may start in the external segment with its
                // known prefix reaching into this one.
                candidate + known as u32 - self.dict_limit,
                known as u32,
            );
            (known + length, smaller)
        } else {
            let known32 = known as u32;
            let length = known + self.count_2segments(curr + known32, candidate + known32);
            let smaller = length < current.len()
                && self.byte::<EXT>(candidate as usize + length)
                    < self.byte::<EXT>(curr as usize + length);
            (length, smaller)
        }
    }
}

/// Hints that `table[index]` will be read soon (`PREFETCH_L1`). The pointer
/// is only handed to the prefetch instruction, never dereferenced, so a
/// hint past the table is harmless; targets without one ignore it.
#[inline(always)]
fn prefetch_l1<T>(table: &[T], index: usize) {
    branches::prefetch_read_data::<T, 0>(table.as_ptr().wrapping_add(index));
}

/// `ZSTD_readMINMATCH` comparison of `length` (3 or 4) bytes.
#[inline(always)]
fn min_match_equal(x: u32, y: u32, length: u32) -> bool {
    if length == 3 {
        (x << 8) == (y << 8)
    } else {
        x == y
    }
}

/// The binary-tree match finder's tables and its parameters for one block.
///
/// The reference reads `ms->cParams` and the window on every search; here
/// whatever is fixed for the block is derived once, so a search loads it
/// rather than recomputing it from the parameters.
struct Finder<'t> {
    hash_table: &'t mut [u32],
    /// The tree: two cells per position, cut to its `2 << btLog` cells.
    chain_table: &'t mut [u32],
    hash_table3: &'t mut [u32],
    /// `ms->nextToUpdate`, stored back after the block.
    next_to_update: u32,
    /// Index from which the 3-byte table is brought up to date.
    next_to_update3: u32,
    hash_log: u32,
    hash_log3: u32,
    /// `(1 << btLog) - 1`.
    bt_mask: u32,
    /// `1 << searchLog`, at least one.
    nb_compares: u32,
    /// `MIN(targetLength, ZSTD_OPT_NUM - 1)`.
    sufficient_len: u32,
    /// The window's reach `1 << windowLog`; with a loaded dictionary every
    /// index from `low_limit` on stays valid, which no distance exceeds.
    max_distance: u32,
    low_limit: u32,
}

impl<'t> Finder<'t> {
    /// The finder over `ms`'s tables for a block whose updates resume at
    /// `ms.next_to_update`, and the pricing state, which the parser keeps
    /// using beside it.
    fn split(ms: &'t mut MatchState) -> (Self, &'t mut OptState) {
        let cparams = ms.cparams;
        let bt_mask = (1u32 << (cparams.chain_log - 1)) - 1;
        let finder = Self {
            hash_table: &mut ms.hash_table,
            chain_table: &mut ms.chain_table[..2 * (bt_mask as usize + 1)],
            hash_table3: &mut ms.hash_table3,
            next_to_update: ms.next_to_update,
            next_to_update3: ms.next_to_update,
            hash_log: cparams.hash_log,
            hash_log3: ms.hash_log3,
            bt_mask,
            nb_compares: 1u32 << cparams.search_log,
            sufficient_len: cparams.target_length.min(OPT_NUM as u32 - 1),
            max_distance: if ms.loaded_dict_end != 0 {
                u32::MAX
            } else {
                1u32 << cparams.window_log
            },
            low_limit: ms.window.low_limit,
        };
        (finder, &mut ms.opt)
    }

    /// `ZSTD_getLowestMatchIndex`.
    #[inline(always)]
    const fn lowest_match_index(&self, curr: u32) -> u32 {
        if curr.wrapping_sub(self.low_limit) > self.max_distance {
            curr - self.max_distance
        } else {
            self.low_limit
        }
    }
}

/// `ZSTD_insertAndFindFirstIndexHash3`.
#[inline(always)]
fn insert_and_find_first_index_hash3<S: Simd>(
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    ip: u32,
) -> u32 {
    let hash_log3 = finder.hash_log3;
    let table = &mut *finder.hash_table3;
    let hash3 = |index: u32| mem::hash3(ctx.read32(index), hash_log3, 0);
    let mut idx = finder.next_to_update3;
    while idx < ip {
        table[hash3(idx)] = idx;
        idx += 1;
    }
    finder.next_to_update3 = ip;
    table[hash3(ip)]
}

/// `ZSTD_insertBt1`: returns how many positions it accounted for.
#[inline(always)]
fn insert_bt1<S: Simd, const EXT: bool, const MLS: u32>(
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    curr: u32,
    target: u32,
) -> u32 {
    let h = ctx.hash::<MLS>(curr, finder.hash_log);
    let match_index = finder.hash_table[h];
    finder.hash_table[h] = curr;
    ctx.prefetch_ahead::<MLS, false>(finder, curr);
    let walk = TreeWalk {
        curr,
        match_index,
        window_low: finder.lowest_match_index(target),
        nb_compares: finder.nb_compares,
        best_length: 8,
    };
    let (best_length, match_end_idx, _) =
        walk.run::<S, EXT>(ctx, finder.chain_table, finder.bt_mask, |_, _| false);
    let positions = if best_length > 384 {
        192.min((best_length - 384) as u32)
    } else {
        0
    };
    positions.max(match_end_idx - (curr + 8))
}

/// One insertion of `curr` into the binary tree: the walk shared by
/// `ZSTD_insertBt1` and `ZSTD_insertBtAndGetAllMatches`.
#[derive(Copy, Clone, Debug)]
struct TreeWalk {
    curr: u32,
    /// The first candidate, from the hash table.
    match_index: u32,
    /// Candidates below this index are out of the window.
    window_low: u32,
    nb_compares: u32,
    /// Length a candidate must exceed to be reported.
    best_length: usize,
}

impl TreeWalk {
    /// Walks the tree from `match_index`, re-linking it so that `curr` is
    /// the new root, and calls `on_longer(index, length)` for each candidate
    /// longer than every earlier one; it returns `true` to stop the walk.
    /// Returns the longest length, the farthest match end (`matchEndIdx`)
    /// and the comparisons left for an attached dictionary's tree: none
    /// after a stop, as the reference's dictMatchState walk sets them.
    ///
    /// `bt` holds the tree's `2 * (bt_mask + 1)` cells. The reference parks
    /// a finished side on a dummy cell; here a side whose walk ended below
    /// `bt_low` is simply not cleared at the end.
    #[inline(always)]
    fn run<S: Simd, const EXT: bool>(
        self,
        ctx: &Ctx<'_, S>,
        bt: &mut [u32],
        bt_mask: u32,
        mut on_longer: impl FnMut(u32, usize) -> bool,
    ) -> (usize, u32, u32) {
        let Self {
            curr,
            mut match_index,
            window_low,
            mut nb_compares,
            mut best_length,
        } = self;
        // The cells the next smaller / larger candidate is linked from.
        let mut smaller = 2 * (curr & bt_mask) as usize;
        let mut match_end_idx = curr + 8 + 1;
        // `nb_compares` is at least one, so only the window can rule out
        // the first candidate: then `curr` becomes a leaf straight away,
        // which incompressible input does at nearly every position.
        if match_index < window_low {
            bt[smaller..smaller + 2].fill(0);
            return (best_length, match_end_idx, nb_compares);
        }
        // Cut to the tree's size and indexed through `cells`: every cell
        // index is below that size already, so the mask changes nothing
        // but lets the compiler drop the bounds checks.
        let cells = (2 * bt_mask + 1) as usize;
        let bt = &mut bt[..=cells];
        let mut larger = smaller + 1;
        let bt_low = curr.saturating_sub(bt_mask);
        let current = ctx.current(curr);
        let mut common_smaller = 0usize;
        let mut common_larger = 0usize;
        let (clear_smaller, clear_larger) = loop {
            let (match_length, match_smaller) = ctx.extend::<EXT>(
                current,
                curr,
                match_index,
                common_smaller.min(common_larger),
            );
            if match_length > best_length {
                best_length = match_length;
                if match_length as u32 > match_end_idx - match_index {
                    match_end_idx = match_index + match_length as u32;
                }
                if on_longer(match_index, match_length) {
                    nb_compares = 0;
                    break (true, true);
                }
            }
            if match_length == current.len() {
                // The run reaches the end of the block: no way to order it.
                nb_compares = 0;
                break (true, true);
            }
            // `cells` is `2 * bt_mask + 1` and an even number has no low
            // bit: this is `2 * (match_index & bt_mask)`.
            let next = ((match_index as usize) << 1) & cells;
            if match_smaller {
                bt[smaller & cells] = match_index;
                common_smaller = match_length;
                if match_index <= bt_low {
                    break (false, true);
                }
                smaller = next + 1;
                match_index = bt[(next + 1) & cells];
            } else {
                bt[larger & cells] = match_index;
                common_larger = match_length;
                if match_index <= bt_low {
                    break (true, false);
                }
                larger = next;
                match_index = bt[next & cells];
            }
            nb_compares -= 1;
            if nb_compares == 0 || match_index < window_low {
                break (true, true);
            }
        };
        if clear_smaller {
            bt[smaller & cells] = 0;
        }
        if clear_larger {
            bt[larger & cells] = 0;
        }
        (best_length, match_end_idx, nb_compares)
    }
}

/// Where and what `get_all_matches` searches: the position, its repeat
/// offsets, whether the literal length is zero, and the length to beat.
#[derive(Copy, Clone, Debug)]
struct MatchQuery<'r> {
    ip: u32,
    rep: &'r [u32; REP_NUM],
    ll0: bool,
    length_to_beat: u32,
}

/// `ZSTD_insertBtAndGetAllMatches`; `DMS` searches the attached
/// dictionary's tree after the block's own (the reference's
/// `dictMode == ZSTD_dictMatchState`).
#[inline(always)]
fn insert_bt_and_get_all_matches<S: Simd, const EXT: bool, const DMS: bool, const MLS: u32>(
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    matches: &mut [Match],
    query: MatchQuery<'_>,
) -> u32 {
    let MatchQuery {
        ip,
        rep,
        ll0,
        length_to_beat,
    } = query;
    let sufficient_len = finder.sufficient_len;
    let curr = ip;
    let mls = MLS;
    let min_match = if mls == 3 { 3 } else { 4 };
    let h = ctx.hash::<MLS>(ip, finder.hash_log);
    let match_index = finder.hash_table[h];
    // Large tables miss the caches: a later search's cell loads while this
    // one runs.
    ctx.prefetch_ahead::<MLS, true>(finder, ip);
    let dict_limit = ctx.dict_limit;
    let window_low = finder.lowest_match_index(curr);
    let match_low = if window_low != 0 { window_low } else { 1 };
    let mut mnum = 0usize;
    let mut best_length = (length_to_beat - 1) as usize;
    let i_limit = ctx.iend;

    // Repeat offsets.
    {
        // Every repeat candidate in the prefix lies before `ip`: its word
        // lies in the bytes up to `ip`'s, so one length test covers all.
        let ip_off = curr - dict_limit;
        let head = &ctx.prefix[..ip_off as usize + 4];
        let word_at = |at: u32| {
            let mut word = [0u8; 4];
            word.copy_from_slice(&head[at as usize..at as usize + 4]);
            u32::from_le_bytes(word)
        };
        let ip_word = word_at(ip_off);
        let ll0 = u32::from(ll0);
        let last_r = REP_NUM as u32 + ll0;
        for rep_code in ll0..last_r {
            let rep_offset = if rep_code == REP_NUM as u32 {
                rep[0].wrapping_sub(1)
            } else {
                rep[rep_code as usize]
            };
            let rep_index = curr.wrapping_sub(rep_offset);
            let mut rep_len = 0u32;
            // `rep_offset - 1 < curr - dict_limit`, spelled so that the
            // candidate's offset is seen to stay at or before `ip_off`.
            if rep_offset != 0 && rep_offset <= ip_off {
                // The candidate lies in the prefix, so its word reads before
                // the window test, as the reference's `&` lets it: most
                // positions settle on the word alone.
                if min_match_equal(ip_word, word_at(ip_off - rep_offset), min_match)
                    && rep_index >= window_low
                {
                    rep_len =
                        ctx.count(ip + min_match, ip + min_match - rep_offset) as u32 + min_match;
                }
            } else if EXT
                && rep_offset.wrapping_sub(1) < curr - window_low
                && dict_limit.wrapping_sub(1).wrapping_sub(rep_index) >= 3
                && min_match_equal(ip_word, ctx.window.ext.read32(rep_index), min_match)
            {
                rep_len =
                    ctx.count_2segments(ip + min_match, rep_index + min_match) as u32 + min_match;
            } else if DMS {
                rep_len =
                    dictionary_repeat_length(ctx, ip, ip_word, rep_offset, window_low, min_match);
            }
            if rep_len as usize > best_length {
                best_length = rep_len as usize;
                matches[mnum] = Match {
                    off: repcode_to_offbase(rep_code - ll0 + 1),
                    len: rep_len,
                };
                mnum += 1;
                if rep_len > sufficient_len || ip + rep_len == i_limit {
                    return mnum as u32;
                }
            }
        }
    }

    // HC3 match finder.
    if mls == 3 && best_length < mls as usize {
        let match_index3 = insert_and_find_first_index_hash3(finder, ctx, ip);
        if match_index3 >= match_low && curr - match_index3 < (1 << 18) {
            let mlen = if !EXT || match_index3 >= dict_limit {
                ctx.count(ip, match_index3)
            } else {
                ctx.count_2segments(ip, match_index3)
            };
            if mlen >= mls as usize {
                best_length = mlen;
                matches[0] = Match {
                    off: offset_to_offbase(curr - match_index3),
                    len: mlen as u32,
                };
                mnum = 1;
                if mlen > sufficient_len as usize || ip as usize + mlen == i_limit as usize {
                    finder.next_to_update = curr + 1;
                    return 1;
                }
            }
        }
    }

    finder.hash_table[h] = curr;
    let walk = TreeWalk {
        curr,
        match_index,
        window_low: match_low,
        nb_compares: finder.nb_compares,
        best_length,
    };
    let (best_length, mut match_end_idx, nb_compares) = walk.run::<S, EXT>(
        ctx,
        finder.chain_table,
        finder.bt_mask,
        |match_index, match_length| {
            matches[mnum] = Match {
                off: offset_to_offbase(curr - match_index),
                len: match_length as u32,
            };
            mnum += 1;
            // A run reaching the block's end stops the walk after this.
            match_length > OPT_NUM
        },
    );
    if DMS && nb_compares != 0 {
        let found = DictionaryFound {
            best_length,
            match_end_idx,
            mnum,
        };
        let found = search_dictionary::<S, MLS>(ctx, matches, curr, window_low, nb_compares, found);
        match_end_idx = found.match_end_idx;
        mnum = found.mnum;
    }
    finder.next_to_update = match_end_idx - 8;
    mnum as u32
}

/// The dictMatchState repeat-offset check of
/// `ZSTD_insertBtAndGetAllMatches` for a repeat offset that reaches before
/// the prefix: the length of the match `rep_offset` back from `ip`, read in
/// the attached dictionary and continued at the prefix start, or `0`.
///
/// `ip_word` holds the four bytes at `ip`; `window_low` is the window's
/// lowest index at `ip`, which places the dictionary's end just below it
/// (`dmsIndexDelta`).
#[inline(always)]
fn dictionary_repeat_length<S: Simd>(
    ctx: &Ctx<'_, S>,
    ip: u32,
    ip_word: u32,
    rep_offset: u32,
    window_low: u32,
    min_match: u32,
) -> u32 {
    let dms = &ctx.dms;
    let curr = ip;
    let rep_index = curr.wrapping_sub(rep_offset);
    let index_delta = window_low.wrapping_sub(dms.high_limit);
    // `curr > repIndex >= dmsLowLimit` in the dictionary's indices, and the
    // word does not straddle the prefix start (`ZSTD_index_overlap_check`).
    if rep_offset.wrapping_sub(1) < curr.wrapping_sub(dms.low_limit.wrapping_add(index_delta))
        && ctx.dict_limit.wrapping_sub(1).wrapping_sub(rep_index) >= 3
    {
        let dict_index = rep_index.wrapping_sub(index_delta);
        if min_match_equal(ip_word, dms.read32(dict_index), min_match) {
            return ctx.count_dictionary(ip + min_match, dict_index + min_match) as u32 + min_match;
        }
    }
    0
}

/// What a search found before the attached dictionary's tree: the length
/// to beat, the farthest match end and the number of matches stored.
#[derive(Copy, Clone, Debug)]
struct DictionaryFound {
    best_length: usize,
    match_end_idx: u32,
    mnum: usize,
}

/// The dictMatchState tail of `ZSTD_insertBtAndGetAllMatches`: walks the
/// attached dictionary's tree with the `nb_compares` comparisons the
/// block's own tree left, storing each candidate longer than every
/// earlier match. The dictionary's tree is only read.
///
/// A dictionary index `i` is window index `i + dmsIndexDelta`, so the
/// dictionary ends just below `window_low`.
#[inline(always)]
fn search_dictionary<S: Simd, const MLS: u32>(
    ctx: &Ctx<'_, S>,
    matches: &mut [Match],
    curr: u32,
    window_low: u32,
    mut nb_compares: u32,
    found: DictionaryFound,
) -> DictionaryFound {
    let DictionaryFound {
        mut best_length,
        mut match_end_idx,
        mut mnum,
    } = found;
    let dms = &ctx.dms;
    let index_delta = window_low.wrapping_sub(dms.high_limit);
    let cells = (2 * dms.bt_mask + 1) as usize;
    let bt = &dms.bt[..=cells];
    let current = ctx.current(curr);
    let mut dict_match_index = dms.hash_table[ctx.hash::<MLS>(curr, dms.hash_log)];
    let mut common_smaller = 0usize;
    let mut common_larger = 0usize;
    while nb_compares != 0 && dict_match_index > dms.low_limit {
        let next = ((dict_match_index as usize) << 1) & cells;
        let known = common_smaller.min(common_larger);
        let match_length =
            known + ctx.count_dictionary(curr + known as u32, dict_match_index + known as u32);
        if match_length > best_length {
            let match_index = dict_match_index.wrapping_add(index_delta);
            if match_length as u32 > match_end_idx.wrapping_sub(match_index) {
                match_end_idx = match_index.wrapping_add(match_length as u32);
            }
            best_length = match_length;
            matches[mnum] = Match {
                off: offset_to_offbase(curr.wrapping_sub(match_index)),
                len: match_length as u32,
            };
            mnum += 1;
            if match_length > OPT_NUM || match_length == current.len() {
                break;
            }
        }
        if dict_match_index <= dms.bt_low {
            break;
        }
        // The candidate's next byte: in the dictionary, or past its end in
        // the prefix, at the window index the reference moves its `match`
        // pointer to.
        let at = dict_match_index.wrapping_add(match_length as u32);
        let candidate_byte = if at < dms.high_limit {
            dms.bytes[at.wrapping_sub(dms.start) as usize]
        } else {
            ctx.prefix[at.wrapping_add(index_delta).wrapping_sub(ctx.dict_limit) as usize]
        };
        let Some(&current_byte) = current.get(match_length) else {
            // A run to the end of the block is longer than any before it,
            // which ended the walk above.
            break;
        };
        if candidate_byte < current_byte {
            common_smaller = match_length;
            dict_match_index = bt[(next + 1) & cells];
        } else {
            common_larger = match_length;
            dict_match_index = bt[next & cells];
        }
        nb_compares -= 1;
    }
    DictionaryFound {
        best_length,
        match_end_idx,
        mnum,
    }
}

/// `ZSTD_btGetAllMatches_internal`: brings the tree up to `query.ip`, then
/// collects the matches there.
///
/// The wrapper inlines into the parser, which calls the level's
/// feature-enabled entry directly; the tree update and the search inline
/// into that entry.
#[simd]
#[inline(always)]
fn get_all_matches<S: Simd, const EXT: bool, const DMS: bool, const MLS: u32>(
    simd: S,
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    matches: &mut [Match],
    query: MatchQuery<'_>,
) -> u32 {
    let _ = simd;
    if query.ip < finder.next_to_update {
        return 0;
    }
    if finder.next_to_update < query.ip {
        update_tree::<S, EXT, MLS>(finder, ctx, query.ip);
    }
    finder.next_to_update = query.ip;
    insert_bt_and_get_all_matches::<S, EXT, DMS, MLS>(finder, ctx, matches, query)
}

/// `ZSTD_updateTree_internal`: inserts the positions a long match skipped.
///
/// Out of line, so that its walk does not crowd the search's registers;
/// it is plain scalar code, and the scan of a long run calls into the
/// backend's feature context as the search's does.
#[inline(never)]
fn update_tree<S: Simd, const EXT: bool, const MLS: u32>(
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    target: u32,
) {
    let mut idx = finder.next_to_update;
    while idx < target {
        idx += insert_bt1::<S, EXT, MLS>(finder, ctx, idx, target);
    }
}

/// [`get_all_matches`] for the block's minimum match length, the
/// reference's `ZSTD_selectBtGetAllMatches`; the branch resolves the same
/// way for the whole block.
#[inline(always)]
fn find_matches<S: Simd, const EXT: bool, const DMS: bool>(
    simd: S,
    finder: &mut Finder<'_>,
    ctx: &Ctx<'_, S>,
    matches: &mut [Match],
    query: MatchQuery<'_>,
) -> u32 {
    match ctx.mls {
        3 => get_all_matches::<S, EXT, DMS, 3>(simd, finder, ctx, matches, query),
        4 => get_all_matches::<S, EXT, DMS, 4>(simd, finder, ctx, matches, query),
        5 => get_all_matches::<S, EXT, DMS, 5>(simd, finder, ctx, matches, query),
        _ => get_all_matches::<S, EXT, DMS, 6>(simd, finder, ctx, matches, query),
    }
}

// ---------------------------------------------------------------------------
// LDM candidates.

/// LDM candidate tracking (`ZSTD_optLdm_t`).
#[derive(Debug)]
struct OptLdm {
    seq_store: RawSeqStore,
    start_pos_in_block: u32,
    end_pos_in_block: u32,
    offset: u32,
}

impl OptLdm {
    /// `ZSTD_opt_getNextMatchAndUpdateSeqStore`.
    fn next_match(&mut self, curr_pos_in_block: u32, block_bytes_remaining: u32) {
        if self.seq_store.size() == 0 || self.seq_store.pos >= self.seq_store.size() {
            self.start_pos_in_block = u32::MAX;
            self.end_pos_in_block = u32::MAX;
            return;
        }
        let curr = self.seq_store.seq[self.seq_store.pos];
        let curr_block_end_pos = curr_pos_in_block + block_bytes_remaining;
        let pos_in_seq = self.seq_store.pos_in_sequence as u32;
        let literals_remaining = curr.lit_length.saturating_sub(pos_in_seq);
        let match_remaining = if literals_remaining == 0 {
            curr.match_length - (pos_in_seq - curr.lit_length)
        } else {
            curr.match_length
        };
        if literals_remaining >= block_bytes_remaining {
            self.start_pos_in_block = u32::MAX;
            self.end_pos_in_block = u32::MAX;
            self.seq_store
                .skip_bytes_opt(block_bytes_remaining as usize);
            return;
        }
        self.start_pos_in_block = curr_pos_in_block + literals_remaining;
        self.end_pos_in_block = self.start_pos_in_block + match_remaining;
        self.offset = curr.offset;
        if self.end_pos_in_block > curr_block_end_pos {
            self.end_pos_in_block = curr_block_end_pos;
            self.seq_store
                .skip_bytes_opt((curr_block_end_pos - curr_pos_in_block) as usize);
        } else {
            self.seq_store
                .skip_bytes_opt((literals_remaining + match_remaining) as usize);
        }
    }

    /// `ZSTD_optLdm_maybeAddMatch`.
    fn maybe_add_match(
        &self,
        matches: &mut [Match],
        nb_matches: &mut u32,
        curr_pos_in_block: u32,
        min_match: u32,
    ) {
        let pos_diff = curr_pos_in_block.wrapping_sub(self.start_pos_in_block);
        let candidate_len = self
            .end_pos_in_block
            .wrapping_sub(self.start_pos_in_block)
            .wrapping_sub(pos_diff);
        if curr_pos_in_block < self.start_pos_in_block
            || curr_pos_in_block >= self.end_pos_in_block
            || candidate_len < min_match
        {
            return;
        }
        if *nb_matches == 0
            || (candidate_len > matches[*nb_matches as usize - 1].len
                && (*nb_matches as usize) < OPT_NUM)
        {
            matches[*nb_matches as usize] = Match {
                off: offset_to_offbase(self.offset),
                len: candidate_len,
            };
            *nb_matches += 1;
        }
    }

    /// `ZSTD_optLdm_processMatchCandidate`.
    ///
    /// Without LDM candidates this is one inline test per position; the
    /// candidate bookkeeping stays out of line.
    #[inline(always)]
    fn process(
        &mut self,
        matches: &mut [Match],
        nb_matches: &mut u32,
        curr_pos_in_block: u32,
        remaining: u32,
        min_match: u32,
    ) {
        if self.seq_store.pos >= self.seq_store.size() {
            return;
        }
        self.process_candidate(matches, nb_matches, curr_pos_in_block, remaining, min_match);
    }

    /// [`Self::process`] with candidates left.
    #[inline(never)]
    fn process_candidate(
        &mut self,
        matches: &mut [Match],
        nb_matches: &mut u32,
        curr_pos_in_block: u32,
        remaining: u32,
        min_match: u32,
    ) {
        if curr_pos_in_block >= self.end_pos_in_block {
            if curr_pos_in_block > self.end_pos_in_block {
                let overshoot = curr_pos_in_block - self.end_pos_in_block;
                self.seq_store.skip_bytes_opt(overshoot as usize);
            }
            self.next_match(curr_pos_in_block, remaining);
        }
        self.maybe_add_match(matches, nb_matches, curr_pos_in_block, min_match);
    }
}

// ---------------------------------------------------------------------------
// The parser.

/// `ZSTD_compressBlock_opt_generic`, in `simd`'s feature context.
///
/// `OPT_LEVEL` (0 for btopt, 2 for the ultra strategies), `EXT` (an
/// external dictionary is in the window) and `DMS` (the attached
/// dictionary `block.dms` is searched too) are the reference's template
/// parameters; all are fixed per block, and at most one of `EXT` and
/// `DMS` is set.
#[simd]
fn opt_generic<S: Simd, const OPT_LEVEL: u32, const EXT: bool, const DMS: bool>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; REP_NUM],
    block: &BlockInput<'_>,
) -> usize {
    let opt_level = OPT_LEVEL;
    let prefix = block.window.prefix;
    let istart = block.start;
    let iend = block.end;
    let ilimit = i64::from(iend) - 8;
    let prefix_start = ms.window.dict_limit;
    let cparams = ms.cparams;
    let sufficient_len = cparams.target_length.min(OPT_NUM as u32 - 1);
    let min_match = if cparams.min_match == 3 { 3 } else { 4 };
    let mls = cparams.min_match.clamp(3, 6);
    let dictionary = match block.dms {
        Some(dms) if DMS => Some(DictSearch::new(&dms)),
        _ => None,
    };
    let ctx = match &dictionary {
        Some(dictionary) => Ctx::with_dictionary(simd, block, mls, dictionary),
        None => Ctx::new(simd, block, mls),
    };

    ms.opt
        .reserve_matches((iend - istart) as usize, 1u32 << cparams.search_log);
    let mut opt_table = core::mem::take(&mut ms.opt.price_table);
    let mut opt: &mut [Optimal] = &mut opt_table;
    // Every match the parser prices ends inside the block, and below
    // `OPT_NUM` positions from where the parse started.
    let reach = OPT_SIZE.min((iend - istart) as usize + 3);
    let mut matches = core::mem::take(&mut ms.opt.match_table);
    let mut last_stretch = Optimal::default();
    // The reference copies the store's header; here the store is moved out
    // for the block and returned with its cursor as it was.
    let lent = ms.ldm_seq_store.take();
    let lent_cursor = lent
        .as_ref()
        .map(|store| (store.pos, store.pos_in_sequence));
    let mut opt_ldm = OptLdm {
        seq_store: lent.unwrap_or_default(),
        start_pos_in_block: 0,
        end_pos_in_block: 0,
        offset: 0,
    };
    let mut ip = istart;
    let mut anchor = istart;
    opt_ldm.next_match(ip - istart, iend - ip);

    ms.opt
        .rescale_freqs(prefix.slice(istart, iend), opt_level, block.symbol_costs);
    ip += u32::from(ip == prefix_start);
    let (mut finder, stats) = Finder::split(ms);

    while i64::from(ip) < ilimit {
        let mut cur: u32;
        let mut last_pos: u32;

        // Find the first match.
        let litlen = ip - anchor;
        let ll0 = litlen == 0;
        let mut nb_matches = find_matches::<S, EXT, DMS>(
            simd,
            &mut finder,
            &ctx,
            &mut matches,
            MatchQuery {
                ip,
                rep,
                ll0,
                length_to_beat: min_match,
            },
        );
        opt_ldm.process(
            &mut matches,
            &mut nb_matches,
            ip - istart,
            iend - ip,
            min_match,
        );
        if nb_matches == 0 {
            ip += 1;
            continue;
        }

        if opt.len() < reach {
            opt = grow_positions(&mut opt_table, reach);
        }
        let max_ml = matches[nb_matches as usize - 1].len;
        let max_off_base = matches[nb_matches as usize - 1].off;

        let os = &*stats;
        opt[0].mlen = 0;
        opt[0].litlen = litlen;
        opt[0].price = os.lit_length_price(litlen);
        opt[0].rep = *rep;
        let shortest_path_ready;
        if max_ml > sufficient_len {
            last_stretch.litlen = 0;
            last_stretch.mlen = max_ml;
            last_stretch.off = max_off_base;
            cur = 0;
            last_pos = max_ml;
            shortest_path_ready = true;
        } else {
            let mut pos = 1u32;
            while pos < min_match {
                opt[pos as usize].price = MAX_PRICE;
                opt[pos as usize].mlen = 0;
                opt[pos as usize].litlen = litlen + pos;
                pos += 1;
            }
            let ll0_price = os.lit_length_price(0);
            for m in &matches[..nb_matches as usize] {
                let off_base = m.off;
                let end = m.len;
                let off_price = os.offset_price(off_base);
                while pos <= end {
                    let match_price = off_price.wrapping_add(os.match_length_price(pos)) as i32;
                    let sequence_price = opt[0].price + match_price;
                    let slot = &mut opt[pos as usize];
                    slot.mlen = pos;
                    slot.off = off_base;
                    slot.litlen = 0;
                    slot.price = sequence_price + ll0_price;
                    pos += 1;
                }
            }
            last_pos = pos - 1;
            opt[pos as usize].price = MAX_PRICE;
            shortest_path_ready = false;
            cur = 0;
        }

        if !shortest_path_ready {
            let mut jumped = false;
            cur = 1;
            while cur <= last_pos {
                let inr = ip + cur;
                {
                    let os = &*stats;
                    let ll_price = |l: u32| os.lit_length_price(l);
                    let ll_inc_price = |l: u32| ll_price(l) - ll_price(l - 1);
                    let litlen = opt[cur as usize - 1].litlen + 1;
                    let price = opt[cur as usize - 1].price
                        + os.literal_price(prefix.byte(ip + cur - 1))
                        + ll_inc_price(litlen);
                    if price <= opt[cur as usize].price {
                        let prev_match = opt[cur as usize];
                        opt[cur as usize] = opt[cur as usize - 1];
                        opt[cur as usize].litlen = litlen;
                        opt[cur as usize].price = price;
                        if opt_level >= 1
                            && prev_match.litlen == 0
                            && ll_inc_price(1) < 0
                            && ip + cur < iend
                        {
                            let next_lit = os.literal_price(prefix.byte(ip + cur));
                            let with1literal = prev_match.price + next_lit + ll_inc_price(1);
                            let with_more_literals = price + next_lit + ll_inc_price(litlen + 1);
                            if with1literal < with_more_literals
                                && with1literal < opt[cur as usize + 1].price
                            {
                                let prev = cur - prev_match.mlen;
                                let new_reps = new_rep(
                                    &opt[prev as usize].rep,
                                    prev_match.off,
                                    opt[prev as usize].litlen == 0,
                                );
                                opt[cur as usize + 1] = prev_match;
                                opt[cur as usize + 1].rep = new_reps;
                                opt[cur as usize + 1].litlen = 1;
                                opt[cur as usize + 1].price = with1literal;
                                if last_pos < cur + 1 {
                                    last_pos = cur + 1;
                                }
                            }
                        }
                    }
                }

                if opt[cur as usize].litlen == 0 {
                    let prev = cur - opt[cur as usize].mlen;
                    let new_reps = new_rep(
                        &opt[prev as usize].rep,
                        opt[cur as usize].off,
                        opt[prev as usize].litlen == 0,
                    );
                    opt[cur as usize].rep = new_reps;
                }

                if i64::from(inr) > ilimit {
                    cur += 1;
                    continue;
                }
                if cur == last_pos {
                    break;
                }
                if opt_level == 0
                    && opt[cur as usize + 1].price
                        <= opt[cur as usize].price + (BITCOST_MULTIPLIER / 2) as i32
                {
                    cur += 1;
                    continue;
                }

                let ll0 = opt[cur as usize].litlen == 0;
                let previous_price = opt[cur as usize].price;
                let base_price = previous_price + stats.lit_length_price(0);
                let cur_rep = opt[cur as usize].rep;
                let mut nb_matches = find_matches::<S, EXT, DMS>(
                    simd,
                    &mut finder,
                    &ctx,
                    &mut matches,
                    MatchQuery {
                        ip: inr,
                        rep: &cur_rep,
                        ll0,
                        length_to_beat: min_match,
                    },
                );
                opt_ldm.process(
                    &mut matches,
                    &mut nb_matches,
                    inr - istart,
                    iend - inr,
                    min_match,
                );
                if nb_matches == 0 {
                    cur += 1;
                    continue;
                }
                let longest_ml = matches[nb_matches as usize - 1].len;
                if longest_ml > sufficient_len
                    || cur + longest_ml >= OPT_NUM as u32
                    || ip + cur + longest_ml >= iend
                {
                    last_stretch.mlen = longest_ml;
                    last_stretch.off = matches[nb_matches as usize - 1].off;
                    last_stretch.litlen = 0;
                    last_pos = cur + longest_ml;
                    jumped = true;
                    break;
                }

                let os = &*stats;
                for match_nb in 0..nb_matches as usize {
                    let offset = matches[match_nb].off;
                    let last_ml = matches[match_nb].len;
                    let start_ml = if match_nb > 0 {
                        matches[match_nb - 1].len + 1
                    } else {
                        min_match
                    };
                    let off_price = os.offset_price(offset);
                    let mut mlen = last_ml;
                    while mlen >= start_ml {
                        let pos = cur + mlen;
                        let price =
                            base_price + off_price.wrapping_add(os.match_length_price(mlen)) as i32;
                        if pos > last_pos || price < opt[pos as usize].price {
                            while last_pos < pos {
                                last_pos += 1;
                                opt[last_pos as usize].price = MAX_PRICE;
                                opt[last_pos as usize].litlen = 1;
                            }
                            let slot = &mut opt[pos as usize];
                            slot.mlen = mlen;
                            slot.off = offset;
                            slot.litlen = 0;
                            slot.price = price;
                        } else if opt_level == 0 {
                            break;
                        }
                        mlen -= 1;
                    }
                }
                opt[last_pos as usize + 1].price = MAX_PRICE;
                cur += 1;
            }

            if !jumped {
                last_stretch = opt[last_pos as usize];
                cur = last_pos - last_stretch.mlen;
            }
        }

        // _shortestPath:
        if last_stretch.mlen == 0 {
            ip += last_pos;
            continue;
        }
        if last_stretch.litlen == 0 {
            *rep = new_rep(
                &opt[cur as usize].rep,
                last_stretch.off,
                opt[cur as usize].litlen == 0,
            );
        } else {
            *rep = last_stretch.rep;
            cur -= last_stretch.litlen;
        }

        let store_end = cur + 2;
        let mut store_start;
        let mut stretch_pos = cur;
        if last_stretch.litlen > 0 {
            opt[store_end as usize].litlen = last_stretch.litlen;
            opt[store_end as usize].mlen = 0;
            store_start = store_end - 1;
            opt[store_start as usize] = last_stretch;
        }
        opt[store_end as usize] = last_stretch;
        store_start = store_end;
        loop {
            let next_stretch = opt[stretch_pos as usize];
            opt[store_start as usize].litlen = next_stretch.litlen;
            if next_stretch.mlen == 0 {
                break;
            }
            store_start -= 1;
            opt[store_start as usize] = next_stretch;
            stretch_pos -= next_stretch.litlen + next_stretch.mlen;
        }

        for store_pos in store_start..=store_end {
            let llen = opt[store_pos as usize].litlen;
            let mlen = opt[store_pos as usize].mlen;
            let off_base = opt[store_pos as usize].off;
            let advance = llen + mlen;
            if mlen == 0 {
                ip = anchor + llen;
                continue;
            }
            let literals = prefix.slice(anchor, anchor + llen);
            stats.update_stats(literals, off_base, mlen);
            seq_store.store_seq(literals, off_base, mlen as usize);
            anchor += advance;
            ip = anchor;
        }
        stats.set_base_prices(opt_level);
    }

    ms.next_to_update = finder.next_to_update;
    ms.opt.price_table = opt_table;
    ms.opt.match_table = matches;
    if let Some((pos, pos_in_sequence)) = lent_cursor {
        let mut store = opt_ldm.seq_store;
        store.pos = pos;
        store.pos_in_sequence = pos_in_sequence;
        ms.ldm_seq_store = Some(store);
    }
    (iend - anchor) as usize
}

/// `ZSTD_compressBlock_btopt`.
pub(crate) fn btopt(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    dispatch!(block.backend.0, simd => opt_generic::<_, 0, false, false>(simd, ms, seq_store, rep, block))
}

/// `ZSTD_compressBlock_btopt_extDict`.
pub(crate) fn btopt_ext(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    dispatch!(block.backend.0, simd => opt_generic::<_, 0, true, false>(simd, ms, seq_store, rep, block))
}

/// `ZSTD_compressBlock_btultra`.
pub(crate) fn btultra(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    dispatch!(block.backend.0, simd => opt_generic::<_, 2, false, false>(simd, ms, seq_store, rep, block))
}

/// `ZSTD_compressBlock_btultra_extDict` (also btultra2 over an external
/// dictionary).
pub(crate) fn btultra_ext(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    dispatch!(block.backend.0, simd => opt_generic::<_, 2, true, false>(simd, ms, seq_store, rep, block))
}

/// `ZSTD_compressBlock_btultra2`: a first pass over the frame's first block
/// seeds the statistics, then the block is compressed for real.
pub(crate) fn btultra2(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let curr = block.start;
    let src_size = block.end - block.start;
    // The reference's "no LDM" test only checks that no sequence was stored
    // yet; long matches offered as candidates do not disable the first pass.
    if ms.opt.lit_length_sum == 0
        && seq_store.nb_seq() == 0
        && ms.window.dict_limit == ms.window.low_limit
        && curr == ms.window.dict_limit
        && src_size as usize > PREDEF_THRESHOLD
    {
        // ZSTD_initStats_ultra.
        let mut tmp_rep = *rep;
        btultra(ms, seq_store, &mut tmp_rep, block);
        seq_store.reset();
        ms.window.base = ms.window.base.add(-i64::from(src_size));
        ms.window.dict_limit += src_size;
        ms.window.low_limit = ms.window.dict_limit;
        ms.next_to_update = ms.window.dict_limit;
        let shifted = BlockInput {
            window: mem::Window {
                prefix: block
                    .window
                    .prefix
                    .rebased(block.window.prefix.start() + src_size),
                ext: mem::Mem::new(&[], ms.window.dict_limit),
            },
            start: block.start + src_size,
            end: block.end + src_size,
            backend: block.backend,
            dms: block.dms,
            symbol_costs: block.symbol_costs,
        };
        return btultra(ms, seq_store, rep, &shifted);
    }
    btultra(ms, seq_store, rep, block)
}

/// Indexes dictionary content up to `target` into the binary tree
/// (`ZSTD_updateTree(ms, iend - HASH_READ_SIZE, iend)`).
///
/// `ZSTD_updateTree` hashes with the unclamped minimum match, so a minimum
/// match of 7 indexes 7-byte hashes, unlike the searches (at most 6).
pub(crate) fn load_dictionary(
    ms: &mut MatchState,
    window: &mem::Window<'_>,
    target: u32,
    end: u32,
) {
    let block = BlockInput {
        window: *window,
        start: target,
        end,
        backend: ms.backend,
        dms: None,
        symbol_costs: None,
    };
    let mls = ms.cparams.min_match.clamp(3, 7);
    dispatch!(block.backend.0, simd => {
        let ctx = Ctx::new(simd, &block, mls);
        let (mut finder, _) = Finder::split(ms);
        match mls {
            3 => update_tree::<_, false, 3>(&mut finder, &ctx, target),
            4 => update_tree::<_, false, 4>(&mut finder, &ctx, target),
            5 => update_tree::<_, false, 5>(&mut finder, &ctx, target),
            6 => update_tree::<_, false, 6>(&mut finder, &ctx, target),
            _ => update_tree::<_, false, 7>(&mut finder, &ctx, target),
        }
    });
    ms.next_to_update = target;
}

/// `ZSTD_compressBlock_btopt_dictMatchState`.
///
/// Without an attached dictionary in `block` this is [`btopt`], the
/// variant `ZSTD_matchState_dictMode` selects then.
pub(crate) fn btopt_dms(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    if block.dms.is_none() {
        return btopt(ms, seq_store, rep, block);
    }
    dispatch!(block.backend.0, simd => opt_generic::<_, 0, false, true>(simd, ms, seq_store, rep, block))
}

/// `ZSTD_compressBlock_btultra_dictMatchState` (also btultra2's, which has
/// no variant of its own for an attached dictionary).
///
/// Without an attached dictionary in `block` this is [`btultra`].
pub(crate) fn btultra_dms(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    if block.dms.is_none() {
        return btultra(ms, seq_store, rep, block);
    }
    dispatch!(block.backend.0, simd => opt_generic::<_, 2, false, true>(simd, ms, seq_store, rep, block))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    use crate::shared::format::BLOCK_SIZE_MAX;
    use alloc::vec;

    /// The reference's `ZSTD_litLengthPrice`, from the frequencies.
    fn reference_lit_length_price(os: &OptState, lit_length: u32, opt_level: u32) -> i32 {
        if os.price_type == PriceType::Predef {
            return weight(lit_length, opt_level) as i32;
        }
        if lit_length as usize == BLOCK_SIZE_MAX {
            return BITCOST_MULTIPLIER as i32
                + reference_lit_length_price(os, lit_length - 1, opt_level);
        }
        let code = usize::from(ll_code(lit_length));
        (u32::from(LL_BITS[code]) * BITCOST_MULTIPLIER + os.lit_length_sum_base_price)
            .wrapping_sub(weight(os.lit_length_freq[code], opt_level)) as i32
    }

    /// The reference's `ZSTD_getMatchPrice`, from the frequencies.
    fn reference_match_price(os: &OptState, off_base: u32, length: u32, opt_level: u32) -> u32 {
        let off_code = highbit32(off_base);
        let ml_base = length - MIN_MATCH as u32;
        if os.price_type == PriceType::Predef {
            return weight(ml_base, opt_level) + (16 + off_code) * BITCOST_MULTIPLIER;
        }
        let mut price = off_code * BITCOST_MULTIPLIER
            + os.off_code_sum_base_price
                .wrapping_sub(weight(os.off_code_freq[off_code as usize], opt_level));
        if opt_level < 2 && off_code >= 20 {
            price += (off_code - 19) * 2 * BITCOST_MULTIPLIER;
        }
        let code = usize::from(ml_code(ml_base));
        price = price.wrapping_add(
            u32::from(ML_BITS[code]) * BITCOST_MULTIPLIER
                + os.match_length_sum_base_price
                    .wrapping_sub(weight(os.match_length_freq[code], opt_level)),
        );
        price + BITCOST_MULTIPLIER / 5
    }

    /// The reference's `ZSTD_rawLiteralsCost` for one literal.
    fn reference_literal_price(os: &OptState, literal: u8, opt_level: u32) -> i32 {
        if !os.compressed_literals() {
            return (8 * BITCOST_MULTIPLIER) as i32;
        }
        if os.price_type == PriceType::Predef {
            return (6 * BITCOST_MULTIPLIER) as i32;
        }
        let max = os.lit_sum_base_price - BITCOST_MULTIPLIER;
        let price = weight(os.lit_freq[usize::from(literal)], opt_level).min(max);
        os.lit_sum_base_price.wrapping_sub(price) as i32
    }

    fn assert_prices_match(os: &OptState, opt_level: u32) {
        for literal in 0..=u8::MAX {
            assert_eq!(
                os.literal_price(literal),
                reference_literal_price(os, literal, opt_level),
                "literal {literal}"
            );
        }
        let lengths = (0..300u32).chain([1000, 65_535, 65_536, 131_071, BLOCK_SIZE_MAX as u32]);
        for lit_length in lengths {
            assert_eq!(
                os.lit_length_price(lit_length),
                reference_lit_length_price(os, lit_length, opt_level),
                "literal length {lit_length}"
            );
        }
        let offsets = (1..70u32).chain((4..32).map(|bit| (1u32 << bit) + 5));
        for off_base in offsets {
            for length in (3..200u32).chain([1000, 65_538, 131_074]) {
                assert_eq!(
                    os.match_price(off_base, length),
                    reference_match_price(os, off_base, length, opt_level),
                    "offset {off_base} length {length}"
                );
            }
        }
    }

    fn sample(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 24) as u8 & 0x3f
            })
            .collect()
    }

    #[test]
    fn price_tables_match_the_reference_formulas() {
        for opt_level in [0, 2] {
            for literal_compression in [ParamSwitch::Auto, ParamSwitch::Disable] {
                let mut os = OptState {
                    literal_compression,
                    ..OptState::default()
                };
                os.allocate();
                let src = sample(4096, opt_level + 1);
                os.rescale_freqs(&src, opt_level, None);
                assert_eq!(os.price_type, PriceType::Dynamic);
                assert_prices_match(&os, opt_level);
                // Statistics updated sequence by sequence, then rescaled.
                for (round, chunk) in src.chunks(37).enumerate() {
                    let round = round as u32;
                    let off_base = if round.is_multiple_of(3) {
                        2
                    } else {
                        4 + (round << (round % 23))
                    };
                    let literals = &chunk[..(round as usize % 37).min(chunk.len())];
                    os.update_stats(literals, off_base, 3 + round * 7);
                    os.set_base_prices(opt_level);
                }
                assert_prices_match(&os, opt_level);
                // Runs long enough to take every weight at once, around
                // the threshold, then short ones again.
                for (round, len) in [64usize, 63, 200, 1000, 65, 3].into_iter().enumerate() {
                    let literals = &src[round * 7..round * 7 + len];
                    os.update_stats(literals, 4 + round as u32, 3 + len as u32);
                    os.set_base_prices(opt_level);
                    assert_prices_match(&os, opt_level);
                }
                os.rescale_freqs(&src, opt_level, None);
                assert_prices_match(&os, opt_level);
            }
        }
    }

    #[test]
    fn finder_bounds_match_the_match_state() {
        use super::super::params::{ParamSwitch as Switch, level_cparams};
        for (level, loaded_dict_end) in [(16, 0), (19, 0), (22, 0), (19, 7)] {
            let mut ms = MatchState::new(level_cparams(level, 1 << 20), Switch::Disable);
            ms.window.low_limit = 5;
            ms.window.dict_limit = 9;
            ms.next_to_update = 40;
            ms.loaded_dict_end = loaded_dict_end;
            let cparams = ms.cparams;
            let currs = [
                5u32,
                9,
                100,
                1 << 20,
                (1 << 20) + 6,
                (1 << 21) + 3,
                u32::MAX - 9,
            ];
            let expected = currs.map(|curr| ms.lowest_match_index(curr, cparams.window_log));
            let (finder, _) = Finder::split(&mut ms);
            assert_eq!(currs.map(|curr| finder.lowest_match_index(curr)), expected);
            assert_eq!(finder.chain_table.len(), 1 << cparams.chain_log);
            assert_eq!(finder.bt_mask, (1 << (cparams.chain_log - 1)) - 1);
            assert_eq!(finder.nb_compares, 1 << cparams.search_log);
            assert_eq!(
                finder.sufficient_len,
                cparams.target_length.min(OPT_NUM as u32 - 1)
            );
            assert_eq!((finder.next_to_update, finder.next_to_update3), (40, 40));
        }
    }

    #[test]
    fn prefetching_ahead_skips_positions_past_the_block() {
        use super::super::params::{ParamSwitch as Switch, level_cparams};
        let bytes = sample(64, 3);
        let mut ms = MatchState::new(level_cparams(19, 64), Switch::Disable);
        let (finder, _) = Finder::split(&mut ms);
        for backend in Backend::available() {
            let block = BlockInput {
                window: mem::Window {
                    prefix: mem::Mem::new(&bytes, 100),
                    ext: mem::Mem::new(&[], 100),
                },
                start: 100,
                end: 164,
                backend,
                dms: None,
                symbol_costs: None,
            };
            dispatch!(backend.0, simd => {
                let ctx = Ctx::new(simd, &block, 3);
                // Hints only: every position up to the block's end, words
                // whole or cut, leaves the tables as they were.
                for ip in 100..164 {
                    ctx.prefetch_ahead::<3, true>(&finder, ip);
                    ctx.prefetch_ahead::<4, false>(&finder, ip);
                    ctx.prefetch_ahead::<5, false>(&finder, ip);
                    ctx.prefetch_ahead::<6, false>(&finder, ip);
                }
            });
        }
        assert!(finder.hash_table.iter().all(|&cell| cell == 0));
        assert!(finder.hash_table3.iter().all(|&cell| cell == 0));
    }

    #[test]
    fn table_weights_match_the_integer_weights() {
        // Every power of two and its neighbours up to 2^24, then values the
        // float form cannot hold, which take the integer fallback.
        let mut stats: Vec<u32> = (0..32)
            .flat_map(|bit| {
                let p = 1u32 << bit;
                [p.wrapping_sub(2), p.wrapping_sub(1), p, p.wrapping_add(1)]
            })
            .filter(|&stat| stat < (1 << 24))
            .collect();
        stats.extend(0..5000);
        for opt_level in [0, 2] {
            let mut out = vec![0u32; stats.len()];
            weights(&stats, opt_level, &mut out);
            for (&stat, &got) in stats.iter().zip(&out) {
                assert_eq!(
                    got,
                    weight(stat, opt_level),
                    "stat {stat} level {opt_level}"
                );
            }
            let large = [1u32 << 24, (1 << 24) + 1, u32::MAX - 1, 7];
            let mut out = [0u32; 4];
            weights(&large, opt_level, &mut out);
            for (&stat, &got) in large.iter().zip(&out) {
                assert_eq!(
                    got,
                    weight(stat, opt_level),
                    "stat {stat} level {opt_level}"
                );
            }
        }
    }

    #[test]
    fn position_tables_grow_with_the_block_up_to_the_parser_window() {
        let mut os = OptState::default();
        os.allocate();
        assert_eq!(os.table_bytes(), 0);
        os.reserve_matches(44, 1 << 12);
        assert_eq!(os.match_table.len(), 47);
        // A smaller block keeps the larger table.
        os.reserve_matches(5, 1 << 12);
        assert_eq!(os.match_table.len(), 47);
        // Few comparisons bound the list below the block.
        os.reserve_matches(BLOCK_SIZE_MAX, 1 << 4);
        assert_eq!(os.match_table.len(), 47);
        os.reserve_matches(BLOCK_SIZE_MAX, 1 << 6);
        assert_eq!(os.match_table.len(), 69);
        os.reserve_matches(BLOCK_SIZE_MAX, 1 << 13);
        assert_eq!(os.match_table.len(), OPT_SIZE);
        // Positions follow the block's remaining length.
        let mut table = core::mem::take(&mut os.price_table);
        assert_eq!(grow_positions(&mut table, 3).len(), 3);
        grow_positions(&mut table, 40);
        assert_eq!(grow_positions(&mut table, 12).len(), 40);
        os.price_table = table;
        assert!(os.table_bytes() >= 40 * core::mem::size_of::<Optimal>());
    }

    #[test]
    fn tiny_blocks_use_predefined_prices() {
        for opt_level in [0, 2] {
            let mut os = OptState::default();
            os.allocate();
            os.rescale_freqs(b"tiny", opt_level, None);
            assert_eq!(os.price_type, PriceType::Predef);
            assert_prices_match(&os, opt_level);
        }
    }

    #[test]
    fn tree_comparisons_read_both_segments_like_the_reference() {
        // Repetitive bytes so that runs cross the segment boundary.
        let ext: Vec<u8> = (0..64u32).map(|i| (i % 5) as u8).collect();
        let mut prefix: Vec<u8> = (0..96u32).map(|i| (i % 5) as u8).collect();
        prefix[70] = 9;
        let dict_limit = 1000u32;
        let window = mem::Window {
            prefix: mem::Mem::new(&prefix, dict_limit),
            ext: mem::Mem::new(&ext, dict_limit - 64),
        };
        let end = dict_limit + prefix.len() as u32;
        for backend in Backend::available() {
            let block = BlockInput {
                window,
                start: dict_limit,
                end,
                backend,
                dms: None,
                symbol_costs: None,
            };
            dispatch!(backend.0, simd => {
                let ctx = Ctx::new(simd, &block, 4);
                for curr in dict_limit + 10..dict_limit + 40 {
                    let current = ctx.current(curr);
                    for candidate in dict_limit - 64..curr {
                        let full = window.count_virtual(curr, candidate, end);
                        for known in [0, 1, 7, 8, 9, 30].map(|k: usize| k.min(full)) {
                            let (length, smaller) =
                                ctx.extend::<true>(current, curr, candidate, known);
                            assert_eq!(length, full, "{backend} {curr} {candidate} {known}");
                            let expected = curr + (full as u32) < end
                                && window.byte_at(candidate + full as u32)
                                    < window.byte_at(curr + full as u32);
                            assert_eq!(smaller, expected, "{backend} {curr} {candidate}");
                            if candidate >= dict_limit {
                                let plain = ctx.extend::<false>(current, curr, candidate, known);
                                assert_eq!(plain, (length, smaller));
                            }
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn specialized_hashes_match_the_shared_ones() {
        let bytes = sample(64, 7);
        for backend in Backend::available() {
            let block = BlockInput {
                window: mem::Window {
                    prefix: mem::Mem::new(&bytes, 100),
                    ext: mem::Mem::new(&[], 100),
                },
                start: 100,
                end: 164,
                backend,
                dms: None,
                symbol_costs: None,
            };
            let prefix = &block.window.prefix;
            dispatch!(backend.0, simd => {
                let ctx = Ctx::new(simd, &block, 4);
                for index in 100..156 {
                    for hash_log in [10, 17, 24] {
                        let expected = |mls| mem::hash_ptr(prefix, index, hash_log, mls);
                        assert_eq!(ctx.hash::<3>(index, hash_log), expected(3));
                        assert_eq!(ctx.hash::<4>(index, hash_log), expected(4));
                        assert_eq!(ctx.hash::<5>(index, hash_log), expected(5));
                        assert_eq!(ctx.hash::<6>(index, hash_log), expected(6));
                        assert_eq!(ctx.hash::<7>(index, hash_log), expected(7));
                    }
                }
            });
        }
    }

    // -----------------------------------------------------------------------
    // Dictionaries.

    use super::super::cctx::{Engine, RequestedParams, compress_bound};
    use super::super::cdict::{AttachPref, CDict};
    use super::super::compress_once_into;
    use super::super::dict_tests::{check, dict_params, inputs, request, trained};
    use super::super::params::Strategy;
    use super::super::stream::{EndOp, Stream};
    use crate::dictionary::DictionaryContent;
    use alloc::boxed::Box;
    use zstd_ffi::corpus;
    use zstd_ffi::oracle::{self, Directive};

    /// A raw-content dictionary: text and some of the records.
    fn raw_dictionary(records: &[Vec<u8>]) -> Vec<u8> {
        [corpus::text(12_000, 9), records[..100].concat()].concat()
    }

    /// Prefixes of the last record, from empty to a few dozen bytes.
    fn tiny_inputs(records: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let record = records[records.len() - 1].repeat(2);
        [0usize, 1, 3, 5, 8, 9, 12, 16, 20, 33, 47, 64]
            .into_iter()
            .map(|len| record[..len.min(record.len())].to_vec())
            .collect()
    }

    /// A handful of record inputs and tiny ones, for wide parameter sweeps.
    fn few_inputs(records: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = [1usize, 3, 40, 300]
            .into_iter()
            .map(|count| records[records.len() - count..].concat())
            .collect();
        out.push(corpus::text(9_000, 5));
        out.extend(tiny_inputs(records).into_iter().step_by(3));
        out
    }

    /// [`check`] on every backend the host offers, each with its own
    /// engine, with the requested maximum block size.
    /// `ZSTD_c_maxBlockSize`, which the shared oracle parameters omit.
    const ZSTD_C_MAX_BLOCK_SIZE: zstd_ffi::ZSTD_cParameter = 1015;

    fn check_backends(requested: &RequestedParams, dict: &[u8], inputs: &[Vec<u8>]) {
        let content = DictionaryContent::Auto;
        let cdict = CDict::new(Box::from(dict), content, requested).expect("prepares");
        let mut params = dict_params(requested, dict, content);
        if requested.max_block_size != 0 {
            params = params.with(ZSTD_C_MAX_BLOCK_SIZE, requested.max_block_size as i32);
        }
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for (index, input) in inputs.iter().enumerate() {
                let expected = oracle::compress2(input, &params).expect("oracle");
                let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                let size =
                    compress_once_into(&mut engine, requested, Some(&cdict), input, 3, &mut dst)
                        .expect("compresses");
                assert!(
                    dst[..size] == expected[..],
                    "{backend} input {index}, {requested:?}"
                );
            }
        }
    }

    /// Streams `input` in chunks with a flush in the middle, with `cdict`.
    fn stream_with_dictionary(
        engine: &mut Engine,
        requested: &RequestedParams,
        cdict: &CDict,
        steps: &[(&[u8], Directive)],
    ) -> Vec<u8> {
        let mut state = Stream::new();
        let mut out = Vec::new();
        let mut buffer = vec![0u8; 1 << 17];
        for &(data, directive) in steps {
            let op = match directive {
                Directive::Continue => EndOp::Continue,
                Directive::Flush => EndOp::Flush,
                Directive::End => EndOp::End,
            };
            let mut in_pos = 0;
            loop {
                let step = state
                    .compress(
                        engine,
                        requested,
                        Some(cdict),
                        &mut buffer[..],
                        &data[in_pos..],
                        op,
                    )
                    .expect("stream");
                in_pos += step.consumed;
                out.extend_from_slice(&buffer[..step.produced]);
                if in_pos == data.len() && (op == EndOp::Continue || step.remaining == 0) {
                    break;
                }
            }
        }
        out
    }

    #[test]
    fn attached_trained_dictionaries_match_the_reference_at_optimal_levels() {
        let (dict, records) = trained();
        let inputs = [inputs(&records), tiny_inputs(&records)].concat();
        // From level 13 on, a dictionary of this size resolves to the
        // optimal parser.
        for level in 13..=22 {
            let requested = request(level, None, AttachPref::ForceAttach);
            check(&requested, &dict, DictionaryContent::Auto, &inputs);
        }
    }

    #[test]
    fn attached_raw_dictionaries_match_the_reference_at_optimal_levels() {
        let (trained, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = inputs(&records);
        for level in 13..=22 {
            let requested = request(level, None, AttachPref::ForceAttach);
            check(&requested, &raw, DictionaryContent::Auto, &inputs);
            // A trained dictionary read as raw content: no entropy tables.
            check(&requested, &trained, DictionaryContent::Raw, &inputs);
        }
    }

    #[test]
    fn default_attachment_matches_the_reference_at_optimal_levels() {
        let (dict, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = [inputs(&records), tiny_inputs(&records)].concat();
        for level in 13..=22 {
            let requested = request(level, None, AttachPref::Default);
            check(&requested, &dict, DictionaryContent::Auto, &inputs);
            check(&requested, &raw, DictionaryContent::Auto, &inputs);
        }
    }

    #[test]
    fn copied_and_loaded_trained_dictionaries_seed_statistics_like_the_reference() {
        let (dict, records) = trained();
        let inputs = [inputs(&records), tiny_inputs(&records)].concat();
        for level in 11..=22 {
            for pref in [AttachPref::ForceCopy, AttachPref::ForceLoad] {
                let requested = request(level, None, pref);
                check(&requested, &dict, DictionaryContent::Auto, &inputs);
            }
        }
    }

    #[test]
    fn explicit_optimal_strategies_match_the_reference_with_dictionaries() {
        let (dict, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = few_inputs(&records);
        for strategy in [Strategy::BtOpt, Strategy::BtUltra, Strategy::BtUltra2] {
            for level in [3, 11, 12, 19] {
                for min_match in [3, 4, 5] {
                    for pref in [AttachPref::ForceAttach, AttachPref::ForceCopy] {
                        let mut requested = request(level, Some(strategy), pref);
                        requested.cparams.min_match = min_match;
                        check(&requested, &dict, DictionaryContent::Auto, &inputs);
                    }
                    let mut requested = request(level, Some(strategy), AttachPref::ForceAttach);
                    requested.cparams.min_match = min_match;
                    check(&requested, &raw, DictionaryContent::Auto, &inputs);
                }
            }
        }
    }

    /// Dictionary loads hash 7 bytes for a minimum match of 7, which the
    /// searches clamp to 6.
    #[test]
    fn dictionaries_load_with_a_minimum_match_of_seven_like_the_reference() {
        let (dict, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = few_inputs(&records);
        for (strategy, prefs) in [
            (
                Strategy::BtOpt,
                &[
                    AttachPref::ForceAttach,
                    AttachPref::ForceCopy,
                    AttachPref::ForceLoad,
                ][..],
            ),
            (
                Strategy::BtUltra2,
                &[AttachPref::ForceAttach, AttachPref::ForceCopy][..],
            ),
            (
                Strategy::BtLazy2,
                &[
                    AttachPref::ForceAttach,
                    AttachPref::ForceCopy,
                    AttachPref::ForceLoad,
                ][..],
            ),
        ] {
            for &pref in prefs {
                for level in [12, 19] {
                    let mut requested = request(level, Some(strategy), pref);
                    requested.cparams.min_match = 7;
                    check(&requested, &dict, DictionaryContent::Auto, &inputs);
                    check(&requested, &raw, DictionaryContent::Auto, &inputs);
                }
            }
        }
    }

    #[test]
    fn small_blocks_carry_dictionary_statistics_like_the_reference() {
        let (dict, records) = trained();
        let inputs = [
            records[records.len() - 300..].concat(),
            corpus::mixed(40_000, 3),
        ];
        for level in [13, 16, 19, 22] {
            for pref in [
                AttachPref::ForceAttach,
                AttachPref::ForceCopy,
                AttachPref::ForceLoad,
            ] {
                let mut requested = request(level, None, pref);
                requested.max_block_size = 1 << 12;
                check_backends(&requested, &dict, &inputs);
            }
        }
    }

    #[test]
    fn long_distance_matching_with_dictionaries_matches_the_reference() {
        let (dict, records) = trained();
        let base = records[records.len() - 600..].concat();
        let input = [base.clone(), corpus::binary(150_000, 4), base].concat();
        let inputs = [input, records[records.len() - 40..].concat()];
        for level in [16, 19, 22] {
            for pref in [AttachPref::ForceAttach, AttachPref::ForceCopy] {
                let mut requested = request(level, None, pref);
                requested.ldm.enable = ParamSwitch::Enable;
                requested.cparams.window_log = 24;
                check(&requested, &dict, DictionaryContent::Auto, &inputs);
            }
        }
    }

    #[test]
    fn every_backend_matches_the_reference_with_attached_dictionaries() {
        let (dict, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = few_inputs(&records);
        for (level, strategy) in [
            (13, None),
            (16, None),
            (19, None),
            (22, None),
            (12, Some(Strategy::BtUltra2)),
        ] {
            let requested = request(level, strategy, AttachPref::ForceAttach);
            check_backends(&requested, &dict, &inputs);
            check_backends(&requested, &raw, &inputs);
            let requested = request(level, strategy, AttachPref::ForceCopy);
            check_backends(&requested, &dict, &inputs);
        }
    }

    #[test]
    fn streaming_with_an_attached_dictionary_matches_the_reference() {
        let (dict, records) = trained();
        let input = records[records.len() - 1200..].concat();
        let chunks: Vec<&[u8]> = input.chunks(7_001).collect();
        let steps: Vec<(&[u8], Directive)> = chunks
            .iter()
            .enumerate()
            .map(|(n, &data)| {
                let directive = if n + 1 == chunks.len() {
                    Directive::End
                } else if n == chunks.len() / 2 {
                    Directive::Flush
                } else {
                    Directive::Continue
                };
                (data, directive)
            })
            .collect();
        for level in [13, 16, 19, 22] {
            for pref in [AttachPref::Default, AttachPref::ForceAttach] {
                let requested = request(level, None, pref);
                let content = DictionaryContent::Auto;
                let cdict =
                    CDict::new(Box::from(&dict[..]), content, &requested).expect("prepares");
                let expected =
                    oracle::compress_stream(&dict_params(&requested, &dict, content), &steps)
                        .expect("oracle");
                for backend in Backend::available() {
                    let mut engine = Engine::with_backend(backend);
                    let ours = stream_with_dictionary(&mut engine, &requested, &cdict, &steps);
                    assert!(ours == expected, "{backend} level {level} {pref:?}");
                }
            }
        }
    }

    #[test]
    fn dictionary_variants_without_a_dictionary_parse_like_the_plain_ones() {
        use super::super::params::{ParamSwitch as Switch, level_cparams};
        let bytes = corpus::text(20_000, 2);
        let block = BlockInput {
            window: mem::Window {
                prefix: mem::Mem::new(&bytes, 2),
                ext: mem::Mem::new(&[], 2),
            },
            start: 2,
            end: 2 + bytes.len() as u32,
            backend: Backend::default(),
            dms: None,
            symbol_costs: None,
        };
        use super::super::match_state::BlockCompressor;
        let pairs: [(BlockCompressor, BlockCompressor, i32); 2] =
            [(btopt, btopt_dms, 16), (btultra, btultra_dms, 18)];
        for (plain, dms, level) in pairs {
            let cparams = level_cparams(level, bytes.len() as u64);
            let mut outputs = Vec::new();
            for compressor in [plain, dms] {
                let mut ms = MatchState::new(cparams, Switch::Disable);
                let mut seq_store = SeqStore::default();
                seq_store.reserve(bytes.len(), cparams.min_match);
                let mut rep = [1, 4, 8];
                let last = compressor(&mut ms, &mut seq_store, &mut rep, &block);
                outputs.push((last, rep, seq_store.sequences, seq_store.literals));
            }
            assert_eq!(outputs[0], outputs[1], "level {level}");
        }
    }

    #[test]
    fn dictionary_runs_continue_at_the_prefix_like_the_reference() {
        let dict_bytes: Vec<u8> = (0..300u32).map(|i| (i % 7) as u8).collect();
        let requested = request(19, Some(Strategy::BtUltra), AttachPref::ForceAttach);
        let cdict = CDict::new(
            Box::from(&dict_bytes[..]),
            DictionaryContent::Raw,
            &requested,
        )
        .expect("prepares");
        let dms = cdict.match_state();
        let dict = dms.window.prefix;
        let mut prefix: Vec<u8> = (0..120u32).map(|i| (i % 7) as u8).collect();
        prefix[90] = 50;
        let dict_limit = 1000u32;
        let block = BlockInput {
            window: mem::Window {
                prefix: mem::Mem::new(&prefix, dict_limit),
                ext: mem::Mem::new(&[], dict_limit),
            },
            start: dict_limit,
            end: dict_limit + prefix.len() as u32,
            backend: Backend::default(),
            dms: Some(dms),
            symbol_costs: None,
        };
        // The virtual run: the dictionary from `candidate`, then the prefix
        // from its start.
        let expected = |ip: u32, candidate: u32| {
            let virtual_bytes: Vec<u8> = dict
                .slice(candidate.min(dict.end()), dict.end())
                .iter()
                .chain(prefix.iter())
                .copied()
                .collect();
            if candidate > dict.end() {
                return 0;
            }
            prefix[(ip - dict_limit) as usize..]
                .iter()
                .zip(&virtual_bytes)
                .take_while(|(a, b)| a == b)
                .count()
        };
        for backend in Backend::available() {
            dispatch!(backend.0, simd => {
                let search = DictSearch::new(&dms);
                let ctx = Ctx::with_dictionary(simd, &block, 4, &search);
                assert_eq!(ctx.dms.high_limit, dict.end());
                assert_eq!(ctx.dms.bytes.len(), dict_bytes.len());
                for ip in dict_limit..dict_limit + 100 {
                    for candidate in (dict.start()..dict.end() + 3).step_by(5) {
                        assert_eq!(
                            ctx.count_dictionary(ip, candidate),
                            expected(ip, candidate),
                            "{backend} {ip} {candidate}"
                        );
                    }
                }
            });
        }
    }

    #[test]
    fn dictionary_tables_seed_the_first_block_statistics() {
        let mut tables = Tables::default();
        tables.huf.repeat = huf::Repeat::Valid;
        tables.huf.table.max_symbol_value = 255;
        for (symbol, bits) in tables.huf.table.nb_bits.iter_mut().enumerate() {
            // Absent symbols and every length up to the scale.
            *bits = (symbol % 12) as u8;
        }
        // Low-probability, absent and larger symbols.
        let normalized = |last: u32| {
            let mut norm: Vec<i16> = (0..=last as usize)
                .map(|s| [1, -1, 0, 3, 9][s % 5])
                .collect();
            let total: i16 = norm.iter().map(|&n| n.abs()).sum();
            norm[0] += 256 - total;
            norm
        };
        tables.fse.matchlength.build(&normalized(MAX_ML), MAX_ML, 8);
        tables.fse.litlength.build(&normalized(MAX_LL), MAX_LL, 8);
        tables.fse.offcode.build(&normalized(MAX_OFF), MAX_OFF, 8);
        let frequency = |bits: u32, scale: u32| {
            if bits == 0 {
                1
            } else {
                1u32.wrapping_shl(scale.wrapping_sub(bits))
            }
        };
        for literal_compression in [ParamSwitch::Auto, ParamSwitch::Disable] {
            for (src, opt_level) in [(&b"tiny"[..], 0), (&[7u8; 300][..], 2)] {
                let mut os = OptState {
                    literal_compression,
                    ..OptState::default()
                };
                os.allocate();
                os.rescale_freqs(src, opt_level, Some(&tables));
                // Dynamic prices even for a tiny block.
                assert_eq!(os.price_type, PriceType::Dynamic);
                if literal_compression == ParamSwitch::Auto {
                    for literal in 0..=MAX_LIT {
                        let bits = u32::from(tables.huf.table.nb_bits[literal as usize]);
                        assert_eq!(os.lit_freq[literal as usize], frequency(bits, 11));
                    }
                    let sum = os.lit_freq.iter().fold(0u32, |sum, &f| sum.wrapping_add(f));
                    assert_eq!(os.lit_sum, sum);
                }
                for (freqs, table, sum) in [
                    (
                        &os.lit_length_freq,
                        &tables.fse.litlength,
                        os.lit_length_sum,
                    ),
                    (
                        &os.match_length_freq,
                        &tables.fse.matchlength,
                        os.match_length_sum,
                    ),
                    (&os.off_code_freq, &tables.fse.offcode, os.off_code_sum),
                ] {
                    for (symbol, &freq) in freqs.iter().enumerate() {
                        let tt = table.symbol_tt[symbol];
                        let bits = (tt.delta_nb_bits + 0xFFFF) >> 16;
                        assert_eq!(freq, frequency(bits, 10), "symbol {symbol}");
                    }
                    assert_eq!(sum, freqs.iter().sum::<u32>());
                }
                assert_prices_match(&os, opt_level);
                // Later blocks scale the statistics down as without one.
                os.rescale_freqs(src, opt_level, Some(&tables));
                assert_prices_match(&os, opt_level);
            }
        }
        // Tables a dictionary did not validate leave the plain seeding.
        tables.huf.repeat = huf::Repeat::Check;
        let mut os = OptState::default();
        os.allocate();
        os.rescale_freqs(b"tiny", 0, Some(&tables));
        assert_eq!(os.price_type, PriceType::Predef);
        assert_eq!(os.match_length_sum, MAX_ML + 1);
    }
}
