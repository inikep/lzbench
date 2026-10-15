//! Compression parameters and their resolution (`clevels.h` and the
//! parameter half of `zstd_compress.c`).
//!
//! A compression is configured by a level, optional overrides of individual
//! parameters, and the expected input size. The reference turns those into
//! the parameters it actually compresses with in several steps — level
//! table lookup, overrides, size-driven shrinking, then the automatic
//! switches for the row match finder, LDM and block splitting — and every
//! step changes the output, so each is ported literally here.

use crate::shared::format::{BLOCK_SIZE_MAX, highbit32};

/// `ZSTD_CONTENTSIZE_UNKNOWN`.
pub(crate) const CONTENT_SIZE_UNKNOWN: u64 = u64::MAX;

/// Highest compression level.
pub(crate) const MAX_CLEVEL: i32 = 22;
/// Level used when none is given.
pub(crate) const DEFAULT_CLEVEL: i32 = 3;
/// Lowest (fastest) compression level: `-ZSTD_TARGETLENGTH_MAX`.
pub(crate) const MIN_CLEVEL: i32 = -(BLOCK_SIZE_MAX as i32);

pub(crate) const WINDOWLOG_MAX: u32 = 31;
pub(crate) const WINDOWLOG_MIN: u32 = 10;
pub(crate) const HASHLOG_MAX: u32 = 30;
pub(crate) const HASHLOG_MIN: u32 = 6;
pub(crate) const CHAINLOG_MAX: u32 = 30;
pub(crate) const CHAINLOG_MIN: u32 = HASHLOG_MIN;
pub(crate) const SEARCHLOG_MAX: u32 = WINDOWLOG_MAX - 1;
pub(crate) const SEARCHLOG_MIN: u32 = 1;
pub(crate) const MINMATCH_MAX: u32 = 7;
pub(crate) const MINMATCH_MIN: u32 = 3;
pub(crate) const TARGETLENGTH_MAX: u32 = BLOCK_SIZE_MAX as u32;
pub(crate) const TARGETLENGTH_MIN: u32 = 0;
pub(crate) const LDM_HASHLOG_MIN: u32 = HASHLOG_MIN;
pub(crate) const LDM_HASHLOG_MAX: u32 = HASHLOG_MAX;
pub(crate) const LDM_MINMATCH_MIN: u32 = 4;
pub(crate) const LDM_MINMATCH_MAX: u32 = 4096;
pub(crate) const LDM_BUCKETSIZELOG_MIN: u32 = 1;
pub(crate) const LDM_BUCKETSIZELOG_MAX: u32 = 8;
pub(crate) const LDM_HASHRATELOG_MIN: u32 = 0;
pub(crate) const LDM_HASHRATELOG_MAX: u32 = WINDOWLOG_MAX - HASHLOG_MIN;
/// Window LDM switches to when enabled without an explicit window.
pub(crate) const LDM_DEFAULT_WINDOW_LOG: u32 = 27;
/// Tag bits of the row match finder (`ZSTD_ROW_HASH_TAG_BITS`).
pub(crate) const ROW_HASH_TAG_BITS: u32 = 8;

/// Match finding strategy, from fastest to strongest (`ZSTD_strategy`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum Strategy {
    Fast = 1,
    DFast = 2,
    Greedy = 3,
    Lazy = 4,
    Lazy2 = 5,
    BtLazy2 = 6,
    BtOpt = 7,
    BtUltra = 8,
    BtUltra2 = 9,
}

impl Strategy {
    /// Every strategy, in reference order.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 9] = [
        Self::Fast,
        Self::DFast,
        Self::Greedy,
        Self::Lazy,
        Self::Lazy2,
        Self::BtLazy2,
        Self::BtOpt,
        Self::BtUltra,
        Self::BtUltra2,
    ];

    /// The reference's numeric value.
    pub(crate) const fn value(self) -> u32 {
        self as u32
    }

    /// The strategy with numeric value `value`, if any.
    #[cfg(test)]
    pub(crate) const fn from_value(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Fast),
            2 => Some(Self::DFast),
            3 => Some(Self::Greedy),
            4 => Some(Self::Lazy),
            5 => Some(Self::Lazy2),
            6 => Some(Self::BtLazy2),
            7 => Some(Self::BtOpt),
            8 => Some(Self::BtUltra),
            9 => Some(Self::BtUltra2),
            _ => None,
        }
    }
}

/// A three-way switch (`ZSTD_ParamSwitch_e`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) enum ParamSwitch {
    /// Let the encoder decide.
    #[default]
    Auto,
    /// Force on.
    Enable,
    /// Force off.
    Disable,
}

/// The core compression parameters (`ZSTD_compressionParameters`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct CParams {
    pub(crate) window_log: u32,
    pub(crate) chain_log: u32,
    pub(crate) hash_log: u32,
    pub(crate) search_log: u32,
    pub(crate) min_match: u32,
    pub(crate) target_length: u32,
    pub(crate) strategy: Strategy,
}

/// Per-parameter overrides; `0` keeps the level's value.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) struct CParamOverrides {
    pub(crate) window_log: u32,
    pub(crate) chain_log: u32,
    pub(crate) hash_log: u32,
    pub(crate) search_log: u32,
    pub(crate) min_match: u32,
    pub(crate) target_length: u32,
    pub(crate) strategy: Option<Strategy>,
}

const fn cp(w: u32, c: u32, h: u32, s: u32, l: u32, t: u32, strategy: Strategy) -> CParams {
    CParams {
        window_log: w,
        chain_log: c,
        hash_log: h,
        search_log: s,
        min_match: l,
        target_length: t,
        strategy,
    }
}

use Strategy::{BtLazy2, BtOpt, BtUltra, BtUltra2, DFast, Fast, Greedy, Lazy, Lazy2};

/// `ZSTD_defaultCParameters`: rows by level, tables by source size class.
const DEFAULT_CPARAMS: [[CParams; MAX_CLEVEL as usize + 1]; 4] = [
    // Any source size above 256 KiB.
    [
        cp(19, 12, 13, 1, 6, 1, Fast),
        cp(19, 13, 14, 1, 7, 0, Fast),
        cp(20, 15, 16, 1, 6, 0, Fast),
        cp(21, 16, 17, 1, 5, 0, DFast),
        cp(21, 18, 18, 1, 5, 0, DFast),
        cp(21, 18, 19, 3, 5, 2, Greedy),
        cp(21, 18, 19, 3, 5, 4, Lazy),
        cp(21, 19, 20, 4, 5, 8, Lazy),
        cp(21, 19, 20, 4, 5, 16, Lazy2),
        cp(22, 20, 21, 4, 5, 16, Lazy2),
        cp(22, 21, 22, 5, 5, 16, Lazy2),
        cp(22, 21, 22, 6, 5, 16, Lazy2),
        cp(22, 22, 23, 6, 5, 32, Lazy2),
        cp(22, 22, 22, 4, 5, 32, BtLazy2),
        cp(22, 22, 23, 5, 5, 32, BtLazy2),
        cp(22, 23, 23, 6, 5, 32, BtLazy2),
        cp(22, 22, 22, 5, 5, 48, BtOpt),
        cp(23, 23, 22, 5, 4, 64, BtOpt),
        cp(23, 23, 22, 6, 3, 64, BtUltra),
        cp(23, 24, 22, 7, 3, 256, BtUltra2),
        cp(25, 25, 23, 7, 3, 256, BtUltra2),
        cp(26, 26, 24, 7, 3, 512, BtUltra2),
        cp(27, 27, 25, 9, 3, 999, BtUltra2),
    ],
    // Source size up to 256 KiB.
    [
        cp(18, 12, 13, 1, 5, 1, Fast),
        cp(18, 13, 14, 1, 6, 0, Fast),
        cp(18, 14, 14, 1, 5, 0, DFast),
        cp(18, 16, 16, 1, 4, 0, DFast),
        cp(18, 16, 17, 3, 5, 2, Greedy),
        cp(18, 17, 18, 5, 5, 2, Greedy),
        cp(18, 18, 19, 3, 5, 4, Lazy),
        cp(18, 18, 19, 4, 4, 4, Lazy),
        cp(18, 18, 19, 4, 4, 8, Lazy2),
        cp(18, 18, 19, 5, 4, 8, Lazy2),
        cp(18, 18, 19, 6, 4, 8, Lazy2),
        cp(18, 18, 19, 5, 4, 12, BtLazy2),
        cp(18, 19, 19, 7, 4, 12, BtLazy2),
        cp(18, 18, 19, 4, 4, 16, BtOpt),
        cp(18, 18, 19, 4, 3, 32, BtOpt),
        cp(18, 18, 19, 6, 3, 128, BtOpt),
        cp(18, 19, 19, 6, 3, 128, BtUltra),
        cp(18, 19, 19, 8, 3, 256, BtUltra),
        cp(18, 19, 19, 6, 3, 128, BtUltra2),
        cp(18, 19, 19, 8, 3, 256, BtUltra2),
        cp(18, 19, 19, 10, 3, 512, BtUltra2),
        cp(18, 19, 19, 12, 3, 512, BtUltra2),
        cp(18, 19, 19, 13, 3, 999, BtUltra2),
    ],
    // Source size up to 128 KiB.
    [
        cp(17, 12, 12, 1, 5, 1, Fast),
        cp(17, 12, 13, 1, 6, 0, Fast),
        cp(17, 13, 15, 1, 5, 0, Fast),
        cp(17, 15, 16, 2, 5, 0, DFast),
        cp(17, 17, 17, 2, 4, 0, DFast),
        cp(17, 16, 17, 3, 4, 2, Greedy),
        cp(17, 16, 17, 3, 4, 4, Lazy),
        cp(17, 16, 17, 3, 4, 8, Lazy2),
        cp(17, 16, 17, 4, 4, 8, Lazy2),
        cp(17, 16, 17, 5, 4, 8, Lazy2),
        cp(17, 16, 17, 6, 4, 8, Lazy2),
        cp(17, 17, 17, 5, 4, 8, BtLazy2),
        cp(17, 18, 17, 7, 4, 12, BtLazy2),
        cp(17, 18, 17, 3, 4, 12, BtOpt),
        cp(17, 18, 17, 4, 3, 32, BtOpt),
        cp(17, 18, 17, 6, 3, 256, BtOpt),
        cp(17, 18, 17, 6, 3, 128, BtUltra),
        cp(17, 18, 17, 8, 3, 256, BtUltra),
        cp(17, 18, 17, 10, 3, 512, BtUltra),
        cp(17, 18, 17, 5, 3, 256, BtUltra2),
        cp(17, 18, 17, 7, 3, 512, BtUltra2),
        cp(17, 18, 17, 9, 3, 512, BtUltra2),
        cp(17, 18, 17, 11, 3, 999, BtUltra2),
    ],
    // Source size up to 16 KiB.
    [
        cp(14, 12, 13, 1, 5, 1, Fast),
        cp(14, 14, 15, 1, 5, 0, Fast),
        cp(14, 14, 15, 1, 4, 0, Fast),
        cp(14, 14, 15, 2, 4, 0, DFast),
        cp(14, 14, 14, 4, 4, 2, Greedy),
        cp(14, 14, 14, 3, 4, 4, Lazy),
        cp(14, 14, 14, 4, 4, 8, Lazy2),
        cp(14, 14, 14, 6, 4, 8, Lazy2),
        cp(14, 14, 14, 8, 4, 8, Lazy2),
        cp(14, 15, 14, 5, 4, 8, BtLazy2),
        cp(14, 15, 14, 9, 4, 8, BtLazy2),
        cp(14, 15, 14, 3, 4, 12, BtOpt),
        cp(14, 15, 14, 4, 3, 24, BtOpt),
        cp(14, 15, 14, 5, 3, 32, BtUltra),
        cp(14, 15, 15, 6, 3, 64, BtUltra),
        cp(14, 15, 15, 7, 3, 256, BtUltra),
        cp(14, 15, 15, 5, 3, 48, BtUltra2),
        cp(14, 15, 15, 6, 3, 128, BtUltra2),
        cp(14, 15, 15, 7, 3, 256, BtUltra2),
        cp(14, 15, 15, 8, 3, 256, BtUltra2),
        cp(14, 15, 15, 8, 3, 512, BtUltra2),
        cp(14, 15, 15, 9, 3, 512, BtUltra2),
        cp(14, 15, 15, 10, 3, 999, BtUltra2),
    ],
];

/// Inclusive bounds of one parameter.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) lower: i64,
    pub(crate) upper: i64,
}

impl Bounds {
    const fn new(lower: i64, upper: i64) -> Self {
        Self { lower, upper }
    }

    /// Whether `value` lies within the bounds.
    pub(crate) const fn contains(self, value: i64) -> bool {
        value >= self.lower && value <= self.upper
    }

    #[cfg(test)]
    const fn clamp_u32(self, value: u32) -> u32 {
        let value = value as i64;
        if value < self.lower {
            self.lower as u32
        } else if value > self.upper {
            self.upper as u32
        } else {
            value as u32
        }
    }
}

pub(crate) const WINDOW_LOG_BOUNDS: Bounds =
    Bounds::new(WINDOWLOG_MIN as i64, WINDOWLOG_MAX as i64);
pub(crate) const CHAIN_LOG_BOUNDS: Bounds = Bounds::new(CHAINLOG_MIN as i64, CHAINLOG_MAX as i64);
pub(crate) const HASH_LOG_BOUNDS: Bounds = Bounds::new(HASHLOG_MIN as i64, HASHLOG_MAX as i64);
pub(crate) const SEARCH_LOG_BOUNDS: Bounds =
    Bounds::new(SEARCHLOG_MIN as i64, SEARCHLOG_MAX as i64);
pub(crate) const MIN_MATCH_BOUNDS: Bounds = Bounds::new(MINMATCH_MIN as i64, MINMATCH_MAX as i64);
pub(crate) const TARGET_LENGTH_BOUNDS: Bounds =
    Bounds::new(TARGETLENGTH_MIN as i64, TARGETLENGTH_MAX as i64);
pub(crate) const LDM_HASH_LOG_BOUNDS: Bounds =
    Bounds::new(LDM_HASHLOG_MIN as i64, LDM_HASHLOG_MAX as i64);
pub(crate) const LDM_MIN_MATCH_BOUNDS: Bounds =
    Bounds::new(LDM_MINMATCH_MIN as i64, LDM_MINMATCH_MAX as i64);
pub(crate) const LDM_BUCKET_SIZE_LOG_BOUNDS: Bounds =
    Bounds::new(LDM_BUCKETSIZELOG_MIN as i64, LDM_BUCKETSIZELOG_MAX as i64);
pub(crate) const LDM_HASH_RATE_LOG_BOUNDS: Bounds =
    Bounds::new(LDM_HASHRATELOG_MIN as i64, LDM_HASHRATELOG_MAX as i64);

impl CParams {
    /// Clamps every parameter into range (`ZSTD_clampCParams`).
    #[cfg(test)]
    pub(crate) const fn clamped(self) -> Self {
        Self {
            window_log: WINDOW_LOG_BOUNDS.clamp_u32(self.window_log),
            chain_log: CHAIN_LOG_BOUNDS.clamp_u32(self.chain_log),
            hash_log: HASH_LOG_BOUNDS.clamp_u32(self.hash_log),
            search_log: SEARCH_LOG_BOUNDS.clamp_u32(self.search_log),
            min_match: MIN_MATCH_BOUNDS.clamp_u32(self.min_match),
            target_length: TARGET_LENGTH_BOUNDS.clamp_u32(self.target_length),
            strategy: self.strategy,
        }
    }

    /// Whether every parameter is in range (`ZSTD_checkCParams`).
    pub(crate) const fn is_valid(&self) -> bool {
        WINDOW_LOG_BOUNDS.contains(self.window_log as i64)
            && CHAIN_LOG_BOUNDS.contains(self.chain_log as i64)
            && HASH_LOG_BOUNDS.contains(self.hash_log as i64)
            && SEARCH_LOG_BOUNDS.contains(self.search_log as i64)
            && MIN_MATCH_BOUNDS.contains(self.min_match as i64)
            && TARGET_LENGTH_BOUNDS.contains(self.target_length as i64)
    }

    /// Replaces every parameter an override sets (`ZSTD_overrideCParams`).
    pub(crate) const fn overridden(mut self, overrides: &CParamOverrides) -> Self {
        if overrides.window_log != 0 {
            self.window_log = overrides.window_log;
        }
        if overrides.hash_log != 0 {
            self.hash_log = overrides.hash_log;
        }
        if overrides.chain_log != 0 {
            self.chain_log = overrides.chain_log;
        }
        if overrides.search_log != 0 {
            self.search_log = overrides.search_log;
        }
        if overrides.min_match != 0 {
            self.min_match = overrides.min_match;
        }
        if overrides.target_length != 0 {
            self.target_length = overrides.target_length;
        }
        if let Some(strategy) = overrides.strategy {
            self.strategy = strategy;
        }
        self
    }
}

/// `ZSTD_cycleLog`.
pub(crate) const fn cycle_log(hash_log: u32, strategy: Strategy) -> u32 {
    let bt_scale = (strategy as u32 >= Strategy::BtLazy2 as u32) as u32;
    hash_log - bt_scale
}

/// Whether `strategy` can use the row match finder.
pub(crate) const fn row_match_finder_supported(strategy: Strategy) -> bool {
    matches!(
        strategy,
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2
    )
}

/// Whether the row match finder is in use (`ZSTD_rowMatchFinderUsed`).
pub(crate) const fn row_match_finder_used(strategy: Strategy, mode: ParamSwitch) -> bool {
    row_match_finder_supported(strategy) && matches!(mode, ParamSwitch::Enable)
}

/// `ZSTD_resolveRowMatchFinderMode`.
pub(crate) const fn resolve_row_match_finder(mode: ParamSwitch, cparams: &CParams) -> ParamSwitch {
    if !matches!(mode, ParamSwitch::Auto) {
        return mode;
    }
    if !row_match_finder_supported(cparams.strategy) {
        return ParamSwitch::Disable;
    }
    if cparams.window_log > 14 {
        ParamSwitch::Enable
    } else {
        ParamSwitch::Disable
    }
}

/// `ZSTD_resolveBlockSplitterMode`.
pub(crate) const fn resolve_block_splitter(mode: ParamSwitch, cparams: &CParams) -> ParamSwitch {
    if !matches!(mode, ParamSwitch::Auto) {
        return mode;
    }
    if cparams.strategy as u32 >= Strategy::BtOpt as u32 && cparams.window_log >= 17 {
        ParamSwitch::Enable
    } else {
        ParamSwitch::Disable
    }
}

/// `ZSTD_resolveEnableLdm`.
pub(crate) const fn resolve_ldm(mode: ParamSwitch, cparams: &CParams) -> ParamSwitch {
    if !matches!(mode, ParamSwitch::Auto) {
        return mode;
    }
    if cparams.strategy as u32 >= Strategy::BtOpt as u32 && cparams.window_log >= 27 {
        ParamSwitch::Enable
    } else {
        ParamSwitch::Disable
    }
}

/// `ZSTD_resolveMaxBlockSize`.
pub(crate) const fn resolve_max_block_size(max_block_size: usize) -> usize {
    if max_block_size == 0 {
        BLOCK_SIZE_MAX
    } else {
        max_block_size
    }
}

/// How a dictionary's size enters parameter selection (`ZSTD_CParamMode_e`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum CParamMode {
    /// No dictionary, or nothing known about how it is used
    /// (`ZSTD_cpm_unknown`).
    Unknown,
    /// A dictionary is attached by reference; its tables have their own
    /// parameters, so only the source counts (`ZSTD_cpm_attachDict`).
    AttachDict,
    /// Parameters for a dictionary's own tables (`ZSTD_cpm_createCDict`).
    CreateCDict,
    /// A dictionary is copied into the context's tables
    /// (`ZSTD_cpm_noAttachDict`).
    NoAttachDict,
}

/// `ZSTD_SHORT_CACHE_TAG_BITS`.
pub(crate) const SHORT_CACHE_TAG_BITS: u32 = 8;

/// Whether a dictionary's fast and dfast tables carry tags in their low bits
/// (`ZSTD_CDictIndicesAreTagged`).
pub(crate) const fn cdict_indices_are_tagged(cparams: &CParams) -> bool {
    matches!(cparams.strategy, Strategy::Fast | Strategy::DFast)
}

/// `ZSTD_dictAndWindowLog`.
fn dict_and_window_log(window_log: u32, src_size: u64, dict_size: u64) -> u32 {
    const MAX_WINDOW_SIZE: u64 = 1 << WINDOWLOG_MAX;
    if dict_size == 0 {
        return window_log;
    }
    let window_size = 1u64 << window_log;
    let dict_and_window_size = dict_size + window_size;
    if window_size >= dict_size + src_size {
        window_log
    } else if dict_and_window_size >= MAX_WINDOW_SIZE {
        WINDOWLOG_MAX
    } else {
        highbit32((dict_and_window_size - 1) as u32) + 1
    }
}

/// Shrinks parameters to fit a source of `src_size` bytes and a dictionary
/// of `dict_size` used as `mode` says (`ZSTD_adjustCParams_internal`).
// Always inlined: callers without a dictionary pass constants that fold
// the dictionary branches away.
#[inline(always)]
pub(crate) fn adjust_cparams_for(
    mut cpar: CParams,
    mut src_size: u64,
    mut dict_size: u64,
    mode: CParamMode,
    use_row_match_finder: ParamSwitch,
) -> CParams {
    const MIN_SRC_SIZE: u64 = 513;
    const MAX_WINDOW_RESIZE: u64 = 1 << (WINDOWLOG_MAX - 1);
    debug_assert!(cpar.is_valid());

    match mode {
        CParamMode::Unknown | CParamMode::NoAttachDict => {}
        CParamMode::CreateCDict => {
            if dict_size != 0 && src_size == CONTENT_SIZE_UNKNOWN {
                src_size = MIN_SRC_SIZE;
            }
        }
        CParamMode::AttachDict => dict_size = 0,
    }

    // Resize the window for small inputs, to use less memory.
    if src_size <= MAX_WINDOW_RESIZE && dict_size <= MAX_WINDOW_RESIZE {
        let t_size = (src_size + dict_size) as u32;
        let hash_size_min = 1u32 << HASHLOG_MIN;
        let src_log = if t_size < hash_size_min {
            HASHLOG_MIN
        } else {
            highbit32(t_size - 1) + 1
        };
        if cpar.window_log > src_log {
            cpar.window_log = src_log;
        }
    }
    if src_size != CONTENT_SIZE_UNKNOWN {
        let dict_and_window_log = dict_and_window_log(cpar.window_log, src_size, dict_size);
        let cycle = cycle_log(cpar.chain_log, cpar.strategy);
        if cpar.hash_log > dict_and_window_log + 1 {
            cpar.hash_log = dict_and_window_log + 1;
        }
        if cycle > dict_and_window_log {
            cpar.chain_log -= cycle - dict_and_window_log;
        }
    }
    if cpar.window_log < crate::shared::format::WINDOW_LOG_ABSOLUTE_MIN {
        cpar.window_log = crate::shared::format::WINDOW_LOG_ABSOLUTE_MIN;
    }

    // Tagged dictionary tables leave 24 bits per index.
    if mode == CParamMode::CreateCDict && cdict_indices_are_tagged(&cpar) {
        let max_short_cache_hash_log = 32 - SHORT_CACHE_TAG_BITS;
        if cpar.hash_log > max_short_cache_hash_log {
            cpar.hash_log = max_short_cache_hash_log;
        }
        if cpar.chain_log > max_short_cache_hash_log {
            cpar.chain_log = max_short_cache_hash_log;
        }
    }

    let use_row_match_finder = if matches!(use_row_match_finder, ParamSwitch::Auto) {
        ParamSwitch::Enable
    } else {
        use_row_match_finder
    };
    if row_match_finder_used(cpar.strategy, use_row_match_finder) {
        let row_log = cpar.search_log.clamp(4, 6);
        let max_row_hash_log = 32 - ROW_HASH_TAG_BITS;
        let max_hash_log = max_row_hash_log + row_log;
        if cpar.hash_log > max_hash_log {
            cpar.hash_log = max_hash_log;
        }
    }
    cpar
}

/// Level-table parameters for a source size hint (`ZSTD_getCParams_internal`
/// without a dictionary).
#[cfg(test)]
pub(crate) fn level_cparams(level: i32, src_size_hint: u64) -> CParams {
    level_cparams_for(level, src_size_hint, 0, CParamMode::Unknown)
}

/// The size that selects a level table row (`ZSTD_getCParamRowSize`).
const fn cparam_row_size(src_size_hint: u64, dict_size: u64, mode: CParamMode) -> u64 {
    let dict_size = if matches!(mode, CParamMode::AttachDict) {
        0
    } else {
        dict_size
    };
    let unknown = src_size_hint == CONTENT_SIZE_UNKNOWN;
    let added_size = if unknown && dict_size > 0 { 500 } else { 0 };
    if unknown && dict_size == 0 {
        CONTENT_SIZE_UNKNOWN
    } else {
        src_size_hint
            .wrapping_add(dict_size)
            .wrapping_add(added_size)
    }
}

/// Level-table parameters for a source size hint and a dictionary of
/// `dict_size` bytes used as `mode` says (`ZSTD_getCParams_internal`).
#[inline(always)]
pub(crate) fn level_cparams_for(
    level: i32,
    src_size_hint: u64,
    dict_size: u64,
    mode: CParamMode,
) -> CParams {
    const KB: u64 = 1 << 10;
    let r_size = cparam_row_size(src_size_hint, dict_size, mode);
    let table_id = usize::from(r_size <= 256 * KB)
        + usize::from(r_size <= 128 * KB)
        + usize::from(r_size <= 16 * KB);
    let row = if level == 0 {
        DEFAULT_CLEVEL
    } else {
        level.clamp(0, MAX_CLEVEL)
    };
    let mut cparams = DEFAULT_CPARAMS[table_id][row as usize];
    if level < 0 {
        let clamped = level.max(MIN_CLEVEL);
        cparams.target_length = clamped.unsigned_abs();
    }
    adjust_cparams_for(cparams, src_size_hint, dict_size, mode, ParamSwitch::Auto)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_built_from_the_reference_rows() {
        // clevels.h, default table, level 3 and level 22.
        assert_eq!(DEFAULT_CPARAMS[0][3], cp(21, 16, 17, 1, 5, 0, DFast));
        assert_eq!(DEFAULT_CPARAMS[0][22], cp(27, 27, 25, 9, 3, 999, BtUltra2));
        assert_eq!(WINDOW_LOG_BOUNDS, Bounds::new(10, 31));
    }

    #[test]
    fn negative_levels_use_the_fast_base_row_with_acceleration() {
        let cparams = level_cparams(-5, CONTENT_SIZE_UNKNOWN);
        assert_eq!(cparams.strategy, Strategy::Fast);
        assert_eq!(cparams.target_length, 5);
        assert_eq!(
            level_cparams(i32::MIN, CONTENT_SIZE_UNKNOWN).target_length,
            TARGETLENGTH_MAX
        );
    }

    #[test]
    fn levels_above_the_maximum_use_the_last_row() {
        assert_eq!(
            level_cparams(99, CONTENT_SIZE_UNKNOWN),
            level_cparams(MAX_CLEVEL, CONTENT_SIZE_UNKNOWN)
        );
        assert_eq!(
            level_cparams(0, CONTENT_SIZE_UNKNOWN),
            level_cparams(DEFAULT_CLEVEL, CONTENT_SIZE_UNKNOWN)
        );
    }

    #[test]
    fn small_sources_shrink_the_window() {
        let cparams = level_cparams(3, 1000);
        assert_eq!(cparams.window_log, 10);
        assert!(cparams.hash_log <= 11);
    }

    #[test]
    fn clamping_and_validation_agree() {
        let wild = CParams {
            window_log: 99,
            chain_log: 0,
            hash_log: 1,
            search_log: 77,
            min_match: 1,
            target_length: u32::MAX,
            strategy: Strategy::Lazy,
        };
        assert!(!wild.is_valid());
        assert!(wild.clamped().is_valid());
    }

    #[test]
    fn strategies_round_trip_through_their_values() {
        for strategy in Strategy::ALL {
            assert_eq!(Strategy::from_value(strategy.value()), Some(strategy));
        }
        assert_eq!(Strategy::from_value(0), None);
    }

    #[test]
    fn automatic_switches_follow_the_reference_thresholds() {
        let lazy = level_cparams(6, CONTENT_SIZE_UNKNOWN);
        assert_eq!(
            resolve_row_match_finder(ParamSwitch::Auto, &lazy),
            ParamSwitch::Enable
        );
        let small_lazy = level_cparams(6, 1000);
        assert_eq!(
            resolve_row_match_finder(ParamSwitch::Auto, &small_lazy),
            ParamSwitch::Disable
        );
        let ultra = level_cparams(22, CONTENT_SIZE_UNKNOWN);
        assert_eq!(resolve_ldm(ParamSwitch::Auto, &ultra), ParamSwitch::Enable);
        assert_eq!(
            resolve_block_splitter(ParamSwitch::Auto, &ultra),
            ParamSwitch::Enable
        );
        assert_eq!(
            resolve_ldm(ParamSwitch::Disable, &ultra),
            ParamSwitch::Disable
        );
        assert_eq!(resolve_max_block_size(0), BLOCK_SIZE_MAX);
        assert_eq!(resolve_max_block_size(4096), 4096);
    }
}
