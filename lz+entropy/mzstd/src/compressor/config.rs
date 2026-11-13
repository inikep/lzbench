//! Encoder configuration: compression level, advanced parameters and frame
//! options.
//!
//! Every value here has a counterpart among the reference's `ZSTD_c_*`
//! parameters, and a configuration produces the same bytes as the reference
//! given the same values (see the crate documentation for the exact
//! contract).

use core::fmt;
use core::num::NonZeroU32;

use thiserror::Error;

use crate::FrameFormat;

use super::core::params::{
    BLOCK_SPLITTER_LEVEL_MAX, CHAIN_LOG_BOUNDS, HASH_LOG_BOUNDS, LDM_BUCKET_SIZE_LOG_BOUNDS,
    LDM_HASH_LOG_BOUNDS, LDM_HASH_RATE_LOG_BOUNDS, LDM_MIN_MATCH_BOUNDS, MAX_BLOCK_SIZE_BOUNDS,
    MIN_MATCH_BOUNDS, SEARCH_LOG_BOUNDS, TARGET_C_BLOCK_SIZE_BOUNDS, TARGET_LENGTH_BOUNDS,
    WINDOW_LOG_BOUNDS,
};

/// A compression level: negative levels trade ratio for speed, `1..=22`
/// select the reference's tuned parameter sets.
///
/// Level `0` is the reference's alias for the default level and converts to
/// [`Level::DEFAULT`].
///
/// # Examples
///
/// ```
/// use mzstd::Level;
///
/// let level = Level::try_from(19)?;
/// assert_eq!(i32::from(level), 19);
/// assert_eq!(Level::try_from(0)?, Level::DEFAULT);
/// assert!(Level::try_from(23).is_err());
/// # Ok::<(), mzstd::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Level(i32);

impl Level {
    /// The fastest level, `-131072`.
    pub const MIN: Self = Self(super::core::params::MIN_CLEVEL);
    /// The strongest level, `22`.
    pub const MAX: Self = Self(super::core::params::MAX_CLEVEL);
    /// The reference's default level, `3`.
    pub const DEFAULT: Self = Self(super::core::params::DEFAULT_CLEVEL);

    /// The level `ZSTD_CCtx_setParameter` stores for `value`: clamped into
    /// range, with `0` as the default.
    pub(crate) const fn clamped(value: i32) -> Self {
        if value == 0 {
            Self::DEFAULT
        } else if value < Self::MIN.0 {
            Self::MIN
        } else if value > Self::MAX.0 {
            Self::MAX
        } else {
            Self(value)
        }
    }

    /// The level as the reference's integer.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(mzstd::Level::MAX.get(), 22);
    /// ```
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<i32> for Level {
    type Error = ConfigError;

    /// Validates a level.
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutOfRange`] outside `-131072..=22`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value == 0 {
            return Ok(Self::DEFAULT);
        }
        if !(Self::MIN.0..=Self::MAX.0).contains(&value) {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::Level,
                value: i64::from(value),
                min: i64::from(Self::MIN.0),
                max: i64::from(Self::MAX.0),
            });
        }
        Ok(Self(value))
    }
}

impl From<Level> for i32 {
    fn from(level: Level) -> Self {
        level.0
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Match finding strategy, from fastest to strongest (`ZSTD_strategy`).
///
/// # Examples
///
/// ```
/// use mzstd::Strategy;
///
/// assert_eq!(Strategy::try_from(9)?, Strategy::BtUltra2);
/// assert_eq!(u8::from(Strategy::Fast), 1);
/// assert!(Strategy::Lazy < Strategy::BtOpt);
/// # Ok::<(), mzstd::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum Strategy {
    /// One hash table, greedy.
    Fast,
    /// Two hash tables, greedy.
    DoubleFast,
    /// Hash chains, greedy.
    Greedy,
    /// Hash chains, one step of lazy evaluation.
    Lazy,
    /// Hash chains, two steps of lazy evaluation.
    Lazy2,
    /// Binary trees, two steps of lazy evaluation.
    BtLazy2,
    /// Optimal parsing over binary trees.
    BtOpt,
    /// Optimal parsing with finer prices.
    BtUltra,
    /// [`Strategy::BtUltra`] with a statistics-seeding first pass.
    BtUltra2,
}

impl Strategy {
    pub(crate) const fn from_core(core: super::core::params::Strategy) -> Self {
        use super::core::params::Strategy as Core;
        match core {
            Core::Fast => Self::Fast,
            Core::DFast => Self::DoubleFast,
            Core::Greedy => Self::Greedy,
            Core::Lazy => Self::Lazy,
            Core::Lazy2 => Self::Lazy2,
            Core::BtLazy2 => Self::BtLazy2,
            Core::BtOpt => Self::BtOpt,
            Core::BtUltra => Self::BtUltra,
            Core::BtUltra2 => Self::BtUltra2,
        }
    }

    pub(crate) const fn to_core(self) -> super::core::params::Strategy {
        use super::core::params::Strategy as Core;
        match self {
            Self::Fast => Core::Fast,
            Self::DoubleFast => Core::DFast,
            Self::Greedy => Core::Greedy,
            Self::Lazy => Core::Lazy,
            Self::Lazy2 => Core::Lazy2,
            Self::BtLazy2 => Core::BtLazy2,
            Self::BtOpt => Core::BtOpt,
            Self::BtUltra => Core::BtUltra,
            Self::BtUltra2 => Core::BtUltra2,
        }
    }
}

impl TryFrom<u8> for Strategy {
    type Error = ConfigError;

    /// The strategy with the reference's numeric value, `1..=9`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutOfRange`] for any other value.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Fast,
            2 => Self::DoubleFast,
            3 => Self::Greedy,
            4 => Self::Lazy,
            5 => Self::Lazy2,
            6 => Self::BtLazy2,
            7 => Self::BtOpt,
            8 => Self::BtUltra,
            9 => Self::BtUltra2,
            _ => {
                return Err(ConfigError::OutOfRange {
                    parameter: Parameter::Strategy,
                    value: i64::from(value),
                    min: 1,
                    max: 9,
                });
            }
        })
    }
}

impl From<Strategy> for u8 {
    fn from(strategy: Strategy) -> Self {
        strategy.to_core().value() as u8
    }
}

/// Explicit match finder parameters; each `None` keeps the level's value.
///
/// Values are validated when the compressor is built. The level's values are
/// still shrunk to fit a known input size afterwards, as the reference does.
///
/// # Examples
///
/// ```
/// use mzstd::{CompressionParams, Strategy};
///
/// let params = CompressionParams::default()
///     .with_window_log(20)
///     .with_strategy(Strategy::Lazy2);
/// assert_eq!(params.window_log(), Some(20));
/// assert_eq!(params.hash_log(), None);
/// // `0` restores the level's value, as in zstd.
/// assert_eq!(params.with_window_log(0).window_log(), None);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CompressionParams {
    window_log: Option<u32>,
    hash_log: Option<u32>,
    chain_log: Option<u32>,
    search_log: Option<u32>,
    min_match: Option<u32>,
    target_length: Option<u32>,
    strategy: Option<Strategy>,
}

macro_rules! param_accessors {
    ($($(#[$doc:meta])* $field:ident, $with:ident, $range:literal;)*) => {$(
        $(#[$doc])*
        #[doc = concat!(
            "\n\nValid values: ", $range, ". `0` restores the default, as it does for \
             `ZSTD_CCtx_setParameter`, so the last call wins and a loop over \
             caller parameters can apply each one directly."
        )]
        #[must_use]
        pub const fn $with(mut self, value: u32) -> Self {
            self.$field = if value == 0 { None } else { Some(value) };
            self
        }

        #[doc = concat!("The value set by [`Self::", stringify!($with), "`], if any.")]
        #[must_use]
        pub const fn $field(&self) -> Option<u32> {
            self.$field
        }
    )*};
}

impl CompressionParams {
    param_accessors! {
        /// Base-2 logarithm of the largest back-reference distance
        /// (`ZSTD_c_windowLog`).
        window_log, with_window_log, "`10..=31`";
        /// Base-2 logarithm of the main hash table size (`ZSTD_c_hashLog`).
        hash_log, with_hash_log, "`6..=30`";
        /// Base-2 logarithm of the chain or tree table size
        /// (`ZSTD_c_chainLog`).
        chain_log, with_chain_log, "`6..=30`";
        /// Base-2 logarithm of the number of searches (`ZSTD_c_searchLog`).
        search_log, with_search_log, "`1..=30`";
        /// Shortest match searched for (`ZSTD_c_minMatch`).
        min_match, with_min_match, "`3..=7`";
        /// Strategy-specific length target (`ZSTD_c_targetLength`).
        target_length, with_target_length, "`1..=131072`";
    }

    /// The parameters the reference's level table gives `level` for an input
    /// of `source_size` bytes and a dictionary of `dictionary_size` bytes
    /// (`ZSTD_getCParams`).
    ///
    /// A `source_size` of `0` means unknown, as in the reference. Every
    /// field is set except a target length of `0`, which is the same as
    /// the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{CompressionParams, Level, Strategy};
    ///
    /// let params = CompressionParams::for_level(Level::try_from(19)?, 0, 0);
    /// assert_eq!(params.strategy(), Some(Strategy::BtUltra2));
    /// assert_eq!(params.window_log(), Some(23));
    /// // A small input shrinks the tables.
    /// let small = CompressionParams::for_level(Level::try_from(19)?, 1000, 0);
    /// assert!(small.window_log() < params.window_log());
    /// # Ok::<(), mzstd::ConfigError>(())
    /// ```
    #[must_use]
    pub fn for_level(level: Level, source_size: u64, dictionary_size: usize) -> Self {
        use super::core::params::{CONTENT_SIZE_UNKNOWN, CParamMode, level_cparams_for};
        let hint = if source_size == 0 {
            CONTENT_SIZE_UNKNOWN
        } else {
            source_size
        };
        let core = level_cparams_for(
            level.get(),
            hint,
            dictionary_size as u64,
            CParamMode::Unknown,
        );
        Self::default()
            .with_window_log(core.window_log)
            .with_hash_log(core.hash_log)
            .with_chain_log(core.chain_log)
            .with_search_log(core.search_log)
            .with_min_match(core.min_match)
            .with_target_length(core.target_length)
            .with_strategy(Strategy::from_core(core.strategy))
    }

    /// Selects a strategy (`ZSTD_c_strategy`).
    #[must_use]
    pub const fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Sets or clears the strategy.
    pub(crate) const fn with_strategy_option(mut self, strategy: Option<Strategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// The strategy set by [`Self::with_strategy`], if any.
    #[must_use]
    pub const fn strategy(&self) -> Option<Strategy> {
        self.strategy
    }
}

/// Parameters of the long distance matcher; each `None` is derived from the
/// other parameters as the reference does.
///
/// # Examples
///
/// ```
/// use mzstd::LdmParams;
///
/// let ldm = LdmParams::default().with_min_match(64);
/// assert_eq!(ldm.min_match(), Some(64));
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct LdmParams {
    hash_log: Option<u32>,
    min_match: Option<u32>,
    bucket_size_log: Option<u32>,
    hash_rate_log: Option<u32>,
}

impl LdmParams {
    /// Every value derived, as [`Default`] has it.
    const NONE: Self = Self {
        hash_log: None,
        min_match: None,
        bucket_size_log: None,
        hash_rate_log: None,
    };

    param_accessors! {
        /// Base-2 logarithm of the LDM table size (`ZSTD_c_ldmHashLog`).
        hash_log, with_hash_log, "`6..=30`";
        /// Shortest long match (`ZSTD_c_ldmMinMatch`).
        min_match, with_min_match, "`4..=4096`";
        /// Base-2 logarithm of the bucket size (`ZSTD_c_ldmBucketSizeLog`).
        bucket_size_log, with_bucket_size_log, "`1..=8`";
        /// Base-2 logarithm of the insertion rate (`ZSTD_c_ldmHashRateLog`).
        hash_rate_log, with_hash_rate_log, "`1..=25`";
    }
}

/// Whether the long distance matcher runs (`ZSTD_c_enableLongDistanceMatching`).
///
/// # Examples
///
/// ```
/// use mzstd::{LdmParams, LongDistanceMatching};
///
/// assert_eq!(LongDistanceMatching::default(), LongDistanceMatching::Auto);
/// let _ = LongDistanceMatching::Enabled(LdmParams::default());
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum LongDistanceMatching {
    /// On for `btopt` and stronger with a window of at least 2^27 bytes.
    #[default]
    Auto,
    /// Off.
    Disabled,
    /// On, with these parameters. Without an explicit window log, the window
    /// grows to 2^27 bytes.
    Enabled(LdmParams),
}

/// A three-way parameter switch (`ZSTD_ParamSwitch_e`): the reference decides
/// ([`Switch::Auto`]), or the caller forces the feature on or off.
///
/// # Examples
///
/// ```
/// use mzstd::Switch;
///
/// assert_eq!(Switch::default(), Switch::Auto);
/// assert_eq!(i32::from(Switch::Disabled), 2);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Switch {
    /// Decided from the other parameters (`ZSTD_ps_auto`).
    #[default]
    Auto,
    /// Forced on (`ZSTD_ps_enable`).
    Enabled,
    /// Forced off (`ZSTD_ps_disable`).
    Disabled,
}

impl Switch {
    pub(crate) const fn to_core(self) -> super::core::params::ParamSwitch {
        use super::core::params::ParamSwitch as Core;
        match self {
            Self::Auto => Core::Auto,
            Self::Enabled => Core::Enable,
            Self::Disabled => Core::Disable,
        }
    }
}

impl From<Switch> for i32 {
    fn from(switch: Switch) -> Self {
        match switch {
            Switch::Auto => 0,
            Switch::Enabled => 1,
            Switch::Disabled => 2,
        }
    }
}

/// How a dictionary prepared with [`DictionaryBuilder`](crate::dictionary::DictionaryBuilder)
/// reaches a frame (`ZSTD_c_forceAttachDict`, `ZSTD_dictAttachPref_e`).
///
/// The choice never changes the decoded data; it changes the compressed
/// bytes and the speed.
///
/// # Examples
///
/// ```
/// use mzstd::DictionaryAttach;
///
/// assert_eq!(DictionaryAttach::default(), DictionaryAttach::Default);
/// assert_eq!(i32::from(DictionaryAttach::Load), 3);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum DictionaryAttach {
    /// Attach for inputs up to a size cutoff that depends on the
    /// dictionary's strategy, copy beyond it (`ZSTD_dictDefaultAttach`).
    #[default]
    Default,
    /// Always search the dictionary's own tables in place
    /// (`ZSTD_dictForceAttach`).
    Attach,
    /// Always copy the dictionary's tables into the context
    /// (`ZSTD_dictForceCopy`).
    Copy,
    /// Index the dictionary's bytes again with the compressor's own
    /// parameters (`ZSTD_dictForceLoad`).
    Load,
}

impl DictionaryAttach {
    pub(crate) const fn to_core(self) -> super::core::cdict::AttachPref {
        use super::core::cdict::AttachPref as Core;
        match self {
            Self::Default => Core::Default,
            Self::Attach => Core::ForceAttach,
            Self::Copy => Core::ForceCopy,
            Self::Load => Core::ForceLoad,
        }
    }
}

impl From<DictionaryAttach> for i32 {
    fn from(attach: DictionaryAttach) -> Self {
        match attach {
            DictionaryAttach::Default => 0,
            DictionaryAttach::Attach => 1,
            DictionaryAttach::Copy => 2,
            DictionaryAttach::Load => 3,
        }
    }
}

/// Everything that configures a [`Compressor`](super::Compressor).
///
/// # Examples
///
/// ```
/// use mzstd::{CompressionParams, EncoderConfig, Level};
///
/// let config = EncoderConfig::default()
///     .with_level(Level::try_from(9)?)
///     .with_checksum(true)
///     .with_params(CompressionParams::default().with_window_log(23));
/// assert!(config.checksum());
/// assert_eq!(config.level().get(), 9);
/// # Ok::<(), mzstd::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct EncoderConfig {
    level: Level,
    params: CompressionParams,
    checksum: bool,
    content_size: bool,
    ldm_switch: Switch,
    ldm: LdmParams,
    size_hint: Option<u32>,
    format: FrameFormat,
    dictionary_id: bool,
    literal_compression: Switch,
    dictionary_attach: DictionaryAttach,
    force_max_window: bool,
    row_match_finder: Switch,
    post_block_splitter: Switch,
    block_splitter_level: u8,
    max_block_size: Option<u32>,
    target_compressed_block_size: Option<NonZeroU32>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderConfig {
    /// The reference's defaults: level 3, content size on, checksum off.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(mzstd::EncoderConfig::new(), mzstd::EncoderConfig::default());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            level: Level::DEFAULT,
            params: CompressionParams {
                window_log: None,
                hash_log: None,
                chain_log: None,
                search_log: None,
                min_match: None,
                target_length: None,
                strategy: None,
            },
            checksum: false,
            content_size: true,
            ldm_switch: Switch::Auto,
            ldm: LdmParams::NONE,
            size_hint: None,
            format: FrameFormat::Standard,
            dictionary_id: true,
            literal_compression: Switch::Auto,
            dictionary_attach: DictionaryAttach::Default,
            force_max_window: false,
            row_match_finder: Switch::Auto,
            post_block_splitter: Switch::Auto,
            block_splitter_level: 0,
            max_block_size: None,
            target_compressed_block_size: None,
        }
    }

    /// Sets the compression level.
    #[must_use]
    pub const fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// The compression level.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// Sets explicit match finder parameters.
    #[must_use]
    pub const fn with_params(mut self, params: CompressionParams) -> Self {
        self.params = params;
        self
    }

    /// The explicit match finder parameters.
    #[must_use]
    pub const fn params(&self) -> CompressionParams {
        self.params
    }

    /// Appends an XXH64 content checksum to every frame (`ZSTD_c_checksumFlag`).
    #[must_use]
    pub const fn with_checksum(mut self, checksum: bool) -> Self {
        self.checksum = checksum;
        self
    }

    /// Whether frames carry a content checksum.
    #[must_use]
    pub const fn checksum(&self) -> bool {
        self.checksum
    }

    /// Records the content size in frame headers when it is known
    /// (`ZSTD_c_contentSizeFlag`).
    #[must_use]
    pub const fn with_content_size(mut self, content_size: bool) -> Self {
        self.content_size = content_size;
        self
    }

    /// Whether frame headers record a known content size.
    #[must_use]
    pub const fn content_size(&self) -> bool {
        self.content_size
    }

    /// Configures the long distance matcher.
    #[must_use]
    pub const fn with_long_distance_matching(mut self, ldm: LongDistanceMatching) -> Self {
        match ldm {
            LongDistanceMatching::Auto => {
                self.ldm_switch = Switch::Auto;
                self.ldm = LdmParams::NONE;
            }
            LongDistanceMatching::Disabled => {
                self.ldm_switch = Switch::Disabled;
                self.ldm = LdmParams::NONE;
            }
            LongDistanceMatching::Enabled(params) => {
                self.ldm_switch = Switch::Enabled;
                self.ldm = params;
            }
        }
        self
    }

    /// Replaces the match finder parameters with `f(params)`.
    pub(crate) fn map_params(
        mut self,
        f: impl FnOnce(CompressionParams) -> CompressionParams,
    ) -> Self {
        self.params = f(self.params);
        self
    }

    /// Replaces the long distance matcher's values with `f(values)`, keeping
    /// its switch.
    pub(crate) fn map_ldm(mut self, f: impl FnOnce(LdmParams) -> LdmParams) -> Self {
        self.ldm = f(self.ldm);
        self
    }

    /// Switches the long distance matcher, keeping its explicit values.
    pub(crate) const fn with_ldm_switch(mut self, switch: Switch) -> Self {
        self.ldm_switch = switch;
        self
    }

    /// The long distance matcher's explicit values, whichever way it is
    /// switched.
    pub(crate) const fn ldm_params(&self) -> LdmParams {
        self.ldm
    }

    /// The long distance matcher's switch.
    pub(crate) const fn ldm_switch(&self) -> Switch {
        self.ldm_switch
    }

    /// The long distance matcher setting.
    #[must_use]
    pub const fn long_distance_matching(&self) -> LongDistanceMatching {
        match self.ldm_switch {
            Switch::Auto => LongDistanceMatching::Auto,
            Switch::Disabled => LongDistanceMatching::Disabled,
            Switch::Enabled => LongDistanceMatching::Enabled(self.ldm),
        }
    }

    /// Expected input size of streams whose size is not declared
    /// (`ZSTD_c_srcSizeHint`); selects parameters as if the size were known.
    #[must_use]
    pub const fn with_size_hint(mut self, size_hint: Option<u32>) -> Self {
        self.size_hint = size_hint;
        self
    }

    /// The size hint, if any.
    #[must_use]
    pub const fn size_hint(&self) -> Option<u32> {
        self.size_hint
    }

    /// Selects the frame format (`ZSTD_c_format`); see [`FrameFormat`].
    ///
    /// Magicless frames are written by [`Compressor`](super::Compressor)
    /// only; the parallel and framed compressors reject them.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Compressor, EncoderConfig, FrameFormat};
    ///
    /// let config = EncoderConfig::default().with_format(FrameFormat::Magicless);
    /// let frame = Compressor::new(config)?.compress(b"payload")?;
    /// // The standard frame without its four magic bytes.
    /// let standard = Compressor::default().compress(b"payload")?;
    /// assert_eq!(frame, standard[4..]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_format(mut self, format: FrameFormat) -> Self {
        self.format = format;
        self
    }

    /// The frame format.
    #[must_use]
    pub const fn format(&self) -> FrameFormat {
        self.format
    }

    /// Writes the dictionary ID into frame headers when a dictionary has one
    /// (`ZSTD_c_dictIDFlag`, default on).
    #[must_use]
    pub const fn with_dictionary_id(mut self, dictionary_id: bool) -> Self {
        self.dictionary_id = dictionary_id;
        self
    }

    /// Whether frame headers name the dictionary's ID.
    #[must_use]
    pub const fn dictionary_id(&self) -> bool {
        self.dictionary_id
    }

    /// Whether literals are entropy coded (`ZSTD_c_literalCompressionMode`):
    /// [`Switch::Disabled`] stores them raw, which decompresses faster and
    /// compresses worse. [`Switch::Auto`] decides from the strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Compressor, EncoderConfig, Switch};
    ///
    /// let payload: String = (0..200).map(|i| format!("item {i} holds {} ", i * 7)).collect();
    /// let raw = EncoderConfig::default().with_literal_compression(Switch::Disabled);
    /// let coded = Compressor::default().compress(payload.as_bytes())?;
    /// let stored = Compressor::new(raw)?.compress(payload.as_bytes())?;
    /// assert_ne!(coded, stored);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_literal_compression(mut self, literal_compression: Switch) -> Self {
        self.literal_compression = literal_compression;
        self
    }

    /// The literal compression switch.
    #[must_use]
    pub const fn literal_compression(&self) -> Switch {
        self.literal_compression
    }

    /// How a dictionary reaches a frame (`ZSTD_c_forceAttachDict`).
    #[must_use]
    pub const fn with_dictionary_attach(mut self, attach: DictionaryAttach) -> Self {
        self.dictionary_attach = attach;
        self
    }

    /// How a dictionary reaches a frame.
    #[must_use]
    pub const fn dictionary_attach(&self) -> DictionaryAttach {
        self.dictionary_attach
    }

    /// Keeps dictionary content reachable only inside the window
    /// (`ZSTD_c_forceMaxWindow`); it also turns attaching a dictionary off.
    #[must_use]
    pub const fn with_force_max_window(mut self, force: bool) -> Self {
        self.force_max_window = force;
        self
    }

    /// Whether the window is enforced on dictionary content.
    #[must_use]
    pub const fn force_max_window(&self) -> bool {
        self.force_max_window
    }

    /// Forces the row-based match finder on or off for the lazy strategies
    /// (`ZSTD_c_useRowMatchFinder`).
    #[must_use]
    pub const fn with_row_match_finder(mut self, row: Switch) -> Self {
        self.row_match_finder = row;
        self
    }

    /// The row match finder switch.
    #[must_use]
    pub const fn row_match_finder(&self) -> Switch {
        self.row_match_finder
    }

    /// Splits blocks after sequence generation (`ZSTD_c_splitAfterSequences`).
    #[must_use]
    pub const fn with_post_block_splitter(mut self, split: Switch) -> Self {
        self.post_block_splitter = split;
        self
    }

    /// The post-sequence block splitter switch.
    #[must_use]
    pub const fn post_block_splitter(&self) -> Switch {
        self.post_block_splitter
    }

    /// Effort of the splitter that cuts blocks before matching
    /// (`ZSTD_c_blockSplitterLevel`): `0` automatic, `1` never, `2..=6`
    /// increasing effort.
    #[must_use]
    pub const fn with_block_splitter_level(mut self, level: u8) -> Self {
        self.block_splitter_level = level;
        self
    }

    /// The pre-matching block splitter level.
    #[must_use]
    pub const fn block_splitter_level(&self) -> u8 {
        self.block_splitter_level
    }

    /// Caps the uncompressed size of a block (`ZSTD_c_maxBlockSize`);
    /// `None` keeps the format's 128 KiB. Valid sizes are `1024..=131072`.
    #[must_use]
    pub const fn with_max_block_size(mut self, size: Option<u32>) -> Self {
        self.max_block_size = size;
        self
    }

    /// The block size cap, if any.
    #[must_use]
    pub const fn max_block_size(&self) -> Option<u32> {
        self.max_block_size
    }

    /// Asks for compressed blocks of about this many bytes
    /// (`ZSTD_c_targetCBlockSize`); `None` (the default) sets no target.
    ///
    /// With a target, each block of up to 128 KiB (or
    /// [`with_max_block_size`](Self::with_max_block_size)) is cut into
    /// sub-blocks whose compressed size is close to `size`, so a receiver can
    /// start decoding before the whole block has arrived. The first
    /// sub-block carries the block's entropy tables and the others reuse
    /// them, which costs compression ratio: the smaller the target, the
    /// larger the frame. The output stays a standard Zstandard frame, byte
    /// for byte the reference's. The target takes the place of the post-match
    /// block splitter ([`with_post_block_splitter`](Self::with_post_block_splitter)),
    /// which then has no effect.
    ///
    /// A target below `1340` (`ZSTD_TARGETCBLOCKSIZE_MIN`, the payload of an
    /// ethernet, Wi-Fi or 4G frame) is raised to it, as
    /// `ZSTD_CCtx_setParameter` does. A target above `131072`
    /// (`ZSTD_TARGETCBLOCKSIZE_MAX`) is refused when the compressor is
    /// built.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU32;
    /// use mzstd::{Compressor, EncoderConfig};
    ///
    /// let text = b"the quick brown fox jumps over the lazy dog. ".repeat(4000);
    /// let config = EncoderConfig::default()
    ///     .with_target_compressed_block_size(NonZeroU32::new(1340));
    /// assert_eq!(
    ///     config.target_compressed_block_size(),
    ///     NonZeroU32::new(1340)
    /// );
    /// let frame = Compressor::new(config)?.compress(&text)?;
    /// // The frame is a standard one: it opens with the Zstandard magic number.
    /// assert_eq!(frame[..4], [0x28, 0xB5, 0x2F, 0xFD]);
    /// // Larger than without a target, as every sub-block but the first
    /// // reuses the first one's tables.
    /// let plain = Compressor::default().compress(&text)?;
    /// assert!(frame.len() >= plain.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn with_target_compressed_block_size(mut self, size: Option<NonZeroU32>) -> Self {
        self.target_compressed_block_size = size;
        self
    }

    /// The target compressed block size, if any, as it was set (before
    /// the minimum is applied).
    #[must_use]
    pub const fn target_compressed_block_size(&self) -> Option<NonZeroU32> {
        self.target_compressed_block_size
    }

    /// Rejects the magicless format, for containers that write the magic
    /// number themselves.
    #[cfg(any(feature = "framed", not(feature = "no_std")))]
    pub(crate) fn validate_standard_frames(&self) -> Result<(), ConfigError> {
        match self.format {
            FrameFormat::Standard => Ok(()),
            FrameFormat::Magicless => Err(ConfigError::Unsupported {
                id: 10,
                value: i32::from(FrameFormat::Magicless),
            }),
        }
    }

    /// Checks every explicit value against its range.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        let p = &self.params;
        check(Parameter::WindowLog, p.window_log, WINDOW_LOG_BOUNDS)?;
        check(Parameter::HashLog, p.hash_log, HASH_LOG_BOUNDS)?;
        check(Parameter::ChainLog, p.chain_log, CHAIN_LOG_BOUNDS)?;
        check(Parameter::SearchLog, p.search_log, SEARCH_LOG_BOUNDS)?;
        check(Parameter::MinMatch, p.min_match, MIN_MATCH_BOUNDS)?;
        check(
            Parameter::TargetLength,
            p.target_length,
            TARGET_LENGTH_BOUNDS,
        )?;
        let ldm = self.ldm;
        check(Parameter::LdmHashLog, ldm.hash_log, LDM_HASH_LOG_BOUNDS)?;
        check(Parameter::LdmMinMatch, ldm.min_match, LDM_MIN_MATCH_BOUNDS)?;
        check(
            Parameter::LdmBucketSizeLog,
            ldm.bucket_size_log,
            LDM_BUCKET_SIZE_LOG_BOUNDS,
        )?;
        check(
            Parameter::LdmHashRateLog,
            ldm.hash_rate_log,
            LDM_HASH_RATE_LOG_BOUNDS,
        )?;
        if self.block_splitter_level > BLOCK_SPLITTER_LEVEL_MAX {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::BlockSplitterLevel,
                value: i64::from(self.block_splitter_level),
                min: 0,
                max: i64::from(BLOCK_SPLITTER_LEVEL_MAX),
            });
        }
        if let Some(size) = self.max_block_size
            && !MAX_BLOCK_SIZE_BOUNDS.contains(i64::from(size))
        {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::MaxBlockSize,
                value: i64::from(size),
                min: MAX_BLOCK_SIZE_BOUNDS.lower,
                max: MAX_BLOCK_SIZE_BOUNDS.upper,
            });
        }
        if let Some(size) = self.target_compressed_block_size
            && i64::from(size.get()) > TARGET_C_BLOCK_SIZE_BOUNDS.upper
        {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::TargetCompressedBlockSize,
                value: i64::from(size.get()),
                min: TARGET_C_BLOCK_SIZE_BOUNDS.lower,
                max: TARGET_C_BLOCK_SIZE_BOUNDS.upper,
            });
        }
        if let Some(hint) = self.size_hint
            && hint > i32::MAX as u32
        {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::SizeHint,
                value: i64::from(hint),
                min: 0,
                max: i64::from(i32::MAX),
            });
        }
        Ok(())
    }
}

fn check(
    parameter: Parameter,
    value: Option<u32>,
    bounds: super::core::params::Bounds,
) -> Result<(), ConfigError> {
    match value {
        Some(value) if !bounds.contains(i64::from(value)) => Err(ConfigError::OutOfRange {
            parameter,
            value: i64::from(value),
            min: bounds.lower,
            max: bounds.upper,
        }),
        _ => Ok(()),
    }
}

/// A configurable parameter, named in [`ConfigError`].
///
/// # Examples
///
/// ```
/// assert_eq!(mzstd::Parameter::WindowLog.to_string(), "window log");
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Parameter {
    /// [`Level`].
    Level,
    /// [`Strategy`].
    Strategy,
    /// [`CompressionParams::with_window_log`].
    WindowLog,
    /// [`CompressionParams::with_hash_log`].
    HashLog,
    /// [`CompressionParams::with_chain_log`].
    ChainLog,
    /// [`CompressionParams::with_search_log`].
    SearchLog,
    /// [`CompressionParams::with_min_match`].
    MinMatch,
    /// [`CompressionParams::with_target_length`].
    TargetLength,
    /// [`LdmParams::with_hash_log`].
    LdmHashLog,
    /// [`LdmParams::with_min_match`].
    LdmMinMatch,
    /// [`LdmParams::with_bucket_size_log`].
    LdmBucketSizeLog,
    /// [`LdmParams::with_hash_rate_log`].
    LdmHashRateLog,
    /// [`EncoderConfig::with_size_hint`].
    SizeHint,
    /// The parallel segment size.
    SegmentSize,
    /// The parallel task count.
    TaskCount,
    /// [`EncoderConfig::with_format`] (`ZSTD_c_format`).
    Format,
    /// [`EncoderConfig::with_dictionary_attach`] (`ZSTD_c_forceAttachDict`).
    DictionaryAttach,
    /// [`EncoderConfig::with_literal_compression`]
    /// (`ZSTD_c_literalCompressionMode`).
    LiteralCompression,
    /// The long distance matching switch (`ZSTD_c_enableLongDistanceMatching`).
    LongDistanceMatching,
    /// [`EncoderConfig::with_post_block_splitter`] (`ZSTD_c_splitAfterSequences`).
    PostBlockSplitter,
    /// [`EncoderConfig::with_block_splitter_level`] (`ZSTD_c_blockSplitterLevel`).
    BlockSplitterLevel,
    /// [`EncoderConfig::with_row_match_finder`] (`ZSTD_c_useRowMatchFinder`).
    RowMatchFinder,
    /// [`EncoderConfig::with_max_block_size`] (`ZSTD_c_maxBlockSize`).
    MaxBlockSize,
    /// [`EncoderConfig::with_target_compressed_block_size`]
    /// (`ZSTD_c_targetCBlockSize`).
    TargetCompressedBlockSize,
    /// The reference parameter with this numeric ID, which has no counterpart
    /// of its own here (see [`CParameter::Ignored`](super::CParameter::Ignored)).
    Other(i32),
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Level => "compression level",
            Self::Strategy => "strategy",
            Self::WindowLog => "window log",
            Self::HashLog => "hash log",
            Self::ChainLog => "chain log",
            Self::SearchLog => "search log",
            Self::MinMatch => "minimum match",
            Self::TargetLength => "target length",
            Self::LdmHashLog => "LDM hash log",
            Self::LdmMinMatch => "LDM minimum match",
            Self::LdmBucketSizeLog => "LDM bucket size log",
            Self::LdmHashRateLog => "LDM hash rate log",
            Self::SizeHint => "size hint",
            Self::SegmentSize => "segment size",
            Self::TaskCount => "task count",
            Self::Format => "frame format",
            Self::DictionaryAttach => "dictionary attach preference",
            Self::LiteralCompression => "literal compression mode",
            Self::LongDistanceMatching => "long distance matching switch",
            Self::PostBlockSplitter => "post-sequence block splitter",
            Self::BlockSplitterLevel => "block splitter level",
            Self::RowMatchFinder => "row match finder switch",
            Self::MaxBlockSize => "maximum block size",
            Self::TargetCompressedBlockSize => "target compressed block size",
            Self::Other(id) => return write!(f, "zstd parameter {id}"),
        })
    }
}

/// A configuration value outside its valid range.
///
/// # Examples
///
/// ```
/// use mzstd::{CompressionParams, Compressor, ConfigError, EncoderConfig, Parameter};
///
/// let config = EncoderConfig::default().with_params(CompressionParams::default().with_window_log(40));
/// match Compressor::new(config) {
///     Err(ConfigError::OutOfRange { parameter, .. }) => assert_eq!(parameter, Parameter::WindowLog),
///     other => panic!("{other:?}"),
/// }
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A reference parameter ID that does not exist.
    #[error("unknown zstd parameter {id}")]
    UnknownParameter {
        /// The rejected numeric `ZSTD_cParameter`.
        id: i32,
    },
    /// A reference parameter this crate does not implement with this value:
    /// multithreading, or a value the single-threaded reference itself
    /// rejects.
    #[error("zstd parameter {id} does not support the value {value}")]
    Unsupported {
        /// The numeric `ZSTD_cParameter`.
        id: i32,
        /// The rejected value.
        value: i32,
    },
    /// A parameter is outside its valid range.
    #[error("{parameter} {value} is outside {min}..={max}")]
    OutOfRange {
        /// The parameter.
        parameter: Parameter,
        /// The rejected value.
        value: i64,
        /// Smallest valid value.
        min: i64,
        /// Largest valid value.
        max: i64,
    },
}

/// The compressed-size bound does not fit in a `usize`.
///
/// # Examples
///
/// ```
/// assert!(mzstd::Compressor::max_compressed_size(usize::MAX).is_err());
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Error)]
#[error("the compressed-size bound does not fit in usize")]
pub struct SizeOverflow;
