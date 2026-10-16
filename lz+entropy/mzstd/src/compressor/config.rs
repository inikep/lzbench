//! Encoder configuration: compression level, advanced parameters and frame
//! options.
//!
//! Every value here has a counterpart among the reference's `ZSTD_c_*`
//! parameters, and a configuration produces the same bytes as the reference
//! given the same values (see the crate documentation for the exact
//! contract).

use core::fmt;

use thiserror::Error;

use super::core::params::{
    CHAIN_LOG_BOUNDS, HASH_LOG_BOUNDS, LDM_BUCKET_SIZE_LOG_BOUNDS, LDM_HASH_LOG_BOUNDS,
    LDM_HASH_RATE_LOG_BOUNDS, LDM_MIN_MATCH_BOUNDS, MIN_MATCH_BOUNDS, SEARCH_LOG_BOUNDS,
    TARGET_LENGTH_BOUNDS, WINDOW_LOG_BOUNDS,
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
        #[doc = concat!("\n\nValid values: ", $range, ".")]
        #[must_use]
        pub const fn $with(mut self, value: u32) -> Self {
            self.$field = Some(value);
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
        /// Strategy-specific length target (`ZSTD_c_targetLength`); `0`
        /// keeps the level's value.
        target_length, with_target_length, "`0..=131072`";
    }

    /// Selects a strategy (`ZSTD_c_strategy`).
    #[must_use]
    pub const fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = Some(strategy);
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
    param_accessors! {
        /// Base-2 logarithm of the LDM table size (`ZSTD_c_ldmHashLog`).
        hash_log, with_hash_log, "`6..=30`";
        /// Shortest long match (`ZSTD_c_ldmMinMatch`).
        min_match, with_min_match, "`4..=4096`";
        /// Base-2 logarithm of the bucket size (`ZSTD_c_ldmBucketSizeLog`).
        bucket_size_log, with_bucket_size_log, "`1..=8`";
        /// Base-2 logarithm of the insertion rate (`ZSTD_c_ldmHashRateLog`).
        hash_rate_log, with_hash_rate_log, "`0..=25`";
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
    long_distance: LongDistanceMatching,
    size_hint: Option<u32>,
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
            long_distance: LongDistanceMatching::Auto,
            size_hint: None,
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
        self.long_distance = ldm;
        self
    }

    /// The long distance matcher setting.
    #[must_use]
    pub const fn long_distance_matching(&self) -> LongDistanceMatching {
        self.long_distance
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
        if let LongDistanceMatching::Enabled(ldm) = self.long_distance {
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
