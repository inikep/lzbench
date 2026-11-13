//! Reference compression parameters by numeric ID.
//!
//! Containers such as OpenZL store `ZSTD_cParameter` IDs and values and
//! apply them, in order, with `ZSTD_CCtx_setParameter`. [`CParameter`] is the
//! checked, typed form of one such pair: building it validates the ID and
//! the value the way the reference does, and
//! [`EncoderConfig::with_parameter`] applies it with the reference's
//! "last call wins" semantics.

use core::num::NonZeroU32;

use super::config::{
    ConfigError, DictionaryAttach, EncoderConfig, Level, Parameter, Strategy, Switch,
};
use super::core::params::{
    BLOCK_SPLITTER_LEVEL_MAX, Bounds, CHAIN_LOG_BOUNDS, HASH_LOG_BOUNDS,
    LDM_BUCKET_SIZE_LOG_BOUNDS, LDM_HASH_LOG_BOUNDS, LDM_HASH_RATE_LOG_BOUNDS,
    LDM_MIN_MATCH_BOUNDS, MAX_BLOCK_SIZE_BOUNDS, MIN_MATCH_BOUNDS, SEARCH_LOG_BOUNDS,
    TARGET_C_BLOCK_SIZE_BOUNDS, TARGET_LENGTH_BOUNDS, WINDOW_LOG_BOUNDS,
};
use crate::FrameFormat;

/// The numeric IDs of the reference's `ZSTD_cParameter` values.
mod id {
    pub(super) const FORMAT: i32 = 10;
    pub(super) const COMPRESSION_LEVEL: i32 = 100;
    pub(super) const WINDOW_LOG: i32 = 101;
    pub(super) const HASH_LOG: i32 = 102;
    pub(super) const CHAIN_LOG: i32 = 103;
    pub(super) const SEARCH_LOG: i32 = 104;
    pub(super) const MIN_MATCH: i32 = 105;
    pub(super) const TARGET_LENGTH: i32 = 106;
    pub(super) const STRATEGY: i32 = 107;
    pub(super) const TARGET_C_BLOCK_SIZE: i32 = 130;
    pub(super) const ENABLE_LDM: i32 = 160;
    pub(super) const LDM_HASH_LOG: i32 = 161;
    pub(super) const LDM_MIN_MATCH: i32 = 162;
    pub(super) const LDM_BUCKET_SIZE_LOG: i32 = 163;
    pub(super) const LDM_HASH_RATE_LOG: i32 = 164;
    pub(super) const CONTENT_SIZE_FLAG: i32 = 200;
    pub(super) const CHECKSUM_FLAG: i32 = 201;
    pub(super) const DICT_ID_FLAG: i32 = 202;
    pub(super) const NB_WORKERS: i32 = 400;
    pub(super) const JOB_SIZE: i32 = 401;
    pub(super) const OVERLAP_LOG: i32 = 402;
    pub(super) const RSYNCABLE: i32 = 500;
    pub(super) const FORCE_MAX_WINDOW: i32 = 1000;
    pub(super) const FORCE_ATTACH_DICT: i32 = 1001;
    pub(super) const LITERAL_COMPRESSION_MODE: i32 = 1002;
    pub(super) const SRC_SIZE_HINT: i32 = 1004;
    pub(super) const ENABLE_DEDICATED_DICT_SEARCH: i32 = 1005;
    pub(super) const STABLE_IN_BUFFER: i32 = 1006;
    pub(super) const STABLE_OUT_BUFFER: i32 = 1007;
    pub(super) const BLOCK_DELIMITERS: i32 = 1008;
    pub(super) const VALIDATE_SEQUENCES: i32 = 1009;
    pub(super) const SPLIT_AFTER_SEQUENCES: i32 = 1010;
    pub(super) const USE_ROW_MATCH_FINDER: i32 = 1011;
    pub(super) const DETERMINISTIC_REF_PREFIX: i32 = 1012;
    pub(super) const PREFETCH_CDICT_TABLES: i32 = 1013;
    pub(super) const ENABLE_SEQ_PRODUCER_FALLBACK: i32 = 1014;
    pub(super) const MAX_BLOCK_SIZE: i32 = 1015;
    pub(super) const REPCODE_RESOLUTION: i32 = 1016;
    pub(super) const BLOCK_SPLITTER_LEVEL: i32 = 1017;
}

/// One reference compression parameter with a validated value
/// (`ZSTD_cParameter` and its value).
///
/// Build it from the reference's numeric pair with
/// [`TryFrom<(i32, i32)>`](#impl-TryFrom%3C(i32,+i32)%3E-for-CParameter), which
/// accepts and rejects exactly what `ZSTD_CCtx_setParameter` does for the
/// single-threaded reference, and apply it with
/// [`EncoderConfig::with_parameter`]. Applying a list in order gives the
/// configuration the reference's context has after the same
/// `ZSTD_CCtx_setParameter` calls, so later parameters override earlier ones.
///
/// Values of `0` restore a parameter's default where the reference does so
/// (window, hash, chain and search logs, minimum match, strategy, the long
/// distance matcher's logs and sizes, source size hint, block size cap and
/// target compressed block size).
///
/// # Examples
///
/// ```
/// use mzstd::{CParameter, Compressor, EncoderConfig, FrameFormat, Level};
///
/// // `ZSTD_c_compressionLevel` is 100, `ZSTD_c_format` is 10 and
/// // `ZSTD_c_windowLog` is 101.
/// let mut config = EncoderConfig::default();
/// for pair in [(100, 9), (10, 1), (101, 20)] {
///     config = config.with_parameter(CParameter::try_from(pair)?);
/// }
/// assert_eq!(config.level(), Level::try_from(9)?);
/// assert_eq!(config.format(), FrameFormat::Magicless);
/// assert_eq!(config.params().window_log(), Some(20));
///
/// // A window log outside `10..=31` is refused, as the reference refuses it.
/// assert!(CParameter::try_from((101, 40)).is_err());
/// // So is an ID the reference does not have.
/// assert!(CParameter::try_from((12345, 0)).is_err());
/// # let _ = Compressor::new(config)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CParameter {
    /// `ZSTD_c_format` (10).
    Format(FrameFormat),
    /// `ZSTD_c_compressionLevel` (100). The reference clamps a level into
    /// `-131072..=22` and reads `0` as the default level.
    CompressionLevel(Level),
    /// `ZSTD_c_windowLog` (101); `0` for the level's value, else `10..=31`.
    WindowLog(u32),
    /// `ZSTD_c_hashLog` (102); `0` for the level's value, else `6..=30`.
    HashLog(u32),
    /// `ZSTD_c_chainLog` (103); `0` for the level's value, else `6..=30`.
    ChainLog(u32),
    /// `ZSTD_c_searchLog` (104); `0` for the level's value, else `1..=30`.
    SearchLog(u32),
    /// `ZSTD_c_minMatch` (105); `0` for the level's value, else `3..=7`.
    MinMatch(u32),
    /// `ZSTD_c_targetLength` (106); `0` for the level's value, else
    /// `1..=131072`.
    TargetLength(u32),
    /// `ZSTD_c_strategy` (107); `None` (`0`) for the level's value.
    Strategy(Option<Strategy>),
    /// `ZSTD_c_targetCBlockSize` (130); `None` (`0`) for no target, else
    /// `1340..=131072`. The reference raises a smaller or negative value to
    /// `1340`, and so does building this parameter.
    TargetCBlockSize(Option<NonZeroU32>),
    /// `ZSTD_c_enableLongDistanceMatching` (160).
    EnableLongDistanceMatching(Switch),
    /// `ZSTD_c_ldmHashLog` (161); `0` derives it, else `6..=30`.
    LdmHashLog(u32),
    /// `ZSTD_c_ldmMinMatch` (162); `0` derives it, else `4..=4096`.
    LdmMinMatch(u32),
    /// `ZSTD_c_ldmBucketSizeLog` (163); `0` derives it, else `1..=8`.
    LdmBucketSizeLog(u32),
    /// `ZSTD_c_ldmHashRateLog` (164); `0` derives it, else `1..=25`.
    LdmHashRateLog(u32),
    /// `ZSTD_c_contentSizeFlag` (200): any non-zero value is on.
    ContentSizeFlag(bool),
    /// `ZSTD_c_checksumFlag` (201): any non-zero value is on.
    ChecksumFlag(bool),
    /// `ZSTD_c_dictIDFlag` (202): any non-zero value is on.
    DictIdFlag(bool),
    /// `ZSTD_c_forceMaxWindow` (1000): any non-zero value is on.
    ForceMaxWindow(bool),
    /// `ZSTD_c_forceAttachDict` (1001).
    ForceAttachDict(DictionaryAttach),
    /// `ZSTD_c_literalCompressionMode` (1002).
    LiteralCompressionMode(Switch),
    /// `ZSTD_c_srcSizeHint` (1004); `None` (`0`) for no hint.
    SrcSizeHint(Option<u32>),
    /// `ZSTD_c_splitAfterSequences` (1010).
    SplitAfterSequences(Switch),
    /// `ZSTD_c_useRowMatchFinder` (1011).
    UseRowMatchFinder(Switch),
    /// `ZSTD_c_maxBlockSize` (1015); `0` for the format's 128 KiB, else
    /// `1024..=131072`.
    MaxBlockSize(u32),
    /// `ZSTD_c_blockSplitterLevel` (1017), `0..=6`.
    BlockSplitterLevel(u8),
    /// A parameter that cannot change the bytes of a single-threaded frame
    /// and is validated and then dropped: `nbWorkers`, `jobSize`,
    /// `overlapLog` and `rsyncable` at `0`; `enableDedicatedDictSearch`;
    /// `stableInBuffer`, `stableOutBuffer`, `blockDelimiters`,
    /// `validateSequences`, `deterministicRefPrefix`, `prefetchCDictTables`,
    /// `enableSeqProducerFallback` and `repcodeResolution`.
    Ignored {
        /// The numeric `ZSTD_cParameter`.
        id: i32,
        /// Its value.
        value: i32,
    },
}

/// A value of `id` checked against `bounds`.
fn bounded(parameter: Parameter, value: i32, bounds: Bounds) -> Result<u32, ConfigError> {
    if bounds.contains(i64::from(value)) {
        // The bounds are non-negative and fit `u32`.
        Ok(value as u32)
    } else {
        Err(ConfigError::OutOfRange {
            parameter,
            value: i64::from(value),
            min: bounds.lower,
            max: bounds.upper,
        })
    }
}

/// Like [`bounded`], but `0` is the parameter's default.
fn bounded_or_default(
    parameter: Parameter,
    value: i32,
    bounds: Bounds,
) -> Result<u32, ConfigError> {
    if value == 0 {
        Ok(0)
    } else {
        bounded(parameter, value, bounds)
    }
}

/// A three-way switch from its numeric value.
fn switch(parameter: Parameter, value: i32) -> Result<Switch, ConfigError> {
    Ok(match bounded(parameter, value, Bounds::inclusive(0, 2))? {
        0 => Switch::Auto,
        1 => Switch::Enabled,
        _ => Switch::Disabled,
    })
}

/// An ignored parameter whose value must lie within `0..=upper`.
fn ignored(id: i32, value: i32, upper: i64) -> Result<CParameter, ConfigError> {
    bounded(Parameter::Other(id), value, Bounds::inclusive(0, upper))?;
    Ok(CParameter::Ignored { id, value })
}

/// A parameter only a multithreaded reference accepts, which here must be `0`.
const fn zero_only(id: i32, value: i32) -> Result<CParameter, ConfigError> {
    if value == 0 {
        Ok(CParameter::Ignored { id, value })
    } else {
        Err(ConfigError::Unsupported { id, value })
    }
}

impl TryFrom<(i32, i32)> for CParameter {
    type Error = ConfigError;

    /// Validates a `(ZSTD_cParameter, value)` pair as
    /// `ZSTD_CCtx_setParameter` does on a single-threaded context.
    ///
    /// # Errors
    ///
    /// - [`ConfigError::UnknownParameter`] for an ID the reference does not
    ///   define.
    /// - [`ConfigError::OutOfRange`] for a value outside the parameter's
    ///   bounds. Compression levels are the exception: like the reference,
    ///   this clamps them, and `targetCBlockSize` raises a value below its
    ///   minimum to the minimum.
    /// - [`ConfigError::Unsupported`] where the single-threaded reference
    ///   answers `parameter_unsupported` (a non-zero `nbWorkers`, `jobSize`,
    ///   `overlapLog` or `rsyncable`).
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{CParameter, ConfigError, Level, Switch};
    ///
    /// // Levels clamp; zero is the default level.
    /// assert_eq!(
    ///     CParameter::try_from((100, 99))?,
    ///     CParameter::CompressionLevel(Level::MAX)
    /// );
    /// assert_eq!(
    ///     CParameter::try_from((100, 0))?,
    ///     CParameter::CompressionLevel(Level::DEFAULT)
    /// );
    /// // `ZSTD_c_literalCompressionMode` 2 is `ZSTD_ps_disable`: literals stay raw.
    /// assert_eq!(
    ///     CParameter::try_from((1002, 2))?,
    ///     CParameter::LiteralCompressionMode(Switch::Disabled)
    /// );
    /// assert!(matches!(
    ///     CParameter::try_from((400, 4)),
    ///     Err(ConfigError::Unsupported { id: 400, value: 4 })
    /// ));
    /// # Ok::<(), ConfigError>(())
    /// ```
    fn try_from((id, value): (i32, i32)) -> Result<Self, Self::Error> {
        Ok(match id {
            id::FORMAT => {
                Self::Format(
                    FrameFormat::try_from(value).map_err(|_| ConfigError::OutOfRange {
                        parameter: Parameter::Format,
                        value: i64::from(value),
                        min: 0,
                        max: 1,
                    })?,
                )
            }
            id::COMPRESSION_LEVEL => Self::CompressionLevel(Level::clamped(value)),
            id::WINDOW_LOG => Self::WindowLog(bounded_or_default(
                Parameter::WindowLog,
                value,
                WINDOW_LOG_BOUNDS,
            )?),
            id::HASH_LOG => Self::HashLog(bounded_or_default(
                Parameter::HashLog,
                value,
                HASH_LOG_BOUNDS,
            )?),
            id::CHAIN_LOG => Self::ChainLog(bounded_or_default(
                Parameter::ChainLog,
                value,
                CHAIN_LOG_BOUNDS,
            )?),
            id::SEARCH_LOG => Self::SearchLog(bounded_or_default(
                Parameter::SearchLog,
                value,
                SEARCH_LOG_BOUNDS,
            )?),
            id::MIN_MATCH => Self::MinMatch(bounded_or_default(
                Parameter::MinMatch,
                value,
                MIN_MATCH_BOUNDS,
            )?),
            id::TARGET_LENGTH => Self::TargetLength(bounded(
                Parameter::TargetLength,
                value,
                TARGET_LENGTH_BOUNDS,
            )?),
            id::STRATEGY => Self::Strategy(match value {
                0 => None,
                other => Some(
                    u8::try_from(other)
                        .ok()
                        .and_then(|strategy| Strategy::try_from(strategy).ok())
                        .ok_or(ConfigError::OutOfRange {
                            parameter: Parameter::Strategy,
                            value: i64::from(other),
                            min: 1,
                            max: 9,
                        })?,
                ),
            }),
            id::TARGET_C_BLOCK_SIZE => Self::TargetCBlockSize(if value == 0 {
                None
            } else {
                // The reference raises small and negative values to the
                // minimum before it checks the upper bound.
                let raised = value.max(TARGET_C_BLOCK_SIZE_BOUNDS.lower as i32);
                NonZeroU32::new(bounded(
                    Parameter::TargetCompressedBlockSize,
                    raised,
                    TARGET_C_BLOCK_SIZE_BOUNDS,
                )?)
            }),
            id::ENABLE_LDM => {
                Self::EnableLongDistanceMatching(switch(Parameter::LongDistanceMatching, value)?)
            }
            id::LDM_HASH_LOG => Self::LdmHashLog(bounded_or_default(
                Parameter::LdmHashLog,
                value,
                LDM_HASH_LOG_BOUNDS,
            )?),
            id::LDM_MIN_MATCH => Self::LdmMinMatch(bounded_or_default(
                Parameter::LdmMinMatch,
                value,
                LDM_MIN_MATCH_BOUNDS,
            )?),
            id::LDM_BUCKET_SIZE_LOG => Self::LdmBucketSizeLog(bounded_or_default(
                Parameter::LdmBucketSizeLog,
                value,
                LDM_BUCKET_SIZE_LOG_BOUNDS,
            )?),
            id::LDM_HASH_RATE_LOG => Self::LdmHashRateLog(bounded_or_default(
                Parameter::LdmHashRateLog,
                value,
                LDM_HASH_RATE_LOG_BOUNDS,
            )?),
            id::CONTENT_SIZE_FLAG => Self::ContentSizeFlag(value != 0),
            id::CHECKSUM_FLAG => Self::ChecksumFlag(value != 0),
            id::DICT_ID_FLAG => Self::DictIdFlag(value != 0),
            id::NB_WORKERS | id::JOB_SIZE | id::OVERLAP_LOG | id::RSYNCABLE => {
                zero_only(id, value)?
            }
            id::FORCE_MAX_WINDOW => Self::ForceMaxWindow(value != 0),
            id::FORCE_ATTACH_DICT => Self::ForceAttachDict(
                match bounded(Parameter::DictionaryAttach, value, Bounds::inclusive(0, 3))? {
                    0 => DictionaryAttach::Default,
                    1 => DictionaryAttach::Attach,
                    2 => DictionaryAttach::Copy,
                    _ => DictionaryAttach::Load,
                },
            ),
            id::LITERAL_COMPRESSION_MODE => {
                Self::LiteralCompressionMode(switch(Parameter::LiteralCompression, value)?)
            }
            id::SRC_SIZE_HINT => Self::SrcSizeHint(
                match bounded_or_default(
                    Parameter::SizeHint,
                    value,
                    Bounds::inclusive(0, i64::from(i32::MAX)),
                )? {
                    0 => None,
                    hint => Some(hint),
                },
            ),
            id::ENABLE_DEDICATED_DICT_SEARCH => Self::Ignored { id, value },
            id::STABLE_IN_BUFFER
            | id::STABLE_OUT_BUFFER
            | id::BLOCK_DELIMITERS
            | id::VALIDATE_SEQUENCES
            | id::DETERMINISTIC_REF_PREFIX
            | id::ENABLE_SEQ_PRODUCER_FALLBACK => ignored(id, value, 1)?,
            id::PREFETCH_CDICT_TABLES | id::REPCODE_RESOLUTION => ignored(id, value, 2)?,
            id::SPLIT_AFTER_SEQUENCES => {
                Self::SplitAfterSequences(switch(Parameter::PostBlockSplitter, value)?)
            }
            id::USE_ROW_MATCH_FINDER => {
                Self::UseRowMatchFinder(switch(Parameter::RowMatchFinder, value)?)
            }
            id::MAX_BLOCK_SIZE => Self::MaxBlockSize(bounded_or_default(
                Parameter::MaxBlockSize,
                value,
                MAX_BLOCK_SIZE_BOUNDS,
            )?),
            id::BLOCK_SPLITTER_LEVEL => Self::BlockSplitterLevel(bounded(
                Parameter::BlockSplitterLevel,
                value,
                Bounds::inclusive(0, i64::from(BLOCK_SPLITTER_LEVEL_MAX)),
            )? as u8),
            unknown => return Err(ConfigError::UnknownParameter { id: unknown }),
        })
    }
}

impl From<CParameter> for (i32, i32) {
    /// The reference's `(ZSTD_cParameter, value)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::CParameter;
    ///
    /// let pair = (1002, 2);
    /// assert_eq!(<(i32, i32)>::from(CParameter::try_from(pair)?), pair);
    /// # Ok::<(), mzstd::ConfigError>(())
    /// ```
    fn from(parameter: CParameter) -> Self {
        match parameter {
            CParameter::Format(format) => (id::FORMAT, format.into()),
            CParameter::CompressionLevel(level) => (id::COMPRESSION_LEVEL, level.into()),
            CParameter::WindowLog(value) => (id::WINDOW_LOG, value as i32),
            CParameter::HashLog(value) => (id::HASH_LOG, value as i32),
            CParameter::ChainLog(value) => (id::CHAIN_LOG, value as i32),
            CParameter::SearchLog(value) => (id::SEARCH_LOG, value as i32),
            CParameter::MinMatch(value) => (id::MIN_MATCH, value as i32),
            CParameter::TargetLength(value) => (id::TARGET_LENGTH, value as i32),
            CParameter::Strategy(strategy) => (
                id::STRATEGY,
                strategy.map_or(0, |strategy| i32::from(u8::from(strategy))),
            ),
            CParameter::TargetCBlockSize(size) => (
                id::TARGET_C_BLOCK_SIZE,
                size.map_or(0, |size| size.get() as i32),
            ),
            CParameter::EnableLongDistanceMatching(value) => (id::ENABLE_LDM, value.into()),
            CParameter::LdmHashLog(value) => (id::LDM_HASH_LOG, value as i32),
            CParameter::LdmMinMatch(value) => (id::LDM_MIN_MATCH, value as i32),
            CParameter::LdmBucketSizeLog(value) => (id::LDM_BUCKET_SIZE_LOG, value as i32),
            CParameter::LdmHashRateLog(value) => (id::LDM_HASH_RATE_LOG, value as i32),
            CParameter::ContentSizeFlag(value) => (id::CONTENT_SIZE_FLAG, i32::from(value)),
            CParameter::ChecksumFlag(value) => (id::CHECKSUM_FLAG, i32::from(value)),
            CParameter::DictIdFlag(value) => (id::DICT_ID_FLAG, i32::from(value)),
            CParameter::ForceMaxWindow(value) => (id::FORCE_MAX_WINDOW, i32::from(value)),
            CParameter::ForceAttachDict(value) => (id::FORCE_ATTACH_DICT, value.into()),
            CParameter::LiteralCompressionMode(value) => {
                (id::LITERAL_COMPRESSION_MODE, value.into())
            }
            CParameter::SrcSizeHint(hint) => {
                (id::SRC_SIZE_HINT, hint.map_or(0, |hint| hint as i32))
            }
            CParameter::SplitAfterSequences(value) => (id::SPLIT_AFTER_SEQUENCES, value.into()),
            CParameter::UseRowMatchFinder(value) => (id::USE_ROW_MATCH_FINDER, value.into()),
            CParameter::MaxBlockSize(value) => (id::MAX_BLOCK_SIZE, value as i32),
            CParameter::BlockSplitterLevel(value) => (id::BLOCK_SPLITTER_LEVEL, i32::from(value)),
            CParameter::Ignored { id, value } => (id, value),
        }
    }
}

impl CParameter {
    /// Refuses a log that would need more memory than `max - log_factor`
    /// allows, for hosts that cap what their users may request.
    ///
    /// Applies to the window, hash, chain and long distance matcher hash
    /// logs, whose largest values are `31`, `30`, `30` and `30`. OpenZL's
    /// zstd node refuses a log above `max - 3`. Zero (the default) always
    /// passes, as does every other parameter.
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutOfRange`] naming the lowered maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::CParameter;
    ///
    /// assert!(CParameter::try_from((101, 28))?.check_memory_cap(3).is_ok());
    /// assert!(CParameter::try_from((101, 29))?.check_memory_cap(3).is_err());
    /// assert!(CParameter::try_from((100, 22))?.check_memory_cap(3).is_ok());
    /// # Ok::<(), mzstd::ConfigError>(())
    /// ```
    pub fn check_memory_cap(self, log_factor: u32) -> Result<Self, ConfigError> {
        let (parameter, value, bounds) = match self {
            Self::WindowLog(value) => (Parameter::WindowLog, value, WINDOW_LOG_BOUNDS),
            Self::HashLog(value) => (Parameter::HashLog, value, HASH_LOG_BOUNDS),
            Self::ChainLog(value) => (Parameter::ChainLog, value, CHAIN_LOG_BOUNDS),
            Self::LdmHashLog(value) => (Parameter::LdmHashLog, value, LDM_HASH_LOG_BOUNDS),
            other => return Ok(other),
        };
        let max = bounds.upper - i64::from(log_factor);
        if i64::from(value) > max {
            return Err(ConfigError::OutOfRange {
                parameter,
                value: i64::from(value),
                min: bounds.lower,
                max,
            });
        }
        Ok(self)
    }
}

impl EncoderConfig {
    /// Applies one reference parameter, as `ZSTD_CCtx_setParameter` does.
    ///
    /// A later parameter replaces an earlier one of the same kind, so a list
    /// of parameters applied in order gives the reference's final state.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{CParameter, EncoderConfig, Level};
    ///
    /// let config = EncoderConfig::default()
    ///     .with_parameter(CParameter::CompressionLevel(Level::try_from(7)?))
    ///     .with_parameter(CParameter::CompressionLevel(Level::try_from(12)?))
    ///     .with_parameter(CParameter::ChecksumFlag(true));
    /// assert_eq!(config.level().get(), 12);
    /// assert!(config.checksum());
    /// # Ok::<(), mzstd::ConfigError>(())
    /// ```
    #[must_use]
    pub fn with_parameter(self, parameter: CParameter) -> Self {
        match parameter {
            CParameter::Format(format) => self.with_format(format),
            CParameter::CompressionLevel(level) => self.with_level(level),
            CParameter::WindowLog(value) => self.map_params(|p| p.with_window_log(value)),
            CParameter::HashLog(value) => self.map_params(|p| p.with_hash_log(value)),
            CParameter::ChainLog(value) => self.map_params(|p| p.with_chain_log(value)),
            CParameter::SearchLog(value) => self.map_params(|p| p.with_search_log(value)),
            CParameter::MinMatch(value) => self.map_params(|p| p.with_min_match(value)),
            CParameter::TargetLength(value) => self.map_params(|p| p.with_target_length(value)),
            CParameter::Strategy(strategy) => self.map_params(|p| p.with_strategy_option(strategy)),
            CParameter::TargetCBlockSize(size) => self.with_target_compressed_block_size(size),
            CParameter::EnableLongDistanceMatching(switch) => self.with_ldm_switch(switch),
            CParameter::LdmHashLog(value) => self.map_ldm(|l| l.with_hash_log(value)),
            CParameter::LdmMinMatch(value) => self.map_ldm(|l| l.with_min_match(value)),
            CParameter::LdmBucketSizeLog(value) => self.map_ldm(|l| l.with_bucket_size_log(value)),
            CParameter::LdmHashRateLog(value) => self.map_ldm(|l| l.with_hash_rate_log(value)),
            CParameter::ContentSizeFlag(value) => self.with_content_size(value),
            CParameter::ChecksumFlag(value) => self.with_checksum(value),
            CParameter::DictIdFlag(value) => self.with_dictionary_id(value),
            CParameter::ForceMaxWindow(value) => self.with_force_max_window(value),
            CParameter::ForceAttachDict(attach) => self.with_dictionary_attach(attach),
            CParameter::LiteralCompressionMode(switch) => self.with_literal_compression(switch),
            CParameter::SrcSizeHint(hint) => self.with_size_hint(hint),
            CParameter::SplitAfterSequences(switch) => self.with_post_block_splitter(switch),
            CParameter::UseRowMatchFinder(switch) => self.with_row_match_finder(switch),
            CParameter::MaxBlockSize(value) => {
                self.with_max_block_size(if value == 0 { None } else { Some(value) })
            }
            CParameter::BlockSplitterLevel(level) => self.with_block_splitter_level(level),
            CParameter::Ignored { .. } => self,
        }
    }
}
