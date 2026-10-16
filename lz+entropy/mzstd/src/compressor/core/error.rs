//! Failures inside the compressor core.
//!
//! The variants mirror the reference error codes the encoder can raise. Most
//! of them never reach a caller: the reference turns many of them into a
//! choice of encoding ("this table does not fit, store the literals raw"),
//! and this port does the same at the same places. [`CoreError::code`]
//! reproduces the reference's `size_t` encoding for the few comparisons that
//! operate on an error value as if it were a size.

use thiserror::Error;

/// A reference error code raised by the compressor core.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Error)]
pub(crate) enum CoreError {
    /// `ZSTD_error_GENERIC`.
    #[error("internal encoder failure")]
    Generic,
    /// `ZSTD_error_tableLog_tooLarge`.
    #[error("entropy table log too large")]
    TableLogTooLarge,
    /// `ZSTD_error_maxSymbolValue_tooLarge`.
    #[error("symbol value too large for the table")]
    MaxSymbolValueTooLarge,
    /// `ZSTD_error_maxSymbolValue_tooSmall`.
    #[error("symbol value above the declared maximum")]
    MaxSymbolValueTooSmall,
    /// `ZSTD_error_dstSize_tooSmall`.
    #[error("destination buffer too small")]
    DstSizeTooSmall,
    /// `ZSTD_error_srcSize_wrong`.
    #[error("source size does not match the declared size")]
    SrcSizeWrong,
    /// `ZSTD_error_stage_wrong`.
    #[error("operation not allowed at this stage")]
    StageWrong,
}

impl CoreError {
    /// The reference's `ZSTD_ErrorCode` value.
    pub(crate) const fn error_code(self) -> usize {
        match self {
            Self::Generic => 1,
            Self::TableLogTooLarge => 44,
            Self::MaxSymbolValueTooLarge => 46,
            Self::MaxSymbolValueTooSmall => 48,
            Self::DstSizeTooSmall => 70,
            Self::SrcSizeWrong => 72,
            Self::StageWrong => 60,
        }
    }

    /// The error as the reference's `size_t` return value, `(size_t)-code`.
    pub(crate) const fn code(self) -> usize {
        0usize.wrapping_sub(self.error_code())
    }
}

/// `ERR_isError` for a reference-style `size_t` result.
pub(crate) const fn is_error_code(value: usize) -> bool {
    value > 0usize.wrapping_sub(120)
}

/// Encodes a result the way the reference returns it.
pub(crate) const fn as_code(result: Result<usize, CoreError>) -> usize {
    match result {
        Ok(value) => value,
        Err(error) => error.code(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_negated_reference_values() {
        assert_eq!(CoreError::Generic.code(), usize::MAX);
        assert_eq!(CoreError::DstSizeTooSmall.code(), usize::MAX - 69);
        for error in [
            CoreError::Generic,
            CoreError::TableLogTooLarge,
            CoreError::MaxSymbolValueTooLarge,
            CoreError::MaxSymbolValueTooSmall,
            CoreError::DstSizeTooSmall,
            CoreError::SrcSizeWrong,
            CoreError::StageWrong,
        ] {
            assert!(is_error_code(error.code()), "{error}");
            assert_eq!(as_code(Err(error)), error.code());
        }
        assert!(!is_error_code(1 << 40));
        assert_eq!(as_code(Ok(5)), 5);
    }
}
