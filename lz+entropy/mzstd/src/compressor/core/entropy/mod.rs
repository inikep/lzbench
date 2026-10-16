//! Entropy coding of blocks: Huffman literals and FSE sequences.
//!
//! The submodules port the reference's `fse_compress.c`, `huf_compress.c`,
//! `hist.c`, `zstd_compress_literals.c` and `zstd_compress_sequences.c`.
//! [`Tables`] is the per-block entropy state (`ZSTD_entropyCTables_t`) that
//! blocks inherit from their predecessor.

pub(crate) mod bits;
pub(crate) mod fse;
pub(crate) mod hist;
pub(crate) mod huf;
pub(crate) mod literals;
pub(crate) mod read;
pub(crate) mod sequences;

/// How a previous sequence table may be reused (`FSE_repeat`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum FseRepeat {
    /// The previous table cannot be used.
    #[default]
    None,
    /// The previous table can be used after checking it.
    Check,
    /// The previous table can be used as is.
    Valid,
}

/// Huffman state carried between blocks (`ZSTD_hufCTables_t`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct HufTables {
    pub(crate) table: huf::CTable,
    pub(crate) repeat: huf::Repeat,
}

/// Sequence table state carried between blocks (`ZSTD_fseCTables_t`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FseTables {
    pub(crate) offcode: fse::CTable,
    pub(crate) matchlength: fse::CTable,
    pub(crate) litlength: fse::CTable,
    pub(crate) offcode_repeat: FseRepeat,
    pub(crate) matchlength_repeat: FseRepeat,
    pub(crate) litlength_repeat: FseRepeat,
}

/// Everything a block's entropy coding inherits (`ZSTD_entropyCTables_t`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Tables {
    pub(crate) huf: HufTables,
    pub(crate) fse: FseTables,
}

impl Tables {
    /// Forgets every reusable table (`ZSTD_reset_compressedBlockState`).
    pub(crate) fn reset(&mut self) {
        self.huf.repeat = huf::Repeat::None;
        self.fse.offcode_repeat = FseRepeat::None;
        self.fse.matchlength_repeat = FseRepeat::None;
        self.fse.litlength_repeat = FseRepeat::None;
    }
}
