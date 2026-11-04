//! Entropy coding of blocks: Huffman literals and FSE sequences.
//!
//! The submodules port the reference's `fse_compress.c`, `huf_compress.c`,
//! `hist.c`, `zstd_compress_literals.c` and `zstd_compress_sequences.c`.
//! [`Tables`] is the per-block entropy state (`ZSTD_entropyCTables_t`) that
//! blocks inherit from their predecessor.

use core::ops::{Deref, DerefMut};

use super::lazy_box::{LazyBox, StaticEmpty};

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
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct HufTables {
    pub(crate) table: LazyBox<huf::CTable>,
    pub(crate) repeat: huf::Repeat,
}

impl Clone for HufTables {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            repeat: self.repeat,
        }
    }

    /// Field by field, so a written table is copied into place
    /// (`ZSTD_memcpy` of the reference) rather than reallocated.
    fn clone_from(&mut self, source: &Self) {
        self.table.clone_from(&source.table);
        self.repeat = source.repeat;
    }
}

/// The three sequence encoding tables, allocated together on the first
/// write to any of them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FseCTables {
    pub(crate) offcode: fse::CTable,
    pub(crate) matchlength: fse::CTable,
    pub(crate) litlength: fse::CTable,
}

impl StaticEmpty for FseCTables {
    const EMPTY_REF: &'static Self = &Self {
        offcode: fse::CTable::EMPTY,
        matchlength: fse::CTable::EMPTY,
        litlength: fse::CTable::EMPTY,
    };
}

/// Sequence table state carried between blocks (`ZSTD_fseCTables_t`).
///
/// It dereferences to its [`FseCTables`]: reading a table never allocates,
/// writing one allocates all three once.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct FseTables {
    tables: LazyBox<FseCTables>,
    pub(crate) offcode_repeat: FseRepeat,
    pub(crate) matchlength_repeat: FseRepeat,
    pub(crate) litlength_repeat: FseRepeat,
}

impl FseTables {
    /// Heap bytes the tables hold.
    pub(crate) const fn heap_bytes(&self) -> usize {
        self.tables.heap_bytes()
    }
}

impl Deref for FseTables {
    type Target = FseCTables;

    #[inline]
    fn deref(&self) -> &FseCTables {
        &self.tables
    }
}

impl DerefMut for FseTables {
    #[inline]
    fn deref_mut(&mut self) -> &mut FseCTables {
        &mut self.tables
    }
}

impl Clone for FseTables {
    fn clone(&self) -> Self {
        Self {
            tables: self.tables.clone(),
            offcode_repeat: self.offcode_repeat,
            matchlength_repeat: self.matchlength_repeat,
            litlength_repeat: self.litlength_repeat,
        }
    }

    /// Field by field, so written tables are copied into place.
    fn clone_from(&mut self, source: &Self) {
        self.tables.clone_from(&source.tables);
        self.offcode_repeat = source.offcode_repeat;
        self.matchlength_repeat = source.matchlength_repeat;
        self.litlength_repeat = source.litlength_repeat;
    }
}

/// Everything a block's entropy coding inherits (`ZSTD_entropyCTables_t`).
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct Tables {
    pub(crate) huf: HufTables,
    pub(crate) fse: FseTables,
}

impl Clone for Tables {
    fn clone(&self) -> Self {
        Self {
            huf: self.huf.clone(),
            fse: self.fse.clone(),
        }
    }

    /// Field by field, so written tables are copied into place.
    fn clone_from(&mut self, source: &Self) {
        self.huf.clone_from(&source.huf);
        self.fse.clone_from(&source.fse);
    }
}

impl Tables {
    /// Heap bytes held by the tables written so far.
    pub(crate) const fn heap_bytes(&self) -> usize {
        self.huf.table.heap_bytes() + self.fse.heap_bytes()
    }

    /// Forgets every reusable table (`ZSTD_reset_compressedBlockState`).
    pub(crate) fn reset(&mut self) {
        self.huf.repeat = huf::Repeat::None;
        self.fse.offcode_repeat = FseRepeat::None;
        self.fse.matchlength_repeat = FseRepeat::None;
        self.fse.litlength_repeat = FseRepeat::None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwritten_tables_read_as_zeroes_without_allocating() {
        let tables = Tables::default();
        assert_eq!(*tables.huf.table, huf::CTable::default());
        assert_eq!(*tables.fse, FseCTables::default());
        assert_eq!(tables.heap_bytes(), 0);
    }

    #[test]
    fn table_sets_clone_field_by_field() {
        let mut source = Tables::default();
        source.huf.table.table_log = 6;
        source.huf.repeat = huf::Repeat::Check;
        source.fse.matchlength.table_log = 6;
        source.fse.offcode_repeat = FseRepeat::Valid;
        source.fse.matchlength_repeat = FseRepeat::Check;
        source.fse.litlength_repeat = FseRepeat::Valid;
        let copy = source.clone();
        assert_eq!(copy, source);
        let mut target = Tables::default();
        target.fse.litlength.table_log = 9;
        target.clone_from(&source);
        assert_eq!(target, source);
        target.clone_from(&Tables::default());
        assert_eq!(target, Tables::default());
        assert_eq!(target.heap_bytes(), 0);
    }

    #[test]
    fn tables_report_their_heap_bytes_and_reset_repeat_modes() {
        let mut tables = Tables::default();
        tables.huf.table.table_log = 1;
        tables.fse.offcode.table_log = 1;
        // One write allocates all three sequence tables.
        tables.fse.litlength.table_log = 1;
        tables.huf.repeat = huf::Repeat::Valid;
        tables.fse.litlength_repeat = FseRepeat::Valid;
        assert_eq!(
            tables.heap_bytes(),
            core::mem::size_of::<huf::CTable>() + core::mem::size_of::<FseCTables>()
        );
        tables.reset();
        assert_eq!(tables.huf.repeat, huf::Repeat::None);
        assert_eq!(tables.fse.litlength_repeat, FseRepeat::None);
    }
}
