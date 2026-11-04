//! Heap state allocated on its first write.
//!
//! A one-shot call builds a fresh engine, so every byte the engine
//! initializes up front is paid per call. The reference leaves its entropy
//! tables and optimal-parser state uninitialized in its workspace until a
//! block writes them; [`LazyBox`] gets the same effect without `unsafe`:
//! the state is allocated, zeroed, on the first mutable access, and reads
//! before that see one shared static empty value.

use alloc::boxed::Box;
use core::ops::{Deref, DerefMut};

/// A type with a static value equal to its [`Default`], which an
/// unallocated [`LazyBox`] reads as.
pub(crate) trait StaticEmpty: Clone + Default + 'static {
    /// The default value, as a static.
    const EMPTY_REF: &'static Self;
}

/// `T` behind a box that is allocated on the first mutable access.
///
/// It dereferences to [`StaticEmpty::EMPTY_REF`] until then, so it must
/// only hold state whose default value is what a reader expects before the
/// first write: callers either read after a write, or check a mode beside
/// it (a table's repeat mode, a sum of zero) first.
#[derive(Debug, Default)]
pub(crate) struct LazyBox<T: StaticEmpty>(Option<Box<T>>);

impl<T: StaticEmpty> LazyBox<T> {
    /// Replaces the value, reusing the allocation when there is one; an
    /// unallocated box is allocated holding `value`, never zeroed first.
    pub(crate) fn set(&mut self, value: T) {
        match &mut self.0 {
            Some(slot) => **slot = value,
            None => self.0 = Some(Box::new(value)),
        }
    }

    /// The value when it has been allocated, without allocating it.
    pub(crate) fn get_mut(&mut self) -> Option<&mut T> {
        self.0.as_deref_mut()
    }

    /// Heap bytes the box holds.
    pub(crate) const fn heap_bytes(&self) -> usize {
        if self.0.is_some() {
            core::mem::size_of::<T>()
        } else {
            0
        }
    }
}

impl<T: StaticEmpty> From<T> for LazyBox<T> {
    fn from(value: T) -> Self {
        Self(Some(Box::new(value)))
    }
}

impl<T: StaticEmpty> Deref for LazyBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.as_deref().unwrap_or(T::EMPTY_REF)
    }
}

impl<T: StaticEmpty> DerefMut for LazyBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.get_or_insert_with(Box::default)
    }
}

impl<T: StaticEmpty> Clone for LazyBox<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    /// Copies in place when both sides are allocated; an unallocated
    /// source leaves `self` unallocated as well.
    fn clone_from(&mut self, source: &Self) {
        match (&mut self.0, &source.0) {
            (Some(value), Some(from)) => (**value).clone_from(from),
            (to, from) => to.clone_from(from),
        }
    }
}

impl<T: StaticEmpty + PartialEq> PartialEq for LazyBox<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: StaticEmpty + Eq> Eq for LazyBox<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct Table {
        log: u32,
        cells: [u16; 32],
    }

    impl StaticEmpty for Table {
        const EMPTY_REF: &'static Self = &Self {
            log: 0,
            cells: [0; 32],
        };
    }

    fn built() -> Table {
        let mut cells = [0; 32];
        cells[3] = 7;
        Table { log: 6, cells }
    }

    #[test]
    fn an_unallocated_box_reads_as_the_default() {
        let mut lazy = LazyBox::<Table>::default();
        assert_eq!(*lazy, Table::default());
        assert_eq!(lazy.heap_bytes(), 0);
        assert!(lazy.get_mut().is_none());
    }

    #[test]
    fn a_write_allocates_a_default_value() {
        let mut lazy = LazyBox::<Table>::default();
        let table: &mut Table = &mut lazy;
        table.log = 5;
        assert_eq!(lazy.heap_bytes(), core::mem::size_of::<Table>());
        assert_eq!(lazy.cells, [0; 32]);
        assert_eq!(lazy.get_mut().map(|table| table.log), Some(5));
    }

    #[test]
    fn set_replaces_in_place_or_allocates() {
        let mut lazy = LazyBox::<Table>::default();
        lazy.set(built());
        assert_eq!(*lazy, built());
        lazy.set(Table::default());
        assert_eq!(*lazy, Table::default());
        assert_eq!(lazy.heap_bytes(), core::mem::size_of::<Table>());
    }

    #[test]
    fn clone_from_follows_the_source() {
        let written = LazyBox::from(built());
        let unwritten = LazyBox::<Table>::default();

        let mut target = LazyBox::<Table>::default();
        target.clone_from(&written);
        assert_eq!(target, written);
        let mut other = LazyBox::from(Table::default());
        other.clone_from(&written);
        assert_eq!(other, written);

        // An unallocated source reads as the default, and so does the copy.
        target.clone_from(&unwritten);
        assert_eq!(target, unwritten);
        assert_eq!(target.heap_bytes(), 0);
        assert_eq!(written.clone(), written);
        assert_ne!(written, unwritten);
    }
}
