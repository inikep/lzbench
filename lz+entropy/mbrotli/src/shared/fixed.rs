//! Fixed-size heap tables whose length is part of their type.
//!
//! A `Box<[T; N]>` indexed by a value the compiler can bound (a `u8`, or a key
//! masked below `N`) needs no bounds check, and unlike a `Vec` it cannot be
//! resized by accident. These helpers build such tables through a `Vec` so a
//! zero fill reaches the allocator as a zeroed allocation instead of a copy of
//! a stack array.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Allocates a fixed-size table filled with `initial`.
#[inline(always)]
pub(crate) fn fixed_table<T: Copy, const N: usize>(initial: T) -> Box<[T; N]> {
    let Ok(table) = vec![initial; N].into_boxed_slice().try_into() else {
        unreachable!("table was created with exactly N entries");
    };
    table
}

/// Resizes an existing vector and transfers its allocation into a fixed-size table.
#[inline(always)]
pub(crate) fn fixed_table_from_vec<T: Copy, const N: usize>(
    mut values: Vec<T>,
    initial: T,
) -> Box<[T; N]> {
    values.resize(N, initial);
    let Ok(table) = values.into_boxed_slice().try_into() else {
        unreachable!("table was resized to exactly N entries");
    };
    table
}

/// Makes `buffer` at least `len` long, zero-filling only the new tail.
///
/// An empty buffer is allocated through `vec![0; len]`, which the allocator
/// serves zeroed: a large buffer most of which is never written then costs no
/// writes at all. A buffer that already holds data grows with
/// [`Vec::resize`], which an allocator may do in place. Existing contents are
/// kept, and a buffer never shrinks.
#[inline]
pub(crate) fn grow_zeroed<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) {
    if buffer.is_empty() {
        *buffer = vec![T::default(); len];
    } else if buffer.len() < len {
        buffer.resize(len, T::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_fixed_tables_initialize_every_entry() {
        assert_eq!(*fixed_table::<u32, 4>(0), [0; 4]);
        assert_eq!(*fixed_table::<u32, 4>(7), [7; 4]);
        assert_eq!(*fixed_table::<u32, 0>(7), [0_u32; 0]);
    }

    #[test]
    fn growing_keeps_contents_zeroes_the_tail_and_never_shrinks() {
        let mut buffer: Vec<u32> = Vec::new();
        grow_zeroed(&mut buffer, 3);
        assert_eq!(buffer, [0, 0, 0]);
        buffer[1] = 7;
        grow_zeroed(&mut buffer, 5);
        assert_eq!(buffer, [0, 7, 0, 0, 0]);
        grow_zeroed(&mut buffer, 2);
        assert_eq!(buffer, [0, 7, 0, 0, 0]);
    }

    #[test]
    fn an_empty_vector_is_resized_into_an_initialized_table() {
        assert_eq!(*fixed_table_from_vec::<u32, 4>(Vec::new(), 7), [7; 4]);
    }

    #[test]
    fn fixed_tables_preserve_existing_values_and_initialize_only_the_extension() {
        assert_eq!(*fixed_table_from_vec::<_, 4>(vec![7, 8], 3), [7, 8, 3, 3]);
        assert_eq!(*fixed_table_from_vec::<_, 1>(vec![7, 8], 3), [7]);
        assert_eq!(*fixed_table_from_vec::<u32, 0>(vec![7, 8], 3), [0_u32; 0]);
        assert_eq!(*fixed_table_from_vec::<u32, 0>(Vec::new(), 3), [0_u32; 0]);
    }
}
