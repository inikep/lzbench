// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 128-bit alignment.
pub struct Aligned128<T>(pub T);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 256-bit alignment.
pub struct Aligned256<T>(pub T);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 512-bit alignment.
pub struct Aligned512<T>(pub T);

/// The actual `Debug` implementation for all `SimdBase` types. This only needs to be monomorphized once per element
/// type, rather than once per vector type.
#[inline(never)]
pub(crate) fn simd_debug_impl<Element: core::fmt::Debug>(
    f: &mut core::fmt::Formatter<'_>,
    type_name: &str,
    token: &dyn core::fmt::Debug,
    items: &[Element],
) -> core::fmt::Result {
    f.debug_struct(type_name)
        .field("val", &items)
        .field("simd", token)
        .finish()
}

/// Selects the input operands to be used for `slignr`/`vext`/etc. when computing a single output block for cross-block
/// "slide" operations. Extracts from [a : b].
#[inline(always)]
#[allow(clippy::allow_attributes, reason = "Only needed in some cfgs.")]
#[allow(dead_code, reason = "Only used in some cfgs.")]
pub(crate) fn cross_block_slide_blocks_at<const N: usize, Block: Copy>(
    a: &[Block; N],
    b: &[Block; N],
    out_idx: usize,
    shift_bytes: usize,
) -> [Block; 2] {
    const BLOCK_BYTES: usize = 16;
    let out_byte_start = out_idx * BLOCK_BYTES + shift_bytes;
    let lo_idx = out_byte_start.div_euclid(BLOCK_BYTES);
    let hi_idx = lo_idx + 1;
    // Concatenation is [a : b], so indices 0..N are from a, indices N..2N are from b
    let lo_block = if lo_idx < N { a[lo_idx] } else { b[lo_idx - N] };
    let hi_block = if hi_idx < N { a[hi_idx] } else { b[hi_idx - N] };
    [lo_block, hi_block]
}
