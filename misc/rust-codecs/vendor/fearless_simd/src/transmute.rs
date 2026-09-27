// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Safe wrappers for SIMD loads/stores and reference casts.
//!
//! This fulfills the purpose of both `safe_unaligned_simd` and `bytemuck` crates.
//! The implementation is bytemuck-like, but far smaller than either of those crates,
//! mostly by virtue of supporting less features (e.g. no by-value transmute).
//!
//! This approach also relies less on the optimizer than using `bytemuck` would.
//! We've verified that it emits the same IR as using intrinsics directly,
//! since they [lower into generic loads/stores anyway](https://github.com/rust-lang/stdarch/pull/2004).
//! With bytemuck the optimizer would need to do more work to remove the abstractions,
//! and for single load/store instructions an inlining failure would be devastating.
//!
//! It's not possible to get rid of `unsafe` here entirely, even if we were to use external crates,
//! because we need to implement Pod for wrappers like Aligned512 which cannot be safely derived.

use core::mem::{align_of, size_of};

use crate::support::{Aligned128, Aligned256, Aligned512};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float32x4_t, float32x4x2_t, float32x4x4_t, float64x2_t, float64x2x2_t, float64x2x4_t,
    int8x16_t, int8x16x2_t, int8x16x4_t, int16x8_t, int16x8x2_t, int16x8x4_t, int32x4_t,
    int32x4x2_t, int32x4x4_t, int64x2_t, int64x2x2_t, int64x2x4_t, uint8x16_t, uint8x16x2_t,
    uint8x16x4_t, uint16x8_t, uint16x8x2_t, uint16x8x4_t, uint32x4_t, uint32x4x2_t, uint32x4x4_t,
    uint64x2_t, uint64x2x2_t, uint64x2x4_t,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::v128;
#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
};

/// Types that can be safely copied through an arbitrary same-sized bit representation.
///
/// This is intentionally narrower than all possible POD types: it only covers scalar and SIMD
/// storage types used by this crate today.
///
/// # Safety
///
/// Implementors must be `Copy`, contain no pointers or padding, and every bit pattern must be a
/// valid value of the type.
///
/// See [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)
/// for the exact requirements and examples.
#[allow(
    unnameable_types,
    reason = "This must be `pub` to avoid `private_bounds` warnings on the generated `ArchTypes` trait, but the containing module remains private"
)]
#[allow(
    unreachable_pub,
    reason = "This must be `pub` to avoid `private_bounds` warnings on the generated `ArchTypes` trait, but the containing module remains private"
)]
pub unsafe trait SimdPod: Copy {}

#[allow(dead_code, reason = "Not all platforms use safe transmute machinery")]
fn assert_simd_pod<T: SimdPod>() {}

// Do not blanket-impl `Aligned*<T: SimdPod>`: alignment wrappers can add
// trailing padding for undersized `T`, e.g. `Aligned128<u8>`.
macro_rules! impl_aligned_simd_pod {
    ($($wrapper:ident<$inner:ty>),+ $(,)?) => {
        $(
            // SAFETY: this enforces that the alignment wrapper adds no trailing padding to the
            // wrapped SIMD storage, preserving the `SimdPod` no-padding invariant,
            // and that the inner type is also SimdPod
            const _: () = assert!(size_of::<$wrapper<$inner>>() == size_of::<$inner>());
            const _: fn() = assert_simd_pod::<$inner>;
            unsafe impl SimdPod for $wrapper<$inner> {}
        )+
    };
}

unsafe impl SimdPod for f32 {}
unsafe impl SimdPod for f64 {}
unsafe impl SimdPod for i8 {}
unsafe impl SimdPod for u8 {}
unsafe impl SimdPod for i16 {}
unsafe impl SimdPod for u16 {}
unsafe impl SimdPod for i32 {}
unsafe impl SimdPod for u32 {}
unsafe impl SimdPod for i64 {}
unsafe impl SimdPod for u64 {}

unsafe impl<T: SimdPod, const N: usize> SimdPod for [T; N] {}

impl_aligned_simd_pod!(
    Aligned128<[f32; 4]>,
    Aligned128<[f64; 2]>,
    Aligned128<[i8; 16]>,
    Aligned128<[i16; 8]>,
    Aligned128<[i32; 4]>,
    Aligned128<[i64; 2]>,
    Aligned128<[u8; 16]>,
    Aligned128<[u16; 8]>,
    Aligned128<[u32; 4]>,
    Aligned128<[u64; 2]>,
    Aligned256<[f32; 8]>,
    Aligned256<[f64; 4]>,
    Aligned256<[i8; 32]>,
    Aligned256<[i16; 16]>,
    Aligned256<[i32; 8]>,
    Aligned256<[i64; 4]>,
    Aligned256<[u8; 32]>,
    Aligned256<[u16; 16]>,
    Aligned256<[u32; 8]>,
    Aligned256<[u64; 4]>,
    Aligned512<[f32; 16]>,
    Aligned512<[f64; 8]>,
    Aligned512<[i8; 64]>,
    Aligned512<[i16; 32]>,
    Aligned512<[i32; 16]>,
    Aligned512<[i64; 8]>,
    Aligned512<[u8; 64]>,
    Aligned512<[u16; 32]>,
    Aligned512<[u32; 16]>,
    Aligned512<[u64; 8]>,
);

// the `const` is just to only use a single cfg annotation, nothing to do with const evaluation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const _: () = {
    // SAFETY: std docs clearly state:
    // > The in-memory representation of this type is the same as the one of an equivalent array
    // > (i.e. the in-memory order of elements is the same,
    // and there is no padding between two consecutive elements);
    // > however, the alignment is different and equal to the size of the type.
    // at https://doc.rust-lang.org/stable/core/arch/x86_64/struct.__m256.html and the other structs
    // Fortunately for us, transmute_copy() does not care about alignment
    unsafe impl SimdPod for __m128 {}
    unsafe impl SimdPod for __m128d {}
    unsafe impl SimdPod for __m128i {}
    unsafe impl SimdPod for __m256 {}
    unsafe impl SimdPod for __m256d {}
    unsafe impl SimdPod for __m256i {}
    unsafe impl SimdPod for __m512 {}
    unsafe impl SimdPod for __m512d {}
    unsafe impl SimdPod for __m512i {}
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe impl SimdPod for v128 {}

#[cfg(target_arch = "aarch64")]
const _: () = {
    // SAFETY:
    // Compound types like float32x4x4_t are defined as #[repr(C)] tuples of basic types,
    // see e.g. https://doc.rust-lang.org/stable/core/arch/aarch64/struct.float32x4x4_t.html
    unsafe impl SimdPod for float32x4_t {}
    unsafe impl SimdPod for float32x4x2_t {}
    unsafe impl SimdPod for float32x4x4_t {}
    unsafe impl SimdPod for float64x2_t {}
    unsafe impl SimdPod for float64x2x2_t {}
    unsafe impl SimdPod for float64x2x4_t {}
    unsafe impl SimdPod for int8x16_t {}
    unsafe impl SimdPod for int8x16x2_t {}
    unsafe impl SimdPod for int8x16x4_t {}
    unsafe impl SimdPod for int16x8_t {}
    unsafe impl SimdPod for int16x8x2_t {}
    unsafe impl SimdPod for int16x8x4_t {}
    unsafe impl SimdPod for int32x4_t {}
    unsafe impl SimdPod for int32x4x2_t {}
    unsafe impl SimdPod for int32x4x4_t {}
    unsafe impl SimdPod for int64x2_t {}
    unsafe impl SimdPod for int64x2x2_t {}
    unsafe impl SimdPod for int64x2x4_t {}
    unsafe impl SimdPod for uint8x16_t {}
    unsafe impl SimdPod for uint8x16x2_t {}
    unsafe impl SimdPod for uint8x16x4_t {}
    unsafe impl SimdPod for uint16x8_t {}
    unsafe impl SimdPod for uint16x8x2_t {}
    unsafe impl SimdPod for uint16x8x4_t {}
    unsafe impl SimdPod for uint32x4_t {}
    unsafe impl SimdPod for uint32x4x2_t {}
    unsafe impl SimdPod for uint32x4x4_t {}
    unsafe impl SimdPod for uint64x2_t {}
    unsafe impl SimdPod for uint64x2x2_t {}
    unsafe impl SimdPod for uint64x2x4_t {}
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_aligned_simd_pod!(
    Aligned128<__m128>,
    Aligned128<__m128d>,
    Aligned128<__m128i>,
    Aligned256<__m256>,
    Aligned256<__m256d>,
    Aligned256<__m256i>,
    Aligned256<[__m128; 2]>,
    Aligned256<[__m128d; 2]>,
    Aligned256<[__m128i; 2]>,
    Aligned512<[__m128; 4]>,
    Aligned512<[__m128d; 4]>,
    Aligned512<[__m128i; 4]>,
    Aligned512<[__m256; 2]>,
    Aligned512<[__m256d; 2]>,
    Aligned512<[__m256i; 2]>,
    Aligned512<__m512>,
    Aligned512<__m512d>,
    Aligned512<__m512i>,
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_aligned_simd_pod!(
    Aligned128<v128>,
    Aligned256<[v128; 2]>,
    Aligned512<[v128; 4]>
);

#[cfg(target_arch = "aarch64")]
impl_aligned_simd_pod!(
    Aligned128<float32x4_t>,
    Aligned128<float64x2_t>,
    Aligned128<int8x16_t>,
    Aligned128<int16x8_t>,
    Aligned128<int32x4_t>,
    Aligned128<int64x2_t>,
    Aligned128<uint8x16_t>,
    Aligned128<uint16x8_t>,
    Aligned128<uint32x4_t>,
    Aligned128<uint64x2_t>,
    Aligned256<float32x4x2_t>,
    Aligned256<float64x2x2_t>,
    Aligned256<int8x16x2_t>,
    Aligned256<int16x8x2_t>,
    Aligned256<int32x4x2_t>,
    Aligned256<int64x2x2_t>,
    Aligned256<uint8x16x2_t>,
    Aligned256<uint16x8x2_t>,
    Aligned256<uint32x4x2_t>,
    Aligned256<uint64x2x2_t>,
    Aligned512<float32x4x4_t>,
    Aligned512<float64x2x4_t>,
    Aligned512<int8x16x4_t>,
    Aligned512<int16x8x4_t>,
    Aligned512<int32x4x4_t>,
    Aligned512<int64x2x4_t>,
    Aligned512<uint8x16x4_t>,
    Aligned512<uint16x8x4_t>,
    Aligned512<uint32x4x4_t>,
    Aligned512<uint64x2x4_t>,
);

/// Like [`core::mem::transmute_copy`], but statically rejects differently-sized
/// types and only accepts this crate's SIMD plain-old-data storage types.
#[inline(always)]
#[allow(
    clippy::disallowed_methods,
    reason = "This is the central checked wrapper around transmute_copy"
)]
pub(crate) fn checked_transmute_copy<Src: SimdPod, Dst: SimdPod>(src: &Src) -> Dst {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_transmute_copy requires source and destination to have the same size"
        );
    }
    // Safety: `SimdPod` guarantees source and destination validity for all bit patterns, and
    // the const assertion above prevents the "destination larger than source" failure mode.
    unsafe { core::mem::transmute_copy(src) }
}

/// Store a plain-old-data value into a differently typed same-sized destination.
///
/// This is the store-side counterpart to [`checked_transmute_copy`].
/// The destination only needs to satisfy its own alignment,
/// not the source type's alignment.
#[inline(always)]
#[allow(dead_code, reason = "Not all backends use this function")]
pub(crate) fn checked_transmute_store<Src: SimdPod, Dst: SimdPod>(src: Src, dest: &mut Dst) {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_transmute_store requires source and destination to have the same size"
        );
    }
    // Safety: `SimdPod` guarantees source and destination validity for all bit patterns, and
    // the const assertion above ensures that the write fully covers exactly one destination.
    // `write_unaligned` avoids making a reference to `Src`, so this is valid even when `dest`
    // has weaker alignment than `Src`.
    // Performance: this lowers into the same LLVM IR as platform-specific store intrinsics.
    // The alternative of ptr::copy_nonoverlapping lowers into a memcpy, which is worse.
    unsafe { core::ptr::write_unaligned((dest as *mut Dst).cast::<Src>(), src) }
}

/// Like `bytemuck::cast_ref`, but rejects incompatible types at compile time
/// and only accepts this crate's SIMD plain-old-data storage types.
#[inline(always)]
#[allow(
    clippy::disallowed_methods,
    reason = "This is the central checked wrapper around transmute"
)]
#[allow(dead_code, reason = "Not all backends use this function")]
pub(crate) fn checked_cast_ref<Src: SimdPod, Dst: SimdPod>(src: &Src) -> &Dst {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_cast_ref requires source and destination to have the same size"
        );
        // alignment is always a power of two as per Rust Reference:
        // https://doc.rust-lang.org/stable/reference/type-layout.html#size-and-alignment
        // so >= is sufficient and won't run into issues with coprime alignments
        assert!(
            align_of::<Src>() >= align_of::<Dst>(),
            "checked_cast_ref requires source to have alignment equal or greater to the destination"
        );
    }
    // Safety: `SimdPod` guarantees source and destination validity for all bit patterns, and
    // the const assertions above enforce compatible size and alignment.
    //
    // The pointer cast has less footguns than `transmute`:
    // `src as *const Src` keeps the same address and provenance from the original reference,
    // so we will never run into lifetime issues.
    unsafe { &*(src as *const Src).cast::<Dst>() }
}

/// Like `bytemuck::cast_mut`, but rejects incompatible types at compile time
/// and only accepts this crate's SIMD plain-old-data storage types.
#[inline(always)]
#[allow(
    clippy::disallowed_methods,
    reason = "This is the central checked wrapper around transmute"
)]
#[allow(dead_code, reason = "Not all backends use this function")]
pub(crate) fn checked_cast_mut<Src: SimdPod, Dst: SimdPod>(src: &mut Src) -> &mut Dst {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_cast_mut requires source and destination to have the same size"
        );
        // alignment is always a power of two as per Rust Reference:
        // https://doc.rust-lang.org/stable/reference/type-layout.html#size-and-alignment
        // so >= is sufficient and won't run into issues with coprime alignments
        assert!(
            align_of::<Src>() >= align_of::<Dst>(),
            "checked_cast_mut requires source to have alignment equal or greater to the destination"
        );
    }
    // Safety: `SimdPod` guarantees source and destination validity for all bit patterns, and
    // the const assertions above enforce compatible size and alignment.
    //
    // The pointer cast has less footguns than `transmute`:
    // `src as *mut Src` keeps the same address and provenance from the original reference,
    // so we will never run into lifetime issues.
    unsafe { &mut *(src as *mut Src).cast::<Dst>() }
}
