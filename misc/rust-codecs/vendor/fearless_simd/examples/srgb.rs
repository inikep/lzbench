// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Converts a single RGBA pixel from linear RGB to sRGB.
//!
//! This example demonstrates:
//! - processing data in fixed-size chunks
//! - safely dropping down to platform-specific intrinsics
//!
//! It follows the usual Fearless SIMD structure:
//!
//! - write the main computation as an `#[inline(always)]` function generic over
//!   [`Simd`];
//! - use [`dispatch!`] at the non-SIMD boundary to run it with the best
//!   available target features;
//! - drop down to [`kernel!`](fearless_simd::kernel) when a small part of the
//!   computation needs a target-specific intrinsic.
//!
//! The RGB channels are converted with portable SIMD operations. The alpha
//! channel is copied unchanged, using an architecture-specific lane-copy
//! intrinsic if one is available and a scalar fallback otherwise.

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{float32x4_t, vcopyq_laneq_f32};
#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128, _mm_blend_ps};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128, _mm_blend_ps};

fearless_simd::kernel!(
    /// Copy the alpha lane on AArch64 using a NEON lane-copy intrinsic.
    #[inline]
    fn copy_alpha_neon(neon: Neon, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vcopyq_laneq_f32::<3, 3>(a, b)
    }
);

fearless_simd::kernel!(
    /// Copy the alpha lane on x86 using the SSE4.2 token to enable SSE4.1 blend instructions.
    #[inline]
    fn copy_alpha_sse4_2(sse4_2: Sse4_2, a: __m128, b: __m128) -> __m128 {
        _mm_blend_ps::<8>(a, b)
    }
);

/// Return `a` with its alpha channel replaced by `b`'s alpha channel.
///
/// This helper shows how portable SIMD code can opportunistically call
/// target-specific kernels while still providing a fallback for every backend.
#[inline(always)] // or #[simd], either works
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(sse4_2) = a.simd.level().as_sse4_2() {
        return copy_alpha_sse4_2(sse4_2, a.into(), b.into()).simd_into(a.simd);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = a.simd.level().as_neon() {
        return copy_alpha_neon(neon, a.into(), b.into()).simd_into(a.simd);
    }

    let mut result = a;
    result[3] = b[3];
    result
}

/// Approximate the linear-RGB to sRGB transfer curve for RGB, preserving alpha.
#[inline(always)] // or #[simd], either works
fn to_srgb<S: Simd>(simd: S, rgba: [f32; 4]) -> [f32; 4] {
    let v: f32x4<S> = rgba.simd_into(simd);
    let vabs = v.abs();
    let x = vabs - 5.358_626_4e-4;
    let x2 = x * x;
    let even1 = x * -9.127_959e-1 + -2.881_431_4e-2;
    let even2 = x2 * -7.291_929e-1 + even1;
    let odd1 = x * 1.061_331_7 + 1.401_945_4;
    let odd2 = x2 * 2.077_583e-1 + odd1;
    let poly = odd2 * x.sqrt() + even2;
    let lin = vabs * 12.92;
    let z = vabs.simd_gt(0.0031308).select(poly, lin);
    let z_signed = z.copysign(v);
    let result = copy_alpha(z_signed, v);
    result.into()
}

fn main() {
    let level = Level::new();
    let rgba = [0.1, -0.2, 0.001, 0.4];
    let srgb = dispatch!(level, simd=> to_srgb(simd, rgba));
    println!("{srgb:?}");
}
