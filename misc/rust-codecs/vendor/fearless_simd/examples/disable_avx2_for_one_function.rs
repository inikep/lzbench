// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This example demonstrates a SIMD function that's never compiled for AVX2.
//!
//! This can be useful if benchmarks show a specific instruction set regressing performance.

use fearless_simd::{Level, dispatch, prelude::*};

#[inline(always)] // or #[simd], either works
fn disable_avx2<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    let level = simd.level();
    match level {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // Downgrade AVX2 to SSE4.2 when calling `sigmoid()`
        Level::Avx2(_) => sigmoid(level.as_sse4_2().unwrap(), x, out),
        _ => sigmoid(simd, x, out),
    }
}

#[inline(always)]
fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    debug_assert!(!matches!(simd.level(), Level::Avx2(_)), "Called with AVX2");

    // If this function calls anything else with its `simd`,
    // the callee will also never get an AVX2 implementation.
    // The downgrade only needs to happen once anywhere in the call chain.

    let n = S::f32s::LEN;
    for (x, y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let a = S::f32s::from_slice(simd, x);
        let b = a / (a * a + 1.0).sqrt();
        b.store_slice(y);
    }
}

fn main() {
    let level = Level::new();
    let inp = [
        0.1, -0.2, 0.001, 0.4, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
    ];
    let mut out = [0.; 16];
    dispatch!(level, simd => disable_avx2(simd, &inp, &mut out));

    println!("{out:?}");
}
