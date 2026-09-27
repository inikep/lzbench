// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This example demonstrates the typical usage Fearless SIMD.
//!
//! The vector size matches the native vector size of the hardware:
//!
//! - SSE and NEON get 128 bit chunks
//! - AVX2 gets 256 bit ones
//! - AVX-512 gets 512-bit ones
//!
//! All from a single function.

use fearless_simd::{Level, dispatch, prelude::*};

/// Applies the sigmoid function to the input and writes to the output
#[inline(always)] // or #[simd], either works
fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    let n = S::f32s::LEN; // CPU's native vector size

    // fast vectorized loop
    for (x, y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let a = S::f32s::from_slice(simd, x);
        let b = a / (a * a + 1.0).sqrt();
        b.store_slice(y);
    }

    // scalar processing of the remainder smaller than a single vector
    let x_remainder = x.chunks_exact(n).remainder();
    let y_remainder = out.chunks_exact_mut(n).into_remainder();
    for (a, b) in x_remainder.iter().zip(y_remainder) {
        *b = a / (a * a + 1.0).sqrt();
    }
}

fn main() {
    let level = Level::new();
    let inp = [
        0.1, -0.2, 0.001, 0.4, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
    ];
    let mut out = [0.; 18];
    // dispatch! selects the best implementation for the CPU we're running on
    dispatch!(level, simd => sigmoid(simd, &inp, &mut out));

    println!("{out:?}");
}
