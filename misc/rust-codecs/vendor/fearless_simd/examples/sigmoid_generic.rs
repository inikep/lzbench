// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Apply the sigmoid function `x / sqrt(x * x + 1)` using either `f32` or `f64`.
//!
//! `SimdFloatElement::Native<S>` selects the native-width vector for the given
//! scalar type and the current SIMD backend.

use fearless_simd::{Level, dispatch, prelude::*};

fn sigmoid<T: SimdFloatElement>(level: Level, input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output lengths must match"
    );
    dispatch!(level, simd => sigmoid_simd(simd, input, output));
}

#[inline(always)]
fn sigmoid_simd<S: Simd, T: SimdFloatElement>(simd: S, input: &[T], output: &mut [T]) {
    let n = T::Native::<S>::LEN;
    let one = T::Native::<S>::splat(simd, T::from(1_u8));
    let mut inputs = input.chunks_exact(n);
    let mut outputs = output.chunks_exact_mut(n);

    for (input, output) in inputs.by_ref().zip(outputs.by_ref()) {
        let a = T::Native::<S>::from_slice(simd, input);
        let b = a / (a * a + one).sqrt();
        b.store_slice(output);
    }

    let input = inputs.remainder();
    let output = outputs.into_remainder();
    if !input.is_empty() {
        // Padding keeps sqrt in SIMD, without requiring a scalar sqrt trait.
        let a = T::Native::<S>::from_fn(simd, |i| input.get(i).copied().unwrap_or_default());
        let b = a / (a * a + one).sqrt();
        output.copy_from_slice(&b.as_slice()[..input.len()]);
    }
}

fn main() {
    let level = Level::new();
    let input_f32 = [
        0.1_f32, -0.2, 0.001, 0.4, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
    ];
    let input_f64 = input_f32.map(f64::from);
    let mut output_f32 = [0.0; 17];
    let mut output_f64 = [0.0; 17];

    sigmoid(level, &input_f32, &mut output_f32);
    sigmoid(level, &input_f64, &mut output_f64);

    println!("f32: {output_f32:?}");
    println!("f64: {output_f64:?}");
}
