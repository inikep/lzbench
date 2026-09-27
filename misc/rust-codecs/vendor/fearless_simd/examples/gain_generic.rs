// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Apply gain to floating-point vectors of different widths and element types.

use fearless_simd::{Level, dispatch, f32x4, f32x16, f64x2, prelude::*};

// V::Element makes the gain match the vector's scalar type: f32 or f64.
#[inline(always)] // or #[simd], either works
fn apply_gain<S: Simd, V: SimdFloat<S>>(samples: V, gain: V::Element) -> V {
    samples * gain
}

fn main() {
    let level = Level::new();
    dispatch!(level, simd => {
        // f32 vectors can be any length
        let samples = f32x4::from_slice(simd, &[0.1, -0.2, 0.3, -0.4]);
        let output = apply_gain(samples, 0.5);
        println!("f32x4: {output:?}");

        let samples = f32x16::from_slice(
            simd,
            &[
                0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
                0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
            ],
        );
        let output = apply_gain(samples, 0.5);
        println!("f32x16: {output:?}");

        // f64 vectors work too through the same helper
        let samples = f64x2::from_slice(simd, &[0.1, -0.2]);
        let output = apply_gain(samples, 0.5);
        println!("f64x2: {output:?}");
    });
}
