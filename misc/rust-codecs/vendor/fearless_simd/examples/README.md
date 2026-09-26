# Fearless SIMD examples

## sigmoid

This example demonstrates the common case: process data in chunks that are optimal for the CPU it's running on.

## sigmoid_generic

This example uses `SimdFloatElement::Native<S>` to process both `f32` and `f64` with the same generic function. It uses the host's native SIMD vectors for optimal performance regardless of the CPU.

## gain_generic

This example uses a helper generic over `SimdFloat` to apply gain to any kind of floating-point vector.

## sRGB

The sRGB example demonstrates:

1. processing data in fixed-sized chunks
2. dropping down to platform intrinsics for a part of the computation

# disable AVX2 for one function

Disables a specific SIMD instruction set for a single function.

See the [crate's README](../README.md) for other ways to control multiversioning, including global controls.
