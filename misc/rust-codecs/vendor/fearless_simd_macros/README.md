# Fearless SIMD macros

This crate provides the `#[simd]` attribute for
[`fearless_simd`](https://crates.io/crates/fearless_simd). It is versioned
separately so that the macro can evolve without adding a procedural-macro
dependency to `fearless_simd` itself.

Add both packages from crates.io to your `Cargo.toml`:

```toml
[dependencies]
fearless_simd = "1.0"
fearless_simd_macros = "0.1"
```

The library must be in scope as `fearless_simd` in the module containing the
annotated function. The dependency declaration above makes that name available
automatically. `fearless_simd` 1.0 or later is required.

Apply `#[simd]` to a function whose first argument carries its SIMD token:

```rust,ignore
use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

#[simd]
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    let mut chunks = values.chunks_exact_mut(S::u32s::LEN);
    for chunk in &mut chunks {
        let value = S::u32s::from_slice(simd, chunk);
        (value * 2).store_slice(chunk);
    }
    for value in chunks.into_remainder() {
        *value *= 2;
    }
}
```

Conceptually, the macro expands the body to:

```rust,ignore
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    ExtractToken::token(&simd).vectorize(
        #[inline(always)]
        || {
            // Original body.
        },
    )
}
```

The actual implementation is more involved, because arguments captured by the
closure are passed as a struct and are often stored on the stack instead of
being passed in registers, which incurs some overhead. We avoid that overhead
by expanding into a more verbose but slightly more optimal code.

Vectors and masks already carry their token, so they do not need a separate
token argument. Both fixed-width and native-width types work:

```rust,ignore
use fearless_simd::{f32x4, prelude::*};
use fearless_simd_macros::simd;

#[simd]
fn double<S: Simd>(value: f32x4<S>) -> f32x4<S> {
    value + value
}

#[simd]
fn double_native<S: Simd>(value: S::f32s) -> S::f32s {
    value + value
}
```

User-defined wrappers can implement the public `ExtractToken` trait. They do
not need to implement `Copy` or any of the vector operation traits:

```rust,ignore
use fearless_simd::{f32x4, prelude::*};
use fearless_simd_macros::simd;

// Four consecutive audio samples, processed together.
struct AudioSamples<S: Simd>(f32x4<S>);

impl<S: Simd> ExtractToken for AudioSamples<S> {
    type S = S;

    #[inline]
    fn token(&self) -> S {
        self.0.token()
    }
}

#[simd]
fn apply_gain<S: Simd>(samples: AudioSamples<S>, gain: f32) -> AudioSamples<S> {
    AudioSamples(samples.0 * gain)
}
```

## Accepted functions

The first typed parameter after an optional `self` receiver is treated as the
SIMD token carrier. Its type must implement `fearless_simd::ExtractToken`;
tokens, vectors, masks, and shared or mutable references to carriers are supported.

An unused carrier may be written as `_: S`; the macro gives it a
private hygienic binding. Destructured and `binding @ pattern`
parameters are not supported for the carrier. Neither `#[cfg]` nor `#[cfg_attr]`
may be placed on that parameter.

The macro accepts synchronous free functions, inherent methods, trait
implementation methods, and default trait methods. Generic parameters, `where`
clauses, return types, `unsafe`, and non-variadic `extern` ABIs are preserved.

`async`, `const`, variadic, bodyless, and specialization `default fn`
functions are rejected. The attributes `#[track_caller]`, `#[unsafe(naked)]`,
and `#[instruction_set]` are also rejected because moving the body into a
closure would invalidate their semantics or body requirements.
Other attributes are preserved.

A trait's `#[track_caller]` attribute is inherited by its implementations, but
is not present in the implementation method's token stream when this macro
runs. Applying `#[simd]` to an implementation of a trait method declared with
`#[track_caller]` is therefore unsupported even though the macro cannot
diagnose it.

## Execution boundaries and captures

The macro calls `ExtractToken::token(&first_argument)` exactly once, before
entering the selected SIMD context, then forwards the original argument to the
body. This borrows the carrier without copying it. Custom `token()`
implementations should be cheap and must not assume the caller already has the
token's target features enabled. An inherent method named `token` is ignored.

Only work performed while the function body is executing is covered by the
SIMD context. Code inside a returned future, closure, or lazy iterator runs
later and is not covered. Named helper functions do not inherit the enabled
target features; make them inlineable or annotate their own SIMD-generic body.
Recursive calls enter the dispatcher again.

The original body becomes an always-inline `FnOnce` closure with explicit
parameters. Receivers and parameters carrying `#[cfg]` or `#[cfg_attr]` remain
captures, preserving their existing semantics and uses inside nested macros.
These captures can still require memory when a helper remains out of line.
Rust infers their capture modes from how the body uses them. As with any closure
conversion, the destruction order of captured values is not a stable
substitute for function-parameter destruction order. Avoid relying on the
relative drop order of by-value parameters with observable destructors in a
`#[simd]` function.

The extracted token must implement the library's `Simd` trait, as required by
`ExtractToken::S`. An `S: Simd` bound already implies `ExtractToken<S = S>`;
`SimdBase<S>` and `SimdMask<S>` also imply `ExtractToken<S = S>`.
The procedural macro invokes
`fearless_simd::__fearless_simd_dispatch!`; the procedural-macro crate itself
does not depend on `fearless_simd`. The library helper owns the unsafe calls
and resolves all proof types through `$crate`, so a lookalike module cannot
substitute counterfeit proof tokens when re-exporting that helper.
Unknown future backends retain the existing `Simd::vectorize` path until the
library helper is updated to generate entries for them.

## Minimum supported Rust version

This version of `fearless_simd_macros` has been verified to compile with Rust
1.89 and later. Future versions may increase this requirement.
