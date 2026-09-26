<div align="center">

# Fearless SIMD

**Safer and easier SIMD**

[![Latest published version.](https://img.shields.io/crates/v/fearless_simd.svg)](https://crates.io/crates/fearless_simd)
[![Documentation build status.](https://img.shields.io/docsrs/fearless_simd.svg)](https://docs.rs/fearless_simd)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/fearless_simd/ci.yml?logo=github&label=CI)](https://github.com/linebender/fearless_simd/actions)
[![Dependency staleness status.](https://deps.rs/crate/fearless_simd/latest/status.svg)](https://deps.rs/crate/fearless_simd/)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=fearless_simd
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here. 
See https://linebender.org/blog/doc-include/ for related discussion. -->

[libm]: https://crates.io/crates/libm
[`f32x4`]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_types/struct.f32x4.html
[`Simd`]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_trait/trait.Simd.html
[`SimdFrom`]: https://docs.rs/fearless_simd/latest/fearless_simd/traits/trait.SimdFrom.html
[SimdBase::from_slice]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_trait/trait.SimdBase.html#tymethod.from_slice
[`dispatch`]: https://docs.rs/fearless_simd/latest/fearless_simd/macro.dispatch.html
[`Level`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html
[`Level::new`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html#method.new
[`Level::dispatch`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html#method.dispatch
[`std::simd`]: https://doc.rust-lang.org/std/simd/index.html
[kernel]: https://docs.rs/fearless_simd/latest/fearless_simd/macro.kernel.html
[Simd::vectorize]: https://docs.rs/fearless_simd/latest/fearless_simd/trait.Simd.html#tymethod.vectorize

<!-- cargo-rdme start -->

`fearless_simd` takes `unsafe` out of SIMD.

No matter what level of abstraction you're after, be it autovectorization and multiversioning, or portable SIMD, or safe access to raw
intrinsics and nothing more, `fearless_simd` has you covered!

The core crate has zero dependencies, safe public APIs, and
[very little](https://shnatsel.github.io/safe-simd-in-rust-even-on-the-inside/)
`unsafe` under the hood.

## Usage

Add the core library and optional `#[simd]` macro to your `Cargo.toml`:

```toml
[dependencies]
fearless_simd = "1.0"
fearless_simd_macros = "0.1"
```

## Automatic vectorization

The easiest way to define a SIMD-generic function is the
[`#[simd]`](https://docs.rs/fearless_simd_macros/latest/fearless_simd_macros/attr.simd.html)
attribute from the separately versioned `fearless_simd_macros` crate. The companion macro
crate is optional: `fearless_simd` does not depend on it, so users of only the core API do not
pay for its procedural-macro dependencies.

[`dispatch!`] generates implementations for the available SIMD levels and selects the best one
at runtime:

```rust
use fearless_simd::{dispatch, Level, Simd};
use fearless_simd_macros::simd;

#[simd]
fn double_u32s<S: Simd>(_: S, values: &mut [u32]) {
    for value in values {
        *value = *value * 2;
    }
}

let mut values = [1, 2, 3, 4, 5];
let level = Level::new();
dispatch!(level, simd => double_u32s(simd, &mut values));
assert_eq!(values, [2, 4, 6, 8, 10]);
```

## Portable SIMD

Use the vector types for explicit lane-wise operations while staying generic over the SIMD level:

```rust
use fearless_simd::{dispatch, prelude::*, Level};
use fearless_simd_macros::simd;

#[simd]
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    let mut chunks = values.chunks_exact_mut(S::u32s::LEN); // the CPU's native SIMD width
    for chunk in &mut chunks {
        let v = S::u32s::from_slice(simd, chunk);
        (v * 2).store_slice(chunk);
    }
    for value in chunks.into_remainder() {
        *value = *value * 2;
    }
}

let mut values = [1, 2, 3, 4, 5];
let level = Level::new();
dispatch!(level, simd => double_u32s(simd, &mut values));
assert_eq!(values, [2, 4, 6, 8, 10]);
```

You can also use fixed-size types such as [u32x8] instead of using the hardware's native SIMD width.

## Explicit intrinsics

If you need access to raw intrinsics, [`kernel!`][kernel] creates a function where they can be called safely:

```rust
use fearless_simd::{prelude::*, Level, u32x4};

fearless_simd::kernel!(
    fn double_u32s_neon(neon: Neon, values: &mut [u32]) {
        use core::arch::aarch64::*;

        let mut chunks = values.chunks_exact_mut(4);
        for chunk in &mut chunks {
            let v: uint32x4_t = u32x4::from_slice(neon, chunk).into(); // safe load
            let doubled = vmulq_u32(v, vdupq_n_u32(2)); // safe access to a NEON intrinsic
            let doubled: u32x4<_> = doubled.simd_into(neon);
            doubled.store_slice(chunk);
        }
        for value in chunks.into_remainder() {
            *value = *value * 2;
        }
    }
);

#[cfg(target_arch = "aarch64")]
{
    let level = Level::new();
    if let Some(neon) = level.as_neon() {
        let mut values = [1, 2, 3, 4, 5];
        double_u32s_neon(neon, &mut values);
        assert_eq!(values, [2, 4, 6, 8, 10]);
    }
}
```

You can also [mix and match](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs)
intrinsics with the other approaches, using high-level code most of the time and dropping down to
hardware-specific intrinsics only when necessary.

### The `#[simd]` annotation

It is recommended to annotate functions that use SIMD with the `#[simd]` attribute from the `fearless_simd_macros` crate. Without it the code will still compile, but requires special care to achieve full performance.

If you cannot use proc macros, [you can achieve the same effect manually](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/MANUAL_INLINING.md), but it requires some care. The use of `#[simd]` is recommended as the more robust and ergonomic option.

## Instruction set support

- x86/x86-64: SSE2 baseline, [v2](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (SSE4.2), [v3](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (AVX2), [Ice Lake](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) (AVX-512, avoiding early slow implementations)
- Aarch64: Baseline [NEON](https://en.wikipedia.org/wiki/Arm_architecture_family#Advanced_SIMD_(Neon))
- WebAssembly: [128-bit packed SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md), [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)

A scalar fallback is also provided for platforms, so your code still works even if SIMD is not available.

## WebAssembly

WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
then select the appropriate one for your user's browser. This can be done via [the `wasm-feature-detect`
library](https://github.com/GoogleChromeLabs/wasm-feature-detect).

You can compile WebAssembly with the SIMD128 feature enabled via the `RUSTFLAGS` environment variable
(`RUSTFLAGS="-Ctarget-feature=+simd128"`), or by adding the compiler flags in your [Cargo
config.toml](https://doc.rust-lang.org/cargo/reference/config.html):

```toml
[target.'cfg(target_arch = "wasm32")']
rustflags = ["-Ctarget-feature=+simd128"]
rustdocflags = ["-Ctarget-feature=+simd128"]
```

If you want to compile both SIMD and non-SIMD versions of your WebAssembly library, your best option right now is to create a shell script
that builds it once with the `RUSTFLAGS` specified, and once without. [Cargo currently does not allow specifying compiler flags
per-profile.](https://github.com/rust-lang/cargo/issues/10271)

### Relaxed SIMD

Fearless SIMD can make use of the [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
WebAssembly instructions, if the requisite target feature is enabled. These instructions can return implementation-dependent results
depending on what is fastest on the underlying hardware. They are only used for operations where we already give hardware-dependent results.

At the time of writing, relaxed SIMD is only supported in Chrome. To make use of it, you'll need to build two versions of your library, one
with relaxed SIMD enabled (`RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd"`) and one with it disabled, and then feature-detect at
runtime.

## Multiversioning on x86

x86 CPUs are not guaranteed to have any SIMD particular instruction set, so `fearless_simd` compiles a version
of each function generic over [`Simd`] for each instruction set, and [`dispatch`] selects the best one at runtime.

This is necessary to take advantage of SIMD, but results in an increased binary size on x86.
If binary size is a concern, the increase can be partially mitigated by setting
[`codegen-units=1`](https://nnethercote.github.io/perf-book/build-configuration.html#codegen-units)
or [`lto=true`](https://nnethercote.github.io/perf-book/build-configuration.html#link-time-optimization) in your Cargo.toml,
at the cost of longer build times.

As a last resort, you can turn off multiversioning for specific SIMD instruction sets by passing
`--cfg disable_dispatch_sse2`, `--cfg disable_dispatch_sse4_2`, `--cfg disable_dispatch_avx2`, or `--cfg disable_dispatch_avx512` in `RUSTFLAGS`.
These configuration flags only control automatic multiversioning. Disabling one does not remove its token type, its
[`Simd`] implementation, or explicit [`kernel`] support; for example, an `Avx2` token can still be used to call an
AVX2 kernel when the CPU supports it.
`disable_dispatch_sse2` has no effect when SSE2 is part of the ambient target baseline, because
that baseline remains the terminal dispatch backend.

Note that later extensions can be beneficial even if you are only using 128-bit vectors:
AVX2 and AVX-512 provide more efficient instructions for some operations,
and AVX-512 also more than doubles the number of vector registers of all sizes.

You can also [disable certain instruction sets for select functions](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/disable_avx2_for_one_function.rs)
without disabling them globally.

## Feature Flags

The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:

- `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
  Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
- `libm`: Use floating point implementations from [libm]. Useful for `#[no_std]`.
- `force_support_fallback`: Force scalar fallback, to be supported, even if your compilation target has a better baseline.

At least one of `std` and `libm` is required; `std` overrides `libm`.

## Credits

This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

[`pulp`]: https://crates.io/crates/pulp

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.89** and later.

Future versions of Fearless SIMD might increase the Rust version requirement. This will be accompanied by a minor version bump.

If you require a fixed MSRV, we recommend using Cargo's [MSRV-aware resolver](https://doc.rust-lang.org/edition-guide/rust-2024/cargo-resolver.html) which will not select a version that fails to build.
We will provide [security backports](SECURITY.md) to older Rust versions released within the last 3 years.

## Community

[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)

Discussion of Fearless SIMD development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically in [#simd](https://xi.zulipchat.com/#narrow/channel/514230-simd).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct
