// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=fearless_simd

//! `fearless_simd` takes `unsafe` out of SIMD.
//!
//! No matter what level of abstraction you're after, be it autovectorization and multiversioning, or portable SIMD, or safe access to raw
//! intrinsics and nothing more, `fearless_simd` has you covered!
//!
//! The core crate has zero dependencies, safe public APIs, and
//! [very little](https://shnatsel.github.io/safe-simd-in-rust-even-on-the-inside/)
//! `unsafe` under the hood.
//!
//! # Usage
//!
//! Add the core library and optional `#[simd]` macro to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! fearless_simd = "1.0"
//! fearless_simd_macros = "0.1"
//! ```
//!
//! # Automatic vectorization
//!
//! The easiest way to define a SIMD-generic function is the
//! [`#[simd]`](https://docs.rs/fearless_simd_macros/latest/fearless_simd_macros/attr.simd.html)
//! attribute from the separately versioned `fearless_simd_macros` crate. The companion macro
//! crate is optional: `fearless_simd` does not depend on it, so users of only the core API do not
//! pay for its procedural-macro dependencies.
//!
//! [`dispatch!`] generates implementations for the available SIMD levels and selects the best one
//! at runtime:
//!
//! ```ignore
//! use fearless_simd::{dispatch, Level, Simd};
//! use fearless_simd_macros::simd;
//!
//! #[simd]
//! fn double_u32s<S: Simd>(_: S, values: &mut [u32]) {
//!     for value in values {
//!         *value = *value * 2;
//!     }
//! }
//!
//! let mut values = [1, 2, 3, 4, 5];
//! let level = Level::new();
//! dispatch!(level, simd => double_u32s(simd, &mut values));
//! assert_eq!(values, [2, 4, 6, 8, 10]);
//! ```
//!
//! # Portable SIMD
//!
//! Use the vector types for explicit lane-wise operations while staying generic over the SIMD level:
//!
//! ```ignore
//! use fearless_simd::{dispatch, prelude::*, Level};
//! use fearless_simd_macros::simd;
//!
//! #[simd]
//! fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
//!     let mut chunks = values.chunks_exact_mut(S::u32s::LEN); // the CPU's native SIMD width
//!     for chunk in &mut chunks {
//!         let v = S::u32s::from_slice(simd, chunk);
//!         (v * 2).store_slice(chunk);
//!     }
//!     for value in chunks.into_remainder() {
//!         *value = *value * 2;
//!     }
//! }
//!
//! let mut values = [1, 2, 3, 4, 5];
//! let level = Level::new();
//! dispatch!(level, simd => double_u32s(simd, &mut values));
//! assert_eq!(values, [2, 4, 6, 8, 10]);
//! ```
//!
//! You can also use fixed-size types such as [u32x8] instead of using the hardware's native SIMD width.
//!
//! # Explicit intrinsics
//!
//! If you need access to raw intrinsics, [`kernel!`][kernel] creates a function where they can be called safely:
//!
//! ```rust
//! use fearless_simd::{prelude::*, Level, u32x4};
//!
//! fearless_simd::kernel!(
//!     fn double_u32s_neon(neon: Neon, values: &mut [u32]) {
//!         use core::arch::aarch64::*;
//!
//!         let mut chunks = values.chunks_exact_mut(4);
//!         for chunk in &mut chunks {
//!             let v: uint32x4_t = u32x4::from_slice(neon, chunk).into(); // safe load
//!             let doubled = vmulq_u32(v, vdupq_n_u32(2)); // safe access to a NEON intrinsic
//!             let doubled: u32x4<_> = doubled.simd_into(neon);
//!             doubled.store_slice(chunk);
//!         }
//!         for value in chunks.into_remainder() {
//!             *value = *value * 2;
//!         }
//!     }
//! );
//!
//! #[cfg(target_arch = "aarch64")]
//! {
//!     let level = Level::new();
//!     if let Some(neon) = level.as_neon() {
//!         let mut values = [1, 2, 3, 4, 5];
//!         double_u32s_neon(neon, &mut values);
//!         assert_eq!(values, [2, 4, 6, 8, 10]);
//!     }
//! }
//! ```
//!
//! You can also [mix and match](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs)
//! intrinsics with the other approaches, using high-level code most of the time and dropping down to
//! hardware-specific intrinsics only when necessary.
//!
//! ## The `#[simd]` annotation
//!
//! It is recommended to annotate functions that use SIMD with the `#[simd]` attribute from the `fearless_simd_macros` crate. Without it the code will still compile, but requires special care to achieve full performance.
//!
//! If you cannot use proc macros, [you can achieve the same effect manually](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/MANUAL_INLINING.md), but it requires some care. The use of `#[simd]` is recommended as the more robust and ergonomic option.
//!
//! # Instruction set support
//!
//! - x86/x86-64: SSE2 baseline, [v2](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (SSE4.2), [v3](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (AVX2), [Ice Lake](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) (AVX-512, avoiding early slow implementations)
//! - Aarch64: Baseline [NEON](https://en.wikipedia.org/wiki/Arm_architecture_family#Advanced_SIMD_(Neon))
//! - WebAssembly: [128-bit packed SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md), [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
//!
//! A scalar fallback is also provided for platforms, so your code still works even if SIMD is not available.
//!
//! # WebAssembly
//!
//! WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
//! then select the appropriate one for your user's browser. This can be done via [the `wasm-feature-detect`
//! library](https://github.com/GoogleChromeLabs/wasm-feature-detect).
//!
//! You can compile WebAssembly with the SIMD128 feature enabled via the `RUSTFLAGS` environment variable
//! (`RUSTFLAGS="-Ctarget-feature=+simd128"`), or by adding the compiler flags in your [Cargo
//! config.toml](https://doc.rust-lang.org/cargo/reference/config.html):
//!
//! ```toml
//! [target.'cfg(target_arch = "wasm32")']
//! rustflags = ["-Ctarget-feature=+simd128"]
//! rustdocflags = ["-Ctarget-feature=+simd128"]
//! ```
//!
//! If you want to compile both SIMD and non-SIMD versions of your WebAssembly library, your best option right now is to create a shell script
//! that builds it once with the `RUSTFLAGS` specified, and once without. [Cargo currently does not allow specifying compiler flags
//! per-profile.](https://github.com/rust-lang/cargo/issues/10271)
//!
//! ## Relaxed SIMD
//!
//! Fearless SIMD can make use of the [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
//! WebAssembly instructions, if the requisite target feature is enabled. These instructions can return implementation-dependent results
//! depending on what is fastest on the underlying hardware. They are only used for operations where we already give hardware-dependent results.
//!
//! At the time of writing, relaxed SIMD is only supported in Chrome. To make use of it, you'll need to build two versions of your library, one
//! with relaxed SIMD enabled (`RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd"`) and one with it disabled, and then feature-detect at
//! runtime.
//!
//! # Multiversioning on x86
//!
//! x86 CPUs are not guaranteed to have any SIMD particular instruction set, so `fearless_simd` compiles a version
//! of each function generic over [`Simd`] for each instruction set, and [`dispatch`] selects the best one at runtime.
//!
//! This is necessary to take advantage of SIMD, but results in an increased binary size on x86.
//! If binary size is a concern, the increase can be partially mitigated by setting
//! [`codegen-units=1`](https://nnethercote.github.io/perf-book/build-configuration.html#codegen-units)
//! or [`lto=true`](https://nnethercote.github.io/perf-book/build-configuration.html#link-time-optimization) in your Cargo.toml,
//! at the cost of longer build times.
//!
//! As a last resort, you can turn off multiversioning for specific SIMD instruction sets by passing
//! `--cfg disable_dispatch_sse2`, `--cfg disable_dispatch_sse4_2`, `--cfg disable_dispatch_avx2`, or `--cfg disable_dispatch_avx512` in `RUSTFLAGS`.
//! These configuration flags only control automatic multiversioning. Disabling one does not remove its token type, its
//! [`Simd`] implementation, or explicit [`kernel`] support; for example, an `Avx2` token can still be used to call an
//! AVX2 kernel when the CPU supports it.
//! `disable_dispatch_sse2` has no effect when SSE2 is part of the ambient target baseline, because
//! that baseline remains the terminal dispatch backend.
//!
//! Note that later extensions can be beneficial even if you are only using 128-bit vectors:
//! AVX2 and AVX-512 provide more efficient instructions for some operations,
//! and AVX-512 also more than doubles the number of vector registers of all sizes.
//!
//! You can also [disable certain instruction sets for select functions](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/disable_avx2_for_one_function.rs)
//! without disabling them globally.
//!
//! # Feature Flags
//!
//! The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:
//!
//! - `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
//!   Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
//! - `libm`: Use floating point implementations from [libm]. Useful for `#[no_std]`.
//! - `force_support_fallback`: Force scalar fallback, to be supported, even if your compilation target has a better baseline.
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//!
//! # Credits
//!
//! This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
//! It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.
//!
//! [`pulp`]: https://crates.io/crates/pulp
#![cfg_attr(feature = "libm", doc = "[libm]: libm")]
#![cfg_attr(not(feature = "libm"), doc = "[libm]: https://crates.io/crates/libm")]
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(not(test), deny(clippy::disallowed_methods))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(non_camel_case_types, reason = "TODO")]
#![expect(clippy::unused_unit, reason = "easier for code generation")]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(all(not(feature = "libm"), not(feature = "std")))]
compile_error!("fearless_simd requires either the `std` or `libm` feature");

// Some native SIMD backends do not need libm unless fallback support is also enabled.
// Keep the optional dependency marked as used for every supported target configuration.
#[cfg(feature = "libm")]
use libm as _;

mod generated;
mod kernel_macros;
mod macros;
mod support;
mod traits;
mod transmute;

pub use generated::*;
pub use traits::*;

/// This prelude module re-exports every SIMD trait defined in this library. It's useful for accessing trait methods.
///
/// Only traits are exported through the prelude; types must be exported separately.
pub mod prelude {
    pub use crate::generated::simd_trait::*;
    pub use crate::traits::*;
}

/// Implementations of [`Simd`] for 64 bit ARM.
#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    pub use crate::generated::Neon;
}

/// Implementations of [`Simd`] for webassembly.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32 {
    pub use crate::generated::WasmSimd128;
}

/// Implementations of [`Simd`] on x86 architectures (both 32 and 64 bit).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    pub use crate::generated::Avx2;
    pub use crate::generated::Avx512;
    pub use crate::generated::Sse2;
    pub use crate::generated::Sse4_2;
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
use std::sync::LazyLock;

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
/// Caches the CPU level detection on x86.
// Aligned128 is used to avoid the data crossing cache line boundary,
// which would incur two cache misses instead of one.
static X86_LEVEL: support::Aligned128<LazyLock<Level>> =
    support::Aligned128(LazyLock::new(detect_x86_level));

// Sourced from `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-cpu=icelake-server`
// and pruned against the features implied by `avx512f` which can be viewed via
// `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-feature='+avx2'`
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn x86_detects_icelake_avx512() -> bool {
    std::arch::is_x86_feature_detected!("adx")
        && std::arch::is_x86_feature_detected!("aes")
        && std::arch::is_x86_feature_detected!("avx512bitalg")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512ifma")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
        && std::arch::is_x86_feature_detected!("avx512vbmi2")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("cmpxchg16b")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("fxsr")
        && std::arch::is_x86_feature_detected!("gfni")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
        && std::arch::is_x86_feature_detected!("popcnt")
        && std::arch::is_x86_feature_detected!("rdrand")
        && std::arch::is_x86_feature_detected!("rdseed")
        && std::arch::is_x86_feature_detected!("sha")
        && std::arch::is_x86_feature_detected!("vaes")
        && std::arch::is_x86_feature_detected!("vpclmulqdq")
        && std::arch::is_x86_feature_detected!("xsave")
        && std::arch::is_x86_feature_detected!("xsavec")
        && std::arch::is_x86_feature_detected!("xsaveopt")
        && std::arch::is_x86_feature_detected!("xsaves")
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[cold]
fn detect_x86_level() -> Level {
    if x86_detects_icelake_avx512() {
        // Safety: All features required by Avx512 were detected above.
        unsafe { Level::Avx512(Avx512::assume_supported()) }
    // Feature list sourced from `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-cpu=x86-64-v3`
    // However, the following features are implied by avx2 and do not need to be spelled out:
    // avx,sse,sse2,sse3,sse4.1,sse4.2,ssse3
    // This can be verified by running:
    // rustc --print=cfg --target=i586-unknown-linux-gnu -C target-feature=+avx2
    } else if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("cmpxchg16b")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("fxsr")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
        && std::arch::is_x86_feature_detected!("popcnt")
        && std::arch::is_x86_feature_detected!("xsave")
    {
        // Safety: All features required by Avx2 were detected above.
        unsafe { Level::Avx2(Avx2::assume_supported()) }
    // All x86 CPUs that ever shipped with sse4.2 also have cmpxchg16b and popcnt:
    // Intel Nehalem, AMD Bulldozer and VIA Isaiah II were the first with SSE4.2
    // and have these extensions already.
    //
    // This set of instructions maps to the x86-64-v2 level:
    // rustc --print=cfg --target=x86_64-unknown-linux-gnu -C target-cpu=x86-64-v2
    //
    // All SSE levels are implied by SSE4.2, which can be verified by running:
    // rustc --print=cfg --target=i586-unknown-linux-gnu -C target-feature=+sse4.2
    } else if std::arch::is_x86_feature_detected!("fxsr")
        && std::arch::is_x86_feature_detected!("sse4.2")
        && std::arch::is_x86_feature_detected!("cmpxchg16b")
        && std::arch::is_x86_feature_detected!("popcnt")
    {
        // Safety: All features required by Sse4_2 were detected above.
        unsafe { Level::Sse4_2(Sse4_2::assume_supported()) }
    } else if std::arch::is_x86_feature_detected!("sse2")
        && std::arch::is_x86_feature_detected!("fxsr")
    {
        // Safety: All features required by Sse2 were detected above.
        unsafe { Level::Sse2(Sse2::assume_supported()) }
    } else {
        Level::baseline()
    }
}

/// The level enum with the specific SIMD capabilities available.
///
/// The contained values serve as a proof that the associated target
/// feature is available.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Level {
    /// Scalar fallback level, i.e. no supported SIMD features are to be used.
    ///
    /// This variant is **absent** on targets that supports a higher baseline
    /// (`aarch64-*`, `i686-*`, `x86_64-*`, WASM with SIMD) unless the `force_support_fallback`
    /// Cargo feature is enabled. Instead of matching on this variant,
    /// call [`is_fallback`](Level::is_fallback) which is always available.
    ///
    /// This can be created with [`Level::fallback`].
    // Keep this predicate in sync with the fallback module and `dispatch!`.
    #[cfg(any(
        feature = "force_support_fallback",
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2",
                target_feature = "fxsr"
            ),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    Fallback(Fallback),
    /// The Neon instruction set on 64 bit ARM.
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    /// The SIMD 128 instructions on 32-bit WebAssembly.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    WasmSimd128(WasmSimd128),
    /// The SSE2 instruction set on (32 and 64 bit) x86.
    ///
    /// This is the baseline for i686 and x86-64 targets.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2(Sse2),
    /// The SSE4.2 instruction set on (32 and 64 bit) x86, plus `popcnt` and `cmpxchg16b`.
    /// Also known as x86-64-v2.
    ///
    /// All production CPUs with SSE4.2 also support the other two extensions, so it is safe to require them.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse4_2(Sse4_2),
    /// Ice Lake-class AVX-512 on (32 and 64 bit) x86.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512(Avx512),
    /// The x86-64-v3 instruction set on (32 and 64 bit) x86, including AVX2 and FMA.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(Avx2),
    // If new variants are added, make sure to handle them in `Level::dispatch`
    // and `dispatch!()`
}

impl Level {
    /// Return the best SIMD level available on the CPU. This value should be passed to [`dispatch`].
    ///
    /// On x86 and x86-64 targets, this detects the available CPU features on the first call and
    /// caches the result. Other targets return their strongest statically supported level.
    /// This may change in the future if runtime-detected levels for other platforms are added.
    ///
    /// This function requires the standard library on targets other than wasm32. On wasm32, the
    /// available level is known statically, so the standard library isn't required.
    ///
    /// On x86-64, it is sometimes possible to detect the available features on `#[no_std]`
    /// by parsing the output of `cpuid` instruction, but this function
    /// [does not do that](https://github.com/linebender/fearless_simd/issues/157).
    /// If you do this, you can create the SIMD token via [`assume_supported`](Avx2::assume_supported)
    /// and then get the [level](Simd::level) from it.
    ///
    /// Libraries that use SIMD on `#[no_std]` should let the user pass the appropriate SIMD level
    /// the user detected through other means (e.g. `cpuid`), to avoid using the fallback level
    /// when a better SIMD level is available in hardware.
    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[must_use]
    #[expect(
        clippy::new_without_default,
        reason = "The `Level::new()` function is not always available, and we also want to be explicit about when runtime feature detection happens"
    )]
    #[inline]
    pub fn new() -> Self {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            *X86_LEVEL.0
        }

        // targets other than x86 do not perform any runtime detection
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::baseline()
        }
    }

    /// Get the target feature level suitable for this run.
    ///
    /// Should be used in libraries if they wish to handle the case where
    /// target features cannot be detected at runtime.
    /// Most users should prefer [`new`](Self::new).
    /// This is discussed in more detail in `new`'s documentation.
    #[allow(clippy::allow_attributes, reason = "Only needed in some cfgs.")]
    #[allow(unreachable_code, reason = "Fallback unreachable in some cfgs.")]
    pub fn try_detect() -> Option<Self> {
        #[cfg(any(feature = "std", target_arch = "wasm32"))]
        return Some(Self::new());
        None
    }

    /// Check whether this is the `Fallback` level; that is, whether no better feature level could
    /// be statically or dynamically detected. This is useful if there's a scalarized version of
    /// your algorithm that runs faster if SIMD isn't supported.
    ///
    /// This method is always available, even when the fallback backend is not compiled. In that
    /// case, it always returns `false`.
    #[inline]
    pub fn is_fallback(self) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        return self.as_sse2().is_none();

        #[cfg(target_arch = "aarch64")]
        return self.as_neon().is_none();

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return self.as_wasm_simd128().is_none();

        #[cfg(any(
            all(target_arch = "wasm32", not(target_feature = "simd128")),
            not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            ))
        ))]
        return true;
    }

    /// If this is a proof that Neon (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `Neon` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of Neon,
    /// this method will return the Neon token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `neon`, there is only one variant."
        )]
        match self {
            Self::Neon(neon) => Some(neon),
            _ => None,
        }
    }

    /// If this is a proof that SIMD 128 (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `WasmSimd128` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of SIMD 128,
    /// this method will return the SIMD 128 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `simd128`, there is only one variant."
        )]
        match self {
            Self::WasmSimd128(simd128) => Some(simd128),
            _ => None,
        }
    }

    /// If this is a proof that SSE2 (or better) is available, access that instruction set.
    ///
    /// See [`Sse2::assume_supported`] for the exact list of CPU features this token enables.
    ///
    /// This method should be preferred over matching against the `Sse2` variant of self,
    /// because if the CPU supports a superset of SSE2 (e.g. SSE4.2, AVX2, or AVX-512),
    /// this method will return the SSE2 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_sse2(self) -> Option<Sse2> {
        match self {
            // Safety: Every stronger x86 SIMD level in this crate includes the `fxsr`,
            // `sse`, and `sse2` features required by Sse2.
            Self::Avx512(_avx512) => unsafe { Some(Sse2::assume_supported()) },
            Self::Avx2(_avx2) => unsafe { Some(Sse2::assume_supported()) },
            Self::Sse4_2(_sse4_2) => unsafe { Some(Sse2::assume_supported()) },
            Self::Sse2(sse2) => Some(sse2),
            #[allow(
                unreachable_patterns,
                reason = "This arm is reachable on x86 targets without SSE2."
            )]
            _ => None,
        }
    }

    /// If this is a proof that x86-64-v2 feature set (or better) is available, access that
    /// instruction set.
    ///
    /// See [`Sse4_2::assume_supported`] for the exact list of CPU features this token enables.
    ///
    /// This method should be preferred over matching against the `Sse4_2` variant of self,
    /// because if the CPU supports a superset of SSE4.2 (e.g. AVX2 or AVX-512),
    /// this method will return the SSE4.2 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_sse4_2(self) -> Option<Sse4_2> {
        match self {
            // Safety: The Avx512 struct represents an Ice Lake feature set, which includes the
            // `sse4.2`, `cmpxchg16b`, and `popcnt` features required by Sse4_2.
            Self::Avx512(_avx512) => unsafe { Some(Sse4_2::assume_supported()) },
            // Safety: The Avx2 struct represents the x86-64-v3 feature set being enabled, which
            // includes the `sse4.2`, `cmpxchg16b`, and `popcnt` features required by Sse4_2.
            Self::Avx2(_avx) => unsafe { Some(Sse4_2::assume_supported()) },
            Self::Sse4_2(sse42) => Some(sse42),
            _ => None,
        }
    }

    /// If this is a proof that the x86-64-v3 feature set (or better) is available, access that
    /// instruction set.
    ///
    /// See [`Avx2::assume_supported`] for the exact list of CPU features this token enables.
    ///
    /// This method should be preferred over matching against the `Avx2` variant of self,
    /// because if the CPU supports a superset of AVX2 (e.g. AVX-512),
    /// this method will return the AVX2 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_avx2(self) -> Option<Avx2> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `avx2`, there is only one variant."
        )]
        match self {
            // Safety: The Ice Lake AVX-512 feature set includes the x86-64-v3 features required by Avx2.
            Self::Avx512(_avx512) => unsafe { Some(Avx2::assume_supported()) },
            Self::Avx2(avx2) => Some(avx2),
            _ => None,
        }
    }

    /// If this is a proof that the Ice Lake AVX-512 feature set is available, access that
    /// instruction set.
    ///
    /// See [`Avx512::assume_supported`] for the exact list of CPU features this token enables.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_avx512(self) -> Option<Avx512> {
        match self {
            Self::Avx512(avx512) => Some(avx512),
            _ => None,
        }
    }

    /// Get the strongest statically supported SIMD level.
    ///
    /// That is, if your compilation run ambiently declares that a target feature is enabled,
    /// this method will take that into account.
    /// In most cases, you should use [`Level::new`] or [`Level::try_detect`].
    /// This method is mainly useful for libraries, where:
    ///
    /// 1) Your crate features request that you not use the standard library, i.e. doesn't enable
    ///    your `"std"` crate feature reason (so you can't use [`Level::new`] and
    ///    [`Level::try_detect`] returns `None`); AND
    /// 2) Your caller does not provide a [`Level`]; AND
    /// 3) The library doesn't want to panic when it can't find a SIMD level.
    ///
    /// Note that in these cases, the library should clearly inform the integrator
    /// that it is using a fallback and so not getting optimal performance (e.g. by panicking if
    /// `debug_assertions` are enabled, and emitting a log with the "error" level otherwise).
    /// The messages given should also provide actionable fixes, such as pointing to the
    /// entry-point which provides a `Level`, or your `"std"` feature.
    ///
    /// Note that this is unaffected by the `force-support-fallback` feature.
    /// Instead, you should use [`Level::fallback`] if you require the fallback level.
    #[inline]
    pub const fn baseline() -> Self {
        // TODO: How do we possibly test that this method works in all cases?
        // Note that you can use the `check_targets.sh` script to at least ensure that it compiles in all reasonable cases.
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            return Self::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            return unsafe { Self::Neon(Neon::assume_supported()) };
            #[cfg(not(target_feature = "neon"))]
            return Self::Fallback(Fallback::new());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(all(
                target_feature = "adx",
                target_feature = "aes",
                target_feature = "avx512bitalg",
                target_feature = "avx512bw",
                target_feature = "avx512cd",
                target_feature = "avx512dq",
                target_feature = "avx512f",
                target_feature = "avx512ifma",
                target_feature = "avx512vbmi",
                target_feature = "avx512vbmi2",
                target_feature = "avx512vl",
                target_feature = "avx512vnni",
                target_feature = "avx512vpopcntdq",
                target_feature = "bmi1",
                target_feature = "bmi2",
                target_feature = "cmpxchg16b",
                target_feature = "fma",
                target_feature = "fxsr",
                target_feature = "gfni",
                target_feature = "lzcnt",
                target_feature = "movbe",
                target_feature = "pclmulqdq",
                target_feature = "popcnt",
                target_feature = "rdrand",
                target_feature = "rdseed",
                target_feature = "sha",
                target_feature = "vaes",
                target_feature = "vpclmulqdq",
                target_feature = "xsave",
                target_feature = "xsavec",
                target_feature = "xsaveopt",
                target_feature = "xsaves"
            ))]
            return unsafe { Self::Avx512(Avx512::assume_supported()) };
            #[cfg(all(
                target_feature = "avx2",
                target_feature = "bmi1",
                target_feature = "bmi2",
                target_feature = "cmpxchg16b",
                target_feature = "f16c",
                target_feature = "fma",
                target_feature = "fxsr",
                target_feature = "lzcnt",
                target_feature = "movbe",
                target_feature = "popcnt",
                target_feature = "xsave",
                not(all(
                    target_feature = "adx",
                    target_feature = "aes",
                    target_feature = "avx512bitalg",
                    target_feature = "avx512bw",
                    target_feature = "avx512cd",
                    target_feature = "avx512dq",
                    target_feature = "avx512f",
                    target_feature = "avx512ifma",
                    target_feature = "avx512vbmi",
                    target_feature = "avx512vbmi2",
                    target_feature = "avx512vl",
                    target_feature = "avx512vnni",
                    target_feature = "avx512vpopcntdq",
                    target_feature = "bmi1",
                    target_feature = "bmi2",
                    target_feature = "cmpxchg16b",
                    target_feature = "fma",
                    target_feature = "fxsr",
                    target_feature = "gfni",
                    target_feature = "lzcnt",
                    target_feature = "movbe",
                    target_feature = "pclmulqdq",
                    target_feature = "popcnt",
                    target_feature = "rdrand",
                    target_feature = "rdseed",
                    target_feature = "sha",
                    target_feature = "vaes",
                    target_feature = "vpclmulqdq",
                    target_feature = "xsave",
                    target_feature = "xsavec",
                    target_feature = "xsaveopt",
                    target_feature = "xsaves"
                ))
            ))]
            return unsafe { Self::Avx2(Avx2::assume_supported()) };
            #[cfg(all(
                all(
                    target_feature = "fxsr",
                    target_feature = "sse4.2",
                    target_feature = "cmpxchg16b",
                    target_feature = "popcnt"
                ),
                not(all(
                    target_feature = "avx2",
                    target_feature = "bmi1",
                    target_feature = "bmi2",
                    target_feature = "cmpxchg16b",
                    target_feature = "f16c",
                    target_feature = "fma",
                    target_feature = "fxsr",
                    target_feature = "lzcnt",
                    target_feature = "movbe",
                    target_feature = "popcnt",
                    target_feature = "xsave"
                ))
            ))]
            return unsafe { Self::Sse4_2(Sse4_2::assume_supported()) };
            #[cfg(all(
                target_feature = "sse2",
                target_feature = "fxsr",
                not(all(
                    target_feature = "fxsr",
                    target_feature = "sse4.2",
                    target_feature = "cmpxchg16b",
                    target_feature = "popcnt"
                ))
            ))]
            return unsafe { Self::Sse2(Sse2::assume_supported()) };
            #[cfg(not(all(target_feature = "sse2", target_feature = "fxsr")))]
            return Self::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            return Self::WasmSimd128(WasmSimd128::assume_supported());
            #[cfg(not(target_feature = "simd128"))]
            return Self::Fallback(Fallback::new());
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn __dispatch_target(self) -> Self {
        // Dispatch compiles only the selected multiversioned backends, but public tokens can
        // still name lower levels even when the ambient target baseline makes those backends
        // redundant. Normalize the proof to the best dispatchable level, while leaving exact
        // token identity available for `kernel!` and explicit token use.
        #[cfg(feature = "force_support_fallback")]
        #[allow(
            irrefutable_let_patterns,
            reason = "On targets without supported SIMD, Fallback is the only Level variant."
        )]
        if let Self::Fallback(fallback) = self {
            return Self::Fallback(fallback);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[allow(unused_variables, reason = "Unused with all cfgs active")]
            let baseline = Self::baseline();

            #[cfg(not(disable_dispatch_avx512))]
            if let Some(avx512) = self.as_avx512().or_else(|| baseline.as_avx512()) {
                return Self::Avx512(avx512);
            }

            #[cfg(not(disable_dispatch_avx2))]
            if let Some(avx2) = self.as_avx2().or_else(|| baseline.as_avx2()) {
                return Self::Avx2(avx2);
            }

            #[cfg(not(disable_dispatch_sse4_2))]
            if let Some(sse4_2) = self.as_sse4_2().or_else(|| baseline.as_sse4_2()) {
                return Self::Sse4_2(sse4_2);
            }

            // The ambient SSE2 baseline is the terminal backend and cannot be disabled. Falling
            // below it would require compiling the otherwise-unneeded fallback implementation.
            #[cfg(any(
                not(disable_dispatch_sse2),
                all(target_feature = "sse2", target_feature = "fxsr")
            ))]
            if let Some(sse2) = self.as_sse2().or_else(|| baseline.as_sse2()) {
                return Self::Sse2(sse2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let baseline = Self::baseline();
            if let Some(neon) = self.as_neon().or_else(|| baseline.as_neon()) {
                return Self::Neon(neon);
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let baseline = Self::baseline();
            if let Some(wasm) = self
                .as_wasm_simd128()
                .or_else(|| baseline.as_wasm_simd128())
            {
                return Self::WasmSimd128(wasm);
            }
        }

        Self::baseline()
    }

    /// Create a scalar fallback level, which uses no SIMD instructions.
    ///
    /// This is primarily intended for tests; most users should prefer [`Level::new`] or [`Level::baseline`].
    ///
    /// Note that enabling the scalar fallback does *not* mean that the fallback branch will not
    /// contain SIMD instructions. This is because the "ambient" compilation environment has SIMD
    /// instructions available, which may be utilised by LLVM to auto-vectorise that path.
    #[inline]
    #[cfg(feature = "force_support_fallback")]
    pub const fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }
}

#[cfg(test)]
mod tests {
    use crate::Level;

    const fn assert_is_send_sync<T: Send + Sync>() {}
    /// If this test compiles, we know that [`Level`] is properly `Send` and `Sync`.
    #[test]
    fn level_is_send_sync() {
        assert_is_send_sync::<Level>();
    }

    #[test]
    fn baseline_reports_whether_fallback_is_required() {
        let has_simd_baseline = cfg!(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2",
                target_feature = "fxsr"
            ),
            all(target_arch = "wasm32", target_feature = "simd128")
        ));

        assert_eq!(Level::baseline().is_fallback(), !has_simd_baseline);
    }

    #[cfg(all(
        any(feature = "std", target_arch = "wasm32"),
        not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))
    ))]
    #[test]
    fn level_new_uses_baseline_outside_x86() {
        assert_eq!(
            core::mem::discriminant(&Level::new()),
            core::mem::discriminant(&Level::baseline())
        );
    }
}
