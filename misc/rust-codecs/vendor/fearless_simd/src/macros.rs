// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

/// Access the applicable [`Simd`] for a given `level`, and perform an operation using it.
///
/// This macro is the root of how any explicitly written SIMD functions in this crate are
/// called from a non-SIMD context.
///
/// The first parameter to the macro is the [`Level`].
/// You should prefer to construct a [`Level`] once and pass it around, rather than
/// frequently calling [`Level::new()`].
/// On x86, the first call to `Level::new` detects which target features are available and caches
/// the result. Passing the resulting value around avoids even the cached atomic load.
///
/// The code of the operation will be repeated literally several times in the output, so you should prefer
/// to keep this code small (as it will be type-checked, etc. for each supported SIMD level on your target).
/// In most cases, it should be a single call to a function which is generic over `Simd` implementations,
/// as seen in [the examples](#examples).
/// For clarity, it will only be executed once per execution of `dispatch`.
///
/// To guarantee target-feature-specific code generation, functions called within the operation
/// should either establish their own context with [`Simd::vectorize`], use the
/// [`#[simd]`](https://docs.rs/fearless_simd_macros/latest/fearless_simd_macros/attr.simd.html)
/// attribute, or be `#[inline(always)]` so that they are incorporated into this operation.
///
/// Note that as an implementation detail of this macro, the operation will be executed inside a closure.
/// This is what enables the target features to be enabled for the code inside the operation.
/// A consequence of this is that early `return` and `?` will not work as expected.
/// Note that in cases where you use `dispatch` to call a single function (which we expect to be the
/// majority of cases), you can use `?` on the return value of dispatch instead.
/// To emulate early return, you can use [`ControlFlow`](core::ops::ControlFlow) instead.
///
/// # Example
///
/// ```rust
/// use fearless_simd::{Level, Simd, dispatch};
///
/// #[inline(always)]
/// fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }
///
/// let level = Level::new();
///
/// dispatch!(level, simd => sigmoid(simd, &[/*...*/], &mut [/*...*/]));
/// ```
///
/// [`Level`]: crate::Level
/// [`Level::new()`]: crate::Level::new
/// [`Simd`]: crate::Simd
/// [`Simd::vectorize`]: crate::Simd::vectorize
#[macro_export]
macro_rules! dispatch {
    ($level:expr, $simd:pat => $op:expr) => {{
        $crate::__fearless_simd_dispatch_inner!($level, $simd => $op)
    }};
    (@impl $level:expr, $simd:pat => $op:expr; $forced_fallback_arm:literal) => {{
        match $crate::Level::__dispatch_target($level) {
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(neon) => {
                $crate::__fearless_simd_dispatch_with_token!(neon, $simd => $op)
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(wasm) => {
                $crate::__fearless_simd_dispatch_with_token!(wasm, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Sse2(sse2) => {
                $crate::__fearless_simd_dispatch_dispatch_sse2!(sse2, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Sse4_2(sse4_2) => {
                $crate::__fearless_simd_dispatch_dispatch_sse4_2!(sse4_2, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(avx2) => {
                $crate::__fearless_simd_dispatch_dispatch_avx2!(avx2, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx512(avx512) => {
                $crate::__fearless_simd_dispatch_dispatch_avx512!(avx512, $simd => $op)
            }
            // Keep this predicate in sync with `Level::Fallback` and the fallback module. The
            // literal carries fearless_simd's feature selection into expansion in a downstream
            // crate, where `cfg(feature = ...)` would be incorrect.
            #[cfg(any(
                $forced_fallback_arm,
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
            $crate::Level::Fallback(fallback) => {
                $crate::__fearless_simd_dispatch_with_token!(fallback, $simd => $op)
            }
            _ => unreachable!(),
        }
    }};
}

// This macro turns whether `force_support_fallback` is enabled into a boolean literal in
// `dispatch!`. Exported macro bodies are expanded in the downstream crate, so evaluating the
// feature cfg directly in `dispatch!` would inspect the wrong crate's features.

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "force_support_fallback")]
macro_rules! __fearless_simd_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(@impl $level, $simd => $op; true)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "force_support_fallback"))]
macro_rules! __fearless_simd_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(@impl $level, $simd => $op; false)
    };
}

// The dispatch helpers are split into cfg-selected macro definitions
// because exported macro bodies are expanded in the downstream crate,
// so selecting the helper definitions here preserves `fearless_simd`'s
// dispatch configuration. Helpers for pruned levels expand to `unreachable!()`
// without mentioning `$op`, so the pruned SIMD body is not typechecked or
// compiled.

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
macro_rules! __fearless_simd_dispatch_with_token {
    ($token:expr, $simd:pat => $op:expr) => {{
        /// Convert the `Simd` value into an `impl Simd`, which enforces that
        /// it is correctly handled.
        // TODO: Just make into a `pub` function in fearless_simd itself?
        #[inline(always)]
        fn launder<S: $crate::Simd>(x: S) -> impl $crate::Simd {
            x
        }

        // $token can be an arbitrary expression,
        // so bind the result of evaluating it to a variable before use
        // so that it's evaluated only once
        let __fearless_simd_token = $token;
        let $simd = launder(__fearless_simd_token);
        $crate::Simd::vectorize(
            __fearless_simd_token,
            #[inline(always)]
            || $op,
        )
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
macro_rules! __fearless_simd_dispatch_pruned {
    ($proof:expr) => {{
        let __fearless_simd_proof = $proof;
        let _ = __fearless_simd_proof;
        unreachable!("this SIMD level was pruned from dispatch")
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(disable_dispatch_avx512))]
macro_rules! __fearless_simd_dispatch_dispatch_avx512 {
    ($avx512:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($avx512, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(disable_dispatch_avx512)]
macro_rules! __fearless_simd_dispatch_dispatch_avx512 {
    ($avx512:expr, $simd:pat => $op:expr) => {{ $crate::__fearless_simd_dispatch_pruned!($avx512) }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(all(
    not(disable_dispatch_avx2),
    any(
        disable_dispatch_avx512,
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
            target_feature = "xsaves",
        ))
    )
))]
macro_rules! __fearless_simd_dispatch_dispatch_avx2 {
    ($avx2:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($avx2, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(any(
    disable_dispatch_avx2,
    all(
        not(disable_dispatch_avx512),
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
        target_feature = "xsaves",
    ),
))]
macro_rules! __fearless_simd_dispatch_dispatch_avx2 {
    ($avx2:expr, $simd:pat => $op:expr) => {{ $crate::__fearless_simd_dispatch_pruned!($avx2) }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(all(
    not(disable_dispatch_sse4_2),
    any(
        disable_dispatch_avx2,
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
    )
))]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2 {
    ($sse4_2:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($sse4_2, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(any(
    disable_dispatch_sse4_2,
    all(
        not(disable_dispatch_avx2),
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
    ),
))]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2 {
    ($sse4_2:expr, $simd:pat => $op:expr) => {{ $crate::__fearless_simd_dispatch_pruned!($sse4_2) }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(all(
    any(
        not(disable_dispatch_sse2),
        all(target_feature = "sse2", target_feature = "fxsr")
    ),
    any(
        disable_dispatch_sse4_2,
        not(all(
            target_feature = "fxsr",
            target_feature = "sse4.2",
            target_feature = "cmpxchg16b",
            target_feature = "popcnt",
        )),
    )
))]
macro_rules! __fearless_simd_dispatch_dispatch_sse2 {
    ($sse2:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($sse2, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(any(
    all(
        disable_dispatch_sse2,
        not(all(target_feature = "sse2", target_feature = "fxsr"))
    ),
    all(
        not(disable_dispatch_sse4_2),
        target_feature = "fxsr",
        target_feature = "sse4.2",
        target_feature = "cmpxchg16b",
        target_feature = "popcnt",
    ),
))]
macro_rules! __fearless_simd_dispatch_dispatch_sse2 {
    ($sse2:expr, $simd:pat => $op:expr) => {{ $crate::__fearless_simd_dispatch_pruned!($sse2) }};
}

#[cfg(test)]
// This expect also validates that we haven't missed any levels!
#[expect(
    unreachable_patterns,
    reason = "Level is non_exhaustive, but you must be exhaustive within the same crate."
)]
mod tests {
    use crate::{Level, Simd};

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum X86DispatchBackend {
        Fallback,
        Sse2,
        Sse4_2,
        Avx2,
        Avx512,
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn x86_dispatch_backend<S: Simd>(_: S) -> X86DispatchBackend {
        use core::any::TypeId;

        #[cfg(any(
            not(all(target_feature = "sse2", target_feature = "fxsr")),
            feature = "force_support_fallback"
        ))]
        if TypeId::of::<S>() == TypeId::of::<crate::Fallback>() {
            return X86DispatchBackend::Fallback;
        }

        if TypeId::of::<S>() == TypeId::of::<crate::Sse2>() {
            X86DispatchBackend::Sse2
        } else if TypeId::of::<S>() == TypeId::of::<crate::Sse4_2>() {
            X86DispatchBackend::Sse4_2
        } else if TypeId::of::<S>() == TypeId::of::<crate::Avx2>() {
            X86DispatchBackend::Avx2
        } else if TypeId::of::<S>() == TypeId::of::<crate::Avx512>() {
            X86DispatchBackend::Avx512
        } else {
            unreachable!("unexpected x86 dispatch backend")
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn expected_x86_dispatch_backend(level: Level) -> X86DispatchBackend {
        if cfg!(not(disable_dispatch_avx512)) && level.as_avx512().is_some() {
            X86DispatchBackend::Avx512
        } else if cfg!(not(disable_dispatch_avx2)) && level.as_avx2().is_some() {
            X86DispatchBackend::Avx2
        } else if cfg!(not(disable_dispatch_sse4_2)) && level.as_sse4_2().is_some() {
            X86DispatchBackend::Sse4_2
        } else if cfg!(any(
            not(disable_dispatch_sse2),
            all(target_feature = "sse2", target_feature = "fxsr")
        )) && level.as_sse2().is_some()
        {
            X86DispatchBackend::Sse2
        } else {
            X86DispatchBackend::Fallback
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(dead_code, reason = "Compile test")]
    fn x86_level_variants_remain_available() {
        let _ = Level::Sse2(unsafe { crate::Sse2::assume_supported() });
        let _ = Level::Sse4_2(unsafe { crate::Sse4_2::assume_supported() });
        let _ = Level::Avx2(unsafe { crate::Avx2::assume_supported() });
        let _ = Level::Avx512(unsafe { crate::Avx512::assume_supported() });
    }

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_generic() {
        fn generic<S: Simd, T>(_: S, x: T) -> T {
            x
        }
        dispatch!(Level::new(), simd => generic::<_, ()>(simd, ()));
    }

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_value() {
        fn make_fn<S: Simd>() -> impl FnOnce(S) {
            |_| ()
        }
        dispatch!(Level::new(), simd => (make_fn())(simd));
    }

    #[test]
    fn dispatch_output() {
        assert_eq!(42, dispatch!(Level::new(), _simd => 42));
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn dispatch_respects_x86_multiversion_features() {
        let level = Level::new();
        let actual = dispatch!(level, simd => x86_dispatch_backend(simd));

        assert_eq!(actual, expected_x86_dispatch_backend(level));
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2",
        not(target_feature = "fxsr")
    ))]
    #[test]
    fn x86_baseline_does_not_use_sse2_without_fxsr() {
        assert!(Level::baseline().is_fallback());

        let actual = dispatch!(Level::Fallback(crate::Fallback::new()), simd =>
            x86_dispatch_backend(simd)
        );

        assert_eq!(actual, X86DispatchBackend::Fallback);
    }

    #[cfg(all(
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64"),
        any(
            disable_dispatch_sse2,
            disable_dispatch_sse4_2,
            disable_dispatch_avx2,
            disable_dispatch_avx512
        )
    ))]
    #[test]
    fn disabled_x86_multiversioning_does_not_filter_token_access() {
        let level = Level::new();

        // If dispatch disabling accidentally removed token access, these method calls stop compiling.
        #[cfg(disable_dispatch_sse2)]
        let _ = level.as_sse2().is_some();

        #[cfg(disable_dispatch_sse4_2)]
        let _ = level.as_sse4_2().is_some();

        #[cfg(disable_dispatch_avx2)]
        let _ = level.as_avx2().is_some();

        #[cfg(disable_dispatch_avx512)]
        let _ = level.as_avx512().is_some();
    }

    #[cfg(all(
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64"),
        disable_dispatch_sse2,
        target_feature = "sse2",
        target_feature = "fxsr"
    ))]
    #[test]
    fn ambient_sse2_is_the_terminal_backend_when_dispatch_is_disabled() {
        let actual = dispatch!(Level::new(), simd => x86_dispatch_backend(simd));

        assert_ne!(actual, X86DispatchBackend::Fallback);
    }

    /// Mostly useful with `RUSTFLAGS='-C target-cpu=x86-64-v3'` and higher, doesn't do much otherwise
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn high_baseline_dispatches_lower_tokens() {
        if let Some(sse2) = Level::baseline().as_sse2() {
            let actual = dispatch!(sse2.level(), simd => x86_dispatch_backend(simd));

            assert_eq!(actual, expected_x86_dispatch_backend(Level::baseline()));
        }

        if let Some(sse4_2) = Level::baseline().as_sse4_2() {
            let actual = dispatch!(sse4_2.level(), simd => x86_dispatch_backend(simd));

            assert_eq!(actual, expected_x86_dispatch_backend(Level::baseline()));
        }
    }

    #[cfg(all(
        not(feature = "force_support_fallback"),
        any(target_arch = "x86", target_arch = "x86_64"),
        not(all(target_feature = "sse2", target_feature = "fxsr"))
    ))]
    #[test]
    fn fallback_dispatches_as_baseline() {
        let actual = dispatch!(Level::Fallback(crate::Fallback::new()), simd =>
            x86_dispatch_backend(simd)
        );

        assert_eq!(actual, expected_x86_dispatch_backend(Level::baseline()));
    }

    mod no_import_simd {
        /// We should be able to use [`dispatch`] in a scope which doesn't import anything.
        #[test]
        fn dispatch_with_no_imports() {
            let res = dispatch!(crate::Level::new(), _ => 1 + 2);
            assert_eq!(res, 3);
        }
    }
}
