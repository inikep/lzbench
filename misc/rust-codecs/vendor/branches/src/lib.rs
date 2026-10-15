#![cfg_attr(not(feature = "std"), no_std)]
// `doc = include_str!(...)` needs rustc 1.54, and the nested cfg_attr is
// load-bearing: pre-1.54 parsers reject this grammar even inside a disabled
// cfg_attr, but never validate the token tree of an inner list-form attribute.
#![cfg_attr(
    rustc_ge_1_54_0,
    cfg_attr(all(), doc = include_str!("../README.md"))
)]
#![cfg_attr(
    not(rustc_ge_1_54_0),
    doc = "Branch prediction hints (`likely`, `unlikely`, `mark_unlikely`), control-flow \
           assumptions (`assume`), `abort`, and CPU cache prefetch helpers for stable Rust. \
           See the project README for the full documentation."
)]
#![warn(missing_docs, missing_debug_implementations)]
#![cfg_attr(branches_nightly, feature(core_intrinsics))]
#![cfg_attr(branches_nightly, allow(internal_features))]
// Provides branch detection functions for Rust, using built-in Rust features
// on stable and core::intrinsics on nightly.

// No one likes to visit this function.
//
// The hint only works while LLVM sees a call to a `#[cold]` function inside
// the branch, so inlining the empty body kills it -- and since ~1.76 rustc's
// cross-crate MIR inlining does that on its own unless `#[inline(never)]`
// stops it. Under `branches_cold_weights` (rustc 1.84+ at -O2/-O3, see
// build.rs) the optimizer converts the call into `!prof` branch weights as it
// inlines, so the call may vanish from callers; at other opt levels no
// weights are recorded and the call has to stay out of line.
#[cfg(all(branches_stable, not(rustc_ge_1_95_0)))]
#[cfg_attr(not(branches_cold_weights), inline(never))]
#[cold]
const fn cold_and_empty() {}

#[cfg(all(branches_stable, rustc_ge_1_95_0))]
use core::hint::cold_path as cold_and_empty;

/// Aborts the execution of the process immediately and without any cleanup.
///
/// This function is used to indicate a critical and unrecoverable error in the program.
/// It terminates the process immediately without performing any cleanup or running destructors.
///
/// This function is safe to call, so it does not require an unsafe block.
/// Therefore, implementations must not require the user to uphold any safety invariants.
///
/// If the std feature is enabled, this function calls `std::process::abort()`
/// on every channel, which raises `SIGABRT` on Unix and honors registered
/// abort handlers.
///
/// If the std feature is disabled, this function executes a trap instruction
/// on nightly, and panics by calling `panic!()` on stable. In the panicking
/// case, `extern "C"` guarantees this function does not unwind, but actual
/// termination depends on the registered panic handler: a handler that loops
/// forever (common in embedded code) hangs instead of aborting.
#[cold]
pub extern "C" fn abort() -> ! {
    #[cfg(feature = "std")]
    std::process::abort();
    #[cfg(all(not(feature = "std"), branches_nightly))]
    core::intrinsics::abort_immediate();
    #[cfg(all(not(feature = "std"), branches_stable))]
    panic!("branches::abort() called");
}

/// Informs the optimizer that a condition is always true.
///
/// If the condition is actually false, the behavior is undefined.
///
/// This intrinsic doesn't generate any code. Instead, it tells the optimizer
/// to preserve the condition for optimization passes. This can interfere with
/// optimization of surrounding code and reduce performance, so avoid using it
/// if the optimizer can already discover the invariant on its own or if it
/// doesn't enable any significant optimizations.
///
/// This function is a `const fn`, so a const fn can carry invariants like
/// `assume(len <= CAP)`. Rust older than 1.57 has no const-legal way to
/// state an assumption (`core::hint::unreachable_unchecked` became callable
/// in const fn in 1.57), so on rustc 1.51-1.56 this compiles to a no-op and
/// the hint is dropped. From rustc 1.59 the hint is exactly as effective as
/// writing the check by hand; 1.57-1.58 may keep a cheap residual compare in
/// some loop shapes.
///
/// # Safety
///
/// This intrinsic is marked unsafe because it can result in undefined behavior
/// if the condition passed to it is false.
#[inline(always)]
pub const unsafe fn assume(b: bool) {
    let _ = b;
    // Rust >= 1.81.0: use the newer `assert_unchecked` hint. Clippy cannot
    // see that the cfg guarantees the API exists.
    #[cfg(all(branches_stable, rustc_ge_1_81_0))]
    #[allow(clippy::incompatible_msrv)]
    {
        core::hint::assert_unchecked(b)
    }
    // Rust 1.57-1.80: `unreachable_unchecked`, const-callable since 1.57.
    #[cfg(all(branches_stable, rustc_ge_1_57_0, not(rustc_ge_1_81_0)))]
    {
        if !b {
            core::hint::unreachable_unchecked()
        }
    }
    #[cfg(all(branches_nightly, rustc_ge_1_57_0))]
    core::intrinsics::assume(b)
    // Pre-1.57 compilers, stable or nightly, fall through to the no-op.
}

/// Hints to the compiler that the branch condition is likely to be true.
/// Returns the value passed to it.
///
/// This intrinsic is primarily used with `if` statements.
/// Using it in other contexts may not have any effect.
///
/// Unlike most intrinsics, this function is safe to call and doesn't require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold any safety invariants.
#[must_use = "the hint only takes effect when the returned value is used as a branch condition"]
#[inline(always)]
pub fn likely(b: bool) -> bool {
    // On 1.95+ `cold_and_empty` is `cold_path`; clippy can't see the cfg.
    #[cfg(branches_stable)]
    #[allow(clippy::incompatible_msrv)]
    {
        if !b {
            cold_and_empty();
        }
        b
    }
    #[cfg(branches_nightly)]
    core::intrinsics::likely(b)
}

/// Marks a code block as cold, indicating to the compiler that it is unlikely to be called.
/// This can help the compiler optimize for the common case.
///
/// This function does not take any arguments and does not return any value.
/// It is primarily used to mark functions or code paths that are rarely executed,
/// such as error handling or panic paths.
///
/// Example: marking the error variant of a match as unlikely.
///
/// In many hot paths a value is expected to be the success variant.
/// By marking the error arm using `mark_unlikely` we give the optimizer a hint
/// that this branch is rarely taken.
///
/// ```rust
/// use branches::{mark_unlikely};
///
/// #[derive(Debug)]
/// enum Status {
///     Ok(i32),
///     Err(String),
/// }
///
/// fn get_value(status: Status) -> i32 {
///     match status {
///         Status::Ok(v) => v,
///         // The error case is rare, hint the compiler accordingly.
///         Status::Err(err) => {
///             mark_unlikely();
///             eprintln!("unexpected error: {:?}", err);
///             -1
///         }
///     }
/// }
/// ```
// Same rules as `cold_and_empty` above.
#[cfg(not(rustc_ge_1_95_0))]
#[cold]
#[cfg_attr(not(branches_cold_weights), inline(never))]
pub const fn mark_unlikely() {}
/// Marks a code block as cold, indicating to the compiler that it is unlikely to be called.
/// This can help the compiler optimize for the common case.
///
/// This function does not take any arguments and does not return any value.
/// It is primarily used to mark functions or code paths that are rarely executed,
/// such as error handling or panic paths.
///
/// Example: marking the error variant of a match as unlikely.
///
/// In many hot paths a value is expected to be the success variant.
/// By marking the error arm using `mark_unlikely` we give the optimizer a hint
/// that this branch is rarely taken.
///
/// ```rust
/// use branches::{mark_unlikely};
///
/// #[derive(Debug)]
/// enum Status {
///     Ok(i32),
///     Err(String),
/// }
///
/// fn get_value(status: Status) -> i32 {
///     match status {
///         Status::Ok(v) => v,
///         // The error case is rare, hint the compiler accordingly.
///         Status::Err(err) => {
///             mark_unlikely();
///             eprintln!("unexpected error: {:?}", err);
///             -1
///         }
///     }
/// }
/// ```
#[cfg(rustc_ge_1_95_0)]
pub use core::hint::cold_path as mark_unlikely;

/// Hints to the compiler that the branch condition is unlikely to be true.
/// Returns the value passed to it.
///
/// This intrinsic is primarily used with `if` statements.
/// Using it in other contexts may not have any effect.
///
/// Unlike most intrinsics, this function is safe to call and doesn't require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold any safety invariants.
#[must_use = "the hint only takes effect when the returned value is used as a branch condition"]
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    // On 1.95+ `cold_and_empty` is `cold_path`; clippy can't see the cfg.
    #[cfg(branches_stable)]
    #[allow(clippy::incompatible_msrv)]
    {
        if b {
            cold_and_empty();
        }
        b
    }
    #[cfg(branches_nightly)]
    core::intrinsics::unlikely(b)
}

/// Prefetches data for reading into the cache.
///
/// This function hints to the CPU that the data at the given address
/// will be read soon, allowing the CPU to load the data into the cache
/// in advance. This can improve performance by reducing cache misses.
///
/// Prefetching is only a hint and never affects the observable behavior of
/// the program: it is safe to call with any pointer, including dangling or
/// out-of-bounds pointers.
///
/// # Arguments
///
/// * `addr` - A pointer to the data to prefetch.
/// * `LOCALITY` - The cache level to prefetch into: `0` = L1, `1` = L2,
///   `2` = L3, any other value = non-temporal. The convention is identical
///   on stable and nightly toolchains.
///
/// # Supported architectures
///
/// On stable, the hint is emitted on rustc 1.59 or newer (the release that
/// stabilized inline assembly) for `x86`/`x86_64` (with the `sse` target
/// feature, enabled by default on `x86_64` and `i686` targets), `aarch64`,
/// and `riscv64` when compiled with the `zicbop` target feature
/// (`-C target-feature=+zicbop`); on rustc 1.84 or newer for `s390x`; and
/// on rustc 1.95 or newer for `powerpc`/`powerpc64` (the releases that
/// stabilized inline assembly for those architectures). `s390x`,
/// `powerpc` and `powerpc64` have a single prefetch instruction with no
/// cache-level selection, so `LOCALITY` is ignored there. On other targets,
/// and on stable compilers older than the listed versions, this compiles to
/// a no-op. On nightly, the hint is lowered by LLVM for every architecture
/// that supports one.
#[inline(always)]
#[cfg(feature = "prefetch")]
pub fn prefetch_read_data<T, const LOCALITY: i32>(addr: *const T) {
    let _ = addr;
    #[cfg(branches_stable)]
    {
        // Inline assembly was stabilized in Rust 1.59, so older stable
        // compilers stay a no-op instead of failing to build.
        #[cfg(all(
            rustc_ge_1_59_0,
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse"
        ))]
        unsafe {
            match LOCALITY {
                0 => core::arch::asm!(
                    "prefetcht0 [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L1 cache
                1 => core::arch::asm!(
                    "prefetcht1 [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L2 cache
                2 => core::arch::asm!(
                    "prefetcht2 [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L3 cache
                _ => core::arch::asm!(
                    "prefetchnta [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // Non-temporal
            }
        }

        // `prfm` only exists on AArch64; 32-bit ARM would need `pld`, which
        // not every 32-bit ARM target supports, so arm stays a no-op.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "aarch64"))]
        unsafe {
            match LOCALITY {
                0 => core::arch::asm!(
                    "prfm pldl1keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L1 cache
                1 => core::arch::asm!(
                    "prfm pldl2keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L2 cache
                2 => core::arch::asm!(
                    "prfm pldl3keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L3 cache
                _ => core::arch::asm!(
                    "prfm pldl1strm, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // Non-temporal (streaming)
            }
        }

        // The Zicbop extension is not part of the baseline riscv64gc target,
        // so the instruction is only emitted when the feature is enabled.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "riscv64", target_feature = "zicbop"))]
        unsafe {
            core::arch::asm!(
                "prefetch.r 0({})",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            );
        }

        // s390x inline assembly was stabilized in Rust 1.84, so older
        // compilers stay a no-op instead of failing to build. `pfd` has no
        // locality levels (LLVM ignores locality on SystemZ too), and the
        // address must go in an address register: `r0` in a base register
        // slot reads as the literal zero, not as the register.
        #[cfg(all(rustc_ge_1_84_0, target_arch = "s390x"))]
        unsafe {
            core::arch::asm!(
                "pfd 1, 0({})",
                in(reg_addr) addr,
                options(nostack, readonly, preserves_flags)
            ); // Prefetch for load
        }

        // PowerPC inline assembly was stabilized in Rust 1.95, so older
        // compilers stay a no-op instead of failing to build. `dcbt` carries
        // no locality levels either, matching LLVM's lowering. The register
        // holds the `RB` operand, where `r0` keeps its normal meaning.
        #[cfg(all(
            rustc_ge_1_95_0,
            any(target_arch = "powerpc", target_arch = "powerpc64")
        ))]
        unsafe {
            core::arch::asm!(
                "dcbt 0, {}",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            );
        }
    }
    #[cfg(branches_nightly)]
    {
        // `core::intrinsics` uses the opposite locality convention
        // (0 = no locality .. 3 = maximally local), so translate to keep
        // stable and nightly behavior identical. The catch-all arm also
        // keeps out-of-range values from reaching LLVM, which only accepts
        // 0..=3 and crashes otherwise.
        match LOCALITY {
            0 => core::intrinsics::prefetch_read_data::<_, 3>(addr),
            1 => core::intrinsics::prefetch_read_data::<_, 2>(addr),
            2 => core::intrinsics::prefetch_read_data::<_, 1>(addr),
            _ => core::intrinsics::prefetch_read_data::<_, 0>(addr),
        }
    }
}

/// Prefetches data for writing into the cache.
///
/// This function hints to the CPU that the data at the given address
/// will be written soon, allowing the CPU to load the data into the cache
/// in advance. This can improve performance by reducing cache misses.
///
/// Prefetching is only a hint and never affects the observable behavior of
/// the program: it is safe to call with any pointer, including dangling or
/// out-of-bounds pointers.
///
/// # Arguments
///
/// * `addr` - A pointer to the data to prefetch.
/// * `LOCALITY` - The cache level to prefetch into: `0` = L1, `1` = L2,
///   `2` = L3, any other value = non-temporal. The convention is identical
///   on stable and nightly toolchains. On `x86_64` there is a single
///   write-prefetch instruction, so `LOCALITY` is ignored there.
///
/// # Supported architectures
///
/// On stable, the hint is emitted on rustc 1.59 or newer (the release that
/// stabilized inline assembly) for `x86`/`x86_64` (with the `sse` target
/// feature, enabled by default on `x86_64` and `i686` targets), `aarch64`,
/// and `riscv64` when compiled with the `zicbop` target feature
/// (`-C target-feature=+zicbop`); on rustc 1.84 or newer for `s390x`; and
/// on rustc 1.95 or newer for `powerpc`/`powerpc64` (the releases that
/// stabilized inline assembly for those architectures). `s390x`,
/// `powerpc` and `powerpc64` have a single write-prefetch instruction with
/// no cache-level selection, so `LOCALITY` is ignored there. On other
/// targets, and on stable compilers older than the listed versions, this
/// compiles to a no-op. On nightly, the hint is lowered by LLVM for every
/// architecture that supports one.
#[inline(always)]
#[cfg(feature = "prefetch")]
pub fn prefetch_write_data<T, const LOCALITY: i32>(addr: *const T) {
    let _ = addr;
    #[cfg(branches_stable)]
    {
        // Inline assembly was stabilized in Rust 1.59, so older stable
        // compilers stay a no-op instead of failing to build.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "prefetchw [{}]",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            ) // Write-prefetch for L1/L2/L3 cache
        }

        // 32-bit x86: `prefetchw` faults on CPUs without the PRFCHW/3DNow!
        // extension, so fall back to a plain read prefetch into L1, the same
        // strategy GCC and Clang use for `__builtin_prefetch(p, 1)` there.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "x86", target_feature = "sse"))]
        unsafe {
            core::arch::asm!(
                "prefetcht0 [{}]",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            )
        }

        // `prfm` only exists on AArch64; 32-bit ARM would need `pldw`, which
        // requires the MP extension, so arm stays a no-op.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "aarch64"))]
        unsafe {
            match LOCALITY {
                0 => core::arch::asm!(
                    "prfm pstl1keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L1 cache
                1 => core::arch::asm!(
                    "prfm pstl2keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L2 cache
                2 => core::arch::asm!(
                    "prfm pstl3keep, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // L3 cache
                _ => core::arch::asm!(
                    "prfm pstl1strm, [{}]",
                    in(reg) addr,
                    options(nostack, readonly, preserves_flags)
                ), // Non-temporal (streaming)
            }
        }

        // The Zicbop extension is not part of the baseline riscv64gc target,
        // so the instruction is only emitted when the feature is enabled.
        #[cfg(all(rustc_ge_1_59_0, target_arch = "riscv64", target_feature = "zicbop"))]
        unsafe {
            core::arch::asm!(
                "prefetch.w 0({})",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            );
        }

        // s390x inline assembly was stabilized in Rust 1.84, so older
        // compilers stay a no-op instead of failing to build. `pfd` has no
        // locality levels (LLVM ignores locality on SystemZ too), and the
        // address must go in an address register: `r0` in a base register
        // slot reads as the literal zero, not as the register.
        #[cfg(all(rustc_ge_1_84_0, target_arch = "s390x"))]
        unsafe {
            core::arch::asm!(
                "pfd 2, 0({})",
                in(reg_addr) addr,
                options(nostack, readonly, preserves_flags)
            ); // Prefetch for store
        }

        // PowerPC inline assembly was stabilized in Rust 1.95, so older
        // compilers stay a no-op instead of failing to build. `dcbtst` carries
        // no locality levels either, matching LLVM's lowering. The register
        // holds the `RB` operand, where `r0` keeps its normal meaning.
        #[cfg(all(
            rustc_ge_1_95_0,
            any(target_arch = "powerpc", target_arch = "powerpc64")
        ))]
        unsafe {
            core::arch::asm!(
                "dcbtst 0, {}",
                in(reg) addr,
                options(nostack, readonly, preserves_flags)
            ); // Write-prefetch
        }
    }
    #[cfg(branches_nightly)]
    {
        // `core::intrinsics` uses the opposite locality convention
        // (0 = no locality .. 3 = maximally local), so translate to keep
        // stable and nightly behavior identical. The catch-all arm also
        // keeps out-of-range values from reaching LLVM, which only accepts
        // 0..=3 and crashes otherwise.
        match LOCALITY {
            0 => core::intrinsics::prefetch_write_data::<_, 3>(addr),
            1 => core::intrinsics::prefetch_write_data::<_, 2>(addr),
            2 => core::intrinsics::prefetch_write_data::<_, 1>(addr),
            _ => core::intrinsics::prefetch_write_data::<_, 0>(addr),
        }
    }
}

// Non-generic instantiations of every architecture-specific code path.
// Not part of the public API: only compiled when CI passes
// `RUSTFLAGS="--cfg branches_check_asm"`, so that plain library cross-builds
// (which never monomorphize the generic prefetch functions) still assemble
// the inline assembly for the target architecture.
#[cfg(branches_check_asm)]
#[doc(hidden)]
pub fn __branches_check_asm(addr: *const u8, cond: bool) -> bool {
    let _ = addr;
    #[cfg(feature = "prefetch")]
    {
        prefetch_read_data::<_, 0>(addr);
        prefetch_read_data::<_, 1>(addr);
        prefetch_read_data::<_, 2>(addr);
        prefetch_read_data::<_, 3>(addr);
        prefetch_read_data::<_, { -1 }>(addr);
        prefetch_write_data::<_, 0>(addr);
        prefetch_write_data::<_, 1>(addr);
        prefetch_write_data::<_, 2>(addr);
        prefetch_write_data::<_, 3>(addr);
        prefetch_write_data::<_, { -1 }>(addr);
    }
    if unlikely(!cond) {
        mark_unlikely();
    }
    likely(cond)
}
