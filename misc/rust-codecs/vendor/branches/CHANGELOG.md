# Changelog

## 0.5.0

- The minimum supported Rust version is reduced from 1.59 to 1.51.0, the release that stabilized the const generics used by the prefetch API. Inline assembly is only stable since 1.59, so on stable 1.51-1.58 the prefetch functions compile to no-ops on every architecture; the branch hint functions are unaffected. Crate-level documentation (the README include) requires rustc 1.54 and falls back to a short doc string on older compilers.
- `assume` is now a `const fn` on every supported compiler, so const fns can carry optimizer invariants such as `assume(len <= CAP)` without any version plumbing on the caller's side. Rust older than 1.57 has no const-legal way to state an assumption (`core::hint::unreachable_unchecked` became callable in const fn in 1.57), so on rustc 1.51-1.56 `assume` compiles to a no-op and the hint is dropped there. From 1.59 the hint is exactly as effective as a hand-inlined check; on 1.57-1.58 old LLVM may keep a cheap loop-invariant compare that a hand-inlined check would fold.
- The `rustc_version` build-dependency is gone: the build script now reads `rustc -vV` itself, leaving the crate with zero dependencies. Recent releases of the `semver` crate (which `rustc_version` pulls in) ship manifests that cargo older than 1.60 cannot parse, which would have made the 1.51 MSRV unusable in practice.
- When built at opt-level 2 or 3 on stable rustc 1.84-1.94, `likely`/`unlikely`/`mark_unlikely` no longer leave a residual call to an empty `#[cold]` helper in callers' cold paths, nor the stack-alignment push/pop that call forced onto the hot path: LLVM 19+ records the coldness as branch weights before the empty call is inlined away, so the helpers drop `#[inline(never)]` there. All other builds - older compilers, and opt-levels 0/1/s/z, whose pipelines record no such weights (also true of `core::hint::cold_path` on 1.95+) - keep the out-of-line call, which is what carries the hint for them. Verified against every stable release from 1.84 through 1.94, and at opt-levels 1, 2, 3 and s; 1.95+ already used `core::hint::cold_path` and is unchanged.
- Prefetch hints are now emitted on `s390x` (`pfd`) and `powerpc`/`powerpc64` (`dcbt`/`dcbtst`), including `powerpc64le`. Both are gated on the rustc release that stabilized inline assembly for the architecture (1.84 for `s390x`, 1.95 for PowerPC); older compilers keep building the crate with prefetch as a no-op. Neither architecture has cache-level selection in its prefetch instruction, so `LOCALITY` is ignored there.

## 0.4.5

Fixes (see MIGRATE.md for details and migration notes):

- `likely`, `unlikely`, and `mark_unlikely` had no effect on stable rustc older than 1.95 since 0.4.2: `#[inline(always)]` on the internal cold helpers let the optimizer erase the cold call the hint relies on. The helpers are `#[inline(never)]` again and the hints work on all supported stable compilers.
- Prefetch `LOCALITY` was interpreted with inverted semantics on nightly (`0` meant non-temporal instead of L1). Nightly now translates to the intrinsics' convention so `0 = L1, 1 = L2, 2 = L3, other = non-temporal` holds on every toolchain, as documented.
- Prefetch functions failed to compile on 32-bit ARM (AArch64-only `prfm`) and riscv64 (x86 mnemonics). 32-bit ARM is now a no-op; riscv64 emits the correct `prefetch.r`/`prefetch.w` (Zicbop) instructions when built with `-C target-feature=+zicbop` and is a no-op otherwise.
- A `LOCALITY` outside `0..=3` crashed the compiler on nightly; out-of-range values are now clamped to non-temporal.
- `abort()` with the `std` feature now always calls `std::process::abort()` (nightly previously executed a trap instruction, raising `SIGILL` instead of `SIGABRT`). Without `std` on stable it now panics with a clear `branches::abort() called` message instead of `unreachable!()`.
- On 32-bit x86, write prefetch now emits `prefetcht0` instead of `prefetchw`, which faults on CPUs without the PRFCHW extension.
- On AArch64, prefetch now maps `LOCALITY` 1/2 to L2/L3 (`pstl2keep`/`pstl3keep`/`pldl2keep`/`pldl3keep`) and other values to streaming prefetches, matching the documented convention and nightly behavior.
- `mark_unlikely` lost its documentation and example on rustc >= 1.95; the docs are back.
- `likely`/`unlikely` are now `#[must_use]`.
- Declared `rust-version = "1.59"`; build script falls back to the stable code path instead of failing cryptically when the compiler version cannot be detected.
- README example fixes: `accumulate` prefetched the address of the slice reference instead of the buffer, and write-prefetched the input instead of the output buffer.

## 0.3.0

Breaking changes:

- Unified nightly/stable: signature now uses const template for locality.
- Function signatures changed from:
  - prefetch_read_data(addr: \*const u8, locality: i32)
  - prefetch_write_data(addr: \*const u8, locality: i32)
    to:
  - fn prefetch_read_data<T, const LOCALITY: i32>(addr: \*const T)
  - fn prefetch_write_data<T, const LOCALITY: i32>(addr: \*const T)
- Pointer now generic (*const T) instead of raw *const u8 (no manual casting needed).
- Locality passed as const generic (compile-time) instead of runtime parameter.
- Prefetch functions are now safe (no unsafe qualifier).

Example (old):
prefetch_read_data(ptr as \*const u8, 0);

Example (new):
prefetch_read_data::<MyType, 0>(ptr);
