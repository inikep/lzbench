# Avoiding #[simd] proc macro with manual inlining

The use of `#[simd]` is recommended as the more ergonomic option. This document describes how to achieve the same effect manually, if you have to.

## TL;DR

- All SIMD functions need `#[inline(always)]`.
- Use `dispatch!` when calling SIMD code from non-SIMD code.
- Use `vectorize()` when calling SIMD from SIMD if you don't want to force inlining.

## A slightly longer explanation

Calls to SIMD functions have to be made from a function annotated `#[target_feature(enable = ...)]`, to let the compiler know that the CPU instructions required for SIMD are available within this function. `dispatch!` and `vectorize()` add the appropriate `#[target_feature(enable = ...)]` annotations, but there is your code between `dispatch!` which sets these annotations and fearless_simd functions like `f32x4.sqrt()` which need that annotation for performance.

The contents of a function annotated `#[inline(always)]` will always be copy-pasted into the caller by the compiler. The compiler may decide to do this as an optimization anyway, but `#[inline(always)]` forces this behavior. This lets you erase the boundaries of code that sits between `dispatch!`/`vectorize()` and the code that wants to emit SIMD instructions.

The `#[simd]` macro solves this by rewriting

```rust
fn foo<S: Simd>(simd: S, ...) {
    // Original body
}
```

into something close to this:

```rust
fn foo<S: Simd>(simd: S, ...) {
    simd.vectorize(
        #[inline(always)]
        || {
            // Original body
        },
    )
}
```

Which solves the problem but requires too much boilerplate to write this out manually every time. Hence the recommendation to annotate functions with the shorter `#[inline(always)]` and insert calls to `vectorize()` when you need to break up the inlining.

## Further reading

[The article describing the design](https://shnatsel.github.io/safe-simd-in-rust-even-on-the-inside/#the-abi-would-like-a-word) explains why this is needed from first principles and how we arrived to this design.

There's also Q&A on [Zulip](https://xi.zulipchat.com/#narrow/channel/514230-simd/topic/inlining/with/546913433).