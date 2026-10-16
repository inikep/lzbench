# borrowed-buf

[![Crates.io](https://img.shields.io/crates/v/borrowed-buf.svg)](https://crates.io/crates/borrowed-buf)
[![Docs.rs](https://img.shields.io/docsrs/borrowed-buf)](https://docs.rs/borrowed-buf)
[![CI](https://github.com/M-Compressors/borrowed-buf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/M-Compressors/borrowed-buf/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/)

A small, fast, `no_std` port of the nightly-only
[`core::io::BorrowedBuf`](https://doc.rust-lang.org/nightly/core/io/struct.BorrowedBuf.html)
and [`BorrowedCursor`](https://doc.rust-lang.org/nightly/core/io/struct.BorrowedCursor.html),
usable on **stable Rust (MSRV 1.89)**.

The API mirrors `core::io` on nightly (including the `borrowed_buf_init` methods), so
switching to the standard library later is a matter of changing the import, unless you use
one of the few [additions](#differences-from-coreio).

It lets you read into uninitialized memory without zeroing it first, while
tracking how much of the buffer is filled and whether the rest is initialized:

```text
[                capacity                ]
[ filled | unfilled (may be initialized) ]
```

- No dependencies, `#![no_std]`; the default `std` feature adds `std::io` helpers.
- Generic over the element type: `BorrowedBuf<'a, T>` for any `T: Copy`.
- `ensure_init` initializes the buffer at most once, so a reused buffer is zeroed only once.
- Every test runs under Miri (Stacked Borrows and Tree Borrows) in CI.

## Why

Most APIs that fill a buffer (`Read::read`, decoders, codecs) take `&mut [T]`,
which must already be initialized. That leaves two options:

- **Zero it first.** `[0; N]` or `vec![0; n]` costs a full memset for every
  fresh buffer, often more than the read itself.
- **Use `MaybeUninit<T>` by hand.** Then you track which prefix is written,
  call `assume_init` only on that prefix, and never create a `&[T]` over
  uninitialized memory. Get any step wrong and it's silent undefined behavior.

`BorrowedBuf` is a **safe wrapper around `&mut [MaybeUninit<T>]`** that does
this bookkeeping for you:

- `filled()`, `into_filled()` and `into_filled_mut()` expose only initialized
  data, as plain `&[T]` / `&mut [T]`, so your code never calls `assume_init`.
- `append()` copies straight into uninitialized memory, with no zeroing;
  `append_fill()` does the same for `n` copies of one value.
- `ensure_init()` initializes the unfilled part at most once per buffer, and
  hands back a `&mut [T]` for APIs like `Read::read`.
- A `BorrowedCursor` is append-only: a callee can't read, overwrite or
  de-initialize what the caller already filled.
- `with_unfilled_buf()` hands a callee a fresh `BorrowedBuf` over the unfilled
  part, so it can inspect what it wrote.
- Writing in place is still possible through a few `unsafe` methods
  (`as_mut`, `advance`, `set_init`), each with one documented precondition,
  so the unsafe surface is small and easy to audit.

## Example

```rust
use borrowed_buf::BorrowedBuf;
use core::mem::MaybeUninit;

let mut storage = [MaybeUninit::<u8>::uninit(); 16];
let mut buf: BorrowedBuf<'_, u8> = BorrowedBuf::from(&mut storage[..]);

// Hand out a cursor: the callee can only append, never touch what's already filled.
let mut cursor = buf.unfilled();
cursor.append(b"hello");

// Write straight into the unfilled region, then record the progress.
// SAFETY: only initialized bytes are written, and the first one is written before advancing.
unsafe {
    cursor.as_mut()[0].write(b'!');
    cursor.advance(1);
}

assert_eq!(buf.filled(), b"hello!");
assert!(!buf.is_init());
```

Any `Copy` element type works, e.g. decoding samples straight into uninitialized memory:

```rust
use borrowed_buf::{BorrowedBuf, BorrowedCursor};
use core::mem::MaybeUninit;

fn decode(input: &[u8], mut out: BorrowedCursor<'_, i16>) {
    let n = out.capacity().min(input.len() / 2);
    // SAFETY: only initialized values are written.
    let slots = unsafe { out.as_mut() };
    for (slot, pair) in slots.iter_mut().zip(input.chunks_exact(2)).take(n) {
        slot.write(i16::from_le_bytes([pair[0], pair[1]]));
    }
    // SAFETY: the first `n` elements of the cursor were just written.
    unsafe { out.advance(n) };
}

let mut storage = [MaybeUninit::<i16>::uninit(); 4];
let mut samples = BorrowedBuf::<i16>::from(&mut storage[..]);
decode(&[1, 0, 0xff, 0xff], samples.unfilled());
assert_eq!(samples.filled(), &[1, -1]);
```

Reading from `std::io::Read` (with the `std` feature):

```rust
# #[cfg(feature = "std")] {
use borrowed_buf::{io, BorrowedBuf};
use core::mem::MaybeUninit;

let mut reader: &[u8] = b"some bytes";
let mut storage = [MaybeUninit::<u8>::uninit(); 4];
let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);

io::read_buf_exact(&mut reader, buf.unfilled()).unwrap();
assert_eq!(buf.filled(), b"some");
# }
```

Note that `BorrowedBuf::from(&mut [MaybeUninit<T>])` can't always infer `T`, since
`MaybeUninit<T>` is itself `Copy` and `From<&mut [T]>` would also apply. Name the element type
(`BorrowedBuf::<u8>::from(..)`) when the compiler asks for it; `core::io` behaves the same.

## Differences from `core::io`

The `BorrowedBuf` and `BorrowedCursor` APIs are a superset of nightly's. Two methods are
additions, so code that uses them won't switch to `core::io` by changing the import alone:

- `BorrowedCursor::append_fill(n, value)` appends `n` copies of `value`, like `append` but
  without needing a source slice (e.g. for run-length or LZ77 distance-1 runs).
- `BorrowedBuf::truncate(len)` shortens the filled region, like `Vec::truncate`; the elements
  cut off stay initialized.

The rest of the additions live in the `io` module, because `std::io::Read` can't be extended
with new methods on stable:

| `core::io` / `std::io` (nightly)  | `borrowed-buf`                                   |
|-----------------------------------|--------------------------------------------------|
| `Read::read_buf(cursor)`          | `io::read_buf(&mut reader, cursor)`              |
| `Read::read_buf_exact(cursor)`    | `io::read_buf_exact(&mut reader, cursor)`        |
| `impl Write for BorrowedCursor<'_, u8>` | `impl Write for BorrowedCursor<'_, u8>`    |

## License

MIT
