# mzstd-ffi

C ABI for [mzstd](https://crates.io/crates/mzstd): one-shot Zstandard
compression and decompression into caller-owned buffers, byte-identical to
zstd 1.5.7's `ZSTD_compress`.

## Build

```sh
cargo build -p mzstd-ffi --release
```

This produces `target/release/libmzstd_ffi.a` and the shared library
(`libmzstd_ffi.so`, `.dylib` or `mzstd_ffi.dll`). The header is
[`include/mzstd.h`](include/mzstd.h).

```sh
cc -Imzstd-ffi/include app.c target/release/libmzstd_ffi.a -o app
```

Linking the static library may also need the system libraries Rust's
standard library uses; `cargo rustc -p mzstd-ffi --release --crate-type
staticlib -- --print native-static-libs` lists them for the host.

## Usage

```c
#include "mzstd.h"

size_t cap = mzstd_compress_bound(input_len);
uint8_t *compressed = malloc(cap);
size_t compressed_len = cap;
if (mzstd_compress(input, input_len, compressed, &compressed_len, 19) != MZSTD_OK) {
    /* handle error */
}

size_t output_len = expected_len;
mzstd_result r = mzstd_decompress(compressed, compressed_len, output, &output_len);
```

| Function | Contract |
| --- | --- |
| `mzstd_compress` | `level` `-131072..=22`, `0` for the default. An output of `mzstd_compress_bound(input_len)` bytes always suffices. |
| `mzstd_decompress` | Any sequence of Zstandard and skippable frames; an empty input decodes to nothing. |
| `mzstd_compress_bound` | Equal to `ZSTD_compressBound`; `0` on overflow. |

`*output_len` is the capacity on entry and the number of bytes written on
return (`0` after any failure). Null pointers with a non-zero length, lengths
above `PTRDIFF_MAX`, overlapping arguments and out-of-range levels return
`MZSTD_INVALID_PARAMETER`. Every call is independent and thread-safe, and a
Rust panic is reported as `MZSTD_ERROR` instead of unwinding into C.

See [architecture/c-abi.md](../architecture/c-abi.md) for the design.
