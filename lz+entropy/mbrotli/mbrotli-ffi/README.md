# mbrotli-ffi

C ABI for [mbrotli](https://crates.io/crates/mbrotli): one-shot Brotli
compression and decompression into caller-owned buffers, byte-identical to
Google's `BrotliEncoderCompress`.

## Build

```sh
cargo build -p mbrotli-ffi --release
```

This produces `target/release/libmbrotli_ffi.a` and the shared library
(`libmbrotli_ffi.so`, `.dylib` or `mbrotli_ffi.dll`). The header is
[`include/mbrotli.h`](include/mbrotli.h).

```sh
cc -Imbrotli-ffi/include app.c target/release/libmbrotli_ffi.a -o app
```

Linking the static library may also need the system libraries Rust's
standard library uses; `cargo rustc -p mbrotli-ffi --release --crate-type
staticlib -- --print native-static-libs` lists them for the host.

## Usage

```c
#include "mbrotli.h"

size_t cap = mbrotli_compress_bound(input_len);
uint8_t *compressed = malloc(cap);
size_t compressed_len = cap;
if (mbrotli_compress(input, input_len, compressed, &compressed_len, 11, 22) != MBROTLI_OK) {
    /* handle error */
}

size_t output_len = expected_len;
mbrotli_result r = mbrotli_decompress(compressed, compressed_len, output, &output_len);
```

| Function | Contract |
| --- | --- |
| `mbrotli_compress` | `quality` `0..=11`, `lgwin` `10..=24`. An output of `mbrotli_compress_bound(input_len)` bytes always suffices. |
| `mbrotli_decompress` | Exactly one complete stream; trailing bytes are `MBROTLI_ERROR`. |
| `mbrotli_compress_bound` | Equal to `BrotliEncoderMaxCompressedSize`; `0` on overflow. |

`*output_len` is the capacity on entry and the number of bytes written on
return (`0` after any failure). Null pointers with a non-zero length, lengths
above `PTRDIFF_MAX` and overlapping arguments return
`MBROTLI_INVALID_PARAMETER`. Every call is independent and thread-safe, and a
Rust panic is reported as `MBROTLI_ERROR` instead of unwinding into C.

See [architecture/c-abi.md](../architecture/c-abi.md) for the design.
