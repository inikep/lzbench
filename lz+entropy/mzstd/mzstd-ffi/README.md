# mzstd-ffi

C ABI for [mzstd](https://crates.io/crates/mzstd): one-shot and streaming
Zstandard compression and decompression into caller-owned buffers,
byte-identical to zstd 1.5.7's `ZSTD_compress` and `ZSTD_compressStream2`.

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

## Sessions

Encoder and decoder handles stream data through buffers of any size and are
reused across frames:

```c
mzstd_encoder *enc;
if (mzstd_encoder_create(3, &enc) != MZSTD_OK) { /* handle error */ }
mzstd_encoder_reset(enc, MZSTD_CONTENT_SIZE_UNKNOWN); /* or the exact size */

mzstd_status status;
do {
    size_t in_len = chunk_len, out_len = sizeof out;
    if (mzstd_encoder_process(enc, chunk, &in_len, out, &out_len,
                              last ? MZSTD_END : MZSTD_CONTINUE,
                              &status) != MZSTD_OK) { /* handle error */ }
    chunk += in_len; chunk_len -= in_len;
    write_out(out, out_len);
} while (chunk_len > 0 || (last && status != MZSTD_FINISHED));
mzstd_encoder_free(enc);
```

| Function | Contract |
| --- | --- |
| `mzstd_encoder_create` | `level` as for `mzstd_compress`; the first frame starts with an undeclared size. |
| `mzstd_encoder_reset` | Starts the next frame, declaring `pledged_size` bytes or `MZSTD_CONTENT_SIZE_UNKNOWN`. |
| `mzstd_encoder_process` | `ZSTD_compressStream2` with `MZSTD_CONTINUE`, `MZSTD_FLUSH` or `MZSTD_END`; repeat `FLUSH`/`END` while it reports `MZSTD_NEEDS_OUTPUT`. |
| `mzstd_decoder_create` | Windows up to `2^window_log_max` (`10..=31`, `0` for 27). |
| `mzstd_decoder_reset` | Starts the next operation. |
| `mzstd_decoder_process` | `ZSTD_decompressStream` with `MZSTD_DECODE_PROCESS`, then `MZSTD_DECODE_FINISH` on every call once the input is known to end, until `MZSTD_FINISHED`. |
| `mzstd_encoder_free`, `mzstd_decoder_free` | Release a handle; `NULL` is ignored. |

On each process call `*input_len` and `*output_len` are the bytes offered and
the room available on entry, and the bytes consumed and produced on return
(`0` after any failure); `*status` is written on `MZSTD_OK`. A codec error
fails the frame until the next reset. A handle may move between threads but
must serve one call at a time; one a panic interrupted reports `MZSTD_ERROR`
until it is freed.

See [architecture/c-abi.md](../architecture/c-abi.md) for the design.
