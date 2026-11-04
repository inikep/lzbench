# mbrotli-ffi

C ABI for [mbrotli](https://crates.io/crates/mbrotli): one-shot and streaming
Brotli compression and decompression into caller-owned buffers. One-shot
compression is byte-identical to Google's `BrotliEncoderCompress`.

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

## Sessions

Opaque `mbrotli_encoder_session` and `mbrotli_decoder_session` handles expose
`create`, `process`, `flush`, `finish`, `reinit` and `destroy`. Each owns its
codec, can be reused with `reinit`, and must be destroyed exactly once. Distinct
handles can run concurrently; access to a single handle must be exclusive.
Only pointers cross the ABI: the C structs are incomplete, so Rust session
layout changes do not affect callers. Never define, allocate, dereference or
copy the handle object itself; use create/destroy from the same library instance.

`process`, `flush` and `finish` accept **uninitialized output**. On entry,
`*input_len` is available input and `*output_len` is capacity; on return they
are consumed and produced bytes. Read only the reported output prefix.
`MBROTLI_NEEDS_OUTPUT` means supply more output space and the remaining input
suffix; `MBROTLI_NEEDS_INPUT` means supply more input (or encoder flush complete).
`MBROTLI_FINISHED` means all output has been delivered.

Encoder process accepts `MBROTLI_PROCESS`, `MBROTLI_FLUSH`, `MBROTLI_FINISH`;
decoder accepts Process and Finish. The separate `flush`/`finish` functions
supply empty input. Finish declares EOF and must be repeated with exactly the
remaining input suffix until consumed; only then may the input-free finish be
used. Decoder flush drains accepted data without declaring EOF. Unlike the
one-shot decoder, the session leaves trailing bytes unconsumed at Finished.

Rejected arguments leave counts unchanged. Codec errors require reinit or
destroy; decoder errors retain exact partial counts and initialized output,
while encoder errors and panics zero counts and leave output unspecified.
Full pointer/lifetime contracts are in the header. Streaming encoding follows
Rust sessions and does not promise the one-shot compressed-size bound.

See [examples/sessions.c](examples/sessions.c) for a complete compilable example.

## License

MIT OR Apache-2.0
