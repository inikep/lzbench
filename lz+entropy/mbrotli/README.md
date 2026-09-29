# mbrotli

**Brotli compression and decompression in safe Rust.**

A port of Google's Brotli encoder with a Rust-native API, plus a native Rust decoder.
Reuse working memory between payloads, integrate directly with `std::io::Read` and
`std::io::Write`, drive either codec incrementally, and opt into caller-scheduled
parallel compression when a workload benefits from it.

[![Crates.io](https://img.shields.io/crates/v/mbrotli.svg)][crate]
[![docs.rs](https://docs.rs/mbrotli/badge.svg)][api]
[![Tests](https://github.com/Mnwa/mbrotli/actions/workflows/ci.yml/badge.svg?branch=master)][ci]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)][license]

[Quick start](#quick-start) · [Benchmarks](#performance) · [`Read` / `Write`](#read-and-write-streaming) · [API guide][guide] · [Compatibility](#compatibility) · [Changelog][changelog]

## Why mbrotli?

- **Safe implementation, not a C wrapper.** Production code in this crate forbids
  `unsafe`. SIMD-accelerated compression uses `fearless_simd`.
- **Reference-compatible encoding.** Qualities **0–11**, with byte-for-byte comparisons
  against Google Brotli under [equivalent streaming settings](#compatibility).
- **Native Rust I/O.** Compress and decompress through one-shot Vec/slice APIs,
  `std::io::Read` / `Write` adapters, or explicit incremental sessions.
- **Reusable state.** Keep codec workspaces and destination buffers across requests instead
  of rebuilding them for every payload.
- **You control the threads.** Run parallel compression with scoped threads, Rayon,
  or your own scheduler. The library does not create a thread pool.

## Performance

Compression performance is generally close to or faster than Google C Brotli,
depending on the workload and quality setting.

<details>
<summary>Compression benchmark results and methodology</summary>

![Compression speed and output size relative to Google C Brotli across qualities 0–11][bench-chart]

**Recorded 2026-09-26 · Intel Core i7-13700KF · WSL2 · window 22 · cold serial APIs.**
Each value is the median of per-dataset ratios across eight equally weighted datasets,
including empty and tiny inputs. A speed ratio above **1×** is faster than C; a size
ratio below **1×** is smaller. These are compression results; the separate decoder comparison follows below.

The benchmark includes encoder construction, allocation, compression, and disposal.
Results apply to the [recorded revision and machine][bench-qualities].

[Per-dataset results and methodology][benchmarks] · [Raw measurements][bench-csv] · [Reproduce the benchmarks][benchmarking]

</details>

<details>
<summary>Decompression benchmark results and methodology</summary>

![Decompression speed relative to Google C Brotli by source quality][decoder-bench-chart]

**Recorded 2026-09-26 · Intel Core i7-13700KF · WSL2 · window 22 · cold serial APIs.**
Google C, mbrotli, Rust brotli and Burli decode identical C-generated streams at
source qualities **0–11**. SIMD Brotli shares Rust brotli's decoder and is omitted;
Burli decodes every source quality. All **384 cases** restore the original bytes.

Across the eight equally weighted inputs, the median speed / Burli is
**1.001×**, and the median speed / Google
C ranges from **3.0× to 3.7×** by source quality. Empty and tiny inputs retain
the same weight as larger datasets. The chart shows per-quality medians of
C-time/decoder-time ratios; above **1×** is faster than C. Throughput in the
quality pages counts restored bytes. Construction, allocation, decode and disposal
are timed. C receives the known output capacity; Rust brotli includes its native
4 KiB I/O adapter. These are cold API measurements on the recorded machine,
with results varying by workload and source quality.

[Per-quality and dataset results][decoder-benchmarks] · [Raw measurements][decoder-bench-csv] · [Methodology and limits][decoder-bench-report]

</details>

## Quick start

Requires **Rust 1.89 or later**. Add it to your project:

```toml
[dependencies]
mbrotli = "0.5.2"
```

```rust
use mbrotli::{Compressor, EncoderConfig, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = "brotli ".repeat(1000);
    let config = EncoderConfig::default().with_quality(Quality::Q5);
    let mut encoder = Compressor::new(config)?;

    let compressed = encoder.compress(input.as_bytes())?;
    println!("{} -> {} bytes", input.len(), compressed.len());
    Ok(())
}
```

Set the quality explicitly for your workload. `EncoderConfig::default()` uses
**quality 11**, the most expensive compression search.

## Choose your API

Compression and decompression expose the same core I/O shapes. Pick the shape that
matches how your application already moves bytes; parallel compression is a separate
execution strategy, not another streaming API.

| I/O shape | Compression | Decompression |
| --- | --- | --- |
| Return a new `Vec<u8>` | `Compressor::compress` | `Decompressor::decompress` |
| Append to an existing Vec | `compress_into` | `decompress_into` |
| Write into a caller-owned slice | `compress_to_slice` | `decompress_to_slice` |
| Pull output through `std::io::Read` | `Compressor::reader` | `Decompressor::reader` |
| Push input through `std::io::Write` | `Compressor::writer` | `Decompressor::writer` |
| Drive input/output incrementally | `start` → `EncoderSession` | `start` → `DecoderSession` |

### Reuse memory between payloads

`Compressor` and `Decompressor` own reusable working state. Keep the codec and your
output buffer alive across operations when allocation reuse matters.

<details>
<summary>Example: reuse the compressor and output buffer</summary>

```rust
use mbrotli::{Compressor, EncoderConfig, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut encoder = Compressor::new(
        EncoderConfig::default().with_quality(Quality::Q5),
    )?;
    let mut output = Vec::new();

    for input in [b"first payload".as_slice(), b"second payload".as_slice()] {
        output.clear(); // Keep the allocation; compress_into appends.
        let written = encoder.compress_into(input, &mut output)?;
        assert_eq!(written, 0..output.len());
    }
    Ok(())
}
```

</details>

## `Read` and `Write` streaming

Both codecs provide synchronous adapters for the standard Rust I/O traits. The two
adapter shapes are complementary:

- `reader(...)` wraps an input `Read` and exposes transformed bytes through `Read`.
- `writer(...)` wraps an output `Write` and accepts source bytes through `Write`.

That means you can plug Brotli into an existing pull-based or push-based pipeline
without first collecting the whole payload in memory.

### Compression I/O

`Compressor::reader` consumes **uncompressed** bytes from a `Read` source and yields
**compressed** bytes. `Compressor::writer` accepts **uncompressed** bytes and writes
**compressed** bytes to its sink. Encoder writers must be explicitly finished;
dropping one abandons the stream.

<details>
<summary>Compression with both <code>Read</code> and <code>Write</code></summary>

```rust
use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let payload = b"streamed payload";
    // Pull model: read compressed bytes from an uncompressed source.
    let mut encoder = Compressor::new(
        EncoderConfig::default().with_quality(Quality::Q5),
    )?;
    let stream = InputSize::Exact(payload.len() as u64).into();
    let mut reader = encoder.reader(&payload[..], stream)?;
    let mut compressed_from_reader = Vec::new();
    reader.read_to_end(&mut compressed_from_reader)?;

    // Push model: write uncompressed bytes into a compressed sink.
    let mut encoder = Compressor::new(
        EncoderConfig::default().with_quality(Quality::Q5),
    )?;
    let stream = InputSize::Exact(payload.len() as u64).into();
    let mut writer = encoder.writer(Vec::new(), stream)?;
    writer.write_all(payload)?;
    let compressed_from_writer = writer
        .finish()
        .map_err(mbrotli::io::FinishError::into_error)?;

    assert_eq!(compressed_from_reader, compressed_from_writer);
    Ok(())
}
```

</details>

`flush()` makes accepted input decodable without ending the encoder stream, and flush
boundaries can affect compressed bytes. Use `finish()` when the stream is complete.

### Decompression I/O

`Decompressor::reader` consumes a **compressed** `Read` source and yields the
**decompressed** payload. `Decompressor::writer` accepts **compressed** bytes and
writes the **decompressed** payload to its sink.

<details>
<summary>Decompression with both <code>Read</code> and <code>Write</code></summary>

```rust
use mbrotli::{DecodeStreamConfig, DecoderConfig, Decompressor};
use std::io::{Read, Write};

fn decode_with_reader(compressed: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut decoder = Decompressor::new(DecoderConfig::default())?;
    let mut reader = decoder.reader(compressed, DecodeStreamConfig::default())?;
    let mut output = Vec::new();
    reader.read_to_end(&mut output)?;
    Ok(output)
}

fn decode_with_writer(compressed: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut decoder = Decompressor::new(DecoderConfig::default())?;
    let mut writer = decoder.writer(Vec::new(), DecodeStreamConfig::default())?;
    writer.write_all(compressed)?;
    Ok(writer
        .finish()
        .map_err(mbrotli::io::FinishError::into_error)?)
}
```

</details>

The decoder adapters use bounded internal buffering. Reader read-ahead can be recovered
with `into_parts()`, while decoder-writer finalization reports truncated or invalid
input instead of silently accepting an incomplete stream.

## Parallel compression

Parallel compression is independent of the `Read`/`Write` adapters. It changes how
one compression job is scheduled: mbrotli splits the input into segments, exposes
work items to the caller, and assembles the completed segments back into one Brotli
stream. The library does not create or own a thread pool.

<details>
<summary>Example: parallel compression with scoped threads</summary>

```rust
use mbrotli::compressor::parallel::{
    BatchConfig, ParallelCompressor, ParallelConfig, TaskCount,
};
use mbrotli::{EncoderConfig, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = vec![b'a'; 8 << 20];
    let mut encoder = ParallelCompressor::new(
        EncoderConfig::default().with_quality(Quality::Q5),
        ParallelConfig::default(),
    )?;
    let mut batch = encoder.prepare_slice(
        &input,
        BatchConfig::auto(TaskCount::available()?),
    )?;
    let tasks = batch.take_tasks()?;

    std::thread::scope(|scope| {
        for task in tasks {
            scope.spawn(move || task.run());
        }
    });

    let mut output = Vec::new();
    batch.finish_into(&mut output)?;
    println!("{} -> {} bytes", input.len(), output.len());
    Ok(())
}
```

</details>

For fixed segment settings, parallel output is deterministic across task counts, but
it can differ in bytes and size from serial compression. The example stages compressed
segments in memory; see the [parallel guide][parallel] for budgets, disk staging,
file input, and other executors.

## Native decompression

`Decompressor` provides reusable Vec/slice APIs, incremental sessions, and synchronous
reader/writer adapters. Vec appends are rolled back on failure. The decoder specializes command loops
and history copies for the selected CPU backend using safe `fearless_simd` abstractions.

**Configure limits for untrusted input.** Numeric budgets are unlimited by default.
This example accepts standard windows and sets explicit input, output, and workspace
budgets; choose limits appropriate for your application.

```rust
use mbrotli::{DecodeLimits, DecoderConfig, Decompressor, WindowLimit};

fn decode_payload(input: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let limits = DecodeLimits::default()
        .with_max_input_bytes(Some(1 << 20))
        .with_max_output_bytes(Some(8 << 20))
        .with_max_workspace_bytes(Some(32 << 20));
    let config = DecoderConfig::default()
        .with_window_limit(WindowLimit::standard(24)?)
        .with_limits(limits);
    let mut decoder = Decompressor::new(config)?;

    Ok(decoder.decompress(input)?)
}
```

The workspace budget excludes caller-owned output and borrowed dictionaries; it is
not a total process-memory limit. Retain the decoder across calls when reuse matters.
See [decoder configuration and semantics][decoder] and [compatibility evidence][decoder-checks].

## Structured framing

With `compression,experimental`, a separate reusable owner encodes ordered
resources and metadata into one container. It supports alloc-backed one-shot
and native sessions; its writer and encoded reader require std APIs.

```rust
use mbrotli::framing::{FramedCompressor, FramedInput, FramedItem, FramedResource};
let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
let mut encoder = FramedCompressor::new(Default::default())?;
let bytes = encoder.compress(FramedInput::from(items.as_slice()))?;
assert_eq!(&bytes[..4], &[0x91, 10, 66, 82]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `encoder.framed_writer(sink, stream_config)` for gradually arriving resource
payload, or `encoder.framed_reader(input, stream_config)` to read encoded bytes
lazily. [API examples and migration](docs/dictionaries.md) cover metadata,
dictionaries and the move from the experimental raw-owner factory.

With `decompression,experimental`, `FramedDecompressor` decodes a container into
`FramedOutput`: resources in wire order with their metadata, global metadata, and
the validated container layout. Keep the decoder across calls to reuse its storage.
Compression support is optional; the decoder can be built on its own.

To decode `bytes` from the encoding example above:

```rust
use mbrotli::framing::FramedDecompressor;
let mut decoder = FramedDecompressor::new(Default::default())?;
let output = decoder.decompress(&bytes)?;
assert_eq!(output.resources.len(), 1);
assert_eq!(output.resources[0].data, b"hello");
# Ok::<(), Box<dyn std::error::Error>>(())
```

For incremental input, `decoder.start(stream_config)` exposes resource and metadata
events with payload fragments. With std APIs, `decoder.framed_reader(source,
stream_config)` wraps a `BufRead` source and exposes `next_event()`. This event
reader preserves resource boundaries; it does not flatten the container into one
`Read` stream.

`FramedDecodeConfig` defaults to `InputMode::FramedOnly`; `InputMode::Auto` also
accepts a raw Brotli member. `FramedDecodeLimits` configures resource, input, output,
metadata and workspace budgets. External dictionary references use an explicit
`DictionaryResolver`; resource checksums are recorded without verification.
The owner, one-shot APIs and native sessions also work with `no_std` and `alloc`;
`FramedReader` requires std APIs. See [framed decoder mechanics](architecture/framed-decoder.md)
for validation, dictionaries and streaming semantics.

## Select only the codecs you need

The default feature set is `std`, `compression`, and `decompression`. Disable default
features to select one codec or use `no_std` with `alloc`. For an alloc-backed decoder:

```toml
[dependencies]
mbrotli = { version = "0.5.2", default-features = false, features = ["no_std", "decompression"] }
```

Add `"compression"` for both codecs, or use `"std"` instead of `"no_std"` for standard
I/O support. `no_std` requires a global allocator; it excludes I/O adapters, parallel
compression, framed I/O adapters, and profiling, and uses compile-time
SIMD selection. The experimental framed decoder supports `no_std` with `alloc`;
its `FramedReader` I/O adapter requires std APIs.
Cargo features are additive: another dependency can re-enable a codec or `std`.
Leave `std` and `hotpath*` disabled throughout the dependency graph for a std-free build.

## Compatibility

The test reference in `brotli-ffi/vendor/brotli` is pinned to
**Google Brotli v1.2.0**, revision `4508218e`. Ordinary encoding at qualities
**0–11** and windows **10–24** is compared
byte-for-byte with equivalent C streaming settings.

That comparison requires matching configuration, dictionary, declared input size,
flush boundaries, and continuation offset. C one-shot shortcuts or arbitrary C chunk
schedules can produce different bytes. Within mbrotli's serial APIs, matching those
settings preserves output across input chunk sizes, SIMD backends, and buffer reuse.
This is format compatibility, not a drop-in replacement for another crate's Rust API.

| Codec capability | Availability |
| --- | --- |
| Standard Brotli (RFC 7932) | Qualities 0–11 |
| Large Window Brotli | Qualities 3–11 |
| Prepared LZ77 prefix dictionaries | Qualities 5–11 |
| Serialized dictionaries and custom static dictionary encoding | `experimental`; qualities 5–11 |
| Headerless stream continuations | `experimental`; qualities 2–11 |
| Shared Brotli framed compressor, structured input and native sessions | `compression,experimental`; std or alloc |
| Framed encoder writer/reader adapters | `compression,experimental`; not available with `no_std` |
| Framed decoder, structured output and native events | `decompression,experimental`; std or alloc |
| Framed decoder event reader | `decompression,experimental`; not available with `no_std` |

Unsupported combinations return errors. External dictionary references require the
same dictionaries at the decoder. Large Window and shared-dictionary streams require
a decoder that supports the corresponding extension. C-based end-to-end validation
of Large Window encoding covers windows up to 30 bits; wider declarations use separate
checks and do not have the same independent C-decoder evidence.

Enable the `experimental` Cargo feature to use the gated formats. Their API may change
in a patch release, and custom static encoding and framing have separate validation
from the ordinary encoder's byte-identity checks. See the [dictionary and format guide][dictionaries].

## Validation

The repository includes differential tests against the pinned C encoder and decoder,
cross-API and cross-backend checks, AFL++ fuzz targets, Miri checks, and AddressSanitizer
workflows. Read the [compatibility and validation guide][validation] and the
[native decoder checks][decoder-checks] for supported configurations,
reproduction commands, and limitations. Testing and fuzzing are evidence, not formal verification.

The crate enforces `#![cfg_attr(not(test), forbid(unsafe_code))]`. This describes
mbrotli's production implementation, not every transitive dependency: SIMD intrinsics
are encapsulated by `fearless_simd`, and differential tests use a C FFI reference.

## Documentation and contributing

[API reference][api] · [User guide][guide] · [Dictionaries][dictionaries] · [Parallel compression][parallel] · [Architecture][architecture]

Bug reports with a minimal input, codec settings, and reproduction steps are welcome.
For performance reports, include the CPU, compiler, feature set, and workload.
The [development guide][development] covers local checks, coverage, and fuzzing.

## License

[MIT][license]. The encoder builds on the work of the [Google Brotli project][google-brotli].

[crate]: https://crates.io/crates/mbrotli
[api]: https://docs.rs/mbrotli
[ci]: https://github.com/Mnwa/mbrotli/actions/workflows/ci.yml
[license]: ./LICENSE
[changelog]: ./CHANGELOG.md
[guide]: ./docs/README.md
[parallel]: ./docs/parallel.md
[dictionaries]: ./docs/dictionaries.md
[decoder]: ./architecture/decompressor.md
[decoder-checks]: ./architecture/decompressor-compatibility.md
[validation]: ./docs/correctness.md
[development]: ./docs/development.md
[architecture]: ./architecture/README.md
[benchmarks]: ./docs/benchmarks/README.md
[bench-qualities]: ./docs/benchmarks/encoders/README.md
[bench-csv]: ./docs/benchmarks/encoder-comparison.csv
[bench-chart]: ./docs/benchmarks/encoders/charts/overview.svg
[benchmarking]: ./docs/benchmarking.md
[google-brotli]: https://github.com/google/brotli

[decoder-benchmarks]: ./docs/benchmarks/decoders/README.md
[decoder-bench-csv]: ./docs/benchmarks/decoder-comparison.csv
[decoder-bench-chart]: ./docs/benchmarks/decoders/charts/overview.svg
[decoder-bench-report]: ./docs/benchmarks/decoder-comparison.md
