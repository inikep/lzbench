# mzstd

Zstandard (RFC 8878) compression and decompression in safe Rust.

[![Crates.io](https://img.shields.io/crates/v/mzstd.svg)][crate]
[![docs.rs](https://docs.rs/mzstd/badge.svg)][api]
[![Tests](https://github.com/M-Compressors/mzstd/actions/workflows/ci.yml/badge.svg?branch=main)][ci]
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![hotpath-cloud](https://img.shields.io/endpoint?url=https://hotpath.rs/badge-cloud.json)](https://hotpath.rs/app/repos/M-Compressors/mzstd)

- **Byte-identical encoder.** For the same input and parameters,
  single-threaded compression writes exactly the bytes of zstd 1.5.7:
  one-shot calls match `ZSTD_compress2`, and sessions match
  `ZSTD_compressStream2` driven with the same chunks, output capacities and
  flushes. This holds for levels -131072 to 22, the explicit compression
  parameters, long distance matching, checksums, content sizes, pledged sizes,
  size hints and dictionaries (raw or Zstandard-format, as
  `ZSTD_CCtx_loadDictionary` loads them).
- **Independent decoder** written from the format specification. It decodes
  every frame zstd writes, with or without a dictionary, with bounded window,
  input and output limits.
- **Parallel compression** with tasks you run on your own executor. Frames
  are valid Zstandard frames and are identical for any task count, but they
  are not byte-identical to zstd's multithreaded mode. Segments are primed
  with the input before them the way zstd's multithreaded mode primes a job.
- **No `unsafe`, no dependencies on zstd.** It runs with `no_std` + `alloc`,
  and there is a C ABI, one-shot and streaming, in `mzstd-ffi`.

## Performance

### Our benchmarks

Compression writes byte-identical frames to zstd 1.5.7 and reaches a median
1.256× its speed across the comparison corpus; decompression reaches 1.502×.
These medians include empty and tiny inputs. Alice text remains slower at
some compression levels and at every decompression level, and on whole
Silesia files both codecs still trail C by 2–6% at most levels.

<details>
<summary>Compression benchmark results and methodology</summary>

![Compression speed and output size relative to zstd 1.5.7 across levels -5 to 22][bench-chart]

**Recorded 2026-10-30 · working tree on `0a4fd87` and main · AMD Ryzen 9 7950X (KVM) · cold one-shot APIs.**
Each value is the median of per-dataset ratios across eight equally weighted datasets,
including empty and tiny inputs. A speed ratio above **1×** is faster than C; every
frame is byte-identical to zstd's, so the size ratio is **1×** at every level. The
median over all 192 dataset/level pairs is **1.256×**, with lower mean latency
in 179 pairs (main: 1.254×, 178 pairs). The slower pairs are Alice at levels
-1, 1, 2, 4-9, 11 and 12 (0.94–0.99×) and random 1 MiB at levels 3 and 4.

The benchmark includes compressor construction, allocation, compression, and disposal.
Results apply to the [recorded revision and machine][bench-levels]; [Silesia][silesia]
covers whole large files with reused contexts.

[Per-dataset results and methodology][benchmarks] · [Raw measurements][bench-csv] · [Reproduce the benchmarks][benchmarking]

</details>

<details>
<summary>Decompression benchmark results and methodology</summary>

![Decompression speed relative to zstd 1.5.7 by source level][decoder-bench-chart]

**Recorded 2026-10-30 · working tree on `0a4fd87` and main · AMD Ryzen 9 7950X (KVM) · cold one-shot APIs.**
zstd C and mzstd decode identical C-written frames at source levels **-5 to 22**.
All **384 cases** restore the original bytes. The median over all 192 pairs is
**1.502×** (main 1.400×), faster in 143 pairs: much faster on empty and tiny
frames and on random data, 1.44× on structured binary and 1.51× on run-length
data, and 0.94× (Alice) and 0.97× (cyclic Alice) on compressible text.
Construction, allocation, decode and disposal are timed; C reads the output
capacity from the frame header.

[Per-level and dataset results][decoder-benchmarks] · [Raw measurements][decoder-bench-csv] · [Methodology and limits][decoder-bench-report]

</details>

```toml
[dependencies]
mzstd = "0.2"
```

## Usage

```rust
use mzstd::{Compressor, Decompressor, EncoderConfig, Level};

let mut encoder = Compressor::new(EncoderConfig::default().with_level(Level::try_from(19)?))?;
let compressed = encoder.compress(b"hello, hello, hello")?;

let restored = Decompressor::default().decompress(&compressed)?;
assert_eq!(restored, b"hello, hello, hello");
# Ok::<(), Box<dyn std::error::Error>>(())
```

Explicit parameters override the level just as `ZSTD_CCtx_setParameter` does:

```rust
use mzstd::{CompressionParams, Compressor, EncoderConfig, LdmParams, Level, LongDistanceMatching, Strategy};

let config = EncoderConfig::default()
    .with_level(Level::try_from(9)?)
    .with_checksum(true)
    .with_params(CompressionParams::default().with_window_log(24).with_strategy(Strategy::Lazy2))
    .with_long_distance_matching(LongDistanceMatching::Enabled(LdmParams::default()));
let mut encoder = Compressor::new(config)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Dictionaries make small inputs compress well. Prepare one for the
configuration you compress with; the same dictionary decodes its frames:

```rust
use mzstd::dictionary::DictionaryBuilder;
use mzstd::{Compressor, Decompressor, EncoderConfig};

let config = EncoderConfig::default();
// Raw content here; a trained Zstandard-format dictionary works the same way.
let dictionary = DictionaryBuilder::new(br#"{"user":"alice","city":"Oslo","tags":["a","b"]}"#)
    .with_config(config)
    .build()?;
let record = br#"{"user":"alice","city":"Oslo","tags":["b"]}"#;
let frame = Compressor::new(config)?.compress_with_dictionary(&dictionary, record)?;
let restored = Decompressor::default().decompress_with_dictionary(&dictionary, &frame)?;
assert_eq!(restored, record);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Choose your API

Compression and decompression expose the same core I/O shapes. Pick the shape that
matches how your application already moves bytes. `ParallelCompressor` (in
`mzstd::compressor::parallel`) is a separate execution strategy for compression: it
splits one frame into independent segments and hands them to your executor as tasks,
either for a whole input at once or for input that arrives in chunks. It requires `std`.

| I/O shape | Compression | Decompression |
| --- | --- | --- |
| Return a new `Vec<u8>` | `Compressor::compress` | `Decompressor::decompress` |
| Append to an existing Vec | `compress_into` | `decompress_into` |
| Write into a caller-owned slice | `compress_to_slice`, `compress_to_uninit` | `decompress_to_slice` |
| Pull output through `std::io::Read` | `Compressor::reader` | `Decompressor::reader` |
| Push input through `std::io::Write` | `Compressor::writer` | `Decompressor::writer` |
| Drive input/output incrementally | `start` → `EncoderSession` | `start` → `DecoderSession` |
| Same, with the session owning the codec | `into_session` → `EncoderSessionOwned` | `into_session` → `DecoderSessionOwned` |
| Any of the above with a dictionary | `compress_with_dictionary*`, `start_with_dictionary`, `into_session_with_dictionary`, `writer_with_dictionary`, `reader_with_dictionary` | `decompress_with_dictionary*`, `start_with_dictionary`, `into_session_with_dictionary`, `writer_with_dictionary`, `reader_with_dictionary` |
| Split one known input across your threads | `ParallelCompressor::prepare_slice` / `prepare_source` → `ScopedParallelBatch` | — |
| Stream chunked input across your threads | `ParallelCompressor::start` → `ParallelEncoderSession` + `ParallelTaskQueue` | — |
| Same, with the session owning the compressor | `ParallelCompressor::into_session` → `ParallelEncoderSessionOwned` | — |
| Push or pull through `std::io` across your threads | `ParallelCompressor::writer`, `ParallelCompressor::reader` | — |

## Features

| Feature | Default | Enables |
| --- | --- | --- |
| `compression` | yes | `mzstd::compressor` |
| `decompression` | yes | `mzstd::decompressor` |
| `std` | yes | `std::error::Error`, `mzstd::io`, parallel compression |
| `no_std` | no | alloc-only build; overrides `std` |
| `framed` | no | `mzstd::framing`: seekable containers of resources and metadata, built only from Zstandard frames (see below) |
| `hotpath`, `hotpath-cpu`, `hotpath-alloc`, `hotpath-callgrind`, `hotpath-cloud` | no | profiling anchors; Callgrind instruction counts and hotpath Cloud upload |

## Framed containers

With `features = ["framed"]`, `mzstd::framing` stores ordered resources,
metadata, padding, a central directory and a footer in one container,
with the same API as mbrotli's framing. The container is only Zstandard
frames: every visible resource is a sequence of regular frames, so
`zstd -d` prints the visible resources concatenated; everything else is
in skippable frames; and the last frame is a seek table in zstd's
`contrib/seekable_format` layout, which seekable readers use to jump to
any offset.

```rust
use mzstd::framing::{FramedCompressor, FramedDecompressor, FramedItem, FramedResource};

let items = [FramedItem::Resource(FramedResource::from(&b"payload"[..]))];
let container = FramedCompressor::new(Default::default())?.compress(items.as_slice().into())?;
let output = FramedDecompressor::new(Default::default())?.decompress(&container)?;
assert_eq!(output.resources[0].data, b"payload");
```

`FramedDecompressor::framed_seek_reader` reads one resource of a container
through any `Read + Seek` source. The format is specified in
[architecture/framing.md](architecture/framing.md).

## Verification

- Differential tests against zstd 1.5.7 (built from the vendored submodule by
  `zstd-ffi`). They cover every level on a standard corpus, thousands of
  random advanced-parameter configurations, streaming schedules with flushes
  and wrapped windows, LDM, short destinations, and a build mode that forces
  index overflow correction on both sides.
- Decoder tests: reference frames through every entry point, upstream's
  golden vectors, truncation and bit-flip corruption, limits.
- AFL++ targets in `fuzz/afl`. Encoder targets: differential, output
  capacity, sessions, lifecycle, parallel and parameter parsing. Decoder
  targets: raw input against the reference decoder, chunked sessions, and
  reference-written frames. ClusterFuzzLite runs the same targets under
  libFuzzer and AddressSanitizer daily and on pull requests that touch code.
- Miri over the code that reaches `unsafe` in dependencies (the
  `fearless_simd` vector kernels and `borrowed-buf` uninitialized output),
  and AddressSanitizer over the integration tests, on every release tag.
- 100% function coverage, checked with `cargo llvm-cov`.
- [Formal verification](architecture/formal-verification.md) in Lean 4 of
  parts of the format against RFC 8878 itself, not the C library: Charon and
  Aeneas translate the Rust code, and machine-checked proofs show that the
  frame and block headers, `Number_of_Sequences`, the length and offset code
  tables, the default distributions, the repeat offset update and the FSE
  state transitions follow the RFC, that encoder and decoder agree on them,
  and that the translated code cannot panic or overflow on the inputs the
  proofs cover.

See [`architecture/`](architecture/README.md) for how the code is organized
and verified.

## Not supported

- Magicless frames
- `targetCBlockSize`
- External sequence producers
- Legacy (v0.x) frames
- Byte identity with zstd's multithreaded mode

## Contributing

Bug reports, differential test cases, fuzz findings and pull requests are
welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the checks a
change must pass, and follow the [Code of Conduct](CODE_OF_CONDUCT.md). If
compression output differs from zstd 1.5.7, open an issue with the
[Output differs from zstd][mismatch] template.

Report security vulnerabilities privately as described in
[SECURITY.md](SECURITY.md), not as public issues.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option. The encoder is a port of zstd, used under its BSD license;
see [NOTICE](NOTICE).

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

[crate]: https://crates.io/crates/mzstd
[api]: https://docs.rs/mzstd
[ci]: https://github.com/M-Compressors/mzstd/actions/workflows/ci.yml
[mismatch]: https://github.com/M-Compressors/mzstd/issues/new?template=output_mismatch.yml
[benchmarks]: ./docs/benchmarks/README.md
[bench-levels]: ./docs/benchmarks/encoders/README.md
[bench-csv]: ./docs/benchmarks/encoder-comparison.csv
[bench-chart]: ./docs/benchmarks/encoders/charts/overview.svg
[benchmarking]: ./docs/benchmarking.md
[decoder-benchmarks]: ./docs/benchmarks/decoders/README.md
[decoder-bench-csv]: ./docs/benchmarks/decoder-comparison.csv
[decoder-bench-chart]: ./docs/benchmarks/decoders/charts/overview.svg
[decoder-bench-report]: ./docs/benchmarks/decoder-comparison.md
[silesia]: ./docs/benchmarks/silesia.md
