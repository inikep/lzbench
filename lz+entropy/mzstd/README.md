# mzstd

Zstandard (RFC 8878) compression and decompression in safe Rust.

[![Crates.io](https://img.shields.io/crates/v/mzstd.svg)][crate]
[![docs.rs](https://docs.rs/mzstd/badge.svg)][api]
[![Tests](https://github.com/M-Compressors/mzstd/actions/workflows/ci.yml/badge.svg?branch=main)][ci]
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

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
  and there is a C ABI in `mzstd-ffi`.

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
| Split one known input across your threads | `ParallelCompressor::prepare_slice` → `ScopedParallelBatch` | — |
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
| `hotpath`, `hotpath-cpu`, `hotpath-alloc` | no | profiling anchors |

## Performance

### Our benchmarks

Compression writes byte-identical frames to zstd 1.5.7 and reaches a median
1.099× its speed across the comparison corpus; decompression reaches 1.024×.
These medians include empty and tiny inputs. Text at the fastest compression
levels and compressible-text decoding remain slower.

<details>
<summary>Compression benchmark results and methodology</summary>

![Compression speed and output size relative to zstd 1.5.7 across levels -5 to 22][bench-chart]

**Recorded 2026-10-15 · mzstd 0.2.5 · Intel Core i7-13700KF · WSL2 · cold one-shot APIs.**
Each value is the median of per-dataset ratios across eight equally weighted datasets,
including empty and tiny inputs. A speed ratio above **1×** is faster than C; every
frame is byte-identical to zstd's, so the size ratio is **1×** at every level. The
median over all 192 dataset/level pairs is **1.099×**, with lower mean latency
in 157 pairs. Across levels, Alice text reaches a median **0.965×** and cyclic
Alice at 1 MiB **1.015×**; every other dataset's median is above 1×. Alice at
levels -5 to 1 remains slower (0.89–0.94×).

The benchmark includes compressor construction, allocation, compression, and disposal.
Results apply to the [recorded revision and machine][bench-levels].

[Per-dataset results and methodology][benchmarks] · [Raw measurements][bench-csv] · [Reproduce the benchmarks][benchmarking]

</details>

<details>
<summary>Decompression benchmark results and methodology</summary>

![Decompression speed relative to zstd 1.5.7 by source level][decoder-bench-chart]

**Recorded 2026-10-15 · mzstd 0.2.5 · Intel Core i7-13700KF · WSL2 · cold one-shot APIs.**
zstd C and mzstd decode identical C-written frames at source levels **-5 to 22**.
All **384 cases** restore the original bytes. The median over all 192 pairs is
**1.024×**: much faster on empty and tiny frames and on random data, and
0.81–1.01× on compressible text. Run-length data measured 0.91× in this sweep
and 1.38× in separate processes; its cold timing depends on allocator state. Construction, allocation, decode and disposal are timed; C reads the output
capacity from the frame header.

[Per-level and dataset results][decoder-benchmarks] · [Raw measurements][decoder-bench-csv] · [Methodology and limits][decoder-bench-report]

</details>

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
  reference-written frames.
- 100% function coverage, checked with `cargo llvm-cov`.

See [`architecture/`](architecture/README.md) for how the code is organized
and verified.

## Not supported

- Magicless frames
- `targetCBlockSize`
- External sequence producers
- Legacy (v0.x) frames
- Byte identity with zstd's multithreaded mode

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
[benchmarks]: ./docs/benchmarks/README.md
[bench-levels]: ./docs/benchmarks/encoders/README.md
[bench-csv]: ./docs/benchmarks/encoder-comparison.csv
[bench-chart]: ./docs/benchmarks/encoders/charts/overview.svg
[benchmarking]: ./docs/benchmarking.md
[decoder-benchmarks]: ./docs/benchmarks/decoders/README.md
[decoder-bench-csv]: ./docs/benchmarks/decoder-comparison.csv
[decoder-bench-chart]: ./docs/benchmarks/decoders/charts/overview.svg
[decoder-bench-report]: ./docs/benchmarks/decoder-comparison.md
