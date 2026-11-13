# mzstd

**Zstandard (RFC 8878) compression and decompression in safe Rust.**

A port of the zstd 1.5.7 encoder that writes the same bytes, plus an independent
Rust decoder. Reuse working memory between payloads, integrate directly with
`std::io::Read` and `std::io::Write`, drive either codec incrementally, and opt into
caller-scheduled parallel compression when a workload benefits from it.

[![Crates.io](https://img.shields.io/crates/v/mzstd.svg)][crate]
[![docs.rs](https://docs.rs/mzstd/badge.svg)][api]
[![Tests](https://github.com/M-Compressors/mzstd/actions/workflows/ci.yml/badge.svg?branch=main)][ci]
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![hotpath-cloud](https://img.shields.io/endpoint?url=https://hotpath.rs/badge-cloud.json)](https://hotpath.rs/app/repos/M-Compressors/mzstd)

**[Documentation][site]** · [Getting started][start] · [Choose your API][guide] · [Examples][examples] · [Benchmarks](#performance) · [API reference][api] · [Changelog](CHANGELOG.md)

## Quick start

Requires **Rust 1.89 or later**.

```toml
[dependencies]
mzstd = "0.2"
```

```rust
use mzstd::{Compressor, Decompressor, EncoderConfig, Level};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EncoderConfig::default().with_level(Level::try_from(19)?);
    let mut encoder = Compressor::new(config)?;
    let compressed = encoder.compress(b"hello, hello, hello")?;

    let restored = Decompressor::default().decompress(&compressed)?;
    assert_eq!(restored, b"hello, hello, hello");
    Ok(())
}
```

Before decoding input you do not control, set limits: the default window limit is
128 MiB, and input and output byte budgets are unlimited. See
[decompression and limits][decoder-guide]. [Getting started][start] continues with
memory reuse, streaming and feature selection.

## Why mzstd?

- **Byte-identical encoder.** For the same input and parameters,
  single-threaded compression writes exactly the bytes of zstd 1.5.7:
  one-shot calls match `ZSTD_compress2`, and sessions match
  `ZSTD_compressStream2` driven with the same chunks, output capacities and
  flushes. This holds for levels -131072 to 22, the explicit compression
  parameters, long distance matching, checksums, content sizes, pledged sizes,
  size hints, target compressed block sizes (`ZSTD_c_targetCBlockSize`) and
  dictionaries (raw or Zstandard-format, as `ZSTD_CCtx_loadDictionary` loads
  them).
- **Independent decoder** written from the format specification. It decodes
  every frame zstd writes, with or without a dictionary, with bounded window,
  input and output limits.
- **Parallel compression** with tasks you run on your own executor. Frames
  are valid Zstandard frames and are identical for any task count, but they
  are not byte-identical to zstd's multithreaded mode. Segments are primed
  with the input before them the way zstd's multithreaded mode primes a job.
- **No `unsafe`, no dependencies on zstd.** It runs with `no_std` + `alloc`,
  and there is a C ABI, one-shot and streaming, in `mzstd-ffi`.
- **Sans-I/O sessions.** `Encoder` and `Decoder` sessions work only with the buffers you
  pass in and never perform I/O themselves, so you can drive them from blocking code,
  async runtimes, or your own event loop.

## Choose your API

Compression and decompression expose the same I/O shapes. Pick the one that matches
how your application already moves bytes.

| I/O shape | Compression | Decompression | Guide |
| --- | --- | --- | --- |
| Return a new `Vec<u8>` | `Compressor::compress` | `Decompressor::decompress` | [Getting started][start] |
| Append to a Vec or write into a slice | `compress_into`, `compress_to_slice` | `decompress_into`, `decompress_to_slice` | [Buffers and memory reuse][memory] |
| Pull through `std::io::Read` | `Compressor::reader` | `Decompressor::reader` | [Read / Write I/O][io] |
| Push through `std::io::Write` | `Compressor::writer` | `Decompressor::writer` | [Read / Write I/O][io] |
| Drive buffers yourself, no I/O | `start` → `EncoderSession` | `start` → `DecoderSession` | Sans I/O [compression][sans-io-enc], [decompression][sans-io-dec] |
| Any of the above with a dictionary | `compress_with_dictionary*`, `*_with_dictionary` | `decompress_with_dictionary*`, `*_with_dictionary` | [Dictionaries][dictionaries] |
| Split one input across your threads | `ParallelCompressor` | Ordinary `Decompressor` | [Parallel compression][parallel] |
| Several resources in one seekable container | `FramedCompressor` | `FramedDecompressor` | [Framing containers][frames], `framed` feature |

The [full API table][guide] also lists owned sessions, uninitialized-output calls and
the parallel streaming entry points.

## Performance

### Our benchmarks

Compression writes byte-identical frames to zstd 1.5.7 and reaches a median
1.216× its speed across the comparison corpus; decompression reaches 1.521×.
These medians include empty and tiny inputs. Alice text remains slower at
some compression levels and at most decompression levels. On the whole
`silesia.tar` in lzbench compression is within about 2% of C up to level 8
and faster above it, and decompression is level with C at the fastest
levels and 3-6% behind from level 1 up.

<details>
<summary>Compression benchmark results and methodology</summary>

![Compression speed and output size relative to zstd 1.5.7 across levels -5 to 22][bench-chart]

**Recorded 2026-11-12 · mzstd 0.2.14 · AMD Ryzen 9 7950X (KVM, host shared with other tenants) · cold one-shot APIs.**
Each value is the median of per-dataset ratios across eight equally weighted datasets,
including empty and tiny inputs. A speed ratio above **1×** is faster than C; every
frame is byte-identical to zstd's, so the size ratio is **1×** at every level. The
median over all 192 dataset/level pairs is **1.216×**, with lower mean latency
in 174 pairs (2026-11-09: 1.248×, 174 pairs). The 18 slower pairs are Alice at
nine levels, cyclic Alice at four, structured binary at two and random data at
three (0.82–0.999×). Other tenants load the host: medians are stable, single
cells good to about ±10%.

The benchmark includes compressor construction, allocation, compression, and disposal.
Results apply to the [recorded revision and machine][bench-levels]; [lzbench](#lzbench)
covers a whole large archive.

[Per-dataset results and methodology][benchmarks] · [Raw measurements][bench-csv] · [Reproduce the benchmarks][benchmarking]

</details>

<details>
<summary>Decompression benchmark results and methodology</summary>

![Decompression speed relative to zstd 1.5.7 by source level][decoder-bench-chart]

**Recorded 2026-11-12 · mzstd 0.2.14 · AMD Ryzen 9 7950X (KVM, host shared with other tenants) · cold one-shot APIs.**
zstd C and mzstd decode identical C-written frames at source levels **-5 to 22**.
All **384 cases** restore the original bytes. The median over all 192 pairs is
**1.521×** (2026-11-09: 1.517×), faster in 152 pairs: much faster on empty and tiny
frames and on random data, 1.44× on structured binary and 1.58× on run-length
data, and 0.91× (Alice) and 0.96× (cyclic Alice) on compressible text.
Construction, allocation, decode and disposal are timed; C reads the output
capacity from the frame header.

[Per-level and dataset results][decoder-benchmarks] · [Raw measurements][decoder-bench-csv] · [Methodology and limits][decoder-bench-report]

</details>

### External benchmarks

#### lzbench

<details>
<summary>lzbench 2.4.1 results for mzstd 0.2.14 and zstd 1.5.7</summary>

[lzbench](https://github.com/inikep/lzbench) is a long-running open-source
in-memory benchmark: every codec is built from source into one binary with the
same compiler and flags and runs on the same data and machine. We ran lzbench
master at [`a6fa8aa`][lzbench-commit] (lzbench 2.4.1, gcc 15.2, C and Rust
codecs built with `-march=native`), with this release's sources in place of
the mzstd 0.2.11 it vendors, on the 211 MB `silesia.tar` of the
[Silesia corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia),
pinned to one thread of an AMD Ryzen 9 7950X VDS, with `-t8,8`, in three
rounds on 2026-11-12. [Command, raw output of every round and the 0.2.13 run][lzbench-run].

mzstd 0.2.14 against zstd 1.5.7; bold marks the faster result. Ratio is the
compressed size as a percentage of the input and matches for both: every frame
is byte-identical.

| Level | Compress mzstd | Compress zstd | Decompress mzstd | Decompress zstd | Ratio, % |
| ----: | -------------: | ------------: | ---------------: | --------------: | -------: |
| -22 | **2.44 MB/s** | 2.17 MB/s | 1524 MB/s | **1628 MB/s** | 24.67 |
| -18 | **4.95 MB/s** | 3.97 MB/s | 1804 MB/s | **1868 MB/s** | 25.15 |
| -15 | **8.71 MB/s** | 8.15 MB/s | 2024 MB/s | **2115 MB/s** | 26.83 |
| -11 | **44.6 MB/s** | 42.5 MB/s | 1975 MB/s | **2059 MB/s** | 27.35 |
| -8 | 102 MB/s | **103 MB/s** | 1941 MB/s | **2043 MB/s** | 28.17 |
| -5 | 205 MB/s | **210 MB/s** | 1806 MB/s | **1872 MB/s** | 29.60 |
| -3 | **379 MB/s** | 375 MB/s | 1812 MB/s | **1896 MB/s** | 31.20 |
| -2 | 536 MB/s | **542 MB/s** | 1872 MB/s | **1977 MB/s** | 32.70 |
| -1 | 640 MB/s | **654 MB/s** | 2039 MB/s | **2132 MB/s** | 34.53 |
| --fast=1 | 718 MB/s | **723 MB/s** | 2678 MB/s | **2712 MB/s** | 41.01 |
| --fast=3 | **819 MB/s** | 817 MB/s | 2862 MB/s | **2868 MB/s** | 44.63 |
| --fast=5 | 886 MB/s | **890 MB/s** | **3062 MB/s** | 3044 MB/s | 48.61 |

**Summary:** byte-identical output at every level; the fastest of three
rounds compresses 4.9-24.7% faster than zstd at levels 11, 15, 18 and 22,
within about 1% of its speed at `--fast=3`, `--fast=5`, `--fast=1`, level 3
and level 8, and 1.1-2.4% slower at levels 1, 2 and 5. It decompresses at
zstd's speed at `--fast=5` and `--fast=3`, 1.3% slower at `--fast=1` and
3.4-6.4% slower from level 1 up. Compression from level 5 up moves by up to
33 points between rounds on this shared host. Against 0.2.13 in the same
rounds, relative to zstd, decompression at the `--fast` levels went from
0.92-0.93 to 0.99-1.01 and gained 1.2-1.7 points at levels 1 to 3, and
compression gained 3.7-4.4 points at the `--fast` levels and 1.6-3.7 points
at levels 1 to 3.
lzbench compresses the whole archive as one frame, so it measures
large-input throughput; the medians above also weight empty and tiny
inputs, where mzstd is much faster.

</details>

#### TurboBench
Results available [here](https://github.com/powturbo/TurboBench).

## Compatibility

Single-threaded compression is byte-identical to zstd **1.5.7**: one-shot calls match
`ZSTD_compress2`, and sessions match `ZSTD_compressStream2` driven with the same
chunks, output capacities, flushes and pledged size. Parallel compression writes
valid, deterministic Zstandard frames, but not the bytes of zstd's multithreaded
mode. [Compatibility and validation][validation] states each claim with its oracle.

Framed containers (`features = ["framed"]`) are built only from Zstandard frames:
`zstd -d` prints the visible resources concatenated, everything else is in skippable
frames, and the last frame is a seek table in zstd's `contrib/seekable_format`
layout. The format is specified in [architecture/framing.md](architecture/framing.md).

Not supported:

- External sequence producers
- Legacy (v0.x) frames
- Byte identity with zstd's multithreaded mode

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

## Cargo features

The default feature set is `std`, `compression`, and `decompression`. Disable default
features to select one codec or to build for `no_std` with `alloc`:

```toml
[dependencies]
mzstd = { version = "0.2", default-features = false, features = ["no_std", "decompression"] }
```

`framed` enables seekable containers of resources and metadata. See
[Cargo features][features] for every feature and what `no_std` excludes.

## Documentation

| If you want to | Read |
| --- | --- |
| Build a first round trip | [Getting started][start] |
| Find the right entry point | [Choose your API][guide] |
| Tune the level, parameters, or features | [Configuration and features][configuration] |
| Control allocations | [Buffers and memory reuse][memory] |
| Decode untrusted input | [Decompression and limits][decoder-guide] |
| Stream with readers, writers, or sessions | [Streaming and completion][streaming] |
| Use several cores | [Parallel compression][parallel] |
| Use dictionaries or magicless frames | [Dictionaries and extended formats][dictionaries] |
| Package several resources | [Framing containers][frames] |
| Look up a type or method | [API reference on docs.rs][api] |
| Understand the implementation | [Architecture][architecture] |

The same pages live in [`docs/`](docs/README.md) and
[`architecture/`](architecture/README.md) in this repository.

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
[lzbench-commit]: https://github.com/inikep/lzbench/tree/a6fa8aa758932a02841c0cc68c07e5abf8870951
[lzbench-run]: ./docs/benchmarks/lzbench-2026-11-12.md
[site]: https://m-compressors.github.io/mzstd/
[start]: https://m-compressors.github.io/mzstd/docs/getting-started.html
[guide]: https://m-compressors.github.io/mzstd/docs/
[examples]: https://m-compressors.github.io/mzstd/docs/examples/
[configuration]: https://m-compressors.github.io/mzstd/docs/configuration.html
[features]: https://m-compressors.github.io/mzstd/docs/configuration.html#cargo-features
[memory]: https://m-compressors.github.io/mzstd/docs/memory.html
[decoder-guide]: https://m-compressors.github.io/mzstd/docs/decompression.html
[streaming]: https://m-compressors.github.io/mzstd/docs/streaming.html
[io]: https://m-compressors.github.io/mzstd/docs/io.html
[sans-io-enc]: https://m-compressors.github.io/mzstd/docs/sans-io-compression.html
[sans-io-dec]: https://m-compressors.github.io/mzstd/docs/sans-io-decompression.html
[parallel]: https://m-compressors.github.io/mzstd/docs/parallel.html
[frames]: https://m-compressors.github.io/mzstd/docs/frames.html
[dictionaries]: https://m-compressors.github.io/mzstd/docs/dictionaries.html
[validation]: https://m-compressors.github.io/mzstd/docs/correctness.html
[architecture]: https://m-compressors.github.io/mzstd/architecture/
