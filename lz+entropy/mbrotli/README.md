# mbrotli

**Brotli compression and decompression in safe Rust.**

A port of Google's Brotli encoder with a Rust-native API, plus a native Rust decoder.
Reuse working memory between payloads, integrate directly with `std::io::Read` and
`std::io::Write`, drive either codec incrementally, and opt into caller-scheduled
parallel compression when a workload benefits from it.

[![Crates.io](https://img.shields.io/crates/v/mbrotli.svg)][crate]
[![docs.rs](https://docs.rs/mbrotli/badge.svg)][api]
[![Tests](https://github.com/M-Compressors/mbrotli/actions/workflows/ci.yml/badge.svg?branch=master)][ci]
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![hotpath-cloud](https://img.shields.io/endpoint?url=https://hotpath.rs/badge-cloud.json)](https://hotpath.rs/app/repos/M-Compressors/mbrotli)

**[Documentation][site]** · [Getting started][start] · [Choose your API][guide] · [Examples][examples] · [Benchmarks](#performance) · [API reference][api] · [Changelog][changelog]

## Quick start

Requires **Rust 1.89 or later**.

```toml
[dependencies]
mbrotli = "0.5.22"
```

```rust
use mbrotli::{Compressor, Decompressor, EncoderConfig, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = "brotli ".repeat(1000);
    let config = EncoderConfig::default().with_quality(Quality::Q5);
    let mut encoder = Compressor::new(config)?;
    let compressed = encoder.compress(input.as_bytes())?;

    let mut decoder = Decompressor::new(Default::default())?;
    assert_eq!(decoder.decompress(&compressed)?, input.as_bytes());

    println!("{} -> {} bytes", input.len(), compressed.len());
    Ok(())
}
```

Two defaults to know before shipping:

- **Set the quality explicitly.** `EncoderConfig::default()` uses quality 11, the most
  expensive compression search. See [configuration][configuration].
- **Set limits for untrusted input.** Decoder budgets are unlimited by default. See
  [decompression and limits][decoder-guide].

[Getting started][start] continues with memory reuse, streaming and feature selection.

## Why mbrotli?

- **Safe implementation, not a C wrapper.** Production code in this crate forbids
  `unsafe`. SIMD-accelerated compression uses `fearless_simd`.
- **Reference-compatible encoding.** Qualities **0–11**, with byte-for-byte comparisons
  against Google Brotli under [equivalent streaming settings](#compatibility).
- **Native Rust I/O.** Compress and decompress through one-shot Vec/slice APIs,
  `std::io::Read` / `Write` adapters, or explicit incremental sessions.
- **Sans-I/O sessions.** `Encoder` and `Decoder` sessions work only with the buffers you
  pass in and never perform I/O themselves, so you can drive them from blocking code,
  async runtimes, or your own event loop.
- **Reusable state.** Keep codec workspaces and destination buffers across requests instead
  of rebuilding them for every payload.
- **You control the threads.** Run parallel compression with scoped threads, Rayon,
  or your own scheduler. The library does not create a thread pool.

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
| Split one input across your threads | `ParallelCompressor` | Ordinary `Decompressor` | [Parallel compression][parallel] |
| Several resources in one container | `FramedCompressor` | `FramedDecompressor` | [Framing containers][frames], `experimental` |

The [full API table][guide] also lists owned sessions and the parallel reader, writer
and streaming entry points.

## Performance

### Compression

![Compression speed and output size relative to Google C Brotli across qualities 0–11][bench-chart]

Across 96 corpus/quality pairs, mbrotli's median compression speed is **1.099×**
Google C Brotli, with lower mean latency in **67/96** cases. A speed ratio above
**1×** is faster; a size ratio below **1×** is smaller.

[Per-dataset results and methodology][benchmarks] · [Raw measurements][bench-csv] · [Reproduce][benchmarking]

### Decompression

![Decompression speed relative to Google C Brotli by source quality][decoder-bench-chart]

Across 96 corpus/quality pairs, mbrotli's median decompression speed is **2.588× C**
and **1.006× Burli**; per-quality medians against C range from **2.547× to 3.461×**.
All decoders restore identical C-generated streams.

[Per-quality and dataset results][decoder-benchmarks] · [Raw measurements][decoder-bench-csv] · [Methodology and limits][decoder-bench-report]

Both charts were recorded on 2026-10-30 on an AMD Ryzen 9 7950X VDS under Linux, with
window 22 and cold serial APIs: construction, allocation, the codec call and disposal
are all timed. Each bar is the median of eight equally weighted per-dataset ratios,
including empty and tiny inputs. Results vary by workload and quality, apply to the
[recorded source and machine][bench-qualities], and do not establish statistical
significance.

### External benchmarks

<details>
<summary>lzbench 2.4.1 results for mbrotli 0.5.20 and Google Brotli 1.2.0</summary>

[lzbench](https://github.com/inikep/lzbench) is a long-running open-source
in-memory benchmark: every codec is built from source into one binary with the
same compiler and flags and runs on the same data and machine. We ran lzbench
master at [`a6fa8aa`][lzbench-commit] (lzbench 2.4.1, gcc 15.2, C and Rust
codecs built with `-march=native`) on the 211 MB `silesia.tar` of the
[Silesia corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia),
pinned to one thread of an AMD Ryzen 9 7950X VDS, with `-t8,8`.
[Command and raw output][lzbench-run]. The lzbench maintainer's independent
run of mbrotli 0.5.2 is in the [lzbench 2.4 table][lzbench-results].

mbrotli 0.5.20 against Google Brotli 1.2.0; bold marks the faster result. Ratio
is the compressed size as a percentage of the input.

| Level | Compress mbrotli | Compress brotli | Decompress mbrotli | Decompress brotli | Ratio, % |
| ----: | ---------------: | --------------: | -----------------: | ----------------: | -------: |
| -11   |   **1.23 MB/s**  |       0.80 MB/s |       **713 MB/s** |          587 MB/s | 23.83 (brotli 23.82) |
| -8    |   **21.8 MB/s**  |       15.5 MB/s |       **862 MB/s** |          662 MB/s | 26.96 |
| -5    |   **78.2 MB/s**  |       71.1 MB/s |       **788 MB/s** |          601 MB/s | 28.10 |
| -2    |        186 MB/s  |    **197 MB/s** |       **703 MB/s** |          542 MB/s | 32.12 |
| -0    |    **549 MB/s**  |        531 MB/s |       **585 MB/s** |          452 MB/s | 37.02 |

**Summary:** the same compressed size as Google Brotli at qualities 0–8 and
+0.03% at 11; **41% and 54% faster** compression at -8 and -11, 10% faster at
-5, and within run-to-run noise at -0 and -2 (a repeat of -2 measured 203 MB/s
for mbrotli and 204 MB/s for brotli); decompression **22–31% faster** at every
level.

</details>

TurboBench results are available in the [TurboBench repository](https://github.com/powturbo/TurboBench).

## Compatibility

The test reference in `brotli-ffi/vendor/brotli` is pinned to
**Google Brotli v1.2.0**, revision `4508218e`. Ordinary encoding at qualities
**0–11** and windows **10–24** is compared byte-for-byte with equivalent C
streaming settings: the same configuration, dictionary, declared input size, flush
boundaries, and continuation offset. C one-shot shortcuts or arbitrary C chunk
schedules can produce different bytes. Within mbrotli's serial APIs, matching those
settings preserves output across input chunk sizes, SIMD backends, and buffer reuse.
This is format compatibility, not a drop-in replacement for another crate's Rust API.

| Format | Availability |
| --- | --- |
| Standard Brotli (RFC 7932) | Qualities 0–11 |
| Large Window Brotli | Qualities 3–11 |
| Prepared LZ77 prefix dictionaries | Qualities 5–11 |
| Serialized and custom static dictionaries, headerless continuations, Shared Brotli framing | `experimental` feature; the API may change in a patch release |

Unsupported combinations return errors. The [capability table][capabilities] has the
per-format details, and [compatibility and validation][validation] states each claim
with its oracle and its limits.

## Validation

The repository includes differential tests against the pinned C encoder and decoder,
cross-API and cross-backend checks, AFL++ fuzz targets, Miri, AddressSanitizer, and
[formal verification][formal] of selected code in Lean 4. See the
[validation guide][validation] and the [native decoder checks][decoder-checks] for
supported configurations, reproduction commands, and limitations.

The crate enforces `#![cfg_attr(not(test), forbid(unsafe_code))]`. This describes
mbrotli's production implementation, not every transitive dependency: SIMD intrinsics
are encapsulated by `fearless_simd`, and differential tests use a C FFI reference.

## Cargo features

The default feature set is `std`, `compression`, and `decompression`. Disable default
features to select one codec or to build for `no_std` with `alloc`:

```toml
[dependencies]
mbrotli = { version = "0.5.22", default-features = false, features = ["no_std", "decompression"] }
```

`experimental` enables framing, serialized dictionaries and continuations. See
[Cargo features][features] for every combination and what `no_std` excludes.

## Documentation

| If you want to | Read |
| --- | --- |
| Build a first round trip | [Getting started][start] |
| Find the right entry point | [Choose your API][guide] |
| Tune quality, window, or features | [Configuration and features][configuration] |
| Control allocations | [Buffers and memory reuse][memory] |
| Decode untrusted input | [Decompression and limits][decoder-guide] |
| Stream with readers, writers, or sessions | [Streaming and completion][streaming] |
| Use several cores | [Parallel compression][parallel] |
| Use dictionaries or Large Window | [Dictionaries and extended formats][dictionaries] |
| Package several resources | [Framing containers][frames] |
| Look up a type or method | [API reference on docs.rs][api] |
| Understand the implementation | [Architecture][architecture] |

The same pages live in [`docs/`](docs/README.md) and
[`architecture/`](architecture/README.md) in this repository.

## Contributing

Bug reports with a minimal input, codec settings, and reproduction steps are welcome.
For performance reports, include the CPU, compiler, feature set, and workload.
If compression output differs from Google Brotli under equivalent streaming settings,
open an issue with the [Output differs from Google Brotli][mismatch] template.

Read [CONTRIBUTING.md][contributing] for the checks a change must pass, and follow
the [Code of Conduct][conduct]. The [development guide][development] covers local
checks, coverage, and fuzzing. Report security vulnerabilities privately as described
in [SECURITY.md][security], not as public issues.

## License

[MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option. The encoder builds on the work of the [Google Brotli project][google-brotli].

[site]: https://m-compressors.github.io/mbrotli/
[start]: https://m-compressors.github.io/mbrotli/docs/getting-started.html
[guide]: https://m-compressors.github.io/mbrotli/docs/
[examples]: https://m-compressors.github.io/mbrotli/docs/examples/
[configuration]: https://m-compressors.github.io/mbrotli/docs/configuration.html
[features]: https://m-compressors.github.io/mbrotli/docs/configuration.html#cargo-features
[capabilities]: https://m-compressors.github.io/mbrotli/docs/configuration.html#what-each-quality-supports
[memory]: https://m-compressors.github.io/mbrotli/docs/memory.html
[decoder-guide]: https://m-compressors.github.io/mbrotli/docs/decompression.html
[streaming]: https://m-compressors.github.io/mbrotli/docs/streaming.html
[io]: https://m-compressors.github.io/mbrotli/docs/io.html
[sans-io-enc]: https://m-compressors.github.io/mbrotli/docs/sans-io-compression.html
[sans-io-dec]: https://m-compressors.github.io/mbrotli/docs/sans-io-decompression.html
[parallel]: https://m-compressors.github.io/mbrotli/docs/parallel.html
[frames]: https://m-compressors.github.io/mbrotli/docs/frames.html
[dictionaries]: https://m-compressors.github.io/mbrotli/docs/dictionaries.html
[architecture]: https://m-compressors.github.io/mbrotli/architecture/
[formal]: ./architecture/formal-verification.md
[crate]: https://crates.io/crates/mbrotli
[api]: https://docs.rs/mbrotli
[ci]: https://github.com/M-Compressors/mbrotli/actions/workflows/ci.yml
[changelog]: ./CHANGELOG.md
[decoder-checks]: ./architecture/decompressor-compatibility.md
[validation]: ./docs/correctness.md
[development]: ./docs/development.md
[contributing]: ./CONTRIBUTING.md
[conduct]: ./CODE_OF_CONDUCT.md
[security]: ./SECURITY.md
[mismatch]: https://github.com/M-Compressors/mbrotli/issues/new?template=output_mismatch.yml
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
[lzbench-results]: https://github.com/inikep/lzbench/blob/2706ba16721e947a9240c70d080b8efa3f752da0/doc/lzbench24_sorted.md
[lzbench-commit]: https://github.com/inikep/lzbench/tree/a6fa8aa758932a02841c0cc68c07e5abf8870951
[lzbench-run]: ./docs/benchmarks/lzbench-2026-11-08.md
