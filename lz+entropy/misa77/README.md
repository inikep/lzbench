# misa77 (0.8.0)

misa77 is an LZ-based compressor that targets the write-once, read-many niche. In particular, it aims to satisfy the following criteria:

1. Extremely high decompression throughput (single-threaded, and in-memory).
2. Good compression ratios relative to decode throughput. For reference, misa77 levels that correspond to LZ4 and zstd-class ratios decode roughly 2 times faster than their counterparts respectively.
3. Constant memory use during compression, regardless of input size. No extra memory use during decompression (except at level 5, which needs 1.4 MB).

Slow compression is the obvious tradeoff that we make to achieve the above.

misa77 has seven compression effort levels as of v0.8.0. For most shapes of data, encode time increases and ratio decreases with increasing level, but decompression throughput is *not* similarly monotonic. In general, level 1 offers the fastest decode throughput. More information about effort levels can be found [here](#levels).

Decompression for all levels is always safe as of v0.8.0, i.e. the decompressor is guaranteed to exit gracefully (i.e. terminate, and not access out-of-bounds memory or engage in any other UB) in the case of corrupt/malicious input.

## Benchmarks

Let's first see some cross-platform results for the [Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip).

Note:

1. "Ratio" ahead is equal to `((compressed size)/(original)) * 100` (so lower is better).
2. misa77 has been merged into [lzbench](https://github.com/inikep/lzbench/) and [TurboBench](https://github.com/powturbo/TurboBench). The results below have been produced using lzbench, and can be reproduced through both benchmarking harnesses.
3. In the tables ahead, rows are grouped by codec family, and ordered from lowest to highest effort level within each family. The recommended misa77 levels are in bold.

---

### Intel x86-64

Details:

- CPU: Intel(R) Core(TM) i7-14650HX (@2.2 GHz) (Intel Turbo disabled).
- Single threaded, pinned to a single performance core.
- CPU governor set to `performance`.

| Compressor name   | Compression | Decompress. | Ratio | Filename    |
| ----------------- | ----------- | ----------- | ----- | ----------- |
| misa77 0.8.0 --1  |    229 MB/s |   4412 MB/s | 46.74 | silesia.tar |
| misa77 0.8.0 -0   |    171 MB/s |   4875 MB/s | 44.60 | silesia.tar |
| **misa77 0.8.0 -1** | **54.2 MB/s** | **5588 MB/s** | **42.64** | silesia.tar |
| misa77 0.8.0 -2   |   41.4 MB/s |   5078 MB/s | 40.33 | silesia.tar |
| misa77 0.8.0 -3   |   10.6 MB/s |   5020 MB/s | 37.76 | silesia.tar |
| **misa77 0.8.0 -4** | **1.75 MB/s** | **5144 MB/s** | **32.34** | silesia.tar |
| **misa77 0.8.0 -5** | **0.85 MB/s** | **2485 MB/s** | **24.79** | silesia.tar |
| lz4 1.11.0        |    382 MB/s |   2530 MB/s | 47.32 | silesia.tar |
| lz4hc 1.11.0 -9   |   21.2 MB/s |   2466 MB/s | 36.75 | silesia.tar |
| lz4hc 1.11.0 -12  |   7.83 MB/s |   2544 MB/s | 36.45 | silesia.tar |
| zxc 1.0.0 -3      |    122 MB/s |   3667 MB/s | 46.59 | silesia.tar |
| zxc 1.0.0 -7      |   5.09 MB/s |   1991 MB/s | 33.09 | silesia.tar |
| lzav 7.2 -1       |    331 MB/s |   1677 MB/s | 39.23 | silesia.tar |
| lzav 7.2 -2       |   64.0 MB/s |   1655 MB/s | 34.54 | silesia.tar |
| snappy 1.3.1      |    300 MB/s |    861 MB/s | 47.89 | silesia.tar |
| zstd 1.5.7 -1     |    297 MB/s |    902 MB/s | 34.54 | silesia.tar |
| zstd 1.5.7 -11    |   25.1 MB/s |    929 MB/s | 27.35 | silesia.tar |
| zstd 1.5.7 -18    |   2.67 MB/s |    838 MB/s | 25.14 | silesia.tar |
| zstd 1.5.7 -22    |   1.39 MB/s |    786 MB/s | 24.67 | silesia.tar |
| brotli 1.2.0 -5   |   38.4 MB/s |    282 MB/s | 28.11 | silesia.tar |
| brotli 1.2.0 -8   |   9.67 MB/s |    307 MB/s | 26.96 | silesia.tar |
| brotli 1.2.0 -11  |   0.41 MB/s |    254 MB/s | 23.79 | silesia.tar |
| xz 5.8.4 -6       |   2.15 MB/s |   77.7 MB/s | 23.35 | silesia.tar |
| xz 5.8.4 -9       |   1.85 MB/s |   78.2 MB/s | 23.01 | silesia.tar |
| bzip3 1.5.4 -5    |   9.19 MB/s |   9.30 MB/s | 22.29 | silesia.tar |
| bzip2 1.0.8 -9    |   8.97 MB/s |   24.4 MB/s | 25.75 | silesia.tar |

---

Results on ARM and AMD x86-64 will be added soon.

---

### Throughput vs ratio

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="misc/lzbench-results-archive/0.8.0/decode_vs_ratio_dark.svg">
  <img alt="Decode throughput vs compression ratio on silesia.tar (Intel), same data as the Intel table" src="misc/lzbench-results-archive/0.8.0/decode_vs_ratio_light.svg">
</picture>

## Levels

TLDR: I recommend using one of levels 1, 4, or 5 depending on your requirements.

Levels are ordered by compression effort. For most shapes of data, encode time increases and ratio decreases with increasing level, but decompression throughput is *not* similarly monotonic.

| Level | Description                                             | Ratio | Encode vs L1 | Decode vs L1 | Encoder memory | Format |
| ----- | ------------------------------------------------------- | ----- | ------------ | ------------ | -------------- | ------ |
| -1    | fastest compression (not recommended as slow to decode) | 46.74 | 4.2x         | 79%          | 1 MB           | light  |
| 0     | fast compression, lz4-class ratio                       | 44.60 | 3.2x         | 87%          | 1 MB           | light  |
| **1**     | **default, fastest decode**                                 | 42.64 | 1x           | 100%         | 3 MB           | light  |
| 2     | better ratio, still cheap to encode                     | 40.33 | 0.76x        | 91%          | 3 MB           | light  |
| 3     | best ratio in the light format                          | 37.76 | 0.20x        | 90%          | 29 MB          | light  |
| **4**     | **best ratio without entropy coding**                       | 32.34 | 0.03x        | 92%          | 71 MB          | heavy  |
| **5**     | **best ratio overall (entropy-coded)**                      | 24.79 | 0.016x       | 44%          | 164 MB         | entropy |

- `Ratio` is as measured on `silesia.tar` in the [Intel run](#intel-x86-64).
- The `Encode`/`Decode vs L1` columns are throughput relative to level 1, from the same run.
- Levels -1 to 3 emit the light format, level 4 emits the heavy format and level 5 emits the entropy format (see [`docs/format.md`](docs/format.md)).
- `Encoder memory` is the compressor's own working state (disjoint from the caller's input and output buffers).  Levels 3 to 5 need less for inputs smaller than their internal block or window, as described in [`docs/format.md`](docs/format.md) (level 5 uses about 11 MB for a 64 KB input and 50 MB for a 1 MB input).
- Decompression needs no working memory at levels -1 to 4, and level 5's decoder allocates 1.4 MB of scratch space per call.

## Requirements

For the library:

- A C++20 compiler (both GCC and Clang are fine).
- CMake >= 3.20.
- A little-endian 64-bit system.

For the CLI:

- The `misa` CLI needs POSIX (Linux, macOS).

Note: On x86-64, AVX2/SSE2 are selected at runtime. ARM has a NEON path. On x86-64 and ARM64 with GCC or Clang, the decoders use hand-written assembly kernels (you can pass `-DMISA77_ASM=OFF` to use the C++ decoders instead, but I *strongly* recommend not doing this).

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces the `misa` CLI at `build/misa`. For a binary tuned to the machine you'll run it on, add `-DMISA77_MARCH=native` (I recommend this). To run the round-trip test:

```sh
ctest --test-dir build
```

## Library Usage

The build produces a static library (CMake target `misa77::misa77`) with a small C++ API in `misa77/misa77.h`. The easiest integration is a git submodule (or CMake `FetchContent`) plus:

```cmake
add_subdirectory(misa77)
target_link_libraries(your_app PRIVATE misa77::misa77)
```

Alternatively, install it (`cmake --install build --prefix <dir>`) and use:

```cmake
find_package(misa77 REQUIRED)
target_link_libraries(your_app PRIVATE misa77::misa77)
```

Pass `-DMISA77_BUILD_CLI=OFF` if you only want the library.

Sample usage:

```cpp
#include <misa77/misa77.h>
#include <vector>

// compress (pick a level with misa77::config(N) with N in [-1, 5], default is 1), returns 0 on failure
misa77::config cfg;
std::vector<uint8_t> compressed(misa77::compress_bound(input_size, cfg));
uint64_t csize = misa77::compress(input, input_size, compressed.data(), compressed.size(), cfg);
compressed.resize(csize);

// decompress, returns original_size on success
uint64_t original_size = misa77::decompressed_size(compressed.data());
std::vector<uint8_t> output(misa77::decompressed_buffer_bound(original_size));
// returns 0 on invalid input
uint64_t written = misa77::decompress(compressed.data(), csize, output.data(), output.size());
```

Two things to keep in mind:

- You must size the destination buffers with `compress_bound` / `decompressed_buffer_bound`, passing `compress_bound` the same `config` you compress with (the bound depends on the level's format).
- The size you pass to `misa77::decompress` must be the exact size of the compressed stream (what `compress` returned), and not the size of the buffer holding it.

The experimental modes are declared in `misa77/experimental.h`, with usage documented in comments (these keep changing frequently so I don't wanna "formally" document them here just yet). You probably shouldn't be using them anyway.

## CLI Usage

`misa` is a single, dependency-free binary with two file-based subcommands. It operates on single files only (there's no directory or pipe support, so `tar` first if you need those).

```sh
misa compress   FILE          # -> FILE.misa77
misa decompress FILE.misa77   # -> FILE
```

`misa compress` takes `-l N` / `--level N` to pick the compression level (default is 1). Everywhere, `-o PATH` sets the output path and `-f` overwrites without asking.

```sh
misa compress -l 0 enwik8                # enwik8 -> enwik8.misa77
misa decompress enwik8.misa77 -f         # back to enwik8
```

## Documentation

The underlying stream formats (used by the library functions) and the container format for `.misa77` files (produced by the CLI) can be found in [`docs/`](docs/).

## Status

1. misa77's format may change unexpectedly as it's still v0.x.y.
2. It's been through local fuzzing with ASan/UBSan, broader hardening is ongoing.

Note: misa77 has evolved out of a less polished endeavour to learn performance engineering, and its history can be found [in this archived repository](https://github.com/welcome-to-the-sunny-side/misa77-archive).

## Acknowledgement

Inspiration has been taken from:

- [LZ4](http://github.com/lz4/lz4/)
- [lizard](https://github.com/inikep/lizard)
- [m77rip](https://github.com/paddor/m77rip)
- Various people on encode.su

Lastly, the following models helped a *lot* with every step of the process, including independently suggesting ideas/techniques that I wouldn't have thought of. Without their assistance, development would have been far slower, and I would likely not have explored all the paths that I did end up exploring (some of which were productive and some of which weren't).

- Claude Opus 4.8
- Claude Fable 5
- Claude Opus 5
- Claude Opus 5.5
- Claude Fable 5.2

## License

MIT (see [LICENSE](LICENSE)).