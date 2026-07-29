# misa77 (0.6.0)

misa77 is an LZ-based codec that targets the write-once, read-many niche. In particular, it aims to satisfy the following criteria:

- Extremely high decompression throughput (single-threaded).
- Modest compression ratios (LZ4 at high effort levels is a good reference point).
- Constant memory use during compression, regardless of input size (under 3 MB for levels -1 to 2, 29 MB for level 3, 176 MB for level 4). Decompression uses no extra memory.

Slow compression is the obvious tradeoff that one makes to achieve the above.

misa77 has six compression effort levels as of v0.6.0. For most shapes of data, encode time increases and ratio decreases with increasing level, but decompression throughput is *not* similarly monotonic. In general, level 1 offers the fastest decode throughput. More information about effort levels can be found [here](#levels).

There are two decompressor modes:

- unsafe: passing invalid input to this mode is UB.
- safe: guaranteed to exit gracefully (ie. provably terminate, and not access out-of-bounds memory or engage in any other UB) in the case of corrupt/malicious input, is typically ~5% slower than unsafe.

Note: As of now, level 4 doesn't have a safe decompressor. It will be added soon.

## Benchmarks

Detailed results are listed ahead, but here's a terse summary:

- misa77 lies on the pareto frontier for decompression throughput vs compression ratio on most shapes of data.
- It very frequently decompresses faster even when competitors have a significantly worse ratio.
- It is somewhat slow at compression.
- Performance is a bit sensitive to codegen, but even with the worst possible codegen I saw in my testing, misa77 was significantly faster than other codecs.

Let's first see some cross-platform results for the [Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip).

Note:

1. "Ratio" ahead is equal to `((compressed size)/(original)) * 100` (so lower is better).
2. misa77 has been merged into [lzbench](https://github.com/inikep/lzbench/) and [TurboBench](https://github.com/powturbo/TurboBench). The results below have been produced using lzbench, and can be reproduced through both benchmarking harnesses.
3. In the tables ahead, rows are grouped by codec family, and ordered from lowest to highest effort level within each family. The misa77 default level (1) is in bold.
4. The tables show unsafe (trusted-input) decompression. The safe decoder costs about 3% of decode throughput on the x86-64 setup and 1-2% on the M3 for these files.

---

### Intel x86-64

Details:

- CPU: Intel(R) Core(TM) i7-14650HX (@2.2 GHz) (Intel Turbo disabled).
- Single threaded, pinned to a single performance core.
- CPU governor set to `performance`.

| Compressor name       | Compression | Decompress. | Ratio | Filename    |
| --------------------- | ----------- | ----------- | ----- | ----------- |
| misa77 0.6.0 --1      |    232 MB/s |   3846 MB/s | 46.74 | silesia.tar |
| misa77 0.6.0 -0       |    170 MB/s |   4593 MB/s | 44.60 | silesia.tar |
| **misa77 0.6.0 -1**       |   **54.2 MB/s** |   **5376 MB/s** | **42.64** | silesia.tar |
| misa77 0.6.0 -2       |   41.5 MB/s |   4689 MB/s | 40.33 | silesia.tar |
| misa77 0.6.0 -3       |   10.7 MB/s |   4635 MB/s | 37.76 | silesia.tar |
| misa77 0.6.0 -4       |   6.97 MB/s |   4366 MB/s | 35.51 | silesia.tar |
| lz4 1.10.0            |    371 MB/s |   2514 MB/s | 47.59 | silesia.tar |
| lz4hc 1.10.0 -9       |   22.0 MB/s |   2457 MB/s | 36.75 | silesia.tar |
| lz4hc 1.10.0 -12      |   7.31 MB/s |   2536 MB/s | 36.45 | silesia.tar |
| lzsse4fast 2019-04-18 |    187 MB/s |   2540 MB/s | 45.26 | silesia.tar |
| lzsse8fast 2019-04-18 |    183 MB/s |   2671 MB/s | 44.80 | silesia.tar |
| zxc 0.13.1 -3         |    114 MB/s |   2840 MB/s | 45.46 | silesia.tar |
| zxc 0.13.1 -4         |   80.6 MB/s |   2731 MB/s | 42.63 | silesia.tar |
| zxc 0.13.1 -5         |   48.2 MB/s |   2597 MB/s | 40.25 | silesia.tar |
| zxc 0.13.1 -7         |   4.26 MB/s |   1642 MB/s | 33.00 | silesia.tar |
| zstd 1.5.7 -1         |    297 MB/s |    903 MB/s | 34.54 | silesia.tar |
| lzav 5.16 -2          |   62.6 MB/s |   1675 MB/s | 34.85 | silesia.tar |
| snappy 1.2.2          |    375 MB/s |    858 MB/s | 47.89 | silesia.tar |

---

### ARM64 (Apple Silicon)

Details: 

- CPU: Apple M3

| Compressor name  | Compression | Decompress. | Ratio | Filename    |
| ---------------- | ----------- | ----------- | ----- | ----------- |
| misa77 0.6.0 --1 |    541 MB/s |   9739 MB/s | 46.74 | silesia.tar |
| misa77 0.6.0 -0  |    385 MB/s |  11777 MB/s | 44.60 | silesia.tar |
| **misa77 0.6.0 -1**  |    **135 MB/s** |  **12638 MB/s** | **42.64** | silesia.tar |
| misa77 0.6.0 -2  |    103 MB/s |  10868 MB/s | 40.33 | silesia.tar |
| misa77 0.6.0 -3  |   19.5 MB/s |  10280 MB/s | 37.76 | silesia.tar |
| misa77 0.6.0 -4  |   13.5 MB/s |   9925 MB/s | 35.51 | silesia.tar |
| lz4 1.10.0       |    882 MB/s |   5195 MB/s | 47.59 | silesia.tar |
| lz4hc 1.10.0 -9  |   53.0 MB/s |   4899 MB/s | 36.74 | silesia.tar |
| lz4hc 1.10.0 -12 |   17.0 MB/s |   4887 MB/s | 36.45 | silesia.tar |
| zxc 0.13.1 -3    |    278 MB/s |   8022 MB/s | 45.77 | silesia.tar |
| zxc 0.13.1 -4    |    193 MB/s |   7642 MB/s | 43.20 | silesia.tar |
| zxc 0.13.1 -5    |    114 MB/s |   7162 MB/s | 40.30 | silesia.tar |
| zstd 1.5.7 -1    |    722 MB/s |   1615 MB/s | 34.54 | silesia.tar |
| lzav 5.16 -2     |    185 MB/s |   4243 MB/s | 34.85 | silesia.tar |
| snappy 1.2.2     |    967 MB/s |   3439 MB/s | 47.91 | silesia.tar |

---

As misa77's performance is quite "spiky" (depending on the shape of the data being compressed), a file-level breakdown for the silesia corpus yields some interesting insights into its performance. 

Note: 

- The visuals that follow are derived from the benchmark results at [misc/lzbench-results-archive/0.6.0/intel.txt](misc/lzbench-results-archive/0.6.0/intel.txt)
- These results are with the same x86-64 (Intel) setup mentioned previously.

### Decode speed relative to lz4

At level 1, misa77 decodes faster than lz4 on all 12 files (some by huge margins). 

![misa77 per-file decode speed vs lz4, levels -1 to 4, Silesia (Intel)](misc/lzbench-results-archive/0.6.0/speedup_vs_lz4.png)

### Throughput vs ratio, against popular fast-decode codecs

On the compressible files, misa77 sits on the decode-throughput/ratio Pareto frontier: it decodes fastest while ~matching or beating the ratio of the other fast-LZ codecs. (`x-ray` and `sao` are exceptions as most compressors have a ratio near 1, and their decode essentially devolves to a `memcpy`).

To spot misa77 in these graphs, just look for the circles near the top :)

![misa77 vs other codecs: per-file decode throughput vs ratio, Silesia (Intel), levels -1 to 4](misc/lzbench-results-archive/0.6.0/pareto_silesia.png)

## Levels

Levels are ordered by compression effort. For most shapes of data, encode time increases and ratio decreases with increasing level, but decompression throughput is *not* similarly monotonic.

| Level | Description                                             | Ratio | Encode vs L1 | Decode vs L1 | Encoder memory | Format |
| ----- | ------------------------------------------------------- | ----- | ------------ | ------------ | -------------- | ------ |
| -1    | fastest compression (not recommended as slow to decode) | 46.74 | 4.0-4.3x     | 72-77%       | 1 MB           | light  |
| 0     | fast compression, lz4-class ratio                       | 44.60 | 2.9-3.1x     | 85-93%       | 1 MB           | light  |
| **1**     | **default, fastest decode**                                 | 42.64 | 1x           | 100%         | 3 MB           | light  |
| 2     | better ratio, still cheap to encode                     | 40.33 | 0.76-0.77x   | 86-87%       | 3 MB           | light  |
| 3     | best ratio in the light format                          | 37.76 | 0.14-0.20x   | 81-86%       | 29 MB          | light  |
| 4     | best ratio overall, prefer on inputs above ~1 MB        | 35.51 | 0.10-0.13x   | 79-81%       | 176 MB         | heavy  |

- `Ratio` is measured on `silesia.tar` in the [Intel run](#intel-x86-64).
- The `Encode`/`Decode vs L1` columns are relative to level 1, given as the range spanned by the Intel and M3 runs.
- `Encoder memory` is the compressor's own working state (disjoint from the caller's input and output buffers). Decompression needs no working memory at all.
- Levels -1 to 3 emit the light format (see [`docs/format.md`](docs/format.md)) and have a safe decoder. Level 4 emits the heavy format, which does not have one yet.

## Requirements

For the library:

- A C++20 compiler (both GCC and Clang are fine).
- CMake >= 3.20.
- A little-endian 64-bit system.

For the CLI:

- The `misa` CLI needs POSIX (Linux, macOS).

Note: On x86-64, AVX2/SSE2 are selected at runtime. ARM has a NEON path.

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces the `misa` CLI at `build/misa`. For a binary tuned to the exact machine you'll run it on, add `-DMISA77_MARCH=native` (I recommend this). To run the round-trip test:

```sh
ctest --test-dir build
```

## Library Usage

The build produces a static library (CMake target `misa77`) with a small C++ API in `misa77/misa77.h`. The easiest integration is a git submodule (or CMake `FetchContent`) plus:

```cmake
add_subdirectory(misa77)
target_link_libraries(your_app PRIVATE misa77)
```

Sample usage:

```cpp
#include <misa77/misa77.h>
#include <vector>

// compress (pick a level with misa77::config(N) with N in [-1, 4], default is 1), returns 0 on failure
misa77::config cfg;
std::vector<uint8_t> compressed(misa77::compress_bound(input_size, cfg));
uint64_t csize = misa77::compress(input, input_size, compressed.data(), compressed.size(), cfg);
compressed.resize(csize);

// decompress, returns original_size on success
uint64_t original_size = misa77::decompressed_size(compressed.data());
std::vector<uint8_t> output(misa77::decompressed_buffer_bound(original_size));
// decompress uses the unsafe decompressor by default, pass dconfig(true) to use the safe decompressor
uint64_t written = misa77::decompress(compressed.data(), csize, output.data(), output.size(), misa77::dconfig(true));
```

Three things to keep in mind:

- You must size the destination buffers with `compress_bound` / `decompressed_buffer_bound`, passing `compress_bound` the same `config` you compress with (the bound depends on the level's format).
- You must pass `misa77::dconfig(true)` to `misa77::decompress` if you want the decompressor to exit gracefully on invalid input.
- Safe decompression does not support level-4 streams yet: `decompress` with `dconfig(true)` returns 0 for them (the unsafe path decodes them fine).

The experimental modes are declared in `misa77/experimental.h`, with usage documented in comments (these keep changing frequently so I don't wanna "formally" document them here just yet).

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
- [zxc](https://github.com/hellobertrand/zxc)
- [lizard](https://github.com/inikep/lizard)

Lastly, the following models helped a *lot* with orchestrating experiments, scripting, tooling, and building the CLI. Without their assistance, development would have been far slower, and I would likely not have explored all the paths that I did end up exploring (some of which were productive and some of which weren't).

- Claude Fable 5
- Claude Opus 5
- Claude Opus 4.8

## License

MIT (see [LICENSE](LICENSE)).