# misa77 (0.5.0)

misa77 is an LZ-based codec that targets the write-once, read-many niche. In particular, it aims to satisfy the following criteria:

- Extremely high decompression throughput (single-threaded).
- Modest compression ratios (LZ4 at high effort levels is a good reference point).
- Constant memory use during compression, regardless of input size (5-16 MB for levels 0-1, ~160 MB for level 2). Decompression uses no extra memory.

Slow compression is the obvious tradeoff that one makes to achieve the above.

In addition, misa77 has a somewhat synergizing tendency to decompress highly compressed files faster. This makes high-effort compression particularly attractive for misa77, and inspires some experimental compression modes (refer to [src/experimental/](src/experimental/)) that aim to spend more effort at compression time to produce a compressed stream that is friendlier to the microarchitectures of most CPUs when decompressing said streams.

misa77 has three compression effort levels as of v0.5.0:

- level 0: offers better decode throughput, slightly worse ratio, similar encode throughput
- level 1 (default): offers slightly worse decode throughput, better ratio, similar encode throughput
- level 2: offers similar decode throughput to level 1, the best ratio (slightly better than `lz4hc -12`), very low encode throughput

There are two decompressor modes:

- unsafe: passing invalid input to this mode is UB.
- safe: guaranteed to exit gracefully (ie. provably terminate, and not access out-of-bounds memory or engage in any other UB) in the case of corrupt/malicious input, is 2-4% slower than unsafe.

Note: as of now, level 2 doesn't have a safe decompressor. It will be added soon.

## Benchmarks

Detailed results are listed ahead, but here's a terse summary:

- misa77 lies on the pareto frontier for decompression throughput vs compression ratio on most shapes of data.
- It very frequently decompresses faster even when competitors have a significantly worse ratio.
- It is quite slow at compression.
- Performance is a bit sensitive to codegen, but even with the worst possible codegen I saw in my testing, misa77 was significantly faster than other codecs.

Let's first see some cross-platform results for the [Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip).

Note:

1. "Ratio" ahead is equal to `((compressed size)/(original)) * 100` (so lower is better).
2. The benchmarking harness is a public fork of lzbench, and can be accessed [here](https://github.com/welcome-to-the-sunny-side/lzbench/tree/add-misa77-0.4.0).
3. In the tables ahead, rows are sorted by decompression speed.

---

### Intel x86-64

Details:

- CPU: Intel(R) Core(TM) i7-14650HX (@2.2 GHz) (Intel Turbo disabled).
- Single threaded, pinned to a single performance core.
- CPU governor set to `performance`.

| Compressor name       | Compression | Decompress. |  Ratio | Filename    |
| --------------------- | ----------- | ----------- | ------ | ----------- |
| misa77 0.5.0 -0       |   54.3 MB/s |   5359 MB/s |  42.64 | silesia.tar |
| misa77 0.5.0 safe -0  |   54.1 MB/s |   5216 MB/s |  42.64 | silesia.tar |
| misa77 0.5.0 -2       |   7.01 MB/s |   4470 MB/s |  35.51 | silesia.tar |
| misa77 0.5.0 -1       |   51.2 MB/s |   4378 MB/s |  39.65 | silesia.tar |
| misa77 0.5.0 safe -1  |   51.2 MB/s |   4252 MB/s |  39.65 | silesia.tar |
| zxc 0.13.1 -3         |    116 MB/s |   2838 MB/s |  45.46 | silesia.tar |
| zxc 0.13.1 -4         |   81.2 MB/s |   2726 MB/s |  42.63 | silesia.tar |
| lzsse8fast 2019-04-18 |    183 MB/s |   2663 MB/s |  44.80 | silesia.tar |
| zxc 0.13.1 -5         |   48.4 MB/s |   2602 MB/s |  40.25 | silesia.tar |
| lz4hc 1.10.0 -12      |   7.31 MB/s |   2531 MB/s |  36.45 | silesia.tar |
| lzsse4fast 2019-04-18 |    187 MB/s |   2525 MB/s |  45.26 | silesia.tar |
| lz4 1.10.0            |    371 MB/s |   2506 MB/s |  47.59 | silesia.tar |
| lz4hc 1.10.0 -9       |   22.0 MB/s |   2454 MB/s |  36.75 | silesia.tar |
| lzav 5.11 -2          |   58.4 MB/s |   1729 MB/s |  34.97 | silesia.tar |
| zxc 0.13.1 -7         |   4.27 MB/s |   1645 MB/s |  33.00 | silesia.tar |
| zstd 1.5.7 -1         |    297 MB/s |    903 MB/s |  34.54 | silesia.tar |
| snappy 1.2.2          |    376 MB/s |    858 MB/s |  47.89 | silesia.tar |

---

### ARM64 (Apple Silicon)

Details: 

- CPU: Apple M3

| Compressor name      | Compression | Decompress. |  Ratio | Filename    |
| -------------------- | ----------- | ----------- | ------ | ----------- |
| misa77 0.5.0 -0      |    134 MB/s |  12660 MB/s |  42.64 | silesia.tar |
| misa77 0.5.0 safe -0 |    134 MB/s |  12484 MB/s |  42.64 | silesia.tar |
| misa77 0.5.0 -1      |    127 MB/s |  10270 MB/s |  39.65 | silesia.tar |
| misa77 0.5.0 safe -1 |    127 MB/s |  10100 MB/s |  39.65 | silesia.tar |
| misa77 0.5.0 -2      |   13.6 MB/s |   9935 MB/s |  35.51 | silesia.tar |
| zxc 0.13.1 -3        |    279 MB/s |   8030 MB/s |  45.77 | silesia.tar |
| zxc 0.13.1 -4        |    193 MB/s |   7663 MB/s |  43.20 | silesia.tar |
| zxc 0.13.1 -5        |    115 MB/s |   7166 MB/s |  40.30 | silesia.tar |
| lz4 1.10.0           |    882 MB/s |   5166 MB/s |  47.59 | silesia.tar |
| lz4hc 1.10.0 -9      |   53.2 MB/s |   4885 MB/s |  36.74 | silesia.tar |
| lz4hc 1.10.0 -12     |   17.0 MB/s |   4883 MB/s |  36.45 | silesia.tar |
| zxc 0.13.1 -7        |   9.78 MB/s |   4310 MB/s |  33.01 | silesia.tar |
| lzav 5.11 -2         |    175 MB/s |   4261 MB/s |  34.97 | silesia.tar |
| snappy 1.2.2         |    967 MB/s |   3438 MB/s |  47.91 | silesia.tar |
| zstd 1.5.7 -1        |    722 MB/s |   1615 MB/s |  34.54 | silesia.tar |

---

As misa77's performance is quite "spiky" (depending on the shape of the data being compressed), a file-level breakdown for the silesia corpus yields some interesting insights into its performance. 

Note: 

- The visuals that follow are derived from the benchmark results at [misc/lzbench-results-archive/0.5.0/intel.txt](misc/lzbench-results-archive/0.5.0/intel.txt)
- These results are with the same x86-64 (Intel) setup mentioned previously.

### Decode speed relative to lz4

At level 0, misa77 decodes faster than lz4 on all 12 files (some by huge margins). Levels 1 and 2 decode faster on 11/12 files each, while compressing substantially better than lz4 everywhere. The exception is `x-ray`, which is highly incompressible (lz4 has a ratio of nearly 1.0 on this file and essentially devolves to a `memcpy`).

![misa77 per-file decode speed vs lz4, levels 0-2, Silesia (Intel)](misc/lzbench-results-archive/0.5.0/speedup_vs_lz4.png)

### Throughput vs ratio, against popular fast-decode codecs

On the compressible files, misa77 sits on the decode-throughput/ratio Pareto frontier: it decodes fastest while ~matching or beating the ratio of the other fast-LZ codecs. `x-ray` is an exception once again.

To spot misa77 in these graphs, just look for the circles near the top :)

![misa77 vs other codecs: per-file decode throughput vs ratio, Silesia (Intel), levels 0-2](misc/lzbench-results-archive/0.5.0/pareto_silesia.png)

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

// compress (pick a level with misa77::config(0/1/2); default is 1), returns 0 on failure
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
- Safe decompression does not support level-2 streams yet: `decompress` with `dconfig(true)` returns 0 for them (the unsafe path decodes them fine).

The experimental modes are declared in `misa77/experimental.h`, with usage documented in comments (these keep changing frequently so I don't wanna "formally" document them here just yet).

## CLI Usage

`misa` is a single, dependency-free binary with three file-based subcommands. It operates on single files only (there's no directory or pipe support, so `tar` first if you need those).

```sh
misa compress   FILE          # -> FILE.misa77
misa decompress FILE.misa77   # -> FILE
misa suggest    FILE          # -> FILE.misap  (tuned params)
```

`misa compress` takes `-l N` / `--level N` to pick the compression level (default is 1).

There are also some experimental compression modes (at most one at a time, not combinable with `--level`):

| Flag | Effect |
| ---- | ------ |
| `--adaptive` | autotune the compressor based on the input for decode speed (only use this with homogeneous data) |
| `--params F.misap` | compress with a vector from `misa suggest` |
| `--yolo` | high-effort, decode-optimized |

`--adaptive` and `suggest` also take `--tune loose` / `--tune tight` (similar tradeoffs as `level 0/1`, and the default is loose) and `--sample MB` (how much input to sample when picking params, default is 2 MB). Everywhere, `-o PATH` sets the output path and `-f` overwrites without asking.

```sh
misa compress -l 0 enwik8                # enwik8 -> enwik8.misa77, fastest-decode level
misa decompress enwik8.misa77            # back to enwik8

# tune on a sample, then reuse the params:
misa suggest --tune tight data.bin       # -> data.misap
misa compress --params data.misap data.bin
```

## Documentation

The underlying stream format (used by the library functions) and the container format for `.misa77` files (produced by the CLI) can be found in [`docs/`](docs/).

## Status

1. misa77's format may change unexpectedly as it's still v0.x.y. 
2. It's been through local fuzzing with ASan/UBSan, broader hardening is ongoing.

Note: misa77 has evolved out of a less polished endeavour to learn performance engineering, and its history can be found [in this archived repository](https://github.com/welcome-to-the-sunny-side/misa77-archive).

## Acknowledgement

Inspiration has been taken from:

- [LZ4](http://github.com/lz4/lz4/)
- [zxc](https://github.com/hellobertrand/zxc)
- [lizard](https://github.com/inikep/lizard)

Lastly, Claude Fable 5 and Opus 4.8 helped a *lot* with orchestrating experiments, scripting, tooling, and building the CLI. Without their assistance, development would have been far slower, and I would likely not have explored all the paths that I did end up exploring (some of which were productive and some of which weren't).

## License

MIT (see [LICENSE](LICENSE)).