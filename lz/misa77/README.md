# misa77 (0.3.0)

misa77 is an LZ-based codec that targets the write-once, read-many niche. In particular, it aims to satisfy the following criteria:

- Extremely high decompression throughput (single-threaded).
- Modest compression ratios (LZ4 at high effort levels is a good reference point).
- Constant memory use, regardless of input size (<= 5 MB across all compression modes, and 0 MB for decompression).

Slow compression is the obvious tradeoff that one makes to achieve the above.

In addition, misa77 has a somewhat synergizing tendency to decompress highly compressed files faster. This makes high-effort compression particularly attractive for misa77, and inspires some experimental compression modes (refer to [src/experimental/](src/experimental/)) that aim to spend more effort at compression time to produce a compressed stream that is friendlier to the microarchitectures of most CPUs when decompressing said streams.

misa77 has two compression effort levels as of v0.3.0:

- level 0: offers better decode throughput, slightly worse ratio, similar encode throughput
- level 1 (default): offers slightly worse decode throughput, better ratio, similar encode throughput

There are two decompressor modes:

- unsafe: passing invalid input to this mode is UB.
- safe: guaranteed to exit gracefully (ie. provably terminate, and not access out-of-bounds memory or engage in any other UB) in the case of corrupt/malicious input, is 2-4% slower than unsafe.

## Benchmarks

Detailed results are listed ahead, but here's a terse summary:

- misa77 lies on the pareto frontier for decompression throughput vs compression ratio on most shapes of data.
- It very frequently decompresses faster even when competitors have a significantly worse ratio.
- It is quite slow at compression.
- Performance is a bit sensitive to codegen, but even with the worst possible codegen I saw in my testing, misa77 was significantly faster than other codecs.

Let's first see some cross-platform results for the [Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip).

Note:

1. "Ratio" ahead is equal to `((compressed size)/(original)) * 100` (so lower is better).
2. The benchmarking harness is a public fork of lzbench, and can be accessed [here](https://github.com/welcome-to-the-sunny-side/lzbench/tree/add-misa77-0.3.0).
3. In the tables ahead, rows are sorted by decompression speed.

---

### Intel x86-64

Details:

- CPU: Intel(R) Core(TM) i7-14650HX (@2.2 GHz) (Intel Turbo disabled).
- Single threaded, pinned to a single performance core.
- CPU governor set to `performance`.

| Compressor name       | Compression | Decompress. |  Ratio | Filename    |
| --------------------- | ----------- | ----------- | ------ | ----------- |
| misa77 0.3.0 -0       |   54.1 MB/s |   5363 MB/s |  42.64 | silesia.tar |
| misa77 0.3.0 safe -0  |   54.1 MB/s |   5205 MB/s |  42.64 | silesia.tar |
| misa77 0.3.0 -1       |   51.2 MB/s |   4372 MB/s |  39.65 | silesia.tar |
| misa77 0.3.0 safe -1  |   51.2 MB/s |   4235 MB/s |  39.65 | silesia.tar |
| zxc 0.13.1 -3         |    116 MB/s |   2830 MB/s |  45.46 | silesia.tar |
| zxc 0.13.1 -4         |   81.0 MB/s |   2721 MB/s |  42.63 | silesia.tar |
| lzsse8fast 2019-04-18 |    183 MB/s |   2668 MB/s |  44.80 | silesia.tar |
| zxc 0.13.1 -5         |   48.4 MB/s |   2599 MB/s |  40.25 | silesia.tar |
| lzsse4fast 2019-04-18 |    187 MB/s |   2538 MB/s |  45.26 | silesia.tar |
| lz4hc 1.10.0 -12      |   7.31 MB/s |   2533 MB/s |  36.45 | silesia.tar |
| lz4 1.10.0            |    371 MB/s |   2510 MB/s |  47.59 | silesia.tar |
| lzav 5.11 -1          |    297 MB/s |   1725 MB/s |  39.94 | silesia.tar |
| zstd 1.5.7 -1         |    296 MB/s |    903 MB/s |  34.54 | silesia.tar |
| snappy 1.2.2          |    375 MB/s |    858 MB/s |  47.89 | silesia.tar |

---

### ARM64 (Apple Silicon)

Details: 

- CPU: Apple M3

| Compressor name       | Compression | Decompress. |  Ratio | Filename    |
| --------------------- | ----------- | ----------- | ------ | ----------- |
| misa77 0.3.0 safe -0  |    133 MB/s |  12561 MB/s |  42.64 | silesia.tar |
| misa77 0.3.0 -0       |    134 MB/s |  12537 MB/s |  42.64 | silesia.tar |
| misa77 0.3.0 -1       |    122 MB/s |  10225 MB/s |  39.65 | silesia.tar |
| misa77 0.3.0 safe -1  |    125 MB/s |  10057 MB/s |  39.65 | silesia.tar |
| zxc 0.13.1 -3         |    271 MB/s |   8011 MB/s |  45.77 | silesia.tar |
| zxc 0.13.1 -4         |    182 MB/s |   7640 MB/s |  43.20 | silesia.tar |
| zxc 0.13.1 -5         |    111 MB/s |   7137 MB/s |  40.30 | silesia.tar |
| lz4 1.10.0            |    919 MB/s |   5311 MB/s |  47.59 | silesia.tar |
| lz4hc 1.10.0 -12      |   16.7 MB/s |   4892 MB/s |  36.45 | silesia.tar |
| lzav 5.11 -1          |    740 MB/s |   4358 MB/s |  39.93 | silesia.tar |
| snappy 1.2.2          |    969 MB/s |   3445 MB/s |  47.91 | silesia.tar |
| zstd 1.5.7 -1         |    717 MB/s |   1619 MB/s |  34.54 | silesia.tar |

---

As misa77's performance is quite "spiky" (depending on the shape of the data being compressed), a file-level breakdown for the silesia corpus yields some interesting insights into its performance. 

Note: 

- The visuals that follow are derived from the benchmark results at [misc/lzbench-results-archive/0.3.0/intel.txt](misc/lzbench-results-archive/0.3.0/intel.txt)
- These results are with the same x86-64 (Intel) setup mentioned previously.

### Decode speed relative to lz4

At level 0, misa77 decodes faster than lz4 on all 12 files (some by huge margins). Level 1 decodes faster on 11/12 files. The exception is `x-ray`, which is highly incompressible (lz4 has a ratio of nearly 1.0 on this file and essentially devolves to a `memcpy`).

![misa77 per-file decode speed vs lz4, levels 0 and 1, Silesia (Intel)](misc/lzbench-results-archive/0.3.0/speedup_vs_lz4.png)

### Throughput vs ratio, against popular fast-decode codecs

On the compressible files, misa77 sits on the decode-throughput/ratio Pareto frontier: it decodes fastest while ~matching or beating the ratio of the other fast-LZ codecs. `x-ray` is an exception once again.

To spot misa77 in these graphs, just look for the circles near the top :)

![misa77 vs other codecs: per-file decode throughput vs ratio, Silesia (Intel)](misc/lzbench-results-archive/0.3.0/pareto_silesia.png)

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

// compress (pick a level with misa77::config(0/1); default is 1), returns 0 on failure
std::vector<uint8_t> compressed(misa77::compress_bound(input_size));
uint64_t csize = misa77::compress(input, input_size, compressed.data(), compressed.size());
compressed.resize(csize);

// decompress, returns original_size on success
uint64_t original_size = misa77::decompressed_size(compressed.data());
std::vector<uint8_t> output(misa77::decompressed_buffer_bound(original_size));
// decompress uses the unsafe decompressor by default, pass dconfig(true) to use the safe decompressor
uint64_t written = misa77::decompress(compressed.data(), csize, output.data(), output.size(), misa77::dconfig(true));
```

Two things to keep in mind:

- You must size the destination buffers with `compress_bound` / `decompressed_buffer_bound`.
- You must pass `misa77::dconfig(true)` to `misa77::decompress` if you want the decompressor to exit gracefully on invalid input.

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

Lastly, Claude Opus 4.8 and Fable 5 helped a lot with scripting, tooling, and building the CLI.

## License

MIT (see [LICENSE](LICENSE)).