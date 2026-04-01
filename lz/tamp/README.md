<div align="center">
  <img src="https://raw.githubusercontent.com/BrianPugh/tamp/main/assets/logo_300w.png">
</div>

<div align="center">

![Python compat](https://img.shields.io/badge/%3E=python-3.9-blue.svg)
[![PyPi](https://img.shields.io/pypi/v/tamp.svg)](https://pypi.python.org/pypi/tamp)
[![GHA Status](https://github.com/BrianPugh/tamp/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/BrianPugh/tamp/actions?query=workflow%3Atests)
[![Coverage](https://codecov.io/github/BrianPugh/tamp/coverage.svg?branch=main)](https://codecov.io/github/BrianPugh/tamp?branch=main)
[![Documentation Status](https://readthedocs.org/projects/tamp/badge/?version=latest)](https://tamp.readthedocs.io/en/latest/?badge=latest)

</div>

---

**Documentation:** <https://tamp.readthedocs.io/en/latest/>

**Source Code:** <https://github.com/BrianPugh/tamp>

**Online Demo:** <https://brianpugh.github.io/tamp>

Tamp is a low-memory, DEFLATE-inspired lossless compression library optimized
for embedded and resource-constrained environments.

Tamp delivers the highest data compression ratios, while using the least amount
of RAM and firmware storage.

# Features

- Various language implementations available:
  - Pure Python reference:
    - `tamp/__init__.py`, `tamp/compressor.py`, `tamp/decompressor.py`
    - `pip install tamp` will use a python-bound C implementation optimized for
      speed.
  - Micropython:
    - Native Module.
      - `mpy_bindings/`
  - C library:
    - `tamp/_c_src/`
  - Javascript/Typescript via Emscripten WASM.
    - `wasm/`
  - Unofficial [rust bindings](https://github.com/tmpfs/tamp-rs).
    - See documentation [here](https://docs.rs/tamp/latest/tamp/index.html).
- High compression ratios, low memory use, and fast.
- Compact compression and decompression implementations.
  - Compiled C library is <5KB (compressor + decompressor).
- Mid-stream flushing.
  - Allows for submission of messages while continuing to compress subsequent
    data.
- Customizable dictionary for greater compression of small messages.
- Fuzz tested with libFuzzer + AddressSanitizer/UBSan.
- Convenient CLI interface.

# Installation

Tamp contains several implementations:

1. A reference desktop CPython implementation that is optimized for readability
   (and **not** speed).
2. A Micropython Native Module implementation (fast).
3. A C implementation (with python bindings) for accelerated desktop use and to
   be used in C projects (very fast).
4. A JavaScript/TypeScript implementation via Emscripten WASM (see `wasm/`).

This section instructs how to install each implementation.

## Desktop Python

The Tamp library requires Python `>=3.9` and can be installed via:

```bash
pip install tamp
```

To also install the `tamp` command line tool:

```bash
pip install tamp[cli]
```

## MicroPython

### MicroPython Native Module

Tamp provides pre-compiled [native modules]{.title-ref} that are easy to
install, are small, and are incredibly fast.

Download the appropriate `.mpy` file from the
[release page](https://github.com/BrianPugh/tamp/releases).

- Match the micropython version.
- Match the architecture to the microcontroller (e.g. `armv6m` for a pi pico).

Rename the file to `tamp.mpy` and transfer it to your board. If using
[Belay](https://github.com/BrianPugh/belay), tamp can be installed by adding the
following to `pyproject.toml`.

```toml
[tool.belay.dependencies]
tamp = "https://github.com/BrianPugh/tamp/releases/download/v1.7.0/tamp-1.7.0-mpy1.23-armv6m.mpy"
```

## C

Copy the `tamp/_c_src/tamp` folder into your project. For more information, see
[the documentation](https://tamp.readthedocs.io/en/latest/c_library.html).

# Usage

Tamp works on desktop python and micropython. On desktop, Tamp can be bundled
with the `tamp` command line tool for compressing and decompressing tamp files.
Install with `pip install tamp[cli]`.

## CLI

### Compression

Use `tamp compress` to compress a file or stream. If no input file is specified,
data from stdin will be read. If no output is specified, the compressed output
stream will be written to stdout.

```bash
$ tamp compress --help
Usage: tamp compress [ARGS] [OPTIONS]

Compress an input file or stream.

╭─ Parameters ───────────────────────────────────────────────────────────────────────────────╮
│ INPUT,--input    -i  Input file to compress. Defaults to stdin.                            │
│ OUTPUT,--output  -o  Output compressed file. Defaults to stdout.                           │
│ --window         -w  Number of bits used to represent the dictionary window. [default: 10] │
│ --literal        -l  Number of bits used to represent a literal. [default: 8]              │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

```bash
tamp compress enwik8 -o enwik8.tamp  # Compress a file
echo "hello world" | tamp compress | wc -c  # Compress a stream and print the compressed size.
```

The following options can impact compression ratios and memory usage:

- `window` - `2^window` plaintext bytes to look back to try and find a pattern.
  A larger window size will increase the chance of finding a longer pattern
  match, but will use more memory, increase compression time, and cause each
  pattern-token to take up more space. Try smaller window values if compressing
  highly repetitive data, or short messages.
- `literal` - Number of bits used in each plaintext byte. For example, if all
  input data is 7-bit ASCII, then setting this to 7 will improve literal
  compression ratios by 11.1%. The default, 8-bits, can encode any binary data.

### Decompression

Use `tamp decompress` to decompress a file or stream. If no input file is
specified, data from stdin will be read. If no output is specified, the
compressed output stream will be written to stdout.

```bash
$ tamp decompress --help
Usage: tamp decompress [ARGS] [OPTIONS]

Decompress an input file or stream.

╭─ Parameters ───────────────────────────────────────────────────────────────────────────────╮
│ INPUT,--input    -i  Input file to decompress. Defaults to stdin.                          │
│ OUTPUT,--output  -o  Output decompressed file. Defaults to stdout.                         │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

```bash
tamp decompress enwik8.tamp -o enwik8
echo "hello world" | tamp compress | tamp decompress
```

## Python

The python library can perform one-shot compression, as well as operate on
files/streams.

```python
import tamp

# One-shot compression
string = b"I scream, you scream, we all scream for ice cream."
compressed_data = tamp.compress(string)
reconstructed = tamp.decompress(compressed_data)
assert reconstructed == string

# Streaming compression
with tamp.open("output.tamp", "wb") as f:
    for _ in range(10):
        f.write(string)

# Streaming decompression
with tamp.open("output.tamp", "rb") as f:
    reconstructed = f.read()
```

# Benchmark

In the following section, we compare Tamp against:

- [zlib](https://docs.python.org/3/library/zlib.html), a python builtin
  gzip-compatible DEFLATE compression library.
- [heatshrink](https://github.com/atomicobject/heatshrink), a data compression
  library for embedded/real-time systems. Heatshrink has similar goals as Tamp.

All of these are LZ-based compression algorithms, and tests were performed using
a 1KB (10 bit) window. Since zlib already uses significantly more memory by
default, the lowest memory level (`memLevel=1`) was used in these benchmarks. It
should be noted that higher zlib memory levels will having greater compression
ratios than Tamp. Currently, there is no micropython-compatible zlib or
heatshrink compression implementation, so these numbers are provided simply as a
reference.

## Compression Ratio

The following table shows compression algorithm performance over a variety of
input data sourced from the
[Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia) and
[Enwik8](https://mattmahoney.net/dc/textdata.html). This should give a general
idea of how these algorithms perform over a variety of input data types.

| dataset         | raw         | tamp        | tamp (LazyMatching) | zlib          | heatshrink |
| --------------- | ----------- | ----------- | ------------------- | ------------- | ---------- |
| enwik8          | 100,000,000 | 51,016,917  | **50,625,930**      | 56,205,166    | 56,110,394 |
| RPI_PICO (.uf2) | 667,648     | **289,454** | 290,577             | 303,763       | -          |
| silesia/dickens | 10,192,446  | 5,538,353   | **5,502,834**       | 6,049,169     | 6,155,768  |
| silesia/mozilla | 51,220,480  | 24,413,362  | **24,229,925**      | 25,104,966    | 25,435,908 |
| silesia/mr      | 9,970,564   | 4,520,091   | **4,391,864**       | 4,864,734     | 5,442,180  |
| silesia/nci     | 33,553,445  | 6,824,403   | 6,772,307           | **5,765,521** | 8,247,487  |
| silesia/ooffice | 6,152,192   | 3,773,003   | **3,755,046**       | 4,077,277     | 3,994,589  |
| silesia/osdb    | 10,085,684  | 8,466,875   | **8,464,328**       | 8,625,159     | 8,747,527  |
| silesia/reymont | 6,627,202   | 2,818,554   | **2,788,774**       | 2,897,661     | 2,910,251  |
| silesia/samba   | 21,606,400  | 8,383,534   | **8,346,076**       | 8,862,423     | 9,223,827  |
| silesia/sao     | 7,251,944   | 6,136,077   | **6,100,061**       | 6,506,417     | 6,400,926  |
| silesia/webster | 41,458,703  | 18,146,641  | **18,010,981**      | 20,212,235    | 19,942,817 |
| silesia/x-ray   | 8,474,240   | 7,509,449   | 7,404,794           | **7,351,750** | 8,059,723  |
| silesia/xml     | 5,345,280   | 1,472,562   | **1,455,641**       | 1,586,985     | 1,665,179  |

Tamp outperforms both heatshrink and zlib on most datasets, winning 12 out of 14
benchmarks. This is while using around 10x less memory than zlib during both
compression and decompression (see next section).

Lazy Matching is a simple technique to improve compression ratios at the expense
of CPU while requiring very little code. One can expect **50-75%** more CPU
usage for modest compression gains (around 0.5 - 2.0%). Because of this
trade-off, it is disabled by default; however, in applications where we want to
compress once on a powerful machine (like a desktop/server) and decompress on an
embedded device, it may be worth it to spend a bit more compute. Lazy matched
compressed data is the exact same format; it appears no different to the tamp
decoder.

### Ablation Study

The following table shows the effect of the `extended` and `lazy_matching`
compression parameters across all benchmark datasets (`window=10`, `literal=8`).

| dataset         | raw         | Baseline   | +lazy              | +extended          | +lazy +extended    |
| --------------- | ----------- | ---------- | ------------------ | ------------------ | ------------------ |
| enwik8          | 100,000,000 | 51,635,633 | 51,252,694 (−0.7%) | 51,016,917 (−1.2%) | 50,625,930 (−2.0%) |
| RPI_PICO (.uf2) | 667,648     | 331,310    | 329,893 (−0.4%)    | 289,454 (−12.6%)   | 290,577 (−12.3%)   |
| silesia/dickens | 10,192,446  | 5,546,761  | 5,511,681 (−0.6%)  | 5,538,353 (−0.2%)  | 5,502,834 (−0.8%)  |
| silesia/mozilla | 51,220,480  | 25,121,385 | 24,937,036 (−0.7%) | 24,413,362 (−2.8%) | 24,229,925 (−3.5%) |
| silesia/mr      | 9,970,564   | 5,027,032  | 4,888,930 (−2.7%)  | 4,520,091 (−10.1%) | 4,391,864 (−12.6%) |
| silesia/nci     | 33,553,445  | 8,643,610  | 8,645,399 (+0.0%)  | 6,824,403 (−21.0%) | 6,772,307 (−21.6%) |
| silesia/ooffice | 6,152,192   | 3,814,938  | 3,798,393 (−0.4%)  | 3,773,003 (−1.1%)  | 3,755,046 (−1.6%)  |
| silesia/osdb    | 10,085,684  | 8,520,835  | 8,518,502 (−0.0%)  | 8,466,875 (−0.6%)  | 8,464,328 (−0.7%)  |
| silesia/reymont | 6,627,202   | 2,847,981  | 2,820,948 (−0.9%)  | 2,818,554 (−1.0%)  | 2,788,774 (−2.1%)  |
| silesia/samba   | 21,606,400  | 9,102,594  | 9,061,143 (−0.5%)  | 8,383,534 (−7.9%)  | 8,346,076 (−8.3%)  |
| silesia/sao     | 7,251,944   | 6,137,755  | 6,101,747 (−0.6%)  | 6,136,077 (−0.0%)  | 6,100,061 (−0.6%)  |
| silesia/webster | 41,458,703  | 18,694,172 | 18,567,618 (−0.7%) | 18,146,641 (−2.9%) | 18,010,981 (−3.7%) |
| silesia/x-ray   | 8,474,240   | 7,510,606  | 7,406,001 (−1.4%)  | 7,509,449 (−0.0%)  | 7,404,794 (−1.4%)  |
| silesia/xml     | 5,345,280   | 1,681,687  | 1,672,827 (−0.5%)  | 1,472,562 (−12.4%) | 1,455,641 (−13.4%) |

The `extended` parameter enables additional Huffman codes for longer pattern
matches, which significantly improves compression on datasets with many long
repeating patterns (e.g., nci, samba, xml). Extended support was added in
v2.0.0.

## Memory Usage

The following table shows approximately how much memory each algorithm uses
during compression and decompression.

|                       | Compression                   | Decompression           |
| --------------------- | ----------------------------- | ----------------------- |
| Tamp                  | (1 << windowBits)             | (1 << windowBits)       |
| ZLib                  | (1 << (windowBits + 2)) + 7KB | (1 << windowBits) + 7KB |
| Heatshrink            | (1 << (windowBits + 1))       | (1 << (windowBits + 1)) |
| Deflate (micropython) | (1 << windowBits)             | (1 << windowBits)       |

All libraries have a few dozen bytes of overhead in addition to the primary
window buffer, but are implementation-specific and ignored for clarity here.
Tamp uses significantly less memory than ZLib, and half the memory of
Heatshrink.

## Runtime

As a rough benchmark, here is the performance (in seconds) of these different
compression algorithms on the 100MB enwik8 dataset. These tests were performed
on an M3 Macbook Air.

|                              | Compression (s) | Decompression (s) |
| ---------------------------- | --------------- | ----------------- |
| Tamp (Pure Python Reference) | 136.2           | 105.0             |
| Tamp (C bindings)            | 5.45            | 0.544             |
| ZLib                         | 3.65            | 0.578             |
| Heatshrink (with index)      | 4.42            | 0.67              |
| Heatshrink (without index)   | 27.40           | 0.67              |

Heatshrink v0.4.1 was used in these benchmarks. When heathshrink uses an index,
an additional `(1 << (windowBits + 1))` bytes of memory are used, resulting in
4x more memory-usage than Tamp. Tamp could use a similar indexing to increase
compression speed, but has chosen not to to focus on the primary goal of a
low-memory compressor.

To give an idea of Tamp's speed on an embedded device, the following table shows
compression/decompression in **bytes/second of the first 100KB of enwik8 on a pi
pico (rp2040)** at the default 125MHz clock rate. The C benchmark **does not**
use a filesystem nor dynamic memory allocation, so it represents the maximum
speed Tamp can achieve. In all tests, a 1KB window (10 bit) was used.

|                                  | Compression (bytes/s) | Decompression (bytes/s) |
| -------------------------------- | --------------------- | ----------------------- |
| Tamp (Micropython Native Module) | 31,328                | 990,099                 |
| Tamp (C)                         | 36,127                | 1,400,600               |
| Deflate (micropython builtin)    | 6,885                 | 294,985                 |

Tamp resulted in a **50841** byte archive, while Micropython's builtin `deflate`
resulted in a larger, **59442** byte archive.

## Binary Size

To give an idea on the resulting binary sizes, Tamp and other libraries were
compiled for the Pi Pico (`armv6m`). All libraries were compiled with `-O3`.
Numbers reported in bytes. Tamp sizes were measured using `arm-none-eabi-gcc`
15.2.1 and MicroPython v1.27, and can be regenerated with `make binary-size`.

|                                  | Compressor | Decompressor | Compressor + Decompressor |
| -------------------------------- | ---------- | ------------ | ------------------------- |
| Tamp (MicroPython Native)        | 4700       | 4347         | 8024                      |
| Tamp (C, no extended, no stream) | 1754       | 1656         | 3172                      |
| Tamp (C, no extended)            | 2036       | 1894         | 3692                      |
| Tamp (C, extended, no stream)    | 2838       | 2452         | 5052                      |
| Tamp (C, extended)               | 3120       | 2690         | 5572                      |
| Heatshrink (C)                   | 2956       | 3876         | 6832                      |
| uzlib (C)                        | 2355       | 3963         | 6318                      |

Tamp C "extended" includes `tamp_compressor_compress_and_flush`. Tamp C includes
a high-level stream API by default. Even with `no stream`, Tamp includes
buffer-looping functions (like `tamp_compressor_compress`) that Heatshrink lacks
(Heatshrink only provides poll/sink primitives).

## Acknowledgement

- Thanks @BitsForPeople for the esp32-optimized compressor implementation.
