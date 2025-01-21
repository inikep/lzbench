<div align="center">
  <img src="https://raw.githubusercontent.com/BrianPugh/tamp/main/assets/logo_300w.png">
</div>

<div align="center">

![Python compat](https://img.shields.io/badge/%3E=python-3.8-blue.svg)
[![PyPi](https://img.shields.io/pypi/v/tamp.svg)](https://pypi.python.org/pypi/tamp)
[![GHA Status](https://github.com/BrianPugh/tamp/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/BrianPugh/tamp/actions?query=workflow%3Atests)
[![Coverage](https://codecov.io/github/BrianPugh/tamp/coverage.svg?branch=main)](https://codecov.io/github/BrianPugh/tamp?branch=main)
[![Documentation Status](https://readthedocs.org/projects/tamp/badge/?version=latest)](https://tamp.readthedocs.io/en/latest/?badge=latest)

</div>

---

**Documentation:** https://tamp.readthedocs.io/en/latest/

**Source Code:** https://github.com/BrianPugh/tamp

---

Tamp is a low-memory, DEFLATE-inspired lossless compression library intended for embedded targets.

Tamp delivers the highest data compression ratios, while using the least amount of RAM and firmware storage.

# Features

* Various language implementations available:
    * Pure Python reference:
        * `tamp/__init__.py`, `tamp/compressor.py`, `tamp/decompressor.py`
        * `pip install tamp` will use a python-bound C implementation
            optimized for speed.
    * Micropython:
        * Native Module (suggested micropython implementation).
            * `mpy_bindings/`
        * Viper.
            * `tamp/__init__.py`, `tamp/compressor_viper.py`, `tamp/decompressor_viper.py`
    * C library:
        * `tamp/_c_src/`
* High compression ratios, low memory use, and fast.
* Compact compression and decompression implementations.
    * Compiled C library is <4KB (compressor + decompressor).
* Mid-stream flushing.
    * Allows for submission of messages while continuing to compress subsequent data.
* Customizable dictionary for greater compression of small messages.
* Convenient CLI interface.

# Installation

Tamp contains 4 implementations:

1.  A reference desktop CPython implementation that is optimized for readability (and **not** speed).
2.  A Micropython Native Module implementation (fast).
3.  A Micropython Viper implementation (not recommended, please use Native Module).
4.  A C implementation (with python bindings) for accelerated desktop
    use and to be used in C projects (very fast).

This section instructs how to install each implementation.

## Desktop Python

The Tamp library and CLI requires Python `>=3.8` and can be installed
via:

``` bash
pip install tamp
```

## MicroPython

### MicroPython Native Module

Tamp provides pre-compiled [native modules]{.title-ref} that are easy to install, are small, and are incredibly fast.

Download the appropriate `.mpy` file from the [release page](https://github.com/BrianPugh/tamp/releases).

* Match the micropython version.
* Match the architecture to the microcontroller (e.g. `armv6m` for a pi pico).

Rename the file to `tamp.mpy` and transfer it to your board. If using [Belay](https://github.com/BrianPugh/belay), tamp can be installed by adding the following to `pyproject.toml`.

``` toml
[tool.belay.dependencies]
tamp = "https://github.com/BrianPugh/tamp/releases/download/v1.6.0/tamp-1.6.0-mpy1.23-armv6m.mpy"
```

### MicroPython Viper

**NOT RECOMMENDED, PLEASE USE NATIVE MODULE**

For micropython use, there are 3 main files:

1.  `tamp/__init__.py` - Always required.
2.  `tamp/decompressor_viper.py` - Required for on-device decompression.
3.  `tamp/compressor_viper.py` - Required for on-device compression.

For example, if on-device decompression isn't used, then do not include `decompressor_viper.py`. If manually installing, just copy these files to your microcontroller's `/lib/tamp` folder.

If using [mip](https://docs.micropython.org/en/latest/reference/packages.html#installing-packages-with-mip), tamp can be installed by specifying the appropriate `package-*.json` file.

``` bash
mip install github:brianpugh/tamp  # Defaults to package.json: Compressor & Decompressor
mip install github:brianpugh/tamp/package-compressor.json  # Compressor only
mip install github:brianpugh/tamp/package-decompressor.json  # Decompressor only
```

If using [Belay](https://github.com/BrianPugh/belay), tamp can be installed by adding the following to `pyproject.toml`.

``` toml
[tool.belay.dependencies]
tamp = [
   "https://github.com/BrianPugh/tamp/blob/main/tamp/__init__.py",
   "https://github.com/BrianPugh/tamp/blob/main/tamp/compressor_viper.py",
   "https://github.com/BrianPugh/tamp/blob/main/tamp/decompressor_viper.py",
]
```

## C

Copy the `tamp/_c_src/tamp` folder into your project. For more information, see [the documentation](https://tamp.readthedocs.io/en/latest/c_library.html).

# Usage

Tamp works on desktop python and micropython. On desktop, Tamp is bundled with the `tamp` command line tool for compressing and decompressing tamp files.

## CLI

### Compression

Use `tamp compress` to compress a file or stream. If no input file is specified, data from stdin will be read. If no output is specified, the compressed output stream will be written to stdout.

``` bash
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

``` bash
tamp compress enwik8 -o enwik8.tamp  # Compress a file
echo "hello world" | tamp compress | wc -c  # Compress a stream and print the compressed size.
```

The following options can impact compression ratios and memory usage:

-   `window` - `2^window` plaintext bytes to look back to try and find a
    pattern. A larger window size will increase the chance of finding a
    longer pattern match, but will use more memory, increase compression
    time, and cause each pattern-token to take up more space. Try
    smaller window values if compressing highly repetitive data, or
    short messages.
-   `literal` - Number of bits used in each plaintext byte. For example,
    if all input data is 7-bit ASCII, then setting this to 7 will
    improve literal compression ratios by 11.1%. The default, 8-bits,
    can encode any binary data.

### Decompression

Use `tamp decompress` to decompress a file or stream. If no input file is specified, data from stdin will be read. If no output is specified, the compressed output stream will be written to stdout.

``` bash
$ tamp decompress --help
Usage: tamp decompress [ARGS] [OPTIONS]

Decompress an input file or stream.

╭─ Parameters ───────────────────────────────────────────────────────────────────────────────╮
│ INPUT,--input    -i  Input file to decompress. Defaults to stdin.                          │
│ OUTPUT,--output  -o  Output decompressed file. Defaults to stdout.                         │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

``` bash
tamp decompress enwik8.tamp -o enwik8
echo "hello world" | tamp compress | tamp decompress
```

## Python

The python library can perform one-shot compression, as well as operate on files/streams.

``` python
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

-   [zlib](https://docs.python.org/3/library/zlib.html), a python builtin gzip-compatible DEFLATE compression library.
-   [heatshrink](https://github.com/atomicobject/heatshrink), a data compression library for embedded/real-time systems. Heatshrink has similar goals as Tamp.

All of these are LZ-based compression algorithms, and tests were performed using a 1KB (10 bit) window. Since zlib already uses significantly more memory by default, the lowest memory level (`memLevel=1`) was used in these benchmarks. It should be noted that higher zlib memory levels will having greater compression ratios than Tamp. Currently, there is no micropython-compatible zlib or heatshrink compression implementation, so these numbers are provided simply as a reference.

## Compression Ratio

The following table shows compression algorithm performance over a variety of input data sourced from the [Silesia Corpus](https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia) and [Enwik8](https://mattmahoney.net/dc/textdata.html). This should give a general idea of how these algorithms perform over a variety of input data types.

  dataset               |   raw         |   tamp           |   zlib           |   heatshrink
----------------------- | ------------- | ---------------- | ---------------- | ------------
  enwik8                |   100,000,000 |   **51,635,633** |   56,205,166     |   56,110,394
  build/silesia/dickens |   10,192,446  |   **5,546,761**  |   6,049,169      |   6,155,768
  build/silesia/mozilla |   51,220,480  |   25,121,385     |   **25,104,966** |   25,435,908
  build/silesia/mr      |   9,970,564   |   5,027,032      |   **4,864,734**  |   5,442,180
  build/silesia/nci     |   33,553,445  |   8,643,610      |   **5,765,521**  |   8,247,487
  build/silesia/ooffice |   6,152,192   |   **3,814,938**  |   4,077,277      |   3,994,589
  build/silesia/osdb    |   10,085,684  |   **8,520,835**  |   8,625,159      |   8,747,527
  build/silesia/reymont |   6,627,202   |   **2,847,981**  |   2,897,661      |   2,910,251
  build/silesia/samba   |   21,606,400  |   9,102,594      |   **8,862,423**  |   9,223,827
  build/silesia/sao     |   7,251,944   |   **6,137,755**  |   6,506,417      |   6,400,926
  build/silesia/webster |   41,458,703  |   **18,694,172** |   20,212,235     |   19,942,817
  build/silesia/x-ray   |   8,474,240   |   7,510,606      |   **7,351,750**  |   8,059,723
  build/silesia/xml     |   5,345,280   |   1,681,687      |   **1,586,985**  |   1,665,179

Tamp usually out-performs heatshrink, and is generally very competitive with zlib. While trying to be an apples-to-apples comparison, zlib still uses significantly more memory during both compression and decompression (see next section). Tamp accomplishes competitive performance while using around 10x less memory.

## Memory Usage

The following table shows approximately how much memory each algorithm
uses during compression and decompression.

|                       | Compression                   | Decompression           |
| --------------------- | ----------------------------- | ----------------------- |
| Tamp                  | (1 << windowBits)             | (1 << windowBits)       |
| ZLib                  | (1 << (windowBits + 2)) + 7KB | (1 << windowBits) + 7KB |
| Heatshrink            | (1 << (windowBits + 1))       | (1 << (windowBits + 1)) |
| Deflate (micropython) | (1 << windowBits)             | (1 << windowBits)       |

All libraries have a few dozen bytes of overhead in addition to the primary window buffer, but are implementation-specific and ignored for clarity here. Tamp uses significantly less memory than ZLib, and half the memory of Heatshrink.

## Runtime

As a rough benchmark, here is the performance (in seconds) of these different compression algorithms on the 100MB enwik8 dataset. These tests were performed on an M1 Macbook Air.

|                            | Compression (s) | Decompression (s) |
| -------------------------- | --------------- | ----------------- |
| Tamp (Python Reference)    | 109.5           | 76.0              |
| Tamp (C)                   | 16.45           | 0.142             |
| ZLib                       | 0.98            | 0.98              |
| Heatshrink (with index)    | 6.22            | 0.82              |
| Heatshrink (without index) | 41.73           | 0.82              |

Heatshrink v0.4.1 was used in these benchmarks. When heathshrink uses an index, an additional `(1 << (windowBits + 1))` bytes of memory are used, resulting in 4x more memory-usage than Tamp. Tamp could use a similar indexing to increase compression speed, but has chosen not to to focus on the primary goal of a low-memory compressor.

To give an idea of Tamp's speed on an embedded device, the following table shows compression/decompression in **bytes/second of the first 100KB of enwik8 on a pi pico (rp2040)** at the default 125MHz clock rate. The C benchmark **does not** use a filesystem nor dynamic memory allocation, so it represents the maximum speed Tamp can achieve. In all tests, a 1KB window (10 bit) was used.

|                                  | Compression (bytes/s) | Decompression (bytes/s) |
| -------------------------------- | --------------------- | ----------------------- |
| Tamp (MicroPython Viper)         | 4,300                 | 42,000                  |
| Tamp (Micropython Native Module) | 12,770                | 644,000                 |
| Tamp (C)                         | 28,500                | 1,042,524               |
| Deflate (micropython builtin)    | 6,715                 | 146,477                 |


Tamp resulted in a **51637** byte archive, while Micropython's builtin `deflate` resulted in a larger, **59442** byte archive.

## Binary Size

To give an idea on the resulting binary sizes, Tamp and other libraries were compiled for the Pi Pico (`armv6m`). All libraries were compiled with `-O3`. Numbers reported in bytes.

|                           | Compressor | Decompressor | Compressor + Decompressor |
| ------------------------- | ---------- | ------------ | ------------------------- |
| Tamp (MicroPython Viper)  | 4429       | 4205         | 7554                      |
| Tamp (MicroPython Native) | 3232       | 3047         | 5505                      |
| Tamp (C)                  | 2008       | 1972         | 3864                      |
| Heatshrink (C)            | 2956       | 3876         | 6832                      |
| uzlib (C)                 | 2355       | 3963         | 6318                      |

Heatshrink doesn't include a high level API; in an apples-to-apples comparison the Tamp library would be even smaller.

## Acknowledgement

* Thanks @BitsForPeople for the esp32-optimized compressor implementation.
