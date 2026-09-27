# LZAV - Fast Data Compression Algorithm (in C/C++)

## Introduction

LZAV is a fast, general-purpose, in-memory data compression algorithm based on
the classic [LZ77](https://wikipedia.org/wiki/LZ77_and_LZ78) lossless
compression method. LZAV holds a strong position on the Pareto frontier among
many similar in-memory (non-streaming) compression algorithms.

LZAV's code is portable, cross-platform, scalar, and header-only. It is also
compatible with C++. It supports big- and little-endian platforms and any
memory alignment model. The algorithm is efficient on both 32- and 64-bit
platforms. It is compatible with WebAssembly and, when compiled against WASI
libc, achieves roughly half the speed of native code. The algorithm does not
measurably increase the size of incompressible data.

LZAV performs internal out-of-bounds (OOB) checks and does not trade safety
for decompression speed. This means that LZAV can be used under strict
conditions where OOB memory reads and writes, which could lead to CPU traps,
are unacceptable (e.g., in real-time, system-level, and server software). LZAV
can be used safely (without crashes or undefined behavior) even when
decompressing malformed or damaged data. Consequently, it does not require
calculating a checksum (or a hash) of the compressed data. Only a checksum of
the uncompressed data may be required, depending on the application's needs.

The internal functions in `lzav.h` allow developers to implement and test
their own compression algorithms. LZAV's stream format allows for high
compression ratios and high decompression speeds.

## Usage

To compress data:

```c
#include "lzav.h"

int max_len = lzav_compress_bound( src_len );
void* comp_buf = malloc( max_len );
int comp_len = lzav_compress_default( src_buf, comp_buf, src_len, max_len );

if( comp_len == 0 && src_len != 0 )
{
    // Error handling
}
```

To decompress data:

```c
#include "lzav.h"

void* decomp_buf = malloc( src_len );
int l = lzav_decompress( comp_buf, decomp_buf, comp_len, src_len );

if( l < 0 )
{
    // Error handling
}
```

To compress data with a higher compression ratio for non-time-critical uses
(e.g., compression of an application's static assets):

```c
#include "lzav.h"

int max_len = lzav_compress_bound_hi( src_len ); // Note the different bound function
void* comp_buf = malloc( max_len );
int comp_len = lzav_compress_hi( src_buf, comp_buf, src_len, max_len );

if( comp_len == 0 && src_len != 0 )
{
    // Error handling
}
```

LZAV's source code conforms to [ISO C99](https://en.wikipedia.org/wiki/C99)
and has been tested with Clang, GCC, MSVC, and the Intel C++ compiler on x86,
x86-64 (Intel, AMD), and AArch64 (Apple Silicon) systems running Windows 10,
Windows 11, AlmaLinux 9.6, and macOS 26.4. Full C++ compatibility is
automatically provided when the source code is compiled with a C++ compiler.

## Ports

* [C#, by AQtun](https://www.nuget.org/packages/AQtun.LZAV)
* [C++, via vcpkg](https://vcpkg.link/ports/lzav)
* [FreeArc, by Shegorat](https://krinkels.org/resources/cls-lzav.579/)
* [Rust, by pkolaczk](https://crates.io/crates/lzav)

## Customizing the C++ Namespace

In C++ environments where it is undesirable to place LZAV symbols in the
global namespace, the `LZAV_NS_CUSTOM` macro can be defined externally:

```c++
#define LZAV_NS_CUSTOM lzav
#include "lzav.h"
```

For example, to place LZAV symbols in a custom namespace alongside your
other data compression functions:

```c++
#define LZAV_NS_CUSTOM my_compressors
#include "lzav.h"
```

As a result, LZAV functions can be referred to as
`my_compressors::lzav_compress_default(...)`. Note that because all LZAV
functions are declared with the `static` specifier, there will be no ABI
conflicts, even if the `lzav.h` header is included in multiple C/C++
translation units.

## Comparisons

The tables below present ballpark performance numbers for the LZAV algorithm.

While LZ4 compresses faster, LZAV provides 16% greater storage space savings.
This is a significant benefit in database and filesystem use cases, since
LZAV's compression is only about 30% slower than LZ4's. In practice, slower
compression is not a limiting factor because writes of compressed data are
deferred to background threads, and disk I/O time is reduced due to better
compression. In general, LZAV holds a very strong position in this class of
data compression algorithms when one considers all factors: compression and
decompression speeds, compression ratio, and, just as importantly, code
maintainability. LZAV is highly portable and has a rather small, independent
codebase.

The performance of LZAV is not limited to the ballpark numbers presented.
Depending on the data being compressed, LZAV can achieve compression speeds
of 800 MB/s and decompression speeds of 5,000 MB/s. Incompressible data is
decompressed at a rate of 10,000 MB/s, which is close to the speed of
`memcpy()`. For datasets such as [enwik9](https://mattmahoney.net/dc/textdata.html),
LZAV provides 22% greater storage space savings than LZ4.

The geometric mean speeds of the LZAV algorithm across a variety of datasets
are 550 &plusmn; 150 MB/s for compression and 3,800 &plusmn; 1,300 MB/s for
decompression. These numbers apply to 64-bit processors released since 2019
that run at 4 GHz or higher. Note that the algorithm exhibits adaptive
qualities, and its actual performance depends on the data being compressed.
LZAV may show exceptional performance on your specific data, including, but
not limited to, sparse databases, log files, and HTML/XML files.

It is also worth noting that compression methods like LZAV and LZ4 usually
have an advantage over explicit dictionary- and entropy-based methods.
Hash-table-based compression has low memory usage and minimal operational
overhead, and classic LZ77 decompression has no overhead at all; this is
especially relevant for smaller datasets.

For a more comprehensive benchmark of in-memory compression algorithms, see
[lzbench](https://github.com/inikep/lzbench).

The benchmarks below use the Silesia compression corpus.

### Apple clang 15.0.0 arm64, macOS 26.4, Apple M1, 3.5 GHz

| Compressor       | Compression | Decompression | Ratio % |
|------------------|------------:|--------------:|:-------:|
| **LZAV 5.17**    | 627 MB/s    | 3,830 MB/s    | 39.91   |
| LZ4 1.9.4        | 700 MB/s    | 4,570 MB/s    | 47.60   |
| Snappy 1.1.10    | 495 MB/s    | 3,230 MB/s    | 48.22   |
| LZF 3.6          | 395 MB/s    | 800 MB/s      | 48.15   |
| **LZAV 5.17 HI** | 146 MB/s    | 3,760 MB/s    | 34.85   |
| LZ4HC 1.9.4 -9   | 40 MB/s     | 4,360 MB/s    | 36.75   |

### LLVM clang 19.1.7 x86-64, AlmaLinux 9.6, Xeon E-2386G (Rocket Lake), 5.1 GHz

| Compressor       | Compression | Decompression | Ratio % |
|------------------|------------:|--------------:|:-------:|
| **LZAV 5.17**    | 655 MB/s    | 3,550 MB/s    | 39.91   |
| LZ4 1.9.4        | 848 MB/s    | 4,980 MB/s    | 47.60   |
| Snappy 1.1.10    | 690 MB/s    | 3,360 MB/s    | 48.22   |
| LZF 3.6          | 455 MB/s    | 1,000 MB/s    | 48.15   |
| **LZAV 5.17 HI** | 128 MB/s    | 3,360 MB/s    | 34.85   |
| LZ4HC 1.9.4 -9   | 43 MB/s     | 4,920 MB/s    | 36.75   |

### LLVM clang-cl 18.1.8 x86-64, Windows 10, Ryzen 3700X (Zen 2), 4.2 GHz

| Compressor       | Compression | Decompression | Ratio % |
|------------------|------------:|--------------:|:-------:|
| **LZAV 5.17**    | 565 MB/s    | 3,270 MB/s    | 39.91   |
| LZ4 1.9.4        | 675 MB/s    | 4,560 MB/s    | 47.60   |
| Snappy 1.1.10    | 415 MB/s    | 2,440 MB/s    | 48.22   |
| LZF 3.6          | 310 MB/s    | 700 MB/s      | 48.15   |
| **LZAV 5.17 HI** | 124 MB/s    | 3,100 MB/s    | 34.85   |
| LZ4HC 1.9.4 -9   | 36 MB/s     | 4,430 MB/s    | 36.75   |

Note: The popular Zstd compressor is not included here because it is not
a pure LZ77 algorithm, it is much harder to integrate, and it has a much
larger codebase. These aspects place it in the same league as zlib. Here are
the author's Zstd measurements with [TurboBench](https://github.com/powturbo/TurboBench/releases)
on a Ryzen 3700X, using the Silesia dataset:

| Compressor         | Compression | Decompression | Ratio % |
|--------------------|------------:|--------------:|:-------:|
| zstd 1.5.5 fast -1 | 460 MB/s    | 1,870 MB/s    | 41.0    |
| zstd 1.5.5 -1      | 436 MB/s    | 1,400 MB/s    | 34.6    |

## Dataset Benchmarks

This section presents compression ratio comparisons for various popular
datasets. Note that each file within these datasets was compressed
individually, which is reflected in the overall ratio.

| Dataset               | Size (MiB) | LZAV 5.17 | LZ4 1.9.4 | Snappy 1.1.10 | LZF 3.6 | Source |
|-----------------------|-----------:|:---------:|:---------:|:-------------:|:-------:|--------|
| 4SICS 151020 PCAP     | 24.5       | 20.46     | 21.82     | 24.95         | 25.34   | [www.netresec.com](https://www.netresec.com/?page=PCAP4SICS) |
| 4SICS 151022 PCAP     | 200.0      | 36.44     | 37.35     | 40.24         | 41.37   | [www.netresec.com](https://www.netresec.com/?page=PCAP4SICS) |
| Calgary Large         | 3.1        | 43.74     | 51.97     | 51.76         | 49.07   | [data-compression.info](https://www.data-compression.info/Corpora/CalgaryCorpus/) |
| Canterbury            | 2.7        | 37.65     | 43.73     | 45.42         | 42.49   | [corpus.canterbury.ac.nz](https://corpus.canterbury.ac.nz/) |
| Canterbury Large      | 10.6       | 37.40     | 51.97     | 48.37         | 54.28   | [corpus.canterbury.ac.nz](https://corpus.canterbury.ac.nz/) |
| Canterbury Artificial | 0.3        | 33.36     | 33.74     | 36.48         | 34.66   | [corpus.canterbury.ac.nz](https://corpus.canterbury.ac.nz/) |
| employees_10KB.json   | 0.01       | 22.64     | 24.68     | 23.92         | 23.52   | [sample.json-format.com](https://sample.json-format.com/) |
| employees_100KB.json  | 0.1        | 15.96     | 17.71     | 19.02         | 21.88   | [sample.json-format.com](https://sample.json-format.com/) |
| employees_50MB.json   | 51.5       | 10.86     | 16.42     | 18.57         | 21.44   | [sample.json-format.com](https://sample.json-format.com/) |
| enwik8                | 95.4       | 44.44     | 57.26     | 56.56         | 53.95   | [www.mattmahoney.net](https://www.mattmahoney.net/dc/textdata.html) |
| enwik9                | 954.7      | 39.32     | 50.92     | 50.79         | 49.30   | [www.mattmahoney.net](https://www.mattmahoney.net/dc/textdata.html) |
| Manzini               | 855.3      | 26.70     | 37.30     | 38.57         | 39.04   | [people.unipmn.it/manzini](https://people.unipmn.it/manzini/boosting/index.html) |
| chr22.dna (Manzini)   | 33.0       | 37.76     | 52.82     | 44.53         | 55.86   | [people.unipmn.it/manzini](https://people.unipmn.it/manzini/boosting/index.html) |
| w3c2 HTML (Manzini)   | 99.4       | 11.33     | 22.20     | 25.35         | 27.20   | [people.unipmn.it/manzini](https://people.unipmn.it/manzini/boosting/index.html) |
| Silesia               | 202.1      | 39.91     | 47.60     | 48.17         | 48.15   | [github.com/MiloszKrajewski](https://github.com/MiloszKrajewski/SilesiaCorpus) |

## Notes

1. The LZAV API is not equivalent to the LZ4 and Snappy APIs. For example, the
`dstlen` parameter of the decompression function must specify the original
uncompressed length, which must have been previously stored in some way,
independently of LZAV.

2. From a technical point of view, LZAV's peak decompression speed is
implicitly limited by its stream format, which is more complex than LZ4's.
LZAV decompression requires extensive branching. Another limiting factor is
a rather large dynamic LZ77 window (2 MiB to 512 MiB), which is not
CPU-cache-friendly. However, without these features, it would not be possible
to achieve competitive compression ratios while maintaining high compression
speeds.

3. LZAV supports compression of data blocks up to 2 GiB. Larger datasets
should be compressed in chunks of at least 16 MiB. Using smaller chunks may
reduce the compression ratio.
