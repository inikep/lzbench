# LZAV - Fast Data Compression Algorithm (in C/C++)

## Introduction

LZAV is a fast general-purpose in-memory data compression algorithm based on
now-classic [LZ77](https://wikipedia.org/wiki/LZ77_and_LZ78) lossless data
compression method. LZAV holds a good position on the Pareto landscape of
factors, among many similar in-memory (non-streaming) compression algorithms.

LZAV algorithm's code is portable, cross-platform, scalar, header-only,
inlineable C (C++ compatible). It supports big- and little-endian platforms,
and any memory alignment models. The algorithm is efficient on both 32- and
64-bit platforms. Incompressible data almost does not expand. Compliant with
WebAssembly (WASI libc), and runs at just twice lower performance than native
code.

LZAV does not sacrifice internal out-of-bounds (OOB) checks for decompression
speed. This means that LZAV can be used in strict conditions where OOB memory
writes (and especially reads) that lead to a trap, are unacceptable (e.g.,
real-time, system, server software). LZAV can be used safely (causing no
crashing nor UB) even when decompressing malformed or damaged compressed data.
Which means that LZAV does not require calculation of a checksum (or hash) of
the compressed data. Only a checksum of the uncompressed data may be required,
depending on application's guarantees.

The internal functions available in the `lzav.h` file allow you to easily
implement, and experiment with, your own compression algorithms. LZAV stream
format and decompressor have a potential of high decompression speeds and
compression ratios, which depends on the way data is compressed.

## Usage

To compress data:

```c
#include "lzav.h"

int max_len = lzav_compress_bound( src_len );
void* comp_buf = malloc( max_len );
int comp_len = lzav_compress_default( src_buf, comp_buf, src_len, max_len );

if( comp_len == 0 && src_len != 0 )
{
    // Error handling.
}
```

To decompress data:

```c
#include "lzav.h"

void* decomp_buf = malloc( src_len );
int l = lzav_decompress( comp_buf, decomp_buf, comp_len, src_len );

if( l < 0 )
{
    // Error handling.
}
```

To compress data with a higher ratio, for non-time-critical uses (e.g.,
compression of application's static assets):

```c
#include "lzav.h"

int max_len = lzav_compress_bound_hi( src_len ); // Note another bound function!
void* comp_buf = malloc( max_len );
int comp_len = lzav_compress_hi( src_buf, comp_buf, src_len, max_len );

if( comp_len == 0 && src_len != 0 )
{
    // Error handling.
}
```

LZAV algorithm and its source code (which is
[ISO C99](https://en.wikipedia.org/wiki/C99)) were quality-tested with:
Clang, GCC, MSVC, Intel C++ compilers; on x86, x86-64 (Intel, AMD), AArch64
(Apple Silicon) architectures; Windows 10, AlmaLinux 9.3, macOS 15.3.1.
Full C++ compliance is enabled conditionally and automatically, when the
source code is compiled with a C++ compiler.

## Customizing C++ namespace

If for some reason, in C++ environment, it is undesired to export LZAV symbols
into the global namespace, the `LZAV_NS_CUSTOM` macro can be defined
externally:

```c++
#define LZAV_NS_CUSTOM lzav
#include "lzav.h"
```

Similarly, LZAV symbols can be placed into any other custom namespace (e.g.,
a namespace with data compression functions):

```c++
#define LZAV_NS_CUSTOM my_namespace
#include "lzav.h"
```

This way, LZAV symbols and functions can be referenced like
`my_namespace::lzav_compress_default(...)`. Note that since all LZAV functions
have a `static inline` specifier, there can be no ABI conflicts, even if the
header is included in unrelated, mixed C/C++, compilation units.

## Comparisons

The tables below present performance ballpark numbers of LZAV algorithm
(based on Silesia dataset).

While LZ4 there seems to be compressing faster, LZAV comparably provides 14.2%
memory storage cost savings. This is a significant benefit in database and
file system use cases since compression is only about 35% slower while CPUs
rarely run at their maximum capacity anyway, and disk I/O times are reduced
due to a better compression. In general, LZAV holds a very strong position in
this class of data compression algorithms, if one considers all factors:
compression and decompression speeds, compression ratio, and not less
important - code maintainability: LZAV is maximally portable and has a rather
small independent codebase.

Performance of LZAV is not limited to the presented ballpark numbers.
Depending on the data being compressed, LZAV can achieve 800 MB/s compression
and 5000 MB/s decompression speeds. Incompressible data decompresses at 10000
MB/s rate, which is not far from the "memcpy". There are cases like the
[enwik9 dataset](https://mattmahoney.net/dc/textdata.html) where LZAV
provides 21.2% higher memory storage savings compared to LZ4. However, on
small data (below 50 KB), compression ratio difference between LZAV and LZ4
diminishes, and LZ4 may have some advantage.

LZAV algorithm's geomean performance on a variety of datasets is 530 +/- 150
MB/s compression and 3800 +/- 1300 MB/s decompression speeds, on 4+ GHz 64-bit
processors released since 2019. Note that the algorithm exhibits adaptive
qualities, and its actual performance depends on the data being compressed.
LZAV may show an exceptional performance on your specific data, including, but
not limited to: sparse databases, log files, HTML/XML files.

It is also worth noting that compression methods like LZAV and LZ4 usually
have an advantage over dictionary- and entropy-based coding in that
hash-table-based compression has a small operation and memory overhead while
the classic LZ77 decompression has no overhead at all - this is especially
relevant for smaller data.

For a more comprehensive in-memory compression algorithms benchmark you may
visit [lzbench](https://github.com/inikep/lzbench).

### Apple clang 15.0.0 arm64, macOS 15.3.1, Apple M1, 3.5 GHz

Silesia compression corpus

|Compressor      |Compression    |Decompression  |Ratio %        |
|----            |----           |----           |----           |
|**LZAV 4.18**   |600 MB/s       |3810 MB/s      |40.83          |
|LZ4 1.9.4       |700 MB/s       |4570 MB/s      |47.60          |
|Snappy 1.1.10   |495 MB/s       |3230 MB/s      |48.22          |
|LZF 3.6         |395 MB/s       |800 MB/s       |48.15          |
|**LZAV 4.18 HI**|129 MB/s       |3770 MB/s      |35.58          |
|LZ4HC 1.9.4 -9  |40 MB/s        |4360 MB/s      |36.75          |

### LLVM clang 18.1.8 x86-64, AlmaLinux 9.3, Xeon E-2386G (RocketLake), 5.1 GHz

Silesia compression corpus

|Compressor      |Compression    |Decompression  |Ratio %        |
|----            |----           |----           |----           |
|**LZAV 4.18**   |590 MB/s       |3620 MB/s      |40.83          |
|LZ4 1.9.4       |848 MB/s       |4980 MB/s      |47.60          |
|Snappy 1.1.10   |690 MB/s       |3360 MB/s      |48.22          |
|LZF 3.6         |455 MB/s       |1000 MB/s      |48.15          |
|**LZAV 4.18 HI**|112 MB/s       |3500 MB/s      |35.58          |
|LZ4HC 1.9.4 -9  |43 MB/s        |4920 MB/s      |36.75          |

### LLVM clang-cl 18.1.8 x86-64, Windows 10, Ryzen 3700X (Zen2), 4.2 GHz

Silesia compression corpus

|Compressor      |Compression    |Decompression  |Ratio %        |
|----            |----           |----           |----           |
|**LZAV 4.18**   |502 MB/s       |3120 MB/s      |40.83          |
|LZ4 1.9.4       |675 MB/s       |4560 MB/s      |47.60          |
|Snappy 1.1.10   |415 MB/s       |2440 MB/s      |48.22          |
|LZF 3.6         |310 MB/s       |700 MB/s       |48.15          |
|**LZAV 4.18 HI**|114 MB/s       |3040 MB/s      |35.58          |
|LZ4HC 1.9.4 -9  |36 MB/s        |4430 MB/s      |36.75          |

P.S. Popular Zstd's benchmark was not included here, because it is not a pure
LZ77, much harder to integrate, and has a much larger code size - a different
league, close to zlib. Here are author's Zstd measurements with
[TurboBench](https://github.com/powturbo/TurboBench/releases), on Ryzen 3700X,
on Silesia dataset:

|Compressor      |Compression    |Decompression  |Ratio %        |
|----            |----           |----           |----           |
|zstd 1.5.5 -1   |460 MB/s       |1870 MB/s      |41.0           |
|zstd 1.5.5 1    |436 MB/s       |1400 MB/s      |34.6           |

## Notes

1. LZAV API is not equivalent to LZ4 nor Snappy API. For example, the "dstl"
parameter in the decompressor should specify the original uncompressed length,
which should have been previously stored in some way, independent of LZAV.

2. From a technical point of view, peak decompression speeds of LZAV have an
implicit limitation arising from its more complex stream format, compared to
LZ4: LZAV decompression requires more code branching. Another limiting factor
is a rather big 8 MiB LZ77 window which is not CPU cache-friendly. On the
other hand, without these features it would not be possible to achieve
competitive compression ratios while having fast compression speeds.

3. LZAV supports compression of continuous data blocks of up to 2 GB. Larger
data should be compressed in chunks of at least 32 MB. Using smaller chunks
may reduce the achieved compression ratio.

## Thanks

* [Paul Dreik](https://github.com/pauldreik), for finding memcpy UB in the
decompressor.
