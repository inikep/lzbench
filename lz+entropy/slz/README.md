SLZ / StateLess Zip version 1.4
-------------------------------

## Intro

SLZ is a fast and memory-less stream compressor which produces an output that
can be decompressed with slz, zlib or gzip. It provides uslz API for
lightweight decompression using 32K sliding buffer. However if decompression
performance is the only priority zlib is perfectly fine for this, though some
significant improvements are available in the development branch.

The purpose is to use SLZ in situations where a zlib-compatible stream is
needed and zlib's resource usage would be too high while the compression ratio
is not critical. The typical use case is in HTTP servers and gateways which
have to compress many streams in parallel with little CPU resources to assign
to this task, and without having to limit the compression ratio due to the
memory usage. In such an environment, the server's memory usage can easily be
divided by 10 and the CPU usage by 3.

## How does it work ?

While zlib uses 256 kB of memory per stream in addition to a few tens of bytes
for the stream descriptor itself, SLZ only stores a stream descriptor made of
40 bytes on 64-bit machines (32 bytes on 32-bit ones). Thus it is particularly
suited to environments having to deal with tens to hundreds of thousands of
concurrent streams.

The key difference between zlib and SLZ is that SLZ is stateless in that it
doesn't consider the data previously compressed as part of its dictionary. It
doesn't hurt compression performance when it is fed in large enough chunks (at
least a few kB at once), but it should preferably not be used if it has to read
less than a few kB at a time without implementing an input buffer to maximize
the compression ratio. For example, an HTTP gateway may defragment all the
parts from a chunked-encoded response buffer into a linear buffer that is then
fed at once to SLZ. This temporary buffer would then be static and shared
between all streams running on the same thread.

A side effect from its stateless nature is that it can emit a zlib-compatible
stream without being vulnerable to CRIME-like attacks by processing the
sensitive and attacker-controlled data in distinct batches. Since no dictionary
persists between calls, there is no risk that attacker-controlled data is used
to guess sensitive data using dictionary lookups.

Another induced benefit which was not initially seeked is that SLZ is about 5
times as fast as zlib in its fastest mode. This is the result of the simpler
matching and encoding mechanisms involved which permit even more optimisations.

Since version 1.4, four compression levels are available. Level 1 is the
fastest one and gives up early on hard-to-compress areas. Level 2 is the
default, it gives up just as early but keeps checking the areas it skips, and
stops skipping as soon as it recognizes something. Level 3 searches more
before giving up. Level 4 never gives up and extends matches further
backwards, for a slightly better ratio. In addition, a text content type may
be selected for HTML, CSS, JS, JSON, SVG and similar contents
(see "content type" below).

## Comparative tests

Comparative tests were performed using some reference datasets that are already
being used for other libraries. This performance report was produced using the
Silesia Corpus to get something comparable to what is reported by the [LZ4
utility here](https://github.com/Cyan4973/lz4) and is also used as a reference
in lzbench. Tests were performed on various hardware, from 16-core ARM to
64-core AMD EPYC, including modern x86_64 and ARMv9 cores.

The following tests were run on a single core of an Intel Core Ultra 7 265K at
5.5 GHz (turbo mode). The input file is the Silesia Corpus tar file placed in a
RAM disk (211947520 bytes), passed 10 times on the command line. The RAM disk
is a tmpfs mounted with huge=always, so that its 2 MB pages amortize the cost
of mapping and unmapping the input file's pages, which is specific to this test
and not to the compressors. With 4 kB pages, they would all lose 2 to 3% of
their speed in this mapping and unmapping. The output size was measured using
"wc -c", but the bandwidth was measured by sending the output to /dev/null so
as not to let the "wc" command slow everything down. The times include the
process startup and the reading of the input. When used as a library, all
compressors will deliver a slightly higher performance by avoiding a few
file-oriented syscalls, but at least here they're all compared under the exact
same conditions.

|  Product              |  Input BW  |  Compression ratio |                              |
|:--------------------- |:----------:|:-----:|:------------------------------------------|
|  lz4-1.8.2 -1         |  853 MB/s  |  2.10 | lz4 -1zc (fastest)                        |
|  lz4-1.8.2 -2         |  853 MB/s  |  2.10 | lz4 -2zc (exact same output as -1)        |
|  zstd-1.5.7 --fast=4  |  863 MB/s  |  2.15 | zstd -c --fast=4                          |
|  zstd-1.5.7 --fast=3  |  837 MB/s  |  2.24 | zstd -c --fast=3                          |
|  zstd-1.5.7 --fast=2  |  798 MB/s  |  2.34 | zstd -c --fast=2                          |
|  zstd-1.5.7 --fast=1  |  756 MB/s  |  2.44 | zstd -c --fast=1 (same as --fast)         |
|  zstd-1.5.7 -1        |  617 MB/s  |  2.89 | zstd -c1                                  |
|  zstd-1.5.7 -2        |  552 MB/s  |**3.05**| zstd -c2                                 |
|  zstd-1.5.7 -3        |  405 MB/s  |**3.20**| zstd -c3                                 |
|  slz-1.4 -1 -D        |**928** MB/s|  2.15 | slz in deflate format, level 1 (fastest)  |
|  slz-1.4 -1 -G        |**902** MB/s|  2.15 | slz in gzip format, level 1 (fastest)     |
|  slz-1.4 -1 -Z        |  899 MB/s  |  2.15 | slz in zlib format, level 1 (fastest)     |
|  slz-1.4 -2 -D        |**903** MB/s|  2.27 | slz in deflate format, level 2 (default)  |
|  slz-1.4 -3 -D        |  792 MB/s  |  2.45 | slz in deflate format, level 3            |
|  slz-1.4 -4 -D        |  752 MB/s  |  2.47 | slz in deflate format, level 4 (best)     |
|  slz-1.3.2 -1 -D      |  619 MB/s  |  2.13 | slz in deflate format, level 1            |
|  slz-1.3.2 -1 -G      |  478 MB/s  |  2.13 | slz in gzip format, level 1               |
|  slz-1.3.2 -1 -Z      |  547 MB/s  |  2.13 | slz in zlib format, level 1               |
|  gzip-1.14 -1         |  163 MB/s  |  2.74 | gzip level 1 (fastest)                    |

![Silesia corpus on a Core Ultra 7 265K](tests/bench/img/slz-cmp-silesia.png)

zstd --fast=N selects zstd's negative levels, which trade compression ratio for
speed as N grows, --fast alone being --fast=1. At --fast=4, zstd already
compresses no more than slz -1, and the higher values continue the trend. Lz4
only takes one file, so it was measured on 10 concatenated copies of the file.
For the last 10 years, SLZ has been part of
[Inikep's lzbench benchmark suite](https://github.com/inikep/lzbench/), which
compares a wide range of compression tools. The project maintains an updated
performance table where other solutions can be compared. Please visit it, it
shows that there are compressors for many different purposes.

**Only gzip/zlib and SLZ** can emit a stream that a regular zlib-compatible browser
can decompress (RFC1950/1951/1952 aka zlib, deflate, gzip). Here SLZ is **4.5 to
5.5 times faster than zlib**, for a payload 11 to 28% larger than zlib's at its
fastest level 1. Slz 1.4's gzip compression is **1.77 times faster** than that of
slz-1.3, its zlib compression is 1.54 times faster, and its deflate compression
is 45% faster, all while compressing slightly more. **Slz-1.4's levels 1 and 2
are both faster and stronger than lz4 on this machine**.

Compared to zstd, slz is not put to shame either. Zstd's regular levels always
compress better, and are always slower. On the fast side, slz -1 compresses as
much as zstd --fast=4 and is 7.5% faster, **slz -2 is both faster and stronger
than zstd --fast=3 and --fast=4, and slz -3 is both faster and stronger than
zstd --fast=1** (792 MB/s for 2.45 against 756 MB/s for 2.44).

Levels 2 and above switch to slz's text tree by themselves when it pays, which
brings the output of levels 3 and 4 within 11% of zlib's on Silesia, as it
contains some text. The -T option of earlier versions, which forced the text
content type, was removed: with it, levels 1 and 2 produced the same output as
level 3. The **gzip and zlib formats cost 2.8% and 3.1%** respectively compared
to raw deflate on this CPU.

Modern ARM cores deliver good performance as well, as seen below on an A720
core of a CIX-P1 processor in Radxa Orion O6 on the Silesia corpus file (passed
3 times, measured as above):

|  Product          |   Input BW  | Compression ratio |                             |
|:------------------|:-----------:|:----:|:-----------------------------------------|
|  slz-1.4 -1 -D    |    519 MB/s | 2.14 | slz in deflate format, level 1 (fastest) |
|  slz-1.4 -1 -G    |    480 MB/s | 2.14 | slz in gzip format, level 1 (fastest)    |
|  slz-1.4 -1 -Z    |    496 MB/s | 2.14 | slz in zlib format, level 1 (fastest)    |
|  slz-1.4 -2 -D    |    501 MB/s | 2.26 | slz in deflate format, level 2 (default) |
|  slz-1.4 -3 -D    |    433 MB/s | 2.45 | slz in deflate format, level 3           |
|  slz-1.4 -4 -D    |    409 MB/s | 2.47 | slz in deflate format, level 4 (best)    |
|  gzip -1          |     81 MB/s | 2.74 | gzip level 1 (fastest)                   |

![Silesia corpus on an A720 core](tests/bench/img/slz-cmp-a720.png)

Here **slz -1 is 6.4 times faster than gzip -1**.

Compressing the sources of a Linux 7.0 kernel better illustrates the gains to
expect from text inputs, on the same Core Ultra 7 265K (a 1643110400 bytes tar
file, passed 3 times, on the same RAM disk):

| Product             |   Input BW  | Compression ratio |                             |
|:--------------------|:-----------:|:----:|:-----------------------------------------|
| zstd-1.5.7 --fast=4 |   945 MB/s  | 4.44 | zstd -c --fast=4                         |
| zstd-1.5.7 --fast=3 |   928 MB/s  | 4.72 | zstd -c --fast=3                         |
| zstd-1.5.7 --fast=2 |   912 MB/s  | 5.01 | zstd -c --fast=2                         |
| zstd-1.5.7 --fast=1 |   895 MB/s  |**5.29**| zstd -c --fast=1 (same as --fast)        |
| slz-1.4 -1 -D       |**1012 MB/s**| 4.28 | slz in deflate format, level 1 (fastest) |
| slz-1.4 -1 -G       |   966 MB/s  | 4.28 | slz in gzip format, level 1 (fastest)    |
| slz-1.4 -1 -Z       |   963 MB/s  | 4.28 | slz in zlib format, level 1 (fastest)    |
| slz-1.4 -2 -D       |   990 MB/s  | 4.81 | slz in deflate format, level 2 (default) |
| slz-1.4 -3 -D       |   984 MB/s  | 4.85 | slz in deflate format, level 3           |
| slz-1.4 -4 -D       |   980 MB/s  | 4.85 | slz in deflate format, level 4 (best)    |
| gzip-1.14 -1        |   284 MB/s  | 4.97 | gzip level 1 (fastest)                   |

![Linux 7.0 sources on a Core Ultra 7 265K](tests/bench/img/slz-cmp-linux.png)

This is about one gigabyte per second of input data that is divided in size by
4.3 to 4.8. It's 3.6 times faster than gzip for a 16% larger output at level 1,
and still 3.5 times faster than gzip for a 3% larger output at level 2.

Here again slz completes the spectrum on the fast side: levels 2 to 4 are both
faster and stronger than zstd --fast=3 and --fast=4, and at its highest ratio,
slz compresses 9.5% faster than zstd --fast=1 for a 9% larger output.

Since SLZ focuses a lot on performance savings gained by keeping most of the
workload in CPU cache, it scales very well when multiple cores are used in
parallel (low memory bandwidth) as shown in this tests run on a 16-core ARM
Cortex A72 (NXP LX2160A) running at 2.2 GHz, using the Silesia corpus at
level 1 in deflate format :

|   #Cores | Input size | BW/core(MB/s) |  Total BW(MB/s) |
|:---------|:----------:|:-------------:| ---------------:|
|       1  |  211947520 |     169.4     |       169.4     |
|       2  |  211947520 |     172.6     |       345.2     |
|       3  |  211947520 |     171.6     |       514.8     |
|       4  |  211947520 |     172.2     |       688.8     |
|       6  |  211947520 |     172.6     |      1035.6     |
|       8  |  211947520 |     173.3     |      1386.4     |
|      12  |  211947520 |     173.1     |      2077.2     |
|      16  |  211947520 |     172.7     |      2763.2     |

![Scalability on a 16-core Cortex A72](tests/bench/img/slz-scale-a72.png)

Here the scaling is perfectly linear: each core compresses as fast with all 16
cores busy as when it runs alone. At level 3, the same machine processes
141.7 MB/s on one core and 2328 MB/s on 16 cores.

On a more powerful machine (64-core 3rd gen AMD EPYC at 3.7 GHz), with the same
settings, we observe this:

|   #Cores | Input size | BW/core(MB/s) |  Total BW(MB/s) |
|:---------|:----------:|:-------------:| ---------------:|
|       1  |  211947520 |     573.8     |       573.8     |
|       2  |  211947520 |     574.9     |      1095.8     |
|       3  |  211947520 |     572.4     |      1717.2     |
|       4  |  211947520 |     569.4     |      2277.6     |
|       8  |  211947520 |     540.1     |      4320.8     |
|      16  |  211947520 |     472.7     |      7563.2     |
|      32  |  211947520 |     461.6     |     14771.2     |
|      64  |  211947520 |     380.0     |     24320.0     |

On the same 64-core AMD EPYC with 2 threads per core:

|   #Cores | Input size  | BW/core(MB/s) |  Total BW(MB/s) |
|:---------|:-----------:|:-------------:| ---------------:|
|       1  | 2*211947520 |      757.0    |        757.0    |
|       2  | 2*211947520 |      762.6    |       1523.2    |
|       3  | 2*211947520 |      754.0    |       2262.0    |
|       4  | 2*211947520 |      756.2    |       3024.8    |
|       8  | 2*211947520 |      720.0    |       5760.0    |
|      16  | 2*211947520 |      624.4    |       9990.4    |
|      32  | 2*211947520 |      610.4    |      19532.8    |
|      64  | 2*211947520 |      447.6    |      28646.4    |

The per-core drop seen here is not caused by SLZ but by the CPU's frequency,
which falls as more cores become active: it runs at 3.7 GHz on up to 4 cores,
3.1 GHz from 8 to 32 cores, and about 2.5 GHz on all 64 cores. With the
frequency fixed at 2.5 GHz, the scaling is almost linear again, with each core
retaining 93% of its single-core performance with all 64 cores busy, and
running two threads per core brings 25 to 35% more.

One thread per core, at 2.5 GHz:

|   #Cores | Input size | BW/core(MB/s) |  Total BW(MB/s) |
|:---------|:----------:|:-------------:| ---------------:|
|       1  |  211947520 |     380.4     |       384.9     |
|       2  |  211947520 |     380.6     |       761.2     |
|       3  |  211947520 |     379.0     |      1137.0     |
|       4  |  211947520 |     378.9     |      1515.6     |
|       8  |  211947520 |     374.0     |      2992.0     |
|      16  |  211947520 |     376.1     |      6017.6     |
|      32  |  211947520 |     371.4     |     11885.1     |
|      64  |  211947520 |     356.0     |     22783.4     |

Two threads per core, at 2.5 GHz:

|   #Cores | Input size  | BW/core(MB/s) |  Total BW(MB/s) |
|:---------|:-----------:|:-------------:| ---------------:|
|       1  | 2*211947520 |     506.3     |       506.3     |
|       2  | 2*211947520 |     507.2     |      1014.3     |
|       3  | 2*211947520 |     505.0     |      1515.0     |
|       4  | 2*211947520 |     504.2     |      2016.8     |
|       8  | 2*211947520 |     496.5     |      3971.8     |
|      16  | 2*211947520 |     494.1     |      7906.1     |
|      32  | 2*211947520 |     490.1     |     15684.0     |
|      64  | 2*211947520 |     450.9     |     28855.6     |

With all 128 threads busy, the machine compresses the equivalent of about **230
Gbps of input TCP traffic to 107 Gbps of output TCP traffic** at level 1. At
level 3, which compresses better (ratio 2.32 instead of 2.15), it still
processes 26.0 GB/s, or 208 Gbps of input traffic compressed to 89.7 Gbps, and
a single core runs at 491 MB/s (650 MB/s with two threads).

![Scalability on a 64-core AMD EPYC](tests/bench/img/slz-scale-epyc.png)

Decompression is not SLZ's main purpose, but "slz -d", based on the built-in
uslz decompressor, is **1.7 to 1.8 times** as fast as zcat on streams produced
by slz, and 1.5 times as fast on streams produced by gzip. These tests were run
on a single core of the same Core Ultra 7 265K, decompressing the Silesia
corpus compressed in gzip format, passed 3 times on the command line. The times
are the best of 4 runs divided by 3, so that they are those of one pass
producing the 211947520 bytes of the original data:

|  Stream produced by  |   Decompressor    |   Time(s)  | Output BW(MB/s) |
|:---------------------|:-----------------:|:----------:|----------------:|
|   slz -G -1          | slz -d (slz-1.4)  |     0.228  |     930         |
|   slz -G -1          | zcat (gzip-1.14)  |     0.391  |     542         |
|   slz -G -3          | slz -d (slz-1.4)  |     0.246  |     862         |
|   slz -G -3          | zcat (gzip-1.14)  |     0.438  |     484         |
|   gzip -1            | slz -d (slz-1.4)  |     0.314  |     674         |
|   gzip -1            | zcat (gzip-1.14)  |     0.464  |     457         |

Since some products automatically throttle the compression level based on the
CPU usage or the available memory, on high speed networks, zlib can end up
showing a much lower average compression ratio in order to fill an uplink. With
SLZ, no extra memory usage is needed for compression, and the CPU usage is
reduced by 2.5-3.5 times, meaning that the throttling will happen at much
higher data rates. The net effect is a significant improvement on the
compression ratio compared to zlib at a given resource usage.

## Increasing compression ratio in HTTP gateways

*Note: the measurements below were performed with version 1.2 and will need to
be performed again with the current version.*

HAProxy running on a single core of a core i5-3320M, with 500 concurrent users,
the default 16kB buffers, and accessing www.haproxy.org (76 kB) shows the
following measures with traffic leaving the machine via a gigabit Ethernet link
(hence the 976 Mbps of IP traffic) after being compressed with zlib and SLZ,
respectively :

At 100% CPU (no CPU usage limit) :
|         |   BW In    |  BW Out   |  BW Saved |  Ratio |  memory VSZ/RSS | hits/s |
|:--------|:----------:|:---------:|:---------:|:------:|:---------------:|-------:|
|  zlib   |   480 Mbps |  188 Mbps |  292 Mbps |  2.55  |   130M / 101M   |  744   |
|  slz    |  1700 Mbps |  810 Mbps |  890 Mbps |  2.10  |  23.7M / 9.7M   | 2382   |

At 85% CPU (limited) :
|         |   BW In    |  BW Out   |  BW Saved |  Ratio |  memory VSZ/RSS | hits/s |
|:--------|:----------:|:---------:|:---------:|:------:|:---------------:|-------:|
|  zlib   |  1240 Mbps |  976 Mbps |  264 Mbps |  1.27  |   130M / 100M   | 1738   |
|  slz    |  1600 Mbps |  976 Mbps |  624 Mbps |  1.64  |  23.7M / 9.7M   | 2210   |

In the CPU-limited test, once CPU usage limit was reached, blocks were sent
uncompressed. This explains how the gigabit link was always saturated, and it
is then the input bandwidth which varies according to the average compression
ratio. In all situations, SLZ resulted in bandwidth savings 3 times higher than
zlib.

## No magics inside!

The "slz" utility (formerly named "zenc"), which was orginally aimed at running
regression tests, found its way into backup systems since it permits to reach
network links limits with compressed streams before saturating the CPU. This
results in a shorter total backup time, while relying on a well-known and
standard output compression format. Such use cases could benefit from a
multithreaded version that would work on different blocks and adapt the running
CRC once blocks are finished, but given that a not-so-fast x86 system easily
saturates a gigabit link using a single core, there's not much demand for this
at the moment.

It is now used mostly like gzip. Given file names, it compresses each file in
place to the same name followed by the format's extension (".gz" for -G, the
default, ".zz" for -Z, ".dfl" for -D), which receives the file's mode, owner
and times, then removes the file. The file's name is not stored in the output.

"slz -d" decompresses files in place the same way, to the file's name without
its extension, whichever it is. One of -G, -Z or -D forces the format.
Otherwise the gzip and zlib formats are detected from the stream, but raw
deflate cannot be, so the ".dfl" extension selects it. A file without
extension can only be decompressed to stdout (-c). Without file name, it works
from stdin to stdout. Its main options are:

```
  -1..-9      compression level (1=fastest, 4 and above=best, 2 by default),
              "-0" only emits the format without compressing
  -c          write to stdout and keep the files (the default without file)
  -d          decompress
  -f          overwrite existing outputs, compress symbolic links and files
              which already carry the extension, write compressed data to a
              terminal
  -k          keep the input files
  -t          test mode: process the inputs without emitting anything; with
              -d, the exit status tells whether they could be decompressed
  -v          report the sizes, ratio and speed
  -K <size>   block size (4 MB by default), or the output ring size with -d
              (32 kB by default and at least), in kB or with a b/k/m suffix
  -m          memory mode: read the input into a buffer of the block size
              instead of mapping it; it is the default for pipes and other
              inputs which cannot be mapped
```

For convenience, the program recognizes names such as "zdec", "zcat", "gzcat"
and "slzcat" as equivalent to "slz -dc", and "gunzip" and "unslz" as equivalent
to "slz -d". Symbolic links will make this possible if needed. By default, only
"zenc" and "zdec" are provided.

An output is only created once its input could be opened, an existing one is
only replaced with -f, and on error or interruption the output is removed and
the input kept. For benchmarks, -l `<loops>` repeats the processing of each
input and -p pre-heats the input and output buffers, both in test mode only. A
development build ("make BUILD=dev") offers debugging features, such as -i to
set the size of the chunks fed to the decompressor (e.g. "-i 1b" for one byte
per call).

There are 6 key points having a large impact on compression speed in any LZ-
based compressor :

- dictionary lookup : looking up a sequence of bytes in a recent window is not
  necessarily fast. SLZ uses a hash-based technique similar to LZO or LZ4,
  focusing on keeping track of only the most recent occurrence of a given
  sequence. The hash table contains the absolute position in the stream of the
  last occurrence of the hashed entry. This table is initialized to impossible
  values prior to compressing so that unused entries are not even evaluated, as
  it was found to be faster than not initializing them. During tests, it was
  found that hashing 4 bytes to one value among 4096-65536 resulted in the best
  compression ratio, and that 8192 entries was the best tradeoff between speed
  and compression ratio. Smaller hash tables cause more collisions hurting the
  compression ratio, and larger ones experience a significant slowdown on
  certain datasets due to the hashtable having more difficulties to fit
  entirely in the CPU cache. The size remains configurable at build time
  using SLZ_HASH_BITS.

- indexing : if sequences of less than 3 bytes are considered for a lookup,
  since we only keep the last occurrence, there's a high risk of not keeping
  the most useful sequence which has large changes of matching a similar word.
  For example in HTML a lot of occurrences of `><`, `</`, `/>` can be found
  so the probability that the one from `</a>` is picked instead of the one from
  `</td>` will make it hard to find long matches. Conversely, sequences larger
  than 4 bytes will only match once in a while and will prevent shorter series
  of characters from being matched. Experimentation has shown that hashing 3
  bytes only provided the same compression ratio as hashing 4 bytes, with up to
  10% faster input processing thanks to the reduced cost of hashing. Indexing
  all the repeated data caused a significant performance drop (about 5%) for
  less than 1% compression ratio improvement, so only the beginning of each
  match is indexed. When a match is found, SLZ also checks whether it extends
  backwards into the pending literals, which recovers bytes that were missed
  without having to index more.

- longest match : the longest match measurement is very similar to an strcmp()
  call except that we count similar characters. In order to avoid this slow
  operation most of the time, we store in the hash table a copy of the 4 bytes
  that were hashed. This makes it easy to perform a comparison and to skip
  those which do not match, then to only start the longest match measurement
  for parts which already match on the first 4 bytes. With a 13-bits hash on
  32-bit character sequence, there's a theorical 1 chance over 524288 that a
  matching hash implies a matching sequence. But in practice on selected text
  this chance is significantly higher since not all 32 bits are used, it's more
  like 24 bits (6 bits per char), so it's one chance over 2048 that an entry
  will match. That's still 2047/2048 useless calls to the memmatch function
  that are saved this way. The match itself is then measured one machine word
  at a time, the first differing byte being located by counting the trailing
  zeroes of the XOR of the two words.

- encoding the output : deflate is often criticized for its bit-oriented stream
  which is expensive to manipulate as it requires bit shifts. Moreover, on top
  of this the deflate stream supports several encoding modes, the most common
  one being dynamic huffman trees, which requires some significant maintenance
  to keep a valid tree up to date and which SLZ doesn't use. Instead, SLZ uses
  static huffman trees, and when too many binary entries are found to keep this
  encoding efficient, it switches to the plain literal mode. Proceeding like
  this comes with a significant benefit : all code sequences, all distances and
  all lengths are stored as their respective huffman output bits. This avoids
  an encoding phase, sequences of bits are immediately sent to the destination
  buffer without any further operation, and remain reasonably cheap compared to
  the more common usages. On 64-bit machines, the bits are accumulated in a
  machine word and a whole match (length and distance) is written with a single
  unaligned store. Long runs of literals that do not match anything are skipped
  and sent as plain literals without being looked up (at levels 1 to 3).

- checksumming : 3 checksum methods exist depending on the encoding format.
  RFC1951 (raw deflate) doesn't involve any checksum. RFC1952 (gzip) uses a
  CRC-32 which is somewhat expensive but can be optimized in various ways.
  On x86 CPUs with the carry-less multiplication instruction (PCLMULQDQ),
  detected at boot time, SLZ computes it by folding 16 bytes at a time, which
  makes it almost free. On ARMv8, the CPU's CRC-32 instructions are used when
  available. Other machines use pre-computed tables processing 4 bytes at a
  time. RFC1950 (zlib format) uses an Adler-32 checksum which theorically
  involves modulus (divide) but which can be optimized as it's done with a
  number very close to a power of two. This format matters more than it seems,
  because it is what browsers call "deflate". Its two sums form a long
  dependency chain which is hard to speed up, but on machines with SIMD units
  (SSE2 on x86, NEON on ARM), SLZ splits the input into 32-byte blocks and sums
  each byte offset of the blocks independently. This is written in plain C that
  the compiler vectorizes, and made the zlib format 12 to 23% faster depending
  on the CPU. It is now about as fast as gzip on x86 and faster than gzip on
  ARM. The checksum is computed over all the input at once before compressing,
  which also contributes to preloading the input data into the caches.

- implementation-specific optimizations : modern CPUs can load unaligned words
  from memory with minimal to no overhead. On architectures that support this
  (ix86, x86_64, armv7, armv8), this is used to limit the number of operations
  and memory accesses, both to compare data and to write the output. The bit
  reversal needed by deflate's huffman codes uses the dedicated instruction on
  ARM. Some CPUs have a smaller cache and the direct mapping between distances
  and huffman sequences can make it thrash a lot. Some experimentations were
  made using a direct mapping only for shortest distances (the most common
  ones), but results were not encouraging for now as a cache miss is not
  completely offset by the amount of extra operations.


These three factors have a significant impact on the compression ratio :

- collisions in the dictionary, caused by suboptimal hash distribution. Here
  with 8k entries we have what looks like the best tradeoff between speed and
  compression ratio for most workloads tested, and it can easily be changed
  using a macro. This impact is dampened a bit by backward lookup upon a match,
  based on the estimate that if a collision happens on the first byte, it will
  not necessarily happen on the next one and if the match is 5 bytes or more,
  it can still be found.

- output encoding : the dynamic huffman trees would definitely shorten the
  output stream but at an important performance cost. But even with fixed trees
  it is possible to waste a lot of space. First, input bytes 144 to 255 are
  encoded on 9 bits per byte. SLZ constantly monitors the output stream to
  decide when it's preferable to switch to plain literals which cost exactly
  52 bits to switch back and forth and will not inflate the output stream any
  further. Second, referencing a match can be longer than the data it replaces.
  This is true for short sequences, but it can be true even for a long sequence
  in the middle of a series of plain literals. By monitoring the output stream
  and verifying the encoding cost before switching, it is possible to limit the
  overhead to the base 5 bytes per 65535 bytes plus headers for non-compressible
  data, plus at most SLZ_MAX_DEBT_BYTES (40 bytes, see slz.h) by which the
  huffman encoding may lag behind before SLZ falls back to plain literals.
  SLZ_MAX_OVERHEAD(len) in slz.h gives the room each call needs on top of its
  input length.

- content type : most of the contents compressed by HTTP servers and gateways
  is text (HTML, CSS, JS, JSON, SVG...), on which zlib's default fixed huffman
  tree is far from optimal, as it encodes most letters on 8 bits and match
  lengths on 7 bits or more. Since 1.4, an application knowing that it
  compresses such contents can call slz_set_content_type(strm, SLZ_CT_TEXT)
  after initializing the stream. Instead of the fixed tree, SLZ then uses a
  static tree that was trained on web contents, declared at the beginning of
  each block in about 80 bytes. It remains as cheap to use as the fixed one, at
  the expense of a few percent of speed, and shrinks the output of such
  contents by 12 to 18%. It is not suited to binary contents nor to contents of
  less than 1-2 kB.

## Get it!

SLZ is provided as a library with a few extra tools, such as the slz utility
mentioned above, whose decompressor, based on uslz, decompresses any gzip, zlib
or raw deflate stream, not only those produced by SLZ, and detects the gzip and
zlib formats by itself. It is distributed under the X11 license, meaning that
you can do a lot of things with it, such as merge it into GPL or BSD-licensed
programs, as well as use it in proprietary software. Contributions are welcome
and should be sent as Git patches (see "git format-patch") and will be made
under the same license exclusively.

Project's webpage : http://www.libslz.org/  
Download sources  : https://github.com/wtarreau/libslz/  
Contact           : Willy Tarreau <w+slz@1wt.eu>  
