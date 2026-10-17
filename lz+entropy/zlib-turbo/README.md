# zlib-turbo

[![zlib-turbo CI](https://github.com/icodywu/zlib-turbo/actions/workflows/turbo.yml/badge.svg)](https://github.com/icodywu/zlib-turbo/actions/workflows/turbo.yml)

**zlib, faster, and byte for byte the same.**

zlib-turbo is zlib 1.3.2 with faster decompression and compression. It has the same API, the same ABI and the
same headers (`zlib.h` is unchanged), and it produces the same compressed bytes as zlib for every input at every
setting zlib has. It decompresses every stream, valid or damaged, with the same output and the same errors. You can
put it in place of zlib in a program, a package or a whole system, and nothing else changes except the speed.

| AMD EPYC 9334, GCC 14 -O2, one core unless noted | zlib 1.3.2 | zlib-turbo | |
|---|---:|---:|---:|
| Decompression, Silesia (level 6 streams) | 438 MB/s | 739 MB/s | **+69%** |
| Decompression, enwik8 (level 6 streams) | 349 MB/s | 621 MB/s | **+78%** |
| Compression, Silesia, level 1 | 114 MB/s | 151 MB/s | **+32%** |
| Compression, Silesia, level 6 | 31.9 MB/s | 36.0 MB/s | **+13%** |
| Compression, Silesia, level 6, 16 threads | 31.9 MB/s | 218 MB/s | **6.8×** |
| Smallest output, Silesia (zlib level 9, zlib-turbo level 12) | 67.64 MB | 64.58 MB | **-4.5%** |
| Decompression, Silesia gzip streams, optional SIMD checksums | 418 MB/s | 806 MB/s | **+93%** |
| Compressed output, levels 0 to 9 | | | **identical** |

zlib-turbo also adds three optional features: multithreaded compression, with output still identical to zlib's;
prepared dictionaries, which make compressing many small messages with a dictionary up to four times as fast (see
below); and levels 10 to 12, which compress smaller than level 9 with an optimal parse (their output is valid
deflate data but not zlib's, since no zlib level has it). Programs that do not use them see plain zlib. A build
option adds SIMD checksums for x86 processors with AVX2, off by default.

## What is new

zlib-turbo is more than known speedups applied to zlib. It contributes:

1. **Speed without changing a byte, as a guarantee.** Every change at levels 0 to 9 alters how zlib does its work,
   never what it decides: the same matches in the same order, the same blocks, the same bits, the same return
   codes. The guarantee is checked against unmodified zlib 1.3.2 rather than against expected outputs: three
   corpora, nearly 35,000 damaged streams, about 80 million differential fuzzing runs, six processor architectures
   and MSVC (see [How it was tested](#how-it-was-tested)). Faster zlib-compatible libraries give it up; zlib-turbo
   shows how much speed there is without giving it up: 66% to 86% in decompression and 4% to 39% in compression on
   one core, and up to 12 times as fast with threads.

2. **Multithreaded compression whose output is zlib's.** At levels 4 to 9 zlib's hash chains depend only on the
   input, so the matches its lazy parse will ask for can be found ahead of it. Each worker thread replays zlib's
   parse over its chunk of input, starting a little before it, so that it searches exactly the positions zlib will
   search, with the chain length zlib will use. The calling thread keeps every decision of zlib's parse and takes
   99.99% of its searches from the workers, and finished blocks are written on workers only where zlib would write
   them too, so even the return codes and pending output match. With 16 threads, level 6 runs 6.8 times as fast as
   zlib 1.3.2 and level 9 up to 12 times, for the same bytes. We know of no other parallel deflate compressor whose
   output equals single-threaded zlib's: pigz, the usual parallel gzip, compresses separate pieces, so its bytes
   differ.

3. **Levels 10 to 12: smaller than libdeflate's best, in zlib's own API.** An optimal-parsing compressor written
   for zlib-turbo: a binary-tree match finder kept from segment to segment, a parse iterated over exact bit costs,
   block boundaries by dynamic programming, and the shortest coding of each block's trees. Level 12 is 0.15% to
   0.33% smaller than libdeflate's level 12 on Silesia, enwik8 and Canterbury + Calgary, 4.0% to 5.7% smaller than
   zlib's level 9, and within 0.2% of Zopfli at 6 to 11 times its speed. Any program that can ask zlib for a level
   can ask for these.

4. **Prepared dictionaries.** zlib indexes a dictionary again for every stream. `deflatePrepareDictionary()`
   indexes it once, and `deflateUseDictionary()` copies the index into each stream, giving exactly the state, and so
   the output, that `deflateSetDictionary()` gives. Small messages compress up to 4.1 times as fast.

5. **A faster decoder and faster checksums.** A 64-bit fast loop with branch-free refills, larger first-level tables
   and wide match copies decompresses 66% to 86% faster, with zlib's results for every valid and damaged stream; it
   is offered back to zlib as [pull request #793](https://github.com/madler/zlib/pull/793). A portable SWAR
   Adler-32, three multiplications per eight bytes, makes the zlib format's checksum 43% faster. An optional build
   adds SIMD checksums for x86, with the same results bit for bit: CRC-32 by carry-less multiplication, 5.5 times as
   fast, and Adler-32 with AVX2, 7.4 times. With them, gzip streams decompress 93% faster than with zlib 1.3.2.

6. **Every number measured and reproducible.** Each change's gain is measured on its own, the ideas that did not
   pay are listed with their measurements in [`turbo/TECHNIQUES.md`](turbo/TECHNIQUES.md), and the scripts in
   [`turbo/`](turbo) repeat every comparison against any zlib source tree.

## Why the same bytes matter

Faster zlib-compatible libraries exist; [zlib-ng](https://github.com/zlib-ng/zlib-ng) is the best known, and on one
core it is faster than zlib-turbo. But they compress differently: the output is valid, yet not the bytes zlib
would write. That breaks anything that hashes or compares compressed data:

- In January 2023, a change of compressor in git made the source archives that GitHub serves differ by a few bytes.
  Every published checksum stopped matching, and builds that verify downloads broke across Homebrew, Bazel and many
  other projects until the change was rolled back.
- Container image layers are identified by the hash of their compressed bytes; a different compressor means new
  digests, missed caches and new uploads.
- Reproducible builds, package checksums and tests with stored compressed files all compare bytes.

zlib-turbo keeps those guarantees: every change speeds up how zlib does its work, never what it decides. The same
matches are found in the same order, the same blocks are emitted, and the same bits are written.

## Speed

Silesia, Canterbury + Calgary and enwik8 (the first 10^8 bytes of enwik9), each file whole; one core of an AMD EPYC
9334 (Zen 4, 2.7 GHz), GCC 14.2 `-O2`, best of two runs. MB/s of uncompressed data.

**Compression** (compressed bytes identical to zlib's in every case)

| Level | Silesia | Canterbury + Calgary | enwik8 |
|---|---|---|---|
| 1 | 114 → 151 (+32%) | 127 → 176 (+39%) | 99 → 138 (+39%) |
| 3 | 75 → 91 (+22%) | 72 → 90 (+24%) | 60 → 75 (+24%) |
| 6 | 31.9 → 36.0 (+13%) | 26.7 → 29.9 (+12%) | 24.4 → 27.2 (+12%) |
| 9 | 13.1 → 14.2 (+9%) | 8.3 → 8.6 (+4%) | 18.9 → 20.7 (+9%) |

**Decompression** of zlib streams made by zlib at each level, in one `inflate()` call / in 16 KiB input and 64 KiB
output pieces

| Level | Silesia | Canterbury + Calgary | enwik8 |
|---|---|---|---|
| 1 | 412 → 681 (+66%) / 385 → 592 (+54%) | 447 → 766 (+71%) / 419 → 659 (+57%) | 343 → 603 (+76%) / 318 → 496 (+56%) |
| 6 | 438 → 739 (+69%) / 409 → 649 (+59%) | 468 → 873 (+86%) / 437 → 755 (+73%) | 349 → 621 (+78%) / 325 → 528 (+62%) |
| 9 | 442 → 744 (+68%) / 413 → 653 (+58%) | 474 → 881 (+86%) / 441 → 762 (+73%) | 350 → 621 (+77%) / 325 → 528 (+62%) |

Other compilers, Silesia level 6, compression / decompression in one call: GCC 11 +11% / +77%, clang 18 +12% /
+62%. On 32-bit x86, where 64-bit arithmetic is emulated, +5% / +8%.

For context, zlib-ng 2.3.3 (compat mode, same machine and compiler) decompresses 15% to 20% faster than zlib-turbo,
which reaches 82% to 86% of its speed without any SIMD code (the default build). zlib-ng compresses two to three times as fast, with
different output; at level 1 its files are 18% to 22% larger than zlib's. All runs:
[`turbo/results/speed_epyc9334.txt`](turbo/results/speed_epyc9334.txt).

## What changed

Nine changes to zlib 1.3.2:

1. **Decompression: a 64-bit bit buffer and wide copies** (`inffast.c`). A new fast loop refills a 64-bit bit
   buffer eight bytes at a time, once per length/distance pair or per up to three literals, and copies matches 16
   or 8 bytes at a time. Larger first-level decoding tables (10 bits for literals and lengths, 8 for distances)
   make second lookups rare. The existing loop finishes near the buffer ends exactly as before.
2. **Adler-32 eight bytes at a time** (`adler32.c`) on 64-bit targets: one multiplication sums eight bytes and two
   weight them, with a modulo only every 2^20 bytes. This speeds up both directions for the zlib format.
3. **Match finding: two- and eight-byte comparisons** (`deflate.c`). Candidates are checked two bytes at a time and
   compared eight bytes at a time. The same candidates are tried, in the same order, with the same lengths found.
4. **A 64-bit bit writer** (`trees.c`) for the compressed data of each block.
5. **Hash tables slid eight entries at a time** (`deflate.c`), so that compilers vectorize it.
6. **Prepared dictionaries** (`deflate.c`, new header `zturbo.h`), below.
7. **Multithreaded compression** (`deflate.c`, `trees.c`, `zturbo.h`), below.
8. **Levels 10 to 12** (`deflate.c`, `zturbo.h`), below: a compressor of their own with an optimal parse, 4% to 6%
   smaller than level 9 and smaller than libdeflate's level 12, and slower. Their output is not zlib's; levels 0 to
   9 are unchanged.
9. **SIMD checksums, optional** (`crc32.c`, `adler32.c`), below: CRC-32 by carry-less multiplication and Adler-32
   with AVX2, in builds that ask for them.

How each change works, why the output stays the same, what each one gains, and what was tried and dropped:
[`turbo/TECHNIQUES.md`](turbo/TECHNIQUES.md).

Everything is portable C89 by default: no SIMD code, no assembly and no run-time CPU detection, so the same source
builds wherever zlib builds, and compilers that target newer CPUs simply generate better code from it. The one
exception is opt-in: SIMD checksums for x86, chosen at build time, below. Threads are POSIX threads, or Windows'
own.

## Multithreaded compression

`deflateSetThreads()` puts worker threads on a stream. At levels 4 to 9 they search for matches ahead of the
parse and write the finished blocks, and the compressed output is still zlib's, byte for byte, for any number of
threads, any sizes of input and output per call, flushes, `deflateParams()` and dictionaries:

```c
#include "zturbo.h"

deflateInit(&strm, 6);
deflateSetThreads(&strm, 16);   /* 0 or 1: back to one thread */
/* deflate() as usual */
```

| Silesia / enwik8, MB/s | 1 thread | 4 threads | 8 threads | 16 threads | 24 threads |
|---|---:|---:|---:|---:|---:|
| Level 4 | 85.8 / 70.7 | 214 / 205 | 223 / 212 | 232 / 219 | 211 / 211 |
| Level 6 | 35.8 / 27.1 | 126 / 103 | 202 / 190 | **218 / 210** | 198 / 206 |
| Level 9 | 14.1 / 20.7 | 51 / 78 | 92 / 155 | 148 / **223** | **160** / 211 |

Each file compressed whole in one `deflate()` call, worker threads on one socket of a 2 × 32-core AMD EPYC 9334, GCC
14 -O2, best of two runs. zlib 1.3.2 itself does 31.9 / 24.4 MB/s at level 6 and 13.1 / 18.9 at level 9. With 16
threads that is 6.8 and 8.6 times zlib at level 6, and 11 to 12 times at level 9. The calling thread still chooses
every match, in zlib's order, so the speed levels off near 220 MB/s; at level 9 on Silesia the workers are the
limit. Levels 1 to 3, `Z_HUFFMAN_ONLY` and `Z_RLE` stay on one thread: their matches depend on the matches before
them, so they cannot be found ahead with the same result.

Two rules come with the threads, both in `zturbo.h`. When `deflate()` returns with input left, the next call must
be given the same bytes (they may be at another address), since the workers read ahead. The workers write a block
only when the output space left for that call can take all of it, so give `deflate()` 256 KiB of output space or
more per call for the speeds above. Each worker takes about 4 MB. All runs:
[`turbo/results/threads_epyc9334.txt`](turbo/results/threads_epyc9334.txt).

## Prepared dictionaries

`deflateSetDictionary()` indexes its dictionary again for every stream, up to 32K hash insertions before the first
byte of input. When many short messages are compressed with the same dictionary (HTTP or RPC payloads, log records,
database rows), that indexing is most of the work. A prepared dictionary is indexed once and then shared, read-only,
by any number of streams and threads:

```c
#include "zturbo.h"

z_dict *dict;
deflatePrepareDictionary(&first, dictionary, dictionary_length, &dict); /* on a just-initialized stream */
...
deflateReset(&strm);
deflateUseDictionary(&strm, dict);  /* same effect and output as deflateSetDictionary() */
...
deflateFreeDictionary(dict);
```

`deflateUseDictionary()` gives a stream exactly the state `deflateSetDictionary()` would: the same return value,
`total_in` and Adler-32, and the same compressed output, byte for byte. On a stream it cannot copy into (other
`windowBits` or `memLevel`, or not just initialized or reset), it calls `deflateSetDictionary()`.

| Message size | Level | zlib 1.3.2 | zlib-turbo, prepared dictionary | |
|---|---|---:|---:|---:|
| 256 B | 1 | 44.2 µs | 10.8 µs | **4.1×** |
| 1 KiB | 1 | 55.7 µs | 20.5 µs | **2.7×** |
| 4 KiB | 1 | 89.3 µs | 47.8 µs | **1.9×** |
| 256 B | 6 | 53.4 µs | 19.5 µs | **2.7×** |
| 1 KiB | 6 | 91.7 µs | 55.4 µs | **1.7×** |

Time per message, each compressed as its own zlib stream with a 32 KiB dictionary; messages from enwik8, one core of
an AMD EPYC 9334, GCC 14. JSON records from a DCLM-baseline shard give the same picture. All runs:
[`turbo/results/dictionary_epyc9334.txt`](turbo/results/dictionary_epyc9334.txt).

These functions are declared in `zturbo.h`, not `zlib.h`, and exported under their own version tag
(`ZLIB_TURBO_1.0`), so programs built for zlib see no difference. `zlibTurboVersion()` tells a program that it is
running with zlib-turbo; `zlibVersion()` still returns the zlib version it is based on.

## Levels 10 to 12

Levels 10, 11 and 12 compress smaller than level 9, and smaller than libdeflate's level 12, at a few MB/s. They are
a compressor of their own, so their output is not zlib's (no zlib level has it), but it is ordinary deflate data that
every inflater decompresses, and it is the same for the same input, flushes and parameters, however the input is
fed. Ask for them as for any level:

```c
deflateInit2(&strm, 12, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY);   /* or compress2(dest, &len, src, n, 12) */
```

| Raw deflate bytes, MB/s | Silesia | enwik8 | Canterbury + Calgary |
|---|---:|---:|---:|
| zlib level 9 | 67,643,201 · 14.2 | 36,475,786 · 20.7 | 1,783,551 · 8.4 |
| libdeflate 1.26 level 9 | 66,715,085 · 35.8 | 36,244,245 · 39.2 | 1,739,956 · 27.4 |
| libdeflate 1.26 level 12 | 64,681,628 · 6.5 | 35,103,237 · 8.5 | 1,688,340 · 6.3 |
| Zopfli 1.0.3 (15 iterations) | 64,713,006 · 0.29 | **34,966,060** · 0.65 | **1,680,016** · 0.32 |
| zlib-turbo level 10 | 64,681,882 · 5.5 | 35,016,775 · 5.6 | 1,687,697 · 5.5 |
| zlib-turbo level 11 | 64,613,145 · 4.4 | 35,012,093 · 5.1 | 1,684,734 · 4.4 |
| zlib-turbo level 12 | **64,584,749** · 3.2 | 35,007,174 · 3.9 | 1,682,786 · 3.1 |

Each file compressed whole on one core of an AMD EPYC 9334, GCC 14 -O2, best time of two runs. Level 12 is 0.15%
to 0.33% smaller than libdeflate's level 12, at half its speed, and 4.0% to 5.7% smaller than level 9. Against
Zopfli it is 0.2% smaller on Silesia and 0.1% to 0.2% larger on the others, at 6 to 11 times its speed. Level 10
is as small as libdeflate's level 12 on Silesia. All runs, per file:
[`turbo/results/levels10-12_epyc9334.txt`](turbo/results/levels10-12_epyc9334.txt).

How they work: the input is parsed 256 KiB at a time, with the 32 KiB before it. A binary tree of the window's
strings gives every match at every position; a parse that finds the cheapest path for given bit costs runs
several times, first with the costs of a lazy parse's statistics, then with each block's own Huffman codes; block
boundaries come from dynamic programming over an entropy estimate, and each block is written dynamic, fixed or
stored, whichever is smallest. Details: [`turbo/TECHNIQUES.md`](turbo/TECHNIQUES.md#8-levels-10-to-12).

They take about 24 MB per stream and keep up to 256 KiB of input before they compress it (a flush compresses what
is there). A stream can change among levels 10 to 12 at any time, but between them and levels 0 to 9 only before
its first `deflate()` call. Threads do not speed them up yet. `Z_BEST_COMPRESSION` is still 9.

## SIMD checksums (optional)

With `./configure --simd` or CMake `-DZLIB_TURBO_SIMD=ON` (or `-DZ_TURBO_SIMD` with `-mpclmul -mavx2`,
`-march=x86-64-v3` or MSVC's `/arch:AVX2`), CRC-32 folds 64 bytes at a time by carry-less multiplication
(PCLMULQDQ) and Adler-32 sums 32 bytes at a time with AVX2. The results are the scalar code's, bit for bit. There
is no check at run time: such a library runs only on processors with AVX2 and PCLMULQDQ (Intel since 2013, AMD
since 2015). The default build is plain C as before.

| AMD EPYC 9334, GCC 14 -O2, one core | zlib 1.3.2 | zlib-turbo | zlib-turbo, SIMD | |
|---|---:|---:|---:|---:|
| CRC-32 | 2.77 GB/s | 2.77 GB/s | 15.3 GB/s | **5.5×** |
| Adler-32 | 3.87 GB/s | 5.52 GB/s | 41.1 GB/s | **7.4×** |
| Decompression, gzip streams, Silesia / enwik8 | 418 / 336 MB/s | 650 / 555 | 806 / 667 | **+24% / +20%** |
| Decompression, zlib streams, Silesia / enwik8 | 437 / 349 MB/s | 736 / 619 | 834 / 685 | **+13% / +11%** |
| Compression, gzip, level 1, Silesia | 113 MB/s | 147 | 154 | **+5%** |

Level 6 streams, each file whole in one call; the last column is SIMD over zlib-turbo without it. In 16 KiB input
and 64 KiB output pieces decompression gains 17% to 24% (gzip) and 9% to 14% (zlib), and compression gains 1% to
5%, with the same output. All runs: [`turbo/results/simd_epyc9334.txt`](turbo/results/simd_epyc9334.txt).

## How it was tested

Every test compares zlib-turbo with unmodified zlib 1.3.2, not only with expected output.

| Test | What it checks |
|---|---|
| Identical output | All files of the three corpora at levels 0 to 9: the same compressed bytes. |
| Damaged streams | 34,860 corrupted streams (bit flips, overwritten bytes, truncations) in random input and output pieces: the same return code, output length and CRC as zlib for each. |
| Valid streams | 2,809 checks over 11 compression settings, in one call, in random pieces each in a buffer of its own size, and with `inflateBack()`. |
| Memory checkers | All of the above under AddressSanitizer and UndefinedBehaviorSanitizer, also built with `INFLATE_STRICT`, `INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR` and `PKZIP_BUG_WORKAROUND`. |
| Architectures | The same results on x86-64, 32-bit x86, arm64, ARMv5, ppc64le and big-endian s390x (QEMU), also with the eight-byte Adler-32 forced on where it is off by default, and with MSVC 2022 on Windows. |
| Differential fuzzing | libFuzzer against zlib 1.3.2. Decompression: 36 million runs under AddressSanitizer and 35 million under UndefinedBehaviorSanitizer. Compression, with all flush modes, `deflateParams()`, dictionaries set or prepared, `deflatePrime()` and `deflateReset()`: 5.7 million runs under AddressSanitizer and 2.8 million under UndefinedBehaviorSanitizer. No difference and no report. |
| Dictionaries | 9,315 combinations of level, window, `memLevel`, strategy and dictionary length per test file: the same results from zlib's `deflateSetDictionary()` and from zlib-turbo's `deflateSetDictionary()` and `deflateUseDictionary()`. |
| Streaming | With `Z_SYNC_FLUSH`, every byte before a flush point comes out as soon as its compressed bytes are in (43,815 messages). |
| Threads | Levels 4 to 9, 23 settings each (input and output in pieces down to one byte, sync and full flushes, `deflateParams()` to other levels, strategies and back, raw and zlib dictionaries, input moved and overwritten between calls) on 12 files, with 3 to 16 threads: every output byte and return code the same as with one thread, 1,656 cases in all, with assertions on. Also under ThreadSanitizer (clang and GCC), AddressSanitizer and UndefinedBehaviorSanitizer, on 32-bit x86, and with MSVC and Windows threads. |
| Levels 10 to 12 | 23 settings per level (input and output in pieces down to one byte, flushes every 31 to 131,072 bytes, window sizes, `memLevel`, raw, zlib and gzip formats, strategies, dictionaries, `deflateCopy()`, `deflateParams()` among the levels) on five files of the corpora, also under AddressSanitizer and UndefinedBehaviorSanitizer: every stream decompresses to its input, also with stock zlib, and the same bytes come out however the input is fed. libFuzzer against zlib 1.3.2's inflate, with the stream fitting in `deflateBound()`: 409,943 runs under AddressSanitizer and 223,448 under UndefinedBehaviorSanitizer, no report. |
| SIMD checksums | CRC-32 and Adler-32 against byte-at-a-time references for every length 0 to 4096 at every alignment 0 to 63, long buffers of random bytes and of all ones (the largest Adler-32 sums) in one call and in pieces, and `crc32_combine()` and `adler32_combine()`: 527,572 cases, the same with GCC 14 (also `-march=x86-64-v3`), clang 18, 32-bit x86, MSVC and under AddressSanitizer and UndefinedBehaviorSanitizer. With SIMD on: identical compressed output at levels 0 to 9, the damaged-stream tests, `make test` and ctest. |
| zlib's own tests | `make test`, also in a `ZLIB_DEBUG` build; zlib's `test/infcover.c` with new cases for the fast loop. |

## Building and installing

Exactly as zlib: `./configure && make test && make install`, or CMake, or the files in `win32/`. The library has
zlib's names (`libz.so.1`, `zlib1.dll`) and installs `zturbo.h` next to `zlib.h`. It is built with threads, and so
linked with `-pthread` where that is needed; `./configure --no-threads` or CMake `-DZLIB_TURBO_THREADS=OFF` builds
it without, and `deflateSetThreads()` then accepts only one thread. The files in `win32/` build without threads
unless `-DZ_TURBO_THREADS` is added. `./configure --simd` or CMake `-DZLIB_TURBO_SIMD=ON` adds the SIMD checksums
(x86 with AVX2 and PCLMULQDQ only); with the files in `win32/`, add `-DZ_TURBO_SIMD -arch:AVX2`.

## Reproducing the results

The scripts in [`turbo/`](turbo) compare any two zlib source trees:

```sh
git clone https://github.com/madler/zlib zlib-1.3.2 && git -C zlib-1.3.2 checkout v1.3.2
sh turbo/tests/run_compress.sh zlib-1.3.2 . silesia/      # identical output and compression speed, levels 0 to 9
sh turbo/bench/run_speed.sh zlib-1.3.2 . silesia/ > speed.txt && python3 turbo/bench/summarize.py speed.txt
sh turbo/tests/run_stress.sh zlib-1.3.2 . inputs/         # damaged streams, under the sanitizers
sh turbo/tests/run_dict.sh zlib-1.3.2 . silesia/xml      # prepared dictionaries
sh turbo/tests/run_threads.sh . 8 silesia/*               # threads: the same bytes as one thread
sh turbo/tests/run_opt.sh zlib-1.3.2 . silesia/*           # levels 10 to 12: round trips, also through zlib
cc -O2 -DZ_TURBO_SIMD -mpclmul -mavx2 -I. adler32.c crc32.c zutil.c turbo/tests/zsum.c -o zsum && ./zsum check
TIME=3600 sh turbo/tests/fuzz/run_fuzz.sh zlib-1.3.2 . inputs/          # decompression, differential
TIME=3600 sh turbo/tests/fuzz/run_fuzz_deflate.sh zlib-1.3.2 . inputs/  # compression, differential
```

## Relation to zlib

zlib-turbo is an altered version of zlib 1.3.2 by Jean-loup Gailly and Mark Adler, plainly marked as such as zlib's
license requires; it is not the original zlib and is not endorsed by its authors. It follows zlib's releases. The
decompression change has also been offered to zlib as [pull request #793](https://github.com/madler/zlib/pull/793).
zlib-turbo is not related to [bitbank2/zlib_turbo](https://github.com/bitbank2/zlib_turbo), an inflate library for
microcontrollers.

## License

The [zlib license](LICENSE), as zlib. zlib is Copyright (C) 1995-2026 Jean-loup Gailly and Mark Adler; the changes
in zlib-turbo are Copyright (C) 2026 Yingquan (Cody) Wu.
