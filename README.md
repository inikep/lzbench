Introduction
-------------------------

lzbench is an in-memory benchmarking tool for open-source compressors. It integrates all compressors into a single executable. Initially, an input file is loaded into memory, after which each compressor is used to compress and decompress the file, ensuring the decompressed output matches the original. This method provides the advantage of compiling all compressors with the same compiler and optimizations. However, it requires access to the source code of each compressor, meaning e.g. Slug and lzturbo are not included.


|Status   |
|---------|
| [![Build Status][AzurePipelinesMasterBadge]][AzurePipelinesLink] |

[AzurePipelinesMasterBadge]: https://dev.azure.com/inikep/lzbench/_apis/build/status%2Finikep.lzbench?branchName=master "gcc and clang tests"
[AzurePipelinesLink]: https://dev.azure.com/inikep/lzbench/_build/latest?definitionId=15&branchName=master

The list of changes in lzbench is available in the [CHANGELOG](CHANGELOG).

Contributor information can be found in [CONTRIBUTING.md](CONTRIBUTING.md).


Usage
-------------------------

```
usage: lzbench [options] input [input2] [input3]

For example:
  lzbench -ezstd filename = selects all levels of zstd
  lzbench -ebrotli,2,5/zstd filename = selects levels 2 & 5 of brotli and zstd
  lzbench -t3,5 fname = 3 sec compression and 5 sec decompression loops
  lzbench -t0,0 -i3,5 fname = 3 compression and 5 decompression iterations
  lzbench -o1c4 fname = output markdown format and sort by 4th column
  lzbench -ezlib -j -r dirname/ = test zlib on all files in directory, recursively
```

For complete list of options refer to [manual](doc/lzbench.7.txt) in [doc](doc/) directory which contains more detailed documentation.


Building
-------------------------
To compile, you need a C and C++ compiler that is GNUC-compatible, such as GCC, LLVM/Clang, or ICC.
It is recommended to use GCC 7.1+ or Clang 6.0+.

For Linux/MacOS/MinGW (Windows):
```
make -j$(nproc)
```

The default linking for Linux is dynamic and static for Windows. This can be changed with `make BUILD_STATIC=0/1`.

For complete building instruction, with troubleshooting refer to [BUILD.md](BUILD.md).


Supported compressors
-------------------------

 - [brieflz 1.3.0](https://github.com/jibsen/brieflz)
 - [brotli 1.1.0](https://github.com/google/brotli)
 - [bsc 3.3.5](https://github.com/IlyaGrebnov/libbsc)
 - [bzip2 1.0.8](http://www.bzip.org/downloads.html)
 - [bzip3 1.5.1](https://github.com/kspalaiologos/bzip3)
 - [crush 1.0](https://sourceforge.net/projects/crush/)
 - [fastlz 0.5.0](https://github.com/ariya/FastLZ)
 - [fast-lzma2 1.0.1](https://github.com/conor42/fast-lzma2)
 - [glza 0.8](https://encode.su/threads/2427-GLZA)
 - [kanzi 2.3](https://github.com/flanglet/kanzi-cpp)
 - [libdeflate v1.23](https://github.com/ebiggers/libdeflate)
 - [lizard v2.1](https://github.com/inikep/lizard)
 - [lz4/lz4hc v1.10.0](https://github.com/lz4/lz4)
 - [lzav 4.15](https://github.com/avaneev/lzav)
 - [lzf 3.6](http://software.schmorp.de/pkg/liblzf.html)
 - [lzfse/lzvn 1.0](https://github.com/lzfse/lzfse)
 - [lzg 1.0.10](https://github.com/mbitsnbites/liblzg)
 - [lzham 1.0](https://github.com/richgel999/lzham_codec)
 - lzjb 2010
 - [lzlib 1.15](http://www.nongnu.org/lzip)
 - [lzma v24.09](http://7-zip.org)
 - [lzo 2.10](http://www.oberhumer.com/opensource/lzo)
 - [lzsse 2019-04-18 (1847c3e827)](https://github.com/ConorStokes/LZSSE)
 - [nvcomp 2.2.0](https://github.com/NVIDIA/nvcomp) - if CUDA is available
 - [ppmd8 24.09](https://github.com/pps83/libppmd)
 - [quicklz 1.5.0](http://www.quicklz.com)
 - [slz 1.2.1](http://www.libslz.org/) - only a compressor, uses zlib for decompression
 - [snappy 1.2.1](https://github.com/google/snappy)
 - [tamp 1.3.1](https://github.com/BrianPugh/tamp)
 - [tornado 0.6a](http://freearc.org)
 - [ucl 1.03](http://www.oberhumer.com/opensource/ucl/)
 - [xz 5.6.3](https://github.com/tukaani-project/xz)
 - [zlib 1.3.1](http://zlib.net)
 - [zlib-ng 2.2.3](https://github.com/zlib-ng/zlib-ng)
 - [zling 2018-10-12](https://github.com/richox/libzling) - according to the author using libzling in a production environment is not a good idea
 - [zstd 1.5.7](https://github.com/facebook/zstd)

**Warning**: The compressors listed below have security issues and/or are
no longer maintained. For information about the security of the various compressors,
see the [CompFuzz Results](https://github.com/nemequ/compfuzz/wiki/Results) page.

 - [csc 2016-10-13](https://github.com/fusiyuan2010/CSC): May cause a segmentation fault when compiled with Apple LLVM version 7.3.0 (clang-703.0.31).
 - [density 0.14.2](https://github.com/g1mv/density/tree/c_archive): Contains bugs leading to shortened decompressed output.
 - [gipfeli 2016-07-13](https://github.com/google/gipfeli): Contains bugs causing decompression file mismatch when compiled with GCC 14.2 using -O3.
 - [lzmat 1.01 v1.0](https://github.com/nemequ/lzmat): Contains decompression bugs and may cause a segmentation fault when compiled with GCC 4.9+ using -O3 optimization.
 - [lzrw 15-Jul-1991](https://en.wikipedia.org/wiki/LZRW): May trigger a segmentation fault when compiled with GCC 4.9+ using -O3.
 - [pithy 2011-12-24](https://github.com/johnezang/pithy): Contains decompression bugs (returns 0).
 - [wflz 2015-09-16](https://github.com/ShaneWF/wflz): May result in a segmentation fault when compiled with GCC 4.9+ using -O3.
 - [yalz77 2022-07-06](https://github.com/ivan-tkatchev/yalz77): A segmentation fault was encountered with GCC 13.3.0 on a 32-bit ARM (arm-linux-gnueabi) target.
 - [yappy 2014-03-22](https://encode.su/threads/2825-Yappy-(working)-compressor): A segmentation fault was observed with GCC 13.3.0 on a 32-bit ARM (arm-linux-gnueabi) system.

Benchmarks
-------------------------

The following results were obtained using `lzbench 2.0.1`, built with `gcc 14.2.0` and executed with the options `-eALL -t8,8 -o1c4`.
The tests were run on a single thread of an AMD EPYC 9554 processor at 3.10 GHz, with the CPU governor set to `performance` and turbo
boost disabled for stability. The operating system was `Ubuntu 24.04.1`, and the benchmark made use of
[`silesia.tar`](https://github.com/DataCompression/corpus-collection/tree/main/Silesia-Corpus), which contains tarred files from the
[Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).

The results sorted by ratio are available [here](doc/lzbench20_sorted.md).

| Compressor name         | Compression| Decompress.| Compr. size | Ratio |
| ---------------         | -----------| -----------| ----------- | ----- |
| memcpy                  | 16332 MB/s | 16362 MB/s |   211947520 |100.00 |
| brieflz 1.3.0 -1        |   199 MB/s |   354 MB/s |    81138803 | 38.28 |
| brieflz 1.3.0 -3        |   132 MB/s |   364 MB/s |    75550736 | 35.65 |
| brieflz 1.3.0 -6        |  21.0 MB/s |   395 MB/s |    67208420 | 31.71 |
| brieflz 1.3.0 -8        |  2.84 MB/s |   419 MB/s |    64531718 | 30.45 |
| brotli 1.1.0 -0         |   341 MB/s |   352 MB/s |    78433298 | 37.01 |
| brotli 1.1.0 -2         |   140 MB/s |   413 MB/s |    68069489 | 32.12 |
| brotli 1.1.0 -5         |  37.1 MB/s |   451 MB/s |    59555446 | 28.10 |
| brotli 1.1.0 -8         |  12.3 MB/s |   477 MB/s |    57148304 | 26.96 |
| brotli 1.1.0 -11        |  0.58 MB/s |   389 MB/s |    50407795 | 23.78 |
| bsc 3.3.5 -m0 -e1       |  16.6 MB/s |  24.1 MB/s |    49143366 | 23.19 |
| bsc 3.3.5 -m4 -e1       |  31.2 MB/s |  14.3 MB/s |    50626974 | 23.89 |
| bsc 3.3.5 -m5 -e1       |  28.8 MB/s |  13.3 MB/s |    49522504 | 23.37 |
| bzip2 1.0.8 -1          |  14.8 MB/s |  46.6 MB/s |    60484813 | 28.54 |
| bzip2 1.0.8 -5          |  14.0 MB/s |  40.0 MB/s |    55724395 | 26.29 |
| bzip2 1.0.8 -9          |  13.1 MB/s |  37.5 MB/s |    54572811 | 25.75 |
| crush 1.0 -0            |  64.3 MB/s |   357 MB/s |    73064603 | 34.47 |
| crush 1.0 -1            |  6.95 MB/s |   407 MB/s |    66494412 | 31.37 |
| crush 1.0 -2            |  0.91 MB/s |   421 MB/s |    63746223 | 30.08 |
| fastlz 0.5.0 -1         |   285 MB/s |   710 MB/s |   104628084 | 49.37 |
| fastlz 0.5.0 -2         |   293 MB/s |   699 MB/s |   100906072 | 47.61 |
| fastlzma2 1.0.1 -1      |  21.9 MB/s |  75.8 MB/s |    59030950 | 27.85 |
| fastlzma2 1.0.1 -3      |  12.1 MB/s |  81.4 MB/s |    54023833 | 25.49 |
| fastlzma2 1.0.1 -5      |  7.31 MB/s |  87.6 MB/s |    51209567 | 24.16 |
| fastlzma2 1.0.1 -8      |  4.46 MB/s |  89.7 MB/s |    49126736 | 23.18 |
| fastlzma2 1.0.1 -10     |  3.31 MB/s |  90.2 MB/s |    48666061 | 22.96 |
| kanzi 2.3 -2            |   155 MB/s |   432 MB/s |    68264304 | 32.21 |
| kanzi 2.3 -3            |   106 MB/s |   321 MB/s |    64963864 | 30.65 |
| kanzi 2.3 -4            |  53.7 MB/s |   152 MB/s |    60767201 | 28.67 |
| kanzi 2.3 -5            |  21.0 MB/s |  50.3 MB/s |    54050463 | 25.50 |
| kanzi 2.3 -6            |  15.8 MB/s |  29.6 MB/s |    49517568 | 23.36 |
| kanzi 2.3 -7            |  10.2 MB/s |  15.2 MB/s |    47308205 | 22.32 |
| kanzi 2.3 -8            |  3.45 MB/s |  3.29 MB/s |    43247149 | 20.40 |
| kanzi 2.3 -9            |  2.44 MB/s |  2.35 MB/s |    41807652 | 19.73 |
| libdeflate 1.23 -1      |   207 MB/s |   860 MB/s |    73502791 | 34.68 |
| libdeflate 1.23 -3      |   136 MB/s |   895 MB/s |    70170816 | 33.11 |
| libdeflate 1.23 -6      |  84.3 MB/s |   912 MB/s |    67510615 | 31.85 |
| libdeflate 1.23 -9      |  28.5 MB/s |   904 MB/s |    66715751 | 31.48 |
| libdeflate 1.23 -12     |  5.14 MB/s |   919 MB/s |    64678723 | 30.52 |
| lizard 2.1 -10          |   482 MB/s |  2172 MB/s |   103402971 | 48.79 |
| lizard 2.1 -12          |   165 MB/s |  2014 MB/s |    86232422 | 40.69 |
| lizard 2.1 -15          |  77.9 MB/s |  2119 MB/s |    81187330 | 38.31 |
| lizard 2.1 -19          |  4.74 MB/s |  1999 MB/s |    77416400 | 36.53 |
| lizard 2.1 -20          |   387 MB/s |  1688 MB/s |    96924204 | 45.73 |
| lizard 2.1 -22          |   162 MB/s |  1721 MB/s |    84866725 | 40.04 |
| lizard 2.1 -25          |  21.9 MB/s |  1750 MB/s |    75131286 | 35.45 |
| lizard 2.1 -29          |  2.10 MB/s |  1819 MB/s |    68694227 | 32.41 |
| lizard 2.1 -30          |   363 MB/s |  1187 MB/s |    85727429 | 40.45 |
| lizard 2.1 -32          |   164 MB/s |  1276 MB/s |    78652654 | 37.11 |
| lizard 2.1 -35          |  85.4 MB/s |  1548 MB/s |    74563583 | 35.18 |
| lizard 2.1 -39          |  4.52 MB/s |  1515 MB/s |    69807522 | 32.94 |
| lizard 2.1 -40          |   297 MB/s |  1150 MB/s |    80843049 | 38.14 |
| lizard 2.1 -42          |   141 MB/s |  1235 MB/s |    73350988 | 34.61 |
| lizard 2.1 -45          |  21.6 MB/s |  1350 MB/s |    66676653 | 31.46 |
| lizard 2.1 -49          |  2.06 MB/s |  1325 MB/s |    60679215 | 28.63 |
| lz4fast 1.10.0 -17      |  1002 MB/s |  4166 MB/s |   131732802 | 62.15 |
| lz4fast 1.10.0 -9       |   820 MB/s |  3922 MB/s |   120130796 | 56.68 |
| lz4fast 1.10.0 -3       |   657 MB/s |  3744 MB/s |   107066190 | 50.52 |
| lz4 1.10.0              |   577 MB/s |  3716 MB/s |   100880800 | 47.60 |
| lz4hc 1.10.0 -1         |   262 MB/s |  3221 MB/s |    89135429 | 42.06 |
| lz4hc 1.10.0 -4         |  76.3 MB/s |  3421 MB/s |    79807909 | 37.65 |
| lz4hc 1.10.0 -9         |  30.9 MB/s |  3527 MB/s |    77884448 | 36.75 |
| lz4hc 1.10.0 -12        |  10.5 MB/s |  3616 MB/s |    77262620 | 36.45 |
| lzav 4.5 -1             |   385 MB/s |  2643 MB/s |    86497609 | 40.81 |
| lzav 4.5 -2             |  74.1 MB/s |  2574 MB/s |    75602661 | 35.67 |
| lzf 3.6 -0              |   339 MB/s |   625 MB/s |   105682088 | 49.86 |
| lzf 3.6 -1              |   339 MB/s |   637 MB/s |   102041092 | 48.14 |
| lzfse 2017-03-08        |  81.3 MB/s |   724 MB/s |    67624281 | 31.91 |
| lzg 1.0.10 -1           |  85.9 MB/s |   524 MB/s |   108553667 | 51.22 |
| lzg 1.0.10 -4           |  47.8 MB/s |   528 MB/s |    95930551 | 45.26 |
| lzg 1.0.10 -6           |  28.6 MB/s |   562 MB/s |    89490220 | 42.22 |
| lzg 1.0.10 -8           |  8.97 MB/s |   611 MB/s |    83606901 | 39.45 |
| lzham 1.0 -d26 -0       |  11.7 MB/s |   233 MB/s |    64089870 | 30.24 |
| lzham 1.0 -d26 -1       |  3.08 MB/s |   309 MB/s |    54740589 | 25.83 |
| lzlib 1.15 -0           |  34.1 MB/s |  58.6 MB/s |    63847386 | 30.12 |
| lzlib 1.15 -3           |  7.69 MB/s |  66.9 MB/s |    56320674 | 26.57 |
| lzlib 1.15 -6           |  2.91 MB/s |  72.0 MB/s |    49777495 | 23.49 |
| lzlib 1.15 -9           |  1.81 MB/s |  72.5 MB/s |    48296889 | 22.79 |
| lzma 24.09 -0           |  31.0 MB/s |  74.1 MB/s |    60509826 | 28.55 |
| lzma 24.09 -2           |  22.4 MB/s |  83.1 MB/s |    57072498 | 26.93 |
| lzma 24.09 -4           |  12.6 MB/s |  85.6 MB/s |    55926363 | 26.39 |
| lzma 24.09 -6           |  4.86 MB/s |  91.5 MB/s |    49544915 | 23.38 |
| lzma 24.09 -9           |  4.01 MB/s |  93.0 MB/s |    48674973 | 22.97 |
| lzo1 2.10 -1            |   236 MB/s |   647 MB/s |   106474519 | 50.24 |
| lzo1 2.10 -99           |   106 MB/s |   682 MB/s |    94946129 | 44.80 |
| lzo1a 2.10 -1           |   234 MB/s |   696 MB/s |   104202251 | 49.16 |
| lzo1a 2.10 -99          |   106 MB/s |   722 MB/s |    92666265 | 43.72 |
| lzo1b 2.10 -1           |   201 MB/s |   642 MB/s |    97036087 | 45.78 |
| lzo1b 2.10 -3           |   206 MB/s |   658 MB/s |    94044578 | 44.37 |
| lzo1b 2.10 -6           |   205 MB/s |   659 MB/s |    91382355 | 43.12 |
| lzo1b 2.10 -9           |   158 MB/s |   655 MB/s |    89261884 | 42.12 |
| lzo1b 2.10 -99          |   104 MB/s |   664 MB/s |    85653376 | 40.41 |
| lzo1b 2.10 -999         |  13.4 MB/s |   752 MB/s |    76594292 | 36.14 |
| lzo1c 2.10 -1           |   209 MB/s |   671 MB/s |    99550904 | 46.97 |
| lzo1c 2.10 -3           |   207 MB/s |   685 MB/s |    96716153 | 45.63 |
| lzo1c 2.10 -6           |   173 MB/s |   681 MB/s |    93303623 | 44.02 |
| lzo1c 2.10 -9           |   141 MB/s |   676 MB/s |    91040386 | 42.95 |
| lzo1c 2.10 -99          |   102 MB/s |   682 MB/s |    88112288 | 41.57 |
| lzo1c 2.10 -999         |  20.7 MB/s |   726 MB/s |    80396741 | 37.93 |
| lzo1f 2.10 -1           |   185 MB/s |   633 MB/s |    99743329 | 47.06 |
| lzo1f 2.10 -999         |  18.9 MB/s |   656 MB/s |    80890206 | 38.17 |
| lzo1x 2.10 -1           |   513 MB/s |   696 MB/s |   100572537 | 47.45 |
| lzo1x 2.10 -11          |   560 MB/s |   710 MB/s |   106604629 | 50.30 |
| lzo1x 2.10 -12          |   545 MB/s |   695 MB/s |   103238859 | 48.71 |
| lzo1x 2.10 -15          |   532 MB/s |   694 MB/s |   101462094 | 47.87 |
| lzo1x 2.10 -999         |  7.13 MB/s |   658 MB/s |    75301903 | 35.53 |
| lzo1y 2.10 -1           |   514 MB/s |   689 MB/s |   101258318 | 47.78 |
| lzo1y 2.10 -999         |  7.37 MB/s |   647 MB/s |    75503849 | 35.62 |
| lzo1z 2.10 -999         |  7.22 MB/s |   643 MB/s |    75061331 | 35.42 |
| lzo2a 2.10 -999         |  23.8 MB/s |   531 MB/s |    82809337 | 39.07 |
| lzsse2 2019-04-18 -1    |  19.0 MB/s |  3286 MB/s |    87976095 | 41.51 |
| lzsse2 2019-04-18 -6    |  8.18 MB/s |  3786 MB/s |    75837101 | 35.78 |
| lzsse2 2019-04-18 -12   |  7.97 MB/s |  3790 MB/s |    75829973 | 35.78 |
| lzsse2 2019-04-18 -16   |  8.00 MB/s |  3788 MB/s |    75829973 | 35.78 |
| lzsse4 2019-04-18 -1    |  17.8 MB/s |  4209 MB/s |    82542106 | 38.94 |
| lzsse4 2019-04-18 -6    |  9.10 MB/s |  4598 MB/s |    76118298 | 35.91 |
| lzsse4 2019-04-18 -12   |  8.89 MB/s |  4611 MB/s |    76113017 | 35.91 |
| lzsse4 2019-04-18 -16   |  8.90 MB/s |  4611 MB/s |    76113017 | 35.91 |
| lzsse8 2019-04-18 -1    |  16.4 MB/s |  4340 MB/s |    81866245 | 38.63 |
| lzsse8 2019-04-18 -6    |  8.70 MB/s |  4752 MB/s |    75469717 | 35.61 |
| lzsse8 2019-04-18 -12   |  8.53 MB/s |  4761 MB/s |    75464339 | 35.61 |
| lzsse8 2019-04-18 -16   |  8.52 MB/s |  4755 MB/s |    75464339 | 35.61 |
| lzvn 2017-03-08         |  69.3 MB/s |   884 MB/s |    80814609 | 38.13 |
| ppmd8 24.09 -4          |  13.2 MB/s |  12.0 MB/s |    51241932 | 24.18 |
| quicklz 1.5.0 -1        |   459 MB/s |   491 MB/s |    94720562 | 44.69 |
| quicklz 1.5.0 -2        |   223 MB/s |   485 MB/s |    84555627 | 39.89 |
| quicklz 1.5.0 -3        |  60.2 MB/s |   835 MB/s |    81822241 | 38.60 |
| slz_gzip 1.2.1 -1       |   310 MB/s |   354 MB/s |    99657946 | 47.02 |
| slz_gzip 1.2.1 -2       |   303 MB/s |   356 MB/s |    96863082 | 45.70 |
| slz_gzip 1.2.1 -3       |   297 MB/s |   357 MB/s |    96187768 | 45.38 |
| snappy 1.2.1            |   401 MB/s |  1077 MB/s |   101415443 | 47.85 |
| tornado 0.6a -1         |   351 MB/s |   499 MB/s |   107381846 | 50.66 |
| tornado 0.6a -2         |   298 MB/s |   458 MB/s |    90076660 | 42.50 |
| tornado 0.6a -3         |   185 MB/s |   283 MB/s |    72662044 | 34.28 |
| tornado 0.6a -4         |   154 MB/s |   296 MB/s |    70513617 | 33.27 |
| tornado 0.6a -5         |  70.9 MB/s |   232 MB/s |    64129604 | 30.26 |
| tornado 0.6a -6         |  45.6 MB/s |   230 MB/s |    62364583 | 29.42 |
| tornado 0.6a -7         |  16.9 MB/s |   223 MB/s |    59026325 | 27.85 |
| tornado 0.6a -10        |  5.58 MB/s |   228 MB/s |    57588241 | 27.17 |
| tornado 0.6a -13        |  5.42 MB/s |   243 MB/s |    55614072 | 26.24 |
| tornado 0.6a -16        |  2.32 MB/s |   255 MB/s |    53257046 | 25.13 |
| ucl_nrv2b 1.03 -1       |  48.5 MB/s |   290 MB/s |    81703168 | 38.55 |
| ucl_nrv2b 1.03 -6       |  18.2 MB/s |   332 MB/s |    73902185 | 34.87 |
| ucl_nrv2b 1.03 -9       |  2.05 MB/s |   363 MB/s |    71031195 | 33.51 |
| ucl_nrv2d 1.03 -1       |  48.8 MB/s |   293 MB/s |    81461976 | 38.43 |
| ucl_nrv2d 1.03 -6       |  18.1 MB/s |   333 MB/s |    73757673 | 34.80 |
| ucl_nrv2d 1.03 -9       |  2.06 MB/s |   364 MB/s |    70053895 | 33.05 |
| ucl_nrv2e 1.03 -1       |  49.0 MB/s |   297 MB/s |    81195560 | 38.31 |
| ucl_nrv2e 1.03 -6       |  18.1 MB/s |   342 MB/s |    73302012 | 34.58 |
| ucl_nrv2e 1.03 -9       |  2.06 MB/s |   372 MB/s |    69645134 | 32.86 |
| xz 5.6.3 -0             |  23.6 MB/s |  98.2 MB/s |    62579435 | 29.53 |
| xz 5.6.3 -3             |  7.52 MB/s |   122 MB/s |    55745125 | 26.30 |
| xz 5.6.3 -6             |  2.97 MB/s |   127 MB/s |    49195929 | 23.21 |
| xz 5.6.3 -9             |  2.57 MB/s |   123 MB/s |    48745306 | 23.00 |
| zlib 1.3.1 -1           |  93.0 MB/s |   323 MB/s |    77259029 | 36.45 |
| zlib 1.3.1 -6           |  25.3 MB/s |   344 MB/s |    68228431 | 32.19 |
| zlib 1.3.1 -9           |  10.3 MB/s |   348 MB/s |    67644548 | 31.92 |
| zlib-ng 2.2.3 -1        |   202 MB/s |   471 MB/s |    94127047 | 44.41 |
| zlib-ng 2.2.3 -6        |  62.1 MB/s |   509 MB/s |    68861129 | 32.49 |
| zlib-ng 2.2.3 -9        |  24.6 MB/s |   518 MB/s |    67582060 | 31.89 |
| zling 2018-10-12 -0     |  78.5 MB/s |   178 MB/s |    62990590 | 29.72 |
| zling 2018-10-12 -1     |  70.0 MB/s |   182 MB/s |    62022546 | 29.26 |
| zling 2018-10-12 -2     |  63.1 MB/s |   184 MB/s |    61503093 | 29.02 |
| zling 2018-10-12 -3     |  56.6 MB/s |   186 MB/s |    60999828 | 28.78 |
| zling 2018-10-12 -4     |  47.9 MB/s |   187 MB/s |    60626768 | 28.60 |
| zstd_fast 1.5.6 --5     |   573 MB/s |  1950 MB/s |   103093752 | 48.64 |
| zstd_fast 1.5.6 --3     |   518 MB/s |  1822 MB/s |    94674672 | 44.67 |
| zstd_fast 1.5.6 --1     |   459 MB/s |  1717 MB/s |    86984009 | 41.04 |
| zstd 1.5.6 -1           |   422 MB/s |  1347 MB/s |    73421914 | 34.64 |
| zstd 1.5.6 -2           |   344 MB/s |  1246 MB/s |    69503444 | 32.79 |
| zstd 1.5.6 -5           |   125 MB/s |  1197 MB/s |    63040310 | 29.74 |
| zstd 1.5.6 -8           |  62.9 MB/s |  1319 MB/s |    60015064 | 28.32 |
| zstd 1.5.6 -11          |  34.4 MB/s |  1332 MB/s |    58262299 | 27.49 |
| zstd 1.5.6 -15          |  8.36 MB/s |  1369 MB/s |    57168834 | 26.97 |
| zstd 1.5.6 -18          |  3.79 MB/s |  1169 MB/s |    53329873 | 25.16 |
| zstd 1.5.6 -22          |  2.08 MB/s |  1073 MB/s |    52333880 | 24.69 |
