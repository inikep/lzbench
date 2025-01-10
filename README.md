Introduction
-------------------------

lzbench is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors. It joins all compressors into a single exe.
At the beginning an input file is read to memory.
Then all compressors are used to compress and decompress the file and decompressed file is verified.
This approach has a big advantage of using the same compiler with the same optimizations for all compressors.
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).

|Status   |
|---------|
| [![Build Status][AzurePipelinesMasterBadge]][AzurePipelinesLink] [![Build status][AppveyorMasterBadge]][AppveyorLink] |

[AzurePipelinesMasterBadge]: https://dev.azure.com/inikep/lzbench/_apis/build/status%2Finikep.lzbench?branchName=master "gcc and clang tests"
[AzurePipelinesLink]: https://dev.azure.com/inikep/lzbench/_build/latest?definitionId=10&branchName=master
[AppveyorMasterBadge]: https://ci.appveyor.com/api/projects/status/u7kjj8ino4gww40v/branch/master?svg=true "mingw tests"
[AppveyorLink]: https://ci.appveyor.com/project/inikep/lzbench

Usage
-------------------------

```
usage: lzbench [options] input [input2] [input3]

where [input] is a file or a directory and [options] are:
 -b#   set block/chunk size to # KB (default = MIN(filesize,1747626 KB))
 -c#   sort results by column # (1=algname, 2=ctime, 3=dtime, 4=comprsize)
 -e#   #=compressors separated by '/' with parameters specified after ',' (deflt=fast)
 -iX,Y set min. number of compression and decompression iterations (default = 1, 1)
 -j    join files in memory but compress them independently (for many small files)
 -l    list of available compressors and aliases
 -m#   set memory limit to # MB (default = no limit)
 -o#   output text format 1=Markdown, 2=text, 3=text+origSize, 4=CSV (default = 2)
 -p#   print time for all iterations: 1=fastest 2=average 3=median (default = 1)
 -r    operate recursively on directories
 -s#   use only compressors with compression speed over # MB (default = 0 MB)
 -tX,Y set min. time in seconds for compression and decompression (default = 1, 2)
 -v    disable progress information
 -x    disable real-time process priority
 -z    show (de)compression times instead of speed

Example usage:
  lzbench -ezstd filename = selects all levels of zstd
  lzbench -ebrotli,2,5/zstd filename = selects levels 2 & 5 of brotli and zstd
  lzbench -t3 -u5 fname = 3 sec compression and 5 sec decompression loops
  lzbench -t0 -u0 -i3 -j5 -ezstd fname = 3 compression and 5 decompression iter.
  lzbench -t0u0i3j5 -ezstd fname = the same as above with aggregated parameters
```


Compilation
-------------------------
For Linux/MacOS/MinGW (Windows):
```
make -j$(nproc)
```

For 32-bit compilation:
```
make BUILD_ARCH=32-bit -j$(nproc)

```

The default linking for Linux is dynamic and static for Windows. This can be changed with `make BUILD_STATIC=0/1`.

To remove one of compressors you can add `-DBENCH_REMOVE_XXX` to `DEFINES` in Makefile (e.g. `DEFINES += -DBENCH_REMOVE_LZ4` to remove LZ4).
You also have to remove corresponding `*.o` files (e.g. `lz4/lz4.o` and `lz4/lz4hc.o`).

lzbench undergoes automated testing using Azure Pipelines and AppVeyor with the following compilers:
- Ubuntu: gcc (versions 7.5 to 14.2) and clang (versions 6.0 to 19), gcc 14.2 (32-bit)
- MacOS: Apple LLVM version 15.0.0
- MinGW (Windows): gcc 5.3 (32-bit) and gcc 9.1 (64-bit)
- Cross-compilation: gcc for ARM (32-bit and 64-bit) and PowerPC (32-bit and 64-bit)


Supported compressors
-------------------------
**Warning**: some of the compressors listed here have security issues and/or are
no longer maintained.  For information about the security of the various compressors,
see the [CompFuzz Results](https://github.com/nemequ/compfuzz/wiki/Results) page.

 - [blosclz 2.0.0](https://github.com/Blosc/c-blosc2)
 - [brieflz 1.3.0](https://github.com/jibsen/brieflz)
 - [brotli 1.0.9](https://github.com/google/brotli)
 - [bsc 3.3.4](https://github.com/IlyaGrebnov/libbsc)
 - [bzip2 1.0.8](http://www.bzip.org/downloads.html)
 - [crush 1.0](https://sourceforge.net/projects/crush/)
 - [csc 2016-10-13](https://github.com/fusiyuan2010/CSC) - WARNING: it can throw SEGFAULT compiled with Apple LLVM version 7.3.0 (clang-703.0.31)
 - [density 0.14.2](https://github.com/centaurean/density) - WARNING: it contains bugs (shortened decompressed output))
 - [fastlz 0.5.0](http://fastlz.org)
 - [fast-lzma2 1.0.1](https://github.com/conor42/fast-lzma2)
 - [gipfeli 2016-07-13](https://github.com/google/gipfeli)
 - [glza 0.8](https://encode.su/threads/2427-GLZA)
 - [kanzi 2.3](https://github.com/flanglet/kanzi-cpp)
 - [libdeflate v1.23](https://github.com/ebiggers/libdeflate)
 - [lizard v1.0 (formerly lz5)](https://github.com/inikep/lizard)
 - [lz4/lz4hc v1.10.0](https://github.com/lz4/lz4)
 - [lzf 3.6](http://software.schmorp.de/pkg/liblzf.html)
 - [lzfse/lzvn 1.0](https://github.com/lzfse/lzfse)
 - [lzg 1.0.10](https://liblzg.bitsnbites.eu/)
 - [lzham 1.0](https://github.com/richgel999/lzham_codec)
 lzjb 2010
 - [lzlib 1.13](http://www.nongnu.org/lzip)
 - [lzma v24.09](http://7-zip.org)
 - [lzmat 1.01 v1.0](https://github.com/nemequ/lzmat) - WARNING: it contains bugs (decompression error; returns 0); it can throw SEGFAULT compiled with gcc 4.9+ -O3
 - [lzo 2.10](http://www.oberhumer.com/opensource/lzo)
 - [lzrw 15-Jul-1991](https://en.wikipedia.org/wiki/LZRW)
 - [lzsse 2019-04-18 (1847c3e827)](https://github.com/ConorStokes/LZSSE)
 - [pithy 2011-12-24](https://github.com/johnezang/pithy) - WARNING: it contains bugs (decompression error; returns 0)
 - [ppmd8 24.09](https://github.com/pps83/libppmd)
 - [quicklz 1.5.0](http://www.quicklz.com)
 - [shrinker 0.1](https://code.google.com/p/data-shrinker) - WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3
 - [slz 1.2.0](http://www.libslz.org/) - only a compressor, uses zlib for decompression
 - [snappy 1.1.10](https://github.com/google/snappy)
 - [tamp 1.3.1](https://github.com/BrianPugh/tamp)
 - [tornado 0.6a](http://freearc.org)
 - [ucl 1.03](http://www.oberhumer.com/opensource/ucl/)
 - [wflz 2015-09-16](https://github.com/ShaneWF/wflz) - WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3
 - [xpack 2016-06-02](https://github.com/ebiggers/xpack)
 - [xz 5.2.4](https://tukaani.org/xz/)
 - [yalz77 2015-09-19](https://github.com/ivan-tkatchev/yalz77) - WARNING: A SEGFAULT was encountered with gcc 13.3.0 on the 32-bit ARM (arm-linux-gnueabi) target
 - [yappy 2014-03-22](https://encode.su/threads/2825-Yappy-(working)-compressor) - WARNING: A SEGFAULT was encountered with gcc 13.3.0 on the 32-bit ARM (arm-linux-gnueabi)
 - [zlib 1.3.1](http://zlib.net)
 - [zling 2018-10-12](https://github.com/richox/libzling) - according to the author using libzling in a production environment is not a good idea
 - [zstd 1.5.6](https://github.com/facebook/zstd)
 - [nvcomp 1.2.3](https://github.com/NVIDIA/nvcomp) - If CUDA is available.


CUDA support
-------------------------

If CUDA is available, lzbench supports additional compressors:
  - [cudaMemcpy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) - similar to the reference `memcpy` benchmark, using GPU memory
  - [nvcomp 1.2.2](https://github.com/NVIDIA/nvcomp) LZ4 GPU-only compressor

The directory where the CUDA compiler and libraries are available can be passed to `make` via the `CUDA_BASE` variable, *e.g.*:
```
make CUDA_BASE=/usr/local/cuda
```

Benchmarks
-------------------------

The following results were obtained using `lzbench 1.9`, built with `gcc 14.2.0` and executed with the options `-eall -t16,16 -o1c4`.
The tests were run on a single core of an AMD EPYC 9554 processor at 3.10 GHz, with the CPU governor set to `performance` and turbo
boost disabled for stability. The operating system was `Ubuntu 24.04.1`, and the benchmark made use of
[`silesia.tar`](https://github.com/DataCompression/corpus-collection/tree/main/Silesia-Corpus), which contains tarred files from the
[Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).

The results sorted by ratio are available [here](doc/lzbench19_sorted.md).

| Compressor name         | Compression| Decompress.| Compr. size | Ratio |
| ---------------         | -----------| -----------| ----------- | ----- |
| memcpy                  | 16341 MB/s | 16349 MB/s |   211947520 |100.00 |
| blosclz 2.0.0 -1        | 11395 MB/s | 15630 MB/s |   211947520 |100.00 |
| blosclz 2.0.0 -3        |   921 MB/s |  7252 MB/s |   199437330 | 94.10 |
| blosclz 2.0.0 -6        |   352 MB/s |   774 MB/s |   137571765 | 64.91 |
| blosclz 2.0.0 -9        |   345 MB/s |   746 MB/s |   135557850 | 63.96 |
| brieflz 1.3.0 -1        |   199 MB/s |   354 MB/s |    81138803 | 38.28 |
| brieflz 1.3.0 -3        |   133 MB/s |   363 MB/s |    75550736 | 35.65 |
| brieflz 1.3.0 -6        |  25.0 MB/s |   395 MB/s |    67208420 | 31.71 |
| brieflz 1.3.0 -8        |  2.89 MB/s |   420 MB/s |    64531718 | 30.45 |
| brotli 1.0.9 -0         |   329 MB/s |   339 MB/s |    78433298 | 37.01 |
| brotli 1.0.9 -2         |   142 MB/s |   403 MB/s |    68060686 | 32.11 |
| brotli 1.0.9 -5         |  36.9 MB/s |   447 MB/s |    59568603 | 28.11 |
| brotli 1.0.9 -8         |  12.4 MB/s |   474 MB/s |    57140168 | 26.96 |
| brotli 1.0.9 -11        |  0.60 MB/s |   392 MB/s |    50407795 | 23.78 |
| bzip2 1.0.8 -1          |  15.0 MB/s |  46.9 MB/s |    60484813 | 28.54 |
| bzip2 1.0.8 -5          |  14.1 MB/s |  40.4 MB/s |    55724395 | 26.29 |
| bzip2 1.0.8 -9          |  13.2 MB/s |  37.5 MB/s |    54572811 | 25.75 |
| crush 1.0 -0            |  65.3 MB/s |   358 MB/s |    73064603 | 34.47 |
| crush 1.0 -1            |  6.85 MB/s |   408 MB/s |    66494412 | 31.37 |
| crush 1.0 -2            |  0.91 MB/s |   422 MB/s |    63746223 | 30.08 |
| csc 2016-10-13 -1       |  21.5 MB/s |  40.4 MB/s |    56201092 | 26.52 |
| csc 2016-10-13 -3       |  8.35 MB/s |  37.7 MB/s |    53477914 | 25.23 |
| csc 2016-10-13 -5       |  3.45 MB/s |  42.1 MB/s |    49801577 | 23.50 |
| density 0.14.2 -1       |  1584 MB/s |  2405 MB/s |   133042166 | 62.77 |
| density 0.14.2 -2       |   789 MB/s |  1276 MB/s |   101651444 | 47.96 |
| density 0.14.2 -3       |   367 MB/s |   382 MB/s |    87649866 | 41.35 |
| fastlz 0.5.0 -1         |   285 MB/s |   712 MB/s |   104628084 | 49.37 |
| fastlz 0.5.0 -2         |   293 MB/s |   699 MB/s |   100906072 | 47.61 |
| fastlzma2 1.0.1 -1      |  22.0 MB/s |  74.8 MB/s |    59030954 | 27.85 |
| fastlzma2 1.0.1 -3      |  12.1 MB/s |  80.4 MB/s |    54023837 | 25.49 |
| fastlzma2 1.0.1 -5      |  7.38 MB/s |  86.4 MB/s |    51209571 | 24.16 |
| fastlzma2 1.0.1 -8      |  4.50 MB/s |  88.5 MB/s |    49126740 | 23.18 |
| fastlzma2 1.0.1 -10     |  3.34 MB/s |  89.0 MB/s |    48666065 | 22.96 |
| gipfeli 2016-07-13      |   312 MB/s |      ERROR |    87931759 | 41.49 |
| libdeflate 1.9 -1       |   195 MB/s |   644 MB/s |    73502791 | 34.68 |
| libdeflate 1.9 -3       |   129 MB/s |   664 MB/s |    70170813 | 33.11 |
| libdeflate 1.9 -6       |  82.6 MB/s |   683 MB/s |    67510615 | 31.85 |
| libdeflate 1.9 -9       |  28.2 MB/s |   682 MB/s |    66715751 | 31.48 |
| libdeflate 1.9 -12      |  6.01 MB/s |   689 MB/s |    64685828 | 30.52 |
| lz4 1.9.3               |   577 MB/s |  3660 MB/s |   100880800 | 47.60 |
| lz4fast 1.9.3 -3        |   656 MB/s |  3673 MB/s |   107066190 | 50.52 |
| lz4fast 1.9.3 -17       |   997 MB/s |  4035 MB/s |   131732802 | 62.15 |
| lz4hc 1.9.3 -1          |   108 MB/s |  3257 MB/s |    83803769 | 39.54 |
| lz4hc 1.9.3 -4          |  72.2 MB/s |  3375 MB/s |    79807909 | 37.65 |
| lz4hc 1.9.3 -9          |  28.8 MB/s |  3478 MB/s |    77884448 | 36.75 |
| lz4hc 1.9.3 -12         |  10.6 MB/s |  3574 MB/s |    77262620 | 36.45 |
| lzf 3.6 -0              |   339 MB/s |   620 MB/s |   105682088 | 49.86 |
| lzf 3.6 -1              |   337 MB/s |   615 MB/s |   102041092 | 48.14 |
| lzfse 2017-03-08        |  81.9 MB/s |   721 MB/s |    67624281 | 31.91 |
| lzg 1.0.10 -1           |  86.7 MB/s |   538 MB/s |   108553667 | 51.22 |
| lzg 1.0.10 -4           |  47.9 MB/s |   537 MB/s |    95930551 | 45.26 |
| lzg 1.0.10 -6           |  28.8 MB/s |   570 MB/s |    89490220 | 42.22 |
| lzg 1.0.10 -8           |  8.96 MB/s |   622 MB/s |    83606901 | 39.45 |
| lzham 1.0 -d26 -0       |  11.8 MB/s |   235 MB/s |    64089870 | 30.24 |
| lzham 1.0 -d26 -1       |  3.11 MB/s |   312 MB/s |    54740589 | 25.83 |
| lzjb 2010               |   304 MB/s |   438 MB/s |   122671613 | 57.88 |
| lzlib 1.13 -0           |  34.1 MB/s |  58.8 MB/s |    63847386 | 30.12 |
| lzlib 1.13 -3           |  7.71 MB/s |  67.4 MB/s |    56320674 | 26.57 |
| lzlib 1.13 -6           |  2.94 MB/s |  72.3 MB/s |    49777495 | 23.49 |
| lzlib 1.13 -9           |  1.82 MB/s |  72.9 MB/s |    48296889 | 22.79 |
| lzma 19.00 -0           |  30.1 MB/s |  67.4 MB/s |    64013917 | 30.20 |
| lzma 19.00 -2           |  24.3 MB/s |  78.3 MB/s |    58867911 | 27.77 |
| lzma 19.00 -4           |  16.8 MB/s |  82.9 MB/s |    57201645 | 26.99 |
| lzma 19.00 -5           |  3.30 MB/s |  91.4 MB/s |    49710307 | 23.45 |
| lzma 19.00 -9           |  2.59 MB/s |  92.6 MB/s |    48707450 | 22.98 |
| lzo1 2.10 -1            |   236 MB/s |   646 MB/s |   106474519 | 50.24 |
| lzo1 2.10 -99           |   106 MB/s |   681 MB/s |    94946129 | 44.80 |
| lzo1a 2.10 -1           |   234 MB/s |   693 MB/s |   104202251 | 49.16 |
| lzo1a 2.10 -99          |   106 MB/s |   721 MB/s |    92666265 | 43.72 |
| lzo1b 2.10 -1           |   201 MB/s |   643 MB/s |    97036087 | 45.78 |
| lzo1b 2.10 -3           |   205 MB/s |   658 MB/s |    94044578 | 44.37 |
| lzo1b 2.10 -6           |   203 MB/s |   660 MB/s |    91382355 | 43.12 |
| lzo1b 2.10 -9           |   157 MB/s |   655 MB/s |    89261884 | 42.12 |
| lzo1b 2.10 -99          |   104 MB/s |   665 MB/s |    85653376 | 40.41 |
| lzo1b 2.10 -999         |  13.5 MB/s |   753 MB/s |    76594292 | 36.14 |
| lzo1c 2.10 -1           |   209 MB/s |   670 MB/s |    99550904 | 46.97 |
| lzo1c 2.10 -3           |   207 MB/s |   684 MB/s |    96716153 | 45.63 |
| lzo1c 2.10 -6           |   174 MB/s |   683 MB/s |    93303623 | 44.02 |
| lzo1c 2.10 -9           |   142 MB/s |   679 MB/s |    91040386 | 42.95 |
| lzo1c 2.10 -99          |   102 MB/s |   684 MB/s |    88112288 | 41.57 |
| lzo1c 2.10 -999         |  21.1 MB/s |   726 MB/s |    80396741 | 37.93 |
| lzo1f 2.10 -1           |   187 MB/s |   639 MB/s |    99743329 | 47.06 |
| lzo1f 2.10 -999         |  19.0 MB/s |   660 MB/s |    80890206 | 38.17 |
| lzo1x 2.10 -1           |   511 MB/s |   692 MB/s |   100572537 | 47.45 |
| lzo1x 2.10 -11          |   561 MB/s |   708 MB/s |   106604629 | 50.30 |
| lzo1x 2.10 -12          |   546 MB/s |   693 MB/s |   103238859 | 48.71 |
| lzo1x 2.10 -15          |   534 MB/s |   690 MB/s |   101462094 | 47.87 |
| lzo1x 2.10 -999         |  7.14 MB/s |   657 MB/s |    75301903 | 35.53 |
| lzo1y 2.10 -1           |   516 MB/s |   690 MB/s |   101258318 | 47.78 |
| lzo1y 2.10 -999         |  7.38 MB/s |   647 MB/s |    75503849 | 35.62 |
| lzo1z 2.10 -999         |  7.20 MB/s |   642 MB/s |    75061331 | 35.42 |
| lzo2a 2.10 -999         |  24.0 MB/s |   527 MB/s |    82809337 | 39.07 |
| lzrw 15-Jul-1991 -1     |   231 MB/s |   537 MB/s |   113761625 | 53.67 |
| lzrw 15-Jul-1991 -3     |   288 MB/s |   595 MB/s |   105424168 | 49.74 |
| lzrw 15-Jul-1991 -4     |   310 MB/s |   533 MB/s |   100131356 | 47.24 |
| lzrw 15-Jul-1991 -5     |   143 MB/s |   539 MB/s |    90818810 | 42.85 |
| lzsse2 2019-04-18 -1    |  23.7 MB/s |  3267 MB/s |    87976095 | 41.51 |
| lzsse2 2019-04-18 -6    |  8.19 MB/s |  3786 MB/s |    75837101 | 35.78 |
| lzsse2 2019-04-18 -12   |  7.98 MB/s |  3787 MB/s |    75829973 | 35.78 |
| lzsse2 2019-04-18 -16   |  8.01 MB/s |  3788 MB/s |    75829973 | 35.78 |
| lzsse4 2019-04-18 -1    |  22.3 MB/s |  4210 MB/s |    82542106 | 38.94 |
| lzsse4 2019-04-18 -6    |  9.11 MB/s |  4603 MB/s |    76118298 | 35.91 |
| lzsse4 2019-04-18 -12   |  8.92 MB/s |  4608 MB/s |    76113017 | 35.91 |
| lzsse4 2019-04-18 -16   |  8.87 MB/s |  4606 MB/s |    76113017 | 35.91 |
| lzsse8 2019-04-18 -1    |  20.0 MB/s |  4326 MB/s |    81866245 | 38.63 |
| lzsse8 2019-04-18 -6    |  8.72 MB/s |  4745 MB/s |    75469717 | 35.61 |
| lzsse8 2019-04-18 -12   |  8.48 MB/s |  4746 MB/s |    75464339 | 35.61 |
| lzsse8 2019-04-18 -16   |  8.58 MB/s |  4758 MB/s |    75464339 | 35.61 |
| lzvn 2017-03-08         |  69.3 MB/s |   907 MB/s |    80814609 | 38.13 |
| pithy 2011-12-24 -0     |   497 MB/s |  1679 MB/s |   103072463 | 48.63 |
| pithy 2011-12-24 -3     |   471 MB/s |  1674 MB/s |    97255186 | 45.89 |
| pithy 2011-12-24 -6     |   440 MB/s |  1747 MB/s |    92090898 | 43.45 |
| pithy 2011-12-24 -9     |   389 MB/s |  1764 MB/s |    90360813 | 42.63 |
| quicklz 1.5.0 -1        |   459 MB/s |   504 MB/s |    94720562 | 44.69 |
| quicklz 1.5.0 -2        |   223 MB/s |   487 MB/s |    84555627 | 39.89 |
| quicklz 1.5.0 -3        |  60.2 MB/s |   836 MB/s |    81822241 | 38.60 |
| slz_gzip 1.2.0 -1       |   311 MB/s |   352 MB/s |    99657946 | 47.02 |
| slz_gzip 1.2.0 -2       |   304 MB/s |   355 MB/s |    96863082 | 45.70 |
| slz_gzip 1.2.0 -3       |   298 MB/s |   356 MB/s |    96187768 | 45.38 |
| snappy 1.1.10           |   418 MB/s |  1002 MB/s |   102146767 | 48.19 |
| tornado 0.6a -1         |   349 MB/s |   495 MB/s |   107381846 | 50.66 |
| tornado 0.6a -2         |   299 MB/s |   457 MB/s |    90076660 | 42.50 |
| tornado 0.6a -3         |   185 MB/s |   281 MB/s |    72662044 | 34.28 |
| tornado 0.6a -4         |   154 MB/s |   295 MB/s |    70513617 | 33.27 |
| tornado 0.6a -5         |  71.3 MB/s |   232 MB/s |    64129604 | 30.26 |
| tornado 0.6a -6         |  45.9 MB/s |   231 MB/s |    62364583 | 29.42 |
| tornado 0.6a -7         |  17.1 MB/s |   226 MB/s |    59026325 | 27.85 |
| tornado 0.6a -10        |  5.72 MB/s |   230 MB/s |    57588241 | 27.17 |
| tornado 0.6a -13        |  5.44 MB/s |   244 MB/s |    55614072 | 26.24 |
| tornado 0.6a -16        |  2.35 MB/s |   256 MB/s |    53257046 | 25.13 |
| ucl_nrv2b 1.03 -1       |  49.5 MB/s |   288 MB/s |    81703168 | 38.55 |
| ucl_nrv2b 1.03 -6       |  18.4 MB/s |   329 MB/s |    73902185 | 34.87 |
| ucl_nrv2b 1.03 -9       |  2.10 MB/s |   360 MB/s |    71031195 | 33.51 |
| ucl_nrv2d 1.03 -1       |  49.7 MB/s |   294 MB/s |    81461976 | 38.43 |
| ucl_nrv2d 1.03 -6       |  18.3 MB/s |   334 MB/s |    73757673 | 34.80 |
| ucl_nrv2d 1.03 -9       |  2.11 MB/s |   364 MB/s |    70053895 | 33.05 |
| ucl_nrv2e 1.03 -1       |  49.6 MB/s |   296 MB/s |    81195560 | 38.31 |
| ucl_nrv2e 1.03 -6       |  18.2 MB/s |   342 MB/s |    73302012 | 34.58 |
| ucl_nrv2e 1.03 -9       |  2.11 MB/s |   370 MB/s |    69645134 | 32.86 |
| xpack 2016-06-02 -1     |   160 MB/s |   746 MB/s |    71090065 | 33.54 |
| xpack 2016-06-02 -6     |  45.1 MB/s |   905 MB/s |    62213845 | 29.35 |
| xpack 2016-06-02 -9     |  16.3 MB/s |   928 MB/s |    61240928 | 28.89 |
| xz 5.2.5 -0             |  23.8 MB/s |  66.2 MB/s |    62579435 | 29.53 |
| xz 5.2.5 -3             |  7.60 MB/s |  80.0 MB/s |    55745125 | 26.30 |
| xz 5.2.5 -6             |  2.95 MB/s |  84.6 MB/s |    49195929 | 23.21 |
| xz 5.2.5 -9             |  2.56 MB/s |  83.8 MB/s |    48745306 | 23.00 |
| yalz77 2015-09-19 -1    |   101 MB/s |   490 MB/s |    93952728 | 44.33 |
| yalz77 2015-09-19 -4    |  63.7 MB/s |   500 MB/s |    87392632 | 41.23 |
| yalz77 2015-09-19 -8    |  44.9 MB/s |   498 MB/s |    85153287 | 40.18 |
| yalz77 2015-09-19 -12   |  34.9 MB/s |   496 MB/s |    84050625 | 39.66 |
| yappy 2014-03-22 -1     |   123 MB/s |  2125 MB/s |   105750956 | 49.89 |
| yappy 2014-03-22 -10    |  98.7 MB/s |  2286 MB/s |   100018673 | 47.19 |
| yappy 2014-03-22 -100   |  78.5 MB/s |  2314 MB/s |    98672514 | 46.56 |
| zlib 1.2.11 -1          |  94.6 MB/s |   324 MB/s |    77259029 | 36.45 |
| zlib 1.2.11 -6          |  25.4 MB/s |   344 MB/s |    68228431 | 32.19 |
| zlib 1.2.11 -9          |  10.3 MB/s |   348 MB/s |    67644548 | 31.92 |
| zling 2018-10-12 -0     |  78.1 MB/s |   180 MB/s |    62990590 | 29.72 |
| zling 2018-10-12 -1     |  69.1 MB/s |   183 MB/s |    62022546 | 29.26 |
| zling 2018-10-12 -2     |  62.0 MB/s |   185 MB/s |    61503093 | 29.02 |
| zling 2018-10-12 -3     |  56.2 MB/s |   187 MB/s |    60999828 | 28.78 |
| zling 2018-10-12 -4     |  47.6 MB/s |   188 MB/s |    60626768 | 28.60 |
| zstd 1.5.5 -1           |   424 MB/s |  1367 MB/s |    73421914 | 34.64 |
| zstd 1.5.5 -2           |   346 MB/s |  1269 MB/s |    69503444 | 32.79 |
| zstd 1.5.5 -5           |   124 MB/s |  1226 MB/s |    63040310 | 29.74 |
| zstd 1.5.5 -8           |  64.4 MB/s |  1352 MB/s |    60015064 | 28.32 |
| zstd 1.5.5 -11          |  35.1 MB/s |  1376 MB/s |    58262299 | 27.49 |
| zstd 1.5.5 -15          |  8.44 MB/s |  1406 MB/s |    57168834 | 26.97 |
| zstd 1.5.5 -18          |  3.85 MB/s |  1201 MB/s |    53420090 | 25.20 |
| zstd 1.5.5 -22          |  2.20 MB/s |  1109 MB/s |    52464642 | 24.75 |
| shrinker 0.1            |   882 MB/s |  3150 MB/s |   172535778 | 81.40 |
| wflz 2015-09-16         |   245 MB/s |  1010 MB/s |   109605264 | 51.71 |
| lzmat 1.01              |  28.9 MB/s |   438 MB/s |    76485353 | 36.09 |
