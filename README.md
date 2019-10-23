Introduction
-------------------------

lzbench is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors. It joins all compressors into a single exe. 
At the beginning an input file is read to memory. 
Then all compressors are used to compress and decompress the file and decompressed file is verified. 
This approach has a big advantage of using the same compiler with the same optimizations for all compressors. 
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).

|Status   |
|---------|
| [![Build Status][travisMasterBadge]][travisLink] [![Build status][AppveyorMasterBadge]][AppveyorLink]  |

[travisMasterBadge]: https://travis-ci.org/inikep/lzbench.svg?branch=master "Continuous Integration test suite"
[travisLink]: https://travis-ci.org/inikep/lzbench
[AppveyorMasterBadge]: https://ci.appveyor.com/api/projects/status/u7kjj8ino4gww40v/branch/master?svg=true "Visual test suite"
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
make
```

For 32-bit compilation:
```
make BUILD_ARCH=32-bit

```

To remove one of compressors you can add `-DBENCH_REMOVE_XXX` to `DEFINES` in Makefile (e.g. `DEFINES += -DBENCH_REMOVE_LZ4` to remove LZ4). 
You also have to remove corresponding `*.o` files (e.g. `lz4/lz4.o` and `lz4/lz4hc.o`).

lzbench was tested with:
- Ubuntu: gcc 4.6.3, 4.8.4 (both 32-bit and 64-bit), 4.9.3, 5.3.0, 6.1.1 and clang 3.4, 3.5, 3.6, 3.8
- MacOS: Apple LLVM version 6.0
- MinGW (Windows): gcc 5.3.0, 4.9.3 (32-bit), 4.8.3 (32-bit)



Supported compressors
-------------------------
**Warning**: some of the compressors listed here have security issues and/or are 
no longer maintained.  For information about the security of the various compressors, 
see the [CompFuzz Results](https://github.com/nemequ/compfuzz/wiki/Results) page.
```
blosclz 2015-11-10
brieflz 1.1.0
brotli 2017-12-12
crush 1.0
csc 2016-10-13 (WARNING: it can throw SEGFAULT compiled with Apple LLVM version 7.3.0 (clang-703.0.31))
density 0.12.5 beta (WARNING: it contains bugs (shortened decompressed output))
fastlz 0.1
gipfeli 2016-07-13
glza 0.8
libdeflate v0.7
lizard v1.0 (formerly lz5)
lz4/lz4hc v1.9.2
lzf 3.6
lzfse/lzvn 2017-03-08
lzg 1.0.8
lzham 1.0
lzjb 2010
lzlib 1.8
lzma v16.04
lzmat 1.01 (WARNING: it contains bugs (decompression error; returns 0); it can throw SEGFAULT compiled with gcc 4.9+ -O3)
lzo 2.09
lzrw 15-Jul-1991
lzsse 2016-05-14
pithy 2011-12-24 (WARNING: it contains bugs (decompression error; returns 0))
quicklz 1.5.0
shrinker 0.1 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
slz 1.0.0 (only a compressor, uses zlib for decompression)
snappy 2019-09-30 (e9e11b84e6)
tornado 0.6a
ucl 1.03
wflz 2015-09-16 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
xpack 2016-06-02
xz 5.2.4
yalz77 2015-09-19
yappy 2014-03-22 (WARNING: fails to decompress properly on ARM)
zlib 1.2.11
zling 2018-10-12 (according to the author using libzling in a production environment is not a good idea)
zstd 1.4.3
```


Benchmarks
-------------------------

The following results are obtained with `lzbench 1.7.1` with the `-t16,16 -eall` options using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 6.3.0)
with ["silesia.tar"](https://drive.google.com/file/d/0BwX7dtyRLxThenZpYU9zLTZhR1k/view?usp=sharing) which contains tarred files from [Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).
The results sorted by ratio are available [here](lzbench171_sorted.md).

| Compressor name         | Compression| Decompress.| Compr. size | Ratio | 
| ---------------         | -----------| -----------| ----------- | ----- |
| memcpy                  |  8657 MB/s |  8891 MB/s |   211947520 |100.00 |
| blosclz 2015-11-10 -1   |   902 MB/s |  5855 MB/s |   211768481 | 99.92 |
| blosclz 2015-11-10 -3   |   492 MB/s |  5190 MB/s |   204507781 | 96.49 |
| blosclz 2015-11-10 -6   |   234 MB/s |   916 MB/s |   113322667 | 53.47 |
| blosclz 2015-11-10 -9   |   220 MB/s |   696 MB/s |   102817442 | 48.51 |
| brieflz 1.1.0           |   105 MB/s |   158 MB/s |    81990651 | 38.68 |
| brotli 2017-03-10 -0    |   225 MB/s |   246 MB/s |    78432913 | 37.01 |
| brotli 2017-03-10 -2    |    98 MB/s |   289 MB/s |    68085200 | 32.12 |
| brotli 2017-03-10 -5    |    19 MB/s |   328 MB/s |    59714719 | 28.17 |
| brotli 2017-03-10 -8    |  4.90 MB/s |   330 MB/s |    57198711 | 26.99 |
| brotli 2017-03-10 -11   |  0.37 MB/s |   269 MB/s |    51136654 | 24.13 |
| crush 1.0 -0            |    28 MB/s |   259 MB/s |    73064603 | 34.47 |
| crush 1.0 -1            |  3.01 MB/s |   293 MB/s |    66494412 | 31.37 |
| crush 1.0 -2            |  0.36 MB/s |   299 MB/s |    63746223 | 30.08 |
| csc 2016-10-13 -1       |    13 MB/s |    48 MB/s |    56201092 | 26.52 |
| csc 2016-10-13 -3       |  5.70 MB/s |    47 MB/s |    53477914 | 25.23 |
| csc 2016-10-13 -5       |  2.39 MB/s |    49 MB/s |    49801577 | 23.50 |
| density 0.12.5 beta -1  |   800 MB/s |  1028 MB/s |   133085162 | 62.79 |
| density 0.12.5 beta -2  |   480 MB/s |   655 MB/s |   101706226 | 47.99 |
| density 0.12.5 beta -3  |   253 MB/s |   235 MB/s |    87622980 | 41.34 |
| fastlz 0.1 -1           |   235 MB/s |   461 MB/s |   104628084 | 49.37 |
| fastlz 0.1 -2           |   243 MB/s |   469 MB/s |   100906072 | 47.61 |
| gipfeli 2016-07-13      |   233 MB/s |   451 MB/s |    87931759 | 41.49 |
| libdeflate 0.7 -1       |   117 MB/s |   570 MB/s |    73318371 | 34.59 |
| libdeflate 0.7 -3       |    96 MB/s |   602 MB/s |    70668968 | 33.34 |
| libdeflate 0.7 -6       |    64 MB/s |   609 MB/s |    67928189 | 32.05 |
| libdeflate 0.7 -9       |    10 MB/s |   584 MB/s |    65701539 | 31.00 |
| libdeflate 0.7 -12      |  4.63 MB/s |   583 MB/s |    64801629 | 30.57 |
| lizard 1.0 -10          |   360 MB/s |  2625 MB/s |   103402971 | 48.79 |
| lizard 1.0 -12          |   105 MB/s |  2471 MB/s |    86232422 | 40.69 |
| lizard 1.0 -15          |    51 MB/s |  2569 MB/s |    81187330 | 38.31 |
| lizard 1.0 -19          |  3.17 MB/s |  2513 MB/s |    77416400 | 36.53 |
| lizard 1.0 -20          |   284 MB/s |  1734 MB/s |    96924204 | 45.73 |
| lizard 1.0 -22          |   105 MB/s |  1719 MB/s |    84866725 | 40.04 |
| lizard 1.0 -25          |    10 MB/s |  1688 MB/s |    75161667 | 35.46 |
| lizard 1.0 -29          |  1.32 MB/s |  1596 MB/s |    68694227 | 32.41 |
| lizard 1.0 -30          |   258 MB/s |   867 MB/s |    85727429 | 40.45 |
| lizard 1.0 -32          |   107 MB/s |   943 MB/s |    78652654 | 37.11 |
| lizard 1.0 -35          |    56 MB/s |  1321 MB/s |    74563583 | 35.18 |
| lizard 1.0 -39          |  3.04 MB/s |  1443 MB/s |    69807522 | 32.94 |
| lizard 1.0 -40          |   206 MB/s |   880 MB/s |    80843049 | 38.14 |
| lizard 1.0 -42          |    90 MB/s |   938 MB/s |    73350988 | 34.61 |
| lizard 1.0 -45          |    11 MB/s |  1061 MB/s |    66692694 | 31.47 |
| lizard 1.0 -49          |  1.28 MB/s |  1013 MB/s |    60679215 | 28.63 |
| lz4 1.7.5               |   452 MB/s |  2244 MB/s |   100880800 | 47.60 |
| lz4fast 1.7.5 -3        |   522 MB/s |  2244 MB/s |   107066190 | 50.52 |
| lz4fast 1.7.5 -17       |   785 MB/s |  2601 MB/s |   131732802 | 62.15 |
| lz4hc 1.7.5 -1          |   100 MB/s |  2056 MB/s |    87591763 | 41.33 |
| lz4hc 1.7.5 -4          |    56 MB/s |  2200 MB/s |    79807909 | 37.65 |
| lz4hc 1.7.5 -9          |    23 MB/s |  2253 MB/s |    77892285 | 36.75 |
| lz4hc 1.7.5 -12         |  3.52 MB/s |  2281 MB/s |    77268977 | 36.46 |
| lzf 3.6 -0              |   244 MB/s |   550 MB/s |   105682088 | 49.86 |
| lzf 3.6 -1              |   251 MB/s |   565 MB/s |   102041092 | 48.14 |
| lzfse 2017-03-08        |    48 MB/s |   592 MB/s |    67624281 | 31.91 |
| lzg 1.0.8 -1            |    52 MB/s |   433 MB/s |   108553667 | 51.22 |
| lzg 1.0.8 -4            |    32 MB/s |   440 MB/s |    95930551 | 45.26 |
| lzg 1.0.8 -6            |    18 MB/s |   463 MB/s |    89490220 | 42.22 |
| lzg 1.0.8 -8            |  6.70 MB/s |   501 MB/s |    83606901 | 39.45 |
| lzham 1.0 -d26 -0       |  6.84 MB/s |   131 MB/s |    64089870 | 30.24 |
| lzham 1.0 -d26 -1       |  1.89 MB/s |   168 MB/s |    54740589 | 25.83 |
| lzjb 2010               |   218 MB/s |   402 MB/s |   122671613 | 57.88 |
| lzlib 1.8 -0            |    22 MB/s |    35 MB/s |    63847386 | 30.12 |
| lzlib 1.8 -3            |  4.35 MB/s |    42 MB/s |    56320674 | 26.57 |
| lzlib 1.8 -6            |  1.88 MB/s |    46 MB/s |    49777495 | 23.49 |
| lzlib 1.8 -9            |  1.18 MB/s |    45 MB/s |    48296889 | 22.79 |
| lzma 16.04 -0           |    18 MB/s |    47 MB/s |    64013917 | 30.20 |
| lzma 16.04 -2           |    16 MB/s |    56 MB/s |    58867911 | 27.77 |
| lzma 16.04 -4           |  8.36 MB/s |    60 MB/s |    57201645 | 26.99 |
| lzma 16.04 -5           |  2.00 MB/s |    66 MB/s |    49720569 | 23.46 |
| lzma 16.04 -9           |  1.55 MB/s |    67 MB/s |    48742901 | 23.00 |
| lzmat 1.01              |    24 MB/s |   288 MB/s |    76485353 | 36.09 |
| lzo1 2.09 -1            |   195 MB/s |   446 MB/s |   106474519 | 50.24 |
| lzo1 2.09 -99           |    81 MB/s |   474 MB/s |    94946129 | 44.80 |
| lzo1a 2.09 -1           |   188 MB/s |   509 MB/s |   104202251 | 49.16 |
| lzo1a 2.09 -99          |    80 MB/s |   535 MB/s |    92666265 | 43.72 |
| lzo1b 2.09 -1           |   168 MB/s |   544 MB/s |    97036087 | 45.78 |
| lzo1b 2.09 -3           |   165 MB/s |   560 MB/s |    94044578 | 44.37 |
| lzo1b 2.09 -6           |   162 MB/s |   564 MB/s |    91382355 | 43.12 |
| lzo1b 2.09 -9           |   121 MB/s |   559 MB/s |    89261884 | 42.12 |
| lzo1b 2.09 -99          |    79 MB/s |   565 MB/s |    85653376 | 40.41 |
| lzo1b 2.09 -999         |  9.02 MB/s |   630 MB/s |    76594292 | 36.14 |
| lzo1c 2.09 -1           |   174 MB/s |   570 MB/s |    99550904 | 46.97 |
| lzo1c 2.09 -3           |   167 MB/s |   581 MB/s |    96716153 | 45.63 |
| lzo1c 2.09 -6           |   145 MB/s |   578 MB/s |    93303623 | 44.02 |
| lzo1c 2.09 -9           |   111 MB/s |   575 MB/s |    91040386 | 42.95 |
| lzo1c 2.09 -99          |    77 MB/s |   579 MB/s |    88112288 | 41.57 |
| lzo1c 2.09 -999         |    15 MB/s |   611 MB/s |    80396741 | 37.93 |
| lzo1f 2.09 -1           |   159 MB/s |   504 MB/s |    99743329 | 47.06 |
| lzo1f 2.09 -999         |    13 MB/s |   526 MB/s |    80890206 | 38.17 |
| lzo1x 2.09 -1           |   394 MB/s |   551 MB/s |   100572537 | 47.45 |
| lzo1x 2.09 -11          |   424 MB/s |   560 MB/s |   106604629 | 50.30 |
| lzo1x 2.09 -12          |   418 MB/s |   550 MB/s |   103238859 | 48.71 |
| lzo1x 2.09 -15          |   406 MB/s |   549 MB/s |   101462094 | 47.87 |
| lzo1x 2.09 -999         |  5.30 MB/s |   528 MB/s |    75301903 | 35.53 |
| lzo1y 2.09 -1           |   397 MB/s |   556 MB/s |   101258318 | 47.78 |
| lzo1y 2.09 -999         |  5.34 MB/s |   529 MB/s |    75503849 | 35.62 |
| lzo1z 2.09 -999         |  5.32 MB/s |   521 MB/s |    75061331 | 35.42 |
| lzo2a 2.09 -999         |    16 MB/s |   400 MB/s |    82809337 | 39.07 |
| lzrw 15-Jul-1991 -1     |   197 MB/s |   392 MB/s |   113761625 | 53.67 |
| lzrw 15-Jul-1991 -3     |   226 MB/s |   449 MB/s |   105424168 | 49.74 |
| lzrw 15-Jul-1991 -4     |   243 MB/s |   392 MB/s |   100131356 | 47.24 |
| lzrw 15-Jul-1991 -5     |   105 MB/s |   414 MB/s |    90818810 | 42.85 |
| lzsse2 2016-05-14 -1    |    12 MB/s |  1986 MB/s |    87976095 | 41.51 |
| lzsse2 2016-05-14 -6    |  5.77 MB/s |  2269 MB/s |    75837101 | 35.78 |
| lzsse2 2016-05-14 -12   |  5.61 MB/s |  2273 MB/s |    75829973 | 35.78 |
| lzsse2 2016-05-14 -16   |  5.58 MB/s |  2272 MB/s |    75829973 | 35.78 |
| lzsse4 2016-05-14 -1    |    11 MB/s |  2556 MB/s |    82542106 | 38.94 |
| lzsse4 2016-05-14 -6    |  6.44 MB/s |  2763 MB/s |    76118298 | 35.91 |
| lzsse4 2016-05-14 -12   |  6.29 MB/s |  2767 MB/s |    76113017 | 35.91 |
| lzsse4 2016-05-14 -16   |  6.30 MB/s |  2768 MB/s |    76113017 | 35.91 |
| lzsse8 2016-05-14 -1    |    10 MB/s |  2624 MB/s |    81866245 | 38.63 |
| lzsse8 2016-05-14 -6    |  6.22 MB/s |  2839 MB/s |    75469717 | 35.61 |
| lzsse8 2016-05-14 -12   |  6.08 MB/s |  2842 MB/s |    75464339 | 35.61 |
| lzsse8 2016-05-14 -16   |  6.08 MB/s |  2840 MB/s |    75464339 | 35.61 |
| lzvn 2017-03-08         |    43 MB/s |   791 MB/s |    80814609 | 38.13 |
| pithy 2011-12-24 -0     |   384 MB/s |  1221 MB/s |   103072463 | 48.63 |
| pithy 2011-12-24 -3     |   352 MB/s |  1222 MB/s |    97255186 | 45.89 |
| pithy 2011-12-24 -6     |   295 MB/s |  1268 MB/s |    92090898 | 43.45 |
| pithy 2011-12-24 -9     |   257 MB/s |  1263 MB/s |    90360813 | 42.63 |
| quicklz 1.5.0 -1        |   346 MB/s |   435 MB/s |    94720562 | 44.69 |
| quicklz 1.5.0 -2        |   176 MB/s |   414 MB/s |    84555627 | 39.89 |
| quicklz 1.5.0 -3        |    42 MB/s |   722 MB/s |    81822241 | 38.60 |
| shrinker 0.1            |   698 MB/s |  1839 MB/s |   172535778 | 81.40 |
| slz_zlib 1.0.0 -1       |   200 MB/s |   228 MB/s |    99657958 | 47.02 |
| slz_zlib 1.0.0 -2       |   195 MB/s |   238 MB/s |    96863094 | 45.70 |
| slz_zlib 1.0.0 -3       |   192 MB/s |   237 MB/s |    96187780 | 45.38 |
| snappy 1.1.4            |   327 MB/s |  1075 MB/s |   102146767 | 48.19 |
| tornado 0.6a -1         |   233 MB/s |   334 MB/s |   107381846 | 50.66 |
| tornado 0.6a -2         |   180 MB/s |   301 MB/s |    90076660 | 42.50 |
| tornado 0.6a -3         |   119 MB/s |   188 MB/s |    72662044 | 34.28 |
| tornado 0.6a -4         |    91 MB/s |   197 MB/s |    70513617 | 33.27 |
| tornado 0.6a -5         |    32 MB/s |   129 MB/s |    64129604 | 30.26 |
| tornado 0.6a -6         |    24 MB/s |   133 MB/s |    62364583 | 29.42 |
| tornado 0.6a -7         |    11 MB/s |   135 MB/s |    59026325 | 27.85 |
| tornado 0.6a -10        |  3.53 MB/s |   136 MB/s |    57588241 | 27.17 |
| tornado 0.6a -13        |  4.27 MB/s |   141 MB/s |    55614072 | 26.24 |
| tornado 0.6a -16        |  1.48 MB/s |   145 MB/s |    53257046 | 25.13 |
| ucl_nrv2b 1.03 -1       |    34 MB/s |   231 MB/s |    81703168 | 38.55 |
| ucl_nrv2b 1.03 -6       |    12 MB/s |   263 MB/s |    73902185 | 34.87 |
| ucl_nrv2b 1.03 -9       |  1.35 MB/s |   284 MB/s |    71031195 | 33.51 |
| ucl_nrv2d 1.03 -1       |    34 MB/s |   238 MB/s |    81461976 | 38.43 |
| ucl_nrv2d 1.03 -6       |    13 MB/s |   270 MB/s |    73757673 | 34.80 |
| ucl_nrv2d 1.03 -9       |  1.37 MB/s |   292 MB/s |    70053895 | 33.05 |
| ucl_nrv2e 1.03 -1       |    34 MB/s |   229 MB/s |    81195560 | 38.31 |
| ucl_nrv2e 1.03 -6       |    12 MB/s |   262 MB/s |    73302012 | 34.58 |
| ucl_nrv2e 1.03 -9       |  1.39 MB/s |   284 MB/s |    69645134 | 32.86 |
| wflz 2015-09-16         |   189 MB/s |   781 MB/s |   109605264 | 51.71 |
| xpack 2016-06-02 -1     |    98 MB/s |   506 MB/s |    71090065 | 33.54 |
| xpack 2016-06-02 -6     |    29 MB/s |   626 MB/s |    62213845 | 29.35 |
| xpack 2016-06-02 -9     |    11 MB/s |   644 MB/s |    61240928 | 28.89 |
| xz 5.2.3 -0             |    15 MB/s |    44 MB/s |    62579435 | 29.53 |
| xz 5.2.3 -3             |  4.18 MB/s |    55 MB/s |    55745125 | 26.30 |
| xz 5.2.3 -6             |  1.89 MB/s |    58 MB/s |    49195929 | 23.21 |
| xz 5.2.3 -9             |  1.70 MB/s |    56 MB/s |    48745306 | 23.00 |
| yalz77 2015-09-19 -1    |    71 MB/s |   358 MB/s |    93952728 | 44.33 |
| yalz77 2015-09-19 -4    |    33 MB/s |   356 MB/s |    87392632 | 41.23 |
| yalz77 2015-09-19 -8    |    19 MB/s |   351 MB/s |    85153287 | 40.18 |
| yalz77 2015-09-19 -12   |    14 MB/s |   344 MB/s |    84050625 | 39.66 |
| yappy 2014-03-22 -1     |   100 MB/s |  1817 MB/s |   105750956 | 49.89 |
| yappy 2014-03-22 -10    |    77 MB/s |  1916 MB/s |   100018673 | 47.19 |
| yappy 2014-03-22 -100   |    56 MB/s |  1933 MB/s |    98672514 | 46.56 |
| zlib 1.2.11 -1          |    66 MB/s |   250 MB/s |    77259029 | 36.45 |
| zlib 1.2.11 -6          |    20 MB/s |   267 MB/s |    68228431 | 32.19 |
| zlib 1.2.11 -9          |  8.30 MB/s |   269 MB/s |    67644548 | 31.92 |
| zling 2016-01-10 -0     |    38 MB/s |   134 MB/s |    63407921 | 29.92 |
| zling 2016-01-10 -1     |    36 MB/s |   136 MB/s |    62438620 | 29.46 |
| zling 2016-01-10 -2     |    32 MB/s |   136 MB/s |    61917662 | 29.21 |
| zling 2016-01-10 -3     |    28 MB/s |   137 MB/s |    61384151 | 28.96 |
| zling 2016-01-10 -4     |    25 MB/s |   137 MB/s |    60997465 | 28.78 |
| zstd 1.1.4 -1           |   242 MB/s |   636 MB/s |    73654014 | 34.75 |
| zstd 1.1.4 -2           |   185 MB/s |   587 MB/s |    70164775 | 33.10 |
| zstd 1.1.4 -5           |    88 MB/s |   553 MB/s |    64998793 | 30.67 |
| zstd 1.1.4 -8           |    30 MB/s |   609 MB/s |    61021141 | 28.79 |
| zstd 1.1.4 -11          |    15 MB/s |   603 MB/s |    59518174 | 28.08 |
| zstd 1.1.4 -15          |  4.78 MB/s |   626 MB/s |    58005265 | 27.37 |
| zstd 1.1.4 -18          |  2.75 MB/s |   573 MB/s |    55288461 | 26.09 |
| zstd 1.1.4 -22          |  1.39 MB/s |   459 MB/s |    52718819 | 24.87 |

