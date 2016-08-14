Introduction
-------------------------

lzbench is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors. It joins all compressors into a single exe. 
At the beginning an input file is read to memory. 
Then all compressors are used to compress and decompress the file and decompressed file is verified. 
This approach has a big advantage of using the same compiler with the same optimizations for all compressors. 
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).

|Branch      |Status   |
|------------|---------|
|master      | [![Build Status](https://travis-ci.org/inikep/lzbench.svg?branch=master)](https://travis-ci.org/inikep/lzbench) |
|dev         | [![Build Status](https://travis-ci.org/inikep/lzbench.svg?branch=dev)](https://travis-ci.org/inikep/lzbench) | 

|Branch      |Status   |
|------------|---------|
|master      | [![Build Status][travisMasterBadge]][travisLink] [![Build status][AppveyorMasterBadge]][AppveyorLink]  |
|dev         | [![Build Status][travisDevBadge]][travisLink]    [![Build status][AppveyorDevBadge]][AppveyorLink]     | 

[travisMasterBadge]: https://travis-ci.org/inikep/lzbench.svg?branch=master "Continuous Integration test suite"
[travisDevBadge]: https://travis-ci.org/inikep/lzbench.svg?branch=dev "Continuous Integration test suite"
[travisLink]: https://travis-ci.org/inikep/lzbench
[AppveyorMasterBadge]: https://ci.appveyor.com/api/projects/status/u7kjj8ino4gww40v/branch/master?svg=true "Visual test suite"
[AppveyorDevBadge]: https://ci.appveyor.com/api/projects/status/u7kjj8ino4gww40v/branch/dev?svg=true "Visual test suite"
[AppveyorLink]: https://ci.appveyor.com/project/inikep/lzbench


Usage
-------------------------

```
usage: lzbench [options] input_file [input_file2] [input_file3]

where [options] are:
 -bX  set block/chunk size to X KB (default = MIN(filesize,1747626 KB))
 -cX  sort results by column number X
 -eX  X = compressors separated by '/' with parameters specified after ','
 -iX  set min. number of compression iterations (default = 1)
 -jX  set min. number of decompression iterations (default = 1)
 -l   list of available compressors and aliases
 -oX  output text format 1=Markdown, 2=text, 3=CSV (default = 2)
 -pX  print time for all iterations: 1=fastest 2=average 3=median (default = 1)
 -r   disable real-time process priority
 -sX  use only compressors with compression speed over X MB (default = 0 MB)
 -tX  set min. time in seconds for compression (default = 1.0)
 -uX  set min. time in seconds for decompression (default = 0.5)
 -v   disable progress information
 -z   show (de)compression times instead of speed

Example usage:
  lzbench -ezstd filename = selects all levels of zstd
  lzbench -ebrotli,2,5/zstd filename = selects levels 2 & 5 of brotli and zstd
  lzbench -t3 -u5 fname = 3 sec compression and 5 sec decompression loops
  lzbench -t0 -u0 -i3 -j5 -elz5 fname = 3 compression and 5 decompression iter.
  lzbench -t0u0i3j5 -elz5 fname = the same as above with aggregated parameters
```


Compilation
-------------------------
For Linux/Unix/MacOS/MinGW (Windows):
```
make
```

For 32-bit compilation:
```
make BUILD_ARCH=32-bit

```

To remove one of compressors you can add `-DBENCH_REMOVE_XXX` to `DEFINES` in Makefile (e.g. `DEFINES += -DBENCH_REMOVE_LZ5` to remove LZ5). 
You also have to remove corresponding `*.o` files (e.g. `lz5/lz5.o` and `lz5/lz5hc.o`).

lzbench was tested with:
- Ubuntu: gcc 4.6.3, 4.8.4, 4.9.3, 5.3.0, 6.1.1 and clang 3.4, 3.5, 3.6, 3.8
- MacOS: Apple LLVM version 6.0
- MinGW (Windows): gcc 5.3.0, 4.9.3 32-bit, 4.8.3 32-bit

Supported compressors
-------------------------
**Warning**: some of the compressors listed here have security issues
and/or are no longer maintained.  For information about the security
of the various compressors, see the
[CompFuzz Results](https://github.com/nemequ/compfuzz/wiki/Results)
page.
```
blosclz 2015-11-10
brieflz 1.1.0
brotli 0.4.0
crush 1.0
csc 3.3
density 0.12.5 beta (WARNING: it contains bugs (shortened decompressed output))
fastlz 0.1
gipfeli 2015-11-30
glza 0.7.1
lz4/lz4hc r131
lz5/lz5hc v1.4.1
lzf 3.6
lzfse/lzvn 2016-06-19
lzg 1.0.8
lzham 1.0
lzjb 2010
lzlib 1.7
lzma 9.38
lzmat 1.01 (WARNING: it contains bugs (decompression error; returns 0); it can throw SEGFAULT compiled with gcc 4.9+ -O3)
lzo 2.09
lzrw 15-Jul-1991
lzsse 2016-05-14
pithy 2011-12-24 (WARNING: it contains bugs (decompression error; returns 0))
quicklz 1.5.0
shrinker 0.1 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
slz 1.0.0 (only a compressor, uses zlib for decompression)
snappy 1.1.3
tornado 0.6a
ucl 1.03
wflz 2015-09-16 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
xpack 2016-06-02
xz 5.2.2
yalz77 2015-09-19
yappy 2014-03-22 (WARNING: fails to decompress properly on ARM)
zlib 1.2.8
zling 2016-04-10 (according to the author using libzling in a production environment is not a good idea)
zstd 0.8.0
```


Benchmarks
-------------------------

The following results are obtained with lzbench ("-t16 -u16 -eall") using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 5.3.0)
with ["silesia.tar"](https://drive.google.com/file/d/0BwX7dtyRLxThenZpYU9zLTZhR1k/view?usp=sharing) which contains tarred files from [Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).
The results sorted by ratio are available [here](lzbench13_sorted.md).

| Compressor name         | Compression| Decompress.| Compr. size | Ratio |
| ---------------         | -----------| -----------| ----------- | ----- |
| memcpy                  |  7332 MB/s |  8719 MB/s |   211947520 |100.00 |
| blosclz 2015-11-10 -1   |   883 MB/s |  5981 MB/s |   211768481 | 99.92 |
| blosclz 2015-11-10 -3   |   477 MB/s |  5267 MB/s |   204507781 | 96.49 |
| blosclz 2015-11-10 -6   |   227 MB/s |   873 MB/s |   113322667 | 53.47 |
| blosclz 2015-11-10 -9   |   214 MB/s |   664 MB/s |   102817442 | 48.51 |
| brieflz 1.1.0           |   101 MB/s |   158 MB/s |    81990651 | 38.68 |
| brotli 0.4.0 -0         |   212 MB/s |   253 MB/s |    78227142 | 36.91 |
| brotli 0.4.0 -2         |    98 MB/s |   297 MB/s |    68088509 | 32.13 |
| brotli 0.4.0 -5         |    23 MB/s |   324 MB/s |    60789033 | 28.68 |
| brotli 0.4.0 -8         |  5.41 MB/s |   339 MB/s |    57379199 | 27.07 |
| brotli 0.4.0 -11        |  0.35 MB/s |   275 MB/s |    51153776 | 24.14 |
| crush 1.0 -0            |    31 MB/s |   252 MB/s |    73064603 | 34.47 |
| crush 1.0 -1            |  3.27 MB/s |   288 MB/s |    66494412 | 31.37 |
| crush 1.0 -2            |  0.38 MB/s |   295 MB/s |    63746223 | 30.08 |
| csc 3.3 -1              |    15 MB/s |    49 MB/s |    56201092 | 26.52 |
| csc 3.3 -3              |  6.41 MB/s |    48 MB/s |    53477914 | 25.23 |
| csc 3.3 -5              |  2.58 MB/s |    52 MB/s |    49801577 | 23.50 |
| density 0.12.5 beta -1  |   835 MB/s |  1140 MB/s |   133085162 | 62.79 |
| density 0.12.5 beta -2  |   490 MB/s |   674 MB/s |   101706226 | 47.99 |
| density 0.12.5 beta -3  |   263 MB/s |   254 MB/s |    87622980 | 41.34 |
| fastlz 0.1 -1           |   232 MB/s |   475 MB/s |   104628084 | 49.37 |
| fastlz 0.1 -2           |   245 MB/s |   456 MB/s |   100906072 | 47.61 |
| gipfeli 2015-11-30      |   231 MB/s |   463 MB/s |    87931759 | 41.49 |
| lz4 r131                |   442 MB/s |  2242 MB/s |   100880800 | 47.60 |
| lz4fast r131 -3         |   509 MB/s |  2254 MB/s |   107066190 | 50.52 |
| lz4fast r131 -17        |   770 MB/s |  2622 MB/s |   131732802 | 62.15 |
| lz4hc r131 -1           |   101 MB/s |  2040 MB/s |    89227392 | 42.10 |
| lz4hc r131 -4           |    57 MB/s |  2177 MB/s |    80485954 | 37.97 |
| lz4hc r131 -9           |    23 MB/s |  2235 MB/s |    77919206 | 36.76 |
| lz4hc r131 -12          |    18 MB/s |  2244 MB/s |    77852851 | 36.73 |
| lz4hc r131 -16          |    12 MB/s |  2245 MB/s |    77841796 | 36.73 |
| lz5 1.4.1               |   210 MB/s |   728 MB/s |    88216194 | 41.62 |
| lz5hc 1.4.1 -1          |   440 MB/s |  1286 MB/s |   113538427 | 53.57 |
| lz5hc 1.4.1 -4          |   149 MB/s |   941 MB/s |    86503541 | 40.81 |
| lz5hc 1.4.1 -9          |    20 MB/s |   815 MB/s |    74228639 | 35.02 |
| lz5hc 1.4.1 -12         |  7.96 MB/s |   772 MB/s |    69485691 | 32.78 |
| lz5hc 1.4.1 -15         |  1.65 MB/s |   684 MB/s |    65029194 | 30.68 |
| lzf 3.6 -0              |   248 MB/s |   548 MB/s |   105682088 | 49.86 |
| lzf 3.6 -1              |   254 MB/s |   564 MB/s |   102041092 | 48.14 |
| lzfse 2016-06-19        |    47 MB/s |   586 MB/s |    67624281 | 31.91 |
| lzg 1.0.8 -1            |    57 MB/s |   421 MB/s |   108553667 | 51.22 |
| lzg 1.0.8 -4            |    35 MB/s |   424 MB/s |    95930551 | 45.26 |
| lzg 1.0.8 -6            |    19 MB/s |   445 MB/s |    89490220 | 42.22 |
| lzg 1.0.8 -8            |  6.81 MB/s |   485 MB/s |    83606901 | 39.45 |
| lzham 1.0 -d26 -0       |  6.30 MB/s |   141 MB/s |    64089870 | 30.24 |
| lzham 1.0 -d26 -1       |  1.94 MB/s |   179 MB/s |    54740589 | 25.83 |
| lzjb 2010               |   222 MB/s |   408 MB/s |   122671613 | 57.88 |
| lzlib 1.7 -0            |    23 MB/s |    37 MB/s |    63847386 | 30.12 |
| lzlib 1.7 -3            |  4.64 MB/s |    43 MB/s |    56320674 | 26.57 |
| lzlib 1.7 -6            |  1.98 MB/s |    45 MB/s |    49777495 | 23.49 |
| lzlib 1.7 -9            |  1.23 MB/s |    47 MB/s |    48296889 | 22.79 |
| lzma 9.38 -0            |    18 MB/s |    47 MB/s |    64013917 | 30.20 |
| lzma 9.38 -2            |    15 MB/s |    56 MB/s |    58867911 | 27.77 |
| lzma 9.38 -4            |  9.06 MB/s |    59 MB/s |    57201645 | 26.99 |
| lzma 9.38 -5            |  2.12 MB/s |    65 MB/s |    49720569 | 23.46 |
| lzmat 1.01              |    25 MB/s |   290 MB/s |    76485353 | 36.09 |
| lzo1 2.09 -1            |   197 MB/s |   435 MB/s |   106474519 | 50.24 |
| lzo1 2.09 -99           |    83 MB/s |   458 MB/s |    94946129 | 44.80 |
| lzo1a 2.09 -1           |   195 MB/s |   508 MB/s |   104202251 | 49.16 |
| lzo1a 2.09 -99          |    83 MB/s |   535 MB/s |    92666265 | 43.72 |
| lzo1b 2.09 -1           |   168 MB/s |   550 MB/s |    97036087 | 45.78 |
| lzo1b 2.09 -3           |   166 MB/s |   565 MB/s |    94044578 | 44.37 |
| lzo1b 2.09 -6           |   166 MB/s |   568 MB/s |    91382355 | 43.12 |
| lzo1b 2.09 -9           |   122 MB/s |   563 MB/s |    89261884 | 42.12 |
| lzo1b 2.09 -99          |    81 MB/s |   569 MB/s |    85653376 | 40.41 |
| lzo1b 2.09 -999         |  9.29 MB/s |   631 MB/s |    76594292 | 36.14 |
| lzo1c 2.09 -1           |   173 MB/s |   571 MB/s |    99550904 | 46.97 |
| lzo1c 2.09 -3           |   168 MB/s |   583 MB/s |    96716153 | 45.63 |
| lzo1c 2.09 -6           |   147 MB/s |   583 MB/s |    93303623 | 44.02 |
| lzo1c 2.09 -9           |   113 MB/s |   579 MB/s |    91040386 | 42.95 |
| lzo1c 2.09 -99          |    76 MB/s |   582 MB/s |    88112288 | 41.57 |
| lzo1c 2.09 -999         |    15 MB/s |   614 MB/s |    80396741 | 37.93 |
| lzo1f 2.09 -1           |   160 MB/s |   495 MB/s |    99743329 | 47.06 |
| lzo1f 2.09 -999         |    13 MB/s |   505 MB/s |    80890206 | 38.17 |
| lzo1x 2.09 -1           |   402 MB/s |   553 MB/s |   100572537 | 47.45 |
| lzo1x 2.09 -11          |   432 MB/s |   563 MB/s |   106604629 | 50.30 |
| lzo1x 2.09 -12          |   424 MB/s |   554 MB/s |   103238859 | 48.71 |
| lzo1x 2.09 -15          |   414 MB/s |   552 MB/s |   101462094 | 47.87 |
| lzo1x 2.09 -999         |  5.68 MB/s |   529 MB/s |    75301903 | 35.53 |
| lzo1y 2.09 -1           |   403 MB/s |   554 MB/s |   101258318 | 47.78 |
| lzo1y 2.09 -999         |  5.78 MB/s |   529 MB/s |    75503849 | 35.62 |
| lzo1z 2.09 -999         |  5.64 MB/s |   513 MB/s |    75061331 | 35.42 |
| lzo2a 2.09 -999         |    16 MB/s |   401 MB/s |    82809337 | 39.07 |
| lzrw 15-Jul-1991 -1     |   197 MB/s |   421 MB/s |   113761625 | 53.67 |
| lzrw 15-Jul-1991 -2     |   202 MB/s |   420 MB/s |   112344608 | 53.01 |
| lzrw 15-Jul-1991 -3     |   229 MB/s |   442 MB/s |   105424168 | 49.74 |
| lzrw 15-Jul-1991 -4     |   244 MB/s |   404 MB/s |   100131356 | 47.24 |
| lzrw 15-Jul-1991 -5     |   109 MB/s |   406 MB/s |    90818810 | 42.85 |
| lzsse2 2016-05-14 -1    |    12 MB/s |  1961 MB/s |    87976095 | 41.51 |
| lzsse2 2016-05-14 -6    |  5.87 MB/s |  2240 MB/s |    75837101 | 35.78 |
| lzsse2 2016-05-14 -12   |  5.69 MB/s |  2238 MB/s |    75829973 | 35.78 |
| lzsse2 2016-05-14 -16   |  5.69 MB/s |  2244 MB/s |    75829973 | 35.78 |
| lzsse4 2016-05-14 -1    |    12 MB/s |  2482 MB/s |    82542106 | 38.94 |
| lzsse4 2016-05-14 -6    |  6.59 MB/s |  2687 MB/s |    76118298 | 35.91 |
| lzsse4 2016-05-14 -12   |  6.41 MB/s |  2685 MB/s |    76113017 | 35.91 |
| lzsse4 2016-05-14 -16   |  6.43 MB/s |  2682 MB/s |    76113017 | 35.91 |
| lzsse8 2016-05-14 -1    |    11 MB/s |  2601 MB/s |    81866245 | 38.63 |
| lzsse8 2016-05-14 -6    |  6.00 MB/s |  2840 MB/s |    75469717 | 35.61 |
| lzsse8 2016-05-14 -12   |  6.16 MB/s |  2838 MB/s |    75464339 | 35.61 |
| lzsse8 2016-05-14 -16   |  6.16 MB/s |  2813 MB/s |    75464339 | 35.61 |
| lzvn 2016-06-19         |    44 MB/s |   778 MB/s |    80814609 | 38.13 |
| pithy 2011-12-24 -0     |   378 MB/s |  1227 MB/s |   103072463 | 48.63 |
| pithy 2011-12-24 -3     |   348 MB/s |  1225 MB/s |    97255186 | 45.89 |
| pithy 2011-12-24 -6     |   295 MB/s |  1274 MB/s |    92090898 | 43.45 |
| pithy 2011-12-24 -9     |   254 MB/s |  1273 MB/s |    90360813 | 42.63 |
| quicklz 1.5.0 -1        |   342 MB/s |   426 MB/s |    94720562 | 44.69 |
| quicklz 1.5.0 -2        |   178 MB/s |   415 MB/s |    84555627 | 39.89 |
| quicklz 1.5.0 -3        |    44 MB/s |   721 MB/s |    81822241 | 38.60 |
| shrinker 0.1            |   703 MB/s |  1853 MB/s |   172535778 | 81.40 |
| snappy 1.1.3            |   317 MB/s |  1059 MB/s |   101382606 | 47.83 |
| slz_zlib 1.0.0 -1       |   200 MB/s |   235 MB/s |    99657958 | 47.02 |
| slz_zlib 1.0.0 -2       |   194 MB/s |   236 MB/s |    96863094 | 45.70 |
| slz_zlib 1.0.0 -3       |   191 MB/s |   237 MB/s |    96187780 | 45.38 |
| tornado 0.6a -1         |   233 MB/s |   351 MB/s |   107381846 | 50.66 |
| tornado 0.6a -2         |   178 MB/s |   312 MB/s |    90076660 | 42.50 |
| tornado 0.6a -3         |   116 MB/s |   188 MB/s |    72662044 | 34.28 |
| tornado 0.6a -4         |    91 MB/s |   197 MB/s |    70513617 | 33.27 |
| tornado 0.6a -5         |    35 MB/s |   130 MB/s |    64129604 | 30.26 |
| tornado 0.6a -6         |    26 MB/s |   133 MB/s |    62364583 | 29.42 |
| tornado 0.6a -7         |    12 MB/s |   138 MB/s |    59026325 | 27.85 |
| tornado 0.6a -10        |  3.75 MB/s |   140 MB/s |    57588241 | 27.17 |
| tornado 0.6a -13        |  4.66 MB/s |   141 MB/s |    55614072 | 26.24 |
| tornado 0.6a -16        |  1.51 MB/s |   146 MB/s |    53257046 | 25.13 |
| ucl_nrv2b 1.03 -1       |    34 MB/s |   228 MB/s |    81703168 | 38.55 |
| ucl_nrv2b 1.03 -6       |    13 MB/s |   258 MB/s |    73902185 | 34.87 |
| ucl_nrv2b 1.03 -9       |  1.28 MB/s |   278 MB/s |    71031195 | 33.51 |
| ucl_nrv2d 1.03 -1       |    35 MB/s |   227 MB/s |    81461976 | 38.43 |
| ucl_nrv2d 1.03 -6       |    12 MB/s |   257 MB/s |    73757673 | 34.80 |
| ucl_nrv2d 1.03 -9       |  1.29 MB/s |   278 MB/s |    70053895 | 33.05 |
| ucl_nrv2e 1.03 -1       |    35 MB/s |   232 MB/s |    81195560 | 38.31 |
| ucl_nrv2e 1.03 -6       |    13 MB/s |   264 MB/s |    73302012 | 34.58 |
| ucl_nrv2e 1.03 -9       |  1.29 MB/s |   285 MB/s |    69645134 | 32.86 |
| wflz 2015-09-16         |   184 MB/s |   776 MB/s |   109605264 | 51.71 |
| xpack 2016-06-02 -1     |    96 MB/s |   514 MB/s |    71090065 | 33.54 |
| xpack 2016-06-02 -6     |    30 MB/s |   633 MB/s |    62213845 | 29.35 |
| xpack 2016-06-02 -9     |    12 MB/s |   651 MB/s |    61240928 | 28.89 |
| xz 5.2.2 -0             |    16 MB/s |    44 MB/s |    62579435 | 29.53 |
| xz 5.2.2 -3             |  4.56 MB/s |    55 MB/s |    55745125 | 26.30 |
| xz 5.2.2 -6             |  1.98 MB/s |    58 MB/s |    49195929 | 23.21 |
| xz 5.2.2 -9             |  1.80 MB/s |    59 MB/s |    48745306 | 23.00 |
| yalz77 2015-09-19 -1    |    79 MB/s |   340 MB/s |    93952728 | 44.33 |
| yalz77 2015-09-19 -4    |    37 MB/s |   340 MB/s |    87392632 | 41.23 |
| yalz77 2015-09-19 -8    |    22 MB/s |   336 MB/s |    85153287 | 40.18 |
| yalz77 2015-09-19 -12   |    17 MB/s |   330 MB/s |    84050625 | 39.66 |
| yappy 2014-03-22 -1     |    97 MB/s |  1807 MB/s |   105750956 | 49.89 |
| yappy 2014-03-22 -10    |    77 MB/s |  1915 MB/s |   100018673 | 47.19 |
| yappy 2014-03-22 -100   |    58 MB/s |  1928 MB/s |    98672514 | 46.56 |
| zlib 1.2.8 -1           |    65 MB/s |   248 MB/s |    77259029 | 36.45 |
| zlib 1.2.8 -6           |    20 MB/s |   266 MB/s |    68228431 | 32.19 |
| zlib 1.2.8 -9           |  8.36 MB/s |   268 MB/s |    67644548 | 31.92 |
| zling 2016-01-10 -0     |    41 MB/s |   136 MB/s |    63407921 | 29.92 |
| zling 2016-01-10 -1     |    37 MB/s |   136 MB/s |    62438620 | 29.46 |
| zling 2016-01-10 -2     |    34 MB/s |   137 MB/s |    61917662 | 29.21 |
| zling 2016-01-10 -3     |    30 MB/s |   138 MB/s |    61384151 | 28.96 |
| zling 2016-01-10 -4     |    27 MB/s |   137 MB/s |    60997465 | 28.78 |
| zstd 0.8.0 -1           |   238 MB/s |   629 MB/s |    73659471 | 34.75 |
| zstd 0.8.0 -2           |   183 MB/s |   560 MB/s |    70168958 | 33.11 |
| zstd 0.8.0 -5           |    88 MB/s |   511 MB/s |    65002227 | 30.67 |
| zstd 0.8.0 -8           |    32 MB/s |   554 MB/s |    61026456 | 28.79 |
| zstd 0.8.0 -11          |    16 MB/s |   547 MB/s |    59523199 | 28.08 |
| zstd 0.8.0 -15          |  5.22 MB/s |   572 MB/s |    58007769 | 27.37 |
| zstd 0.8.0 -18          |  3.10 MB/s |   492 MB/s |    55540622 | 26.20 |
| zstd 0.8.0 -22          |  1.54 MB/s |   464 MB/s |    52787120 | 24.91 |

