Introduction
-------------------------

lzbench is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors. It joins all compressors into a single exe. 
At the beginning an input file is read to memory. 
Then all compressors are used to compress and decompress the file and decompressed file is verified. 
This approach has a big advantage of using the same compiler with the same optimizations for all compressors. 
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).

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
  lzbench -t0 -u0 -i3 -j5 -elz5 fname = 3 compression and 5 decompression iter.
  lzbench -t0u0i3j5 -elz5 fname = the same as above with aggregated parameters
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

To remove one of compressors you can add `-DBENCH_REMOVE_XXX` to `DEFINES` in Makefile (e.g. `DEFINES += -DBENCH_REMOVE_LZ5` to remove LZ5). 
You also have to remove corresponding `*.o` files (e.g. `lz5/lz5.o` and `lz5/lz5hc.o`).

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
brotli 0.5.2
crush 1.0
csc 2016-10-13 (WARNING: it can throw SEGFAULT compiled with Apple LLVM version 7.3.0 (clang-703.0.31))
density 0.12.5 beta (WARNING: it contains bugs (shortened decompressed output))
fastlz 0.1
gipfeli 2016-07-13
glza 0.7.1
libdeflate v0.6
lz4/lz4hc v1.7.3
lz5 v2.0 RC2
lzf 3.6
lzfse/lzvn 2016-08-16
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
zstd 1.1.1
```


Benchmarks
-------------------------

The following results are obtained with lzbench ("-t16,16 -eall") using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 6.2.0)
with ["silesia.tar"](https://drive.google.com/file/d/0BwX7dtyRLxThenZpYU9zLTZhR1k/view?usp=sharing) which contains tarred files from [Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).
The results sorted by ratio are available [here](lzbench15_sorted.md).

| Compressor name         | Compression| Decompress.| Compr. size | Ratio |
| ---------------         | -----------| -----------| ----------- | ----- |
| memcpy                  |  7332 MB/s |  8719 MB/s |   211947520 |100.00 |
| blosclz 2015-11-10 -1   |   892 MB/s |  5679 MB/s |   211768481 | 99.92 |
| blosclz 2015-11-10 -3   |   487 MB/s |  5047 MB/s |   204507781 | 96.49 |
| blosclz 2015-11-10 -6   |   231 MB/s |   907 MB/s |   113322667 | 53.47 |
| blosclz 2015-11-10 -9   |   217 MB/s |   688 MB/s |   102817442 | 48.51 |
| brieflz 1.1.0           |   104 MB/s |   158 MB/s |    81990651 | 38.68 |
| brotli 0.5.2 -0         |   217 MB/s |   244 MB/s |    78226979 | 36.91 |
| brotli 0.5.2 -2         |    96 MB/s |   283 MB/s |    68066621 | 32.11 |
| brotli 0.5.2 -5         |    24 MB/s |   312 MB/s |    60801716 | 28.69 |
| brotli 0.5.2 -8         |  5.56 MB/s |   324 MB/s |    57382470 | 27.07 |
| brotli 0.5.2 -11        |  0.39 MB/s |   266 MB/s |    51138054 | 24.13 |
| crush 1.0 -0            |    32 MB/s |   256 MB/s |    73064603 | 34.47 |
| crush 1.0 -1            |  3.28 MB/s |   291 MB/s |    66494412 | 31.37 |
| crush 1.0 -2            |  0.39 MB/s |   296 MB/s |    63746223 | 30.08 |
| csc 2016-10-13 -1       |    14 MB/s |    48 MB/s |    56201092 | 26.52 |
| csc 2016-10-13 -3       |  5.98 MB/s |    46 MB/s |    53477914 | 25.23 |
| csc 2016-10-13 -5       |  2.49 MB/s |    50 MB/s |    49801577 | 23.50 |
| density 0.12.5 beta -1  |   806 MB/s |  1028 MB/s |   133085162 | 62.79 |
| density 0.12.5 beta -2  |   479 MB/s |   660 MB/s |   101706226 | 47.99 |
| density 0.12.5 beta -3  |   254 MB/s |   234 MB/s |    87622980 | 41.34 |
| fastlz 0.1 -1           |   237 MB/s |   462 MB/s |   104628084 | 49.37 |
| fastlz 0.1 -2           |   240 MB/s |   468 MB/s |   100906072 | 47.61 |
| gipfeli 2016-07-13      |   235 MB/s |   454 MB/s |    87931759 | 41.49 |
| libdeflate 16-08-29 -1  |   117 MB/s |   568 MB/s |    73318371 | 34.59 |
| libdeflate 16-08-29 -3  |    96 MB/s |   602 MB/s |    70668968 | 33.34 |
| libdeflate 16-08-29 -6  |    63 MB/s |   610 MB/s |    67928189 | 32.05 |
| libdeflate 16-08-29 -9  |    10 MB/s |   587 MB/s |    65701539 | 31.00 |
| libdeflate 16-08-29 -12 |  4.57 MB/s |   586 MB/s |    64801629 | 30.57 |
| lz4 1.7.3               |   440 MB/s |  2318 MB/s |   100880800 | 47.60 |
| lz4fast 1.7.3 -17       |   781 MB/s |  2696 MB/s |   131732802 | 62.15 |
| lz4fast 1.7.3 -3        |   516 MB/s |  2317 MB/s |   107066190 | 50.52 |
| lz4hc 1.7.3 -1          |    98 MB/s |  2121 MB/s |    87591763 | 41.33 |
| lz4hc 1.7.3 -4          |    55 MB/s |  2259 MB/s |    79807909 | 37.65 |
| lz4hc 1.7.3 -9          |    22 MB/s |  2315 MB/s |    77892285 | 36.75 |
| lz4hc 1.7.3 -12         |    17 MB/s |  2323 MB/s |    77849762 | 36.73 |
| lz4hc 1.7.3 -16         |    10 MB/s |  2323 MB/s |    77841782 | 36.73 |
| lz5 2.0 RC2 -10         |   346 MB/s |  2610 MB/s |   103402971 | 48.79 |
| lz5 2.0 RC2 -12         |   103 MB/s |  2458 MB/s |    86232422 | 40.69 |
| lz5 2.0 RC2 -15         |    50 MB/s |  2552 MB/s |    81187330 | 38.31 |
| lz5 2.0 RC2 -19         |  3.04 MB/s |  2497 MB/s |    77416400 | 36.53 |
| lz5 2.0 RC2 -20         |   157 MB/s |  1795 MB/s |    89239174 | 42.10 |
| lz5 2.0 RC2 -22         |    30 MB/s |  1778 MB/s |    81097176 | 38.26 |
| lz5 2.0 RC2 -25         |  6.63 MB/s |  1734 MB/s |    74503695 | 35.15 |
| lz5 2.0 RC2 -29         |  1.37 MB/s |  1634 MB/s |    68694227 | 32.41 |
| lz5 2.0 RC2 -30         |   246 MB/s |   909 MB/s |    85727429 | 40.45 |
| lz5 2.0 RC2 -32         |    94 MB/s |  1244 MB/s |    76929454 | 36.30 |
| lz5 2.0 RC2 -35         |    47 MB/s |  1435 MB/s |    73850400 | 34.84 |
| lz5 2.0 RC2 -39         |  2.94 MB/s |  1502 MB/s |    69807522 | 32.94 |
| lz5 2.0 RC2 -40         |   126 MB/s |   961 MB/s |    76100661 | 35.91 |
| lz5 2.0 RC2 -42         |    28 MB/s |  1101 MB/s |    70955653 | 33.48 |
| lz5 2.0 RC2 -45         |  6.25 MB/s |  1073 MB/s |    65413061 | 30.86 |
| lz5 2.0 RC2 -49         |  1.27 MB/s |  1064 MB/s |    60679215 | 28.63 |
| lzf 3.6 -0              |   242 MB/s |   529 MB/s |   105682088 | 49.86 |
| lzf 3.6 -1              |   250 MB/s |   541 MB/s |   102041092 | 48.14 |
| lzfse 2016-08-16        |    46 MB/s |   561 MB/s |    67624281 | 31.91 |
| lzg 1.0.8 -1            |    56 MB/s |   434 MB/s |   108553667 | 51.22 |
| lzg 1.0.8 -4            |    34 MB/s |   440 MB/s |    95930551 | 45.26 |
| lzg 1.0.8 -6            |    19 MB/s |   465 MB/s |    89490220 | 42.22 |
| lzg 1.0.8 -8            |  6.66 MB/s |   503 MB/s |    83606901 | 39.45 |
| lzham 1.0 -d26 -0       |  6.94 MB/s |   135 MB/s |    64089870 | 30.24 |
| lzham 1.0 -d26 -1       |  1.95 MB/s |   176 MB/s |    54740589 | 25.83 |
| lzjb 2010               |   217 MB/s |   415 MB/s |   122671613 | 57.88 |
| lzlib 1.7 -0            |    22 MB/s |    36 MB/s |    63847386 | 30.12 |
| lzlib 1.7 -3            |  4.54 MB/s |    41 MB/s |    56320674 | 26.57 |
| lzlib 1.7 -6            |  1.97 MB/s |    45 MB/s |    49777495 | 23.49 |
| lzlib 1.7 -9            |  1.21 MB/s |    45 MB/s |    48296889 | 22.79 |
| lzma 9.38 -0            |    18 MB/s |    46 MB/s |    64013917 | 30.20 |
| lzma 9.38 -2            |    15 MB/s |    55 MB/s |    58867911 | 27.77 |
| lzma 9.38 -4            |  8.79 MB/s |    60 MB/s |    57201645 | 26.99 |
| lzma 9.38 -5            |  2.08 MB/s |    65 MB/s |    49720569 | 23.46 |
| lzmat 1.01              |    24 MB/s |   294 MB/s |    76485353 | 36.09 |
| lzo1 2.09 -1            |   197 MB/s |   439 MB/s |   106474519 | 50.24 |
| lzo1 2.09 -99           |    82 MB/s |   464 MB/s |    94946129 | 44.80 |
| lzo1a 2.09 -1           |   193 MB/s |   513 MB/s |   104202251 | 49.16 |
| lzo1a 2.09 -99          |    81 MB/s |   544 MB/s |    92666265 | 43.72 |
| lzo1b 2.09 -1           |   168 MB/s |   550 MB/s |    97036087 | 45.78 |
| lzo1b 2.09 -3           |   164 MB/s |   564 MB/s |    94044578 | 44.37 |
| lzo1b 2.09 -6           |   163 MB/s |   572 MB/s |    91382355 | 43.12 |
| lzo1b 2.09 -9           |   123 MB/s |   564 MB/s |    89261884 | 42.12 |
| lzo1b 2.09 -99          |    80 MB/s |   570 MB/s |    85653376 | 40.41 |
| lzo1b 2.09 -999         |  9.15 MB/s |   628 MB/s |    76594292 | 36.14 |
| lzo1c 2.09 -1           |   174 MB/s |   573 MB/s |    99550904 | 46.97 |
| lzo1c 2.09 -3           |   167 MB/s |   583 MB/s |    96716153 | 45.63 |
| lzo1c 2.09 -6           |   146 MB/s |   580 MB/s |    93303623 | 44.02 |
| lzo1c 2.09 -9           |   108 MB/s |   576 MB/s |    91040386 | 42.95 |
| lzo1c 2.09 -99          |    74 MB/s |   580 MB/s |    88112288 | 41.57 |
| lzo1c 2.09 -999         |    15 MB/s |   614 MB/s |    80396741 | 37.93 |
| lzo1f 2.09 -1           |   160 MB/s |   501 MB/s |    99743329 | 47.06 |
| lzo1f 2.09 -999         |    13 MB/s |   521 MB/s |    80890206 | 38.17 |
| lzo1x 2.09 -1           |   401 MB/s |   548 MB/s |   100572537 | 47.45 |
| lzo1x 2.09 -11          |   430 MB/s |   558 MB/s |   106604629 | 50.30 |
| lzo1x 2.09 -12          |   423 MB/s |   548 MB/s |   103238859 | 48.71 |
| lzo1x 2.09 -15          |   414 MB/s |   547 MB/s |   101462094 | 47.87 |
| lzo1x 2.09 -999         |  5.34 MB/s |   526 MB/s |    75301903 | 35.53 |
| lzo1y 2.09 -1           |   404 MB/s |   557 MB/s |   101258318 | 47.78 |
| lzo1y 2.09 -999         |  5.44 MB/s |   530 MB/s |    75503849 | 35.62 |
| lzo1z 2.09 -999         |  5.41 MB/s |   520 MB/s |    75061331 | 35.42 |
| lzo2a 2.09 -999         |    16 MB/s |   411 MB/s |    82809337 | 39.07 |
| lzrw 15-Jul-1991 -1     |   198 MB/s |   426 MB/s |   113761625 | 53.67 |
| lzrw 15-Jul-1991 -2     |   199 MB/s |   438 MB/s |   112344608 | 53.01 |
| lzrw 15-Jul-1991 -3     |   227 MB/s |   455 MB/s |   105424168 | 49.74 |
| lzrw 15-Jul-1991 -4     |   241 MB/s |   398 MB/s |   100131356 | 47.24 |
| lzrw 15-Jul-1991 -5     |   105 MB/s |   422 MB/s |    90818810 | 42.85 |
| lzsse2 2016-05-14 -1    |    11 MB/s |  2051 MB/s |    87976095 | 41.51 |
| lzsse2 2016-05-14 -6    |  5.72 MB/s |  2382 MB/s |    75837101 | 35.78 |
| lzsse2 2016-05-14 -12   |  5.61 MB/s |  2383 MB/s |    75829973 | 35.78 |
| lzsse2 2016-05-14 -16   |  5.63 MB/s |  2369 MB/s |    75829973 | 35.78 |
| lzsse4 2016-05-14 -1    |    12 MB/s |  2545 MB/s |    82542106 | 38.94 |
| lzsse4 2016-05-14 -6    |  6.44 MB/s |  2767 MB/s |    76118298 | 35.91 |
| lzsse4 2016-05-14 -12   |  6.40 MB/s |  2772 MB/s |    76113017 | 35.91 |
| lzsse4 2016-05-14 -16   |  6.30 MB/s |  2761 MB/s |    76113017 | 35.91 |
| lzsse8 2016-05-14 -1    |    11 MB/s |  2630 MB/s |    81866245 | 38.63 |
| lzsse8 2016-05-14 -6    |  6.19 MB/s |  2842 MB/s |    75469717 | 35.61 |
| lzsse8 2016-05-14 -12   |  6.00 MB/s |  2859 MB/s |    75464339 | 35.61 |
| lzsse8 2016-05-14 -16   |  6.05 MB/s |  2858 MB/s |    75464339 | 35.61 |
| lzvn 2016-08-16         |    41 MB/s |   803 MB/s |    80814609 | 38.13 |
| pithy 2011-12-24 -0     |   384 MB/s |  1229 MB/s |   103072463 | 48.63 |
| pithy 2011-12-24 -3     |   354 MB/s |  1235 MB/s |    97255186 | 45.89 |
| pithy 2011-12-24 -6     |   297 MB/s |  1285 MB/s |    92090898 | 43.45 |
| pithy 2011-12-24 -9     |   257 MB/s |  1269 MB/s |    90360813 | 42.63 |
| quicklz 1.5.0 -1        |   349 MB/s |   439 MB/s |    94720562 | 44.69 |
| quicklz 1.5.0 -2        |   177 MB/s |   410 MB/s |    84555627 | 39.89 |
| quicklz 1.5.0 -3        |    43 MB/s |   720 MB/s |    81822241 | 38.60 |
| shrinker 0.1            |   698 MB/s |  1843 MB/s |   172535778 | 81.40 |
| slz_zlib 1.0.0 -1       |   198 MB/s |   232 MB/s |    99657958 | 47.02 |
| slz_zlib 1.0.0 -2       |   192 MB/s |   234 MB/s |    96863094 | 45.70 |
| slz_zlib 1.0.0 -3       |   189 MB/s |   235 MB/s |    96187780 | 45.38 |
| snappy 1.1.3            |   314 MB/s |   926 MB/s |   101382606 | 47.83 |
| tornado 0.6a -1         |   228 MB/s |   330 MB/s |   107381846 | 50.66 |
| tornado 0.6a -2         |   179 MB/s |   294 MB/s |    90076660 | 42.50 |
| tornado 0.6a -3         |   118 MB/s |   188 MB/s |    72662044 | 34.28 |
| tornado 0.6a -4         |    91 MB/s |   196 MB/s |    70513617 | 33.27 |
| tornado 0.6a -5         |    34 MB/s |   130 MB/s |    64129604 | 30.26 |
| tornado 0.6a -6         |    25 MB/s |   133 MB/s |    62364583 | 29.42 |
| tornado 0.6a -7         |    12 MB/s |   135 MB/s |    59026325 | 27.85 |
| tornado 0.6a -10        |  3.73 MB/s |   136 MB/s |    57588241 | 27.17 |
| tornado 0.6a -13        |  4.37 MB/s |   140 MB/s |    55614072 | 26.24 |
| tornado 0.6a -16        |  1.56 MB/s |   146 MB/s |    53257046 | 25.13 |
| ucl_nrv2b 1.03 -1       |    33 MB/s |   232 MB/s |    81703168 | 38.55 |
| ucl_nrv2b 1.03 -6       |    13 MB/s |   262 MB/s |    73902185 | 34.87 |
| ucl_nrv2b 1.03 -9       |  1.29 MB/s |   287 MB/s |    71031195 | 33.51 |
| ucl_nrv2d 1.03 -1       |    33 MB/s |   239 MB/s |    81461976 | 38.43 |
| ucl_nrv2d 1.03 -6       |    13 MB/s |   270 MB/s |    73757673 | 34.80 |
| ucl_nrv2d 1.03 -9       |  1.30 MB/s |   293 MB/s |    70053895 | 33.05 |
| ucl_nrv2e 1.03 -1       |    33 MB/s |   230 MB/s |    81195560 | 38.31 |
| ucl_nrv2e 1.03 -6       |    13 MB/s |   264 MB/s |    73302012 | 34.58 |
| ucl_nrv2e 1.03 -9       |  1.30 MB/s |   286 MB/s |    69645134 | 32.86 |
| wflz 2015-09-16         |   186 MB/s |   773 MB/s |   109605264 | 51.71 |
| xpack 2016-06-02 -1     |    98 MB/s |   512 MB/s |    71090065 | 33.54 |
| xpack 2016-06-02 -6     |    29 MB/s |   636 MB/s |    62213845 | 29.35 |
| xpack 2016-06-02 -9     |    11 MB/s |   654 MB/s |    61240928 | 28.89 |
| xz 5.2.2 -0             |    15 MB/s |    44 MB/s |    62579435 | 29.53 |
| xz 5.2.2 -3             |  4.45 MB/s |    54 MB/s |    55745125 | 26.30 |
| xz 5.2.2 -6             |  1.97 MB/s |    58 MB/s |    49195929 | 23.21 |
| xz 5.2.2 -9             |  1.80 MB/s |    58 MB/s |    48745306 | 23.00 |
| yalz77 2015-09-19 -1    |    71 MB/s |   363 MB/s |    93952728 | 44.33 |
| yalz77 2015-09-19 -4    |    33 MB/s |   360 MB/s |    87392632 | 41.23 |
| yalz77 2015-09-19 -8    |    19 MB/s |   355 MB/s |    85153287 | 40.18 |
| yalz77 2015-09-19 -12   |    14 MB/s |   351 MB/s |    84050625 | 39.66 |
| yappy 2014-03-22 -1     |    98 MB/s |  1851 MB/s |   105750956 | 49.89 |
| yappy 2014-03-22 -10    |    77 MB/s |  1954 MB/s |   100018673 | 47.19 |
| yappy 2014-03-22 -100   |    56 MB/s |  1969 MB/s |    98672514 | 46.56 |
| zlib 1.2.8 -1           |    66 MB/s |   244 MB/s |    77259029 | 36.45 |
| zlib 1.2.8 -6           |    20 MB/s |   263 MB/s |    68228431 | 32.19 |
| zlib 1.2.8 -9           |  8.37 MB/s |   266 MB/s |    67644548 | 31.92 |
| zling 2016-01-10 -0     |    40 MB/s |   132 MB/s |    63407921 | 29.92 |
| zling 2016-01-10 -1     |    36 MB/s |   135 MB/s |    62438620 | 29.46 |
| zling 2016-01-10 -2     |    32 MB/s |   135 MB/s |    61917662 | 29.21 |
| zling 2016-01-10 -3     |    29 MB/s |   137 MB/s |    61384151 | 28.96 |
| zling 2016-01-10 -4     |    26 MB/s |   135 MB/s |    60997465 | 28.78 |
| zstd 1.1.1 -1           |   235 MB/s |   645 MB/s |    73659468 | 34.75 |
| zstd 1.1.1 -2           |   181 MB/s |   600 MB/s |    70168955 | 33.11 |
| zstd 1.1.1 -5           |    88 MB/s |   565 MB/s |    65002208 | 30.67 |
| zstd 1.1.1 -8           |    31 MB/s |   619 MB/s |    61026497 | 28.79 |
| zstd 1.1.1 -11          |    16 MB/s |   613 MB/s |    59523167 | 28.08 |
| zstd 1.1.1 -15          |  4.97 MB/s |   639 MB/s |    58007773 | 27.37 |
| zstd 1.1.1 -18          |  2.87 MB/s |   583 MB/s |    55294241 | 26.09 |
| zstd 1.1.1 -22          |  1.44 MB/s |   505 MB/s |    52731930 | 24.88 |
