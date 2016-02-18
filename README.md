Introduction
-------------------------

lzbench is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors. It joins all compressors into a single exe. 
At the beginning an input file is read to memory. 
Then all compressors are used to compress and decompress the file and decompressed file is verified. 
This approach has a big advantage of using the same compiler with the same optimizations for all compressors. 
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).


Usage
-------------------------

```
usage: lzbench [options] input_file

where [options] are:
 -bX  set block/chunk size to X KB (default = filesize or 2097152 KB)
 -cX  sort results by column number X
 -eX  X = compressors separated by '/' with parameters specified after ','
 -iX  set min. number of compression iterations (default = 1)
 -jX  set min. number of decompression iterations (default = 1)
 -l   list of available compressors and aliases
 -oX  output text format 1=Markdown, 2=text, 3=CSV (default = 2)
 -pX  print time for all iterations: 1=fastest 2=average 3=median (default = 1)
 -sX  use only compressors with compression speed over X MB (default = 0 MB)
 -tX  set min. time in seconds for compression (default = 1.0)
 -uX  set min. time in seconds for decompression (default = 0.5)

Example usage:
  lzbench -ebrotli filename - selects all levels of brotli
  lzbench -ebrotli,2,5/zstd filename - selects levels 2 & 5 of brotli and zstd
```


Compilation
-------------------------
For Linux/Unix:
```
make BUILD_SYSTEM=linux
```

For Windows (MinGW):
```
make
```

For 32-bit compilation:
```
make BUILD_ARCH=32-bit

```

To remove one of compressors you can add -DBENCH_REMOVE_XXX to $DEFINES in Makefile (e.g. DEFINES += -DBENCH_REMOVE_LZ5 to remove LZ5).

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
brotli 2016-02-04
crush 1.0
csc 3.3
density 0.12.5 beta (WARNING: it contains bugs (shortened decompressed output))
fastlz 0.1
gipfeli 2015-11-30
lz4/lz4hc r131
lz5/lz5hc v1.4.1
lzf 3.6
lzg 1.0.8
lzham 1.0
lzjb 2010
lzlib 1.7
lzma 9.38
lzmat 1.01 (WARNING: it contains bugs (decompression error; returns 0); it can throw SEGFAULT compiled with gcc 4.9+ -O3)
lzo 2.09
lzrw 15-Jul-1991
pithy 2011-12-24 (WARNING: it contains bugs (decompression error; returns 0))
quicklz 1.5.0
shrinker 0.1 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
snappy 1.1.3
tornado 0.6a
ucl 1.03
wflz 2015-09-16 (WARNING: it can throw SEGFAULT compiled with gcc 4.9+ -O3)
xz 5.2.2
yalz77 2015-09-19
yappy 2014-03-22 (WARNING: fails to decompress properly on ARM)
zlib 1.2.8
zling 2016-01-05+bugfix (according to the author using libzling in a production environment is not a good idea)
zstd v0.5.1
```


Benchmarks
-------------------------

The following results are obtained with lzbench using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 4.8.3) with 3 iterations. 
The ["win81"] input file (100 MB) is a concatanation of carefully selected files from installed version of Windows 8.1 64-bit. 

["win81"]: https://docs.google.com/uc?id=0BwX7dtyRLxThRzBwT0xkUy1TMFE&export=download 


| Compressor name             | Compression| Decompress.| Compr. size | Ratio |
| ---------------             | -----------| -----------| ----------- | ----- |
| memcpy                      |  8368 MB/s |  8406 MB/s |   104857600 |100.00 |
| blosclz 2015-11-10 level 1  |  1008 MB/s |  5486 MB/s |   101118592 | 96.43 |
| blosclz 2015-11-10 level 3  |   559 MB/s |  5105 MB/s |    98716389 | 94.14 |
| blosclz 2015-11-10 level 6  |   233 MB/s |  1181 MB/s |    71944073 | 68.61 |
| blosclz 2015-11-10 level 9  |   198 MB/s |   677 MB/s |    64269967 | 61.29 |
| brieflz 1.1.0               |    80 MB/s |   152 MB/s |    55001889 | 52.45 |
| brotli 2015-10-29 level 0   |    84 MB/s |   208 MB/s |    47882059 | 45.66 |
| brotli 2015-10-29 level 2   |    66 MB/s |   207 MB/s |    47605131 | 45.40 |
| brotli 2015-10-29 level 5   |    16 MB/s |   215 MB/s |    43363897 | 41.36 |
| brotli 2015-10-29 level 8   |  3.00 MB/s |   212 MB/s |    41031551 | 39.13 |
| brotli 2015-10-29 level 11  |  0.26 MB/s |   167 MB/s |    37394612 | 35.66 |
| crush 1.0 level 0           |    21 MB/s |   202 MB/s |    50419812 | 48.08 |
| crush 1.0 level 1           |  4.30 MB/s |   214 MB/s |    48195021 | 45.96 |
| crush 1.0 level 2           |  0.61 MB/s |   216 MB/s |    47105187 | 44.92 |
| csc 3.3 level 1             |    10 MB/s |    32 MB/s |    39201748 | 37.39 |
| csc 3.3 level 2             |  6.60 MB/s |    32 MB/s |    38433849 | 36.65 |
| csc 3.3 level 3             |  5.34 MB/s |    31 MB/s |    37947503 | 36.19 |
| csc 3.3 level 4             |  3.74 MB/s |    32 MB/s |    37427899 | 35.69 |
| csc 3.3 level 5             |  3.07 MB/s |    30 MB/s |    37016660 | 35.30 |
| density 0.12.5 beta level 1 |   713 MB/s |   838 MB/s |    77139532 | 73.57 |
| density 0.12.5 beta level 2 |   433 MB/s |   572 MB/s |    65904712 | 62.85 |
| density 0.12.5 beta level 3 |   201 MB/s |   180 MB/s |    60230248 | 57.44 |
| fastlz 0.1 level 1          |   175 MB/s |   502 MB/s |    65163214 | 62.14 |
| fastlz 0.1 level 2          |   207 MB/s |   497 MB/s |    63462293 | 60.52 |
| gipfeli 2015-11-30          |   220 MB/s |   434 MB/s |    59292275 | 56.55 |
| lz4 r131                    |   487 MB/s |  2452 MB/s |    64872315 | 61.87 |
| lz4fast r131 level 3        |   610 MB/s |  2577 MB/s |    67753409 | 64.61 |
| lz4fast r131 level 17       |   964 MB/s |  3112 MB/s |    77577906 | 73.98 |
| lz4hc r131 level 1          |    83 MB/s |  1939 MB/s |    59448496 | 56.69 |
| lz4hc r131 level 4          |    47 MB/s |  2029 MB/s |    55670801 | 53.09 |
| lz4hc r131 level 9          |    24 MB/s |  2042 MB/s |    54773517 | 52.24 |
| lz4hc r131 level 12         |    18 MB/s |  2051 MB/s |    54747494 | 52.21 |
| lz5 v1.3.3                  |   188 MB/s |   893 MB/s |    56183327 | 53.58 |
| lz5hc v1.3.3 level 1        |   483 MB/s |  1736 MB/s |    68770655 | 65.58 |
| lz5hc v1.3.3 level 4        |   128 MB/s |   939 MB/s |    55011906 | 52.46 |
| lz5hc v1.3.3 level 9        |    17 MB/s |   707 MB/s |    48718531 | 46.46 |
| lz5hc v1.3.3 level 12       |  7.64 MB/s |   748 MB/s |    47063261 | 44.88 |
| lz5hc v1.3.3 level 16       |  0.79 MB/s |   720 MB/s |    46125742 | 43.99 |
| lzf 3.6 level 0             |   214 MB/s |   527 MB/s |    66219900 | 63.15 |
| lzf 3.6 level 1             |   216 MB/s |   543 MB/s |    63913133 | 60.95 |
| lzg 1.0.8 level 1           |    44 MB/s |   411 MB/s |    65173949 | 62.15 |
| lzg 1.0.8 level 4           |    30 MB/s |   413 MB/s |    61218435 | 58.38 |
| lzg 1.0.8 level 6           |    18 MB/s |   428 MB/s |    58591217 | 55.88 |
| lzg 1.0.8 level 8           |  6.16 MB/s |   450 MB/s |    55268743 | 52.71 |
| lzham 1.0 -d26 level 0      |  6.21 MB/s |   114 MB/s |    42178467 | 40.22 |
| lzham 1.0 -d26 level 1      |  1.75 MB/s |   131 MB/s |    38407249 | 36.63 |
| lzjb 2010                   |   206 MB/s |   410 MB/s |    73436239 | 70.03 |
| lzlib 1.7 level 0           |    16 MB/s |    26 MB/s |    43911286 | 41.88 |
| lzlib 1.7 level 3           |  3.48 MB/s |    29 MB/s |    38565696 | 36.78 |
| lzlib 1.7 level 6           |  2.14 MB/s |    31 MB/s |    35911569 | 34.25 |
| lzlib 1.7 level 9           |  1.62 MB/s |    30 MB/s |    35718249 | 34.06 |
| lzma 9.38 level 0           |    13 MB/s |    33 MB/s |    43768712 | 41.74 |
| lzma 9.38 level 2           |    10 MB/s |    37 MB/s |    40675661 | 38.79 |
| lzma 9.38 level 4           |  6.54 MB/s |    40 MB/s |    39191481 | 37.38 |
| lzma 9.38 level 5           |  2.47 MB/s |    42 MB/s |    36052585 | 34.38 |
| lzmat 1.01                  |    23 MB/s |      ERROR |    52691815 | 50.25 |
| lzo1 2.09 level 1           |   150 MB/s |   475 MB/s |    66048927 | 62.99 |
| lzo1 2.09 level 99          |    65 MB/s |   480 MB/s |    61246849 | 58.41 |
| lzo1a 2.09 level 1          |   151 MB/s |   515 MB/s |    64369332 | 61.39 |
| lzo1a 2.09 level 99         |    64 MB/s |   518 MB/s |    59522850 | 56.77 |
| lzo1b 2.09 level 1          |   126 MB/s |   536 MB/s |    62277761 | 59.39 |
| lzo1b 2.09 level 2          |   126 MB/s |   541 MB/s |    61501385 | 58.65 |
| lzo1b 2.09 level 3          |   123 MB/s |   542 MB/s |    60949402 | 58.13 |
| lzo1b 2.09 level 4          |   144 MB/s |   498 MB/s |    60260856 | 57.47 |
| lzo1b 2.09 level 5          |   143 MB/s |   527 MB/s |    59539396 | 56.78 |
| lzo1b 2.09 level 6          |   138 MB/s |   535 MB/s |    59032880 | 56.30 |
| lzo1b 2.09 level 7          |    99 MB/s |   522 MB/s |    59328953 | 56.58 |
| lzo1b 2.09 level 8          |    96 MB/s |   535 MB/s |    58578366 | 55.86 |
| lzo1b 2.09 level 9          |   102 MB/s |   525 MB/s |    58343947 | 55.64 |
| lzo1b 2.09 level 99         |    58 MB/s |   522 MB/s |    57075974 | 54.43 |
| lzo1b 2.09 level 999        |  8.90 MB/s |   562 MB/s |    53498464 | 51.02 |
| lzo1c 2.09 level 1          |   125 MB/s |   560 MB/s |    63395252 | 60.46 |
| lzo1c 2.09 level 2          |   122 MB/s |   564 MB/s |    62701074 | 59.80 |
| lzo1c 2.09 level 3          |   121 MB/s |   564 MB/s |    62195255 | 59.31 |
| lzo1c 2.09 level 4          |   110 MB/s |   539 MB/s |    61087271 | 58.26 |
| lzo1c 2.09 level 5          |   110 MB/s |   543 MB/s |    60379996 | 57.58 |
| lzo1c 2.09 level 6          |   110 MB/s |   544 MB/s |    59987222 | 57.21 |
| lzo1c 2.09 level 7          |    72 MB/s |   534 MB/s |    59968920 | 57.19 |
| lzo1c 2.09 level 8          |    70 MB/s |   542 MB/s |    59450542 | 56.70 |
| lzo1c 2.09 level 9          |    82 MB/s |   534 MB/s |    59173072 | 56.43 |
| lzo1c 2.09 level 99         |    56 MB/s |   527 MB/s |    58250149 | 55.55 |
| lzo1c 2.09 level 999        |    14 MB/s |   555 MB/s |    55182562 | 52.63 |
| lzo1f 2.09 level 1          |   116 MB/s |   522 MB/s |    63167952 | 60.24 |
| lzo1f 2.09 level 999        |    12 MB/s |   478 MB/s |    54841880 | 52.30 |
| lzo1x 2.09 level 1          |   411 MB/s |   635 MB/s |    64904436 | 61.90 |
| lzo1x 2.09 level 11         |   456 MB/s |   677 MB/s |    67004005 | 63.90 |
| lzo1x 2.09 level 12         |   447 MB/s |   656 MB/s |    65865366 | 62.81 |
| lzo1x 2.09 level 15         |   428 MB/s |   639 MB/s |    65236411 | 62.21 |
| lzo1x 2.09 level 999        |  4.89 MB/s |   490 MB/s |    52280907 | 49.86 |
| lzo1y 2.09 level 1          |   413 MB/s |   637 MB/s |    65233337 | 62.21 |
| lzo1y 2.09 level 999        |  4.88 MB/s |   477 MB/s |    52581195 | 50.15 |
| lzo1z 2.09 level 999        |  5.03 MB/s |   468 MB/s |    51729363 | 49.33 |
| lzo2a 2.09 level 999        |    15 MB/s |   377 MB/s |    55743639 | 53.16 |
| lzrw 15-Jul-1991 level 1    |   155 MB/s |   375 MB/s |    69138188 | 65.94 |
| lzrw 15-Jul-1991 level 2    |   166 MB/s |   407 MB/s |    68803677 | 65.62 |
| lzrw 15-Jul-1991 level 3    |   204 MB/s |   419 MB/s |    66253542 | 63.18 |
| lzrw 15-Jul-1991 level 4    |   220 MB/s |   330 MB/s |    64382024 | 61.40 |
| lzrw 15-Jul-1991 level 5    |    95 MB/s |   315 MB/s |    61293136 | 58.45 |
| pithy 2011-12-24 level 0    |   445 MB/s |  1533 MB/s |    65569609 | 62.53 |
| pithy 2011-12-24 level 3    |   402 MB/s |      ERROR |    63403946 | 60.47 |
| pithy 2011-12-24 level 6    |   314 MB/s |  1462 MB/s |    61219685 | 58.38 |
| pithy 2011-12-24 level 9    |   252 MB/s |  1338 MB/s |    59407478 | 56.66 |
| quicklz 1.5.0 level 1       |   326 MB/s |   338 MB/s |    62896807 | 59.98 |
| quicklz 1.5.0 level 2       |   149 MB/s |   291 MB/s |    57784302 | 55.11 |
| quicklz 1.5.0 level 3       |    39 MB/s |   569 MB/s |    55938979 | 53.35 |
| shrinker 0.1                |   280 MB/s |   859 MB/s |    60900075 | 58.08 |
| snappy 1.1.3                |   326 MB/s |  1147 MB/s |    64864200 | 61.86 |
| tornado 0.6a level 1        |   228 MB/s |   320 MB/s |    71907303 | 68.58 |
| tornado 0.6a level 2        |   194 MB/s |   274 MB/s |    60989163 | 58.16 |
| tornado 0.6a level 3        |   101 MB/s |   140 MB/s |    47942540 | 45.72 |
| tornado 0.6a level 4        |    70 MB/s |   145 MB/s |    45984872 | 43.85 |
| tornado 0.6a level 5        |    23 MB/s |    94 MB/s |    42800284 | 40.82 |
| tornado 0.6a level 6        |    18 MB/s |    94 MB/s |    42135261 | 40.18 |
| tornado 0.6a level 7        |  8.11 MB/s |    97 MB/s |    40993890 | 39.09 |
| tornado 0.6a level 10       |  2.49 MB/s |    97 MB/s |    40664357 | 38.78 |
| tornado 0.6a level 13       |  4.93 MB/s |    96 MB/s |    39439514 | 37.61 |
| tornado 0.6a level 16       |  2.37 MB/s |    98 MB/s |    38726511 | 36.93 |
| ucl_nrv2b 1.03 level 1      |    17 MB/s |   202 MB/s |    54524452 | 52.00 |
| ucl_nrv2b 1.03 level 6      |    11 MB/s |   217 MB/s |    50950304 | 48.59 |
| ucl_nrv2b 1.03 level 9      |  1.24 MB/s |   223 MB/s |    49001893 | 46.73 |
| ucl_nrv2d 1.03 level 1      |    25 MB/s |   206 MB/s |    54430708 | 51.91 |
| ucl_nrv2d 1.03 level 6      |    11 MB/s |   225 MB/s |    50952760 | 48.59 |
| ucl_nrv2d 1.03 level 9      |  1.25 MB/s |   228 MB/s |    48561867 | 46.31 |
| ucl_nrv2e 1.03 level 1      |    24 MB/s |   214 MB/s |    54408737 | 51.89 |
| ucl_nrv2e 1.03 level 6      |    11 MB/s |   232 MB/s |    50832861 | 48.48 |
| ucl_nrv2e 1.03 level 9      |  1.28 MB/s |   235 MB/s |    48462802 | 46.22 |
| wflz 2015-09-16             |   133 MB/s |   809 MB/s |    68272262 | 65.11 |
| xz 5.2.2 level 0            |    10 MB/s |    31 MB/s |    41795581 | 39.86 |
| xz 5.2.2 level 3            |  4.22 MB/s |    35 MB/s |    38842485 | 37.04 |
| xz 5.2.2 level 6            |  2.40 MB/s |    36 MB/s |    35963930 | 34.30 |
| xz 5.2.2 level 9            |  2.17 MB/s |    35 MB/s |    35883407 | 34.22 |
| yalz77 2015-09-19 level 1   |    46 MB/s |   395 MB/s |    60275588 | 57.48 |
| yalz77 2015-09-19 level 4   |    23 MB/s |   377 MB/s |    58110443 | 55.42 |
| yalz77 2015-09-19 level 8   |    13 MB/s |   381 MB/s |    56559159 | 53.94 |
| yalz77 2015-09-19 level 12  |    10 MB/s |   370 MB/s |    55748814 | 53.17 |
| yappy 2014-03-22 level 1    |    78 MB/s |  1954 MB/s |    66362536 | 63.29 |
| yappy 2014-03-22 level 10   |    66 MB/s |  2063 MB/s |    64110300 | 61.14 |
| yappy 2014-03-22 level 100  |    57 MB/s |  2067 MB/s |    63584665 | 60.64 |
| zlib 1.2.8 level 1          |    39 MB/s |   201 MB/s |    51131815 | 48.76 |
| zlib 1.2.8 level 6          |    18 MB/s |   214 MB/s |    47681614 | 45.47 |
| zlib 1.2.8 level 9          |  7.29 MB/s |   215 MB/s |    47516720 | 45.32 |
| zling 2015-09-16 level 0    |    23 MB/s |   111 MB/s |    45169630 | 43.08 |
| zling 2015-09-16 level 1    |    23 MB/s |   102 MB/s |    44776544 | 42.70 |
| zling 2015-09-16 level 2    |    22 MB/s |   115 MB/s |    44604367 | 42.54 |
| zling 2015-09-16 level 3    |    20 MB/s |   110 MB/s |    44393939 | 42.34 |
| zling 2015-09-16 level 4    |    19 MB/s |   115 MB/s |    44288238 | 42.24 |
| zstd v0.4.1 level 1         |   249 MB/s |   537 MB/s |    51160301 | 48.79 |
| zstd v0.4.1 level 2         |   183 MB/s |   505 MB/s |    49719335 | 47.42 |
| zstd v0.4.1 level 5         |    72 MB/s |   461 MB/s |    46389082 | 44.24 |
| zstd v0.4.1 level 9         |    17 MB/s |   474 MB/s |    43892280 | 41.86 |
| zstd v0.4.1 level 13        |    10 MB/s |   487 MB/s |    42321163 | 40.36 |
| zstd v0.4.1 level 17        |  1.97 MB/s |   476 MB/s |    42009876 | 40.06 |
| zstd v0.4.1 level 20        |  1.70 MB/s |   459 MB/s |    41880158 | 39.94 |
