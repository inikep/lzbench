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
usage: lzbench [options] input_file [input_file2] [input_file3]

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
For Linux/Unix/MinGW (Windows):
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
brotli 0.4.0
crush 1.0
csc 3.3
density 0.12.5 beta (WARNING: it contains bugs (shortened decompressed output))
fastlz 0.1
gipfeli 2015-11-30
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
zstd 0.7.0
```


Benchmarks
-------------------------

The following results are obtained with lzbench using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 4.8.3). 
The ["win81"] input file (100 MB) is a concatanation of carefully selected files from installed version of Windows 8.1 64-bit. 
The results sorted by ratio are available [here](lzbench10_win81sorted.md).

["win81"]: https://docs.google.com/uc?id=0BwX7dtyRLxThRzBwT0xkUy1TMFE&export=download 


| Compressor name             | Compression| Decompress.| Compr. size | Ratio |
| ---------------             | -----------| -----------| ----------- | ----- |
| memcpy                      |  8368 MB/s |  8406 MB/s |   104857600 |100.00 |
| blosclz 2015-11-10 level 1  |  1041 MB/s |  5621 MB/s |   101118592 | 96.43 |
| blosclz 2015-11-10 level 3  |   565 MB/s |  5247 MB/s |    98716389 | 94.14 |
| blosclz 2015-11-10 level 6  |   240 MB/s |  1226 MB/s |    71944073 | 68.61 |
| blosclz 2015-11-10 level 9  |   201 MB/s |   696 MB/s |    64269967 | 61.29 |
| brieflz 1.1.0               |    76 MB/s |   156 MB/s |    55001889 | 52.45 |
| brotli 2016-03-22 -0        |   210 MB/s |   195 MB/s |    52629581 | 50.19 |
| brotli 2016-03-22 -2        |    88 MB/s |   187 MB/s |    48030385 | 45.81 |
| brotli 2016-03-22 -5        |    19 MB/s |   220 MB/s |    43208885 | 41.21 |
| brotli 2016-03-22 -8        |  3.37 MB/s |   224 MB/s |    41009167 | 39.11 |
| brotli 2016-03-22 -11       |  0.25 MB/s |   170 MB/s |    37358056 | 35.63 |
| crush 1.0 level 0           |    21 MB/s |   203 MB/s |    50419812 | 48.08 |
| crush 1.0 level 1           |  4.32 MB/s |   213 MB/s |    48195021 | 45.96 |
| crush 1.0 level 2           |  0.59 MB/s |   221 MB/s |    47105187 | 44.92 |
| csc 3.3 level 1             |    10 MB/s |    33 MB/s |    39201748 | 37.39 |
| csc 3.3 level 3             |  5.41 MB/s |    31 MB/s |    37947503 | 36.19 |
| csc 3.3 level 5             |  3.06 MB/s |    32 MB/s |    37016660 | 35.30 |
| density 0.12.5 beta level 1 |   720 MB/s |   854 MB/s |    77139532 | 73.57 |
| density 0.12.5 beta level 2 |   448 MB/s |   598 MB/s |    65904712 | 62.85 |
| density 0.12.5 beta level 3 |   207 MB/s |   178 MB/s |    60230248 | 57.44 |
| fastlz 0.1 level 1          |   181 MB/s |   510 MB/s |    65163214 | 62.14 |
| fastlz 0.1 level 2          |   208 MB/s |   508 MB/s |    63462293 | 60.52 |
| gipfeli 2015-11-30          |   214 MB/s |   439 MB/s |    59292275 | 56.55 |
| lz4fast r131 level 17       |   994 MB/s |  3172 MB/s |    77577906 | 73.98 |
| lz4fast r131 level 3        |   623 MB/s |  2598 MB/s |    67753409 | 64.61 |
| lz4 r131                    |   497 MB/s |  2492 MB/s |    64872315 | 61.87 |
| lz4hc r131 level 1          |    83 MB/s |  2041 MB/s |    59448496 | 56.69 |
| lz4hc r131 level 4          |    49 MB/s |  2067 MB/s |    55670801 | 53.09 |
| lz4hc r131 level 9          |    25 MB/s |  2147 MB/s |    54773517 | 52.24 |
| lz4hc r131 level 12         |    19 MB/s |  2155 MB/s |    54747494 | 52.21 |
| lz4hc r131 level 16         |    13 MB/s |  2157 MB/s |    54741717 | 52.21 |
| lz5 v1.4.1                  |   198 MB/s |   893 MB/s |    56183327 | 53.58 |
| lz5hc v1.4.1 level 1        |   494 MB/s |  1758 MB/s |    68860852 | 65.67 |
| lz5hc v1.4.1 level 4        |   128 MB/s |   982 MB/s |    56306606 | 53.70 |
| lz5hc v1.4.1 level 9        |    17 MB/s |   744 MB/s |    49862164 | 47.55 |
| lz5hc v1.4.1 level 12       |  8.06 MB/s |   757 MB/s |    47057399 | 44.88 |
| lz5hc v1.4.1 level 15       |  2.29 MB/s |   724 MB/s |    45767126 | 43.65 |
| lzf 3.6 level 0             |   218 MB/s |   533 MB/s |    66219900 | 63.15 |
| lzf 3.6 level 1             |   220 MB/s |   541 MB/s |    63913133 | 60.95 |
| lzg 1.0.8 level 1           |    44 MB/s |   412 MB/s |    65173949 | 62.15 |
| lzg 1.0.8 level 4           |    31 MB/s |   415 MB/s |    61218435 | 58.38 |
| lzg 1.0.8 level 6           |    18 MB/s |   430 MB/s |    58591217 | 55.88 |
| lzg 1.0.8 level 8           |  6.51 MB/s |   456 MB/s |    55268743 | 52.71 |
| lzham 1.0 -d26 level 0      |  6.31 MB/s |   106 MB/s |    42178467 | 40.22 |
| lzham 1.0 -d26 level 1      |  1.77 MB/s |   132 MB/s |    38407249 | 36.63 |
| lzjb 2010                   |   208 MB/s |   394 MB/s |    73436239 | 70.03 |
| lzlib 1.7 level 0           |    16 MB/s |    26 MB/s |    43911286 | 41.88 |
| lzlib 1.7 level 3           |  3.52 MB/s |    30 MB/s |    38565696 | 36.78 |
| lzlib 1.7 level 6           |  2.18 MB/s |    31 MB/s |    35911569 | 34.25 |
| lzlib 1.7 level 9           |  1.64 MB/s |    31 MB/s |    35718249 | 34.06 |
| lzma 9.38 level 0           |    14 MB/s |    34 MB/s |    43768712 | 41.74 |
| lzma 9.38 level 2           |    11 MB/s |    37 MB/s |    40675661 | 38.79 |
| lzma 9.38 level 4           |  6.54 MB/s |    40 MB/s |    39191481 | 37.38 |
| lzma 9.38 level 5           |  2.48 MB/s |    42 MB/s |    36052585 | 34.38 |
| lzmat 1.01                  |    22 MB/s |      ERROR |    52691815 | 50.25 |
| lzo1 2.09 level 1           |   152 MB/s |   462 MB/s |    66048927 | 62.99 |
| lzo1 2.09 level 99          |    66 MB/s |   472 MB/s |    61246849 | 58.41 |
| lzo1a 2.09 level 1          |   153 MB/s |   526 MB/s |    64369332 | 61.39 |
| lzo1a 2.09 level 99         |    62 MB/s |   534 MB/s |    59522850 | 56.77 |
| lzo1b 2.09 level 1          |   129 MB/s |   559 MB/s |    62277761 | 59.39 |
| lzo1b 2.09 level 5          |   144 MB/s |   539 MB/s |    59539396 | 56.78 |
| lzo1b 2.09 level 9          |   106 MB/s |   480 MB/s |    58343947 | 55.64 |
| lzo1b 2.09 level 99         |    57 MB/s |   534 MB/s |    57075974 | 54.43 |
| lzo1b 2.09 level 999        |  9.06 MB/s |   572 MB/s |    53498464 | 51.02 |
| lzo1c 2.09 level 1          |   128 MB/s |   578 MB/s |    63395252 | 60.46 |
| lzo1c 2.09 level 5          |   112 MB/s |   553 MB/s |    60379996 | 57.58 |
| lzo1c 2.09 level 9          |    87 MB/s |   539 MB/s |    59173072 | 56.43 |
| lzo1c 2.09 level 99         |    57 MB/s |   544 MB/s |    58250149 | 55.55 |
| lzo1c 2.09 level 999        |    14 MB/s |   553 MB/s |    55182562 | 52.63 |
| lzo1f 2.09 level 1          |   119 MB/s |   535 MB/s |    63167952 | 60.24 |
| lzo1f 2.09 level 999        |    12 MB/s |   487 MB/s |    54841880 | 52.30 |
| lzo1x 2.09 level 1          |   410 MB/s |   647 MB/s |    64904436 | 61.90 |
| lzo1x 2.09 level 11         |   469 MB/s |   676 MB/s |    67004005 | 63.90 |
| lzo1x 2.09 level 15         |   436 MB/s |   666 MB/s |    65236411 | 62.21 |
| lzo1x 2.09 level 999        |  4.87 MB/s |   485 MB/s |    52280907 | 49.86 |
| lzo1y 2.09 level 1          |   411 MB/s |   645 MB/s |    65233337 | 62.21 |
| lzo1y 2.09 level 999        |  4.88 MB/s |   478 MB/s |    52581195 | 50.15 |
| lzo1z 2.09 level 999        |  5.08 MB/s |   483 MB/s |    51729363 | 49.33 |
| lzo2a 2.09 level 999        |    15 MB/s |   386 MB/s |    55743639 | 53.16 |
| lzrw 15-Jul-1991 level 1    |   153 MB/s |   382 MB/s |    69138188 | 65.94 |
| lzrw 15-Jul-1991 level 2    |   158 MB/s |   413 MB/s |    68803677 | 65.62 |
| lzrw 15-Jul-1991 level 3    |   203 MB/s |   428 MB/s |    66253542 | 63.18 |
| lzrw 15-Jul-1991 level 4    |   221 MB/s |   338 MB/s |    64382024 | 61.40 |
| lzrw 15-Jul-1991 level 5    |    94 MB/s |   323 MB/s |    61293136 | 58.45 |
| pithy 2011-12-24 level 0    |   458 MB/s |  1587 MB/s |    65569609 | 62.53 |
| pithy 2011-12-24 level 3    |   406 MB/s |      ERROR |    63403946 | 60.47 |
| pithy 2011-12-24 level 6    |   321 MB/s |  1512 MB/s |    61219685 | 58.38 |
| pithy 2011-12-24 level 9    |   253 MB/s |  1401 MB/s |    59407478 | 56.66 |
| quicklz 1.5.0 level 1       |   319 MB/s |   337 MB/s |    62896807 | 59.98 |
| quicklz 1.5.0 level 2       |   145 MB/s |   292 MB/s |    57784302 | 55.11 |
| quicklz 1.5.0 level 3       |    39 MB/s |   583 MB/s |    55938979 | 53.35 |
| shrinker 0.1                |   289 MB/s |   867 MB/s |    60900075 | 58.08 |
| snappy 1.1.3                |   328 MB/s |  1177 MB/s |    64864200 | 61.86 |
| tornado 0.6a level 1        |   233 MB/s |   328 MB/s |    71907303 | 68.58 |
| tornado 0.6a level 2        |   202 MB/s |   276 MB/s |    60989163 | 58.16 |
| tornado 0.6a level 3        |   102 MB/s |   138 MB/s |    47942540 | 45.72 |
| tornado 0.6a level 4        |    71 MB/s |   145 MB/s |    45984872 | 43.85 |
| tornado 0.6a level 5        |    23 MB/s |    94 MB/s |    42800284 | 40.82 |
| tornado 0.6a level 6        |    18 MB/s |    92 MB/s |    42135261 | 40.18 |
| tornado 0.6a level 7        |  8.66 MB/s |    94 MB/s |    40993890 | 39.09 |
| tornado 0.6a level 10       |  2.46 MB/s |    96 MB/s |    40664357 | 38.78 |
| tornado 0.6a level 13       |  4.73 MB/s |    91 MB/s |    39439514 | 37.61 |
| tornado 0.6a level 16       |  2.34 MB/s |    98 MB/s |    38726511 | 36.93 |
| ucl_nrv2b 1.03 level 1      |    27 MB/s |   203 MB/s |    54524452 | 52.00 |
| ucl_nrv2b 1.03 level 6      |    11 MB/s |   201 MB/s |    50950304 | 48.59 |
| ucl_nrv2b 1.03 level 9      |  1.32 MB/s |   221 MB/s |    49001893 | 46.73 |
| ucl_nrv2d 1.03 level 1      |    27 MB/s |   211 MB/s |    54430708 | 51.91 |
| ucl_nrv2d 1.03 level 6      |    12 MB/s |   237 MB/s |    50952760 | 48.59 |
| ucl_nrv2d 1.03 level 9      |  1.33 MB/s |   241 MB/s |    48561867 | 46.31 |
| ucl_nrv2e 1.03 level 1      |    26 MB/s |   212 MB/s |    54408737 | 51.89 |
| ucl_nrv2e 1.03 level 6      |    12 MB/s |   231 MB/s |    50832861 | 48.48 |
| ucl_nrv2e 1.03 level 9      |  1.46 MB/s |   240 MB/s |    48462802 | 46.22 |
| wflz 2015-09-16             |   133 MB/s |   815 MB/s |    68272262 | 65.11 |
| xz 5.2.2 level 0            |    11 MB/s |    31 MB/s |    41795581 | 39.86 |
| xz 5.2.2 level 3            |  4.24 MB/s |    35 MB/s |    38842485 | 37.04 |
| xz 5.2.2 level 6            |  2.43 MB/s |    37 MB/s |    35963930 | 34.30 |
| xz 5.2.2 level 9            |  2.19 MB/s |    36 MB/s |    35883407 | 34.22 |
| yalz77 2015-09-19 level 1   |    49 MB/s |   400 MB/s |    60275588 | 57.48 |
| yalz77 2015-09-19 level 4   |    23 MB/s |   389 MB/s |    58110443 | 55.42 |
| yalz77 2015-09-19 level 8   |    13 MB/s |   370 MB/s |    56559159 | 53.94 |
| yalz77 2015-09-19 level 12  |    10 MB/s |   378 MB/s |    55748814 | 53.17 |
| yappy 2014-03-22 level 1    |    74 MB/s |  1953 MB/s |    66362536 | 63.29 |
| yappy 2014-03-22 level 10   |    66 MB/s |  2089 MB/s |    64110300 | 61.14 |
| yappy 2014-03-22 level 100  |    57 MB/s |  2178 MB/s |    63584665 | 60.64 |
| zlib 1.2.8 level 1          |    45 MB/s |   197 MB/s |    51131815 | 48.76 |
| zlib 1.2.8 level 6          |    18 MB/s |   214 MB/s |    47681614 | 45.47 |
| zlib 1.2.8 level 9          |  7.50 MB/s |   216 MB/s |    47516720 | 45.32 |
| zling 2016-04-10 level 0    |    23 MB/s |   100 MB/s |    44381730 | 42.33 |
| zling 2016-04-10 level 2    |    20 MB/s |   105 MB/s |    43836149 | 41.81 |
| zling 2016-04-10 level 4    |    17 MB/s |    98 MB/s |    43491149 | 41.48 |
| zstd v0.6.0 -1              |   231 MB/s |   595 MB/s |    51081337 | 48.71 |
| zstd v0.6.0 -2              |   170 MB/s |   556 MB/s |    49649612 | 47.35 |
| zstd v0.6.0 -5              |    66 MB/s |   508 MB/s |    46175896 | 44.04 |
| zstd v0.6.0 -8              |    25 MB/s |   521 MB/s |    44051111 | 42.01 |
| zstd v0.6.0 -11             |    16 MB/s |   559 MB/s |    42444816 | 40.48 |
| zstd v0.6.0 -15             |  5.92 MB/s |   545 MB/s |    42090097 | 40.14 |
| zstd v0.6.0 -18             |  3.85 MB/s |   513 MB/s |    40724929 | 38.84 |
| zstd v0.6.0 -22             |  2.25 MB/s |   446 MB/s |    38805650 | 37.01 |
