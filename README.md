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

where options are:
 -bX: divides input data in blocks/chunks of size X KB (default = 2097152 KB)
 -cX: sort results by column number X
 -eX: X = compressors separated by '/' with parameters specified after ','
 -iX: selects number of iterations (default 1) and displays best time of X iterations.
 -sX: use only compressors with compression speed over X MB (default = 0 MB)

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
```
brieflz 1.1.0
brotli 2015-10-29
crush 1.0
csc 3.3
density 0.12.5 beta
fastlz 0.1
lz4/lz4hc r131
lz5/lz5hc r131b
lzf 3.6
lzg 1.0.8
lzham 1.0
lzjb 2010
lzlib 1.7
lzma 9.38
lzmat 1.01
lzo 2.09
lzrw 15-Jul-1991
pithy 2011-12-24
quicklz 1.5.0
shrinker 0.1
snappy 1.1.3
tornado 0.6a
ucl 1.03
wflz 2015-09-16
xz 5.2.2
yalz77 2015-09-19
yappy 2014-03-22
zlib 1.2.8
zling 2015-09-15
zstd v0.3.6
zstd_HC v0.3.6
```

Benchmarks
-------------------------

The following results are obtained with lzbench using 1 core of Intel Core i5-4300U, Windows 10 64-bit (MinGW-w64 compilation under gcc 4.8.3) with 3 iterations. 
The ["win81"] input file (100 MB) is a concatanation of carefully selected files from installed version of Windows 8.1 64-bit. 

["win81"]: https://docs.google.com/uc?id=0BwX7dtyRLxThRzBwT0xkUy1TMFE&export=download 


| Compressor name             | Compression| Decompress.| Compr. size | Ratio |
| ---------------             | -----------| -----------| ----------- | ----- |
| memcpy                      |  9309 MB/s |  9309 MB/s |   104857600 |100.00 |
| brotli 2015-10-29 level 1   |    90 MB/s |   213 MB/s |    47882059 | 45.66 |
| brotli 2015-10-29 level 2   |    71 MB/s |   217 MB/s |    47605131 | 45.40 |
| brotli 2015-10-29 level 5   |    18 MB/s |   221 MB/s |    43363897 | 41.36 |
| brotli 2015-10-29 level 8   |  3.44 MB/s |   233 MB/s |    41031551 | 39.13 |
| brotli 2015-10-29 level 11  |  0.27 MB/s |   176 MB/s |    37394612 | 35.66 |
| crush 1.0 level 0           |    22 MB/s |   207 MB/s |    50419812 | 48.08 |
| crush 1.0 level 1           |  4.61 MB/s |   219 MB/s |    48195021 | 45.96 |
| csc 3.3 level 1             |    11 MB/s |    33 MB/s |    39227867 | 37.41 |
| csc 3.3 level 2             |  7.65 MB/s |    34 MB/s |    38447672 | 36.67 |
| csc 3.3 level 3             |  5.91 MB/s |    33 MB/s |    37965236 | 36.21 |
| csc 3.3 level 4             |  4.36 MB/s |    33 MB/s |    37461214 | 35.73 |
| csc 3.3 level 5             |  3.41 MB/s |    34 MB/s |    37030552 | 35.32 |
| density 0.12.5 beta level 1 |   731 MB/s |   860 MB/s |    77139532 | 73.57 |
| density 0.12.5 beta level 2 |   449 MB/s |   602 MB/s |    65904712 | 62.85 |
| density 0.12.5 beta level 3 |   214 MB/s |   196 MB/s |    60230248 | 57.44 |
| fastlz 0.1 level 1          |   181 MB/s |   527 MB/s |    65163214 | 62.14 |
| fastlz 0.1 level 2          |   212 MB/s |   514 MB/s |    63462293 | 60.52 |
| lz4 r131                    |   492 MB/s |  2560 MB/s |    64872315 | 61.87 |
| lz4fast r131 acc=3          |   620 MB/s |  2694 MB/s |    67753409 | 64.61 |
| lz4fast r131 acc=17         |   984 MB/s |  3303 MB/s |    77577906 | 73.98 |
| lz4hc r131 -1               |    85 MB/s |  2048 MB/s |    59448496 | 56.69 |
| lz4hc r131 -4               |    49 MB/s |  2089 MB/s |    55670801 | 53.09 |
| lz4hc r131 -9               |    25 MB/s |  2133 MB/s |    54773517 | 52.24 |
| lz5 r131b                   |   209 MB/s |  1003 MB/s |    55884927 | 53.30 |
| lz5hc r131b -1              |    34 MB/s |   775 MB/s |    52927122 | 50.48 |
| lz5hc r131b -4              |    16 MB/s |   747 MB/s |    50389567 | 48.06 |
| lz5hc r131b -9              |  2.93 MB/s |   701 MB/s |    49346894 | 47.06 |
| lzf level 0                 |   218 MB/s |   541 MB/s |    66219900 | 63.15 |
| lzf level 1                 |   221 MB/s |   556 MB/s |    63913133 | 60.95 |
| lzham 1.0 -m0d26 -0         |  6.64 MB/s |   117 MB/s |    42178467 | 40.22 |
| lzham 1.0 -m0d26 -1         |  1.83 MB/s |   137 MB/s |    38407249 | 36.63 |
| lzjb 2010                   |   213 MB/s |   414 MB/s |    73436239 | 70.03 |
| lzma 9.38 level 0           |    14 MB/s |    34 MB/s |    43768712 | 41.74 |
| lzma 9.38 level 1           |    13 MB/s |    37 MB/s |    42167199 | 40.21 |
| lzma 9.38 level 2           |    12 MB/s |    39 MB/s |    40675661 | 38.79 |
| lzma 9.38 level 3           |  7.73 MB/s |    40 MB/s |    40118385 | 38.26 |
| lzma 9.38 level 4           |  6.79 MB/s |    42 MB/s |    39191481 | 37.38 |
| lzma 9.38 level 5           |  2.61 MB/s |    44 MB/s |    36052585 | 34.38 |
| lzmat 1.01                  |    23 MB/s |      ERROR |    52691815 | 50.25 |
| lzo1b 2.09 -1               |   129 MB/s |   562 MB/s |    62277761 | 59.39 |
| lzo1b 2.09 -9               |   103 MB/s |   556 MB/s |    58343947 | 55.64 |
| lzo1b 2.09 -99              |    61 MB/s |   550 MB/s |    57075974 | 54.43 |
| lzo1b 2.09 -999             |  9.32 MB/s |   578 MB/s |    53498464 | 51.02 |
| lzo1c 2.09 -1               |   128 MB/s |   581 MB/s |    63395252 | 60.46 |
| lzo1c 2.09 -9               |    86 MB/s |   553 MB/s |    59173072 | 56.43 |
| lzo1c 2.09 -99              |    58 MB/s |   553 MB/s |    58250149 | 55.55 |
| lzo1c 2.09 -999             |    15 MB/s |   572 MB/s |    55182562 | 52.63 |
| lzo1f 2.09 -1               |   119 MB/s |   538 MB/s |    63167952 | 60.24 |
| lzo1f 2.09 -999             |    12 MB/s |   492 MB/s |    54841880 | 52.30 |
| lzo1x 2.09 -1               |   417 MB/s |   648 MB/s |    64904436 | 61.90 |
| lzo1x 2.09 -999             |  5.08 MB/s |   494 MB/s |    52280907 | 49.86 |
| lzo1y 2.09 -1               |   421 MB/s |   652 MB/s |    65233337 | 62.21 |
| lzo1y 2.09 -999             |  5.10 MB/s |   492 MB/s |    52581195 | 50.15 |
| lzo1z 2.09 -999             |  5.23 MB/s |   485 MB/s |    51729363 | 49.33 |
| lzo2a 2.09 -999             |    15 MB/s |   396 MB/s |    55743639 | 53.16 |
| lzrw1                       |   158 MB/s |   389 MB/s |    69138188 | 65.94 |
| lzrw1a                      |   167 MB/s |   421 MB/s |    68803677 | 65.62 |
| lzrw2                       |   210 MB/s |   435 MB/s |    66253542 | 63.18 |
| lzrw3                       |   227 MB/s |   345 MB/s |    64382024 | 61.40 |
| lzrw3a                      |    97 MB/s |   326 MB/s |    61293136 | 58.45 |
| pithy 2011-12-24 level 0    |   471 MB/s |  1600 MB/s |    65569609 | 62.53 |
| pithy 2011-12-24 level 3    |   417 MB/s |      ERROR |    63403946 | 60.47 |
| pithy 2011-12-24 level 6    |   332 MB/s |  1505 MB/s |    61219685 | 58.38 |
| pithy 2011-12-24 level 9    |   266 MB/s |  1402 MB/s |    59407478 | 56.66 |
| quicklz 1.5.0 -1            |   325 MB/s |   339 MB/s |    62896807 | 59.98 |
| quicklz 1.5.0 -2            |   150 MB/s |   295 MB/s |    57784302 | 55.11 |
| quicklz 1.5.0 -3            |    40 MB/s |   598 MB/s |    55938979 | 53.35 |
| quicklz 1.5.1 b7 -1         |   332 MB/s |   342 MB/s |    62896808 | 59.98 |
| shrinker                    |   290 MB/s |   882 MB/s |    60900075 | 58.08 |
| snappy 1.1.3                |   327 MB/s |  1163 MB/s |    64864200 | 61.86 |
| tornado 0.6a -1             |   237 MB/s |   329 MB/s |    71907303 | 68.58 |
| tornado 0.6a -2             |   203 MB/s |   286 MB/s |    60989163 | 58.16 |
| tornado 0.6a -3             |   104 MB/s |   144 MB/s |    47942540 | 45.72 |
| tornado 0.6a -4             |    76 MB/s |   151 MB/s |    45984872 | 43.85 |
| tornado 0.6a -5             |    25 MB/s |    97 MB/s |    42800284 | 40.82 |
| tornado 0.6a -6             |    19 MB/s |    98 MB/s |    42135261 | 40.18 |
| tornado 0.6a -7             |  9.33 MB/s |   100 MB/s |    40993890 | 39.09 |
| tornado 0.6a -10            |  2.56 MB/s |   100 MB/s |    40664357 | 38.78 |
| tornado 0.6a -13            |  5.35 MB/s |    98 MB/s |    39439514 | 37.61 |
| tornado 0.6a -16            |  2.47 MB/s |   100 MB/s |    38726511 | 36.93 |
| tornado 0.6a h16k b1m       |   237 MB/s |   329 MB/s |    71907303 | 68.58 |
| tornado 0.6a h128k b2m      |   209 MB/s |   329 MB/s |    66953110 | 63.85 |
| tornado 0.6a h128k b8m      |   207 MB/s |   327 MB/s |    66583452 | 63.50 |
| tornado 0.6a h4m b8m        |    88 MB/s |   317 MB/s |    62198875 | 59.32 |
| tornado h128k b8m bitio     |   186 MB/s |   283 MB/s |    59603082 | 56.84 |
| tornado h4m b8m bitio       |    78 MB/s |   281 MB/s |    55424435 | 52.86 |
| tornado h4m b32m bitio      |    71 MB/s |   274 MB/s |    55325986 | 52.76 |
| ucl_nrv2b 1.03 -1           |    28 MB/s |   204 MB/s |    54524452 | 52.00 |
| ucl_nrv2b 1.03 -6           |    12 MB/s |   222 MB/s |    50950304 | 48.59 |
| ucl_nrv2d 1.03 -1           |    28 MB/s |   217 MB/s |    54430708 | 51.91 |
| ucl_nrv2d 1.03 -6           |    12 MB/s |   235 MB/s |    50952760 | 48.59 |
| ucl_nrv2e 1.03 -1           |    28 MB/s |   218 MB/s |    54408737 | 51.89 |
| ucl_nrv2e 1.03 -6           |    12 MB/s |   237 MB/s |    50832861 | 48.48 |
| wflz 2015-09-16             |   147 MB/s |   867 MB/s |    68272262 | 65.11 |
| yappy 1                     |    80 MB/s |  1828 MB/s |    66362536 | 63.29 |
| yappy 10                    |    68 MB/s |  2007 MB/s |    64110300 | 61.14 |
| yappy 100                   |    59 MB/s |  2048 MB/s |    63584665 | 60.64 |
| zlib 1.2.8 -1               |    49 MB/s |   203 MB/s |    51131815 | 48.76 |
| zlib 1.2.8 -6               |    18 MB/s |   216 MB/s |    47681614 | 45.47 |
| zlib 1.2.8 -9               |  7.53 MB/s |   217 MB/s |    47516720 | 45.32 |
| zling 2015-09-15 level 0    |    26 MB/s |   118 MB/s |    45169630 | 43.08 |
| zling 2015-09-15 level 1    |    24 MB/s |   119 MB/s |    44776544 | 42.70 |
| zling 2015-09-15 level 2    |    23 MB/s |   119 MB/s |    44604367 | 42.54 |
| zling 2015-09-15 level 3    |    21 MB/s |   119 MB/s |    44393939 | 42.34 |
| zling 2015-09-15 level 4    |    20 MB/s |   120 MB/s |    44288238 | 42.24 |
| zstd v0.3                   |   266 MB/s |   538 MB/s |    51231016 | 48.86 |
| zstd_HC v0.3 -1             |   266 MB/s |   541 MB/s |    51231016 | 48.86 |
| zstd_HC v0.3 -5             |    44 MB/s |   487 MB/s |    45628362 | 43.51 |
| zstd_HC v0.3 -9             |    15 MB/s |   497 MB/s |    44840562 | 42.76 |
| zstd_HC v0.3 -13            |  9.97 MB/s |   487 MB/s |    43114895 | 41.12 |
| zstd_HC v0.3 -17            |  6.48 MB/s |   494 MB/s |    42989971 | 41.00 |
| zstd_HC v0.3 -21            |  3.63 MB/s |   478 MB/s |    42956964 | 40.97 |
