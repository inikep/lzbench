Introduction
-------------------------

lzbench is an in-memory benchmark of the fastest open-source LZ77/LZSS compressors. It joins all compressors into a single exe. 
At the beginning an input file is read to memory. 
Then all compressors are used to compress and decompress the file and decompressed file is verified. 
This approach has a big advantage of using the same compiler with the same optimizations for all compressors. 
The disadvantage is that it requires source code of each compressor (therefore Slug or lzturbo are not included).


Usage
-------------------------

```
usage: lzbench [options] input_file

where options are:
 -iX: selects number of iterations (default 1) and displays best time of X iterations.
 -bX: divides input data in blocks/chunks of size X KB (default = 2097152 KB)
 -sX: selects only compressors with compression speed over X MB (default = 100 MB) - so far it's only approximation
```


Compilation
-------------------------
For Linux/Unix:
```
make BUILD_SYSTEM=linux
```

For Windows (MinGW)
```
make
```

To remove one of compressors you can add -DBENCH_REMOVE_XXX to $DEFINES in Makefile (e.g. DEFINES += -DBENCH_REMOVE_LZ5 to remove LZ5).

Supported compressors
-------------------------
```
brotli 2015-10-29
crush 1.0
csc 3.3
density 0.12.5 beta
fastlz 0.1
lz4/lz4hc r131
lz5/lz5hc r131
lzf
lzham 1.0
lzjb 2010
lzma 9.38
lzmat 1.01
lzo 2.09
lzrw
pithy 2011-12-24
quicklz 1.5.0
quicklz 1.5.1 b7
shrinker
snappy 1.1.3
tornado 0.6
ucl 1.03
yappy
zlib 1.2.8
zling 2015-09-15
zstd v0.3
zstd_HC v0.3
```

