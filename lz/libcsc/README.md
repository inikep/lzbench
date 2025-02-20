# CSC (CSArc)

It includes
* libcsc: A Loss-less data compression algorithm inspired by LZMA
* csarc: Experimental command line archiver running on Linux/Windows based on libcsc


### Usage of CSArc:
* Create a new archive:
    ```csarc a [options] archive_name file2 file2 ...```
    [options] can be:
    ```-m[1..5] Compression level from most efficient to strongest```
    ```-d##[k|m] Dictionary size, must be in range of [32k, 1024m]```
    ```-r Recursively scan files in directories```
    ```-f Forcely overwrite existing archive```
    ```-p## Only works with single file compression, split the work for multi-threading```
    ```-t# Multithreading-number, ranges in [1,8], memory usage will be multiplied by this number```

* Extract file(s) from an archive:
    ```csarc x [options] archive_name [file1_in_arc file2_in_arc ...]```
    [options] can be:
    ```-t# Multithreading-number, ranges in [1,8], memory usage will be multiplied by this number```
    ```-o out_dir Extraction output directory```

* List file(s) in archive:
    ```csarc l [options] archive_name [file1_in_arc file2_in_arc ...]```
    [options] can be:
    ```-v Shows fragments information with Adler32 hash```

* Test to extract file(s) in archive:
    ```csarc t [options] archive_name [file1_in_arc file2_in_arc ...]```
    [options] can be:
    ```-t# Multithreading-number, ranges in [1,8]```

* Examples:
csarc a -m2 -d64m -r -t2 out.csa /disk2/*
csarc x -t2 -o /tmp/ out.csa *.jpg
csarc l out.csa
csarc t out.csa *.dll

### Introduction for libcsc:
* The whole compressor was mostly inspired by LZMA, with some ideas from others or my own mixed.
* Based on LZ77 Sliding Window + Range Coder. Literals are with pure order-1, literal/match flag was coded with state of previous 3 flag. Match length / distance was coded into different slots. The schema is quite similar with LZMA.
* Match finder for LZ77 is using 2-bytes hash table and 3-bytes hash table with only one candidate. And for -m1 .. -m4, another 6-bytes hash table was used, with width(candidates) of 1, 8, 2, 8. For -m5, there is binary tree with 6-bytes hash searching head. The implementation for binary tree was from LZMA. The binary tree match finder does not hold the whole dictionary range, instead it is smaller, to avoid 10x dictionary size memory usage. But the searching head table for BT does not have such limitation.
* Two kinds of parser: Lazy one and advanced one. Advanced parser is calculating the lowest price path for every several KB chunk. -m1 , -m2 is using the lazy one, -m3 .. -m5 is with advanced. Details for m1 .. m5 configuration can be find in csc_enc.cpp
* A simple data analyzer is being used on every 8KB of raw data block. It calculates: the order-0 bits/Byte of current block, data similarity of every 1,2,3,4,8 bytes for potential data tables, heuristic method to tell EXE codes or English text.
* Continuous blocks with same type analyzed will be compressed as one chunk, base on its type, on it may apply:
  * e8e9 preprocessor, Engeue Shelwien gave me the code before.
  * A very simple & fast English preprocessor, just replacing some words into char 128 to 255. The dictionary was from Sami Runsas.
  * Delta preprocessor on corresponding channel, and then use order-1 direct coding instead of LZ77. The entropy for delta-processed block was also calculated and compared before to decide coding to avoid false positive.
  * No compression is applied on high entropy data, which is measured by order-0 bpB.
  * For delta and high entropy data, the original data was stored into LZ77 window and match finder. Before it is compressed, it will be checked whether the block is duplicated. So for two same block in one file, it still will be compressed into one.
* libcsc API is very LZMA liked, which is using the same Types.h in LZMA.
* Memory usage is about 2 - 4x dictionary size.

### Introduction For CSArc:
* Cross-platform thread / file / directory operation codes were from ZPAQ.
* Files were sorted by extension and size (if it is not a very small file), then split into tasks by extension.
* Each task is compressed by one thread, and there can be several threads working for multiple tasks.
* Each worker task contains separate I/O thread and buffer.
* Multi-threading also works in same way when doing extraction.
* For single big file, -p# switch can be used to force split the file into multiple tasks for multi-threading, however will hurt compression.
* Memory usage will be multiplied by the thread number.
* Adler32 checksum is being used.
* Meta info of all files is also compressed by libcsc, appended at the end of all other compressed data.


