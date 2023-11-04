kanzi
=====


Kanzi is a modern, modular, portable and efficient lossless data compressor implemented in C++.

* modern: state-of-the-art algorithms are implemented and multi-core CPUs can take advantage of the built-in multi-threading.
* modular: entropy codec and a combination of transforms can be provided at runtime to best match the kind of data to compress.
* portable: many OSes, compilers and C++ versions are supported (see below).
* expandable: clean design with heavy use of interfaces as contracts makes integrating and expanding the code easy. No dependencies.
* efficient: the code is optimized for efficiency (trade-off between compression ratio and speed).

Unlike the most common lossless data compressors, Kanzi uses a variety of different compression algorithms and supports a wider range of compression ratios as a result. Most usual compressors do not take advantage of the many cores and threads available on modern CPUs (what a waste!). Kanzi is multithreaded by design and uses several threads by default to compress blocks concurrently. It is not compatible with standard compression formats. Kanzi is a lossless data compressor, not an archiver. It uses checksums (optional but recommended) to validate data integrity but does not have a mechanism for data recovery. It also lacks data deduplication across files.

For more details, check https://github.com/flanglet/kanzi/wiki.

See how to reuse the C and C++ APIs here: https://github.com/flanglet/kanzi-cpp/wiki/Using-and-extending-the-code

There is a Java implementation available here: https://github.com/flanglet/kanzi

There is Go implementation available here: https://github.com/flanglet/kanzi-go

Credits

Matt Mahoney,
Yann Collet,
Jan Ondrus,
Yuta Mori,
Ilya Muravyov,
Neal Burns,
Fabian Giesen,
Jarek Duda, 
Ilya Grebnov

Disclaimer

Use at your own risk. Always keep a copy of your original files.

![Build Status](https://github.com/flanglet/kanzi-cpp/actions/workflows/c-cpp.yml/badge.svg)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=flanglet_kanzi-cpp&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=flanglet_kanzi-cpp)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=flanglet_kanzi-cpp&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=flanglet_kanzi-cpp)
<a href="https://scan.coverity.com/projects/flanglet-kanzi-cpp">
  <img alt="Coverity Scan Build Status"
       src="https://img.shields.io/coverity/scan/16859.svg"/>
</a>


Silesia corpus benchmark
-------------------------

i7-7700K @4.20GHz, 32GB RAM, Ubuntu 22.04

clang++ 14.0.0, tcmalloc

Kanzi version 2.1 C++ implementation. Block size is 100 MB. 


|        Compressor               | Encoding (sec)  | Decoding (sec)  |    Size          |
|---------------------------------|-----------------|-----------------|------------------|
|Original     	                  |                 |                 |   211,938,580    |
|**Kanzi -l 1 -j 1**              |    	 **1.1**    |     **0.5**     |  **69,399,477**  |
|**Kanzi -l 1 -j 6**              |      **0.4**    |     **0.2**     |  **69,399,477**  |
|Pigz 2.6 -5 -p6                  |        1.0      |       0.7       |    69,170,603    |
|Gzip 1.10 -5                     |        4.8      |       1.0       |    69,143,980    |
|Zstd 1.5.3 -2 --long=30          |	       0.9      |       0.5       |    68,694,316    |
|Zstd 1.5.3 -2 -T6 --long=30      |	       0.4      |       0.3       |    68,694,316    |
|Brotli 1.0.9 -2 --large_window=30|        1.5      |       0.8       |    68,033,377    |
|Pigz 2.6 -9 -p6                  |        3.0      |       0.6       |    67,656,836    |
|Gzip 1.10 -9                     |       15.5      |       1.0       |    67,631,990    |
|Brotli 1.0.9 -4 --large_window=30|        4.1      |       0.7       |    64,267,169    |
|**Kanzi -l 2 -j 1**              |      **2.3**    |     **0.7**     |  **63,808,747**  |
|**Kanzi -l 2 -j 6**              |      **0.9**    |     **0.3**     |  **63,808,747**  |
|Zstd 1.5.3 -9 --long=30          |	       3.7      |       0.3       |    59,272,590    |
|Zstd 1.5.3 -9 -T6 --long=30      |	       2.3      |       0.3       |    59,272,590    |
|**Kanzi -l 3 -j 1**              |      **3.5**    |     **1.3**     |  **59,199,795**  |
|**Kanzi -l 3 -j 6**              |      **1.2**    |     **0.4**     |  **59,199,795**  |
|Orz 1.5.0                        |	       7.7      |       2.0       |    57,564,831    |
|Brotli 1.0.9 -9 --large_window=30|       36.7      |       0.7       |    56,232,817    |
|Lzma 5.2.2 -3	                  |       24.1	    |       2.6       |    55,743,540    |
|**Kanzi -l 4 -j 1**              |      **6.2**    |     **4.2**     |  **54,998,198**  |
|**Kanzi -l 4 -j 6**              |      **2.0**    |     **1.2**     |  **54,998,198**  |
|Bzip2 1.0.6 -9	                  |       14.9      |       5.2       |    54,506,769    |
|Zstd 1.5.3 -19 --long=30         |       62.0      |       0.3       |    52,828,057    |
|Zstd 1.5.3 -19	-T6 --long=30     |       62.0      |       0.4       |    52,828,057    |
|**Kanzi -l 5 -j 1**              |     **11.3**    |     **4.5**     |  **51,760,244**  |
|**Kanzi -l 5 -j 6**              |      **3.6**    |     **1.5**     |  **51,760,244**  |
|Brotli 1.0.9 --large_window=30   |      356.2	    |       0.9       |    49,383,136    |
|Lzma 5.2.2 -9                    |       65.6	    |       2.5       |    48,780,457    |
|**Kanzi -l 6 -j 1**              |     **13.6**    |     **6.2**     |  **48,068,000**  |
|**Kanzi -l 6 -j 6**              |      **4.2**    |     **2.1**     |  **48,068,000**  |
|bsc 3.2.3 -b100 -T -t            |        8.8      |       6.0       |    46,932,394    |
|bsc 3.2.3 -b100                  |        5.4      |       4.9       |    46,932,394    |
|BCM 1.65 -b100                   |       15.5      |      21.1       |    46,506,716    |
|**Kanzi -l 7 -j 1**              |     **16.7**    |    **11.1**     |  **46,447,003**  |
|**Kanzi -l 7 -j 6**              |      **5.2**    |     **3.7**     |  **46,447,003**  |
|Tangelo 2.4                      |       83.2      |      85.9       |    44,862,127    |
|zpaq v7.14 m4 t1                 |      107.3	    |     112.2       |    42,628,166    |
|zpaq v7.14 m4 t12                |      108.1	    |     111.5       |    42,628,166    |
|**Kanzi -l 8 -j 1**              |     **47.8**    |    **49.4**     |  **41,821,127**  |
|**Kanzi -l 8 -j 6**              |     **15.8**    |    **15.5**     |  **41,821,127**  |
|Tangelo 2.0                      |      302.0      |     310.9       |    41,267,068    |
|**Kanzi -l 9 -j 1**              |     **72.4**    |    **74.5**     |  **40,361,391**  |
|**Kanzi -l 9 -j 6**              |     **26.1**    |    **26.9**     |  **40,361,391**  |
|zpaq v7.14 m5 t1                 |      343.1	    |     352.0       |    39,112,924    |
|zpaq v7.14 m5 t12                |	     344.3	    |     350.4       |    39,112,924    |



enwik8
-------

i7-7700K @4.20GHz, 32GB RAM, Ubuntu 22.04

clang++ 14.0.0, tcmalloc

Kanzi version 2.1 C++ implementation. Block size is 100 MB. 1 thread


|        Compressor           | Encoding (sec)  | Decoding (sec)  |    Size          |
|-----------------------------|-----------------|-----------------|------------------|
|Original     	              |                 |                 |   100,000,000    |
|**Kanzi -l 1 -j 1**          |     **0.78**    |    **0.33**     |  **37,969,539**  |
|**Kanzi -l 2 -j 1**          |     **1.65**    |    **0.56**     |  **30,953,719**  |
|**Kanzi -l 3 -j 1**          |     **2.02**    |    **0.80**     |  **27,362,969**  |
|**Kanzi -l 4 -j 1**          |	    **3.37**    |    **2.18**     |  **25,670,924**  |
|**Kanzi -l 5 -j 1**          |	    **5.14**    |    **1.82**     |  **22,490,875**  |
|**Kanzi -l 6 -j 1**          |	    **6.88**    |    **2.80**     |  **21,232,300**  |
|**Kanzi -l 7 -j 1**          |	    **8.80**    |    **5.02**     |  **20,935,519**  |
|**Kanzi -l 8 -j 1**          |	   **18.84**    |   **18.95**     |  **19,671,786**  |
|**Kanzi -l 9 -j 1**          |	   **28.25**    |   **29.03**     |  **19,097,946**  |


Build Kanzi
-----------

The C++ code can be built on Windows with Visual Studio, Linux, macOS and Android with g++ and/or clang++.
There are no dependencies. Porting to other operating systems should be straightforward.

### Visual Studio 2008
Unzip the file "Kanzi_VS2008.zip" in place.
The solution generates a Windows 32 binary. Multithreading is not supported with this version.

### Visual Studio 2022
Unzip the file "Kanzi_VS2022.zip" in place.
The solution generates a Windows 64 binary and library. Multithreading is supported with this version.

### mingw-w64
Go to the source directory and run 'make clean && mingw32-make.exe'. The Makefile contains 
all the necessary targets. Tested successfully on Win64 with mingw-w64 g++ 8.1.0. 
Multithreading is supportedwith g++ version 5.0.0 or newer.
Builds successfully with C++11, C++14, C++17.

### Linux
Go to the source directory and run 'make clean && make'. The Makefile contains all the necessary
targets. Build successfully on Ubuntu with many versions of g++ and clang++.
Multithreading is supported with g++ version 5.0.0 or newer.
Builds successfully with C++98, C++11, C++14, C++17, C++20.

### MacOS
Go to the source directory and run 'make clean && make'. The Makefile contains all the necessary
targets. Build successfully on Ubuntu with many versions of clang++.
Multithreading is supported.

### BSD
The makefile uses the gnu-make syntax. First, make sure gmake is present (or install it: 'pkg_add gmake').
Go to the source directory and run 'gmake clean && gmake'. The Makefile contains all the necessary
targets. Multithreading is supported.

### Makefile targets
```
clean:     removes objects, libraries and binaries
kanzi:     builds the kanzi executable
lib:       builds static and dynamic libraries
test:      builds test binaries
all:       kanzi + lib + test
install:   installs libraries, headers and executable
uninstall: removes installed libraries, headers and executable
```


