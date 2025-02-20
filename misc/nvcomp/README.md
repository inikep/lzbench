# What is nvCOMP?

nvCOMP is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications.

## Version 2.2 Release

This minor release of nvCOMP introduces a new high-level interface and 2 extremely fast entropy-only compressors: ANS and gdeflate::ENTROPY_ONLY (see performance charts below).

The redesigned [**high-level**](doc/highlevel_cpp_quickstart.md) interface in nvCOMP 2.2 enhances user experience by storing metadata in the compressed buffer. It can also manage the required scratch space for the user. Finally, unlike the low-level API, the high level interface automatically splits the data into independent chunks for parallel processing. This enables the easiest way to ramp up and use nvCOMP in applications, maintaining similar level of performance as the low-level interface. In nvCOMP 2.2 all compressors are available through both low-level and high-level APIs.

## nvCOMP Compression algorithms

- Cascaded: Novel high-throughput compressor ideal for analytical or structured/tabular data.
- LZ4: General-purpose no-entropy byte-level compressor well-suited for a wide range of datasets.
- Snappy: Similar to LZ4, this byte-level compressor is a popular existing format used for tabular data.
- GDeflate: Proprietary compressor with entropy encoding and LZ77, high compression ratios on arbitrary data.
- Bitcomp: Proprietary compressor designed for floating point data in Scientific Computing applications.
- ANS: Proprietary entropy encoder based on asymmetric numeral systems (ANS).

## Compression algorithm sample results

Compression ratio and performance plots for each of the compression methods available in nvCOMP are now provided. Each column shows results for a single column from an analytical dataset derived from [Fannie Mae’s Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html). The presented results are from the 2009Q2 dataset. Instructions for generating the column files used here are provided in the benchmark section below. The numbers were collected on a NVIDIA A100 40GB GPU (with ECC on). 

<center><strong>CompressionRatios</strong></center>

![compression ratio](/doc/CompressionRatios.svg)

<center><strong>CompressionThroughput</strong></center>

![compression performance](/doc/CompressionThroughput.svg)

<center><strong>DecompressionThroughput</strong></center>

![decompression performance](/doc/DecompressionThroughput.svg)

## Known issues
* Cascaded, GDeflate and Bitcomp decompressors can only operate on valid input data (data that was compressed using the same compressor). Other decompressors can sometimes detect errors in the compressed stream. However, there are no implicit checksums implemented for any of the compressors. For full verification of the stream, it's recommended to run checksum separately.  
* Cascaded and Bitcomp batched decompression C APIs cannot currently accept nullptr for actual_decompressed_bytes or device_statuses values.
* The Bitcomp low-level batched decompression function is not fully asynchronous.

## Requirements
To build / use nvCOMP, the following are required:
* Compiler with full C++ 14 support (e.g. GCC 5, Clang 3.4, MSVC 2019)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 10.2 or higher
  * If the CUDA Toolkit is version 10.2, [CUB version 1.8](https://github.com/thrust/cub/tree/1.8.0) is also required
* [CMake](https://cmake.org/) 3.18 or higher
* GNU make on Linux, or Microsoft Visual C++ 2019 on Windows
* Pascal (sm60) or higher GPU architecture is required. 
  * Volta (sm70)+ GPU architecture is recommended for best results. GDeflate requires Volta+.

# Getting Started
Below you can find instructions on how to build the library, reproduce our benchmarking results, a guide on how to integrate into your application and a detailed description of the compression methods. Enjoy!

## Building the library, optionally with nvCOMP extensions
To begin, you can download the [nvCOMP source code](https://github.com/NVIDIA/nvcomp.git) or use `git clone` to download it to a subdirectory of the current directory by running:
```
git clone https://github.com/NVIDIA/nvcomp.git
```

Create a directory named "build" inside the directory to which nvCOMP was downloaded and make that the current directory, e.g. from the command line after cloning with git:
```
cd nvcomp
mkdir build
cd build
```

To use the nvCOMP extensions, first, download nvCOMP extensions from the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp), into a directory that is not inside the nvCOMP source directory.
There are three available extensions.
1. Bitcomp
2. GDeflate
3. ANS

To build on Linux with GNU make:
```
cmake -DNVCOMP_EXTS_ROOT=/path/to/nvcomp_exts/${CUDA_VERSION} ..
make -j
```

To build on Windows with Microsoft Visual C++ 2019:
```
cmake -DNVCOMP_EXTS_ROOT=C:/path/to/nvcomp_exts/${CUDA_VERSION} -G "Visual Studio 16 2019" -A x64 -Tcuda=${CUDA_VERSION} ..
```
Then open the Visual Studio solution file nvcomp.sln in VS2019, and select "Build Solution" from the Build menu.

For either platform, optionally specify `-DBUILD_BENCHMARKS=ON` or `-DBUILD_EXAMPLES=ON` or `-DBUILD_TESTS=ON` on the cmake command line to include any combination of benchmarks, examples, or tests.  To omit the nvCOMP extensions, omit `-DNVCOMP_EXTS_ROOT=<path to exts>` from the `cmake` command line.  When building using CUDA 10.2, you will need to specify a path to [CUB] on your system by adding a `-DCUB_DIR=<path to cub repository>` argument.  To specify where to install the built nvCOMP libraries, specify `-DCMAKE_INSTALL_PREFIX=<path to install to>`

After the library is built, it can then be installed on Linux via:
```
make install
```
or on Windows by right-clicking the "INSTALL" project in Visual Studio and selecting "Build".  This will copy the `libnvcomp.so` or `nvcomp.lib` file into `<path to install to>/lib/libnvcomp.so` and the header files into `<path to install to>/include/`, with the path specified by `CMAKE_INSTALL_PREFIX` as above.

## How to use the library in your code

* [High-level Quick Start Guide](doc/highlevel_cpp_quickstart.md)
* [Low-level Quick Start Guide](doc/lowlevel_c_quickstart.md)


## Further information about some of our compression algorithms

* [Algorithms overview](doc/algorithms_overview.md)

## Running benchmarks

By default the benchmarks are not built. To build them, pass `-DBUILD_BENCHMARKS=ON` to cmake as described above.  This will result in the benchmarks being placed inside of the `bin/` directory.

To obtain TPC-H data:
- Clone and compile https://github.com/electrum/tpch-dbgen
- Run `./dbgen -s <scale factor>`, then grab `lineitem.tbl`

To obtain Mortgage data:
- Download any of the archives from https://docs.rapids.ai/datasets/mortgage-data
- Unpack and grab `perf/Perforamnce_<year><quarter>.txt`, e.g. `Perforamnce_2000Q4.txt`

Convert CSV files to binary files:
- `benchmarks/text_to_binary.py` is provided to read a `.csv` or text file and output a chosen column of data into a binary file
- For example, run `python benchmarks/text_to_binary.py lineitem.tbl <column number> <datatype> column_data.bin '|'` to generate the binary dataset `column_data.bin` for TPC-H lineitem column `<column number>` using `<datatype>` as the type
- *Note*: make sure that the delimiter is set correctly, default is `,`

Run benchmarks:
- Various benchmarks are provided in the benchmarks/ folder. For example, here we demonstrate execution of the low-level and high level benchmarks for lz4.
Other formats can be executed similarly.

Below are some example benchmark results on a A100 for the Mortgage 2009Q2 column 0:

```
./bin/benchmark_hlif lz4 -f /data/nvcomp/benchmark/mortgage-2009Q2-col0-long.bin
----------
uncompressed (B): 329055928
comp_size: 8582564, compressed ratio: 38.34
compression throughput (GB/s): 90.48
decompression throughput (GB/s): 312.81
```

```
./bin/benchmark_lz4_chunked -f /data/nvcomp/benchmark/mortgage-2009Q2-col0-long.bin
----------
uncompressed (B): 329055928
comp_size: 8461988, compressed ratio: 38.89
compression throughput (GB/s): 95.87
decompression throughput (GB/s): 320.70
```

## Running examples

By default the examples are not built. To build the CPU compression examples, pass `-DBUILD_EXAMPLES=ON` to cmake as described above.  This will result in the examples being placed inside of the `bin/` directory.

These examples require some external dependencies namely:
- [zlib](https://github.com/madler/zlib) for the GDeflate CPU compression example (`zlib1g-dev` on debian based systems)
- [LZ4](https://github.com/lz4/lz4) for the LZ4 CPU compression example (`liblz4-dev` on debian based systems)
- [GPU Direct Storage](https://developer.nvidia.com/blog/gpudirect-storage/) for the corresponding example

Run examples:
- Run `./bin/gdeflate_cpu_compression` or `./bin/lz4_cpu_compression` with `-f </path/to/datafile>` to compress the data on the CPU and decompress on the GPU.

Below are the CPU compression example results on a RTX A6000 for the Mortgage 2000Q4 column 12:
```
$ ./bin/gdeflate_cpu_compression -f /Data/mortgage/mortgage-2009Q2-col12-string.bin 
----------
files: 1
uncompressed (B): 164527964
chunks: 2511
comp_size: 1785796, compressed ratio: 92.13
decompression validated :)
decompression throughput (GB/s): 152.88

$ ./bin/lz4_cpu_compression -f /Data/mortgage/mortgage-2009Q2-col12-string.bin 
----------
files: 1
uncompressed (B): 164527964
chunks: 2511
comp_size: 2018066, compressed ratio: 81.53
decompression validated :)
decompression throughput (GB/s): 160.35
```
