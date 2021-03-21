# What is nvcomp?
nvcomp is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors in their applications.

nvcomp 1.2 includes Cascaded and LZ4 compression methods. Cascaded compression methods demonstrate high performance with up to 500GB/s throughput and a high compression ratio of up to 80x on numerical data from analytical workloads. LZ4 methods feature up to 60 GB/s decompression throughput and good compression ratios for arbitrary byte streams.

Below are performance vs. compression ratio scatter plots for a few numerical columns from TPC-H and Fanny Mae’s Mortgage datasets for Cascaded compression (R1D1B1), and string columns from TPC-H for LZ4 compression. For the TPC-H dataset we used SF10 lineitem table, and the following columns: column 0 (L_ORDERKEY) as 8B integers, and columns 8, 9, 13, 14, 15 as byte streams. From the Mortgage dataset we used 2009 Q2 performance table: column 0 (LOAN_ID) as 8B integers and column 10 (CURRENT_LOAN_DELINQUENCY_STATUS) as 4B integers. The numbers were collected on a Tesla V100 PCIe card (with ECC on). Note that you can tune the Cascaded format settings (e.g. the number of RLE layers) for even better compression ratio for some of these datasets.  We also provide a fast auto-selector that can be used to quickly determine the best Cascaded format settings to use for your dataset (details are in the [Cascaded Selector Guide](doc/selector-quickstart.md)).

![Cascaded compression performance](/doc/cascaded-perf.png)

![LZ4 performance](/doc/LZ4-perf.png)

The library is designed to be modular with ability to add new implementations without changing the high-level user interface. We’re working on additional schemes, and also on a “how to” guide for developers to add their own custom algorithms. Stay tuned for updates!

Below you can find instructions on how to build the library, reproduce our benchmarking results, a guide on how to integrate into your application and a detailed description of the compression methods. Enjoy!

# Version 1.2 Release

This version of nvcomp adds the
[Cascaded Selector](/dec/selector-quickstart) set of interfaces,
for automating the process of configuring Cascaded compression as well as other
improvements.
Full details in [CHANGELOG.md](CHANGELOG.md).

## Known issues

* Cascaded compression requires a large amount of temporary workspace to
operate. Current workaround is to compress/decompress large datasets in pieces,
re-using temporary workspace for each piece.

# Building the library
NVComp uses CMake for building. Generally, it is best to do an out of source build:
```
mkdir build/
cd build
cmake ..
make
```

If you're building using CUDA 10 or less, you will need to specify a path to
[CUB](https://github.com/thrust/cub) on your system, of at least version
1.9.10.

```
cmake -DCUB_DIR=<path to cub repository>
```

# Running benchmarks
GPU requirement:
* It's recommended to run on Volta architecture or higher (the library was not tested on Pascal and below, but may work)

To obtain TPC-H data:
- Clone and compile https://github.com/electrum/tpch-dbgen
- Run `./dbgen -s <scale factor>`, then grab `lineitem.tbl`

To obtain Mortgage data:
- Download [16 Years](http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2015.tgz) archive from https://rapidsai.github.io/demos/datasets/mortgage-data
- Unpack and copy `mortgage/perf/Perforamnce_2009Q2.txt`

Convert CSV files to binary files:
- `benchmarks/text_to_binary.py` is provided to read a `.csv` or text file and output a chosen column of data into a binary file
- For example, run `python benchmarks/text_to_binary.py lineitem.tbl <column number> <datatype> column_data.bin '|'` to generate the binary dataset `column_data.bin` for TPC-H lineitem column `<column number>` using `<datatype>` as the type
- *Note*: make sure that the delimiter is set correctly, default is `,`

Run tests:
- Run `./bin/benchmark_cascaded` or `./bin/benchmark_lz4` with `-f column_data.bin <options>` to measure throughput.

Below are some reference benchmark results on a Tesla V100 for TPC-H.

Example of compressing and decompressing TPC-H SF10 lineitem column 0 (L_ORDERKEY) using Cascaded with RLE + Delta + Bit-packing (input stream is treated as 8-byte integers):

```
$ ./bin/benchmark_cascaded -f /tmp/lineitem-col0-long.bin -t long -r 1 -d 1 -b 1 -m
----------
uncompressed (B): 479888416
compression memory (input+output+temp) (B): 2400694079
compression temp space (B): 1200972879
compression output space (B): 719832784
comp_size: 15000160, compressed ratio: 31.99
compression throughput (GB/s): 184.06
decompression memory (input+output+temp) (B): 930155136
decompression temp space (B): 435266560
decompression throughput (GB/s): 228.53
```

Example of compressing and decompressing TPC-H SF10 lineitem column 15 (L_COMMENT) using LZ4:

```
$ ./bin/benchmark_lz4 -f lineitem-col15-string.bin -m
----------
uncompressed (B): 2579400236
compression memory (input+output+temp) (B): 7761407820
compression temp space (B): 2591870472
compression output space (B): 2590137112
comp_size: 1033546459, compressed ratio: 2.50
compression throughput (GB/s): 3.00
decompression memory (input+output+temp) (B): 3613891311
decompression temp space (B): 944616
decompression throughput (GB/s): 27.48
```

*Note*: Your TPC-H performance results may not precisely match the output above, since the TPC-H data generator produces random data - you need to use the same seed to have byte-matching input files.

# How to use the library in your code

* [C++ Quick Start Guide](doc/cpp_quickstart.md)
* [Batched Compression/Decompression Guide](doc/batched-quickstart.md)
* [Cascaded Format Selector Guide](doc/selector-quickstart.md)

# Further information about our compression algorithms

* [Algorithms overview](doc/algorithms_overview.md)
