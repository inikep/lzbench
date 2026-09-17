AOCL-Compression
================

AOCL-Compression is a software framework of various lossless compression and
decompression methods tuned and optimized for AMD Zen based CPUs.
This framework offers a single set of unified APIs for all the supported
compression and decompression methods which facilitate the applications to
easily integrate and use them.
AOCL-Compression supports lz4, zlib/deflate, lzma, zstd, bzip2, snappy, and lz4hc
based compression and decompression methods along with their native APIs.
The library offers openMP based multi-threaded implementation for all the methods
(for LZMA, only multi-threaded compression is supported).
It supports the dynamic dispatcher feature that executes the most optimal
function variant implemented using Function Multi-versioning thereby offering
a single optimized library portable across different x86 CPU architectures.
AOCL-Compression framework is developed in C for UNIX® and Windows® based systems.
A test suite is provided for the validation and performance benchmarking
of the supported compression and decompression methods. This suite also
supports the benchmarking of IPP compression methods, such as, lz4, lz4hc, zlib and bzip2.
The library build framework offers CTest based testing of the test cases
implemented using GTest and the library test suite.


Installation
------------

1. Download the latest stable release from the Github repository:<br>
https://github.com/amd/aocl-compression
2. Install CMake (version 3.13.0+) on the machine where the sources are to be compiled.
3. Make any one of the supported compilers (GCC 8.5+ or Clang 11.0+) available on the machine.
4. Then, use the cmake based build system to compile and generate AOCL-Compression <br>
library and testsuite binary as explained below for Linux® and Windows® platforms.

Building on Linux
-----------------

1. To create a build directory and configure the build system in it, run the following:
   ```
    cmake -B <build directory> <directory containing CMakeList.txt>
   ```
   Additional options that can be specified for build configuration are:
   ```
   cmake -B <build directory> <directory containing CMakeList.txt> 
      -DCMAKE_INSTALL_PREFIX=<install path> 
      -DCMAKE_BUILD_TYPE=<Debug or Release> 
      -DBUILD_STATIC_LIBS=ON
      <Additional Library Build Options>
   ```

   To use clang compiler for the build, specify `-DCMAKE_C_COMPILER=clang` as the option.
2. Compile using the following command:
   ```
   cmake --build <build directory> --target install -j
   ```
   The library is generated in "lib" directory. <br>
   The test bench executable is generated in "build". <br>
   The additional option `--target install` will install the library, and <br>
   interface header files in the installation path as specified with <br>
   `-DCMAKE_INSTALL_PREFIX` option or in the local system path. <br>
   The option `-j` will run the compilation process using multiple cores.
3. To uninstall the installed files, run the following custom command:
   ```
   cmake --build <build directory> --target uninstall
   ```
   To uninstall and then install the build package, run the following command:
   ```
   cmake --build <build directory> --target uninstall --target install -j -v
   ```
   The option `-v` will print verbose build logs on the console.
4. To clear or delete the build folder or files, manually remove the build directory or its files.

Building with GNU Make (Linux - Limited Support)
-------------------------------------------------
GNU Make based build is provided for Linux systems with limited functional features only.

1. Ensure required tools are available: `make`, `gcc`/`clang`, `g++`/`clang++`, and `bash`.
2. From the source root, build using default configuration:
   ```
   make
   ```
3. Configure build options by overriding variables on command line (same options as `config.mk`):
   ```
   make BUILD_TYPE=Debug BUILD_STATIC_LIBS=1 AOCL_ENABLE_THREADS=1
   ```
4. Run tests when enabled:
   ```
   make test AOCL_TEST_COVERAGE=1
   ```
5. Install and uninstall using make targets:
   ```
   make install PREFIX=/usr/local
   make uninstall PREFIX=/usr/local
   ```
* NOTE: <br>
   1. **GoogleTest**: Required to build and execute the test suite

GNU Make Example
----------------
Build a Debug static library with OpenMP threading support and run tests:
```
make clean
make BUILD_TYPE=Debug BUILD_STATIC_LIBS=1 AOCL_ENABLE_THREADS=1 -j
make test AOCL_TEST_COVERAGE=1
```

Building on Windows
-------------------
As a prerequisite, make Microsoft Visual Studio® available along with <br>
__Desktop development with C++__ toolset that includes the Clang compiler.

Building with Visual Studio IDE (GUI)
-------------------------------------
1. Launch CMake GUI and set the locations for source package and build output.
2.  Click __Configure__ option and select:
      - __Generator__ as the Installed Microsoft Visual Studio Version
      - __Platform__ as __x64__
      - __Optional toolset__ as __ClangCl__
3. Select additional library config and build options.
4. Configure CMAKE_INSTALL_PREFIX appropriately.
5. Click __Generate__.
   Microsoft Visual Studio project is generated.
6. Click __Open Project__.
   Microsoft Visual Studio project for the source package __is launched__.
7. Build the entire solution or the required projects.

Building with Visual Studio IDE (command line)
----------------------------------------------
1. Go to AOCL-Compression source package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build the library and test bench executable.
```
cmake .. -T ClangCl -G <installed Visual Studio version> && cmake --build . --config Release --target INSTALL
```
You can pass additional library configuration and build options in the command.

Additional Library Build Options
--------------------------------
Use the following additional options to configure your build:

Option                              |  Description
------------------------------------|----------------------------------------------------------------------------------------
AOCL_LZ4_OPT_PREFETCH_BACKWARDS     |  Enable LZ4 optimizations related to backward prefetching of data (Disabled by default)
SNAPPY_MATCH_SKIP_OPT               |  Enable Snappy match skipping optimization (Enabled by default)
SNAPPY_HIGH_COMPRESSION             |  Enable Snappy high compression to get better ratio by compromising on speed (Disabled by default)
SNAPPY_ENABLE_DECOMPRESS_BRANCHLESS |  Enable Snappy branchless decompression optimization (Disabled by default for GCC and enabled for all other compilers)
LZ4_FRAME_FORMAT_SUPPORT            |  Enable building LZ4 with Frame format and API support (Disabled by default)
AOCL_LZ4HC_DISABLE_PATTERN_ANALYSIS |  Disable Pattern Analysis in LZ4HC for level 9 (Enabled by default)
AOCL_ZSTD_SEARCH_SKIP_OPT           |  Enable ZSTD match skipping optimization that steps more aggresively when matches are not found (Enabled by default)
AOCL_ZSTD_DYN_BLOCK_SIZE            |  Enable ZSTD dynamic block size determination (Disabled by default)
AOCL_DECOMPRESS_FAST                |  Enable fast decompression modes that might compromise on compression speed / ratio to produce streams that decompress faster. Supported values: {1,2,3} ZSTD, {1,2} Snappy, {1} LZ4. (Disabled by default)
AOCL_COMPRESS_FAST                  |  Enable fast compression modes that might compromise on compression ratio but compress faster. Supported values: {1,2} ZSTD. (Disabled by default)
AOCL_TEST_COVERAGE                  |  Enable GTest, AOCL test bench and third party test bench based CTest suite (Disabled by default)
AOCL_ENABLE_LOG_FEATURE             |  Enables logging through environment variable `AOCL_ENABLE_LOG` (Disabled by default)
CODE_COVERAGE                       |  Enable code coverage (GCC/gcov for Linux and Clang/llvm-cov for both Linux and Windows) (Disabled by default)
ASAN                                |  Enable Address Sanitizer checks. Only supported on Linux/Debug build (Disabled by default)
VALGRIND                            |  Enable Valgrind checks. Only supported on Linux/Debug and incompatible with ASAN=ON (Disabled by default)
BUILD_DOC                           |  Build documentation for this library (Disabled by default)
BUILD_EXAMPLE                       |  Build examples for aocl-compression (Disabled by default)
AOCL_LZ4_MATCH_SKIP_OPT_LDS_STRAT1  |  Enable LZ4 match skipping optimization strategy-1 based on a larger base step size applied for long distance search (Disabled by default)
AOCL_LZ4_MATCH_SKIP_OPT_LDS_STRAT2  |  Enable LZ4 match skipping optimization strategy-2 by aggressively setting search distance on top of strategy-1. Preferred to be used with Silesia corpus (Disabled by default)
AOCL_LZ4_NEW_PRIME_NUMBER           |  Enable the usage of a new prime number for LZ4 hashing function. Preferred to be used with Silesia corpus (Disabled by default)
AOCL_LZ4_EXTRA_HASH_TABLE_UPDATES   |  Enable storing of additional potential matches to improve compression ratio. Recommended for higher compressibility use cases (Disabled by default)
AOCL_LZ4_HASH_BITS_USED             |  Control the number of bits used for LZ4 hashing, allowed values are OFF, LOW (low perf gain and less CR regression) and HIGH (high perf gain and high CR regression) (LOW by default)
AOCL_EXCLUDE_BZIP2                  |  Exclude BZIP2 compression method from the library build (Disabled by default)
AOCL_EXCLUDE_LZ4                    |  Exclude LZ4 compression method from the library build. LZ4HC also gets excluded (Disabled by default)
AOCL_EXCLUDE_LZ4HC                  |  Exclude LZ4HC compression method from the library build (Disabled by default)
AOCL_EXCLUDE_LZMA                   |  Exclude LZMA compression method from the library build (Disabled by default)
AOCL_EXCLUDE_SNAPPY                 |  Exclude SNAPPY compression method from the library build (Disabled by default)
AOCL_EXCLUDE_ZLIB                   |  Exclude ZLIB compression method from the library build (Disabled by default)
AOCL_EXCLUDE_ZSTD                   |  Exclude ZSTD compression method from the library build (Disabled by default)
AOCL_XZ_UTILS_LZMA_API_EXPERIMENTAL |  Build with xz utils lzma APIs. Experimental feature with limited API support (Disabled by default)
AOCL_ENABLE_THREADS                 |  Enable multi-threaded compression and decompression using SMP based openMP threads (Disabled by default)
TEST_COVERAGE_THIRD_PARTY           |  Enable third party test bench based CTest suite (Disabled by default)
NATIVE_ENABLE_THREADS               |  Enable native multi-threaded compression for supported methods (Disabled by default)
AOCL_TEST_FUZZER                    |  Enable fuzz test along with GTest. Only supported on Linux with the Clang compiler (Disabled by default)
AOCL_TEST_FUZZER_WITH_CORPUS        |  Run fuzz tests with corpus. Only supported on Linux with the Clang compiler (Disabled by default)
ENABLE_FAST_MATH                    |  Enable fast-math optimizations (Disabled by default)
BUILD_UTILITY                       |  Enable third party utility build: minigzip(zlib), zstd_utility(zstd) (Disabled by default)
AOCL_BZIP2_HUFFMAN_ITERATIONS       |  Control number of BZIP2 Huffman tables refinement iterations (1-4). Lower values are faster but reduce compression ratio. (Default: 3)
AOCL_LLC_PREFIX                     |  Prefix library symbols (Disabled by default)

* NOTE: <br>
   1. ZLIB supports quicker compression strategy for Level 1 by trading off compression ratio. Enable it by
   setting environment variable AOCL_ZLIB_QUICK_MODE=ON. It also improves performance for levels 2, 3 and 5
   while trading off compression ratio. <br>
   2. **Threading Options Conflict**: If both `AOCL_ENABLE_THREADS` and `NATIVE_ENABLE_THREADS` are enabled, 
   `NATIVE_ENABLE_THREADS` will be automatically disabled to avoid conflicts. <br>
   3. **BUILD_UTILITY Forces Static Build**: When `BUILD_UTILITY=ON`, the build system automatically forces 
   `BUILD_STATIC_LIBS=ON` as some utilities cannot link to shared libraries. <br>
   4. AOCL LZ4HC optimizations are disabled when `LZ4_FRAME_FORMAT_SUPPORT` is enabled. <br>


Running AOCL-Compression Test Bench On Linux
--------------------------------------------

* CAUTION: <br>
   Before running the test bench, check whether it points to the right library dependency. <br>

Test bench supports several options to validate, benchmark or debug the supported
compression methods.
It uses the unified API set to invoke the compression methods supported by AOCL-Compression.
Test bench can invoke and benchmark some of the IPP's compression methods as well.

* To check various options supported by the test bench, use one of the following commands:<br>
  `aocl_compression_bench -h`  
  `aocl_compression_bench --help`

* To check all the supported compression methods, use the command:<br>
  `aocl_compression_bench -l`

* To run the test bench with requested number of iterations, use the command:<br>
  `aocl_compression_bench -i`

* To run the test bench to check the performance of all the supported compression <br>
   and decompression methods for a given input file, use the command:<br>
   `aocl_compression_bench -a -p <input filename>`

* To run the test bench to validate the outputs from all the supported compression <br>
   and decompression methods for a given input file, use the command:<br>
   `aocl_compression_bench -a -t <input filename>`

* To run the test bench to check the performance of a compression and decompression <br>
   method for a given input file, use the command:<br>
   `aocl_compression_bench -ezstd:5:0 -p <input filename>`<br>
Here, 5 is the level and 0 is the additional parameter passed to ZSTD method.


* To run the test bench to validate the output of a compression and decompression <br>
   method for a given input file, use the command:<br>
   `aocl_compression_bench -ezstd:5:0 -t <input filename>`<br>
   Here, 5 is the level and 0 is the additional parameter passed to ZSTD method.
  

* To run the test bench with error/debug/trace/info logs, build the library by using `-DAOCL_ENABLE_LOG_FEATURE=ON` & set the environment variable `AOCL_ENABLE_LOG` to any of the following:<br>
   * `AOCL_ENABLE_LOG=ERR`   for Error logs.
   * `AOCL_ENABLE_LOG=INFO`  for Error, Info logs.
   * `AOCL_ENABLE_LOG=DEBUG` for Error, Info, Debug logs.
   * `AOCL_ENABLE_LOG=TRACE` for Error, Info, Debug, Trace logs.<br>
  Note: When building the library for highest performance, do not enable `AOCL_ENABLE_LOG_FEATURE`.


* To run the test bench but only compression or decompression <br>
   for a given input file, use the command:<br>
   `aocl_compression_bench -rcompress <input filename>` or <br>
   `aocl_compression_bench -rdecompress -ezstd <compressed input filename>` or <br>
   `aocl_compression_bench -rdecompress -ezstd -t -f<uncompressed file for validation> <compressed input filename>` <br>
   Note: In -rdecompress mode, compression method must be specified using -e option. <br>
   If validation of decompressed data is needed, specify -t and -f options additionally.

* To run the test bench and dump output data generated <br>
   for a given input file, use the command:<br>
   `aocl_compression_bench -d<dump filename> -ezstd:1 <input filename>` or <br>
   `aocl_compression_bench -d<dump filename> -rcompress -ezstd:1 <input filename>` or <br>
   `aocl_compression_bench -d<dump filename> -rdecompress -ezstd <compressed input filename>` <br>
   Here, when -rcompress operation is selected, compressed file gets dumped <br>
   and when -rdecompress operation is selected, decompressed file gets dumped. <br>
   Method name and level must be specified using -e for default and -rcompress modes. <br>
   Method name must be specified using -e for -rdecompress mode. <br>

* To run the test bench and test native APIs, use the command: <br>
   `aocl_compression_bench -n -p <input filename>` <br>
   Other options -e, -i, -t, -r are supported when running with -n <br>

* To run the test bench and test multi-threaded native APIs, <br>
  for supported methods, use the command: <br>
  `aocl_compression_bench -e<method>:<level>:<num-of-workers> -n -p <input filename>` <br>

* To run the test bench and test native APIs with an external dictionary file <br>
  for supported methods, use the command: <br>
  `aocl_compression_bench -e<method> -n -p -y<dictionary filename> <input filename>` <br>

* NOTE: <br>
   1. Compression and decompression of large files (>1GB) are supported in the test bench. <br>
   2. Decompression of compressed files (> 1GB) that are not generated by aocl-compression <br> 
      is not guaranteed by the test bench. <br>
 
---
  
To test and benchmark the performance of IPP's compression methods, use the
test bench option `-c<path to IPP library method>` along with other relevant options (as explained above).
IPP's lz4, lz4hc, zlib and bzip2 methods are supported by the test bench.
Check the following details for the exact steps:
1. Set the library path environment variable (export LD_LIBRARY_PATH on <br>
   Linux) to point to the installed IPP library path. <br>
   Alternatively, you can also run vars.sh that comes along with the <br>
   IPP installation to setup the environment variable.
2. Download lz4-1.9.3, zlib-1.2.11 and bzip2-1.0.8 source packages.
3. Apply IPP patch files using the command:<br>
   `patch -p1 < path to corresponding patch file>`

4. Build the patched IPP lz4, zlib and bzip2 libraries per the steps <br>
   in the IPP readme files in the corresponding patch file <br>
   locations for these compression methods.
5. Append the library path to `-c` option and pass it to executable as command line argument <br>
   (Linux is only supported) for running patched IPP lz4, zlib and bzip2 libraries.
6. Run the test bench to benchmark the IPP library methods as follows:
```
    aocl_compression_bench -a -p -c/path/to/ipp_patch <input filename>
    aocl_compression_bench -elz4 -p -c/path/to/ipp_patch <input filename>
    aocl_compression_bench -elz4hc -p -c/path/to/ipp_patch <input filename>
    aocl_compression_bench -ezlib -p -c/path/to/ipp_patch <input filename>
    aocl_compression_bench -ebzip2 -p -c/path/to/ipp_patch <input filename>
```

Running AOCL-Compression Test Bench On Windows
----------------------------------------------

* CAUTION: <br>
   Before running the test bench, ensure it points to the right library dependencies for aocl_compression, openMP, etc. <br>

Test bench on Windows supports all the user options as Linux,
except for the `-c` option to link and test IPP compression methods.
For more information on various user options, refer to the previous section on Linux.
To set and launch the test bench with a specific user option,
go to project aocl_compression_bench -> Properties -> Debugging;
specify the user options and the input test file.

Running AOCL-Compression Examples
---------------------------------

* CAUTION: <br>
   Before running the example programs, ensure it points to the right library dependencies for aocl_compression, openMP, etc. <br>

Example programs are provided for both unified API and native APIs of each compression method.
The library should be built with -DBUILD_EXAMPLE=ON. Other cmake options including 
-DAOCL_ENABLE_THREADS=ON can be enabled as desired.

* To run example program for unified API, use the command:<br>
  `example_unified_api <input filename>`

* To run example program for LZ4 native API, use the command:<br>
  `example_LZ4_compress_default <input filename>`

* To run example program that demonstrates obtaining format compliant compressed stream from multithreaded unified API,
  build the library by using -DAOCL_ENABLE_THREADS=ON and run the command:<br>
  `example_aocl_llc_skip_rap_frame <input filename>`

* To run example program that demonstrates obtaining format compliant gzip compressed stream from multithreaded API,
  build the library by using -DAOCL_ENABLE_THREADS=ON and run the command:<br>
  `example_compress2_gzip <input filename>`

Running tests with CTest
------------------------

CTest is configured in CMake build system to run the test cases implemented with GTest and AOCL Test Bench for Silesia, Calgary, and Canterbury datasets.
To enable testing with CTest, use AOCL_TEST_COVERAGE option while configuring the CMake build.

Following are a few sample commands that can be executed in the build directory to run the test cases with CTest.

 To run all the tests (GTest and Test bench)<br>
 `ctest` 
 
 To only run Test bench<br>
 `ctest -R BENCH`
 
 To run GTest test cases for a specific method<br>
 `ctest -R <METHOD_NAME_IN_CAPITALS>`

Running fuzzer tests
--------------------

To list all the fuzz tests available for a method, use the following command:
   `<METHOD_GTEST_EXECUTABLE> --list_fuzz_tests`
   example: `zlib_gtest --list_fuzz_tests`

Fuzzer test can be run in two modes:

1. Unit test mode: Default operation mode of AOCL_TEST_FUZZER. Can be run as part of ctest. No sanitizer and coverage instrumentation.
   `ctest -R <TestSuiteName>.<FuzzTestName>`
2. Fuzzing mode: Enabled with cmake option FUZZTEST_FUZZING_MODE. Runs each fuzz test with sanitizer and coverage instrumentation

   To run all fuzz tests for a specified duration, use the following command:
   `<METHOD_GTEST_EXECUTABLE> --fuzz_for=<DURATION>`
   example: `zlib_gtest --fuzz_for=60s`

   To run a single fuzz test until a bug is found or until manually stopped:
   `<METHOD_GTEST_EXECUTABLE> --fuzz=<TestSuiteName>.<FuzzTestName>`
   example: `zlib_gtest --fuzz=AOCL_Compression_zlib.compress2_fuzz`

   To run a single fuzz test by feeding in an external corpus of seeds: Enabled with cmake option AOCL_TEST_FUZZER_WITH_CORPUS.
   Place folders containing seed files in the directory pointed by environment variable AOCL_FUZZ_CORPUS_DIR.
   Sub-folders under this must be as follows:
   *   /compress_fuzz : Must contain uncompressed raw files for compress API fuzz tests.
   *   /*_fuzz        : Folders with individual fuzz test names must contain compressed files 
                        for respective methods used for decompress API fuzz tests.
                        Example: /LZ4_decompress_safe_fuzz, /RawUncompress_fuzz, etc
   Run the single fuzz test:
   `<METHOD_GTEST_EXECUTABLE> --fuzz=<TestSuiteName>.<FuzzTestName>`
   example: `zlib_gtest --fuzz=AOCL_Compression_zlib.compress2_fuzz`
   Additional seed properties can be specified by environment variables:
   *  AOCL_FUZZ_SIZE_MAX : Max size in bytes to use for i/o buffers used in fuzz testing.
   *  AOCL_FUZZ_CPR_RATIO : Compression ratio estimate of compressed files used for decompress API fuzz tests.

Running source code coverage
---------------------------------------

To measure source code coverage, use CODE_COVERAGE option while configuring the CMake build. Run CMake with the custom target option 'code-coverage' to execute tests and generate code coverage data. The code coverage reports are generated in the build directory under subdirectory called 'coverage/html_report'. Open the HTML files in browser to view the coverage information.
Supports Linux (GCC/Clang) and Windows (ClangCL). The build system automatically detects the compiler type and uses the appropriate coverage tool.

Following is the sample command usage to run code coverage:
`cmake -B <build directory> <directory containing CMakeList.txt> 
      -DCMAKE_INSTALL_PREFIX=<install path> 
      -DCMAKE_BUILD_TYPE=Debug 
      -DBUILD_STATIC_LIBS=ON
      -DCODE_COVERAGE=ON
      <Additional Library Build Options>`
`cmake --build <build directory> --target install code-coverage`

Running Valgrind and ASAN memory checks using CTest
---------------------------------------------------

Use VALGRIND option for Valgrind memory check and ASAN option for ASAN memory check while configuring the CMake build. VALGRIND and ASAN options can not be enabled together.

Following are the commands to execute in the 'build' directory to run memory checks.

 To run Valgrind memory check<br>
 `ctest -T memcheck` 
 
 To run ASAN memory check<br>
 `ctest`

Running Performance Benchmarking
--------------------------------

Use test_speed.py script to benchmark performance and compare AOCL-Compression library with other 
compression libraries such as open-source reference or IPP. It generates summary reports describing compression/decompression speeds and compression ratio.

Following are a few sample commands to use the script available in the 'scripts' directory.

 To print usage options<br>
 `python3 test_speed.py --help` 
 
 To run AOCL optimized vs Reference methods for lz4, snappy and zlib levels 1 and 2:<br>
 `python3 test_speed.py --dataset $PATH_DATASETS_DIR -m lz4 snappy zlib:1 zlib:2 -cw vanilla`
 
 To run AOCL optimized vs IPP for lz4 method:<br>
 `python3 test_speed.py --dataset $PATH_DATASETS_DIR -m lz4 -cw ipp --ipp $IPP_PATCHED_LZ4_LIBS_PATH` 
 

Generating Documentation
------------------------
- To generate documentation, specify the `-DBUILD_DOC=ON` option while building.
- Documents will be generated in HTML format in the folder __docs/sphinx/html__ . Open index.html file from the folder in any browser to view the documentation.
- The following packages are expected before running CMake with `-DBUILD_DOC=ON` option:
   1. Doxygen.
   2. Python packages:
      - Sphinx
      - rocm_docs
      - breathe
      - myst_parser
- CMake halts if required packages are missing by providing directives for installing the absent packages.

Enabling/disabling optimizations
--------------------------------
- AOCL optimizations can be disabled by setting the environment variable AOCL_DISABLE_OPT to ON.
- Reference code paths are taken in such a scenario.
- This needs to be set before launching the application for it to take effect.
- If optimization is turned off via aocl_compression_desc::optOff (= 1) passed to aocl_llc_setup(), then reference code paths are taken.
- If optimization is turned on  via aocl_compression_desc::optOff (= 0) passed to aocl_llc_setup(), then AOCL_DISABLE_OPT is checked 
  additionally to override aocl_compression_desc::optOff value.

Enabling specific instructions (ISA)
------------------------------------
- AOCL optimizations can be restricted to certain ISAs by setting the environment variable 
  AOCL_ENABLE_INSTRUCTIONS. Supported values are SSE2, AVX, AVX2 and AVX512.
- This ensures optimized code paths with ISAs above the set value are not taken. E.g. If 
  it is set to AVX, no AVX2 and AVX512 optimized code paths are taken.
- This needs to be set before launching the application for it to take effect.
- It takes precedence over aocl_compression_desc::optLevel setting passed to aocl_llc_setup().
- Note: When calling aocl_llc_setup() API from multiple threads, changing aocl_compression_desc::optOff
  and aocl_compression_desc::optLevel values between threads can lead to undefined behaviour.

Multi-threaded Compression and Decompression
--------------------------------------------
- AOCL-Compression provides parallel compression and decompression capabilities for multiple formats:
  lz4, lz4hc, zlib (including zlib, deflate, and gzip formats), zstd, snappy, bzip2, and lzma.
  Note: For lzma, only multi-threaded compression is currently supported.
- The parallel processing is implemented using OpenMP multi-threading. To enable parallel decompression
  of compressed streams and files, AOCL-Compression introduces a RAP (Random Access Point) frame format.
- Enable multi-threading support by using the `AOCL_ENABLE_THREADS` configuration option.
- A stream compressed with multi-threaded AOCL-Compression library can be decompressed using any
  single-threaded standard decompressor by simply skipping the initial block of bytes containing
  the RAP frame present at the start of the stream.
- The multi-threaded compression support is optimally tuned for AMD CPUs on Linux® OS whereas
  this support is experimental for Windows® platforms.


CONTACTS
--------
AOCL-Compression is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.