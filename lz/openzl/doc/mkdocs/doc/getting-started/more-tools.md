# More tools and compilation modes

<!-- Note: this file is currently about "storing things that used to be part of quick-start.md".
     The different parts will be re-employed in proper context later on. -->

## Build & Install the Python Extension

Optionally, you can build and install the Python extension:

```
# Optionally create a virtualenv
python -m venv openzl-virtualenv
source ./openzl-virtualenv/bin/activate

# Install the Python extension
cd py
pip install .
```

See the [Python API Reference](../api/py/introduction.md) to start using it.

## Building OpenZL

Build OpenZL and its dependencies with CMake with the following commands. These commands will:

1. Download `googletest` and `zstd` from GitHub & build them.
2. Build OpenZL.
3. Run tests.
4. Installs OpenZL and Zstd to `install/`. You can change the `CMAKE_INSTALL_PREFIX` to set the install directory, or leave it unset to install system-wide.

!!! note

    This installs `zstd` as well as `openzl`, because we currently don't use the system `zstd`.
    We have the GitHub [Issue#1728](https://github.com/facebook/openzl/issues/1728) to fix this.

```bash
mkdir build-cmake
cd build-cmake
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENZL_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=install
make -j
ctest . -j 10
make install
```

## Running an example

Find your favorite numeric data, or use this generated data:

```bash
$ yes | head -c 1000000 > my-data
```

Run help to see the options:

```bash
$ ./examples/numeric_array --help
Usage: ./examples/numeric_array [standard|sorted|int|bfloat16|float16|float32|brute_force] <width> <input>
        Compresses native-endian numeric data of the given width using the specified compressor
```

The first option is the compression profile, which is the compressor that should be used to compress the data.
When in doubt, you can try `brute_force`, which attempts using every compressor and selects the best one.
The next argument is the integer width, in this example we'll use `4` because the pattern is `"yes\n"`, which is
4 bytes.
The last argument is the input file to compress.

```bash
$ ./examples/numeric_array standard 4 my-data
Compressed 1000000 bytes to 63 bytes
```

This particular data file is not very interesting, so feel free to bring your own numeric data & try it out.

## Building against an installed OpenZL

```cpp title="main.cpp"
#include <openzl/zl_compress.h>
#include <openzl/openzl.hpp>

int main(int argc, const char** argv)
{
    ZL_CCtx* cctx = ZL_CCtx_create();
    ZL_CCtx_free(cctx);
    openzl::DCtx dctx; // C++ is also supported
}
```

Once you install OpenZL, you can build arbitrary code and link against it.

```bash
export OPENZL_INSTALL_PREFIX=install/
g++ main.cpp -I"$OPENZL_INSTALL_PREFIX/include" -L"$OPENZL_INSTALL_PREFIX/lib" -L"$OPENZL_INSTALL_PREFIX/lib64" -lopenzl_cpp -lopenzl -lzstd -o main
```


## Using find_package to build OpenZL

You can also have CMake find & build OpenZL in your own project:

```cpp title="main.cpp"
#include <openzl/zl_compress.h>
#include <openzl/openzl.hpp>

int main(int argc, const char** argv)
{
    ZL_CCtx* cctx = ZL_CCtx_create();
    ZL_CCtx_free(cctx);
    openzl::DCtx dctx; // C++ is also supported
}
```

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.20.2 FATAL_ERROR)
project(tester)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
find_package(openzl)
if(NOT openzl_FOUND)
    FetchContent_Declare(
        openzl
        GIT_REPOSITORY https://github.com/facebook/openzl.git
        GIT_TAG dev
        OVERRIDE_FIND_PACKAGE
    )
    find_package(openzl REQUIRED)
    message(STATUS "Downloaded openzl")
else()
    message(STATUS "Found openzl")
endif()

add_executable(main main.cpp)
target_link_libraries(main openzl openzl_cpp)
```

## Next steps

* Read the [Introduction](introduction.md) and [Concepts](concepts.md) pages to learn more about OpenZL
* Check out the [numeric_array](examples/c/numeric-array.md) example
* Check out the [how to use OpenZL](using-openzl.md) for getting started with custom compression
* Check out our [API Reference](../api/c/compressor.md)
* See our [PyTorch Model Compressor](https://github.com/facebook/openzl/blob/dev/custom_parsers/pytorch_model_compressor.cpp) for a complete production compressor.
