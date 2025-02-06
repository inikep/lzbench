Building
------------
These are generic building and installation instructions.


Requirements
------------
To compile, you need a C and C++ compiler that is GNUC-compatible, such as GCC, LLVM/Clang, or ICC.
It is recommended to use GCC 7.1+ or Clang 6.0+. You can find GCC at http://gcc.gnu.org.


Testing
------------
lzbench undergoes automated testing using Azure Pipelines with the following compilers:
- Ubuntu: gcc (versions 7.5 to 14.2) and clang (versions 6.0 to 19), gcc 14.2 (32-bit)
- MacOS: Apple LLVM version 15.0.0
- Windows: mingw32-gcc 14.2.0 (32-bit) and mingw64-gcc 14.2.0 (64-bit)
- Cross-compilation: gcc for ARM (32-bit and 64-bit) and PowerPC (32-bit and 64-bit)


Get the source code
-------------------
### From git repository
Clone git repository with:
```
git clone https://github.com/inikep/lzbench.git
cd lzbench
```

### Download an archive
Another option is to download zip or tar ball from repository or release page at https://github.com/inikep/lzbench/releases/.

Unpack the archive:

	unzip lzbench-master.zip
or

	tar -xzf lzbench-[version].tar.gz

This creates the directory `./lzbench-[version]` containing the source
from the main archive.


Compilation
-----------
Change to lzbench directory and run make.

	make -j$(nproc)


To order static or dynamic linking set `BUILD_STATIC` to 1, or 0 respectively:

	make BUILD_STATIC=1

For 32-bit compilation:

	make BUILD_ARCH=32-bit

For non-default compiler:

	make CC=gcc-14 CXX=g++-14

For an optimized but non-portable build, use:

	make MOREFLAGS="-march=native"
or

	make USER_CFLAGS="-march=native" USER_CXXFLAGS="-march=native"


USER_CFLAGS, USER_CXXFLAGS and USER_LDFLAGS variables allow user to add
or replace existing values of corresponding variables without completely
replacing them.


With certain, old compilers, systems or combination some compressor
may cause issues. Most of it should be chosen automatically but in
case of problems, particular compressors can be excluded from build setting
`DONT_BUILD_XXX` to 1, where `XXX` is target compressor, for example:

	make DONT_BUILD_GLZA=1


CUDA support
-------------------------

If CUDA is available, lzbench supports additional compressors:
- [cudaMemcpy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) - similar to the reference `memcpy` benchmark, using GPU memory
- [nvcomp](https://github.com/NVIDIA/nvcomp) - LZ4 GPU-only compressor

The directory where the CUDA compiler and libraries are available can be passed to `make` via the `CUDA_BASE` variable, *e.g.*:

	make CUDA_BASE=/usr/local/cuda


Compilation for various CPU architectures
-----------------------------------------
- `x86_64` (`amd64`) is our main configuration as is well tested
- `x86` (32-bit) works fine with all codecs except LZSSE in our tests
- `ppc/ppc64` (PowerPC 32-bit and 64-bit) works fine with additional `DONT_BUILD_YAPPY=1 DONT_BUILD_DENSITY=1 DONT_BUILD_ZLING=1` parameters
- `riscv32/riscv64` - we have reports it works fine with `DONT_BUILD_TORNADO=1`
- `mips/mips64` - waiting for reports
- `loongarch64` - we have reports it works fine


Known issues
------------
With all issues refer to `Makefile`, as they may already be suppressed.

### All operating systems
- LZSSE requires 64-bit Intel CPU with SSE4.1 for compilation and execution (a support for `__SSE4_1__` is auto-detected)

### Windows
- `mingw64-gcc` works correctly with `DONT_BUILD_LZHAM=1`
- `mingw32-gcc` functions properly with `DONT_BUILD_GLZA=1 DONT_BUILD_LZHAM=1`

### MacOS
- `Apple clang` works fine with `DONT_BUILD_LZHAM=1 DONT_BUILD_CSC=1`, which should be added automatically


------------------------
Copyright (C) 2025 tansy

This file is free documentation: you have unlimited permission to copy,
distribute, and modify it.
