# Contributing to lzbench

Thank you for considering contributing to lzbench! Please follow the guidelines below to ensure a smooth and efficient contribution process.

## 1. Passing Tests
All contributions should pass (green tick) all Azure Pipeline tests for [lzbench pipeline](https://dev.azure.com/inikep/lzbench/_build?definitionId=19&_a=summary), which will be triggered automatically.

## 2. No Git Submodules
Codec sources are copied into this repository rather than pulled in as git submodules. This is a
deliberate choice — please do not propose converting a codec to a submodule. The main reasons:

- **Codecs need local patches.** lzbench builds on ~30 CI configurations (GCC 10-15, Clang 12-22,
  MSVC, macOS, 32/64-bit ARM, big-endian PowerPC, RISC-V, CUDA). Upstreams rarely test any of that,
  so many codecs need small portability fixes. A submodule cannot be patched without forking it.
- **Many upstreams are dead or were never in git.** Some codecs are decades old or were published
  only as forum attachments, so there is no repository to track.
- **Reproducibility.** A benchmark number is only meaningful if you know exactly which code produced
  it. One lzbench commit pins every codec.
- **Clone reliability.** A single renamed or deleted upstream would break `git clone --recursive`
  for everyone.
- **Build integration.** Every codec is built from lzbench's single `Makefile` with per-codec flags,
  so a submodule's own build system would be bypassed anyway.

The trade-off is that local patches must be re-applied when a codec is updated. If you carry one
forward, say so in the commit message so it is not silently lost on the next update.

## 3. Updating Existing Codecs
When updating an existing codec, please follow these steps:

- Update the codec files (e.g., `lz/zlib-ng/*`).
- Update `Makefile` if there are new source files that need to be built.
- Update the codec version in `bench/lzbench.h` and `README.md`.
- Add a new entry in `CHANGELOG`
- Refer to example commit: [Update zlib-ng to 2.2.5](https://github.com/inikep/lzbench/commit/5eed568).

## 4. Adding New Codecs
Before proposing a new codec for inclusion, please make sure it is a good fit for lzbench:

- The codec must be open source, with source code that can be included in or built with lzbench.
- The codec should have a license that allows redistribution and benchmarking as part of this project.
- The codec should preferably be written in C or C++, or provide a C-compatible API that can be called from the existing benchmark harness.
- The codec should support in-memory compression and decompression APIs so lzbench can verify that decompressed output matches the original input.
- The codec should be stable and should not crash frequently on valid inputs.
- The codec should be significant in at least one benchmark dimension, such as compression speed, decompression speed, compression ratio, memory usage, or another useful trade-off. A codec that is worse than existing codecs in every measurement is unlikely to be a good candidate for inclusion.

When adding a new codec, please follow these steps:

- Create a new subdirectory with the codec files (e.g., `xxxx`) in `lz`, `bwt`, or `misc` directory.
- Add a new codec to `README.md` with a proper link to the upstream repository.
- Add a new entry in `CHANGELOG`
- Add declarations of compression and decompression functions in `bench/codecs.h`, e.g.:

```
#ifndef BENCH_REMOVE_XXXX
int64_t lzbench_xxxx_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options);
int64_t lzbench_xxxx_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options);
#else
#define lzbench_xxxx_compress NULL
#define lzbench_xxxx_decompress NULL
#endif // BENCH_REMOVE_XXXX
```

- Add definitions of compression and decompression functions in `bench/lz_codecs.cpp`, `bench/symmetric_codecs.cpp` (BWT, PPM-based), or `bench/misc_codecs.cpp`, e.g.:

```
#ifndef BENCH_REMOVE_XXXX
#include "XXXX/YYYY.h"
int64_t lzbench_xxxx_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options) { }
int64_t lzbench_xxxx_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options) { }
#endif
```
- If a codec supports multi-threading, it should use a number of threads provided with `codec_options->threads`.

- Update `Makefile`:

```
ifeq "$(DONT_BUILD_XXXX)" "1"
    DEFINES += -DBENCH_REMOVE_XXXX
else
    XXXX_FILES = XXXX/YYYY.o XXXX/YYYY_Dec.o XXXX/YYYY_Enc.o
endif
```

And ensure the new codec in `Makefile` is linked in:

```
lzbench: $(BZIP2_FILES) $(KANZI_FILES) ... $(XXXX_FILES)
```

- Refer to example commit: [Add zpaq 7.15](https://github.com/inikep/lzbench/commit/20f553b).
