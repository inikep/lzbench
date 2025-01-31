# Contributing to lzbench

Thank you for considering contributing to lzbench! Please follow the guidelines below to ensure a smooth and efficient contribution process.

## 1. Passing Tests
All contributions should pass (green tick) all Azure Pipeline tests for [lzbench pipeline](https://dev.azure.com/inikep/lzbench/_build?definitionId=12&_a=summary), which will be triggered automatically.

## 2. Updating Existing Codecs
When updating an existing codec, please follow these steps:

- Update the codec files (e.g., `brotli/*`).
- Update `Makefile` if there are new source files that need to be built.
- Update the codec version in `_lzbench/lzbench.h` and `README.md`.
- Add a new entry in `CHANGELOG`
- Refer to example commit: [update brotli to 1.1.0](https://github.com/inikep/lzbench/commit/f46161a8).

## 3. Adding New Codecs
When adding a new codec, please follow these steps:

- Create a new directory with the codec files (e.g., `xxxx`).
- Add a new codec to `README.md` with a proper link to the upstream repository.
- Add a new entry in `CHANGELOG`
- Add the new codec to `_lzbench/lzbench.h` and increase `LZBENCH_COMPRESSOR_COUNT`.
- Add declarations of compression and decompression functions in `_lzbench/compressors.h`, e.g.:

```
#ifndef BENCH_REMOVE_XXXX
int64_t lzbench_xxxx_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, size_t level, size_t, char*);
int64_t lzbench_xxxx_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, size_t, size_t, char*);
#else
#define lzbench_xxxx_compress NULL
#define lzbench_xxxx_decompress NULL
#endif // BENCH_REMOVE_XXXX
```

- Add definitions of compression and decompression functions in `_lzbench/compressors.cpp`, e.g.:

```
#ifndef BENCH_REMOVE_XXXX
#include "XXXX/YYYY.h"
int64_t lzbench_xxxx_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, size_t level, size_t, char*) { }
int64_t lzbench_xxxx_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, size_t, size_t, char*) { }
#endif
```

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

- Refer to example commit: [Add ppmd8 based on 7-zip 24.09](https://github.com/inikep/lzbench/commit/10d1e6f0).
