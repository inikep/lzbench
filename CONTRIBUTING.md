# Contributing to lzbench

Thank you for considering contributing to lzbench! Please follow the guidelines below to ensure a smooth and efficient contribution process.

## 1. Passing Tests
All contributions should pass (green tick) all Azure Pipeline tests for [lzbench pipeline](https://dev.azure.com/inikep/lzbench/_build?definitionId=19&_a=summary), which will be triggered automatically.

## 2. Updating Existing Codecs
When updating an existing codec, please follow these steps:

- Update the codec files (e.g., `lz/zlib-ng/*`).
- Update `Makefile` if there are new source files that need to be built.
- Update the codec version in `bench/lzbench.h` and `README.md`.
- Add a new entry in `CHANGELOG`
- Refer to example commit: [Update zlib-ng to 2.2.5](https://github.com/inikep/lzbench/commit/5eed568).

## 3. Adding New Codecs
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
