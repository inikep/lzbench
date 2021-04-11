# nvcomp 1.2.3 (2021-04-07)

- Fix bug in LZ4 compression kernel for the Pascal architecture.

# nvcomp 1.2.2 (2021-02-08)

- Fix linking errors in Clang++.
- Fix error being incorrectly returned by Cascaded compression when output memory was initialized to
all `-1`'s.
- Fix C++17 style static assert.
- Fix prematurely freeing memory in Cascaded compression.
- Fix input format and usage messaging for benchmarks.

# nvcomp 1.2.1 (2020-12-21)

- Fix compile error and unit tests for cascaded selector.

# nvcomp 1.2.0 (2020-12-19)

- Add the Cascaded Selector and Cascaded Auto set of interfaces for
automatically configuring cascaded compression.
- Generally improve error handling and messaging.
- Update CMake configuration to support CCache.

# nvcomp 1.1.1 (2020-12-02)

- Add all-gather benchmark.
- Add sm80 target if CUDA version is 11 or greater.

# nvcomp 1.1.0 (2020-10-05)

- Add batch C interface for LZ4, allowing compressing/decompressing multiple
inputs at once.
- Significantly improve performance of LZ4 compression.

# nvcomp 1.0.2 (2020-08-12)

- Fix metadata freeing for LZ4, to avoid possible mismatch of `new[]` and
`delete`.


# nvcomp 1.0.1 (2020-08-07)

- Fixed naming of nvcompLZ4CompressX functions in `include/lz4.h`, to have the
`nvcomp` prefix.
- Changed CascadedMetadata::Header struct initialization to work around
internal compiler error.


# nvcomp 1.0.0 (2020-07-31)

- Initial public release.
