# NX GZIP compression library

libnxz implements a zlib-compatible API for Linux userspace programs that
exploit the NX GZIP accelerator available on POWER9 and newer processors.

[![Packaging status](https://repology.org/badge/vertical-allrepos/libnxz.svg)](https://repology.org/project/libnxz/versions)

## How to Use
- If want to use nxzlib to substitute zlib, following the steps as below:
1. Build libnxz.so
```
./configure
make
```
2. Use libnxz.so to substitute libz.so (replace 0.0 with the version being used)
```
cp lib/libnxz.so.0.0 /usr/lib/
mv /usr/lib/libz.so /usr/lib/libz.so.bak
ln -s /usr/lib/libnxz.so.0.0 /usr/lib/libz.so
```
- If don't want to override the libz.so, use LD_PRELOAD to run. Something like:
```
LD_PRELOAD=./libnxz.so /home/your_program
```

## How to Run Test
```
cd test
./configure
make
make check
```

## How to Select NXs

By default, the NX-GZIP device with the nearest process to cpu affinity is
selected. Consider using numactl -N 0 (or 8) to force your process attach to a
particular device

## How to enable log and trace for debug
The default log will be /tmp/nx.log. Use `export NX_GZIP_LOGFILE=your.log`
to specify a different log. By default, only errors will be recorded in log.

Use `export NX_GZIP_VERBOSE=2` to record the more information.

Use `export NX_GZIP_TRACE=1` to enable logic trace.

Use `export NX_GZIP_TRACE=8` to enable statistics trace.

## Supported Functions List

All the zlib supported functions are listed and described at libnxz.h.

If want to use nxzlib standalone, add a prefix 'nx_' before the function.
For example, use `nx_compress` instead of `compress`.

## Code organization

- libnxz.h - Provides the  zlib-compatible API.
- doc/ - Provides documentation about the library.
- inc_nx/ - Internal header files.
- lib/ - Implements the library functions.
- [oct/](oct/README.md) - Provide output comparison tests validating that data
can be compressed and decompressed with other libraries maintaining their
integrity.
- samples/ - Provide example application that use the libnxz API.
- [selftest/](selftest/README.md) - Small set of tests for the NX GZIP
accelerator.  These tests are reused in Linux.
- [test/](test/README.md) - Unit tests for libnxz.
