# Libnxz Output Comparison Tests

This directory provides a series of tests with different file formats to
validate the output from Libnxz.

These tests include:

- Compress: compress files with Libnxz and decompress them with the system's
  software and compare the final result.
- Decompress: compress files with the system's software and decompress with
  Libnxz and compare the final result.
- Compress and Decompress: use Libnxz to compress and decompress a file and
  compare the final result.

Some of the files used in the tests are too large, taking a lot of disk space
and time to execute.  In order to run the tests with those files too, set
`LARGEFILES`, e.g.

```
make LARGEFILES=1 -j$(nproc)
```

## Requirements

In order to work, oct requires 2 files:

1. Copy libnxz.so.0 to oct/.
2. Provide shared linked minigzipsh in oct/.
