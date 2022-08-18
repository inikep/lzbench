# Libnxz tests

This directory contains test cases.
To be compliant with zlib, usually we use libnxz to deflate,
the compressed data will use both zlib and libnxz to inflate for verification.
We also use zlib to deflate, the compressed data will use libnxz to inflate
for verification.
