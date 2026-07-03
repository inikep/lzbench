# misa77

A fast-decode LZ compressor.

- Upstream: https://github.com/welcome-to-the-sunny-side/misa77
- License:  MIT

Registered lzbench codecs:

| name              | description                                              |
| ----------------- | -------------------------------------------------------- |
| `misa77`          | the library codec; levels 0 (fastest decompression) / 1 (best ratio, the library default) |
| `misa77_adaptive` | decoder-friendly adaptive parse for homogeneous data; levels 0 (loose) / 1 (tight) |

Both emit bitstreams conforming to the same format and share one decompressor.
