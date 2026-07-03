LZ4 - Introduction
==================

LZ4 is a lossless compression algorithm
providing a compression speed > 500 MB/s per core,
scalable with multi-cores CPU.
It features an extremely fast decoder
with speed in multiple GB/s per core,
reaching RAM speed limits on multi-core systems.

Speed can be tuned dynamically, selecting an "acceleration" factor
which trades compression ratio for faster speed.
On the other end, a high compression derivative, LZ4_HC, is also provided,
trading CPU time for improved compression ratio.
All the versions feature the same decompression speed.

LZ4 is also compatible with [dictionary compression](https://github.com/facebook/zstd#the-case-for-small-data-compression),
both at [API](https://github.com/lz4/lz4/blob/v1.8.3/lib/lz4frame.h#L481) and [CLI](https://github.com/lz4/lz4/blob/v1.8.3/programs/lz4.1.md#operation-modifiers) levels.
It can ingest any input file as dictionary, though only the final 64 KB is used.
This capability can be combined with the [Zstandard Dictionary Builder](https://github.com/facebook/zstd/blob/v1.3.5/programs/zstd.1.md#dictionary-builder)
to improve the compression performance on small files.


LZ4 library is provided as an open-source software using BSD 2-Clause license.



Documentation
-------------------------

The raw LZ4 block compression format is detailed within [lz4_Block_format].

Arbitrarily, for streaming requirements, long files or data streams are compressed using multiple blocks. These blocks are organized into a frame,
defined into [lz4_Frame_format].
Interoperable versions of LZ4 must also respect the frame format.

[lz4_Block_format]: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
[lz4_Frame_format]: https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md


Other source versions
-------------------------

Beyond the C reference source,
many contributors have created versions of LZ4 in multiple languages
(Java, C#, Python, Perl, Ruby, and so on).
A list of known source ports is maintained on the [LZ4 Homepage].

[LZ4 Homepage]: http://www.lz4.org
