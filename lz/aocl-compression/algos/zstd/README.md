<p align="center"><img src="https://raw.githubusercontent.com/facebook/zstd/dev/doc/images/zstd_logo86.png" alt="Zstandard"></p>

__Zstandard__, or `zstd` as short version, is a fast lossless compression algorithm,
targeting real-time compression scenarios at zlib-level and better compression ratios.
It's backed by a very fast entropy stage, provided by [Huff0 and FSE library](https://github.com/Cyan4973/FiniteStateEntropy).

Zstandard's format is stable and documented in [RFC8878](https://datatracker.ietf.org/doc/html/rfc8878). Multiple independent implementations are already available.
This repository represents the reference implementation, provided as an open-source dual [BSD](LICENSE) OR [GPLv2](COPYING) licensed **C** library,
and a command line utility producing and decoding `.zst`, `.gz`, `.xz` and `.lz4` files.
Should your project require another programming language,
a list of known ports and bindings is provided on [Zstandard homepage](https://facebook.github.io/zstd/#other-languages).

**Development branch status:**

[![Build Status][travisDevBadge]][travisLink]
[![Build status][CircleDevBadge]][CircleLink]
[![Build status][CirrusDevBadge]][CirrusLink]
[![Fuzzing Status][OSSFuzzBadge]][OSSFuzzLink]

[travisDevBadge]: https://api.travis-ci.com/facebook/zstd.svg?branch=dev "Continuous Integration test suite"
[travisLink]: https://travis-ci.com/facebook/zstd
[CircleDevBadge]: https://circleci.com/gh/facebook/zstd/tree/dev.svg?style=shield "Short test suite"
[CircleLink]: https://circleci.com/gh/facebook/zstd
[CirrusDevBadge]: https://api.cirrus-ci.com/github/facebook/zstd.svg?branch=dev
[CirrusLink]: https://cirrus-ci.com/github/facebook/zstd
[OSSFuzzBadge]: https://oss-fuzz-build-logs.storage.googleapis.com/badges/zstd.svg
[OSSFuzzLink]: https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:zstd

## Benchmarks

For reference, several fast compression algorithms were tested and compared
on a desktop featuring a Core i7-9700K CPU @ 4.9GHz
and running Ubuntu 20.04 (`Linux ubu20 5.15.0-101-generic`),
using [lzbench], an open-source in-memory benchmark by @inikep
compiled with [gcc] 9.4.0,
on the [Silesia compression corpus].

[lzbench]: https://github.com/inikep/lzbench
[Silesia compression corpus]: https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia
[gcc]: https://gcc.gnu.org/

| Compressor name         | Ratio | Compression| Decompress.|
| ---------------         | ------| -----------| ---------- |
| **zstd 1.5.6 -1**       | 2.887 |   510 MB/s |  1580 MB/s |
| [zlib] 1.2.11 -1        | 2.743 |    95 MB/s |   400 MB/s |
| brotli 1.0.9 -0         | 2.702 |   395 MB/s |   430 MB/s |
| **zstd 1.5.6 --fast=1** | 2.437 |   545 MB/s |  1890 MB/s |
| **zstd 1.5.6 --fast=3** | 2.239 |   650 MB/s |  2000 MB/s |
| quicklz 1.5.0 -1        | 2.238 |   525 MB/s |   750 MB/s |
| lzo1x 2.10 -1           | 2.106 |   650 MB/s |   825 MB/s |
| [lz4] 1.9.4             | 2.101 |   700 MB/s |  4000 MB/s |
| lzf 3.6 -1              | 2.077 |   420 MB/s |   830 MB/s |
| snappy 1.1.9            | 2.073 |   530 MB/s |  1660 MB/s |

[zlib]: https://www.zlib.net/
[lz4]: https://lz4.github.io/lz4/

The negative compression levels, specified with `--fast=#`,
offer faster compression and decompression speed
at the cost of compression ratio.

Zstd can also offer stronger compression ratios at the cost of compression speed.
Speed vs Compression trade-off is configurable by small increments.
Decompression speed is preserved and remains roughly the same at all settings,
a property shared by most LZ compression algorithms, such as [zlib] or lzma.

The following tests were run
on a server running Linux Debian (`Linux version 4.14.0-3-amd64`)
with a Core i7-6700K CPU @ 4.0GHz,
using [lzbench], an open-source in-memory benchmark by @inikep
compiled with [gcc] 7.3.0,
on the [Silesia compression corpus].

Compression Speed vs Ratio | Decompression Speed
---------------------------|--------------------
![Compression Speed vs Ratio](doc/images/CSpeed2.png "Compression Speed vs Ratio") | ![Decompression Speed](doc/images/DSpeed3.png "Decompression Speed")

A few other algorithms can produce higher compression ratios at slower speeds, falling outside of the graph.
For a larger picture including slow modes, [click on this link](doc/images/DCspeed5.png).


## The case for Small Data compression

Previous charts provide results applicable to typical file and stream scenarios (several MB). Small data comes with different perspectives.

The smaller the amount of data to compress, the more difficult it is to compress. This problem is common to all compression algorithms, and reason is, compression algorithms learn from past data how to compress future data. But at the beginning of a new data set, there is no "past" to build upon.

To solve this situation, Zstd offers a __training mode__, which can be used to tune the algorithm for a selected type of data.
Training Zstandard is achieved by providing it with a few samples (one file per sample). The result of this training is stored in a file called "dictionary", which must be loaded before compression and decompression.
Using this dictionary, the compression ratio achievable on small data improves dramatically.

The following example uses the `github-users` [sample set](https://github.com/facebook/zstd/releases/tag/v1.1.3), created from [github public API](https://developer.github.com/v3/users/#get-all-users).
It consists of roughly 10K records weighing about 1KB each.

Compression Ratio | Compression Speed | Decompression Speed
------------------|-------------------|--------------------
![Compression Ratio](doc/images/dict-cr.png "Compression Ratio") | ![Compression Speed](doc/images/dict-cs.png "Compression Speed") | ![Decompression Speed](doc/images/dict-ds.png "Decompression Speed")


These compression gains are achieved while simultaneously providing _faster_ compression and decompression speeds.

Training works if there is some correlation in a family of small data samples. The more data-specific a dictionary is, the more efficient it is (there is no _universal dictionary_).
Hence, deploying one dictionary per type of data will provide the greatest benefits.
Dictionary gains are mostly effective in the first few KB. Then, the compression algorithm will gradually use previously decoded content to better compress the rest of the file.

### Dictionary compression How To:

1. Create the dictionary

   `zstd --train FullPathToTrainingSet/* -o dictionaryName`

2. Compress with dictionary

   `zstd -D dictionaryName FILE`

3. Decompress with dictionary

   `zstd -D dictionaryName --decompress FILE.zst`



## Status

Zstandard is currently deployed within Facebook and many other large cloud infrastructures.
It is run continuously to compress large amounts of data in multiple formats and use cases.
Zstandard is considered safe for production environments.

## License

Zstandard is dual-licensed under [BSD](LICENSE) OR [GPLv2](COPYING).

## Contributing

The `dev` branch is the one where all contributions are merged before reaching `release`.
If you plan to propose a patch, please commit into the `dev` branch, or its own feature branch.
Direct commit to `release` are not permitted.
For more information, please read [CONTRIBUTING](CONTRIBUTING.md).
