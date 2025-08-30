Density
========
Superfast compression library

**Density** is a free, **pure rust** (formerly C99) open-source compression library.

It is focused on high-speed compression, at the best ratio possible. **Density**'s algorithms are currently at the **pareto frontier** of compression speed vs ratio (cf. [here](https://quixdb.github.io/squash-benchmark/?dataset=dickens&machine=s-desktop) for an independent benchmark).

**Density** features a simple API to enable quick integration in any project.

[![MIT licensed](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE-MIT)
[![Apache-2.0 licensed](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE-APACHE)
[![Crates.io](https://img.shields.io/crates/v/density-rs.svg)](https://crates.io/crates/density-rs)
[![Build Status](https://github.com/g1mv/density/actions/workflows/ci.yml/badge.svg)](https://github.com/g1mv/density/actions)

Why is it so fast ?
-------------------

One of the biggest assets of **density** is its work unit: unlike most compression algorithms, it is **not a byte**
but **a group of 4 bytes**.

When other libraries consume one byte of data to apply their algorithmic processing, **density** consumes 4 bytes.

That's why **density**'s algorithms have been designed from scratch. They cater for 4-byte work units and still provide interesting compression ratios.

**Speed pedigree traits**

* 4-byte work units
* heavy use of registers as opposed to memory
* avoidance of or use of minimal branching when possible
* use of low memory data structures to favor processor cache Lx storage
* library wide inlining

A "blowup protection" is provided, dramatically increasing the processing speed of incompressible input data. The aim
is to never exceed original data size, even for incompressible inputs.

Benchmarks
----------

**Quick benchmark**

**Density** features an integrated single-core **in-memory benchmark**.

Just use ```cargo bench``` to assess the performance of the library on your platform, as well as lz4's and snappy's,
using the
*dickens* file from the renowned [silesia corpus](https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia).

It is also possible to run the benchmark with your own files, using the following command: ```FILE=... cargo bench``` (replace ... with a file path).
Popular compression test files include
the [silesia corpus files](https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia)
or [enwik8](https://mattmahoney.net/dc/textdata.html).

[>> Link to a sample run log](benchmark.log)

**Third-party benchmarks**

Other benchmarks featuring **density** (non-exhaustive list) :

* [**squash**](https://github.com/quixdb/squash) is an abstraction layer for compression algorithms, and has an
  extremely exhaustive set of benchmark results, including
  density's, [available here](https://quixdb.github.io/squash-benchmark/?dataset=dickens&machine=s-desktop).

* [**lzbench**](https://github.com/inikep/lzbench) is an in-memory benchmark of open-source LZ77/LZSS/LZMA compressors.

Build
-----
**Density** can be built on rust-compatible platforms. First use [rustup](https://rustup.rs) to install
rust.

a) get the source code:

```shell
    git clone https://github.com/g1mv/density.git
    cd density
```

b) build and test:

```shell
    cargo build
    cargo test
```

c) run benchmarks with or without your own files:

```shell
    RUSTFLAGS="-C target-cpu=native" cargo bench
    FILE=... cargo bench
```

About the algorithms
--------------------

**Density**'s algorithms are general purpose, very fast algorithms. They empirically exhibit strong performance (pareto frontier speed/ratio) on
voluminous datasets and text-based datasets, and are less performant on very small datasets. Their 4-byte work unit
approach makes them "slower
learners" than Lempel-Ziv-based algorithms, albeit being much faster theoretically.

| Algorithm | Speed rank | Ratio rank | Dictionary unit(s) | Prediction unit(s) | Sig. size (bytes) |
|-----------|------------|------------|--------------------|--------------------|-------------------|
| chameleon | 1st        | 3rd        | 1                  | 0                  | 8                 |
| cheetah   | 2nd        | 2nd        | 2                  | 1                  | 8                 |
| lion      | 3rd        | 1st        | 2                  | 5                  | 6                 |

**Chameleon** is a dictionary lookup based compression algorithm. It is designed for absolute speed (GB/s order) both for compression
and decompression.

**Cheetah**, developed with inputs from [Piotr Tarsa](https://github.com/tarsa), is derived from chameleon and
uses swapped dual dictionary lookups with a single prediction unit.

**Lion** is derived from chameleon/cheetah. It uses different data structures, dual dictionary lookups and 5 prediction units, giving it a compression ratio advantage on moderate to highly compressible data.

Quick start
--------------------------------------------
Using **density** in your rust code is a straightforward process.

Include the required dependency in the *Cargo.toml* file:

```toml
[dependencies]
density-rs = "0.16.6"
```

Use the API:

```rust
use std::fs::read;
use density_rs::algorithms::chameleon::chameleon::Chameleon;

fn main() {
    let file_mem = read("file_path").unwrap();

    let mut encoded_mem = vec![0_u8; Chameleon::safe_encode_buffer_size(file_mem.len())];
    let encoded_size = Chameleon.encode(&file_mem, &mut encoded_mem).unwrap();
      
    let mut decoded_mem = vec![0_u8; file_mem.len()];
    let decoded_size = Chameleon::decode(&encoded_mem[0..encoded_size], &mut decoded_mem).unwrap();
}
```

And that's it! We've done a compression/decompression round trip with a few lines!