# Introduction

XPACK is an experimental compression format.  It is intended to have better
performance than DEFLATE as implemented in the zlib library and also produce a
notably better compression ratio on most inputs.  The format is not yet stable.

XPACK has been inspired by the DEFLATE, LZX, and Zstandard formats, among
others.  Originally envisioned as a DEFLATE replacement, it won't necessarily
see a lot of additional development since other solutions such as Zstandard seem
to have gotten much closer to that goal first (great job to those involved!).
But I am releasing the code anyway for anyone who may find it useful.

# Format overview

Like many other common compression formats, XPACK is based on the LZ77 method
(decomposition into literals and length/offset copy commands) with a number of
tricks on top.  Features include:

* Increased sliding window, or "dictionary", size (like LZX and Zstd)
* Entropy encoding with finite state entropy (FSE) codes, also known as
  table-based asymmetric numeral systems (tANS) (like Zstd)
* Minimum match length of 2 (like LZX)
* Lowest three bits of match offsets can be entropy-encoded (like LZX)
* Aligned and verbatim blocks (like LZX)
* Recent match offsets queue with three entries (like LZX)
* Literals packed separately from matches, and with two FSE streams (like older
  Zstd versions)
* Literal runs (like Zstd)
* Concise FSE header (state count list) representation
* Decoder reads in forwards direction, encoder writes in backwards direction
* Optional preprocessing step for x86 machine code (like LZX)

# Implementation overview

libxpack is a library containing an optimized, portable implementation of an
XPACK compressor and decompressor.  Features currently include:

* Whole-buffer compression and decompression only
* Multiple compression levels
* Fast hash chains-based matchfinder
* Greedy and lazy parsers
* Decompressor automatically uses Intel BMI2 instructions when supported

In addition, the following command-line programs using libxpack are provided:

* xpack (or xunpack), a program which behaves like a standard UNIX command-line
  compressor such as gzip (or gunzip).  The command-line interface should be
  compatible enough that xpack can be used as a drop-in gzip replacement in many
  cases --- though the on-disk format is incompatible, of course.
* benchmark, a program for benchmarking in-memory compression and decompression

Note that currently, all the programs internally use "chunks", as the library
does not yet support streaming.  This will worsen the compression ratio
slightly, compared to what is possible.

All files may be modified and/or redistributed under the terms of the MIT
license.  There is NO WARRANTY, to the extent permitted by law.  See the COPYING
file for details.

# Building

## For UNIX

Just run `make`.  You need GNU Make and either GCC or Clang.  GCC is recommended
because it builds slightly faster binaries.  There is no `make install` yet;
just copy the file(s) to where you want.

By default, all targets are built, including the library and programs.  `make
help` shows the available targets.  There are also several options which can be
set on the `make` command line.  See the Makefile for details.

## For Windows

MinGW (GCC) is the recommended compiler to use when building binaries for
Windows.  MinGW can be used on either Windows or Linux.  Use a command like:

    $ make CC=x86_64-w64-mingw32-gcc

Windows binaries prebuilt with MinGW may also be downloaded from
https://github.com/ebiggers/xpack/releases.

Alternatively, a separate Makefile, `Makefile.msc`, is provided for the tools
that come with Visual Studio, for those who strongly prefer that toolchain.

As usual, 64-bit binaries are faster than 32-bit binaries and should be
preferred whenever possible.
