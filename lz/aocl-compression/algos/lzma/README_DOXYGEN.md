LZMA - Introduction
===================
LZMA / LZMA2 are default and general compression methods of 7z format in the 7-Zip program.
LZMA provides a high compression ratio and fast decompression, so it is very suitable for 
embedded applications. For example, it can be used for ROM (firmware) compressing.

LZMA features:

Compression speed: 3 MB/s on 3 GHz dual-core CPU.
Decompression speed:
20-50 MB/s on modern 3 GHz CPU (Intel, AMD, ARM).
Small memory requirements for decompression: 8-32 KB + DictionarySize
Small code size for decompression: 2-8 KB (depending on speed optimizations)

The LZMA decoder uses only CPU integer instructions and can be implemented for any modern 32-bit CPU.