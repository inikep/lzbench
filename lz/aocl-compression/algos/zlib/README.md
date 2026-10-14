ZLIB - Introduction
===================

zlib 1.3.1 is a general-purpose data compression library.  All the code is
thread safe.  The data format used by the zlib library is described by
Request for Comments (RFCs) 1950 to 1952 in the files
http://tools.ietf.org/html/rfc1950 (zlib format), rfc1951 (deflate format), and
rfc1952 (gzip format).

All functions of the compression library are documented in the file zlib.h (volunteer to write man pages welcome, contact zlib@gzip.org).


Notes:
======

- For 64-bit Irix, deflate.c must be compiled without any optimization. With
  -O, one libpng test fails. The test works in 32-bit mode (with the -n32
  compiler flag). The compiler bug has been reported to SGI.

- zlib doesn't work with GCC 2.6.3 on a DEC 3000/300LX under OSF/1 2.1. It works
  when compiled with cc.

- On Digital Unix 4.0D (formerly OSF/1) on AlphaServer, the CC option -std1 is
  necessary to get gzprintf working correctly. This is done by configure.

- zlib doesn't work on HP-UX 9.05 with some versions of /bin/cc. It works with
  the other compilers. Use "make test" to check your compiler.

- gzdopen is not supported on RISCOS or BEOS.

- For PalmOs, refer to http://palmzlib.sourceforge.net/
