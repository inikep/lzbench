# lbzip2 in lzbench

Vendored from https://github.com/caius72/lbzip2 at v2.6.5 (commit 26461f6),
which is a maintained fork of https://github.com/kjn/lbzip2. GPL-3.0-or-later;
see COPYING.

`crctab.c`, `decode.c`, `divbwt.c`, `encode.c`, `parse.c` and the headers are
upstream's `src/` files, unmodified. They are the low-level codec; the rest of
upstream (`main.c`, `process.c`, `compress.c`, `expand.c`, ...) is the
multi-threaded command-line tool and is not used here.

`lbzip2_lzbench.c` is the lzbench-side wrapper: it drives that codec
sequentially, block by block, the way the tool does around its scheduler, and
supplies the one function (`xmalloc`) that `decode.c` expects from the tool.

`arpa/inet.h` is a shim for MinGW, which lacks that header; three of the
vendored sources include it for `ntohl`/`htonl` alone. It is reachable only
through `-Ibwt/lbzip2`.

To update: copy those files from a newer lbzip2 `src/`, and check that
`collect`/`encode`/`transmit` and `parse`/`retrieve`/`decode`/`emit` still
have the signatures the wrapper uses.
