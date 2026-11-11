# lzmat (buggy: may crash)
CODECS += LZMAT
LZMAT_UPSTREAM := frozen LZMAT 1.01 from 2008 is the last release; github.com/nemequ/lzmat is a mirror
LZMAT_OBJS := lz/lzmat/lzmat_dec.o lz/lzmat/lzmat_enc.o
# -O2: segfaults when built with -O3 by GCC 4.9+
LZMAT_OPT  := O2
