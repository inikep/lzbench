# lzmat (buggy: may crash)
CODECS += LZMAT
LZMAT_OBJS := lz/lzmat/lzmat_dec.o lz/lzmat/lzmat_enc.o
# -O2: segfaults when built with -O3 by GCC 4.9+
LZMAT_OPT  := O2
