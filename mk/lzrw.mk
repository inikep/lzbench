# lzrw (buggy: may crash)
CODECS += LZRW
LZRW_OBJS := $(addprefix lz/lzrw/, lzrw1-a.o lzrw1.o lzrw2.o lzrw3.o lzrw3-a.o)
# -O2: segfaults when built with -O3 by GCC 4.9+
LZRW_OPT  := O2
