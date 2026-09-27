# yappy (buggy: may crash)
CODECS += YAPPY

# yappy doesn't work on big-endian PowerPC
ifeq ($(UNAME_P),powerpc)
    DONT_BUILD_YAPPY ?= 1
endif

YAPPY_OBJS := lz/yappy/yappy.o
# -O2: segfaults when built with -O3 by GCC 4.9+
YAPPY_OPT  := O2
