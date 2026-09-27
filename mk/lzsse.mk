# lzsse: needs SSE4.1 and a 64-bit CPU
CODECS += LZSSE

ifeq ($(BUILD_ARCH),32-bit)
    DONT_BUILD_LZSSE ?= 1
endif
ifneq ($(shell echo|$(CC) -dM -E - -march=native 2>/dev/null|egrep -c '__(SSE4_1|x86_64)__'), 2)
    DONT_BUILD_LZSSE ?= 1
endif

LZSSE_OBJS  := lz/lzsse/lzsse2/lzsse2.o lz/lzsse/lzsse4/lzsse4.o lz/lzsse/lzsse8/lzsse8.o
LZSSE_FLAGS := -std=c++0x -msse4.1
