# slz: compressor only, lzbench decompresses its output with zlib
CODECS += SLZ
SLZ_OBJS  := lz+entropy/slz/src/slz.o lz+entropy/slz/src/slz_common.o
SLZ_FLAGS := -std=gnu99
