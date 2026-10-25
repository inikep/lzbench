# slz: compressor (slz) and decompressor (uslz)
CODECS += SLZ
SLZ_OBJS  := lz+entropy/slz/src/slz.o lz+entropy/slz/src/uslz.o lz+entropy/slz/src/slz_common.o
SLZ_FLAGS := -std=gnu99
