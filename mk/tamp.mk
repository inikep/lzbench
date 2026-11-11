# tamp
CODECS += TAMP
TAMP_UPSTREAM := git-tags https://github.com/BrianPugh/tamp
TAMP_OBJS  := lz/tamp/common.o lz/tamp/compressor.o lz/tamp/decompressor.o
TAMP_FLAGS := -std=gnu99
