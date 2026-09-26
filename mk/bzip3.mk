# bzip3
CODECS += BZIP3
BZIP3_OBJS  := bwt/bzip3/src/libbz3.o
BZIP3_FLAGS := -DVERSION=\"1.5.4\" -I$(SRC)bwt/bzip3/include
