# zlib
CODECS += ZLIB
ZLIB_OBJS := $(addprefix lz+entropy/zlib/, \
    adler32.o compress.o crc32.o deflate.o gzclose.o gzlib.o gzread.o gzwrite.o infback.o inffast.o \
    inflate.o inftrees.o trees.o uncompr.o zutil.o)
ZLIB_FLAGS := -DZ_HAVE_UNISTD_H
