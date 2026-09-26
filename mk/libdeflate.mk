# libdeflate
CODECS += LIBDEFLATE
LIBDEFLATE_OBJS := $(addprefix lz/libdeflate/lib/, \
    adler32.o crc32.o deflate_compress.o deflate_decompress.o gzip_compress.o gzip_decompress.o \
    utils.o zlib_compress.o zlib_decompress.o x86/cpu_features.o arm/cpu_features.o)
LIBDEFLATE_FLAGS := -I$(SRC)lz/libdeflate
