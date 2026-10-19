# zlib-turbo: built with Z_PREFIX to link beside zlib; Z_PREFIX leaves z_errmsg
# as it is, so it is renamed here. Z_TURBO_SIMD: CRC-32 and Adler-32 with
# PCLMULQDQ and AVX2 where the target has them (e.g. MOREFLAGS="-march=native")
CODECS += ZLIB_TURBO
ZLIB_TURBO_OBJS := bench/zlib_turbo_codecs.o $(addprefix lz+entropy/zlib-turbo/, \
    adler32.o compress.o crc32.o deflate.o infback.o inffast.o inflate.o inftrees.o trees.o uncompr.o zutil.o)
ZLIB_TURBO_FLAGS := -DZ_PREFIX -Dz_errmsg=zlib_turbo_errmsg -DZ_TURBO_SIMD -I$(SRC)lz+entropy/zlib-turbo

# levels 4-9 find matches in threads of their own with -I# (the output is that of one thread)
ifneq ($(DISABLE_THREADING),1)
    ZLIB_TURBO_FLAGS += -DZ_TURBO_THREADS
endif
