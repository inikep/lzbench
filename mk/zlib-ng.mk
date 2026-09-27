# zlib-ng: only the generic C implementations (arch/generic) are built
CODECS += ZLIB_NG
ZLIB_NG_OBJS := $(addprefix lz/zlib-ng/, \
    adler32.o crc32.o deflate_medium.o deflate_stored.o inftrees.o uncompr.o compress.o deflate.o \
    deflate_quick.o functable.o insert_string.o zutil.o cpu_features.o deflate_fast.o deflate_rle.o \
    infback.o insert_string_roll.o crc32_braid_comb.o deflate_huff.o deflate_slow.o inflate.o \
    trees.o arch/generic/adler32_c.o arch/generic/chunkset_c.o arch/generic/crc32_braid_c.o \
    arch/generic/slide_hash_c.o arch/generic/adler32_fold_c.o arch/generic/compare256_c.o \
    arch/generic/crc32_fold_c.o)
ZLIB_NG_FLAGS := -DWITH_ALL_FALLBACKS -Ilz/zlib-ng
