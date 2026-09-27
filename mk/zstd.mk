# zstd
CODECS += ZSTD
ZSTD_OBJS := $(addprefix lz/zstd/lib/, \
    common/zstd_common.o common/fse_decompress.o common/xxhash.o common/error_private.o \
    common/entropy_common.o common/pool.o common/debug.o common/threading.o \
    compress/zstd_compress.o compress/zstd_compress_literals.o compress/zstd_compress_sequences.o \
    compress/zstd_compress_superblock.o compress/zstdmt_compress.o compress/zstd_double_fast.o \
    compress/zstd_fast.o compress/zstd_lazy.o compress/zstd_ldm.o compress/zstd_opt.o \
    compress/zstd_preSplit.o compress/fse_compress.o compress/huf_compress.o compress/hist.o \
    decompress/zstd_decompress.o decompress/huf_decompress.o decompress/zstd_ddict.o \
    decompress/zstd_decompress_block.o dictBuilder/cover.o dictBuilder/divsufsort.o \
    dictBuilder/fastcover.o dictBuilder/zdict.o)
# passed to the link command as is: the compiler driver assembles it
ZSTD_OBJS += lz/zstd/lib/decompress/huf_decompress_amd64.S

ifneq ($(DISABLE_THREADING),1)
    ZSTD_FLAGS := -DZSTD_MULTITHREAD
endif
