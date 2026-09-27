# xz (liblzma), built with the hand-written lz/xz/src/config.h
CODECS += XZ
XZ_OBJS := $(addprefix lz/xz/src/liblzma/, \
    lzma/lzma_decoder.o lzma/lzma_encoder.o lzma/lzma_encoder_optimum_fast.o \
    lzma/lzma_encoder_optimum_normal.o lzma/fastpos_table.o lzma/lzma_encoder_presets.o \
    lz/lz_decoder.o lz/lz_encoder.o lz/lz_encoder_mf.o common/common.o rangecoder/price_table.o \
    common/block_decoder.o common/block_util.o common/outqueue.o common/stream_flags_common.o \
    common/index.o check/check.o common/stream_encoder_mt.o common/stream_decoder_mt.o \
    common/filter_common.o common/stream_flags_decoder.o common/stream_flags_encoder.o \
    common/block_buffer_encoder.o check/crc32_fast.o common/block_header_encoder.o \
    common/vli_encoder.o common/vli_size.o common/filter_flags_encoder.o common/filter_encoder.o \
    lzma/lzma2_encoder.o common/easy_preset.o common/block_encoder.o common/index_encoder.o \
    common/filter_decoder.o lzma/lzma2_decoder.o common/block_header_decoder.o common/vli_decoder.o \
    common/filter_flags_decoder.o common/index_hash.o)
XZ_FLAGS := $(addprefix -I$(SOURCE_PATH),. lz/xz/src lz/xz/src/common lz/xz/src/liblzma/delta \
                lz/xz/src/liblzma/simple lz/xz/src/liblzma/api lz/xz/src/liblzma/common \
                lz/xz/src/liblzma/lzma lz/xz/src/liblzma/lz lz/xz/src/liblzma/check \
                lz/xz/src/liblzma/rangecoder) \
            -DHAVE_CHECK_CRC32 -DMYTHREAD_POSIX -DHAVE_CONFIG_H
