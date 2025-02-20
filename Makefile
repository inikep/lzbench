# Multi-threaded build:
#	make -j$(nproc)
#
# To order static or dynamic linking set `BUILD_STATIC` to 1, or 0 respectively:
#	make BUILD_STATIC=1
#
# For 32-bit compilation:
#	make BUILD_ARCH=32-bit
#
# For non-default compiler:
#	make CC=gcc-14 CXX=g++-14
#
# For an optimized but non-portable build, use:
#	make MOREFLAGS="-march=native"
# or
#	make USER_CFLAGS="-march=native" USER_CXXFLAGS="-march=native"


# direct GNU Make to search the directories relative to the
# parent directory of this file
SOURCE_PATH=$(dir $(lastword $(MAKEFILE_LIST)))
vpath
vpath %.c $(SOURCE_PATH)
vpath %.cc $(SOURCE_PATH)
vpath %.cpp $(SOURCE_PATH)
vpath bench/lzbench.h $(SOURCE_PATH)
vpath wflz/wfLZ.h $(SOURCE_PATH)

ifeq ($(BUILD_ARCH),32-bit)
	CODE_FLAGS += -m32
	LDFLAGS += -m32
	DONT_BUILD_LZSSE ?= 1
endif

CC?=gcc

COMPILER = $(shell $(CC) -v 2>&1 | grep -q "clang version" && echo clang || echo gcc)
GCC_VERSION = $(shell echo | $(CC) -dM -E - | grep __VERSION__  | sed -e 's:\#define __VERSION__ "\([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')
CLANG_VERSION = $(shell $(CC) -v 2>&1 | grep "clang version" | sed -e 's:.*version \([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')

# LZSSE requires compiler with __SSE4_1__ support and 64-bit CPU
ifneq ($(shell echo|$(CC) -dM -E - -march=native|egrep -c '__(SSE4_1|x86_64)__'), 2)
    DONT_BUILD_LZSSE ?= 1
endif

# detect Windows
ifneq (,$(filter Windows%,$(OS)))
	ifeq ($(COMPILER),clang)
		DONT_BUILD_GLZA ?= 1
	endif
	BUILD_STATIC ?= 1
	ifeq ($(BUILD_STATIC),1)
		LDFLAGS += -lshell32 -lole32 -loleaut32 -static
	endif
else
	ifeq ($(shell uname -p),powerpc)
		# density and yappy don't work with big-endian PowerPC
		DONT_BUILD_DENSITY ?= 1
		DONT_BUILD_YAPPY ?= 1
		DONT_BUILD_ZLING ?= 1
	endif

	# detect MacOS
	detected_OS := $(shell uname -s)
	ifeq ($(detected_OS), Darwin)
		DONT_BUILD_LZHAM ?= 1
		DONT_BUILD_CSC ?= 1
	endif

	LDFLAGS	+= -pthread

	ifeq ($(BUILD_STATIC),1)
		LDFLAGS	+= -static -static-libstdc++
	endif
endif


DEFINES     += -I.
CODE_FLAGS  += -Wno-unknown-pragmas -Wno-sign-compare -Wno-conversion

# don't use "-ffast-math" for clang < 10.0
ifeq (1, $(shell [ "$(COMPILER)" = "clang" ] && expr $(CLANG_VERSION) \< 100000 ))
	OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing
else
	OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing -ffast-math
endif

ifeq ($(BUILD_TYPE),debug)
	OPT_FLAGS_O2 = $(OPT_FLAGS) -Og
	OPT_FLAGS_O3 = $(OPT_FLAGS) -Og
else
	OPT_FLAGS_O2 = $(OPT_FLAGS) -O2 -DNDEBUG
	OPT_FLAGS_O3 = $(OPT_FLAGS) -O3 -DNDEBUG
endif

CXXFLAGS  = $(CODE_FLAGS) $(OPT_FLAGS_O3) $(DEFINES) $(MOREFLAGS) $(USER_CXXFLAGS)
CFLAGS    = $(CODE_FLAGS) $(OPT_FLAGS_O3) $(DEFINES) $(MOREFLAGS) $(USER_CFLAGS)
CFLAGS_O2 = $(CODE_FLAGS) $(OPT_FLAGS_O2) $(DEFINES) $(MOREFLAGS) $(USER_CFLAGS)
LDFLAGS  += $(MOREFLAGS) $(USER_LDFLAGS)
ifeq ($(detected_OS), Darwin)
    CXXFLAGS += -std=c++14
endif


LZ_CODECS     = bench/lz_codecs.o
BUGGY_CODECS  = bench/buggy_codecs.o
LZBENCH_FILES = $(LZ_CODECS) $(BUGGY_CODECS) bench/lzbench.o  bench/symmetric_codecs.o bench/misc_codecs.o


ifeq "$(DONT_BUILD_BLOSCLZ)" "1"
	DEFINES += -DBENCH_REMOVE_BLOSCLZ
else
    MISC_FILES += lz/blosclz/blosc/blosclz.o lz/blosclz/blosc/fastcopy.o
endif


ifeq "$(DONT_BUILD_BRIEFLZ)" "1"
	DEFINES += -DBENCH_REMOVE_BRIEFLZ
else
    BRIEFLZ_FILES = lz/brieflz/brieflz.o lz/brieflz/depack.o lz/brieflz/depacks.o
endif


ifeq "$(DONT_BUILD_BROTLI)" "1"
	DEFINES += -DBENCH_REMOVE_BROTLI
else
    BROTLI_FILES = lz/brotli/common/constants.o lz/brotli/common/context.o lz/brotli/common/dictionary.o lz/brotli/common/platform.o lz/brotli/common/transform.o
    BROTLI_FILES += lz/brotli/dec/bit_reader.o lz/brotli/dec/decode.o lz/brotli/dec/huffman.o lz/brotli/dec/state.o
    BROTLI_FILES += lz/brotli/enc/backward_references.o lz/brotli/enc/block_splitter.o lz/brotli/enc/brotli_bit_stream.o lz/brotli/enc/encode.o lz/brotli/enc/encoder_dict.o
    BROTLI_FILES += lz/brotli/enc/entropy_encode.o lz/brotli/enc/fast_log.o lz/brotli/enc/histogram.o lz/brotli/enc/command.o lz/brotli/enc/literal_cost.o lz/brotli/enc/memory.o
    BROTLI_FILES += lz/brotli/enc/metablock.o lz/brotli/enc/static_dict.o lz/brotli/enc/utf8_util.o lz/brotli/enc/compress_fragment.o lz/brotli/enc/compress_fragment_two_pass.o
    BROTLI_FILES += lz/brotli/enc/cluster.o lz/brotli/enc/bit_cost.o lz/brotli/enc/backward_references_hq.o lz/brotli/enc/dictionary_hash.o lz/brotli/common/shared_dictionary.o
    BROTLI_FILES += lz/brotli/enc/compound_dictionary.o
endif


ifeq "$(DONT_BUILD_CRUSH)" "1"
	DEFINES += -DBENCH_REMOVE_CRUSH
else
    MISC_FILES += lz/crush/crush.o
endif


ifeq "$(DONT_BUILD_FASTLZ)" "1"
	DEFINES += -DBENCH_REMOVE_FASTLZ
else
    MISC_FILES += lz/fastlz/fastlz.o
endif


ifeq "$(DONT_BUILD_FASTLZMA2)" "1"
	DEFINES += -DBENCH_REMOVE_FASTLZMA2
else
	FASTLZMA2_SRC = $(wildcard lz/fast-lzma2/*.c)
	FASTLZMA2_OBJ = $(FASTLZMA2_SRC:.c=.o)
endif


ifeq "$(DONT_BUILD_KANZI)" "1"
    DEFINES += -DBENCH_REMOVE_KANZI
else
    KANZI_FILES = misc/kanzi-cpp/src/io/CompressedOutputStream.o misc/kanzi-cpp/src/io/CompressedInputStream.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/EntropyUtils.o misc/kanzi-cpp/src/entropy/ExpGolombEncoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/FPAQEncoder.o misc/kanzi-cpp/src/entropy/ANSRangeEncoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/ANSRangeDecoder.o misc/kanzi-cpp/src/entropy/BinaryEntropyDecoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/BinaryEntropyEncoder.o misc/kanzi-cpp/src/entropy/ExpGolombDecoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/HuffmanEncoder.o misc/kanzi-cpp/src/entropy/FPAQDecoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/TPAQPredictor.o misc/kanzi-cpp/src/entropy/CMPredictor.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/HuffmanCommon.o misc/kanzi-cpp/src/entropy/RangeDecoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/RangeEncoder.o misc/kanzi-cpp/src/entropy/BinaryEntropyEncoder.o
    KANZI_FILES += misc/kanzi-cpp/src/entropy/HuffmanDecoder.o misc/kanzi-cpp/src/entropy/BinaryEntropyDecoder.o
    KANZI_FILES += misc/kanzi-cpp/src/bitstream/DefaultInputBitStream.o misc/kanzi-cpp/src/bitstream/DebugOutputBitStream.o
    KANZI_FILES += misc/kanzi-cpp/src/bitstream/DebugInputBitStream.o misc/kanzi-cpp/src/bitstream/DefaultOutputBitStream.o
    KANZI_FILES += misc/kanzi-cpp/src/Event.o misc/kanzi-cpp/src/Global.o misc/kanzi-cpp/src/transform/AliasCodec.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/BWT.o misc/kanzi-cpp/src/transform/RLT.o misc/kanzi-cpp/src/transform/TextCodec.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/EXECodec.o misc/kanzi-cpp/src/transform/SBRT.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/ROLZCodec.o misc/kanzi-cpp/src/transform/LZCodec.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/SRT.o misc/kanzi-cpp/src/transform/DivSufSort.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/BWTBlockCodec.o misc/kanzi-cpp/src/transform/BWTS.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/UTFCodec.o misc/kanzi-cpp/src/transform/ZRLT.o
    KANZI_FILES += misc/kanzi-cpp/src/transform/FSDCodec.o
endif


ifeq "$(DONT_BUILD_LIBDEFLATE)" "1"
    DEFINES += -DBENCH_REMOVE_LIBDEFLATE
else
    LIBDEFLATE_FILES  = lz/libdeflate/lib/adler32.o lz/libdeflate/lib/crc32.o lz/libdeflate/lib/deflate_compress.o
    LIBDEFLATE_FILES += lz/libdeflate/lib/deflate_decompress.o lz/libdeflate/lib/gzip_compress.o
    LIBDEFLATE_FILES += lz/libdeflate/lib/gzip_decompress.o lz/libdeflate/lib/utils.o lz/libdeflate/lib/zlib_compress.o
    LIBDEFLATE_FILES += lz/libdeflate/lib/zlib_decompress.o lz/libdeflate/lib/x86/cpu_features.o
    LIBDEFLATE_FILES += lz/libdeflate/lib/arm/cpu_features.o
endif


ifeq "$(DONT_BUILD_LIZARD)" "1"
	DEFINES += -DBENCH_REMOVE_LIZARD
else
    LIZARD_FILES = lz/lizard/lizard_compress.o lz/lizard/lizard_decompress.o
    LIZARD_FILES += lz/lizard/entropy/huf_compress.o lz/lizard/entropy/huf_decompress.o lz/lizard/entropy/entropy_common.o
    LIZARD_FILES += lz/lizard/entropy/fse_compress.o lz/lizard/entropy/fse_decompress.o lz/lizard/entropy/hist.o
endif


ifeq "$(DONT_BUILD_LZ4)" "1"
	DEFINES += -DBENCH_REMOVE_LZ4
else
    LZ4_FILES = lz/lz4/lib/lz4.o lz/lz4/lib/lz4hc.o
endif


ifeq "$(DONT_BUILD_LZF)" "1"
	DEFINES += -DBENCH_REMOVE_LZF
else
    LZF_FILES = lz/lzf/lzf_c_ultra.o lz/lzf/lzf_c_very.o lz/lzf/lzf_d.o
endif


ifeq "$(DONT_BUILD_LZFSE)" "1"
	DEFINES += -DBENCH_REMOVE_LZFSE
else
    LZFSE_FILES  = lz/lzfse/lzfse_decode.o lz/lzfse/lzfse_decode_base.o lz/lzfse/lzfse_encode.o lz/lzfse/lzfse_encode_base.o
    LZFSE_FILES += lz/lzfse/lzfse_fse.o lz/lzfse/lzvn_decode.o lz/lzfse/lzvn_decode_base.o lz/lzfse/lzvn_encode_base.o
endif


ifeq "$(DONT_BUILD_LZG)" "1"
	DEFINES += -DBENCH_REMOVE_LZG
else
    LIBLZG_FILES = lz/liblzg/decode.o lz/liblzg/encode.o lz/liblzg/checksum.o
endif


ifeq "$(DONT_BUILD_LZHAM)" "1"
    DEFINES += -DBENCH_REMOVE_LZHAM
else
    LZHAM_FILES  = lz/lzham/lzham_assert.o lz/lzham/lzham_checksum.o lz/lzham/lzham_huffman_codes.o lz/lzham/lzham_lzbase.o
    LZHAM_FILES += lz/lzham/lzham_lzcomp.o lz/lzham/lzham_lzcomp_internal.o lz/lzham/lzham_lzdecomp.o lz/lzham/lzham_lzdecompbase.o
    LZHAM_FILES += lz/lzham/lzham_match_accel.o lz/lzham/lzham_mem.o lz/lzham/lzham_platform.o lz/lzham/lzham_lzcomp_state.o
    LZHAM_FILES += lz/lzham/lzham_prefix_coding.o lz/lzham/lzham_symbol_codec.o lz/lzham/lzham_timer.o lz/lzham/lzham_vector.o lz/lzham/lzham_lib.o
endif


ifeq "$(DONT_BUILD_LZLIB)" "1"
	DEFINES += -DBENCH_REMOVE_LZLIB
else
	MISC_FILES += lz/lzlib/lzlib.o
endif


ifeq "$(DONT_BUILD_LZMA)" "1"
    DEFINES += -DBENCH_REMOVE_LZMA
else
    LZMA_FILES  = misc/7-zip/CpuArch.o misc/7-zip/LzFind.o misc/7-zip/LzFindOpt.o misc/7-zip/LzFindMt.o
    LZMA_FILES += misc/7-zip/LzmaDec.o misc/7-zip/LzmaEnc.o misc/7-zip/Threads.o
endif


ifeq "$(DONT_BUILD_LZO)" "1"
    DEFINES += -DBENCH_REMOVE_LZO
else
    LZO_FILES = lz/lzo/lzo1.o lz/lzo/lzo1a.o lz/lzo/lzo1a_99.o lz/lzo/lzo1b_1.o lz/lzo/lzo1b_2.o lz/lzo/lzo1b_3.o lz/lzo/lzo1b_4.o lz/lzo/lzo1b_5.o
    LZO_FILES += lz/lzo/lzo1b_6.o lz/lzo/lzo1b_7.o lz/lzo/lzo1b_8.o lz/lzo/lzo1b_9.o lz/lzo/lzo1b_99.o lz/lzo/lzo1b_9x.o lz/lzo/lzo1b_cc.o
    LZO_FILES += lz/lzo/lzo1b_d1.o lz/lzo/lzo1b_d2.o lz/lzo/lzo1b_rr.o lz/lzo/lzo1b_xx.o lz/lzo/lzo1c_1.o lz/lzo/lzo1c_2.o lz/lzo/lzo1c_3.o
    LZO_FILES += lz/lzo/lzo1c_4.o lz/lzo/lzo1c_5.o lz/lzo/lzo1c_6.o lz/lzo/lzo1c_7.o lz/lzo/lzo1c_8.o lz/lzo/lzo1c_9.o lz/lzo/lzo1c_99.o
    LZO_FILES += lz/lzo/lzo1c_9x.o lz/lzo/lzo1c_cc.o lz/lzo/lzo1c_d1.o lz/lzo/lzo1c_d2.o lz/lzo/lzo1c_rr.o lz/lzo/lzo1c_xx.o lz/lzo/lzo1f_1.o
    LZO_FILES += lz/lzo/lzo1f_9x.o lz/lzo/lzo1f_d1.o lz/lzo/lzo1f_d2.o lz/lzo/lzo1x_1.o lz/lzo/lzo1x_1k.o lz/lzo/lzo1x_1l.o lz/lzo/lzo1x_1o.o
    LZO_FILES += lz/lzo/lzo1x_9x.o lz/lzo/lzo1x_d1.o lz/lzo/lzo1x_d2.o lz/lzo/lzo1x_d3.o lz/lzo/lzo1x_o.o lz/lzo/lzo1y_1.o lz/lzo/lzo1y_9x.o
    LZO_FILES += lz/lzo/lzo1y_d1.o lz/lzo/lzo1y_d2.o lz/lzo/lzo1y_d3.o lz/lzo/lzo1y_o.o lz/lzo/lzo1z_9x.o lz/lzo/lzo1z_d1.o lz/lzo/lzo1z_d2.o
    LZO_FILES += lz/lzo/lzo1z_d3.o lz/lzo/lzo1_99.o lz/lzo/lzo2a_9x.o lz/lzo/lzo2a_d1.o lz/lzo/lzo2a_d2.o lz/lzo/lzo_crc.o lz/lzo/lzo_init.o
    LZO_FILES += lz/lzo/lzo_ptr.o lz/lzo/lzo_str.o lz/lzo/lzo_util.o
endif


ifeq "$(DONT_BUILD_LZSSE)" "1"
    DEFINES += -DBENCH_REMOVE_LZSSE
else
    LZSSE_FILES = lz/lzsse/lzsse2/lzsse2.o lz/lzsse/lzsse4/lzsse4.o lz/lzsse/lzsse8/lzsse8.o
endif


ifeq "$(DONT_BUILD_QUICKLZ)" "1"
	DEFINES += -DBENCH_REMOVE_QUICKLZ
else
    QUICKLZ_FILES = lz/quicklz/quicklz151b7.o lz/quicklz/quicklz1.o lz/quicklz/quicklz2.o lz/quicklz/quicklz3.o
endif


ifeq "$(DONT_BUILD_SLZ)" "1"
	DEFINES += -DBENCH_REMOVE_SLZ
else
	MISC_FILES += lz/slz/src/slz.o
endif


ifeq "$(DONT_BUILD_SNAPPY)" "1"
	DEFINES += -DBENCH_REMOVE_SNAPPY
else
	SNAPPY_FILES = lz/snappy/snappy-sinksource.o lz/snappy/snappy-stubs-internal.o lz/snappy/snappy.o
endif


ifeq "$(DONT_BUILD_TORNADO)" "1"
    DEFINES += "-DBENCH_REMOVE_TORNADO"
    LZMA_FILES += misc/7-zip/Alloc.o
else
    MISC_FILES += lz/tornado/tor_test.o
endif


ifeq "$(DONT_BUILD_UCL)" "1"
	DEFINES += -DBENCH_REMOVE_UCL
else
    UCL_FILES = lz/ucl/alloc.o lz/ucl/n2b_99.o lz/ucl/n2b_d.o lz/ucl/n2b_ds.o lz/ucl/n2b_to.o lz/ucl/n2d_99.o lz/ucl/n2d_d.o lz/ucl/n2d_ds.o
    UCL_FILES += lz/ucl/n2d_to.o lz/ucl/n2e_99.o lz/ucl/n2e_d.o lz/ucl/n2e_ds.o lz/ucl/n2e_to.o lz/ucl/ucl_crc.o lz/ucl/ucl_init.o
    UCL_FILES += lz/ucl/ucl_ptr.o lz/ucl/ucl_str.o lz/ucl/ucl_util.o
endif


ifeq "$(DONT_BUILD_XPACK)" "1"
	DEFINES += -DBENCH_REMOVE_XPACK
else
	XPACK_FILES = lz/xpack/lib/x86_cpu_features.o lz/xpack/lib/xpack_common.o lz/xpack/lib/xpack_compress.o lz/xpack/lib/xpack_decompress.o
endif


ifeq "$(DONT_BUILD_XZ)" "1"
	DEFINES += -DBENCH_REMOVE_XZ
else
    XZ_FILES = lz/xz/src/liblzma/lzma/lzma_decoder.o lz/xz/src/liblzma/lzma/lzma_encoder.o lz/xz/src/liblzma/lzma/lzma_encoder_optimum_fast.o lz/xz/src/liblzma/lzma/lzma_encoder_optimum_normal.o lz/xz/src/liblzma/lzma/fastpos_table.o
    XZ_FILES += lz/xz/src/liblzma/lzma/lzma_encoder_presets.o lz/xz/src/liblzma/lz/lz_decoder.o lz/xz/src/liblzma/lz/lz_encoder.o lz/xz/src/liblzma/lz/lz_encoder_mf.o lz/xz/src/liblzma/common/common.o lz/xz/src/liblzma/rangecoder/price_table.o
    XZ_FILES += lz/xz/src/liblzma/common/alone_encoder.o lz/xz/src/liblzma/common/alone_decoder.o lz/xz/src/liblzma/check/crc32_table.o
    XZ_FLAGS = $(addprefix -I$(SOURCE_PATH),. lz/xz/src lz/xz/src/common lz/xz/src/liblzma/api lz/xz/src/liblzma/common lz/xz/src/liblzma/lzma lz/xz/src/liblzma/lz lz/xz/src/liblzma/check lz/xz/src/liblzma/rangecoder)
endif


ifeq "$(DONT_BUILD_ZLIB)" "1"
    DEFINES += -DBENCH_REMOVE_ZLIB
else
    ZLIB_FILES  = lz/zlib/adler32.o lz/zlib/compress.o lz/zlib/crc32.o lz/zlib/deflate.o lz/zlib/gzclose.o
    ZLIB_FILES += lz/zlib/gzlib.o lz/zlib/gzread.o lz/zlib/gzwrite.o lz/zlib/infback.o lz/zlib/inffast.o
    ZLIB_FILES += lz/zlib/inflate.o lz/zlib/inftrees.o lz/zlib/trees.o lz/zlib/uncompr.o lz/zlib/zutil.o
endif


ifeq "$(DONT_BUILD_ZLIB_NG)" "1"
    DEFINES += -DBENCH_REMOVE_ZLIB_NG
else
    ZLIB_NG_FILES  = lz/zlib-ng/adler32.o lz/zlib-ng/crc32.o lz/zlib-ng/deflate_medium.o lz/zlib-ng/deflate_stored.o lz/zlib-ng/inftrees.o lz/zlib-ng/uncompr.o
    ZLIB_NG_FILES += lz/zlib-ng/compress.o lz/zlib-ng/deflate.o lz/zlib-ng/deflate_quick.o lz/zlib-ng/functable.o lz/zlib-ng/insert_string.o lz/zlib-ng/zutil.o
    ZLIB_NG_FILES += lz/zlib-ng/cpu_features.o lz/zlib-ng/deflate_fast.o lz/zlib-ng/deflate_rle.o lz/zlib-ng/infback.o lz/zlib-ng/insert_string_roll.o
    ZLIB_NG_FILES += lz/zlib-ng/crc32_braid_comb.o lz/zlib-ng/deflate_huff.o lz/zlib-ng/deflate_slow.o lz/zlib-ng/inflate.o lz/zlib-ng/trees.o

    ZLIB_NG_FILES += lz/zlib-ng/arch/generic/adler32_c.o lz/zlib-ng/arch/generic/chunkset_c.o lz/zlib-ng/arch/generic/crc32_braid_c.o lz/zlib-ng/arch/generic/slide_hash_c.o
    ZLIB_NG_FILES += lz/zlib-ng/arch/generic/adler32_fold_c.o lz/zlib-ng/arch/generic/compare256_c.o lz/zlib-ng/arch/generic/crc32_fold_c.o

#    ZLIB_NG_FILES += lz/zlib-ng/arch/x86/adler32_avx2.o lz/zlib-ng/arch/x86/adler32_ssse3.o lz/zlib-ng/arch/x86/chunkset_ssse3.o lz/zlib-ng/arch/x86/crc32_vpclmulqdq.o
#    ZLIB_NG_FILES += lz/zlib-ng/arch/x86/adler32_avx512.o lz/zlib-ng/arch/x86/chunkset_avx2.o lz/zlib-ng/arch/x86/compare256_avx2.o lz/zlib-ng/arch/x86/slide_hash_avx2.o
#    ZLIB_NG_FILES += lz/zlib-ng/arch/x86/adler32_avx512_vnni.o lz/zlib-ng/arch/x86/chunkset_avx512.o lz/zlib-ng/arch/x86/compare256_sse2.o lz/zlib-ng/arch/x86/slide_hash_sse2.o
#    ZLIB_NG_FILES += lz/zlib-ng/arch/x86/adler32_sse42.o lz/zlib-ng/arch/x86/chunkset_sse2.o lz/zlib-ng/arch/x86/crc32_pclmulqdq.o lz/zlib-ng/arch/x86/x86_features.o
endif


ifeq "$(DONT_BUILD_ZLING)" "1"
    DEFINES += -DBENCH_REMOVE_ZLING
else
    ZLING_FILES = lz/libzling/libzling.o lz/libzling/libzling_huffman.o lz/libzling/libzling_lz.o lz/libzling/libzling_utils.o
endif


ifeq "$(DONT_BUILD_ZSTD)" "1"
    DEFINES += -DBENCH_REMOVE_ZSTD
else
	ZSTD_FILES  = lz/zstd/lib/common/zstd_common.o
	ZSTD_FILES += lz/zstd/lib/common/fse_decompress.o
	ZSTD_FILES += lz/zstd/lib/common/xxhash.o
	ZSTD_FILES += lz/zstd/lib/common/error_private.o
	ZSTD_FILES += lz/zstd/lib/common/entropy_common.o
	ZSTD_FILES += lz/zstd/lib/common/pool.o
	ZSTD_FILES += lz/zstd/lib/common/debug.o
	ZSTD_FILES += lz/zstd/lib/common/threading.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_compress.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_compress_literals.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_compress_sequences.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_compress_superblock.o
	ZSTD_FILES += lz/zstd/lib/compress/zstdmt_compress.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_double_fast.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_fast.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_lazy.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_ldm.o
	ZSTD_FILES += lz/zstd/lib/compress/zstd_opt.o
	ZSTD_FILES += lz/zstd/lib/compress/fse_compress.o
	ZSTD_FILES += lz/zstd/lib/compress/huf_compress.o
	ZSTD_FILES += lz/zstd/lib/compress/hist.o
	ZSTD_FILES += lz/zstd/lib/decompress/zstd_decompress.o
	ZSTD_FILES += lz/zstd/lib/decompress/huf_decompress.o
	ZSTD_FILES += lz/zstd/lib/decompress/zstd_ddict.o
	ZSTD_FILES += lz/zstd/lib/decompress/zstd_decompress_block.o
	ZSTD_FILES += lz/zstd/lib/dictBuilder/cover.o
	ZSTD_FILES += lz/zstd/lib/dictBuilder/divsufsort.o
	ZSTD_FILES += lz/zstd/lib/dictBuilder/fastcover.o
	ZSTD_FILES += lz/zstd/lib/dictBuilder/zdict.o
	MISC_FILES += lz/zstd/lib/decompress/huf_decompress_amd64.S
endif



# Symmetric codecs
ifeq "$(DONT_BUILD_BSC)" "1"
    DEFINES += -DBENCH_REMOVE_BSC
else
    BSC_FLAGS = -DLIBBSC_SORT_TRANSFORM_SUPPORT -DLIBBSC_ALLOW_UNALIGNED_ACCESS

    BSC_C_FILES = bwt/libbsc/libbsc/bwt/libsais/libsais.o

    BSC_CXX_FILES  = bwt/libbsc/libbsc/adler32/adler32.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/bwt/bwt.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/coder/coder.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/coder/qlfc/qlfc.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/coder/qlfc/qlfc_model.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/filters/detectors.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/filters/preprocessing.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/libbsc/libbsc.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/lzp/lzp.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/platform/platform.o
    BSC_CXX_FILES += bwt/libbsc/libbsc/st/st.o
endif


ifeq "$(DONT_BUILD_BZIP2)" "1"
    DEFINES += -DBENCH_REMOVE_BZIP2
else
    BZIP2_FILES += bwt/bzip2/blocksort.o bwt/bzip2/huffman.o bwt/bzip2/crctable.o bwt/bzip2/randtable.o
    BZIP2_FILES += bwt/bzip2/compress.o bwt/bzip2/decompress.o bwt/bzip2/bzlib.o
endif


ifeq "$(DONT_BUILD_PPMD)" "1"
    DEFINES += -DBENCH_REMOVE_PPMD
else
    PPMD_FILES += misc/7-zip/Ppmd8.o misc/7-zip/Ppmd8Dec.o misc/7-zip/Ppmd8Enc.o
endif



# Misc codecs
ifeq "$(DONT_BUILD_GLZA)" "1"
    DEFINES += -DBENCH_REMOVE_GLZA
else
    MISC_FILES += misc/glza/GLZAcomp.o misc/glza/GLZAformat.o misc/glza/GLZAcompress.o
    MISC_FILES += misc/glza/GLZAencode.o misc/glza/GLZAdecode.o misc/glza/GLZAmodel.o
endif


ifeq "$(DONT_BUILD_LZJB)" "1"
	DEFINES += -DBENCH_REMOVE_LZJB
else
	MISC_FILES += lz/lzjb/lzjb2010.o
endif


ifeq "$(BENCH_HAS_NAKAMICHI)" "1"
    DEFINES += -DBENCH_HAS_NAKAMICHI
    MISC_FILES += misc/nakamichi/Nakamichi_Okamigan.o
endif


ifeq "$(DONT_BUILD_TAMP)" "1"
	DEFINES += -DBENCH_REMOVE_TAMP
else
	MISC_FILES += lz/tamp/common.o lz/tamp/compressor.o lz/tamp/decompressor.o
endif



# Buggy codecs
ifeq "$(DONT_BUILD_CSC)" "1"
    DEFINES += -DBENCH_REMOVE_CSC
else
	CSC_FILES += lz/libcsc/csc_analyzer.o lz/libcsc/csc_coder.o lz/libcsc/csc_dec.o lz/libcsc/csc_enc.o
    CSC_FILES += lz/libcsc/csc_encoder_main.o lz/libcsc/csc_filters.o lz/libcsc/csc_lz.o lz/libcsc/csc_memio.o
	CSC_FILES += lz/libcsc/csc_mf.o lz/libcsc/csc_model.o lz/libcsc/csc_profiler.o lz/libcsc/csc_default_alloc.o
endif


ifeq "$(DONT_BUILD_DENSITY)" "1"
    DEFINES += -DBENCH_REMOVE_DENSITY
else
    BUGGY_FILES += lz/density/globals.o lz/density/buffers/buffer.o
    BUGGY_FILES += lz/density/algorithms/cheetah/core/cheetah_decode.o lz/density/algorithms/cheetah/core/cheetah_encode.o
    BUGGY_FILES += lz/density/algorithms/lion/forms/lion_form_model.o lz/density/algorithms/lion/core/lion_decode.o
    BUGGY_FILES += lz/density/algorithms/lion/core/lion_encode.o lz/density/algorithms/dictionaries.o
    BUGGY_FILES += lz/density/algorithms/chameleon/core/chameleon_decode.o lz/density/algorithms/chameleon/core/chameleon_encode.o
    BUGGY_FILES += lz/density/algorithms/algorithms.o lz/density/structure/header.o
endif


ifeq "$(DONT_BUILD_GIPFELI)" "1"
    DEFINES += -DBENCH_REMOVE_GIPFELI
else
    BUGGY_CC_FILES += lz/gipfeli/decompress.o lz/gipfeli/entropy.o lz/gipfeli/entropy_code_builder.o lz/gipfeli/gipfeli-internal.o lz/gipfeli/lz77.o
endif


ifeq "$(DONT_BUILD_LZMAT)" "1"
    DEFINES += -DBENCH_REMOVE_LZMAT
else
    BUGGY_FILES += lz/lzmat/lzmat_dec.o lz/lzmat/lzmat_enc.o
endif


ifeq "$(DONT_BUILD_LZRW)" "1"
    DEFINES += -DBENCH_REMOVE_LZRW
else
    BUGGY_FILES += lz/lzrw/lzrw1-a.o lz/lzrw/lzrw1.o lz/lzrw/lzrw2.o lz/lzrw/lzrw3.o lz/lzrw/lzrw3-a.o
endif


ifeq "$(DONT_BUILD_PITHY)" "1"
    DEFINES += -DBENCH_REMOVE_PITHY
else
    BUGGY_CXX_FILES += lz/pithy/pithy.o
endif


ifeq "$(DONT_BUILD_SHRINKER)" "1"
    DEFINES += -DBENCH_REMOVE_SHRINKER
else
    BUGGY_FILES += lz/shrinker/shrinker.o
endif


ifeq "$(DONT_BUILD_WFLZ)" "1"
    DEFINES += -DBENCH_REMOVE_WFLZ
else
    BUGGY_FILES += lz/wflz/wfLZ.o
endif


ifeq "$(DONT_BUILD_YAPPY)" "1"
    DEFINES += -DBENCH_REMOVE_YAPPY
else
    BUGGY_CXX_FILES += lz/yappy/yappy.o
endif



# CUDA-based codecs
CUDA_BASE ?= /usr/local/cuda
LIBCUDART=$(wildcard $(CUDA_BASE)/lib64/libcudart.so)

ifeq "$(LIBCUDART)" ""
    $(info CUDA Toolkit not found at $(CUDA_BASE), CUDA support will be disabled.)
    $(info Run "make CUDA_BASE=..." to use a different path.)
    CUDA_BASE =
    LIBCUDART =
else
    DEFINES += -DBENCH_HAS_CUDA -I$(CUDA_BASE)/include
    LDFLAGS += -L$(CUDA_BASE)/lib64 -lcudart -Wl,-rpath=$(CUDA_BASE)/lib64
    CUDA_COMPILER = nvcc
    CUDA_CC = $(CUDA_BASE)/bin/nvcc --compiler-bindir $(CXX)
    CUDA_ARCH = 50 52 60 61 70 75 80 86 89
    CUDA_CXXFLAGS = -x cu -std=c++14 -O3 $(foreach ARCH, $(CUDA_ARCH), --generate-code=arch=compute_$(ARCH),code=[compute_$(ARCH),sm_$(ARCH)]) --expt-extended-lambda -forward-unknown-to-host-compiler -Wno-deprecated-gpu-targets

ifneq "$(DONT_BUILD_NVCOMP)" "1"
    DEFINES += -DBENCH_HAS_NVCOMP
    NVCOMP_CPP_SRC = $(wildcard misc/nvcomp/src/*.cpp misc/nvcomp/src/lowlevel/*.cpp)
    NVCOMP_CPP_OBJ = $(NVCOMP_CPP_SRC:%=%.o)
    NVCOMP_CU_SRC  = $(wildcard misc/nvcomp/src/*.cu misc/nvcomp/src/lowlevel/*.cu)
    NVCOMP_CU_OBJ  = $(NVCOMP_CU_SRC:%=%.o)
    NVCOMP_FILES   = $(NVCOMP_CU_OBJ) $(NVCOMP_CPP_OBJ)
endif

ifneq "$(DONT_BUILD_BSC)" "1"
    BSC_FLAGS += -DLIBBSC_CUDA_SUPPORT
    BSC_CUDA_FILES = bwt/libbsc/libbsc/bwt/libcubwt/libcubwt.cu.o bwt/libbsc/libbsc/st/st.cu.o
endif
endif # ifneq "$(LIBCUDART)"



MKDIR = mkdir -p

lzbench: $(BUGGY_FILES) $(BUGGY_CC_FILES) $(BUGGY_CXX_FILES) $(CSC_FILES) $(BSC_C_FILES) $(BSC_CXX_FILES) $(BSC_CUDA_FILES) $(BZIP2_FILES) $(KANZI_FILES) $(FASTLZMA2_OBJ) $(ZSTD_FILES) $(LZSSE_FILES) $(LZFSE_FILES) $(XPACK_FILES) $(XZ_FILES) $(LIBLZG_FILES) $(BRIEFLZ_FILES) $(LZF_FILES) $(BROTLI_FILES) $(LZMA_FILES) $(ZLING_FILES) $(QUICKLZ_FILES) $(SNAPPY_FILES) $(ZLIB_FILES) $(ZLIB_NG_FILES) $(LZHAM_FILES) $(LZO_FILES) $(UCL_FILES) $(LZ4_FILES) $(LIZARD_FILES) $(LIBDEFLATE_FILES) $(MISC_FILES) $(NVCOMP_FILES) $(LZBENCH_FILES) $(PPMD_FILES)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo Linked GCC_VERSION=$(GCC_VERSION) CLANG_VERSION=$(CLANG_VERSION) COMPILER=$(COMPILER)

bench/lzbench.o: bench/lzbench.cpp bench/lzbench.h

# disable the implicit rule for making a binary out of a single object file
%: %.o

.c.o:
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) $< -std=gnu99 -c -o $@

.cc.o:
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) $< -c -o $@

.cpp.o:
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) $< -c -o $@

# FIX for SEGFAULT on GCC 4.9+
$(BUGGY_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS_O2) $< -c -o $@

$(BUGGY_CC_FILES): %.o : %.cc
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS_O2) $< -c -o $@

$(BUGGY_CXX_FILES): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS_O2) $< -c -o $@

$(CSC_FILES): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS_O2) -Ilz/libcsc $< -c -o $@

$(LIZARD_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS_O2) $< -c -o $@

$(LZ_CODECS): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -Ilz -Ilz/brotli/include $< -c -o $@

$(BUGGY_CODECS): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -Ilz/libcsc $< -c -o $@

$(BROTLI_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz/brotli/include $< -c -o $@

$(LIBDEFLATE_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz/libdeflate $< -c -o $@

$(LZO_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz $< -c -o $@

$(UCL_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz $< -c -o $@

$(XPACK_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz $< -c -o $@

$(LZSSE_FILES): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -std=c++0x -msse4.1 $< -c -o $@

$(FASTLZMA2_OBJ): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -DFL2_SINGLETHREAD -DNO_XXHASH $< -c -o $@

$(XZ_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) $(XZ_FLAGS) -DHAVE_CONFIG_H $< -c -o $@

$(ZLIB_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -DZ_HAVE_UNISTD_H $< -c -o $@

$(ZLIB_NG_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -Ilz/zlib-ng $< -c -o $@

$(NVCOMP_CU_OBJ): %.cu.o: %.cu
	@$(MKDIR) $(dir $@)
	$(CUDA_CC) $(CUDA_CXXFLAGS) $(CXXFLAGS) -Imisc/nvcomp/include -Imisc/nvcomp/src -Imisc/nvcomp/src/lowlevel -c $< -o $@

$(NVCOMP_CPP_OBJ): %.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -Imisc/nvcomp/include -Imisc/nvcomp/src -Imisc/nvcomp/src/lowlevel -c $< -o $@

$(BSC_C_FILES): %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) $(BSC_FLAGS) $< -c -o $@

$(BSC_CXX_FILES): %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) $(BSC_FLAGS) $< -c -o $@

$(BSC_CUDA_FILES): %.cu.o: %.cu
	@$(MKDIR) $(dir $@)
	$(CUDA_CC) $(CUDA_CXXFLAGS) $(CXXFLAGS) $(BSC_FLAGS) -c $< -o $@

nakamichi/Nakamichi_Okamigan.o: nakamichi/Nakamichi_Okamigan.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -mavx $< -c -o $@

clean:
	rm -rf lzbench lzbench.exe
	find . -type f -name "*.o" -exec rm -f {} +
