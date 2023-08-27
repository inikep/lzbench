# direct GNU Make to search the directories relative to the
# parent directory of this file
SOURCE_PATH=$(dir $(lastword $(MAKEFILE_LIST)))
vpath
vpath %.c $(SOURCE_PATH)
vpath %.cc $(SOURCE_PATH)
vpath %.cpp $(SOURCE_PATH)
vpath _lzbench/lzbench.h $(SOURCE_PATH)
vpath wflz/wfLZ.h $(SOURCE_PATH)

#BUILD_ARCH = 32-bit
#BUILD_STATIC = 1

ifeq ($(BUILD_ARCH),32-bit)
	CODE_FLAGS += -m32
	LDFLAGS += -m32
	DONT_BUILD_LZSSE ?= 1
endif

CC?=gcc

COMPILER = $(shell $(CC) -v 2>&1 | grep -q "clang version" && echo clang || echo gcc)
GCC_VERSION = $(shell echo | $(CC) -dM -E - | grep __VERSION__  | sed -e 's:\#define __VERSION__ "\([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')
CLANG_VERSION = $(shell $(CC) -v 2>&1 | grep "clang version" | sed -e 's:.*version \([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')

# glza doesn't work with gcc < 4.9 and clang < 3.6 (missing stdatomic.h)
ifeq (1,$(filter 1,$(shell [ "$(COMPILER)" = "gcc" ] && expr $(GCC_VERSION) \< 40900) $(shell [ "$(COMPILER)" = "clang" ] && expr $(CLANG_VERSION) \< 30600)))
    DONT_BUILD_GLZA ?= 1
endif

# LZSSE requires compiler with __SSE4_1__ support and 64-bit CPU
ifneq ($(shell echo|$(CC) -dM -E - -march=native|egrep -c '__(SSE4_1|x86_64)__'), 2)
    DONT_BUILD_LZSSE ?= 1
endif

# zling requires c++14
ifeq (1,$(filter 1,$(shell [ "$(COMPILER)" = "gcc" ] && expr $(GCC_VERSION) \< 60000) $(shell [ "$(COMPILER)" = "clang" ] && expr $(CLANG_VERSION) \< 60000)))
  DONT_BUILD_ZLING ?= 1
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
	ifeq ($(shell uname -s),Darwin)
		DONT_BUILD_LZHAM ?= 1
		DONT_BUILD_CSC ?= 1
	endif

	LDFLAGS	+= -pthread -lrt

	ifeq ($(BUILD_STATIC),1)
		LDFLAGS	+= -lrt -static
		ifeq (1, $(shell [ "$(COMPILER)" = "gcc" ] && [ "$(shell uname -m)" != "aarch64" ] && expr $(GCC_VERSION) \>= 80000 ))
		  LDFLAGS += -lmvec
		endif
	endif
endif


DEFINES     += $(addprefix -I$(SOURCE_PATH),. brotli/include libcsc libdeflate xpack/common xz xz/api xz/check xz/common xz/lz xz/lzma xz/rangecoder zstd/lib zstd/lib/common)
DEFINES     += -DHAVE_CONFIG_H -DFL2_SINGLETHREAD
CODE_FLAGS  += -Wno-unknown-pragmas -Wno-sign-compare -Wno-conversion
OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing -ffast-math


ifeq ($(BUILD_TYPE),debug)
	OPT_FLAGS_O2 = $(OPT_FLAGS) -g
	OPT_FLAGS_O3 = $(OPT_FLAGS) -g
else
	OPT_FLAGS_O2 = $(OPT_FLAGS) -O2 -DNDEBUG
	OPT_FLAGS_O3 = $(OPT_FLAGS) -O3 -DNDEBUG
endif

CFLAGS = $(MOREFLAGS) $(CODE_FLAGS) $(OPT_FLAGS_O3) $(DEFINES)
CFLAGS_O2 = $(MOREFLAGS) $(CODE_FLAGS) $(OPT_FLAGS_O2) $(DEFINES)
LDFLAGS += $(MOREFLAGS)


LZO_FILES = lzo/lzo1.o lzo/lzo1a.o lzo/lzo1a_99.o lzo/lzo1b_1.o lzo/lzo1b_2.o lzo/lzo1b_3.o lzo/lzo1b_4.o lzo/lzo1b_5.o
LZO_FILES += lzo/lzo1b_6.o lzo/lzo1b_7.o lzo/lzo1b_8.o lzo/lzo1b_9.o lzo/lzo1b_99.o lzo/lzo1b_9x.o lzo/lzo1b_cc.o
LZO_FILES += lzo/lzo1b_d1.o lzo/lzo1b_d2.o lzo/lzo1b_rr.o lzo/lzo1b_xx.o lzo/lzo1c_1.o lzo/lzo1c_2.o lzo/lzo1c_3.o
LZO_FILES += lzo/lzo1c_4.o lzo/lzo1c_5.o lzo/lzo1c_6.o lzo/lzo1c_7.o lzo/lzo1c_8.o lzo/lzo1c_9.o lzo/lzo1c_99.o
LZO_FILES += lzo/lzo1c_9x.o lzo/lzo1c_cc.o lzo/lzo1c_d1.o lzo/lzo1c_d2.o lzo/lzo1c_rr.o lzo/lzo1c_xx.o lzo/lzo1f_1.o
LZO_FILES += lzo/lzo1f_9x.o lzo/lzo1f_d1.o lzo/lzo1f_d2.o lzo/lzo1x_1.o lzo/lzo1x_1k.o lzo/lzo1x_1l.o lzo/lzo1x_1o.o
LZO_FILES += lzo/lzo1x_9x.o lzo/lzo1x_d1.o lzo/lzo1x_d2.o lzo/lzo1x_d3.o lzo/lzo1x_o.o lzo/lzo1y_1.o lzo/lzo1y_9x.o
LZO_FILES += lzo/lzo1y_d1.o lzo/lzo1y_d2.o lzo/lzo1y_d3.o lzo/lzo1y_o.o lzo/lzo1z_9x.o lzo/lzo1z_d1.o lzo/lzo1z_d2.o
LZO_FILES += lzo/lzo1z_d3.o lzo/lzo1_99.o lzo/lzo2a_9x.o lzo/lzo2a_d1.o lzo/lzo2a_d2.o lzo/lzo_crc.o lzo/lzo_init.o
LZO_FILES += lzo/lzo_ptr.o lzo/lzo_str.o lzo/lzo_util.o

UCL_FILES = ucl/alloc.o ucl/n2b_99.o ucl/n2b_d.o ucl/n2b_ds.o ucl/n2b_to.o ucl/n2d_99.o ucl/n2d_d.o ucl/n2d_ds.o
UCL_FILES += ucl/n2d_to.o ucl/n2e_99.o ucl/n2e_d.o ucl/n2e_ds.o ucl/n2e_to.o ucl/ucl_crc.o ucl/ucl_init.o
UCL_FILES += ucl/ucl_ptr.o ucl/ucl_str.o ucl/ucl_util.o

ZLIB_FILES = zlib/adler32.o zlib/compress.o zlib/crc32.o zlib/deflate.o zlib/gzclose.o zlib/gzlib.o zlib/gzread.o
ZLIB_FILES += zlib/gzwrite.o zlib/infback.o zlib/inffast.o zlib/inflate.o zlib/inftrees.o zlib/trees.o
ZLIB_FILES += zlib/uncompr.o zlib/zutil.o

LZMAT_FILES = lzmat/lzmat_dec.o lzmat/lzmat_enc.o

LZRW_FILES = lzrw/lzrw1-a.o lzrw/lzrw1.o lzrw/lzrw2.o lzrw/lzrw3.o lzrw/lzrw3-a.o

LZMA_FILES = lzma/LzFind.o lzma/LzmaDec.o lzma/LzmaEnc.o

LZ4_FILES = lz4/lz4.o lz4/lz4hc.o

LZF_FILES = lzf/lzf_c_ultra.o lzf/lzf_c_very.o lzf/lzf_d.o

LZFSE_FILES = lzfse/lzfse_decode.o lzfse/lzfse_decode_base.o lzfse/lzfse_encode.o lzfse/lzfse_encode_base.o lzfse/lzfse_fse.o lzfse/lzvn_decode.o lzfse/lzvn_decode_base.o lzfse/lzvn_encode_base.o

QUICKLZ_FILES = quicklz/quicklz151b7.o quicklz/quicklz1.o quicklz/quicklz2.o quicklz/quicklz3.o

BROTLI_FILES = brotli/common/constants.o brotli/common/context.o brotli/common/dictionary.o brotli/common/platform.o brotli/common/transform.o
BROTLI_FILES += brotli/dec/bit_reader.o brotli/dec/decode.o brotli/dec/huffman.o brotli/dec/state.o
BROTLI_FILES += brotli/enc/backward_references.o brotli/enc/block_splitter.o brotli/enc/brotli_bit_stream.o brotli/enc/encode.o brotli/enc/encoder_dict.o
BROTLI_FILES += brotli/enc/entropy_encode.o brotli/enc/fast_log.o brotli/enc/histogram.o brotli/enc/command.o brotli/enc/literal_cost.o brotli/enc/memory.o
BROTLI_FILES += brotli/enc/metablock.o brotli/enc/static_dict.o brotli/enc/utf8_util.o brotli/enc/compress_fragment.o brotli/enc/compress_fragment_two_pass.o
BROTLI_FILES += brotli/enc/cluster.o brotli/enc/bit_cost.o brotli/enc/backward_references_hq.o brotli/enc/dictionary_hash.o

ZSTD_FILES = zstd/lib/common/zstd_common.o
ZSTD_FILES += zstd/lib/common/fse_decompress.o
ZSTD_FILES += zstd/lib/common/xxhash.o
ZSTD_FILES += zstd/lib/common/error_private.o
ZSTD_FILES += zstd/lib/common/entropy_common.o
ZSTD_FILES += zstd/lib/common/pool.o
ZSTD_FILES += zstd/lib/common/debug.o
ZSTD_FILES += zstd/lib/common/threading.o
ZSTD_FILES += zstd/lib/compress/zstd_compress.o
ZSTD_FILES += zstd/lib/compress/zstd_compress_literals.o
ZSTD_FILES += zstd/lib/compress/zstd_compress_sequences.o
ZSTD_FILES += zstd/lib/compress/zstd_compress_superblock.o
ZSTD_FILES += zstd/lib/compress/zstdmt_compress.o
ZSTD_FILES += zstd/lib/compress/zstd_double_fast.o
ZSTD_FILES += zstd/lib/compress/zstd_fast.o
ZSTD_FILES += zstd/lib/compress/zstd_lazy.o
ZSTD_FILES += zstd/lib/compress/zstd_ldm.o
ZSTD_FILES += zstd/lib/compress/zstd_opt.o
ZSTD_FILES += zstd/lib/compress/fse_compress.o
ZSTD_FILES += zstd/lib/compress/huf_compress.o
ZSTD_FILES += zstd/lib/compress/hist.o
ZSTD_FILES += zstd/lib/decompress/zstd_decompress.o
ZSTD_FILES += zstd/lib/decompress/huf_decompress.o
ZSTD_FILES += zstd/lib/decompress/zstd_ddict.o
ZSTD_FILES += zstd/lib/decompress/huf_decompress_amd64.S
ZSTD_FILES += zstd/lib/decompress/zstd_decompress_block.o
ZSTD_FILES += zstd/lib/dictBuilder/cover.o
ZSTD_FILES += zstd/lib/dictBuilder/divsufsort.o
ZSTD_FILES += zstd/lib/dictBuilder/fastcover.o
ZSTD_FILES += zstd/lib/dictBuilder/zdict.o

BRIEFLZ_FILES = brieflz/brieflz.o brieflz/depack.o brieflz/depacks.o

LIBLZG_FILES = liblzg/decode.o liblzg/encode.o liblzg/checksum.o

XZ_FILES = xz/lzma/lzma_decoder.o xz/lzma/lzma_encoder.o xz/lzma/lzma_encoder_optimum_fast.o xz/lzma/lzma_encoder_optimum_normal.o xz/lzma/fastpos_table.o
XZ_FILES += xz/lzma/lzma_encoder_presets.o xz/lz/lz_decoder.o xz/lz/lz_encoder.o xz/lz/lz_encoder_mf.o xz/common/common.o xz/rangecoder/price_table.o
XZ_FILES += xz/common/alone_encoder.o xz/common/alone_decoder.o xz/check/crc32_table.o xz/alone.o

GIPFELI_FILES = gipfeli/decompress.o gipfeli/entropy.o gipfeli/entropy_code_builder.o gipfeli/gipfeli-internal.o gipfeli/lz77.o

LIBDEFLATE_FILES = libdeflate/lib/adler32.o libdeflate/lib/utils.o libdeflate/lib/crc32.o libdeflate/lib/deflate_compress.o
LIBDEFLATE_FILES += libdeflate/lib/deflate_decompress.o libdeflate/lib/gzip_compress.o libdeflate/lib/gzip_decompress.o
LIBDEFLATE_FILES += libdeflate/lib/x86/cpu_features.o libdeflate/lib/arm/cpu_features.o libdeflate/lib/zlib_compress.o libdeflate/lib/zlib_decompress.o

MISC_FILES = crush/crush.o shrinker/shrinker.o fastlz/fastlz.o pithy/pithy.o lzjb/lzjb2010.o wflz/wfLZ.o
MISC_FILES += lzlib/lzlib.o blosclz/blosclz.o blosclz/fastcopy.o slz/slz.o

LZBENCH_FILES = _lzbench/lzbench.o _lzbench/compressors.o _lzbench/csc_codec.o

ifeq "$(DONT_BUILD_BZIP2)" "1"
    DEFINES += -DBENCH_REMOVE_BZIP2
else
    BZIP2_FILES += bzip2/blocksort.o bzip2/huffman.o bzip2/crctable.o bzip2/randtable.o bzip2/compress.o bzip2/decompress.o bzip2/bzlib.o
endif

ifeq "$(DONT_BUILD_SNAPPY)" "1"
	DEFINES += -DBENCH_REMOVE_SNAPPY
else
	SNAPPY_FILES = snappy/snappy-sinksource.o snappy/snappy-stubs-internal.o snappy/snappy.o
endif

ifeq "$(DONT_BUILD_FASTLZMA2)" "1"
	DEFINES += -DBENCH_REMOVE_FASTLZMA2
else
	FASTLZMA2_SRC = $(wildcard fast-lzma2/*.c)
	FASTLZMA2_OBJ = $(FASTLZMA2_SRC:.c=.o)
endif

ifeq "$(DONT_BUILD_XPACK)" "1"
	DEFINES += -DBENCH_REMOVE_XPACK
else
	XPACK_FILES = xpack/lib/x86_cpu_features.o xpack/lib/xpack_common.o xpack/lib/xpack_compress.o xpack/lib/xpack_decompress.o
endif

ifeq "$(DONT_BUILD_CSC)" "1"
    DEFINES += -DBENCH_REMOVE_CSC
else
	CSC_FILES = libcsc/csc_analyzer.o libcsc/csc_coder.o libcsc/csc_dec.o libcsc/csc_enc.o libcsc/csc_encoder_main.o
	CSC_FILES += libcsc/csc_filters.o libcsc/csc_lz.o libcsc/csc_memio.o libcsc/csc_mf.o libcsc/csc_model.o libcsc/csc_profiler.o libcsc/csc_default_alloc.o
endif

ifeq "$(DONT_BUILD_DENSITY)" "1"
    DEFINES += -DBENCH_REMOVE_DENSITY
else
#    DENSITY_SRC = $(shell find ./density -name '*.c')
#    DENSITY_FILES = $(DENSITY_SRC:.c=.o)
    DENSITY_FILES  = density/buffers/buffer.o
    DENSITY_FILES += density/algorithms/cheetah/core/cheetah_decode.o
    DENSITY_FILES += density/algorithms/cheetah/core/cheetah_encode.o
    DENSITY_FILES += density/algorithms/lion/forms/lion_form_model.o
    DENSITY_FILES += density/algorithms/lion/core/lion_decode.o
    DENSITY_FILES += density/algorithms/lion/core/lion_encode.o
    DENSITY_FILES += density/algorithms/dictionaries.o
    DENSITY_FILES += density/algorithms/chameleon/core/chameleon_decode.o
    DENSITY_FILES += density/algorithms/chameleon/core/chameleon_encode.o
    DENSITY_FILES += density/algorithms/algorithms.o
    DENSITY_FILES += density/structure/header.o
    DENSITY_FILES += density/globals.o
endif

ifeq "$(DONT_BUILD_GLZA)" "1"
    DEFINES += -DBENCH_REMOVE_GLZA
else
    GLZA_FILES = glza/GLZAcomp.o glza/GLZAformat.o glza/GLZAcompress.o glza/GLZAencode.o glza/GLZAdecode.o glza/GLZAmodel.o
endif

ifeq "$(DONT_BUILD_LZHAM)" "1"
    DEFINES += -DBENCH_REMOVE_LZHAM
else
    LZHAM_FILES = lzham/lzham_assert.o lzham/lzham_checksum.o lzham/lzham_huffman_codes.o lzham/lzham_lzbase.o
    LZHAM_FILES += lzham/lzham_lzcomp.o lzham/lzham_lzcomp_internal.o lzham/lzham_lzdecomp.o lzham/lzham_lzdecompbase.o
    LZHAM_FILES += lzham/lzham_match_accel.o lzham/lzham_mem.o lzham/lzham_platform.o lzham/lzham_lzcomp_state.o
    LZHAM_FILES += lzham/lzham_prefix_coding.o lzham/lzham_symbol_codec.o lzham/lzham_timer.o lzham/lzham_vector.o lzham/lzham_lib.o
endif

ifeq "$(DONT_BUILD_LZSSE)" "1"
    DEFINES += -DBENCH_REMOVE_LZSSE
else
    LZSSE_FILES = lzsse/lzsse2/lzsse2.o lzsse/lzsse4/lzsse4.o lzsse/lzsse8/lzsse8.o
endif

ifeq "$(DONT_BUILD_TORNADO)" "1"
    DEFINES += "-DBENCH_REMOVE_TORNADO"
    LZMA_FILES += lzma/Alloc.o
else
    MISC_FILES += tornado/tor_test.o
endif

ifeq "$(DONT_BUILD_YAPPY)" "1"
    DEFINES += -DBENCH_REMOVE_YAPPY
else
    MISC_FILES += yappy/yappy.o
endif

detected_OS := $(shell uname)

ifeq "$(DONT_BUILD_ZLING)" "1"
    DEFINES += -DBENCH_REMOVE_ZLING
else
    ZLING_FILES = libzling/libzling.o libzling/libzling_huffman.o libzling/libzling_lz.o libzling/libzling_utils.o
ifeq ($(detected_OS), Darwin)
    CFLAGS += -std=c++14
endif
endif

ifeq "$(BENCH_HAS_NAKAMICHI)" "1"
    DEFINES += -DBENCH_HAS_NAKAMICHI
    MISC_FILES += nakamichi/Nakamichi_Okamigan.o
endif

CUDA_BASE ?= /usr/local/cuda
LIBCUDART=$(wildcard $(CUDA_BASE)/lib64/libcudart.so)

ifneq "$(LIBCUDART)" ""
    DEFINES += -DBENCH_HAS_CUDA -I$(CUDA_BASE)/include
    LDFLAGS += -L$(CUDA_BASE)/lib64 -lcudart -Wl,-rpath=$(CUDA_BASE)/lib64
    CUDA_COMPILER = nvcc
    CUDA_CC = $(CUDA_BASE)/bin/nvcc --compiler-bindir $(CXX)
    CUDA_ARCH = 50 60 70 80
    CUDA_CFLAGS = -x cu -std=c++14 -O3 $(foreach ARCH, $(CUDA_ARCH), --generate-code=arch=compute_$(ARCH),code=[compute_$(ARCH),sm_$(ARCH)]) --expt-extended-lambda -forward-unknown-to-host-compiler -Wno-deprecated-gpu-targets
else
    $(info CUDA Toolkit not found at $(CUDA_BASE), CUDA support will be disabled.)
    $(info Run "make CUDA_BASE=..." to use a different path.)
    CUDA_BASE =
    LIBCUDART =
endif

ifneq "$(LIBCUDART)" ""
ifneq "$(DONT_BUILD_NVCOMP)" "1"
    DEFINES += -DBENCH_HAS_NVCOMP
    NVCOMP_CPP_SRC = $(wildcard nvcomp/*.cpp)
    NVCOMP_CPP_OBJ = $(NVCOMP_CPP_SRC:%=%.o)
    NVCOMP_CU_SRC  = $(wildcard nvcomp/*.cu)
    NVCOMP_CU_OBJ  = $(NVCOMP_CU_SRC:%=%.o)
    NVCOMP_FILES   = $(NVCOMP_CU_OBJ) $(NVCOMP_CPP_OBJ)
endif
endif

all: lzbench

MKDIR = mkdir -p

# FIX for SEGFAULT on GCC 4.9+
wflz/wfLZ.o shrinker/shrinker.o lzmat/lzmat_dec.o lzmat/lzmat_enc.o lzrw/lzrw1-a.o lzrw/lzrw1.o: %.o : %.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS_O2) $< -c -o $@

pithy/pithy.o: pithy/pithy.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS_O2) $< -c -o $@

_lzbench/compressors.o: %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 $< -c -o $@

snappy/snappy-sinksource.o snappy/snappy-stubs-internal.o snappy/snappy.o: %.o : %.cc
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 $< -c -o $@

lzsse/lzsse2/lzsse2.o lzsse/lzsse4/lzsse4.o lzsse/lzsse8/lzsse8.o: %.o : %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) -std=c++0x -msse4.1 $< -c -o $@

nakamichi/Nakamichi_Okamigan.o: nakamichi/Nakamichi_Okamigan.c
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -mavx $< -c -o $@

$(NVCOMP_CU_OBJ): %.cu.o: %.cu
	@$(MKDIR) $(dir $@)
	$(CUDA_CC) $(CUDA_CFLAGS) $(CFLAGS) -c $< -o $@

$(NVCOMP_CPP_OBJ): %.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) -c $< -o $@

# disable the implicit rule for making a binary out of a single object file
%: %.o


_lzbench/lzbench.o: _lzbench/lzbench.cpp _lzbench/lzbench.h

lzbench: $(BZIP2_FILES) $(DENSITY_FILES) $(FASTLZMA2_OBJ) $(ZSTD_FILES) $(GLZA_FILES) $(LZSSE_FILES) $(LZFSE_FILES) $(XPACK_FILES) $(GIPFELI_FILES) $(XZ_FILES) $(LIBLZG_FILES) $(BRIEFLZ_FILES) $(LZF_FILES) $(LZRW_FILES) $(BROTLI_FILES) $(CSC_FILES) $(LZMA_FILES) $(ZLING_FILES) $(QUICKLZ_FILES) $(SNAPPY_FILES) $(ZLIB_FILES) $(LZHAM_FILES) $(LZO_FILES) $(UCL_FILES) $(LZMAT_FILES) $(LZ4_FILES) $(LIBDEFLATE_FILES) $(MISC_FILES) $(NVCOMP_FILES) $(LZBENCH_FILES)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo Linked GCC_VERSION=$(GCC_VERSION) CLANG_VERSION=$(CLANG_VERSION) COMPILER=$(COMPILER)

.c.o:
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) $< -std=gnu99 -c -o $@

.cc.o:
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) $< -c -o $@

.cpp.o:
	@$(MKDIR) $(dir $@)
	$(CXX) $(CFLAGS) $< -c -o $@

clean:
	rm -rf lzbench lzbench.exe *.o _lzbench/*.o bzip2/*.o fast-lzma2/*.o slz/*.o zstd/lib/*.o zstd/lib/*.a zstd/lib/common/*.o zstd/lib/compress/*.o zstd/lib/decompress/*.o zstd/lib/dictBuilder/*.o lzsse/lzsse2/*.o lzsse/lzsse4/*.o lzsse/lzsse8/*.o lzfse/*.o xpack/lib/*.o blosclz/*.o gipfeli/*.o xz/*.o xz/common/*.o xz/check/*.o xz/lzma/*.o xz/lz/*.o xz/rangecoder/*.o liblzg/*.o lzlib/*.o brieflz/*.o brotli/common/*.o brotli/enc/*.o brotli/dec/*.o libcsc/*.o wflz/*.o lzjb/*.o lzma/*.o density/buffers/*.o density/algorithms/*.o density/algorithms/cheetah/core/*.o density/algorithms/*.o density/algorithms/lion/forms/*.o density/algorithms/lion/core/*.o density/algorithms/chameleon/core/*.o density/*.o density/structure/*.o pithy/*.o glza/*.o libzling/*.o yappy/*.o shrinker/*.o fastlz/*.o ucl/*.o zlib/*.o lzham/*.o lzmat/*.o lz4/*.o crush/*.o lzf/*.o lzrw/*.o lzo/*.o snappy/*.o quicklz/*.o tornado/*.o libdeflate/lib/*.o libdeflate/lib/x86/*.o libdeflate/lib/arm/*.o nakamichi/*.o nvcomp/*.o
