#BUILD_SYSTEM = linux
BUILD_ARCH = 64-bit

ifeq ($(BUILD_SYSTEM),linux)
	DEFINES += -DFREEARC_UNIX
	LDFLAGS = -lrt
	GCC		= gcc
	GPP		= g++
else
#	DEFINES = -march=core2 -march=nocona -march=k8 -march=native
	DEFINES += -DFREEARC_WIN
	LDFLAGS	= -lshell32 -lole32 -loleaut32 
	GCC		= gcc.exe
	GPP		= g++.exe
endif


ifeq ($(BUILD_ARCH),64-bit)
	DEFINES	+= -D__x86_64__ -D__SSE2__
	LDFLAGS	+= -static  -L C:\Aplikacje\win-builds64\lib
else
	LDFLAGS	+= -static  -L C:\Aplikacje\win-builds32\lib
endif


#DEFINES		+= -DBENCH_REMOVE_XXX
DEFINES		+= -I. -DFREEARC_INTEL_BYTE_ORDER -D_UNICODE -DUNICODE
CODE_FLAGS  = -Wno-unknown-pragmas -Wno-sign-compare -Wno-conversion
OPT_FLAGS   = -fomit-frame-pointer -fstrict-aliasing -fforce-addr -ffast-math

#BUILD_TYPE = debug

ifeq ($(BUILD_TYPE),debug)
	OPT_FLAGS += -g
else
	OPT_FLAGS += -O3 -DNDEBUG
endif

CFLAGS = $(CODE_FLAGS) $(OPT_FLAGS) $(DEFINES)


all: lzbench


ZLING_FILES = libzling/libzling.o libzling/libzling_huffman.o libzling/libzling_lz.o libzling/libzling_utils.o

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

LZHAM_FILES = lzham/lzham_assert.o lzham/lzham_checksum.o lzham/lzham_huffman_codes.o lzham/lzham_lzbase.cpp
LZHAM_FILES += lzham/lzham_lzcomp.o lzham/lzham_lzcomp_internal.o lzham/lzham_lzdecomp.o lzham/lzham_lzdecompbase.o
LZHAM_FILES += lzham/lzham_match_accel.o lzham/lzham_mem.o lzham/lzham_platform.o lzham/lzham_lzcomp_state.o
LZHAM_FILES += lzham/lzham_prefix_coding.o lzham/lzham_symbol_codec.o lzham/lzham_timer.o lzham/lzham_vector.o lzham/lzham_lib.o

ZLIB_FILES = zlib/adler32.o zlib/compress.o zlib/crc32.o zlib/deflate.o zlib/gzclose.o zlib/gzlib.o zlib/gzread.o
ZLIB_FILES += zlib/gzwrite.o zlib/infback.o zlib/inffast.o zlib/inflate.o zlib/inftrees.o zlib/trees.o
ZLIB_FILES += zlib/uncompr.o zlib/zutil.o

LZMAT_FILES = lzmat/lzmat_dec.o lzmat/lzmat_enc.o 

LZRW_FILES = lzrw/lzrw1-a.o lzrw/lzrw1.o lzrw/lzrw2.o lzrw/lzrw3.o lzrw/lzrw3-a.o

LZMA_FILES = lzma/LzFind.o lzma/LzmaDec.o lzma/LzmaEnc.o

LZ4_FILES = lz5/lz5.o lz5/lz5hc.o lz4/lz4.o lz4/lz4hc.o 

LZF_FILES = lzf/lzf_c_ultra.o lzf/lzf_c_very.o lzf/lzf_d.o

QUICKLZ_FILES = quicklz/quicklz151b7.o quicklz/quicklz1.o quicklz/quicklz2.o quicklz/quicklz3.o

DENSITY_FILES = density/block_decode.o density/block_encode.o density/block_footer.o density/block_header.o density/block_mode_marker.o
DENSITY_FILES += density/buffer.o density/globals.o density/kernel_chameleon_decode.o density/kernel_chameleon_dictionary.o
DENSITY_FILES += density/kernel_chameleon_encode.o density/kernel_cheetah_decode.o density/kernel_cheetah_dictionary.o
DENSITY_FILES += density/kernel_cheetah_encode.o density/kernel_lion_decode.o density/kernel_lion_dictionary.o
DENSITY_FILES += density/kernel_lion_encode.o density/kernel_lion_form_model.o density/main_decode.o density/main_encode.o
DENSITY_FILES += density/main_footer.o density/main_header.o density/memory_location.o density/memory_teleport.o density/stream.o
DENSITY_FILES += density/spookyhash/spookyhash.o density/spookyhash/context.o

SNAPPY_FILES = snappy/snappy-sinksource.o snappy/snappy-stubs-internal.o snappy/snappy.o 

CSC_FILES = libcsc/csc_analyzer.o libcsc/csc_coder.o libcsc/csc_dec.o libcsc/csc_enc.o libcsc/csc_encoder_main.o
CSC_FILES += libcsc/csc_filters.o libcsc/csc_lz.o libcsc/csc_memio.o libcsc/csc_mf.o libcsc/csc_model.o libcsc/csc_profiler.o 

BROTLI_FILES = brotli/dec/bit_reader.o brotli/dec/decode.o brotli/dec/dictionary.o brotli/dec/huffman.o brotli/dec/state.o brotli/dec/streams.o
BROTLI_FILES += brotli/enc/backward_references.o brotli/enc/block_splitter.o brotli/enc/brotli_bit_stream.o brotli/enc/encode.o
BROTLI_FILES += brotli/enc/encode_parallel.o brotli/enc/entropy_encode.o brotli/enc/histogram.o brotli/enc/literal_cost.o
BROTLI_FILES += brotli/enc/metablock.o brotli/enc/static_dict.o brotli/enc/streams.o brotli/enc/utf8_util.o

ZSTD_FILES = zstd/fse.o zstd/huff0.o zstd/zstd.o zstd/zstdhc.o 

LZLIB_FILES = lzlib/lzlib.o

BRIEFLZ_FILES = brieflz/brieflz.o brieflz/depacks.o 

LIBLZG_FILES = liblzg/decode.o liblzg/encode.o liblzg/checksum.o 

MISC_FILES = crush/crush.o shrinker/shrinker.o yappy/yappy.o fastlz/fastlz.o tornado/tor_test.o pithy/pithy.o lzjb/lzjb2010.o wflz/wfLZ.o

_lzbench/lzbench.o: _lzbench/lzbench.h

lzbench: $(LIBLZG_FILES) $(BRIEFLZ_FILES) $(LZLIB_FILES) $(LZF_FILES) $(LZRW_FILES) $(ZSTD_FILES) $(BROTLI_FILES) $(CSC_FILES) $(LZMA_FILES) $(DENSITY_FILES) $(ZLING_FILES) $(QUICKLZ_FILES) $(SNAPPY_FILES) $(ZLIB_FILES) $(LZHAM_FILES) $(LZO_FILES) $(UCL_FILES) $(LZMAT_FILES) $(LZ4_FILES) $(MISC_FILES) _lzbench/lzbench.o _lzbench/compressors.o
	$(GPP) $^ -o $@ $(LDFLAGS)

.c.o:
	$(GCC) $(CFLAGS) $< -std=c99 -c -o $@

.cc.o:
	$(GPP) $(CFLAGS) $< -c -o $@

.cpp.o:
	$(GPP) $(CFLAGS) $< -c -o $@

clean:
	rm -f _lzbench/*.o liblzg/*.o lzlib/*.o brieflz/*.o brotli/enc/*.o brotli/dec/*.o  libcsc/*.o wflz/*.o lzjb/*.o lzma/*.o density/spookyhash/*.o density/*.o pithy/*.o zstd/*.o libzling/*.o yappy/*.o shrinker/*.o fastlz/*.o ucl/*.o zlib/*.o lzham/*.o lzmat/*.o lz5/*.o lz4/*.o crush/*.o lzf/*.o lzrw/*.o lzo/*.o snappy/*.o quicklz/*.o tornado/*.o *.o *.exe
