# AOCL-Compression (AMD optimized): opt-in, x86_64 Linux only.
# Enable: make DONT_BUILD_AOCL=0
CODECS += AOCL

DONT_BUILD_AOCL ?= 1
NM      ?= nm
OBJCOPY ?= objcopy

# A command line "DONT_BUILD_AOCL=0" overrides a plain assignment here, so the
# unsupported targets use "override": opting in cannot make them build.
ifneq ($(DONT_BUILD_AOCL),1)
    ifneq ($(detected_OS),Linux)                      # non-Linux (Windows/macOS/etc.)
        override DONT_BUILD_AOCL := 1
    else ifeq ($(BUILD_ARCH),32-bit)                  # -m32 on x86_64 host: AOCL is x86_64 only
        override DONT_BUILD_AOCL := 1
    else ifeq ($(CROSS_BUILD),1)                      # skip cross-compilation
        override DONT_BUILD_AOCL := 1
    else ifeq (,$(filter x86_64%,$(TARGET_ARCH)))     # only x86_64 (AMD Zen)
        override DONT_BUILD_AOCL := 1
    else ifeq (,$(shell command -v $(OBJCOPY) 2>/dev/null))  # need binutils objcopy
        override DONT_BUILD_AOCL := 1
    else ifeq (,$(shell command -v $(NM) 2>/dev/null))       # need binutils nm
        override DONT_BUILD_AOCL := 1
    else ifeq (,$(wildcard $(SRC)lz/aocl-compression/api/api.cpp))  # AOCL source must be present
        override DONT_BUILD_AOCL := 1
        $(info AOCL-Compression source not found at lz/aocl-compression - skipping AOCL build)
    endif
endif

ifneq ($(DONT_BUILD_AOCL),1)

AOCL_DIR = lz/aocl-compression
AOCL_SRC = $(SRC)lz/aocl-compression
AOCL_GEN = $(AOCL_SRC)/generated

AOCL_BZIP2_OBJ = $(addprefix $(AOCL_DIR)/algos/bzip2/, \
    blocksort.o bzlib.o code_alignment.o compress.o crctable.o decompress.o \
    huffman.o libsais.o randtable.o)
AOCL_LZ4_OBJ = $(addprefix $(AOCL_DIR)/algos/lz4/, \
    code_alignment_lz4.o code_alignment_lz4hc.o lz4.o lz4hc.o xxhash.o)
AOCL_LZMA_OBJ = $(addprefix $(AOCL_DIR)/algos/lzma/, \
    Alloc.o code_alignment.o LzFind.o LzmaDec.o LzmaEnc.o)
AOCL_ZLIB_OBJ = $(addprefix $(AOCL_DIR)/algos/zlib/, \
    adler32.o adler32_x86.o aocl_zlib_utils.o code_alignment.o compress.o \
    crc32.o crc32_x86_pclmul.o crc32_x86_vpclmulqdq.o deflate.o deflate_medium.o \
    deflate_quick.o gzclose.o gzlib.o gzread.o gzwrite.o infback.o inffast.o \
    inflate.o inftrees.o longest_match_x86.o slide_hash_x86.o trees.o uncompr.o \
    zlib-ng/arch/x86/chunkset_avx512.o zlib-ng/arch/x86/chunkset_sse2.o zutil.o)
AOCL_ZSTD_OBJ = $(addprefix $(AOCL_DIR)/algos/zstd/, \
    lib/common/code_alignment.o lib/common/debug.o lib/common/entropy_common.o \
    lib/common/error_private.o lib/common/fse_decompress.o lib/common/pool.o \
    lib/common/threading.o lib/common/xxhash.o lib/common/zstd_common.o \
    lib/compress/fse_compress.o lib/compress/hist.o lib/compress/huf_compress.o \
    lib/compress/zstd_compress.o lib/compress/zstd_compress_literals.o \
    lib/compress/zstd_compress_sequences.o lib/compress/zstd_compress_superblock.o \
    lib/compress/zstd_double_fast.o lib/compress/zstd_fast.o lib/compress/zstd_lazy.o \
    lib/compress/zstd_ldm.o lib/compress/zstdmt_compress.o lib/compress/zstd_opt.o \
    lib/compress/zstd_preSplit.o lib/decompress/huf_decompress.o \
    lib/decompress/huf_decompress_amd64.o \
    lib/decompress/zstd_ddict.o lib/decompress/zstd_decompress_block.o \
    lib/decompress/zstd_decompress.o lib/dictBuilder/cover.o \
    lib/dictBuilder/divsufsort.o lib/dictBuilder/fastcover.o lib/dictBuilder/zdict.o)
AOCL_SNAPPY_OBJ = $(addprefix $(AOCL_DIR)/algos/snappy/, \
    code_alignment.o snappy.o snappy-c.o snappy-sinksource.o snappy-stubs-internal.o)
AOCL_API_OBJ = $(addprefix $(AOCL_DIR)/api/, api.o code_alignment.o codec.o)
AOCL_UTILS_OBJ = $(addprefix $(AOCL_DIR)/utils/, \
    code_alignment.o cpu_features.o dispatcher.o utils.o)

AOCL_S_OBJ   = $(AOCL_DIR)/algos/zstd/lib/decompress/huf_decompress_amd64.o
AOCL_C_OBJ   = $(AOCL_BZIP2_OBJ) $(AOCL_LZ4_OBJ) $(AOCL_LZMA_OBJ) $(AOCL_ZLIB_OBJ) $(filter-out $(AOCL_S_OBJ),$(AOCL_ZSTD_OBJ))
AOCL_CC_OBJ  = $(AOCL_SNAPPY_OBJ)
AOCL_CPP_OBJ = $(AOCL_API_OBJ) $(AOCL_UTILS_OBJ)
AOCL_LIB_OBJ = $(AOCL_C_OBJ) $(AOCL_CC_OBJ) $(AOCL_CPP_OBJ) $(AOCL_S_OBJ)

AOCL_RAW     = $(AOCL_DIR)/libaocl_compression.a
AOCL_ARCHIVE = $(AOCL_DIR)/libaocl_lzbench.a

# The archive is linked after bench/aocl_codecs.o, which calls into it: the
# Makefile puts the %.a of a codec last.
AOCL_OBJS   := bench/aocl_codecs.o $(AOCL_ARCHIVE)
AOCL_FLAGS  := -I$(AOCL_SRC)/api
CLEAN_FILES += $(AOCL_RAW) $(AOCL_ARCHIVE)

# Common flags mirror AOCL 5.3's Linux Release build. -fPIC replaces AOCL's
# -fpie since these objects go into a static archive (superset, link-safe).
# -Werror (ENABLE_STRICT_WARNINGS) is intentionally kept out so third-party
# warnings on newer compilers don't fail the build.
AOCL_INC = -I$(AOCL_SRC) -I$(AOCL_GEN) -I$(AOCL_GEN)/algos/common \
           -I$(AOCL_SRC)/algos/common -I$(AOCL_SRC)/utils -I$(AOCL_SRC)/api
# BUILD_TYPE=debug builds AOCL with -O0 -g like the rest of lzbench
# (_FORTIFY_SOURCE needs optimization, so it is left out there).
ifeq ($(BUILD_TYPE),debug)
    AOCL_OPT = -O0 -g
else
    AOCL_OPT = -O3 -DNDEBUG -D_FORTIFY_SOURCE=2
endif
AOCL_COMMON_FLAGS = $(AOCL_OPT) -fomit-frame-pointer -fstrict-aliasing \
    -fstack-protector-strong -Wformat -Wformat-security \
    -Wall -Wpedantic -fPIC -DAOCL_DYNAMIC_DISPATCHER \
    -DAOCL_BUILD_VERSION='"lzbench"' -DAOCL_MAX_ISA_LEVEL=4 $(AOCL_INC)

# AOCL sources are compiled directly by lzbench with per-codec flags mirroring
# AOCL's Release build. Generated headers are vendored in $(AOCL_GEN). The
# user's MOREFLAGS and USER_CFLAGS/USER_CXXFLAGS come last, as for the other
# codecs, and DEPFLAGS writes the .d files included at the end of this file.
AOCL_USER_CFLAGS   = $(MOREFLAGS) $(USER_CFLAGS) $(DEPFLAGS)
AOCL_USER_CXXFLAGS = $(MOREFLAGS) $(USER_CXXFLAGS) $(DEPFLAGS)
$(AOCL_C_OBJ):   %.o : %.c   ; @$(MKDIR) $(dir $@); $(CC)  -std=gnu11 -D_POSIX_C_SOURCE=200809L $(AOCL_COMMON_FLAGS) $(AOCL_EXTRA) $(AOCL_ARCH) $(AOCL_USER_CFLAGS) -c $< -o $@
$(AOCL_CC_OBJ):  %.o : %.cc  ; @$(MKDIR) $(dir $@); $(CXX) $(AOCL_COMMON_FLAGS) $(AOCL_EXTRA) $(AOCL_USER_CXXFLAGS) -c $< -o $@
$(AOCL_CPP_OBJ): %.o : %.cpp ; @$(MKDIR) $(dir $@); $(CXX) $(AOCL_COMMON_FLAGS) $(AOCL_EXTRA) $(AOCL_USER_CXXFLAGS) -c $< -o $@
$(AOCL_S_OBJ):   %.o : %.S   ; @$(MKDIR) $(dir $@); $(CC)  $(AOCL_COMMON_FLAGS) $(AOCL_EXTRA) $(AOCL_ARCH) $(AOCL_USER_CFLAGS) -c $< -o $@

# Per-codec extra flags (from AOCL's own build).
$(AOCL_BZIP2_OBJ):  AOCL_EXTRA = -DAOCL_BZIP2_HUFFMAN_ITERATIONS=3
$(AOCL_LZ4_OBJ):    AOCL_EXTRA = -DAOCL_LZ4_HASH_BITS_USED=41 -DAOCL_LZ4HC_DISABLE_PATTERN_ANALYSIS
$(AOCL_ZSTD_OBJ):   AOCL_EXTRA = -DAOCL_ZSTD_SEARCH_SKIP_OPT
$(AOCL_ZLIB_OBJ):   AOCL_EXTRA = -DHAVE_BUILTIN_CTZ=1 -DHAVE_BUILTIN_EXPECT=1 -DHAVE_HIDDEN -I$(AOCL_SRC)/algos/zlib -I$(AOCL_SRC)/algos/zlib/zlib-ng
$(AOCL_SNAPPY_OBJ): AOCL_EXTRA = -msse4.1 -DHAVE_ATTRIBUTE_ALWAYS_INLINE=1 -DHAVE_BUILTIN_CTZ=1 -DHAVE_BUILTIN_EXPECT=1 -DHAVE_BUILTIN_PREFETCH=1 -DHAVE_SYS_UIO_H -DSNAPPY_HAVE_X86_CRC32=1 -DAOCL_SNAPPY_MATCH_SKIP_OPT

# AOCL defines HAVE_ATTRIBUTE_ALIGNED for the zlib-ng sources only: crc32_x86.h
# defines it itself, and a -D for all of zlib warns about the redefinition.
$(filter $(AOCL_DIR)/algos/zlib/zlib-ng/%,$(AOCL_ZLIB_OBJ)): AOCL_EXTRA += -DHAVE_ATTRIBUTE_ALIGNED=1

# Clang has no per-function optimization attributes, so AOCL builds LZ4 with -O2
# under it (LZ4_CFLAGS in AOCL's Makefile). Debug builds keep -O0.
ifeq ($(COMPILER),clang)
ifneq ($(BUILD_TYPE),debug)
$(AOCL_LZ4_OBJ):    AOCL_EXTRA += -O2
endif
endif

# Per-file ISA flags for the two runtime-dispatched zlib files.
$(AOCL_DIR)/algos/zlib/crc32_x86_vpclmulqdq.o:             AOCL_ARCH = -mavx512f -mpclmul -mvpclmulqdq
$(AOCL_DIR)/algos/zlib/zlib-ng/arch/x86/chunkset_avx512.o: AOCL_ARCH = -mavx512bw -mavx512f

# GCC 15 scheduling workarounds from AOCL 5.3.2.
ifeq (1, $(shell [ "$(COMPILER)" = "gcc" ] && expr $(GCC_VERSION) \>= 150000))
$(AOCL_DIR)/algos/zlib/zlib-ng/arch/x86/chunkset_avx512.o: AOCL_ARCH += -fno-schedule-insns2 -fno-code-hoisting
$(AOCL_DIR)/algos/zstd/lib/decompress/zstd_decompress_block.o: AOCL_ARCH += -fno-schedule-insns -fno-tree-sink -fno-if-conversion
endif

$(AOCL_RAW): $(AOCL_LIB_OBJ) ; @$(MKDIR) $(dir $@); $(AR) rcs $@ $(AOCL_LIB_OBJ)

# Rename AOCL's internal symbols to AOCLLZB_* to avoid linker collisions,
# keeping the public aocl_llc_* API. C = common symbols
$(AOCL_ARCHIVE): $(AOCL_RAW) ; @$(MKDIR) $(dir $@); $(NM) -g --defined-only $(AOCL_RAW) | awk '$$2 ~ /^[TDBRGVWC]$$/ {print $$3}' | grep -v '^aocl_llc_' | sort -u | awk 'NF{print $$1" AOCLLZB_"$$1}' > $@.redef
	$(OBJCOPY) --redefine-syms=$@.redef $(AOCL_RAW) $@
	rm -f $@.redef

# The Makefile includes the .d files of LZBENCH_OBJS only, which name the
# archive and not the objects in it.
-include $(AOCL_LIB_OBJ:.o=.d)

endif
