# zxc: the *_default/_avx2/_avx512/_neon32 objects are built from the same
# sources with different ISA flags, selected at run time
CODECS += ZXC
ZXC_DIR := lz/zxc/src/lib

# bench/lz_codecs.cpp includes the zxc headers too
ifneq ($(DONT_BUILD_ZXC),1)
    DEFINES += -DZXC_STATIC_DEFINE
endif

ZXC_OBJS := $(addprefix $(ZXC_DIR)/, zxc_common.o zxc_dict.o zxc_dispatch.o zxc_driver.o \
    zxc_pivco_tables.o zxc_pstream.o zxc_seekable.o \
    zxc_compress_default.o zxc_decompress_default.o zxc_huffman_default.o)
ifneq (,$(filter x86_64% amd64%,$(TARGET_ARCH)))
    ZXC_OBJS += $(addprefix $(ZXC_DIR)/, zxc_compress_avx2.o zxc_decompress_avx2.o zxc_huffman_avx2.o \
        zxc_compress_avx512.o zxc_decompress_avx512.o zxc_huffman_avx512.o)
endif
# 32-bit ARM only (AArch64's NEON tier is _default).
ifneq (,$(filter arm% aarch64%,$(TARGET_ARCH)))
    ifeq (,$(filter arm64% aarch64%,$(TARGET_ARCH)))
        ZXC_OBJS += $(addprefix $(ZXC_DIR)/, zxc_compress_neon32.o zxc_decompress_neon32.o zxc_huffman_neon32.o)
        ZXC_NEON_FLAGS := -march=armv7-a -mfpu=neon
    endif
endif
ZXC_FLAGS = -I$(ZXC_DIR)/vendors $(ZXC_ISA_FLAGS)

$(ZXC_DIR)/%_default.o: ZXC_ISA_FLAGS := -DZXC_FUNCTION_SUFFIX=_default
$(ZXC_DIR)/%_avx2.o:    ZXC_ISA_FLAGS := -mavx2 -mbmi -mbmi2 -mlzcnt -mno-avx512f -DZXC_FUNCTION_SUFFIX=_avx2 -DZXC_USE_AVX2
$(ZXC_DIR)/%_avx512.o:  ZXC_ISA_FLAGS := -mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2 -mbmi -mbmi2 -mlzcnt -DZXC_FUNCTION_SUFFIX=_avx512 -DZXC_USE_AVX512
$(ZXC_DIR)/%_neon32.o:  ZXC_ISA_FLAGS  = $(ZXC_NEON_FLAGS) -DZXC_FUNCTION_SUFFIX=_neon32

# One rule per ISA: a pattern rule with several targets would mean that one run
# of the recipe makes all of them.
$(ZXC_DIR)/%_default.o: $(ZXC_DIR)/%.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)

$(ZXC_DIR)/%_avx2.o: $(ZXC_DIR)/%.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)

$(ZXC_DIR)/%_avx512.o: $(ZXC_DIR)/%.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)

$(ZXC_DIR)/%_neon32.o: $(ZXC_DIR)/%.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)
