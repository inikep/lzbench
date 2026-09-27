# csc (buggy: may crash)
CODECS += CSC

ifeq ($(detected_OS),Darwin)
    DONT_BUILD_CSC ?= 1
endif

CSC_OBJS := $(addprefix lz/libcsc/, \
    csc_analyzer.o csc_coder.o csc_dec.o csc_enc.o csc_encoder_main.o csc_filters.o csc_lz.o \
    csc_memio.o csc_mf.o csc_model.o csc_profiler.o csc_default_alloc.o)
CSC_FLAGS := -Ilz/libcsc
# -O2: segfaults when built with -O3 by GCC 4.9+
CSC_OPT   := O2
