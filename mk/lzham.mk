# lzham
CODECS += LZHAM

# lzham uses a 64 MB dictionary (m_dict_size_log2=26) and multiplies that working
# set across helper threads; on 32-bit x86 it overflows the limited address space
# and every chunk fails to compress (seen on 32-bit Windows under -T). Disable it
# for -m32 builds and on native 32-bit x86 (mingw32 etc.).
# (Pattern is i%86 -- a single '%' wildcard, matching i386/i586/i686. GNU make
# allows only one '%' per word, so the old i%86% never matched anything.)
ifeq ($(BUILD_ARCH),32-bit)
    DONT_BUILD_LZHAM ?= 1
endif
ifneq (,$(filter i%86,$(TARGET_ARCH)))
    DONT_BUILD_LZHAM ?= 1
endif
ifeq ($(detected_OS),Darwin)
    DONT_BUILD_LZHAM ?= 1
endif

LZHAM_OBJS := $(addprefix lz/lzham/, \
    lzhamdecomp/lzham_assert.o lzhamdecomp/lzham_checksum.o lzhamdecomp/lzham_huffman_codes.o \
    lzhamdecomp/lzham_lzdecomp.o lzhamdecomp/lzham_lzdecompbase.o lzhamdecomp/lzham_mem.o \
    lzhamdecomp/lzham_platform.o lzhamdecomp/lzham_prefix_coding.o lzhamdecomp/lzham_timer.o \
    lzhamdecomp/lzham_symbol_codec.o lzhamdecomp/lzham_vector.o lzhamlib/lzham_lib.o \
    lzhamcomp/lzham_lzbase.o lzhamcomp/lzham_lzcomp.o lzhamcomp/lzham_lzcomp_internal.o \
    lzhamcomp/lzham_lzcomp_state.o lzhamcomp/lzham_match_accel.o)
LZHAM_FLAGS := -Ilz/lzham/include -Ilz/lzham/lzhamcomp -Ilz/lzham/lzhamdecomp

ifneq ($(DISABLE_THREADING),1)
    ifeq ($(THREAD_MODEL),win32)
        LZHAM_OBJS += lz/lzham/lzhamcomp/lzham_win32_threading.o
    else
        LZHAM_OBJS  += lz/lzham/lzhamcomp/lzham_pthreads_threading.o
        LZHAM_FLAGS += -DTHREAD_MODEL_POSIX
    endif
endif
