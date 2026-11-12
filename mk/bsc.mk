# bsc (libbsc)
CODECS += BSC
BSC_UPSTREAM := git-tags https://github.com/IlyaGrebnov/libbsc desc=bsc1

# bsc needs a little-endian CPU (3.4.0 has an #error for big-endian targets)
ifeq ($(TARGET_BIG_ENDIAN),1)
    DONT_BUILD_BSC ?= 1
endif

# On 32-bit ARM (armv5/v7) bsc crashes in its multithreaded decompress path
# (lzbench#293).
ifneq (,$(filter arm armeb armv%,$(TARGET_ARCH)))
    DONT_BUILD_BSC ?= 1
endif

BSC_OBJS := bwt/libbsc/libbsc/bwt/libsais/libsais.o
BSC_OBJS += $(addprefix bwt/libbsc/libbsc/, \
    adler32/adler32.o bwt/bwt.o coder/coder.o coder/qlfc/qlfc.o coder/qlfc/qlfc_model.o \
    filters/detectors.o filters/preprocessing.o libbsc/libbsc.o lzp/lzp.o platform/platform.o \
    st/st.o)

ifeq ($(HAVE_OPENMP),1)
    BSC_FLAGS += -fopenmp -DLIBBSC_OPENMP_SUPPORT -DLIBSAIS_OPENMP
endif

ifeq ($(HAVE_CUDA),1)
    BSC_FLAGS += -DLIBBSC_CUDA_SUPPORT
    BSC_OBJS  += bwt/libbsc/libbsc/bwt/libcubwt/libcubwt.cu.o bwt/libbsc/libbsc/st/st.cu.o
endif

ifneq ($(DONT_BUILD_BSC),1)
    LDFLAGS_LIBDL = $(LIBDL)
endif
