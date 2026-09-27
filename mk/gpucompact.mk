# GPUCompact: CUDA only (make ENABLE_CUDA=1)
ifeq ($(HAVE_CUDA),1)
    CODECS += GPUCOMPACT
    GPUCOMPACT_OBJS := $(addprefix lz/gpucompact/, kernels.cu.o context.cu.o gpucompact_lzbench.o)
endif
