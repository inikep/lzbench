# aceapex
CODECS += ACEAPEX
ACEAPEX_OBJS  := lz/aceapex/aceapex_lzbench.o
ACEAPEX_FLAGS := -Ilz/aceapex -Ibench

# Optional CUDA decoder for the aceapex format (make ENABLE_CUDA=1). It is not
# switched off by DONT_BUILD_ACEAPEX.
ifeq ($(HAVE_CUDA),1)
    OBJ_GROUPS += ACEAPEX_CUDA
    ACEAPEX_CUDA_OBJS := lz/aceapex/cuda/aceapex_cuda.cu.o lz/aceapex/cuda/aceapex_cuda_lzbench.o
endif
