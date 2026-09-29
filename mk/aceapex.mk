# aceapex
CODECS += ACEAPEX
ACEAPEX_OBJS  := lz+entropy/aceapex/aceapex_lzbench.o
ACEAPEX_FLAGS := -I$(SRC)lz+entropy/aceapex -I$(SRC)bench

# Optional CUDA decoder for the aceapex format (make ENABLE_CUDA=1). It is not
# switched off by DONT_BUILD_ACEAPEX.
ifeq ($(HAVE_CUDA),1)
    OBJ_GROUPS += ACEAPEX_CUDA
    ACEAPEX_CUDA_OBJS := lz+entropy/aceapex/cuda/aceapex_cuda.cu.o lz+entropy/aceapex/cuda/aceapex_cuda_lzbench.o
endif
