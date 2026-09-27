# nvcomp: CUDA only (make ENABLE_CUDA=1). Enabled with BENCH_HAS_NVCOMP rather
# than disabled with BENCH_REMOVE_NVCOMP, so it is an object group, not a codec.
ifeq ($(HAVE_CUDA),1)
ifneq ($(DONT_BUILD_NVCOMP),1)
    DEFINES    += -DBENCH_HAS_NVCOMP
    OBJ_GROUPS += NVCOMP
    NVCOMP_OBJS  := $(addsuffix .o,$(wildcard misc/nvcomp/src/*.cu misc/nvcomp/src/lowlevel/*.cu \
                                              misc/nvcomp/src/*.cpp misc/nvcomp/src/lowlevel/*.cpp))
    NVCOMP_FLAGS := -Imisc/nvcomp/include -Imisc/nvcomp/src -Imisc/nvcomp/src/lowlevel
    LDFLAGS_LIBDL = $(LIBDL)
endif
endif
