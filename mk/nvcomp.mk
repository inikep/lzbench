# nvcomp: CUDA only (make ENABLE_CUDA=1). Enabled with BENCH_HAS_NVCOMP rather
# than disabled with BENCH_REMOVE_NVCOMP, so it is an object group, not a codec.
ifeq ($(HAVE_CUDA),1)
ifneq ($(DONT_BUILD_NVCOMP),1)
    DEFINES    += -DBENCH_HAS_NVCOMP
    OBJ_GROUPS += NVCOMP
    NVCOMP_OBJS  := $(patsubst $(SRC)%,%.o,$(wildcard $(addprefix $(SRC)misc/nvcomp/src/, \
                        *.cu lowlevel/*.cu *.cpp lowlevel/*.cpp)))
    NVCOMP_FLAGS := -I$(SRC)misc/nvcomp/include -I$(SRC)misc/nvcomp/src -I$(SRC)misc/nvcomp/src/lowlevel
    LDFLAGS_LIBDL = $(LIBDL)
endif
endif
