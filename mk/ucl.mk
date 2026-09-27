# ucl
CODECS += UCL
UCL_OBJS := $(addprefix lz/ucl/, \
    alloc.o n2b_99.o n2b_d.o n2b_ds.o n2b_to.o n2d_99.o n2d_d.o n2d_ds.o n2d_to.o n2e_99.o n2e_d.o \
    n2e_ds.o n2e_to.o ucl_crc.o ucl_init.o ucl_ptr.o ucl_str.o ucl_util.o)
UCL_FLAGS := -Ilz
