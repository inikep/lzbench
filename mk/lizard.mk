# lizard
CODECS += LIZARD
LIZARD_OBJS := $(addprefix lz/lizard/, \
    lizard_compress.o lizard_decompress.o entropy/huf_compress.o entropy/huf_decompress.o \
    entropy/entropy_common.o entropy/fse_compress.o entropy/fse_decompress.o entropy/hist.o)
LIZARD_OPT := O2
