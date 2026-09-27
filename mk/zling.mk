# zling (libzling)
CODECS += ZLING

# zling doesn't work on big-endian PowerPC
ifeq ($(UNAME_P),powerpc)
    DONT_BUILD_ZLING ?= 1
endif

ZLING_OBJS := $(addprefix lz/libzling/, libzling.o libzling_huffman.o libzling_lz.o libzling_utils.o)
