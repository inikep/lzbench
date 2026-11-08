# zling (libzling)
CODECS += ZLING
ZLING_UPSTREAM := git-head https://github.com/richox/libzling

# zling doesn't work on big-endian PowerPC
ifeq ($(UNAME_P),powerpc)
    DONT_BUILD_ZLING ?= 1
endif

ZLING_OBJS := $(addprefix lz+entropy/libzling/, libzling.o libzling_huffman.o libzling_lz.o libzling_utils.o)
