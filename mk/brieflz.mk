# brieflz
CODECS += BRIEFLZ
BRIEFLZ_OBJS := $(addprefix lz/brieflz/, \
    brieflz.o depack.o depacks.o)
BRIEFLZ_FLAGS := -std=gnu99
