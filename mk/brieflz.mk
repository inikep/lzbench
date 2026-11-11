# brieflz
CODECS += BRIEFLZ
BRIEFLZ_UPSTREAM := git-tags https://github.com/jibsen/brieflz pattern=v(\d+\.\d+\.\d+)
BRIEFLZ_OBJS := $(addprefix lz/brieflz/, \
    brieflz.o depack.o depacks.o)
BRIEFLZ_FLAGS := -std=gnu99
