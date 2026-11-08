# fast-lzma2
CODECS += FASTLZMA2
FASTLZMA2_UPSTREAM := git-tags https://github.com/conor42/fast-lzma2
FASTLZMA2_OBJS  := $(patsubst $(SRC)%.c,%.o,$(wildcard $(SRC)lz+entropy/fast-lzma2/*.c))
FASTLZMA2_FLAGS := -DNO_XXHASH
ifeq ($(DISABLE_THREADING),1)
    FASTLZMA2_FLAGS += -DFL2_SINGLETHREAD
endif
