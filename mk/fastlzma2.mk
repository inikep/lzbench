# fast-lzma2
CODECS += FASTLZMA2
FASTLZMA2_OBJS  := $(patsubst %.c,%.o,$(wildcard lz/fast-lzma2/*.c))
FASTLZMA2_FLAGS := -DNO_XXHASH
ifeq ($(DISABLE_THREADING),1)
    FASTLZMA2_FLAGS += -DFL2_SINGLETHREAD
endif
