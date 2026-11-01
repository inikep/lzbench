# lzfse and lzvn
CODECS += LZFSE

# lzfse and lzvn assume a little-endian CPU: their round trip fails on s390x
ifeq ($(TARGET_BIG_ENDIAN),1)
    DONT_BUILD_LZFSE ?= 1
endif

LZFSE_OBJS := $(addprefix lz+entropy/lzfse/, \
    lzfse_decode.o lzfse_decode_base.o lzfse_encode.o lzfse_encode_base.o lzfse_fse.o lzvn_decode.o \
    lzvn_decode_base.o lzvn_encode_base.o)
LZFSE_FLAGS := -std=gnu99
