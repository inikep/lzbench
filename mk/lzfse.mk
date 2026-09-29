# lzfse and lzvn
CODECS += LZFSE
LZFSE_OBJS := $(addprefix lz+entropy/lzfse/, \
    lzfse_decode.o lzfse_decode_base.o lzfse_encode.o lzfse_encode_base.o lzfse_fse.o lzvn_decode.o \
    lzvn_decode_base.o lzvn_encode_base.o)
LZFSE_FLAGS := -std=gnu99
