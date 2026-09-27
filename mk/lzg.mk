# lzg (liblzg)
CODECS += LZG
LZG_OBJS  := $(addprefix lz/liblzg/, decode.o encode.o checksum.o)
LZG_FLAGS := -std=gnu99
