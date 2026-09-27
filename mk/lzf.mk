# lzf
CODECS += LZF
LZF_OBJS  := $(addprefix lz/lzf/, lzf_c_ultra.o lzf_c_very.o lzf_d.o)
LZF_FLAGS := -std=gnu99
