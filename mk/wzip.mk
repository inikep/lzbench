# wzip
CODECS += WZIP
WZIP_OBJS := $(addprefix lz+entropy/wzip/, \
    WZIP_L.o WZIP_M.o WZIP_wrapper.o Huffman_Compress.o Huffman_Decompress.o)
WZIP_OPT := O2

# levels 7-13 find matches in threads of their own with -I# (the output is that of one thread)
ifneq ($(DISABLE_THREADING),1)
    WZIP_FLAGS := -DWZIP_MULTITHREAD=1
endif
