# quicklz: quicklz151b7.c is compiled once per compression level
CODECS += QUICKLZ
QUICKLZ_OBJS  := lz/quicklz/quicklz_lvl1.o lz/quicklz/quicklz_lvl2.o lz/quicklz/quicklz_lvl3.o
QUICKLZ_FLAGS  = -DQLZ_COMPRESSION_LEVEL=$*

lz/quicklz/quicklz_lvl%.o: lz/quicklz/quicklz151b7.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)
