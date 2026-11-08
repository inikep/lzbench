# bzip2
CODECS += BZIP2
BZIP2_UPSTREAM := git-tags https://sourceware.org/git/bzip2.git pattern=bzip2-(\d+(?:\.\d+)*)
BZIP2_OBJS := $(addprefix bwt/bzip2/, \
    blocksort.o huffman.o crctable.o randtable.o compress.o decompress.o bzlib.o)
BZIP2_FLAGS := -std=gnu99
