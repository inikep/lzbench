# slz: compressor (slz) and decompressor (uslz)
CODECS += SLZ
SLZ_UPSTREAM := git-tags http://git.1wt.eu/git/libslz.git desc=slz_deflate prerelease
SLZ_OBJS  := lz+entropy/slz/src/slz.o lz+entropy/slz/src/uslz.o lz+entropy/slz/src/slz_common.o
SLZ_FLAGS := -std=gnu99
