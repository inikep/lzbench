# lzma (7-Zip LZMA SDK)
CODECS += LZMA
LZMA_UPSTREAM := git-tags https://github.com/ip7z/7zip
LZMA_OBJS := $(addprefix misc/7-zip/, \
    CpuArch.o LzFind.o LzFindOpt.o LzFindMt.o LzmaDec.o LzmaEnc.o Threads.o 7zStream.o Alloc.o \
    Lzma2Dec.o Lzma2DecMt.o Lzma2Enc.o MtCoder.o MtDec.o)
LZMA_FLAGS := -std=gnu99
