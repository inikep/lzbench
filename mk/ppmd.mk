# ppmd8 (7-Zip)
CODECS += PPMD
PPMD_UPSTREAM := git-tags https://github.com/ip7z/7zip desc=ppmd8
PPMD_OBJS  := misc/7-zip/Ppmd8.o misc/7-zip/Ppmd8Dec.o misc/7-zip/Ppmd8Enc.o
PPMD_FLAGS := -std=gnu99
