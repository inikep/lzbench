# lzlib
CODECS += LZLIB
LZLIB_UPSTREAM := http https://download.savannah.gnu.org/releases/lzip/lzlib/ pattern=lzlib-(\d[\d.]*)\.tar
LZLIB_OBJS  := lz+entropy/lzlib/lzlib.o
LZLIB_FLAGS := -std=gnu99
