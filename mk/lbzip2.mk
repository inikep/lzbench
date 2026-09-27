# lbzip2
CODECS += LBZIP2
LBZIP2_OBJS := $(addprefix bwt/lbzip2/, \
    crctab.o decode.o divbwt.o encode.o parse.o lbzip2_lzbench.o)
# -Ibwt/lbzip2 also picks up the <arpa/inet.h> shim there, which MinGW needs.
# zstd's dictBuilder exports a divbwt() too, and xmalloc() is a name anything
# might take, so rename both rather than patch the vendored source.
LBZIP2_FLAGS := -Ibwt/lbzip2 -Ddivbwt=lbzip2_divbwt -Dxmalloc=lbzip2_xmalloc
