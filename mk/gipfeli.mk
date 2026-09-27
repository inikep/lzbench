# gipfeli (buggy: may crash)
CODECS += GIPFELI
GIPFELI_OBJS := $(addprefix lz/gipfeli/, decompress.o entropy.o entropy_code_builder.o gipfeli-internal.o lz77.o)
# -O2: segfaults when built with -O3 by GCC 4.9+
GIPFELI_OPT  := O2
