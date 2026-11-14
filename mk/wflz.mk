# wflz (buggy: may crash)
CODECS += WFLZ
WFLZ_UPSTREAM := git-head https://github.com/ShaneYCG/wflz
WFLZ_OBJS := lz/wflz/wfLZ.o
# -O2: segfaults when built with -O3 by GCC 4.9+
WFLZ_OPT  := O2
