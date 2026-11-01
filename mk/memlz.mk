# memlz: header-only, compiled into bench/lz_codecs.cpp
CODECS += MEMLZ

# memlz assumes a little-endian CPU: its round trip fails on s390x
ifeq ($(TARGET_BIG_ENDIAN),1)
    DONT_BUILD_MEMLZ ?= 1
endif
