# indexed_bzip2
CODECS += INDEXED_BZIP2
INDEXED_BZIP2_OBJS := bench/indexed_bzip2_codecs.o
INDEXED_BZIP2_FLAGS := -std=c++17 -DLIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION \
    -I$(SRC)bwt/librapidarchive -I$(SRC)bwt/bzip2
INDEXED_BZIP2_NEEDS := BZIP2
INDEXED_BZIP2_UPSTREAM := git-tags https://github.com/mxmlnkn/librapidarchive

# SharedFileReader uses std::atomic<uint64_t>, which needs libatomic on 32-bit ARM.
ifneq (,$(filter arm armv%,$(TARGET_ARCH)))
    LDFLAGS += -latomic
endif
