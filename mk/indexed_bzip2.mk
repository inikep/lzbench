# indexed_bzip2
CODECS += INDEXED_BZIP2
INDEXED_BZIP2_OBJS := bench/indexed_bzip2_codecs.o
INDEXED_BZIP2_FLAGS := -std=c++17 -DLIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION \
    -I$(SRC)bwt/librapidarchive/src \
    -I$(SRC)bwt/librapidarchive/src/external/bzip2 \
    -I$(SRC)bwt/librapidarchive/src/external/bzip2_version \
    -include $(SRC)bwt/librapidarchive/src/external/bzip2/bz2_prefix.h

# The compression path uses the librapidarchive bzip2, which has the precomputed-CRC entry point and
# the libsais forward BWT. Prefix the bzip2 and libsais symbols because the stock bzip2 and libbsc
# codecs are linked as well.
ifeq (,$(filter 1,$(DONT_BUILD_INDEXED_BZIP2)))
INDEXED_BZIP2_BZIP2_OBJS := $(addprefix bwt/librapidarchive/src/external/bzip2/, \
    blocksort.o huffman.o crctable.o randtable.o compress.o bzlib.o) \
    bwt/librapidarchive/src/external/libsais/src/libsais.o
INDEXED_BZIP2_BZIP2_FLAGS := -std=gnu99 \
    -I$(SRC)bwt/librapidarchive/src/external/bzip2 \
    -I$(SRC)bwt/librapidarchive/src/external/bzip2_version \
    -I$(SRC)bwt/librapidarchive/src/external/libsais/include \
    -include $(SRC)bwt/librapidarchive/src/external/bzip2/bz2_prefix.h
OBJ_GROUPS += INDEXED_BZIP2_BZIP2
endif
INDEXED_BZIP2_UPSTREAM := git-tags https://github.com/mxmlnkn/librapidarchive desc=indexed_bzip2

# SharedFileReader uses std::atomic<uint64_t>, which needs libatomic on 32-bit ARM.
ifneq (,$(filter arm armv%,$(TARGET_ARCH)))
    LDFLAGS += -latomic
endif
