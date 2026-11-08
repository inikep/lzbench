# lz4, lz4hc and lz4frame
CODECS += LZ4
LZ4_UPSTREAM := git-tags https://github.com/lz4/lz4
LZ4_OBJS  := lz/lz4/lib/lz4.o lz/lz4/lib/lz4hc.o lz/lz4/lib/lz4frame.o lz/lz4/lib/xxhash.o
# XXH_NAMESPACE: aceapex exports unprefixed XXH32/XXH64 too
LZ4_FLAGS := -std=gnu99 -DXXH_NAMESPACE=LZ4_
ifneq ($(DISABLE_THREADING),1)
    # lz4frame's built-in multithreaded compression (LZ4F_preferences_t.nbWorkers)
    LZ4_FLAGS += -DLZ4F_MULTITHREAD=1
endif
