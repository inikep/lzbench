// Copyright Advanced Micro Devices, Inc.

#include "codecs.h"
#include <stdint.h>
#include <stdlib.h>
#include "aocl_compression.h"
/* AOCL-Compression (AMD optimized) wrappers.
 * Codec setup is done once.
 * The handle is reused for each benchmark call. */
typedef struct {
    aocl_compression_desc desc;
    aocl_compression_type codec;
 } aocl_bench_t;

static char* lzbench_aocl_init_generic(aocl_compression_type codec, size_t insize, size_t level)
{
    aocl_bench_t* b = (aocl_bench_t*) calloc(1, sizeof(aocl_bench_t));
    if (!b) return NULL;

    b->codec = codec;
    b->desc.inSize       = insize;   // worst-case chunk size; sizes workBuf
    b->desc.level        = level;
    b->desc.numThreads   = 1;        // unused by these codecs' API path
    b->desc.optOff       = 0;        // keep AMD optimizations enabled
    b->desc.optVar       = 0;        // matches codec_list[].extra_param
    b->desc.measureStats = 0;        // lzbench performs its own timing

    // Only zstd requires an allocated work buffer; other codecs return NULL on success.
    int setup_result = aocl_llc_setup(&b->desc, codec);
    if (setup_result != 0 || (codec == ZSTD && !b->desc.workBuf)) {
        if (setup_result == 0) aocl_llc_destroy(&b->desc, codec);
        free(b);
        return NULL;
    }
    return (char*) b;
}

static void lzbench_aocl_deinit_generic(char* workmem)
{
    aocl_bench_t* b = (aocl_bench_t*) workmem;
    if (!b) return;
    aocl_llc_destroy(&b->desc, b->codec);
    free(b);
}

static int64_t lzbench_aocl_compress_generic(char *inbuf, size_t insize,
    char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    aocl_bench_t* b = (aocl_bench_t*) codec_options->work_mem;
    if (!b) return 0;

    // Only the per-call buffers change; the handle (setup + workBuf) is reused.
    b->desc.inBuf   = inbuf;
    b->desc.outBuf  = outbuf;
    b->desc.inSize  = insize;
    b->desc.outSize = outsize;

    int64_t res = aocl_llc_compress(&b->desc, b->codec);
    return (res > 0) ? res : 0;
}

static int64_t lzbench_aocl_decompress_generic(char *inbuf, size_t insize,
    char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    aocl_bench_t* b = (aocl_bench_t*) codec_options->work_mem;
    if (!b) return 0;

    b->desc.inBuf   = inbuf;
    b->desc.outBuf  = outbuf;
    b->desc.inSize  = insize;
    b->desc.outSize = outsize;

    int64_t res = aocl_llc_decompress(&b->desc, b->codec);
    return (res > 0) ? res : 0;
}

#define LZBENCH_AOCL_COMPRESSION(name, codec)                                    \
    char* lzbench_aocl_##name##_init(size_t insize, size_t level, size_t)        \
    {                                                                            \
        return lzbench_aocl_init_generic(codec, insize, level);                  \
    }                                                                            \
    void lzbench_aocl_##name##_deinit(char* workmem)                             \
    {                                                                            \
        lzbench_aocl_deinit_generic(workmem);                                    \
    }                                                                            \
    int64_t lzbench_aocl_##name##_compress(char *inbuf, size_t insize,           \
        char *outbuf, size_t outsize, codec_options_t *codec_options)            \
    {                                                                            \
        return lzbench_aocl_compress_generic(inbuf, insize,                      \
                                             outbuf, outsize, codec_options);    \
    }                                                                            \
    int64_t lzbench_aocl_##name##_decompress(char *inbuf, size_t insize,         \
        char *outbuf, size_t outsize, codec_options_t *codec_options)            \
    {                                                                            \
        return lzbench_aocl_decompress_generic(inbuf, insize,                    \
                                               outbuf, outsize, codec_options);  \
    }

LZBENCH_AOCL_COMPRESSION(lz4,    LZ4)
LZBENCH_AOCL_COMPRESSION(lz4hc,  LZ4HC)
LZBENCH_AOCL_COMPRESSION(lzma,   LZMA)
LZBENCH_AOCL_COMPRESSION(bzip2,  BZIP2)
LZBENCH_AOCL_COMPRESSION(snappy, SNAPPY)
LZBENCH_AOCL_COMPRESSION(zlib,   ZLIB)
LZBENCH_AOCL_COMPRESSION(zstd,   ZSTD)

#undef LZBENCH_AOCL_COMPRESSION
