#ifndef LZBENCH_H
#define LZBENCH_H

#include "compressors.h"

typedef struct
{
    char* name;
    char* version;
    compress_func compress;
    compress_func decompress;
} compressor_desc_t;

static const compressor_desc_t comp_desc[2] =
{
    { "brotli", "2015-10-29", lzbench_brotli_compress, lzbench_brotli_decompress },
    { "brotli", "2015-10-29", lzbench_brotli_compress, lzbench_brotli_decompress }
}

#endif