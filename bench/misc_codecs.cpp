/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * misc_codecs.cpp: codecs that are neither LZ, BWT nor PPM/CM (glza, skim), and cudaMemcpy
 */

#include "codecs.h"
#include <stdio.h> // FILE


#ifndef BENCH_REMOVE_GLZA
#include "misc/glza/GLZAcomp.h"
#include "misc/glza/GLZAdecode.h"

int64_t lzbench_glza_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (GLZAcomp(insize, (uint8_t *)inbuf, &outsize, (uint8_t *)outbuf, (FILE *)0, NULL) == 0) return(0);
    return outsize;
}

int64_t lzbench_glza_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (GLZAdecode(insize, (uint8_t *)inbuf, &outsize, (uint8_t *)outbuf, (FILE *)0, NULL) == 0) return(0);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_SKIM
extern "C" {
#include "misc/skim/skim.h"
}

struct lzbench_skim_state {
    skim_encoder_t* encoder;
    skim_decoder_t* decoder;
};

char* lzbench_skim_init(size_t insize, size_t level, size_t)
{
    lzbench_skim_state* state = (lzbench_skim_state*)malloc(sizeof(lzbench_skim_state));
    if (!state) return NULL;
    
    state->encoder = skim_encoder_create();
    state->decoder = skim_decoder_create();
    return (char*)state;
}

void lzbench_skim_deinit(char* workmem)
{
    lzbench_skim_state* state = (lzbench_skim_state*)workmem;
    if (state) {
        if (state->encoder) skim_encoder_destroy(state->encoder);
        if (state->decoder) skim_decoder_destroy(state->decoder);
        free(state);
    }
}

int64_t lzbench_skim_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzbench_skim_state* state = (lzbench_skim_state*)codec_options->work_mem;
    if (!state || !state->encoder) return 0;
    
    skim_encoder_reset(state->encoder);
    
    return skim_encoder_compress(state->encoder, (const uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_skim_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzbench_skim_state* state = (lzbench_skim_state*)codec_options->work_mem;
    if (!state || !state->decoder) return 0;

    skim_decoder_reset(state->decoder);
    
    size_t consumed = skim_decoder_decompress(state->decoder, (const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize);
    if (consumed == 0) return 0;

    return skim_decoder_exact_output_length((const uint8_t*)inbuf, insize);
}
#endif 

#ifdef BENCH_HAS_CUDA
#include <cuda_runtime.h>

char* lzbench_cuda_init(size_t insize, size_t, size_t)
{
    char* workmem = NULL;
    // Report failure (no CUDA device, out of device memory) so that lzbench skips
    // the codec. Without this the cudaMemcpy calls below silently do nothing and
    // the codec is reported as a decompression ERROR.
    if (cudaMalloc(& workmem, insize) != cudaSuccess) return NULL;
    return workmem;
}

void lzbench_cuda_deinit(char* workmem)
{
    cudaFree(workmem);
}

int64_t lzbench_cuda_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    cudaMemcpy(codec_options->work_mem, inbuf, insize, cudaMemcpyHostToDevice);
    cudaMemcpy(outbuf, codec_options->work_mem, insize, cudaMemcpyDeviceToHost);
    return insize;
}

#endif    // BENCH_HAS_CUDA
