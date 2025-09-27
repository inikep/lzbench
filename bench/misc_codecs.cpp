/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * misc_codecs.cpp: miscellaneous codecs not included in the "ALL" alias
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



#ifndef BENCH_REMOVE_LZJB
#include "lz/lzjb/lzjb2010.h"

int64_t lzbench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzjb_compress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
}

int64_t lzbench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzjb_decompress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
}

#endif



#ifndef BENCH_REMOVE_TAMP
#include "lz/tamp/compressor.h"
#include "lz/tamp/decompressor.h"

char* lzbench_tamp_init(size_t, size_t level, size_t)
{
    return (char*) malloc(1 << level);
}

void lzbench_tamp_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_tamp_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t compressed_size = 0;
    TampConf conf = {
       /* Describes the size of the decompression buffer in bits.
       A 10-bit window represents a 1024-byte buffer.
       Must be in range [8, 15], representing [256, 32678] byte windows. */
       .window = (uint16_t)codec_options->level,
       .literal = 8,
       .use_custom_dictionary = false
    };
    TampCompressor compressor;
    tamp_compressor_init(&compressor, &conf, (unsigned char *)codec_options->work_mem);

    tamp_compressor_compress_and_flush(
            &compressor,
            (unsigned char*) outbuf,
            outsize,
            (size_t *)&compressed_size,
            (unsigned char *)inbuf,
            insize,
            NULL,
            false
    );
    return compressed_size;
}

int64_t lzbench_tamp_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t decompressed_size = 0;
    TampConf conf;
    TampDecompressor decompressor;
    size_t compressed_consumed_size;

    tamp_decompressor_init(&decompressor, NULL, (unsigned char *)codec_options->work_mem);

    tamp_decompressor_decompress(
        &decompressor,
        (unsigned char *)outbuf,
        outsize,
        (size_t *)&decompressed_size,
        (unsigned char *)inbuf,
        insize,
        NULL
    );

    return decompressed_size;
}
#endif



#ifdef BENCH_HAS_CUDA
#include <cuda_runtime.h>
#include <algorithm> // std::min

#define CUDA_CHECK(cond)                                               \
    do {                                                               \
        int err = cond;                                                \
        if (err != nvcompSuccess) {                                    \
            fprintf(stderr, "CUDA failure at %s:%d - Error Code: %d\n",\
                    __FILE__, __LINE__, err);                          \
            return 0;                                                  \
        }                                                              \
    } while (false)

char* lzbench_cuda_init(size_t insize, size_t, size_t)
{
    char* workmem;
    cudaMalloc(& workmem, insize);
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

#ifdef BENCH_HAS_NVCOMP
#include "misc/nvcomp/include/nvcomp/lz4.h"

typedef struct {
    cudaStream_t stream;
    size_t max_out_bytes;
    size_t batch_size;

    char* device_input_data;
    void ** device_uncompressed_ptrs;
    size_t* device_uncompressed_bytes;

    char* device_output_data;
    void** device_compressed_ptrs;
    size_t *device_compressed_bytes;

    char* device_temp_ptr;
    size_t device_temp_bytes;

    void ** host_compressed_ptrs;
    size_t* host_compressed_bytes;

    void ** host_uncompressed_ptrs;
    size_t* host_uncompressed_bytes;
    nvcompLZ4FormatOpts opts;
} nvcomp_params_s;

// allocate the host and device memory buffers for the nvcom LZ4 compression and decompression
// the chunk size is configured by the compression level, 0 to 5 inclusive, corresponding to a chunk size from 32 kB to 1 MB
char* lzbench_nvcomp_init(size_t in_bytes, size_t level, size_t)
{
    // allocate the host memory for the algorithm options
    nvcomp_params_s* params = (nvcomp_params_s*) malloc(sizeof(nvcomp_params_s));
    if (!params) return NULL;

    // create a CUDA stream to run the compression/decompression
    int status = 0;
    CUDA_CHECK(cudaStreamCreate(&params->stream));

    // set the chunk size based on the compression level
    params->opts.chunk_size = 1 << (15 + level);
    params->batch_size = (in_bytes + params->opts.chunk_size - 1) / params->opts.chunk_size;

    // allocate device memory for the data to be compressed
    CUDA_CHECK(cudaMalloc(&params->device_input_data, in_bytes));

    // Setup an array of chunk sizes
    CUDA_CHECK(cudaMallocHost((void**)&params->host_uncompressed_bytes, sizeof(size_t)*params->batch_size));
    for (size_t i = 0; i < params->batch_size; ++i) {
        if (i + 1 < params->batch_size) {
            params->host_uncompressed_bytes[i] = params->opts.chunk_size;
        } else {
            // last chunk may be smaller
            params->host_uncompressed_bytes[i] = in_bytes - (params->opts.chunk_size*i);
        }
    }

    // Setup an array of pointers to the start of each chunk
    CUDA_CHECK(cudaMallocHost((void**)&params->host_uncompressed_ptrs, sizeof(size_t)*params->batch_size));
    for (size_t ix_chunk = 0; ix_chunk < params->batch_size; ++ix_chunk) {
        params->host_uncompressed_ptrs[ix_chunk] = params->device_input_data + params->opts.chunk_size*ix_chunk;
    }

    CUDA_CHECK(cudaMalloc((void**)&params->device_uncompressed_bytes, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMalloc((void**)&params->device_uncompressed_ptrs, sizeof(size_t) * params->batch_size));

    CUDA_CHECK(cudaMemcpyAsync(params->device_uncompressed_bytes, params->host_uncompressed_bytes, sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));
    CUDA_CHECK(cudaMemcpyAsync(params->device_uncompressed_ptrs, params->host_uncompressed_ptrs, sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));

    // determine the size of the temporary buffer
    CUDA_CHECK(nvcompBatchedLZ4CompressGetTempSize(params->batch_size, params->opts.chunk_size, nvcompBatchedLZ4DefaultOpts, &params->device_temp_bytes));

    // allocate device memory for the temporary buffer
    CUDA_CHECK(cudaMalloc(&params->device_temp_ptr, params->device_temp_bytes));

    // get the maxmimum output size for each chunk
    CUDA_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(params->opts.chunk_size, nvcompBatchedLZ4DefaultOpts, &params->max_out_bytes));

    // allocate device memory for the data to be compressed
    CUDA_CHECK(cudaMalloc(&params->device_output_data, params->batch_size * params->max_out_bytes));

    // Next, allocate output space on the device
    CUDA_CHECK(cudaMallocHost((void**)&params->host_compressed_bytes, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMallocHost((void**)&params->host_compressed_ptrs, sizeof(size_t) * params->batch_size));
    for(size_t ix_chunk = 0; ix_chunk < params->batch_size; ++ix_chunk) {
        params->host_compressed_ptrs[ix_chunk] = params->device_output_data + params->max_out_bytes*ix_chunk;
    }

    CUDA_CHECK(cudaMalloc((void**)&params->device_compressed_ptrs, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMemcpyAsync(
        params->device_compressed_ptrs, params->host_compressed_ptrs,
        sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));

    // allocate space for compressed chunk sizes to be written to
    CUDA_CHECK(cudaMalloc((void**)&params->device_compressed_bytes, sizeof(size_t) * params->batch_size));

    return (char*) params;
}

void lzbench_nvcomp_deinit(char* nvcomp_params)
{
    nvcomp_params_s* params = (nvcomp_params_s*) nvcomp_params;
    if (!params) return;

    // free all the device memory
    cudaFree(params->device_input_data);
    cudaFree(params->device_uncompressed_ptrs);
    cudaFree(params->device_uncompressed_bytes);
    cudaFree(params->device_output_data);
    cudaFree(params->device_compressed_ptrs);
    cudaFree(params->device_compressed_bytes);
    cudaFree(params->device_temp_ptr);
    cudaFreeHost(params->host_compressed_ptrs);
    cudaFreeHost(params->host_compressed_bytes);
    cudaFreeHost(params->host_uncompressed_ptrs);
    cudaFreeHost(params->host_uncompressed_bytes);

    // release the CUDA stream
    cudaStreamDestroy(params->stream);

    // free the host memory for the algorithm options
    free(params);
}

int64_t lzbench_nvcomp_compress(char *inbuf, size_t in_bytes, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    nvcomp_params_s* params = (nvcomp_params_s*) codec_options->work_mem;
    int status = 0;

    // copy the uncompressed data to the device
    CUDA_CHECK(cudaMemcpyAsync(params->device_input_data, inbuf, in_bytes, cudaMemcpyHostToDevice, params->stream));

#if 0
    fprintf(stderr, "COMPRESS device_uncompressed_ptrs=%p device_uncompressed_bytes=%p\n", params->device_uncompressed_ptrs, params->device_uncompressed_bytes);
    fprintf(stderr, "COMPRESS chunk_size=%ld batch_size=%ld\n", params->opts.chunk_size, params->batch_size);
    fprintf(stderr, "COMPRESS device_temp_ptr=%p device_temp_bytes=%ld\n", params->device_temp_ptr, params->device_temp_bytes);
    fprintf(stderr, "COMPRESS device_compressed_ptrs=%p device_compressed_bytes=%p\n", params->device_compressed_ptrs, params->device_compressed_bytes);
#endif

    // call the API to compress the data
    CUDA_CHECK(nvcompBatchedLZ4CompressAsync(
        params->device_uncompressed_ptrs,
        params->device_uncompressed_bytes,
        params->opts.chunk_size, // The maximum chunk size
        params->batch_size,
        params->device_temp_ptr,
        params->device_temp_bytes,
        params->device_compressed_ptrs,
        params->device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        params->stream));

    // limit the data to be copied back to the size available on the host
    size_t out_bytes = std::min(outsize, params->batch_size * params->max_out_bytes);

    // copy the compressed data back to the host
    CUDA_CHECK(cudaMemcpyAsync(outbuf, params->device_output_data, out_bytes, cudaMemcpyDeviceToHost, params->stream));
    CUDA_CHECK(cudaMemcpyAsync(params->host_compressed_bytes, params->device_compressed_bytes, sizeof(size_t) * params->batch_size, cudaMemcpyDeviceToHost, params->stream));

    // ensure that all operations and copies are complete, and that params->device_compressed_bytes is available
    CUDA_CHECK(cudaStreamSynchronize(params->stream));

    size_t total_out_bytes = 0;
    for (size_t i = 0; i < params->batch_size; ++i) {
        //fprintf(stderr, "COMPRESS host_compressed_bytes[%ld]=%ld\n", i, params->host_compressed_bytes[i]);
        total_out_bytes += params->host_compressed_bytes[i];
    }

    return total_out_bytes;
}

int64_t lzbench_nvcomp_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    nvcomp_params_s* params = (nvcomp_params_s*) codec_options->work_mem;
    int status = 0;
    size_t uncompressed_size = outsize;

    // make sure that original data is cleared from device
    size_t in_bytes = std::min(insize, params->batch_size * params->max_out_bytes);
    CUDA_CHECK(cudaMemsetAsync(params->device_input_data, 0, uncompressed_size));
    CUDA_CHECK(cudaMemsetAsync(params->device_output_data, 0, in_bytes));

    // copy the compressed data to the device
    CUDA_CHECK(cudaMemcpyAsync(params->device_output_data, inbuf, in_bytes, cudaMemcpyHostToDevice, params->stream));

    // decompression the data on the device
    CUDA_CHECK(nvcompBatchedLZ4DecompressAsync(
        params->device_compressed_ptrs,
        params->device_compressed_bytes,
        params->device_uncompressed_bytes,
        nullptr,
        params->batch_size,
        params->device_temp_ptr,
        params->device_temp_bytes,
        params->device_uncompressed_ptrs,
        nullptr,
        params->stream));

    // copy the uncompressed data back to the host
    CUDA_CHECK(cudaMemcpyAsync(outbuf, params->device_input_data, uncompressed_size, cudaMemcpyDeviceToHost, params->stream));

    // ensure that all operations and copies are complete
    CUDA_CHECK(cudaStreamSynchronize(params->stream));

    return uncompressed_size;
}

#endif    // BENCH_HAS_NVCOMP

#endif    // BENCH_HAS_CUDA
