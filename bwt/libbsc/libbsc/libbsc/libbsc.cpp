/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Compression/decompression functions                       */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2026 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#include <stdlib.h>
#include <string.h>
#include <memory.h>

#include "../platform/platform.h"
#include "../libbsc.h"

#include "../adler32/adler32.h"
#include "../bwt/bwt.h"
#include "../lzp/lzp.h"
#include "../coder/coder.h"
#include "../st/st.h"

/* The block header can be at any address (e.g. a block inside a larger buffer), so it is */
/* accessed with memcpy. Casting it to int * is undefined behavior, and on strict-alignment */
/* targets such as 32-bit ARM the compiler can merge such accesses into ldm/ldrd, which fault */
/* (SIGBUS) on unaligned addresses. */
static INLINE int bsc_load_int(const unsigned char * p)
{
    int v; memcpy(&v, p, sizeof(v)); return v;
}

static INLINE void bsc_store_int(unsigned char * p, int v)
{
    memcpy(p, &v, sizeof(v));
}

#define LIBBSC_COPY_MIN_BLOCK_SIZE (1024 * 1024)

static void bsc_memcpy(unsigned char * destination, const unsigned char * source, int n, int features)
{

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        int numThreads = omp_get_max_threads() / omp_get_num_threads();
        if (numThreads > n / LIBBSC_COPY_MIN_BLOCK_SIZE) numThreads = n / LIBBSC_COPY_MIN_BLOCK_SIZE;

        if (numThreads > 1)
        {
            int blockSize = (n / numThreads) & (-64);

            #pragma omp parallel for schedule(static) num_threads(numThreads)
            for (int blockId = 0; blockId < numThreads; ++blockId)
            {
                int blockStart = blockId * blockSize;
                memcpy(destination + blockStart, source + blockStart, blockId != numThreads - 1 ? blockSize : n - blockStart);
            }

            return;
        }
    }

#endif

    memcpy(destination, source, n);
}

const char * bsc_version(void)
{
    return LIBBSC_VERSION_STRING;
}

int bsc_init_full(int features, void* (* malloc)(size_t size), void* (* zero_malloc)(size_t size), void (* free)(void* address))
{
    int result = LIBBSC_NO_ERROR;

    if (result == LIBBSC_NO_ERROR) result = bsc_platform_init(features, malloc, zero_malloc, free);
    if (result == LIBBSC_NO_ERROR) result = bsc_coder_init(features);
    if (result == LIBBSC_NO_ERROR) result = bsc_bwt_init(features);

#ifndef LIBBSC_NO_SORT_TRANSFORM

    if (result == LIBBSC_NO_ERROR) result = bsc_st_init(features);

#endif

    return result;
}

int bsc_init(int features)
{
    return bsc_init_full(features, NULL, NULL, NULL);
}

int bsc_store(const unsigned char * input, unsigned char * output, int n, int features)
{
    unsigned int adler32_data = bsc_adler32(input, n, features);

    memmove(output + LIBBSC_HEADER_SIZE, input, n);
    bsc_store_int(output +  0, n + LIBBSC_HEADER_SIZE);
    bsc_store_int(output +  4, n);
    bsc_store_int(output +  8, 0);
    bsc_store_int(output + 12, 0);
    bsc_store_int(output + 16, adler32_data);
    bsc_store_int(output + 20, adler32_data);
    bsc_store_int(output + 24, bsc_adler32(output, 24, features));
    return n + LIBBSC_HEADER_SIZE;
}

int bsc_compress_inplace(unsigned char * data, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int coder, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    int mode = 0;

    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }

    switch (coder)
    {
        case LIBBSC_CODER_QLFC_STATIC   : mode += (LIBBSC_CODER_QLFC_STATIC   << 5); break;
        case LIBBSC_CODER_QLFC_ADAPTIVE : mode += (LIBBSC_CODER_QLFC_ADAPTIVE << 5); break;
        case LIBBSC_CODER_QLFC_FAST     : mode += (LIBBSC_CODER_QLFC_FAST     << 5); break;

        default : return LIBBSC_BAD_PARAMETER;
    }

    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n < 0 || n > 2146435072) return LIBBSC_BAD_PARAMETER;
    if (n <= LIBBSC_HEADER_SIZE)
    {
        return bsc_store(data, data, n, features);
    }

    unsigned int adler32_data = bsc_adler32(data, n, features);

    int lzSize = n;
    if (mode != (mode & 0xff))
    {
        lzSize = bsc_lzp_compress(data, data, n, lzpHashSize, lzpMinLen, features);
        if (lzSize < LIBBSC_NO_ERROR)
        {
            lzSize = n; mode &= 0xff;
        }
    }

    if (lzSize <= LIBBSC_HEADER_SIZE)
    {
        blockSorter = LIBBSC_BLOCKSORTER_BWT;
        mode = (mode & 0xffffffe0) | LIBBSC_BLOCKSORTER_BWT;
    }

    int index = LIBBSC_BAD_PARAMETER; num_indexes = 0;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(data, lzSize, &num_indexes, indexes, features); break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st_encode(data, lzSize, 3, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st_encode(data, lzSize, 4, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st_encode(data, lzSize, 5, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : index = bsc_st_encode(data, lzSize, 6, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : index = bsc_st_encode(data, lzSize, 7, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : index = bsc_st_encode(data, lzSize, 8, features); break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }

    if (n < 64 * 1024) num_indexes = 0;

    if (index < LIBBSC_NO_ERROR)
    {
        return index;
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize + 4096))
    {
        int result = bsc_coder_compress(data, buffer, lzSize, coder, features);
        if (result >= LIBBSC_NO_ERROR) bsc_memcpy(data + LIBBSC_HEADER_SIZE, buffer, result, features);
        bsc_free(buffer);
        if ((result < LIBBSC_NO_ERROR) || (result + 1 + 4 * num_indexes >= n))
        {
            return LIBBSC_NOT_COMPRESSIBLE;
        }
        {
            if (num_indexes > 0)
            {
                memcpy(data + LIBBSC_HEADER_SIZE + result, indexes, 4 * num_indexes);
            }
            data[LIBBSC_HEADER_SIZE + result + 4 * num_indexes] = num_indexes;
            result += 1 + 4 * num_indexes;
        }
        bsc_store_int(data +  0, result + LIBBSC_HEADER_SIZE);
        bsc_store_int(data +  4, n);
        bsc_store_int(data +  8, mode);
        bsc_store_int(data + 12, index);
        bsc_store_int(data + 16, adler32_data);
        bsc_store_int(data + 20, bsc_adler32(data + LIBBSC_HEADER_SIZE, result, features));
        bsc_store_int(data + 24, bsc_adler32(data, 24, features));
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_compress(const unsigned char * input, unsigned char * output, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int coder, int features)
{
    if (input == output)
    {
        return bsc_compress_inplace(output, n, lzpHashSize, lzpMinLen, blockSorter, coder, features);
    }

    int             indexes[256];
    unsigned char   num_indexes;

    int mode = 0;

    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }

    switch (coder)
    {
        case LIBBSC_CODER_QLFC_STATIC   : mode += (LIBBSC_CODER_QLFC_STATIC   << 5); break;
        case LIBBSC_CODER_QLFC_ADAPTIVE : mode += (LIBBSC_CODER_QLFC_ADAPTIVE << 5); break;
        case LIBBSC_CODER_QLFC_FAST     : mode += (LIBBSC_CODER_QLFC_FAST     << 5); break;

        default : return LIBBSC_BAD_PARAMETER;
    }

    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n < 0 || n > 1073741824) return LIBBSC_BAD_PARAMETER;
    if (n <= LIBBSC_HEADER_SIZE)
    {
        return bsc_store(input, output, n, features);
    }
    int lzSize = 0;
    if (mode != (mode & 0xff))
    {
        lzSize = bsc_lzp_compress(input, output, n, lzpHashSize, lzpMinLen, features);
        if (lzSize < LIBBSC_NO_ERROR)
        {
            mode &= 0xff;
        }
    }
    if (mode == (mode & 0xff))
    {
        lzSize = n; bsc_memcpy(output, input, n, features);
    }

    if (lzSize <= LIBBSC_HEADER_SIZE)
    {
        blockSorter = LIBBSC_BLOCKSORTER_BWT;
        mode = (mode & 0xffffffe0) | LIBBSC_BLOCKSORTER_BWT;
    }

    int index = LIBBSC_BAD_PARAMETER; num_indexes = 0;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(output, lzSize, &num_indexes, indexes, features); break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st_encode(output, lzSize, 3, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st_encode(output, lzSize, 4, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st_encode(output, lzSize, 5, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : index = bsc_st_encode(output, lzSize, 6, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : index = bsc_st_encode(output, lzSize, 7, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : index = bsc_st_encode(output, lzSize, 8, features); break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }

    if (n < 64 * 1024) num_indexes = 0;

    if (index < LIBBSC_NO_ERROR)
    {
        return index;
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize + 4096))
    {
        int result = bsc_coder_compress(output, buffer, lzSize, coder, features);
        if (result >= LIBBSC_NO_ERROR) bsc_memcpy(output + LIBBSC_HEADER_SIZE, buffer, result, features);
        bsc_free(buffer);
        if ((result < LIBBSC_NO_ERROR) || (result + 1 + 4 * num_indexes >= n))
        {
            return bsc_store(input, output, n, features);
        }
        {
            if (num_indexes > 0)
            {
                memcpy(output + LIBBSC_HEADER_SIZE + result, indexes, 4 * num_indexes);
            }
            output[LIBBSC_HEADER_SIZE + result + 4 * num_indexes] = num_indexes;
            result += 1 + 4 * num_indexes;
        }
        bsc_store_int(output +  0, result + LIBBSC_HEADER_SIZE);
        bsc_store_int(output +  4, n);
        bsc_store_int(output +  8, mode);
        bsc_store_int(output + 12, index);
        bsc_store_int(output + 16, bsc_adler32(input, n, features));
        bsc_store_int(output + 20, bsc_adler32(output + LIBBSC_HEADER_SIZE, result, features));
        bsc_store_int(output + 24, bsc_adler32(output, 24, features));
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_block_info(const unsigned char * blockHeader, int headerSize, int * pBlockSize, int * pDataSize, int features)
{
    if (headerSize < LIBBSC_HEADER_SIZE)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if ((unsigned int)bsc_load_int(blockHeader + 24) != bsc_adler32(blockHeader, 24, features))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int blockSize    = bsc_load_int(blockHeader +  0);
    int dataSize     = bsc_load_int(blockHeader +  4);
    int mode         = bsc_load_int(blockHeader +  8);
    int index        = bsc_load_int(blockHeader + 12);

    int lzpHashSize  = (mode >> 16) & 0xff;
    int lzpMinLen    = (mode >>  8) & 0xff;
    int coder        = (mode >>  5) & 0x7;
    int blockSorter  = (mode >>  0) & 0x1f;

    int test_mode = 0;

    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : test_mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : test_mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : test_mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : test_mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : test_mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : test_mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : test_mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

        default : if (blockSorter > 0) return LIBBSC_DATA_CORRUPT;
    }

    switch (coder)
    {
        case LIBBSC_CODER_QLFC_STATIC   : test_mode += (LIBBSC_CODER_QLFC_STATIC   << 5); break;
        case LIBBSC_CODER_QLFC_ADAPTIVE : test_mode += (LIBBSC_CODER_QLFC_ADAPTIVE << 5); break;
        case LIBBSC_CODER_QLFC_FAST     : test_mode += (LIBBSC_CODER_QLFC_FAST     << 5); break;

        default : if (coder > 0) return LIBBSC_DATA_CORRUPT;
    }

    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_DATA_CORRUPT;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_DATA_CORRUPT;
        test_mode += (lzpMinLen << 8);
        test_mode += (lzpHashSize << 16);
    }

    if (test_mode != mode)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (blockSize < LIBBSC_HEADER_SIZE || blockSize > LIBBSC_HEADER_SIZE + dataSize)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (index < 0 || index > dataSize)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (pBlockSize != NULL) *pBlockSize = blockSize;
    if (pDataSize != NULL) *pDataSize = dataSize;

    return LIBBSC_NO_ERROR;
}

int bsc_decompress_inplace(unsigned char * data, int inputSize, int outputSize, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    int blockSize = 0, dataSize = 0;

    int info = bsc_block_info(data, inputSize, &blockSize, &dataSize, features);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if ((unsigned int)bsc_load_int(data + 20) != bsc_adler32(data + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, features))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int mode = bsc_load_int(data + 8);
    if (mode == 0)
    {
        memmove(data, data + LIBBSC_HEADER_SIZE, dataSize);
        return LIBBSC_NO_ERROR;
    }

    int             index           = bsc_load_int(data + 12);
    unsigned int    adler32_data    = bsc_load_int(data + 16);

    num_indexes = data[blockSize - 1];
    if (num_indexes > 0)
    {
        memcpy(indexes, data + blockSize - 1 - 4 * num_indexes, 4 * num_indexes);
    }

    int lzpHashSize  = (mode >> 16) & 0xff;
    int lzpMinLen    = (mode >>  8) & 0xff;
    int coder        = (mode >>  5) & 0x7;
    int blockSorter  = (mode >>  0) & 0x1f;

    int lzSize = LIBBSC_NO_ERROR;
    {
        unsigned char * buffer = (unsigned char *)bsc_malloc(blockSize);
        if (buffer == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;

        bsc_memcpy(buffer, data, blockSize, features);

        lzSize = bsc_coder_decompress(buffer + LIBBSC_HEADER_SIZE, data, coder, features);

        bsc_free(buffer);
    }
    if (lzSize < LIBBSC_NO_ERROR)
    {
        return lzSize;
    }

    int result;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(data, lzSize, index, num_indexes, indexes, features); break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st_decode(data, lzSize, 3, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st_decode(data, lzSize, 4, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st_decode(data, lzSize, 5, index, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : result = bsc_st_decode(data, lzSize, 6, index, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : result = bsc_st_decode(data, lzSize, 7, index, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : result = bsc_st_decode(data, lzSize, 8, index, features); break;

#endif

        default : return LIBBSC_DATA_CORRUPT;
    }
    if (result < LIBBSC_NO_ERROR)
    {
        return result;
    }

    if (mode != (mode & 0xff))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize))
        {
            unsigned int adler32_lzp = 0;

            bsc_memcpy(buffer, data, lzSize, features);
            result = bsc_lzp_decompress(buffer, data, lzSize, lzpHashSize, lzpMinLen, &adler32_lzp, features);
            bsc_free(buffer);
            if (result < LIBBSC_NO_ERROR)
            {
                return result;
            }
            return result == dataSize ? (adler32_data == adler32_lzp ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT) : LIBBSC_DATA_CORRUPT;
        }
        return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    return lzSize == dataSize ? (adler32_data == bsc_adler32(data, dataSize, features) ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT) : LIBBSC_DATA_CORRUPT;
}

int bsc_decompress(const unsigned char * input, int inputSize, unsigned char * output, int outputSize, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    if (input == output)
    {
        return bsc_decompress_inplace(output, inputSize, outputSize, features);
    }

    int blockSize = 0, dataSize = 0;

    int info = bsc_block_info(input, inputSize, &blockSize, &dataSize, features);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if ((unsigned int)bsc_load_int(input + 20) != bsc_adler32(input + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, features))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int mode = bsc_load_int(input + 8);
    if (mode == 0)
    {
        bsc_memcpy(output, input + LIBBSC_HEADER_SIZE, dataSize, features);
        return LIBBSC_NO_ERROR;
    }

    int             index           = bsc_load_int(input + 12);
    unsigned int    adler32_data    = bsc_load_int(input + 16);

    num_indexes = input[blockSize - 1];
    if (num_indexes > 0)
    {
        memcpy(indexes, input + blockSize - 1 - 4 * num_indexes, 4 * num_indexes);
    }

    int lzpHashSize  = (mode >> 16) & 0xff;
    int lzpMinLen    = (mode >>  8) & 0xff;
    int coder        = (mode >>  5) & 0x7;
    int blockSorter  = (mode >>  0) & 0x1f;

    int lzSize = bsc_coder_decompress(input + LIBBSC_HEADER_SIZE, output, coder, features);
    if (lzSize < LIBBSC_NO_ERROR)
    {
        return lzSize;
    }

    int result;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(output, lzSize, index, num_indexes, indexes, features); break;

#ifndef LIBBSC_NO_SORT_TRANSFORM

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st_decode(output, lzSize, 3, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st_decode(output, lzSize, 4, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st_decode(output, lzSize, 5, index, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : result = bsc_st_decode(output, lzSize, 6, index, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : result = bsc_st_decode(output, lzSize, 7, index, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : result = bsc_st_decode(output, lzSize, 8, index, features); break;

#endif

        default : return LIBBSC_DATA_CORRUPT;
    }
    if (result < LIBBSC_NO_ERROR)
    {
        return result;
    }

    if (mode != (mode & 0xff))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize))
        {
            unsigned int adler32_lzp = 0;

            bsc_memcpy(buffer, output, lzSize, features);
            result = bsc_lzp_decompress(buffer, output, lzSize, lzpHashSize, lzpMinLen, &adler32_lzp, features);
            bsc_free(buffer);
            if (result < LIBBSC_NO_ERROR)
            {
                return result;
            }
            return result == dataSize ? (adler32_data == adler32_lzp ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT) : LIBBSC_DATA_CORRUPT;
        }
        return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    return lzSize == dataSize ? (adler32_data == bsc_adler32(output, dataSize, features) ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT) : LIBBSC_DATA_CORRUPT;
}

/*-------------------------------------------------*/
/* End                                  libbsc.cpp */
/*-------------------------------------------------*/
