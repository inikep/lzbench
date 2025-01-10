/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Second stage encoding functions                           */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2024 Ilya Grebnov <ilya.grebnov@gmail.com>

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
#include <memory.h>

#include "coder.h"

#include "../libbsc.h"
#include "../platform/platform.h"

#include "qlfc/qlfc.h"

int bsc_coder_init(int features)
{
    int result = LIBBSC_NO_ERROR;

    if (result == LIBBSC_NO_ERROR) result = bsc_qlfc_init(features);

    return result;
}

static INLINE int bsc_coder_num_blocks(int n)
{
    if (n <       256 * 1024)   return 1;
    if (n <  4 * 1024 * 1024)   return 2;
    if (n < 16 * 1024 * 1024)   return 4;

    return 8;
}

int bsc_coder_encode_block(const unsigned char * input, unsigned char * output, int inputSize, int outputSize, int coder)
{
    if (coder == LIBBSC_CODER_QLFC_STATIC)   return bsc_qlfc_static_encode_block  (input, output, inputSize, outputSize);
    if (coder == LIBBSC_CODER_QLFC_ADAPTIVE) return bsc_qlfc_adaptive_encode_block(input, output, inputSize, outputSize);
    if (coder == LIBBSC_CODER_QLFC_FAST)     return bsc_qlfc_fast_encode_block    (input, output, inputSize, outputSize);

    return LIBBSC_BAD_PARAMETER;
}

void bsc_coder_split_blocks(const unsigned char * input, int n, int nBlocks, int * blockStart, int * blockSize)
{
    int rankSize = 0;
    for (int i = 1; i < n; i += 32)
    {
        if (input[i] != input[i - 1]) rankSize++;
    }

    if (rankSize > nBlocks)
    {
        int blockRankSize = rankSize / nBlocks;

        blockStart[0] = 0; rankSize = 0;
        for (int id = 0, i = 1; i < n; i += 32)
        {
            if (input[i] != input[i - 1])
            {
                rankSize++;
                if (rankSize == blockRankSize)
                {
                    rankSize = 0;

                    blockSize[id] = i - blockStart[id];
                    id++; blockStart[id] = i;

                    if (id == nBlocks - 1) break;
                }
            }
        }
        blockSize[nBlocks - 1] = n - blockStart[nBlocks - 1];
    }
    else
    {
        for (int p = 0; p < nBlocks; ++p)
        {
            blockStart[p] = (n / nBlocks) * p;
            blockSize[p]  = (p != nBlocks - 1) ? n / nBlocks : n - (n / nBlocks) * (nBlocks - 1);
        }
    }
}

int bsc_coder_compress_serial(const unsigned char * input, unsigned char * output, int n, int coder)
{
    if (bsc_coder_num_blocks(n) == 1)
    {
        int result = bsc_coder_encode_block(input, output + 1, n, n - 1, coder);
        if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);

        return result;
    }

    int compressedStart[ALPHABET_SIZE];
    int compressedSize[ALPHABET_SIZE];

    int nBlocks   = bsc_coder_num_blocks(n);
    int outputPtr = 1 + 8 * nBlocks;

    bsc_coder_split_blocks(input, n, nBlocks, compressedStart, compressedSize);

    output[0] = nBlocks;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        int inputStart  = compressedStart[blockId];
        int inputSize   = compressedSize[blockId];
        int outputSize  = inputSize; if (outputSize > n - outputPtr) outputSize = n - outputPtr;

        int result = bsc_coder_encode_block(input + inputStart, output + outputPtr, inputSize, outputSize, coder);
        if (result < LIBBSC_NO_ERROR)
        {
            if (outputPtr + inputSize >= n) return LIBBSC_NOT_COMPRESSIBLE;
            result = inputSize; memcpy(output + outputPtr, input + inputStart, inputSize);
        }

#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
        *(int *)(output + 1 + 8 * blockId + 0) = inputSize;
        *(int *)(output + 1 + 8 * blockId + 4) = result;
#else
        memcpy(output + 1 + 8 * blockId + 0, &inputSize, sizeof(int));
        memcpy(output + 1 + 8 * blockId + 4, &result, sizeof(int));
#endif

        outputPtr += result;
    }

    return outputPtr;
}

#ifdef LIBBSC_OPENMP

int bsc_coder_compress_parallel(const unsigned char * input, unsigned char * output, int n, int coder)
{
    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        int compressionResult[ALPHABET_SIZE];
        int compressedStart[ALPHABET_SIZE];
        int compressedSize[ALPHABET_SIZE];

        int nBlocks = bsc_coder_num_blocks(n);
        int result  = LIBBSC_NO_ERROR;

        int numThreads = omp_get_max_threads();
        if (numThreads > nBlocks) numThreads = nBlocks;

        output[0] = nBlocks;
        #pragma omp parallel num_threads(numThreads) if(numThreads > 1)
        {
            if (omp_get_num_threads() == 1)
            {
                result = bsc_coder_compress_serial(input, output, n, coder);
            }
            else
            {
                #pragma omp single
                {
                    bsc_coder_split_blocks(input, n, nBlocks, compressedStart, compressedSize);
                }

                #pragma omp for schedule(dynamic)
                for (int blockId = 0; blockId < nBlocks; ++blockId)
                {
                    int blockStart   = compressedStart[blockId];
                    int blockSize    = compressedSize[blockId];

                    compressionResult[blockId] = bsc_coder_encode_block(input + blockStart, buffer + blockStart, blockSize, blockSize, coder);
                    if (compressionResult[blockId] < LIBBSC_NO_ERROR) compressionResult[blockId] = blockSize;

                    *(int *)(output + 1 + 8 * blockId + 0) = blockSize;
                    *(int *)(output + 1 + 8 * blockId + 4) = compressionResult[blockId];
                }

                #pragma omp single
                {
                    result = 1 + 8 * nBlocks;
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        result += compressionResult[blockId];
                    }

                    if (result >= n) result = LIBBSC_NOT_COMPRESSIBLE;
                }

                if (result >= LIBBSC_NO_ERROR)
                {
                    #pragma omp for schedule(dynamic)
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        int blockStart   = compressedStart[blockId];
                        int blockSize    = compressedSize[blockId];

                        int outputPtr = 1 + 8 * nBlocks;
                        for (int p = 0; p < blockId; ++p) outputPtr += compressionResult[p];

                        if (compressionResult[blockId] != blockSize)
                        {
                            memcpy(output + outputPtr, buffer + blockStart, compressionResult[blockId]);
                        }
                        else
                        {
                            memcpy(output + outputPtr, input + blockStart, compressionResult[blockId]);
                        }
                    }
                }
            }
        }

        bsc_free(buffer);

        return result;
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_coder_compress(const unsigned char * input, unsigned char * output, int n, int coder, int features)
{
    if ((coder != LIBBSC_CODER_QLFC_STATIC) && (coder != LIBBSC_CODER_QLFC_ADAPTIVE) && (coder != LIBBSC_CODER_QLFC_FAST))
    {
        return LIBBSC_BAD_PARAMETER;
    }

#ifdef LIBBSC_OPENMP

    if ((bsc_coder_num_blocks(n) != 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        return bsc_coder_compress_parallel(input, output, n, coder);
    }

#endif

    return bsc_coder_compress_serial(input, output, n, coder);
}


int bsc_coder_decode_block(const unsigned char * input, unsigned char * output, int coder)
{
    if (coder == LIBBSC_CODER_QLFC_STATIC)   return bsc_qlfc_static_decode_block  (input, output);
    if (coder == LIBBSC_CODER_QLFC_ADAPTIVE) return bsc_qlfc_adaptive_decode_block(input, output);
    if (coder == LIBBSC_CODER_QLFC_FAST)     return bsc_qlfc_fast_decode_block    (input, output);

    return LIBBSC_BAD_PARAMETER;
}

int bsc_coder_decompress(const unsigned char * input, unsigned char * output, int coder, int features)
{
    if ((coder != LIBBSC_CODER_QLFC_STATIC) && (coder != LIBBSC_CODER_QLFC_ADAPTIVE) && (coder != LIBBSC_CODER_QLFC_FAST))
    {
        return LIBBSC_BAD_PARAMETER;
    }

    int nBlocks = input[0];
    if (nBlocks == 1)
    {
        return bsc_coder_decode_block(input + 1, output, coder);
    }

    int decompressionResult[ALPHABET_SIZE];

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr  = 0; for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_coder_decode_block(input + inputPtr, output + outputPtr, coder);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }
    else

#endif

    {
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr  = 0; for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_coder_decode_block(input + inputPtr, output + outputPtr, coder);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }

    int dataSize = 0, result = LIBBSC_NO_ERROR;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        if (decompressionResult[blockId] < LIBBSC_NO_ERROR) result = decompressionResult[blockId];
        dataSize += decompressionResult[blockId];
    }

    return (result == LIBBSC_NO_ERROR) ? dataSize : result;
}

/*-----------------------------------------------------------*/
/* End                                             coder.cpp */
/*-----------------------------------------------------------*/
