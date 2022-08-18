/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Detectors of blocksize, recordsize and contexts reorder.  */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

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

#include "../filters.h"

#include "../platform/platform.h"
#include "../libbsc.h"

#include "tables.h"

#define DETECTORS_MAX_RECORD_SIZE   4
#define DETECTORS_NUM_BLOCKS        48
#define DETECTORS_BLOCK_SIZE        24576

struct BscSegmentationModel
{
    struct
    {
        int left, right;
    } contextsCount[ALPHABET_SIZE];
    struct
    {
        struct
        {
            int left, right;
        } Frequencies[ALPHABET_SIZE];
    } contexts[ALPHABET_SIZE];
};

struct BscReorderingModel
{
    struct
    {
        int frequencies[ALPHABET_SIZE];
    } contexts[DETECTORS_MAX_RECORD_SIZE][ALPHABET_SIZE];
};

int bsc_detect_segments_serial(BscSegmentationModel * RESTRICT model, const unsigned char * RESTRICT input, int n)
{
    memset(model, 0, sizeof(BscSegmentationModel));

    for (int context = 0, i = 0; i < n; ++i)
    {
        unsigned char symbol = input[i];
        model->contexts[context].Frequencies[symbol].right++;
        context = (unsigned char)((context << 5) ^ symbol);
    }

    long long entropy = 0;
    for (int context = 0; context < ALPHABET_SIZE; ++context)
    {
        int count = 0;
        for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
        {
            int frequency = model->contexts[context].Frequencies[symbol].right;
            count += frequency; entropy -= bsc_entropy(frequency);
        }
        model->contextsCount[context].right = count; entropy += bsc_entropy(count);
    }

    int blockSize = n;

    long long localEntropy = entropy, bestEntropy = entropy - (entropy >> 5) - (65536LL * 12 * 1024);
    for (int context = 0, i = 0; i < n; ++i)
    {
        if (localEntropy < bestEntropy)
        {
            bestEntropy = localEntropy;
            blockSize   = i;
        }

        unsigned char symbol = input[i];

        localEntropy    += bsc_delta(--model->contexts[context].Frequencies[symbol].right);
        localEntropy    -= bsc_delta(model->contexts[context].Frequencies[symbol].left++);
        localEntropy    -= bsc_delta(--model->contextsCount[context].right);
        localEntropy    += bsc_delta(model->contextsCount[context].left++);

        context = (unsigned char)((context << 5) ^ symbol);
    }

    return blockSize;
}

#ifdef LIBBSC_OPENMP

int bsc_detect_segments_parallel(BscSegmentationModel * RESTRICT model0, BscSegmentationModel * RESTRICT model1, const unsigned char * RESTRICT input, int n)
{
    int globalBlockSize = n; long long globalEntropy, globalBestEntropy;

    #pragma omp parallel num_threads(2)
    {
        int nThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();

        if (nThreads == 1)
        {
            globalBlockSize = bsc_detect_segments_serial(model0, input, n);
        }
        else
        {
            int median = n / 2;

            {
                if (threadId == 0)
                {
                    memset(model0, 0, sizeof(BscSegmentationModel));

                    int context = 0;
                    for (int i = 0; i < median; ++i)
                    {
                        unsigned char symbol = input[i];
                        model0->contexts[context].Frequencies[symbol].right++;
                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }
                else
                {
                    memset(model1, 0, sizeof(BscSegmentationModel));

                    int context = (unsigned char)((input[median - 2] << 5) ^ input[median - 1]);
                    for (int i = median; i < n; ++i)
                    {
                        unsigned char symbol = input[i];
                        model1->contexts[context].Frequencies[symbol].left++;
                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }

                #pragma omp barrier
            }

            {
                #pragma omp single
                {
                    long long entropy = 0;
                    for (int context = 0; context < ALPHABET_SIZE; ++context)
                    {
                        int count = 0;
                        for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
                        {
                            int frequency = model0->contexts[context].Frequencies[symbol].right + model1->contexts[context].Frequencies[symbol].left;
                            model0->contexts[context].Frequencies[symbol].right = model1->contexts[context].Frequencies[symbol].left = frequency;

                            count += frequency; entropy -= bsc_entropy(frequency);
                        }
                        model0->contextsCount[context].right = model1->contextsCount[context].left = count; entropy += bsc_entropy(count);
                    }

                    globalEntropy = entropy; globalBestEntropy = entropy - (entropy >> 5) - (65536LL * 12 * 1024);
                }
            }

            {
                int localBlockSize = n; long long localBestEntropy = globalEntropy - (globalEntropy >> 5) - (65536LL * 12 * 1024);

                if (threadId == 0)
                {
                    long long localEntropy = globalEntropy;
                    for (int context = 0, i = 0; i < median; ++i)
                    {
                        if (localEntropy < localBestEntropy)
                        {
                            localBestEntropy = localEntropy;
                            localBlockSize   = i;
                        }

                        unsigned char symbol = input[i];

                        localEntropy    += bsc_delta(--model0->contexts[context].Frequencies[symbol].right);
                        localEntropy    -= bsc_delta(model0->contexts[context].Frequencies[symbol].left++);
                        localEntropy    -= bsc_delta(--model0->contextsCount[context].right);
                        localEntropy    += bsc_delta(model0->contextsCount[context].left++);

                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }
                else
                {
                    long long localEntropy = globalEntropy;
                    for (int i = n - 1; i >= median; --i)
                    {
                        unsigned char   symbol  = input[i];
                        int             context = (unsigned char)((input[i - 2] << 5) ^ input[i - 1]);

                        localEntropy    -= bsc_delta(model1->contexts[context].Frequencies[symbol].right++);
                        localEntropy    += bsc_delta(--model1->contexts[context].Frequencies[symbol].left);
                        localEntropy    += bsc_delta(model1->contextsCount[context].right++);
                        localEntropy    -= bsc_delta(--model1->contextsCount[context].left);

                        if (localEntropy <= localBestEntropy)
                        {
                            localBestEntropy = localEntropy;
                            localBlockSize   = i;
                        }
                    }
                }

                if (globalBestEntropy > localBestEntropy)
                {
                    #pragma omp critical
                    {
                        if (globalBestEntropy > localBestEntropy)
                        {
                            globalBlockSize = localBlockSize; globalBestEntropy = localBestEntropy;
                        }
                    }
                }
            }
        }
    }

    return globalBlockSize;
}


#endif

int bsc_detect_segments_recursive(BscSegmentationModel * model0, BscSegmentationModel * model1, const unsigned char * input, int n, int * segments, int k, int features)
{
    if (n < DETECTORS_BLOCK_SIZE || k == 1)
    {
        segments[0] = n;
        return 1;
    }

    int blockSize = n;

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        blockSize = bsc_detect_segments_parallel(model0, model1, input, n);
    }
    else

#endif

    {
        blockSize = bsc_detect_segments_serial(model0, input, n);
    }

    if (blockSize == n)
    {
        segments[0] = n;
        return 1;
    }

    int leftResult = bsc_detect_segments_recursive(model0, model1, input, blockSize, segments, k - 1, features);
    if (leftResult < LIBBSC_NO_ERROR) return leftResult;

    int rightResult = bsc_detect_segments_recursive(model0, model1, input + blockSize, n - blockSize, segments + leftResult, k - leftResult, features);
    if (rightResult < LIBBSC_NO_ERROR) return rightResult;

    return leftResult + rightResult;
}

int bsc_detect_segments(const unsigned char * input, int n, int * segments, int k, int features)
{
    if (n < DETECTORS_BLOCK_SIZE || k == 1)
    {
        segments[0] = n;
        return 1;
    }

    if (BscSegmentationModel * model0 = (BscSegmentationModel *)bsc_malloc(sizeof(BscSegmentationModel)))
    {
        if (BscSegmentationModel * model1 = (BscSegmentationModel *)bsc_malloc(sizeof(BscSegmentationModel)))
        {
            int result = bsc_detect_segments_recursive(model0, model1, input, n, segments, k, features);

            bsc_free(model1); bsc_free(model0);

            return result;
        }
        bsc_free(model0);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static long long bsc_estimate_contextsorder(const unsigned char * input, int n)
{
    int frequencies[ALPHABET_SIZE][3];

    memset(frequencies, 0, sizeof(frequencies));

    unsigned char MTF0 = 0;
    unsigned char MTF1 = 1;
    unsigned char MTFC = 0;

    for (int i = 0; i < n; ++i)
    {
        unsigned char C = input[i];
        if (C == MTF0)
        {
            frequencies[MTFC][0]++; MTFC = MTFC << 2;
        }
        else
        {
            if (C == MTF1)
            {
                frequencies[MTFC][1]++; MTFC = (MTFC << 2) | 1;
            }
            else
            {
                frequencies[MTFC][2]++; MTFC = (MTFC << 2) | 2;
            }
            MTF1 = MTF0; MTF0 = C;
        }
    }

    long long entropy = 0;
    for (int context = 0; context < ALPHABET_SIZE; ++context)
    {
        int count = 0;
        for (int rank = 0; rank < 3; ++rank)
        {
            count += frequencies[context][rank];
            entropy -= bsc_entropy(frequencies[context][rank]);
        }
        entropy += bsc_entropy(count);
    }

    return entropy;
}

int bsc_detect_contextsorder(const unsigned char * RESTRICT input, int n, int features)
{
    int sortingContexts = LIBBSC_NOT_ENOUGH_MEMORY;

    if ((n > DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) && (features & LIBBSC_FEATURE_FASTMODE))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE * sizeof(unsigned char)))
        {
            int blockStride = (((n - DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) / DETECTORS_NUM_BLOCKS) / 48) * 48;

            for (int block = 0; block < DETECTORS_NUM_BLOCKS; ++block)
            {
                memcpy(buffer + block * DETECTORS_BLOCK_SIZE, input + block * (DETECTORS_BLOCK_SIZE + blockStride), DETECTORS_BLOCK_SIZE);
            }

            sortingContexts = bsc_detect_contextsorder(buffer, DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE, features);

            bsc_free(buffer);
        }

        return sortingContexts;
    }

    if (unsigned char * RESTRICT buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        if (int * RESTRICT bucket0 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            if (int * RESTRICT bucket1 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
            {
                unsigned char C0 = input[n - 1];
                for (int i = 0; i < n; ++i)
                {
                    unsigned char C1 = input[i];
                    bucket0[(C0 << 8) | C1]++;
                    bucket1[(C1 << 8) | C0]++;
                    C0 = C1;
                }

                for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                {
                    int tmp = sum; sum += bucket0[i]; bucket0[i] = tmp;
                }

                unsigned char F0 = input[n - 2];
                unsigned char F1 = input[n - 1];
                for (int i = 0; i < n; ++i)
                {
                    unsigned char F2 = input[i];
                    buffer[bucket0[(F1 << 8) | F2]++] = F0;
                    F0 = F1; F1 = F2;
                }

                long long following = bsc_estimate_contextsorder(buffer, n);

                for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                {
                    int tmp = sum; sum += bucket1[i]; bucket1[i] = tmp;
                }

                unsigned char P0 = input[1];
                unsigned char P1 = input[0];
                for (int i = n - 1; i >= 0; --i)
                {
                    unsigned char P2 = input[i];
                    buffer[bucket1[(P1 << 8) | P2]++] = P0;
                    P0 = P1; P1 = P2;
                }

                long long preceding = bsc_estimate_contextsorder(buffer, n);

                sortingContexts = (preceding < following) ? LIBBSC_CONTEXTS_PRECEDING : LIBBSC_CONTEXTS_FOLLOWING;

                bsc_free(bucket1);
            }
            bsc_free(bucket0);
        };
        bsc_free(buffer);
    }

    return sortingContexts;
}

long long bsc_estimate_reordering(BscReorderingModel * model, int recordSize)
{
    long long entropy = 0;
    for (int record = 0; record < recordSize; ++record)
    {
        for (int context = 0; context < ALPHABET_SIZE; ++context)
        {
            int count = 0;
            for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
            {
                int frequency = model->contexts[record][context].frequencies[symbol];
                count += frequency; entropy -= bsc_entropy(frequency);
            }
            entropy += (65536LL * 8 * (count < 256 ? count : 256)) + bsc_entropy(count);
        }
    }
    return entropy;
}

int bsc_detect_recordsize(const unsigned char * RESTRICT input, int n, int features)
{
    int result = LIBBSC_NOT_ENOUGH_MEMORY;

    if ((n > DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) && (features & LIBBSC_FEATURE_FASTMODE))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE * sizeof(unsigned char)))
        {
            int blockStride = (((n - DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) / DETECTORS_NUM_BLOCKS) / 48) * 48;

            for (int block = 0; block < DETECTORS_NUM_BLOCKS; ++block)
            {
                memcpy(buffer + block * DETECTORS_BLOCK_SIZE, input + block * (DETECTORS_BLOCK_SIZE + blockStride), DETECTORS_BLOCK_SIZE);
            }

            result = bsc_detect_recordsize(buffer, DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE, features);

            bsc_free(buffer);
        }

        return result;
    }

    if (BscReorderingModel * RESTRICT model = (BscReorderingModel *)bsc_malloc(sizeof(BscReorderingModel)))
    {
        long long Entropy[DETECTORS_MAX_RECORD_SIZE];

        if ((n % 48) != 0) n = n - (n % 48);

        for (int recordSize = 1; recordSize <= DETECTORS_MAX_RECORD_SIZE; ++recordSize)
        {
            memset(model, 0, sizeof(BscReorderingModel));

            if (recordSize == 1)
            {
                int ctx0 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[0][ctx0].frequencies[c1]++; ctx0 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[0][ctx0].frequencies[c2]++; ctx0 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[0][ctx0].frequencies[c3]++; ctx0 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[0][ctx0].frequencies[c5]++; ctx0 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[0][ctx0].frequencies[c6]++; ctx0 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[0][ctx0].frequencies[c7]++; ctx0 = c7;
                }
            }

            if (recordSize == 2)
            {
                int ctx0 = 0, ctx1 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[0][ctx0].frequencies[c2]++; ctx0 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[1][ctx1].frequencies[c3]++; ctx1 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[1][ctx1].frequencies[c5]++; ctx1 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[0][ctx0].frequencies[c6]++; ctx0 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[1][ctx1].frequencies[c7]++; ctx1 = c7;
                }
            }

            if (recordSize == 3)
            {
                int ctx0 = 0, ctx1 = 0, ctx2 = 0;
                for (int i = 0; i < n; i += 6)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[2][ctx2].frequencies[c2]++; ctx2 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[0][ctx0].frequencies[c3]++; ctx0 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[1][ctx1].frequencies[c4]++; ctx1 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[2][ctx2].frequencies[c5]++; ctx2 = c5;
                }
            }

            if (recordSize == 4)
            {
                int ctx0 = 0, ctx1 = 0, ctx2 = 0, ctx3 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[2][ctx2].frequencies[c2]++; ctx2 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[3][ctx3].frequencies[c3]++; ctx3 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[1][ctx1].frequencies[c5]++; ctx1 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[2][ctx2].frequencies[c6]++; ctx2 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[3][ctx3].frequencies[c7]++; ctx3 = c7;
                }
            }

            if (recordSize > 4)
            {
                int Context[DETECTORS_MAX_RECORD_SIZE] = { 0 };
                for (int record = 0, i = 0; i < n; ++i)
                {
                    model->contexts[record][Context[record]].frequencies[input[i]]++;
                    Context[record] = input[i]; record++; if (record == recordSize) record = 0;
                }
            }

            Entropy[recordSize - 1] = bsc_estimate_reordering(model, recordSize);
        }

        long long bestSize = Entropy[0] - (Entropy[0] >> 4) - (65536LL * 8 * 1024);

        result = 1;
        for (int recordSize = 1; recordSize <= DETECTORS_MAX_RECORD_SIZE; ++recordSize)
        {
            if (bestSize > Entropy[recordSize - 1]) { bestSize = Entropy[recordSize - 1]; result = recordSize; }
        }

        bsc_free(model);
    };

    return result;
}

/*-------------------------------------------------*/
/* End                               detectors.cpp */
/*-------------------------------------------------*/
