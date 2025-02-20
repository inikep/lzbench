/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Sort Transform                                            */
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

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

#include <stdlib.h>
#include <memory.h>

#include "st.h"

#include "../libbsc.h"
#include "../platform/platform.h"

#include "st.cuh"

#define ALPHABET_SQRT_SIZE  (16)

int bsc_st_init(int features)
{
#ifdef LIBBSC_CUDA_SUPPORT
    return bsc_st_cuda_init(features);
#else
    return LIBBSC_NO_ERROR;
#endif
}

static int bsc_st3_transform_serial(unsigned char * RESTRICT T, unsigned short * RESTRICT P, int * RESTRICT bucket, int n)
{
    unsigned int count[ALPHABET_SIZE]; memset(count, 0, ALPHABET_SIZE * sizeof(unsigned int));

    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C1 = T[i];
        count[C1]++; bucket[(C0 << 8) | C1]++;
        C0 = C1;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        int tmp = sum; sum += bucket[i]; bucket[i] = tmp;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE; ++i)
    {
        int tmp = sum; sum += count[i]; count[i] = tmp;
    }

    int pos = bucket[(T[1] << 8) | T[2]];

    unsigned int W = (T[n - 1] << 16) | (T[0] << 8) | T[1];
    for (int i = 0; i < n; ++i)
    {
        W = (W << 8) | T[i + 2];
        P[bucket[W & 0x0000ffff]++] = W >> 16;
    }

    for (int i = 0; i < pos; ++i)
    {
        T[count[P[i] & 0x00ff]++] = (unsigned char)(P[i] >> 8);
    }
    int index = count[P[pos] & 0x00ff];
    for (int i = pos; i < n; ++i)
    {
        T[count[P[i] & 0x00ff]++] = (unsigned char)(P[i] >> 8);
    }

    return index;
}

static int bsc_st4_transform_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket, int n)
{
    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C1 = T[i];
        bucket[(C0 << 8) | C1]++;
        C0 = C1;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        int tmp = sum; sum += bucket[i]; bucket[i] = tmp;
    }

    int pos = bucket[(T[2] << 8) | T[3]];

    unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C = (unsigned char)(W >> 24);
        W = (W << 8) | T[i + 3];
        P[bucket[W & 0x0000ffff]++] = (W & 0xffff0000) | C;
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 16]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 16];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 16]] = P[i] & 0xff;
    }

    return index;
}

static int bsc_st5_transform_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket, int n)
{
    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 2] & 0xf;
    unsigned char C1 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C2 = T[i];
        bucket[(C0 << 16) | (C1 << 8) | C2]++;
        C0 = C1 & 0xf; C1 = C2;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        int tmp = sum; sum += bucket[i]; bucket[i] = tmp;
    }

    int pos = bucket[((T[2] & 0xf) << 16) | (T[3] << 8) | T[4]];

    unsigned char L = T[n - 1];
    unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
    for (int i = 0; i < n; ++i)
    {
        unsigned int V = (W & 0xfffff000) | L;
        L = (unsigned char)(W >> 24); W = (W << 8) | T[i + 4];
        P[bucket[W & 0x000fffff]++] = V;
    }

    memset(bucket, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

    unsigned char P0 = T[n - 2];
    unsigned char P1 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char P2 = T[i];
        bucket[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
        P0 = P1; P1 = P2;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i]; bucket[i] = sum;
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 12]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 12];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 12]] = P[i] & 0xff;
    }

    return index;
}

static int bsc_st6_transform_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket, int n)
{
    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned int W = (T[n - 2] << 16) | (T[n - 1] << 8) | T[0];
    for (int i = 0; i < n; ++i)
    {
        W = (W << 8) | T[i + 1]; bucket[W >> 8]++;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        int tmp = sum; sum += bucket[i]; bucket[i] = tmp;
    }

    int pos = bucket[(T[3] << 16) | (T[4] << 8) | T[5]];

    unsigned int W0 = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
    unsigned int W1 = (T[    2] << 24) | (T[    3] << 16) | (T[4] << 8) | T[5];
    for (int i = 0; i < n; ++i)
    {
        W0 = (W0 << 8) | T[i + 2]; W1 = (W1 << 8) | T[i + 6];
        P[bucket[W1 >> 8]++] = (W0 << 8) | (W0 >> 24);
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 8]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 8];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 8]] = P[i] & 0xff;
    }

    return index;
}

#ifdef LIBBSC_OPENMP

static int bsc_st3_transform_parallel(unsigned char * RESTRICT T, unsigned short * RESTRICT P, int * RESTRICT bucket0, int n)
{
    unsigned int count0[ALPHABET_SIZE]; memset(count0, 0, ALPHABET_SIZE * sizeof(unsigned int));
    unsigned int count1[ALPHABET_SIZE]; memset(count1, 0, ALPHABET_SIZE * sizeof(unsigned int));

    if (int * RESTRICT bucket1 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        int pos, index = 0;

        for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

        #pragma omp parallel num_threads(2)
        {
            int nThreads = omp_get_num_threads();
            int threadId = omp_get_thread_num();

            if (nThreads == 1)
            {
                index = bsc_st3_transform_serial(T, P, bucket0, n);
            }
            else
            {
                int median = n / 2;

                {
                    if (threadId == 0)
                    {
                        unsigned char C0 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char C1 = T[i];
                            count0[C1]++; bucket0[(C0 << 8) | C1]++;
                            C0 = C1;
                        }
                    }
                    else
                    {
                        unsigned char C0 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char C1 = T[i];
                            count1[C1]++; bucket1[(C0 << 8) | C1]++;
                            C0 = C1;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    #pragma omp single
                    {
                        for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int tmp = sum; sum += bucket0[i] + bucket1[i]; bucket0[i] = tmp; bucket1[i] = sum - 1;
                        }

                        for (int sum = 0, i = 0; i < ALPHABET_SIZE; ++i)
                        {
                            int tmp = sum; sum += count0[i] + count1[i]; count0[i] = tmp; count1[i] = sum - 1;
                        }

                        pos = bucket0[(T[1] << 8) | T[2]];
                    }
                }

                {
                    if (threadId == 0)
                    {
                        unsigned int W = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                        for (int i = 0; i < median; ++i)
                        {
                            W = (W << 8) | T[i + 2];
                            P[bucket0[W & 0x0000ffff]++] = W >> 16;
                        }
                    }
                    else
                    {
                        unsigned int W = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                        for (int i = n - 1; i >= median; --i)
                        {
                            P[bucket1[W & 0x0000ffff]--] = W >> 16;
                            W = (W >> 8) | (T[i - 2] << 24);
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        if (pos < median)
                        {
                            for (int i = 0; i < pos; ++i)
                            {
                                T[count0[P[i] & 0x00ff]++] = (unsigned char)(P[i] >> 8);
                            }
                            index = count0[P[pos] & 0x00ff];
                            for (int i = pos; i < median; ++i)
                            {
                                T[count0[P[i] & 0x00ff]++] = (unsigned char)(P[i] >> 8);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < median; ++i)
                            {
                                T[count0[P[i] & 0x00ff]++] = (unsigned char)(P[i] >> 8);
                            }
                        }
                    }
                    else
                    {
                        if (pos >= median)
                        {
                            for (int i = n - 1; i > pos; --i)
                            {
                                T[count1[P[i] & 0x00ff]--] = (unsigned char)(P[i] >> 8);
                            }
                            index = count1[P[pos] & 0x00ff];
                            for (int i = pos; i >= median; --i)
                            {
                                T[count1[P[i] & 0x00ff]--] = (unsigned char)(P[i] >> 8);
                            }
                        }
                        else
                        {
                            for (int i = n - 1; i >= median; --i)
                            {
                                T[count1[P[i] & 0x00ff]--] = (unsigned char)(P[i] >> 8);
                            }
                        }
                    }
                }
            }
        }

        bsc_free(bucket1);
        return index;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st4_transform_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket, int n)
{
    if (int * RESTRICT bucket0 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        if (int * RESTRICT bucket1 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int pos, index = 0;

            for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

            #pragma omp parallel num_threads(2)
            {
                int nThreads = omp_get_num_threads();
                int threadId = omp_get_thread_num();

                if (nThreads == 1)
                {
                    index = bsc_st4_transform_serial(T, P, bucket, n);
                }
                else
                {
                    int median = n / 2;

                    {
                        if (threadId == 0)
                        {
                            unsigned char C0 = T[n - 1];
                            for (int i = 0; i < median; ++i)
                            {
                                unsigned char C1 = T[i];
                                bucket0[(C0 << 8) | C1]++;
                                C0 = C1;
                            }
                        }
                        else
                        {
                            unsigned char C0 = T[median - 1];
                            for (int i = median; i < n; ++i)
                            {
                                unsigned char C1 = T[i];
                                bucket1[(C0 << 8) | C1]++;
                                C0 = C1;
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        #pragma omp single
                        {
                            for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                            {
                                int tmp = sum; sum += bucket0[i] + bucket1[i]; bucket[i] = bucket0[i] = tmp; bucket1[i]= sum - 1;
                            }

                            pos = bucket[(T[2] << 8) | T[3]];
                        }
                    }

                    {
                        if (threadId == 0)
                        {
                            unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            for (int i = 0; i < median; ++i)
                            {
                                unsigned char C = (unsigned char)(W >> 24);
                                W = (W << 8) | T[i + 3];
                                P[bucket0[W & 0x0000ffff]++] = (W & 0xffff0000) | C;
                            }
                        }
                        else
                        {
                            unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            for (int i = n - 1; i >= median; --i)
                            {
                                unsigned char C = T[i - 1];
                                P[bucket1[W & 0x0000ffff]--] = (W & 0xffff0000) | C;
                                W = (W >> 8) | (C << 24);
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i) bucket0[i] = bucket[i];

                            if (pos < median)
                            {
                                for (int i = 0; i < pos; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                                index = bucket0[P[pos] >> 16];
                                for (int i = pos; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = 0; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE - 1; ++i) bucket1[i] = bucket[i + 1] - 1;
                            bucket1[ALPHABET_SIZE * ALPHABET_SIZE - 1] = n - 1;

                            if (pos >= median)
                            {
                                for (int i = n - 1; i > pos; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                                index = bucket1[P[pos] >> 16];
                                for (int i = pos; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = n - 1; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                            }
                        }
                    }
                }
            }

            bsc_free(bucket1); bsc_free(bucket0);
            return index;
        };
        bsc_free(bucket0);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st5_transform_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket0, int n)
{
    if (int * RESTRICT bucket1 = (int *)bsc_zero_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        int pos, index = 0;

        for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

        #pragma omp parallel num_threads(2)
        {
            int nThreads = omp_get_num_threads();
            int threadId = omp_get_thread_num();

            if (nThreads == 1)
            {
                index = bsc_st5_transform_serial(T, P, bucket0, n);
            }
            else
            {
                int median = n / 2;

                {
                    if (threadId == 0)
                    {
                        unsigned char C0 = T[n - 2] & 0xf;
                        unsigned char C1 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char C2 = T[i];
                            bucket0[(C0 << 16) | (C1 << 8) | C2]++;
                            C0 = C1 & 0xf; C1 = C2;
                        }
                    }
                    else
                    {
                        unsigned char C0 = T[median - 2] & 0xf;
                        unsigned char C1 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char C2 = T[i];
                            bucket1[(C0 << 16) | (C1 << 8) | C2]++;
                            C0 = C1 & 0xf; C1 = C2;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    #pragma omp single
                    {
                        for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int tmp = sum; sum += bucket0[i] + bucket1[i]; bucket0[i] = tmp; bucket1[i] = sum - 1;
                        }

                        pos = bucket0[((T[2] & 0xf) << 16) | (T[3] << 8) | T[4]];
                    }
                }

                {
                    if (threadId == 0)
                    {
                        unsigned char L = T[n - 1];
                        unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned int V = (W & 0xfffff000) | L;

                            L = (unsigned char)(W >> 24); W = (W << 8) | T[i + 4];
                            P[bucket0[W & 0x000fffff]++] = V;
                        }

                        memset(bucket0, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[n - 2];
                        unsigned char P1 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char P2 = T[i];
                            bucket0[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
                            P0 = P1; P1 = P2;
                        }
                    }
                    else
                    {
                        unsigned char L = T[n - 1];
                        unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
                        for (int i = n - 1; i >= median; --i)
                        {
                            unsigned int S = W & 0x000fffff;

                            W = (W >> 8) | (L << 24); L = T[i - 1];
                            P[bucket1[S]--] = (W & 0xfffff000) | L;
                        }

                        memset(bucket1, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[median - 2];
                        unsigned char P1 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char P2 = T[i];
                            bucket1[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
                            P0 = P1; P1 = P2;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    #pragma omp single
                    {
                        for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int tmp = sum; sum += bucket0[i] + bucket1[i]; bucket0[i] = tmp; bucket1[i] = sum - 1;
                        }
                    }
                }

                {
                    if (threadId == 0)
                    {
                        if (pos < median)
                        {
                            for (int i = 0; i < pos; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                            index = bucket0[P[pos] >> 12];
                            for (int i = pos; i < median; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                        }
                        else
                        {
                            for (int i = 0; i < median; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                        }
                    }
                    else
                    {
                        if (pos >= median)
                        {
                            for (int i = n - 1; i > pos; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                            index = bucket1[P[pos] >> 12];
                            for (int i = pos; i >= median; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                        }
                        else
                        {
                            for (int i = n - 1; i >= median; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                        }
                    }
                }
            }
        }

        bsc_free(bucket1);
        return index;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st6_transform_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, int * RESTRICT bucket, int n)
{
    if (int * RESTRICT bucket0 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        if (int * RESTRICT bucket1 = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int pos, index = 0;

            for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

            #pragma omp parallel num_threads(2)
            {
                int nThreads = omp_get_num_threads();
                int threadId = omp_get_thread_num();

                if (nThreads == 1)
                {
                    index = bsc_st6_transform_serial(T, P, bucket, n);
                }
                else
                {
                    int median = n / 2;

                    {
                        if (threadId == 0)
                        {
                            unsigned int W = (T[n - 2] << 16) | (T[n - 1] << 8) | T[0];
                            for (int i = 0; i < median; ++i)
                            {
                                W = (W << 8) | T[i + 1]; bucket0[W >> 8]++;
                            }
                        }
                        else
                        {
                            unsigned int W = (T[median - 2] << 16) | (T[median - 1] << 8) | T[median];
                            for (int i = median; i < n; ++i)
                            {
                                W = (W << 8) | T[i + 1]; bucket1[W >> 8]++;
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2; ++i)
                            {
                                int tmp = sum; sum = sum + bucket0[i] + bucket1[i]; bucket[i] = bucket0[i] = tmp; bucket1[i] = sum - 1;
                            }
                        }
                        else
                        {
                            for (int sum = n, i = ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE - 1; i >= ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2; --i)
                            {
                                int tmp = sum; sum = sum - bucket0[i] - bucket1[i]; bucket[i] = bucket0[i] = sum; bucket1[i] = tmp - 1;
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            pos = bucket0[(T[3] << 16) | (T[4] << 8) | T[5]];

                            unsigned int W0 = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                            unsigned int W1 = (T[    2] << 24) | (T[    3] << 16) | (T[4] << 8) | T[5];
                            for (int i = 0; i < median; ++i)
                            {
                                W0 = (W0 << 8) | T[i + 2]; W1 = (W1 << 8) | T[i + 6];
                                P[bucket0[W1 >> 8]++] = (W0 << 8) | (W0 >> 24);
                            }
                        }
                        else
                        {
                            unsigned int W0 = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            unsigned int W1 = (T[    3] << 24) | (T[4] << 16) | (T[5] << 8) | T[6];
                            for (int i = n - 1; i >= median; --i)
                            {
                                W0 = (W0 >> 8) | (T[i - 1] << 24); W1 = (W1 >> 8) | (T[i + 3] << 24);
                                P[bucket1[W1 >> 8]--] = (W0 << 8) | (W0 >> 24);
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            memcpy(bucket1, bucket + 1, (ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2) * sizeof(int));
                        }
                        else
                        {
                            memcpy(bucket1 + ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2, bucket  + ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2 + 1, (ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE / 2- 1) * sizeof(int));
                            bucket1[ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE - 1] = n;
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            if (pos < median)
                            {
                                for (int i = 0; i < pos; ++i)
                                {
                                    T[bucket[P[i] >> 8]++] = P[i] & 0xff;
                                }
                                index = bucket[P[pos] >> 8];
                                for (int i = pos; i < median; ++i)
                                {
                                    T[bucket[P[i] >> 8]++] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = 0; i < median; ++i)
                                {
                                    T[bucket[P[i] >> 8]++] = P[i] & 0xff;
                                }
                            }
                        }
                        else
                        {
                            if (pos >= median)
                            {
                                for (int i = n - 1; i >= pos; --i)
                                {
                                    T[--bucket1[P[i] >> 8]] = P[i] & 0xff;
                                }
                                index = bucket1[P[pos] >> 8];
                                for (int i = pos - 1; i >= median; --i)
                                {
                                    T[--bucket1[P[i] >> 8]] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = n - 1; i >= median; --i)
                                {
                                    T[--bucket1[P[i] >> 8]] = P[i] & 0xff;
                                }
                            }
                        }
                    }
                }
            }

            bsc_free(bucket1); bsc_free(bucket0);
            return index;
        };
        bsc_free(bucket0);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_st3_encode(unsigned char * T, int n, int features)
{
    if (unsigned short * P = (unsigned short *)bsc_malloc(n * sizeof(unsigned short)))
    {
        if (int * bucket = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef LIBBSC_OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st3_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st3_transform_serial(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st4_encode(unsigned char * T, int n, int features)
{
    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef LIBBSC_OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st4_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st4_transform_serial(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st5_encode(unsigned char * T, int n, int features)
{
    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_zero_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef LIBBSC_OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st5_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st5_transform_serial(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st6_encode(unsigned char * T, int n, int features)
{
    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef LIBBSC_OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 6 * 1024 * 1024))
            {
                index = bsc_st6_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st6_transform_serial(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st_encode(unsigned char * T, int n, int k, int features)
{
    if ((T == NULL) || (n < 0)) return LIBBSC_BAD_PARAMETER;
    if ((k < 3) || (k > 8))     return LIBBSC_BAD_PARAMETER;
    if (n <= 1)                 return 0;

#ifdef LIBBSC_CUDA_SUPPORT

    if (features & LIBBSC_FEATURE_CUDA)
    {
        int index = bsc_st_encode_cuda(T, n, k, features);
        if (index >= LIBBSC_NO_ERROR || k >= 7) return index;
    }

#endif

    if (k == 3) return bsc_st3_encode(T, n, features);
    if (k == 4) return bsc_st4_encode(T, n, features);
    if (k == 5) return bsc_st5_encode(T, n, features);
    if (k == 6) return bsc_st6_encode(T, n, features);

    return LIBBSC_NOT_SUPPORTED;
}

static bool bsc_unst_sort_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, unsigned int * RESTRICT bucket, int n, int k)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    bool failBack = false;
    {
        for (int i = 0; i < n; ++i) count[T[i]]++;
        for (int sum = 0, c = 0; c < ALPHABET_SIZE; ++c)
        {
            if (count[c] >= 0x800000) failBack = true;

            int tmp = sum; sum += count[c]; count[c] = tmp;
            if ((int)count[c] != sum)
            {
                unsigned int * RESTRICT bucket_p = &bucket[c << 8];
                for (int i = count[c]; i < sum; ++i) bucket_p[T[i]]++;
            }
        }
    }

    for (int c = 0; c < ALPHABET_SIZE; ++c)
    {
        for (int d = 0; d < c; ++d)
        {
            int tmp = bucket[(d << 8) | c]; bucket[(d << 8) | c] = bucket[(c << 8) | d]; bucket[(c << 8) | d] = tmp;
        }
    }

    if (k == 3)
    {
        for (int sum = 0, w = 0; w < ALPHABET_SIZE * ALPHABET_SIZE; ++w)
        {
            if (bucket[w] > 0)
            {
                P[sum] = 1; sum += bucket[w];
            }
        }

        return failBack;
    }

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int sum = 0, w = 0; w < ALPHABET_SIZE * ALPHABET_SIZE; ++w)
    {
        int tmp = sum; sum += bucket[w]; bucket[w] = tmp;
        for (int i = bucket[w]; i < sum; ++i)
        {
            unsigned char c = T[i];
            if (group[c] != w)
            {
                group[c] = w; P[index[c]] = 0x80000000;
            }
            index[c]++;
        }
    }

    unsigned int mask0 = 0x80000000, mask1 = 0x40000000;
    for (int round = 4; round < k; ++round, mask0 >>= 1, mask1 >>= 1)
    {
        memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
        memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

        for (int g = 0, i = 0; i < n; ++i)
        {
            if (P[i] & mask0) g = i;

            unsigned char c = T[i];
            if (group[c] != g)
            {
                group[c] = g; P[index[c]] += mask1;
            }
            index[c]++;
        }
    }

    return failBack;
}

static void bsc_unst_reconstruct_case1_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, int n, int start)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = (c << 24) | index[c];
        }
        else
        {
            P[i] = (c << 24) | 0x800000 | group[c]; P[group[c]]++;
        }
        index[c]++;
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = u & 0x7fffff;
            u = P[p];
        }

        T[i] = u >> 24; P[p]--; p = u & 0x7fffff;
    }
}

static void bsc_unst_reconstruct_case2_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, int n, int start)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    memset(index, 0, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = (c << 24) | index[c];
        }
        else
        {
            P[i] = (c << 24) | 0x800000 | (i - group[c]); P[group[c]]++;
        }
        index[c]++;
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = p - (u & 0x7fffff);
            u = P[p];
        }

        unsigned char c = u >> 24;
        T[i] = c; P[p]--; p = (u & 0x7fffff) + count[c];
    }
}

static INLINE int bsc_unst_search(int index, unsigned int * p, unsigned int v)
{
    while (p[index] <= v) { index++; } return index;
}

#define ST_NUM_FASTBITS (10)

static void bsc_unst_reconstruct_case3_serial(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, int n, int start)
{
    unsigned char   fastbits[1 << ST_NUM_FASTBITS];
    unsigned int    index[ALPHABET_SIZE];
             int    group[ALPHABET_SIZE];

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = index[c];
        }
        else
        {
            P[i] = 0x80000000 | group[c]; P[group[c]]++;
        }
        index[c]++;
    }

    {
        int shift = 0; while (((n - 1) >> shift) >= (1 << ST_NUM_FASTBITS)) shift++;

        {
            for (int v = 0, c = 0; c < ALPHABET_SIZE; ++c)
            {
                index[c] = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
                if (count[c] != index[c])
                {
                    for (; v <= (int)((index[c] - 1) >> shift); ++v) fastbits[v] = c;
                }
            }
        }

        if (P[start] & 0x80000000)
        {
            start = P[start] & 0x7fffffff;
        }

        T[0] = bsc_unst_search(fastbits[start >> shift], index, start);
        P[start]--; start = P[start] + 1;

        for (int p = start, i = n - 1; i >= 1; --i)
        {
            unsigned int u = P[p];
            if (u & 0x80000000)
            {
                p = u & 0x7fffffff;
                u = P[p];
            }

            T[i] = bsc_unst_search(fastbits[p >> shift], index, p);
            P[p]--; p = u;
        }
    }
}

static void bsc_unst_reconstruct_serial(unsigned char * T, unsigned int * P, unsigned int * count, int n, int index, bool failBack)
{
    if (n < 0x800000)   return bsc_unst_reconstruct_case1_serial(T, P, count, n, index);
    if (!failBack)      return bsc_unst_reconstruct_case2_serial(T, P, count, n, index);
    if (failBack)       return bsc_unst_reconstruct_case3_serial(T, P, count, n, index);
}

#ifdef LIBBSC_OPENMP

static bool bsc_unst_sort_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, unsigned int * RESTRICT bucket, int n, int k)
{
    bool failBack = false;
    {
        #pragma omp parallel
        {
            unsigned int count_local[ALPHABET_SIZE];

            memset(count_local, 0, ALPHABET_SIZE * sizeof(unsigned int));

            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; ++i) count_local[T[i]]++;

            #pragma omp critical
            for (int c = 0; c < ALPHABET_SIZE; ++c) count[c] += count_local[c];
        }

        for (int sum = 0, c = 0; c < ALPHABET_SIZE; ++c)
        {
            if (count[c] >= 0x800000) failBack = true;
            int tmp = sum; sum += count[c]; count[c] = tmp;
        }

        #pragma omp parallel for schedule(static, 1)
        for (int c = 0; c < ALPHABET_SIZE; ++c)
        {
            int start = count[c], end = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
            if (start != end)
            {
                unsigned int * RESTRICT bucket_p = &bucket[c << 8];
                for (int i = start; i < end; ++i) bucket_p[T[i]]++;
            }
        }
    }

    for (int sum = 0, C0 = 0; C0 < ALPHABET_SIZE; ++C0)
    {
        for (int C1 = 0; C1 < ALPHABET_SIZE; ++C1)
        {
            if (bucket[(C1 << 8) | C0] > 0)
            {
                P[sum] = 0x80000000; sum += bucket[(C1 << 8) | C0];
            }
        }
    }

    {
        unsigned int index[ALPHABET_SIZE];

        memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
        for (int C0 = 0; C0 < ALPHABET_SIZE; ++C0)
        {
            unsigned int * RESTRICT bucket_p = &bucket[C0 << 8];
            for (int C1 = 0; C1 < ALPHABET_SIZE; ++C1)
            {
                int tmp = index[C1]; index[C1] += bucket_p[C1]; bucket_p[C1] = tmp;
            }
        }
    }

    unsigned int mask0 = 0x80000000, mask1 = 0x40000000;
    for (int round = 3; round < k; ++round, mask0 >>= 1, mask1 >>= 1)
    {
        #pragma omp parallel for schedule(static, 1)
        for (int c = 0; c < ALPHABET_SIZE; ++c)
        {
            unsigned int index[ALPHABET_SIZE]; memcpy(index, &bucket[c << 8], ALPHABET_SIZE * sizeof(unsigned int));
                     int group[ALPHABET_SIZE]; memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

            int start = count[c], end = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
            for (int g = 0, i = start; i < end; ++i)
            {
                if (P[i] & mask0) g = i;

                unsigned char c = T[i];
                if (group[c] != g)
                {
                    group[c] = g; P[index[c]] += mask1;
                }
                index[c]++;
            }
        }
    }

    return failBack;
}

static void bsc_unst_reconstruct_case1_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, unsigned int * RESTRICT bucket, int n, int start)
{
    #pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < ALPHABET_SIZE; ++c)
    {
        unsigned int index[ALPHABET_SIZE]; memcpy(index, &bucket[c << 8], ALPHABET_SIZE * sizeof(unsigned int));
                 int group[ALPHABET_SIZE]; memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

        int start = count[c], end = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
        for (int g = 0, i = start; i < end; ++i)
        {
            if (P[i] > 0) g = i;

            unsigned char c = T[i];
            if (group[c] < g)
            {
                group[c] = i; P[i] = (c << 24) | index[c];
            }
            else
            {
                P[i] = (c << 24) | 0x800000 | group[c]; P[group[c]]++;
            }
            index[c]++;
        }
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = u & 0x7fffff;
            u = P[p];
        }

        T[i] = u >> 24; P[p]--; p = u & 0x7fffff;
    }
}

static void bsc_unst_reconstruct_case2_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, unsigned int * RESTRICT bucket, int n, int start)
{
    #pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < ALPHABET_SIZE; ++c)
    {
        unsigned int index[ALPHABET_SIZE]; memcpy(index, &bucket[c << 8], ALPHABET_SIZE * sizeof(unsigned int));
                 int group[ALPHABET_SIZE]; memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

        for (int i = 0; i < ALPHABET_SIZE; ++i) index[i] -= count[i];

        int start = count[c], end = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
        for (int g = 0, i = start; i < end; ++i)
        {
            if (P[i] > 0) g = i;

            unsigned char c = T[i];
            if (group[c] < g)
            {
                group[c] = i; P[i] = (c << 24) | index[c];
            }
            else
            {
                P[i] = (c << 24) | 0x800000 | (i - group[c]); P[group[c]]++;
            }
            index[c]++;
        }
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = p - (u & 0x7fffff);
            u = P[p];
        }

        unsigned char c = u >> 24;
        T[i] = c; P[p]--; p = (u & 0x7fffff) + count[c];
    }
}

static void bsc_unst_reconstruct_case3_parallel(unsigned char * RESTRICT T, unsigned int * RESTRICT P, unsigned int * RESTRICT count, unsigned int * RESTRICT bucket, int n, int start)
{
    #pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < ALPHABET_SIZE; ++c)
    {
        unsigned int index[ALPHABET_SIZE]; memcpy(index, &bucket[c << 8], ALPHABET_SIZE * sizeof(unsigned int));
                 int group[ALPHABET_SIZE]; memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

        int start = count[c], end = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
        for (int g = 0, i = start; i < end; ++i)
        {
            if (P[i] > 0) g = i;

            unsigned char c = T[i];
            if (group[c] < g)
            {
                group[c] = i; P[i] = index[c];
            }
            else
            {
                P[i] = 0x80000000 | group[c]; P[group[c]]++;
            }
            index[c]++;
        }
    }

    unsigned char   fastbits[1 << ST_NUM_FASTBITS];
    unsigned int    index[ALPHABET_SIZE];

    {
        int shift = 0; while (((n - 1) >> shift) >= (1 << ST_NUM_FASTBITS)) shift++;

        {
            for (int v = 0, c = 0; c < ALPHABET_SIZE; ++c)
            {
                index[c] = (c + 1 < ALPHABET_SIZE) ? count[c + 1] : n;
                if (count[c] != index[c])
                {
                    for (; v <= (int)((index[c] - 1) >> shift); ++v) fastbits[v] = c;
                }
            }
        }

        if (P[start] & 0x80000000)
        {
            start = P[start] & 0x7fffffff;
        }

        T[0] = bsc_unst_search(fastbits[start >> shift], index, start);
        P[start]--; start = P[start] + 1;

        for (int p = start, i = n - 1; i >= 1; --i)
        {
            unsigned int u = P[p];
            if (u & 0x80000000)
            {
                p = u & 0x7fffffff;
                u = P[p];
            }

            T[i] = bsc_unst_search(fastbits[p >> shift], index, p);
            P[p]--; p = u;
        }
    }
}

static void bsc_unst_reconstruct_parallel(unsigned char * T, unsigned int * P, unsigned int * count, unsigned int * bucket, int n, int index, bool failBack)
{
    if (n < 0x800000)   return bsc_unst_reconstruct_case1_parallel(T, P, count, bucket, n, index);
    if (!failBack)      return bsc_unst_reconstruct_case2_parallel(T, P, count, bucket, n, index);
    if (failBack)       return bsc_unst_reconstruct_case3_parallel(T, P, count, bucket, n, index);
}

#endif

int bsc_st_decode(unsigned char * T, int n, int k, int index, int features)
{
    if ((T == NULL) || (n < 0))      return LIBBSC_BAD_PARAMETER;
    if ((index < 0) || (index >= n)) return LIBBSC_BAD_PARAMETER;
    if ((k < 3) || (k > 8))          return LIBBSC_BAD_PARAMETER;
    if (n <= 1)                      return LIBBSC_NO_ERROR;

    if (unsigned int * P = (unsigned int *)bsc_zero_malloc(n * sizeof(unsigned int)))
    {
        if (unsigned int * bucket = (unsigned int *)bsc_zero_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int)))
        {
            unsigned int count[ALPHABET_SIZE]; memset(count, 0, ALPHABET_SIZE * sizeof(unsigned int));

#ifdef LIBBSC_OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                bool failBack = bsc_unst_sort_parallel(T, P, count, bucket, n, k);
                bsc_unst_reconstruct_parallel(T, P, count, bucket, n, index, failBack);
            }
            else

#endif

            {
                bool failBack = bsc_unst_sort_serial(T, P, count, bucket, n, k);
                bsc_unst_reconstruct_serial(T, P, count, n, index, failBack);
            }

            bsc_free(bucket); bsc_free(P);
            return LIBBSC_NO_ERROR;
        };
        bsc_free(P);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

/*-----------------------------------------------------------*/
/* End                                                st.cpp */
/*-----------------------------------------------------------*/
