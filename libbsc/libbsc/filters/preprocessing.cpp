/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Data preprocessing functions                              */
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

int bsc_reverse_block(unsigned char * T, int n, int features)
{

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        #pragma omp parallel for
        for (int i = 0; i < n / 2; ++i)
        {
            unsigned char tmp = T[i]; T[i] = T[n - 1 - i]; T[n - 1 - i] = tmp;
        }
    }
    else

#endif

    {
        for (int i = 0, j = n - 1; i < j; ++i, --j)
        {
            unsigned char tmp = T[i]; T[i] = T[j]; T[j] = tmp;
        }
    }

    return LIBBSC_NO_ERROR;
}

int bsc_reorder_forward(unsigned char * T, int n, int recordSize, int features)
{
    if (recordSize <= 0) return LIBBSC_BAD_PARAMETER;
    if (recordSize == 1) return LIBBSC_NO_ERROR;

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n))
    {
        memcpy(buffer, T, n);

        unsigned char * RESTRICT S = buffer;
        unsigned char * RESTRICT D = T;

        int chunk = (n / recordSize);

#ifdef LIBBSC_OPENMP

        if (features & LIBBSC_FEATURE_MULTITHREADING)
        {
            switch (recordSize)
            {
                case 2:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[2 * i]; D[chunk + i] = S[2 * i + 1]; } break;
                case 3:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[3 * i]; D[chunk + i] = S[3 * i + 1]; D[chunk * 2 + i] = S[3 * i + 2]; } break;
                case 4:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[4 * i]; D[chunk + i] = S[4 * i + 1]; D[chunk * 2 + i] = S[4 * i + 2]; D[chunk * 3 + i] = S[4 * i + 3]; } break;
                default:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j * chunk + i] = S[recordSize * i + j]; }
            }
        }
        else

#endif

        {
            switch (recordSize)
            {
                case 2: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D++; S += 2; } break;
                case 3: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D[chunk * 2] = S[2]; D++; S += 3; } break;
                case 4: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D[chunk * 2] = S[2]; D[chunk * 3] = S[3]; D++; S += 4; } break;
                default:
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j * chunk] = S[j]; D++; S += recordSize; }
            }
        }

        bsc_free(buffer); return LIBBSC_NO_ERROR;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_reorder_reverse(unsigned char * T, int n, int recordSize, int features)
{
    if (recordSize <= 0) return LIBBSC_BAD_PARAMETER;
    if (recordSize == 1) return LIBBSC_NO_ERROR;

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n))
    {
        memcpy(buffer, T, n);

        unsigned char * RESTRICT S = buffer;
        unsigned char * RESTRICT D = T;

        int chunk = (n / recordSize);

#ifdef LIBBSC_OPENMP

        if (features & LIBBSC_FEATURE_MULTITHREADING)
        {
            switch (recordSize)
            {
                case 2:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[2 * i] = S[i]; D[2 * i + 1] = S[chunk + i]; } break;
                case 3:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[3 * i] = S[i]; D[3 * i + 1] = S[chunk + i]; D[3 * i + 2] = S[chunk * 2 + i]; } break;
                case 4:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[4 * i] = S[i]; D[4 * i + 1] = S[chunk + i]; D[4 * i + 2] = S[chunk * 2 + i]; D[4 * i + 3] = S[chunk * 3 + i]; } break;
                default:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[recordSize * i + j] = S[j * chunk + i]; }
            }
        }
        else

#endif

        {
            switch (recordSize)
            {
                case 2: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D += 2; S++; } break;
                case 3: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D[2] = S[chunk * 2]; D += 3; S++; } break;
                case 4: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D[2] = S[chunk * 2]; D[3] = S[chunk * 3]; D += 4; S++; } break;
                default:
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j] = S[j * chunk]; D += recordSize; S++; }
            }
        }

        bsc_free(buffer); return LIBBSC_NO_ERROR;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-------------------------------------------------*/
/* End                           preprocessing.cpp */
/*-------------------------------------------------*/
