/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Burrows Wheeler Transform                                 */
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

#include "bwt.h"

#include "../platform/platform.h"
#include "../libbsc.h"

#include "libsais/libsais.h"

int bsc_bwt_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features)
{
    if (int * RESTRICT A = (int *)bsc_malloc(n * sizeof(int)))
    {
        int mod = n / 8;
        {
            mod |= mod >> 1;  mod |= mod >> 2;
            mod |= mod >> 4;  mod |= mod >> 8;
            mod |= mod >> 16; mod >>= 1;
        }

#ifdef LIBBSC_OPENMP
        int index = libsais_bwt_aux_omp(T, T, A, n, 0, NULL, mod + 1, indexes, (features & LIBBSC_FEATURE_MULTITHREADING) > 0 ? 0 : 1);
#else
        int index = libsais_bwt_aux(T, T, A, n, 0, NULL, mod + 1, indexes);
#endif

        bsc_free(A);

        switch (index)
        {
            case -1 : return LIBBSC_BAD_PARAMETER;
            case -2 : return LIBBSC_NOT_ENOUGH_MEMORY;
        }

        num_indexes[0] = (unsigned char)((n - 1) / (mod + 1));
        index = indexes[0]; for (int t = 0; t < num_indexes[0]; ++t) indexes[t] = indexes[t + 1] - 1;

        return index;
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_bwt_decode(unsigned char * T, int n, int index, unsigned char num_indexes, int * indexes, int features)
{
    if ((T == NULL) || (n < 0) || (index <= 0) || (index > n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }
    if (int * P = (int *)bsc_malloc((n + 1) * sizeof(int)))
    {
        int mod = n / 8;
        {
            mod |= mod >> 1;  mod |= mod >> 2;
            mod |= mod >> 4;  mod |= mod >> 8;
            mod |= mod >> 16; mod >>= 1;
        }

        if (num_indexes == (unsigned char)((n - 1) / (mod + 1)) && indexes != NULL)
        {
            int I[256]; I[0] = index; for (int t = 0; t < num_indexes; ++t) { I[t + 1] = indexes[t] + 1; }

#ifdef LIBBSC_OPENMP
            index = libsais_unbwt_aux_omp(T, T, P, n, NULL, mod + 1, I, (features & LIBBSC_FEATURE_MULTITHREADING) > 0 ? 0 : 1);
#else
            index = libsais_unbwt_aux(T, T, P, n, NULL, mod + 1, I);
#endif
        }
        else
        {
#ifdef LIBBSC_OPENMP
            index = libsais_unbwt_omp(T, T, P, n, NULL, index, (features & LIBBSC_FEATURE_MULTITHREADING) > 0 ? 0 : 1);
#else
            index = libsais_unbwt(T, T, P, n, NULL, index);
#endif
        }

        bsc_free(P);

        switch (index)
        {
            case -1 : return LIBBSC_BAD_PARAMETER;
            case -2 : return LIBBSC_NOT_ENOUGH_MEMORY;
        }       

        return LIBBSC_NO_ERROR;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-----------------------------------------------------------*/
/* End                                               bwt.cpp */
/*-----------------------------------------------------------*/
