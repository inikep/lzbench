/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Burrows Wheeler Transform                                 */
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

#include "bwt.h"

#include "../platform/platform.h"
#include "../libbsc.h"

#include "libcubwt/libcubwt.cuh"
#include "libsais/libsais.h"

#if defined(LIBBSC_CUDA_SUPPORT) && defined(LIBBSC_OPENMP)

omp_lock_t bwt_cuda_lock;
void *     bwt_cuda_device_storage = NULL;
int        bwt_cuda_device_storage_size = 0;

int bsc_bwt_init(int features)
{
    if (features & LIBBSC_FEATURE_CUDA)
    {
        omp_init_lock(&bwt_cuda_lock);
    }

    return LIBBSC_NO_ERROR;
}

#else

int bsc_bwt_init(int features)
{
    return LIBBSC_NO_ERROR;
}

#endif

int bsc_bwt_gpu_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features)
{
    int index = -1;

    if (features & LIBBSC_FEATURE_CUDA)
    {
#ifdef LIBBSC_CUDA_SUPPORT
        if (num_indexes != NULL && indexes != NULL)
        {
            int I[256];

            int mod = n / 8;
            {
                mod |= mod >> 1;  mod |= mod >> 2;
                mod |= mod >> 4;  mod |= mod >> 8;
                mod |= mod >> 16; mod >>= 1;
            }

#ifdef LIBBSC_OPENMP
            omp_set_lock(&bwt_cuda_lock);

            if (bwt_cuda_device_storage_size < n)
            {
                if (bwt_cuda_device_storage != NULL)
                {
                    libcubwt_free_device_storage(bwt_cuda_device_storage);

                    bwt_cuda_device_storage = NULL;
                    bwt_cuda_device_storage_size = 0;
                }

                if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, n + (n / 32)) == LIBCUBWT_NO_ERROR)
                {
                    bwt_cuda_device_storage_size = n + (n / 32);
                }
            }

            if (bwt_cuda_device_storage_size >= n)
            {
                index = (int)libcubwt_bwt_aux(bwt_cuda_device_storage, T, T, n, mod + 1, (unsigned int *)I);
            } 

            omp_unset_lock(&bwt_cuda_lock);
#else
            void * bwt_cuda_device_storage = NULL;

            if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, n) == LIBCUBWT_NO_ERROR)
            {
                index = (int)libcubwt_bwt_aux(bwt_cuda_device_storage, T, T, n, mod + 1, (unsigned int *)I);

                libcubwt_free_device_storage(bwt_cuda_device_storage);
            }
#endif

            if (index == 0)
            {
                num_indexes[0] = (unsigned char)((n - 1) / (mod + 1));
                index = I[0]; for (int t = 0; t < num_indexes[0]; ++t) indexes[t] = I[t + 1] - 1;
            }
        }
        else
        {
#ifdef LIBBSC_OPENMP
            omp_set_lock(&bwt_cuda_lock);

            if (bwt_cuda_device_storage_size < n)
            {
                if (bwt_cuda_device_storage != NULL)
                {
                    libcubwt_free_device_storage(bwt_cuda_device_storage);

                    bwt_cuda_device_storage = NULL;
                    bwt_cuda_device_storage_size = 0;
                }

                if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, n + (n / 32)) == LIBCUBWT_NO_ERROR)
                {
                    bwt_cuda_device_storage_size = n + (n / 32);
                }
            }

            if (bwt_cuda_device_storage_size >= n)
            {
                index = (int)libcubwt_bwt(bwt_cuda_device_storage, T, T, n);
            } 

            omp_unset_lock(&bwt_cuda_lock);
#else
            void * bwt_cuda_device_storage = NULL;

            if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, n) == LIBCUBWT_NO_ERROR)
            {
                index = (int)libcubwt_bwt(bwt_cuda_device_storage, T, T, n);

                libcubwt_free_device_storage(bwt_cuda_device_storage);
            }
#endif
        }
#endif
    }

    return index;
}


int bsc_bwt_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features)
{
    int index = bsc_bwt_gpu_encode(T, n, num_indexes, indexes, features);
    if (index >= 0)
    {
        return index;
    }

    if (int * RESTRICT A = (int *)bsc_malloc(n * sizeof(int)))
    {
        if (num_indexes != NULL && indexes != NULL)
        {
            int I[256];

            int mod = n / 8;
            {
                mod |= mod >> 1;  mod |= mod >> 2;
                mod |= mod >> 4;  mod |= mod >> 8;
                mod |= mod >> 16; mod >>= 1;
            }

#ifdef LIBBSC_OPENMP
            index = libsais_bwt_aux_omp(T, T, A, n, 0, NULL, mod + 1, I, (features & LIBBSC_FEATURE_MULTITHREADING) > 0 ? 0 : 1);
#else
            index = libsais_bwt_aux(T, T, A, n, 0, NULL, mod + 1, I);
#endif

            if (index == 0)
            {
                num_indexes[0] = (unsigned char)((n - 1) / (mod + 1));
                index = I[0]; for (int t = 0; t < num_indexes[0]; ++t) indexes[t] = I[t + 1] - 1;
            }
        }
        else
        {
#ifdef LIBBSC_OPENMP
            index = libsais_bwt_omp(T, T, A, n, 0, NULL, (features & LIBBSC_FEATURE_MULTITHREADING) > 0 ? 0 : 1);
#else
            index = libsais_bwt(T, T, A, n, 0, NULL);
#endif
        }

        bsc_free(A);

        switch (index)
        {
            case -1 : return LIBBSC_BAD_PARAMETER;
            case -2 : return LIBBSC_NOT_ENOUGH_MEMORY;
        }

        return index;
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_bwt_gpu_decode(unsigned char * T, int n, int index, int features)
{
    int result = -1;

#ifdef LIBBSC_CUDA_SUPPORT
    if (features & LIBBSC_FEATURE_CUDA)
    {
        int storage_approx_length = (n / 3) | 0x1fffff;

#ifdef LIBBSC_OPENMP
        omp_set_lock(&bwt_cuda_lock);

        if (bwt_cuda_device_storage_size < storage_approx_length)
        {
            if (bwt_cuda_device_storage != NULL)
            {
                libcubwt_free_device_storage(bwt_cuda_device_storage);

                bwt_cuda_device_storage = NULL;
                bwt_cuda_device_storage_size = 0;
            }

            if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, storage_approx_length + (storage_approx_length / 32)) == LIBCUBWT_NO_ERROR)
            {
                bwt_cuda_device_storage_size = storage_approx_length + (storage_approx_length / 32);
            }
        }

        if (bwt_cuda_device_storage_size >= storage_approx_length)
        {
            result = (int)libcubwt_unbwt(bwt_cuda_device_storage, T, T, n, NULL, index);
        } 

        omp_unset_lock(&bwt_cuda_lock);
#else
        void * bwt_cuda_device_storage = NULL;

        if (libcubwt_allocate_device_storage(&bwt_cuda_device_storage, storage_approx_length) == LIBCUBWT_NO_ERROR)
        {
            result = (int)libcubwt_unbwt(bwt_cuda_device_storage, T, T, n, NULL, index);

            libcubwt_free_device_storage(bwt_cuda_device_storage);
        }
#endif
    }
#endif

    return result;
}

int bsc_bwt_decode(unsigned char * T, int n, int index, unsigned char num_indexes, int * indexes, int features)
{
    if ((T == NULL) || (n < 0) || (index <= 0) || (index > n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1 || bsc_bwt_gpu_decode(T, n, index, features) == 0)
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
