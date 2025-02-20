/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Sort Transform (GPU version)                              */
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

#if defined(LIBBSC_SORT_TRANSFORM_SUPPORT) && defined(LIBBSC_CUDA_SUPPORT)

#if defined(_MSC_VER)
  #pragma warning(disable : 4267)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "st.cuh"

#include "../libbsc.h"
#include "../platform/platform.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

#ifdef LIBBSC_OPENMP

omp_lock_t      st_cuda_lock;
cudaStream_t    st_cuda_stream;

int bsc_st_cuda_init(int features)
{
    if (features & LIBBSC_FEATURE_CUDA)
    {
        omp_init_lock(&st_cuda_lock);
        st_cuda_stream = NULL;
    }

    return LIBBSC_NO_ERROR;
}

#else

int bsc_st_cuda_init(int features)
{
    return LIBBSC_NO_ERROR;
}

#endif

#ifndef __CUDA_ARCH__
  #define CUDA_DEVICE_ARCH              0
#else
  #define CUDA_DEVICE_ARCH              __CUDA_ARCH__
#endif

#define CUDA_DEVICE_PADDING             1024
#define CUDA_NUM_THREADS_IN_BLOCK       256

cudaError_t bsc_cuda_safe_call(const char * filename, int line, cudaError_t result, cudaError_t status = cudaSuccess)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "\n%s(%d): bsc_cuda_safe_call failed %d: '%s'.", filename, line, result, cudaGetErrorString(result));
        fflush(stderr);
    }

    return result != cudaSuccess ? result : status;
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK)
void bsc_st567_encode_cuda_presort(unsigned char * RESTRICT T_device, unsigned long long * RESTRICT K_device)
{
    __shared__ unsigned int staging[1 + CUDA_NUM_THREADS_IN_BLOCK + 6];

    unsigned int * RESTRICT thread_staging = &staging[threadIdx.x];
    {
        int index = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK + threadIdx.x;

        {
                                 thread_staging[0                        ] = T_device[index - 1                            ];
            if (threadIdx.x < 7) thread_staging[CUDA_NUM_THREADS_IN_BLOCK] = T_device[index - 1 + CUDA_NUM_THREADS_IN_BLOCK];

            __syncthreads();
        }

        {
            unsigned int lo = __byte_perm(thread_staging[4], thread_staging[5], 0x0411) | __byte_perm(thread_staging[6], thread_staging[7], 0x1104);
            unsigned int hi = __byte_perm(thread_staging[0], thread_staging[1], 0x0411) | __byte_perm(thread_staging[2], thread_staging[3], 0x1104);

            K_device[index] = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);
        }
    }
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK)
void bsc_st8_encode_cuda_presort(unsigned char * RESTRICT T_device, unsigned long long * RESTRICT K_device, unsigned char * RESTRICT V_device)
{
    __shared__ unsigned int staging[1 + CUDA_NUM_THREADS_IN_BLOCK + 7];

    unsigned int * RESTRICT thread_staging = &staging[threadIdx.x];
    {
        int index = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK + threadIdx.x;

        {
                                 thread_staging[0                        ] = T_device[index - 1                            ];
            if (threadIdx.x < 8) thread_staging[CUDA_NUM_THREADS_IN_BLOCK] = T_device[index - 1 + CUDA_NUM_THREADS_IN_BLOCK];

            __syncthreads();
        }

        {
            unsigned int lo = __byte_perm(thread_staging[5], thread_staging[6], 0x0411) | __byte_perm(thread_staging[7], thread_staging[8], 0x1104);
            unsigned int hi = __byte_perm(thread_staging[1], thread_staging[2], 0x0411) | __byte_perm(thread_staging[3], thread_staging[4], 0x1104);

            K_device[index] = (((unsigned long long)hi) << 32) | ((unsigned long long)lo); V_device[index] = thread_staging[0];
        }
    }
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK)
void bsc_st567_encode_cuda_postsort(unsigned char * RESTRICT T_device, unsigned long long * RESTRICT K_device, unsigned long long lookup, int * RESTRICT I_device)
{
    int index = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK + threadIdx.x;
    if (K_device[index] == lookup) { atomicMin(I_device, index); }

    T_device[index] = (unsigned char)(K_device[index] >> 56);
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK)
void bsc_st8_encode_cuda_postsort(unsigned long long * RESTRICT K_device, unsigned long long lookup, int * RESTRICT I_device)
{
    int index = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK + threadIdx.x;
    if (K_device[index] == lookup) { atomicMin(I_device, index); }
}

int bsc_st567_encode_cuda(unsigned char * T, unsigned char * T_device, int n, int num_blocks, int k, cudaStream_t st_cuda_stream)
{
    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned long long * K_device = NULL;
        unsigned long long * K_device_sorted = NULL;

        if (bsc_cuda_safe_call(__FILE__, __LINE__, cudaMallocAsync((void **)&K_device, 2 * (n + 2 * CUDA_DEVICE_PADDING) * sizeof(unsigned long long), st_cuda_stream)) == cudaSuccess)
        {
            index              = LIBBSC_GPU_ERROR;
            cudaError_t status = cudaSuccess;

            bsc_st567_encode_cuda_presort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK, 0, st_cuda_stream>>>(T_device, K_device);

            if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
            {
                K_device_sorted = K_device + ((n + 2 * CUDA_DEVICE_PADDING) / CUDA_DEVICE_PADDING) * CUDA_DEVICE_PADDING;

                cub::DoubleBuffer<unsigned long long> d_keys(K_device, K_device_sorted);

                void * d_temp_storage = NULL; size_t temp_storage_bytes = 0;

                status = bsc_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n, (7 - k) * 8, 56, st_cuda_stream), status);
                if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                {
                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMallocAsync(&d_temp_storage, temp_storage_bytes, st_cuda_stream), status);
                    if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                    {
                        status = bsc_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n, (7 - k) * 8, 56, st_cuda_stream), status);

                        if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                        {
                            K_device_sorted = d_keys.Current();

                            unsigned long long lookup;
                            {
                                unsigned int lo = (T[3    ] << 24) | (T[4] << 16) | (T[5] << 8) | T[6];
                                unsigned int hi = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];

                                lookup = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);

                                status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T_device - sizeof(int), &n, sizeof(int), cudaMemcpyHostToDevice, st_cuda_stream), status);
                            }

                            if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                            {
                                bsc_st567_encode_cuda_postsort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK, 0, st_cuda_stream>>>(T_device, K_device_sorted, lookup, (int *)(T_device - sizeof(int)));

                                if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                                {
                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T_device + n, T_device - sizeof(int), sizeof(int), cudaMemcpyDeviceToDevice, st_cuda_stream), status);
                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T, T_device, n + sizeof(int), cudaMemcpyDeviceToHost, st_cuda_stream), status);
                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(st_cuda_stream), status);
                                }

                                status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaFreeAsync(d_temp_storage, st_cuda_stream), status);
                                status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaFreeAsync(K_device, st_cuda_stream), status);

                                if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                                {
                                    index = *(int *)(T + n);
                                }

                                return index;
                            }
                        }

                        cudaFreeAsync(d_temp_storage, st_cuda_stream);
                    }
                }
            }

            cudaFreeAsync(K_device, st_cuda_stream);
        }
    }

    return index;
}

int bsc_st8_encode_cuda(unsigned char * T, unsigned char * T_device, int n, int num_blocks, cudaStream_t st_cuda_stream)
{
    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned char * V_device = NULL;
        unsigned char * V_device_sorted = NULL;

        if (bsc_cuda_safe_call(__FILE__, __LINE__, cudaMallocAsync((void **)&V_device, 2 * (n + 2 * CUDA_DEVICE_PADDING) * sizeof(unsigned char), st_cuda_stream)) == cudaSuccess)
        {
            unsigned long long * K_device = NULL;
            unsigned long long * K_device_sorted = NULL;

            if (bsc_cuda_safe_call(__FILE__, __LINE__, cudaMallocAsync((void **)&K_device, 2 * (n + 2 * CUDA_DEVICE_PADDING) * sizeof(unsigned long long), st_cuda_stream)) == cudaSuccess)
            {
                index              = LIBBSC_GPU_ERROR;
                cudaError_t status = cudaSuccess;

                bsc_st8_encode_cuda_presort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK, 0, st_cuda_stream>>>(T_device, K_device, V_device);

                if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                {
                    K_device_sorted = K_device + ((n + 2 * CUDA_DEVICE_PADDING) / CUDA_DEVICE_PADDING) * CUDA_DEVICE_PADDING;
                    V_device_sorted = V_device + ((n + 2 * CUDA_DEVICE_PADDING) / CUDA_DEVICE_PADDING) * CUDA_DEVICE_PADDING;

                    cub::DoubleBuffer<unsigned long long> d_keys(K_device, K_device_sorted);
                    cub::DoubleBuffer<unsigned char>      d_values(V_device, V_device_sorted);

                    void * d_temp_storage = NULL; size_t temp_storage_bytes = 0;

                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n, 0, 64, st_cuda_stream), status);
                    if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                    {
                        status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMallocAsync(&d_temp_storage, temp_storage_bytes, st_cuda_stream), status);
                        if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                        {
                            status = bsc_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n, 0, 64, st_cuda_stream), status);

                            if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                            {
                                K_device_sorted = d_keys.Current();
                                V_device_sorted = d_values.Current();

                                unsigned long long lookup;
                                {
                                    unsigned int lo = (T[4] << 24) | (T[5] << 16) | (T[6] << 8) | T[7];
                                    unsigned int hi = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];

                                    lookup = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);

                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(V_device_sorted + ((n + sizeof(int) - 1) / sizeof(int)) * sizeof(int), &n, sizeof(int), cudaMemcpyHostToDevice, st_cuda_stream), status);
                                }

                                if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                                {
                                    bsc_st8_encode_cuda_postsort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK, 0, st_cuda_stream>>>(K_device_sorted, lookup, (int *)(V_device_sorted + ((n + sizeof(int) - 1) / sizeof(int)) * sizeof(int)));

                                    if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                                    {
                                        status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T, V_device_sorted, n + 2 * sizeof(int), cudaMemcpyDeviceToHost, st_cuda_stream), status);
                                        status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(st_cuda_stream), status);
                                    }

                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaFreeAsync(d_temp_storage, st_cuda_stream), status);
                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaFreeAsync(K_device, st_cuda_stream), status);
                                    status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaFreeAsync(V_device, st_cuda_stream), status);

                                    if (bsc_cuda_safe_call(__FILE__, __LINE__, status) == cudaSuccess)
                                    {
                                        index = *(int *)(T + ((n + sizeof(int) - 1) / sizeof(int)) * sizeof(int));
                                    }

                                    return index;
                                }
                            }

                            cudaFreeAsync(d_temp_storage, st_cuda_stream);
                        }
                    }
                }

                cudaFreeAsync(K_device, st_cuda_stream);
            }

            cudaFreeAsync(V_device, st_cuda_stream);
        }
    }

    return index;
}

int bsc_st_encode_cuda(unsigned char * T, int n, int k, int features)
{
    if ((T == NULL) || (n < 0)) return LIBBSC_BAD_PARAMETER;
    if ((k < 5) || (k > 8))     return LIBBSC_BAD_PARAMETER;
    if (n <= 1)                 return 0;

    int num_blocks = 1;
    {
        cudaDeviceProp deviceProperties;
        {
            int deviceId; if (cudaGetDevice(&deviceId) != cudaSuccess || cudaGetDeviceProperties(&deviceProperties, deviceId) != cudaSuccess)
            {
                return LIBBSC_GPU_NOT_SUPPORTED;
            }
        }

        if (deviceProperties.major * 10 + deviceProperties.minor < 35) return LIBBSC_GPU_NOT_SUPPORTED;

        num_blocks = (n + CUDA_NUM_THREADS_IN_BLOCK - 1) / CUDA_NUM_THREADS_IN_BLOCK;
    }

    #ifdef LIBBSC_OPENMP
        omp_set_lock(&st_cuda_lock);
    #else
        cudaStream_t st_cuda_stream = NULL;
    #endif

    if (st_cuda_stream == NULL)
    {
        if (bsc_cuda_safe_call(__FILE__, __LINE__, cudaStreamCreate(&st_cuda_stream)) != cudaSuccess)
        {
            st_cuda_stream = NULL;
        }
    }

    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned char * T_device = NULL;
        if (st_cuda_stream != NULL && cudaMallocAsync((void **)&T_device, n + 2 * CUDA_DEVICE_PADDING, st_cuda_stream) == cudaSuccess)
        {
            index = LIBBSC_GPU_ERROR;

            cudaError_t status = cudaSuccess;
            status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T_device + CUDA_DEVICE_PADDING    , T                             , n                  , cudaMemcpyHostToDevice  , st_cuda_stream), status);
            status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T_device + CUDA_DEVICE_PADDING + n, T_device + CUDA_DEVICE_PADDING, CUDA_DEVICE_PADDING, cudaMemcpyDeviceToDevice, st_cuda_stream), status);
            status = bsc_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(T_device                          , T_device + n                  , CUDA_DEVICE_PADDING, cudaMemcpyDeviceToDevice, st_cuda_stream), status);

            if (status == cudaSuccess)
            {
                if (k >= 5 && k <= 7) index = bsc_st567_encode_cuda(T, T_device + CUDA_DEVICE_PADDING, n, num_blocks, k, st_cuda_stream);
                if (k == 8)           index = bsc_st8_encode_cuda  (T, T_device + CUDA_DEVICE_PADDING, n, num_blocks   , st_cuda_stream);
            }

            cudaFreeAsync(T_device, st_cuda_stream);
        }
    }

    #ifdef LIBBSC_OPENMP
        omp_unset_lock(&st_cuda_lock);
    #else
        if (st_cuda_stream != NULL)
        {
            bsc_cuda_safe_call(__FILE__, __LINE__, cudaStreamDestroy(st_cuda_stream));
        }        
    #endif

    return index;
}

#endif

/*-----------------------------------------------------------*/
/* End                                                 st.cu */
/*-----------------------------------------------------------*/
