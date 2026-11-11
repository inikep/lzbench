/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file aocl_threading.c
 *
 *  @brief OpenMP backend implementation of the AOCL threading API
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCL_USE_TBB

#include <omp.h>
#include "aocl_threading.h"

AOCL_INT32 aocl_parallel_for(AOCL_UINT32 num_threads,
                             aocl_parallel_work_fn_t work_fn,
                             void *context)
{
    if (num_threads == 0 || work_fn == NULL)
        return -1;

    /* C callback never throws, so the OpenMP region always completes. */
    #pragma omp parallel num_threads(num_threads)
    {
        work_fn((AOCL_UINT32)omp_get_thread_num(), context);
    }
    return 0;
}

AOCL_UINT32 aocl_get_max_threads(void)
{
    /* AOCL_NUM_THREADS overrides the OpenMP default so AOCL threading can be
     * sized independently of OMP_NUM_THREADS. When unset, defer to the OpenMP
     * runtime (which already honors OMP_NUM_THREADS and omp_set_num_threads()). */
    AOCL_UINT32 env_threads = aocl_env_num_threads();
    if (env_threads > 0)
        return env_threads;

    return (AOCL_UINT32)omp_get_max_threads();
}

#endif /* !AOCL_USE_TBB */
