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

/** @file aocl_threading.h
 *
 *  @brief Backend-agnostic threading API for AOCL compression library
 *
 *  Provides a C-callable abstraction over OpenMP or TBB threading backends.
 *  The active backend is selected at compile time via AOCL_USE_TBB.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCL_THREADING_H
#define AOCL_THREADING_H

#include <stdlib.h>
#include "api/types.h"

/**
 * Returns the AOCL-specific thread-count override requested via the
 * AOCL_NUM_THREADS environment variable, or 0 if it is unset or not a
 * positive integer.
 *
 * This override is honored by every threading backend so the number of
 * threads used by AOCL can be controlled independently of any co-resident
 * OpenMP runtime (which is driven by OMP_NUM_THREADS and may be shared with
 * other libraries in the same process). When it returns 0, each backend falls
 * back to its native default (OpenMP: omp_get_max_threads(); TBB: OMP_NUM_THREADS
 * then hardware concurrency). Defined as static inline so both the C (OpenMP)
 * and C++ (TBB) backend translation units resolve AOCL_NUM_THREADS identically.
 */
static inline AOCL_UINT32 aocl_env_num_threads(void)
{
    const char *env = getenv("AOCL_NUM_THREADS");
    if (env)
    {
        int val = atoi(env);
        if (val > 0)
            return (AOCL_UINT32)val;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*aocl_parallel_work_fn_t)(AOCL_UINT32 thread_id, void *context);

/**
 * Execute work_fn for each thread_id in [0, num_threads) in parallel.
 * All invocations may run concurrently. Returns after all complete.
 *
 * Returns 0 on success, non-zero if the region did not complete (invalid
 * arguments, or a TBB exception contained at this C boundary). Callers whose
 * phase has no other status channel (e.g. the compress copy-out) must treat a
 * non-zero return as failure.
 */
AOCL_INT32 aocl_parallel_for(AOCL_UINT32 num_threads,
                             aocl_parallel_work_fn_t work_fn,
                             void *context);

/**
 * Returns the maximum number of threads available for parallel execution.
 *
 * Resolution order (same for every backend): the AOCL_NUM_THREADS environment
 * variable (if set to a positive integer) takes precedence, otherwise the
 * backend's native default is used (OpenMP: omp_get_max_threads(), which
 * reflects OMP_NUM_THREADS/omp_set_num_threads(); TBB: OMP_NUM_THREADS then the
 * hardware concurrency).
 */
AOCL_UINT32 aocl_get_max_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* AOCL_THREADING_H */
