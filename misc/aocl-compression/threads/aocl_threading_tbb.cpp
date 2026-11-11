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

/** @file aocl_threading_tbb.cpp
 *
 *  @brief TBB (oneTBB) backend implementation of the AOCL threading API
 *
 *  @author S. Biplab Raut
 */

#ifdef AOCL_USE_TBB

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <thread>
#include <cstdlib>
#include <memory>

#include "aocl_threading.h"

extern "C" {

AOCL_INT32 aocl_parallel_for(AOCL_UINT32 num_threads,
                             aocl_parallel_work_fn_t work_fn,
                             void *context)
{
    if (num_threads == 0 || work_fn == nullptr)
        return -1;

    /* The arena is cached per calling thread (thread_local) rather than in a
     * shared global. This keeps the cross-call reuse optimization while making
     * concurrent calls from multiple application threads race-free: there is no
     * shared mutable state, so one thread rebuilding its arena can never free an
     * arena another thread is executing in. The unique_ptr frees the arena at
     * thread exit, so the cache does not leak. */
    static thread_local std::unique_ptr<tbb::task_arena> t_arena;
    static thread_local AOCL_UINT32 t_arena_threads = 0;

    /* Contain any C++ exception (e.g. TBB std::bad_alloc): it must not unwind
     * across this extern "C" boundary. Report it via a non-zero return. */
    try {
        if (!t_arena || t_arena_threads != num_threads) {
            t_arena.reset(new tbb::task_arena(static_cast<int>(num_threads)));
            t_arena_threads = num_threads;
        }
        t_arena->execute([&] {
            tbb::parallel_for(
                AOCL_UINT32(0), num_threads, AOCL_UINT32(1),
                [=](AOCL_UINT32 thread_id) {
                    work_fn(thread_id, context);
                }
            );
        });
    } catch (...) {
        return -1;
    }
    return 0;
}

AOCL_UINT32 aocl_get_max_threads(void)
{
    /* AOCL_NUM_THREADS (resolved by the shared helper, identical to the OpenMP
     * backend) takes precedence. There is no OpenMP runtime under TBB, so fall
     * back to OMP_NUM_THREADS and then the hardware concurrency. */
    AOCL_UINT32 env_threads = aocl_env_num_threads();
    if (env_threads > 0)
        return env_threads;

    const char *env = std::getenv("OMP_NUM_THREADS");
    if (env) {
        int val = std::atoi(env);
        if (val > 0)
            return (AOCL_UINT32)val;
    }
    unsigned n = std::thread::hardware_concurrency();
    return (n > 0) ? (AOCL_UINT32)n : 1;
}

} /* extern "C" */

#endif /* AOCL_USE_TBB */
