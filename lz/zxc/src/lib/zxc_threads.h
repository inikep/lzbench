/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_threads.h
 * @brief POSIX-shaped threading layer shared by the stream engine
 *        (zxc_driver.c) and the seekable reader (zxc_seekable.c).
 *
 * On POSIX this is @c <pthread.h> verbatim. On Windows it maps the handful of
 * primitives both engines use onto the Win32 API under their POSIX names, so
 * one body of threading logic compiles everywhere. Include this instead of
 * @c <pthread.h>.
 */

#ifndef ZXC_THREADS_H
#define ZXC_THREADS_H

#include "zxc_internal.h"

// LCOV_EXCL_START - Windows platform layer, not reachable on POSIX CI
#if defined(_WIN32)

#include <process.h> /* _beginthreadex */
#include <windows.h>

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;

#define pthread_mutex_init(m, a) InitializeCriticalSection(m)
#define pthread_mutex_destroy(m) DeleteCriticalSection(m)
#define pthread_mutex_lock(m) EnterCriticalSection(m)
#define pthread_mutex_unlock(m) LeaveCriticalSection(m)

#define pthread_cond_init(c, a) InitializeConditionVariable(c)
#define pthread_cond_destroy(c) (void)(0)
#define pthread_cond_wait(c, m) SleepConditionVariableCS(c, m, INFINITE)
#define pthread_cond_signal(c) WakeConditionVariable(c)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)

/**
 * @brief Trampoline payload bridging the POSIX @c void*(*)(void*) worker
 *        signature to the @c _beginthreadex entry point.
 *
 * Heap-allocated by the @c pthread_create shim and freed by
 * @ref zxc_win_thread_entry once the captured worker has started.
 */
typedef struct {
    void* (*func)(void*); /* worker to invoke */
    void* arg;            /* argument forwarded to @c func */
} zxc_win_thread_arg_t;

/**
 * @brief @c _beginthreadex entry point: unpacks the trampoline payload, frees
 *        it, then runs the captured POSIX-style worker.
 *
 * @param[in] p  Heap @ref zxc_win_thread_arg_t handed over by the creator;
 *               ownership transfers to this function.
 * @return Always 0 (the worker's @c void* result is discarded, as on POSIX).
 */
static inline unsigned __stdcall zxc_win_thread_entry(void* p) {
    zxc_win_thread_arg_t* a = (zxc_win_thread_arg_t*)p;
    void* (*f)(void*) = a->func;
    void* arg = a->arg;
    ZXC_FREE(a);
    f(arg);
    return 0;
}

/**
 * @brief @c pthread_create shim: spawns @p start_routine(@p arg) via
 *        @c _beginthreadex, matching the POSIX prototype.
 *
 * @param[out] thread        Receives the thread handle on success.
 * @param[in]  attr          Unused (POSIX attribute object); ignored.
 * @param[in]  start_routine Worker to run on the new thread.
 * @param[in]  arg           Opaque argument forwarded to @p start_routine.
 * @return 0 on success, @ref ZXC_ERROR_MEMORY on allocation or spawn failure.
 */
static inline int pthread_create(pthread_t* thread, const void* attr, void* (*start_routine)(void*),
                                 void* arg) {
    (void)attr;
    zxc_win_thread_arg_t* wrapper = ZXC_MALLOC(sizeof(zxc_win_thread_arg_t));
    if (UNLIKELY(!wrapper)) return ZXC_ERROR_MEMORY;
    wrapper->func = start_routine;
    wrapper->arg = arg;
    uintptr_t handle = _beginthreadex(NULL, 0, zxc_win_thread_entry, wrapper, 0, NULL);
    if (UNLIKELY(handle == 0)) {
        ZXC_FREE(wrapper);
        return ZXC_ERROR_MEMORY;
    }
    *thread = (HANDLE)handle;
    return 0;
}

/**
 * @brief @c pthread_join shim: blocks until @p thread finishes, then closes its
 *        handle.
 *
 * @param[in] thread  Handle from a successful @c pthread_create.
 * @param[in] retval  Unused (POSIX exit-value out-param); ignored.
 * @return Always 0.
 */
static inline int pthread_join(pthread_t thread, void** retval) {
    (void)retval;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

/**
 * @brief Returns the number of logical processors reported by the OS.
 * @return Processor count from @c GetSystemInfo (always >= 1 in practice).
 */
static ZXC_ALWAYS_INLINE int zxc_num_procs(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}

#else /* POSIX */
// LCOV_EXCL_STOP

#include <pthread.h>
#include <unistd.h>

/**
 * @brief Returns the number of online logical processors.
 * @return @c _SC_NPROCESSORS_ONLN, clamped to a minimum of 1 if the query fails.
 */
static ZXC_ALWAYS_INLINE int zxc_num_procs(void) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

#endif /* _WIN32 */

#endif /* ZXC_THREADS_H */
