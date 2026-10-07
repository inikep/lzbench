/*
 * WZIP - the little threading the library uses (internal): threads, and acquire/release loads and stores of a long
 * Copyright (c) 2026-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 *
 * Built only with WZIP_MULTITHREAD: Win32 threads on Windows, pthreads elsewhere.
 */
#ifndef WZ_THREADS_H_2026
#define WZ_THREADS_H_2026

#ifndef WZIP_MULTITHREAD
#  define WZIP_MULTITHREAD 0
#endif

#if WZIP_MULTITHREAD
#  if defined(_WIN32)
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    undef far                                   /* old Win32 keywords, macros here */
#    undef near
typedef HANDLE WZ_Thread;
#    define WZ_THREAD_FN(name, arg)      static DWORD WINAPI name(void* arg)
#    define WZ_THREAD_START(t, fn, arg)  ((*(t) = CreateThread(NULL, 0, fn, arg, 0, NULL)) != NULL)
#    define WZ_THREAD_JOIN(t)            (WaitForSingleObject(t, INFINITE), CloseHandle(t))
#    define WZ_LOAD(p)                   InterlockedCompareExchange((volatile LONG*)(p), 0, 0)
#    define WZ_STORE(p, v)               InterlockedExchange((volatile LONG*)(p), (LONG)(v))
#    define WZ_FETCH_ADD(p, v)           InterlockedExchangeAdd((volatile LONG*)(p), (LONG)(v))
#    define WZ_YIELD()                   SwitchToThread()
#  else
#    include <pthread.h>
#    include <sched.h>
typedef pthread_t WZ_Thread;
#    define WZ_THREAD_FN(name, arg)      static void* name(void* arg)
#    define WZ_THREAD_START(t, fn, arg)  (pthread_create(t, NULL, fn, arg) == 0)
#    define WZ_THREAD_JOIN(t)            pthread_join(t, NULL)
#    define WZ_LOAD(p)                   __atomic_load_n((p), __ATOMIC_ACQUIRE)
#    define WZ_STORE(p, v)               __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#    define WZ_FETCH_ADD(p, v)           __atomic_fetch_add((p), (v), __ATOMIC_ACQ_REL)
#    define WZ_YIELD()                   sched_yield()
#  endif
#endif

#endif
