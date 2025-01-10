/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Platform specific functions and constants                 */
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
#include <string.h>
#include <memory.h>

#include "platform.h"

#include "../libbsc.h"

#if defined(_WIN32)
  #include <windows.h>
  SIZE_T g_LargePageSize = 0;
#endif

#if (LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2)

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

static void bsc_cpuid(unsigned int regs[4], unsigned int level)
{
#if defined(_MSC_VER)
    __cpuid((int *)regs, (int)level);
#else
    __asm__ __volatile__
    (
        "xchg %%ebx, %%edi\n\t"
        "cpuid\n\t"
        "xchg %%ebx, %%edi" 
        : "=a"(regs[0]), "=D"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(level), "c"(0)
    );
#endif
}

static unsigned long long bsc_xgetbv()
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int eax = 0, edx = 0;
    __asm__ __volatile__
    (
        "xgetbv" 
        : "=a"(eax), "=d"(edx) 
        : "c"(0)
    );
    return ((unsigned long long)edx << 32) | eax;
#endif
}

int bsc_get_cpu_features(void)
{
    static int g_cpu_features = -1; if (g_cpu_features >= 0) { return g_cpu_features; }

    unsigned int regs[4] = { 0, 0, 0, 0 };

    bsc_cpuid(regs, 0); if (regs[0] < 1) { return g_cpu_features = LIBBSC_CPU_FEATURE_NONE; }

    bsc_cpuid(regs, 1);
    if ((regs[3] & (1 << 26)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_NONE; }    // no SSE2
    if ((regs[2] & (1 <<  0)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE2; }    // no SSE3
    if ((regs[2] & (1 <<  9)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE3; }    // no SSSE3
    if ((regs[2] & (1 << 19)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSSE3; }   // no SSE4.1
    if ((regs[2] & (1 << 23)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE41; }   // no POPCNT
    if ((regs[2] & (1 << 20)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE41; }   // no SSE4.2
    if ((regs[2] & (1 << 28)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE42; }   // no AVX
    if ((regs[2] & (1 << 27)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE42; }   // no XSAVE
    if ((bsc_xgetbv() & 0x6) != 0x6)    { return g_cpu_features = LIBBSC_CPU_FEATURE_SSE42; }   // AVX not enabled by OS

    bsc_cpuid(regs, 0); if (regs[0] < 7) { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX; }

    bsc_cpuid(regs, 7);
    if ((regs[1] & (1 <<  5)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX; }     // no AVX2
    if ((regs[1] & (1 << 16)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX2; }    // no AVX512F
    if ((regs[1] & (1 << 28)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX2; }    // no AVX512CD
    if ((bsc_xgetbv() & 0xE0) != 0xE0)  { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX2; }    // AVX512 not enabled by OS
    if ((regs[1] & (1 << 17)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX512F; } // no AVX512DQ
    if ((regs[1] & (1 << 31)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX512F; } // no AVX512VL
    if ((regs[1] & (1 << 30)) == 0)     { return g_cpu_features = LIBBSC_CPU_FEATURE_AVX512F; } // no AVX512BW
    
    return g_cpu_features = LIBBSC_CPU_FEATURE_AVX512BW;
}

#else

int bsc_get_cpu_features(void)
{
    return LIBBSC_CPU_FEATURE_NONE;
}

#endif

static void * bsc_default_malloc(size_t size)
{
#if defined(_WIN32)
    if ((g_LargePageSize != 0) && (size >= 256 * 1024))
    {
        void * address = VirtualAlloc(0, (size + g_LargePageSize - 1) & (~(g_LargePageSize - 1)), MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (address != NULL) return address;
    }
    return VirtualAlloc(0, size, MEM_COMMIT, PAGE_READWRITE);
#else
    return malloc(size);
#endif
}

static void * bsc_default_zero_malloc(size_t size)
{
#if defined(_WIN32)
    if ((g_LargePageSize != 0) && (size >= 256 * 1024))
    {
        void * address = VirtualAlloc(0, (size + g_LargePageSize - 1) & (~(g_LargePageSize - 1)), MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (address != NULL) return address;
    }
    return VirtualAlloc(0, size, MEM_COMMIT, PAGE_READWRITE);
#else
    return calloc(1, size);
#endif
}

static void * bsc_wrap_zero_malloc(size_t size)
{
    void *address = bsc_malloc(size);
    if(address != NULL)
    {
	memset(address, 0, size);
    }
    return address;
}

static void bsc_default_free(void * address)
{
#if defined(_WIN32)
    VirtualFree(address, 0, MEM_RELEASE);
#else
    free(address);
#endif
}

static void* (* bsc_malloc_fn)(size_t size) = bsc_default_malloc;
static void* (* bsc_zero_malloc_fn)(size_t size) = bsc_default_zero_malloc;
static void  (* bsc_free_fn)(void* address) = bsc_default_free;

void* bsc_malloc(size_t size)
{
    return bsc_malloc_fn(size);
}

void* bsc_zero_malloc(size_t size)
{
    return bsc_zero_malloc_fn(size);
}

void bsc_free(void* address)
{
    return bsc_free_fn(address);
}

int bsc_platform_init(int features, void* (* malloc)(size_t size), void* (* zero_malloc)(size_t size), void (* free)(void* address))
{
    /* If the caller provides a malloc function but not a zero_malloc
       function, we want to use malloc to implement zero_malloc.
       Otherwise we'll use the default function which may be slightly
       faster on some platforms. */
    if (zero_malloc != NULL)
    {
	bsc_zero_malloc_fn = zero_malloc;
    }
    else if (malloc != NULL)
    {
	bsc_zero_malloc_fn = bsc_wrap_zero_malloc;
    }

    if (malloc != NULL)
    {
	bsc_malloc_fn = malloc;
    }

    if (free != NULL)
    {
	bsc_free_fn = free;
    }

#if defined(_WIN32)

    if (features & LIBBSC_FEATURE_LARGEPAGES)
    {
        HANDLE hToken = 0;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken))
        {
            LUID luid;
            if (LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &luid))
            {
                TOKEN_PRIVILEGES tp;

                tp.PrivilegeCount = 1;
                tp.Privileges[0].Luid = luid;
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

                AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), 0, 0);
            }

            CloseHandle(hToken);
        }

        {
            if (HMODULE hKernel = GetModuleHandle(TEXT("kernel32.dll")))
            {
                typedef SIZE_T (WINAPI * GetLargePageMinimumProcT)();

                GetLargePageMinimumProcT largePageMinimumProc = (GetLargePageMinimumProcT)GetProcAddress(hKernel, "GetLargePageMinimum");
                if (largePageMinimumProc != NULL)
                {
                    SIZE_T largePageSize = largePageMinimumProc();

                    if ((largePageSize & (largePageSize - 1)) != 0) largePageSize = 0;

                    g_LargePageSize = largePageSize;
                }
            }
        }
    }

#endif

    return LIBBSC_NO_ERROR;
}

/*-----------------------------------------------------------*/
/* End                                          platform.cpp */
/*-----------------------------------------------------------*/
