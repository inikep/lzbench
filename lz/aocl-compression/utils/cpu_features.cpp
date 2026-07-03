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

#include "cpu_features.h"
#include <string.h>

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    #include <omp.h>
#endif

#if !defined(_M_IX86) && !defined(_M_X64) && !defined(__i386__) && !defined(__x86_64__)
    #error "cpu_features implementation only supports x86/x86_64 architectures."
#endif

/**
 * Platform-Specific Includes for CPUID Instruction
 * 
 * Each compiler provides different intrinsics and APIs for executing CPUID:
 * - MSVC: __cpuidex intrinsic from intrin.h
 * - GCC/Clang: __get_cpuid_count from cpuid.h (requires GCC 4.4+ or Clang 3.0+)
 */
#if defined(COMPILER_MSVC)
    #include <intrin.h>   /* MSVC: __cpuidex, __cpuid */
#elif defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    #include <cpuid.h>    /* GCC/Clang: __get_cpuid, __get_cpuid_count */
#else
    #error "Unsupported compiler for CPUID detection"
#endif

/**
 * CPUID Result Cache
 * 
 * Caches CPUID results to avoid redundant instruction executions.
 * Initial value (0xffffffff) indicates cache is uninitialized.
 *
 * Concurrency model:
 * - AOCL_ENABLE_THREADS + _OPENMP:
 *   - One-time lock creation via OpenMP critical (no busy-wait).
 *   - Cache initialization guarded by OpenMP lock.
 * - Otherwise:
 *   - One-time initialization guarded by compiler atomic spin lock.
 *
 * Both paths provide single-initializer semantics for CPUID cache publication.
 */
static CpuidResult g_cpuid_leaf_1 = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
static CpuidResult g_cpuid_leaf_7 = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
static CpuidResult g_cpuid_leaf_7_subleaf_1 = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
static CpuidResult g_cpuid_leaf_ext_80000001 = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
static uint32_t g_max_cpuid_leaf = 0xffffffff;
static uint32_t g_max_cpuid_leaf_ext = 0xffffffff;
static int g_cpuid_initialized = 0;

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
static omp_lock_t g_cpuid_init_lock;
static int g_cpuid_omp_lock_initialized = 0;
#else
static unsigned char g_cpuid_init_lock = 0;
#endif

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
/*
 * Initializes the OpenMP lock exactly once.
 *
 * No nested lock acquisition occurs here: this routine only enters a named
 * OpenMP critical region and initializes g_cpuid_init_lock when needed.
 */
static void CPU_EnsureInitLockReady(void) {
    if (__atomic_load_n(&g_cpuid_omp_lock_initialized, __ATOMIC_ACQUIRE)) {
        return;
    }

    #pragma omp critical(cpu_features_lock_init)
    {
        if (!g_cpuid_omp_lock_initialized) {
            omp_init_lock(&g_cpuid_init_lock);
            __atomic_store_n(&g_cpuid_omp_lock_initialized, 1, __ATOMIC_RELEASE);
        }
    }
}
#endif

static int CPUID_Query(uint32_t leaf, uint32_t subleaf, CpuidResult* result) {
    uint32_t max_leaf = 0;

#if defined(COMPILER_MSVC)
    int cpu_info[4];

    if (leaf & 0x80000000u) {
        __cpuidex(cpu_info, (int)0x80000000u, 0);
        max_leaf = (uint32_t)cpu_info[0];
    } else {
        __cpuidex(cpu_info, 0, 0);
        max_leaf = (uint32_t)cpu_info[0];
    }

    if (leaf > max_leaf) {
        result->eax = 0;
        result->ebx = 0;
        result->ecx = 0;
        result->edx = 0;
        return 0;
    }

    if (leaf == 7u) {
        __cpuidex(cpu_info, 7, 0);
        if (subleaf > (uint32_t)cpu_info[0]) {
            result->eax = 0;
            result->ebx = 0;
            result->ecx = 0;
            result->edx = 0;
            return 0;
        }
    }

    __cpuidex(cpu_info, (int)leaf, (int)subleaf);
    result->eax = (uint32_t)cpu_info[0];
    result->ebx = (uint32_t)cpu_info[1];
    result->ecx = (uint32_t)cpu_info[2];
    result->edx = (uint32_t)cpu_info[3];
    return 1;
#elif defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    if (leaf & 0x80000000u) {
        max_leaf = __get_cpuid_max(0x80000000u, 0);
    } else {
        max_leaf = __get_cpuid_max(0, 0);
    }

    if (leaf > max_leaf) {
        result->eax = 0;
        result->ebx = 0;
        result->ecx = 0;
        result->edx = 0;
        return 0;
    }

    if (leaf == 7u) {
        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            result->eax = 0;
            result->ebx = 0;
            result->ecx = 0;
            result->edx = 0;
            return 0;
        }
        if (subleaf > eax) {
            result->eax = 0;
            result->ebx = 0;
            result->ecx = 0;
            result->edx = 0;
            return 0;
        }
    }

    if (!__get_cpuid_count(leaf, subleaf, &result->eax, &result->ebx, &result->ecx, &result->edx)) {
        result->eax = 0;
        result->ebx = 0;
        result->ecx = 0;
        result->edx = 0;
        return 0;
    }
    return 1;
#else
    (void)leaf;
    (void)subleaf;
    result->eax = 0;
    result->ebx = 0;
    result->ecx = 0;
    result->edx = 0;
    return 0;
#endif
}

static void CPU_EnsureCpuidCacheInitialized(void) {
    /*
     * Locking model and deadlock safety:
     * - OpenMP path: CPU_EnsureInitLockReady() may enter an OpenMP critical
     *   region before omp_set_lock(g_cpuid_init_lock). The critical region is
     *   exited before omp_set_lock() is called, so lock acquisition is never
     *   nested in reverse order.
     * - Non-OpenMP path: single atomic spin lock guards one-time init.
     */

    /* Fast path: cache already published. */
    if (__atomic_load_n(&g_cpuid_initialized, __ATOMIC_ACQUIRE)) {
        return;
    }

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    CPU_EnsureInitLockReady();
    omp_set_lock(&g_cpuid_init_lock);
#else
    while (__atomic_test_and_set(&g_cpuid_init_lock, __ATOMIC_ACQUIRE)) {
        /* spin */
    }
#endif

    /* Re-check under lock to preserve one-time initialization semantics. */
    if (!__atomic_load_n(&g_cpuid_initialized, __ATOMIC_RELAXED)) {
        CpuidResult leaf0 = {0, 0, 0, 0};
        g_cpuid_leaf_1 = {0, 0, 0, 0};
        g_cpuid_leaf_7 = {0, 0, 0, 0};
        g_cpuid_leaf_7_subleaf_1 = {0, 0, 0, 0};
        g_cpuid_leaf_ext_80000001 = {0, 0, 0, 0};
        g_max_cpuid_leaf = 0;
        g_max_cpuid_leaf_ext = 0;

        if (CPUID_Query(0, 0, &leaf0)) {
            g_max_cpuid_leaf = leaf0.eax;
        }

        if (g_max_cpuid_leaf >= 1) {
            (void)CPUID_Query(1, 0, &g_cpuid_leaf_1);
        }

        if (g_max_cpuid_leaf >= 7) {
            (void)CPUID_Query(7, 0, &g_cpuid_leaf_7);
            (void)CPUID_Query(7, 1, &g_cpuid_leaf_7_subleaf_1);
        }

        {
            CpuidResult leaf_ext = {0, 0, 0, 0};
            if (CPUID_Query(0x80000000u, 0, &leaf_ext)) {
                g_max_cpuid_leaf_ext = leaf_ext.eax;
                if (g_max_cpuid_leaf_ext >= 0x80000001u) {
                    (void)CPUID_Query(0x80000001u, 0, &g_cpuid_leaf_ext_80000001);
                }
            }
        }

        /* Publish fully initialized cache to all threads. */
        __atomic_store_n(&g_cpuid_initialized, 1, __ATOMIC_RELEASE);
    }

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    omp_unset_lock(&g_cpuid_init_lock);
#else
    __atomic_clear(&g_cpuid_init_lock, __ATOMIC_RELEASE);
#endif
}

/**
 * Execute CPUID Instruction
 * 
 * Direct CPUID execution without caching. For feature detection, use the
 * cached variants via CPU_InitializeCpuidCache() to avoid redundant calls.
 * 
 * Cross-compiler implementation:
 * - MSVC: __cpuidex intrinsic
 * - GCC/Clang: __get_cpuid_count (requires GCC 4.4+, Clang 3.0+)
 * 
 * @param leaf CPUID leaf index
 * @param subleaf CPUID subleaf index
 * @return CPUID result with EAX, EBX, ECX, EDX register values
 */
CpuidResult CPU_Cpuid(uint32_t leaf, uint32_t subleaf) {
    CpuidResult result = {0, 0, 0, 0};
    (void)CPUID_Query(leaf, subleaf, &result);
    return result;
}

/**
 * Initialize CPUID Caches
 * 
 * Queries and caches CPUID results for leaves 0, 1, and 7 on first call.
 * Subsequent calls return immediately.
 *
 * Thread safety behavior:
 * - AOCL_ENABLE_THREADS + _OPENMP: OpenMP lock protects one-time init.
 * - Otherwise: atomic spin lock protects one-time init.
 */
void CPU_InitializeCpuidCache(void) {
    CPU_EnsureCpuidCacheInitialized();
}

int CPU_IsCpuidSupported(void) {
    return 1;
}

uint32_t CPU_GetMaxCpuidLeaf(void) {
    CPU_EnsureCpuidCacheInitialized();
    return g_max_cpuid_leaf;
}

void CPU_GetVendorString(char vendor[13]) {
    CpuidResult result = CPU_Cpuid(0, 0);

    memcpy(vendor + 0, &result.ebx, 4);
    memcpy(vendor + 4, &result.edx, 4);
    memcpy(vendor + 8, &result.ecx, 4);
    vendor[12] = '\0';
}

int CPU_HasSSE2(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_1.edx & (1u << 26)) != 0;
}

int CPU_HasAVX(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_1.ecx & (1u << 28)) != 0;
}

int CPU_HasAVX2(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 5)) != 0;
}

int CPU_HasFMA(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_1.ecx & (1u << 12)) != 0;
}

int CPU_HasBMI1(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 3)) != 0;
}

int CPU_HasBMI2(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 8)) != 0;
}

int CPU_HasLZCNT(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_ext_80000001.ecx & (1u << 5)) != 0;
}

int CPU_HasPCLMUL(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_1.ecx & (1u << 1)) != 0;
}

int CPU_HasVAES(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 9)) != 0;
}

int CPU_HasGFNI(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 8)) != 0;
}

int CPU_HasVPCLMULQDQ(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 10)) != 0;
}

int CPU_HasAVXVNNI(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7_subleaf_1.eax & (1u << 4)) != 0;
}

int CPU_HasAVX512F(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 16)) != 0;
}

int CPU_HasAVX512DQ(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 17)) != 0;
}

int CPU_HasAVX512IFMA(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 21)) != 0;
}

int CPU_HasAVX512CD(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 28)) != 0;
}

int CPU_HasAVX512BW(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 30)) != 0;
}

int CPU_HasAVX512VL(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ebx & (1u << 31)) != 0;
}

int CPU_HasAVX512VBMI(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 1)) != 0;
}

int CPU_HasAVX512VBMI2(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 6)) != 0;
}

int CPU_HasAVX512VNNI(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 11)) != 0;
}

int CPU_HasAVX512BITALG(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 12)) != 0;
}

int CPU_HasAVX512VPOPCNTDQ(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.ecx & (1u << 14)) != 0;
}

int CPU_HasAVX512VP2INTERSECT(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7.edx & (1u << 8)) != 0;
}

int CPU_HasAVX512BF16(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7_subleaf_1.eax & (1u << 5)) != 0;
}

int CPU_HasAVX512FP16(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_7_subleaf_1.eax & (1u << 23)) != 0;
}

int CPU_IsXGETBVSupported(void) {
    CPU_EnsureCpuidCacheInitialized();
    return (g_cpuid_leaf_1.ecx & (1u << 27)) != 0;
}

uint64_t CPU_GetXCR0(void) {
    if (!CPU_IsXGETBVSupported()) {
        return 0;
    }

#if defined(COMPILER_MSVC)
    return _xgetbv(0);

#elif defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    uint32_t eax, edx;
    __asm__ __volatile__(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0)
    );
    return ((uint64_t)edx << 32) | eax;

#else
    return 0;
#endif
}

int CPU_HasAVXSupport(void) {
    uint64_t xcr0 = CPU_GetXCR0();
    const uint64_t AVX_BITS = 0x06;
    return (xcr0 & AVX_BITS) == AVX_BITS;
}

int CPU_HasAVX512Support(void) {
    uint64_t xcr0 = CPU_GetXCR0();
    const uint64_t AVX512_BITS = 0xE6;
    return (xcr0 & AVX512_BITS) == AVX512_BITS;
}

int CPU_CanUseAVX(void) {
    return CPU_HasAVX() && CPU_HasAVXSupport();
}

int CPU_CanUseAVX2(void) {
    return CPU_HasAVX2() && CPU_CanUseAVX();
}

int CPU_CanUseFMA(void) {
    return CPU_HasFMA() && CPU_CanUseAVX();
}

int CPU_CanUseVAES(void) {
    return CPU_HasVAES() && CPU_CanUseAVX();
}

int CPU_CanUseVPCLMULQDQ(void) {
    return CPU_HasVPCLMULQDQ() && CPU_CanUseAVX();
}

int CPU_CanUseAVXVNNI(void) {
    return CPU_HasAVXVNNI() && CPU_CanUseAVX();
}

int CPU_CanUseAVX512F(void) {
    return CPU_HasAVX512F() && CPU_HasAVX512Support();
}

int CPU_CanUseAVX512DQ(void) {
    return CPU_HasAVX512DQ() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512IFMA(void) {
    return CPU_HasAVX512IFMA() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512CD(void) {
    return CPU_HasAVX512CD() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512BW(void) {
    return CPU_HasAVX512BW() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VL(void) {
    return CPU_HasAVX512VL() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VBMI(void) {
    return CPU_HasAVX512VBMI() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VBMI2(void) {
    return CPU_HasAVX512VBMI2() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VNNI(void) {
    return CPU_HasAVX512VNNI() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512BITALG(void) {
    return CPU_HasAVX512BITALG() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VPOPCNTDQ(void) {
    return CPU_HasAVX512VPOPCNTDQ() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512VP2INTERSECT(void) {
    return CPU_HasAVX512VP2INTERSECT() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512BF16(void) {
    return CPU_HasAVX512BF16() && CPU_CanUseAVX512F();
}

int CPU_CanUseAVX512FP16(void) {
    return CPU_HasAVX512FP16() && CPU_CanUseAVX512F();
}
