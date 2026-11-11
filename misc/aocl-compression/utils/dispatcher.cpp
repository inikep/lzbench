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

#include "dispatcher.h"
#include "cpu_features.h"
#include <ctype.h>
#include <stdlib.h>

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    #include <omp.h>
#endif

/*
 * Initialization state and cached feature mask.
 * g_initialized is published with release semantics after DetectCapabilities().
 */
static int g_initialized = 0;
static CpuFeatures g_available_features = 0;

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
static omp_lock_t g_dispatcher_init_lock;
static int g_dispatcher_omp_lock_initialized = 0;
#else
static unsigned char g_dispatcher_init_lock = 0;
#endif

static void Dispatcher_EnsureInitialized(void);

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
/* Initialize OpenMP lock exactly once before first use. */
static void Dispatcher_EnsureInitLockReady(void) {
    if (__atomic_load_n(&g_dispatcher_omp_lock_initialized, __ATOMIC_ACQUIRE)) {
        return;
    }

    #pragma omp critical(dispatcher_lock_init)
    {
        if (!g_dispatcher_omp_lock_initialized) {
            omp_init_lock(&g_dispatcher_init_lock);
            __atomic_store_n(&g_dispatcher_omp_lock_initialized, 1, __ATOMIC_RELEASE);
        }
    }
}
#endif

/* Case-insensitive equality check for ASCII environment values. */
static int StringsEqualCaseInsensitive(const char* a, const char* b) {
    if (!a || !b) {
        return 0;
    }
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

OptimizationLevel Dispatcher_IntToLevel(int level) {
    switch (level) {
        case -1: return OPTLEVEL_AUTO;
        case 0: return OPTLEVEL_SCALAR;
        case 1: return OPTLEVEL_SSE2;
        case 2: return OPTLEVEL_AVX;
        case 3: return OPTLEVEL_AVX2;
        case 4: return OPTLEVEL_AVX512;
        default: return OPTLEVEL_AUTO;
    }
}

/*
 * Collect runtime-usable feature bits.
 * CPU_CanUse* APIs already include required OS-state gating.
 */
static void DetectCapabilities(void) {
    g_available_features = 0;

    g_available_features |= CPU_HasSSE2() ? FEATURE_SSE2 : 0;
    g_available_features |= CPU_CanUseAVX() ? FEATURE_AVX : 0;
    g_available_features |= CPU_CanUseAVX2() ? FEATURE_AVX2 : 0;
    g_available_features |= CPU_CanUseAVX512F() ? FEATURE_AVX512F : 0;
    g_available_features |= CPU_CanUseAVX512VL() ? FEATURE_AVX512VL : 0;
    g_available_features |= CPU_CanUseAVX512VNNI() ? FEATURE_AVX512VNNI : 0;
    g_available_features |= CPU_CanUseAVX512BW() ? FEATURE_AVX512BW : 0;
    g_available_features |= CPU_HasBMI2() ? FEATURE_BMI2 : 0;
    g_available_features |= CPU_HasPCLMUL() ? FEATURE_PCLMUL : 0;
    g_available_features |= CPU_CanUseVPCLMULQDQ() ? FEATURE_VPCLMULQDQ : 0;
}

/*
 * Thread-safe lazy initialization with double-check locking.
 * Fast path is a single acquire load after initialization.
 */
static void Dispatcher_EnsureInitialized(void) {
    if (__atomic_load_n(&g_initialized, __ATOMIC_ACQUIRE)) {
        return;
    }

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    Dispatcher_EnsureInitLockReady();
    omp_set_lock(&g_dispatcher_init_lock);
#else
    while (__atomic_test_and_set(&g_dispatcher_init_lock, __ATOMIC_ACQUIRE)) {
        /* spin */
    }
#endif

    if (!__atomic_load_n(&g_initialized, __ATOMIC_RELAXED)) {
        CPU_InitializeCpuidCache();
        DetectCapabilities();
        __atomic_store_n(&g_initialized, 1, __ATOMIC_RELEASE);
    }

#if defined(AOCL_ENABLE_THREADS) && defined(_OPENMP)
    omp_unset_lock(&g_dispatcher_init_lock);
#else
    __atomic_clear(&g_dispatcher_init_lock, __ATOMIC_RELEASE);
#endif
}

/* Build cumulative feature mask for requested optimization level. */
CpuFeatures Dispatcher_GetSupportedFeaturesForLevel(OptimizationLevel level) {
    CpuFeatures avail;
    CpuFeatures result = 0;

    Dispatcher_EnsureInitialized();
    avail = g_available_features;

    if (level == OPTLEVEL_AUTO) {
        return avail;
    }

    if (level >= OPTLEVEL_SSE2 && (avail & FEATURE_SSE2)) {
        result |= FEATURE_SSE2;
    }
    if (level >= OPTLEVEL_AVX && (avail & FEATURE_AVX)) {
        result |= FEATURE_AVX;
        if (avail & FEATURE_VPCLMULQDQ) {
            result |= FEATURE_VPCLMULQDQ;
        }
    }
    if (avail & FEATURE_BMI2) {
        result |= FEATURE_BMI2;
    }
    if (avail & FEATURE_PCLMUL) {
        result |= FEATURE_PCLMUL;
    }

    if (level >= OPTLEVEL_AVX2) {
        if (avail & FEATURE_AVX2) {
            result |= FEATURE_AVX2;
        }
    }
    if (level >= OPTLEVEL_AVX512) {
        if (avail & FEATURE_AVX512F) {
            result |= FEATURE_AVX512F;
        }
        if (avail & FEATURE_AVX512VL) {
            result |= FEATURE_AVX512VL;
        }
        if (avail & FEATURE_AVX512VNNI) {
            result |= FEATURE_AVX512VNNI;
        }
        if (avail & FEATURE_AVX512BW) {
            result |= FEATURE_AVX512BW;
        }
    }

    return result;
}

/* Parse AOCL_ENABLE_INSTRUCTIONS into requested optimization tier. */
OptimizationLevel Dispatcher_GetEnvRequestedLevel(void) {
    const char* env_val = getenv("AOCL_ENABLE_INSTRUCTIONS");

    if (env_val != NULL) {
        if (StringsEqualCaseInsensitive(env_val, "avx512")) {
            return OPTLEVEL_AVX512;
        }
        if (StringsEqualCaseInsensitive(env_val, "avx2")) {
            return OPTLEVEL_AVX2;
        }
        if (StringsEqualCaseInsensitive(env_val, "avx")) {
            return OPTLEVEL_AVX;
        }
        if (StringsEqualCaseInsensitive(env_val, "sse2")) {
            return OPTLEVEL_SSE2;
        }
        return OPTLEVEL_SCALAR;
    }
#ifndef AOCL_DYNAMIC_DISPATCHER
    return OPTLEVEL_AUTO;
#else
    return OPTLEVEL_AVX512;
#endif
}

/* Resolve env-requested level into cumulative supported feature mask. */
CpuFeatures Dispatcher_GetFeaturesFromEnv(void) {
    OptimizationLevel level = Dispatcher_GetEnvRequestedLevel();
    return Dispatcher_GetSupportedFeaturesForLevel(level);
}
