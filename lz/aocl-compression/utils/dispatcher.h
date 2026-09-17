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

#ifndef DISPATCH_DISPATCHER_H
#define DISPATCH_DISPATCHER_H

#include <stdint.h>

#if defined(WIN32) && defined(AOCL_UNIT_TEST)
#define EXPORT_UTILS_DYN_TEST __declspec(dllexport)
#else
/**
 * For Linux EXPORT_UTILS_DYN_TEST is NULL, by default the symbols are publicly exposed.
 */
#define EXPORT_UTILS_DYN_TEST
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Dispatcher module: maps runtime CPU capabilities to optimization-level
 * feature sets used by codec paths.
 *
 * Responsibilities:
 * - Perform one-time CPU capability detection on first use.
 * - Expose cumulative feature masks for each OptimizationLevel.
 * - Parse AOCL_ENABLE_INSTRUCTIONS and map it to an OptimizationLevel.
 * - Resolve environment-requested level to a feature mask.
 *
 * Threading model:
 * - AOCL_ENABLE_THREADS + _OPENMP: lazy init synchronized via OpenMP lock.
 * - Otherwise: lazy init synchronized via compiler atomics/spin lock.
 */

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

typedef uint64_t CpuFeatures;

/* Feature Bit Definitions */
#define FEATURE_SSE2            (1ULL << 0)
#define FEATURE_AVX             (1ULL << 1)
#define FEATURE_AVX2            (1ULL << 2)
#define FEATURE_AVX512F         (1ULL << 3)
#define FEATURE_AVX512VL        (1ULL << 4)
#define FEATURE_AVX512VNNI      (1ULL << 5)
#define FEATURE_AVX512BW        (1ULL << 6)
#define FEATURE_BMI2            (1ULL << 7)
#define FEATURE_PCLMUL          (1ULL << 8)
#define FEATURE_VPCLMULQDQ      (1ULL << 9)

typedef enum {
    OPTLEVEL_AUTO   = -1,
    OPTLEVEL_SCALAR = 0,
    OPTLEVEL_SSE2   = 1,
    OPTLEVEL_AVX    = 2,
    OPTLEVEL_AVX2   = 3,
    OPTLEVEL_AVX512 = 4
} OptimizationLevel;


/* ============================================================================
 * Level Mapping APIs
 * ============================================================================ */

/**
 * Convert integer input to OptimizationLevel.
 *
 * Mapping: -1=>AUTO, 0=>SCALAR, 1=>SSE2, 2=>AVX, 3=>AVX2, 4=>AVX512.
 * Out-of-range values return OPTLEVEL_AUTO.
 */
OptimizationLevel Dispatcher_IntToLevel(int level);

/**
 * Return cumulative supported features for a given optimization level.
 *
 * Tier behavior:
 * - OPTLEVEL_AUTO: returns all detected supported features.
 * - OPTLEVEL_SSE2: includes SSE2 tier only.
 * - OPTLEVEL_AVX: includes SSE2 + AVX tier and AVX-gated VPCLMULQDQ.
 * - OPTLEVEL_AVX2: includes AVX tier + AVX2 tier.
 * - OPTLEVEL_AVX512: includes AVX2 tier + AVX-512 tier.
 *
 * Cross-tier features:
 * - BMI2 and PCLMUL are included whenever supported, independent of AVX tier.
 */
EXPORT_UTILS_DYN_TEST CpuFeatures Dispatcher_GetSupportedFeaturesForLevel(OptimizationLevel level);


/* ============================================================================
 * Environment-Driven APIs
 * ============================================================================ */

/**
 * Parse AOCL_ENABLE_INSTRUCTIONS and return requested OptimizationLevel.
 *
 * Accepted values (case-insensitive): avx512, avx2, avx, sse2.
 * Behavior:
 * - Unset: returns OPTLEVEL_AUTO (or OPTLEVEL_AVX512 when
 *   AOCL_DYNAMIC_DISPATCHER is enabled).
 * - Unknown value: returns OPTLEVEL_SCALAR.
 */
OptimizationLevel Dispatcher_GetEnvRequestedLevel(void);

/**
 * Read AOCL_ENABLE_INSTRUCTIONS and return the corresponding cumulative
 * supported feature set using Dispatcher_GetSupportedFeaturesForLevel().
 */
CpuFeatures Dispatcher_GetFeaturesFromEnv(void);


#ifdef __cplusplus
}
#endif

#endif /* DISPATCH_DISPATCHER_H */
