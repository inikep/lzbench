/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_deps.h
 * @brief Single point of override for the C-library dependencies of libzxc.
 *
 * The core compression/decompression code only ever reaches the standard
 * library through the macros and headers declared here.
 * Freestanding consumers vendor the source tree and replace this file with
 * an environment-specific version that maps the same macros onto their own
 * allocator, sort, and basic header set.
 *
 * @par Stock (hosted) build
 * Pulls in @c <assert.h>, @c <limits.h>, @c <stdint.h>, @c <stdlib.h>,
 * @c <string.h> and expands the macros to their libc equivalents.
 *
 * Per-symbol @c -D overrides are also accepted (each macro is guarded by
 * an @c ifndef, the aligned pair as one), so vendoring is optional for ad-hoc
 * consumers.
 */

#ifndef ZXC_DEPS_H
#define ZXC_DEPS_H

/**
 * @addtogroup internal
 * @{
 */

/**
 * @name Standard Headers
 * @brief Pulled in by the stock libzxc build to provide @c static_assert,
 * @c size_t, @c uintN_t / @c intN_t, @c CHAR_BIT, @c malloc / @c calloc /
 * @c realloc / @c free, and @c memcpy / @c memset / @c memmove /
 * @c memcmp.
 *
 * Vendored overrides of this file typically substitute @c <linux/build_bug.h>,
 * @c <linux/limits.h>, @c <linux/types.h>, @c <linux/string.h>, @c <linux/slab.h>,
 * @c <linux/sort.h>.
 * @{
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/** @} */ /* end of Standard Headers */

/**
 * @name Heap Allocator Abstraction
 * @brief Macros around the libc allocators so non-libc targets (Linux kernel,
 * embedded freestanding builds, custom arenas) can override them via @c -D
 * flags **before** including any zxc header, or by vendoring this file.
 *
 * @note Aligned allocations go through @ref ZXC_ALIGNED_MALLOC /
 * @ref ZXC_ALIGNED_FREE (see below), not these.
 * @{
 */

/** @def ZXC_MALLOC
 *  @brief Heap allocator. Default: libc @c malloc. */
#ifndef ZXC_MALLOC
#define ZXC_MALLOC(size) malloc(size)
#endif

/** @def ZXC_CALLOC
 *  @brief Zero-initialised heap allocator. Default: libc @c calloc. */
#ifndef ZXC_CALLOC
#define ZXC_CALLOC(nmemb, size) calloc(nmemb, size)
#endif

/** @def ZXC_REALLOC
 *  @brief In-place / move heap reallocator. Default: libc @c realloc. */
#ifndef ZXC_REALLOC
#define ZXC_REALLOC(ptr, size) realloc(ptr, size)
#endif

/** @def ZXC_FREE
 *  @brief Heap deallocator. Default: libc @c free. */
#ifndef ZXC_FREE
#define ZXC_FREE(ptr) free(ptr)
#endif

/** @} */ /* end of Heap Allocator Abstraction */

/**
 * @name Aligned Allocator Abstraction
 * @brief Macros around the cache-line-aligned allocator used for compression
 * workspace and per-context scratch buffers.
 *
 * The default is the pair defined below, wrapping @c _aligned_malloc on Windows
 * and @c posix_memalign elsewhere - the core's only use of either. A host
 * overrides both macros or neither; a vendored copy of this file defines both.
 *
 * Kernel builds align by hand over the slab allocator (see contrib/linux-kernel):
 * @c kmalloc guarantees only @c ARCH_KMALLOC_MINALIGN, 8 bytes on x86 and on
 * arm64 since 6.5, short of the cache line the workspace layout assumes.
 * @{
 */

#if defined(ZXC_ALIGNED_MALLOC) != defined(ZXC_ALIGNED_FREE)
#error "ZXC_ALIGNED_MALLOC and ZXC_ALIGNED_FREE are overridden together"
#endif

#ifndef ZXC_ALIGNED_MALLOC

/** @brief Default cache-line-aligned allocator; release with @ref zxc_aligned_free. */
static inline void* zxc_aligned_malloc(const size_t size, const size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#endif
}

/** @brief Counterpart of @ref zxc_aligned_malloc; NULL is a no-op. */
static inline void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/** @def ZXC_ALIGNED_MALLOC
 *  @brief Cache-line-aligned allocator. */
#define ZXC_ALIGNED_MALLOC(size, alignment) zxc_aligned_malloc(size, alignment)
/** @def ZXC_ALIGNED_FREE
 *  @brief Counterpart deallocator for @ref ZXC_ALIGNED_MALLOC. */
#define ZXC_ALIGNED_FREE(ptr) zxc_aligned_free(ptr)

#endif  // ZXC_ALIGNED_MALLOC

/** @} */ /* end of Aligned Allocator Abstraction */

/**
 * @name Population Count
 * @brief @c ZXC_POPCOUNT32 / @c ZXC_POPCOUNT64. Default: the builtins, libgcc
 * calls without a hardware count; a kernel maps them onto @c hweight32/64.
 * @{
 */

#if defined(ZXC_POPCOUNT32) != defined(ZXC_POPCOUNT64)
#error "ZXC_POPCOUNT32 and ZXC_POPCOUNT64 are overridden together"
#endif

#ifndef ZXC_POPCOUNT32
#if defined(__GNUC__) || defined(__clang__)
#define ZXC_POPCOUNT32(v) __builtin_popcount(v)
#define ZXC_POPCOUNT64(v) __builtin_popcountll(v)
#else
/** @brief Portable SWAR popcount (MSVC). */
static inline int zxc_popcount32(const uint32_t v) {
    uint32_t x = v - ((v >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int)((x * 0x01010101U) >> 24);
}
static inline int zxc_popcount64(const uint64_t v) {
    return zxc_popcount32((uint32_t)v) + zxc_popcount32((uint32_t)(v >> 32));
}
#define ZXC_POPCOUNT32(v) zxc_popcount32(v)
#define ZXC_POPCOUNT64(v) zxc_popcount64(v)
#endif
#endif  // ZXC_POPCOUNT32

/** @} */ /* end of Population Count */

/** @} */ /* end of addtogroup internal */

#endif /* ZXC_DEPS_H */
