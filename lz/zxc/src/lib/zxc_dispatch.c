/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "zxc_internal.h"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__linux__) && (defined(__arm__) || defined(_M_ARM))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

/*
 * ============================================================================
 * PROTOTYPES FOR MULTI-VERSIONED VARIANTS
 * ============================================================================
 * These are compiled in separate translation units with different flags.
 */

// Decompression Prototypes
int zxc_decompress_chunk_wrapper_default(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                         uint8_t* dst, size_t dst_cap);

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
int zxc_decompress_chunk_wrapper_avx2(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                      uint8_t* dst, size_t dst_cap);
int zxc_decompress_chunk_wrapper_avx512(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                        uint8_t* dst, size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_decompress_chunk_wrapper_neon(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                      uint8_t* dst, size_t dst_cap);
#endif
#endif

// Compression Prototypes
int zxc_compress_chunk_wrapper_default(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                       uint8_t* dst, size_t dst_cap);

#if defined(__x86_64__) || defined(_M_X64)
int zxc_compress_chunk_wrapper_avx2(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                    uint8_t* dst, size_t dst_cap);
int zxc_compress_chunk_wrapper_avx512(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                      uint8_t* dst, size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_compress_chunk_wrapper_neon(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                    uint8_t* dst, size_t dst_cap);
#endif

/*
 * ============================================================================
 * CPU DETECTION LOGIC
 * ============================================================================
 */

typedef enum {
    ZXC_CPU_GENERIC = 0,
    ZXC_CPU_AVX2 = 1,
    ZXC_CPU_AVX512 = 2,
    ZXC_CPU_NEON = 3
} zxc_cpu_feature_t;

static zxc_cpu_feature_t zxc_detect_cpu_features(void) {
#ifdef ZXC_ONLY_DEFAULT
    return ZXC_CPU_GENERIC;
#else
    zxc_cpu_feature_t features = ZXC_CPU_GENERIC;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    // MSVC detection using __cpuid
    // Function ID 1: EAX=1. ECX: Bit 28=AVX.
    // Function ID 7: EAX=7, ECX=0. EBX: Bit 5=AVX2, Bit 16=AVX512F, Bit 30=AVX512BW.
    int regs[4];
    int avx = 0;
    int avx2 = 0;
    int avx512 = 0;

    __cpuid(regs, 1);
    if (regs[2] & (1 << 28)) avx = 1;

    if (avx) {
        __cpuidex(regs, 7, 0);
        if (regs[1] & (1 << 5)) avx2 = 1;
        if ((regs[1] & (1 << 16)) && (regs[1] & (1 << 30))) avx512 = 1;
    }

    if (avx512) {
        features = ZXC_CPU_AVX512;
    } else if (avx2) {
        features = ZXC_CPU_AVX2;
    }
#else
    // GCC/Clang built-in detection
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
        features = ZXC_CPU_AVX512;
    } else if (__builtin_cpu_supports("avx2")) {
        features = ZXC_CPU_AVX2;
    }
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 usually guarantees NEON
    features = ZXC_CPU_NEON;

#elif defined(__arm__) || defined(_M_ARM)
    // ARM32 Runtime detection for Linux
#if defined(__linux__)
    unsigned long hwcaps = getauxval(AT_HWCAP);
    if (hwcaps & HWCAP_NEON) {
        features = ZXC_CPU_NEON;
    }
#else
// Fallback for non-Linux: rely on compiler flags.
// If compiled with -mfpu=neon, we assume target supports it.
// Otherwise, safe default is GENERIC.
#if defined(__ARM_NEON)
    features = ZXC_CPU_NEON;
#endif
#endif
#endif

    return features;
#endif
}

/*
 * ============================================================================
 * DISPATCHERS
 * ============================================================================
 * We use a function pointer initialized on first use (lazy initialization).
 */

typedef int (*zxc_decompress_func_t)(zxc_cctx_t*, const uint8_t*, size_t, uint8_t*, size_t);
typedef int (*zxc_compress_func_t)(zxc_cctx_t*, const uint8_t*, size_t, uint8_t*, size_t);

static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_ptr = NULL;
static ZXC_ATOMIC zxc_compress_func_t zxc_compress_ptr = NULL;

// Initializer for Decompression
static int zxc_decompress_dispatch_init(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                        uint8_t* dst, size_t dst_cap) {
    zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_decompress_func_t zxc_decompress_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512)
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_avx512;
    else if (cpu == ZXC_CPU_AVX2)
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_avx2;
    else
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON)
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_neon;
    else
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
#else
    (void)cpu;
    zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
#endif
#else
    (void)cpu;
    zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_decompress_ptr, zxc_decompress_ptr_local, memory_order_release);
#else
    zxc_decompress_ptr = zxc_decompress_ptr_local;
#endif
    return zxc_decompress_ptr_local(ctx, src, src_sz, dst, dst_cap);
}

// Initializer for Compression
static int zxc_compress_dispatch_init(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz,
                                      uint8_t* dst, size_t dst_cap) {
    zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_compress_func_t zxc_compress_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_avx512;
    else if (cpu == ZXC_CPU_AVX2)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_avx2;
    else
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_neon;
    else
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#else
    (void)cpu;
    zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#endif
#else
    (void)cpu;
    zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_compress_ptr, zxc_compress_ptr_local, memory_order_release);
#else
    zxc_compress_ptr = zxc_compress_ptr_local;
#endif
    return zxc_compress_ptr_local(ctx, src, src_sz, dst, dst_cap);
}

// Public Wrappers (Dispatcher and Main API)

int zxc_decompress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz, uint8_t* dst,
                                 size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    zxc_decompress_func_t func = atomic_load_explicit(&zxc_decompress_ptr, memory_order_acquire);
#else
    zxc_decompress_func_t func = zxc_decompress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_decompress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

int zxc_compress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz, uint8_t* dst,
                               size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    zxc_compress_func_t func = atomic_load_explicit(&zxc_compress_ptr, memory_order_acquire);
#else
    zxc_compress_func_t func = zxc_compress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_compress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/*
 * ============================================================================
 * PUBLIC UTILITY API
 * ============================================================================
 * These wrapper functions provide a simplified interface by managing context
 * allocation and looping over blocks. They call the dispatched wrappers above.
 */

// cppcheck-suppress unusedFunction
size_t zxc_compress(const void* src, size_t src_size, void* dst, size_t dst_capacity, int level,
                    int checksum_enabled) {
    if (UNLIKELY(!src || !dst || src_size == 0 || dst_capacity == 0)) return 0;

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;

    zxc_cctx_t ctx;
    if (zxc_cctx_init(&ctx, ZXC_BLOCK_SIZE, 1, level, checksum_enabled) != 0) return 0;

    int h_size = zxc_write_file_header(op, (size_t)(op_end - op));
    if (UNLIKELY(h_size < 0)) {
        zxc_cctx_free(&ctx);
        return 0;
    }
    op += h_size;

    size_t pos = 0;
    while (pos < src_size) {
        size_t chunk_len = (src_size - pos > ZXC_BLOCK_SIZE) ? ZXC_BLOCK_SIZE : (src_size - pos);
        size_t rem_cap = (size_t)(op_end - op);

        int res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap);
        if (UNLIKELY(res < 0)) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        op += res;
        pos += chunk_len;
    }

    zxc_cctx_free(&ctx);
    return (size_t)(op - op_start);
}

// cppcheck-suppress unusedFunction
size_t zxc_decompress(const void* src, size_t src_size, void* dst, size_t dst_capacity,
                      int checksum_enabled) {
    if (UNLIKELY(!src || !dst || src_size < ZXC_FILE_HEADER_SIZE)) return 0;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;

    // File header verification
    if (zxc_read_file_header(ip, src_size, &runtime_chunk_size) != 0) return 0;

    zxc_cctx_t ctx;
    if (zxc_cctx_init(&ctx, runtime_chunk_size, 0, 0, checksum_enabled) != 0) return 0;

    ip += ZXC_FILE_HEADER_SIZE;

    // Block decompression loop
    while (ip < ip_end) {
        size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        // Read the block header to determine the compressed size
        if (zxc_read_block_header(ip, rem_src, &bh) != 0) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        // Safety check: ensure the block (header + data + checksum) fits in the input buffer
        size_t checksum_sz =
            (bh.block_flags & ZXC_BLOCK_FLAG_CHECKSUM) ? ZXC_BLOCK_CHECKSUM_SIZE : 0;
        size_t total_block_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size + checksum_sz;

        if (UNLIKELY(total_block_sz > rem_src)) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        size_t rem_cap = (size_t)(op_end - op);
        int res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, op, rem_cap);
        if (UNLIKELY(res < 0)) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        ip += ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
        ip += (bh.block_flags & ZXC_BLOCK_FLAG_CHECKSUM) ? ZXC_BLOCK_CHECKSUM_SIZE : 0;
        op += res;
    }

    zxc_cctx_free(&ctx);
    return (size_t)(op - op_start);
}
