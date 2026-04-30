/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_dispatch.c
 * @brief Runtime CPU feature detection and SIMD dispatch layer.
 *
 * Detects AVX2/AVX512/NEON at runtime and routes compress/decompress calls
 * to the best available implementation via lazy-initialised function pointers.
 * Also contains the public one-shot buffer API (@ref zxc_compress,
 * @ref zxc_decompress, @ref zxc_get_decompressed_size).
 */

#include "../../include/zxc_error.h"
#include "../../include/zxc_seekable.h"
#include "zxc_internal.h"

/*
 * ZXC_DISABLE_SIMD => force ZXC_ONLY_DEFAULT so the dispatcher never selects
 * an AVX2/AVX512/NEON variant.
 */
#if defined(ZXC_DISABLE_SIMD) && !defined(ZXC_ONLY_DEFAULT)
#define ZXC_ONLY_DEFAULT
#endif

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
int zxc_decompress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                         const size_t src_sz, uint8_t* RESTRICT dst,
                                         const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                              const size_t src_sz, uint8_t* RESTRICT dst,
                                              const size_t dst_cap);

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
int zxc_decompress_chunk_wrapper_avx2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_decompress_chunk_wrapper_avx512(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_avx2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                           const size_t src_sz, uint8_t* RESTRICT dst,
                                           const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_avx512(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                             const size_t src_sz, uint8_t* RESTRICT dst,
                                             const size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_decompress_chunk_wrapper_neon(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_neon(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                           const size_t src_sz, uint8_t* RESTRICT dst,
                                           const size_t dst_cap);
#endif
#endif

// Compression Prototypes
int zxc_compress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                       const size_t src_sz, uint8_t* RESTRICT dst,
                                       const size_t dst_cap);

// Huffman Prototypes (variant TUs of zxc_huffman.c). The compressor and
// decompressor variants resolve their Huffman calls to the matching suffixed
// symbol at compile time (zero dispatch overhead in the hot path); the thin
// wrappers below expose the un-suffixed names for tests and external callers.
int zxc_huf_build_code_lengths_default(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                                       void* RESTRICT scratch);
int zxc_huf_encode_section_default(const uint8_t* RESTRICT literals, const size_t n_literals,
                                   const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                   const size_t dst_cap);
int zxc_huf_decode_section_default(const uint8_t* RESTRICT payload, const size_t payload_size,
                                   uint8_t* RESTRICT dst, const size_t n_literals);

#if defined(__x86_64__) || defined(_M_X64)
int zxc_compress_chunk_wrapper_avx2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap);
int zxc_compress_chunk_wrapper_avx512(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_compress_chunk_wrapper_neon(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap);
#endif

/*
 * ============================================================================
 * CPU DETECTION LOGIC
 * ============================================================================
 */

/**
 * @enum zxc_cpu_feature_t
 * @brief Detected CPU SIMD capability level.
 */
typedef enum {
    ZXC_CPU_GENERIC = 0, /**< @brief Scalar-only fallback.   */
    ZXC_CPU_AVX2 = 1,    /**< @brief x86-64 AVX2 available.  */
    ZXC_CPU_AVX512 = 2,  /**< @brief x86-64 AVX-512F+BW available. */
    ZXC_CPU_NEON = 3     /**< @brief ARM NEON available.      */
} zxc_cpu_feature_t;

/**
 * @brief Probes the running CPU for SIMD support.
 *
 * Uses CPUID on x86-64 (MSVC and GCC/Clang paths), `getauxval` on
 * 32-bit ARM Linux, and compile-time constants on AArch64.
 *
 * @return The highest @ref zxc_cpu_feature_t level supported.
 */
// LCOV_EXCL_START
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
    const unsigned long hwcaps = getauxval(AT_HWCAP);
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
// LCOV_EXCL_STOP

/*
 * ============================================================================
 * DISPATCHERS
 * ============================================================================
 * We use a function pointer initialized on first use (lazy initialization).
 */

/** @brief Function pointer type for the chunk decompressor. */
typedef int (*zxc_decompress_func_t)(zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT, const size_t,
                                     uint8_t* RESTRICT, const size_t);
/** @brief Function pointer type for the chunk compressor. */
typedef int (*zxc_compress_func_t)(zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT, const size_t,
                                   uint8_t* RESTRICT, const size_t);

/** @brief Lazily-resolved pointer to the best decompression variant. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_ptr = (zxc_decompress_func_t)0;
/** @brief Lazily-resolved pointer to the best safe-decompression variant. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_safe_ptr = (zxc_decompress_func_t)0;
/** @brief Lazily-resolved pointer to the best compression variant. */
static ZXC_ATOMIC zxc_compress_func_t zxc_compress_ptr = (zxc_compress_func_t)0;

/**
 * @brief First-call initialiser for the decompression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 */
// LCOV_EXCL_START
static int zxc_decompress_dispatch_init(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
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
// LCOV_EXCL_STOP

/**
 * @brief First-call initialiser for the safe-decompression dispatcher.
 *
 * Mirrors @ref zxc_decompress_dispatch_init but selects the `_safe_*`
 * decoder variants used by @ref zxc_decompress_block_safe.
 */
// LCOV_EXCL_START
static int zxc_decompress_safe_dispatch_init(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                             const size_t src_sz, uint8_t* RESTRICT dst,
                                             const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_decompress_func_t zxc_decompress_safe_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_avx512;
    else if (cpu == ZXC_CPU_AVX2)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_avx2;
    else
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_neon;
    else
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#else
    (void)cpu;
    zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#endif
#else
    (void)cpu;
    zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_decompress_safe_ptr, zxc_decompress_safe_ptr_local,
                          memory_order_release);
#else
    zxc_decompress_safe_ptr = zxc_decompress_safe_ptr_local;
#endif
    return zxc_decompress_safe_ptr_local(ctx, src, src_sz, dst, dst_cap);
}
// LCOV_EXCL_STOP

/**
 * @brief First-call initialiser for the compression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 */
// LCOV_EXCL_START
static int zxc_compress_dispatch_init(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
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
// LCOV_EXCL_STOP

/**
 * @brief Public decompression dispatcher (calls lazily-resolved implementation).
 *
 * @param[in,out] ctx    Decompression context.
 * @param[in]     src    Compressed input chunk (header + payload + optional checksum).
 * @param[in]     src_sz Size of @p src in bytes.
 * @param[out]    dst    Destination buffer for decompressed data.
 * @param[in]     dst_cap Capacity of @p dst.
 * @return Decompressed size in bytes, or a negative @ref zxc_error_t code.
 */
int zxc_decompress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    const zxc_decompress_func_t func =
        atomic_load_explicit(&zxc_decompress_ptr, memory_order_acquire);
#else
    const zxc_decompress_func_t func = zxc_decompress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_decompress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/**
 * @brief Internal safe-decompression dispatcher (strict dst_capacity == uncompressed_size).
 */
static int zxc_decompress_chunk_wrapper_safe_public(zxc_cctx_t* RESTRICT ctx,
                                                    const uint8_t* RESTRICT src,
                                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                                    const size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    const zxc_decompress_func_t func =
        atomic_load_explicit(&zxc_decompress_safe_ptr, memory_order_acquire);
#else
    const zxc_decompress_func_t func = zxc_decompress_safe_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_decompress_safe_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/**
 * @brief Public compression dispatcher (calls lazily-resolved implementation).
 *
 * @param[in,out] ctx    Compression context.
 * @param[in]     src    Uncompressed input chunk.
 * @param[in]     src_sz Size of @p src in bytes.
 * @param[out]    dst    Destination buffer for compressed data.
 * @param[in]     dst_cap Capacity of @p dst.
 * @return Compressed size in bytes, or a negative @ref zxc_error_t code.
 */
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    const zxc_compress_func_t func = atomic_load_explicit(&zxc_compress_ptr, memory_order_acquire);
#else
    const zxc_compress_func_t func = zxc_compress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_compress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/*
 * ============================================================================
 * HUFFMAN TRAMPOLINES
 * ============================================================================
 * The Huffman codec is built per-variant (default / avx2 / avx512 / neon)
 * alongside zxc_compress.c and zxc_decompress.c, so the LZ77 stages and the
 * Huffman stage in a given variant share the same ISA flags (e.g. -mbmi2 on
 * the AVX2/AVX512 variants). The compress/decompress variant TUs resolve
 * their Huffman calls to the matching suffixed symbol at compile time, so
 * the production hot path has zero dispatch overhead.
 *
 * These thin wrappers exist only for tests and external callers that link
 * against the un-suffixed names. They forward to the default (scalar) variant.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch) {
    return zxc_huf_build_code_lengths_default(freq, code_len, scratch);
}

int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                           const size_t dst_cap) {
    return zxc_huf_encode_section_default(literals, n_literals, code_len, dst, dst_cap);
}

int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n_literals) {
    return zxc_huf_decode_section_default(payload, payload_size, dst, n_literals);
}

/*
 * ============================================================================
 * PUBLIC UTILITY API
 * ============================================================================
 * These wrapper functions provide a simplified interface by managing context
 * allocation and looping over blocks. They call the dispatched wrappers above.
 */

/**
 * @brief Compresses an entire buffer in one call.
 *
 * Manages context allocation internally, loops over blocks, writes the
 * file header / EOF block / footer, and accumulates the global checksum.
 *
 * @param[in]  src              Uncompressed input data.
 * @param[in]  src_size         Size of @p src in bytes.
 * @param[out] dst              Destination buffer (use zxc_compress_bound() to size).
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  level            Compression level (1-5).
 * @param[in]  checksum_enabled Non-zero to enable per-block and global checksums.
 * @return Total compressed size in bytes, or a negative @ref zxc_error_t code.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_compress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                     const size_t dst_capacity, const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!src || !dst || src_size == 0 || dst_capacity == 0)) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int seekable = opts ? opts->seekable : 0;
    const int level = (opts && opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    uint32_t global_hash = 0;
    zxc_cctx_t ctx;

    // LCOV_EXCL_START
    if (UNLIKELY(zxc_cctx_init(&ctx, block_size, 1, level, checksum_enabled) != ZXC_OK))
        return ZXC_ERROR_MEMORY;
    // LCOV_EXCL_STOP

    const int h_val =
        zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled);
    // LCOV_EXCL_START
    if (UNLIKELY(h_val < 0)) {
        zxc_cctx_free(&ctx);
        return h_val;
    }
    // LCOV_EXCL_STOP
    op += h_val;

    /* Seekable: dynamic array for per-block compressed sizes */
    uint32_t* seek_comp = NULL;
    uint32_t seek_count = 0;
    uint32_t seek_cap = 0;
    if (seekable) {
        const size_t block_count = src_size / block_size;
        if (UNLIKELY(block_count > (size_t)UINT32_MAX - 2)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_BLOCK_SIZE;
        }
        seek_cap = (uint32_t)(block_count + 2);
        seek_comp = (uint32_t*)malloc(seek_cap * sizeof(uint32_t));
        // LCOV_EXCL_START
        if (UNLIKELY(!seek_comp)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_MEMORY;
        }
        // LCOV_EXCL_STOP
    }

    size_t pos = 0;
    while (pos < src_size) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        const int res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap);
        if (UNLIKELY(res < 0)) {
            free(seek_comp);
            zxc_cctx_free(&ctx);
            return res;
        }

        if (checksum_enabled) {
            // Update Global Hash (Rotation + XOR)
            // Block checksum is at the end of the written block data
            if (LIKELY(res >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                const uint32_t block_hash = zxc_le32(op + res - ZXC_GLOBAL_CHECKSUM_SIZE);
                global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
            }
        }

        /* Seekable: record compressed block size */
        if (seekable) {
            // LCOV_EXCL_START
            if (UNLIKELY(seek_count >= seek_cap)) {
                seek_cap = seek_cap * 2;
                uint32_t* nc = (uint32_t*)realloc(seek_comp, seek_cap * sizeof(uint32_t));
                if (UNLIKELY(!nc)) {
                    free(seek_comp);
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_MEMORY;
                }
                seek_comp = nc;
            }
            // LCOV_EXCL_STOP
            seek_comp[seek_count] = (uint32_t)res;
            seek_count++;
        }

        op += res;
        pos += chunk_len;
    }

    zxc_cctx_free(&ctx);

    // Write EOF Block
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    // LCOV_EXCL_START
    if (UNLIKELY(eof_val < 0)) {
        free(seek_comp);
        return eof_val;
    }
    // LCOV_EXCL_STOP
    op += eof_val;

    /* Seekable: write seek table between EOF block and footer */
    if (seekable && seek_count > 0) {
        const size_t st_cap = (size_t)(op_end - op);
        const int64_t st_val = zxc_write_seek_table(op, st_cap, seek_comp, seek_count);
        free(seek_comp);
        if (UNLIKELY(st_val < 0)) return (int64_t)st_val;  // LCOV_EXCL_LINE
        op += st_val;
    } else {
        free(seek_comp);
    }

    if (UNLIKELY((size_t)(op_end - op) < ZXC_FILE_FOOTER_SIZE))
        return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE

    // Write 12-byte Footer: [Source Size (8)] + [Global Hash (4)]
    const int footer_val =
        zxc_write_file_footer(op, (size_t)(op_end - op), src_size, global_hash, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;  // LCOV_EXCL_LINE
    op += footer_val;

    return (int64_t)(op - op_start);
}

/**
 * @brief Decompresses an entire buffer in one call.
 *
 * Validates the file header and footer, loops over compressed blocks,
 * and verifies the global checksum when enabled.
 *
 * @param[in]  src              Compressed input data.
 * @param[in]  src_size         Size of @p src in bytes.
 * @param[out] dst              Destination buffer for decompressed data.
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  checksum_enabled Non-zero to verify per-block and global checksums.
 * @return Total decompressed size in bytes, or a negative @ref zxc_error_t code.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_decompress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                       const size_t dst_capacity, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!src || !dst || src_size < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    zxc_cctx_t ctx;

    int file_has_checksums = 0;
    // File header verification and context initialization
    if (UNLIKELY(zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums) !=
                     ZXC_OK ||
                 zxc_cctx_init(&ctx, runtime_chunk_size, 0, 0,
                               file_has_checksums && checksum_enabled) != ZXC_OK)) {
        return ZXC_ERROR_BAD_HEADER;
    }

    ip += ZXC_FILE_HEADER_SIZE;

    // GLO/GHI wild copies (zxc_copy32) overshoot by up to ZXC_PAD_SIZE bytes.
    // Decode into a padded scratch buffer, then memcpy the exact result out.
    const size_t work_sz = runtime_chunk_size + ZXC_PAD_SIZE;
    if (ctx.work_buf_cap < work_sz) {
        free(ctx.work_buf);
        ctx.work_buf = (uint8_t*)malloc(work_sz);
        // LCOV_EXCL_START
        if (UNLIKELY(!ctx.work_buf)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_MEMORY;
        }
        // LCOV_EXCL_STOP
        ctx.work_buf_cap = work_sz;
    }

    // Block decompression loop
    uint32_t global_hash = 0;

    while (ip < ip_end) {
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        // Read the block header to determine the compressed size
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != ZXC_OK)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_HEADER;
        }

        // Handle EOF block separately (not a real chunk to decompress)
        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            // Footer is always the last ZXC_FILE_FOOTER_SIZE bytes of the source,
            // even when a seek table is inserted between EOF block and footer.
            // LCOV_EXCL_START
            if (UNLIKELY(src_size < ZXC_FILE_FOOTER_SIZE)) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_SRC_TOO_SMALL;
            }
            // LCOV_EXCL_STOP
            const uint8_t* const footer = (const uint8_t*)src + src_size - ZXC_FILE_FOOTER_SIZE;

            // Validate source size matches what we decompressed
            const uint64_t stored_size = zxc_le64(footer);
            if (UNLIKELY(stored_size != (uint64_t)(op - op_start))) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_CORRUPT_DATA;
            }

            // Validate global checksum if enabled and file has checksums
            if (checksum_enabled && file_has_checksums) {
                const uint32_t stored_hash = zxc_le32(footer + sizeof(uint64_t));
                if (UNLIKELY(stored_hash != global_hash)) {
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_BAD_CHECKSUM;
                }
            }
            break;  // EOF reached, exit loop
        }

        int res;
        const size_t rem_cap = (size_t)(op_end - op);
        if (LIKELY(rem_cap >= work_sz)) {
            // Fast path: decode directly into dst. Cap dst_cap to chunk_size + PAD
            res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, op, work_sz);
        } else {
            // Safe path: decode into bounce buffer, then copy exact result.
            res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, ctx.work_buf, ctx.work_buf_cap);
            if (LIKELY(res > 0)) {
                // LCOV_EXCL_START
                if (UNLIKELY((size_t)res > rem_cap)) {
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_DST_TOO_SMALL;
                }
                // LCOV_EXCL_STOP
                ZXC_MEMCPY(op, ctx.work_buf, (size_t)res);
            }
        }
        if (UNLIKELY(res < 0)) {
            zxc_cctx_free(&ctx);
            return res;
        }

        // Update global hash from block checksum
        if (checksum_enabled && file_has_checksums) {
            const uint32_t block_hash = zxc_le32(ip + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
            global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
        }

        ip += ZXC_BLOCK_HEADER_SIZE + bh.comp_size +
              (file_has_checksums ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
        op += res;
    }

    zxc_cctx_free(&ctx);
    return (int64_t)(op - op_start);
}

/**
 * @brief Reads the decompressed size from a ZXC-compressed buffer.
 *
 * The size is stored in the file footer (last @ref ZXC_FILE_FOOTER_SIZE bytes).
 *
 * @param[in] src      Compressed data.
 * @param[in] src_size Size of @p src in bytes.
 * @return Original uncompressed size, or 0 on error.
 */
uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE)) return 0;

    const uint8_t* const p = (const uint8_t*)src;
    if (UNLIKELY(zxc_le32(p) != ZXC_MAGIC_WORD)) return 0;

    const uint8_t* const footer = p + src_size - ZXC_FILE_FOOTER_SIZE;
    return zxc_le64(footer);
}

/*
 * ============================================================================
 * REUSABLE CONTEXT API (Opaque)
 * ============================================================================
 *
 * Provides heap-allocated, opaque contexts that integrators can reuse across
 * multiple compress / decompress calls, eliminating per-call malloc/free
 * overhead.
 */

/* --- Compression --------------------------------------------------------- */

struct zxc_cctx_s {
    zxc_cctx_t inner;       /* existing internal context */
    int initialized;        /* 1 if inner has live allocations */
    size_t last_block_size; /* block size used for last init */
    /* Sticky options (remembered from create or last compress call). */
    int stored_level;
    int stored_checksum;
    size_t stored_block_size;
};

zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts) {
    zxc_cctx* const cctx = (zxc_cctx*)calloc(1, sizeof(zxc_cctx));
    if (UNLIKELY(!cctx)) return NULL;  // LCOV_EXCL_LINE

    /* Resolve and store sticky defaults. */
    cctx->stored_level = (opts && opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    cctx->stored_block_size =
        (opts && opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    cctx->stored_checksum = opts ? opts->checksum_enabled : 0;

    if (opts) {
        // LCOV_EXCL_START
        if (UNLIKELY(!zxc_validate_block_size(cctx->stored_block_size) ||
                     zxc_cctx_init(&cctx->inner, cctx->stored_block_size, 1, cctx->stored_level,
                                   cctx->stored_checksum) != ZXC_OK)) {
            free(cctx);
            return NULL;
        }
        // LCOV_EXCL_STOP
        cctx->last_block_size = cctx->stored_block_size;
        cctx->initialized = 1;
    }

    return cctx;
}

void zxc_free_cctx(zxc_cctx* cctx) {
    if (UNLIKELY(!cctx)) return;
    if (cctx->initialized) zxc_cctx_free(&cctx->inner);
    free(cctx);
}

int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                          void* RESTRICT dst, const size_t dst_capacity,
                          const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!cctx)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(!src || !dst || src_size == 0 || dst_capacity == 0)) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int level = (opts && opts->level > 0) ? opts->level : cctx->stored_level;
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : cctx->stored_block_size;

    cctx->stored_level = level;
    cctx->stored_block_size = block_size;
    cctx->stored_checksum = checksum_enabled;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    /* Re-init only when block_size changed (it drives buffer sizes). */
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != block_size)) {
        if (cctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, block_size, 1, level, checksum_enabled) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = block_size;
        cctx->initialized = 1;
    } else {
        /* Same block_size: update level + checksum without realloc. */
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    zxc_cctx_t* const ctx = &cctx->inner;

    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    const uint8_t* const ip = (const uint8_t*)src;
    uint32_t global_hash = 0;

    const int h_val =
        zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled);
    if (UNLIKELY(h_val < 0)) return h_val;  // LCOV_EXCL_LINE
    op += h_val;

    size_t pos = 0;
    while (pos < src_size) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        const int res = zxc_compress_chunk_wrapper(ctx, ip + pos, chunk_len, op, rem_cap);
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled) {
            if (LIKELY(res >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                const uint32_t block_hash = zxc_le32(op + res - ZXC_GLOBAL_CHECKSUM_SIZE);
                global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
            }
        }

        op += res;
        pos += chunk_len;
    }

    /* EOF block */
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    if (UNLIKELY(eof_val < 0)) return eof_val;  // LCOV_EXCL_LINE
    op += eof_val;

    if (UNLIKELY(rem_cap < (size_t)eof_val + ZXC_FILE_FOOTER_SIZE))
        return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE

    const int footer_val =
        zxc_write_file_footer(op, (size_t)(op_end - op), src_size, global_hash, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;  // LCOV_EXCL_LINE
    op += footer_val;

    return (int64_t)(op - op_start);
}

/* --- Decompression ------------------------------------------------------- */

struct zxc_dctx_s {
    zxc_cctx_t inner;       /* reuses the same internal context type */
    size_t last_block_size; /* block size from last header parse */
    int initialized;        /* 1 if inner has live allocations */
};

zxc_dctx* zxc_create_dctx(void) {
    zxc_dctx* const dctx = (zxc_dctx*)calloc(1, sizeof(zxc_dctx));
    return dctx;
}

void zxc_free_dctx(zxc_dctx* dctx) {
    if (UNLIKELY(!dctx)) return;
    if (dctx->initialized) zxc_cctx_free(&dctx->inner);
    free(dctx);
}

int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                            void* RESTRICT dst, const size_t dst_capacity,
                            const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_FILE_HEADER_SIZE))
        return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* const ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    int file_has_checksums = 0;
    uint32_t global_hash = 0;

    if (UNLIKELY(zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums) !=
                 ZXC_OK))
        return ZXC_ERROR_BAD_HEADER;

    /* Re-init only when block size changed. */
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != runtime_chunk_size)) {
        if (dctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, runtime_chunk_size, 0, 0,
                                   file_has_checksums && checksum_enabled) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = runtime_chunk_size;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = file_has_checksums && checksum_enabled;
    }

    zxc_cctx_t* const ctx = &dctx->inner;
    ip += ZXC_FILE_HEADER_SIZE;

    /* Ensure scratch buffer is large enough. */
    const size_t work_sz = runtime_chunk_size + ZXC_PAD_SIZE;
    if (UNLIKELY(ctx->work_buf_cap < work_sz)) {
        free(ctx->work_buf);
        ctx->work_buf = (uint8_t*)malloc(work_sz);
        if (UNLIKELY(!ctx->work_buf)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        ctx->work_buf_cap = work_sz;
    }

    while (ip < ip_end) {
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != ZXC_OK))
            return ZXC_ERROR_BAD_HEADER;

        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            if (UNLIKELY(rem_src < ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE))
                return ZXC_ERROR_SRC_TOO_SMALL;

            const uint8_t* const footer = ip + ZXC_BLOCK_HEADER_SIZE;
            const uint64_t stored_size = zxc_le64(footer);
            if (UNLIKELY(stored_size != (uint64_t)(op - op_start))) return ZXC_ERROR_CORRUPT_DATA;

            if (checksum_enabled && file_has_checksums) {
                const uint32_t stored_hash = zxc_le32(footer + sizeof(uint64_t));
                if (UNLIKELY(stored_hash != global_hash)) return ZXC_ERROR_BAD_CHECKSUM;
            }
            break;
        }

        const size_t rem_cap = (size_t)(op_end - op);
        int res;
        if (LIKELY(rem_cap >= work_sz)) {
            // Fast path: decode directly into dst (enough padding for wild copies).
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, op, rem_cap);
        } else {
            // Safe path: decode into bounce buffer, then copy exact result.
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, ctx->work_buf, ctx->work_buf_cap);
            if (LIKELY(res > 0)) {
                if (UNLIKELY((size_t)res > rem_cap))
                    return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE
                ZXC_MEMCPY(op, ctx->work_buf, (size_t)res);
            }
        }
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled && file_has_checksums) {
            const uint32_t block_hash = zxc_le32(ip + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
            global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
        }

        ip += ZXC_BLOCK_HEADER_SIZE + bh.comp_size +
              (file_has_checksums ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
        op += res;
    }

    return (int64_t)(op - op_start);
}

/* ========================================================================= */
/*  Block-Level API (no file framing)                                        */
/* ========================================================================= */

int64_t zxc_compress_block(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                           void* RESTRICT dst, const size_t dst_capacity,
                           const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!cctx || !src || !dst || src_size == 0 || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int level = (opts && opts->level > 0) ? opts->level : cctx->stored_level;
    /* For block API, block_size == src_size (the caller compresses one block at a time). */
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : cctx->stored_block_size;
    const size_t min_bs = zxc_block_size_ceil(src_size);

    /* Always ensure internal buffers can hold src_size. */
    const size_t effective_block_size = (block_size > min_bs) ? block_size : min_bs;

    cctx->stored_level = level;
    cctx->stored_block_size = effective_block_size;
    cctx->stored_checksum = checksum_enabled;

    /* Re-init only when block_size changed. */
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != effective_block_size)) {
        if (cctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, effective_block_size, 1, level,
                                   checksum_enabled) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = effective_block_size;
        cctx->initialized = 1;
    } else {
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    const int res = zxc_compress_chunk_wrapper(&cctx->inner, (const uint8_t*)src, src_size,
                                               (uint8_t*)dst, dst_capacity);
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

int64_t zxc_decompress_block(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                             void* RESTRICT dst, const size_t dst_capacity,
                             const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    /* Derive the block_size from dst_capacity (callers know the original size). */
    const size_t block_size = zxc_block_size_ceil(dst_capacity);
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != block_size)) {
        if (dctx->initialized) {
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, block_size, 0, 0, checksum_enabled) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = block_size;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = checksum_enabled;
    }

    zxc_cctx_t* const ctx = &dctx->inner;

    /* Ensure scratch buffer for safe-path wild copies. */
    const size_t work_sz = block_size + ZXC_PAD_SIZE;
    if (ctx->work_buf_cap < work_sz) {
        free(ctx->work_buf);
        ctx->work_buf = (uint8_t*)malloc(work_sz);
        if (UNLIKELY(!ctx->work_buf)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        ctx->work_buf_cap = work_sz;
    }

    int res;
    if (LIKELY(dst_capacity >= work_sz)) {
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                           dst_capacity);
    } else {
        /* Bounce through work_buf when output can't absorb wild copies. */
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, ctx->work_buf,
                                           ctx->work_buf_cap);
        if (LIKELY(res > 0)) {
            if (UNLIKELY((size_t)res > dst_capacity)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, ctx->work_buf, (size_t)res);
        }
    }
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/**
 * @brief Safe-variant block decompressor: accepts dst_capacity == uncompressed_size.
 *
 * Router: NUM/RAW blocks (which never wild-write past dst_capacity) are
 * forwarded to the existing fast path. GLO/GHI blocks use the strict safe
 * decoder, avoiding the bounce buffer and the +ZXC_DECOMPRESS_TAIL_PAD
 * requirement of @ref zxc_decompress_block.
 */
int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                                  void* RESTRICT dst, const size_t dst_capacity,
                                  const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    const uint8_t type = ((const uint8_t*)src)[0];
    /* NUM/RAW never wild-write past dst_capacity: route to the existing fast API. */
    if (type == ZXC_BLOCK_NUM || type == ZXC_BLOCK_RAW) {
        return zxc_decompress_block(dctx, src, src_size, dst, dst_capacity, opts);
    }

    /* GLO/GHI: use the strict-tail decoder (no bounce buffer required). */
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const size_t block_size = zxc_block_size_ceil(dst_capacity);
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != block_size)) {
        if (dctx->initialized) {
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, block_size, 0, 0, checksum_enabled) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = block_size;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = checksum_enabled;
    }

    const int res = zxc_decompress_chunk_wrapper_safe_public(&dctx->inner, (const uint8_t*)src,
                                                             src_size, (uint8_t*)dst, dst_capacity);
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}
