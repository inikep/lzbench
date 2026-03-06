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
int zxc_decompress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
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
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_decompress_chunk_wrapper_neon(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
#endif
#endif

// Compression Prototypes
int zxc_compress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                       const size_t src_sz, uint8_t* RESTRICT dst,
                                       const size_t dst_cap);

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
/** @brief Lazily-resolved pointer to the best compression variant. */
static ZXC_ATOMIC zxc_compress_func_t zxc_compress_ptr = (zxc_compress_func_t)0;

/**
 * @brief First-call initialiser for the decompression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 */
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

/**
 * @brief First-call initialiser for the compression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 */
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
                     const size_t dst_capacity, const int level, const int checksum_enabled) {
    if (UNLIKELY(!src || !dst || src_size == 0 || dst_capacity == 0)) return ZXC_ERROR_NULL_INPUT;

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    uint32_t global_hash = 0;
    zxc_cctx_t ctx;

    if (UNLIKELY(zxc_cctx_init(&ctx, ZXC_BLOCK_SIZE, 1, level, checksum_enabled) != 0))
        return ZXC_ERROR_MEMORY;

    const int h_val = zxc_write_file_header(op, (size_t)(op_end - op), checksum_enabled);
    if (UNLIKELY(h_val < 0)) {
        zxc_cctx_free(&ctx);
        return h_val;
    }
    op += h_val;

    size_t pos = 0;
    while (pos < src_size) {
        const size_t chunk_len =
            (src_size - pos > ZXC_BLOCK_SIZE) ? ZXC_BLOCK_SIZE : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        const int res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap);
        if (UNLIKELY(res < 0)) {
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

        op += res;
        pos += chunk_len;
    }

    zxc_cctx_free(&ctx);

    // Write EOF Block (Checksum flag handled by Block Header, but we zero it out now)
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    if (UNLIKELY(eof_val < 0)) return eof_val;
    op += eof_val;

    if (UNLIKELY(rem_cap < (size_t)eof_val + ZXC_FILE_FOOTER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    // Write 12-byte Footer: [Source Size (8)] + [Global Hash (4)]
    const int footer_val =
        zxc_write_file_footer(op, (size_t)(op_end - op), src_size, global_hash, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;
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
                       const size_t dst_capacity, const int checksum_enabled) {
    if (UNLIKELY(!src || !dst || src_size < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_NULL_INPUT;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    zxc_cctx_t ctx;

    int file_has_checksums = 0;
    // File header verification and context initialization
    if (UNLIKELY(
            zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums) != 0 ||
            zxc_cctx_init(&ctx, runtime_chunk_size, 0, 0, file_has_checksums && checksum_enabled) !=
                0)) {
        return ZXC_ERROR_BAD_HEADER;
    }

    ip += ZXC_FILE_HEADER_SIZE;

    // GLO/GHI wild copies (zxc_copy32) overshoot by up to ZXC_PAD_SIZE bytes.
    // Decode into a padded scratch buffer, then memcpy the exact result out.
    const size_t work_sz = runtime_chunk_size + ZXC_PAD_SIZE;
    if (ctx.work_buf_cap < work_sz) {
        free(ctx.work_buf);
        ctx.work_buf = (uint8_t*)malloc(work_sz);
        if (UNLIKELY(!ctx.work_buf)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_MEMORY;
        }
        ctx.work_buf_cap = work_sz;
    }

    // Block decompression loop
    uint32_t global_hash = 0;

    while (ip < ip_end) {
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        // Read the block header to determine the compressed size
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != 0)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_HEADER;
        }

        // Handle EOF block separately (not a real chunk to decompress)
        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            // Validate we have the footer after the header
            if (UNLIKELY(rem_src < ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE)) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_SRC_TOO_SMALL;
            }
            const uint8_t* const footer = ip + ZXC_BLOCK_HEADER_SIZE;

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

        const size_t rem_cap = (size_t)(op_end - op);
        const int res =
            zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, ctx.work_buf, runtime_chunk_size);
        if (UNLIKELY(res < 0 || (size_t)res > rem_cap)) {
            zxc_cctx_free(&ctx);
            return (res < 0) ? res : ZXC_ERROR_DST_TOO_SMALL;
        }
        ZXC_MEMCPY(op, ctx.work_buf, (size_t)res);

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
