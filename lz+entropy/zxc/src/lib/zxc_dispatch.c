// SPDX-License-Identifier: BSD-3-Clause
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_dispatch.c
 * @brief Runtime CPU feature detection and SIMD dispatch layer.
 *
 * Detects AVX2/AVX512 (x86-64) and NEON (32-bit ARM) at runtime and routes
 * compress/decompress calls to the best available implementation via
 * lazy-initialised function pointers. SSE2 on x86-64 and NEON on AArch64 are
 * baseline ISA guarantees, so the _default variant already covers those tiers.
 * Also contains the public one-shot buffer API (@ref zxc_compress,
 * @ref zxc_decompress, @ref zxc_get_decompressed_size).
 */

#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"

// ZXC_DISABLE_SIMD => force ZXC_ONLY_DEFAULT so the dispatcher never selects
// an AVX2/AVX512/NEON variant.
#if defined(ZXC_DISABLE_SIMD) && !defined(ZXC_ONLY_DEFAULT)
#define ZXC_ONLY_DEFAULT
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64)
#include <immintrin.h>  // _xgetbv (x86-specific header; x64 AVX state check)
#endif
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER) && !defined(ZXC_ONLY_DEFAULT)
#include <cpuid.h>  // __cpuid_count: CPUID probes in zxc_detect_cpu_features
#endif

#if defined(__linux__) && (defined(__arm__) || defined(_M_ARM)) && !defined(ZXC_ONLY_DEFAULT)
#include <sys/auxv.h>
// musl does not ship <asm/hwcap.h>; HWCAP_NEON is stable arm32 UAPI. Nested:
// naming __has_include in a single #if is a syntax error where it is missing.
#ifdef __has_include
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif
#ifndef HWCAP_NEON
#define HWCAP_NEON (1 << 12)
#endif
#endif

// ============================================================================
// PROTOTYPES FOR MULTI-VERSIONED VARIANTS
// ============================================================================
// These are compiled in separate translation units with different flags.

// Decompression Prototypes
int zxc_decompress_chunk_wrapper_default(const zxc_cctx_t* RESTRICT ctx,
                                         const uint8_t* RESTRICT src, const size_t src_sz,
                                         uint8_t* RESTRICT dst, const size_t dst_cap,
                                         const uint64_t block_index);
int zxc_decompress_chunk_wrapper_dict_default(const zxc_cctx_t* RESTRICT ctx,
                                              const uint8_t* RESTRICT src, const size_t src_sz,
                                              uint8_t* RESTRICT dst, const size_t dst_cap,
                                              const uint64_t block_index);
int zxc_decompress_chunk_wrapper_safe_default(const zxc_cctx_t* RESTRICT ctx,
                                              const uint8_t* RESTRICT src, const size_t src_sz,
                                              uint8_t* RESTRICT dst, const size_t dst_cap,
                                              const uint64_t block_index);

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
int zxc_decompress_chunk_wrapper_avx2(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap, const uint64_t block_index);
int zxc_decompress_chunk_wrapper_dict_avx2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap,
                                           const uint64_t block_index);
int zxc_decompress_chunk_wrapper_avx512(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap, const uint64_t block_index);
int zxc_decompress_chunk_wrapper_dict_avx512(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap,
                                             const uint64_t block_index);
int zxc_decompress_chunk_wrapper_safe_avx2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap,
                                           const uint64_t block_index);
int zxc_decompress_chunk_wrapper_safe_avx512(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap,
                                             const uint64_t block_index);
#elif defined(__arm__) || defined(_M_ARM)
int zxc_decompress_chunk_wrapper_neon32(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap, const uint64_t block_index);
int zxc_decompress_chunk_wrapper_dict_neon32(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap,
                                             const uint64_t block_index);
int zxc_decompress_chunk_wrapper_safe_neon32(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap,
                                             const uint64_t block_index);
#endif
#endif

// Compression Prototypes
int zxc_compress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                       const size_t src_sz, uint8_t* RESTRICT dst,
                                       const size_t dst_cap, const uint64_t block_index);

// Huffman prototypes (variant TUs of zxc_huffman.c). Compressor and decompressor
// variants bind to the matching suffixed symbol at compile time, so the hot path
// pays no dispatch; the wrappers below expose un-suffixed names to callers.
int zxc_huf_build_code_lengths_default(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                                       void* RESTRICT scratch, int max_code_len);
size_t zxc_huf_calc_size_default(const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                                 int with_header);
int zxc_huf_encode_section_default(const uint8_t* RESTRICT literals, size_t n_literals,
                                   const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                                   uint8_t* RESTRICT dst, size_t dst_cap);
int zxc_huf_decode_section_default(const uint8_t* RESTRICT payload, size_t payload_size,
                                   uint8_t* RESTRICT dst, size_t n, uint8_t* RESTRICT scratch);
int zxc_huf_encode_section_dict_default(const uint8_t* RESTRICT literals, size_t n_literals,
                                        const uint32_t* RESTRICT freq,
                                        const uint8_t* RESTRICT code_len,
                                        const zxc_pivco_tree_t* RESTRICT tree,
                                        const uint32_t* RESTRICT codes, uint8_t* RESTRICT dst,
                                        size_t dst_cap);
int zxc_huf_decode_section_dict_default(const uint8_t* RESTRICT payload, size_t payload_size,
                                        uint8_t* RESTRICT dst, size_t n,
                                        const zxc_pivco_tree_t* RESTRICT tree,
                                        uint8_t* RESTRICT scratch);
size_t zxc_huf_calc_size_dict_default(const uint32_t* RESTRICT freq,
                                      const uint8_t* RESTRICT code_len,
                                      const zxc_pivco_tree_t* RESTRICT tree);
void zxc_huf_pack_lengths_default(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out);
int zxc_huf_unpack_lengths_default(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len);

#if defined(__x86_64__) || defined(_M_X64)
int zxc_compress_chunk_wrapper_avx2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap, const uint64_t block_index);
int zxc_compress_chunk_wrapper_avx512(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap, const uint64_t block_index);
#elif defined(__arm__) || defined(_M_ARM)
int zxc_compress_chunk_wrapper_neon32(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap, const uint64_t block_index);
#endif

// ============================================================================
// CPU DETECTION LOGIC
// ============================================================================

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(ZXC_ONLY_DEFAULT)

/** @brief Reads CPUID leaf @p leaf, subleaf @p sub into @p regs (EAX,EBX,ECX,EDX). */
static inline void zxc_cpuid(const uint32_t leaf, const uint32_t sub, uint32_t regs[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, (int)leaf, (int)sub);
    regs[0] = (uint32_t)r[0];
    regs[1] = (uint32_t)r[1];
    regs[2] = (uint32_t)r[2];
    regs[3] = (uint32_t)r[3];
#else
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
}

/** @brief Reads XCR0 (@c XGETBV with ECX=0). Callers must check OSXSAVE first. */
static inline uint64_t zxc_xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    // Raw encoding: the xgetbv intrinsic needs -mxsave, which the baseline
    // translation unit is not compiled with.
    uint32_t lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif /* x86-64 && !ZXC_ONLY_DEFAULT */

/** @brief Function pointer type for the chunk decompressor. */
typedef int (*zxc_decompress_func_t)(const zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT,
                                     const size_t, uint8_t* RESTRICT, const size_t, const uint64_t);
/** @brief Function pointer type for the chunk compressor. */
typedef int (*zxc_compress_func_t)(zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT, const size_t,
                                   uint8_t* RESTRICT, const size_t, const uint64_t);

#ifndef ZXC_ONLY_DEFAULT

/**
 * @enum zxc_cpu_feature_t
 * @brief Detected CPU SIMD capability level.
 */
typedef enum {
    ZXC_CPU_GENERIC = 0, /**< @brief Scalar-only fallback.   */
    ZXC_CPU_AVX2 = 1,    /**< @brief x86-64 AVX2 available.  */
    ZXC_CPU_AVX512 = 2,  /**< @brief x86-64 AVX-512F+BW available. */
    ZXC_CPU_NEON = 3,    /**< @brief ARM NEON available (dedicated variant on 32-bit ARM only;
                          *          AArch64 baseline, served by _default there). */
    ZXC_CPU_SSE2 = 4     /**< @brief x86 SSE2 available (no AVX2); x86-64 baseline,
                          *          served by _default (no dedicated variant). */
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
    zxc_cpu_feature_t features = ZXC_CPU_GENERIC;

#if defined(__x86_64__) || defined(_M_X64)
    // AVX2/AVX512 need OS-enabled YMM/ZMM state: gate on OSXSAVE + XGETBV/XCR0,
    // not CPUID alone (else a VEX/EVEX op faults #UD when the OS hasn't enabled it).
    uint32_t regs[4];
    int sse2 = 0;
    int avx2 = 0;
    int avx512 = 0;
    int bmi_lzcnt = 0;

    zxc_cpuid(0, 0, regs);
    const uint32_t max_leaf = regs[0];

    zxc_cpuid(1, 0, regs);
    if (regs[3] & (1U << 26)) sse2 = 1;             // CPUID.1:EDX[26]
    if ((regs[2] & (1U << 27)) && max_leaf >= 7) {  // OSXSAVE, and leaf 7 is real
        const uint64_t xcr0 = zxc_xgetbv0();
        if ((xcr0 & 0x6) == 0x6) {  // SSE+YMM enabled
            zxc_cpuid(7, 0, regs);
            const int bmi1 = (regs[1] >> 3) & 1;  // CPUID.7.0:EBX[3]
            const int bmi2 = (regs[1] >> 8) & 1;  // CPUID.7.0:EBX[8]
            if (regs[1] & (1U << 5)) avx2 = 1;
            // AVX512 also needs XCR0[5..7] (opmask/ZMM)
            if ((regs[1] & (1U << 16)) && (regs[1] & (1U << 30)) && (regs[2] & (1U << 1)) &&
                (regs[2] & (1U << 6)) && (xcr0 & 0xE0) == 0xE0)
                avx512 = 1; /* AVX512 tier = F+BW+VBMI+VBMI2, as the variant is built */
            // The AVX2/AVX512 variants are compiled with BMI1/BMI2/LZCNT enabled,
            // so both gates must prove those bits too. LZCNT (ABM) lives in
            // CPUID.80000001H:ECX[5]; that leaf is architectural on x86-64.
            zxc_cpuid(0x80000001U, 0, regs);
            bmi_lzcnt = bmi1 && bmi2 && ((regs[2] >> 5) & 1);
        }
    }

    if (avx512 && bmi_lzcnt) {
        features = ZXC_CPU_AVX512;
    } else if (avx2 && bmi_lzcnt) {
        features = ZXC_CPU_AVX2;
    } else if (sse2) {
        features = ZXC_CPU_SSE2;
    }

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
}
// LCOV_EXCL_STOP

#endif  // ZXC_ONLY_DEFAULT

// ============================================================================
// DISPATCHERS
// ============================================================================
// One constant variant set per ISA, one published pointer to the selected one;
// ZXC_ONLY_DEFAULT binds _default at compile time.

/**
 * @struct zxc_variant_set_t
 * @brief The three chunk entry points of one ISA variant; adding a tier is one
 *        line here.
 *
 * The `_safe` decoder is kept out on purpose: a table entry is a reference the
 * linker cannot prove dead, so every static link would carry its decoders.
 */
typedef struct {
    zxc_decompress_func_t decompress;
    zxc_decompress_func_t decompress_dict;
    zxc_compress_func_t compress;
} zxc_variant_set_t;

/** @brief Defines the constant variant set of suffix @p sfx. */
#define ZXC_VARIANT_SET(sfx)                                                                    \
    static const zxc_variant_set_t zxc_variants##sfx = {zxc_decompress_chunk_wrapper##sfx,      \
                                                        zxc_decompress_chunk_wrapper_dict##sfx, \
                                                        zxc_compress_chunk_wrapper##sfx}

ZXC_VARIANT_SET(_default);
#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
ZXC_VARIANT_SET(_avx2);
ZXC_VARIANT_SET(_avx512);
#elif defined(__arm__) || defined(_M_ARM)
ZXC_VARIANT_SET(_neon32);
#endif
#endif
#undef ZXC_VARIANT_SET

#ifdef ZXC_ONLY_DEFAULT
/** @brief The variant set in use: `_default`, bound at compile time. */
static ZXC_ALWAYS_INLINE const zxc_variant_set_t* zxc_variants(void) {
    return &zxc_variants_default;
}
#else
/**
 * @brief Detects the CPU tier and returns the matching variant set.
 *
 * Falls back to the `_default` (baseline) set when no ISA extension applies.
 */
// LCOV_EXCL_START
static const zxc_variant_set_t* zxc_select_variants(void) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512) return &zxc_variants_avx512;
    if (cpu == ZXC_CPU_AVX2) return &zxc_variants_avx2;
#elif defined(__arm__) || defined(_M_ARM)
    // 32-bit ARM: the only arch with a real runtime NEON probe (getauxval).
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON) return &zxc_variants_neon32;
#else
    (void)cpu;
#endif
    return &zxc_variants_default;
}
// LCOV_EXCL_STOP

/** @brief The selection, NULL until the first call. */
static const zxc_variant_set_t* ZXC_ATOMIC zxc_variants_ptr = NULL;

/** @brief First call: selects and publishes. Cold and out of line, so the
 *         dispatchers inline only the load and the branch. */
// LCOV_EXCL_START
static ZXC_COLD ZXC_NOINLINE const zxc_variant_set_t* zxc_variants_resolve(void) {
    const zxc_variant_set_t* v = zxc_select_variants();
#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_variants_ptr, v, memory_order_release);
#else
    zxc_variants_ptr = v;
#endif
    return v;
}
// LCOV_EXCL_STOP

/** @brief The variant set in use, selected on the first call. */
static ZXC_ALWAYS_INLINE const zxc_variant_set_t* zxc_variants(void) {
#if ZXC_USE_C11_ATOMICS
    const zxc_variant_set_t* v = atomic_load_explicit(&zxc_variants_ptr, memory_order_acquire);
#else
    const zxc_variant_set_t* v = zxc_variants_ptr;
#endif
    if (UNLIKELY(!v)) v = zxc_variants_resolve();
    return v;
}
#endif  // ZXC_ONLY_DEFAULT

#ifdef ZXC_ONLY_DEFAULT
/** @brief The `_safe` decoder in use: `_default`, bound at compile time. */
static ZXC_ALWAYS_INLINE zxc_decompress_func_t zxc_decompress_safe_variant(void) {
    return zxc_decompress_chunk_wrapper_safe_default;
}
#else
/** @brief The `_safe` decoder in use, NULL until the first call. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_safe_ptr = (zxc_decompress_func_t)0;

/** @brief First call: the `_safe` decoder of the selected set, published. */
// LCOV_EXCL_START
static ZXC_COLD ZXC_NOINLINE zxc_decompress_func_t zxc_decompress_safe_resolve(void) {
    const zxc_variant_set_t* v = zxc_variants();
    zxc_decompress_func_t f = zxc_decompress_chunk_wrapper_safe_default;
#if defined(__x86_64__) || defined(_M_X64)
    if (v == &zxc_variants_avx512) f = zxc_decompress_chunk_wrapper_safe_avx512;
    if (v == &zxc_variants_avx2) f = zxc_decompress_chunk_wrapper_safe_avx2;
#elif defined(__arm__) || defined(_M_ARM)
    if (v == &zxc_variants_neon32) f = zxc_decompress_chunk_wrapper_safe_neon32;
#else
    (void)v;
#endif
#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_decompress_safe_ptr, f, memory_order_release);
#else
    zxc_decompress_safe_ptr = f;
#endif
    return f;
}
// LCOV_EXCL_STOP

/** @brief The `_safe` decoder in use, selected on the first call. */
static ZXC_ALWAYS_INLINE zxc_decompress_func_t zxc_decompress_safe_variant(void) {
#if ZXC_USE_C11_ATOMICS
    zxc_decompress_func_t f = atomic_load_explicit(&zxc_decompress_safe_ptr, memory_order_acquire);
#else
    zxc_decompress_func_t f = zxc_decompress_safe_ptr;
#endif
    if (UNLIKELY(!f)) f = zxc_decompress_safe_resolve();
    return f;
}
#endif  // ZXC_ONLY_DEFAULT

/** @brief Public decompression dispatcher. */
int zxc_decompress_chunk_wrapper(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                                 const uint64_t block_index) {
    // dict_size is constant for a stream; this per-block branch (outside the decode
    // loop) routes to the dict variant only when a dictionary is active, so the
    // no-dict path runs the dict-free chunk wrapper (identical codegen to main).
    const zxc_variant_set_t* v = zxc_variants();
    return (ctx->dict_size ? v->decompress_dict : v->decompress)(ctx, src, src_sz, dst, dst_cap,
                                                                 block_index);
}

/**
 * @brief Internal safe-decompression dispatcher (strict dst_capacity == uncompressed_size).
 *
 * @param[in]  ctx      Decompression context.
 * @param[in]  src      Compressed input chunk.
 * @param[in]  src_sz   Size of @p src in bytes.
 * @param[out] dst      Destination buffer (exactly the decoded size).
 * @param[in]  dst_cap  Capacity of @p dst.
 * @param[in]  block_index Frame position of the block: the checksum seed.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static int zxc_decompress_chunk_wrapper_safe_public(const zxc_cctx_t* RESTRICT ctx,
                                                    const uint8_t* RESTRICT src,
                                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                                    const size_t dst_cap,
                                                    const uint64_t block_index) {
    return zxc_decompress_safe_variant()(ctx, src, src_sz, dst, dst_cap, block_index);
}

/** @brief Public compression dispatcher. */
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                               const uint64_t block_index) {
    return zxc_variants()->compress(ctx, src, src_sz, dst, dst_cap, block_index);
}

// ============================================================================
// HUFFMAN TRAMPOLINES
// ============================================================================
// The Huffman codec is built per-variant (default / avx2 / avx512, plus neon32
// on 32-bit ARM)
// alongside zxc_compress.c and zxc_decompress.c, so the LZ77 stages and the
// Huffman stage in a given variant share the same ISA flags (e.g. -mbmi2 on
// the AVX2/AVX512 variants). The compress/decompress variant TUs resolve
// their Huffman calls to the matching suffixed symbol at compile time, so
// the production hot path has zero dispatch overhead.
//
// These thin wrappers exist only for tests and external callers that link
// against the un-suffixed names. They forward to the default (scalar) variant.
/**
 * @brief Build length-limited per-symbol Huffman code lengths from frequencies.
 *
 * Un-suffixed entry forwarding to `zxc_huf_build_code_lengths_default`; full
 * contract in @c zxc_internal.h.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch, const int max_code_len) {
    return zxc_huf_build_code_lengths_default(freq, code_len, scratch, max_code_len);
}

/** @brief Un-suffixed forwarders for the PivCo section codec (tests, tools). */
size_t zxc_huf_calc_size(const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                         const int with_header) {
    return zxc_huf_calc_size_default(freq, code_len, with_header);
}

int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                           uint8_t* RESTRICT dst, const size_t dst_cap) {
    return zxc_huf_encode_section_default(literals, n_literals, freq, code_len, dst, dst_cap);
}

int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n, uint8_t* RESTRICT scratch) {
    return zxc_huf_decode_section_default(payload, payload_size, dst, n, scratch);
}

int zxc_huf_encode_section_dict(const uint8_t* RESTRICT literals, const size_t n_literals,
                                const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                                const zxc_pivco_tree_t* RESTRICT tree,
                                const uint32_t* RESTRICT codes, uint8_t* RESTRICT dst,
                                const size_t dst_cap) {
    return zxc_huf_encode_section_dict_default(literals, n_literals, freq, code_len, tree, codes,
                                               dst, dst_cap);
}

int zxc_huf_decode_section_dict(const uint8_t* RESTRICT payload, const size_t payload_size,
                                uint8_t* RESTRICT dst, const size_t n,
                                const zxc_pivco_tree_t* RESTRICT tree, uint8_t* RESTRICT scratch) {
    return zxc_huf_decode_section_dict_default(payload, payload_size, dst, n, tree, scratch);
}

size_t zxc_huf_calc_size_dict(const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                              const zxc_pivco_tree_t* RESTRICT tree) {
    return zxc_huf_calc_size_dict_default(freq, code_len, tree);
}

/**
 * @brief Pack per-symbol code lengths into the 128-byte nibble header.
 *
 * Un-suffixed entry forwarding to `zxc_huf_pack_lengths_default`; full
 * contract in @c zxc_internal.h.
 */
void zxc_huf_pack_lengths(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out) {
    zxc_huf_pack_lengths_default(code_len, out);
}

/**
 * @brief Unpack and validate a 128-byte packed lengths header.
 *
 * Un-suffixed entry forwarding to `zxc_huf_unpack_lengths_default`; full
 * contract in @c zxc_internal.h.
 */
int zxc_huf_unpack_lengths(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len) {
    return zxc_huf_unpack_lengths_default(in, code_len);
}

// ============================================================================
// PUBLIC UTILITY API
// ============================================================================
// These wrapper functions provide a simplified interface by managing context
// allocation and looping over blocks. They call the dispatched wrappers above.

/**
 * @brief Writes the whole archive an empty source produces: file header, EOF
 *        block, the empty seek table when @p seekable, then the footer.
 *
 * Both entry points return through here: same options, same empty archive, by
 * construction. Neither carves a workspace: there is no block to encode.
 *
 * @return Bytes written, or a negative @ref zxc_error_t.
 */
static int64_t zxc_write_empty_frame(uint8_t* RESTRICT dst, const size_t dst_capacity,
                                     const size_t block_size, const int checksum_enabled,
                                     const uint32_t did, const int seekable) {
    const int h =
        zxc_write_file_header(dst, dst_capacity, block_size, checksum_enabled, did, seekable);
    if (UNLIKELY(h < 0)) return h;
    size_t off = (size_t)h;

    const zxc_block_header_t eof = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int e = zxc_write_block_header(dst + off, dst_capacity - off, &eof);
    if (UNLIKELY(e < 0)) return e;
    off += (size_t)e;

    // The flag promised a table: an empty one, a bare SEK header.
    if (seekable) {
        const int t = zxc_seek_table_header(dst + off, dst_capacity - off, 0);
        if (UNLIKELY(t < 0)) return t;
        off += (size_t)t;
    }

    const int f = zxc_write_file_footer(dst + off, dst_capacity - off, off, 0, 0, checksum_enabled);
    if (UNLIKELY(f < 0)) return f;
    return (int64_t)(off + (size_t)f);
}

/**
 * @brief Writes the seek table of the @p num_blocks blocks in
 *        [@p blocks, @p blocks_end), sizes read back from their headers.
 *
 * Rereading spares the context API a per-call allocation. The walk must land
 * exactly on the EOF block, or @c dst changed under us.
 *
 * @return Bytes written, or a negative @ref zxc_error_t.
 */
static int64_t zxc_write_seek_table_from_blocks(uint8_t* RESTRICT dst, const size_t dst_capacity,
                                                const uint8_t* RESTRICT blocks,
                                                const uint8_t* const blocks_end,
                                                const uint64_t num_blocks,
                                                const int checksum_enabled) {
    const size_t total = zxc_seek_table_size(num_blocks);
    if (UNLIKELY(total == 0)) return ZXC_ERROR_OVERFLOW;  // LCOV_EXCL_LINE
    if (UNLIKELY(dst_capacity < total)) return ZXC_ERROR_DST_TOO_SMALL;

    const int h = zxc_seek_table_header(dst, dst_capacity, num_blocks);
    if (UNLIKELY(h < 0)) return h;  // LCOV_EXCL_LINE
    uint8_t* p = dst + h;

    const size_t trailer = ZXC_BLOCK_HEADER_SIZE + (checksum_enabled ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    uint64_t anchor = ZXC_FILE_HEADER_SIZE;
    for (uint64_t i = 0; i < num_blocks; i++) {
        if (i % ZXC_SEEK_GROUP == 0) {
            zxc_store_le64(p, anchor);
            p += ZXC_SEEK_ANCHOR_SIZE;
        }
        const size_t left = (size_t)(blocks_end - blocks);
        if (UNLIKELY(left < ZXC_BLOCK_HEADER_SIZE))
            return ZXC_ERROR_CORRUPT_DATA;                           // LCOV_EXCL_LINE
        const size_t size = (size_t)zxc_le32(blocks + 3) + trailer;  // comp_size at offset 3
        if (UNLIKELY(size > left)) return ZXC_ERROR_CORRUPT_DATA;    // LCOV_EXCL_LINE
        zxc_store_le32(p, (uint32_t)size);
        p += ZXC_SEEK_SIZE_ENTRY;
        anchor += size;
        blocks += size;
    }
    if (UNLIKELY(blocks != blocks_end)) return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE
    return (int64_t)(p - dst);
}

/**
 * @brief Compresses an entire buffer in one call.
 *
 * Manages context allocation internally, loops over blocks and writes the
 * file header / EOF block / footer.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_compress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                     const size_t dst_capacity, const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!dst || dst_capacity == 0 || (src_size > 0 && !src))) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int seekable = opts ? opts->seekable : 0;
    const int level = ZXC_OPTS_LEVEL(opts, ZXC_LEVEL_DEFAULT);
    const size_t block_size = ZXC_OPTS_BLOCK_SIZE(opts, ZXC_BLOCK_SIZE_DEFAULT);
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    const uint8_t* dict_huf = ZXC_OPTS_DICT_HUF(opts);

    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    const uint32_t did = (dict && dict_size > 0) ? zxc_dict_id(dict, dict_size, dict_huf) : 0;

    if (UNLIKELY(src_size == 0))
        return zxc_write_empty_frame((uint8_t*)dst, dst_capacity, block_size, checksum_enabled, did,
                                     seekable);

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    zxc_cctx_t ctx;

    const size_t eff_chunk =
        dict_size > 0 ? zxc_block_size_ceil(dict_size + block_size) : block_size;
    // LCOV_EXCL_START
    if (UNLIKELY(zxc_cctx_init(&ctx, eff_chunk, 1, level, checksum_enabled, dict_size) != ZXC_OK))
        return ZXC_ERROR_MEMORY;
    // LCOV_EXCL_STOP
    if (UNLIKELY(zxc_cctx_attach_dict_huf(&ctx, dict_huf) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&ctx);
        return ZXC_ERROR_CORRUPT_DATA;
        // LCOV_EXCL_STOP
    }

    // Dict input buffer: [dict_content | block_data] for the encoder, carved
    // into the cctx workspace (NULL when no dictionary is active).
    uint8_t* const dict_input = ctx.dict_buffer;
    if (dict_input) ZXC_MEMCPY(dict_input, dict, dict_size);

    const int h_val = zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled,
                                            did, seekable);
    // LCOV_EXCL_START
    if (UNLIKELY(h_val < 0)) {
        zxc_cctx_free(&ctx);
        return h_val;
    }
    // LCOV_EXCL_STOP
    op += h_val;

    // Seekable: one compressed size per block, at most block_count + 1 of them.
    uint32_t* seek_comp = NULL;
    uint64_t seek_count = 0;
    if (seekable) {
        const size_t block_count = src_size / block_size;
        if (UNLIKELY(block_count + 2 > SIZE_MAX / sizeof(uint32_t))) {
            // LCOV_EXCL_START
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_BLOCK_SIZE;
            // LCOV_EXCL_STOP
        }
        seek_comp = (uint32_t*)ZXC_MALLOC((block_count + 2) * sizeof(uint32_t));
        // LCOV_EXCL_START
        if (UNLIKELY(!seek_comp)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_MEMORY;
        }
        // LCOV_EXCL_STOP
    }

    uint64_t digest = 0;
    size_t pos = 0;
    for (uint64_t bi = 0; pos < src_size; bi++) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        int res;
        if (dict_input) {
            ZXC_MEMCPY(dict_input + dict_size, ip + pos, chunk_len);
            res = zxc_compress_chunk_wrapper(&ctx, dict_input, dict_size + chunk_len, op, rem_cap,
                                             bi);
        } else {
            res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap, bi);
        }
        if (UNLIKELY(res < 0)) {
            ZXC_FREE(seek_comp);
            zxc_cctx_free(&ctx);
            return res;
        }

        // Seekable: record compressed block size
        if (seekable) seek_comp[seek_count++] = (uint32_t)res;
        if (checksum_enabled && LIKELY(res >= ZXC_BLOCK_CHECKSUM_SIZE))
            digest = zxc_digest_combine(digest, zxc_le32(op + res - ZXC_BLOCK_CHECKSUM_SIZE));

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
        ZXC_FREE(seek_comp);
        return eof_val;
    }
    // LCOV_EXCL_STOP
    op += eof_val;

    // Seekable: write seek table between EOF block and footer
    if (seekable) {
        const size_t st_cap = (size_t)(op_end - op);
        const int64_t st_val = zxc_write_seek_table(op, st_cap, seek_comp, seek_count);
        ZXC_FREE(seek_comp);
        if (UNLIKELY(st_val < 0)) return st_val;  // LCOV_EXCL_LINE
        op += st_val;
    }

    const int footer_val = zxc_write_file_footer(
        op, (size_t)(op_end - op), (uint64_t)(op - op_start), src_size, digest, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;  // LCOV_EXCL_LINE
    op += footer_val;

    return (int64_t)(op - op_start);
}

// One-shot decode for zxc_decompress and the no-destination probe.
static int64_t zxc_decompress_oneshot(const uint8_t* src, size_t src_size, uint8_t* dst,
                                      size_t dst_capacity, const zxc_decompress_opts_t* opts);

// The frame walk behind every buffer decode, in place when @p inplace is set.
static int64_t zxc_dctx_decode_frame(zxc_dctx* dctx, const uint8_t* src, size_t src_size,
                                     uint8_t* dst, size_t dst_capacity,
                                     const zxc_decompress_opts_t* opts, int inplace,
                                     size_t* frame_end);

// The container walk: frames back to back.
static int64_t zxc_dctx_decode_container(zxc_dctx* dctx, const uint8_t* src, size_t src_size,
                                         uint8_t* dst, size_t dst_capacity,
                                         const zxc_decompress_opts_t* opts);

/**
 * @brief Validates a frame without decoding it: the check of
 *        zxc_read_frame_info() on a whole buffer, header first.
 *
 * The footer's compressed size must be the whole input: one frame, starting at
 * offset 0. Each caller adds its own policy on the result.
 *
 * @param[in]  src        The frame.
 * @param[in]  src_size   Its size.
 * @param[out] info       Filled on success.
 * @param[out] footer_len Footer bytes, digest included; may be NULL.
 */
static int zxc_buffer_frame_info(const uint8_t* src, const size_t src_size, zxc_frame_info_t* info,
                                 size_t* footer_len) {
    if (UNLIKELY(!src)) return ZXC_ERROR_NULL_INPUT;
    const zxc_scan_src_t scan = zxc_scan_src_mem(src, src_size);
    return zxc_scan_whole_frame(&scan, info, footer_len);
}

/**
 * @brief Answers a decode request that carries no destination.
 *
 * Walks the surviving frame for real on a stand-in buffer, so the verdict is
 * the one a caller with a destination would have got, by construction rather
 * than by keeping a second copy of the checks in step.
 */
static int64_t zxc_probe_without_dst(const uint8_t* RESTRICT src, const size_t src_size,
                                     const zxc_decompress_opts_t* opts,
                                     const zxc_container_info_t* scanned) {
    // Only an empty archive succeeds without a buffer. @p scanned: the caller's walk, if any.
    zxc_container_info_t info;
    if (!scanned) {
        const zxc_scan_src_t scan = zxc_scan_src_mem(src, src_size);
        const int rc = zxc_scan_container(&scan, 0, &info);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
        scanned = &info;
    }
    if (scanned->dsize != 0) return ZXC_ERROR_DST_TOO_SMALL;
    uint8_t probe_dst[1];
    return zxc_decompress_oneshot(src, src_size, probe_dst, 0, opts);
}

/**
 * @brief Decompresses an entire buffer in one call.
 *
 * Validates the file header, loops over compressed blocks and checks the
 * footer size.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_decompress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                       const size_t dst_capacity, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!src || (!dst && dst_capacity != 0))) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(ZXC_OPTS_DICT_SIZE(opts) > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(src_size < ZXC_FRAME_MIN_SIZE))
        return zxc_short_input_verdict((const uint8_t*)src, src_size);

    if (UNLIKELY(!dst || dst_capacity == 0))
        return zxc_probe_without_dst(src, src_size, opts, NULL);

    return zxc_decompress_oneshot((const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, opts);
}

/**
 * @brief Bytes an in-place decode needs on top of the decompressed size.
 *
 * The walk never writes into unread input; the margin keeps a full block of room
 * for an archive its header and footer describe, so every block takes the fast
 * path. Flush-right placement puts block 0 at `capacity - comp_size`, so the read
 * cursor before block k sits at `capacity - sum_{j>=k} (c_j + H) - trailing`, and
 * `sum_{j<k} o_j + chunk_size + PAD <= R_k` requires
 *
 *     capacity >= max_k [ sum_{j<k} o_j + chunk_size + sum_{j>=k} (c_j + H) ] + PAD + trailing
 *
 * Incompressible input (all RAW, `c_j = o_j = chunk_size`) maximises the
 * bracket at `dsize + chunk_size + nblocks * H`: the margin carries the whole
 * accumulated per-block overhead, not just one block's.
 *
 * `trailing` is everything written after the last data block: EOF header, the
 * seek table when HAS_SEEK_TABLE announces one, and the footer. Like has_cs, the
 * flag is taken on trust here and checked by the walk: a tail it does not announce
 * is trailing bytes the bound did not count, as inserted padding is, and fails the
 * tail check. Leaving out an announced table used to push the bound *below*
 * comp_size for a seekable archive of incompressible data in small blocks.
 *
 * @param[in] dsize      Decompressed size, in bytes.
 * @param[in] chunk_size Block size from the file header (0 = no blocks).
 * @param[in] has_cs     Non-zero if blocks carry checksums.
 * @param[in] has_seek   Non-zero if the header announces a seek table.
 * @param[in] footer_len The archive's footer bytes, digest included.
 * @return Bytes to reserve beyond @p dsize.
 */
static uint64_t zxc_inplace_margin(const uint64_t dsize, const size_t chunk_size, const int has_cs,
                                   const int has_seek, const size_t footer_len) {
    const uint64_t nblocks = zxc_seek_block_count(dsize, chunk_size);
    const uint64_t per_block =
        (uint64_t)ZXC_BLOCK_HEADER_SIZE + (has_cs ? (uint64_t)ZXC_BLOCK_CHECKSUM_SIZE : 0);
    const uint64_t seek_table =
        has_seek ? (uint64_t)ZXC_BLOCK_HEADER_SIZE + zxc_seek_table_bytes(nblocks) : 0;
    const uint64_t trailing = (uint64_t)ZXC_BLOCK_HEADER_SIZE +  // EOF block header
                              seek_table + (uint64_t)footer_len;
    return (uint64_t)chunk_size + nblocks * per_block + trailing +
           (uint64_t)ZXC_DECOMPRESS_TAIL_PAD;
}

/**
 * @brief Shared archive probe for the in-place entry points: validates the
 *        magic + file header, then reads the footer's decompressed size and
 *        derives the in-place margin.
 *
 * Keeping this parse in one place guarantees @ref zxc_decompress_inplace_bound
 * and @ref zxc_decompress_inplace always agree on what a buffer of at least
 * the bound must satisfy.
 *
 * The envelope is untrusted, so it goes through zxc_buffer_frame_info() like
 * @ref zxc_get_decompressed_size does; only the error mapping differs.
 *
 * @param[in]  comp      Compressed archive; only the header and footer are read.
 * @param[in]  comp_size Size of the archive in bytes. The caller guarantees it
 *                       covers at least the file header and footer.
 * @param[out] dsize     Decompressed size read from the footer.
 * @param[out] margin    In-place margin for @p dsize, from @ref zxc_inplace_margin.
 * @param[out] floor     Minimum @c off: what has to separate the flush-right
 *                       archive from the head of the buffer.
 * @return ZXC_OK, or a negative @ref zxc_error_t on an invalid archive.
 */
static int zxc_inplace_probe(const uint8_t* comp, const size_t comp_size, uint64_t* dsize,
                             uint64_t* margin, uint64_t* floor) {
    zxc_frame_info_t info;
    size_t footer_len = 0;

    // Own mapping: only a wrong magic word and a forged footer keep their code,
    // everything else reads as a bad header.
    const int rc = zxc_buffer_frame_info(comp, comp_size, &info, &footer_len);
    if (UNLIKELY(rc != ZXC_OK))
        return (rc == ZXC_ERROR_BAD_MAGIC || rc == ZXC_ERROR_CORRUPT_DATA) ? rc
                                                                           : ZXC_ERROR_BAD_HEADER;

    *dsize = info.decompressed_size;
    *margin = zxc_inplace_margin(info.decompressed_size, info.block_size, info.has_checksum,
                                 info.has_seek_table, footer_len);
    *floor = (uint64_t)info.block_size + (uint64_t)ZXC_DECOMPRESS_TAIL_PAD;
    return ZXC_OK;
}

/**
 * @brief Minimum single-buffer size for a safe in-place decode of @p src.
 *
 * Reads the archive header (block size) and footer (decompressed size) without
 * decoding, and returns `decompressed_size + zxc_inplace_margin(...)` (one
 * block + accumulated per-block overhead + footer + wild-copy tail). A buffer
 * of at least this size lets @ref zxc_decompress_inplace decode with the
 * compressed data placed flush-right, the write cursor never overtaking the
 * read cursor.
 *
 * Because src_size is attacker-controlled, inserted padding bytes can slide decoding dangerously
 * close to the output boundary. The src_size + floor term keeps block 0 clear; the walk checks
 * every later block. For authentic archives, this adjustment grows the bound by at most 16 bytes
 * and guarantees it never shrinks.
 */
// cppcheck-suppress unusedFunction
size_t zxc_decompress_inplace_bound(const void* src, const size_t src_size) {
    if (UNLIKELY(!src || src_size < ZXC_FRAME_MIN_SIZE)) return 0;
    uint64_t dsize = 0;
    uint64_t margin = 0;
    uint64_t floor = 0;
    if (UNLIKELY(zxc_inplace_probe((const uint8_t*)src, src_size, &dsize, &margin, &floor) !=
                 ZXC_OK))
        return 0;
    if (UNLIKELY(margin > (uint64_t)SIZE_MAX || dsize > (uint64_t)SIZE_MAX - margin)) return 0;
    const uint64_t by_payload = dsize + margin;

    if (UNLIKELY(floor > (uint64_t)SIZE_MAX - (uint64_t)src_size)) return 0;
    const uint64_t by_placement = (uint64_t)src_size + floor;

    return (size_t)(by_payload > by_placement ? by_payload : by_placement);
}

/**
 * @brief Argument and buffer checks shared by the in-place entry points.
 *
 * Runs before any allocation, so both entry points refuse the same calls with
 * the same codes. On success @p comp is the start of the flush-right archive.
 *
 * @param[in]  buffer          Caller buffer holding the flush-right archive.
 * @param[in]  buffer_capacity Total size of @p buffer.
 * @param[in]  comp_size       Size of the archive.
 * @param[in]  opts            Decompression options, or NULL.
 * @param[out] comp            Start of the archive inside @p buffer.
 * @return ZXC_OK, or a negative @ref zxc_error_t.
 */
static int zxc_inplace_prepare(const void* buffer, const size_t buffer_capacity,
                               const size_t comp_size, const zxc_decompress_opts_t* opts,
                               const uint8_t** comp) {
    if (UNLIKELY(!buffer || comp_size < ZXC_FRAME_MIN_SIZE || comp_size > buffer_capacity))
        return ZXC_ERROR_NULL_INPUT;
    const uint8_t* const c = (const uint8_t*)buffer + (buffer_capacity - comp_size);
    uint64_t dsize = 0;
    uint64_t margin = 0;
    uint64_t floor = 0;
    const int rc = zxc_inplace_probe(c, comp_size, &dsize, &margin, &floor);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(dsize > (uint64_t)buffer_capacity || (uint64_t)buffer_capacity - dsize < margin))
        return ZXC_ERROR_DST_TOO_SMALL;
    /* The check above sizes the buffer against the payload, this one against the
     * archive where it lies: block 0 starts a full block clear. */
    if (UNLIKELY((uint64_t)(buffer_capacity - comp_size) < floor)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(ZXC_OPTS_DICT_SIZE(opts) > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    *comp = c;
    return ZXC_OK;
}

/**
 * @brief Decompresses in place, inside a single caller-owned buffer.
 *
 * Public API; full contract in @c zxc_buffer.h. Same walk as
 * @ref zxc_decompress_inplace_dctx, on a context created for the call once the
 * buffer checks have passed.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_decompress_inplace(void* buffer, const size_t buffer_capacity, const size_t comp_size,
                               const zxc_decompress_opts_t* opts) {
    const uint8_t* comp = NULL;
    const int rc = zxc_inplace_prepare(buffer, buffer_capacity, comp_size, opts, &comp);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    zxc_dctx* const dctx = zxc_create_dctx();
    if (UNLIKELY(!dctx)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
    const int64_t res = zxc_dctx_decode_frame(dctx, comp, comp_size, (uint8_t*)buffer,
                                              buffer_capacity, opts, 1, NULL);
    zxc_free_dctx(dctx);
    return res;
}

/**
 * @brief Reads the decompressed size from a ZXC-compressed buffer.
 *
 * Sums the footers through @ref zxc_scan_container(), which checks each frame:
 * a container that does not hold up returns 0.
 */
uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size) {
    if (UNLIKELY(!src)) return 0;
    const zxc_scan_src_t scan = zxc_scan_src_mem((const uint8_t*)src, src_size);
    zxc_container_info_t info;
    if (UNLIKELY(zxc_scan_container(&scan, 0, &info) != ZXC_OK)) return 0;
    return info.dsize;
}

/**
 * @brief Reads the dictionary id from a compressed archive's file header.
 *
 * Public API; see @c zxc_buffer.h. Validates the magic, then returns the
 * header's @c dict_id field when the dictionary flag is set. Does not decompress.
 */
// cppcheck-suppress unusedFunction
uint32_t zxc_get_dict_id(const void* src, const size_t src_size) {
    if (UNLIKELY(!src || src_size < ZXC_FILE_HEADER_SIZE)) return 0;

    const uint8_t* const p = (const uint8_t*)src;
    if (UNLIKELY(zxc_le32(p) != ZXC_MAGIC_WORD)) return 0;

    return (p[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? zxc_le32(p + 7) : 0;
}

/**
 * @brief Reads a frame's header and footer, without decoding.
 *
 * Public API; see @c zxc_buffer.h.
 */
// cppcheck-suppress unusedFunction
int zxc_get_frame_info(const void* src, const size_t src_size, zxc_frame_info_t* info,
                       const size_t info_size) {
    if (UNLIKELY(!info)) return ZXC_ERROR_NULL_INPUT;
    zxc_frame_info_t got;
    const int rc = zxc_buffer_frame_info((const uint8_t*)src, src_size, &got, NULL);
    if (rc == ZXC_OK) zxc_frame_info_copy(info, info_size, &got);
    return rc;
}

/**
 * @brief Reads the frame that ends @p src, without decoding.
 *
 * Public API; see @c zxc_buffer.h.
 */
// cppcheck-suppress unusedFunction
int zxc_get_last_frame_info(const void* src, const size_t src_size, zxc_frame_info_t* info,
                            const size_t info_size) {
    if (UNLIKELY(!src || !info)) return ZXC_ERROR_NULL_INPUT;
    const zxc_scan_src_t scan = zxc_scan_src_mem((const uint8_t*)src, src_size);
    zxc_frame_info_t got;
    uint64_t start = 0;
    const int rc = zxc_scan_frame(&scan, src_size, &start, &got, NULL);
    if (rc == ZXC_OK) zxc_frame_info_copy(info, info_size, &got);
    return rc;
}

// ============================================================================
// REUSABLE CONTEXT API (Opaque)
// ============================================================================
//
// Provides heap-allocated, opaque contexts that integrators can reuse across
// multiple compress / decompress calls, eliminating per-call malloc/free
// overhead.

// --- Compression ---------------------------------------------------------

/**
 * @brief Opaque reusable compression context (public handle @ref zxc_cctx).
 *
 * Wraps one internal @ref zxc_cctx_t plus the sticky options and bookkeeping
 * needed to reuse buffers across calls and re-init only when the block size
 * changes.
 */
struct zxc_cctx_s {
    zxc_cctx_t inner;       /* existing internal context */
    int initialized;        /* 1 if inner has live allocations */
    int owns_workspace;     /* 0 = library-allocated (free in zxc_free_cctx),
                               1 = caller-supplied static workspace (no-op free,
                               block_size pinned at init) */
    size_t last_block_size; /* block size used for last init */
    // Sticky options (remembered from create or last compress call).
    int stored_level;
    int stored_checksum;
    int stored_seekable;
    size_t stored_block_size;
    int huf_cached;                        /* inner carries the table below */
    uint8_t huf_cache[ZXC_HUF_TABLE_SIZE]; /* last table attached */
};

/**
 * @brief Attaches the shared literal table to @p inner; NULL detaches.
 *
 * The tree is rebuilt only when the 128 bytes differ from @p cache, as a build
 * costs about as much as decoding a small block. Clear @p cached whenever
 * @p inner is re-initialised: the attached state dies with it.
 */
static int zxc_ctx_sync_dict_huf(zxc_cctx_t* RESTRICT inner, uint8_t* RESTRICT cache,
                                 int* RESTRICT cached, const uint8_t* RESTRICT dict_huf) {
    if (dict_huf) {
        if (!*cached || memcmp(cache, dict_huf, ZXC_HUF_TABLE_SIZE) != 0) {
            *cached = 0; /* a failed attach leaves inner without a tree */
            if (UNLIKELY(zxc_cctx_attach_dict_huf(inner, dict_huf) != ZXC_OK))
                return ZXC_ERROR_CORRUPT_DATA;
            ZXC_MEMCPY(cache, dict_huf, ZXC_HUF_TABLE_SIZE);
            *cached = 1;
        }
    } else if (*cached) {
        (void)zxc_cctx_attach_dict_huf(inner, NULL);
        *cached = 0;
    }
    return ZXC_OK;
}

zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts) {
    zxc_cctx* const cctx = (zxc_cctx*)ZXC_CALLOC(1, sizeof(zxc_cctx));
    if (UNLIKELY(!cctx)) return NULL;  // LCOV_EXCL_LINE

    // Resolve and store sticky defaults.
    cctx->stored_level = ZXC_OPTS_LEVEL(opts, ZXC_LEVEL_DEFAULT);
    cctx->stored_block_size = ZXC_OPTS_BLOCK_SIZE(opts, ZXC_BLOCK_SIZE_DEFAULT);
    cctx->stored_checksum = opts ? opts->checksum_enabled : 0;
    cctx->stored_seekable = opts ? opts->seekable : 0;

    if (opts) {
        // LCOV_EXCL_START
        if (UNLIKELY(!zxc_validate_block_size(cctx->stored_block_size) ||
                     zxc_cctx_init(&cctx->inner, cctx->stored_block_size, 1, cctx->stored_level,
                                   cctx->stored_checksum, 0) != ZXC_OK)) {
            ZXC_FREE(cctx);
            return NULL;
        }
        // LCOV_EXCL_STOP
        cctx->last_block_size = cctx->stored_block_size;
        cctx->initialized = 1;
    }

    return cctx;
}

/**
 * @brief Releases a reusable compression context.
 *
 * Public API; see @c zxc_buffer.h. Frees the inner buffers and the handle.
 * NULL-safe. For a static (caller-workspace) context this is a no-op, since the
 * caller owns the workspace.
 */
void zxc_free_cctx(zxc_cctx* cctx) {
    if (UNLIKELY(!cctx)) return;
    // Static cctx: handle + inner buffers live inside the caller's workspace,
    // which we do not own. Free is a no-op; the caller owns the workspace.
    if (cctx->owns_workspace) return;
    if (cctx->initialized) zxc_cctx_free(&cctx->inner);
    ZXC_FREE(cctx);
}

/**
 * @brief Compresses data using a reusable context.
 *
 * Public API; full contract in @c zxc_buffer.h. Same frame walk and dictionary
 * binding as zxc_compress(), empty source included; buffers re-carved only when
 * [dict | block] or the parser tier changes, shared table rebuilt only when it
 * changes.
 */
int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                          void* RESTRICT dst, const size_t dst_capacity,
                          const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!cctx)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(!dst || dst_capacity == 0 || (src_size > 0 && !src))) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int seekable = opts ? opts->seekable : cctx->stored_seekable;
    const int level = ZXC_OPTS_LEVEL(opts, cctx->stored_level);
    const size_t block_size = ZXC_OPTS_BLOCK_SIZE(opts, cctx->stored_block_size);
    // Dictionary options are never sticky: a remembered pointer would dangle.
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    const uint8_t* dict_huf = ZXC_OPTS_DICT_HUF(opts);

    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    // Static cctx: no room for a dictionary prefix, and the workspace cannot
    // grow, so reject a block_size change or a level raise into the
    // optimal-parser tier it was carved without.
    if (UNLIKELY(cctx->owns_workspace && dict_size > 0)) return ZXC_ERROR_DICT_UNSUPPORTED;
    if (UNLIKELY(cctx->owns_workspace && block_size != cctx->last_block_size))
        return ZXC_ERROR_BAD_BLOCK_SIZE;
    if (UNLIKELY(cctx->owns_workspace && level >= ZXC_LEVEL_DENSITY && !cctx->inner.opt_scratch))
        return ZXC_ERROR_BAD_LEVEL;

    cctx->stored_level = level;
    cctx->stored_block_size = block_size;
    cctx->stored_checksum = checksum_enabled;
    cctx->stored_seekable = seekable;

    // The encoder sees [dict | block], so the carved chunk covers both.
    const size_t eff_chunk =
        dict_size > 0 ? zxc_block_size_ceil(dict_size + block_size) : block_size;
    const uint32_t did = (dict && dict_size > 0) ? zxc_dict_id(dict, dict_size, dict_huf) : 0;

    // The sticky settings above are already stored, so an empty source is done:
    // same writer as the one-shot, and the workspace is left alone.
    if (UNLIKELY(src_size == 0))
        return zxc_write_empty_frame((uint8_t*)dst, dst_capacity, block_size, checksum_enabled, did,
                                     seekable);

    // Re-init when the chunk changed, a level raise needs the optimal-parser
    // scratch, or a dictionary arrives on a context carved without its prefix.
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != eff_chunk ||
                 (level >= ZXC_LEVEL_DENSITY && !cctx->inner.opt_scratch) ||
                 (dict_size > 0 && !cctx->inner.dict_buffer))) {
        if (cctx->initialized) {
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, eff_chunk, 1, level, checksum_enabled,
                                   dict_size) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = eff_chunk;
        cctx->initialized = 1;
        cctx->huf_cached = 0;
    } else {
        // Same chunk: update level + checksum without realloc.
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    zxc_cctx_t* const ctx = &cctx->inner;
    ctx->dict_size = dict_size;
    if (UNLIKELY(zxc_ctx_sync_dict_huf(ctx, cctx->huf_cache, &cctx->huf_cached, dict_huf) !=
                 ZXC_OK))
        return ZXC_ERROR_CORRUPT_DATA;

    // [dict | block] input for the encoder, NULL without a dictionary.
    uint8_t* const dict_input = dict_size > 0 ? ctx->dict_buffer : NULL;
    if (dict_input) ZXC_MEMCPY(dict_input, dict, dict_size);

    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    const uint8_t* const ip = (const uint8_t*)src;

    const int h_val = zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled,
                                            did, seekable);
    if (UNLIKELY(h_val < 0)) return h_val;  // LCOV_EXCL_LINE
    op += h_val;

    uint64_t digest = 0;
    size_t pos = 0;
    uint64_t bi = 0;
    for (; pos < src_size; bi++) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        int res;
        if (dict_input) {
            ZXC_MEMCPY(dict_input + dict_size, ip + pos, chunk_len);
            res =
                zxc_compress_chunk_wrapper(ctx, dict_input, dict_size + chunk_len, op, rem_cap, bi);
        } else {
            res = zxc_compress_chunk_wrapper(ctx, ip + pos, chunk_len, op, rem_cap, bi);
        }
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled && LIKELY(res >= ZXC_BLOCK_CHECKSUM_SIZE))
            digest = zxc_digest_combine(digest, zxc_le32(op + res - ZXC_BLOCK_CHECKSUM_SIZE));

        op += res;
        pos += chunk_len;
    }
    // EOF block
    uint8_t* const eof_at = op;
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    if (UNLIKELY(eof_val < 0)) return eof_val;  // LCOV_EXCL_LINE
    op += eof_val;

    // Seek table between the EOF block and the footer.
    if (seekable) {
        const int64_t st_val = zxc_write_seek_table_from_blocks(
            op, (size_t)(op_end - op), op_start + h_val, eof_at, bi, checksum_enabled);
        if (UNLIKELY(st_val < 0)) return st_val;
        op += st_val;
    }

    const int footer_val = zxc_write_file_footer(
        op, (size_t)(op_end - op), (uint64_t)(op - op_start), src_size, digest, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;
    op += footer_val;

    return (int64_t)(op - op_start);
}

// --- Decompression -------------------------------------------------------

/**
 * @brief Opaque reusable decompression context (public handle @ref zxc_dctx).
 *
 * Reuses the internal @ref zxc_cctx_t type for decode, tracking the last block
 * and dict sizes so the inner buffers are re-carved only when they change.
 */
struct zxc_dctx_s {
    zxc_cctx_t inner;       /* reuses the same internal context type */
    size_t last_block_size; /* block size from last header parse */
    size_t last_dict_size;  /* dict_size the inner buffer was carved for (drives re-init) */
    int initialized;        /* 1 if inner has live allocations */
    int owns_workspace;     /* 0 = library-allocated (free in zxc_free_dctx),
                               1 = caller-supplied static workspace (no-op free,
                               block_size pinned at init) */
    int huf_cached;         /* inner carries the table below */
    uint8_t huf_cache[ZXC_HUF_TABLE_SIZE]; /* last table attached */
};

zxc_dctx* zxc_create_dctx(void) {
    zxc_dctx* const dctx = (zxc_dctx*)ZXC_CALLOC(1, sizeof(zxc_dctx));
    return dctx;
}

/**
 * @brief Releases a reusable decompression context.
 *
 * Public API; see @c zxc_buffer.h. Frees the inner buffers and the handle.
 * NULL-safe; a no-op for a static (caller-workspace) context.
 */
void zxc_free_dctx(zxc_dctx* dctx) {
    if (UNLIKELY(!dctx)) return;
    // Static dctx: handle + inner buffers live inside the caller's workspace,
    // which we do not own. Free is a no-op; the caller owns the workspace.
    if (dctx->owns_workspace) return;
    if (dctx->initialized) zxc_cctx_free(&dctx->inner);
    ZXC_FREE(dctx);
}

/**
 * @brief Answers a no-destination decode on a reusable context.
 *
 * This context's own rules come first: one that could never decode the archive
 * has to say why, not "give me a buffer". The answer then comes from the shared
 * probe, which walks a private workspace, so a read-only query leaves the
 * buffers this context has carved untouched.
 */
static int64_t zxc_dctx_probe(const zxc_dctx* dctx, const uint8_t* RESTRICT src,
                              const size_t src_size, const zxc_decompress_opts_t* opts) {
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    const zxc_scan_src_t scan = zxc_scan_src_mem(src, src_size);
    zxc_container_info_t info;
    const int rc =
        zxc_scan_container(&scan, dctx->owns_workspace ? dctx->last_block_size : 0, &info);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(dctx->owns_workspace && (info.uses_dict || dict_size != 0)))
        return ZXC_ERROR_DICT_UNSUPPORTED;
    return zxc_probe_without_dst(src, src_size, opts, &info);
}

/**
 * @brief Decompresses a framed archive into @p dst, reusing @p dctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Same frame walk and dictionary
 * binding as zxc_decompress(); the inner buffers are re-carved only when the
 * block or dictionary size changes, the shared table only when it changes.
 */
int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                            void* RESTRICT dst, const size_t dst_capacity,
                            const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || (!dst && dst_capacity != 0))) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(ZXC_OPTS_DICT_SIZE(opts) > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(src_size < ZXC_FRAME_MIN_SIZE))
        return zxc_short_input_verdict((const uint8_t*)src, src_size);
    if (UNLIKELY(!dst || dst_capacity == 0)) return zxc_dctx_probe(dctx, src, src_size, opts);
    return zxc_dctx_decode_container(dctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                     dst_capacity, opts);
}

/**
 * @brief Decompresses in place inside one caller buffer, reusing @p dctx.
 *
 * Public API; full contract in @c zxc_buffer.h. The archive sits flush-right, at
 * `buffer + buffer_capacity - comp_size`, and decodes left-to-right into
 * `buffer[0..]`. Dictionary archives decode through the context's own bounce
 * buffer, which does not alias @p buffer.
 */
int64_t zxc_decompress_inplace_dctx(zxc_dctx* dctx, void* buffer, const size_t buffer_capacity,
                                    const size_t comp_size, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx)) return ZXC_ERROR_NULL_INPUT;
    const uint8_t* comp = NULL;
    const int rc = zxc_inplace_prepare(buffer, buffer_capacity, comp_size, opts, &comp);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    return zxc_dctx_decode_frame(dctx, comp, comp_size, (uint8_t*)buffer, buffer_capacity, opts, 1,
                                 NULL);
}

/**
 * @brief Readies @p dctx for a frame: re-carves the inner buffers when the block
 *        or dictionary size changed (the block API shares this context), then
 *        syncs the checksum flag and the shared literal table.
 *
 * @return ZXC_OK, or a negative @ref zxc_error_t.
 */
static int zxc_dctx_carve(zxc_dctx* dctx, const size_t chunk_size, const size_t dict_size,
                          const int checksum, const uint8_t* dict_huf) {
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != chunk_size ||
                 dctx->last_dict_size != dict_size)) {
        if (dctx->initialized) {
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, chunk_size, 0, 0, checksum, dict_size) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = chunk_size;
        dctx->last_dict_size = dict_size;
        dctx->initialized = 1;
        dctx->huf_cached = 0;
    } else {
        dctx->inner.checksum_enabled = checksum;
    }
    if (UNLIKELY(zxc_ctx_sync_dict_huf(&dctx->inner, dctx->huf_cache, &dctx->huf_cached,
                                       dict_huf) != ZXC_OK))
        return ZXC_ERROR_CORRUPT_DATA;
    return ZXC_OK;
}

/**
 * @brief The frame walk behind every buffer decode: zxc_decompress(), its
 *        no-destination probe, zxc_decompress_dctx() and both in-place entry
 *        points.
 *
 * No RESTRICT between @p src and @p dst: in place (@p inplace set) they overlap,
 * and a block's room ends at the unread input, which keeps the per-block
 * RESTRICT valid. Arguments are already checked by the callers.
 */
static int64_t zxc_dctx_decode_frame(zxc_dctx* dctx, const uint8_t* src, const size_t src_size,
                                     uint8_t* dst, const size_t dst_capacity,
                                     const zxc_decompress_opts_t* opts, const int inplace,
                                     size_t* frame_end) {
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    const uint8_t* dict_huf = ZXC_OPTS_DICT_HUF(opts);

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* const ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    int file_has_checksums = 0;
    int file_has_seek = 0;
    uint32_t header_dict_id = 0;

    const int hrc = zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums,
                                         &header_dict_id, &file_has_seek);
    if (UNLIKELY(hrc != ZXC_OK)) return hrc;

    // Static dctx: block_size is locked at workspace init; reject any
    // archive whose declared block_size would require a re-partition.
    if (UNLIKELY(dctx->owns_workspace && runtime_chunk_size != dctx->last_block_size))
        return ZXC_ERROR_BAD_BLOCK_SIZE;
    // Static dctx: no room for a dictionary prefix.
    if (UNLIKELY(dctx->owns_workspace && (header_dict_id != 0 || dict_size != 0)))
        return ZXC_ERROR_DICT_UNSUPPORTED;

    // Dictionary binding: same contract as zxc_decompress().
    if (header_dict_id != 0) {
        if (UNLIKELY(!dict || dict_size == 0)) return ZXC_ERROR_DICT_REQUIRED;
        if (UNLIKELY(zxc_dict_id(dict, dict_size, dict_huf) != header_dict_id))
            return ZXC_ERROR_DICT_MISMATCH;
    }

    ip += ZXC_FILE_HEADER_SIZE;

    // work_buf is sized to this by the carve below.
    const size_t work_sz = runtime_chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
    zxc_cctx_t* const ctx = &dctx->inner;

    // Carved on the first block that needs it: an archive storing nothing, and a
    // no-destination probe of one, never allocate.
    int ctx_ready = 0;
    // [dict | decode + PAD] scratch, NULL without a dictionary.
    uint8_t* dict_dec = NULL;

    uint64_t block_index = 0;
    uint64_t digest = 0;
    for (;;) {
        // Only the EOF block may end the walk.
        if (UNLIKELY(ip >= ip_end)) return ZXC_ERROR_CORRUPT_DATA;
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != ZXC_OK))
            return ZXC_ERROR_BAD_HEADER;

        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            if (UNLIKELY(zxc_check_eof_header(ip) != ZXC_OK)) return ZXC_ERROR_BAD_HEADER;

            // After the EOF block: the SEK block when announced, then the footer
            // the frame implies. Another frame may follow it.
            const uint64_t total_out = (uint64_t)(op - op_start);
            size_t pos = (size_t)(ip - src) + ZXC_BLOCK_HEADER_SIZE;
            if (file_has_seek) {
                uint64_t sek_bytes = 0;
                const int src_rc = zxc_check_seek_header(src + pos, src_size - pos, total_out,
                                                         runtime_chunk_size, &sek_bytes);
                if (UNLIKELY(src_rc != ZXC_OK)) return src_rc;
                pos += ZXC_BLOCK_HEADER_SIZE;
                if (UNLIKELY(sek_bytes > (uint64_t)(src_size - pos)))
                    return ZXC_ERROR_SRC_TOO_SMALL;
                pos += (size_t)sek_bytes;
            }
            const zxc_footer_layout_t fl = zxc_footer_layout(pos, total_out, file_has_checksums);
            uint64_t stored_digest = 0;
            const int frc =
                zxc_check_file_footer(src + pos, src_size - pos, &fl, total_out, &stored_digest);
            if (UNLIKELY(frc != ZXC_OK)) return frc;
            if (checksum_enabled && file_has_checksums && UNLIKELY(stored_digest != digest))
                return ZXC_ERROR_BAD_CHECKSUM;
            pos += fl.len;
            // In place, the archive is one frame flush against the buffer's end.
            if (frame_end)
                *frame_end = pos;
            else if (UNLIKELY(pos != src_size))
                return ZXC_ERROR_CORRUPT_DATA;
            break;  // EOF reached, stop decoding
        }

        // The decoder only requires the payload, the step also covers the
        // checksum.
        const size_t advance = ZXC_BLOCK_HEADER_SIZE + bh.comp_size +
                               (file_has_checksums ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
        if (UNLIKELY(advance > rem_src)) return ZXC_ERROR_SRC_TOO_SMALL;

        if (UNLIKELY(!ctx_ready)) {
            const int crc = zxc_dctx_carve(dctx, runtime_chunk_size, dict_size,
                                           file_has_checksums && checksum_enabled, dict_huf);
            if (UNLIKELY(crc != ZXC_OK)) return crc;
            dict_dec = ctx->dict_buffer;
            if (dict_dec) ZXC_MEMCPY(dict_dec, dict, dict_size);
            ctx_ready = 1;
        }

        // The block's room: the rest of dst, in place cut at the unread input. The
        // fast decoder never writes past its capacity; a block that does not fit
        // decodes aside and copies only its real output.
        const size_t rem_cap = (size_t)(op_end - op);
        size_t room = rem_cap;
        if (inplace) {
            const size_t ahead = ip > op ? (size_t)(ip - op) : 0;
            if (ahead < room) room = ahead;
        }
        int res;
        const uint8_t* bounce = NULL;
        if (dict_dec) {
            // Decode behind the prefix so back-references into it resolve.
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, dict_dec + dict_size, work_sz,
                                               block_index);
            bounce = dict_dec + dict_size;
        } else if (LIKELY(room >= work_sz)) {
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, op, work_sz, block_index);
        } else {
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, ctx->work_buf, ctx->work_buf_cap,
                                               block_index);
            bounce = ctx->work_buf;
        }
        if (bounce && LIKELY(res > 0)) {
            // A no-destination probe lands here too. In place, output past the
            // room means padding or forged sizes.
            if (UNLIKELY((size_t)res > room))
                return (size_t)res > rem_cap ? ZXC_ERROR_DST_TOO_SMALL : ZXC_ERROR_CORRUPT_DATA;
            ZXC_MEMCPY(op, bounce, (size_t)res);
        }
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled && file_has_checksums)
            digest =
                zxc_digest_combine(digest, zxc_le32(ip + ZXC_BLOCK_HEADER_SIZE + bh.comp_size));

        ip += advance;
        op += res;
        block_index++;
    }

    return (int64_t)(op - op_start);
}

/** @brief The container walk behind zxc_decompress() and zxc_decompress_dctx(). */
static int64_t zxc_dctx_decode_container(zxc_dctx* dctx, const uint8_t* src, const size_t src_size,
                                         uint8_t* dst, const size_t dst_capacity,
                                         const zxc_decompress_opts_t* opts) {
    const zxc_scan_src_t scan = zxc_scan_src_mem(src, src_size);
    uint64_t at = 0;
    size_t out = 0;
    int rc;
    while ((rc = zxc_container_next(&scan, at)) == 1) {
        size_t frame_size = 0;
        const int64_t res = zxc_dctx_decode_frame(dctx, src + at, src_size - (size_t)at, dst + out,
                                                  dst_capacity - out, opts, 0, &frame_size);
        if (UNLIKELY(res < 0)) return res;
        out += (size_t)res;
        at += frame_size;
    }
    if (UNLIKELY(rc < 0)) return rc;
    if (UNLIKELY(at == 0)) return ZXC_ERROR_SRC_TOO_SMALL;
    return (int64_t)out;
}

// One-shot decode: the same walk, on a context that lives for the call.
static int64_t zxc_decompress_oneshot(const uint8_t* src, const size_t src_size, uint8_t* dst,
                                      const size_t dst_capacity,
                                      const zxc_decompress_opts_t* opts) {
    zxc_dctx dctx;
    ZXC_MEMSET(&dctx, 0, sizeof(dctx));
    const int64_t res = zxc_dctx_decode_container(&dctx, src, src_size, dst, dst_capacity, opts);
    if (dctx.initialized) zxc_cctx_free(&dctx.inner);
    return res;
}

// =========================================================================
// Block-Level API (no file framing)
// =========================================================================

/**
 * @brief Compresses a single block (no file framing), reusing @p cctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Produces one format-conformant
 * block with no header / EOF / footer, so @p src_size must not exceed
 * @c ZXC_BLOCK_SIZE_MAX (use the frame or streaming APIs for larger inputs).
 * With a dictionary in @p opts, [dict | block] is assembled in the cctx-owned
 * bounce buffer before encoding. Inner buffers are re-initialised when the
 * effective block size changes, or when a per-call level raise into the
 * optimal-parser tier requires the opt_scratch region; static contexts
 * reject both cases instead (the workspace cannot grow).
 */
int64_t zxc_compress_block(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                           void* RESTRICT dst, const size_t dst_capacity,
                           const zxc_compress_opts_t* opts) {
    return zxc_compress_block_at(cctx, src, src_size, dst, dst_capacity, opts, 0);
}

int64_t zxc_compress_block_at(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                              void* RESTRICT dst, const size_t dst_capacity,
                              const zxc_compress_opts_t* opts, const uint64_t block_index) {
    if (UNLIKELY(!cctx || !src || !dst || src_size == 0 || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    // Block API processes a single format-conformant block: src_size must not
    // exceed ZXC_BLOCK_SIZE_MAX. Callers with larger inputs should use the
    // frame or streaming APIs which chunk transparently.
    if (UNLIKELY(src_size > ZXC_BLOCK_SIZE_MAX)) return ZXC_ERROR_BAD_BLOCK_SIZE;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int level = ZXC_OPTS_LEVEL(opts, cctx->stored_level);
    // For block API, block_size == src_size (the caller compresses one block at a time).
    const size_t block_size = ZXC_OPTS_BLOCK_SIZE(opts, cctx->stored_block_size);
    const size_t min_bs = zxc_block_size_ceil(src_size);

    // Always ensure internal buffers can hold src_size.
    // When a dictionary is active, offset_bits must accommodate dict + block.
    const uint8_t* b_dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t b_dict_size = ZXC_OPTS_DICT_SIZE(opts);
    if (UNLIKELY(b_dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    const size_t base_block_size = (block_size > min_bs) ? block_size : min_bs;
    const size_t effective_block_size =
        b_dict_size > 0 ? zxc_block_size_ceil(b_dict_size + base_block_size) : base_block_size;

    // Static cctx: the workspace cannot grow, so reject anything forcing a
    // re-partition - another block size, or a level raise into the
    // optimal-parser tier it carries no opt_scratch for. Re-initing on the heap
    // would break the no-allocation contract and leak: zxc_free_cctx is a no-op
    // for static contexts.
    if (UNLIKELY(cctx->owns_workspace && b_dict_size > 0)) return ZXC_ERROR_DICT_UNSUPPORTED;
    if (UNLIKELY(cctx->owns_workspace && effective_block_size != cctx->last_block_size))
        return ZXC_ERROR_BAD_BLOCK_SIZE;  // LCOV_EXCL_LINE
    if (UNLIKELY(cctx->owns_workspace && level >= ZXC_LEVEL_DENSITY && !cctx->inner.opt_scratch))
        return ZXC_ERROR_BAD_LEVEL;

    cctx->stored_level = level;
    cctx->stored_block_size = base_block_size;
    cctx->stored_checksum = checksum_enabled;

    // Re-init when block_size changed, a level raise needs the optimal-parser
    // scratch, or a dictionary arrives on a context carved without its prefix
    // (a block_size switch can round [dict | block] back to the same size).
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != effective_block_size ||
                 (level >= ZXC_LEVEL_DENSITY && !cctx->inner.opt_scratch) ||
                 (b_dict_size > 0 && !cctx->inner.dict_buffer))) {
        if (cctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, effective_block_size, 1, level, checksum_enabled,
                                   b_dict_size) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = effective_block_size;
        cctx->initialized = 1;
        cctx->huf_cached = 0;
    } else {
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    cctx->inner.dict_size = b_dict_size;
    if (UNLIKELY(zxc_ctx_sync_dict_huf(&cctx->inner, cctx->huf_cache, &cctx->huf_cached,
                                       ZXC_OPTS_DICT_HUF(opts)) != ZXC_OK))
        return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE

    int res;
    if (b_dict && b_dict_size > 0) {
        // [dict | block] assembled in the cctx-owned dict_buffer
        uint8_t* const combined = cctx->inner.dict_buffer;
        ZXC_MEMCPY(combined, b_dict, b_dict_size);
        ZXC_MEMCPY(combined + b_dict_size, src, src_size);
        res = zxc_compress_chunk_wrapper(&cctx->inner, combined, b_dict_size + src_size,
                                         (uint8_t*)dst, dst_capacity, block_index);
    } else {
        res = zxc_compress_chunk_wrapper(&cctx->inner, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                         dst_capacity, block_index);
    }
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/**
 * @brief Block decode through work_buf, for a @p dst too tight for the
 *        speculative writes, then copied out. @p carved bounds the decoded size
 *        on a static context; 0 leaves it unbounded.
 */
static int64_t zxc_decode_bounce(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_size, uint8_t* RESTRICT dst,
                                 const size_t dst_capacity, const size_t carved) {
    // Block API only: frameless blocks carry index 0.
    const int res =
        zxc_decompress_chunk_wrapper(ctx, src, src_size, ctx->work_buf, ctx->work_buf_cap, 0);
    if (UNLIKELY(res < 0)) return res;
    if (UNLIKELY(carved != 0 && (size_t)res > carved)) return ZXC_ERROR_BAD_BLOCK_SIZE;
    if (UNLIKELY((size_t)res > dst_capacity)) return ZXC_ERROR_DST_TOO_SMALL;
    ZXC_MEMCPY(dst, ctx->work_buf, (size_t)res);
    return res;
}

/**
 * @brief Block decode on a static dctx: the carved block is the effective capacity.
 *
 * Fast decoder: carved block plus wild-copy margin; strict: at most that. A
 * larger block still within the margin is @ref ZXC_ERROR_BAD_BLOCK_SIZE, beyond
 * it the decoder's own too-small-destination code, never reinterpreted.
 */
static int64_t zxc_static_decompress_block(zxc_dctx* RESTRICT dctx, const uint8_t* RESTRICT src,
                                           const size_t src_size, uint8_t* RESTRICT dst,
                                           const size_t dst_capacity, const size_t dict_size,
                                           const int checksum_enabled, const int strict) {
    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(dict_size > 0)) return ZXC_ERROR_DICT_UNSUPPORTED;
    const size_t carved = dctx->last_block_size;
    const size_t work_sz = carved + ZXC_DECOMPRESS_TAIL_PAD;
    zxc_cctx_t* const ctx = &dctx->inner;
    ctx->checksum_enabled = checksum_enabled;
    ctx->dict_size = 0;
    int res;
    if (strict) {
        const size_t cap = dst_capacity < work_sz ? dst_capacity : work_sz;
        res = zxc_decompress_chunk_wrapper_safe_public(ctx, src, src_size, dst, cap, 0);
    } else if (dst_capacity >= work_sz) {
        res = zxc_decompress_chunk_wrapper(ctx, src, src_size, dst, work_sz, 0);
    } else {
        // Bounce through work_buf when dst cannot absorb wild copies.
        return zxc_decode_bounce(ctx, src, src_size, dst, dst_capacity, carved);
    }
    if (UNLIKELY(res > 0 && (size_t)res > carved)) return ZXC_ERROR_BAD_BLOCK_SIZE;
    return res;
}

/**
 * @brief Recarves a heap dctx when @p block_size or @p dict_size changed;
 *        otherwise only refreshes the checksum flag.
 */
static int zxc_dctx_prepare(zxc_dctx* RESTRICT dctx, const size_t block_size,
                            const size_t dict_size, const int checksum_enabled) {
    if (LIKELY(dctx->initialized && dctx->last_block_size == block_size &&
               dctx->last_dict_size == dict_size)) {
        dctx->inner.checksum_enabled = checksum_enabled;
        return ZXC_OK;
    }
    if (dctx->initialized) {
        zxc_cctx_free(&dctx->inner);
        dctx->initialized = 0;
    }
    // LCOV_EXCL_START
    if (UNLIKELY(zxc_cctx_init(&dctx->inner, block_size, 0, 0, checksum_enabled, dict_size) !=
                 ZXC_OK))
        return ZXC_ERROR_MEMORY;
    // LCOV_EXCL_STOP
    dctx->last_block_size = block_size;
    dctx->last_dict_size = dict_size;
    dctx->initialized = 1;
    dctx->huf_cached = 0;
    return ZXC_OK;
}

/**
 * @brief Decompresses a single block (no file framing), reusing @p dctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Decodes one format-conformant
 * block; the decoded payload cannot exceed @c ZXC_BLOCK_SIZE_MAX, so
 * @p dst_capacity is bounded by @c ZXC_BLOCK_SIZE_MAX + @c ZXC_DECOMPRESS_TAIL_PAD.
 * With a dictionary in @p opts the decode runs through the [dict | decode]
 * bounce buffer; otherwise it goes straight into @p dst, and is redone through
 * @c work_buf if a tight tail aborted it, leaving @p dst partly written.
 */
int64_t zxc_decompress_block(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                             void* RESTRICT dst, const size_t dst_capacity,
                             const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    // One format-conformant block, so the payload cannot exceed
    // ZXC_BLOCK_SIZE_MAX; dst_capacity is bounded to that plus the tail-pad the
    // wild copies need. Larger outputs belong to the frame or streaming APIs.
    if (UNLIKELY(dst_capacity > ZXC_BLOCK_SIZE_MAX + ZXC_DECOMPRESS_TAIL_PAD))
        return ZXC_ERROR_BAD_BLOCK_SIZE;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(dctx->owns_workspace))
        return zxc_static_decompress_block(dctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                           dst_capacity, dict_size, checksum_enabled, 0);

    // Carved for dst_capacity: the context holds any block dst can hold.
    const size_t block_size = zxc_block_size_ceil(dst_capacity);
    const int rc = zxc_dctx_prepare(dctx, block_size, dict_size, checksum_enabled);
    if (UNLIKELY(rc != ZXC_OK)) return rc;  // LCOV_EXCL_LINE

    zxc_cctx_t* const ctx = &dctx->inner;
    ctx->dict_size = dict_size;
    if (UNLIKELY(zxc_ctx_sync_dict_huf(ctx, dctx->huf_cache, &dctx->huf_cached,
                                       ZXC_OPTS_DICT_HUF(opts)) != ZXC_OK))
        return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE

    // The [dict | decode] buffer carries block_size + ZXC_DECOMPRESS_TAIL_PAD.
    const size_t work_sz = block_size + ZXC_DECOMPRESS_TAIL_PAD;

    int res;
    if (dict && dict_size > 0) {
        // [dict | decode] assembled in the cctx-owned dict_buffer
        uint8_t* const dec_buf = ctx->dict_buffer;
        ZXC_MEMCPY(dec_buf, dict, dict_size);
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, dec_buf + dict_size,
                                           work_sz, 0);
        if (LIKELY(res > 0)) {
            if (UNLIKELY((size_t)res > dst_capacity)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, dec_buf + dict_size, (size_t)res);
        }
    } else {
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                           dst_capacity, 0);
        // A tight tail aborts a block that fits: work_buf has the margin.
        // Corrupt blocks report OVERFLOW too, and pay the second decode.
        if (UNLIKELY(res == ZXC_ERROR_OVERFLOW))
            return zxc_decode_bounce(ctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                     dst_capacity, 0);
    }
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/**
 * @brief Safe-variant block decompressor: accepts dst_capacity == uncompressed_size.
 *
 * Static dctx: shared static path. Heap dctx: dict inputs and RAW route to
 * @ref zxc_decompress_block, plain GLO/GHI use the strict decoder (no bounce
 * buffer, no +ZXC_DECOMPRESS_TAIL_PAD).
 *
 * Public API; full contract in @c zxc_buffer.h.
 */
int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                                  void* RESTRICT dst, const size_t dst_capacity,
                                  const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    // Strict-tail variant: dst_capacity matches the exact uncompressed size
    if (UNLIKELY(dst_capacity > ZXC_BLOCK_SIZE_MAX)) return ZXC_ERROR_BAD_BLOCK_SIZE;
    if (UNLIKELY(dctx->owns_workspace))
        return zxc_static_decompress_block(dctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                           dst_capacity, ZXC_OPTS_DICT_SIZE(opts),
                                           opts ? opts->checksum_enabled : 0, 1);

    // A dict needs the [dict|payload] bounce; route to the bounce-capable path.
    if (opts && opts->dict && opts->dict_size > 0) {
        return zxc_decompress_block(dctx, src, src_size, dst, dst_capacity, opts);
    }

    const uint8_t type = ((const uint8_t*)src)[0];
    // RAW never wild-writes past dst_capacity: route to the existing fast API.
    if (type == ZXC_BLOCK_RAW) {
        return zxc_decompress_block(dctx, src, src_size, dst, dst_capacity, opts);
    }

    // GLO/GHI: use the strict-tail decoder (no bounce buffer required).
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int rc = zxc_dctx_prepare(dctx, zxc_block_size_ceil(dst_capacity), 0, checksum_enabled);
    if (UNLIKELY(rc != ZXC_OK)) return rc;  // LCOV_EXCL_LINE
    dctx->inner.dict_size = 0;

    const int res = zxc_decompress_chunk_wrapper_safe_public(
        &dctx->inner, (const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, 0);
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

// ============================================================================
// STATIC CONTEXT API (caller-allocated workspace)
// ============================================================================
// Places the public handle struct at the start of the workspace, then carves
// the persistent buffer (via zxc_cctx_init_in_workspace) in the remaining
// cache-line-aligned tail.  The caller owns the whole workspace; free
// functions become no-ops via the owns_workspace flag.

// Size occupied by the opaque handle at the start of the workspace, rounded
// up to a cache-line boundary so the persistent buffer (which expects 64 B
// alignment for the hot zones) starts aligned.
#define ZXC_STATIC_CCTX_HDR_SIZE ZXC_ALIGN_CL(sizeof(struct zxc_cctx_s))
#define ZXC_STATIC_DCTX_HDR_SIZE ZXC_ALIGN_CL(sizeof(struct zxc_dctx_s))

/**
 * @brief Workspace size needed for a static compression context.
 *
 * Public API; see @c zxc_buffer.h. Sum of the cache-line-aligned handle header
 * and the persistent buffer that @ref zxc_init_static_cctx carves for the given
 * @p block_size / @p level. Performs no allocation.
 */
size_t zxc_static_cctx_workspace_size(const size_t block_size, const int level) {
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return 0;
    if (UNLIKELY(level < ZXC_LEVEL_FASTEST || level > ZXC_LEVEL_ULTRA)) return 0;
    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 1, level, 0);
    if (UNLIKELY(inner_sz == 0)) return 0;
    return ZXC_STATIC_CCTX_HDR_SIZE + inner_sz;
}

zxc_cctx* zxc_init_static_cctx(void* RESTRICT workspace, const size_t workspace_size,
                               const zxc_compress_opts_t* RESTRICT opts) {
    if (UNLIKELY(!workspace || !opts)) return NULL;

    const int level = (opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    const size_t block_size = (opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    const int checksum_enabled = opts->checksum_enabled;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return NULL;
    if (UNLIKELY(level < ZXC_LEVEL_FASTEST || level > ZXC_LEVEL_ULTRA)) return NULL;

    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 1, level, 0);
    if (UNLIKELY(inner_sz == 0)) return NULL;
    if (UNLIKELY(workspace_size < ZXC_STATIC_CCTX_HDR_SIZE + inner_sz)) return NULL;

    zxc_cctx* const cctx = (zxc_cctx*)workspace;
    ZXC_MEMSET(cctx, 0, sizeof(*cctx));

    uint8_t* const inner_ws = (uint8_t*)workspace + ZXC_STATIC_CCTX_HDR_SIZE;
    if (UNLIKELY(zxc_cctx_init_in_workspace(&cctx->inner, inner_ws, inner_sz, block_size, 1, level,
                                            checksum_enabled, 0, 0) != ZXC_OK))
        return NULL;

    cctx->owns_workspace = 1;
    cctx->initialized = 1;
    cctx->last_block_size = block_size;
    cctx->stored_level = level;
    cctx->stored_block_size = block_size;
    cctx->stored_checksum = checksum_enabled;
    cctx->stored_seekable = opts->seekable;
    return cctx;
}

/**
 * @brief Workspace size needed for a static decompression context.
 *
 * Public API; see @c zxc_buffer.h. Sum of the cache-line-aligned handle header
 * and the persistent buffer that @ref zxc_init_static_dctx carves for the given
 * @p block_size. Performs no allocation.
 */
size_t zxc_static_dctx_workspace_size(const size_t block_size) {
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return 0;
    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 0, 0, 0);
    if (UNLIKELY(inner_sz == 0)) return 0;
    return ZXC_STATIC_DCTX_HDR_SIZE + inner_sz;
}

zxc_dctx* zxc_init_static_dctx(void* RESTRICT workspace, const size_t workspace_size,
                               const size_t block_size) {
    if (UNLIKELY(!workspace)) return NULL;
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return NULL;

    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 0, 0, 0);
    if (UNLIKELY(inner_sz == 0)) return NULL;
    if (UNLIKELY(workspace_size < ZXC_STATIC_DCTX_HDR_SIZE + inner_sz)) return NULL;

    zxc_dctx* const dctx = (zxc_dctx*)workspace;
    ZXC_MEMSET(dctx, 0, sizeof(*dctx));

    uint8_t* const inner_ws = (uint8_t*)workspace + ZXC_STATIC_DCTX_HDR_SIZE;
    // mode == 0 init: checksum_enabled is updated per-call from the file
    // header flags, so it does not need to be locked at workspace init.
    if (UNLIKELY(zxc_cctx_init_in_workspace(&dctx->inner, inner_ws, inner_sz, block_size, 0, 0, 0,
                                            0, 0) != ZXC_OK))
        return NULL;

    dctx->owns_workspace = 1;
    dctx->initialized = 1;
    dctx->last_block_size = block_size;
    return dctx;
}
