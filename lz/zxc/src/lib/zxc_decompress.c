/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_decompress.c
 * @brief Block-level decompression: GLO / GHI / RAW decoding with
 *        SIMD-accelerated bit-unpacking and overlapping copies.
 *
 * Like @ref zxc_compress.c, this file is compiled multiple times with
 * @c ZXC_FUNCTION_SUFFIX to produce per-ISA variants.
 */

// Function Multi-Versioning Support
// With ZXC_FUNCTION_SUFFIX defined (e.g. _avx2), rename the entry point AND the
// Huffman decoder this TU consumes. The defines precede zxc_internal.h so the
// header's prototypes get the suffix too, keeping callers and callees matched.
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_decompress_chunk_wrapper ZXC_CAT(zxc_decompress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#define zxc_decompress_chunk_wrapper_dict \
    ZXC_CAT(zxc_decompress_chunk_wrapper_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_decompress_chunk_wrapper_safe \
    ZXC_CAT(zxc_decompress_chunk_wrapper_safe, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section ZXC_CAT(zxc_huf_decode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section_dict ZXC_CAT(zxc_huf_decode_section_dict, ZXC_FUNCTION_SUFFIX)
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/**
 * @brief Reads a Prefix Varint encoded integer.
 *
 * Unary prefix bits in the first byte give the total length, at most 3 bytes
 * here since that covers every length this decoder can meet:
 *
 * Format:
 * - 1 byte  (0xxxxxxx):  7-bit payload (val < 2^7  = 128)
 * - 2 bytes (10xxxxxx): 14-bit payload (val < 2^14 = 16384)
 * - 3 bytes (110xxxxx): 21-bit payload (val < 2^21 = 2097152)
 *
 * @param[in,out] ptr Pointer to a pointer to the current position in the stream.
 * @param[in] end Pointer to the end of the readable stream (for bounds checking).
 * @return The decoded 32-bit integer, or 0 if reading would overflow bounds (safe default).
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_read_varint(const uint8_t** ptr, const uint8_t* end) {
    const uint8_t* p = *ptr;
    if (UNLIKELY(p >= end)) return 0;

    const uint32_t b0 = p[0];

    // 1 Byte: 0xxxxxxx (7 bits) -> val < 128 (2^7)
    if (LIKELY(b0 < 0x80)) {
        *ptr = p + 1;
        return b0;
    }

    // 2 Bytes: 10xxxxxx xxxxxxxx (14 bits) -> val < 16384 (2^14)
    if (LIKELY(b0 < 0xC0)) {
        if (UNLIKELY(p + 1 >= end)) {
            *ptr = end;
            return 0;
        }
        *ptr = p + 2;
        return (b0 & 0x3F) | ((uint32_t)p[1] << 6);
    }

    // 3 Bytes: 110xxxxx xxxxxxxx xxxxxxxx (21 bits) -> val < 2^21. The longest
    // a legitimate varint can be: values are (ll - MASK) or (ml - MASK), always
    // strictly below block_size_max = 2^21.
    if (LIKELY(b0 < 0xE0)) {
        if (UNLIKELY(p + 2 >= end)) {
            *ptr = end;
            return 0;
        }
        *ptr = p + 3;
        return (b0 & 0x1F) | ((uint32_t)p[1] << 5) | ((uint32_t)p[2] << 13);
    }

    // extra encoding: out-of-spec for the current format, reject.
    *ptr = end;
    return 0;
}

#if defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32) || defined(ZXC_USE_AVX2) || \
    defined(ZXC_USE_AVX512)
/**
 * @brief Periodic pattern masks, `mask[off][i] = i % off`, for off in [2, 31].
 *
 * Rows 0 and 1 stay zero and unused: offset 1 is a byte splat, not a pattern.
 */
static const ZXC_ALIGN(32) uint8_t zxc_overlap_masks32[32][32] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // off=0 (unused)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // off=1 (RLE handled separately)
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},  // off=2
    {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
     1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1},  // off=3
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
     0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},  // off=4
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0,
     1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1},  // off=5
    {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
     4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1},  // off=6
    {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1,
     2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3},  // off=7
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},  // off=8
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4},  // off=9
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
     6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1},  // off=10
    {0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2, 3, 4,
     5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7, 8, 9},  // off=11
    {0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10, 11, 0, 1, 2, 3,
     4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2,  3,  4, 5, 6, 7},  // off=12
    {0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12, 0, 1, 2,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0,  1,  2,  3, 4, 5},  // off=13
    {0, 1, 2, 3, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13, 0, 1,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0,  1,  2, 3},  // off=14
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 0,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,  1},  // off=15
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  // off=16
    {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14},  // off=17
    {0,  1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13},  // off=18
    {0,  1,  2,  3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12},  // off=19
    {0,  1,  2,  3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11},  // off=20
    {0,  1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 0, 1, 2, 3, 4, 5,  6,  7,  8,  9,  10},  // off=21
    {0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4,  5,  6,  7,  8,  9},  // off=22
    {0,  1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 0, 1, 2, 3,  4,  5,  6,  7,  8},  // off=23
    {0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2,  3,  4,  5,  6,  7},  // off=24
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1,  2,  3,  4,  5,  6},  // off=25
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0,  1,  2,  3,  4,  5},  // off=26
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0,  1,  2,  3,  4},  // off=27
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0,  1,  2,  3},  // off=28
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0,  1,  2},  // off=29
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0,  1},  // off=30
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0}  // off=31
};
#endif

/**
 * @brief Store stride for the 32-byte pattern: `(32 / off) * off`.
 *
 * Past 16 no second period fits, so the stride is the offset itself and each
 * store's tail gets rewritten by the next one.
 */
static const uint8_t zxc_overlap_strides32[32] = {32, 32, 32, 30, 32, 30, 30, 28, 32, 27, 30,
                                                  22, 24, 26, 28, 30, 32, 17, 18, 19, 20, 21,
                                                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

/**
 * @brief Copies an @p ml-byte run of period @p off (2..31) with 32-byte stores.
 *
 * Builds the run's first 32 bytes in a register pair, then stores them
 * repeatedly. Nothing is re-shuffled: advancing by a multiple of @p off keeps
 * the pattern in phase.
 *
 * Replaced a 16-byte form that reloaded the source every iteration, which for
 * offsets of 17 and up straddled two of its own recent stores and stalled on
 * store forwarding - worth up to 7x on periodic data.
 *
 * Reading 32 source bytes touches destination bytes not written yet on the
 * smallest offsets. They are never selected (every mask index is `< off`) and
 * sit inside the @ref ZXC_PAD_SIZE headroom the caller owes.
 *
 * May overshoot up to 31 bytes past @p ml, like the 32-byte copy ladder.
 *
 * @param[out] dst Output cursor; the run source is `dst - off`.
 * @param[in]  off Back-reference distance, in [2, 31].
 * @param[in]  ml  Run length in bytes (>= 1).
 */
// codeql[cpp/unused-static-function] : False positive
static ZXC_ALWAYS_INLINE void zxc_decode_copy_overlap_run32(uint8_t* dst, const uint32_t off,
                                                            const uint64_t ml) {
    const size_t stride = zxc_overlap_strides32[off];
    size_t copied = 0;
#if defined(ZXC_USE_NEON64)
    uint8x16x2_t tbl;
    tbl.val[0] = vld1q_u8(dst - off);
    tbl.val[1] = vld1q_u8(dst - off + 16);
    const uint8x16_t pat_lo = vqtbl2q_u8(tbl, vld1q_u8(zxc_overlap_masks32[off]));
    const uint8x16_t pat_hi = vqtbl2q_u8(tbl, vld1q_u8(zxc_overlap_masks32[off] + 16));
    do {
        vst1q_u8(dst + copied, pat_lo);
        vst1q_u8(dst + copied + 16, pat_hi);
        copied += stride;
    } while (copied < ml);

#elif defined(ZXC_USE_NEON32)
    // VTBL reaches 8 bytes per lookup from a 4-register (32-byte) table, so the
    // pattern takes four lookups instead of NEON64's two.
    uint8x8x4_t tbl;
    tbl.val[0] = vld1_u8(dst - off);
    tbl.val[1] = vld1_u8(dst - off + 8);
    tbl.val[2] = vld1_u8(dst - off + 16);
    tbl.val[3] = vld1_u8(dst - off + 24);
    const uint8_t* const m = zxc_overlap_masks32[off];
    const uint8x8_t p0 = vtbl4_u8(tbl, vld1_u8(m));
    const uint8x8_t p1 = vtbl4_u8(tbl, vld1_u8(m + 8));
    const uint8x8_t p2 = vtbl4_u8(tbl, vld1_u8(m + 16));
    const uint8x8_t p3 = vtbl4_u8(tbl, vld1_u8(m + 24));
    do {
        vst1_u8(dst + copied, p0);
        vst1_u8(dst + copied + 8, p1);
        vst1_u8(dst + copied + 16, p2);
        vst1_u8(dst + copied + 24, p3);
        copied += stride;
    } while (copied < ml);

#elif defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    // pshufb only reaches 16 bytes per lane, so the 32-byte table is emulated:
    // shuffle both halves, then pick per byte on bit 4 of the index.
    const __m128i t0 = _mm_loadu_si128((const __m128i*)(dst - off));
    const __m128i t1 = _mm_loadu_si128((const __m128i*)(dst - off + 16));
    const __m128i sixteen = _mm_set1_epi8(16);
    const __m128i m_lo = _mm_load_si128((const __m128i*)zxc_overlap_masks32[off]);
    const __m128i m_hi = _mm_load_si128((const __m128i*)(zxc_overlap_masks32[off] + 16));
    const __m128i pat_lo =
        _mm_blendv_epi8(_mm_shuffle_epi8(t0, m_lo),
                        _mm_shuffle_epi8(t1, _mm_sub_epi8(m_lo, sixteen)), _mm_slli_epi16(m_lo, 3));
    const __m128i pat_hi =
        _mm_blendv_epi8(_mm_shuffle_epi8(t0, m_hi),
                        _mm_shuffle_epi8(t1, _mm_sub_epi8(m_hi, sixteen)), _mm_slli_epi16(m_hi, 3));
    do {
        _mm_storeu_si128((__m128i*)(dst + copied), pat_lo);
        _mm_storeu_si128((__m128i*)(dst + copied + 16), pat_hi);
        copied += stride;
    } while (copied < ml);

#else
    // SSE2 (no PSHUFB) and non-SIMD tiers: build the 32-byte pattern with a
    // wrap counter (no per-byte modulo), then store it as two 16-byte halves.
    const uint8_t* src = dst - off;
    uint8_t pat[32];
    uint32_t k = 0;
    for (size_t i = 0; i < 32; i++) {
        pat[i] = src[k];
        if (++k == off) k = 0;
    }
    do {
        zxc_copy16(dst + copied, pat);
        zxc_copy16(dst + copied + 16, pat + 16);
        copied += stride;
    } while (copied < ml);
#endif
}

/**
 * @brief Overlap copy for a run of period @p off (2..31) bounded by @c ml <= 32.
 *
 * Bounded length buys two things over zxc_decode_copy_overlap_run32(): the
 * loop disappears, and the upper 16 bytes are skipped whenever @p ml allows -
 * which is most of the time, GLO matches averaging 7 to 11 bytes here.
 *
 * The lower half never needs the second source register: its mask indices are
 * `i % off` for `i < 16`, hence always below 16. Only the upper half can reach
 * past 15, and only when `off > 16`.
 *
 * Overshoots to 16 or 32 bytes, so the caller owes @ref ZXC_PAD_SIZE of
 * headroom. Reading 32 source bytes is safe because no mask index reaches
 * `dst` (all are `< off`).
 *
 * @param[out] dst Output cursor; the run source is `dst - off`.
 * @param[in]  off Back-reference distance, in [2, 31].
 * @param[in]  ml  Match length, `<= 32`.
 */
// codeql[cpp/unused-static-function] : False positive
static ZXC_ALWAYS_INLINE void zxc_decode_copy_overlap_short(uint8_t* dst, const uint32_t off,
                                                            const uint64_t ml) {
#if defined(ZXC_USE_NEON64)
    const uint8x16_t s0 = vld1q_u8(dst - off);
    vst1q_u8(dst, vqtbl1q_u8(s0, vld1q_u8(zxc_overlap_masks32[off])));
    if (UNLIKELY(ml > 16)) {
        uint8x16x2_t tbl;
        tbl.val[0] = s0;
        tbl.val[1] = vld1q_u8(dst - off + 16);
        vst1q_u8(dst + 16, vqtbl2q_u8(tbl, vld1q_u8(zxc_overlap_masks32[off] + 16)));
    }

#elif defined(ZXC_USE_NEON32)
    uint8x8x2_t lo_tbl;
    lo_tbl.val[0] = vld1_u8(dst - off);
    lo_tbl.val[1] = vld1_u8(dst - off + 8);
    const uint8_t* const m = zxc_overlap_masks32[off];
    vst1_u8(dst, vtbl2_u8(lo_tbl, vld1_u8(m)));
    vst1_u8(dst + 8, vtbl2_u8(lo_tbl, vld1_u8(m + 8)));
    if (UNLIKELY(ml > 16)) {
        uint8x8x4_t tbl;
        tbl.val[0] = lo_tbl.val[0];
        tbl.val[1] = lo_tbl.val[1];
        tbl.val[2] = vld1_u8(dst - off + 16);
        tbl.val[3] = vld1_u8(dst - off + 24);
        vst1_u8(dst + 16, vtbl4_u8(tbl, vld1_u8(m + 16)));
        vst1_u8(dst + 24, vtbl4_u8(tbl, vld1_u8(m + 24)));
    }

#elif defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    // pshufb reaches 16 bytes, so only the upper half needs two tables: shuffle
    // both and pick per byte on bit 4 of the index.
    const __m128i t0 = _mm_loadu_si128((const __m128i*)(dst - off));
    _mm_storeu_si128(
        (__m128i*)dst,
        _mm_shuffle_epi8(t0, _mm_load_si128((const __m128i*)zxc_overlap_masks32[off])));
    if (UNLIKELY(ml > 16)) {
        const __m128i t1 = _mm_loadu_si128((const __m128i*)(dst - off + 16));
        const __m128i m_hi = _mm_load_si128((const __m128i*)(zxc_overlap_masks32[off] + 16));
        _mm_storeu_si128(
            (__m128i*)(dst + 16),
            _mm_blendv_epi8(_mm_shuffle_epi8(t0, m_hi),
                            _mm_shuffle_epi8(t1, _mm_sub_epi8(m_hi, _mm_set1_epi8(16))),
                            _mm_slli_epi16(m_hi, 3)));
    }

#else
    const uint8_t* src = dst - off;
    const size_t n = (ml > 16) ? 32 : 16;
    uint32_t k = 0;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[k];
        if (++k == off) k = 0;
    }
#endif
}

/**
 * @brief Fills an @p ml-byte single-byte run (LZ offset == 1) with wild stores.
 *
 * Splats @p byte into a vector register and emits 32-byte chunks, avoiding a
 * libc memset call on the typically short runs of the hot path. Like the other
 * run copiers it may **overshoot** up to 31 bytes past @p ml; the caller must
 * guarantee @ref ZXC_PAD_SIZE bytes of headroom. Falls back to
 * `ZXC_MEMSET` on non-SIMD builds.
 *
 * @param[out] dst  Output cursor.
 * @param[in]  byte Byte value to replicate.
 * @param[in]  ml   Run length in bytes (>= 1).
 */
// codeql[cpp/unused-static-function] : False positive, used in DECODE_MATCH_SAFE/FAST macros
static ZXC_ALWAYS_INLINE void zxc_decode_fill_run(uint8_t* dst, const uint8_t byte,
                                                  const uint64_t ml) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    const __m256i v = _mm256_set1_epi8((char)byte);
    _mm256_storeu_si256((__m256i*)dst, v);
    if (UNLIKELY(ml > 32)) {
        uint8_t* out = dst + 32;
        size_t rem = ml - 32;
        while (rem > 32) {
            _mm256_storeu_si256((__m256i*)out, v);
            out += 32;
            rem -= 32;
        }
        _mm256_storeu_si256((__m256i*)out, v);
    }
#elif defined(ZXC_USE_SSE2)
    const __m128i v = _mm_set1_epi8((char)byte);
    _mm_storeu_si128((__m128i*)dst, v);
    _mm_storeu_si128((__m128i*)(dst + 16), v);
    if (UNLIKELY(ml > 32)) {
        uint8_t* out = dst + 32;
        size_t rem = ml - 32;
        while (rem > 32) {
            _mm_storeu_si128((__m128i*)out, v);
            _mm_storeu_si128((__m128i*)(out + 16), v);
            out += 32;
            rem -= 32;
        }
        _mm_storeu_si128((__m128i*)out, v);
        _mm_storeu_si128((__m128i*)(out + 16), v);
    }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    const uint8x16_t v = vdupq_n_u8(byte);
    vst1q_u8(dst, v);
    vst1q_u8(dst + 16, v);
    if (UNLIKELY(ml > 32)) {
        uint8_t* out = dst + 32;
        size_t rem = ml - 32;
        while (rem > 32) {
            vst1q_u8(out, v);
            vst1q_u8(out + 16, v);
            out += 32;
            rem -= 32;
        }
        vst1q_u8(out, v);
        vst1q_u8(out + 16, v);
    }
#else
    ZXC_MEMSET(dst, byte, ml);
#endif
}

// ==========================================================================
// Shared decode macros for the GLO and GHI decoders (fast + safe variants).
// Defined at file scope to avoid four identical copies inside each function.
// They reference the local names l_ptr, d_ptr, d_floor that every call site
// has in scope. #undef-ed at the end of the last consumer.
// ==========================================================================

/**
 * @brief Copies @p ll literal bytes from @p src to @p dst using 32-byte wild copies.
 *
 * Writes in 32-byte chunks -- the width of @ref zxc_copy32 -- and may
 * **overshoot** by up to 31 bytes past @p ll; the caller must guarantee @p dst
 * has at least @ref ZXC_PAD_SIZE bytes of writable headroom (the unrolled loops
 * and the trailing-literal margins ensure this). Pointers are taken by value and
 * the caller advances its own cursors by @p ll, keeping them in registers on the
 * hot path.
 *
 * @param[out] dst Output cursor. Must not overlap @p src and must have
 *                 @ref ZXC_PAD_SIZE bytes of overshoot headroom.
 * @param[in]  src Literal-stream source. Must not overlap @p dst (RESTRICT).
 * @param[in]  ll  Number of literal bytes to copy.
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_literals(uint8_t* RESTRICT dst,
                                                       const uint8_t* RESTRICT src,
                                                       const uint64_t ll) {
    zxc_copy32(dst, src);
    if (UNLIKELY(ll > 32)) {
        dst += 32;
        src += 32;
        size_t rem = ll - 32;
        while (rem > 32) {
            zxc_copy32(dst, src);
            dst += 32;
            src += 32;
            rem -= 32;
        }
        zxc_copy32(dst, src);
    }
}

/**
 * @brief Copies an @p ml-byte LZ match from @c d_ptr-off to @p d_ptr, handling overlap.
 *
 * The source @c d_ptr-off may overlap the destination (the LZ repeat case), so the
 * copy strategy is chosen by back-reference distance:
 *  - @p off >= 32   : 32-byte wild copies (no overlap within a chunk);
 *  - @p off == 1    : single-byte run via @ref zxc_decode_fill_run;
 *  - otherwise 2-31 : pattern-replicating overlap copy.
 *
 * The 32 and 16 below are the widths of @ref zxc_copy32 and @ref zxc_copy16 --
 * a copy may never be wider than the distance it reads back, or it would read
 * bytes it is still writing. They are not @ref ZXC_PAD_SIZE, which measures the
 * overshoot the caller leaves writable and merely happens to equal 32.
 *
 * Like @ref zxc_decode_copy_literals it may **overshoot** up to 31 bytes past
 * @p ml, so @p d_ptr must have @ref ZXC_PAD_SIZE bytes of headroom. @p d_ptr
 * is taken by value; the caller advances its cursor by @p ml.
 *
 * @param[in,out] d_ptr Output cursor; the match source is @c d_ptr-off. Must have
 *                      @ref ZXC_PAD_SIZE bytes of overshoot headroom.
 * @param[in]     off   Resolved (bias-removed) back-reference distance, @c >= 1.
 * @param[in]     ml    Match length in bytes (@c >= ZXC_LZ_MIN_MATCH_LEN).
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_match(uint8_t* RESTRICT d_ptr, const uint32_t off,
                                                    const uint64_t ml) {
    const uint8_t* match_src = d_ptr - off;
    if (LIKELY(off >= 32)) {
        zxc_copy32(d_ptr, match_src);
        if (UNLIKELY(ml > 32)) {
            uint8_t* out = d_ptr + 32;
            const uint8_t* ref = match_src + 32;
            size_t rem = ml - 32;
            while (rem > 32) {
                zxc_copy32(out, ref);
                out += 32;
                ref += 32;
                rem -= 32;
            }
            zxc_copy32(out, ref);
        }
    } else if (off == 1) {
        zxc_decode_fill_run(d_ptr, match_src[0], ml);
    } else {
        zxc_decode_copy_overlap_run32(d_ptr, off, ml);
    }
}

/**
 * @brief Match copy for a length that cannot exceed 32 bytes.
 *
 * @ref zxc_decode_copy_match without its length ladder, which a bounded @p ml
 * makes unreachable. Below 32 a full-width copy would read destination bytes
 * not written yet, hence the overlap form.
 *
 * The 32 is the width of @ref zxc_copy32, not @ref ZXC_PAD_SIZE - they only
 * happen to share a value.
 *
 * @param[in,out] d_ptr Output cursor; match source is @c d_ptr-off. Needs
 *                      @ref ZXC_PAD_SIZE bytes of overshoot headroom.
 * @param[in]     off   Resolved back-reference distance, @c >= 1.
 * @param[in]     ml    Match length, @c <= 32 (caller's obligation).
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_match_short(uint8_t* RESTRICT d_ptr,
                                                          const uint32_t off, const uint64_t ml) {
    const uint8_t* match_src = d_ptr - off;
    if (LIKELY(off >= 32)) {
        zxc_copy32(d_ptr, match_src);
    } else if (off == 1) {
        zxc_decode_fill_run(d_ptr, match_src[0], ml);
    } else {
        zxc_decode_copy_overlap_short(d_ptr, off, ml);
    }
}

/**
 * @brief GLO match copy: bounded form when the ml nibble did not saturate.
 *
 * The test is exact (the escape always yields more) and statically decidable
 * from either predecessor, so jump threading folds it away. GHI uses
 * @ref zxc_decode_copy_match directly - its inline ml reaches 259.
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_match_glo(uint8_t* RESTRICT d_ptr, const uint32_t off,
                                                        const uint64_t ml) {
    if (LIKELY(ml <= ZXC_GLO_MAX_INLINE_ML)) {
        zxc_decode_copy_match_short(d_ptr, off, ml);
    } else {
        zxc_decode_copy_match(d_ptr, off, ml);
    }
}

/**
 * @brief Exact-size match copy for the tail loops (no overshoot headroom).
 *
 * The tail/remaining-sequence loops validate against @c d_end exactly, so the
 * wild-copy ladder above (which overshoots up to @ref ZXC_PAD_SIZE) is off
 * limits there. For overlapping matches this expands the run by doubling:
 * every ZXC_MEMCPY source range ends at or before its destination start, and
 * nothing is written past @c d_ptr+ml. Since each copied prefix length is a
 * multiple of @p off, the result is byte-identical to the naive per-byte
 * copy, in O(log(ml/off)) calls instead of O(ml) iterations.
 *
 * @param[in,out] d_ptr     Output cursor (match source is @c d_ptr-off).
 * @param[in]     match_src Match source, equal to @c d_ptr-off.
 * @param[in]     off       Back-reference distance, @c >= 1.
 * @param[in]     ml        Match length in bytes.
 */
static ZXC_NOINLINE void zxc_decode_copy_match_exact(uint8_t* d_ptr, const uint8_t* match_src,
                                                     const size_t off, const size_t ml) {
    if (off >= ml) {
        ZXC_MEMCPY(d_ptr, match_src, ml);
    } else if (ml < 16) {
        for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
    } else {
        size_t n = off;
        ZXC_MEMCPY(d_ptr, match_src, n);
        while (n <= ml - n) {
            ZXC_MEMCPY(d_ptr + n, d_ptr, n);
            n <<= 1;
        }
        ZXC_MEMCPY(d_ptr + n, d_ptr, ml - n);
    }
}

// Match emission only. The literal copy lives in the per-format sequence macros
// because the two formats want different strategies: GLO's inline ll is at most
// 14, so one zxc_copy16 covers it, while GHI's reaches 254 and needs the
// 32-byte ladder.

// SAFE version: rejects a match reaching below d_floor. COPY is the format's
// match-copy helper, so GLO can pass the length-bounded one.
#define DECODE_MATCH_SAFE(ml, off, COPY)                                              \
    do {                                                                              \
        if (UNLIKELY((size_t)(d_ptr - d_floor) < (off))) return ZXC_ERROR_BAD_OFFSET; \
        COPY(d_ptr, off, ml);                                                         \
        d_ptr += ml;                                                                  \
    } while (0)

// FAST version: no offset check. Only reached past d_bounds, where no encodable
// offset can reach below d_floor, so the check above would always pass.
#define DECODE_MATCH_FAST(ml, off, COPY) \
    do {                                 \
        COPY(d_ptr, off, ml);            \
        d_ptr += ml;                     \
    } while (0)

/**
 * @brief True when the GLO offset stream stores 1-byte offsets, 2-byte otherwise.
 *
 * The single definition of the offset width: the stream-size check, the
 * validation span and every read site expand from this, so a bound can never
 * drift from the width actually consumed. A macro rather than a local flag on
 * purpose -- holding one more value live across the decode costs more spills in
 * these register-starved loops than re-reading the header field.
 *
 * References the call site's `gh`, like the DECODE_* macros below.
 */
#define GLO_OFF8 (gh.enc_off == 1)

/**
 * @brief Decodes one GLO sequence of a 4x batch: extracts ll/ml, applies the
 *        varint extensions with their bounds checks, then emits via @p DECODE.
 *
 * Like `DECODE_MATCH_SAFE`, references the call site's local names (e_ptr,
 * e_end, l_ptr, l_end, d_ptr, d_end). @p RESERVE is the sum of the inline
 * (pre-varint) literal lengths of the batch's remaining sequences, so one
 * l_ptr bound covers the whole batch; @p N_REM counts the remaining
 * sequences, reserving their worst-case inline output in the d_ptr bound
 * (0 for the last sequence - the compiler folds the dead terms). @p ON_FAIL
 * is `goto rollback_*` in the state-saving safe variants and
 * `return ZXC_ERROR_OVERFLOW` otherwise.
 */
#define DECODE_GLO_SEQ(LL, ML, OFF, RESERVE, N_REM, DECODE, ON_FAIL)                            \
    do {                                                                                        \
        uint64_t ll = (LL);                                                                     \
        uint64_t ml = (ML);                                                                     \
        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) {                                                \
            ll += zxc_read_varint(&e_ptr, e_end);                                               \
            const uint64_t reserve = (RESERVE);                                                 \
            /* An extended ll eats the batch's destination budget, so reserve                   \
             * this sequence's match and the remaining ones too. ml is still                    \
             * the raw nibble here; an escaped one re-checks itself below. */                   \
            if (UNLIKELY(ll + reserve > (size_t)(l_end - l_ptr) ||                              \
                         ll + ml + ZXC_LZ_MIN_MATCH_LEN +                                       \
                                 (N_REM) * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >      \
                             (size_t)(d_end - d_ptr)))                                          \
                ON_FAIL;                                                                        \
            zxc_decode_copy_literals(d_ptr, l_ptr, ll);                                         \
        } else {                                                                                \
            /* ll <= 14 here, so one 16-byte store covers it. */                                \
            zxc_copy16(d_ptr, l_ptr);                                                           \
        }                                                                                       \
        l_ptr += ll;                                                                            \
        d_ptr += ll;                                                                            \
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) {                                                \
            ml += zxc_read_varint(&e_ptr, e_end);                                               \
            /* d_ptr already carries the literals, hence no `ll +` term here. */                \
            if (UNLIKELY(ml + ZXC_LZ_MIN_MATCH_LEN + (N_REM) * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + \
                             ZXC_PAD_SIZE >                                                     \
                         (size_t)(d_end - d_ptr)))                                              \
                ON_FAIL;                                                                        \
        }                                                                                       \
        ml += ZXC_LZ_MIN_MATCH_LEN;                                                             \
        DECODE(ml, OFF, zxc_decode_copy_match_glo);                                             \
    } while (0)

/**
 * @brief One full GLO 4x batch: token word, four offsets (1- or 2-byte form),
 *        four @ref DECODE_GLO_SEQ emissions, sequence-count update.
 */
#define DECODE_GLO_BATCH_4X(DECODE, ON_FAIL)                                                      \
    do {                                                                                          \
        uint32_t tokens = zxc_le32(t_ptr);                                                        \
        t_ptr += sizeof(uint32_t);                                                                \
        uint32_t off1 = ZXC_LZ_OFFSET_BIAS, off2 = ZXC_LZ_OFFSET_BIAS, off3 = ZXC_LZ_OFFSET_BIAS, \
                 off4 = ZXC_LZ_OFFSET_BIAS;                                                       \
        if (GLO_OFF8) {                                                                           \
            uint32_t offsets = zxc_le32(o_ptr);                                                   \
            o_ptr += sizeof(uint32_t);                                                            \
            off1 += offsets & 0xFF;                                                               \
            off2 += (offsets >> 8) & 0xFF;                                                        \
            off3 += (offsets >> 16) & 0xFF;                                                       \
            off4 += (offsets >> 24) & 0xFF;                                                       \
        } else {                                                                                  \
            uint64_t offsets = zxc_le64(o_ptr);                                                   \
            o_ptr += sizeof(uint64_t);                                                            \
            off1 += (uint32_t)(offsets & 0xFFFF);                                                 \
            off2 += (uint32_t)((offsets >> 16) & 0xFFFF);                                         \
            off3 += (uint32_t)((offsets >> 32) & 0xFFFF);                                         \
            off4 += (uint32_t)((offsets >> 48) & 0xFFFF);                                         \
        }                                                                                         \
        DECODE_GLO_SEQ((tokens & 0x0F0) >> 4, (tokens & 0x00F), off1,                             \
                       ((tokens >> 12) & 0xF) + ((tokens >> 20) & 0xF) + (tokens >> 28), 3U,      \
                       DECODE, ON_FAIL);                                                          \
        DECODE_GLO_SEQ((tokens & 0x0F000) >> 12, (tokens & 0x00F00) >> 8, off2,                   \
                       ((tokens >> 20) & 0xF) + (tokens >> 28), 2U, DECODE, ON_FAIL);             \
        DECODE_GLO_SEQ((tokens & 0x0F00000) >> 20, (tokens & 0x00F0000) >> 16, off3,              \
                       (tokens >> 28), 1U, DECODE, ON_FAIL);                                      \
        DECODE_GLO_SEQ((tokens >> 28), (tokens >> 24) & 0x0F, off4, 0, 0U, DECODE, ON_FAIL);      \
        n_seq -= 4;                                                                               \
    } while (0)

/**
 * @brief GHI twin of @ref DECODE_GLO_SEQ, decoding one sequence word @p S
 *        (ll in the top byte, ml bits, 16-bit offset). References the call
 *        site's extras_ptr/extras_end instead of e_ptr/e_end.
 */
#define DECODE_GHI_SEQ(S, RESERVE, N_REM, DECODE, ON_FAIL)                                   \
    do {                                                                                     \
        uint64_t ll = (S) >> 24;                                                             \
        const uint32_t mb = ((S) >> 16) & 0xFF;                                              \
        uint64_t ml = mb + ZXC_LZ_MIN_MATCH_LEN;                                             \
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) {                                               \
            ll += zxc_read_varint(&extras_ptr, extras_end);                                  \
            const uint64_t reserve = (RESERVE);                                              \
            /* Same reservation as the GLO twin; GHI's inline ml reaches 259. */             \
            if (UNLIKELY(ll + reserve > (size_t)(l_end - l_ptr) ||                           \
                         ll + ml + (N_REM) * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE > \
                             (size_t)(d_end - d_ptr)))                                       \
                ON_FAIL;                                                                     \
        }                                                                                    \
        if (UNLIKELY(mb == ZXC_SEQ_ML_MASK)) {                                               \
            ml += zxc_read_varint(&extras_ptr, extras_end);                                  \
            if (UNLIKELY(ll + ml + (N_REM) * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE > \
                         (size_t)(d_end - d_ptr)))                                           \
                ON_FAIL;                                                                     \
        }                                                                                    \
        const uint32_t off = ((S) & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;                            \
        /* GHI keeps the ladder: its inline ll reaches 254. */                               \
        zxc_decode_copy_literals(d_ptr, l_ptr, ll);                                          \
        l_ptr += ll;                                                                         \
        d_ptr += ll;                                                                         \
        DECODE(ml, off, zxc_decode_copy_match);                                              \
    } while (0)

/**
 * @brief One full GHI 4x batch. @p PREFETCH is the literal-stream prefetch
 *        statement of the post-threshold FAST loops ((void)0 elsewhere),
 *        placed exactly where the hand-unrolled bodies had it.
 */
#define DECODE_GHI_BATCH_4X(DECODE, ON_FAIL, PREFETCH)                                 \
    do {                                                                               \
        uint32_t s1 = zxc_le32(seq_ptr);                                               \
        uint32_t s2 = zxc_le32(seq_ptr + sizeof(uint32_t));                            \
        uint32_t s3 = zxc_le32(seq_ptr + 2 * sizeof(uint32_t));                        \
        uint32_t s4 = zxc_le32(seq_ptr + 3 * sizeof(uint32_t));                        \
        seq_ptr += 4 * sizeof(uint32_t);                                               \
        PREFETCH;                                                                      \
        DECODE_GHI_SEQ(s1, (s2 >> 24) + (s3 >> 24) + (s4 >> 24), 3U, DECODE, ON_FAIL); \
        DECODE_GHI_SEQ(s2, (s3 >> 24) + (s4 >> 24), 2U, DECODE, ON_FAIL);              \
        DECODE_GHI_SEQ(s3, (s4 >> 24), 1U, DECODE, ON_FAIL);                           \
        DECODE_GHI_SEQ(s4, 0, 0U, DECODE, ON_FAIL);                                    \
        n_seq -= 4;                                                                    \
    } while (0)

/**
 * @brief Ensures the entropy decode scratch (tok_buffer + pivco_scratch) is
 *        available before decoding an entropy section.
 *
 * Heap contexts defer this scratch to the first entropy section
 * (@ref zxc_cctx_alloc_entropy_scratch allocates it once); static workspaces
 * pre-carve it, making this a no-op. The const cast is sound: every context
 * lives in writable memory (heap allocation or caller workspace) - the
 * decode chain is const only because steady-state decoding never mutates
 * the context.
 *
 * @param[in] ctx  Decompression context (mutated on the first entropy block).
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_MEMORY on allocation failure.
 */
static ZXC_NOINLINE ZXC_COLD int zxc_ensure_entropy_scratch(const zxc_cctx_t* RESTRICT ctx) {
    if (LIKELY(ctx->pivco_scratch != NULL)) return ZXC_OK;
    return zxc_cctx_alloc_entropy_scratch((zxc_cctx_t*)(uintptr_t)ctx);
}

static ZXC_NOINLINE ZXC_COLD int zxc_decode_lit_pivco(const zxc_cctx_t* RESTRICT ctx,
                                                      const uint8_t* RESTRICT payload,
                                                      const size_t psize,
                                                      const size_t required_size) {
    const int arc = zxc_ensure_entropy_scratch(ctx);
    if (UNLIKELY(arc != ZXC_OK)) return arc;
    if (UNLIKELY(ctx->lit_buffer_cap < required_size + ZXC_PAD_SIZE ||
                 ctx->pivco_scratch_cap < required_size + ZXC_PIVCO_SCRATCH_PAD))
        return ZXC_ERROR_CORRUPT_DATA;
    return zxc_huf_decode_section(payload, psize, ctx->lit_buffer, required_size,
                                  ctx->pivco_scratch);
}

static ZXC_NOINLINE ZXC_COLD int zxc_decode_lit_pivco_dict(const zxc_cctx_t* RESTRICT ctx,
                                                           const uint8_t* RESTRICT payload,
                                                           const size_t psize,
                                                           const size_t required_size) {
    if (UNLIKELY(!ctx->dict_huf_tree_ok)) return ZXC_ERROR_DICT_REQUIRED;
    const int arc = zxc_ensure_entropy_scratch(ctx);
    if (UNLIKELY(arc != ZXC_OK)) return arc;
    if (UNLIKELY(ctx->lit_buffer_cap < required_size + ZXC_PAD_SIZE ||
                 ctx->pivco_scratch_cap < required_size + ZXC_PIVCO_SCRATCH_PAD))
        return ZXC_ERROR_CORRUPT_DATA;
    return zxc_huf_decode_section_dict(payload, psize, ctx->lit_buffer, required_size,
                                       &ctx->dict_huf->tree, &ctx->dict_huf->dec,
                                       ctx->pivco_scratch);
}

static ZXC_NOINLINE ZXC_COLD int zxc_decode_tok_pivco(const zxc_cctx_t* RESTRICT ctx,
                                                      const uint8_t* RESTRICT payload,
                                                      const size_t psize, const size_t n_tok) {
    const int arc = zxc_ensure_entropy_scratch(ctx);
    if (UNLIKELY(arc != ZXC_OK)) return arc;
    if (UNLIKELY(n_tok + ZXC_PAD_SIZE > ctx->tok_buffer_cap ||
                 n_tok + ZXC_PIVCO_SCRATCH_PAD > ctx->pivco_scratch_cap))
        return ZXC_ERROR_CORRUPT_DATA;
    return zxc_huf_decode_section(payload, psize, ctx->tok_buffer, n_tok, ctx->pivco_scratch);
}

static ZXC_NOINLINE int zxc_decode_block_glo_entropy(const zxc_cctx_t* RESTRICT ctx,
                                                     const uint8_t* RESTRICT src, size_t src_size,
                                                     uint8_t* RESTRICT dst, size_t dst_capacity);
static ZXC_NOINLINE int zxc_decode_block_glo_entropy_dict(const zxc_cctx_t* RESTRICT ctx,
                                                          const uint8_t* RESTRICT src,
                                                          size_t src_size, uint8_t* RESTRICT dst,
                                                          size_t dst_capacity);
static ZXC_NOINLINE int zxc_decode_block_glo_entropy_safe(const zxc_cctx_t* RESTRICT ctx,
                                                          const uint8_t* RESTRICT src,
                                                          size_t src_size, uint8_t* RESTRICT dst,
                                                          size_t dst_capacity);

/**
 * @brief Unified GLO (General Low) block decoder body, shared by the fast,
 *        safe, dictionary and entropy-token variants.
 *
 * Decodes a block in the internal GLO format. The decompressed size is not on
 * the wire: it falls out of walking the sequences. @p safe, @p has_dict and
 * @p tok_entropy must be compile-time constants (0 or 1): the 4x-unrolled
 * loops are duplicated inside @c if(safe)/else branches so each variant keeps
 * single-assignment @c const save pointers, and after constant propagation
 * only one branch survives per wrapper (codegen equivalent to a hand-written
 * pair).
 *
 * The @p tok_entropy dimension keeps the token pointer on a SINGLE provenance
 * per instantiation (in-place @c src tokens vs @c ctx->tok_buffer): the
 * tok_entropy=0 instantiations tail-call their entropy twin right after the
 * header parse when they meet enc_tok == 2.
 *
 * @param[in,out] ctx          Decompression context (dict buffer, scratch).
 * @param[in]     src          Compressed block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer for decoded bytes.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @param[in]     safe         Compile-time flag: 1 = strict bounds-checked loop.
 * @param[in]     has_dict     Compile-time flag: 1 = resolve matches against a dict prefix.
 * @param[in]     tok_entropy  Compile-time flag: 1 = tokens are a PivCo section
 *                             decoded into @c ctx->tok_buffer (level 7).
 * @return Bytes written to @p dst on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decode_block_glo_impl(const zxc_cctx_t* RESTRICT ctx,
                                                       const uint8_t* RESTRICT src,
                                                       const size_t src_size, uint8_t* RESTRICT dst,
                                                       const size_t dst_capacity, const int safe,
                                                       const int has_dict, const int tok_entropy) {
    zxc_gnr_header_t gh;

    // Constant 0 when !has_dict, so `d_floor` folds to `dst`.
    const size_t dict_size = has_dict ? ctx->dict_size : 0;
    uint32_t lit_comp;
    uint32_t tok_comp;

    const int hdr_sz = zxc_read_glo_header_and_desc(src, src_size, &gh, &lit_comp, &tok_comp);
    if (UNLIKELY(hdr_sz < 0)) return ZXC_ERROR_BAD_HEADER;

    // Only 0 and 1 are defined widths.
    if (UNLIKELY(gh.enc_off > 1)) return ZXC_ERROR_CORRUPT_DATA;

    // Entropy-coded tokens (level 7): tail-call the dedicated instantiation
    // before any section work - it restarts from the header, so only the parse
    // above is duplicated. Flags are constant per instantiation, so exactly one
    // call survives and t_ptr keeps a single provenance below.
    if (!tok_entropy && UNLIKELY(gh.enc_tok == ZXC_SECTION_ENCODING_HUFFMAN)) {
        if (safe) return zxc_decode_block_glo_entropy_safe(ctx, src, src_size, dst, dst_capacity);
        if (has_dict)
            return zxc_decode_block_glo_entropy_dict(ctx, src, src_size, dst, dst_capacity);
        return zxc_decode_block_glo_entropy(ctx, src, src_size, dst, dst_capacity);
    }

    const uint8_t* p_data = src + (size_t)hdr_sz;
    const uint8_t* p_curr = p_data;

    // --- Literal Stream Setup ---
    const uint8_t* l_ptr;
    const uint8_t* l_end;
    uint8_t* rle_buf = NULL;

    size_t lit_stream_size = lit_comp;
    // Decoded size of an encoded literal section (RAW ignores it).
    const size_t required_size = gh.n_literals;

    if (gh.enc_lit == ZXC_SECTION_ENCODING_HUFFMAN ||
        gh.enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT) {
        if (UNLIKELY(lit_stream_size > (size_t)(src + src_size - p_curr)))
            return ZXC_ERROR_CORRUPT_DATA;
        if (required_size == 0) {
            l_ptr = p_curr;
            l_end = p_curr;
        } else {
            if (UNLIKELY(required_size > dst_capacity || required_size > SIZE_MAX - ZXC_PAD_SIZE))
                return ZXC_ERROR_DST_TOO_SMALL;
            const int rc =
                (gh.enc_lit == ZXC_SECTION_ENCODING_HUFFMAN)
                    ? zxc_decode_lit_pivco(ctx, p_curr, lit_stream_size, required_size)
                    : zxc_decode_lit_pivco_dict(ctx, p_curr, lit_stream_size, required_size);
            if (UNLIKELY(rc != ZXC_OK)) return rc;
            l_ptr = ctx->lit_buffer;
            l_end = ctx->lit_buffer + required_size;
        }
    } else if (gh.enc_lit == ZXC_SECTION_ENCODING_RLE) {
        if (required_size > 0) {
            if (UNLIKELY(required_size > dst_capacity || required_size > SIZE_MAX - ZXC_PAD_SIZE))
                return ZXC_ERROR_DST_TOO_SMALL;
            const size_t alloc_size = required_size + ZXC_PAD_SIZE;

            // lit_buffer is pre-allocated to chunk_size + ZXC_PAD_SIZE by
            // zxc_cctx_init (mode == 0).
            if (UNLIKELY(ctx->lit_buffer_cap < alloc_size)) return ZXC_ERROR_CORRUPT_DATA;

            rle_buf = ctx->lit_buffer;
            if (UNLIKELY(!rle_buf || lit_stream_size > (size_t)(src + src_size - p_curr)))
                return ZXC_ERROR_CORRUPT_DATA;

            const uint8_t* r_ptr = p_curr;
            const uint8_t* r_end = r_ptr + lit_stream_size;
            uint8_t* w_ptr = rle_buf;
            const uint8_t* const w_end = rle_buf + required_size;

            while (r_ptr < r_end && w_ptr < w_end) {
                uint8_t token = *r_ptr++;
                if (LIKELY(!(token & ZXC_LIT_RLE_FLAG))) {
                    // Raw copy (most common path): 32-byte wild copies (zxc_copy32 width)
                    // token is 7-bit (0-127), so len is 1-128 bytes
                    const uint32_t len = (uint32_t)token + 1;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr + len > r_end))
                        return ZXC_ERROR_CORRUPT_DATA;

                    // Destination has ZXC_PAD_SIZE bytes of safe overrun space.
                    // Source may not - check before wild copy.
                    // Fast path: source holds a full zxc_copy32 read (most common)
                    if (LIKELY(r_ptr + 32 <= r_end)) {
                        // Single copy covers len <= 32 (most tokens)
                        zxc_copy32(w_ptr, r_ptr);

                        if (UNLIKELY(len > 32)) {
                            // Unroll: max len=128, so max 4 copies total. The last
                            // copy is placed to end exactly on len, overlapping the
                            // previous one - unconditional stores beat branches here.
                            if (len <= 64) {
                                zxc_copy32(w_ptr + len - 32, r_ptr + len - 32);
                            } else if (len <= 96) {
                                zxc_copy32(w_ptr + 32, r_ptr + 32);
                                zxc_copy32(w_ptr + len - 32, r_ptr + len - 32);
                            } else {
                                zxc_copy32(w_ptr + 32, r_ptr + 32);
                                zxc_copy32(w_ptr + 64, r_ptr + 64);
                                zxc_copy32(w_ptr + len - 32, r_ptr + len - 32);
                            }
                        }
                    } else {
                        // Near end of source: safe copy (rare cold path)
                        ZXC_MEMCPY(w_ptr, r_ptr, len);
                    }

                    w_ptr += len;
                    r_ptr += len;
                } else {
                    // RLE run: fill with single byte
                    const uint32_t len = (token & ZXC_LIT_LEN_MASK) + 4;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr >= r_end))
                        return ZXC_ERROR_CORRUPT_DATA;
                    ZXC_MEMSET(w_ptr, *r_ptr++, len);
                    w_ptr += len;
                }
            }
            if (UNLIKELY(w_ptr != w_end)) return ZXC_ERROR_CORRUPT_DATA;
            l_ptr = rle_buf;
            l_end = rle_buf + required_size;
        } else {
            l_ptr = p_curr;
            l_end = p_curr;
        }
    } else if (gh.enc_lit == ZXC_SECTION_ENCODING_RAW) {
        l_ptr = p_curr;
        l_end = p_curr + lit_stream_size;
    } else {
        return ZXC_ERROR_CORRUPT_DATA;
    }

    p_curr += lit_stream_size;

    // --- Stream Pointers & Validation ---
    // Only the literal and token sizes are on the wire; offsets follow from the
    // sequence count and width, and extras take the payload residue.
    const size_t sz_tokens = tok_comp;
    const uint64_t sz_offsets = GLO_OFF8 ? (uint64_t)gh.n_sequences : (uint64_t)gh.n_sequences * 2;

    const size_t payload_avail = (size_t)(src + src_size - p_data);
    const uint64_t consumed = (uint64_t)lit_stream_size + (uint64_t)sz_tokens + sz_offsets;
    if (UNLIKELY(consumed > (uint64_t)payload_avail)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t sz_extras = payload_avail - (size_t)consumed; /* slack padding included */

    // RAW literals point into the caller's buffer and the wild copy overshoots,
    // so it needs ZXC_BLOCK_LIT_SLACK readable bytes behind it. v7 staged short
    // streams into a padded scratch here; v8 rejects instead. No underflow:
    // lit_stream_size <= consumed <= payload_avail.
    if (UNLIKELY(payload_avail - lit_stream_size < ZXC_BLOCK_LIT_SLACK))
        return ZXC_ERROR_CORRUPT_DATA;

    // Offsets/extras follow the on-disk token SECTION; sz_tokens is its size
    // (== n_sequences when RAW, the Huffman payload size when enc_tok set).
    const uint8_t* o_ptr = p_curr + sz_tokens;
    const uint8_t* e_ptr = o_ptr + (size_t)sz_offsets;
    const uint8_t* const e_end = e_ptr + sz_extras;

    const uint8_t* RESTRICT t_ptr;
    if (!tok_entropy) {
        // enc_tok == 2 was re-routed right after the header parse; any other
        // value would have mis-sized the section descriptors above.
        if (UNLIKELY(gh.enc_tok != 0)) return ZXC_ERROR_CORRUPT_DATA;
        t_ptr = p_curr;
    } else {
        if (UNLIKELY(gh.enc_tok != ZXC_SECTION_ENCODING_HUFFMAN)) return ZXC_ERROR_CORRUPT_DATA;
        const int rc = zxc_decode_tok_pivco(ctx, p_curr, sz_tokens, gh.n_sequences);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
        t_ptr = ctx->tok_buffer;
    }

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    // Lowest address a match may reach: the dictionary prefix if any, else dst.
    const uint8_t* const d_floor = dst - dict_size;
    // Destination safe margin for 4x loop: max output without varint extension.
    // ll_max = 14, ml_max = 14 + 5 = 19, per-seq = 33, 4x = 132.
    // Plus the overshoot the wild copies are allowed (ZXC_PAD_SIZE) + 4 safety = 168.
    const uint8_t* const d_end_safe = d_end - (132 + ZXC_PAD_SIZE + 4);

    // Literal margin for the 4x loops: without a varint, ll <= 14 per sequence,
    // so 4 * 14 = 56. Past that margin only the cold varint path checks l_ptr.
    const size_t glo_sz_lit = (size_t)(l_end - l_ptr);
    const size_t glo_margin_4x = 4 * (ZXC_TOKEN_LL_MASK - 1);  // 56
    const size_t glo_margin_1x = ZXC_TOKEN_LL_MASK - 1;        // 14
    const uint8_t* const l_end_safe_4x =
        (glo_sz_lit > glo_margin_4x) ? l_end - glo_margin_4x : l_ptr;
    const uint8_t* const l_end_safe_1x =
        (glo_sz_lit > glo_margin_1x) ? l_end - glo_margin_1x : l_ptr;

    uint32_t n_seq = gh.n_sequences;

    // Offsets only need checking until the output is wider than the widest offset
    // the format can encode (256 for 1-byte offsets, 65536 for 2-byte); past that
    // none can reach below d_floor. A dictionary counts as already written, so a
    // large one puts d_bounds at dst and skips the SAFE loop.
    const size_t off_span = GLO_OFF8 ? (1U << 8) : (1U << 16);
    const size_t span = (dict_size >= off_span) ? 0 : off_span - dict_size;
    const uint8_t* const d_bounds = (span >= dst_capacity) ? d_end : dst + span;

    // --- SAFE Loop: offset validation until d_bounds (4x unroll) ---
    if (safe) {
        // SAFE variant: save per-batch state so overflow can rollback.
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x && d_ptr < d_bounds) {
            const uint8_t* const t_save = t_ptr;
            const uint8_t* const o_save = o_ptr;
            const uint8_t* const e_save = e_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            DECODE_GLO_BATCH_4X(DECODE_MATCH_SAFE, goto rollback_safe_4x);
            continue;

        rollback_safe_4x:
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x && d_ptr < d_bounds) {
            DECODE_GLO_BATCH_4X(DECODE_MATCH_SAFE, return ZXC_ERROR_OVERFLOW);
        }
    }

    // --- FAST Loop: After threshold, no offset validation needed (4x unroll) ---
    if (safe) {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x) {
            const uint8_t* const t_save = t_ptr;
            const uint8_t* const o_save = o_ptr;
            const uint8_t* const e_save = e_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            DECODE_GLO_BATCH_4X(DECODE_MATCH_FAST, goto rollback_fast_4x);
            continue;

        rollback_fast_4x:
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x) {
            DECODE_GLO_BATCH_4X(DECODE_MATCH_FAST, return ZXC_ERROR_OVERFLOW);
        }
    }

    // Validate vbyte reads didn't overflow
    if (UNLIKELY(e_ptr > e_end)) return ZXC_ERROR_CORRUPT_DATA;

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe && l_ptr < l_end_safe_1x) {
        // Save pointers before reading (in case we need to fall back to Safe Path)
        const uint8_t* t_save = t_ptr;
        const uint8_t* o_save = o_ptr;
        const uint8_t* e_save = e_ptr;

        uint8_t token = *t_ptr++;
        uint64_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint64_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset = ZXC_LZ_OFFSET_BIAS;
        if (GLO_OFF8) {
            offset += *o_ptr++;  // 1-byte offset (biased)
        } else {
            offset += zxc_le16(o_ptr);  // 2-byte offset (biased)
            o_ptr += sizeof(uint16_t);
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) {
            ll += zxc_read_varint(&e_ptr, e_end);
            if (UNLIKELY(l_ptr + ll > l_end)) {
                t_ptr = t_save;
                o_ptr = o_save;
                e_ptr = e_save;
                break;
            }
        }
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_varint(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        // Check bounds before wild copies - if too close to end, fall back to Safe Path
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr))) {
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            break;
        }

        // Unlike the 4x batch the copy cannot sit in the varint branch's else -
        // the rollback above must run first - so re-test ll instead. Exact:
        // the escape leaves ll >= ZXC_TOKEN_LL_MASK.
        if (LIKELY(ll < ZXC_TOKEN_LL_MASK)) {
            zxc_copy16(d_ptr, l_ptr);
        } else {
            zxc_decode_copy_literals(d_ptr, l_ptr, ll);
        }
        l_ptr += ll;
        d_ptr += ll;

        if (UNLIKELY(d_ptr < d_bounds && (size_t)(d_ptr - d_floor) < offset))
            return ZXC_ERROR_BAD_OFFSET;

        // The loop entry check guarantees ll + ml + ZXC_PAD_SIZE bytes of
        // headroom, so the wild-copy ladder (incl. overlap/fill runs) is safe.
        zxc_decode_copy_match_glo(d_ptr, offset, ml);
        d_ptr += ml;
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint8_t token = *t_ptr++;
        uint64_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint64_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset = ZXC_LZ_OFFSET_BIAS;
        if (GLO_OFF8) {
            offset += *o_ptr++;  // 1-byte offset (biased)
        } else {
            offset += zxc_le16(o_ptr);  // 2-byte offset (biased)
            o_ptr += sizeof(uint16_t);
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) ll += zxc_read_varint(&e_ptr, e_end);
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_varint(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        if (UNLIKELY(ll + ml > (size_t)(d_end - d_ptr) || l_ptr + ll > l_end))
            return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        if (UNLIKELY((size_t)(d_ptr - d_floor) < offset)) return ZXC_ERROR_BAD_OFFSET;
        const uint8_t* match_src = d_ptr - offset;

        zxc_decode_copy_match_exact(d_ptr, match_src, offset, ml);
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    // Copy remaining literals from source stream (literal exhaustion)
    if (UNLIKELY(l_ptr > l_end)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(d_ptr > d_end)) return ZXC_ERROR_OVERFLOW;

    const size_t remaining_literals = (size_t)(l_end - l_ptr);
    if (UNLIKELY(remaining_literals > (size_t)(d_end - d_ptr))) return ZXC_ERROR_OVERFLOW;
    ZXC_MEMCPY(d_ptr, l_ptr, remaining_literals);
    d_ptr += remaining_literals;

    return (int)(d_ptr - dst);
}

/**
 * @brief Unified GHI (General High) block decoder body, shared by the fast, safe
 *        and dictionary variants.
 *
 * Decodes a block in the internal GHI format. The decompressed size is not on
 * the wire: it falls out of walking the sequences. @p safe and @p has_dict must be
 * compile-time constants (0 or 1): the 4x-unrolled loops are duplicated inside
 * @c if(safe)/else branches so each variant keeps single-assignment @c const
 * save pointers, and after constant propagation only one branch survives per
 * wrapper.
 *
 * @param[in,out] ctx          Decompression context (dict buffer, tables).
 * @param[in]     src          Compressed block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer for decoded bytes.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @param[in]     safe         Compile-time flag: 1 = strict bounds-checked loop.
 * @param[in]     has_dict     Compile-time flag: 1 = resolve matches against a dict prefix.
 * @return Bytes written to @p dst on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decode_block_ghi_impl(const zxc_cctx_t* RESTRICT ctx,
                                                       const uint8_t* RESTRICT src,
                                                       const size_t src_size, uint8_t* RESTRICT dst,
                                                       const size_t dst_capacity, const int safe,
                                                       const int has_dict) {
    zxc_gnr_header_t gh;

    // 0 when !has_dict (safe path) -> folds `d_floor` to `dst`.
    const size_t dict_size = has_dict ? ctx->dict_size : 0;

    if (UNLIKELY(zxc_read_ghi_header(src, src_size, &gh) != ZXC_OK)) return ZXC_ERROR_BAD_HEADER;
    if (UNLIKELY(gh.enc_lit != ZXC_SECTION_ENCODING_RAW || gh.enc_tok != 0))
        return ZXC_ERROR_CORRUPT_DATA;

    const uint8_t* const p_data = src + ZXC_GHI_HEADER_BINARY_SIZE;
    const uint8_t* p_curr = p_data;

    // --- Stream Pointers & Validation ---
    // GHI carries no section descriptors: literals are always RAW, the sequence
    // stream is four bytes per sequence, and extras run to the payload end.
    const size_t sz_lit = gh.n_literals;
    const uint64_t sz_seqs = (uint64_t)gh.n_sequences * sizeof(uint32_t);

    const size_t payload_avail = (size_t)(src + src_size - p_data);
    const uint64_t consumed = (uint64_t)sz_lit + sz_seqs;
    if (UNLIKELY(consumed > (uint64_t)payload_avail)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t sz_exts = payload_avail - (size_t)consumed; /* slack padding included */

    // GHI literals are always RAW, so the slack is always load-bearing here.
    // No underflow: sz_lit <= consumed <= payload_avail.
    if (UNLIKELY(payload_avail - sz_lit < ZXC_BLOCK_LIT_SLACK)) return ZXC_ERROR_CORRUPT_DATA;

    const uint8_t* l_ptr = p_curr;
    const uint8_t* l_end = l_ptr + sz_lit;
    p_curr += sz_lit;

    const uint8_t* seq_ptr = p_curr;
    const uint8_t* extras_ptr = p_curr + (size_t)sz_seqs;
    const uint8_t* const extras_end = extras_ptr + sz_exts;

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    // Lowest address a match may reach: the dictionary prefix if any, else dst.
    const uint8_t* const d_floor = dst - dict_size;
    const uint8_t* const d_end_safe = d_end - (ZXC_PAD_SIZE * 4);  // 128
    // Safety margin for 4x unrolled loop: 4 * (ZXC_SEQ_LL_MASK LL +
    // ZXC_SEQ_ML_MASK+ZXC_LZ_MIN_MATCH_LEN ML) + ZXC_PAD_SIZE Pad = 4 x (255 + 255 + 5) + 32 = 2092
    const uint8_t* const d_end_fast = d_end - ZXC_DECOMPRESS_TAIL_PAD;  // 2112

    // Literal margin for the GHI loops: without a varint, ll <= 254 per sequence,
    // so 4 * 254 = 1016. Past that only the cold varint path checks l_ptr.
    const size_t ghi_margin_4x = 4 * (ZXC_SEQ_LL_MASK - 1);  // 1016
    const size_t ghi_margin_1x = ZXC_SEQ_LL_MASK - 1;        // 254
    const uint8_t* const l_end_safe_4x = (sz_lit > ghi_margin_4x) ? l_end - ghi_margin_4x : l_ptr;
    const uint8_t* const l_end_safe_1x = (sz_lit > ghi_margin_1x) ? l_end - ghi_margin_1x : l_ptr;

    uint32_t n_seq = gh.n_sequences;

    // Same handover as GLO, but the GHI offset is inline in the sequence word and
    // always 16 bits -- enc_off is not a width here, so the span is fixed.
    const size_t off_span = 1U << 16;
    const size_t span = (dict_size >= off_span) ? 0 : off_span - dict_size;
    const uint8_t* const d_bounds = (span >= dst_capacity) ? d_end : dst + span;

    // --- SAFE loop: validate offsets until d_bounds (4x unroll) ---
    if (safe) {
        // SAFE variant: save per-batch state so an OVERFLOW can rollback and
        // hand over to the 1x loop / Safe Path. Wild writes already committed
        // are deterministically overwritten when the 1x loop replays.
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x && d_ptr < d_bounds) {
            const uint8_t* const t_save = seq_ptr;
            const uint8_t* const e_save = extras_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            DECODE_GHI_BATCH_4X(DECODE_MATCH_SAFE, goto rollback_safe_4x, (void)0);
            continue;

        rollback_safe_4x:
            seq_ptr = t_save;
            extras_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x && d_ptr < d_bounds) {
            DECODE_GHI_BATCH_4X(DECODE_MATCH_SAFE, return ZXC_ERROR_OVERFLOW, (void)0);
        }
    }

    // --- SAFE Loop tail: remaining sequences with offset validation (1x) ---
    while (n_seq > 0 && d_ptr < d_end_safe && d_ptr < d_bounds) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        // Strict bounds check: sequence must fit, AND wild copies must not overshoot
        // Check both destination (d_ptr) and source literal stream (l_ptr)
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr) ||
                     ll + ZXC_PAD_SIZE > (size_t)(l_end - l_ptr))) {
            // Fallback to exact copy (slow but safe)
            if (UNLIKELY(d_ptr + ll > d_end || l_ptr + ll > l_end)) return ZXC_ERROR_OVERFLOW;
            ZXC_MEMCPY(d_ptr, l_ptr, ll);
            l_ptr += ll;
            d_ptr += ll;

            if (UNLIKELY(d_ptr + ml > d_end)) return ZXC_ERROR_OVERFLOW;
            if (UNLIKELY((size_t)(d_ptr - d_floor) < offset)) return ZXC_ERROR_BAD_OFFSET;
            const uint8_t* match_src = d_ptr - offset;

            zxc_decode_copy_match_exact(d_ptr, match_src, offset, ml);
            d_ptr += ml;
        } else {
            zxc_decode_copy_literals(d_ptr, l_ptr, ll);
            l_ptr += ll;
            d_ptr += ll;
            DECODE_MATCH_SAFE(ml, offset, zxc_decode_copy_match);
        }
        n_seq--;
    }

    // --- FAST Loop: After threshold, check large margin to avoid individual bounds checks ---
    if (safe) {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x) {
            const uint8_t* const t_save = seq_ptr;
            const uint8_t* const e_save = extras_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            DECODE_GHI_BATCH_4X(DECODE_MATCH_FAST, goto rollback_fast_4x,
                                ZXC_PREFETCH_READ(l_ptr + ZXC_CACHE_LINE_SIZE));
            continue;

        rollback_fast_4x:
            seq_ptr = t_save;
            extras_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x) {
            DECODE_GHI_BATCH_4X(DECODE_MATCH_FAST, return ZXC_ERROR_OVERFLOW,
                                ZXC_PREFETCH_READ(l_ptr + ZXC_CACHE_LINE_SIZE));
        }
    }

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe && l_ptr < l_end_safe_1x) {
        // Save state for fallback
        const uint8_t* seq_save = seq_ptr;
        const uint8_t* ext_save = extras_ptr;

        const uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) {
            ll += zxc_read_varint(&extras_ptr, extras_end);
            if (UNLIKELY(l_ptr + ll > l_end)) {
                seq_ptr = seq_save;
                extras_ptr = ext_save;
                break;
            }
        }

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);

        // Strict bounds checks (including wild copy overrun safety)
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr) ||
                     ll + ZXC_PAD_SIZE > (size_t)(l_end - l_ptr))) {
            seq_ptr = seq_save;
            extras_ptr = ext_save;
            break;
        }
        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        zxc_decode_copy_literals(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        if (UNLIKELY(d_ptr < d_bounds && (size_t)(d_ptr - d_floor) < offset))
            return ZXC_ERROR_BAD_OFFSET;

        // The loop entry check guarantees ll + ml + ZXC_PAD_SIZE bytes of
        // headroom, so the wild-copy ladder (incl. overlap/fill runs) is safe.
        zxc_decode_copy_match(d_ptr, offset, ml);
        d_ptr += ml;
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);
        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        if (UNLIKELY(ll + ml > (size_t)(d_end - d_ptr) || l_ptr + ll > l_end))
            return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        if (UNLIKELY((size_t)(d_ptr - d_floor) < offset)) return ZXC_ERROR_BAD_OFFSET;
        const uint8_t* match_src = d_ptr - offset;

        zxc_decode_copy_match_exact(d_ptr, match_src, offset, ml);
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    // Copy remaining literals from source stream (literal exhaustion)
    if (UNLIKELY(l_ptr > l_end)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(d_ptr > d_end)) return ZXC_ERROR_OVERFLOW;

    const size_t remaining_literals = (size_t)(l_end - l_ptr);
    if (UNLIKELY(remaining_literals > (size_t)(d_end - d_ptr))) return ZXC_ERROR_OVERFLOW;
    ZXC_MEMCPY(d_ptr, l_ptr, remaining_literals);
    d_ptr += remaining_literals;

    return (int)(d_ptr - dst);
}

/**
 * Cold specializations for entropy-coded GLO tokens (level 7). Each wrapper fixes
 * tok_entropy=1 plus one (safe, has_dict) combo as compile-time constants, so the
 * inlined impl is fully specialized and the RAW-token fast path stays untouched.
 */
static ZXC_NOINLINE int zxc_decode_block_glo_entropy(const zxc_cctx_t* RESTRICT ctx,
                                                     const uint8_t* RESTRICT src,
                                                     const size_t src_size, uint8_t* RESTRICT dst,
                                                     const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 0, 1);
}

static ZXC_NOINLINE int zxc_decode_block_glo_entropy_dict(const zxc_cctx_t* RESTRICT ctx,
                                                          const uint8_t* RESTRICT src,
                                                          const size_t src_size,
                                                          uint8_t* RESTRICT dst,
                                                          const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 1, 1);
}

static ZXC_NOINLINE int zxc_decode_block_glo_entropy_safe(const zxc_cctx_t* RESTRICT ctx,
                                                          const uint8_t* RESTRICT src,
                                                          const size_t src_size,
                                                          uint8_t* RESTRICT dst,
                                                          const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 1, 0, 1);
}

/**
 * @brief Decode a no-dict GLO block (plain, inlinable path).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=0, @c has_dict=0, so
 * the no-dict chunk wrapper inlines it exactly like the dict-free build.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static int zxc_decode_block_glo(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_size, uint8_t* RESTRICT dst,
                                const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 0, 0);
}

/**
 * @brief Decode a no-dict GHI block (plain, inlinable path).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=0, @c has_dict=0.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static int zxc_decode_block_ghi(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_size, uint8_t* RESTRICT dst,
                                const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 0, 0);
}

/**
 * @brief Decode a GLO block against a dictionary prefix (cold path).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=0, @c has_dict=1.
 * NOINLINE: only reached on the cold dict path (@ref zxc_decompress_chunk_wrapper_dict),
 * so it never loads into I-cache on a no-dict stream.
 *
 * @param[in,out] ctx          Decompression context (dict prefix in its buffer).
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_glo_dict(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 1, 0);
}

/**
 * @brief Decode a GHI block against a dictionary prefix (cold path).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=0, @c has_dict=1
 * (NOINLINE; see @ref zxc_decode_block_glo_dict).
 *
 * @param[in,out] ctx          Decompression context (dict prefix in its buffer).
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_ghi_dict(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 0, 1);
}

/**
 * @brief Decode a GLO block with the strict-tail safe loop (no wild copies).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=1, @c has_dict=0.
 * The safe path never carries a dict (block_safe routes dict inputs to the
 * bounce path), so @c has_dict=0 folds the dead dict handling.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_glo_safe(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 1, 0, 0);
}

/**
 * @brief Decode a GHI block with the strict-tail safe loop (no wild copies).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=1, @c has_dict=0
 * (the strict-tail safe path never carries a dict; see
 * @ref zxc_decode_block_glo_safe).
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_ghi_safe(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 1, 0);
}

#undef DECODE_GHI_BATCH_4X
#undef DECODE_GHI_SEQ
#undef DECODE_GLO_BATCH_4X
#undef DECODE_GLO_SEQ
#undef DECODE_MATCH_FAST
#undef GLO_OFF8
#undef DECODE_MATCH_SAFE

/**
 * @brief Shared chunk-decode body: validates the block header, verifies the
 *        optional checksum, then dispatches on block type.
 *
 * @p has_dict and @p safe are compile-time constants: the no-dict instantiation
 * folds the GLO/GHI selection to the plain (inlinable) decoders, so
 * @ref zxc_decompress_chunk_wrapper carries no dict code and matches the
 * dict-free build; the dict and safe instantiations call the NOINLINE @c _dict
 * / @c _safe decoders (the strict-tail safe path never carries a dict).
 *
 * @param[in,out] ctx       Decompression context.
 * @param[in]     src       Compressed block (header + payload + optional checksum).
 * @param[in]     src_sz    Size of @p src in bytes.
 * @param[out]    dst       Destination buffer for the decoded block.
 * @param[in]     dst_cap   Capacity of @p dst in bytes.
 * @param[in]     has_dict  Compile-time flag: 1 = dictionary-aware decoders.
 * @param[in]     safe      Compile-time flag: 1 = strict-tail safe decoders.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decompress_chunk_wrapper_body(
    const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src, const size_t src_sz,
    uint8_t* RESTRICT dst, const size_t dst_cap, const int has_dict, const int safe) {
    if (UNLIKELY(src_sz < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t type = src[0];
    const uint32_t comp_sz = zxc_le32(src + 3);
    const int has_checksum = ctx->checksum_enabled;

    // Check bounds: Header + Body + Checksum(if any)
    const size_t expected_sz =
        (size_t)ZXC_BLOCK_HEADER_SIZE + comp_sz + (has_checksum ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    if (UNLIKELY(src_sz < expected_sz)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t* data = src + ZXC_BLOCK_HEADER_SIZE;

    if (has_checksum) {
        const uint32_t stored = zxc_le32(data + comp_sz);
        const uint32_t calc = zxc_checksum(data, comp_sz, ZXC_CHECKSUM_RAPIDHASH);
        if (UNLIKELY(stored != calc)) return ZXC_ERROR_BAD_CHECKSUM;
    }

    int decoded_sz = ZXC_ERROR_BAD_BLOCK_TYPE;

    switch (type) {
        case ZXC_BLOCK_GLO:
            decoded_sz = safe       ? zxc_decode_block_glo_safe(ctx, data, comp_sz, dst, dst_cap)
                         : has_dict ? zxc_decode_block_glo_dict(ctx, data, comp_sz, dst, dst_cap)
                                    : zxc_decode_block_glo(ctx, data, comp_sz, dst, dst_cap);
            break;
        case ZXC_BLOCK_GHI:
            decoded_sz = safe       ? zxc_decode_block_ghi_safe(ctx, data, comp_sz, dst, dst_cap)
                         : has_dict ? zxc_decode_block_ghi_dict(ctx, data, comp_sz, dst, dst_cap)
                                    : zxc_decode_block_ghi(ctx, data, comp_sz, dst, dst_cap);
            break;
        case ZXC_BLOCK_RAW:
            // For RAW blocks, comp_sz == raw_sz (uncompressed data stored as-is)
            if (UNLIKELY(comp_sz > dst_cap)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, data, comp_sz);
            decoded_sz = (int)comp_sz;
            break;
        case ZXC_BLOCK_EOF:
            // EOF should be handled by the dispatcher, not here
            return ZXC_ERROR_CORRUPT_DATA;
        default:
            return ZXC_ERROR_BAD_BLOCK_TYPE;
    }

    return decoded_sz;
}

/**
 * @brief Public no-dict chunk decoder (decompression hot path).
 *
 * Inlines the plain GLO/GHI decoders via @ref zxc_decompress_chunk_wrapper_body
 * with @c has_dict=0, so it carries no dict code and matches the dict-free build.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
    return zxc_decompress_chunk_wrapper_body(ctx, src, src_sz, dst, dst_cap, 0, 0);
}

/**
 * @brief Public dictionary chunk decoder.
 *
 * Routes through @ref zxc_decompress_chunk_wrapper_body with @c has_dict=1,
 * which calls the NOINLINE @c _dict decoders (slower: dict back-refs read the
 * prepended dictionary). Used only when @c ctx->dict_size != 0.
 *
 * @param[in,out] ctx     Decompression context (dict prefix in its buffer).
 * @param[in]     src     Compressed block bytes.
 * @param[in]     src_sz  Size of @p src in bytes.
 * @param[out]    dst     Destination buffer for the decoded block.
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper_dict(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    return zxc_decompress_chunk_wrapper_body(ctx, src, src_sz, dst, dst_cap, 1, 0);
}

/**
 * @brief Public strict-tail safe chunk decoder (dst_cap == exact decoded size).
 *
 * Routes through @ref zxc_decompress_chunk_wrapper_body with @c safe=1, which
 * calls the NOINLINE @c _safe decoders (no bounce buffer, no tail padding);
 * RAW blocks are copied directly. Dict inputs are not handled here (the
 * caller routes them to the bounce-capable path).
 *
 * @param[in,out] ctx     Decompression context.
 * @param[in]     src     Compressed block bytes.
 * @param[in]     src_sz  Size of @p src in bytes.
 * @param[out]    dst     Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper_safe(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    return zxc_decompress_chunk_wrapper_body(ctx, src, src_sz, dst, dst_cap, 0, 1);
}
