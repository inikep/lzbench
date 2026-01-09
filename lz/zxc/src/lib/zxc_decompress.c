/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../../include/zxc_sans_io.h"
#include "zxc_internal.h"

/*
 * Function Multi-Versioning Support
 * If ZXC_FUNCTION_SUFFIX is defined (e.g. _avx2), rename the public entry point.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_decompress_chunk_wrapper ZXC_CAT(zxc_decompress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#endif

#define ZXC_DEC_BATCH 32  // Number of sequences to decode in a batch

/**
 * @brief Consumes a specified number of bits from the bit reader buffer without
 * performing safety checks.
 *
 * This function advances the bit reader's state by `n` bits. It is marked as
 * always inline for performance critical paths.
 *
 * @warning This is a "fast" variant, meaning it assumes the buffer has enough
 * bits available. The caller is responsible for ensuring that at least `n` bits
 * are present in the accumulator or buffer before calling this function to
 * avoid undefined behavior or reading past valid memory.
 *
 * @param[in,out] br Pointer to the bit reader instance.
 * @param[in] n The number of bits to consume (must be <= 32, typically <= 24
 * depending on implementation).
 * @return The value of the consumed bits as a 32-bit unsigned integer.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_br_consume_fast(zxc_bit_reader_t* br, uint8_t n) {
#if defined(__BMI2__) && defined(__x86_64__)
    // BMI2 Optimization: _bzhi_u64(x, n) copies the lower n bits of x to dst and
    // clears the rest. It is equivalent to x & ((1ULL << n) - 1) but executes in
    // a single cycle without dependency chains.
    uint32_t val = (uint32_t)_bzhi_u64(br->accum, n);
#else
    uint32_t val = (uint32_t)(br->accum & ((1ULL << n) - 1));
#endif
    br->accum >>= n;
    br->bits -= n;
    return val;
}

/**
 * @brief Reads a variable-length byte (VByte) encoded integer from a stream.
 *
 * This function decodes a 32-bit unsigned integer encoded in VByte format from
 * the provided byte stream. VByte encoding uses one or more bytes, where each byte
 * has its most significant bit (MSB) set to 1 if there are more bytes to read,
 * and 0 if it is the last byte. The remaining 7 bits of each byte contribute
 * to the integer value.
 *
 * @param[in,out] ptr Pointer to a pointer to the current position in the stream.
 * @param[in] end Pointer to the end of the readable stream (for bounds checking).
 * @return The decoded 32-bit integer, or 0 if reading would overflow bounds (safe default).
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_read_vbyte(const uint8_t** ptr, const uint8_t* end) {
    const uint8_t* p = *ptr;
    // Bounds check: need at least 1 byte
    if (UNLIKELY(p >= end)) return 0;  // Safe default: prevents crash, detected later

    uint32_t b0 = p[0];
    if (LIKELY(b0 < ZXC_VBYTE_MSB)) {
        *ptr = p + 1;
        return b0;
    }

    // 2-byte path (second most common)
    if (UNLIKELY(p + 1 >= end)) {
        *ptr = p + 1;
        return 0;
    }
    uint32_t b1 = p[1];
    if (LIKELY(b1 < ZXC_VBYTE_MSB)) {
        *ptr = p + 2;
        return (b0 & ZXC_VBYTE_MASK) | (b1 << 7);
    }

    // 3-byte path (last possible for ZXC_BLOCK_SIZE = 256KB)
    if (UNLIKELY(p + 2 >= end)) {
        *ptr = p + 2;
        return 0;
    }
    uint32_t b2 = p[2];
    uint32_t val = (b0 & ZXC_VBYTE_MASK) | ((b1 & ZXC_VBYTE_MASK) << 7);
    if (b2 < ZXC_VBYTE_MSB) {
        *ptr = p + 3;
        return val | (b2 << 14);
    }

    if (UNLIKELY(p + 3 >= end)) {
        *ptr = p + 3;
        return 0;
    }
    val |= (b2 & ZXC_VBYTE_MASK) << 14;
    uint32_t b3 = p[3];
    if (b3 < ZXC_VBYTE_MSB) {
        *ptr = p + 4;
        return val | (b3 << 21);
    }

    if (UNLIKELY(p + 4 >= end)) {
        *ptr = p + 4;
        return 0;
    }
    *ptr = p + 5;
    // 5th byte: only 4 bits used (32 - 28 = 4). Mask 0x0F for robustness against corrupted data.
    return val | ((b3 & ZXC_VBYTE_MASK) << 21) | (((uint32_t)p[4] & 0x0F) << 28);
}

/**
 * @brief Shuffle masks for overlapping copies with small offsets (0-15).
 *
 * Shared between ARM NEON and x86 SSSE3. Each row defines how to replicate
 * source bytes to fill 16 bytes when offset < 16.
 */
#if defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32) || defined(ZXC_USE_AVX2) || \
    defined(ZXC_USE_AVX512)
/**
 * @brief Precomputed masks for handling overlapping data during decompression.
 *
 * This 16x16 lookup table contains 128-bit aligned masks used to efficiently
 * mask off or combine bytes when processing overlapping copy operations or
 * boundary conditions in the ZXC decompression algorithm.
 *
 * The alignment to 16 bytes ensures compatibility with SIMD instructions
 * (like SSE/AVX) for optimized memory operations.
 */
static const ZXC_ALIGN(16) uint8_t zxc_overlap_masks[16][16] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // off=0 (unused)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // off=1 (RLE handled separately)
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},      // off=2
    {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},      // off=3
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},      // off=4
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0},      // off=5
    {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3},      // off=6
    {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1},      // off=7
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},      // off=8
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6},      // off=9
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},      // off=10
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4},     // off=11
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3},    // off=12
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2},   // off=13
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1},  // off=14
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0}  // off=15
};
#endif

/**
 * @brief Copies 16 bytes from an overlapping source to the destination.
 *
 * This function is designed to handle memory copies where the source and
 * destination regions might overlap, specifically copying 16 bytes from
 * `dst - off` to `dst`. It is typically used in decompression routines
 * (like LZ77) where repeating a previous sequence is required.
 *
 * Handles NEON64, NEON32, SSSE3/AVX2 and generic scalar fallback.
 *
 * @param[out] dst Pointer to the destination buffer where bytes will be written.
 * @param[in]  off The offset backwards from the destination pointer to read from.
 *                 (i.e., source address is `dst - off`).
 */
// codeql[cpp/unused-static-function] : False positive, used in DECODE_SEQ_SAFE/FAST macros
static ZXC_ALWAYS_INLINE void zxc_copy_overlap16(uint8_t* dst, uint32_t off) {
    // If off==0 (invalid), we force off=1 to mimic a safe RLE of the previous byte.
    // This prevents the division-by-zero crash in the scalar fallback (i % off)
    // and prevents reading uninitialized memory (dst[0] = dst[-0]), maintaining
    // memory safety even with corrupt data.
    off = off ? off : 1;
#if defined(ZXC_USE_NEON64)
    uint8x16_t mask = vld1q_u8(zxc_overlap_masks[off]);
    uint8x16_t src_data = vld1q_u8(dst - off);
    vst1q_u8(dst, vqtbl1q_u8(src_data, mask));

#elif defined(ZXC_USE_NEON32)
    uint8x8x2_t src_tbl;
    src_tbl.val[0] = vld1_u8(dst - off);
    src_tbl.val[1] = vld1_u8(dst - off + 8);
    uint8x8_t mask_lo = vld1_u8(zxc_overlap_masks[off]);
    uint8x8_t mask_hi = vld1_u8(zxc_overlap_masks[off] + 8);
    vst1_u8(dst, vtbl2_u8(src_tbl, mask_lo));
    vst1_u8(dst + 8, vtbl2_u8(src_tbl, mask_hi));

#elif defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    __m128i mask = _mm_load_si128((const __m128i*)zxc_overlap_masks[off]);
    __m128i src_data = _mm_loadu_si128((const __m128i*)(dst - off));
    _mm_storeu_si128((__m128i*)dst, _mm_shuffle_epi8(src_data, mask));

#else
    const uint8_t* src = dst - off;
    for (size_t i = 0; i < 16; i++) {
        dst[i] = src[i % off];
    }
#endif
}



#if defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
/**
 * @brief Computes the prefix sum of a 128-bit vector of 32-bit unsigned
 * integers using NEON intrinsics.
 *
 * This function calculates the running total of the elements in the input
 * vector `v`. If the input vector is `[a, b, c, d]`, the result will be `[a,
 * a+b, a+b+c, a+b+c+d]`. This operation is typically used for calculating
 * cumulative distributions or offsets in parallel.
 *
 * @param[in] v The input vector containing four 32-bit unsigned integers.
 * @return A uint32x4_t vector containing the prefix sums.
 */
static ZXC_ALWAYS_INLINE uint32x4_t zxc_neon_prefix_sum_u32(uint32x4_t v) {
    uint32x4_t zero = vdupq_n_u32(0);  // Create a vector of zeros

    // Rotate right by 1 element (shift 4 bytes)
    uint32x4_t s1 =
        vreinterpretq_u32_u8(vextq_u8(vreinterpretq_u8_u32(zero), vreinterpretq_u8_u32(v), 12));
    v = vaddq_u32(v, s1);  // Add shifted version: [a, b, c, d] + [0, a, b, c] ->
                           // [a, a+b, b+c, c+d]

    // Rotate right by 2 elements (shift 8 bytes)
    uint32x4_t s2 =
        vreinterpretq_u32_u8(vextq_u8(vreinterpretq_u8_u32(zero), vreinterpretq_u8_u32(v), 8));
    v = vaddq_u32(v, s2);  // Add shifted version to complete prefix sum

    return v;
}
#endif

#if defined(ZXC_USE_AVX2)
/**
 * @brief Computes the prefix sum of 32-bit integers within a 256-bit vector.
 *
 * This function calculates the cumulative sum of the eight 32-bit integers
 * contained in the input vector `v`.
 *
 * Operation logic (conceptually):
 *   out[0] = v[0]
 *   out[1] = v[0] + v[1]
 *   ...
 *   out[7] = v[0] + v[1] + ... + v[7]
 *
 * @param[in] v The input 256-bit vector containing eight 32-bit integers.
 * @return A 256-bit vector containing the prefix sums of the input elements.
 */
// codeql[cpp/unused-static-function] : Used conditionally when ZXC_USE_AVX2 is defined
static ZXC_ALWAYS_INLINE __m256i zxc_mm256_prefix_sum_epi32(__m256i v) {
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));  // Add value shifted by 1 element
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));  // Add value shifted by 2 elements

    // Use permute/shuffle to bridge the 128-bit lane gap
    __m256i v_bridge = _mm256_permute2x128_si256(v, v, 0x00);  // Duplicate lower 128 to upper
    v_bridge = _mm256_shuffle_epi32(v_bridge,
                                    0xFF);  // Broadcast last element of lower 128
    v_bridge = _mm256_blend_epi32(_mm256_setzero_si256(), v_bridge,
                                  0xF0);  // Only apply to upper lane

    return _mm256_add_epi32(v, v_bridge);  // Add bridge value to upper lane
}
#endif

#if defined(ZXC_USE_AVX512)
/**
 * @brief Computes the prefix sum of 32-bit integers within a 512-bit vector.
 *
 * This function calculates the running sum of the 16 packed 32-bit integers
 * in the input vector `v`.
 *
 * For an input vector v = [x0, x1, x2, ... x15], the result will be:
 * [x0, x0+x1, x0+x1+x2, ... , sum(x0..x15)].
 *
 * @note This function is forced inline for performance reasons.
 *
 * @param[in] v The input 512-bit vector containing sixteen 32-bit integers.
 * @return A 512-bit vector containing the prefix sums of the input elements.
 */
static ZXC_ALWAYS_INLINE __m512i zxc_mm512_prefix_sum_epi32(__m512i v) {
    __m512i t = _mm512_bslli_epi128(v, 4);  // Shift left by 4 bytes (1 int)
    v = _mm512_add_epi32(v, t);             // Add shifted value
    t = _mm512_bslli_epi128(v, 8);          // Shift left by 8 bytes (2 ints)
    v = _mm512_add_epi32(v, t);             // Add shifted value

    // Propagate sums across 128-bit lanes (sequential dependency)
    __m512i v_l0 = _mm512_shuffle_i32x4(v, v, 0x00);  // Broadcast lane 0
    v_l0 = _mm512_shuffle_epi32(v_l0, 0xFF);          // Broadcast last element of lane 0
    v = _mm512_mask_add_epi32(v, 0x00F0, v, v_l0);    // Add to lane 1 only

    __m512i v_l1 = _mm512_shuffle_i32x4(v, v, 0x55);  // Broadcast lane 1
    v_l1 = _mm512_shuffle_epi32(v_l1, 0xFF);          // Broadcast last element of lane 1
    v = _mm512_mask_add_epi32(v, 0x0F00, v, v_l1);    // Add to lane 2 only

    __m512i v_l2 = _mm512_shuffle_i32x4(v, v, 0xAA);  // Broadcast lane 2
    v_l2 = _mm512_shuffle_epi32(v_l2, 0xFF);          // Broadcast last element of lane 2
    v = _mm512_mask_add_epi32(v, 0xF000, v, v_l2);    // Add to lane 3 only

    return v;
}
#endif

/**
 * @brief Decodes a block of numerical data compressed with the ZXC format.
 *
 * This function reads a compressed numerical block from the source buffer,
 * parses the header to determine the number of values and encoding parameters,
 * and then decompresses the data into the destination buffer.
 *
 * **Algorithm Details:**
 * 1. **Header Parsing:** Reads the `zxc_num_header_t` to get the count of
 * values.
 * 2. **Bit Unpacking:** For each chunk of values, it initializes a bit reader.
 *    - **Unrolling:** The main loop is unrolled 4x to minimize branch overhead
 *      and maximize instruction throughput.
 * 3. **ZigZag Decoding:** Converts the unsigned unpacked value back to a signed
 * delta using `(n >> 1) ^ -(n & 1)`.
 * 4. **Delta Reconstruction:** Adds the signed delta to a `running_val`
 * accumulator to recover the original integer sequence.
 *
 * @param[in] src Pointer to the source buffer containing compressed data.
 * @param[in] src_size Size of the source buffer in bytes.
 * @param[out] dst Pointer to the destination buffer where decompressed data will be
 * written.
 * @param[in] dst_capacity Maximum capacity of the destination buffer in bytes.
 * @param[in] expected_raw_size Expected size of the uncompressed data (unused in
 * current implementation).
 *
 * @return The number of bytes written to the destination buffer on success,
 *         or -1 if an error occurs (e.g., buffer overflow, invalid header,
 *         or malformed compressed stream).
 */
static int zxc_decode_block_num(const uint8_t* RESTRICT src, size_t src_size, uint8_t* RESTRICT dst,
                                size_t dst_capacity, uint32_t expected_raw_size) {
    (void)expected_raw_size;

    zxc_num_header_t nh;
    if (UNLIKELY(zxc_read_num_header(src, src_size, &nh) != 0)) return -1;

    const uint8_t* p = src + ZXC_NUM_HEADER_BINARY_SIZE;
    const uint8_t* p_end = src + src_size;
    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    uint64_t vals_remaining = nh.n_values;
    uint32_t running_val = 0;

    ZXC_ALIGN(ZXC_CACHE_LINE_SIZE)
    uint32_t deltas[ZXC_DEC_BATCH];

    while (vals_remaining > 0) {
        if (UNLIKELY(p + 16 > p_end)) return -1;
        uint16_t nvals = zxc_le16(p + 0);
        uint16_t bits = zxc_le16(p + 2);
        uint32_t psize = zxc_le32(p + 12);
        p += 16;
        if (UNLIKELY(p + psize > p_end || d_ptr + nvals * 4 > d_end ||
                     bits > (sizeof(uint32_t) * ZXC_BITS_PER_BYTE)))
            return -1;

        zxc_bit_reader_t br;
        zxc_br_init(&br, p, psize);
        size_t i = 0;

        for (; i + ZXC_DEC_BATCH <= nvals; i += ZXC_DEC_BATCH) {
            for (int k = 0; k < ZXC_DEC_BATCH; k += 4) {
                zxc_br_ensure(&br, bits);
                deltas[k + 0] = zxc_zigzag_decode(zxc_br_consume_fast(&br, (uint8_t)bits));
                zxc_br_ensure(&br, bits);
                deltas[k + 1] = zxc_zigzag_decode(zxc_br_consume_fast(&br, (uint8_t)bits));
                zxc_br_ensure(&br, bits);
                deltas[k + 2] = zxc_zigzag_decode(zxc_br_consume_fast(&br, (uint8_t)bits));
                zxc_br_ensure(&br, bits);
                deltas[k + 3] = zxc_zigzag_decode(zxc_br_consume_fast(&br, (uint8_t)bits));
            }

            uint32_t* batch_dst = (uint32_t*)d_ptr;

#if defined(ZXC_USE_AVX512)
            for (int k = 0; k < ZXC_DEC_BATCH; k += 16) {
                __m512i v_deltas = _mm512_load_si512((void*)&deltas[k]);  // Load 16 deltas
                __m512i v_run = _mm512_set1_epi32(running_val);  // Broadcast current running total

                __m512i v_sum = zxc_mm512_prefix_sum_epi32(v_deltas);  // Compute local prefix sums
                v_sum = _mm512_add_epi32(v_sum, v_run);                // Add base running total

                _mm512_storeu_si512((void*)&batch_dst[k],
                                    v_sum);  // Store decoded values

                // Extract the last value (15th element) to update running_val for next
                // batch
                __m128i v_last128 = _mm512_extracti32x4_epi32(v_sum, 3);
                running_val = (uint32_t)_mm_cvtsi128_si32(_mm_shuffle_epi32(v_last128, 0xFF));
            }

#elif defined(ZXC_USE_AVX2)
            for (int k = 0; k < ZXC_DEC_BATCH; k += 8) {
                __m256i v_deltas = _mm256_load_si256((const __m256i*)&deltas[k]);  // Load 8 deltas
                __m256i v_run = _mm256_set1_epi32(running_val);  // Broadcast running total

                __m256i v_sum = zxc_mm256_prefix_sum_epi32(v_deltas);  // Compute local prefix sums
                v_sum = _mm256_add_epi32(v_sum, v_run);                // Add base

                _mm256_storeu_si256((__m256i*)&batch_dst[k],
                                    v_sum);                   // Store decoded values
                running_val = ((uint32_t*)&batch_dst[k])[7];  // Update running_val
            }

#elif defined(ZXC_USE_NEON64)
            uint32x4_t v_run = vdupq_n_u32(running_val);  // Broadcast running total
            for (int k = 0; k < ZXC_DEC_BATCH; k += 4) {
                uint32x4_t v_deltas = vld1q_u32(&deltas[k]);  // Load 4 deltas

                uint32x4_t v_sum = zxc_neon_prefix_sum_u32(v_deltas);  // Compute local prefix sums
                v_sum = vaddq_u32(v_sum, v_run);                       // Add base

                vst1q_u32(&batch_dst[k], v_sum);  // Store decoded values

                running_val = vgetq_lane_u32(v_sum, 3);  // Extract last element
                v_run = vdupq_n_u32(running_val);        // Update vector for next iter
            }

#elif defined(ZXC_USE_NEON32)
            uint32x4_t v_run = vdupq_n_u32(running_val);
            for (int k = 0; k < ZXC_DEC_BATCH; k += 4) {
                uint32x4_t v_deltas = vld1q_u32(&deltas[k]);

                uint32x4_t v_sum = zxc_neon_prefix_sum_u32(v_deltas);
                v_sum = vaddq_u32(v_sum, v_run);

                vst1q_u32(&batch_dst[k], v_sum);

                running_val = vgetq_lane_u32(v_sum, 3);
                v_run = vdupq_n_u32(running_val);
            }

#else
            for (int k = 0; k < ZXC_DEC_BATCH; k++) {
                running_val += deltas[k];
                batch_dst[k] = running_val;
            }
#endif
            d_ptr += ZXC_DEC_BATCH * 4;
        }

        for (; i < nvals; i++) {
            zxc_br_ensure(&br, bits);
            uint32_t delta = zxc_zigzag_decode(zxc_br_consume_fast(&br, (uint8_t)bits));
            running_val += delta;
            zxc_store_le32(d_ptr, running_val);
            d_ptr += 4;
        }

        p += psize;
        vals_remaining -= nvals;
    }
    return (int)(d_ptr - dst);
}

/**
 * @brief Decompresses a "GLO" (General) encoded block of data.
 *
 * This function handles the decoding of a compressed block formatted with the
 * internal GLO structure.
 *
 * @param[in,out] ctx Pointer to the compression context (`zxc_cctx_t`) containing
 * @param[in] src Pointer to the source buffer containing compressed data.
 * @param[in] src_size Size of the source buffer in bytes.
 * @param[out] dst Pointer to the destination buffer for decompressed data.
 * @param[in] dst_capacity Maximum capacity of the destination buffer.
 * @param[in] expected_raw_size The expected size of the decompressed data (used for
 * validation and trailing literals).
 *
 * @return The number of bytes written to the destination buffer on success, or
 * -1 on failure (e.g., invalid header, buffer overflow, or corrupted data).
 */
static int zxc_decode_block_glo(zxc_cctx_t* ctx, const uint8_t* RESTRICT src, size_t src_size,
                                uint8_t* RESTRICT dst, size_t dst_capacity,
                                uint32_t expected_raw_size) {
    zxc_gnr_header_t gh;
    zxc_section_desc_t desc[ZXC_GLO_SECTIONS];

    int res = zxc_read_glo_header_and_desc(src, src_size, &gh, desc);
    if (UNLIKELY(res != 0)) return -1;

    const uint8_t* p_data =
        src + ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;
    const uint8_t* p_curr = p_data;

    // --- Literal Stream Setup ---
    const uint8_t* l_ptr;
    const uint8_t* l_end;
    uint8_t* rle_buf = NULL;

    size_t lit_stream_size = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);

    if (gh.enc_lit == 1) {
        size_t required_size = (size_t)(desc[0].sizes >> 32);

        if (required_size > 0) {
            if (UNLIKELY(required_size > dst_capacity)) return -1;

            if (ctx->lit_buffer_cap < required_size + ZXC_PAD_SIZE) {
                uint8_t* new_buf = (uint8_t*)realloc(ctx->lit_buffer, required_size + ZXC_PAD_SIZE);
                if (UNLIKELY(!new_buf)) {
                    free(ctx->lit_buffer);
                    ctx->lit_buffer = NULL;
                    ctx->lit_buffer_cap = 0;
                    return -1;
                }
                ctx->lit_buffer = new_buf;
                ctx->lit_buffer_cap = required_size + ZXC_PAD_SIZE;
            }

            rle_buf = ctx->lit_buffer;
            if (UNLIKELY(!rle_buf || lit_stream_size > (size_t)(src + src_size - p_curr)))
                return -1;

            const uint8_t* r_ptr = p_curr;
            const uint8_t* r_end = r_ptr + lit_stream_size;
            uint8_t* w_ptr = rle_buf;
            const uint8_t* const w_end = rle_buf + required_size;

            while (r_ptr < r_end && w_ptr < w_end) {
                uint8_t token = *r_ptr++;
                if (LIKELY(!(token & ZXC_LIT_RLE_FLAG))) {
                    // Raw copy (most common path): use ZXC_PAD_SIZE-byte wild copies
                    // token is 7-bit (0-127), so len is 1-128 bytes
                    uint32_t len = (uint32_t)token + 1;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr + len > r_end)) return -1;

                    // Destination has ZXC_PAD_SIZE bytes of safe overrun space.
                    // Source may not - check before wild copy.
                    // Fast path: source has ZXC_PAD_SIZE-byte read headroom (most common)
                    if (LIKELY(r_ptr + ZXC_PAD_SIZE <= r_end)) {
                        // Single 32-byte copy covers len <= ZXC_PAD_SIZE (most tokens)
                        zxc_copy32(w_ptr, r_ptr);

                        if (UNLIKELY(len > ZXC_PAD_SIZE)) {
                            // Unroll: max len=128, so max 4 copies total
                            // Use unconditional stores with overlap - faster than branches
                            if (len <= 2 * ZXC_PAD_SIZE) {
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
                            } else if (len <= 3 * ZXC_PAD_SIZE) {
                                zxc_copy32(w_ptr + ZXC_PAD_SIZE, r_ptr + ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
                            } else {
                                zxc_copy32(w_ptr + ZXC_PAD_SIZE, r_ptr + ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + 2 * ZXC_PAD_SIZE, r_ptr + 2 * ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
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
                    uint32_t len = (token & ZXC_LIT_LEN_MASK) + 4;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr >= r_end)) return -1;
                    ZXC_MEMSET(w_ptr, *r_ptr++, len);
                    w_ptr += len;
                }
            }
            if (UNLIKELY(w_ptr != w_end)) return -1;
            l_ptr = rle_buf;
            l_end = rle_buf + required_size;
        } else {
            l_ptr = p_curr;
            l_end = p_curr;
        }
    } else {
        l_ptr = p_curr;
        l_end = p_curr + lit_stream_size;
    }

    p_curr += lit_stream_size;

    // --- Stream Pointers & Validation ---
    size_t sz_tokens = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    size_t sz_offsets = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);
    size_t sz_extras = (size_t)(desc[3].sizes & ZXC_SECTION_SIZE_MASK);

    // Validate stream sizes match sequence count (early rejection of malformed data)
    size_t expected_off_size =
        (gh.enc_off == 1) ? (size_t)gh.n_sequences : (size_t)gh.n_sequences * 2;

    if (UNLIKELY(sz_tokens < gh.n_sequences || sz_offsets < expected_off_size)) return -1;

    const uint8_t* t_ptr = p_curr;
    const uint8_t* o_ptr = t_ptr + sz_tokens;
    const uint8_t* e_ptr = o_ptr + sz_offsets;
    const uint8_t* const e_end = e_ptr + sz_extras;  // For vbyte overflow detection

    // Validate streams don't overflow source buffer
    if (UNLIKELY(e_end != src + src_size)) return -1;

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    const uint8_t* const d_end_safe = d_end - 128;

    uint32_t n_seq = gh.n_sequences;

    // Track bytes written for offset validation
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written (max 8-bit offset)
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written (max 16-bit offset)
    // After threshold, all offsets are guaranteed valid (can't exceed written bytes)
    size_t written = 0;

// Macro for copy literal + match (uses 32-byte wild copies)
// SAFE version: validates offset against written bytes
#define DECODE_SEQ_SAFE(ll, ml, off)                                     \
    do {                                                                 \
        {                                                                \
            const uint8_t* src_lit = l_ptr;                              \
            uint8_t* dst_lit = d_ptr;                                    \
            zxc_copy32(dst_lit, src_lit);                                \
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {                           \
                dst_lit += ZXC_PAD_SIZE;                                 \
                src_lit += ZXC_PAD_SIZE;                                 \
                size_t rem = ll - ZXC_PAD_SIZE;                          \
                while (rem > ZXC_PAD_SIZE) {                             \
                    zxc_copy32(dst_lit, src_lit);                        \
                    dst_lit += ZXC_PAD_SIZE;                             \
                    src_lit += ZXC_PAD_SIZE;                             \
                    rem -= ZXC_PAD_SIZE;                                 \
                }                                                        \
                zxc_copy32(dst_lit, src_lit);                            \
            }                                                            \
            l_ptr += ll;                                                 \
            d_ptr += ll;                                                 \
            written += ll;                                               \
        }                                                                \
        {                                                                \
            if (UNLIKELY(off > written)) return -1;                      \
            const uint8_t* match_src = d_ptr - off;                      \
            if (LIKELY(off >= ZXC_PAD_SIZE)) {                           \
                zxc_copy32(d_ptr, match_src);                            \
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {                       \
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;                 \
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;       \
                    size_t rem = ml - ZXC_PAD_SIZE;                      \
                    while (rem > ZXC_PAD_SIZE) {                         \
                        zxc_copy32(out, ref);                            \
                        out += ZXC_PAD_SIZE;                             \
                        ref += ZXC_PAD_SIZE;                             \
                        rem -= ZXC_PAD_SIZE;                             \
                    }                                                    \
                    zxc_copy32(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else if (off >= (ZXC_PAD_SIZE / 2)) {                      \
                zxc_copy16(d_ptr, match_src);                            \
                if (UNLIKELY(ml > (ZXC_PAD_SIZE / 2))) {                 \
                    uint8_t* out = d_ptr + (ZXC_PAD_SIZE / 2);           \
                    const uint8_t* ref = match_src + (ZXC_PAD_SIZE / 2); \
                    size_t rem = ml - (ZXC_PAD_SIZE / 2);                \
                    while (rem > (ZXC_PAD_SIZE / 2)) {                   \
                        zxc_copy16(out, ref);                            \
                        out += (ZXC_PAD_SIZE / 2);                       \
                        ref += (ZXC_PAD_SIZE / 2);                       \
                        rem -= (ZXC_PAD_SIZE / 2);                       \
                    }                                                    \
                    zxc_copy16(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else if (off == 1) {                                       \
                ZXC_MEMSET(d_ptr, match_src[0], ml);                     \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else {                                                     \
                size_t copied = 0;                                       \
                while (copied < ml) {                                    \
                    zxc_copy_overlap16(d_ptr + copied, off);             \
                    copied += (ZXC_PAD_SIZE / 2);                        \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            }                                                            \
        }                                                                \
    } while (0)

// FAST version: no offset validation (for use after written >= 256 or 65536)
#define DECODE_SEQ_FAST(ll, ml, off)                                     \
    do {                                                                 \
        {                                                                \
            const uint8_t* src_lit = l_ptr;                              \
            uint8_t* dst_lit = d_ptr;                                    \
            zxc_copy32(dst_lit, src_lit);                                \
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {                           \
                dst_lit += ZXC_PAD_SIZE;                                 \
                src_lit += ZXC_PAD_SIZE;                                 \
                size_t rem = ll - ZXC_PAD_SIZE;                          \
                while (rem > ZXC_PAD_SIZE) {                             \
                    zxc_copy32(dst_lit, src_lit);                        \
                    dst_lit += ZXC_PAD_SIZE;                             \
                    src_lit += ZXC_PAD_SIZE;                             \
                    rem -= ZXC_PAD_SIZE;                                 \
                }                                                        \
                zxc_copy32(dst_lit, src_lit);                            \
            }                                                            \
            l_ptr += ll;                                                 \
            d_ptr += ll;                                                 \
        }                                                                \
        {                                                                \
            const uint8_t* match_src = d_ptr - off;                      \
            if (LIKELY(off >= ZXC_PAD_SIZE)) {                           \
                zxc_copy32(d_ptr, match_src);                            \
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {                       \
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;                 \
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;       \
                    size_t rem = ml - ZXC_PAD_SIZE;                      \
                    while (rem > ZXC_PAD_SIZE) {                         \
                        zxc_copy32(out, ref);                            \
                        out += ZXC_PAD_SIZE;                             \
                        ref += ZXC_PAD_SIZE;                             \
                        rem -= ZXC_PAD_SIZE;                             \
                    }                                                    \
                    zxc_copy32(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
            } else if (off >= (ZXC_PAD_SIZE / 2)) {                      \
                zxc_copy16(d_ptr, match_src);                            \
                if (UNLIKELY(ml > (ZXC_PAD_SIZE / 2))) {                 \
                    uint8_t* out = d_ptr + (ZXC_PAD_SIZE / 2);           \
                    const uint8_t* ref = match_src + (ZXC_PAD_SIZE / 2); \
                    size_t rem = ml - (ZXC_PAD_SIZE / 2);                \
                    while (rem > (ZXC_PAD_SIZE / 2)) {                   \
                        zxc_copy16(out, ref);                            \
                        out += (ZXC_PAD_SIZE / 2);                       \
                        ref += (ZXC_PAD_SIZE / 2);                       \
                        rem -= (ZXC_PAD_SIZE / 2);                       \
                    }                                                    \
                    zxc_copy16(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
            } else if (off == 1) {                                       \
                ZXC_MEMSET(d_ptr, match_src[0], ml);                     \
                d_ptr += ml;                                             \
            } else {                                                     \
                size_t copied = 0;                                       \
                while (copied < ml) {                                    \
                    zxc_copy_overlap16(d_ptr + copied, off);             \
                    copied += (ZXC_PAD_SIZE / 2);                        \
                }                                                        \
                d_ptr += ml;                                             \
            }                                                            \
        }                                                                \
    } while (0)

    // --- SAFE Loop: offset validation until threshold (4x unroll) ---
    // For 1-byte offsets: bounds check until 256 bytes written
    // For 2-byte offsets: bounds check until 65536 bytes written
    size_t bounds_threshold = (gh.enc_off == 1) ? (1U << 8) : (1U << 16);

    while (n_seq >= 4 && d_ptr < d_end_safe && written < bounds_threshold) {
        uint32_t tokens = zxc_le32(t_ptr);
        t_ptr += 4;

        uint32_t off1, off2, off3, off4;
        if (gh.enc_off == 1) {
            // Read 4 x 1-byte offsets
            uint32_t offsets = zxc_le32(o_ptr);
            o_ptr += 4;
            off1 = offsets & 0xFF;
            off2 = (offsets >> 8) & 0xFF;
            off3 = (offsets >> 16) & 0xFF;
            off4 = (offsets >> 24) & 0xFF;
        } else {
            // Read 4 x 2-byte offsets
            uint64_t offsets = zxc_le64(o_ptr);
            o_ptr += 8;
            off1 = (uint32_t)(offsets & 0xFFFF);
            off2 = (uint32_t)((offsets >> 16) & 0xFFFF);
            off3 = (uint32_t)((offsets >> 32) & 0xFFFF);
            off4 = (uint32_t)((offsets >> 48) & 0xFFFF);
        }

        // Reject zero offsets
        if (UNLIKELY((off1 == 0) | (off2 == 0) | (off3 == 0) | (off4 == 0))) return -1;

        uint32_t ll1 = (tokens & 0x0F0) >> 4;
        uint32_t ml1 = (tokens & 0x00F);
        if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) ll1 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) ml1 += zxc_read_vbyte(&e_ptr, e_end);
        ml1 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll1 + ml1 > d_end)) return -1;
        DECODE_SEQ_SAFE(ll1, ml1, off1);

        uint32_t ll2 = (tokens & 0x0F000) >> 12;
        uint32_t ml2 = (tokens & 0x00F00) >> 8;
        if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) ll2 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) ml2 += zxc_read_vbyte(&e_ptr, e_end);
        ml2 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll2 + ml2 > d_end)) return -1;
        DECODE_SEQ_SAFE(ll2, ml2, off2);

        uint32_t ll3 = (tokens & 0x0F00000) >> 20;
        uint32_t ml3 = (tokens & 0x00F0000) >> 16;
        if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) ll3 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) ml3 += zxc_read_vbyte(&e_ptr, e_end);
        ml3 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll3 + ml3 > d_end)) return -1;
        DECODE_SEQ_SAFE(ll3, ml3, off3);

        uint32_t ll4 = (tokens >> 28);
        uint32_t ml4 = (tokens >> 24) & 0x0F;
        if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) ll4 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) ml4 += zxc_read_vbyte(&e_ptr, e_end);
        ml4 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll4 + ml4 > d_end)) return -1;
        DECODE_SEQ_SAFE(ll4, ml4, off4);

        n_seq -= 4;
    }

    // --- FAST Loop: After threshold, no offset validation needed (4x unroll) ---
    while (n_seq >= 4 && d_ptr < d_end_safe) {
        uint32_t tokens = zxc_le32(t_ptr);
        t_ptr += 4;

        uint32_t off1, off2, off3, off4;
        if (gh.enc_off == 1) {
            // Read 4 x 1-byte offsets
            uint32_t offsets = zxc_le32(o_ptr);
            o_ptr += 4;
            off1 = offsets & 0xFF;
            off2 = (offsets >> 8) & 0xFF;
            off3 = (offsets >> 16) & 0xFF;
            off4 = (offsets >> 24) & 0xFF;
        } else {
            // Read 4 x 2-byte offsets
            uint64_t offsets = zxc_le64(o_ptr);
            o_ptr += 8;
            off1 = (uint32_t)(offsets & 0xFFFF);
            off2 = (uint32_t)((offsets >> 16) & 0xFFFF);
            off3 = (uint32_t)((offsets >> 32) & 0xFFFF);
            off4 = (uint32_t)((offsets >> 48) & 0xFFFF);
        }

        uint32_t ll1 = (tokens & 0x0F0) >> 4;
        uint32_t ml1 = (tokens & 0x00F);
        if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) ll1 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) ml1 += zxc_read_vbyte(&e_ptr, e_end);
        ml1 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll1 + ml1 > d_end)) return -1;
        DECODE_SEQ_FAST(ll1, ml1, off1);

        uint32_t ll2 = (tokens & 0x0F000) >> 12;
        uint32_t ml2 = (tokens & 0x00F00) >> 8;
        if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) ll2 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) ml2 += zxc_read_vbyte(&e_ptr, e_end);
        ml2 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll2 + ml2 > d_end)) return -1;
        DECODE_SEQ_FAST(ll2, ml2, off2);

        uint32_t ll3 = (tokens & 0x0F00000) >> 20;
        uint32_t ml3 = (tokens & 0x00F0000) >> 16;
        if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) ll3 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) ml3 += zxc_read_vbyte(&e_ptr, e_end);
        ml3 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll3 + ml3 > d_end)) return -1;
        DECODE_SEQ_FAST(ll3, ml3, off3);

        uint32_t ll4 = (tokens >> 28);
        uint32_t ml4 = (tokens >> 24) & 0x0F;
        if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) ll4 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) ml4 += zxc_read_vbyte(&e_ptr, e_end);
        ml4 += ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(d_ptr + ll4 + ml4 > d_end)) return -1;
        DECODE_SEQ_FAST(ll4, ml4, off4);

        n_seq -= 4;
    }

#undef DECODE_SEQ_SAFE
#undef DECODE_SEQ_FAST

    // Validate vbyte reads didn't overflow
    if (UNLIKELY(e_ptr > e_end)) return -1;

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe) {
        // Save pointers before reading (in case we need to fall back to Safe Path)
        const uint8_t* t_save = t_ptr;
        const uint8_t* o_save = o_ptr;
        const uint8_t* e_save = e_ptr;

        uint8_t token = *t_ptr++;
        uint32_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset;
        if (gh.enc_off == 1) {
            offset = *o_ptr++;  // 1-byte offset
        } else {
            offset = (uint32_t)o_ptr[0] | ((uint32_t)o_ptr[1] << 8);
            o_ptr += 2;
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) ll += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_vbyte(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        // Check bounds before wild copies - if too close to end, fall back to Safe Path
        if (UNLIKELY(d_ptr + ll + ml + ZXC_PAD_SIZE > d_end)) {
            // Restore pointers and let Safe Path handle this sequence
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            break;
        }

        {
            const uint8_t* src_lit = l_ptr;
            uint8_t* dst_lit = d_ptr;
            zxc_copy32(dst_lit, src_lit);
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                dst_lit += ZXC_PAD_SIZE;
                src_lit += ZXC_PAD_SIZE;
                size_t rem = ll - ZXC_PAD_SIZE;
                while (rem > ZXC_PAD_SIZE) {
                    zxc_copy32(dst_lit, src_lit);
                    dst_lit += ZXC_PAD_SIZE;
                    src_lit += ZXC_PAD_SIZE;
                    rem -= ZXC_PAD_SIZE;
                }
                zxc_copy32(dst_lit, src_lit);
            }
            l_ptr += ll;
            d_ptr += ll;
            written += ll;
        }

        {
            // Skip check if written >= bounds_threshold (256 for 8-bit, 65536 for 16-bit)
            if (UNLIKELY(written < bounds_threshold && (offset == 0 || offset > written)))
                return -1;

            const uint8_t* match_src = d_ptr - offset;
            if (LIKELY(offset >= ZXC_PAD_SIZE)) {
                zxc_copy32(d_ptr, match_src);
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;
                    size_t rem = ml - ZXC_PAD_SIZE;
                    while (rem > ZXC_PAD_SIZE) {
                        zxc_copy32(out, ref);
                        out += ZXC_PAD_SIZE;
                        ref += ZXC_PAD_SIZE;
                        rem -= ZXC_PAD_SIZE;
                    }
                    zxc_copy32(out, ref);
                }
                d_ptr += ml;
                written += ml;
            } else if (offset == 1) {
                ZXC_MEMSET(d_ptr, match_src[0], ml);
                d_ptr += ml;
                written += ml;
            } else {
                for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
                d_ptr += ml;
                written += ml;
            }
        }
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint8_t token = *t_ptr++;
        uint32_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset;
        if (gh.enc_off == 1) {
            offset = *o_ptr++;  // 1-byte offset
        } else {
            offset = (uint32_t)o_ptr[0] | ((uint32_t)o_ptr[1] << 8);  // 2-byte offset
            o_ptr += 2;
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) ll += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_vbyte(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        if (UNLIKELY(d_ptr + ll > d_end)) return -1;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        const uint8_t* match_src = d_ptr - offset;
        if (UNLIKELY(match_src < dst || d_ptr + ml > d_end)) return -1;

        if (offset < ml) {
            for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
        } else {
            ZXC_MEMCPY(d_ptr, match_src, ml);
        }
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    size_t generated = d_ptr - dst;
    if (generated < expected_raw_size) {
        size_t rem = expected_raw_size - generated;
        if (UNLIKELY(d_ptr + rem > d_end || l_ptr + rem > l_end)) return -1;
        ZXC_MEMCPY(d_ptr, l_ptr, rem);
        d_ptr += rem;
    }

    // Final validation: decoded size must match expected
    if (UNLIKELY((size_t)(d_ptr - dst) != expected_raw_size)) return -1;

    return (int)(d_ptr - dst);
}

/**
 * @brief Decodes a GHI format compressed block.
 *
 * This function handles the decoding of a compressed block formatted with the
 * internal GHI structure.
 *
 * @param[in] ctx Pointer to the decompression context (unused in current implementation).
 * @param[in] src Pointer to the source buffer containing compressed data.
 * @param[in] src_size Size of the source buffer in bytes.
 * @param[out] dst Pointer to the destination buffer for decompressed data.
 * @param[in] dst_capacity Capacity of the destination buffer in bytes.
 * @param[in] expected_raw_size Expected size of the decompressed data in bytes.
 * @return int Returns 0 on success, or a negative error code on failure.
 */
static int zxc_decode_block_ghi(zxc_cctx_t* ctx, const uint8_t* RESTRICT src, size_t src_size,
                                uint8_t* RESTRICT dst, size_t dst_capacity,
                                uint32_t expected_raw_size) {
    (void)ctx;
    zxc_gnr_header_t gh;
    zxc_section_desc_t desc[ZXC_GHI_SECTIONS];

    int res = zxc_read_ghi_header_and_desc(src, src_size, &gh, desc);
    if (UNLIKELY(res != 0)) return -1;

    const uint8_t* p_curr =
        src + ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    // --- Stream Pointers & Validation ---
    size_t sz_lit = (uint32_t)desc[0].sizes;
    size_t sz_seqs = (uint32_t)desc[1].sizes;
    size_t sz_exts = (uint32_t)desc[2].sizes;
    const uint8_t* l_ptr = p_curr;
    const uint8_t* l_end = l_ptr + sz_lit;
    p_curr += sz_lit;

    const uint8_t* seq_ptr = p_curr;
    const uint8_t* extras_ptr = p_curr + sz_seqs;
    const uint8_t* const extras_end = extras_ptr + sz_exts;

    // Validate streams don't overflow source buffer
    if (UNLIKELY(extras_end != src + src_size)) return -1;

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    const uint8_t* const d_end_safe = d_end - (ZXC_PAD_SIZE * 4);  // 128
    // Safety margin for 4x unrolled loop: 4 * (ZXC_SEQ_LL_MASK LL +
    // ZXC_SEQ_ML_MASK+ZXC_LZ_MIN_MATCH_LEN ML) + ZXC_PAD_SIZE Pad = 4 x (255 + 255 + 5) + 32 = 2092
    const uint8_t* const d_end_fast = d_end - (ZXC_PAD_SIZE * 66);  // 2112

    uint32_t n_seq = gh.n_sequences;

    // Track bytes written for offset validation
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written (max 8-bit offset)
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written (max 16-bit offset)
    // After threshold, all offsets are guaranteed valid (can't exceed written bytes)
    size_t written = 0;

// Macro for copy literal + match (uses 32-byte wild copies)
// SAFE version: validates offset against written bytes
#define DECODE_SEQ_SAFE(ll, ml, off)                                     \
    do {                                                                 \
        {                                                                \
            const uint8_t* src_lit = l_ptr;                              \
            uint8_t* dst_lit = d_ptr;                                    \
            zxc_copy32(dst_lit, src_lit);                                \
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {                           \
                dst_lit += ZXC_PAD_SIZE;                                 \
                src_lit += ZXC_PAD_SIZE;                                 \
                size_t rem = ll - ZXC_PAD_SIZE;                          \
                while (rem > ZXC_PAD_SIZE) {                             \
                    zxc_copy32(dst_lit, src_lit);                        \
                    dst_lit += ZXC_PAD_SIZE;                             \
                    src_lit += ZXC_PAD_SIZE;                             \
                    rem -= ZXC_PAD_SIZE;                                 \
                }                                                        \
                zxc_copy32(dst_lit, src_lit);                            \
            }                                                            \
            l_ptr += ll;                                                 \
            d_ptr += ll;                                                 \
            written += ll;                                               \
        }                                                                \
        {                                                                \
            if (UNLIKELY(off > written)) return -1;                      \
            const uint8_t* match_src = d_ptr - off;                      \
            if (LIKELY(off >= ZXC_PAD_SIZE)) {                           \
                zxc_copy32(d_ptr, match_src);                            \
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {                       \
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;                 \
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;       \
                    size_t rem = ml - ZXC_PAD_SIZE;                      \
                    while (rem > ZXC_PAD_SIZE) {                         \
                        zxc_copy32(out, ref);                            \
                        out += ZXC_PAD_SIZE;                             \
                        ref += ZXC_PAD_SIZE;                             \
                        rem -= ZXC_PAD_SIZE;                             \
                    }                                                    \
                    zxc_copy32(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else if (off >= (ZXC_PAD_SIZE / 2)) {                      \
                zxc_copy16(d_ptr, match_src);                            \
                if (UNLIKELY(ml > (ZXC_PAD_SIZE / 2))) {                 \
                    uint8_t* out = d_ptr + (ZXC_PAD_SIZE / 2);           \
                    const uint8_t* ref = match_src + (ZXC_PAD_SIZE / 2); \
                    size_t rem = ml - (ZXC_PAD_SIZE / 2);                \
                    while (rem > (ZXC_PAD_SIZE / 2)) {                   \
                        zxc_copy16(out, ref);                            \
                        out += (ZXC_PAD_SIZE / 2);                       \
                        ref += (ZXC_PAD_SIZE / 2);                       \
                        rem -= (ZXC_PAD_SIZE / 2);                       \
                    }                                                    \
                    zxc_copy16(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else if (off == 1) {                                       \
                ZXC_MEMSET(d_ptr, match_src[0], ml);                     \
                d_ptr += ml;                                             \
                written += ml;                                           \
            } else {                                                     \
                size_t copied = 0;                                       \
                while (copied < ml) {                                    \
                    zxc_copy_overlap16(d_ptr + copied, off);             \
                    copied += (ZXC_PAD_SIZE / 2);                        \
                }                                                        \
                d_ptr += ml;                                             \
                written += ml;                                           \
            }                                                            \
        }                                                                \
    } while (0)

// FAST version: no offset validation (for use after written >= 256 or 65536)
#define DECODE_SEQ_FAST(ll, ml, off)                                     \
    do {                                                                 \
        {                                                                \
            const uint8_t* src_lit = l_ptr;                              \
            uint8_t* dst_lit = d_ptr;                                    \
            zxc_copy32(dst_lit, src_lit);                                \
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {                           \
                dst_lit += ZXC_PAD_SIZE;                                 \
                src_lit += ZXC_PAD_SIZE;                                 \
                size_t rem = ll - ZXC_PAD_SIZE;                          \
                while (rem > ZXC_PAD_SIZE) {                             \
                    zxc_copy32(dst_lit, src_lit);                        \
                    dst_lit += ZXC_PAD_SIZE;                             \
                    src_lit += ZXC_PAD_SIZE;                             \
                    rem -= ZXC_PAD_SIZE;                                 \
                }                                                        \
                zxc_copy32(dst_lit, src_lit);                            \
            }                                                            \
            l_ptr += ll;                                                 \
            d_ptr += ll;                                                 \
        }                                                                \
        {                                                                \
            const uint8_t* match_src = d_ptr - off;                      \
            if (LIKELY(off >= ZXC_PAD_SIZE)) {                           \
                zxc_copy32(d_ptr, match_src);                            \
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {                       \
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;                 \
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;       \
                    size_t rem = ml - ZXC_PAD_SIZE;                      \
                    while (rem > ZXC_PAD_SIZE) {                         \
                        zxc_copy32(out, ref);                            \
                        out += ZXC_PAD_SIZE;                             \
                        ref += ZXC_PAD_SIZE;                             \
                        rem -= ZXC_PAD_SIZE;                             \
                    }                                                    \
                    zxc_copy32(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
            } else if (off >= (ZXC_PAD_SIZE / 2)) {                      \
                zxc_copy16(d_ptr, match_src);                            \
                if (UNLIKELY(ml > (ZXC_PAD_SIZE / 2))) {                 \
                    uint8_t* out = d_ptr + (ZXC_PAD_SIZE / 2);           \
                    const uint8_t* ref = match_src + (ZXC_PAD_SIZE / 2); \
                    size_t rem = ml - (ZXC_PAD_SIZE / 2);                \
                    while (rem > (ZXC_PAD_SIZE / 2)) {                   \
                        zxc_copy16(out, ref);                            \
                        out += (ZXC_PAD_SIZE / 2);                       \
                        ref += (ZXC_PAD_SIZE / 2);                       \
                        rem -= (ZXC_PAD_SIZE / 2);                       \
                    }                                                    \
                    zxc_copy16(out, ref);                                \
                }                                                        \
                d_ptr += ml;                                             \
            } else if (off == 1) {                                       \
                ZXC_MEMSET(d_ptr, match_src[0], ml);                     \
                d_ptr += ml;                                             \
            } else {                                                     \
                size_t copied = 0;                                       \
                while (copied < ml) {                                    \
                    zxc_copy_overlap16(d_ptr + copied, off);             \
                    copied += (ZXC_PAD_SIZE / 2);                        \
                }                                                        \
                d_ptr += ml;                                             \
            }                                                            \
        }                                                                \
    } while (0)

    // --- SAFE Loop: offset validation until threshold ---
    // Since offset is 16-bit, threshold is 65536.
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written
    size_t bounds_threshold = (gh.enc_off == 1) ? (1U << 8) : (1U << 16);

    while (n_seq > 0 && d_ptr < d_end_safe && written < bounds_threshold) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += 4;

        uint32_t ll = (uint32_t)(seq >> 24);
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_vbyte(&extras_ptr, extras_end);

        uint32_t m_bits = (uint32_t)((seq >> 16) & 0xFF);
        uint32_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_vbyte(&extras_ptr, extras_end);

        uint32_t offset = (uint32_t)(seq & 0xFFFF);

        // Strict bounds check: sequence must fit, AND wild copies must not overshoot
        if (UNLIKELY(d_ptr + ll + ml + ZXC_PAD_SIZE > d_end)) {
            // Fallback to exact copy (slow but safe)
            if (UNLIKELY(d_ptr + ll > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll);
            l_ptr += ll;
            d_ptr += ll;
            written += ll;

            if (UNLIKELY(offset == 0 || offset > written)) return -1;
            const uint8_t* match_src = d_ptr - offset;
            if (UNLIKELY(match_src < dst || d_ptr + ml > d_end)) return -1;  // Bounds check

            if (offset < ml) {
                for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
            } else {
                ZXC_MEMCPY(d_ptr, match_src, ml);
            }
            d_ptr += ml;
            written += ml;
        } else {
            // Safe to process with wild copies
            if (UNLIKELY(offset == 0)) return -1;
            DECODE_SEQ_SAFE(ll, ml, offset);
        }
        n_seq--;
    }

    // --- FAST Loop: After threshold, check large margin to avoid individual bounds checks ---
    while (n_seq >= 4 && d_ptr < d_end_fast) {
        uint32_t s1 = zxc_le32(seq_ptr);
        uint32_t s2 = zxc_le32(seq_ptr + 4);
        uint32_t s3 = zxc_le32(seq_ptr + 8);
        uint32_t s4 = zxc_le32(seq_ptr + 12);
        seq_ptr += 16;

        uint32_t ll1 = (uint32_t)(s1 >> 24);
        if (UNLIKELY(ll1 == ZXC_SEQ_LL_MASK)) {
            ll1 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll1 > d_end)) return -1;
        }
        uint32_t m1b = (uint32_t)((s1 >> 16) & 0xFF);
        uint32_t ml1 = m1b + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m1b == ZXC_SEQ_ML_MASK)) {
            ml1 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll1 + ml1 > d_end)) return -1;
        }
        uint32_t of1 = (uint32_t)(s1 & 0xFFFF);
        if (UNLIKELY(l_ptr + ll1 > l_end || d_ptr + ll1 + ml1 > d_end)) return -1;
        DECODE_SEQ_FAST(ll1, ml1, of1);

        uint32_t ll2 = (uint32_t)(s2 >> 24);
        if (UNLIKELY(ll2 == ZXC_SEQ_LL_MASK)) {
            ll2 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll2 > d_end)) return -1;
        }
        uint32_t m2b = (uint32_t)((s2 >> 16) & 0xFF);
        uint32_t ml2 = m2b + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m2b == ZXC_SEQ_ML_MASK)) {
            ml2 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll2 + ml2 > d_end)) return -1;
        }
        uint32_t of2 = (uint32_t)(s2 & 0xFFFF);
        if (UNLIKELY(l_ptr + ll2 > l_end || d_ptr + ll2 + ml2 > d_end)) return -1;
        DECODE_SEQ_FAST(ll2, ml2, of2);

        uint32_t ll3 = (uint32_t)(s3 >> 24);
        if (UNLIKELY(ll3 == ZXC_SEQ_LL_MASK)) {
            ll3 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll3 > d_end)) return -1;
        }
        uint32_t m3b = (uint32_t)((s3 >> 16) & 0xFF);
        uint32_t ml3 = m3b + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m3b == ZXC_SEQ_ML_MASK)) {
            ml3 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll3 + ml3 > d_end)) return -1;
        }
        uint32_t of3 = (uint32_t)(s3 & 0xFFFF);
        if (UNLIKELY(l_ptr + ll3 > l_end || d_ptr + ll3 + ml3 > d_end)) return -1;
        DECODE_SEQ_FAST(ll3, ml3, of3);

        uint32_t ll4 = (uint32_t)(s4 >> 24);
        if (UNLIKELY(ll4 == ZXC_SEQ_LL_MASK)) {
            ll4 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll4 > d_end)) return -1;
        }
        uint32_t m4b = (uint32_t)((s4 >> 16) & 0xFF);
        uint32_t ml4 = m4b + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m4b == ZXC_SEQ_ML_MASK)) {
            ml4 += zxc_read_vbyte(&extras_ptr, extras_end);
            if (UNLIKELY(d_ptr + ll4 + ml4 > d_end)) return -1;
        }
        uint32_t of4 = (uint32_t)(s4 & 0xFFFF);
        if (UNLIKELY(l_ptr + ll4 > l_end || d_ptr + ll4 + ml4 > d_end)) return -1;
        DECODE_SEQ_FAST(ll4, ml4, of4);

        n_seq -= 4;
    }

#undef DECODE_SEQ_SAFE
#undef DECODE_SEQ_FAST

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe) {
        // Save state for fallback
        const uint8_t* seq_save = seq_ptr;
        const uint8_t* ext_save = extras_ptr;

        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += 4;

        uint32_t ll = (uint32_t)(seq >> 24);
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_vbyte(&extras_ptr, extras_end);

        uint32_t m_bits = (uint32_t)((seq >> 16) & 0xFF);
        uint32_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_vbyte(&extras_ptr, extras_end);

        // Strict bounds checks (including wild copy overrun safety)
        if (UNLIKELY(d_ptr + ll + ml + ZXC_PAD_SIZE > d_end || l_ptr + ll > l_end)) {
            // Restore state and break to Safe Path
            seq_ptr = seq_save;
            extras_ptr = ext_save;
            break;
        }
        uint32_t offset = (uint32_t)(seq & 0xFFFF);

        {
            const uint8_t* src_lit = l_ptr;
            uint8_t* dst_lit = d_ptr;
            zxc_copy32(dst_lit, src_lit);
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                dst_lit += ZXC_PAD_SIZE;
                src_lit += ZXC_PAD_SIZE;
                size_t rem = ll - ZXC_PAD_SIZE;
                while (rem > ZXC_PAD_SIZE) {
                    zxc_copy32(dst_lit, src_lit);
                    dst_lit += ZXC_PAD_SIZE;
                    src_lit += ZXC_PAD_SIZE;
                    rem -= ZXC_PAD_SIZE;
                }
                zxc_copy32(dst_lit, src_lit);
            }
            l_ptr += ll;
            d_ptr += ll;
            written += ll;
        }

        {
            // Skip check if written >= bounds_threshold (256 for 8-bit, 65536 for 16-bit)
            if (UNLIKELY(written < bounds_threshold && (offset == 0 || offset > written)))
                return -1;

            const uint8_t* match_src = d_ptr - offset;
            if (LIKELY(offset >= ZXC_PAD_SIZE)) {
                zxc_copy32(d_ptr, match_src);
                if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
                    uint8_t* out = d_ptr + ZXC_PAD_SIZE;
                    const uint8_t* ref = match_src + ZXC_PAD_SIZE;
                    size_t rem = ml - ZXC_PAD_SIZE;
                    while (rem > ZXC_PAD_SIZE) {
                        zxc_copy32(out, ref);
                        out += ZXC_PAD_SIZE;
                        ref += ZXC_PAD_SIZE;
                        rem -= ZXC_PAD_SIZE;
                    }
                    zxc_copy32(out, ref);
                }
                d_ptr += ml;
                written += ml;
            } else if (offset == 1) {
                ZXC_MEMSET(d_ptr, match_src[0], ml);
                d_ptr += ml;
                written += ml;
            } else {
                for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
                d_ptr += ml;
                written += ml;
            }
        }
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += 4;

        uint32_t ll = (uint32_t)(seq >> 24);
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_vbyte(&extras_ptr, extras_end);

        uint32_t m_bits = (uint32_t)((seq >> 16) & 0xFF);
        uint32_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_vbyte(&extras_ptr, extras_end);
        uint32_t offset = (uint32_t)(seq & 0xFFFF);

        if (UNLIKELY(d_ptr + ll > d_end)) return -1;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        const uint8_t* match_src = d_ptr - offset;
        if (UNLIKELY(offset == 0 || match_src < dst || d_ptr + ml > d_end)) return -1;

        if (offset < ml) {
            for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
        } else {
            ZXC_MEMCPY(d_ptr, match_src, ml);
        }
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    size_t generated = d_ptr - dst;
    if (generated < expected_raw_size) {
        size_t rem = expected_raw_size - generated;
        if (UNLIKELY(d_ptr + rem > d_end || l_ptr + rem > l_end)) return -1;
        ZXC_MEMCPY(d_ptr, l_ptr, rem);
        d_ptr += rem;
    }

    // Final validation: decoded size must match expected
    if (UNLIKELY((size_t)(d_ptr - dst) != expected_raw_size)) return -1;

    return (int)(d_ptr - dst);
}

// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz, uint8_t* dst,
                                 size_t dst_cap) {
    if (UNLIKELY(src_sz < ZXC_BLOCK_HEADER_SIZE)) return -1;

    uint8_t type = src[0];
    uint8_t flags = src[1];
    uint32_t comp_sz = zxc_le32(src + 4);
    uint32_t raw_sz = zxc_le32(src + 8);

    int has_crc = (flags & ZXC_BLOCK_FLAG_CHECKSUM);
    size_t header_len = ZXC_BLOCK_HEADER_SIZE + (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);

    if (UNLIKELY(src_sz < header_len + comp_sz)) return -1;

    const uint8_t* data = src + header_len;
    int decoded_sz = -1;

    switch (type) {
        case ZXC_BLOCK_GLO:
            decoded_sz = zxc_decode_block_glo(ctx, data, comp_sz, dst, dst_cap, raw_sz);
            break;
        case ZXC_BLOCK_GHI:
            decoded_sz = zxc_decode_block_ghi(ctx, data, comp_sz, dst, dst_cap, raw_sz);
            break;
        case ZXC_BLOCK_RAW:
            if (UNLIKELY(raw_sz > dst_cap || raw_sz > comp_sz)) return -1;
            ZXC_MEMCPY(dst, data, raw_sz);
            decoded_sz = (int)raw_sz;
            break;
        case ZXC_BLOCK_NUM:
            decoded_sz = zxc_decode_block_num(data, comp_sz, dst, dst_cap, raw_sz);
            break;
        default:
            return -1;
    }

    if (decoded_sz >= 0 && has_crc && ctx->checksum_enabled) {
        uint8_t algo = flags & ZXC_CHECKSUM_TYPE_MASK;
        uint64_t stored = zxc_le64(src + ZXC_BLOCK_HEADER_SIZE);
        uint64_t calc = zxc_checksum(dst, (size_t)decoded_sz, algo);

        if (UNLIKELY(stored != calc)) return -1;
    }

    return decoded_sz;
}
