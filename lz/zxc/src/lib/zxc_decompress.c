/*
 * Copyright (c) 2025, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_sans_io.h"
#include "zxc_internal.h"

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
 * @brief Ensures that the bit reader buffer contains at least the specified
 * number of bits.
 *
 * This function checks if the internal buffer of the bit reader has enough bits
 * available to satisfy a subsequent read operation of `needed` bits. If not, it
 * refills the buffer from the source.
 *
 * @param[in,out] br Pointer to the bit reader context.
 * @param[in] needed The number of bits required to be available in the buffer.
 */
static ZXC_ALWAYS_INLINE void zxc_br_ensure(zxc_bit_reader_t* br, int needed) {
    if (UNLIKELY(br->bits < needed)) {
        int safe_bits = (br->bits < 0) ? 0 : br->bits;
        br->bits = safe_bits;

// Mask out garbage bits
#if defined(__BMI2__) && defined(__x86_64__)
        // BMI2 Optimization: _bzhi_u64 isolates the valid bits we want to keep.
        br->accum = _bzhi_u64(br->accum, safe_bits);
#else
        br->accum &= ((1ULL << safe_bits) - 1);
#endif

        const uint8_t* p_loc = br->ptr;

        uint64_t raw = zxc_le64(p_loc);
        int consumed = (64 - safe_bits) >> 3;
        br->accum |= (raw << safe_bits);
        p_loc += consumed;
        br->bits = safe_bits + consumed * 8;

        br->ptr = p_loc;
    }
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
    if (LIKELY(b0 < 128)) {
        *ptr = p + 1;
        return b0;
    }

    // 2-byte path (second most common)
    if (UNLIKELY(p + 1 >= end)) {
        *ptr = p + 1;
        return 0;
    }
    uint32_t b1 = p[1];
    if (LIKELY(b1 < 128)) {
        *ptr = p + 2;
        return (b0 & 0x7F) | (b1 << 7);
    }

    if (UNLIKELY(p + 2 >= end)) {
        *ptr = p + 2;
        return 0;
    }
    uint32_t b2 = p[2];
    uint32_t val = (b0 & 0x7F) | ((b1 & 0x7F) << 7);
    if (b2 < 128) {
        *ptr = p + 3;
        return val | (b2 << 14);
    }

    if (UNLIKELY(p + 3 >= end)) {
        *ptr = p + 3;
        return 0;
    }
    val |= (b2 & 0x7F) << 14;
    uint32_t b3 = p[3];
    if (b3 < 128) {
        *ptr = p + 4;
        return val | (b3 << 21);
    }

    if (UNLIKELY(p + 4 >= end)) {
        *ptr = p + 4;
        return 0;
    }
    *ptr = p + 5;
    return val | ((b3 & 0x7F) << 21) | ((uint32_t)p[4] << 28);
}

/**
 * @brief Shuffle masks for overlapping copies with small offsets (0-15).
 *
 * Shared between ARM NEON and x86 SSSE3. Each row defines how to replicate
 * source bytes to fill 16 bytes when offset < 16.
 */
#if defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32) || defined(__SSSE3__) || \
    defined(ZXC_USE_AVX2)
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

#elif defined(__SSSE3__) || defined(ZXC_USE_AVX2)
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

    ZXC_ALIGN(64)
    uint32_t deltas[ZXC_DEC_BATCH];

    while (vals_remaining > 0) {
        if (UNLIKELY(p + 16 > p_end)) return -1;
        uint16_t nvals = zxc_le16(p + 0);
        uint16_t bits = zxc_le16(p + 2);
        uint32_t psize = zxc_le32(p + 12);
        p += 16;
        if (UNLIKELY(p + psize > p_end || d_ptr + nvals * 4 > d_end ||
                     bits > (sizeof(uint32_t) * 8)))
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
 * @brief Decompresses a "GNR" (General) encoded block of data.
 *
 * This function handles the decoding of a compressed block formatted with the
 * internal GNR structure.
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
static int zxc_decode_block_gnr(zxc_cctx_t* ctx, const uint8_t* RESTRICT src, size_t src_size,
                                uint8_t* RESTRICT dst, size_t dst_capacity,
                                uint32_t expected_raw_size) {
    zxc_gnr_header_t gh;
    zxc_section_desc_t desc[4];
    if (UNLIKELY(zxc_read_gnr_header_and_desc(src, src_size, &gh, desc) != 0)) return -1;

    const uint8_t* p_data = src + ZXC_GNR_HEADER_BINARY_SIZE + 4 * ZXC_SECTION_DESC_BINARY_SIZE;
    const uint8_t* p_curr = p_data;

    // --- Literal Stream Setup ---
    const uint8_t* l_ptr;
    const uint8_t* l_end;
    uint8_t* rle_buf = NULL;

    size_t lit_stream_size = (size_t)(desc[0].sizes & 0xFFFFFFFF);

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
                if (LIKELY(!(token & 0x80))) {
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
                    uint32_t len = (token & 0x7F) + 4;
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
    size_t sz_tokens = (size_t)(desc[1].sizes & 0xFFFFFFFF);
    size_t sz_offsets = (size_t)(desc[2].sizes & 0xFFFFFFFF);
    size_t sz_extras = (size_t)(desc[3].sizes & 0xFFFFFFFF);

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
#define V2_DECODE_SEQ_SAFE(ll, ml, off)                                  \
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
#define V2_DECODE_SEQ_FAST(ll, ml, off)                                  \
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
    size_t bounds_threshold = (gh.enc_off == 1) ? 256 : 65536;

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
        ml1 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll1 + ml1 > d_end)) return -1;
        V2_DECODE_SEQ_SAFE(ll1, ml1, off1);

        uint32_t ll2 = (tokens & 0x0F000) >> 12;
        uint32_t ml2 = (tokens & 0x00F00) >> 8;
        if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) ll2 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) ml2 += zxc_read_vbyte(&e_ptr, e_end);
        ml2 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll2 + ml2 > d_end)) return -1;
        V2_DECODE_SEQ_SAFE(ll2, ml2, off2);

        uint32_t ll3 = (tokens & 0x0F00000) >> 20;
        uint32_t ml3 = (tokens & 0x00F0000) >> 16;
        if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) ll3 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) ml3 += zxc_read_vbyte(&e_ptr, e_end);
        ml3 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll3 + ml3 > d_end)) return -1;
        V2_DECODE_SEQ_SAFE(ll3, ml3, off3);

        uint32_t ll4 = (tokens >> 28);
        uint32_t ml4 = (tokens >> 24) & 0x0F;
        if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) ll4 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) ml4 += zxc_read_vbyte(&e_ptr, e_end);
        ml4 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll4 + ml4 > d_end)) return -1;
        V2_DECODE_SEQ_SAFE(ll4, ml4, off4);

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
        ml1 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll1 + ml1 > d_end)) return -1;
        V2_DECODE_SEQ_FAST(ll1, ml1, off1);

        uint32_t ll2 = (tokens & 0x0F000) >> 12;
        uint32_t ml2 = (tokens & 0x00F00) >> 8;
        if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) ll2 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) ml2 += zxc_read_vbyte(&e_ptr, e_end);
        ml2 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll2 + ml2 > d_end)) return -1;
        V2_DECODE_SEQ_FAST(ll2, ml2, off2);

        uint32_t ll3 = (tokens & 0x0F00000) >> 20;
        uint32_t ml3 = (tokens & 0x00F0000) >> 16;
        if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) ll3 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) ml3 += zxc_read_vbyte(&e_ptr, e_end);
        ml3 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll3 + ml3 > d_end)) return -1;
        V2_DECODE_SEQ_FAST(ll3, ml3, off3);

        uint32_t ll4 = (tokens >> 28);
        uint32_t ml4 = (tokens >> 24) & 0x0F;
        if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) ll4 += zxc_read_vbyte(&e_ptr, e_end);
        if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) ml4 += zxc_read_vbyte(&e_ptr, e_end);
        ml4 += ZXC_LZ_MIN_MATCH;
        if (UNLIKELY(d_ptr + ll4 + ml4 > d_end)) return -1;
        V2_DECODE_SEQ_FAST(ll4, ml4, off4);

        n_seq -= 4;
    }

#undef V2_DECODE_SEQ_SAFE
#undef V2_DECODE_SEQ_FAST

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
        ml += ZXC_LZ_MIN_MATCH;

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
        ml += ZXC_LZ_MIN_MATCH;

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

    if (LIKELY(type == ZXC_BLOCK_GNR)) {
        decoded_sz = zxc_decode_block_gnr(ctx, data, comp_sz, dst, dst_cap, raw_sz);
    } else if (type == ZXC_BLOCK_RAW) {
        if (UNLIKELY(raw_sz > dst_cap || raw_sz > comp_sz)) return -1;
        ZXC_MEMCPY(dst, data, raw_sz);
        decoded_sz = (int)raw_sz;
    } else if (type == ZXC_BLOCK_NUM) {
        decoded_sz = zxc_decode_block_num(data, comp_sz, dst, dst_cap, raw_sz);
    } else {
        return -1;
    }

    if (decoded_sz >= 0 && has_crc && ctx->checksum_enabled) {
        uint64_t stored = zxc_le64(src + ZXC_BLOCK_HEADER_SIZE);
        uint64_t calc = zxc_checksum(dst, (size_t)decoded_sz);

        if (UNLIKELY(stored != calc)) return -1;
    }

    return decoded_sz;
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
        zxc_block_header_t bh;
        // Read the block header to determine the compressed size
        if (zxc_read_block_header(ip, (size_t)(ip_end - ip), &bh) != 0) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        size_t rem_cap = (size_t)(op_end - op);
        int res = zxc_decompress_chunk_wrapper(&ctx, ip, (size_t)(ip_end - ip), op, rem_cap);
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