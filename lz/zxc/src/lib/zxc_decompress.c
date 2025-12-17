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
 * @brief Reads a variable-length encoded integer (VByte) from a byte stream.
 *
 * This function decodes a 32-bit unsigned integer stored in the VByte format
 * (also known as VarInt). In this format, the 7 least significant bits of each
 * byte hold data, and the most significant bit (MSB) serves as a continuation
 * flag. If the MSB is set (1), the next byte is part of the integer. If the MSB
 * is clear (0), the byte is the last one in the sequence.
 *
 * The function updates the source pointer to the position immediately following
 * the read integer.
 *
 * @note This function is marked as ALWAYS_INLINE for performance reasons, as it
 *       is likely called frequently in a tight decompression loop.
 * @brief Reads a Variable Byte (VByte) encoded integer from the stream.
 *
 * This function handles values that were too large to fit in the standard 4-bit
 * token field. VByte encoding uses the high bit of each byte as a continuation
 * flag (1 = more bytes, 0 = last byte).
 *
 * @param[in,out] ptr Address of the pointer to the current read position. The pointer
 * is advanced.
 * @return The decoded 32-bit integer.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_read_vbyte(const uint8_t** ptr) {
    const uint8_t* p = *ptr;
    uint32_t val = *p++;
    // Fast path: Single byte value (< 128)
    if (LIKELY(val < 128)) {
        *ptr = p;
        return val;
    }
    // Slow path: Multi-byte value
    val &= 0x7F;
    uint32_t shift = 7;
    uint32_t b;
    do {
        b = *p++;
        val |= (b & 0x7F) << shift;
        shift += 7;
    } while (b & 0x80);
    *ptr = p;
    return val;
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

    // Propagate sums across 128-bit lanes
    __m512i v_l0 = _mm512_shuffle_i32x4(v, v, 0x00);  // Broadcast lane 0
    v_l0 = _mm512_shuffle_epi32(v_l0, 0xFF);          // Broadcast last element of lane 0
    v = _mm512_mask_add_epi32(v, 0xFFF0, v, v_l0);    // Add to lanes 1, 2, 3

    __m512i v_l1 = _mm512_shuffle_i32x4(v, v, 0x55);  // Broadcast lane 1
    v_l1 = _mm512_shuffle_epi32(v_l1, 0xFF);          // Broadcast last element of lane 1
    v = _mm512_mask_add_epi32(v, 0xFF00, v, v_l1);    // Add to lanes 2, 3

    __m512i v_l2 = _mm512_shuffle_i32x4(v, v, 0xAA);  // Broadcast lane 2
    v_l2 = _mm512_shuffle_epi32(v_l2, 0xFF);          // Broadcast last element of lane 2
    v = _mm512_mask_add_epi32(v, 0xF000, v, v_l2);    // Add to lane 3

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
static int zxc_decode_block_num(const uint8_t* restrict src, size_t src_size, uint8_t* restrict dst,
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

    ZXC_ALIGN(64) uint32_t deltas[ZXC_DEC_BATCH];

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
 * **Optimization: Fast Path vs. Safe Path**
 * - **Fast Path:** When there is sufficient space in the output buffer
 * (`d_end_safe`), we use "Wild Copies" (16-byte `ZXC_MEMCPY` or SIMD
 * loads/stores) even for short matches or literals. This avoids conditional
 * checks for every byte copy.
 *   - *Trick:* We copy 16 bytes even if the match length is 4. The extra bytes
 *     will simply be overwritten by the next sequence.
 * - **Safe Path:** Near the end of the buffer, we switch to a standard
 * byte-by-byte copy loop to prevent buffer overflows.
 *
 * **Sequence Decoding:**
 * - Reads tokens (Literal Length, Match Length) in batches of 4.
 * - Handles "VByte" variable-length integers for lengths >= 15.
 * - Reconstructs the data by copying literals from the literal stream and
 *   matches from the previously decoded output (history).
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
static int zxc_decode_block_gnr(zxc_cctx_t* ctx, const uint8_t* restrict src, size_t src_size,
                                uint8_t* restrict dst, size_t dst_capacity,
                                uint32_t expected_raw_size) {
    zxc_gnr_header_t gh;
    zxc_section_desc_t desc[4];
    if (UNLIKELY(zxc_read_gnr_header_and_desc(src, src_size, &gh, desc) != 0)) return -1;

    const uint8_t* p_data = src + ZXC_GNR_HEADER_BINARY_SIZE + 4 * ZXC_SECTION_DESC_BINARY_SIZE;
    const uint8_t* p_curr = p_data;

    const uint8_t* l_ptr;
    const uint8_t* l_end;
    uint8_t* rle_buf = NULL;

    if (gh.enc_lit == 1) {
        // RLE Encoded Literals
        size_t required_size = (size_t)(desc[0].sizes >> 32);
        size_t rle_stream_size = (size_t)(desc[0].sizes & 0xFFFFFFFF);

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

            if (UNLIKELY(!rle_buf || rle_stream_size > (size_t)(src + src_size - p_curr)))
                return -1;

            const uint8_t* r_ptr = p_curr;
            const uint8_t* r_end = r_ptr + rle_stream_size;
            const uint8_t* const w_end = rle_buf + required_size;
            uint8_t* w_ptr = rle_buf;

            while (r_ptr < r_end && w_ptr < w_end) {
                uint8_t token = *r_ptr++;
                if (token & 0x80) {
                    // Repeat Run
                    size_t len = (token & 0x7F) + 4;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr >= r_end)) return -1;

                    uint8_t val = *r_ptr++;
                    ZXC_MEMSET(w_ptr, val, len);
                    w_ptr += len;
                } else {
                    // Literal Run
                    size_t len = token + 1;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr + len > r_end)) return -1;
                    ZXC_MEMCPY(w_ptr, r_ptr, len);
                    w_ptr += len;
                    r_ptr += len;
                }
            }
            if (UNLIKELY(w_ptr != w_end)) return -1;
            // RLE stream ended prematurely or overran
            l_ptr = rle_buf;
            l_end = rle_buf + required_size;
        } else {
            l_ptr = p_curr;
            l_end = p_curr;
        }
    } else {
        // RAW Literals
        l_ptr = p_curr;
        l_end = p_curr + (size_t)(desc[0].sizes & 0xFFFFFFFF);
    }

    p_curr += (size_t)(desc[0].sizes & 0xFFFFFFFF);

    size_t sz_tokens = (size_t)(desc[1].sizes & 0xFFFFFFFF);
    size_t sz_offsets = (size_t)(desc[2].sizes & 0xFFFFFFFF);
    size_t sz_extras = (size_t)(desc[3].sizes & 0xFFFFFFFF);

    const uint8_t* ptr_tokens = p_curr;
    const uint8_t* ptr_offsets = ptr_tokens + sz_tokens;
    const uint8_t* ptr_extras = ptr_offsets + sz_offsets;

    if (UNLIKELY(sz_tokens < gh.n_sequences || sz_offsets < (size_t)gh.n_sequences * 2 ||
                 ptr_extras + sz_extras > src + src_size)) {
        return -1;
    }

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    const uint8_t* d_end_safe = d_end - 64;
    const uint8_t* t_ptr = ptr_tokens;
    const uint8_t* o_ptr = ptr_offsets;
    const uint8_t* e_ptr = ptr_extras;

    uint32_t n_seq = gh.n_sequences;

#if defined(__APPLE__) && defined(__aarch64__)
    while (n_seq >= 4) {
        // --- Token & Offset Loading (4x) ---
        // Batch read 4 tokens (4 bytes) and 4 offsets (8 bytes)
        uint32_t tokens = zxc_le32(t_ptr);
        t_ptr += 4;
        uint64_t offsets = zxc_le64(o_ptr);
        o_ptr += 8;

        uint8_t token1 = (uint8_t)(tokens);
        uint32_t ll1 = token1 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml1 = token1 & ZXC_TOKEN_ML_MASK;
        uint32_t off1 = (uint32_t)(offsets & 0xFFFF);

        uint8_t token2 = (uint8_t)(tokens >> 8);
        uint32_t ll2 = token2 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml2 = token2 & ZXC_TOKEN_ML_MASK;
        uint32_t off2 = (uint32_t)((offsets >> 16) & 0xFFFF);

        uint8_t token3 = (uint8_t)(tokens >> 16);
        uint32_t ll3 = token3 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml3 = token3 & ZXC_TOKEN_ML_MASK;
        uint32_t off3 = (uint32_t)((offsets >> 32) & 0xFFFF);

        uint8_t token4 = (uint8_t)(tokens >> 24);
        uint32_t ll4 = token4 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml4 = token4 & ZXC_TOKEN_ML_MASK;
        uint32_t off4 = (uint32_t)((offsets >> 48) & 0xFFFF);

        // --- Extra Length Handling ---
        if (UNLIKELY(ll1 == 15)) {
            ll1 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml1 == 15)) {
            ml1 = zxc_read_vbyte(&e_ptr);
        }
        ml1 += ZXC_LZ_MIN_MATCH;

        if (UNLIKELY(ll2 == 15)) {
            ll2 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml2 == 15)) {
            ml2 = zxc_read_vbyte(&e_ptr);
        }
        ml2 += ZXC_LZ_MIN_MATCH;

        if (UNLIKELY(ll3 == 15)) {
            ll3 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml3 == 15)) {
            ml3 = zxc_read_vbyte(&e_ptr);
        }
        ml3 += ZXC_LZ_MIN_MATCH;

        if (UNLIKELY(ll4 == 15)) {
            ll4 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml4 == 15)) {
            ml4 = zxc_read_vbyte(&e_ptr);
        }
        ml4 += ZXC_LZ_MIN_MATCH;

        ZXC_PREFETCH_READ(l_ptr + 256);

        if (LIKELY(d_ptr + ll1 + ml1 + ll2 + ml2 + ll3 + ml3 + ll4 + ml4 < d_end_safe)) {
            // Sequence 1
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll1;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll1;
                l_ptr += ll1;
                uint8_t* match_src = d_ptr - off1;
                if (off1 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml1;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml1;
                } else {
                    if (off1 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml1);
                        d_ptr += ml1;
                    } else {
                        for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml1;
                    }
                }
            }
            // Sequence 2
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll2;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll2;
                l_ptr += ll2;
                uint8_t* match_src = d_ptr - off2;
                if (off2 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml2;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml2;
                } else {
                    if (off2 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml2);
                        d_ptr += ml2;
                    } else {
                        for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml2;
                    }
                }
            }
            // Sequence 3
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll3;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll3;
                l_ptr += ll3;
                uint8_t* match_src = d_ptr - off3;
                if (off3 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml3;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml3;
                } else {
                    if (off3 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml3);
                        d_ptr += ml3;
                    } else {
                        for (size_t i = 0; i < ml3; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml3;
                    }
                }
            }
            // Sequence 4
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll4;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll4;
                l_ptr += ll4;
                uint8_t* match_src = d_ptr - off4;
                if (off4 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml4;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml4;
                } else {
                    if (off4 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml4);
                        d_ptr += ml4;
                    } else {
                        for (size_t i = 0; i < ml4; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml4;
                    }
                }
            }
        } else {
            // Safe path for Sequence 1
            if (UNLIKELY(d_ptr + ll1 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll1);
            l_ptr += ll1;
            d_ptr += ll1;
            const uint8_t* match_src1 = d_ptr - off1;
            if (UNLIKELY(match_src1 < dst || d_ptr + ml1 > d_end)) return -1;
            if (off1 < ml1)
                for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src1[i];
            else
                ZXC_MEMCPY(d_ptr, match_src1, ml1);
            d_ptr += ml1;

            // Safe path for Sequence 2
            if (UNLIKELY(d_ptr + ll2 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll2);
            l_ptr += ll2;
            d_ptr += ll2;
            const uint8_t* match_src2 = d_ptr - off2;
            if (UNLIKELY(match_src2 < dst || d_ptr + ml2 > d_end)) return -1;
            if (off2 < ml2)
                for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src2[i];
            else
                ZXC_MEMCPY(d_ptr, match_src2, ml2);
            d_ptr += ml2;

            // Safe path for Sequence 3
            if (UNLIKELY(d_ptr + ll3 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll3);
            l_ptr += ll3;
            d_ptr += ll3;
            const uint8_t* match_src3 = d_ptr - off3;
            if (UNLIKELY(match_src3 < dst || d_ptr + ml3 > d_end)) return -1;
            if (off3 < ml3)
                for (size_t i = 0; i < ml3; i++) d_ptr[i] = match_src3[i];
            else
                ZXC_MEMCPY(d_ptr, match_src3, ml3);
            d_ptr += ml3;

            // Safe path for Sequence 4
            if (UNLIKELY(d_ptr + ll4 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll4);
            l_ptr += ll4;
            d_ptr += ll4;
            const uint8_t* match_src4 = d_ptr - off4;
            if (UNLIKELY(match_src4 < dst || d_ptr + ml4 > d_end)) return -1;
            if (off4 < ml4)
                for (size_t i = 0; i < ml4; i++) d_ptr[i] = match_src4[i];
            else
                ZXC_MEMCPY(d_ptr, match_src4, ml4);
            d_ptr += ml4;
        }
        n_seq -= 4;
    }

    while (n_seq >= 2) {
        uint8_t token1 = *t_ptr++;
        uint32_t ll1 = token1 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml1 = token1 & ZXC_TOKEN_ML_MASK;
        uint32_t off1 = zxc_le16(o_ptr);
        o_ptr += 2;
        if (UNLIKELY(ll1 == 15)) {
            ll1 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml1 == 15)) {
            ml1 = zxc_read_vbyte(&e_ptr);
        }
        ml1 += ZXC_LZ_MIN_MATCH;

        uint8_t token2 = *t_ptr++;
        uint32_t ll2 = token2 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml2 = token2 & ZXC_TOKEN_ML_MASK;
        uint32_t off2 = zxc_le16(o_ptr);
        o_ptr += 2;

        if (UNLIKELY(ll2 == 15)) {
            ll2 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml2 == 15)) {
            ml2 = zxc_read_vbyte(&e_ptr);
        }
        ml2 += ZXC_LZ_MIN_MATCH;

        ZXC_PREFETCH_READ(l_ptr + 128);

        if (LIKELY(d_ptr + ll1 + ml1 + ll2 + ml2 < d_end_safe)) {
            // --- Sequence 1 ---
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll1;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll1;
                l_ptr += ll1;
                uint8_t* match_src = d_ptr - off1;
                if (off1 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml1;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml1;
                } else {
                    if (off1 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml1);
                        d_ptr += ml1;
                    } else {
                        for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml1;
                    }
                }
            }
            // --- Sequence 2 ---
            {
                const uint8_t* src_lit = l_ptr;
                uint8_t* dst_lit = d_ptr;
                const uint8_t* target_lit_end = d_ptr + ll2;
                do {
                    zxc_copy16(dst_lit, src_lit);
                    dst_lit += 16;
                    src_lit += 16;
                } while (dst_lit < target_lit_end);
                d_ptr += ll2;
                l_ptr += ll2;
                uint8_t* match_src = d_ptr - off2;
                if (off2 >= 16) {
                    uint8_t* out = d_ptr;
                    const uint8_t* target_match_end = d_ptr + ml2;
                    do {
                        zxc_copy16(out, match_src);
                        out += 16;
                        match_src += 16;
                    } while (out < target_match_end);
                    d_ptr += ml2;
                } else {
                    if (off2 == 1) {
                        ZXC_MEMSET(d_ptr, match_src[0], ml2);
                        d_ptr += ml2;
                    } else {
                        for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src[i];
                        d_ptr += ml2;
                    }
                }
            }
        } else {
            // Safe path for Sequence 1
            if (UNLIKELY(d_ptr + ll1 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll1);
            l_ptr += ll1;
            d_ptr += ll1;
            const uint8_t* match_src1 = d_ptr - off1;
            if (UNLIKELY(match_src1 < dst || d_ptr + ml1 > d_end)) return -1;
            if (off1 < ml1)
                for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src1[i];
            else
                ZXC_MEMCPY(d_ptr, match_src1, ml1);
            d_ptr += ml1;

            // Safe path for Sequence 2
            if (UNLIKELY(d_ptr + ll2 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll2);
            l_ptr += ll2;
            d_ptr += ll2;
            const uint8_t* match_src2 = d_ptr - off2;
            if (UNLIKELY(match_src2 < dst || d_ptr + ml2 > d_end)) return -1;
            if (off2 < ml2)
                for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src2[i];
            else
                ZXC_MEMCPY(d_ptr, match_src2, ml2);
            d_ptr += ml2;
        }
        n_seq -= 2;
    }
#else
    while (n_seq >= 2) {
        // Optimization: Load 2 tokens (2 bytes) and 2 offsets (4 bytes) at once
        uint16_t tokens = zxc_le16(t_ptr);
        t_ptr += 2;
        uint32_t offsets = zxc_le32(o_ptr);
        o_ptr += 4;

        uint8_t token1 = (uint8_t)(tokens);
        uint32_t ll1 = token1 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml1 = token1 & ZXC_TOKEN_ML_MASK;
        uint32_t off1 = offsets & 0xFFFF;

        uint8_t token2 = (uint8_t)(tokens >> 8);
        uint32_t ll2 = token2 >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml2 = token2 & ZXC_TOKEN_ML_MASK;
        uint32_t off2 = offsets >> 16;

        if (UNLIKELY(ll1 == 15)) {
            ll1 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml1 == 15)) {
            ml1 = zxc_read_vbyte(&e_ptr);
        }
        ml1 += ZXC_LZ_MIN_MATCH;

        if (UNLIKELY(ll2 == 15)) {
            ll2 = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml2 == 15)) {
            ml2 = zxc_read_vbyte(&e_ptr);
        }
        ml2 += ZXC_LZ_MIN_MATCH;

        ZXC_PREFETCH_READ(l_ptr + 128);

        if (LIKELY(d_ptr + ll1 + ml1 + ll2 + ml2 < d_end_safe)) {
            // --- Sequence 1 ---
            {
                // Wild Copy for Literals
                zxc_copy16(d_ptr, l_ptr);
                if (UNLIKELY(ll1 > 16)) {
                    ZXC_MEMCPY(d_ptr + 16, l_ptr + 16, ll1 - 16);
                }
                d_ptr += ll1;
                l_ptr += ll1;

                uint8_t* match_src = d_ptr - off1;
                // Optimized Match Copy
                if (LIKELY(off1 >= 16)) {
                    zxc_copy16(d_ptr, match_src);  // Wild copy
                    if (UNLIKELY(ml1 > 16)) {
                        uint8_t* out = d_ptr + 16;
                        const uint8_t* target_match_end = d_ptr + ml1;
                        match_src += 16;
                        do {
                            zxc_copy16(out, match_src);
                            out += 16;
                            match_src += 16;
                        } while (out < target_match_end);
                    }
                    d_ptr += ml1;
                } else {
                    switch (off1) {
                        case 1:
                            ZXC_MEMSET(d_ptr, match_src[0], ml1);
                            d_ptr += ml1;
                            break;
                        default:
                            for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src[i];
                            d_ptr += ml1;
                            break;
                    }
                }
            }
            // --- Sequence 2 ---
            {
                // Wild Copy for Literals
                zxc_copy16(d_ptr, l_ptr);
                if (UNLIKELY(ll2 > 16)) {
                    ZXC_MEMCPY(d_ptr + 16, l_ptr + 16, ll2 - 16);
                }
                d_ptr += ll2;
                l_ptr += ll2;

                uint8_t* match_src = d_ptr - off2;
                // Optimized Match Copy
                if (LIKELY(off2 >= 16)) {
                    zxc_copy16(d_ptr, match_src);  // Wild copy
                    if (UNLIKELY(ml2 > 16)) {
                        uint8_t* out = d_ptr + 16;
                        const uint8_t* target_match_end = d_ptr + ml2;
                        match_src += 16;
                        do {
                            zxc_copy16(out, match_src);
                            out += 16;
                            match_src += 16;
                        } while (out < target_match_end);
                    }
                    d_ptr += ml2;
                } else {
                    switch (off2) {
                        case 1:
                            ZXC_MEMSET(d_ptr, match_src[0], ml2);
                            d_ptr += ml2;
                            break;
                        default:
                            for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src[i];
                            d_ptr += ml2;
                            break;
                    }
                }
            }
        } else {
            // Safe path for Sequence 1
            if (UNLIKELY(d_ptr + ll1 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll1);
            l_ptr += ll1;
            d_ptr += ll1;
            const uint8_t* match_src1 = d_ptr - off1;
            if (UNLIKELY(match_src1 < dst || d_ptr + ml1 > d_end)) return -1;
            if (off1 < ml1)
                for (size_t i = 0; i < ml1; i++) d_ptr[i] = match_src1[i];
            else
                ZXC_MEMCPY(d_ptr, match_src1, ml1);
            d_ptr += ml1;

            // Safe path for Sequence 2
            if (UNLIKELY(d_ptr + ll2 > d_end)) return -1;
            ZXC_MEMCPY(d_ptr, l_ptr, ll2);
            l_ptr += ll2;
            d_ptr += ll2;
            const uint8_t* match_src2 = d_ptr - off2;
            if (UNLIKELY(match_src2 < dst || d_ptr + ml2 > d_end)) return -1;
            if (off2 < ml2)
                for (size_t i = 0; i < ml2; i++) d_ptr[i] = match_src2[i];
            else
                ZXC_MEMCPY(d_ptr, match_src2, ml2);
            d_ptr += ml2;
        }
        n_seq -= 2;
    }
#endif
    if (n_seq) {
        uint8_t token = *t_ptr++;
        uint32_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint32_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t off = zxc_le16(o_ptr);

        if (UNLIKELY(ll == 15)) {
            ll = zxc_read_vbyte(&e_ptr);
        }
        if (UNLIKELY(ml == 15)) {
            ml = zxc_read_vbyte(&e_ptr);
        }
        ml += ZXC_LZ_MIN_MATCH;

        if (UNLIKELY(d_ptr + ll > d_end)) return -1;

        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        const uint8_t* match_src = d_ptr - off;
        if (UNLIKELY(match_src < dst || d_ptr + ml > d_end)) return -1;

        if (off < ml) {
            for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
        } else {
            ZXC_MEMCPY(d_ptr, match_src, ml);
        }
        d_ptr += ml;
    }

    size_t generated = d_ptr - dst;
    if (generated < expected_raw_size) {
        size_t rem = expected_raw_size - generated;

        if (UNLIKELY(d_ptr + rem > d_end || l_ptr + rem > l_end)) return -1;

        ZXC_MEMCPY(d_ptr, l_ptr, rem);
        d_ptr += rem;
    }

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

    if (type == ZXC_BLOCK_GNR) {
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
    if (!src || !dst || src_size < ZXC_FILE_HEADER_SIZE) return 0;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;

    zxc_cctx_t ctx;
    if (zxc_cctx_init(&ctx, ZXC_CHUNK_SIZE, 0, 0, checksum_enabled) != 0) return 0;

    // File header verification
    if (zxc_read_file_header(ip, src_size) != 0) {
        zxc_cctx_free(&ctx);
        return 0;
    }
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
        if (res < 0) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        int has_crc = (bh.block_flags & ZXC_BLOCK_FLAG_CHECKSUM);
        size_t header_overhead = ZXC_BLOCK_HEADER_SIZE + (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);

        ip += header_overhead + bh.comp_size;
        op += res;
    }

    zxc_cctx_free(&ctx);
    return (size_t)(op - op_start);
}