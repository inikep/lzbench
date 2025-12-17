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

#define ZXC_NUM_FRAME_SIZE \
    128  // Maximum number of frames that can be processed in a single compression operation.
#define ZXC_EPOCH_BITS \
    14  // Number of bits reserved for epoch tracking in compressed pointers.
        // Derived from chunk size: 2^18 = ZXC_CHUNK_SIZE => 32 - 18 = 14 bits.
#define ZXC_OFFSET_MASK              \
    ((1U << (32 - ZXC_EPOCH_BITS)) - \
     1)  // Mask to extract the offset bits from a compressed pointer.
#define ZXC_MAX_EPOCH \
    (1U << ZXC_EPOCH_BITS)  // Maximum number of epochs supported by the compression system.

#if defined(ZXC_USE_AVX2)
/**
 * @brief Reduces a 256-bit integer vector to a single scalar by finding the maximum unsigned 32-bit
 * integer element.
 *
 * This function performs a horizontal reduction across the 8 packed 32-bit unsigned integers
 * in the source vector to determine the maximum value.
 *
 * @param v The 256-bit vector containing 8 unsigned 32-bit integers.
 * @return The maximum unsigned 32-bit integer found in the vector.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_mm256_reduce_max_epu32(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);        // Extract the lower 128 bits
    __m128i vhigh = _mm256_extracti128_si256(v, 1);  // Extract the upper 128 bits
    vlow = _mm_max_epu32(vlow, vhigh);               // Element-wise max of lower and upper halves
    __m128i vshuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(1, 0, 3, 2));  // Shuffle to swap pairs
    vlow = _mm_max_epu32(vlow, vshuf);  // Max of original and swapped
    vshuf =
        _mm_shuffle_epi32(vlow, _MM_SHUFFLE(2, 3, 0, 1));  // Shuffle to bring remaining candidates
    vlow = _mm_max_epu32(vlow, vshuf);                     // Final max comparison
    return (uint32_t)_mm_cvtsi128_si32(vlow);              // Extract the scalar result
}
#endif

/**
 * @brief Encodes a block of numerical data using delta encoding and
 * bit-packing.
 *
 * This function compresses a source buffer of 32-bit integers. It processes the
 * data in frames defined by `ZXC_NUM_FRAME_SIZE`.
 *
 * **Algorithm Steps:**
 * 1. **Delta Encoding:** Calculates `delta = value[i] - value[i-1]`. This
 * reduces the magnitude of numbers if the data is sequential or correlated.
 *    - **SIMD Optimization:** Uses AVX2 (`_mm256_sub_epi32`) to compute deltas
 * for 8 integers at once.
 * 2. **ZigZag Encoding:** Maps signed deltas to unsigned integers (`(n << 1) ^
 * (n >> 31)`). This ensures small negative numbers become small positive
 * numbers (e.g., -1 -> 1, 1 -> 2).
 * 3. **Bit Width Calculation:** Finds the maximum value in the frame to
 * determine the minimum number of bits (`b`) needed to represent all deltas.
 * 4. **Bit Packing:** Packs the ZigZag-encoded deltas into a compact bitstream
 *    using `b` bits per value.
 *
 * @param[in] src Pointer to the source buffer containing raw 32-bit integer data.
 * @param[in] src_size Size of the source buffer in bytes. Must be a multiple of 4
 * and non-zero.
 * @param[out] dst Pointer to the destination buffer where compressed data will be
 * written.
 * @param[in] dst_cap Capacity of the destination buffer in bytes.
 * @param[out] out_sz Pointer to a variable where the total size of the compressed
 * output will be stored.
 * @param[in] crc_val The pre-calculated XXH3 value (if checksum is enabled).
 *
 * @return 0 on success, or -1 on failure (e.g., invalid input size, destination
 * buffer too small).
 */
static int zxc_encode_block_num(const zxc_cctx_t* ctx, const uint8_t* src, size_t src_size,
                                uint8_t* dst, size_t dst_cap, size_t* out_sz, uint64_t crc_val) {
    if (UNLIKELY(src_size % 4 != 0 || src_size == 0)) return -1;
    int chk = ctx->checksum_enabled;

    size_t count = src_size / 4;
    size_t h_gap = ZXC_BLOCK_HEADER_SIZE + (chk ? ZXC_BLOCK_CHECKSUM_SIZE : 0);

    if (UNLIKELY(dst_cap < h_gap + ZXC_NUM_HEADER_BINARY_SIZE)) return -1;

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_NUM, .raw_size = (uint32_t)src_size};
    uint8_t* p_curr = dst + h_gap;
    size_t rem = dst_cap - h_gap;
    zxc_num_header_t nh = {.n_values = count, .frame_size = ZXC_NUM_FRAME_SIZE};

    int hs = zxc_write_num_header(p_curr, rem, &nh);
    if (UNLIKELY(hs < 0)) return -1;

    p_curr += hs;
    rem -= hs;

    uint32_t deltas[ZXC_NUM_FRAME_SIZE];
    const uint8_t* in_ptr = src;
    uint32_t prev = 0;

    for (size_t i = 0; i < count; i += ZXC_NUM_FRAME_SIZE) {
        size_t frames = (count - i < ZXC_NUM_FRAME_SIZE) ? (count - i) : ZXC_NUM_FRAME_SIZE;
        uint32_t max_d = 0, base = prev;
        size_t j = 0;

#if defined(ZXC_USE_AVX512)
        if (frames >= 16) {
            __m512i v_max_accum = _mm512_setzero_si512();  // Initialize max accumulator to 0

            for (; j < (frames & ~15); j += 16) {
                if (UNLIKELY(i == 0 && j == 0)) goto _scalar;

                // Load 16 consecutive integers
                __m512i vc = _mm512_loadu_si512((const void*)(in_ptr + j * 4));
                // Load 16 integers offset by -1 to get previous values
                __m512i vp = _mm512_loadu_si512((const void*)(in_ptr + j * 4 - 4));

                __m512i diff = _mm512_sub_epi32(vc, vp);  // Compute deltas: curr - prev

                // ZigZag encode: (diff << 1) ^ (diff >> 31)
                __m512i zigzag =
                    _mm512_xor_si512(_mm512_slli_epi32(diff, 1), _mm512_srai_epi32(diff, 31));

                _mm512_storeu_si512((void*)&deltas[j], zigzag);  // Store results
                v_max_accum =
                    _mm512_max_epu32(v_max_accum, zigzag);  // Update max value seen so far
            }
            max_d = _mm512_reduce_max_epu32(v_max_accum);  // Horizontal max reduction

            if (j > 0) prev = zxc_le32(in_ptr + (j - 1) * 4);
        }
#elif defined(ZXC_USE_AVX2)
        if (frames >= 8) {
            __m256i v_max_accum = _mm256_setzero_si256();  // Initialize max accumulator to 0

            for (; j < (frames & ~7); j += 8) {
                if (UNLIKELY(i == 0 && j == 0)) goto _scalar;

                // Load 8 consecutive integers
                __m256i vc = _mm256_loadu_si256((const __m256i*)(in_ptr + j * 4));
                // Load 8 integers offset by -1
                __m256i vp = _mm256_loadu_si256((const __m256i*)(in_ptr + j * 4 - 4));

                __m256i diff = _mm256_sub_epi32(vc, vp);  // Compute deltas

                // ZigZag encode: (diff << 1) ^ (diff >> 31)
                __m256i zigzag =
                    _mm256_xor_si256(_mm256_slli_epi32(diff, 1), _mm256_srai_epi32(diff, 31));
                _mm256_storeu_si256((__m256i*)&deltas[j], zigzag);    // Store results
                v_max_accum = _mm256_max_epu32(v_max_accum, zigzag);  // Update max accumulator
            }

            max_d = zxc_mm256_reduce_max_epu32(v_max_accum);  // Horizontal max reduction

            if (j > 0) {
                prev = zxc_le32(in_ptr + (j - 1) * 4);
            }
        }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
        // NEON processes 128-bit vectors (4 uint32 integers)
        if (frames >= 4) {
            uint32x4_t v_max_accum = vdupq_n_u32(0);  // Initialize vector with zeros

            for (; j < (frames & ~3); j += 4) {
                if (UNLIKELY(i == 0 && j == 0)) goto _scalar;

                // Load 4 32-bit integers
                uint32x4_t vc = vld1q_u32((const uint32_t*)(in_ptr + j * 4));
                uint32x4_t vp = vld1q_u32((const uint32_t*)(in_ptr + j * 4 - 4));

                uint32x4_t diff = vsubq_u32(vc, vp);  // Calc deltas

                // ZigZag encode: (diff << 1) ^ (diff >> 31)
                uint32x4_t z1 = vshlq_n_u32(diff, 1);
                // Arithmetic shift right to duplicate sign bit
                uint32x4_t z2 = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(diff), 31));
                uint32x4_t zigzag = veorq_u32(z1, z2);

                vst1q_u32(&deltas[j], zigzag);                 // Store results
                v_max_accum = vmaxq_u32(v_max_accum, zigzag);  // Update max accumulator
            }

#if defined(ZXC_USE_NEON64)
            max_d = vmaxvq_u32(v_max_accum);  // Reduce vector to single max value (AArch64)
#else
            // NEON 32-bit (ARMv7) fallback for horizontal max using standard shifts
            // Reduce 4 elements -> 2
            uint32x4_t v_swap =
                vextq_u32(v_max_accum, v_max_accum, 2);  // Swap low/high 64-bit halves
            uint32x4_t v_max2 = vmaxq_u32(v_max_accum, v_swap);
            // Reduce 2 -> 1
            v_swap = vextq_u32(v_max2, v_max2, 1);  // Shift by 32 bits
            uint32x4_t v_max1 = vmaxq_u32(v_max2, v_swap);
            max_d = vgetq_lane_u32(v_max1, 0);
#endif

            if (j > 0) prev = zxc_le32(in_ptr + (j - 1) * 4);
        }
#endif
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32)
    _scalar:
#ifndef _MSC_VER
        __attribute__((unused));
#endif
#endif
        for (; j < frames; j++) {
            uint32_t v = zxc_le32(in_ptr + j * 4);
            uint32_t diff = zxc_zigzag_encode((int32_t)(v - prev));
            deltas[j] = diff;
            if (diff > max_d) max_d = diff;
            prev = v;
        }
        in_ptr += frames * 4;

        uint8_t bits = zxc_highbit32(max_d);
        size_t packed = ((frames * bits) + 7) / 8;
        if (UNLIKELY(rem < 16 + packed)) return -1;

        zxc_store_le16(p_curr, (uint16_t)frames);
        zxc_store_le16(p_curr + 2, bits);
        zxc_store_le64(p_curr + 4, (uint64_t)base);
        zxc_store_le32(p_curr + 12, (uint32_t)packed);

        p_curr += 16;
        rem -= 16;

        int pb = zxc_bitpack_stream_32(deltas, frames, p_curr, rem, bits);
        if (UNLIKELY(pb < 0)) return -1;
        p_curr += pb;
        rem -= pb;
    }

    uint32_t p_sz = (uint32_t)(p_curr - (dst + h_gap));
    if (chk)
        bh.block_flags |= ZXC_BLOCK_FLAG_CHECKSUM;
    else
        bh.block_flags &= ~ZXC_BLOCK_FLAG_CHECKSUM;

    bh.comp_size = p_sz;
    int hw = zxc_write_block_header(dst, dst_cap, &bh);

    if (chk) zxc_store_le64(dst + hw, crc_val);
    *out_sz = hw + (chk ? ZXC_BLOCK_CHECKSUM_SIZE : 0) + p_sz;
    return 0;
}

/**
 * @brief Encodes a data block using the General (GNR) compression format.
 *
 * This function implements the core LZ77 compression logic. It dynamically
 * adjusts compression parameters (search depth, lazy matching strategy, and
 * step skipping) based on the compression level configured in the context.
 *
 * **LZ77 Implementation Details:**
 * 1. **Hash Chain:** Uses a hash table (`ctx->hash_table`) to find potential
 * match positions. Collisions are handled via a `chain_table`, allowing us to
 * search deeper into the history for a better match.
 * 2. **Lazy Matching:** If a match is found, we check the *next* byte to see if
 *    it produces a longer match. If so, we output a literal and take the better
 * match. This is enabled for levels >= 3.
 * 3. **Step Skipping:** For lower levels (1-3), we skip bytes when updating the
 *    hash table to increase speed (`step > 1`). For levels 4+, we process every
 * byte to maximize compression ratio.
 * 4. **SIMD Match Finding:** Uses AVX2/AVX512/NEON to compare 32/64 bytes at a
 * time during match length calculation, significantly speeding up long match
 * verification.
 * 5. **RLE Detection:** Analyzes literals to see if Run-Length Encoding would
 * be beneficial (saving > 10% space).
 *
 * The encoding process consists of:
 * 1. **LZ77 Parsing**: The function iterates through the source data,
 * maintaining a hash chain to find repeated patterns (matches). It supports
 * "Lazy Matching" for higher compression levels to optimize match selection.
 * 2. **Sequence Storage**: Matches are converted into sequences consisting of
 *    literal lengths, match lengths, and offsets.
 * 3. **Bitpacking & Serialization**: The sequences are analyzed to determine
 * optimal bit-widths. The function then writes the block header, encodes
 * literals (using Raw or RLE encoding), and bit-packs the sequence streams into
 * the destination buffer.
 *
 * @param[in,out] ctx       Pointer to the compression context containing hash tables
 * and configuration.
 * @param[in] src       Pointer to the input source data.
 * @param[in] src_size  Size of the input data in bytes.
 * @param[out] dst       Pointer to the destination buffer where compressed data will
 * be written.
 * @param[in] dst_cap   Maximum capacity of the destination buffer.
 * @param[out] out_sz    [Out] Pointer to a variable that will receive the total size
 * of the compressed output.
 * @param[in] crc_val   The pre-calculated XXH3 value (if checksum is enabled).
 *
 * @return 0 on success, or -1 if an error occurs (e.g., buffer overflow).
 */
static int zxc_encode_block_gnr(zxc_cctx_t* ctx, const uint8_t* src, size_t src_size, uint8_t* dst,
                                size_t dst_cap, size_t* out_sz, uint64_t crc_val) {
    int level = ctx->compression_level;
    int chk = ctx->checksum_enabled;

    int search_depth;
    int use_lazy;
    int sufficient_len = 256;

    uint32_t step_base = 1;
    uint32_t step_shift = 31;

    if (level <= 2) {
        search_depth = 4;
        use_lazy = 0;
        sufficient_len = 16;
        step_base = 3;
        step_shift = 3;
    } else if (level <= 3) {
        search_depth = 4;
        use_lazy = 1;
        sufficient_len = 32;
        step_base = 1;
        step_shift = 4;
    } else if (level <= 4) {
        search_depth = 4;
        use_lazy = 1;
        sufficient_len = 32;
        step_base = 1;
        step_shift = 5;
    } else {
        search_depth = 64;
        use_lazy = 1;
    }

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ZXC_MAX_EPOCH)) {
        ZXC_MEMSET(ctx->hash_table, 0, 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ctx->epoch = 1;
    }
    const uint32_t epoch_mark = ctx->epoch << (32 - ZXC_EPOCH_BITS);
    const uint8_t *ip = src, *iend = src + src_size, *anchor = ip, *mflimit = iend - 12;

    uint32_t* hash_table = ctx->hash_table;
    uint16_t* chain_table = ctx->chain_table;
    uint8_t* literals = ctx->literals;

    uint32_t seq_c = 0;
    size_t lit_c = 0;

    // --- TOKEN ENCODING PRE-INIT ---
    uint8_t* buf_tokens = ctx->buf_tokens;
    uint16_t* buf_offsets = ctx->buf_offsets;
    uint32_t* buf_extras = ctx->buf_extras;
    size_t n_extras = 0;
    size_t vbyte_size = 0;

    while (LIKELY(ip < mflimit)) {
        size_t dist = (size_t)(ip - anchor);
        size_t step = step_base + (dist >> step_shift);

        if (UNLIKELY(ip + step >= mflimit)) step = 1;

        ZXC_PREFETCH_READ(ip + step * 4 + 64);

        uint32_t cur_val = zxc_le32(ip);
        uint32_t h = zxc_hash_func(cur_val) & (ZXC_LZ_HASH_SIZE - 1);
        int32_t cur_pos = (uint32_t)(ip - src);

        uint32_t raw_head = hash_table[2 * h];
        uint32_t match_idx =
            (raw_head & ~ZXC_OFFSET_MASK) == epoch_mark ? (raw_head & ZXC_OFFSET_MASK) : 0;

        hash_table[2 * h] = epoch_mark | cur_pos;
        // cppcheck-suppress knownConditionTrueFalse ; false positive
        if (match_idx > 0 && (cur_pos - match_idx) < 0x10000)
            chain_table[cur_pos] = (uint16_t)(cur_pos - match_idx);
        else
            chain_table[cur_pos] = 0;

        const uint8_t* best_ref = NULL;
        uint32_t best_len = ZXC_LZ_MIN_MATCH - 1;

        int attempts = search_depth;
        while (match_idx > 0 && attempts-- >= 0) {
            if (cur_pos - match_idx >= ZXC_LZ_MAX_DIST) break;

            const uint8_t* ref = src + match_idx;

            if (zxc_le32(ref) == cur_val && ref[best_len] == ip[best_len]) {
                uint32_t mlen = 4;
#if defined(ZXC_USE_AVX512)
                const uint8_t* limit_64 = iend - 64;
                while (ip + mlen < limit_64) {
                    // AVX-512 Optimization: Compare 64 bytes at once
                    __m512i v_src = _mm512_loadu_si512((const void*)(ip + mlen));
                    __m512i v_ref = _mm512_loadu_si512((const void*)(ref + mlen));

                    // _mm512_cmpeq_epi8_mask returns a 64-bit mask where each bit represents a byte
                    // match
                    __mmask64 mask = _mm512_cmpeq_epi8_mask(v_src, v_ref);

                    // If mask is all 1s (UINT64_MAX), all 64 bytes match
                    if (mask == 0xFFFFFFFFFFFFFFFF)
                        mlen += 64;
                    else {
                        // Count trailing zeros of negated mask to find first mismatch
                        mlen += (uint32_t)zxc_ctz64(~mask);
                        goto _match_len_done;
                    }
                }
#elif defined(ZXC_USE_AVX2)
                const uint8_t* limit_32 = iend - 32;
                while (ip + mlen < limit_32) {
                    // AVX2 Optimization: Compare 32 bytes at once
                    __m256i v_src = _mm256_loadu_si256((const __m256i*)(ip + mlen));
                    __m256i v_ref = _mm256_loadu_si256((const __m256i*)(ref + mlen));
                    __m256i v_cmp = _mm256_cmpeq_epi8(v_src, v_ref);
                    // _mm256_movemask_epi8 creates a 32-bit mask from the most significant bit of
                    // each byte
                    uint32_t mask = (uint32_t)_mm256_movemask_epi8(v_cmp);
                    // If mask is all 1s (0xFFFFFFFF), all 32 bytes match
                    if (mask == 0xFFFFFFFF)
                        mlen += 32;
                    else {
                        // Count trailing zeros of negated mask to find first mismatch
                        mlen += zxc_ctz32(~mask);
                        goto _match_len_done;
                    }
                }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
                const uint8_t* limit_16 = iend - 16;
                while (ip + mlen < limit_16) {
                    // NEON Optimization: Compare 16 bytes at once
                    uint8x16_t v_src = vld1q_u8(ip + mlen);
                    uint8x16_t v_ref = vld1q_u8(ref + mlen);
                    // vceqq_u8 performs byte-wise equality comparison, result is 0xFF for equal,
                    // 0x00 for not equal
                    uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);

                    // Check if all bytes are equal (min value of comparison result is 0xFF)
#if defined(ZXC_USE_NEON64)
                    // AArch64 unified min
                    if (vminvq_u8(v_cmp) == 0xFF)
                        mlen += 16;
                    else {
                        // NEON lacks a direct movemask instruction like x86.
                        // We invert the comparison result (0xFF -> 0x00, 0x00 -> 0xFF)
                        // Then we can use ctz on the 64-bit lanes to find the first non-zero byte.
                        uint8x16_t v_diff = vmvnq_u8(v_cmp);
                        uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(v_diff), 0);
                        if (lo != 0)
                            mlen += (zxc_ctz64(lo) >> 3);  // Divide by 8 to get byte index
                        else {
                            uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(v_diff), 1);
                            mlen += 8 + (zxc_ctz64(hi) >> 3);
                        }
                        goto _match_len_done;
                    }
#else
                    // NEON 32-bit (ARMv7) fallback for min scan
                    uint8x16_t p1 = vpminq_u8(v_cmp, v_cmp);
                    uint8x16_t p2 = vpminq_u8(p1, p1);
                    uint8x16_t p3 = vpminq_u8(p2, p2);
                    // Now reduced to 2 bytes, convert to scalar
                    uint8_t min_val = vgetq_lane_u8(p3, 0);
                    min_val = min_val < vgetq_lane_u8(p3, 8) ? min_val : vgetq_lane_u8(p3, 8);

                    if (min_val == 0xFF)
                        mlen += 16;
                    else {
                        uint8x16_t v_diff = vmvnq_u8(v_cmp);
                        // Access as 32-bit lanes to reconstruct 64-bit values or check directly
                        // Reconstructing 64-bit for compatibility with zxc_ctz64 usage
                        uint64_t lo =
                            (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 0) |
                            ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 1) << 32);

                        if (lo != 0)
                            mlen += (zxc_ctz64(lo) >> 3);
                        else {
                            uint64_t hi =
                                (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 2) |
                                ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 3) << 32);
                            mlen += 8 + (zxc_ctz64(hi) >> 3);
                        }
                        goto _match_len_done;
                    }
#endif
                }
#endif
                const uint8_t* limit_8 = iend - 8;
                while (ip + mlen < limit_8) {
                    if (zxc_le64(ip + mlen) == zxc_le64(ref + mlen))
                        mlen += 8;
                    else {
                        // XOR to find differing bits, trailing zeros / 8 = byte index
                        mlen += (zxc_ctz64(zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen)) >> 3);
                        goto _match_len_done;
                    }
                }
                while (ip + mlen < iend && ref[mlen] == ip[mlen]) mlen++;

            _match_len_done:
                if (mlen > best_len) {
                    best_len = mlen;
                    best_ref = ref;
                    if (best_len >= (uint32_t)sufficient_len) break;  // Sufficient match found

                    if (ip + best_len >= iend) break;  // Prevent overruns
                }
            }
            uint16_t delta = chain_table[match_idx];
            if (delta == 0) break;
            match_idx -= delta;
        }

        if (use_lazy && best_ref && best_len < 128 && ip + 1 < mflimit) {
            uint32_t next_val = zxc_le32(ip + 1);
            uint32_t h2 = zxc_hash_func(next_val) & (ZXC_LZ_HASH_SIZE - 1);
            uint32_t next_head = hash_table[2 * h2];
            uint32_t next_idx =
                (next_head & ~ZXC_OFFSET_MASK) == epoch_mark ? (next_head & ZXC_OFFSET_MASK) : 0;

            uint32_t max_lazy = 0;
            int lazy_att = (level == 3 || level == 4) ? 1 : 8;

            while (next_idx > 0 && lazy_att-- > 0) {
                if ((uint32_t)(ip + 1 - src) - next_idx >= ZXC_LZ_MAX_DIST) break;
                const uint8_t* ref2 = src + next_idx;
                if (zxc_le32(ref2) == next_val) {
                    uint32_t l2 = 4;
                    while (ip + 1 + l2 < iend && ref2[l2] == ip[1 + l2]) l2++;
                    if (l2 > max_lazy) max_lazy = l2;
                }
                uint16_t delta = chain_table[next_idx];
                if (delta == 0) break;
                next_idx -= delta;
            }
            if (max_lazy > best_len + 1) best_ref = NULL;
        }

        if (best_ref) {
            while (ip > anchor && best_ref > src && ip[-1] == best_ref[-1]) {
                ip--;
                best_ref--;
                best_len++;
            }
            uint32_t ll = (uint32_t)(ip - anchor);
            uint32_t ml = (uint32_t)(best_len - ZXC_LZ_MIN_MATCH);
            uint32_t off = (uint32_t)(ip - best_ref);

            // cppcheck-suppress knownConditionTrueFalse ; false positive
            if (ll > 0) {
                ZXC_MEMCPY(literals + lit_c, anchor, ll);
                lit_c += ll;
            }

            // Token & Offset
            // cppcheck-suppress knownConditionTrueFalse ; false positive
            uint8_t ll_code = (ll >= 15) ? 15 : (uint8_t)ll;
            uint8_t ml_code = (ml >= 15) ? 15 : (uint8_t)ml;
            buf_tokens[seq_c] = (ll_code << 4) | ml_code;
            buf_offsets[seq_c] = (uint16_t)off;

            // Extras & VByte size
            // cppcheck-suppress knownConditionTrueFalse ; false positive
            if (ll >= 15) {
                buf_extras[n_extras++] = ll;
                if (LIKELY(ll < 128)) {
                    vbyte_size += 1;
                } else {
                    if (ll < 16384)
                        vbyte_size += 2;
                    else if (ll < 2097152)
                        vbyte_size += 3;
                    else
                        vbyte_size += 5;
                }
            }
            if (ml >= 15) {
                buf_extras[n_extras++] = ml;
                if (LIKELY(ml < 128)) {
                    vbyte_size += 1;
                } else {
                    if (ml < 16384)
                        vbyte_size += 2;
                    else if (ml < 2097152)
                        vbyte_size += 3;
                    else
                        vbyte_size += 5;
                }
            }
            seq_c++;

            if (best_len > 2 && level > 4) {
                const uint8_t* match_end = ip + best_len;
                // Check that we can read 4 bytes for the hash at (end-2)
                if (match_end < iend - 3) {
                    uint32_t pos_u = (uint32_t)((match_end - 2) - src);
                    uint32_t val_u = zxc_le32(match_end - 2);
                    uint32_t h_u = zxc_hash_func(val_u) & (ZXC_LZ_HASH_SIZE - 1);

                    // Retrieve the old head to maintain the chain
                    uint32_t prev_head = hash_table[2 * h_u];
                    uint32_t prev_idx = (prev_head & ~ZXC_OFFSET_MASK) == epoch_mark
                                            ? (prev_head & ZXC_OFFSET_MASK)
                                            : 0;

                    // Update the hash table and chain table
                    hash_table[2 * h_u] = epoch_mark | pos_u;
                    if (prev_idx > 0 && (pos_u - prev_idx) < 0x10000)
                        chain_table[pos_u] = (uint16_t)(pos_u - prev_idx);
                    else
                        chain_table[pos_u] = 0;
                }
            }

            ip += best_len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    size_t last_lits = iend - anchor;
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

    // --- RLE ANALYSIS ---
    size_t rle_size = 0;
    int use_rle = 0;

    if (lit_c > 0 && level >= 2) {
        size_t k = 0;
        while (k < lit_c) {
            uint8_t b = literals[k];
            size_t run = 1;
            while (k + run < lit_c && literals[k + run] == b) run++;
            if (run >= 4) {
                // Repeat Run: Header (1 byte) + Value (1 byte) = 2 bytes
                // Header: 1xxxxxxx (Length = (val & 0x7F) + 4)
                // We can encode runs up to 127 + 4 = 131
                size_t rem_run = run;
                while (rem_run >= 4) {
                    size_t chunk = rem_run > 131 ? 131 : rem_run;
                    rle_size += 2;
                    rem_run -= chunk;
                }
                if (rem_run > 0) rle_size += 1 + rem_run;
            } else {
                // Literal Run: Header (1 byte) + Length bytes
                // Header: 0xxxxxxx (Length = val + 1)
                // We can encode runs up to 128
                size_t lit_run = run;
                // Check ahead for more literals
                size_t j = k + run;
                while (j < lit_c) {
                    uint8_t nb = literals[j];
                    size_t nrun = 1;
                    while (j + nrun < lit_c && literals[j + nrun] == nb) nrun++;
                    if (nrun >= 4) break;
                    lit_run += nrun;
                    j += nrun;
                }

                size_t rem_lit = lit_run;
                while (rem_lit > 0) {
                    size_t chunk = rem_lit > 128 ? 128 : rem_lit;
                    rle_size += 1 + chunk;
                    rem_lit -= chunk;
                }
                run = lit_run;
            }
            k += run;
        }

        // Threshold: 3% savings
        if (rle_size < lit_c * 0.97) use_rle = 1;
    }

    size_t h_gap = ZXC_BLOCK_HEADER_SIZE + (chk ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_GNR, .raw_size = (uint32_t)src_size};
    uint8_t* p = dst + h_gap;
    size_t rem = dst_cap - h_gap;

    zxc_gnr_header_t gh = {.n_sequences = seq_c,
                           .n_literals = (uint32_t)lit_c,
                           .enc_lit = (uint8_t)use_rle,
                           .enc_litlen = 0,
                           .enc_mlen = 0,
                           .enc_off = 0};

    zxc_section_desc_t desc[4] = {0};
    desc[0].sizes = (uint64_t)(use_rle ? (uint32_t)rle_size : (uint32_t)lit_c) |
                    ((uint64_t)(uint32_t)lit_c << 32);
    desc[1].sizes = (uint64_t)seq_c | ((uint64_t)seq_c << 32);
    desc[2].sizes = (uint64_t)(seq_c * 2) | ((uint64_t)(seq_c * 2) << 32);
    desc[3].sizes = (uint64_t)(uint32_t)vbyte_size | ((uint64_t)(uint32_t)vbyte_size << 32);

    int ghs = zxc_write_gnr_header_and_desc(p, rem, &gh, desc);
    if (UNLIKELY(ghs < 0)) return -1;

    uint8_t* p_curr = p + ghs;
    rem -= ghs;

    if (rem < (desc[0].sizes & 0xFFFFFFFF)) return -1;

    if (use_rle) {
        // Write RLE
        size_t rle_pos = 0;
        while (rle_pos < lit_c) {
            uint8_t b = literals[rle_pos];
            size_t run = 1;
            while (rle_pos + run < lit_c && literals[rle_pos + run] == b) run++;

            if (run >= 4) {
                size_t rem_run = run;
                while (rem_run >= 4) {
                    size_t chunk = rem_run > 131 ? 131 : rem_run;
                    *p_curr++ = (uint8_t)(0x80 | (chunk - 4));
                    *p_curr++ = b;
                    rem_run -= chunk;
                }
                if (rem_run > 0) {
                    *p_curr++ = (uint8_t)(rem_run - 1);
                    ZXC_MEMCPY(p_curr, literals + rle_pos + run - rem_run, rem_run);
                    p_curr += rem_run;
                }
            } else {
                size_t lit_run = run;
                size_t j = rle_pos + run;
                while (j < lit_c) {
                    uint8_t nb = literals[j];
                    size_t nrun = 1;
                    while (j + nrun < lit_c && literals[j + nrun] == nb) nrun++;
                    if (nrun >= 4) break;
                    lit_run += nrun;
                    j += nrun;
                }

                size_t rem_lit = lit_run;
                size_t offset = 0;
                while (rem_lit > 0) {
                    size_t chunk = rem_lit > 128 ? 128 : rem_lit;
                    *p_curr++ = (uint8_t)(chunk - 1);
                    ZXC_MEMCPY(p_curr, literals + rle_pos + offset, chunk);
                    p_curr += chunk;
                    offset += chunk;
                    rem_lit -= chunk;
                }
                run = lit_run;
            }
            rle_pos += run;
        }
    } else {
        ZXC_MEMCPY(p_curr, literals, lit_c);
        p_curr += lit_c;
    }
    rem -= (desc[0].sizes & 0xFFFFFFFF);

    if (rem < (desc[1].sizes & 0xFFFFFFFF)) return -1;
    ZXC_MEMCPY(p_curr, buf_tokens, seq_c);
    p_curr += seq_c;
    rem -= seq_c;

    if (rem < (desc[2].sizes & 0xFFFFFFFF)) return -1;
    ZXC_MEMCPY(p_curr, buf_offsets, seq_c * 2);
    p_curr += seq_c * 2;
    rem -= seq_c * 2;

    if (rem < (desc[3].sizes & 0xFFFFFFFF)) return -1;
    // Write VByte stream
    for (size_t j = 0; j < n_extras; j++) {
        uint32_t val = buf_extras[j];
        while (val >= 128) {
            *p_curr++ = (uint8_t)(val | 0x80);
            val >>= 7;
        }
        *p_curr++ = (uint8_t)val;
    }

    uint32_t p_sz = (uint32_t)(p_curr - (dst + h_gap));
    if (chk)
        bh.block_flags |= ZXC_BLOCK_FLAG_CHECKSUM;
    else
        bh.block_flags &= ~ZXC_BLOCK_FLAG_CHECKSUM;
    bh.comp_size = p_sz;
    int hw = zxc_write_block_header(dst, dst_cap, &bh);

    if (chk) zxc_store_le64(dst + hw, crc_val);
    *out_sz = hw + (chk ? ZXC_BLOCK_CHECKSUM_SIZE : 0) + p_sz;
    return 0;
}

/**
 * @brief Encodes a raw data block (uncompressed).
 *
 * This function prepares and writes a "RAW" type block into the destination
 * buffer. It handles the block header, copying of source data, and optionally
 * the calculation and storage of a checksum.
 *
 * @param[in] src Pointer to the source data to encode.
 * @param[in] src_sz Size of the source data in bytes.
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] dst_cap Maximum capacity of the destination buffer.
 * @param[out] out_sz Pointer to a variable receiving the total written size
 * (header
 * + data + checksum).
 * @param[in] chk Boolean flag: if non-zero, a checksum is calculated and added.
 * @param[in] crc_val The pre-calculated XXH3 value (if checksum is enabled).
 *
 * @return 0 on success, -1 if the destination buffer capacity is
 * insufficient.
 */
static int zxc_encode_block_raw(const uint8_t* src, size_t src_sz, uint8_t* dst, size_t dst_cap,
                                size_t* out_sz, int chk, uint64_t crc_val) {
    size_t h_gap = ZXC_BLOCK_HEADER_SIZE + (chk ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    if (UNLIKELY(dst_cap < h_gap + src_sz)) return -1;

    // Compute block RAW
    zxc_block_header_t bh;
    bh.block_type = ZXC_BLOCK_RAW;
    bh.block_flags = chk ? ZXC_BLOCK_FLAG_CHECKSUM : 0;
    bh.reserved = 0;
    bh.comp_size = (uint32_t)src_sz;
    bh.raw_size = (uint32_t)src_sz;

    zxc_write_block_header(dst, dst_cap, &bh);

    if (chk) {
        zxc_store_le64(dst + ZXC_BLOCK_HEADER_SIZE, crc_val);
    }

    ZXC_MEMCPY(dst + h_gap, src, src_sz);
    *out_sz = h_gap + src_sz;
    return 0;
}

/**
 * @brief Probes data to see if it looks like a sequence of correlated
 * integers.
 * * Heuristic:
 * 1. Must be aligned to 4 bytes.
 * 2. Samples the first 64 integers.
 * 3. Calculates the delta between consecutive values.
 * 4. If the deltas are small (fit in few bits), it's a candidate for NUM.
 *
 * @return 1 if NUM encoding is recommended, 0 otherwise.
 */
static int zxc_probe_is_numeric(const uint8_t* src, size_t size) {
    if (UNLIKELY(size % 4 != 0 || size < 16)) return 0;

    size_t count = size / 4;
    if (count > 64) count = 64;

    uint32_t prev = zxc_le32(src);
    const uint8_t* p = src + 4;
    uint32_t small_deltas = 0;

    for (size_t i = 1; i < count; i++) {
        uint32_t curr = zxc_le32(p);

        int32_t diff = (int32_t)(curr - prev);
        uint32_t zigzag = zxc_zigzag_encode(diff);

        if (zigzag < 256) {
            small_deltas++;
        } else if (zigzag < 65536) {
            small_deltas++;
        }

        prev = curr;
        p += 4;
    }

    return (small_deltas > (count * 90) / 100);
}

int zxc_compress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* chunk, size_t src_sz, uint8_t* dst,
                               size_t dst_cap) {
    int chk = ctx->checksum_enabled;

    size_t w = 0;
    uint64_t crc = 0;
    int res = -1;
    int try_num = 0;

    if (chk) {
        crc = zxc_checksum(chunk, src_sz);
    }

    if (zxc_probe_is_numeric(chunk, src_sz)) {
        try_num = 1;
    }

    if (try_num) {
        res = zxc_encode_block_num(ctx, chunk, src_sz, dst, dst_cap, &w, crc);
        if (res != 0 || w > ((src_sz >> 1) + (src_sz >> 3))) {  // Ratio > 0.625
            try_num = 0;
        }
    }

    if (!try_num) {
        res = zxc_encode_block_gnr(ctx, chunk, src_sz, dst, dst_cap, &w, crc);
    }

    if (UNLIKELY(res != 0 || w >= src_sz)) {
        res = zxc_encode_block_raw(chunk, src_sz, dst, dst_cap, &w, chk, crc);
        if (UNLIKELY(res != 0)) {
            return res;
        }
    }

    return (int)w;
}

// cppcheck-suppress unusedFunction
size_t zxc_compress(const void* src, size_t src_size, void* dst, size_t dst_capacity, int level,
                    int checksum_enabled) {
    if (!src || !dst || src_size == 0 || dst_capacity == 0) return 0;

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;

    zxc_cctx_t ctx;
    if (zxc_cctx_init(&ctx, ZXC_CHUNK_SIZE, 1, level, checksum_enabled) != 0) return 0;

    int h_size = zxc_write_file_header(op, (size_t)(op_end - op));
    if (h_size < 0) {
        zxc_cctx_free(&ctx);
        return 0;
    }
    op += h_size;

    size_t pos = 0;
    while (pos < src_size) {
        size_t chunk_len = (src_size - pos > ZXC_CHUNK_SIZE) ? ZXC_CHUNK_SIZE : (src_size - pos);
        size_t rem_cap = (size_t)(op_end - op);

        int res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap);
        if (res < 0) {
            zxc_cctx_free(&ctx);
            return 0;
        }

        op += res;
        pos += chunk_len;
    }

    zxc_cctx_free(&ctx);
    return (size_t)(op - op_start);
}