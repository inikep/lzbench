/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
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
#define zxc_compress_chunk_wrapper ZXC_CAT(zxc_compress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#endif

#define ZXC_NUM_FRAME_SIZE \
    128  // Maximum number of frames that can be processed in a single compression operation.
#define ZXC_EPOCH_BITS \
    14  // Number of bits reserved for epoch tracking in compressed pointers.
        // Derived from chunk size: 2^18 = ZXC_BLOCK_SIZE => 32 - 18 = 14 bits.
#define ZXC_OFFSET_MASK              \
    ((1U << (32 - ZXC_EPOCH_BITS)) - \
     1)  // Mask to extract the offset bits from a compressed pointer.
#define ZXC_MAX_EPOCH \
    (1U << ZXC_EPOCH_BITS)  // Maximum number of epochs supported by the compression system.

/**
 * @brief Computes a hash value optimized for LZ77 pattern matching speed.
 *
 * Knuth's multiplicative hash constant: 2654435761 (golden ratio * 2^32)
 * Returns upper bits which have the best avalanche properties
 * The caller applies the mask (& (ZXC_LZ_HASH_SIZE - 1))
 *
 * @param[in] val The 32-bit integer sequence (e.g., 4 bytes from the input stream).
 * @return uint32_t A hash value suitable for indexing the match table.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_func(const uint32_t val) {
    return (val * 2654435761U) >> (32 - ZXC_LZ_HASH_BITS);
}

#if defined(ZXC_USE_AVX2)
/**
 * @brief Reduces a 256-bit integer vector to a single scalar by finding the maximum unsigned 32-bit
 * integer element.
 *
 * This function performs a horizontal reduction across the 8 packed 32-bit unsigned integers
 * in the source vector to determine the maximum value.
 *
 * @param[in] v The 256-bit vector containing 8 unsigned 32-bit integers.
 * @return The maximum unsigned 32-bit integer found in the vector.
 */
// codeql[cpp/unused-static-function] : Used conditionally when ZXC_USE_AVX2 is defined
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
 * @brief Writes a Prefix Varint encoded value to a buffer.
 *
 * This function encodes a 32-bit unsigned integer using Prefix Varint encoding
 * and writes it to the destination buffer. Unary prefix bits in the first
 * byte determine the total length (1-5 bytes), allowing for branchless or
 * predictable decoding.
 *
 * Format:
 * - 0xxxxxxx (1 byte)
 * - 10xxxxxx ... (2 bytes)
 * - 110xxxxx ... (3 bytes)
 * ...
 *
 * @param[out] dst Pointer to the destination buffer where the encoded value will be written.
 * @param[in] val The 32-bit unsigned integer value to encode.
 * @return The number of bytes written to the destination buffer.
 */
static ZXC_ALWAYS_INLINE size_t zxc_write_varint(uint8_t* RESTRICT dst, uint32_t val) {
    // Prefix Varint Encoding
    // 1 byte: 0xxxxxxx (7 bits) -> val < 128
    if (LIKELY(val < 128)) {
        dst[0] = (uint8_t)val;
        return 1;
    }

    // 2 bytes: 10xxxxxx xxxxxxxx (14 bits) -> val < 16384 (2^14)
    if (LIKELY(val < 16384)) {
        dst[0] = (uint8_t)(0x80 | (val & 0x3F));
        dst[1] = (uint8_t)(val >> 6);
        return 2;
    }

    // 3 bytes: 110xxxxx xxxxxxxx xxxxxxxx (21 bits) -> val < 2097152 (2^21)
    if (LIKELY(val < 2097152)) {
        dst[0] = (uint8_t)(0xC0 | (val & 0x1F));
        dst[1] = (uint8_t)(val >> 5);
        dst[2] = (uint8_t)(val >> 13);
        return 3;
    }

    // 4 bytes: 1110xxxx xxxxxxxx xxxxxxxx xxxxxxxx (28 bits) -> val < 268435456 (2^28)
    if (LIKELY(val < 268435456)) {
        dst[0] = (uint8_t)(0xE0 | (val & 0x0F));
        dst[1] = (uint8_t)(val >> 4);
        dst[2] = (uint8_t)(val >> 12);
        dst[3] = (uint8_t)(val >> 20);
        return 4;
    }

    // 5 bytes: 11110xxx ... (35 bits) -> Full 32-bit range
    dst[0] = (uint8_t)(0xF0 | (val & 0x07));
    dst[1] = (uint8_t)(val >> 3);
    dst[2] = (uint8_t)(val >> 11);
    dst[3] = (uint8_t)(val >> 19);
    dst[4] = (uint8_t)(val >> 27);
    return 5;
}

/**
 * @brief Structure representing a match found during compression.
 *
 * This structure holds information about a matching sequence found
 * in the input data during the compression process.
 *
 * @param ref       Pointer to the reference data where the match was found.
 * @param len       Length of the matching sequence in bytes.
 * @param backtrack Distance to backtrack from the current position to find the match.
 */
typedef struct {
    const uint8_t* ref;
    uint32_t len;
    uint32_t backtrack;
} zxc_match_t;

/**
 * @brief Finds the best matching sequence for LZ77 compression
 *
 * This function searches for the longest matching sequence in the
 * sliding window dictionary for LZ77 compression algorithm.
 * It is marked as always inline for performance optimization.
 *
 * @param[in] src Pointer to the start of the source buffer.
 * @param[in] ip Current input position pointer.
 * @param[in] iend Pointer to the end of the input buffer.
 * @param[in] mflimit Pointer to the match finding limit.
 * @param[in] anchor Pointer to the current anchor position.
 * @param[in,out] hash_table Pointer to the hash table for match finding.
 * @param[in,out] chain_table Pointer to the chain table for collision handling.
 * @param[in] epoch_mark Current epoch marker for hash table invalidation.
 * @param[in] p LZ77 parameters controlling search depth, lazy matching, and stepping.
 * @return zxc_match_t Structure containing the best match information
 *         (reference pointer, length of the match, and backtrack distance).
 */
static ZXC_ALWAYS_INLINE zxc_match_t zxc_lz77_find_best_match(
    const uint8_t* src, const uint8_t* ip, const uint8_t* iend, const uint8_t* mflimit,
    const uint8_t* anchor, uint32_t* RESTRICT hash_table, uint16_t* RESTRICT chain_table,
    uint32_t epoch_mark, const int level, const zxc_lz77_params_t p) {
    // Track the best match found so far.
    //  ref is the pointer to the start of the match in the history buffer,
    //  len is the match length, and backtrack is the distance from ip to ref.
    //  Start with a sentinel length just below the minimum so any valid match will replace it.
    zxc_match_t best = (zxc_match_t){NULL, ZXC_LZ_MIN_MATCH_LEN - 1, 0};

    // Load the 4-byte sequence at the current position and hash it.
    // The hash value h is used to index into the LZ77 hash table.
    uint32_t cur_val = zxc_le32(ip);
    uint32_t h = zxc_hash_func(cur_val);

    // For levels 1-2, enhance tag with byte5 info via XOR (preserves byte4 info)
    // High byte becomes (byte4 ^ byte5), keeping discrimination from both bytes
    uint32_t cur_tag = (level <= 2) ? (cur_val ^ ((uint32_t)ip[4] << 24)) : cur_val;

    // Current position in the input buffer expressed as a 32-bit index.
    // This index is what we store in / retrieve from the hash/chain tables.
    uint32_t cur_pos = (uint32_t)(ip - src);

    // Each hash bucket stores:
    // - raw_head: compressed pointer (epoch in high bits, position in low bits)
    // - stored_tag: 4-byte tag (or XOR-enhanced for levels 1-2) to quickly reject mismatches.
    // Epoch bits allow the tables to be lazily invalidated without clearing all entries.
    uint32_t raw_head = hash_table[2 * h];
    uint32_t stored_tag = hash_table[2 * h + 1];

    // If the epoch in raw_head matches the current epoch_mark, extract the
    // stored position; otherwise treat this bucket as empty (index 0).
    // Branchless optimization:
    // Create a mask that is 0xFFFFFFFF if epochs match, 0 otherwise.
    uint32_t epoch_mask = -((int32_t)((raw_head & ~ZXC_OFFSET_MASK) == epoch_mark));
    uint32_t match_idx = (raw_head & ZXC_OFFSET_MASK) & epoch_mask;

    // Decide whether to skip the head entry of the hash chain.
    int skip_head = (match_idx != 0) & (stored_tag != cur_tag);

    // If we should skip the head and level is low (<= 2), we drop the match entirely (match_idx =
    // 0). drop_mask is 0 if we drop (skip_head && level <= 2 is true becomes 1, 1-1=0), -1
    // otherwise.
    uint32_t drop_mask = (uint32_t)((skip_head & (level <= 2)) - 1);
    match_idx &= drop_mask;

    hash_table[2 * h] = epoch_mark | cur_pos;
    hash_table[2 * h + 1] = cur_tag;

    // Branchless chain table update
    uint32_t dist = cur_pos - match_idx;
    uint32_t valid_mask = -((int32_t)((match_idx != 0) & (dist < ZXC_LZ_WINDOW_SIZE)));
    chain_table[cur_pos] = (uint16_t)(dist & valid_mask);

    if (match_idx == 0) return best;

    int attempts = p.search_depth;

    // Optimization: If head tag doesn't match, advance immediately without loading the first
    // mismatch.
    if (skip_head) {
        uint16_t delta = chain_table[match_idx];
        uint32_t next_idx = match_idx - delta;
        match_idx = (delta != 0) ? next_idx : 0;
        attempts--;
    }

    while (match_idx > 0 && attempts-- >= 0) {
        if (UNLIKELY(cur_pos - match_idx > ZXC_LZ_MAX_DIST)) break;
        const uint8_t* ref = src + match_idx;

        uint32_t ref_val = zxc_le32(ref);
        int tag_match = (ref_val == cur_val);
        // Simplified check: only tag match and next-byte match required
        int should_compare = tag_match && (ref[best.len] == ip[best.len]);

        if (should_compare) {
            uint32_t mlen = sizeof(uint32_t);  // We already know the first 4 bytes match

            // Fast path: Scalar 64-bit comparison for short matches (=< 64 bytes)
            // Most matches are short, so this avoids SIMD overhead for common cases
            const uint8_t* limit_8 = iend - sizeof(uint64_t);
            const uint8_t* scalar_limit = ip + mlen + 64;
            if (scalar_limit > limit_8) scalar_limit = limit_8;

            while (ip + mlen < scalar_limit) {
                uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen);
                if (diff == 0)
                    mlen += sizeof(uint64_t);
                else {
                    mlen += (zxc_ctz64(diff) >> 3);
                    goto _match_len_done;
                }
            }

            // Long match path: Use SIMD for matches exceeding 64 bytes
#if defined(ZXC_USE_AVX512)
            const uint8_t* limit_64 = iend - 64;
            while (ip + mlen < limit_64) {
                __m512i v_src = _mm512_loadu_si512((const void*)(ip + mlen));
                __m512i v_ref = _mm512_loadu_si512((const void*)(ref + mlen));
                __mmask64 mask = _mm512_cmpeq_epi8_mask(v_src, v_ref);
                if (mask == 0xFFFFFFFFFFFFFFFF)
                    mlen += 64;
                else {
                    mlen += (uint32_t)zxc_ctz64(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_AVX2)
            const uint8_t* limit_32 = iend - 32;
            while (ip + mlen < limit_32) {
                __m256i v_src = _mm256_loadu_si256((const __m256i*)(ip + mlen));
                __m256i v_ref = _mm256_loadu_si256((const __m256i*)(ref + mlen));
                __m256i v_cmp = _mm256_cmpeq_epi8(v_src, v_ref);
                uint32_t mask = (uint32_t)_mm256_movemask_epi8(v_cmp);
                if (mask == 0xFFFFFFFF)
                    mlen += 32;
                else {
                    mlen += zxc_ctz32(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
            const uint8_t* limit_16 = iend - 16;
            while (ip + mlen < limit_16) {
                uint8x16_t v_src = vld1q_u8(ip + mlen), v_ref = vld1q_u8(ref + mlen);
                uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);
#if defined(ZXC_USE_NEON64)
                if (vminvq_u8(v_cmp) == 0xFF)
                    mlen += 16;
                else {
                    uint8x16_t v_diff = vmvnq_u8(v_cmp);
                    uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(v_diff), 0);
                    if (lo != 0)
                        mlen += (zxc_ctz64(lo) >> 3);
                    else
                        mlen +=
                            8 + (zxc_ctz64(vgetq_lane_u64(vreinterpretq_u64_u8(v_diff), 1)) >> 3);
                    goto _match_len_done;
                }
#else
                uint8x8_t p1 = vpmin_u8(vget_low_u8(v_cmp), vget_high_u8(v_cmp));
                uint8x8_t p2 = vpmin_u8(p1, p1);
                uint8x8_t p3 = vpmin_u8(p2, p2);
                uint8_t min_val = vget_lane_u8(p3, 0);
                if (min_val == 0xFF)
                    mlen += 16;
                else {
                    uint8x16_t v_diff = vmvnq_u8(v_cmp);
                    uint64_t lo = (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 0) |
                                  ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 1) << 32);
                    if (lo != 0)
                        mlen += (zxc_ctz64(lo) >> 3);
                    else
                        mlen +=
                            8 +
                            (zxc_ctz64((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 2) |
                                       ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 3)
                                        << 32)) >>
                             3);
                    goto _match_len_done;
                }
#endif
            }
#endif
            while (ip + mlen < limit_8) {
                uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen);
                if (diff == 0)
                    mlen += sizeof(uint64_t);
                else {
                    mlen += (zxc_ctz64(diff) >> 3);
                    goto _match_len_done;
                }
            }
            while (ip + mlen < iend && ref[mlen] == ip[mlen]) mlen++;

        _match_len_done:;
            int better = (mlen > best.len);
            best.len = better ? mlen : best.len;
            best.ref = better ? ref : best.ref;

            if (UNLIKELY(best.len >= (uint32_t)p.sufficient_len || ip + best.len >= iend)) break;
        }

        uint16_t delta = chain_table[match_idx];
        uint32_t next_idx = match_idx - delta;
        ZXC_PREFETCH_READ(src + next_idx);

        match_idx = (delta != 0) ? next_idx : 0;
    }

    if (best.ref) {
        // Backtrack to extend match backwards
        const uint8_t* b_ip = ip;
        const uint8_t* b_ref = best.ref;
        while (b_ip > anchor && b_ref > src && b_ip[-1] == b_ref[-1]) {
            b_ip--;
            b_ref--;
            best.len++;
            best.backtrack++;
        }
        best.ref = b_ref;
    }

    if (p.use_lazy && best.ref && best.len < 128 && ip + 1 < mflimit) {
        uint32_t next_val = zxc_le32(ip + 1);
        uint32_t h2 = zxc_hash_func(next_val);
        uint32_t next_head = hash_table[2 * h2];
        uint32_t next_stored_tag = hash_table[2 * h2 + 1];
        uint32_t next_idx =
            (next_head & ~ZXC_OFFSET_MASK) == epoch_mark ? (next_head & ZXC_OFFSET_MASK) : 0;
        int skip_lazy_head = (next_idx > 0 && next_stored_tag != next_val);
        uint32_t max_lazy = 0;
        int lazy_att = p.lazy_attempts;
        int is_lazy_first = 1;

        while (next_idx > 0 && lazy_att-- > 0) {
            if (UNLIKELY((uint32_t)(ip + 1 - src) - next_idx > ZXC_LZ_MAX_DIST)) break;
            const uint8_t* ref2 = src + next_idx;
            if ((!is_lazy_first || !skip_lazy_head) && zxc_le32(ref2) == next_val) {
                uint32_t l2 = sizeof(uint32_t);
                const uint8_t* limit8 = iend - sizeof(uint64_t);
                while (ip + 1 + l2 < limit8) {
                    uint64_t v1 = zxc_le64(ip + 1 + l2);
                    uint64_t v2 = zxc_le64(ref2 + l2);
                    if (v1 != v2) {
                        l2 += zxc_ctz64(v1 ^ v2) >> 3;
                        goto lazy1_done;
                    }
                    l2 += sizeof(uint64_t);
                }
                while (ip + 1 + l2 < iend && ref2[l2] == ip[1 + l2]) l2++;
            lazy1_done:
                if (l2 > max_lazy) max_lazy = l2;
            }
            uint16_t delta = chain_table[next_idx];
            if (UNLIKELY(delta == 0)) break;
            next_idx -= delta;
            is_lazy_first = 0;
        }

        if (max_lazy > best.len + 1) {
            best.ref = NULL;
        } else if (level >= 4 && ip + 2 < mflimit) {
            uint32_t val3 = zxc_le32(ip + 2);
            uint32_t h3 = zxc_hash_func(val3);
            uint32_t head3 = hash_table[2 * h3];
            uint32_t tag3 = hash_table[2 * h3 + 1];
            uint32_t idx3 =
                (head3 & ~ZXC_OFFSET_MASK) == epoch_mark ? (head3 & ZXC_OFFSET_MASK) : 0;
            int skip_head3 = (idx3 > 0 && tag3 != val3);
            int is_first3 = 1;
            uint32_t max_lazy3 = 0;
            lazy_att = p.lazy_attempts;
            while (idx3 > 0 && lazy_att-- > 0) {
                if (UNLIKELY((uint32_t)(ip + 2 - src) - idx3 > ZXC_LZ_MAX_DIST)) break;
                const uint8_t* ref3 = src + idx3;
                if ((!is_first3 || !skip_head3) && zxc_le32(ref3) == val3) {
                    uint32_t l3 = sizeof(uint32_t);
                    const uint8_t* limit8_3 = iend - sizeof(uint64_t);
                    while (ip + 2 + l3 < limit8_3) {
                        uint64_t v1 = zxc_le64(ip + 2 + l3);
                        uint64_t v2 = zxc_le64(ref3 + l3);
                        if (v1 != v2) {
                            l3 += zxc_ctz64(v1 ^ v2) >> 3;
                            goto lazy2_done;
                        }
                        l3 += sizeof(uint64_t);
                    }
                    while (ip + 2 + l3 < iend && ref3[l3] == ip[2 + l3]) l3++;
                lazy2_done:
                    if (l3 > max_lazy3) max_lazy3 = l3;
                }
                uint16_t delta = chain_table[idx3];
                if (UNLIKELY(delta == 0)) break;
                idx3 -= delta;
                is_first3 = 0;
            }
            if (max_lazy3 > best.len + 2) best.ref = NULL;
        }
    }

    return best;
}

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
 * @param[in] src_sz Size of the source buffer in bytes. Must be a multiple of 4
 * and non-zero.
 * @param[out] dst Pointer to the destination buffer where compressed data will be
 * written.
 * @param[in] dst_cap Capacity of the destination buffer in bytes.
 * @param[out] out_sz Pointer to a variable where the total size of the compressed
 * output will be stored.
 *
 * @return 0 on success, or -1 on failure (e.g., invalid input size, destination
 * buffer too small).
 */
static int zxc_encode_block_num(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, size_t dst_cap,
                                size_t* RESTRICT out_sz) {
    if (UNLIKELY(src_sz % sizeof(uint32_t) != 0 || src_sz == 0 ||
                 dst_cap < ZXC_BLOCK_HEADER_SIZE + ZXC_NUM_HEADER_BINARY_SIZE))
        return -1;

    const size_t count = src_sz / sizeof(uint32_t);

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_NUM};
    uint8_t* p_curr = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;
    zxc_num_header_t nh = {.n_values = count, .frame_size = ZXC_NUM_FRAME_SIZE};

    const int hs = zxc_write_num_header(p_curr, rem, &nh);
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

            if (j > 0) prev = zxc_le32(in_ptr + (j - 1) * sizeof(uint32_t));
        }
#endif
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32)
    _scalar:
#endif
        for (; j < frames; j++) {
            uint32_t v = zxc_le32(in_ptr + j * sizeof(uint32_t));
            uint32_t diff = zxc_zigzag_encode((int32_t)(v - prev));
            deltas[j] = diff;
            if (diff > max_d) max_d = diff;
            prev = v;
        }
        in_ptr += frames * sizeof(uint32_t);

        uint8_t bits = zxc_highbit32(max_d);
        size_t packed = ((frames * bits) + ZXC_BITS_PER_BYTE - 1) / ZXC_BITS_PER_BYTE;
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

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return -1;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return 0;
}

/**
 * @brief Encodes a data block using the General (GLO) compression format.
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
 * @param[in] src_sz  Size of the input data in bytes.
 * @param[out] dst       Pointer to the destination buffer where compressed data will
 * be written.
 * @param[in] dst_cap   Maximum capacity of the destination buffer.
 * @param[out] out_sz    [Out] Pointer to a variable that will receive the total size
 * of the compressed output.
 *
 * @return 0 on success, or -1 if an error occurs (e.g., buffer overflow).
 */
static int zxc_encode_block_glo(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, size_t dst_cap,
                                size_t* RESTRICT out_sz) {
    const int level = ctx->compression_level;

    const zxc_lz77_params_t lzp = zxc_get_lz77_params(level);

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ZXC_MAX_EPOCH)) {
        ZXC_MEMSET(ctx->hash_table, 0, 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ctx->epoch = 1;
    }
    const uint32_t epoch_mark = ctx->epoch << (32 - ZXC_EPOCH_BITS);
    const uint8_t *ip = src, *iend = src + src_sz, *anchor = ip, *mflimit = iend - 12;

    uint32_t* const hash_table = ctx->hash_table;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint8_t* const buf_tokens = ctx->buf_tokens;
    uint16_t* const buf_offsets = ctx->buf_offsets;
    uint8_t* const buf_extras = ctx->buf_extras;

    uint32_t seq_c = 0;
    size_t lit_c = 0;
    size_t extras_sz = 0;
    uint16_t max_offset = 0;  // Track max offset for 1-byte/2-byte mode decision

    while (LIKELY(ip < mflimit)) {
        size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= mflimit)) step = 1;

        ZXC_PREFETCH_READ(ip + step * 4 + ZXC_CACHE_LINE_SIZE);

        const zxc_match_t m = zxc_lz77_find_best_match(src, ip, iend, mflimit, anchor, hash_table,
                                                       chain_table, epoch_mark, level, lzp);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = (uint32_t)(m.len - ZXC_LZ_MIN_MATCH_LEN);
            const uint32_t off = (uint32_t)(ip - m.ref);

            if (ll > 0) {
                if (ll <= 16)
                    zxc_copy16(literals + lit_c, anchor);
                else if (ll <= 32)
                    zxc_copy32(literals + lit_c, anchor);
                else
                    ZXC_MEMCPY(literals + lit_c, anchor, ll);
                lit_c += ll;
            }

            const uint8_t ll_code = (ll >= ZXC_TOKEN_LL_MASK) ? ZXC_TOKEN_LL_MASK : (uint8_t)ll;
            const uint8_t ml_code = (ml >= ZXC_TOKEN_ML_MASK) ? ZXC_TOKEN_ML_MASK : (uint8_t)ml;
            buf_tokens[seq_c] = (ll_code << ZXC_TOKEN_LIT_BITS) | ml_code;
            buf_offsets[seq_c] = (uint16_t)off;
            if (off > max_offset) max_offset = (uint16_t)off;

            if (ll >= ZXC_TOKEN_LL_MASK) {
                extras_sz += zxc_write_varint(buf_extras + extras_sz, ll - ZXC_TOKEN_LL_MASK);
            }
            if (ml >= ZXC_TOKEN_ML_MASK) {
                extras_sz += zxc_write_varint(buf_extras + extras_sz, ml - ZXC_TOKEN_ML_MASK);
            }
            seq_c++;

            if (m.len > 2 && level > 4) {
                const uint8_t* match_end = ip + m.len;
                if (match_end < iend - 3) {
                    uint32_t pos_u = (uint32_t)((match_end - 2) - src);
                    uint32_t val_u = zxc_le32(match_end - 2);
                    uint32_t h_u = zxc_hash_func(val_u);
                    uint32_t prev_head = hash_table[2 * h_u];
                    uint32_t prev_idx = (prev_head & ~ZXC_OFFSET_MASK) == epoch_mark
                                            ? (prev_head & ZXC_OFFSET_MASK)
                                            : 0;
                    hash_table[2 * h_u] = epoch_mark | pos_u;
                    hash_table[2 * h_u + 1] = val_u;
                    chain_table[pos_u] = (prev_idx > 0 && (pos_u - prev_idx) < ZXC_LZ_WINDOW_SIZE)
                                             ? (uint16_t)(pos_u - prev_idx)
                                             : 0;
                }
            }

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = iend - anchor;
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

    // --- RLE ANALYSIS ---
    size_t rle_size = 0;
    int use_rle = 0;

    if (lit_c > 0) {
        const uint8_t* p = literals;
        const uint8_t* const p_end = literals + lit_c;
        const uint8_t* const p_end_4 = p_end - 3;  // Safe limit for 4-byte lookahead

        while (LIKELY(p < p_end)) {
            uint8_t b = *p;
            const uint8_t* run_start = p++;

            // Fast run counting with early SIMD exit
#if defined(ZXC_USE_AVX512)
            __m512i vb = _mm512_set1_epi8((char)b);
            while (p <= p_end - 64) {
                __m512i v = _mm512_loadu_si512((const void*)p);
                __mmask64 mask = _mm512_cmpeq_epi8_mask(v, vb);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) {
                    p += (size_t)zxc_ctz64(~mask);
                    goto _run_done;
                }
                p += 64;
            }
#elif defined(ZXC_USE_AVX2)
            __m256i vb = _mm256_set1_epi8((char)b);
            while (p <= p_end - 32) {
                __m256i v = _mm256_loadu_si256((const __m256i*)p);
                uint32_t mask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vb));
                if (mask != 0xFFFFFFFF) {
                    p += zxc_ctz32(~mask);
                    goto _run_done;
                }
                p += 32;
            }
#elif defined(ZXC_USE_NEON64)
            uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                uint8x16_t v = vld1q_u8(p);
                uint8x16_t eq = vceqq_u8(v, vb);
                uint8x16_t not_eq = vmvnq_u8(eq);
                uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(not_eq), 0);
                if (lo != 0) {
                    p += (zxc_ctz64(lo) >> 3);
                    goto _run_done;
                }
                uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(not_eq), 1);
                if (hi != 0) {
                    p += 8 + (zxc_ctz64(hi) >> 3);
                    goto _run_done;
                }
                p += 16;
            }
#elif defined(ZXC_USE_NEON32)
            uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                uint8x16_t v = vld1q_u8(p);
                uint8x16_t eq = vceqq_u8(v, vb);
                uint8x16_t not_eq = vmvnq_u8(eq);

                // 32-bit ARM NEON doesn't always support vgetq_lane_u64 / vreinterpretq_u64_u8 so
                // we treat the 128-bit vector as 4 x 32-bit lanes */
                uint32x4_t neq32 = vreinterpretq_u32_u8(not_eq);
                uint32_t l0 = vgetq_lane_u32(neq32, 0);
                uint32_t l1 = vgetq_lane_u32(neq32, 1);

                uint64_t lo = ((uint64_t)l1 << 32) | l0;
                if (lo != 0) {
                    p += (zxc_ctz64(lo) >> 3);
                    goto _run_done;
                }

                uint32_t h0 = vgetq_lane_u32(neq32, 2);
                uint32_t h1 = vgetq_lane_u32(neq32, 3);
                uint64_t hi = ((uint64_t)h1 << 32) | h0;

                if (hi != 0) {
                    p += 8 + (zxc_ctz64(hi) >> 3);
                    goto _run_done;
                }
                p += 16;
            }
#endif
            while (p < p_end && *p == b) p++;

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32)
        _run_done:;
#endif
            size_t run = (size_t)(p - run_start);

            if (run >= 4) {
                // RLE run: 2 bytes per 131 values, then remainder
                // Branchless: full_chunks * 2 + remainder handling
                size_t full_chunks = run / 131;
                size_t rem = run - full_chunks * 131;  // Avoid modulo
                rle_size += full_chunks * 2;
                // Remainder: if >= 4 -> 2 bytes (RLE), else 1 + rem (literal)
                if (rem >= 4)
                    rle_size += 2;
                else if (rem > 0)
                    rle_size += 1 + rem;
            } else {
                // Literal run: scan ahead with fast SIMD lookahead
                const uint8_t* lit_start = run_start;

#if defined(ZXC_USE_AVX512)
                while (p <= p_end_4 - 64) {
                    __m512i v0 = _mm512_loadu_si512((const void*)p);
                    __m512i v1 = _mm512_loadu_si512((const void*)(p + 1));
                    __m512i v2 = _mm512_loadu_si512((const void*)(p + 2));
                    __m512i v3 = _mm512_loadu_si512((const void*)(p + 3));
                    __mmask64 mask = _mm512_cmpeq_epi8_mask(v0, v1) &
                                     _mm512_cmpeq_epi8_mask(v1, v2) &
                                     _mm512_cmpeq_epi8_mask(v2, v3);
                    if (mask != 0) {
                        p += (size_t)zxc_ctz64(mask);
                        goto _lit_done;
                    }
                    p += 64;
                }
#elif defined(ZXC_USE_AVX2)
                while (p <= p_end_4 - 32) {
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)p);
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 1));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(p + 2));
                    __m256i v3 = _mm256_loadu_si256((const __m256i*)(p + 3));
                    __m256i vend = _mm256_and_si256(
                        _mm256_cmpeq_epi8(v0, v1),
                        _mm256_and_si256(_mm256_cmpeq_epi8(v1, v2), _mm256_cmpeq_epi8(v2, v3)));
                    uint32_t mask = (uint32_t)_mm256_movemask_epi8(vend);
                    if (mask != 0) {
                        p += zxc_ctz32(mask);
                        goto _lit_done;
                    }
                    p += 32;
                }
#elif defined(ZXC_USE_NEON64)
                while (p <= p_end_4 - 16) {
                    uint8x16_t v0 = vld1q_u8(p);
                    uint8x16_t v1 = vld1q_u8(p + 1);
                    uint8x16_t v2 = vld1q_u8(p + 2);
                    uint8x16_t v3 = vld1q_u8(p + 3);
                    uint8x16_t eq =
                        vandq_u8(vceqq_u8(v0, v1), vandq_u8(vceqq_u8(v1, v2), vceqq_u8(v2, v3)));
                    uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(eq), 0);
                    if (lo != 0) {
                        p += (zxc_ctz64(lo) >> 3);
                        goto _lit_done;
                    }
                    uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(eq), 1);
                    if (hi != 0) {
                        p += 8 + (zxc_ctz64(hi) >> 3);
                        goto _lit_done;
                    }
                    p += 16;
                }
#elif defined(ZXC_USE_NEON32)
                while (p <= p_end_4 - 16) {
                    uint8x16_t v0 = vld1q_u8(p);
                    uint8x16_t v1 = vld1q_u8(p + 1);
                    uint8x16_t v2 = vld1q_u8(p + 2);
                    uint8x16_t v3 = vld1q_u8(p + 3);
                    uint8x16_t eq =
                        vandq_u8(vceqq_u8(v0, v1), vandq_u8(vceqq_u8(v1, v2), vceqq_u8(v2, v3)));

                    uint32x4_t eq32 = vreinterpretq_u32_u8(eq);
                    uint32_t l0 = vgetq_lane_u32(eq32, 0);
                    uint32_t l1 = vgetq_lane_u32(eq32, 1);
                    uint64_t lo = ((uint64_t)l1 << 32) | l0;

                    if (lo != 0) {
                        p += (zxc_ctz64(lo) >> 3);
                        goto _lit_done;
                    }

                    uint32_t h0 = vgetq_lane_u32(eq32, 2);
                    uint32_t h1 = vgetq_lane_u32(eq32, 3);
                    uint64_t hi = ((uint64_t)h1 << 32) | h0;

                    if (hi != 0) {
                        p += 8 + (zxc_ctz64(hi) >> 3);
                        goto _lit_done;
                    }
                    p += 16;
                }
#endif
                while (p < p_end_4) {
                    // Check for RLE opportunity (4 identical bytes)
                    if (UNLIKELY(p[0] == p[1] && p[1] == p[2] && p[2] == p[3])) break;
                    p++;
                }
                // Handle remaining bytes near end
                while (p < p_end) {
                    if (UNLIKELY(p + 3 < p_end && p[0] == p[1] && p[1] == p[2] && p[2] == p[3]))
                        break;
                    p++;
                }

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32)
            _lit_done:;
#endif
                size_t lit_run = (size_t)(p - lit_start);
                // 1 header per 128 bytes + all data bytes
                // = lit_run + ceil(lit_run / 128)
                rle_size += lit_run + ((lit_run + 127) >> 7);
            }
        }

        // Threshold: ~3% savings using integer math (97% ~= 1 - 1/32)
        if (rle_size < lit_c - (lit_c >> 5)) use_rle = 1;
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_GLO};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    // Decide offset encoding mode: 1-byte if all offsets <= 255
    int use_8bit_off = (max_offset <= 255) ? 1 : 0;
    size_t off_stream_size = use_8bit_off ? seq_c : (seq_c * 2);

    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = (uint8_t)use_rle,
                                 .enc_litlen = 0,
                                 .enc_mlen = 0,
                                 .enc_off = (uint8_t)use_8bit_off};

    zxc_section_desc_t desc[ZXC_GLO_SECTIONS] = {0};
    desc[0].sizes = (uint64_t)(use_rle ? rle_size : lit_c) | ((uint64_t)lit_c << 32);
    desc[1].sizes = (uint64_t)seq_c | ((uint64_t)seq_c << 32);
    desc[2].sizes = (uint64_t)off_stream_size | ((uint64_t)off_stream_size << 32);
    desc[3].sizes = (uint64_t)extras_sz | ((uint64_t)extras_sz << 32);

    int ghs = zxc_write_glo_header_and_desc(p, rem, &gh, desc);
    if (UNLIKELY(ghs < 0)) return -1;

    uint8_t* p_curr = p + ghs;
    rem -= ghs;

    // Extract stream sizes once
    const size_t sz_lit = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_tok = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_off = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_ext = (size_t)(desc[3].sizes & ZXC_SECTION_SIZE_MASK);

    if (UNLIKELY(rem < sz_lit)) return -1;

    if (use_rle) {
        // Write RLE - optimized single-pass encoding
        const uint8_t* lit_ptr = literals;
        const uint8_t* const lit_end = literals + lit_c;

        while (lit_ptr < lit_end) {
            uint8_t b = *lit_ptr;
            const uint8_t* run_start = lit_ptr++;

            // Count run length
            while (lit_ptr < lit_end && *lit_ptr == b) lit_ptr++;
            size_t run = (size_t)(lit_ptr - run_start);

            if (run >= 4) {
                // RLE runs: emit 2-byte tokens (header + value)
                while (run >= 4) {
                    size_t chunk = (run > 131) ? 131 : run;
                    *p_curr++ = (uint8_t)(ZXC_LIT_RLE_FLAG | (chunk - 4));
                    *p_curr++ = b;
                    run -= chunk;
                }
                // Leftover < 4 bytes: emit as literal
                if (run > 0) {
                    *p_curr++ = (uint8_t)(run - 1);
                    ZXC_MEMCPY(p_curr, lit_ptr - run, run);
                    p_curr += run;
                }
            } else {
                // Literal run: scan ahead to find next RLE opportunity
                const uint8_t* lit_run_start = run_start;

                while (lit_ptr < lit_end) {
                    // Quick check: need 4 identical bytes to break
                    if (UNLIKELY(lit_ptr + 3 < lit_end && lit_ptr[0] == lit_ptr[1] &&
                                 lit_ptr[1] == lit_ptr[2] && lit_ptr[2] == lit_ptr[3])) {
                        break;
                    }
                    lit_ptr++;
                }

                size_t lit_run = (size_t)(lit_ptr - lit_run_start);
                const uint8_t* src_ptr = lit_run_start;

                // Emit literal chunks (max 128 bytes each)
                while (lit_run > 0) {
                    size_t chunk = (lit_run > 128) ? 128 : lit_run;
                    *p_curr++ = (uint8_t)(chunk - 1);
                    ZXC_MEMCPY(p_curr, src_ptr, chunk);
                    p_curr += chunk;
                    src_ptr += chunk;
                    lit_run -= chunk;
                }
            }
        }
    } else {
        ZXC_MEMCPY(p_curr, literals, lit_c);
        p_curr += lit_c;
    }
    rem -= sz_lit;

    if (UNLIKELY(rem < sz_tok)) return -1;

    ZXC_MEMCPY(p_curr, buf_tokens, seq_c);
    p_curr += seq_c;
    rem -= sz_tok;

    if (UNLIKELY(rem < sz_off)) return -1;

    if (use_8bit_off) {
        // Write 1-byte offsets - unroll for better throughput
        uint32_t i = 0;
        for (; i + 8 <= seq_c; i += 8) {
            p_curr[0] = (uint8_t)buf_offsets[i + 0];
            p_curr[1] = (uint8_t)buf_offsets[i + 1];
            p_curr[2] = (uint8_t)buf_offsets[i + 2];
            p_curr[3] = (uint8_t)buf_offsets[i + 3];
            p_curr[4] = (uint8_t)buf_offsets[i + 4];
            p_curr[5] = (uint8_t)buf_offsets[i + 5];
            p_curr[6] = (uint8_t)buf_offsets[i + 6];
            p_curr[7] = (uint8_t)buf_offsets[i + 7];
            p_curr += 8;
        }
        for (; i < seq_c; i++) {
            *p_curr++ = (uint8_t)buf_offsets[i];
        }
    } else {
        // Write 2-byte offsets in little-endian order
#ifdef ZXC_BIG_ENDIAN
        for (uint32_t i = 0; i < seq_c; i++) {
            zxc_store_le16(p_curr, buf_offsets[i]);
            p_curr += sizeof(uint16_t);
        }
#else
        ZXC_MEMCPY(p_curr, buf_offsets, seq_c * sizeof(uint16_t));
        p_curr += seq_c * sizeof(uint16_t);
#endif
    }
    rem -= sz_off;

    if (UNLIKELY(rem < sz_ext)) return -1;

    ZXC_MEMCPY(p_curr, buf_extras, extras_sz);
    p_curr += extras_sz;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return -1;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return 0;
}

/**
 * @brief Encodes a data block using the General High Velocity (GHI) compression format.
 *
 * 1. Compression Strategy
 * It uses an LZ77-based algorithm with a sliding window (64KB) and a hash table/chain table
 * mechanism.
 *
 * 2. Token Format (Fixed-Width)
 * Unlike the standard GLO block which uses 1-byte tokens (4-bit literal length / 4-bit match
 * length), GHI uses 4-byte (32-bit) sequence records for better performance on long runs:
 * Literal Length (LL): 8 bits (stores 0-254; 255 indicates overflow).
 * Match Length (ML): 8 bits (stores 0-254; 255 indicates overflow).
 * Offset: 16 bits (supports the full 64KB window).
 * This format minimizes the number of expensive VByte reads during decompression for common
 * sequences where lengths are between 16 and 255.
 *
 * @param[in,out] ctx       Pointer to the compression context containing hash tables
 * and configuration.
 * @param[in] src       Pointer to the input source data.
 * @param[in] src_sz  Size of the input data in bytes.
 * @param[out] dst       Pointer to the destination buffer where compressed data will
 * be written.
 * @param[in] dst_cap   Maximum capacity of the destination buffer.
 * @param[out] out_sz    [Out] Pointer to a variable that will receive the total size
 * of the compressed output.
 *
 * @return 0 on success, or -1 if an error occurs (e.g., buffer overflow).
 */
static int zxc_encode_block_ghi(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    const int level = ctx->compression_level;

    const zxc_lz77_params_t lzp = zxc_get_lz77_params(level);

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ZXC_MAX_EPOCH)) {
        ZXC_MEMSET(ctx->hash_table, 0, 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ctx->epoch = 1;
    }
    const uint32_t epoch_mark = ctx->epoch << (32 - ZXC_EPOCH_BITS);
    const uint8_t *ip = src, *iend = src + src_sz, *anchor = ip, *mflimit = iend - 12;

    uint32_t* const hash_table = ctx->hash_table;
    uint8_t* const buf_extras = ctx->buf_extras;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint32_t* const buf_sequences = ctx->buf_sequences;

    uint32_t seq_c = 0;
    size_t extras_c = 0;
    size_t lit_c = 0;
    uint16_t max_offset = 0;

    while (LIKELY(ip < mflimit)) {
        size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= mflimit)) step = 1;

        ZXC_PREFETCH_READ(ip + step * 4 + 64);

        const zxc_match_t m = zxc_lz77_find_best_match(src, ip, iend, mflimit, anchor, hash_table,
                                                       chain_table, epoch_mark, level, lzp);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = (uint32_t)(m.len - ZXC_LZ_MIN_MATCH_LEN);
            const uint32_t off = (uint32_t)(ip - m.ref);

            if (ll > 0) {
                if (ll <= 16)
                    zxc_copy16(literals + lit_c, anchor);
                else if (ll <= 32)
                    zxc_copy32(literals + lit_c, anchor);
                else
                    ZXC_MEMCPY(literals + lit_c, anchor, ll);
                lit_c += ll;
            }

            const uint32_t ll_write = (ll >= ZXC_SEQ_LL_MASK) ? 255U : ll;
            const uint32_t ml_write = (ml >= ZXC_SEQ_ML_MASK) ? 255U : ml;
            const uint32_t seq_val = (ll_write << (ZXC_SEQ_ML_BITS + ZXC_SEQ_OFF_BITS)) |
                                     (ml_write << ZXC_SEQ_OFF_BITS) | (off & ZXC_SEQ_OFF_MASK);
            if (off > max_offset) max_offset = (uint16_t)off;
            buf_sequences[seq_c] = seq_val;
            seq_c++;

            if (ll >= ZXC_SEQ_LL_MASK) {
                extras_c += zxc_write_varint(buf_extras + extras_c, ll - ZXC_SEQ_LL_MASK);
            }
            if (ml >= ZXC_SEQ_ML_MASK) {
                extras_c += zxc_write_varint(buf_extras + extras_c, ml - ZXC_SEQ_ML_MASK);
            }

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = iend - anchor;
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_GHI};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    // Decide offset encoding mode
    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = 0,
                                 .enc_litlen = 0,
                                 .enc_mlen = 0,
                                 .enc_off = (uint8_t)(max_offset <= 255) ? 1 : 0};

    zxc_section_desc_t desc[ZXC_GHI_SECTIONS] = {0};
    desc[0].sizes = (uint64_t)lit_c | ((uint64_t)lit_c << 32);
    size_t sz_seqs = seq_c * sizeof(uint32_t);
    desc[1].sizes = (uint64_t)sz_seqs | ((uint64_t)sz_seqs << 32);
    desc[2].sizes = (uint64_t)extras_c | ((uint64_t)extras_c << 32);

    const int ghs = zxc_write_ghi_header_and_desc(p, rem, &gh, desc);
    if (UNLIKELY(ghs < 0)) return -1;

    uint8_t* p_curr = p + ghs;
    rem -= ghs;

    // Extract stream sizes once
    const size_t sz_lit = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_seq = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_ext = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);

    if (UNLIKELY(rem < sz_lit + sz_seq + sz_ext)) return -1;

    ZXC_MEMCPY(p_curr, literals, lit_c);
    p_curr += lit_c;
    rem -= sz_lit;

    if (UNLIKELY(rem < sz_seq)) return -1;
    // Write sequences in little-endian order
#ifdef ZXC_BIG_ENDIAN
    for (uint32_t i = 0; i < seq_c; i++) {
        zxc_store_le32(p_curr, buf_sequences[i]);
        p_curr += sizeof(uint32_t);
    }
#else
    ZXC_MEMCPY(p_curr, buf_sequences, sz_seq);
    p_curr += sz_seq;
#endif

    // --- WRITE EXTRAS ---
    ZXC_MEMCPY(p_curr, buf_extras, sz_ext);
    p_curr += sz_ext;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return -1;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
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
 *
 * @return 0 on success, -1 if the destination buffer capacity is
 * insufficient.
 */
static int zxc_encode_block_raw(const uint8_t* RESTRICT src, const size_t src_sz,
                                uint8_t* RESTRICT const dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE + src_sz)) return -1;

    // Compute block RAW
    zxc_block_header_t bh;
    bh.block_type = ZXC_BLOCK_RAW;
    bh.block_flags = 0;  // Checksum flag moved to file header
    bh.reserved = 0;
    bh.comp_size = (uint32_t)src_sz;

    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return -1;

    ZXC_MEMCPY(dst + ZXC_BLOCK_HEADER_SIZE, src, src_sz);

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + src_sz;
    return 0;
}

/**
 * @brief Checks if the given byte array represents a numeric value.
 *
 * This function examines the provided buffer to determine if it contains
 * only numeric characters (e.g., ASCII digits '0'-'9').
 *
 * Improved heuristic:
 * 1. Must be aligned to 4 bytes.
 * 2. Samples the first 128 integers (more accurate).
 * 3. Calculates bit width of deltas (fewer bits = better for NUM).
 * 4. Estimates compression ratio: if NUM would save >20% vs raw, use it.
 *
 * @param[in] src Pointer to the input byte array to be checked.
 * @param[in] size The number of bytes in the input array.
 * @return int Returns 1 if the array is numeric, 0 otherwise.
 */
static int zxc_probe_is_numeric(const uint8_t* src, const size_t size) {
    if (UNLIKELY(size % 4 != 0 || size < 16)) return 0;

    size_t count = size / 4;
    if (count > 128) count = 128;  // Sample more values for accuracy

    uint32_t prev = zxc_le32(src);
    const uint8_t* p = src + 4;

    uint32_t max_zigzag = 0;
    uint32_t small_count = 0;   // Deltas < 256 (8 bits)
    uint32_t medium_count = 0;  // Deltas < 65536 (16 bits)

    for (size_t i = 1; i < count; i++) {
        const uint32_t curr = zxc_le32(p);
        const int32_t diff = (int32_t)(curr - prev);
        const uint32_t zigzag = zxc_zigzag_encode(diff);

        if (zigzag > max_zigzag) max_zigzag = zigzag;

        if (zigzag < 256) {
            small_count++;
        } else if (zigzag < 65536) {
            medium_count++;
        }

        prev = curr;
        p += 4;
    }

    // Calculate bit width needed for max delta
    uint32_t bits_needed = 0;
    uint32_t tmp = max_zigzag;
    while (tmp > 0) {
        bits_needed++;
        tmp >>= 1;
    }

    // Estimate compression ratio:
    // NUM uses ~bits_needed per value, Raw uses 32 bits per value
    // Worth it if bits_needed <= 20 (saves >37.5%)
    if (bits_needed <= 16) return 1;
    if (bits_needed <= 20 && (small_count + medium_count) >= (count * 85) / 100) return 1;

    // Fallback: if 90% of deltas are small, still use NUM
    if ((small_count + medium_count) >= (count * 90) / 100) return 1;

    return 0;
}

// cppcheck-suppress unusedFunction
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT chunk,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
    size_t w = 0;
    int res = -1;
    int try_num = 0;

    if (UNLIKELY(zxc_probe_is_numeric(chunk, src_sz))) try_num = 1;
    if (UNLIKELY(try_num)) {
        res = zxc_encode_block_num(ctx, chunk, src_sz, dst, dst_cap, &w);
        if (res != 0 || w > (src_sz - (src_sz >> 2)))  // w > 75% of src_sz
            try_num = 0;  // NUM didn't compress well, try GLO/GHI instead
    }

    if (LIKELY(!try_num)) {
        if (ctx->compression_level <= 2)
            res = zxc_encode_block_ghi(ctx, chunk, src_sz, dst, dst_cap, &w);
        else
            res = zxc_encode_block_glo(ctx, chunk, src_sz, dst, dst_cap, &w);
    }

    // Check expansion. W contains Header + Payload.
    if (UNLIKELY(res != 0 || w >= src_sz)) {
        res = zxc_encode_block_raw(chunk, src_sz, dst, dst_cap, &w);
        if (UNLIKELY(res != 0)) return res;
    }

    if (ctx->checksum_enabled) {
        // Calculate checksum on the compressed payload (w currently excludes checksum)
        // Header is at dst, data starts at dst + ZXC_BLOCK_HEADER_SIZE
        if (UNLIKELY(w < ZXC_BLOCK_HEADER_SIZE || w + ZXC_BLOCK_CHECKSUM_SIZE > dst_cap)) return -1;

        uint32_t payload_sz = (uint32_t)(w - ZXC_BLOCK_HEADER_SIZE);
        uint32_t crc =
            zxc_checksum(dst + ZXC_BLOCK_HEADER_SIZE, payload_sz, ZXC_CHECKSUM_RAPIDHASH);
        zxc_store_le32(dst + w, crc);
        w += ZXC_BLOCK_CHECKSUM_SIZE;
    }

    return (int)w;
}
