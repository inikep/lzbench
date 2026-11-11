// SPDX-License-Identifier: BSD-3-Clause
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_compress.c
 * @brief Block-level compression: LZ77 parsing, SPL / PAK / RAW encoding,
 *        and the chunk-wrapper entry point.
 *
 * Compiled multiple times with different @c ZXC_FUNCTION_SUFFIX values to
 * produce AVX2, AVX-512, NEON, and scalar variants dispatched at runtime
 * by @ref zxc_dispatch.c.
 */

// Function Multi-Versioning Support
// With ZXC_FUNCTION_SUFFIX defined (e.g. _avx2), rename the entry point AND the
// Huffman entry points this TU consumes. The defines precede zxc_internal.h so
// the header's prototypes get the suffix too, keeping callers and callees matched.
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_compress_chunk_wrapper ZXC_CAT(zxc_compress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_build_code_lengths ZXC_CAT(zxc_huf_build_code_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_unpack_lengths ZXC_CAT(zxc_huf_unpack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size ZXC_CAT(zxc_huf_calc_size, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size_dict ZXC_CAT(zxc_huf_calc_size_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section ZXC_CAT(zxc_huf_encode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section_dict ZXC_CAT(zxc_huf_encode_section_dict, ZXC_FUNCTION_SUFFIX)
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/**
 * @brief Computes a hash value for either a 4-byte or 5-byte sequence.
 *
 * @param[in] val The 64-bit integer sequence (e.g., 8 bytes read from input stream).
 * @param[in] use_hash5 Non-zero to use the 5-byte xorshift64* hash (Marsaglia/Vigna), zero for
 * 4-byte Marsaglia hash.
 * @return uint32_t A hash value suitable for indexing the match table.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_func(const uint64_t val, const int use_hash5) {
    if (use_hash5) {
        const uint64_t v5 = val & 0xFFFFFFFFFFULL;
        return (uint32_t)((v5 * ZXC_HASH_XORSHIFT64) >> (64 - ZXC_LZ_HASH_BITS));
    } else {
        const uint64_t v4 = val ^ (val >> 15);
        return ((uint32_t)v4 * ZXC_HASH_MULT32) >> (32 - ZXC_LZ_HASH_BITS);
    }
}

/**
 * @brief 8-bit rejection tag: XOR fold of the first four bytes at a position.
 *
 * One tag per hash slot, so a mismatching head is rejected without touching the
 * referenced data.
 */
static ZXC_ALWAYS_INLINE uint8_t zxc_hash_tag(const uint32_t val4) {
    return (uint8_t)(val4 ^ (val4 >> 16));
}

/**
 * @brief Decodes a hash-table slot into a block position, 0 if it is stale.
 *
 * Slots carry the epoch above @c offset_mask, so an entry left from an earlier
 * block reads as "no match" and the table never needs clearing.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_epoch_pos(const uint32_t head, const uint32_t offset_mask,
                                                const uint32_t epoch_mark) {
    return ((head & ~offset_mask) == epoch_mark) ? (head & offset_mask) : 0;
}

/**
 * @brief Chain link from a position back to the one the hash slot held.
 *
 * Branchless: a missing predecessor, or one beyond the window, gives 0, which
 * the chain walk reads as end of chain.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_chain_delta(const uint32_t cur_pos, const uint32_t prev_idx) {
    const uint32_t dist = cur_pos - prev_idx;
    // cppcheck-suppress bitwiseOnBoolean
    const uint32_t valid = 0U - (uint32_t)((prev_idx != 0) & (dist < ZXC_LZ_WINDOW_SIZE));
    return (uint16_t)(dist & valid);
}

/**
 * @brief Pair-equality mask for the RLE lookahead: element i is set when
 *        p[i] == p[i+1].
 *
 * The only thing that differs between vector widths; the scan that consumes it
 * is written once. @c ZXC_RLE_WIN is the bytes examined per step and
 * @c ZXC_RLE_BPB the mask bits each byte occupies (4 on NEON, which has no
 * byte-movemask). 32-bit NEON has no cheap equivalent and keeps its own scan.
 */
#if defined(ZXC_USE_AVX512)
#define ZXC_RLE_WIN 64
#define ZXC_RLE_BPB 1
#define ZXC_RLE_CTZ(x) zxc_ctz64(x)
typedef uint64_t zxc_rle_mask_t;
static ZXC_ALWAYS_INLINE zxc_rle_mask_t zxc_rle_pair_mask(const uint8_t* p) {
    return (uint64_t)_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)p),
                                            _mm512_loadu_si512((const void*)(p + 1)));
}
#elif defined(ZXC_USE_AVX2)
#define ZXC_RLE_WIN 32
#define ZXC_RLE_BPB 1
#define ZXC_RLE_CTZ(x) zxc_ctz32(x)
typedef uint32_t zxc_rle_mask_t;
static ZXC_ALWAYS_INLINE zxc_rle_mask_t zxc_rle_pair_mask(const uint8_t* p) {
    const __m256i v0 = _mm256_loadu_si256((const __m256i*)p);
    const __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 1));
    return (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v1));
}
#elif defined(ZXC_USE_SSE2)
#define ZXC_RLE_WIN 16
#define ZXC_RLE_BPB 1
#define ZXC_RLE_CTZ(x) zxc_ctz32(x)
typedef uint32_t zxc_rle_mask_t;
static ZXC_ALWAYS_INLINE zxc_rle_mask_t zxc_rle_pair_mask(const uint8_t* p) {
    const __m128i v0 = _mm_loadu_si128((const __m128i*)p);
    const __m128i v1 = _mm_loadu_si128((const __m128i*)(p + 1));
    return (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(v0, v1));
}
#elif defined(ZXC_USE_NEON64)
#define ZXC_RLE_WIN 16
#define ZXC_RLE_BPB 4
#define ZXC_RLE_CTZ(x) zxc_ctz64(x)
typedef uint64_t zxc_rle_mask_t;
static ZXC_ALWAYS_INLINE zxc_rle_mask_t zxc_rle_pair_mask(const uint8_t* p) {
    const uint8x16_t eq = vceqq_u8(vld1q_u8(p), vld1q_u8(p + 1));
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
}
#endif

// SSE2 selected, not merely available: guards the two helpers below.
#if defined(ZXC_USE_SSE2) && !(defined(ZXC_USE_AVX512) && defined(__AVX512VL__)) && \
    !defined(ZXC_USE_AVX2) && !defined(ZXC_USE_NEON64) && !defined(ZXC_USE_NEON32)
#define ZXC_OPT_SSE2
#endif

#if defined(ZXC_OPT_SSE2)
/**
 * @brief SSE2 emulation of SSE4.1 @c _mm_blendv_epi8.
 *
 * Selects bytes from @p b where the corresponding @p mask byte has its high bit
 * set, else from @p a. In every call site the mask lanes are full-width compare
 * results (all-ones or all-zero per element), so a plain bitwise select is exact.
 *
 * @param[in] a     Lanes chosen where @p mask is clear.
 * @param[in] b     Lanes chosen where @p mask is set.
 * @param[in] mask  Per-byte selector (a full-width compare result).
 * @return The blended 128-bit vector.
 */
static ZXC_ALWAYS_INLINE __m128i zxc_mm_blendv_epi8_sse2(__m128i a, __m128i b, __m128i mask) {
    return _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a));
}

/**
 * @brief SSE2 emulation of SSE4.1 @c _mm_packus_epi32 (saturating u32 -> u16).
 *
 * SSE2 only has signed @c _mm_packs_epi32 (saturates to int16). Bias each lane
 * by -0x8000 so values in [0, 0xFFFF] land in the signed int16 range with no
 * saturation, pack, then add 0x8000 back per 16-bit lane. Exact for inputs in
 * [0, 0xFFFF] (all call sites pass match lengths < 2^16).
 *
 * @param[in] a  Four u32 lanes forming the low half of the result.
 * @param[in] b  Four u32 lanes forming the high half of the result.
 * @return The eight u16 lanes packed from @p a then @p b.
 */
static ZXC_ALWAYS_INLINE __m128i zxc_mm_packus_epi32_sse2(__m128i a, __m128i b) {
    const __m128i bias32 = _mm_set1_epi32(0x8000);
    const __m128i bias16 = _mm_set1_epi16((short)0x8000);
    const __m128i pa = _mm_sub_epi32(a, bias32);
    const __m128i pb = _mm_sub_epi32(b, bias32);
    return _mm_add_epi16(_mm_packs_epi32(pa, pb), bias16);
}
#endif

/**
 * @brief Writes a value in Prefix Varint form.
 *
 * Unary prefix bits in the first byte give the total length (1 to 5 bytes), so
 * the reader learns the size before it reads the payload:
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
static ZXC_ALWAYS_INLINE size_t zxc_write_varint(uint8_t* RESTRICT dst, const uint32_t val) {
    // Values above ZXC_MAX_VARINT_VALUE are out-of-spec and the decoder would
    // reject them. Unreachable from the Block API, so this is defence in depth;
    // callers must treat a return of 0 as an encoding error.
    if (UNLIKELY(val > ZXC_MAX_VARINT_VALUE)) {
        return 0;
    }

    // 1 byte: 0xxxxxxx (7 bits)
    if (LIKELY(val < ZXC_VARINT_1B_LIMIT)) {
        dst[0] = (uint8_t)val;
        return 1;
    }

    // 2 bytes: 10xxxxxx xxxxxxxx (14 bits)
    if (LIKELY(val < ZXC_VARINT_2B_LIMIT)) {
        dst[0] = (uint8_t)(0x80 | (val & 0x3F));
        dst[1] = (uint8_t)(val >> 6);
        return 2;
    }

    // 3 bytes: 110xxxxx xxxxxxxx xxxxxxxx (21 bits) -> max emittable value,
    // matching ZXC_MAX_VARINT_VALUE = ZXC_BLOCK_SIZE_MAX - 1.
    dst[0] = (uint8_t)(0xC0 | (val & 0x1F));
    dst[1] = (uint8_t)(val >> 5);
    dst[2] = (uint8_t)(val >> 13);
    return 3;
}

/**
 * @brief Appends a saturated token field's excess to the extras stream.
 *
 * A field holds values below its mask; a larger one stores the mask in the
 * token and the excess here as a varint. Returns 0 if it does not fit one.
 */
static ZXC_ALWAYS_INLINE int zxc_emit_extra(uint8_t* RESTRICT extras, size_t* RESTRICT used,
                                            const uint32_t val, const uint32_t mask) {
    if (LIKELY(val < mask)) return 1;
    const size_t n = zxc_write_varint(extras + *used, val - mask);
    if (UNLIKELY(n == 0)) return 0;
    *used += n;
    return 1;
}

/**
 * @brief Emits one SPL sequence (token, offset, LL/ML escapes) and updates
 *        @p max_offset, the block's widest biased offset.
 *
 * @param[out]    buf_tokens  Token stream; the token goes at index @p *seq_c.
 * @param[out]    buf_offsets Offset stream, same index.
 * @param[out]    buf_extras  Extras stream, for the LL/ML excess varints.
 * @param[in,out] seq_c       Sequence count, advanced on success.
 * @param[in,out] extras_sz   Bytes used in @p buf_extras.
 * @param[in,out] max_offset  Widest biased offset so far.
 * @param[in]     ll          Literal length (raw).
 * @param[in]     ml          Match length minus ZXC_LZ_MIN_MATCH_LEN.
 * @param[in]     off_biased  Offset minus ZXC_LZ_OFFSET_BIAS.
 * @return 1, or 0 when a length exceeds ZXC_MAX_VARINT_VALUE. The extras buffer
 *         is sized by the layout, not checked here.
 */
static ZXC_ALWAYS_INLINE int zxc_spl_put_seq(uint8_t* RESTRICT buf_tokens,
                                             uint16_t* RESTRICT buf_offsets,
                                             uint8_t* RESTRICT buf_extras, uint32_t* RESTRICT seq_c,
                                             size_t* RESTRICT extras_sz,
                                             uint16_t* RESTRICT max_offset, const uint32_t ll,
                                             const uint32_t ml, const uint16_t off_biased) {
    const uint8_t ll_code = (ll >= ZXC_TOKEN_LL_MASK) ? ZXC_TOKEN_LL_MASK : (uint8_t)ll;
    const uint8_t ml_code = (ml >= ZXC_TOKEN_ML_MASK) ? ZXC_TOKEN_ML_MASK : (uint8_t)ml;
    buf_tokens[*seq_c] = (uint8_t)((ll_code << ZXC_TOKEN_LIT_BITS) | ml_code);
    buf_offsets[*seq_c] = off_biased;
    if (off_biased > *max_offset) *max_offset = off_biased;
    if (UNLIKELY(!zxc_emit_extra(buf_extras, extras_sz, ll, ZXC_TOKEN_LL_MASK) ||
                 !zxc_emit_extra(buf_extras, extras_sz, ml, ZXC_TOKEN_ML_MASK)))
        return 0;
    (*seq_c)++;
    return 1;
}

/** @brief Moves extras[*r, to) down to extras[*w] (w <= r), advancing both. */
static ZXC_ALWAYS_INLINE void zxc_extras_move(uint8_t* extras, size_t* RESTRICT w,
                                              size_t* RESTRICT r, const uint8_t* to) {
    while (extras + *r < to) extras[(*w)++] = extras[(*r)++];
}

/** @brief Keeps the varint at extras[*r]: moves it down to extras[*w]. */
static ZXC_ALWAYS_INLINE void zxc_varint_keep(uint8_t* extras, size_t* RESTRICT w,
                                              size_t* RESTRICT r, const uint8_t* end) {
    const uint8_t* p = extras + *r;
    (void)zxc_read_varint(&p, end);
    zxc_extras_move(extras, w, r, p);
}

/** @brief Inline pieces a match of length code @p code splits into: ceil(length / 19). */
static ZXC_ALWAYS_INLINE uint32_t zxc_spl_match_split_pieces(const uint32_t code) {
    return (code + ZXC_LZ_MIN_MATCH_LEN + ZXC_SPL_MAX_INLINE_ML - 1) / ZXC_SPL_MAX_INLINE_ML;
}

/** @brief Next piece of a match of length code @p code above the inline reach: the
 *         longest inline one that leaves at least a minimum match. */
static ZXC_ALWAYS_INLINE uint32_t zxc_spl_match_split_piece(const uint32_t code) {
    const uint32_t rest = code - ZXC_LZ_MIN_MATCH_LEN;
    return (rest < ZXC_SPL_INLINE_ML_CODE) ? rest : ZXC_SPL_INLINE_ML_CODE;
}

/**
 * @brief Predictability of the block's ML-escape branch, for the split decision.
 *
 * Contexts are the last `order` escape bits; @p side records each sequence's.
 * @p ctx_bad marks the contexts where an escape is the minority outcome: the
 * decoder's predictor guesses "no escape" and misses. Four rotating histogram
 * lanes keep an inline run off one store-forwarding chain.
 *
 * @return 1 when the minority counts, a static predictor's misses, reach
 *         ZXC_SPL_MATCH_SPLIT_MIN_MISPREDICT_PCT of the escapes and some escape sits
 *         in a bad context, so that a split can happen; else 0.
 */
static int zxc_spl_match_split_analyze(const uint8_t* RESTRICT tokens, const uint32_t n_seq,
                                       zxc_spl_match_split_hist_t* RESTRICT hist_buf,
                                       uint8_t* RESTRICT ctx_bad, uint8_t* RESTRICT side) {
    uint32_t order = ZXC_SPL_MATCH_SPLIT_CTX_BITS;
    while (order > 2 && (n_seq >> order) < 16) order--;
    const uint32_t hmask = (1U << order) - 1U;
    uint32_t (*const ctx_hist)[2][1U << ZXC_SPL_MATCH_SPLIT_CTX_BITS] = *hist_buf;
    for (uint32_t l = 0; l < 4; l++) {
        ZXC_MEMSET(ctx_hist[l][0], 0, (hmask + 1) * sizeof(uint32_t));
        ZXC_MEMSET(ctx_hist[l][1], 0, (hmask + 1) * sizeof(uint32_t));
    }
    uint32_t hist = 0, n_esc = 0;
    for (uint32_t i = 0; i < n_seq; i++) {
        const uint32_t esc = (tokens[i] & ZXC_TOKEN_ML_MASK) == ZXC_TOKEN_ML_MASK;
        side[i] = (uint8_t)hist;
        ctx_hist[i & 3][esc][hist]++;
        hist = ((hist << 1) | esc) & hmask;
        n_esc += esc;
    }
    uint32_t errors = 0, bad_esc = 0;
    for (uint32_t c = 0; c <= hmask; c++) {
        uint32_t n0 = 0, n1 = 0;
        for (uint32_t l = 0; l < 4; l++) {
            n0 += ctx_hist[l][0][c];
            n1 += ctx_hist[l][1][c];
        }
        ctx_bad[c] = (n1 < n0);  // escape a strict minority in its context
        bad_esc += ctx_bad[c] ? n1 : 0;
        errors += (n0 < n1) ? n0 : n1;
    }
    return bad_esc != 0 &&
           (uint64_t)errors * 100 >= (uint64_t)ZXC_SPL_MATCH_SPLIT_MIN_MISPREDICT_PCT * n_esc;
}

/**
 * @brief Per-block match splitting: escaped matches re-emitted as inline pieces.
 *
 * A match whose length code exceeds the inline reach but not @p cap becomes a
 * chain of inline matches at the same offset: the decoder never takes its ML
 * escape. Each extra piece costs a token and an offset, less the varint it
 * replaces.
 *
 * Two gates (zxc_spl_match_split_analyze): the block must look mispredicted, which
 * leaves regular data whole, then only escapes of mispredicted contexts split.
 *
 * In place: pass 1 turns @p side from each sequence's context into its length
 * code to split (0 = keep) and compacts the kept extras forward; pass 2 expands
 * tokens and offsets backward, each write at or past its read. A piece
 * consumes at least ZXC_LZ_MIN_MATCH_LEN input bytes, so the count fits the
 * block's sequence buffers.
 *
 * @param[in,out] tokens    The block's tokens, rewritten in place.
 * @param[in,out] offsets   Its offsets, rewritten in place.
 * @param[in,out] extras    Its extras, compacted in place.
 * @param[in,out] extras_sz Bytes used in @p extras.
 * @param[out]    side      Scratch, one byte per sequence (zxc_cctx_t::buf_split).
 * @param[out]    hist_buf  Scratch (zxc_cctx_t::buf_split_hist).
 * @param[in]     n_seq     Sequence count of the block.
 * @param[in]     cap       zxc_lz77_params_t::split_max.
 * @return The new sequence count (unchanged when the block is not split).
 */
static uint32_t zxc_spl_match_split_block(uint8_t* RESTRICT tokens, uint16_t* RESTRICT offsets,
                                          uint8_t* RESTRICT extras, size_t* RESTRICT extras_sz,
                                          uint8_t* RESTRICT side,
                                          zxc_spl_match_split_hist_t* RESTRICT hist_buf,
                                          const uint32_t n_seq, const uint8_t cap) {
    if (cap <= ZXC_SPL_INLINE_ML_CODE || n_seq == 0) return n_seq;
    uint8_t ctx_bad[1U << ZXC_SPL_MATCH_SPLIT_CTX_BITS];
    if (!zxc_spl_match_split_analyze(tokens, n_seq, hist_buf, ctx_bad, side)) return n_seq;

    // 1. Mark the matches to split (bad context, within the cap) and drop their
    // varints; the other extras move forward.
    const uint8_t* const extras_end = extras + *extras_sz;
    size_t r = 0, w = 0;
    uint32_t extra = 0;
    for (uint32_t i = 0; i < n_seq; i++) {
        const uint8_t tok = tokens[i];
        const int bad = ctx_bad[side[i]];
        side[i] = 0;
        if ((tok >> ZXC_TOKEN_LIT_BITS) == ZXC_TOKEN_LL_MASK)
            zxc_varint_keep(extras, &w, &r, extras_end);
        if ((tok & ZXC_TOKEN_ML_MASK) != ZXC_TOKEN_ML_MASK) continue;
        const uint8_t* p = extras + r;
        const uint32_t code = zxc_read_varint(&p, extras_end) + ZXC_TOKEN_ML_MASK;
        if (bad && code <= cap) {
            side[i] = (uint8_t)code;
            extra += zxc_spl_match_split_pieces(code) - 1;
            r = (size_t)(p - extras);
        } else {
            zxc_extras_move(extras, &w, &r, p);
        }
    }
    *extras_sz = w;
    if (extra == 0) return n_seq;

    // 2. Expand backward. A marked sequence spans its pieces, written forward: the
    // first keeps the literal nibble and its escape, the last takes the remainder.
    uint32_t out = n_seq + extra;
    for (uint32_t i = n_seq; i-- > 0;) {
        const uint8_t tok = tokens[i];
        const uint16_t off = offsets[i];
        if (side[i] == 0) {
            out--;
            tokens[out] = tok;
            offsets[out] = off;
            continue;
        }
        uint32_t code = side[i];
        uint8_t ll_bits = tok & (uint8_t)(ZXC_TOKEN_LL_MASK << ZXC_TOKEN_LIT_BITS);
        out -= zxc_spl_match_split_pieces(code);
        uint32_t k = out;
        while (code > ZXC_SPL_INLINE_ML_CODE) {
            const uint32_t p = zxc_spl_match_split_piece(code);
            tokens[k] = (uint8_t)(ll_bits | p);
            offsets[k++] = off;
            ll_bits = 0;
            code -= p + ZXC_LZ_MIN_MATCH_LEN;
        }
        tokens[k] = (uint8_t)(ll_bits | code);
        offsets[k] = off;
    }
    return n_seq + extra;
}

/**
 * @brief Stages a run of literals into the literal stream.
 *
 * One 32-byte wild copy covers the common short run. Within 32 bytes of the
 * block end that copy would read past it, so the tail takes an exact copy.
 */
static ZXC_ALWAYS_INLINE void zxc_flush_literals(uint8_t* literals, size_t* lit_c,
                                                 const uint8_t* anchor, const uint32_t ll,
                                                 const uint8_t* iend) {
    if (LIKELY(anchor + ZXC_PAD_SIZE <= iend)) {
        zxc_copy32(literals + *lit_c, anchor);
        if (UNLIKELY(ll > ZXC_PAD_SIZE))
            ZXC_MEMCPY(literals + *lit_c + ZXC_PAD_SIZE, anchor + ZXC_PAD_SIZE, ll - ZXC_PAD_SIZE);
    } else {
        ZXC_MEMCPY(literals + *lit_c, anchor, ll);
    }
    *lit_c += ll;
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
 * @brief Tells whether a match reaches a given length, without measuring it.
 *
 * The lazy evaluation only needs a yes or no, so the compare stops at @p need
 * or at the first difference. Bytes below @p from are taken as already equal.
 *
 * @param[in] cur  Position being matched.
 * @param[in] ref  Candidate, earlier in the same buffer.
 * @param[in] from Bytes already known to match.
 * @param[in] need Length to reach; at most @p iend - @p cur.
 * @param[in] iend End of the input buffer.
 * @return 1 if cur[from, need) equals ref[from, need), else 0.
 */
static ZXC_ALWAYS_INLINE int zxc_match_reaches(const uint8_t* cur, const uint8_t* ref,
                                               uint32_t from, const uint32_t need,
                                               const uint8_t* iend) {
    while (from < need) {
        if (UNLIKELY((size_t)(iend - cur) - from < sizeof(uint64_t))) {
            while (from < need && cur[from] == ref[from]) from++;
            return from >= need;
        }
        const uint64_t d = zxc_le64(cur + from) ^ zxc_le64(ref + from);
        if (d != 0) return from + (uint32_t)(zxc_ctz64(d) >> 3) >= need;
        from += sizeof(uint64_t);
    }
    return 1;
}

/**
 * @brief Tells whether @p lp holds a match of at least @p need bytes, within
 *        the lazy search budget. Read-only; 8 bytes at @p lp must be readable.
 *
 * Candidates are gated on the last 4 of the @p need bytes, [need-4, need), and
 * the first to reach @p need settles it.
 */
static ZXC_ALWAYS_INLINE int zxc_lazy_probe(const uint8_t* src, const uint8_t* lp,
                                            const uint8_t* iend, const uint32_t need,
                                            const uint32_t* RESTRICT hash_table,
                                            const uint8_t* RESTRICT hash_tags,
                                            const uint16_t* RESTRICT chain_table,
                                            const uint32_t epoch_mark, const uint32_t offset_mask,
                                            const int use_hash5, const zxc_lz77_params_t p) {
    if (need > (size_t)(iend - lp)) return 0;  // too close to the end to win

    const uint64_t val8 = zxc_le64(lp);
    const uint32_t val = (uint32_t)val8;
    const uint32_t h = zxc_hash_func(val8, use_hash5);
    const uint32_t pos = (uint32_t)(lp - src);
    const uint32_t gate = zxc_le32(lp + need - 4);
    uint32_t idx = zxc_epoch_pos(hash_table[h], offset_mask, epoch_mark);
    // A head with another tag is another string: walk past it.
    int skip = (hash_tags[h] != zxc_hash_tag(val));

    for (int att = p.lazy_attempts; idx > 0 && att > 0; att--) {
        if (UNLIKELY(pos - idx > ZXC_LZ_MAX_DIST)) break;
        const uint8_t* const ref = src + idx;

        // Filtered here too: a lazy hit only cancels best.ref, so a match
        // we could never emit would drop a legal one for nothing.
        if (pos - idx >= p.min_offset && !skip && zxc_le32(ref + need - 4) == gate &&
            zxc_le32(ref) == val && zxc_match_reaches(lp, ref, sizeof(uint32_t), need, iend))
            return 1;

        const uint16_t delta = chain_table[idx & ZXC_LZ_WINDOW_MASK];
        if (UNLIKELY(delta == 0)) break;
        idx -= delta;
        skip = 0;
    }
    return 0;
}

/**
 * @brief Finds the best matching sequence for LZ77 compression
 *
 * Uses a split hash table layout:
 * - hash_table[h]  : uint32_t position + epoch (128 KB for 15-bit hash)
 * - hash_tags[h]   : uint8_t tag for fast rejection (32 KB, L1-resident)
 *
 * @param[in] src Pointer to the start of the source buffer.
 * @param[in] ip Current input position pointer.
 * @param[in] iend Pointer to the end of the input buffer.
 * @param[in] search_limit Pointer to the match finding limit.
 * @param[in] anchor Pointer to the current anchor position.
 * @param[in,out] hash_table Pointer to the position table for match finding.
 * @param[in,out] hash_tags Pointer to the tag table for fast rejection.
 * @param[in,out] chain_table Pointer to the chain table for collision handling.
 * @param[in] epoch_mark Current epoch marker for hash table invalidation.
 * @param[in] offset_mask Mask isolating the position bits in chain/table entries.
 * @param[in] level Compression level (selects search depth and matcher behaviour).
 * @param[in] p LZ77 parameters controlling search depth, lazy matching, and stepping.
 * @param[in] last_off Most recently accepted match offset, used as a repeat-offset
 *            seed probed before the hash-chain walk. Pass 0 to disable (all callers
 *            except the level-6 optimal parser do so).
 * @return zxc_match_t Structure containing the best match information
 *         (reference pointer, length of the match, and backtrack distance).
 */
static ZXC_ALWAYS_INLINE zxc_match_t zxc_lz77_find_best_match(
    const uint8_t* src, const uint8_t* ip, const uint8_t* iend, const uint8_t* search_limit,
    const uint8_t* anchor, uint32_t* RESTRICT hash_table, uint8_t* RESTRICT hash_tags,
    uint16_t* RESTRICT chain_table, const uint32_t epoch_mark, const uint32_t offset_mask,
    const int level, const zxc_lz77_params_t p, const uint32_t last_off) {
    const int use_hash5 = (level >= ZXC_LEVEL_DEFAULT);
    // Sentinel length just below the minimum, so any valid match replaces it.
    zxc_match_t best = (zxc_match_t){NULL, ZXC_LZ_MIN_MATCH_LEN - 1, 0};

    uint64_t cur_val8 = zxc_le64(ip);
    uint32_t cur_val = (uint32_t)cur_val8;
    uint32_t h = zxc_hash_func(cur_val8, use_hash5);

    // 8-bit tag: XOR fold of first 4 bytes for fast rejection
    const uint8_t cur_tag = zxc_hash_tag(cur_val);

    const uint32_t cur_pos = (uint32_t)(ip - src);

    // Tag-first filter on fast levels.
    const uint8_t stored_tag = hash_tags[h];
    uint32_t match_idx;
    if (level <= ZXC_LEVEL_FAST && stored_tag != cur_tag) {
        match_idx = 0;
    } else {
        match_idx = zxc_epoch_pos(hash_table[h], offset_mask, epoch_mark);
    }

    // skip_head still drives the chain walk on level >= 3 (advances past the
    // mismatched head without comparing). On level <= 2 it is always 0 here:
    // either match_idx == 0 (filter-skip) or stored_tag == cur_tag.
    // cppcheck-suppress bitwiseOnBoolean
    const int skip_head = (match_idx != 0) & (stored_tag != cur_tag);

    // Split table writes
    hash_table[h] = epoch_mark | cur_pos;
    hash_tags[h] = cur_tag;

    chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = zxc_chain_delta(cur_pos, match_idx);

    int attempts = p.search_depth;

    // Repeat-offset seed (level-6 parser passes last_off; others pass 0).
    // Probing the previous offset first often finds the longest match right
    // away, speeding up the chain walk. It only raises best.len, never lowers
    // it, so the result is unchanged - this is purely a speed optimization.
    if (last_off != 0U && last_off >= p.min_offset && last_off <= (uint32_t)ZXC_LZ_MAX_DIST &&
        last_off <= cur_pos) {
        const uint8_t* const rep_ref = src + (cur_pos - last_off);
        if (zxc_le32(rep_ref) == cur_val) {
            uint32_t mlen = sizeof(uint32_t);
            const uint8_t* const limit_8 = iend - sizeof(uint64_t);
            while (ip + mlen < limit_8) {
                const uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(rep_ref + mlen);
                if (diff == 0) {
                    mlen += sizeof(uint64_t);
                } else {
                    mlen += (uint32_t)(zxc_ctz64(diff) >> 3);
                    goto _rep_done;
                }
            }
            while (ip + mlen < iend && rep_ref[mlen] == ip[mlen]) mlen++;
        _rep_done:;
            best.len = mlen;
            best.ref = rep_ref;
            if (UNLIKELY(best.len >= (uint32_t)p.sufficient_len || ip + best.len >= iend))
                goto _finalize_match;
        }
    }

    if (match_idx == 0) goto _finalize_match;

    // Optimization: If head tag doesn't match, advance immediately without loading the first
    // mismatch.
    if (skip_head) {
        const uint16_t delta = chain_table[match_idx & ZXC_LZ_WINDOW_MASK];
        const uint32_t next_idx = match_idx - delta;
        match_idx = (delta != 0) ? next_idx : 0;
        attempts--;
    }

    while (match_idx > 0) {
        if (UNLIKELY(attempts-- < 0 || cur_pos - match_idx > ZXC_LZ_MAX_DIST)) break;
        const uint8_t* ref = src + match_idx;

        // Load the next chain link early (before the compare) so its address
        // resolves while we prefetch. The prefetch aims at the next gate for
        // the current best.len; a longer match found here moves that gate.
        const uint16_t delta = chain_table[match_idx & ZXC_LZ_WINDOW_MASK];
        const uint32_t next_idx = match_idx - delta;
        ZXC_PREFETCH_READ(src + next_idx + best.len - 3);

        // Gate first: most candidates share the head and fail on bytes
        // [best.len-3, best.len], which a longer match must also cover.
        const int should_compare = (cur_pos - match_idx) >= p.min_offset &&
                                   zxc_le32(ref + best.len - 3) == zxc_le32(ip + best.len - 3) &&
                                   zxc_le32(ref) == cur_val;

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
                    mlen += (uint32_t)zxc_ctz64(diff) >> 3;
                    goto _match_len_done;
                }
            }

            // Long match path: Use SIMD for matches exceeding 64 bytes
#if defined(ZXC_USE_AVX512)
            const uint8_t* limit_64 = iend - 64;
            while (ip + mlen < limit_64) {
                const __m512i v_src = _mm512_loadu_si512((const void*)(ip + mlen));
                const __m512i v_ref = _mm512_loadu_si512((const void*)(ref + mlen));
                const __mmask64 mask = _mm512_cmpeq_epi8_mask(v_src, v_ref);
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
                const __m256i v_src = _mm256_loadu_si256((const __m256i*)(ip + mlen));
                const __m256i v_ref = _mm256_loadu_si256((const __m256i*)(ref + mlen));
                const __m256i v_cmp = _mm256_cmpeq_epi8(v_src, v_ref);
                const uint32_t mask = (uint32_t)_mm256_movemask_epi8(v_cmp);
                if (mask == 0xFFFFFFFF)
                    mlen += 32;
                else {
                    mlen += (uint32_t)zxc_ctz32(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_SSE2)
            const uint8_t* limit_16 = iend - 16;
            while (ip + mlen < limit_16) {
                const __m128i v_src = _mm_loadu_si128((const __m128i*)(ip + mlen));
                const __m128i v_ref = _mm_loadu_si128((const __m128i*)(ref + mlen));
                const __m128i v_cmp = _mm_cmpeq_epi8(v_src, v_ref);
                const uint32_t mask = (uint32_t)_mm_movemask_epi8(v_cmp);
                if (mask == 0xFFFFU)
                    mlen += 16;
                else {
                    // mask != 0xFFFF => a differing byte exists in bits 0..15,
                    // so the lowest set bit of ~mask lies in that range.
                    mlen += (uint32_t)zxc_ctz32(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_NEON64)
            {
                const uint8_t* limit_32 = iend - 32;
                while (ip + mlen < limit_32) {
                    const uint8x16_t s0 = vld1q_u8(ip + mlen);
                    const uint8x16_t r0 = vld1q_u8(ref + mlen);
                    const uint8x16_t c0 = vceqq_u8(s0, r0);
                    const uint64_t m0 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)), 0);
                    if (UNLIKELY(m0 != ~(uint64_t)0)) {
                        mlen += (uint32_t)(zxc_ctz64(~m0) >> 2);
                        goto _match_len_done;
                    }
                    const uint8x16_t s1 = vld1q_u8(ip + mlen + 16);
                    const uint8x16_t r1 = vld1q_u8(ref + mlen + 16);
                    const uint8x16_t c1 = vceqq_u8(s1, r1);
                    const uint64_t m1 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c1), 4)), 0);
                    if (UNLIKELY(m1 != ~(uint64_t)0)) {
                        mlen += 16 + (uint32_t)(zxc_ctz64(~m1) >> 2);
                        goto _match_len_done;
                    }
                    mlen += 32;
                }
                if (ip + mlen < iend - 16) {
                    const uint8x16_t v_src = vld1q_u8(ip + mlen);
                    const uint8x16_t v_ref = vld1q_u8(ref + mlen);
                    const uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);
                    const uint64_t mask = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v_cmp), 4)), 0);
                    if (LIKELY(mask == ~(uint64_t)0))
                        mlen += 16;
                    else {
                        mlen += (uint32_t)(zxc_ctz64(~mask) >> 2);
                        goto _match_len_done;
                    }
                }
            }
#elif defined(ZXC_USE_NEON32)
            {
                const uint8_t* limit_16 = iend - 16;
                while (ip + mlen < limit_16) {
                    const uint8x16_t v_src = vld1q_u8(ip + mlen);
                    const uint8x16_t v_ref = vld1q_u8(ref + mlen);
                    const uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);
                    uint8x8_t p1 = vpmin_u8(vget_low_u8(v_cmp), vget_high_u8(v_cmp));
                    uint8x8_t p2 = vpmin_u8(p1, p1);
                    uint8x8_t p3 = vpmin_u8(p2, p2);
                    uint8x8_t p4 = vpmin_u8(p3, p3);
                    uint8_t min_val = vget_lane_u8(p4, 0);
                    if (min_val == 0xFF)
                        mlen += 16;
                    else {
                        uint8x16_t v_diff = vmvnq_u8(v_cmp);
                        uint64_t lo =
                            (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 0) |
                            ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 1) << 32);
                        if (lo != 0)
                            mlen += (zxc_ctz64(lo) >> 3);
                        else
                            mlen +=
                                8 + (zxc_ctz64(
                                         (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 2) |
                                         ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 3)
                                          << 32)) >>
                                     3);
                        goto _match_len_done;
                    }
                }
            }
#endif
            while (ip + mlen < limit_8) {
                const uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen);
                if (diff == 0)
                    mlen += sizeof(uint64_t);
                else {
                    mlen += (uint32_t)zxc_ctz64(diff) >> 3;
                    goto _match_len_done;
                }
            }
            while (ip + mlen < iend && ref[mlen] == ip[mlen]) mlen++;

        _match_len_done:;
            const int better = (mlen > best.len);
            best.len = better ? mlen : best.len;
            best.ref = better ? ref : best.ref;

            if (UNLIKELY(best.len >= (uint32_t)p.sufficient_len || ip + best.len >= iend)) break;
        }

        match_idx = (delta != 0) ? next_idx : 0;
    }

_finalize_match:
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

    // Lazy: a match two bytes longer at ip+1, or from level 4 three bytes longer
    // at ip+2, cancels this one.
    if (p.use_lazy && best.ref && best.len < (uint32_t)p.lazy_len_threshold &&
        ip + 1 < search_limit) {
        if (zxc_lazy_probe(src, ip + 1, iend, best.len + 2, hash_table, hash_tags, chain_table,
                           epoch_mark, offset_mask, use_hash5, p) ||
            (level >= ZXC_LEVEL_BALANCED && ip + 2 < search_limit &&
             zxc_lazy_probe(src, ip + 2, iend, best.len + 3, hash_table, hash_tags, chain_table,
                            epoch_mark, offset_mask, use_hash5, p)))
            best.ref = NULL;
    }

    return best;
}

/**
 * @brief Relaxes dp[p + L] with a match of length @p L costing @p nxt in total.
 *
 * Strict comparison: on a tie the earlier transition keeps the cell.
 */
static ZXC_ALWAYS_INLINE void zxc_opt_relax(uint32_t* RESTRICT dp, uint16_t* RESTRICT parent_len,
                                            uint16_t* RESTRICT parent_off, const size_t p,
                                            const size_t L, const uint32_t nxt,
                                            const uint16_t off_biased) {
    if (nxt < dp[p + L]) {
        dp[p + L] = nxt;
        parent_len[p + L] = (uint16_t)L;
        parent_off[p + L] = off_biased;
    }
}

/**
 * @brief Update dp[p + L_start .. p + L_end) with a constant transition
 *        cost, in parallel where the target ISA allows.
 *
 * For each L in [L_start, L_end), if @p nxt is strictly less than the
 * current dp[p+L], rewrite dp/parent_len/parent_off in lockstep: same
 * semantics as the scalar update inside ::zxc_lz77_optimal_parse_spl.
 * Caller guarantees @p nxt is independent of L (the cost of the L-th
 * transition does not vary across the requested span).
 *
 * Vectorized prologue per ISA, falling through to a scalar tail:
 *   - AVX-512 BW + VL : 16-wide via vpcmpud + vmask{store,storeu}.
 *                       Falls back to AVX2 if VL is absent.
 *   - AVX2            : 8-wide via biased vpcmpgt + vpblendvb (no 32-bit
 *                       unsigned cmpgt before AVX-512). parent_off is
 *                       updated with a packed 8x16 mask + 128-bit blend.
 *   - NEON64 / NEON32 : 4-wide via vcgtq_u32 + vbslq_u32, with vmovn_u32
 *                       to narrow the mask for the 4x16 parent_off update.
 *
 * @param[in,out] dp         DP cost array; dp[p + L] is relaxed when
 *                           @p nxt < dp[p + L].
 * @param[in,out] parent_len Backtrack length array, written in lockstep
 *                           with @p dp; receives the L of the relaxing
 *                           transition.
 * @param[in,out] parent_off Backtrack offset array, written in lockstep;
 *                           receives @p off_biased on relaxation.
 * @param[in]     p          Source DP position the transitions originate
 *                           from. Indexing into the three arrays is
 *                           `p + L`.
 * @param[in]     L          Initial L value (start of the span, inclusive).
 * @param[in]     L_end      End of the span (exclusive). Must satisfy
 *                           @p L_end <= UINT16_MAX so every written length
 *                           fits in @c parent_len's @c uint16_t cells.
 * @param[in]     nxt        Constant successor cost `dp[p] + transition`,
 *                           shared across the [L, L_end) span.
 * @param[in]     off_biased Match offset minus ::ZXC_LZ_OFFSET_BIAS, the
 *                           value stored when a transition wins.
 * @return The first L value not processed (i.e., @p L_end on success).
 */
// codeql[cpp/unused-static-function]: false positive
static ZXC_ALWAYS_INLINE size_t zxc_opt_dp_update_const_cost(
    uint32_t* RESTRICT dp, uint16_t* RESTRICT parent_len, uint16_t* RESTRICT parent_off,
    const size_t p, size_t L, const size_t L_end, const uint32_t nxt, const uint16_t off_biased) {
#if defined(ZXC_USE_AVX512) && defined(__AVX512VL__)
    if (L + 16 <= L_end) {
        const __m512i v_inc =
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i v_nxt = _mm512_set1_epi32((int)nxt);
        const __m256i v_off = _mm256_set1_epi16((int16_t)off_biased);
        for (; L + 16 <= L_end; L += 16) {
            const __m512i v_L_lanes = _mm512_add_epi32(v_inc, _mm512_set1_epi32((int)L));
            const __m512i v_dp = _mm512_loadu_si512((const void*)&dp[p + L]);
            const __mmask16 m = _mm512_cmplt_epu32_mask(v_nxt, v_dp);
            _mm512_mask_storeu_epi32(&dp[p + L], m, v_nxt);
            const __m256i v_L_u16 = _mm512_cvtusepi32_epi16(v_L_lanes);
            _mm256_mask_storeu_epi16((void*)&parent_len[p + L], m, v_L_u16);
            _mm256_mask_storeu_epi16((void*)&parent_off[p + L], m, v_off);
        }
    }
#elif defined(ZXC_USE_AVX2)
    if (L + 8 <= L_end) {
        const __m256i v_inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i v_nxt = _mm256_set1_epi32((int)nxt);
        const __m256i v_bias = _mm256_set1_epi32((int)0x80000000);
        const __m256i v_nxt_b = _mm256_xor_si256(v_nxt, v_bias);
        const __m128i v_off = _mm_set1_epi16((int16_t)off_biased);
        for (; L + 8 <= L_end; L += 8) {
            const __m256i v_L_lanes = _mm256_add_epi32(v_inc, _mm256_set1_epi32((int)L));
            const __m256i v_dp = _mm256_loadu_si256((const __m256i*)&dp[p + L]);
            // Unsigned-compare-via-bias trick:
            //   (dp ^ 0x80000000) > (nxt ^ 0x80000000)  iff  dp > nxt
            // because XOR with the sign bit maps unsigned ordering to
            // signed ordering. AVX2 only has signed cmpgt for 32-bit.
            const __m256i v_dp_b = _mm256_xor_si256(v_dp, v_bias);
            const __m256i v_mask = _mm256_cmpgt_epi32(v_dp_b, v_nxt_b);
            const __m256i v_dp_new = _mm256_blendv_epi8(v_dp, v_nxt, v_mask);
            _mm256_storeu_si256((__m256i*)&dp[p + L], v_dp_new);
            // Pack 8x int32 mask -> 8x int16 mask with signed saturation:
            // 0xFFFFFFFF -> 0xFFFF, 0x00000000 -> 0x0000.
            const __m128i v_mask16 = _mm_packs_epi32(_mm256_castsi256_si128(v_mask),
                                                     _mm256_extracti128_si256(v_mask, 1));
            const __m128i v_L_u16 = _mm_packus_epi32(_mm256_castsi256_si128(v_L_lanes),
                                                     _mm256_extracti128_si256(v_L_lanes, 1));
            const __m128i v_pl = _mm_loadu_si128((const __m128i*)&parent_len[p + L]);
            const __m128i v_pl_new = _mm_blendv_epi8(v_pl, v_L_u16, v_mask16);
            _mm_storeu_si128((__m128i*)&parent_len[p + L], v_pl_new);
            const __m128i v_po = _mm_loadu_si128((const __m128i*)&parent_off[p + L]);
            const __m128i v_po_new = _mm_blendv_epi8(v_po, v_off, v_mask16);
            _mm_storeu_si128((__m128i*)&parent_off[p + L], v_po_new);
        }
    }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    if (L + 4 <= L_end) {
        static const uint32_t k_inc_array[4] = {0, 1, 2, 3};
        const uint32x4_t v_inc = vld1q_u32(k_inc_array);
        const uint32x4_t v_nxt = vdupq_n_u32(nxt);
        const uint16x4_t v_off = vdup_n_u16(off_biased);
        for (; L + 4 <= L_end; L += 4) {
            const uint32x4_t v_L_lanes = vaddq_u32(v_inc, vdupq_n_u32((uint32_t)L));
            const uint32x4_t v_dp = vld1q_u32(&dp[p + L]);
            const uint32x4_t v_mask = vcgtq_u32(v_dp, v_nxt);
            vst1q_u32(&dp[p + L], vbslq_u32(v_mask, v_nxt, v_dp));
            const uint16x4_t v_mask16 = vmovn_u32(v_mask);
            const uint16x4_t v_L_u16 = vqmovn_u32(v_L_lanes);
            const uint16x4_t v_pl = vld1_u16(&parent_len[p + L]);
            vst1_u16(&parent_len[p + L], vbsl_u16(v_mask16, v_L_u16, v_pl));
            const uint16x4_t v_po = vld1_u16(&parent_off[p + L]);
            vst1_u16(&parent_off[p + L], vbsl_u16(v_mask16, v_off, v_po));
        }
    }
#elif defined(ZXC_OPT_SSE2)
    if (L + 4 <= L_end) {
        const __m128i v_inc = _mm_setr_epi32(0, 1, 2, 3);
        const __m128i v_nxt = _mm_set1_epi32((int)nxt);
        const __m128i v_bias = _mm_set1_epi32((int)0x80000000);
        const __m128i v_nxt_b = _mm_xor_si128(v_nxt, v_bias);
        const __m128i v_off = _mm_set1_epi16((short)off_biased);
        for (; L + 4 <= L_end; L += 4) {
            const __m128i v_L_lanes = _mm_add_epi32(v_inc, _mm_set1_epi32((int)L));
            const __m128i v_dp = _mm_loadu_si128((const __m128i*)&dp[p + L]);
            // Unsigned compare via sign-bit bias (SSE2 cmpgt is signed only):
            //   (dp ^ 0x80000000) > (nxt ^ 0x80000000)  iff  dp > nxt.
            const __m128i v_dp_b = _mm_xor_si128(v_dp, v_bias);
            const __m128i v_mask = _mm_cmpgt_epi32(v_dp_b, v_nxt_b);
            const __m128i v_dp_new = zxc_mm_blendv_epi8_sse2(v_dp, v_nxt, v_mask);
            _mm_storeu_si128((__m128i*)&dp[p + L], v_dp_new);
            // Narrow the 4x int32 mask / length lanes to 4x int16 (low 64 bits).
            // packs: 0xFFFFFFFF -> 0xFFFF, 0 -> 0; packus (SSE2-emulated): u32->u16.
            const __m128i v_mask16 = _mm_packs_epi32(v_mask, v_mask);
            const __m128i v_L_u16 = zxc_mm_packus_epi32_sse2(v_L_lanes, v_L_lanes);
            __m128i v_pl = _mm_loadl_epi64((const __m128i*)&parent_len[p + L]);
            v_pl = zxc_mm_blendv_epi8_sse2(v_pl, v_L_u16, v_mask16);
            _mm_storel_epi64((__m128i*)&parent_len[p + L], v_pl);
            __m128i v_po = _mm_loadl_epi64((const __m128i*)&parent_off[p + L]);
            v_po = zxc_mm_blendv_epi8_sse2(v_po, v_off, v_mask16);
            _mm_storel_epi64((__m128i*)&parent_off[p + L], v_po);
        }
    }
#endif
    // Scalar tail (and full path on archs without SIMD).
    // L < L_end <= UINT16_MAX (caller precondition), so the cast is lossless.
    for (; L < L_end; L++) zxc_opt_relax(dp, parent_len, parent_off, p, L, nxt, off_biased);
    return L;
}

/**
 * @brief Estimate per-block literal cost from a sampled histogram passed
 *        through the actual length-limited Huffman builder.
 *
 * Strategy: build a strided sample of @p src (4096 entries), run the same
 * length-limited Huffman code construction the encoder uses, and report the
 * sample-weighted average code length. This is the predicted bits/byte
 * for Huffman-encoded literals on this distribution: no calibration
 * constants, no per-corpus tuning. The cap at 8 reflects that RAW is
 * always available at exactly that cost; if Huffman doesn't beat 8 on the
 * sample, the encoder will pick RAW and 8 is the right price.
 *
 * @param[in] src     Source buffer for the block.
 * @param[in] src_sz  Length of @p src in bytes.
 * @param[in] scratch Package-merge scratch (pre-allocated in the cctx for
 *                    level >= 6). May be `NULL`, in which case the builder
 *                    allocates its own working memory.
 * @return Estimated literal cost in bits, in `[1, 8]`.
 */
// codeql[cpp/unused-static-function]: false positive
static uint32_t zxc_opt_estimate_lit_bits(const uint8_t* RESTRICT src, const size_t src_sz,
                                          void* RESTRICT scratch) {
    if (UNLIKELY(src_sz < ZXC_OPT_LIT_SAMPLE_MIN)) return CHAR_BIT;

    uint32_t hist[ZXC_HUF_NUM_SYMBOLS] = {0};
    const size_t step =
        (src_sz > ZXC_OPT_LIT_SAMPLE_TARGET) ? src_sz / ZXC_OPT_LIT_SAMPLE_TARGET : 1U;
    size_t sampled = 0;
    for (size_t i = 0; i < src_sz; i += step) {
        hist[src[i]]++;
        sampled++;
    }

    uint8_t code_len[ZXC_HUF_NUM_SYMBOLS];
    if (UNLIKELY(zxc_huf_build_code_lengths(hist, code_len, scratch,
                                            ZXC_HUF_MAX_CODE_LEN_DENSITY) != ZXC_OK))
        return CHAR_BIT;

    // Sample-weighted sum of code lengths = predicted Huffman bits; divide by
    // the sample count for bits/byte, rounded up. The DP is integer, and
    // rounding up favours matches over fractional-cost literals.
    uint32_t total_bits = 0;
    for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++) {
        total_bits += hist[k] * (uint32_t)code_len[k];
    }
    const uint32_t n = (uint32_t)sampled;
    const uint32_t avg = (total_bits + n - 1) / n;

    // Cap at RAW cost: if Huffman can't beat 8 bits/byte on the sample,
    // the encoder will pick RAW anyway and 8 is the actual literal cost.
    return (avg < CHAR_BIT) ? avg : CHAR_BIT;
}

/**
 * @brief Static price-based optimal LZ77 parser for level 6.
 *
 * Forward DP over the block's positions: `dp[p]` = min bit cost to encode
 * `src[0..p)`. Per-position transitions are
 *   - literal: `dp[p+1] = min(dp[p+1], dp[p] + lit_cost`
 *   - match  : `dp[p+L] = min(dp[p+L], dp[p] + match_cost(L))` for L in
 *              `[MIN_MATCH, max_L]`
 * where `max_L` is the longest match found by ::zxc_lz77_find_best_match at
 * `p` (with lazy disabled, the DP itself handles position-based
 * optimization). Backtracking from `dp[src_sz]` reconstructs the
 * optimal token sequence.
 *
 * Complexity guard: ::ZXC_OPT_LONG_MATCH_SKIP causes ::zxc_lz77_find_best_match
 * to be skipped at positions strictly inside a long match, without this
 * guard, highly repetitive data (e.g. Lorem-loop with multi-MB matches at
 * every offset) makes the parser quadratic and unit tests run for minutes.
 * The inner sub-length update visits every L from `MIN_MATCH` to `max_L`,
 * except at a covered position (see the loop), which relaxes at most two
 * cells; the skip threshold means each long-match region only pays its O(L)
 * cost once at the starting position, keeping total work O(N).
 *
 * @param[in,out] ctx           Compression context. The lazy-allocated
 *                              `opt_scratch` field provides the DP arrays;
 *                              it is grown on first use and reused on
 *                              subsequent blocks.
 * @param[in]  src              Source buffer to parse.
 * @param[in]  src_sz           Length of @p src in bytes.
 * @param[in,out] hash_table    LZ77 hash table (epoch | position entries).
 * @param[in,out] hash_tags     8-bit fast-rejection tags paired with @p hash_table.
 * @param[in,out] chain_table   Hash-chain link table (ring buffer).
 * @param[in]  epoch_mark       Current epoch shifted into the high bits.
 * @param[in]  offset_mask      Mask isolating the position bits in chain entries.
 * @param[in]  level            Compression level (used to size the matcher).
 * @param[out] literals         Buffer receiving the gathered literal bytes.
 * @param[out] buf_tokens       Buffer receiving the per-sequence token bytes.
 * @param[out] buf_offsets      Buffer receiving the per-sequence offsets.
 * @param[out] buf_extras       Buffer receiving variable-length overflow data.
 * @param[out] seq_c_out        Number of emitted sequences.
 * @param[out] lit_c_out        Number of literal bytes written into @p literals.
 * @param[out] extras_sz_out    Number of bytes written into @p buf_extras.
 * @param[out] max_offset_out   Largest biased offset emitted (used by the caller
 *                              to choose 1-byte vs 2-byte offset encoding).
 * @return @c ZXC_OK on success, or a negative @ref zxc_error_t.
 */
static int zxc_lz77_optimal_parse_spl(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint32_t* RESTRICT hash_table,
                                      uint8_t* RESTRICT hash_tags, uint16_t* RESTRICT chain_table,
                                      const uint32_t epoch_mark, const uint32_t offset_mask,
                                      const int level, uint8_t* RESTRICT literals,
                                      uint8_t* RESTRICT buf_tokens, uint16_t* RESTRICT buf_offsets,
                                      uint8_t* RESTRICT buf_extras, uint32_t* RESTRICT seq_c_out,
                                      size_t* RESTRICT lit_c_out, size_t* RESTRICT extras_sz_out,
                                      uint16_t* RESTRICT max_offset_out) {
    zxc_lz77_params_t lzp_opt = zxc_get_lz77_params(level);
    lzp_opt.use_lazy = 0;  // guard

    // With a dictionary, src = [dict | block]. The DP arrays index from the block
    // start, so src_base points at the first block byte for literal copies while
    // src stays the match finder's base.
    const size_t dict_sz = ctx->dict_size;
    const size_t block_sz = src_sz - dict_sz;
    const uint8_t* const src_base = src + dict_sz;
    const uint8_t* const iend = src + src_sz;

    // Block too small for any match: emit all as literals.
    if (UNLIKELY(block_sz < ZXC_LZ_SEARCH_MARGIN + 1)) {
        if (block_sz > 0) ZXC_MEMCPY(literals, src_base, block_sz);
        *lit_c_out = block_sz;
        *seq_c_out = 0;
        *extras_sz_out = 0;
        *max_offset_out = 0;
        return 0;
    }

    const size_t search_limit_pos = block_sz - ZXC_LZ_SEARCH_MARGIN;
    const uint8_t* const search_limit = src + search_limit_pos;

    // DP arrays carved from ctx->opt_scratch, one allocation reused across
    // blocks, each sub-buffer cache-line padded.
    //
    //   dp             : (chunk+1) x u32: min cost to reach position p.
    //   parent_len     : (chunk+1) x u16: 0 = literal, >= MIN_MATCH = match.
    //   parent_off     : (chunk+1) x u16: biased match offset (distance-1).
    //   match_end_bits : one bit per position, set when a match on the chosen
    //                    path ends there - a forward actions[] stack at 1/64
    //                    the cost.
    //
    // The same buffer doubles as package-merge scratch for the code-length
    // builder: that scratch is live before the DP and after the parse is read
    // out, never during, so capacity is just the larger of the two.
    size_t sz_dp;
    size_t sz_pl;
    size_t sz_po;
    size_t sz_bm;
    zxc_opt_dp_sizes(ctx->chunk_size, &sz_dp, &sz_pl, &sz_po, &sz_bm);
    const size_t n_bm_words = ZXC_BITMAP_WORDS(ctx->chunk_size + 1);

    // Per-block literal cost (sample only block data, not dict prefix):
    const uint32_t lit_cost = zxc_opt_estimate_lit_bits(src_base, block_sz, ctx->opt_scratch);

    uint32_t* const dp = (uint32_t*)ctx->opt_scratch;
    uint16_t* const parent_len = (uint16_t*)(ctx->opt_scratch + sz_dp);
    uint16_t* const parent_off = (uint16_t*)(ctx->opt_scratch + sz_dp + sz_pl);
    uint64_t* const match_end_bits = (uint64_t*)(ctx->opt_scratch + sz_dp + sz_pl + sz_po);

    dp[0] = 0;
    ZXC_MEMSET(dp + 1, 0xFF, block_sz * sizeof(uint32_t));
    ZXC_MEMSET(match_end_bits, 0, sz_bm);

    // Forward DP: visit every position, update reachable successors.
    // `skip_until` suppresses find_best_match strictly inside the last long
    // match - the transition from its start already covers dp[p+1..p+L], and
    // re-searching there is what makes the parser quadratic on repetitive data.
    size_t skip_until = 0;
    // Rolling repeat-offset seed for find_best_match
    uint32_t last_off = 0;
    // Previous matched position and the length relaxed from it.
    size_t prev_p = 0;
    size_t prev_len = 0;
    for (size_t p = 0; p < search_limit_pos; p++) {
        if (UNLIKELY(dp[p] == UINT32_MAX)) continue;

        // Literal transition.
        const uint32_t lit_next = dp[p] + lit_cost;
        if (lit_next < dp[p + 1]) {
            dp[p + 1] = lit_next;
            parent_len[p + 1] = 0;
        }

        if (p < skip_until) continue;

        // Match transition: no lazy, no backtrack (anchor=ip). Sub-lengths are
        // iterated because any L <= max_L matches at the same offset and may end
        // at a more useful DP position. ip is absolute so dict references
        // resolve against src.
        const uint8_t* ip = src_base + p;
        const zxc_match_t m = zxc_lz77_find_best_match(
            src, ip, iend, search_limit, /*anchor=*/ip, hash_table, hash_tags, chain_table,
            epoch_mark, offset_mask, level, lzp_opt, last_off);

        if (m.ref) {
            const uint32_t off = (uint32_t)(ip - m.ref);
            if (off > 0 && off <= ZXC_LZ_WINDOW_SIZE) {
                last_off = off;
                const size_t L_max_raw = (m.len > block_sz - p) ? (block_sz - p) : (size_t)m.len;
                const size_t L_max = (L_max_raw > UINT16_MAX) ? UINT16_MAX : L_max_raw;

                // Cost is piecewise constant in varint segments, so split
                // [MIN_MATCH, L_max] into:
                //   1. cheap   : v < ML_MASK                   -> base
                //   2. varint1 : v in [ML_MASK, ML_MASK + VARINT_1B_LIMIT) -> base + 8
                //   3. varint2+: v >= ML_MASK + VARINT_1B_LIMIT            -> base + 16, ...
                // The first two have a constant nxt and vectorize; the third is
                // rare enough to stay scalar.
                const uint16_t off_biased = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);
                const size_t L_max_plus = L_max + 1;
                const size_t L_cheap_end = ZXC_LZ_MIN_MATCH_LEN + ZXC_TOKEN_ML_MASK;
                const size_t L_v1_end = L_cheap_end + ZXC_VARINT_1B_LIMIT;
                const uint32_t nxt_cheap = dp[p] + ZXC_OPT_MATCH_COST_BASE;
                const uint32_t nxt_v1 = nxt_cheap + CHAR_BIT;

                // Covered: each processed position q bounds dp[q+l] by
                // dp[q] + cost(l), by relaxing or through its own predecessor.
                // With dp[p] >= dp[p-1], length l only wins where cost(l) <
                // cost(l+1): the last length of the two classes below
                // ZXC_OPT_LONG_MATCH_SKIP. Costs ignore the offset.
                const int covered = prev_len > L_max && prev_p + 1 == p && dp[p] >= dp[p - 1];
                prev_p = p;
                prev_len = L_max;

                if (covered) {
                    if (L_max_plus >= L_cheap_end)
                        zxc_opt_relax(dp, parent_len, parent_off, p, L_cheap_end - 1, nxt_cheap,
                                      off_biased);
                    if (L_max_plus >= L_v1_end)
                        zxc_opt_relax(dp, parent_len, parent_off, p, L_v1_end - 1, nxt_v1,
                                      off_biased);
                } else {
                    // 1. Cheap range.
                    size_t L = zxc_opt_dp_update_const_cost(
                        dp, parent_len, parent_off, p, ZXC_LZ_MIN_MATCH_LEN,
                        (L_max_plus < L_cheap_end) ? L_max_plus : L_cheap_end, nxt_cheap,
                        off_biased);

                    // 2. First varint level (1-byte extension).
                    if (L < L_max_plus)
                        L = zxc_opt_dp_update_const_cost(
                            dp, parent_len, parent_off, p, L,
                            (L_max_plus < L_v1_end) ? L_max_plus : L_v1_end, nxt_v1, off_biased);

                    // 3. Higher varint levels: variable cost, kept scalar.
                    // Reached only by L >= ML_MASK + 128 + MIN_MATCH, so the
                    // v >= ML_MASK guard from the original loop is implied.
                    for (; L < L_max_plus; L++) {
                        uint32_t cost = ZXC_OPT_MATCH_COST_BASE;
                        uint32_t v = (uint32_t)(L - ZXC_LZ_MIN_MATCH_LEN) - ZXC_TOKEN_ML_MASK;
                        cost += CHAR_BIT;
                        while (v >= 128) {
                            v >>= 7;
                            cost += CHAR_BIT;
                        }
                        zxc_opt_relax(dp, parent_len, parent_off, p, L, dp[p] + cost, off_biased);
                    }
                }
                if (UNLIKELY(L_max >= ZXC_OPT_LONG_MATCH_SKIP)) skip_until = p + L_max - 1;
            }
        }
    }

    // Tail (last ZXC_LZ_SEARCH_MARGIN bytes) can only be literals: the match finder
    // stops at search_limit so its 8-byte probe reads stay in bounds.
    for (size_t p = search_limit_pos; p < block_sz; p++) {
        if (UNLIKELY(dp[p] == UINT32_MAX)) continue;
        const uint32_t lit_next = dp[p] + lit_cost;
        if (lit_next < dp[p + 1]) {
            dp[p + 1] = lit_next;
            parent_len[p + 1] = 0;
        }
    }

    // Backtrack from src_sz to 0. Only match endpoints are recorded; literals
    // are the unmarked runs between them, rebuilt at emission from lit_start,
    // so they need no backtrack storage.
    {
        size_t pos = block_sz;
        while (pos > 0) {
            const uint32_t L = parent_len[pos];
            if (L == 0) {
                pos -= 1;
            } else {
                match_end_bits[pos >> 6] |= (uint64_t)1 << (pos & 63);
                pos -= L;
            }
        }
    }

    // Forward emission: walk match_end_bits word-by-word, peeling set bits
    // with ctzll. Each set bit gives a match endpoint; parent_len/parent_off
    // at that position recover (length, offset).
    uint32_t seq_c = 0;
    size_t lit_c = 0;
    size_t extras_sz = 0;
    uint16_t max_offset = 0;
    size_t lit_start = 0;

    for (size_t word_idx = 0; word_idx < n_bm_words; word_idx++) {
        uint64_t w = match_end_bits[word_idx];
        while (w) {
            const size_t pos = (word_idx << 6) + (size_t)zxc_ctz64(w);
            w &= w - 1;
            const uint32_t L = parent_len[pos];
            const uint16_t off_biased = parent_off[pos];
            const size_t match_start = pos - L;

            const size_t LL = match_start - lit_start;
            if (LL > 0) {
                ZXC_MEMCPY(literals + lit_c, src_base + lit_start, LL);
                lit_c += LL;
            }
            const uint32_t ll = (uint32_t)LL;
            const uint32_t ml = L - ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(!zxc_spl_put_seq(buf_tokens, buf_offsets, buf_extras, &seq_c, &extras_sz,
                                          &max_offset, ll, ml, off_biased)))
                return ZXC_ERROR_OVERFLOW;
            lit_start = pos;
        }
    }

    // Tail literals after the last match (or all literals if no match).
    if (lit_start < block_sz) {
        const size_t tail = block_sz - lit_start;
        ZXC_MEMCPY(literals + lit_c, src_base + lit_start, tail);
        lit_c += tail;
    }

    *seq_c_out = seq_c;
    *lit_c_out = lit_c;
    *extras_sz_out = extras_sz;
    *max_offset_out = max_offset;
    return 0;
}

/**
 * @brief Seeds the hash/chain tables from dictionary content prepended to @p src.
 *
 * When a dictionary is active, @p src is laid out as [dict_content | block_data].
 * This function inserts hash entries for dictionary positions [0, dict_size) so
 * the match finder can reference them during block encoding.
 *
 * @param[in]     src         Source buffer starting with dictionary content.
 * @param[in]     dict_size   Size of the dictionary prefix in bytes.
 * @param[in,out] hash_table  Hash table to seed with dictionary positions.
 * @param[in,out] hash_tags   Tag table for fast match rejection.
 * @param[in,out] chain_table Chain table for collision resolution.
 * @param[in]     epoch_mark  Current epoch marker for hash table entries.
 * @param[in]     offset_mask Position mask for epoch/offset encoding.
 * @param[in]     level       Compression level (controls hash function variant).
 */
static void zxc_lz_seed_dict(const uint8_t* RESTRICT src, const size_t dict_size,
                             uint32_t* RESTRICT hash_table, uint8_t* RESTRICT hash_tags,
                             uint16_t* RESTRICT chain_table, const uint32_t epoch_mark,
                             const uint32_t offset_mask, const int level) {
    if (UNLIKELY(dict_size < ZXC_LZ_MIN_MATCH_LEN)) return;

    const int use_hash5 = (level >= ZXC_LEVEL_DEFAULT);
    const size_t limit = dict_size - (ZXC_LZ_MIN_MATCH_LEN - 1);

    // Sparse seeding for the first half, dense for the second half.
    // Positions near the end of the dict produce shorter offsets and are
    // more likely to yield matches, so they deserve full coverage.
    const size_t half = limit / 2;
    for (size_t i = 0; i < half; i += 4) {
        const uint64_t val8 = zxc_le64(src + i);
        const uint32_t h = zxc_hash_func(val8, use_hash5);
        const uint32_t cur_pos = (uint32_t)i;

        hash_table[h] = epoch_mark | cur_pos;
        hash_tags[h] = zxc_hash_tag((uint32_t)val8);
        chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = 0;
    }
    for (size_t i = half; i < limit; i++) {
        const uint64_t val8 = zxc_le64(src + i);
        const uint32_t h = zxc_hash_func(val8, use_hash5);
        const uint32_t cur_pos = (uint32_t)i;
        const uint32_t prev_idx = zxc_epoch_pos(hash_table[h], offset_mask, epoch_mark);

        hash_table[h] = epoch_mark | cur_pos;
        hash_tags[h] = zxc_hash_tag((uint32_t)val8);
        chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = zxc_chain_delta(cur_pos, prev_idx);
    }
}

/**
 * @brief Encodes one block in the SPL format (levels 3 and up).
 *
 * Two parsers feed the same serialiser. Levels 6 and 7 hand the whole parse to
 * zxc_lz77_optimal_parse_spl() and rejoin at @c parse_done; levels 3 to 5 run
 * the lazy loop here, with the search depth, lazy attempts and hash step taken
 * from zxc_get_lz77_params(). Either way the parse yields a literal run plus
 * token, offset and extras streams.
 *
 * The tail then picks an encoding per section - RAW against RLE for the
 * literals, plus a Huffman candidate from level 6 - sizes the sequence streams,
 * and writes the header, its descriptors and the payload.
 *
 * @param[in,out] ctx     Compression context: hash and chain tables, buffers, level.
 * @param[in]     src     Input, prefixed by the dictionary content when one is active.
 * @param[in]     src_sz  Size of @p src, that prefix included.
 * @param[out]    dst     Destination buffer.
 * @param[in]     dst_cap Capacity of @p dst.
 * @param[out]    out_sz  Receives the number of bytes written.
 * @return ZXC_OK, or a negative @ref zxc_error_t, typically
 *         @ref ZXC_ERROR_DST_TOO_SMALL.
 */
static int zxc_encode_block_spl(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, size_t dst_cap,
                                size_t* RESTRICT out_sz) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    const int level = ctx->compression_level;
    const size_t dict_sz = ctx->dict_size;

    zxc_lz77_params_t lzp = zxc_get_lz77_params(level);
    // The floor is a candidate until the block itself clears it.
    if (lzp.min_offset > 1 && zxc_block_is_short_dist_bound(src + dict_sz, src_sz - dict_sz))
        lzp.min_offset = 1;

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ctx->max_epoch)) {
        ZXC_MEMSET(ctx->hash_table, 0, ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ZXC_MEMSET(ctx->hash_tags, 0, ZXC_LZ_HASH_SIZE * sizeof(uint8_t));
        ctx->epoch = 1;
    }
    const uint32_t offset_bits = ctx->offset_bits;
    const uint32_t offset_mask = ctx->offset_mask;
    const uint32_t epoch_mark = ctx->epoch << offset_bits;

    if (dict_sz > 0)
        zxc_lz_seed_dict(src, dict_sz, ctx->hash_table, ctx->hash_tags, ctx->chain_table,
                         epoch_mark, offset_mask, level);

    const uint8_t* ip = src + dict_sz;
    const uint8_t* iend = src + src_sz;
    const uint8_t* anchor = ip;
    const uint8_t* search_limit = iend - ZXC_LZ_SEARCH_MARGIN;

    uint32_t* const hash_table = ctx->hash_table;
    uint8_t* const hash_tags = ctx->hash_tags;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint8_t* const buf_tokens = ctx->buf_tokens;
    uint16_t* const buf_offsets = ctx->buf_offsets;
    uint8_t* const buf_extras = ctx->buf_extras;

    uint32_t seq_c = 0;
    size_t lit_c = 0;
    size_t extras_sz = 0;
    uint16_t max_offset = 0;  // Track max offset for 1-byte/2-byte mode decision

    // Level 6+: price-based optimal parser (fills outputs and skips the
    // lazy loop + last_lits handling below via `goto parse_done`).
    if (level >= ZXC_LEVEL_DENSITY) {
        const int rc = zxc_lz77_optimal_parse_spl(
            ctx, src, src_sz, hash_table, hash_tags, chain_table, epoch_mark, offset_mask, level,
            literals, buf_tokens, buf_offsets, buf_extras, &seq_c, &lit_c, &extras_sz, &max_offset);
        if (UNLIKELY(rc != 0)) return rc;
        goto parse_done;
    }

    while (LIKELY(ip < search_limit)) {
        const size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= search_limit)) step = 1;

        if (LIKELY(ip + step + sizeof(uint64_t) <= iend)) {
            const uint64_t v_next = zxc_le64(ip + step);
            // cppcheck-suppress unreadVariable
            const uint32_t h_next = zxc_hash_func(v_next, 1);
            ZXC_PREFETCH_READ(&hash_tags[h_next]);
            ZXC_PREFETCH_READ(&hash_table[h_next]);
        }

        const zxc_match_t m =
            zxc_lz77_find_best_match(src, ip, iend, search_limit, anchor, hash_table, hash_tags,
                                     chain_table, epoch_mark, offset_mask, level, lzp,
                                     /*last_off=*/0U);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = m.len - ZXC_LZ_MIN_MATCH_LEN;
            const uint32_t off = (uint32_t)(ip - m.ref);
            const uint16_t off_biased = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);

            if (ll > 0) zxc_flush_literals(literals, &lit_c, anchor, ll, iend);

            if (UNLIKELY(!zxc_spl_put_seq(buf_tokens, buf_offsets, buf_extras, &seq_c, &extras_sz,
                                          &max_offset, ll, ml, off_biased)))
                return ZXC_ERROR_OVERFLOW;

            if (m.len > 2 && level > ZXC_LEVEL_BALANCED) {
                const uint8_t* match_end = ip + m.len;
                if (match_end < iend - 7) {
                    const uint32_t pos_u = (uint32_t)((match_end - 2) - src);
                    const uint64_t val_u8 = zxc_le64(match_end - 2);
                    const uint32_t val_u = (uint32_t)val_u8;
                    const uint32_t h_u = zxc_hash_func(val_u8, 1);
                    const uint32_t prev_idx =
                        zxc_epoch_pos(hash_table[h_u], offset_mask, epoch_mark);
                    hash_table[h_u] = epoch_mark | pos_u;
                    hash_tags[h_u] = zxc_hash_tag(val_u);
                    chain_table[pos_u & ZXC_LZ_WINDOW_MASK] = zxc_chain_delta(pos_u, prev_idx);
                }
            }

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = (size_t)(iend - anchor);
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

parse_done:;
    // Per-block match splitting on the parsed block (see zxc_spl_match_split_block).
    seq_c = zxc_spl_match_split_block(buf_tokens, buf_offsets, buf_extras, &extras_sz,
                                      ctx->buf_split, ctx->buf_split_hist, seq_c, lzp.split_max);
    // Dictionary-table trainer hook: accumulate the REAL post-LZ literal
    // frequencies (see zxc_dict_train_huf). Cold path, NULL outside training.
    if (UNLIKELY(ctx->lit_freq_acc)) {
        for (size_t i = 0; i < lit_c; i++) ctx->lit_freq_acc[literals[i]]++;
    }

    // --- RLE ANALYSIS ---
    size_t rle_size = 0;
    int enc_lit = ZXC_SECTION_ENCODING_RAW;
    size_t best_j = lit_c;

    if (lit_c > 0) {
        const uint8_t* p = literals;
        const uint8_t* const p_end = literals + lit_c;
        const uint8_t* const p_end_4 = p_end - 3;  // Safe limit for 4-byte lookahead

        while (LIKELY(p < p_end)) {
            const uint8_t b = *p;
            const uint8_t* run_start = p++;

            // Fast run counting with early SIMD exit
#if defined(ZXC_USE_AVX512)
            const __m512i vb = _mm512_set1_epi8((char)b);
            while (p <= p_end - 64) {
                const __m512i v = _mm512_loadu_si512((const void*)p);
                const __mmask64 mask = _mm512_cmpeq_epi8_mask(v, vb);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) {
                    p += (size_t)zxc_ctz64(~mask);
                    goto _run_done;
                }
                p += 64;
            }
#elif defined(ZXC_USE_AVX2)
            const __m256i vb = _mm256_set1_epi8((char)b);
            while (p <= p_end - 32) {
                const __m256i v = _mm256_loadu_si256((const __m256i*)p);
                const uint32_t mask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vb));
                if (mask != 0xFFFFFFFF) {
                    p += zxc_ctz32(~mask);
                    goto _run_done;
                }
                p += 32;
            }
#elif defined(ZXC_USE_SSE2)
            const __m128i vb = _mm_set1_epi8((char)b);
            while (p <= p_end - 16) {
                const __m128i v = _mm_loadu_si128((const __m128i*)p);
                const uint32_t mask = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(v, vb));
                if (mask != 0xFFFFU) {
                    p += zxc_ctz32(~mask);
                    goto _run_done;
                }
                p += 16;
            }
#elif defined(ZXC_USE_NEON64)
            const uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                const uint8x16_t v = vld1q_u8(p);
                const uint8x16_t eq = vceqq_u8(v, vb);
                // SHRN nibble-mask: see find_best_match above for rationale.
                const uint64_t mask =
                    vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
                if (LIKELY(mask == ~(uint64_t)0)) {
                    p += 16;
                } else {
                    p += (size_t)(zxc_ctz64(~mask) >> 2);
                    goto _run_done;
                }
            }
#elif defined(ZXC_USE_NEON32)
            uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                uint8x16_t v = vld1q_u8(p);
                uint8x16_t eq = vceqq_u8(v, vb);
                uint8x16_t not_eq = vmvnq_u8(eq);

                // 32-bit ARM NEON doesn't always support vgetq_lane_u64 / vreinterpretq_u64_u8 so
                // we treat the 128-bit vector as 4 x 32-bit lanes */
                const uint32x4_t neq32 = vreinterpretq_u32_u8(not_eq);
                const uint32_t l0 = vgetq_lane_u32(neq32, 0);
                const uint32_t l1 = vgetq_lane_u32(neq32, 1);

                const uint64_t lo = ((uint64_t)l1 << 32) | l0;
                if (lo != 0) {
                    p += (size_t)(zxc_ctz64(lo) >> 3);
                    goto _run_done;
                }

                const uint32_t h0 = vgetq_lane_u32(neq32, 2);
                const uint32_t h1 = vgetq_lane_u32(neq32, 3);
                const uint64_t hi = ((uint64_t)h1 << 32) | h0;

                if (hi != 0) {
                    p += 8 + (zxc_ctz64(hi) >> 3);
                    goto _run_done;
                }
                p += 16;
            }
#endif
            while (p < p_end && *p == b) p++;

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32) || defined(ZXC_USE_SSE2)
        _run_done:;
#endif
            const size_t run = (size_t)(p - run_start);

            if (run >= ZXC_LIT_RLE_RUN_MIN) {
                // RLE run: 2 bytes per ZXC_LIT_RLE_RUN_MAX values, then remainder
                // Branchless: full_chunks * 2 + remainder handling
                const size_t full_chunks = run / ZXC_LIT_RLE_RUN_MAX;
                const size_t rem = run - full_chunks * ZXC_LIT_RLE_RUN_MAX;  // Avoid modulo
                rle_size += full_chunks * 2;
                // Remainder: if >= 4 -> 2 bytes (RLE), else 1 + rem (literal)
                if (rem >= ZXC_LIT_RLE_RUN_MIN)
                    rle_size += 2;
                else if (rem > 0)
                    rle_size += 1 + rem;
            } else {
                // Literal run: scan ahead with fast SIMD lookahead
                const uint8_t* lit_start = run_start;

#if defined(ZXC_RLE_WIN)
                while (p <= p_end_4 - ZXC_RLE_WIN) {
                    const zxc_rle_mask_t m = zxc_rle_pair_mask(p);
                    const zxc_rle_mask_t mask = m & (m >> ZXC_RLE_BPB) & (m >> (2 * ZXC_RLE_BPB));
                    if (LIKELY(mask == 0)) {
                        p += ZXC_RLE_WIN - 2;
                        continue;
                    }
                    const unsigned rpos = (unsigned)(ZXC_RLE_CTZ(mask) / ZXC_RLE_BPB);
                    const unsigned rpairs =
                        (unsigned)(ZXC_RLE_CTZ(~(m >> (rpos * ZXC_RLE_BPB))) / ZXC_RLE_BPB);
                    if (UNLIKELY(rpos + rpairs >= ZXC_RLE_WIN)) {
                        p += rpos;
                        goto _lit_done; /* run may extend past the window */
                    }
                    const size_t seg = (size_t)(p + rpos - lit_start);
                    rle_size += seg + (seg + ZXC_LIT_LEN_MASK) / ZXC_LIT_RUN_MAX;
                    rle_size += 2; /* run fits one chunk: full_chunks=0, rem>=4 */
                    p += rpos + rpairs + 1;
                    lit_start = p;
                }
#elif defined(ZXC_USE_NEON32)
                while (p <= p_end_4 - 16) {
                    uint8x16_t v0 = vld1q_u8(p);
                    uint8x16_t v1 = vld1q_u8(p + 1);
                    const uint8x16_t pe = vceqq_u8(v0, v1);
                    const uint8x16_t zv = vdupq_n_u8(0);
                    const uint8x16_t eq =
                        vandq_u8(pe, vandq_u8(vextq_u8(pe, zv, 1), vextq_u8(pe, zv, 2)));

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
                    p += 14;
                }
#endif
                while (p < p_end_4) {
                    if (UNLIKELY(p[0] == p[1] && p[1] == p[2] && p[2] == p[3])) break;
                    p++;
                }
                if (p >= p_end_4) p = p_end;

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32) || defined(ZXC_USE_SSE2)
            _lit_done:;
#endif
                const size_t lit_run = (size_t)(p - lit_start);
                // 1 header per ZXC_LIT_RUN_MAX bytes + all data bytes
                rle_size += lit_run + (lit_run + ZXC_LIT_LEN_MASK) / ZXC_LIT_RUN_MAX;
            }
        }

        // RLE over RAW: space-speed J comparison. The per-level premium
        // reproduces the historical margin exactly below ULTRA.
        const size_t rle_j = rle_size + ZXC_SS_TAX(lit_c, zxc_ss_prem_rle_q8(level));
        if (rle_j < best_j) {
            enc_lit = ZXC_SECTION_ENCODING_RLE;
            best_j = rle_j;
        }
    }

    // Level >= 6: Huffman as a third literal-encoding candidate. Histogram,
    // length-limited code lengths, exact PivCo section size, and switch if its
    // J (size + decode tax) beats the current winner's.
    uint8_t huf_code_len[ZXC_HUF_NUM_SYMBOLS];
    size_t huf_total_size = SIZE_MAX;
    uint32_t lit_freq[ZXC_HUF_NUM_SYMBOLS]; /* valid iff huf_total_size != SIZE_MAX */
    if (level >= ZXC_LEVEL_DENSITY && lit_c >= ZXC_HUF_MIN_LITERALS) {
        uint32_t freq0[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq1[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq2[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq3[ZXC_HUF_NUM_SYMBOLS] = {0};
        {
            size_t i = 0;
            for (; i + 4 <= lit_c; i += 4) {
                freq0[literals[i + 0]]++;
                freq1[literals[i + 1]]++;
                freq2[literals[i + 2]]++;
                freq3[literals[i + 3]]++;
            }
            for (; i < lit_c; i++) freq0[literals[i]]++;
        }
        uint32_t* const freq = lit_freq;
        for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++) {
            freq[k] = freq0[k] + freq1[k] + freq2[k] + freq3[k];
        }

        if (zxc_huf_build_code_lengths(freq, huf_code_len, ctx->opt_scratch,
                                       zxc_huf_enc_max_code_len(level)) == ZXC_OK) {
            (void)zxc_huf_nudge_code_lengths(freq, huf_code_len, ctx->opt_scratch,
                                             ctx->opt_scratch_cap, zxc_huf_enc_max_code_len(level));
            huf_total_size = zxc_huf_calc_size(freq, huf_code_len, 1);
            // Space-speed: the entropy candidate must beat the current winner's
            // J, paying its own decode tax over the copy path.
            if (huf_total_size != SIZE_MAX) {
                const size_t huf_j = huf_total_size + ZXC_SS_TAX(lit_c, zxc_ss_prem_huf_q8(level));
                if (huf_j < best_j) {
                    enc_lit = ZXC_SECTION_ENCODING_HUFFMAN;
                    best_j = huf_j;
                }
            }
        }
    }

    // Shared dictionary table (level >= 6, dict with a table): the HUFFMAN
    // bitstream minus the 128-byte header, so it stays viable far below
    // ZXC_HUF_MIN_LITERALS. A literal without a code drops the candidate.
    size_t huf_dict_total_size = SIZE_MAX;
    if (level >= ZXC_LEVEL_DENSITY && ctx->dict_huf_tree_ok && lit_c > 0) {
        // The dict candidate runs on sections too small for the per-block
        // table, so the histogram may not have been built above.
        if (huf_total_size == SIZE_MAX) {
            ZXC_MEMSET(lit_freq, 0, sizeof(uint32_t) * ZXC_HUF_NUM_SYMBOLS);
            for (size_t i = 0; i < lit_c; i++) lit_freq[literals[i]]++;
        }
        // zxc_huf_calc_size returns SIZE_MAX for unencodable candidates
        // (a literal byte without a code in the shared table).
        huf_dict_total_size =
            zxc_huf_calc_size_dict(lit_freq, ctx->dict_huf->code_len, &ctx->dict_huf->tree);
        if (huf_dict_total_size != SIZE_MAX) {
            // Same J rule against the running winner: vs the per-block Huffman
            // candidate the equal entropy taxes cancel (pure size comparison),
            // while RAW/RLE baselines keep their decode-tax edge.
            const size_t huf_dict_j =
                huf_dict_total_size + ZXC_SS_TAX(lit_c, zxc_ss_prem_huf_q8(level));
            // Last candidate: no need to keep best_j updated past this point.
            if (huf_dict_j < best_j) enc_lit = ZXC_SECTION_ENCODING_HUFFMAN_DICT;
        }
    }

    // Level-7 (ULTRA): Huffman-code the token stream with the literal codec -
    // tokens are a <= 256-symbol alphabet. Same J rule; the decoder picks the
    // path from gh.enc_tok. opt_scratch is free by now.
    uint8_t tok_code_len[ZXC_HUF_NUM_SYMBOLS];
    uint8_t enc_tok = ZXC_SECTION_ENCODING_RAW;
    size_t tok_huf_size = 0;
    uint32_t tfreq[ZXC_HUF_NUM_SYMBOLS];
    if (level >= ZXC_LEVEL_ULTRA && seq_c >= ZXC_HUF_MIN_LITERALS) {
        ZXC_MEMSET(tfreq, 0, sizeof(tfreq)); /* only the ULTRA path reads tfreq */
        for (uint32_t i = 0; i < seq_c; i++) tfreq[buf_tokens[i]]++;
        if (zxc_huf_build_code_lengths(tfreq, tok_code_len, ctx->opt_scratch,
                                       zxc_huf_enc_max_code_len(level)) == ZXC_OK) {
            (void)zxc_huf_nudge_code_lengths(tfreq, tok_code_len, ctx->opt_scratch,
                                             ctx->opt_scratch_cap, zxc_huf_enc_max_code_len(level));
            tok_huf_size = zxc_huf_calc_size(tfreq, tok_code_len, 1);
            // Space-speed J comparison (this path is ULTRA-only): the PivCo
            // token section pays the same decode tax as PivCo literals.
            if (tok_huf_size != SIZE_MAX &&
                tok_huf_size + ZXC_SS_TAX((size_t)seq_c, zxc_ss_prem_huf_q8(level)) < (size_t)seq_c)
                enc_tok = ZXC_SECTION_ENCODING_HUFFMAN;
        }
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_SPL};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    const int use_8bit_off = (max_offset <= 255) ? 1 : 0;
    const size_t off_stream_size = use_8bit_off ? seq_c : (seq_c * 2);

    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = (uint8_t)enc_lit,
                                 .enc_tok = enc_tok,
                                 .enc_mlen = 0,
                                 .enc_off = (uint8_t)use_8bit_off};

    const size_t sz_lit = (enc_lit == ZXC_SECTION_ENCODING_RLE)            ? rle_size
                          : (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN)      ? huf_total_size
                          : (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT) ? huf_dict_total_size
                                                                           : lit_c;
    const size_t sz_tok = (enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) ? tok_huf_size : seq_c;
    const size_t sz_off = off_stream_size;
    const size_t sz_ext = extras_sz;

    const int ghs = zxc_write_spl_header_and_desc(p, rem, &gh, (uint32_t)sz_lit, (uint32_t)sz_tok);
    if (UNLIKELY(ghs < 0)) return ghs;

    uint8_t* p_curr = p + ghs;
    rem -= (size_t)ghs;

    if (UNLIKELY(rem < sz_lit)) return ZXC_ERROR_DST_TOO_SMALL;

    if (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN) {
        const int written =
            zxc_huf_encode_section(literals, lit_c, lit_freq, huf_code_len, p_curr, rem);
        if (UNLIKELY(written < 0)) return written;
        if (UNLIKELY((size_t)written != huf_total_size)) return ZXC_ERROR_DST_TOO_SMALL;
        p_curr += written;
    } else if (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT) {
        const int written =
            zxc_huf_encode_section_dict(literals, lit_c, lit_freq, ctx->dict_huf->code_len,
                                        &ctx->dict_huf->tree, ctx->dict_huf->codes, p_curr, rem);
        if (UNLIKELY(written < 0)) return written;
        if (UNLIKELY((size_t)written != huf_dict_total_size)) return ZXC_ERROR_DST_TOO_SMALL;
        p_curr += written;
    } else if (enc_lit == ZXC_SECTION_ENCODING_RLE) {
        // Write RLE - optimized single-pass encoding
        const uint8_t* lit_ptr = literals;
        const uint8_t* const lit_end = literals + lit_c;

        while (lit_ptr < lit_end) {
            uint8_t b = *lit_ptr;
            const uint8_t* run_start = lit_ptr++;

            while (lit_ptr < lit_end && *lit_ptr == b) lit_ptr++;
            size_t run = (size_t)(lit_ptr - run_start);

            if (run >= ZXC_LIT_RLE_RUN_MIN) {
                // RLE runs: emit 2-byte tokens (header + value)
                while (run >= ZXC_LIT_RLE_RUN_MIN) {
                    size_t chunk = (run > ZXC_LIT_RLE_RUN_MAX) ? ZXC_LIT_RLE_RUN_MAX : run;
                    *p_curr++ = (uint8_t)(ZXC_LIT_RLE_FLAG | (chunk - ZXC_LIT_RLE_RUN_MIN));
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
                    if (UNLIKELY(lit_ptr + 3 < lit_end && lit_ptr[0] == lit_ptr[1] &&
                                 lit_ptr[1] == lit_ptr[2] && lit_ptr[2] == lit_ptr[3])) {
                        break;
                    }
                    lit_ptr++;
                }

                size_t lit_run = (size_t)(lit_ptr - lit_run_start);
                const uint8_t* src_ptr = lit_run_start;

                // Emit literal chunks (ZXC_LIT_RUN_MAX bytes each at most)
                while (lit_run > 0) {
                    size_t chunk = (lit_run > ZXC_LIT_RUN_MAX) ? ZXC_LIT_RUN_MAX : lit_run;
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

    if (UNLIKELY(rem < sz_tok)) return ZXC_ERROR_DST_TOO_SMALL;

    if (enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) {
        const int written =
            zxc_huf_encode_section(buf_tokens, seq_c, tfreq, tok_code_len, p_curr, rem);
        if (UNLIKELY(written < 0)) return written;
        if (UNLIKELY((size_t)written != tok_huf_size)) return ZXC_ERROR_DST_TOO_SMALL;
        p_curr += written;
    } else {
        ZXC_MEMCPY(p_curr, buf_tokens, seq_c);
        p_curr += seq_c;
    }
    rem -= sz_tok;

    if (UNLIKELY(rem < sz_off)) return ZXC_ERROR_DST_TOO_SMALL;

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

    // The sections behind the literals must total ZXC_BLOCK_LIT_SLACK bytes so
    // the decoder's literal wild-copy can overshoot. Tokens and offsets alone
    // clear it from 16 sequences on, so pad is 0 on any real block. It rides
    // inside the extras section, which the decoder sizes from the payload
    // residue - hence no wire field.
    const size_t behind_lit = sz_tok + sz_off + sz_ext;
    const size_t pad = (behind_lit < ZXC_BLOCK_LIT_SLACK) ? ZXC_BLOCK_LIT_SLACK - behind_lit : 0;

    if (UNLIKELY(rem < sz_ext + pad)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMCPY(p_curr, buf_extras, extras_sz);
    p_curr += extras_sz;

    ZXC_MEMSET(p_curr, 0, pad);
    p_curr += pad;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return ZXC_OK;
}

/**
 * @brief Encodes one block in the PAK format (levels 1 and 2, the speed path).
 *
 * What sets PAK apart is the sequence record. SPL packs a 1-byte token (4-bit
 * literal length, 4-bit match length) and escapes anything longer through a
 * varint; PAK spends a fixed 4 bytes instead - 8-bit LL, 8-bit ML (both
 * escaping at 255) and a 16-bit offset covering the whole 64 KB window. Lengths
 * between 16 and 255 are common, and this keeps them out of the varint reader,
 * which is what the decoder actually pays for.
 *
 * @param[in,out] ctx     Compression context: hash and chain tables, buffers, level.
 * @param[in]     src     Input, prefixed by the dictionary content when one is active.
 * @param[in]     src_sz  Size of @p src, that prefix included.
 * @param[out]    dst     Destination buffer.
 * @param[in]     dst_cap Capacity of @p dst.
 * @param[out]    out_sz  Receives the number of bytes written.
 * @return ZXC_OK, or a negative @ref zxc_error_t, typically
 *         @ref ZXC_ERROR_DST_TOO_SMALL.
 */
static int zxc_encode_block_pak(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    const int level = ctx->compression_level;
    const size_t dict_sz = ctx->dict_size;

    zxc_lz77_params_t lzp = zxc_get_lz77_params(level);
    // The floor is a candidate until the block itself clears it.
    if (lzp.min_offset > 1 && zxc_block_is_short_dist_bound(src + dict_sz, src_sz - dict_sz))
        lzp.min_offset = 1;

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ctx->max_epoch)) {
        ZXC_MEMSET(ctx->hash_table, 0, ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ZXC_MEMSET(ctx->hash_tags, 0, ZXC_LZ_HASH_SIZE * sizeof(uint8_t));
        ctx->epoch = 1;
    }
    const uint32_t offset_bits = ctx->offset_bits;
    const uint32_t offset_mask = ctx->offset_mask;
    const uint32_t epoch_mark = ctx->epoch << offset_bits;

    if (dict_sz > 0)
        zxc_lz_seed_dict(src, dict_sz, ctx->hash_table, ctx->hash_tags, ctx->chain_table,
                         epoch_mark, offset_mask, level);

    const uint8_t* ip = src + dict_sz;
    const uint8_t* iend = src + src_sz;
    const uint8_t* anchor = ip;
    const uint8_t* search_limit = iend - ZXC_LZ_SEARCH_MARGIN;

    uint32_t* const hash_table = ctx->hash_table;
    uint8_t* const hash_tags = ctx->hash_tags;
    uint8_t* const buf_extras = ctx->buf_extras;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint32_t* const buf_sequences = ctx->buf_sequences;

    uint32_t seq_c = 0;
    size_t extras_c = 0;
    size_t lit_c = 0;

    while (LIKELY(ip < search_limit)) {
        size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= search_limit)) step = 1;

        ZXC_PREFETCH_READ(ip + step * 4 + ZXC_CACHE_LINE_SIZE);

        if (LIKELY(ip + step + sizeof(uint64_t) <= iend)) {
            const uint64_t v_next = zxc_le64(ip + step);
            // cppcheck-suppress unreadVariable
            const uint32_t h_next = zxc_hash_func(v_next, 0);
            ZXC_PREFETCH_READ(&hash_tags[h_next]);
            ZXC_PREFETCH_READ(&hash_table[h_next]);
        }

        const zxc_match_t m =
            zxc_lz77_find_best_match(src, ip, iend, search_limit, anchor, hash_table, hash_tags,
                                     chain_table, epoch_mark, offset_mask, level, lzp,
                                     /*last_off=*/0U);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = m.len - ZXC_LZ_MIN_MATCH_LEN;
            const uint32_t off = (uint32_t)(ip - m.ref);

            if (ll > 0) zxc_flush_literals(literals, &lit_c, anchor, ll, iend);

            const uint32_t ll_write = (ll >= ZXC_SEQ_LL_MASK) ? ZXC_SEQ_LL_MASK : ll;
            const uint32_t ml_write = (ml >= ZXC_SEQ_ML_MASK) ? ZXC_SEQ_ML_MASK : ml;
            const uint32_t seq_val = (ll_write << ZXC_SEQ_LL_SHIFT) |
                                     (ml_write << ZXC_SEQ_ML_SHIFT) |
                                     ((off - ZXC_LZ_OFFSET_BIAS) & ZXC_SEQ_OFF_MASK);
            buf_sequences[seq_c] = seq_val;
            seq_c++;

            if (UNLIKELY(!zxc_emit_extra(buf_extras, &extras_c, ll, ZXC_SEQ_LL_MASK) ||
                         !zxc_emit_extra(buf_extras, &extras_c, ml, ZXC_SEQ_ML_MASK)))
                return ZXC_ERROR_OVERFLOW;

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = (size_t)(iend - anchor);
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_PAK};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = ZXC_SECTION_ENCODING_RAW,
                                 .enc_tok = 0,
                                 .enc_mlen = 0,
                                 .enc_off = 0};

    const int ghs = zxc_write_pak_header(p, rem, &gh);
    if (UNLIKELY(ghs < 0)) return ghs;

    uint8_t* p_curr = p + ghs;
    rem -= (size_t)ghs;

    const size_t sz_lit = lit_c;
    const size_t sz_seq = seq_c * sizeof(uint32_t);
    const size_t sz_ext = extras_c;

    // See the SPL twin. PAK literals are always RAW, so the slack always
    // matters here.
    const size_t behind_lit = sz_seq + sz_ext;
    const size_t pad = (behind_lit < ZXC_BLOCK_LIT_SLACK) ? ZXC_BLOCK_LIT_SLACK - behind_lit : 0;

    if (UNLIKELY(rem < sz_lit + sz_seq + sz_ext + pad)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMCPY(p_curr, literals, lit_c);
    p_curr += lit_c;

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

    ZXC_MEMSET(p_curr, 0, pad);
    p_curr += pad;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return ZXC_OK;
}

/**
 * @brief Encodes a RAW (stored) block: header plus a verbatim copy.
 *
 * Any block checksum is appended by the caller, not here.
 *
 * @param[in] src Pointer to the source data to encode.
 * @param[in] src_sz Size of the source data in bytes.
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] dst_cap Maximum capacity of the destination buffer.
 * @param[out] out_sz Pointer to a variable receiving the total written size
 * (header + data).
 *
 * @return ZXC_OK on success, or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) if the
 * destination buffer capacity is insufficient.
 */
static int zxc_encode_block_raw(const uint8_t* RESTRICT src, const size_t src_sz,
                                uint8_t* RESTRICT const dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE + src_sz)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_block_header_t bh;
    bh.block_type = ZXC_BLOCK_RAW;
    bh.block_flags = 0;  // Checksum flag moved to file header
    bh.reserved = 0;
    bh.comp_size = (uint32_t)src_sz;

    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    ZXC_MEMCPY(dst + ZXC_BLOCK_HEADER_SIZE, src, src_sz);

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + src_sz;
    return ZXC_OK;
}

/**
 * @brief Compresses one chunk into a single ZXC block (the compression hot path).
 *
 * Selects the PAK encoder at level <= 2, otherwise SPL; falls back to a RAW
 * block when the coded form would not shrink the data. When @c ctx->dict_size
 * is > 0, @p chunk is the [dict | block] concat and only the block tail counts
 * toward the expansion check. Appends the checksum when enabled, seeded with
 * @p block_index.
 */
// cppcheck-suppress unusedFunction
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT chunk,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                               const uint64_t block_index) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    const size_t dict_sz = ctx->dict_size;
    const size_t block_sz = src_sz - dict_sz;
    const uint8_t* block_data = chunk + dict_sz;
    size_t w = 0;
    int res = ZXC_OK;

    if (ctx->compression_level <= 2)
        res = zxc_encode_block_pak(ctx, chunk, src_sz, dst, dst_cap, &w);
    else
        res = zxc_encode_block_spl(ctx, chunk, src_sz, dst, dst_cap, &w);

    // Check expansion against block data size (excluding dict prefix).
    if (UNLIKELY(res != ZXC_OK || w >= block_sz)) {
        res = zxc_encode_block_raw(block_data, block_sz, dst, dst_cap, &w);
        if (UNLIKELY(res != ZXC_OK)) return res;
    }

    if (ctx->checksum_enabled) {
        if (UNLIKELY(w + ZXC_BLOCK_CHECKSUM_SIZE > dst_cap)) return ZXC_ERROR_OVERFLOW;

        zxc_store_le32(dst + w,
                       zxc_checksum(block_data, block_sz, block_index, ZXC_CHECKSUM_RAPIDHASH));
        w += ZXC_BLOCK_CHECKSUM_SIZE;
    }

    return (int)w;
}
