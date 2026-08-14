/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */

/**
 * @file zxc_huffman.c
 * @brief Canonical, length-limited Huffman codec in the PivCo layout (GLO
 *        entropy sections, wire values enc 2/3).
 *
 * Covers the whole entropy pipeline for GLO literal sections (level >= 6)
 * and level-7 token sections:
 *  - boundary package-merge -> optimal length-limited code lengths
 *    (<= ZXC_HUF_MAX_CODE_LEN_DENSITY at level 6, <= _ULTRA at level 7);
 *  - 128-byte packed lengths header (4-bit nibbles) pack/unpack + validation;
 *  - exact section sizing for the encoder's space-speed selection
 *    (zxc_huf_calc_size returns SIZE_MAX for unencodable candidates);
 *  - PivCo section encode/decode: the code is classical canonical Huffman,
 *    only the bit LAYOUT differs -- bits are grouped by tree level into
 *    per-node branch runs (see the section banner below and FORMAT.md
 *    section 5.2.1), so decoding runs data-parallel list merges
 *    (NEON TBL / SSSE3 pshufb / AVX-512-VBMI2 vpexpandb kernels, plus flat
 *    subtree unpacking and a leaf-pair fast path) instead of a serial
 *    bit chain.
 *
 * Public declarations live in zxc_internal.h; the shuffle-index tables live
 * in zxc_pivco_tables.c (single TU, shared by all variants); the rest is
 * private to this translation unit.
 */

/*
 * Function Multi-Versioning Support
 * With ZXC_FUNCTION_SUFFIX defined (e.g. _avx2), each variant TU gets its own
 * copy under a unique symbol, which the runtime dispatcher routes to. The
 * defines precede zxc_internal.h so the header's prototypes take the same
 * suffix as the definitions below.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_huf_build_code_lengths ZXC_CAT(zxc_huf_build_code_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_pack_lengths ZXC_CAT(zxc_huf_pack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_unpack_lengths ZXC_CAT(zxc_huf_unpack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size ZXC_CAT(zxc_huf_calc_size, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size_dict ZXC_CAT(zxc_huf_calc_size_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section ZXC_CAT(zxc_huf_encode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section_dict ZXC_CAT(zxc_huf_encode_section_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section ZXC_CAT(zxc_huf_decode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section_dict ZXC_CAT(zxc_huf_decode_section_dict, ZXC_FUNCTION_SUFFIX)
#endif

/* Mark the primary variant (only _default, or a no-suffix build) so ISA-
 * independent cold code compiles once, not in every per-ISA copy. Keyed off the
 * suffix value, so every build gets it with no extra flag. */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_PRIMARY__default 1
#define ZXC_PRIMARY_CAT_(a, b) a##b
#define ZXC_PRIMARY_CAT(a, b) ZXC_PRIMARY_CAT_(a, b)
#if ZXC_PRIMARY_CAT(ZXC_PRIMARY_, ZXC_FUNCTION_SUFFIX)
#define ZXC_VARIANT_PRIMARY 1
#endif
#else
#define ZXC_VARIANT_PRIMARY 1
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"
#include "zxc_pivco_tables.h"

/* ===========================================================================
 * Length-limited Huffman: boundary package-merge
 * ===========================================================================
 *
 * Builds optimal length-limited Huffman code lengths (max length
 * ZXC_HUF_MAX_CODE_LEN_ULTRA) on 256-symbol alphabets. Package-merge is run for
 * ZXC_HUF_MAX_CODE_LEN_ULTRA levels; each level holds up to 2N items (leaves +
 * paired packages). Selection of the cheapest 2N - 2 items at level
 * ZXC_HUF_MAX_CODE_LEN_ULTRA gives the appearance count of each leaf, which is
 * its code length.
 */

typedef zxc_huf_pm_item_t pm_item_t;

typedef struct {
    uint32_t w;
    int16_t sym;
} pm_leaf_t;

typedef zxc_huf_pm_frame_t frame_t;

/**
 * @brief Sort `pm_leaf_t` array by ascending weight, ties broken by ascending symbol.
 *
 * Bucket sort on `floor(log2(weight))` (32 buckets), with insertion sort
 * inside each bucket. Replaces a libc `qsort` call: the comparator's
 * indirect call dominated, and frequency distributions cluster naturally
 * across ~10-14 magnitude buckets, so intra-bucket lists stay short and
 * insertion sort is branch-friendly. Deterministic tie-break on `sym` is
 * applied inside the insertion sort.
 *
 * Precondition: all weights are > 0 (zero-frequency symbols are filtered
 * by the caller before this runs).
 *
 * @param[in,out] leaves  Leaf array, sorted in place (ascending weight, then
 *                        ascending @c sym on ties).
 * @param[in]     n       Number of leaves; @c n < 2 is effectively a no-op.
 */
static void pm_leaves_sort(pm_leaf_t* RESTRICT leaves, const int n) {
    /* One bucket per possible value of floor(log2(weight)) for a 32-bit
     * weight, i.e. 32 buckets. */
    enum { NUM_BUCKETS = 32 };
    int count[NUM_BUCKETS];
    int offset[NUM_BUCKETS + 1]; /* +1 sentinel = n, avoids end-of-bucket branch. */
    uint8_t bucket_of[ZXC_HUF_NUM_SYMBOLS];
    pm_leaf_t tmp[ZXC_HUF_NUM_SYMBOLS];

    ZXC_MEMSET(count, 0, sizeof(count));
    for (int i = 0; i < n; i++) {
        const unsigned b = zxc_log2_u32(leaves[i].w);
        bucket_of[i] = (uint8_t)b;
        count[b]++;
    }

    int acc = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offset[b] = acc;
        acc += count[b];
    }
    offset[NUM_BUCKETS] = n;

    int pos[NUM_BUCKETS];
    ZXC_MEMCPY(pos, offset, sizeof(pos));
    for (int i = 0; i < n; i++) {
        tmp[pos[bucket_of[i]]++] = leaves[i];
    }

    for (int b = 0; b < NUM_BUCKETS; b++) {
        if (count[b] < 2) continue;
        const int s = offset[b];
        const int e = offset[b + 1];
        for (int i = s + 1; i < e; i++) {
            const pm_leaf_t key = tmp[i];
            int j = i - 1;
            while (j >= s && (tmp[j].w > key.w || (tmp[j].w == key.w && tmp[j].sym > key.sym))) {
                tmp[j + 1] = tmp[j];
                j--;
            }
            tmp[j + 1] = key;
        }
    }

    ZXC_MEMCPY(leaves, tmp, (size_t)n * sizeof(pm_leaf_t));
}

/**
 * @brief Build length-limited canonical Huffman code lengths.
 *
 * Runs the boundary package-merge algorithm capped at `ZXC_HUF_MAX_CODE_LEN_ULTRA`.
 * Symbols with `freq[i] == 0` get `code_len[i] == 0`; every other symbol
 * receives a length in `[1, ZXC_HUF_MAX_CODE_LEN_ULTRA]`. The single-present-symbol
 * case is handled as a degenerate code of length 1.
 *
 * @param[in]  freq     Frequency table indexed by symbol (0..255).
 * @param[out] code_len Output code-length array, written in full.
 * @param[in]  scratch  Optional scratch of `ZXC_HUF_BUILD_SCRATCH_SIZE` bytes
 *                      (carved into items / counts / stack regions). If
 *                      `NULL`, the function allocates its own working memory
 *                      for the duration of the call.
 * @return `ZXC_OK` on success, `ZXC_ERROR_MEMORY` or `ZXC_ERROR_CORRUPT_DATA`
 *         on failure.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch, const int max_code_len) {
    ZXC_MEMSET(code_len, 0, ZXC_HUF_NUM_SYMBOLS);

    pm_leaf_t leaves[ZXC_HUF_NUM_SYMBOLS];
    int n = 0;
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i++) {
        if (freq[i] > 0) {
            leaves[n].w = freq[i];
            leaves[n].sym = (int16_t)i;
            n++;
        }
    }
    if (UNLIKELY(n == 0)) return ZXC_ERROR_CORRUPT_DATA;
    if (n == 1) {
        code_len[leaves[0].sym] = 1;
        return ZXC_OK;
    }

    pm_leaves_sort(leaves, n);

    /* Callers pass max_code_len >= 8, and n <= 256 <= 2^max_code_len, so the
     * length limit is always feasible (a full 8-bit code covers all 256 symbols). */
    const int max_per_level = 2 * n;

    /* Working buffers: either carve from caller-provided scratch (sized for
     * the worst-case alphabet) or fall back to per-call malloc/free. */
    pm_item_t* items;
    int* counts;
    frame_t* stack;
    pm_item_t* owned_items = NULL;
    int* owned_counts = NULL;
    frame_t* owned_stack = NULL;
    if (scratch) {
        uint8_t* p = (uint8_t*)scratch;
        items = (pm_item_t*)p;
        p +=
            (size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA * (size_t)ZXC_HUF_PM_LEVEL_BOUND * sizeof(pm_item_t);
        p = (uint8_t*)(((uintptr_t)p + 7U) & ~(uintptr_t)7U);
        counts = (int*)p;
        ZXC_MEMSET(counts, 0, (size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA * sizeof(int));
        p += (size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA * sizeof(int);
        p = (uint8_t*)(((uintptr_t)p + 7U) & ~(uintptr_t)7U);
        stack = (frame_t*)p;
    } else {
        owned_items = (pm_item_t*)ZXC_MALLOC((size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA *
                                             (size_t)max_per_level * sizeof(pm_item_t));
        owned_counts = (int*)ZXC_CALLOC((size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA, sizeof(int));
        owned_stack = (frame_t*)ZXC_MALLOC((size_t)ZXC_HUF_MAX_CODE_LEN_ULTRA *
                                           (size_t)max_per_level * sizeof(frame_t));
        if (UNLIKELY(!owned_items || !owned_counts || !owned_stack)) {
            ZXC_FREE(owned_items);
            ZXC_FREE(owned_counts);
            ZXC_FREE(owned_stack);
            return ZXC_ERROR_MEMORY;
        }
        items = owned_items;
        counts = owned_counts;
        stack = owned_stack;
    }
#define ITEM(k, i) items[(size_t)(k) * (size_t)max_per_level + (size_t)(i)]

    /* Level 0 (logical level 1): the leaves themselves, already sorted. */
    for (int i = 0; i < n; i++) {
        ITEM(0, i).weight = leaves[i].w;
        ITEM(0, i).left = -1;
        ITEM(0, i).right = -1;
        ITEM(0, i).sym = leaves[i].sym;
    }
    counts[0] = n;

    /* Levels 1..max_code_len-1: merge sorted leaves with sorted packages from the previous
     * level. */
    for (int k = 1; k < max_code_len; k++) {
        const int prev = counts[k - 1];
        const int packs = prev / 2;
        int li = 0;
        int pi = 0;
        int n_lvl = 0;
        while (li < n || pi < packs) {
            const uint32_t wl = (li < n) ? leaves[li].w : UINT32_MAX;
            const uint32_t wp =
                (pi < packs)
                    ? (uint32_t)(ITEM(k - 1, 2 * pi).weight + ITEM(k - 1, 2 * pi + 1).weight)
                    : UINT32_MAX;
            if (wl <= wp && li < n) {
                ITEM(k, n_lvl).weight = wl;
                ITEM(k, n_lvl).left = -1;
                ITEM(k, n_lvl).right = -1;
                ITEM(k, n_lvl).sym = leaves[li].sym;
                li++;
            } else {
                ITEM(k, n_lvl).weight = wp;
                ITEM(k, n_lvl).left = (int16_t)(2 * pi);
                ITEM(k, n_lvl).right = (int16_t)(2 * pi + 1);
                ITEM(k, n_lvl).sym = -1;
                pi++;
            }
            n_lvl++;
        }
        counts[k] = n_lvl;
    }

    /* Step 3: take first 2n-2 items at level max_code_len-1; trace back, counting leaf
     * appearances. */
    int n_take = 2 * n - 2;
    if (n_take > counts[max_code_len - 1]) n_take = counts[max_code_len - 1];

    /* Worst case stack depth: (max_code_len * n_take) frames; bounded by
     * max_code_len * 2n <= ZXC_HUF_MAX_CODE_LEN_ULTRA * 2n. `stack` was set up earlier from
     * scratch (or the local malloc fallback), sized for the ceiling. */
    int sp = 0;
    for (int i = 0; i < n_take; i++) {
        stack[sp].lvl = (int8_t)(max_code_len - 1);
        stack[sp].idx = (int16_t)i;
        sp++;
    }
    while (sp > 0) {
        frame_t f = stack[--sp];
        const pm_item_t* it = &ITEM(f.lvl, f.idx);
        if (it->sym >= 0) {
            code_len[it->sym]++;
        } else {
            stack[sp].lvl = (int8_t)(f.lvl - 1);
            stack[sp].idx = it->left;
            sp++;
            stack[sp].lvl = (int8_t)(f.lvl - 1);
            stack[sp].idx = it->right;
            sp++;
        }
    }

    if (owned_items) {
        ZXC_FREE(owned_items);
        ZXC_FREE(owned_counts);
        ZXC_FREE(owned_stack);
    }
#undef ITEM
    return ZXC_OK;
}

/* ===========================================================================
 * Encoder-side joint flat/length nudge (idea: pivco-huffman issue #20)
 * ===========================================================================
 *
 * Package-merge minimizes bits alone; the PivCo decoder's cost also depends on
 * the SHAPE of the length histogram (pass count = max_depth + 1, and maximal
 * complete subtrees collapse into single unpacks -- see the decode section).
 * The functions below reshape the class counts toward power-of-two boundaries
 * and shallower caps under an adoption guard, trading a bounded ratio loss for
 * fewer modeled decode level-touches.
 *
 * Everything is exact integer arithmetic on the canonical code space. With
 * leaves assigned in (length, symbol) ascending order -- the canonical rule of
 * zxc_pivco_tree_build -- the depth-l leaves occupy one contiguous ALIGNED run
 * [S_{l-1}, S_l) of the code space at 2^-11 slot granularity, where
 * S_l = sum_{j<=l} bl_count[j] << (11-j) (each prior term is divisible by
 * 2^(12-l), so runs start on even slot boundaries). The maximal dyadic (buddy)
 * blocks of each run are EXACTLY the decoder's maximal flat subtrees, so any
 * candidate histogram is priced without building a tree.
 *
 * Compiled once in the primary variant: this is ISA-independent cold encoder
 * policy, and a single copy guarantees cross-ISA identical archives. For an
 * A/B build without the nudge, override the guard from CFLAGS
 * (-DZXC_HUF_NUDGE_MERGE_Q8=0 rejects every candidate).
 */
#if defined(ZXC_VARIANT_PRIMARY)

/** @brief Modeled cost of one code-length candidate (see zxc_huf_nudge_eval). */
typedef struct {
    uint64_t bits;    /**< Payload bits: sum(len * freq); per-node padding excluded. */
    uint64_t touches; /**< Modeled decode cost: occurrence-weighted level-touches. */
} zxc_huf_nudge_cost_t;

/**
 * @brief Per-length class counts of a code-length vector.
 *
 * @param[in]  code_len Per-symbol code lengths (0 = absent symbol).
 * @param[out] blc      Class counts indexed by length, entries
 *                      `0..ZXC_HUF_MAX_CODE_LEN_ULTRA` (entry 0 stays 0).
 * @return Number of present symbols (nonzero lengths).
 */
static int zxc_huf_nudge_classes(const uint8_t* RESTRICT code_len, uint32_t* RESTRICT blc) {
    ZXC_MEMSET(blc, 0, (ZXC_HUF_MAX_CODE_LEN_ULTRA + 1) * sizeof(uint32_t));
    int n = 0;
    for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
        const int l = code_len[s];
        if (!l) continue;
        blc[l]++;
        n++;
    }
    return n;
}

/**
 * @brief Frequency prefix sums over the leaves in canonical (length, symbol)
 *        order: `pf[i]` = mass of the first `i` leaves of the code space.
 *
 * @param[in]  code_len Per-symbol code lengths.
 * @param[in]  freq     Frequency table indexed by symbol.
 * @param[in]  blc      Class counts of @p code_len (from ::zxc_huf_nudge_classes).
 * @param[out] pf       Prefix sums, `n_present + 1` entries with `pf[0] = 0`.
 */
static void zxc_huf_nudge_pf_canonical(const uint8_t* RESTRICT code_len,
                                       const uint32_t* RESTRICT freq, const uint32_t* RESTRICT blc,
                                       uint64_t* RESTRICT pf) {
    uint32_t pos[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
    uint32_t acc = 0;
    for (int l = 1; l <= ZXC_HUF_MAX_CODE_LEN_ULTRA; l++) {
        pos[l] = acc;
        acc += blc[l];
    }
    uint32_t val[ZXC_HUF_NUM_SYMBOLS];
    for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
        const int l = code_len[s];
        if (l) val[pos[l]++] = freq[s];
    }
    pf[0] = 0;
    for (uint32_t i = 0; i < acc; i++) pf[i + 1] = pf[i] + val[i];
}

/**
 * @brief Modeled (bits, level-touches) of one class-count histogram.
 *
 * The buddy decomposition of each same-depth leaf run mirrors
 * zxc_pivco_decode_core row by row: a lone leaf at depth l is memset once and
 * merged l times (l+1 touches); a leaf pair is emitted by its parent's
 * XOR-blend (l touches); a flat root at depth t = l-D is unpacked once and
 * merged t times (t+1 touches, plus a penalty past the SIMD unpackers); each
 * pass adds a fixed overhead term.
 *
 * @param[in]  blc Class counts of the candidate (Kraft-exact by construction).
 * @param[in]  pf  Frequency prefix sums of the leaves in code order
 *                 (`pf[0] = 0`): canonical (length, symbol)-order sums give
 *                 exact costs; frequency-rank sums give the cheap scores used
 *                 inside the walk (bits are identical either way, only the
 *                 within-run mass split differs).
 * @param[out] out Modeled bits and level-touches.
 */
static void zxc_huf_nudge_eval(const uint32_t* RESTRICT blc, const uint64_t* RESTRICT pf,
                               zxc_huf_nudge_cost_t* RESTRICT out) {
    enum { LU = ZXC_HUF_MAX_CODE_LEN_ULTRA };
    uint64_t bits = 0;
    uint64_t touches = 0;
    int max_len = 0;
    uint32_t S = 0;    /* code-space cursor, in 2^-LU slots */
    uint32_t base = 0; /* leaf index where the current run starts */
    for (int l = 1; l <= LU; l++) {
        if (!blc[l]) continue;
        max_len = l;
        bits += (uint64_t)l * (pf[base + blc[l]] - pf[base]);
        const uint32_t w = 1U << (LU - l);
        const uint32_t end = S + blc[l] * w;
        uint32_t x = S;
        while (x < end) {
            const uint32_t wx = x ? (x & (0U - x)) : (1U << LU);
            const uint32_t wr = 1U << zxc_log2_u32(end - x);
            const uint32_t W = wx < wr ? wx : wr;
            const int D = (int)zxc_log2_u32(W) - (LU - l);
            const uint32_t i0 = base + ((x - S) >> (LU - l));
            const uint64_t mass = pf[i0 + (W >> (LU - l))] - pf[i0];
            int t;
            if (D == 0) {
                t = l + 1; /* lone leaf: one memset + merges above */
            } else if (D == 1) {
                t = l; /* leaf pair: parent XOR-blend at depth l-1 */
            } else {
                t = (l - D) + 1; /* flat root: one unpack + merges above */
                if (D > ZXC_HUF_NUDGE_FLAT_SIMD_MAX) t += ZXC_HUF_NUDGE_DEEP_FLAT_PENALTY;
            }
            touches += mass * (uint64_t)t;
            x += W;
        }
        S = end;
        base += blc[l];
    }
    touches += (uint64_t)ZXC_HUF_NUDGE_LEVEL_COST * (uint64_t)(max_len + 1);
    out->bits = bits;
    out->touches = touches;
}

/**
 * @brief Candidate cost `J = 256*bits + lambda*touches` (Q8 bit units).
 *
 * @param[in] c Modeled cost from ::zxc_huf_nudge_eval.
 * @return Scalar cost; lower is better.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_huf_nudge_j(const zxc_huf_nudge_cost_t* c) {
    return 256U * c->bits + (uint64_t)ZXC_HUF_NUDGE_LAMBDA_Q8 * c->touches;
}

/**
 * @brief Clamp a desired class count into the feasible set at one walk level.
 *
 * A count c is feasible iff a Kraft-exact completion within @p cap still
 * exists: either c == n_rem == s (a finishing level fills every slot), or,
 * continuing with i = s - c internal nodes, i >= 1 and
 * 2*i <= n_rem - c <= i << (cap - l). Under the walk invariant
 * (s <= n_rem <= s << (cap - l)) the continuing window
 * [max(0, 2s - n_rem), min(s-1, n_rem-1, (s*M - n_rem)/(M-1))] is nonempty
 * whenever n_rem > s, and n_rem == s forces the finishing count (continuing
 * would need both c >= s and c <= s-1).
 *
 * @param[in] want  Desired class count at this level.
 * @param[in] s     Open slots at level @p l.
 * @param[in] n_rem Symbols left to place (including this level's).
 * @param[in] l     Current level.
 * @param[in] cap   Code-length cap of the walk.
 * @return The nearest feasible count to @p want.
 */
static uint32_t zxc_huf_nudge_clamp(const uint32_t want, const uint32_t s, const uint32_t n_rem,
                                    const int l, const int cap) {
    if (n_rem <= s || l >= cap) return n_rem; /* forced finish */
    const uint64_t m = (uint64_t)1 << (cap - l);
    const uint32_t lo = (2 * s > n_rem) ? 2 * s - n_rem : 0;
    uint32_t hi = s - 1; /* s < n_rem here, so min(s-1, n_rem-1) == s-1 */
    const uint64_t cap_hi = ((uint64_t)s * m - n_rem) / (m - 1);
    if (cap_hi < hi) hi = (uint32_t)cap_hi;
    const uint32_t c = want < lo ? lo : want;
    return c > hi ? hi : c;
}

/**
 * @brief Baseline-following completion: fill levels `l+1..cap` with the
 *        baseline class counts clamped into feasibility.
 *
 * @param[in,out] blc   Candidate histogram; levels above @p l are written.
 * @param[in]     l     Last level already fixed by the caller.
 * @param[in]     s     Open slots at level `l+1`.
 * @param[in]     n_rem Symbols still to place.
 * @param[in]     blc0  Baseline class counts to follow.
 * @param[in]     cap   Code-length cap of the walk.
 */
static void zxc_huf_nudge_complete(uint32_t* RESTRICT blc, const int l, uint32_t s, uint32_t n_rem,
                                   const uint32_t* RESTRICT blc0, const int cap) {
    for (int j = l + 1; j <= cap && n_rem; j++) {
        const uint32_t c = zxc_huf_nudge_clamp(blc0[j], s, n_rem, j, cap);
        blc[j] = c;
        n_rem -= c;
        s = 2 * (s - c);
    }
}

/**
 * @brief Greedy shallow-to-deep walk over the slot ledger.
 *
 * At each level, up to six feasible class counts are tried: follow the
 * baseline; make the internal-node count the nearest power of two (down/up)
 * or a two-bit value (power-of-two internal-node counts keep the run
 * boundaries aligned, which is what lets deeper runs decompose into large
 * flat blocks); or finish the whole remainder as uniform-depth subtrees.
 * Each count is expanded into a full candidate histogram and scored with the
 * rank-order evaluator; the cheapest count is fixed and the walk descends.
 * Fixed candidate order + strict improvement keeps the result deterministic.
 * A chosen uniform-depth finish needs no lock: at the next level it reappears
 * as the (c = 0, D-1) finish candidate, so the greedy can hold or drop it.
 *
 * @param[in]  blc0    Baseline class counts (package-merge output).
 * @param[in]  pf_rank Frequency-rank prefix sums (rank 0 = most frequent).
 * @param[in]  n       Number of present symbols (>= 4).
 * @param[in]  cap     Code-length cap.
 * @param[out] out_blc Chosen class counts (Kraft-exact by construction).
 */
static void zxc_huf_nudge_walk(const uint32_t* RESTRICT blc0, const uint64_t* RESTRICT pf_rank,
                               const int n, const int cap, uint32_t* RESTRICT out_blc) {
    enum { LU = ZXC_HUF_MAX_CODE_LEN_ULTRA };
    ZXC_MEMSET(out_blc, 0, (LU + 1) * sizeof(uint32_t));
    uint32_t s = 2; /* the root's two slots at level 1 (n >= 4 => root is internal) */
    uint32_t n_rem = (uint32_t)n;
    for (int l = 1; l <= cap && n_rem; l++) {
        struct {
            uint32_t c;
            int flat_d; /* > 0: complete as one uniform run at depth l + flat_d */
        } cand[6];
        int n_cand = 0;
        const uint32_t c_base = zxc_huf_nudge_clamp(blc0[l], s, n_rem, l, cap);
        cand[n_cand].c = c_base;
        cand[n_cand++].flat_d = 0;
        if (c_base < n_rem) {
            /* Shape the internal-node count i = s - c toward few set bits. */
            const uint32_t i_base = s - c_base;
            const uint32_t i_dn = 1U << zxc_log2_u32(i_base);
            const uint32_t rest = i_base - i_dn;
            const uint32_t shapes[3] = {i_dn, i_dn << 1,
                                        i_dn | (rest ? 1U << zxc_log2_u32(rest) : 0U)};
            for (int k = 0; k < 3; k++) {
                const uint32_t want = s > shapes[k] ? s - shapes[k] : 0;
                const uint32_t c = zxc_huf_nudge_clamp(want, s, n_rem, l, cap);
                int dup = 0;
                for (int p = 0; p < n_cand; p++) dup |= (cand[p].c == c);
                if (!dup) {
                    cand[n_cand].c = c;
                    cand[n_cand++].flat_d = 0;
                }
            }
            /* Uniform finish: c leaves here, the rest as i complete depth-D
             * subtrees (c*(2^D - 1) == s*2^D - n_rem must divide exactly). */
            for (int d = 1; d <= cap - l && n_cand < 6; d++) {
                const uint64_t den = ((uint64_t)1 << d) - 1;
                const int64_t num = (int64_t)((uint64_t)s << d) - (int64_t)n_rem;
                if (num < 0 || (uint64_t)num % den) continue;
                const uint64_t c64 = (uint64_t)num / den;
                if (c64 >= s || c64 >= n_rem) continue; /* need i >= 1 and a remainder */
                cand[n_cand].c = (uint32_t)c64;
                cand[n_cand++].flat_d = d;
                break; /* the shallowest exact finish is the aggressive one */
            }
        }
        uint64_t best_j = UINT64_MAX;
        uint32_t best_c = c_base;
        for (int k = 0; k < n_cand; k++) {
            uint32_t tmp[LU + 1];
            ZXC_MEMCPY(tmp, out_blc, sizeof(tmp));
            tmp[l] = cand[k].c;
            const uint32_t rem = n_rem - cand[k].c;
            if (rem) {
                if (cand[k].flat_d)
                    tmp[l + cand[k].flat_d] = rem;
                else
                    zxc_huf_nudge_complete(tmp, l, 2 * (s - cand[k].c), rem, blc0, cap);
            }
            zxc_huf_nudge_cost_t cc;
            zxc_huf_nudge_eval(tmp, pf_rank, &cc);
            const uint64_t j = zxc_huf_nudge_j(&cc);
            if (j < best_j) {
                best_j = j;
                best_c = cand[k].c;
            }
        }
        out_blc[l] = best_c;
        n_rem -= best_c;
        s = 2 * (s - best_c);
    }
}

/**
 * @brief Exact rank-weighted J of one leaf run placed at coarse level @p lc.
 *
 * The DP prices in REAL units through a coarse lens: one coarse slot stands
 * for a complete depth-@p g_log2 subtree of 2^g_log2 frequency-adjacent
 * leaves, so a coarse buddy block of 2^d slots is a real block of depth
 * d + g_log2 whose leaves have real length lc + g_log2. Because canonical
 * codes fill the code space left to right, a ledger state with @p s open
 * slots pins the run start at 2^lu - s * 2^(lu - lc): the cost of taking
 * @p c slots here depends only on (lc, s, c, rank interval) -- which is what
 * makes the level costs additive and the DP exact for this model.
 *
 * @param[in] lu     Coarse code-space scale (`ZXC_HUF_MAX_CODE_LEN_ULTRA - g_log2`).
 * @param[in] lc     Coarse level the run is placed at.
 * @param[in] g_log2 Granularity: one coarse slot = 2^g_log2 real leaves.
 * @param[in] s      Open coarse slots at level @p lc.
 * @param[in] c      Coarse slots taken as leaves at this level.
 * @param[in] pfg    Group-mass prefix sums in frequency-rank order.
 * @param[in] k      Groups already placed (rank of the run's first group).
 * @return J contribution of the run (Q8 bits + lambda * touches).
 */
static uint64_t zxc_huf_nudge_dp_run_j(const int lu, const int lc, const int g_log2,
                                       const uint32_t s, const uint32_t c,
                                       const uint64_t* RESTRICT pfg, const uint32_t k) {
    if (!c) return 0;
    const int lr = lc + g_log2; /* real code length of these leaves */
    const uint32_t w = 1U << (lu - lc);
    const uint32_t S = (1U << lu) - s * w;
    const uint32_t end = S + c * w;
    const uint64_t bits = (uint64_t)lr * (pfg[k + c] - pfg[k]);
    uint64_t touches = 0;
    uint32_t x = S;
    while (x < end) {
        const uint32_t wx = x ? (x & (0U - x)) : (1U << lu);
        const uint32_t wr = 1U << zxc_log2_u32(end - x);
        const uint32_t W = wx < wr ? wx : wr;
        const int d = (int)zxc_log2_u32(W >> (lu - lc)) + g_log2;
        const uint32_t i0 = k + ((x - S) >> (lu - lc));
        const uint64_t mass = pfg[i0 + (W >> (lu - lc))] - pfg[i0];
        int t;
        if (d == 0) {
            t = lr + 1;
        } else if (d == 1) {
            t = lr;
        } else {
            t = (lr - d) + 1;
            if (d > ZXC_HUF_NUDGE_FLAT_SIMD_MAX) t += ZXC_HUF_NUDGE_DEEP_FLAT_PENALTY;
        }
        touches += mass * (uint64_t)t;
        x += W;
    }
    return 256U * bits + (uint64_t)ZXC_HUF_NUDGE_LAMBDA_Q8 * touches;
}

/**
 * @brief Slot-ledger dynamic program: optimal class counts for the model.
 *
 * States are (symbols placed, open slots) per level -- the (k, s) plane of
 * pivco-huffman #20 -- relaxed level by level with the exact per-run cost
 * above; a state with s == remaining symbols is a forced finishing level
 * (continuing needs both c >= s and c <= s - 1). The finishing transition
 * adds the pass-loop overhead for the resulting max depth. Backtracking the
 * per-level arrival choices reconstructs the coarse class counts.
 *
 * Iteration order and strict-improvement relaxation are fixed, so the result
 * is deterministic.
 *
 * @param[in]  pfg      Group-mass prefix sums in frequency-rank order.
 * @param[in]  m        Number of coarse symbols (groups), >= 2.
 * @param[in]  cap_c    Coarse code-length cap (`real cap - g_log2`).
 * @param[in]  lu       Coarse code-space scale (see ::zxc_huf_nudge_dp_run_j).
 * @param[in]  g_log2   Granularity exponent.
 * @param[out] out_cblc Optimal coarse class counts (indexed by coarse level).
 * @return 1 on success; 0 on allocation failure or no feasible finish
 *         (callers then simply keep their other candidates).
 */
static int zxc_huf_nudge_dp_solve(const uint64_t* RESTRICT pfg, const int m, const int cap_c,
                                  const int lu, const int g_log2, uint32_t* RESTRICT out_cblc) {
    if (UNLIKELY(m < 2 || cap_c < 1)) return 0;

    const size_t plane = (size_t)(m + 1) * (size_t)(m + 1);
    const size_t arrive_cnt = (size_t)(cap_c + 1) * plane;
    const size_t arrive_slots =
        (arrive_cnt * sizeof(uint16_t) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    const size_t pool_slots = 2 * plane + arrive_slots;

    uint64_t* pool = (uint64_t*)ZXC_MALLOC(pool_slots * sizeof(uint64_t));
    if (UNLIKELY(!pool)) return 0;

    uint64_t* jcur = pool;
    uint64_t* jnxt = pool + plane;
    uint16_t* arrive = (uint16_t*)(pool + 2 * plane);
    int ok = 0;

#define ZXC_NUDGE_DP_IDX(k, s) ((size_t)(k) * (size_t)(m + 1) + (size_t)(s))
    for (size_t i = 0; i < plane; i++) jcur[i] = UINT64_MAX;
    jcur[ZXC_NUDGE_DP_IDX(0, 2)] = 0;
    uint64_t best_j = UINT64_MAX;
    int best_l = 0;
    int best_k = 0;
    int best_s = 0;
    for (int lc = 1; lc <= cap_c; lc++) {
        for (size_t i = 0; i < plane; i++) jnxt[i] = UINT64_MAX;
        for (int k = 0; k < m; k++) {
            const uint32_t n_rem = (uint32_t)(m - k);
            for (uint32_t s = 1; s <= n_rem; s++) {
                const size_t from = ZXC_NUDGE_DP_IDX(k, s);
                if (UNLIKELY(from >= plane)) continue;
                const uint64_t j0 = jcur[from];
                if (j0 == UINT64_MAX) continue;
                if (s == n_rem) {
                    /* Forced finish: fill every slot, close the tree here. */
                    const uint64_t j =
                        j0 + zxc_huf_nudge_dp_run_j(lu, lc, g_log2, s, s, pfg, (uint32_t)k) +
                        (uint64_t)ZXC_HUF_NUDGE_LAMBDA_Q8 * (uint64_t)ZXC_HUF_NUDGE_LEVEL_COST *
                            (uint64_t)(lc + g_log2 + 1);
                    if (j < best_j) {
                        best_j = j;
                        best_l = lc;
                        best_k = k;
                        best_s = (int)s;
                    }
                    continue;
                }
                if (lc == cap_c) continue; /* must finish at the cap */
                const uint64_t mm = (uint64_t)1 << (cap_c - lc);
                const uint32_t lo = (2 * s > n_rem) ? 2 * s - n_rem : 0;
                uint32_t hi = s - 1;
                const uint64_t cap_hi = ((uint64_t)s * mm - n_rem) / (mm - 1);
                if (cap_hi < hi) hi = (uint32_t)cap_hi;
                for (uint32_t c = lo; c <= hi; c++) {
                    const uint64_t j =
                        j0 + zxc_huf_nudge_dp_run_j(lu, lc, g_log2, s, c, pfg, (uint32_t)k);
                    const size_t to = ZXC_NUDGE_DP_IDX(k + (int)c, 2 * (s - c));
                    const size_t arr = (size_t)(lc + 1) * plane + to;

                    if (UNLIKELY(to >= plane || arr >= arrive_cnt)) continue;
                    if (j < jnxt[to]) {
                        jnxt[to] = j;
                        arrive[arr] = (uint16_t)c;
                    }
                }
            }
        }
        uint64_t* tmp = jcur;
        jcur = jnxt;
        jnxt = tmp;
    }
    if (best_j == UINT64_MAX) goto done;
    ZXC_MEMSET(out_cblc, 0, (ZXC_HUF_MAX_CODE_LEN_ULTRA + 1) * sizeof(uint32_t));
    out_cblc[best_l] = (uint32_t)best_s;
    {
        int k = best_k;
        int s = best_s;
        for (int lc = best_l; lc > 1; lc--) {
            const uint32_t c = arrive[(size_t)lc * plane + ZXC_NUDGE_DP_IDX(k, s)];
            out_cblc[lc - 1] = c;
            s = s / 2 + (int)c;
            k -= (int)c;
        }
        if (UNLIKELY(k != 0 || s != 2)) goto done; /* backtrack corruption: drop candidate */
    }
    ok = 1;
done:
#undef ZXC_NUDGE_DP_IDX
    ZXC_FREE(pool);
    return ok;
}

/**
 * @brief Assign a class-count multiset to symbols: frequency rank r takes the
 *        r-th shortest slot (rearrangement-optimal, deterministic).
 *
 * @param[in]  blc       Class counts to realize (sum = number of ranks).
 * @param[in]  sym_order Symbols by descending frequency (rank 0 first).
 * @param[out] cand_len  Resulting per-symbol code lengths, written in full.
 */
static void zxc_huf_nudge_assign(const uint32_t* RESTRICT blc, const int16_t* RESTRICT sym_order,
                                 uint8_t* RESTRICT cand_len) {
    ZXC_MEMSET(cand_len, 0, ZXC_HUF_NUM_SYMBOLS);
    int r = 0;
    for (int l = 1; l <= ZXC_HUF_MAX_CODE_LEN_ULTRA; l++)
        for (uint32_t k = 0; k < blc[l]; k++) cand_len[sym_order[r++]] = (uint8_t)l;
}

/**
 * @brief Modeled (bits, level-touches) decode cost of one code-length vector.
 *
 * Introspection hook over the exact evaluator (canonical-order weighting);
 * full contract in zxc_internal.h. The unit tests cross-check it against the
 * real tree built by the decoder.
 */
void zxc_huf_nudge_cost(const uint8_t* RESTRICT code_len, const uint32_t* RESTRICT freq,
                        uint64_t* RESTRICT bits, uint64_t* RESTRICT touches) {
    uint32_t blc[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
    (void)zxc_huf_nudge_classes(code_len, blc);
    uint64_t pf[ZXC_HUF_NUM_SYMBOLS + 1];
    zxc_huf_nudge_pf_canonical(code_len, freq, blc, pf);
    zxc_huf_nudge_cost_t c;
    zxc_huf_nudge_eval(blc, pf, &c);
    *bits = c.bits;
    *touches = c.touches;
}

/**
 * @brief Optionally reshape freshly built code lengths for faster PivCo decode.
 *
 * Candidate generation (walk, reduced-cap rebuilds, slot-ledger DP), exact
 * pricing and the adoption guard; full contract in zxc_internal.h. On
 * rejection @p code_len is left byte-for-byte untouched.
 */
int zxc_huf_nudge_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch, const int max_code_len) {
    enum { LU = ZXC_HUF_MAX_CODE_LEN_ULTRA };
    uint32_t blc0[LU + 1];
    const int n = zxc_huf_nudge_classes(code_len, blc0);
    /* Degenerate alphabets have no shape freedom (and n == 1 is format-pinned
     * to code length exactly 1): leave them untouched. */
    if (n < 4) return 0;

    /* Baseline cost, exact (canonical-order frequency weighting). */
    uint64_t pf[ZXC_HUF_NUM_SYMBOLS + 1];
    zxc_huf_nudge_pf_canonical(code_len, freq, blc0, pf);
    zxc_huf_nudge_cost_t c0;
    zxc_huf_nudge_eval(blc0, pf, &c0);

    /* Frequency-rank order shared by every candidate (rank 0 = most frequent;
     * pm_leaves_sort is ascending, ties on symbol, so read it backwards). */
    pm_leaf_t leaves[ZXC_HUF_NUM_SYMBOLS];
    int k = 0;
    for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
        if (!freq[s]) continue;
        leaves[k].w = freq[s];
        leaves[k].sym = (int16_t)s;
        k++;
    }
    pm_leaves_sort(leaves, n);
    int16_t sym_order[ZXC_HUF_NUM_SYMBOLS];
    uint64_t pf_rank[ZXC_HUF_NUM_SYMBOLS + 1];
    pf_rank[0] = 0;
    for (int r = 0; r < n; r++) {
        sym_order[r] = leaves[n - 1 - r].sym;
        pf_rank[r + 1] = pf_rank[r] + leaves[n - 1 - r].w;
    }

    /* Candidates: the greedy ledger walk, package-merge rebuilt at reduced caps
     * (the pass loop runs max_depth + 1 times, so depth cuts attack the decoder's
     * biggest fixed cost), and the slot-ledger DP below. All Kraft-exact. */
    uint8_t cand[4][ZXC_HUF_NUM_SYMBOLS];
    int n_cand = 0;
    {
        uint32_t blc_w[LU + 1];
        zxc_huf_nudge_walk(blc0, pf_rank, n, max_code_len, blc_w);
        zxc_huf_nudge_assign(blc_w, sym_order, cand[n_cand]);
        n_cand++;
    }
    int max_len0 = 0;
    for (int l = LU; l >= 1; l--) {
        if (blc0[l]) {
            max_len0 = l;
            break;
        }
    }
    if (max_len0 >= 2) {
        for (int cut = 1; cut <= 2; cut++) {
            const int cap2 = max_len0 - cut;
            if (cap2 < 2 || ((uint32_t)1 << cap2) < (uint32_t)n) break;
            if (zxc_huf_build_code_lengths(freq, cand[n_cand], scratch, cap2) != ZXC_OK) break;
            n_cand++;
        }
    }

    /* Slot-ledger DP: optimal class counts under the rank-weighted model, over
     * groups of 2^G frequency-adjacent symbols (G = 0 is exact; coarser tiers
     * keep the plane cache-resident, bounding the call to a few hundred us).
     * When G does not divide n the lightest group is padded with zero-freq
     * ghost leaves on unused byte values - wire-legal, empty runs, and the
     * evaluator prices the burnt code space like anything else. */
    {
        const int g_log2 = n <= 64 ? 0 : (n <= 128 ? 1 : 2);
        const int g = 1 << g_log2;
        const int m = (n + g - 1) / g;
        const int cap_c = max_code_len - g_log2;
        if (m >= 2 && cap_c >= 1 && m <= (1 << cap_c)) {
            uint64_t pfg[ZXC_HUF_NUM_SYMBOLS + 1];
            for (int j2 = 0; j2 <= m; j2++) {
                int r = j2 * g;
                if (r > n) r = n;
                pfg[j2] = pf_rank[r];
            }
            uint32_t cblc[LU + 1];
            if (zxc_huf_nudge_dp_solve(pfg, m, cap_c, LU - g_log2, g_log2, cblc)) {
                uint8_t* const cl = cand[n_cand];
                ZXC_MEMSET(cl, 0, ZXC_HUF_NUM_SYMBOLS);
                int r = 0;
                int ghosts = 0;
                uint8_t ghost_len = 0;
                for (int lc = 1; lc <= cap_c; lc++) {
                    for (uint32_t q = 0; q < cblc[lc]; q++) {
                        for (int e = 0; e < g; e++, r++) {
                            if (r < n) {
                                cl[sym_order[r]] = (uint8_t)(lc + g_log2);
                            } else {
                                ghost_len = (uint8_t)(lc + g_log2);
                                ghosts++;
                            }
                        }
                    }
                }
                for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS && ghosts; s++) {
                    if (freq[s] == 0 && cl[s] == 0) {
                        cl[s] = ghost_len;
                        ghosts--;
                    }
                }
                n_cand++;
            }
        }
    }

    /* Adopt the cheapest candidate clearing both guard rails and strictly
     * beating the baseline J; otherwise leave code_len byte-for-byte alone, so a
     * rejection never depends on package-merge tie assignment. */
    const uint64_t j0 = zxc_huf_nudge_j(&c0);
    uint64_t best_j = j0;
    int best = -1;
    for (int ci = 0; ci < n_cand; ci++) {
        /* Every live symbol must keep a code; ghost-padded candidates carry
         * MORE nonzero lengths than n, which is fine. */
        int valid = 1;
        for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
            if (freq[s] != 0 && cand[ci][s] == 0) {
                valid = 0;
                break;
            }
        }
        if (!valid) continue;
        uint32_t blc[LU + 1];
        (void)zxc_huf_nudge_classes(cand[ci], blc);
        zxc_huf_nudge_pf_canonical(cand[ci], freq, blc, pf);
        zxc_huf_nudge_cost_t c1;
        zxc_huf_nudge_eval(blc, pf, &c1);
        if (c1.bits * 1000 > c0.bits * ZXC_HUF_NUDGE_BITS_PERMIL) continue;
        if (c1.touches * 256 > c0.touches * ZXC_HUF_NUDGE_MERGE_Q8) continue;
        const uint64_t j = zxc_huf_nudge_j(&c1);
        if (j < best_j) {
            best_j = j;
            best = ci;
        }
    }
    if (best < 0) return 0;
    ZXC_MEMCPY(code_len, cand[best], ZXC_HUF_NUM_SYMBOLS);
    return 1;
}
#endif /* ZXC_VARIANT_PRIMARY */

/* ===========================================================================
 * 128-byte length header: 256 x 4-bit lengths, low nibble first.
 * =========================================================================*/

/**
 * @brief Pack per-symbol code lengths into the 128-byte (4-bit nibble) header.
 *
 * The packing is little-endian within each byte: low nibble holds
 * `code_len[2*i]`, high nibble holds `code_len[2*i + 1]`. The function
 * silently truncates any length > 15; callers must enforce the cap of
 * `ZXC_HUF_MAX_CODE_LEN_ULTRA` (<= 15) before calling.
 *
 * @param[in]  code_len Per-symbol code lengths (length `ZXC_HUF_NUM_SYMBOLS`).
 * @param[out] out      Output header buffer of `ZXC_HUF_TABLE_SIZE` bytes.
 */
void zxc_huf_pack_lengths(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out) {
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i += 2) {
        const uint8_t lo = code_len[i] & 0x0F;
        const uint8_t hi = code_len[i + 1] & 0x0F;
        out[i >> 1] = (uint8_t)(lo | (hi << 4));
    }
}

/**
 * @brief Unpack and structurally validate a 128-byte packed lengths header.
 *
 * Inverts ::zxc_huf_pack_lengths and validates the two structural invariants:
 * no length exceeds `ZXC_HUF_MAX_CODE_LEN_ULTRA`, and at least one symbol is
 * present. (Kraft consistency is checked later by the tree build.)
 *
 * @param[in]  in       128-byte packed lengths header.
 * @param[out] code_len Output code-length array of length `ZXC_HUF_NUM_SYMBOLS`.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` if a length is too
 *         large or the table is empty.
 */
int zxc_huf_unpack_lengths(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len) {
    int max_len = 0;
    int n_present = 0;
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i += 2) {
        const uint8_t b = in[i >> 1];
        const uint8_t lo = b & 0x0F;
        const uint8_t hi = (uint8_t)(b >> 4);
        code_len[i] = lo;
        code_len[i + 1] = hi;
        if (lo > max_len) max_len = lo;
        if (hi > max_len) max_len = hi;
        if (lo) n_present++;
        if (hi) n_present++;
    }
    if (UNLIKELY(max_len > ZXC_HUF_MAX_CODE_LEN_ULTRA || n_present == 0))
        return ZXC_ERROR_CORRUPT_DATA;
    return ZXC_OK;
}

/* ===========================================================================
 * PivCo-Huffman section codec (enc 2/3)
 * ===========================================================================
 *
 * Level-ordered layout and the merge-based decode are from PivCo-Huffman
 * by Marcin Zukowski (https://github.com/MarcinZukowski/pivco-huffman).
 * Implemented independently here.
 *
 * Layout: [128-byte packed code lengths (literal sections only)] then, for
 * every EMITTING node of the canonical code tree in BFS order, that node's
 * run: one branch bit per symbol routed through the node (0 = left child,
 * 1 = right), in sequence order -- or, for flat subtree roots, D packed bits
 * per symbol (bit j = branch at relative depth j). Runs are LSB-first within
 * bytes and byte-padded; descendants of a flat root emit nothing.
 *
 * The decoder recovers each node's symbol count for free: the root handles
 * n symbols, and popcounting a node's bits yields its right child's count.
 * Reconstruction runs bottom-up, one level at a time: leaves are runs of a
 * single symbol; an internal node MERGES its two children's sequences under
 * the control of its bits. Merges are branch-free shuffles (16 outputs per
 * step on AArch64 via a two-register TBL, 8 on ARMv7 via VTBL4, 64 on
 * AVX-512-VBMI2 via vpexpandb), which is what makes this layout
 * decode faster than the serial bit-chain of the classic 4-stream layout on
 * any target with a 16-byte shuffle.
 *
 * Level buffers ping-pong between `scratch` (odd depths) and `dst` (even
 * depths, final output at depth 0), so one n-sized scratch suffices. Both
 * buffers need read slack past `n` (speculative 16-byte kernel loads):
 * ZXC_PAD_SIZE on dst (already true for lit/token buffers), and
 * ZXC_PIVCO_SCRATCH_PAD on scratch.
 */

static ZXC_ALWAYS_INLINE int zxc_pivco_popcnt32(const uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(v);
#else
    /* Portable SWAR popcount for MSVC */
    uint32_t x = v - ((v >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int)((x * 0x01010101U) >> 24);
#endif
}

static ZXC_ALWAYS_INLINE int zxc_pivco_popcnt64(const uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    return zxc_pivco_popcnt32((uint32_t)v) + zxc_pivco_popcnt32((uint32_t)(v >> 32));
#endif
}

/**
 * @brief Build the canonical code tree (and codes) from per-symbol lengths.
 *
 * Canonical MSB-first assignment: symbols sorted by (length, symbol) receive
 * sequential codes. Insertion collisions or code-space overflow (malformed
 * lengths on the decode path) fail with a nonzero return.
 *
 * @param[in]  code_len Per-symbol code lengths (0 = absent).
 * @param[out] t        Tree + BFS ordering, fully initialised on success.
 * @param[out] codes    Optional per-symbol canonical codes (encoder only).
 * @return 0 on success, -1 on malformed lengths.
 */
static int zxc_pivco_tree_build(const uint8_t* RESTRICT code_len, zxc_pivco_tree_t* RESTRICT t,
                                uint32_t* RESTRICT codes) {
    uint32_t bl_count[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1] = {0};
    int n_present = 0;
    for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
        const int l = code_len[s];
        if (l == 0) continue;
        if (UNLIKELY(l > ZXC_HUF_MAX_CODE_LEN_ULTRA)) return -1;
        bl_count[l]++;
        n_present++;
    }
    if (UNLIKELY(n_present == 0)) return -1;

    if (n_present >= 2) {
        uint32_t kraft = 0;
        for (int l = 1; l <= ZXC_HUF_MAX_CODE_LEN_ULTRA; l++)
            kraft += bl_count[l] << (ZXC_HUF_MAX_CODE_LEN_ULTRA - l);
        if (UNLIKELY(kraft != (1U << ZXC_HUF_MAX_CODE_LEN_ULTRA))) return -1;
    } else {
        /* Degenerate single-symbol table: the format pins the lone symbol at code
         * length exactly 1, so reject longer unary chains. */
        if (UNLIKELY(bl_count[1] != 1)) return -1;
    }

    uint32_t next_code[ZXC_HUF_MAX_CODE_LEN_ULTRA + 2] = {0};
    uint32_t code = 0;
    for (int l = 1; l <= ZXC_HUF_MAX_CODE_LEN_ULTRA; l++) {
        code = (code + bl_count[l - 1]) << 1;
        next_code[l] = code;
    }

    t->n_nodes = 1; /* root */
    t->nd[0].child[0] = -1;
    t->nd[0].child[1] = -1;
    t->nd[0].sym = -1;
    int max_depth = 0;

    for (int s = 0; s < ZXC_HUF_NUM_SYMBOLS; s++) {
        const int l = code_len[s];
        if (l == 0) continue;
        const uint32_t c = next_code[l]++;
        if (UNLIKELY(c >> l)) return -1; /* code space overflow */
        if (codes) codes[s] = c;
        int cur = 0;
        for (int d = l - 1; d >= 0; d--) {
            if (UNLIKELY(t->nd[cur].sym >= 0)) return -1; /* prefix collision */
            const int bit = (int)((c >> d) & 1U);
            int nxt = t->nd[cur].child[bit];
            if (nxt < 0) {
                if (UNLIKELY(t->n_nodes >= ZXC_PIVCO_MAX_NODES)) return -1;
                nxt = t->n_nodes++;
                t->nd[nxt].child[0] = -1;
                t->nd[nxt].child[1] = -1;
                t->nd[nxt].sym = -1;
                t->nd[cur].child[bit] = (int16_t)nxt;
            }
            cur = nxt;
        }
        if (UNLIKELY(t->nd[cur].child[0] >= 0 || t->nd[cur].child[1] >= 0)) return -1;
        t->nd[cur].sym = (int16_t)s;
        if (l > max_depth) max_depth = l;
    }
    t->max_depth = max_depth;

    /* BFS order (parents before children, left before right): this is both
     * the wire order of the node bit runs and the property that makes each
     * parent's children CONTIGUOUS in the next level's sequence buffer. */
    int head = 0;
    int tail = 0;
    t->bfs[tail++] = 0;
    int depth_end = 1; /* index in bfs where the current depth ends */
    int depth = 0;
    t->lvl_start[0] = 0;
    while (head < tail) {
        if (head == depth_end) {
            depth++;
            t->lvl_start[depth] = (int16_t)head;
            depth_end = tail;
        }
        const int nid = t->bfs[head++];
        for (int b = 0; b < 2; b++) {
            const int ch = t->nd[nid].child[b];
            if (ch >= 0) t->bfs[tail++] = (int16_t)ch;
        }
    }
    for (int d = depth + 1; d <= max_depth + 1; d++) t->lvl_start[d] = (int16_t)tail;

    /* Flat-subtree detection: min/max leaf depth per node in one reverse-BFS
     * sweep, then maximality by masking descendants of the first flat node on
     * each root-to-leaf path. */
    {
        int8_t mn[ZXC_PIVCO_MAX_NODES];
        int8_t mx[ZXC_PIVCO_MAX_NODES];
        for (int i = t->n_nodes - 1; i >= 0; i--) {
            const int nid = t->bfs[i];
            const zxc_pivco_node_t* nd = &t->nd[nid];
            if (nd->sym >= 0) {
                mn[nid] = 0;
                mx[nid] = 0;
            } else if (nd->child[0] >= 0 && nd->child[1] >= 0) {
                const int a = mn[nd->child[0]];
                const int b = mn[nd->child[1]];
                const int c = mx[nd->child[0]];
                const int d2 = mx[nd->child[1]];
                mn[nid] = (int8_t)(1 + (a < b ? a : b));
                mx[nid] = (int8_t)(1 + (c > d2 ? c : d2));
            } else { /* single-child (degenerate) : never flat */
                mn[nid] = 0;
                mx[nid] = (int8_t)ZXC_HUF_MAX_CODE_LEN_ULTRA;
            }
        }
        for (int i = 0; i < t->n_nodes; i++) {
            const int nid = t->bfs[i];
            t->flat_d[nid] = 0;
            if (i == 0) t->covered[nid] = 0; /* root */
            /* Any complete subtree unpacks flat, beating the merge cascade
             * (FORMAT RULE, both sides): D = 2-6 have SIMD unpackers, D >= 7 the
             * scalar one. (D = 1 is a leaf pair, handled on the merge path.) */
            if (!t->covered[nid] && t->nd[nid].sym < 0 && mn[nid] == mx[nid] && mn[nid] >= 2)
                t->flat_d[nid] = (uint8_t)mn[nid];
            const int ch0 = t->nd[nid].child[0];
            const int ch1 = t->nd[nid].child[1];
            const uint8_t cov = (uint8_t)(t->covered[nid] || t->flat_d[nid]);
            if (ch0 >= 0) t->covered[ch0] = cov;
            if (ch1 >= 0) t->covered[ch1] = cov;
        }
    }
    return 0;
}

/**
 * @brief Per-node symbol counts from the histogram (encoder / size estimate).
 *
 * Leaf count = freq[sym]; internal count = sum of children, computed in one
 * reverse-BFS sweep (children precede their parent in that direction).
 */
static void zxc_pivco_counts(const zxc_pivco_tree_t* RESTRICT t, const uint32_t* RESTRICT freq,
                             uint32_t* RESTRICT count) {
    for (int i = t->n_nodes - 1; i >= 0; i--) {
        const int nid = t->bfs[i];
        const zxc_pivco_node_t* nd = &t->nd[nid];
        if (nd->sym >= 0) {
            count[nid] = freq[nd->sym];
        } else {
            const uint32_t c0 = nd->child[0] >= 0 ? count[nd->child[0]] : 0;
            const uint32_t c1 = nd->child[1] >= 0 ? count[nd->child[1]] : 0;
            count[nid] = c0 + c1;
        }
    }
}

/**
 * @brief Byte size of one PivCo node's on-wire bit run.
 *
 * A wire-visible node emits one of two run kinds, both bit-packed then rounded
 * up to whole bytes:
 * - @b Flat root (@p flat_d > 0): a complete subtree of depth @p flat_d, storing
 *   @p count packed codes of @p flat_d bits each: `ceil(count * flat_d / 8)`.
 * - @b Merge node (@p flat_d == 0): one partition bit per symbol routed through
 *   the node: `ceil(count / 8)`.
 *
 * This is THE wire run-boundary rule. It is the single definition shared by the
 * size estimator (@ref zxc_huf_calc_size), the encoder (zxc_pivco_encode_core),
 * and the decoder's pass-1 pointer walk (zxc_pivco_decode_core), so the three
 * cannot drift: the encoder asserts bytes-written == estimate, and the decoder
 * advances from run to run by this exact arithmetic.
 *
 * @param[in] count  Number of symbols the node covers (packed codes if flat,
 *                    else partition bits).
 * @param[in] flat_d Flat-subtree depth (bits per packed code), or 0 for a merge
 *                    node.
 * @return Byte length of the node's run (0 when @p count is 0).
 */
static ZXC_ALWAYS_INLINE size_t zxc_pivco_run_bytes(const uint32_t count, const uint8_t flat_d) {
    return flat_d ? ((size_t)count * flat_d + 7) / 8 : ((size_t)count + 7) / 8;
}

size_t zxc_huf_calc_size(const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                         const int with_header) {
    zxc_pivco_tree_t t;
    if (UNLIKELY(zxc_pivco_tree_build(code_len, &t, NULL) != 0)) return SIZE_MAX;
    const size_t body = zxc_huf_calc_size_dict(freq, code_len, &t);
    if (UNLIKELY(body == SIZE_MAX)) return SIZE_MAX;
    return body + (with_header ? (size_t)ZXC_HUF_TABLE_SIZE : 0);
}

/**
 * @brief Sizing core shared with zxc_huf_calc_size: exact payload size of one
 *        PivCo section for a prebuilt tree (tree-at-attach dict sections call
 *        it directly; they carry no inline lengths header).
 */
size_t zxc_huf_calc_size_dict(const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                              const zxc_pivco_tree_t* RESTRICT tree) {
    /* Encodability is part of the estimate: a histogram symbol the table has no
     * code for would be silently ignored below and undercount. Report the
     * candidate unencodable instead. */
    for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++)
        if (UNLIKELY(freq[k] != 0 && code_len[k] == 0)) return SIZE_MAX;
    uint32_t count[ZXC_PIVCO_MAX_NODES];
    zxc_pivco_counts(tree, freq, count);
    size_t total = 0;
    for (int i = 0; i < tree->n_nodes; i++) {
        const int nid = tree->bfs[i];
        if (tree->covered[nid] || tree->nd[nid].sym >= 0) continue;
        total += zxc_pivco_run_bytes(count[nid], tree->flat_d[nid]);
    }
    return total;
}

/**
 * @brief Shared PivCo section encoder body.
 *
 * Walks each input symbol root-to-leaf once, appending its branch bit to the
 * per-node write cursor; node bit runs land in BFS order, byte-aligned.
 */
static int zxc_pivco_encode_core(const uint8_t* RESTRICT literals, const size_t n_literals,
                                 const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                                 const zxc_pivco_tree_t* RESTRICT t, const uint32_t* RESTRICT codes,
                                 uint8_t* RESTRICT dst, const size_t dst_cap,
                                 const int with_header) {
    if (UNLIKELY(n_literals == 0)) return ZXC_ERROR_CORRUPT_DATA;
    /* Every literal present in the histogram must have a code (a dict table
     * may lack codes for symbols unseen in training: the caller then falls
     * back to a per-block table, but a direct call must fail loudly). */
    for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++)
        if (UNLIKELY(freq[k] != 0 && code_len[k] == 0)) return ZXC_ERROR_CORRUPT_DATA;

    uint32_t count[ZXC_PIVCO_MAX_NODES];
    zxc_pivco_counts(t, freq, count);

    /* Byte offsets of every wire-visible internal node's run, in BFS order:
     * bitmap runs (1 bit/symbol) or, for flat roots, packed code runs
     * (flat_d bits/symbol). Covered nodes have no run. */
    uint32_t bit_off[ZXC_PIVCO_MAX_NODES];
    size_t payload = 0;
    for (int i = 0; i < t->n_nodes; i++) {
        const int nid = t->bfs[i];
        if (t->covered[nid] || t->nd[nid].sym >= 0) continue;
        bit_off[nid] = (uint32_t)payload;
        payload += zxc_pivco_run_bytes(count[nid], t->flat_d[nid]);
    }
    const size_t hdr = with_header ? (size_t)ZXC_HUF_TABLE_SIZE : 0;
    /* +2: the packed-code emitter uses a 3-byte read-modify-write that may
     * touch up to 2 bytes past the payload end. */
    if (UNLIKELY(hdr + payload + 2 > dst_cap)) return ZXC_ERROR_DST_TOO_SMALL;

    if (with_header) zxc_huf_pack_lengths(code_len, dst);
    uint8_t* const out = dst + hdr;
    ZXC_MEMSET(out, 0, payload + 2);

    /* Per-node bit cursors, emitting MSB-first while descending. At a flat root,
     * emit the symbol's packed D-bit residual in one shot and stop. Batching the
     * per-bit read-modify-write buys nothing: accumulators would still touch
     * memory once per bit, and flat roots already absorb the dense levels. */
    uint32_t wpos[ZXC_PIVCO_MAX_NODES];
    ZXC_MEMSET(wpos, 0, (size_t)t->n_nodes * sizeof(uint32_t));
    for (size_t i = 0; i < n_literals; i++) {
        const uint8_t s = literals[i];
        const uint32_t c = codes[s];
        int cur = 0;
        for (int d = code_len[s] - 1; d >= 0; d--) {
            const int fd = t->flat_d[cur];
            if (fd) {
                /* residual = low fd bits of the canonical code, bit-reversed
                 * so that packed bit j is the branch taken at level j. */
                uint32_t r = 0;
                for (int j = 0; j < fd; j++) r |= ((c >> (fd - 1 - j)) & 1U) << j;
                const uint32_t p = wpos[cur];
                wpos[cur] += (uint32_t)fd;
                uint8_t* q = out + bit_off[cur] + (p >> 3);
                const uint32_t sh = p & 7U;
                uint32_t w = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
                w |= r << sh; /* fd <= 11, sh <= 7 -> fits 24 bits */
                q[0] = (uint8_t)w;
                q[1] = (uint8_t)(w >> 8);
                q[2] = (uint8_t)(w >> 16);
                break;
            }
            const uint32_t bit = (c >> d) & 1U;
            const uint32_t p = wpos[cur]++;
            out[bit_off[cur] + (p >> 3)] |= (uint8_t)(bit << (p & 7));
            cur = t->nd[cur].child[bit];
        }
    }
    return (int)(hdr + payload);
}

/**
 * @brief Encode a literal/token section (PivCo layout, inline lengths header).
 *
 * Emits the 128-byte packed code-length header followed by the per-node
 * branch runs. The caller supplies the histogram and code lengths it already
 * built for the selection step.
 *
 * @param[in]  literals   Symbols to encode (n_literals bytes).
 * @param[in]  n_literals Symbol count (> 0).
 * @param[in]  freq       Symbol histogram of @p literals.
 * @param[in]  code_len   Per-symbol code lengths (every symbol with
 *                        freq != 0 must have a code).
 * @param[out] dst        Destination buffer.
 * @param[in]  dst_cap    Destination capacity in bytes.
 * @return Encoded size in bytes, or a negative ZXC_ERROR_* value
 *         (ZXC_ERROR_DST_TOO_SMALL, ZXC_ERROR_CORRUPT_DATA).
 */
int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                           uint8_t* RESTRICT dst, const size_t dst_cap) {
    zxc_pivco_tree_t t;
    uint32_t codes[ZXC_HUF_NUM_SYMBOLS];
    if (UNLIKELY(zxc_pivco_tree_build(code_len, &t, codes) != 0)) return ZXC_ERROR_CORRUPT_DATA;
    return zxc_pivco_encode_core(literals, n_literals, freq, code_len, &t, codes, dst, dst_cap, 1);
}

/**
 * @brief Encode a literal section against a shared dictionary table.
 *
 * Same payload as @ref zxc_huf_encode_section with the 128-byte lengths
 * header omitted: @p code_len / @p tree / @p codes come from the dictionary's
 * shared literal table, prebuilt ONCE at attach by @ref zxc_huf_dict_tree_build
 * (wire value enc_lit = 3).
 */
int zxc_huf_encode_section_dict(const uint8_t* RESTRICT literals, const size_t n_literals,
                                const uint32_t* RESTRICT freq, const uint8_t* RESTRICT code_len,
                                const zxc_pivco_tree_t* RESTRICT tree,
                                const uint32_t* RESTRICT codes, uint8_t* RESTRICT dst,
                                const size_t dst_cap) {
    return zxc_pivco_encode_core(literals, n_literals, freq, code_len, tree, codes, dst, dst_cap,
                                 0);
}

/* ISA-independent cold dict setup: emit once in the primary variant, not in
 * every per-ISA copy (dead weight). zxc_pivco_tree_build stays per-variant. */
#if defined(ZXC_VARIANT_PRIMARY)
/**
 * @brief Precompute the topology-derived decoder tables for @p t.
 *
 * Mirrors exactly what zxc_pivco_decode_core builds inline for per-section
 * trees: the leaf-pair skip flags, and each flat root's packed-code -> symbol
 * table (bit j of the code = branch taken at subtree level j). Both depend
 * only on the tree, so a frame-constant (dictionary) tree computes them once
 * here instead of once per decoded section.
 */
static void zxc_pivco_decode_aux_build(const zxc_pivco_tree_t* RESTRICT t,
                                       zxc_pivco_decode_aux_t* RESTRICT aux) {
    ZXC_MEMSET(aux->skip, 0, sizeof(aux->skip));
    for (int i = 0; i < t->n_nodes; i++) {
        const zxc_pivco_node_t* nd = &t->nd[t->bfs[i]];
        if (nd->sym >= 0) continue;
        const int ch0 = nd->child[0];
        const int ch1 = nd->child[1];
        if (ch0 >= 0 && ch1 >= 0 && t->nd[ch0].sym >= 0 && t->nd[ch1].sym >= 0) {
            aux->skip[ch0] = 1;
            aux->skip[ch1] = 1;
        }
    }

    /* Flat subtrees have disjoint leaves, so the concatenated tables fit in
     * ZXC_HUF_NUM_SYMBOLS pool entries (see zxc_pivco_decode_aux_t). */
    uint32_t pool_off = 0;
    for (int i = 0; i < t->n_nodes; i++) {
        const int nid = t->bfs[i];
        if (t->covered[nid] || !t->flat_d[nid]) continue;
        aux->c2s_off[nid] = (uint16_t)pool_off;
        uint8_t* const c2s = aux->c2s_pool + pool_off;
        pool_off += 1U << t->flat_d[nid];
        int16_t stk_n[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
        uint16_t stk_p[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
        uint8_t stk_l[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
        int sp = 0;
        stk_n[0] = (int16_t)nid;
        stk_p[0] = 0;
        stk_l[0] = 0;
        while (sp >= 0) {
            const int cn = stk_n[sp];
            const uint32_t cp = stk_p[sp];
            const int cl = stk_l[sp];
            sp--;
            if (t->nd[cn].sym >= 0) {
                c2s[cp] = (uint8_t)t->nd[cn].sym;
                continue;
            }
            sp++;
            stk_n[sp] = t->nd[cn].child[0];
            stk_p[sp] = (uint16_t)cp;
            stk_l[sp] = (uint8_t)(cl + 1);
            sp++;
            stk_n[sp] = t->nd[cn].child[1];
            stk_p[sp] = (uint16_t)(cp | (1U << cl));
            stk_l[sp] = (uint8_t)(cl + 1);
        }
    }
}

/**
 * @brief Prebuild a dict table's decode/encode state (tree-at-attach).
 *
 * Unpacks the 128-byte packed lengths and builds the PivCo tree, canonical
 * codes and decoder tables once; the outputs are frame-constant, so per-block
 * encode, estimate and decode reuse them instead of rebuilding (the pre-v7
 * cached-decode-table form, restored for the PivCo layout).
 */
int zxc_huf_dict_tree_build(const uint8_t* RESTRICT packed_lengths, zxc_pivco_tree_t* RESTRICT tree,
                            uint32_t* RESTRICT codes, uint8_t* RESTRICT code_len,
                            zxc_pivco_decode_aux_t* RESTRICT aux) {
    const int rc = zxc_huf_unpack_lengths(packed_lengths, code_len);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(zxc_pivco_tree_build(code_len, tree, codes) != 0)) return ZXC_ERROR_CORRUPT_DATA;
    zxc_pivco_decode_aux_build(tree, aux);
    return ZXC_OK;
}
#endif /* ZXC_VARIANT_PRIMARY */

/**
 * @brief Merge two adjacent child sequences under control bits.
 *
 * src[0..nl) is the left sequence, src[nl..nl+nr) the right one (children are
 * contiguous by construction). Control bits are LSB-first; bit 0 takes the
 * next left element, bit 1 the next right one. Speculative 16-byte loads read
 * up to 15 bytes past each cursor: callers guarantee read slack after the
 * level buffers. Writes stay within out[0..nl+nr).
 */
static ZXC_ALWAYS_INLINE void zxc_pivco_merge(uint8_t* RESTRICT out, const uint8_t* RESTRICT src,
                                              const size_t nl, const size_t nr,
                                              const uint8_t* RESTRICT bits) {
    const uint8_t* L = src;
    const uint8_t* R = src + nl;
    const size_t n = nl + nr;
    size_t i = 0;
    size_t lp = 0;
    size_t rp = 0;
#if defined(ZXC_USE_NEON64)
    /* 16 outputs per step: two-register TBL over {L[0..15], R[0..15]} with the
     * signed-index tables (see zxc_pivco_tables.h). */
    while (i + 16 <= n) {
        const uint8_t b0 = bits[i >> 3];
        const uint8_t b1 = bits[(i >> 3) + 1];
        uint8x16x2_t tb;
        tb.val[0] = vld1q_u8(L + lp);
        tb.val[1] = vld1q_u8(R + rp);
        const int pc0 = zxc_pivco_popcnt32(b0);
        const uint8x8_t ia = vld1_u8(zxc_pivco_idxa_u8[b0]);
        const uint8x8_t ib = vld1_u8(zxc_pivco_idxb_pre[pc0][b1]);
        vst1q_u8(out + i, vqtbl2q_u8(tb, vcombine_u8(ia, ib)));
        const int pc = pc0 + zxc_pivco_popcnt32(b1);
        rp += (size_t)pc;
        lp += (size_t)(16 - pc);
        i += 16;
    }
#else /* x86 tiers */
#if defined(ZXC_USE_AVX512) && defined(__AVX512VBMI2__)
    /* 64 outputs per step: the merge IS a pair of byte expands - L's bytes fill
     * the control word's 0-bit lanes, R's the 1-bit lanes. Masked loads keep the
     * speculative reads fault-safe without extra buffer slack. */
    while (i + 64 <= n) {
        uint64_t ctrl;
        ZXC_MEMCPY(&ctrl, bits + (i >> 3), 8);
        const int pc = zxc_pivco_popcnt64(ctrl);
        const __mmask64 lmask = (pc == 0) ? ~(__mmask64)0 : (((__mmask64)1 << (64 - pc)) - 1U);
        const __mmask64 rmask = (pc == 64) ? ~(__mmask64)0 : (((__mmask64)1 << pc) - 1U);
        const __m512i vl = _mm512_maskz_loadu_epi8(lmask, (const void*)(L + lp));
        const __m512i vr = _mm512_maskz_loadu_epi8(rmask, (const void*)(R + rp));
        const __m512i outl = _mm512_maskz_expand_epi8((__mmask64)~ctrl, vl);
        const __m512i outv = _mm512_mask_expand_epi8(outl, (__mmask64)ctrl, vr);
        _mm512_storeu_si512((void*)(out + i), outv);
        rp += (size_t)pc;
        lp += (size_t)(64 - pc);
        i += 64;
    }
#endif
#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || \
    (defined(ZXC_USE_SSE2) && defined(__SSSE3__))
    /* 16 outputs/step: pshufb(L, ix+0x70) | pshufb(R, ix-16) selects L[ix] or
     * R[ix-16] (out-of-range lanes zero out); 32/step unrolls it twice. */
    while (i + 32 <= n) {
        const uint8_t* cb = bits + (i >> 3);
        const int pcA0 = zxc_pivco_popcnt32(cb[0]);
        const int pcA = pcA0 + zxc_pivco_popcnt32(cb[1]);
        const int pcB0 = zxc_pivco_popcnt32(cb[2]);
        const int pcB = pcB0 + zxc_pivco_popcnt32(cb[3]);
        const size_t lpB = lp + (size_t)(16 - pcA);
        const size_t rpB = rp + (size_t)pcA;
        const __m128i vlA = _mm_loadu_si128((const __m128i*)(const void*)(L + lp));
        const __m128i vrA = _mm_loadu_si128((const __m128i*)(const void*)(R + rp));
        const __m128i vlB = _mm_loadu_si128((const __m128i*)(const void*)(L + lpB));
        const __m128i vrB = _mm_loadu_si128((const __m128i*)(const void*)(R + rpB));
        const __m128i iaA = _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxa_u8[cb[0]]);
        const __m128i ibA =
            _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxb_pre[pcA0][cb[1]]);
        const __m128i iaB = _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxa_u8[cb[2]]);
        const __m128i ibB =
            _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxb_pre[pcB0][cb[3]]);
        const __m128i ixA = _mm_unpacklo_epi64(iaA, ibA);
        const __m128i ixB = _mm_unpacklo_epi64(iaB, ibB);
        const __m128i selA =
            _mm_or_si128(_mm_shuffle_epi8(vlA, _mm_add_epi8(ixA, _mm_set1_epi8(0x70))),
                         _mm_shuffle_epi8(vrA, _mm_sub_epi8(ixA, _mm_set1_epi8(16))));
        const __m128i selB =
            _mm_or_si128(_mm_shuffle_epi8(vlB, _mm_add_epi8(ixB, _mm_set1_epi8(0x70))),
                         _mm_shuffle_epi8(vrB, _mm_sub_epi8(ixB, _mm_set1_epi8(16))));
        _mm_storeu_si128((__m128i*)(void*)(out + i), selA);
        _mm_storeu_si128((__m128i*)(void*)(out + i + 16), selB);
        lp = lpB + (size_t)(16 - pcB);
        rp = rpB + (size_t)pcB;
        i += 32;
    }
    while (i + 16 <= n) {
        const uint8_t b0 = bits[i >> 3];
        const uint8_t b1 = bits[(i >> 3) + 1];
        const __m128i vl = _mm_loadu_si128((const __m128i*)(const void*)(L + lp));
        const __m128i vr = _mm_loadu_si128((const __m128i*)(const void*)(R + rp));
        const int pc0 = zxc_pivco_popcnt32(b0);
        const __m128i ia = _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxa_u8[b0]);
        const __m128i ib =
            _mm_loadl_epi64((const __m128i*)(const void*)zxc_pivco_idxb_pre[pc0][b1]);
        const __m128i ix = _mm_unpacklo_epi64(ia, ib);
        const __m128i sell = _mm_shuffle_epi8(vl, _mm_add_epi8(ix, _mm_set1_epi8(0x70)));
        const __m128i selr = _mm_shuffle_epi8(vr, _mm_sub_epi8(ix, _mm_set1_epi8(16)));
        _mm_storeu_si128((__m128i*)(void*)(out + i), _mm_or_si128(sell, selr));
        const int pc = pc0 + zxc_pivco_popcnt32(b1);
        rp += (size_t)pc;
        lp += (size_t)(16 - pc);
        i += 16;
    }
#elif defined(ZXC_USE_NEON32)
    /* 8 outputs per step: four-register VTBL over {L[0..7], -, R[0..7], -}. The
     * shared index tables address a 32-lane view, and an 8-output step only
     * touches lanes 0..7 and 16..23, so the two spare d-registers stay zero. */
    while (i + 8 <= n) {
        const uint8_t b = bits[i >> 3];
        uint8x8x4_t tb;
        tb.val[0] = vld1_u8(L + lp);
        tb.val[1] = vdup_n_u8(0);
        tb.val[2] = vld1_u8(R + rp);
        tb.val[3] = vdup_n_u8(0);
        vst1_u8(out + i, vtbl4_u8(tb, vld1_u8(zxc_pivco_idxa_u8[b])));
        const int pc = zxc_pivco_popcnt32(b);
        rp += (size_t)pc;
        lp += (size_t)(8 - pc);
        i += 8;
    }
#endif
#endif /* x86 tiers */
    /* Portable 8-output path: byte selects from a 24-byte scratch laid out
     * as the low half of the 32-lane view the shared index tables assume
     * (L at offset 0, R at offset 16). */
    while (i + 8 <= n) {
        const uint8_t b = bits[i >> 3];
        uint8_t comb[24];
        ZXC_MEMCPY(comb, L + lp, 8);
        ZXC_MEMCPY(comb + 16, R + rp, 8);
        const uint8_t* ix = zxc_pivco_idxa_u8[b];
        for (int j = 0; j < 8; j++) out[i + (size_t)j] = comb[ix[j]];
        const int pc = zxc_pivco_popcnt32(b);
        rp += (size_t)pc;
        lp += (size_t)(8 - pc);
        i += 8;
    }
    while (i < n) {
        const int bit = (bits[i >> 3] >> (i & 7)) & 1;
        out[i++] = bit ? R[rp++] : L[lp++];
    }
}

/**
 * @brief Decode a flat subtree's packed D-bit code run into symbols.
 *
 * codes are packed LSB-first, D in [2, ZXC_HUF_MAX_CODE_LEN_ULTRA]; c2s maps
 * a packed path to its leaf symbol. SIMD paths cover the frequent D = 2 / 4
 * (in-register unpack + single 16-byte table lookup); the scalar bit-reader
 * handles every D.
 */
static void zxc_pivco_unpack_flat(uint8_t* RESTRICT out, const size_t n, const int D,
                                  const uint8_t* RESTRICT bits, const uint8_t* RESTRICT c2s) {
    size_t i = 0;
#if defined(ZXC_USE_NEON64)
    if (D == 4) {
        const uint8x16_t vc2s = vld1q_u8(c2s); /* 16 entries */
        static const uint8_t rep2[16] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
        static const int8_t sh4[16] = {0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4};
        const uint8x16_t vrep = vld1q_u8(rep2);
        const int8x16_t vsh = vld1q_s8(sh4);
        const uint8x16_t vmask = vdupq_n_u8(0x0F);
        while (i + 16 <= n) {
            uint8x16_t raw = vdupq_n_u8(0);
            raw = vreinterpretq_u8_u64(
                vsetq_lane_u64(zxc_le64(bits + ((i * 4) >> 3)), vreinterpretq_u64_u8(raw), 0));
            const uint8x16_t rep = vqtbl1q_u8(raw, vrep);
            const uint8x16_t codes = vandq_u8(vshlq_u8(rep, vsh), vmask);
            vst1q_u8(out + i, vqtbl1q_u8(vc2s, codes));
            i += 16;
        }
    } else if (D == 2) {
        uint8_t c2s16[16];
        for (int k = 0; k < 16; k++) c2s16[k] = c2s[k & 3];
        const uint8x16_t vc2s = vld1q_u8(c2s16);
        static const uint8_t rep4[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        static const int8_t sh2[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
        const uint8x16_t vrep = vld1q_u8(rep4);
        const int8x16_t vsh = vld1q_s8(sh2);
        const uint8x16_t vmask = vdupq_n_u8(0x03);
        while (i + 16 <= n) {
            uint8x16_t raw = vdupq_n_u8(0);
            raw = vreinterpretq_u8_u32(
                vsetq_lane_u32(zxc_le32(bits + ((i * 2) >> 3)), vreinterpretq_u32_u8(raw), 0));
            const uint8x16_t rep = vqtbl1q_u8(raw, vrep);
            const uint8x16_t codes = vandq_u8(vshlq_u8(rep, vsh), vmask);
            vst1q_u8(out + i, vqtbl1q_u8(vc2s, codes));
            i += 16;
        }
    } else if (D == 3) {
        /* Odd D straddle bytes: build a 16-bit window {byte_j,byte_j+1} per lane
         * (byte_j = Di>>3), right-shift by Di&7, mask D bits. 16/step, two u16x8
         * halves sharing the shift vector. D=3: 6 source bytes. */
        uint8_t c2s16[16];
        for (int k = 0; k < 16; k++) c2s16[k] = c2s[k & 7];
        const uint8x16_t vc2s = vld1q_u8(c2s16); /* 8 entries, padded to 16 */
        static const uint8_t idxA[16] = {0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3};
        static const uint8_t idxB[16] = {3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6};
        static const int16_t sh8[8] = {0, -3, -6, -1, -4, -7, -2, -5};
        const uint8x16_t vidxA = vld1q_u8(idxA);
        const uint8x16_t vidxB = vld1q_u8(idxB);
        const int16x8_t vsh = vld1q_s16(sh8);
        const uint16x8_t vmask = vdupq_n_u16(0x07);
        while (i + 16 <= n) {
            const size_t off = (i * 3) >> 3;
            uint32_t w0;
            uint16_t w1;
            ZXC_MEMCPY(&w0, bits + off, 4);
            ZXC_MEMCPY(&w1, bits + off + 4, 2); /* exactly 6 bytes read */
            uint8x16_t raw = vdupq_n_u8(0);
            raw = vreinterpretq_u8_u32(vsetq_lane_u32(w0, vreinterpretq_u32_u8(raw), 0));
            raw = vreinterpretq_u8_u16(vsetq_lane_u16(w1, vreinterpretq_u16_u8(raw), 2));
            const uint16x8_t winA = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxA));
            const uint16x8_t winB = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxB));
            const uint8x8_t cA = vmovn_u16(vandq_u16(vshlq_u16(winA, vsh), vmask));
            const uint8x8_t cB = vmovn_u16(vandq_u16(vshlq_u16(winB, vsh), vmask));
            vst1q_u8(out + i, vqtbl1q_u8(vc2s, vcombine_u8(cA, cB)));
            i += 16;
        }
    } else if (D == 5) {
        /* Same window scheme as D=3; c2s has 32 entries (vqtbl2q). 10 bytes. */
        uint8x16x2_t vc2s;
        vc2s.val[0] = vld1q_u8(c2s);
        vc2s.val[1] = vld1q_u8(c2s + 16);
        static const uint8_t idxA[16] = {0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5};
        static const uint8_t idxB[16] = {5, 6, 5, 6, 6, 7, 6, 7, 7, 8, 8, 9, 8, 9, 9, 10};
        static const int16_t sh8[8] = {0, -5, -2, -7, -4, -1, -6, -3};
        const uint8x16_t vidxA = vld1q_u8(idxA);
        const uint8x16_t vidxB = vld1q_u8(idxB);
        const int16x8_t vsh = vld1q_s16(sh8);
        const uint16x8_t vmask = vdupq_n_u16(0x1F);
        while (i + 16 <= n) {
            const size_t off = (i * 5) >> 3;
            uint64_t w0;
            uint16_t w1;
            ZXC_MEMCPY(&w0, bits + off, 8);
            ZXC_MEMCPY(&w1, bits + off + 8, 2);
            uint8x16_t raw = vdupq_n_u8(0);
            raw = vreinterpretq_u8_u64(vsetq_lane_u64(w0, vreinterpretq_u64_u8(raw), 0));
            raw = vreinterpretq_u8_u16(vsetq_lane_u16(w1, vreinterpretq_u16_u8(raw), 4));
            const uint16x8_t winA = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxA));
            const uint16x8_t winB = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxB));
            const uint8x8_t cA = vmovn_u16(vandq_u16(vshlq_u16(winA, vsh), vmask));
            const uint8x8_t cB = vmovn_u16(vandq_u16(vshlq_u16(winB, vsh), vmask));
            vst1q_u8(out + i, vqtbl2q_u8(vc2s, vcombine_u8(cA, cB)));
            i += 16;
        }
    } else if (D == 6) {
        /* Same window scheme as D=3; c2s has 64 entries (vqtbl4q). 12 bytes. */
        uint8x16x4_t vc2s;
        vc2s.val[0] = vld1q_u8(c2s);
        vc2s.val[1] = vld1q_u8(c2s + 16);
        vc2s.val[2] = vld1q_u8(c2s + 32);
        vc2s.val[3] = vld1q_u8(c2s + 48);
        static const uint8_t idxA[16] = {0, 1, 0, 1, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 5, 6};
        static const uint8_t idxB[16] = {6, 7, 6, 7, 7, 8, 8, 9, 9, 10, 9, 10, 10, 11, 11, 12};
        static const int16_t sh8[8] = {0, -6, -4, -2, 0, -6, -4, -2};
        const uint8x16_t vidxA = vld1q_u8(idxA);
        const uint8x16_t vidxB = vld1q_u8(idxB);
        const int16x8_t vsh = vld1q_s16(sh8);
        const uint16x8_t vmask = vdupq_n_u16(0x3F);
        while (i + 16 <= n) {
            const size_t off = (i * 6) >> 3;
            uint64_t w0;
            uint32_t w1;
            ZXC_MEMCPY(&w0, bits + off, 8);
            ZXC_MEMCPY(&w1, bits + off + 8, 4);
            uint8x16_t raw = vdupq_n_u8(0);
            raw = vreinterpretq_u8_u64(vsetq_lane_u64(w0, vreinterpretq_u64_u8(raw), 0));
            raw = vreinterpretq_u8_u32(vsetq_lane_u32(w1, vreinterpretq_u32_u8(raw), 2));
            const uint16x8_t winA = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxA));
            const uint16x8_t winB = vreinterpretq_u16_u8(vqtbl1q_u8(raw, vidxB));
            const uint8x8_t cA = vmovn_u16(vandq_u16(vshlq_u16(winA, vsh), vmask));
            const uint8x8_t cB = vmovn_u16(vandq_u16(vshlq_u16(winB, vsh), vmask));
            vst1q_u8(out + i, vqtbl4q_u8(vc2s, vcombine_u8(cA, cB)));
            i += 16;
        }
    }
#elif defined(ZXC_USE_NEON32)
    /* d-register mirrors of the AArch64 kernels: 8 codes per step, c2s lookup
     * via VTBL (16-entry table = two d-registers). */
    if (D == 4) {
        uint8x8x2_t vc2s;
        vc2s.val[0] = vld1_u8(c2s);
        vc2s.val[1] = vld1_u8(c2s + 8);
        static const uint8_t rep2[8] = {0, 0, 1, 1, 2, 2, 3, 3};
        static const int8_t sh4[8] = {0, -4, 0, -4, 0, -4, 0, -4};
        const uint8x8_t vrep = vld1_u8(rep2);
        const int8x8_t vsh = vld1_s8(sh4);
        const uint8x8_t vmask = vdup_n_u8(0x0F);
        while (i + 8 <= n) {
            uint32_t w;
            ZXC_MEMCPY(&w, bits + ((i * 4) >> 3), 4);
            const uint8x8_t raw = vreinterpret_u8_u32(vdup_n_u32(w));
            const uint8x8_t rep = vtbl1_u8(raw, vrep);
            const uint8x8_t codes = vand_u8(vshl_u8(rep, vsh), vmask);
            vst1_u8(out + i, vtbl2_u8(vc2s, codes));
            i += 8;
        }
    } else if (D == 2) {
        uint8_t c2s8[8];
        for (int k = 0; k < 8; k++) c2s8[k] = c2s[k & 3];
        const uint8x8_t vc2s = vld1_u8(c2s8);
        static const uint8_t rep4[8] = {0, 0, 0, 0, 1, 1, 1, 1};
        static const int8_t sh2[8] = {0, -2, -4, -6, 0, -2, -4, -6};
        const uint8x8_t vrep = vld1_u8(rep4);
        const int8x8_t vsh = vld1_s8(sh2);
        const uint8x8_t vmask = vdup_n_u8(0x03);
        while (i + 8 <= n) {
            const uint16_t w = (uint16_t)((uint16_t)bits[(i * 2) >> 3] |
                                          ((uint16_t)bits[((i * 2) >> 3) + 1] << 8));
            const uint8x8_t raw = vreinterpret_u8_u16(vdup_n_u16(w));
            const uint8x8_t rep = vtbl1_u8(raw, vrep);
            const uint8x8_t codes = vand_u8(vshl_u8(rep, vsh), vmask);
            vst1_u8(out + i, vtbl1_u8(vc2s, codes));
            i += 8;
        }
    } else if (D == 3) {
        /* 8 codes/step. 3-bit codes straddle bytes, so build a 16-bit window
         * {byte_j, byte_j+1} per lane via two VTBLs, then extract bits [s,s+3)
         * by multiply-by-2^(13-s) then >>13 - ARMv7 has no per-lane u16 variable
         * shift. 8 codes fit in 3 source bytes. */
        const uint8x8_t vc2s = vld1_u8(c2s); /* 8 entries (2^3) */
        static const uint8_t idxlo[8] = {0, 1, 0, 1, 0, 1, 1, 2};
        static const uint8_t idxhi[8] = {1, 2, 1, 2, 2, 3, 2, 3};
        static const uint16_t mul8[8] = {8192, 1024, 128, 4096, 512, 64, 2048, 256};
        const uint8x8_t vidxlo = vld1_u8(idxlo);
        const uint8x8_t vidxhi = vld1_u8(idxhi);
        const uint16x8_t vmul = vld1q_u16(mul8);
        const uint16x8_t vmask = vdupq_n_u16(0x07);
        while (i + 8 <= n) {
            const size_t off = (i * 3) >> 3;
            const uint32_t w = (uint32_t)bits[off] | ((uint32_t)bits[off + 1] << 8) |
                               ((uint32_t)bits[off + 2] << 16); /* exactly 3 bytes */
            const uint8x8_t raw = vreinterpret_u8_u32(vdup_n_u32(w));
            const uint8x8_t wlo = vtbl1_u8(raw, vidxlo);
            const uint8x8_t whi = vtbl1_u8(raw, vidxhi);
            const uint16x8_t win = vreinterpretq_u16_u8(vcombine_u8(wlo, whi));
            const uint16x8_t codes16 = vandq_u16(vshrq_n_u16(vmulq_u16(win, vmul), 13), vmask);
            vst1_u8(out + i, vtbl1_u8(vc2s, vmovn_u16(codes16)));
            i += 8;
        }
    } else if (D == 5) {
        /* Same window scheme as D=3; c2s has 32 entries (VTBL4). 5 source bytes. */
        uint8x8x4_t vc2s;
        vc2s.val[0] = vld1_u8(c2s);
        vc2s.val[1] = vld1_u8(c2s + 8);
        vc2s.val[2] = vld1_u8(c2s + 16);
        vc2s.val[3] = vld1_u8(c2s + 24);
        static const uint8_t idxlo[8] = {0, 1, 0, 1, 1, 2, 1, 2};
        static const uint8_t idxhi[8] = {2, 3, 3, 4, 3, 4, 4, 5};
        static const uint16_t mul8[8] = {2048, 64, 512, 16, 128, 1024, 32, 256};
        const uint8x8_t vidxlo = vld1_u8(idxlo);
        const uint8x8_t vidxhi = vld1_u8(idxhi);
        const uint16x8_t vmul = vld1q_u16(mul8);
        const uint16x8_t vmask = vdupq_n_u16(0x1F);
        while (i + 8 <= n) {
            uint64_t w = 0;
            ZXC_MEMCPY(&w, bits + ((i * 5) >> 3), 5);
            const uint8x8_t raw = vcreate_u8(w);
            const uint16x8_t win =
                vreinterpretq_u16_u8(vcombine_u8(vtbl1_u8(raw, vidxlo), vtbl1_u8(raw, vidxhi)));
            const uint8x8_t codes =
                vmovn_u16(vandq_u16(vshrq_n_u16(vmulq_u16(win, vmul), 11), vmask));
            vst1_u8(out + i, vtbl4_u8(vc2s, codes));
            i += 8;
        }
    } else if (D == 6) {
        /* Same window scheme; c2s has 64 entries: two VTBL4 halves, select by
         * code bit 5. 6 source bytes. */
        uint8x8x4_t lo, hi;
        lo.val[0] = vld1_u8(c2s);
        lo.val[1] = vld1_u8(c2s + 8);
        lo.val[2] = vld1_u8(c2s + 16);
        lo.val[3] = vld1_u8(c2s + 24);
        hi.val[0] = vld1_u8(c2s + 32);
        hi.val[1] = vld1_u8(c2s + 40);
        hi.val[2] = vld1_u8(c2s + 48);
        hi.val[3] = vld1_u8(c2s + 56);
        static const uint8_t idxlo[8] = {0, 1, 0, 1, 1, 2, 2, 3};
        static const uint8_t idxhi[8] = {3, 4, 3, 4, 4, 5, 5, 6};
        static const uint16_t mul8[8] = {1024, 16, 64, 256, 1024, 16, 64, 256};
        const uint8x8_t vidxlo = vld1_u8(idxlo);
        const uint8x8_t vidxhi = vld1_u8(idxhi);
        const uint16x8_t vmul = vld1q_u16(mul8);
        const uint16x8_t vmask = vdupq_n_u16(0x3F);
        while (i + 8 <= n) {
            uint64_t w = 0;
            ZXC_MEMCPY(&w, bits + ((i * 6) >> 3), 6);
            const uint8x8_t raw = vcreate_u8(w);
            const uint16x8_t win =
                vreinterpretq_u16_u8(vcombine_u8(vtbl1_u8(raw, vidxlo), vtbl1_u8(raw, vidxhi)));
            const uint8x8_t codes =
                vmovn_u16(vandq_u16(vshrq_n_u16(vmulq_u16(win, vmul), 10), vmask));
            const uint8x8_t sel =
                vbsl_u8(vcge_u8(codes, vdup_n_u8(32)), vtbl4_u8(hi, vsub_u8(codes, vdup_n_u8(32))),
                        vtbl4_u8(lo, codes));
            vst1_u8(out + i, sel);
            i += 8;
        }
    }
#elif defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || \
    (defined(ZXC_USE_SSE2) && defined(__SSE4_1__))
    if (D == 4) {
        const __m128i vc2s = _mm_loadu_si128((const __m128i*)(const void*)c2s);
        const __m128i vrep = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        const __m128i vmask = _mm_set1_epi8(0x0F);
        /* even lanes keep low nibble, odd lanes take the high nibble: shift the
         * replicated bytes right by 4 via a 16-bit shift + odd-lane select. */
        const __m128i vodd = _mm_set1_epi16(0x0F00 - 0x0F00 + (short)0xFF00);
        while (i + 16 <= n) {
            const __m128i raw =
                _mm_loadl_epi64((const __m128i*)(const void*)(bits + ((i * 4) >> 3)));
            const __m128i rep = _mm_shuffle_epi8(raw, vrep);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(rep, 4), _mm_set1_epi8(0x0F));
            const __m128i lo = _mm_and_si128(rep, vmask);
            const __m128i codes = _mm_blendv_epi8(lo, hi, vodd);
            _mm_storeu_si128((__m128i*)(void*)(out + i), _mm_shuffle_epi8(vc2s, codes));
            i += 16;
        }
    } else if (D == 2) {
        /* 16 codes from 4 bytes. SSE has no per-lane byte shift, so shift by
         * multiply: (b << (6 - 2j)) >> 6 keeps bits [2j+1:2j]. Bytes spread to
         * u16 lanes, times {64,16,4,1}, shifted, then packed back. */
        uint8_t c2s16[16];
        for (int k = 0; k < 16; k++) c2s16[k] = c2s[k & 3];
        const __m128i vc2s = _mm_loadu_si128((const __m128i*)(const void*)c2s16);
        const __m128i vspreadA =
            _mm_setr_epi8(0, (char)0x80, 0, (char)0x80, 0, (char)0x80, 0, (char)0x80, 1, (char)0x80,
                          1, (char)0x80, 1, (char)0x80, 1, (char)0x80);
        const __m128i vspreadB =
            _mm_setr_epi8(2, (char)0x80, 2, (char)0x80, 2, (char)0x80, 2, (char)0x80, 3, (char)0x80,
                          3, (char)0x80, 3, (char)0x80, 3, (char)0x80);
        const __m128i vmul = _mm_setr_epi16(64, 16, 4, 1, 64, 16, 4, 1);
        while (i + 16 <= n) {
            const __m128i raw = _mm_cvtsi32_si128(
                (int)((uint32_t)bits[(i * 2) >> 3] | ((uint32_t)bits[((i * 2) >> 3) + 1] << 8) |
                      ((uint32_t)bits[((i * 2) >> 3) + 2] << 16) |
                      ((uint32_t)bits[((i * 2) >> 3) + 3] << 24)));
            const __m128i a16 =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, vspreadA), vmul), 6);
            const __m128i b16 =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, vspreadB), vmul), 6);
            const __m128i codes = _mm_and_si128(_mm_packus_epi16(a16, b16), _mm_set1_epi8(3));
            _mm_storeu_si128((__m128i*)(void*)(out + i), _mm_shuffle_epi8(vc2s, codes));
            i += 16;
        }
    } else if (D == 3) {
        /* 16-bit window {byte_j,byte_j+1} per lane (byte_j = Di>>3); SSE has no
         * per-lane u16 shift, so extract bits [s,s+D) of window w by w*2^(16-D-s)
         * (field lands in the top D bits) then >>(16-D); s_j = Dj&7. D=3: 6 bytes. */
        uint8_t c2s16[16];
        for (int k = 0; k < 16; k++) c2s16[k] = c2s[k & 7];
        const __m128i vc2s = _mm_loadu_si128((const __m128i*)(const void*)c2s16);
        const __m128i shufA = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3);
        const __m128i shufB = _mm_setr_epi8(3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6);
        const __m128i vmul = _mm_setr_epi16(8192, 1024, 128, 4096, 512, 64, 2048, 256);
        while (i + 16 <= n) {
            const size_t off = (i * 3) >> 3;
            uint32_t w0;
            uint16_t w1;
            ZXC_MEMCPY(&w0, bits + off, 4);
            ZXC_MEMCPY(&w1, bits + off + 4, 2); /* exactly 6 bytes read */
            const __m128i raw = _mm_insert_epi16(_mm_cvtsi32_si128((int)w0), w1, 2);
            const __m128i cA =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufA), vmul), 13);
            const __m128i cB =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufB), vmul), 13);
            const __m128i codes = _mm_and_si128(_mm_packus_epi16(cA, cB), _mm_set1_epi8(7));
            _mm_storeu_si128((__m128i*)(void*)(out + i), _mm_shuffle_epi8(vc2s, codes));
            i += 16;
        }
    } else if (D == 5) {
        /* c2s has 32 entries: pshufb does 16, so select lo/hi by code bit 4. */
        const __m128i vlo = _mm_loadu_si128((const __m128i*)(const void*)c2s);
        const __m128i vhi = _mm_loadu_si128((const __m128i*)(const void*)(c2s + 16));
        const __m128i shufA = _mm_setr_epi8(0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5);
        const __m128i shufB = _mm_setr_epi8(5, 6, 5, 6, 6, 7, 6, 7, 7, 8, 8, 9, 8, 9, 9, 10);
        const __m128i vmul = _mm_setr_epi16(2048, 64, 512, 16, 128, 1024, 32, 256);
        while (i + 16 <= n) {
            const size_t off = (i * 5) >> 3;
            uint16_t w1;
            ZXC_MEMCPY(&w1, bits + off + 8, 2); /* 10 source bytes */
            const __m128i raw =
                _mm_insert_epi16(_mm_loadl_epi64((const __m128i*)(const void*)(bits + off)), w1, 4);
            const __m128i cA =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufA), vmul), 11);
            const __m128i cB =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufB), vmul), 11);
            const __m128i codes = _mm_and_si128(_mm_packus_epi16(cA, cB), _mm_set1_epi8(0x1F));
            const __m128i sel =
                _mm_blendv_epi8(_mm_shuffle_epi8(vlo, codes), _mm_shuffle_epi8(vhi, codes),
                                _mm_slli_epi16(codes, 3));
            _mm_storeu_si128((__m128i*)(void*)(out + i), sel);
            i += 16;
        }
    } else if (D == 6) {
        /* c2s has 64 entries: 4 pshufb sub-tables, select by code bits 4-5. */
        const __m128i t0 = _mm_loadu_si128((const __m128i*)(const void*)c2s);
        const __m128i t1 = _mm_loadu_si128((const __m128i*)(const void*)(c2s + 16));
        const __m128i t2 = _mm_loadu_si128((const __m128i*)(const void*)(c2s + 32));
        const __m128i t3 = _mm_loadu_si128((const __m128i*)(const void*)(c2s + 48));
        const __m128i shufA = _mm_setr_epi8(0, 1, 0, 1, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 5, 6);
        const __m128i shufB = _mm_setr_epi8(6, 7, 6, 7, 7, 8, 8, 9, 9, 10, 9, 10, 10, 11, 11, 12);
        const __m128i vmul = _mm_setr_epi16(1024, 16, 64, 256, 1024, 16, 64, 256);
        while (i + 16 <= n) {
            const size_t off = (i * 6) >> 3;
            uint32_t w1;
            ZXC_MEMCPY(&w1, bits + off + 8, 4); /* 12 source bytes */
            const __m128i raw = _mm_insert_epi32(
                _mm_loadl_epi64((const __m128i*)(const void*)(bits + off)), (int)w1, 2);
            const __m128i cA =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufA), vmul), 10);
            const __m128i cB =
                _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(raw, shufB), vmul), 10);
            const __m128i codes = _mm_and_si128(_mm_packus_epi16(cA, cB), _mm_set1_epi8(0x3F));
            const __m128i m4 = _mm_slli_epi16(codes, 3);
            const __m128i lo =
                _mm_blendv_epi8(_mm_shuffle_epi8(t0, codes), _mm_shuffle_epi8(t1, codes), m4);
            const __m128i hi =
                _mm_blendv_epi8(_mm_shuffle_epi8(t2, codes), _mm_shuffle_epi8(t3, codes), m4);
            _mm_storeu_si128((__m128i*)(void*)(out + i),
                             _mm_blendv_epi8(lo, hi, _mm_slli_epi16(codes, 2)));
            i += 16;
        }
    }
#endif
    /* Generic scalar bit-reader (any D). */
    {
        uint64_t bitpos = i * (uint64_t)D;
        const uint32_t m = (1U << D) - 1U;
        const size_t fbytes = (size_t)(((uint64_t)n * (uint64_t)D + 7U) >> 3);
        while (i < n) {
            const size_t byte = (size_t)(bitpos >> 3);
            uint32_t w = bits[byte];
            if (byte + 1 < fbytes) w |= (uint32_t)bits[byte + 1] << 8;
            if (D > 8 && byte + 2 < fbytes) w |= (uint32_t)bits[byte + 2] << 16;
            out[i++] = c2s[(w >> (bitpos & 7U)) & m];
            bitpos += (uint64_t)D;
        }
    }
}

/**
 * @brief Direct emission for a node whose two children are BOTH leaves.
 *
 * out[k] = bit_k ? sym1 : sym0, i.e. sym0 ^ (delta & mask(bit_k)). Skips
 * materialising the two child runs entirely (idea from the PivCo reference
 * implementation: sequential stores + bit-test/XOR-blend beat a table merge
 * of two constant runs by 2-3x).
 */
static ZXC_ALWAYS_INLINE void zxc_pivco_emit_leaf_pair(uint8_t* RESTRICT out, const size_t n,
                                                       const uint8_t sym0, const uint8_t sym1,
                                                       const uint8_t* RESTRICT bits) {
    size_t i = 0;
#if defined(ZXC_USE_NEON64)
    const uint8x16_t vsym0 = vdupq_n_u8(sym0);
    const uint8x16_t vdelta = vdupq_n_u8((uint8_t)(sym0 ^ sym1));
    static const uint8_t rep_idx[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static const uint8_t bit_sel[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t vrep = vld1q_u8(rep_idx);
    const uint8x16_t vsel = vld1q_u8(bit_sel);
    while (i + 16 <= n) {
        uint8x16_t ctrl = vdupq_n_u8(0);
        ctrl = vsetq_lane_u8(bits[i >> 3], ctrl, 0);
        ctrl = vsetq_lane_u8(bits[(i >> 3) + 1], ctrl, 1);
        const uint8x16_t rep = vqtbl1q_u8(ctrl, vrep);
        const uint8x16_t mask = vtstq_u8(rep, vsel);
        vst1q_u8(out + i, veorq_u8(vsym0, vandq_u8(vdelta, mask)));
        i += 16;
    }
#elif defined(ZXC_USE_NEON32)
    const uint8x8_t vsym0 = vdup_n_u8(sym0);
    const uint8x8_t vdelta = vdup_n_u8((uint8_t)(sym0 ^ sym1));
    static const uint8_t bit_sel8[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t vsel = vld1_u8(bit_sel8);
    while (i + 8 <= n) {
        const uint8x8_t mask = vtst_u8(vdup_n_u8(bits[i >> 3]), vsel);
        vst1_u8(out + i, veor_u8(vsym0, vand_u8(vdelta, mask)));
        i += 8;
    }
#elif defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || \
    (defined(ZXC_USE_SSE2) && defined(__SSSE3__))
    const __m128i vsym0 = _mm_set1_epi8((char)sym0);
    const __m128i vdelta = _mm_set1_epi8((char)(uint8_t)(sym0 ^ sym1));
    const __m128i vrep = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i vsel =
        _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
    while (i + 16 <= n) {
        const __m128i ctrl = _mm_cvtsi32_si128(bits[i >> 3] | ((int)bits[(i >> 3) + 1] << 8));
        const __m128i rep = _mm_shuffle_epi8(ctrl, vrep);
        /* lanes where the selected bit is set -> 0xFF */
        const __m128i mask = _mm_cmpeq_epi8(_mm_and_si128(rep, vsel), vsel);
        _mm_storeu_si128((__m128i*)(void*)(out + i),
                         _mm_xor_si128(vsym0, _mm_and_si128(vdelta, mask)));
        i += 16;
    }
#endif
    while (i < n) {
        const int bit = (bits[i >> 3] >> (i & 7)) & 1;
        out[i++] = bit ? sym1 : sym0;
    }
}

/**
 * @brief Shared PivCo section decoder body (code lengths already unpacked).
 *
 * Pass 1 (sizing) walks the wire's BFS order once: each internal node's bit
 * run is located, bounds-checked and popcounted, which yields both children's
 * symbol counts. Pass 2 rebuilds sequences bottom-up, one level at a time,
 * ping-ponging between dst (even depths, depth 0 = final output) and scratch
 * (odd depths); a level's writes never alias its reads.
 *
 * @p aux carries the topology-derived tables (leaf-pair skip flags, flat-root
 * c2s) precomputed at attach for a frame-constant dictionary tree; NULL for
 * per-section trees, which rebuild them inline below.
 */
static int zxc_pivco_decode_core(const uint8_t* RESTRICT payload, const size_t payload_size,
                                 uint8_t* RESTRICT dst, const size_t n,
                                 const zxc_pivco_tree_t* RESTRICT t,
                                 const zxc_pivco_decode_aux_t* RESTRICT aux,
                                 uint8_t* RESTRICT scratch) {
    if (UNLIKELY(n == 0)) return ZXC_ERROR_CORRUPT_DATA;

    /* Pass 1: node counts + bit-run pointers, straight from the wire order. */
    uint32_t count[ZXC_PIVCO_MAX_NODES];
    const uint8_t* bit_ptr[ZXC_PIVCO_MAX_NODES];
    count[0] = (uint32_t)n;
    const uint8_t* p = payload;
    const uint8_t* const pend = payload + payload_size;
    for (int i = 0; i < t->n_nodes; i++) {
        const int nid = t->bfs[i];
        const zxc_pivco_node_t* nd = &t->nd[nid];
        if (t->covered[nid] || nd->sym >= 0) continue;
        const uint32_t c = count[nid];
        if (t->flat_d[nid]) {
            /* Packed-code run: no partition below, nothing to popcount. */
            const size_t fbytes = zxc_pivco_run_bytes(c, t->flat_d[nid]);
            if (UNLIKELY((size_t)(pend - p) < fbytes)) return ZXC_ERROR_CORRUPT_DATA;
            bit_ptr[nid] = p;
            p += fbytes;
            continue;
        }
        const size_t nbytes = zxc_pivco_run_bytes(c, 0); /* merge node: c partition bits */
        if (UNLIKELY((size_t)(pend - p) < nbytes)) return ZXC_ERROR_CORRUPT_DATA;
        bit_ptr[nid] = p;
        /* popcount of the c valid bits = the right child's count. All but the
         * last byte counted fast, the last one with its padding masked - if the
         * 8-byte loop swallowed it unmasked, padding bits would inflate `ones`. */
        uint32_t ones = 0;
        if (nbytes) {
            const size_t full = nbytes - 1;
            size_t k = 0;
            for (; k + 8 <= full; k += 8) {
                uint64_t w;
                ZXC_MEMCPY(&w, p + k, 8);
                ones += (uint32_t)zxc_pivco_popcnt64(w);
            }
            for (; k < full; k++) ones += (uint32_t)zxc_pivco_popcnt32(p[k]);
            uint8_t last = p[full];
            if (c & 7U) last &= (uint8_t)((1U << (c & 7U)) - 1U);
            ones += (uint32_t)zxc_pivco_popcnt32(last);
        }
        p += nbytes;
        if (UNLIKELY(ones > c)) return ZXC_ERROR_CORRUPT_DATA;
        const int ch0 = nd->child[0];
        const int ch1 = nd->child[1];
        if (ch1 >= 0)
            count[ch1] = ones;
        else if (UNLIKELY(ones != 0))
            return ZXC_ERROR_CORRUPT_DATA;
        if (ch0 >= 0)
            count[ch0] = c - ones;
        else if (UNLIKELY(c - ones != 0))
            return ZXC_ERROR_CORRUPT_DATA;
    }

    /* Per-level sequence offsets (BFS order => children contiguous). */
    uint32_t seq_off[ZXC_PIVCO_MAX_NODES];
    for (int d = 0; d <= t->max_depth; d++) {
        uint32_t off = 0;
        for (int i = t->lvl_start[d]; i < t->lvl_start[d + 1]; i++) {
            const int nid = t->bfs[i];
            if (t->covered[nid]) continue; /* not materialised anywhere */
            seq_off[nid] = off;
            off += count[nid];
        }
    }

    /* Leaf-pair parents emit both runs directly from their bits (XOR-blend),
     * so their children never need materialising: flag them for skipping.
     * A dictionary tree carries these flags precomputed in aux. */
    uint8_t skip_local[ZXC_PIVCO_MAX_NODES];
    const uint8_t* skip;
    if (aux) {
        skip = aux->skip;
    } else {
        ZXC_MEMSET(skip_local, 0, sizeof(skip_local));
        for (int i = 0; i < t->n_nodes; i++) {
            const zxc_pivco_node_t* nd = &t->nd[t->bfs[i]];
            if (nd->sym >= 0) continue;
            const int ch0 = nd->child[0];
            const int ch1 = nd->child[1];
            if (ch0 >= 0 && ch1 >= 0 && t->nd[ch0].sym >= 0 && t->nd[ch1].sym >= 0) {
                skip_local[ch0] = 1;
                skip_local[ch1] = 1;
            }
        }
        skip = skip_local;
    }

    /* Pass 2: bottom-up level reconstruction. */
    for (int d = t->max_depth; d >= 0; d--) {
        uint8_t* const buf_d = (d & 1) ? scratch : dst;
        const uint8_t* const buf_c = (d & 1) ? dst : scratch; /* children at d+1 (read-only) */
        for (int i = t->lvl_start[d]; i < t->lvl_start[d + 1]; i++) {
            const int nid = t->bfs[i];
            if (t->covered[nid]) continue; /* lives inside a flat root's run */
            const zxc_pivco_node_t* nd = &t->nd[nid];
            const uint32_t c = count[nid];
            if (c == 0 || skip[nid]) continue;
            if (nd->sym >= 0) {
                ZXC_MEMSET(buf_d + seq_off[nid], (uint8_t)nd->sym, c);
            } else if (t->flat_d[nid]) {
                /* Packed-path -> symbol table (complete subtree of depth D:
                 * 2^D leaves): precomputed at attach for a dictionary tree,
                 * else built here, then unpack the code run directly. */
                const int D = t->flat_d[nid];
                uint8_t c2s_local[1U << ZXC_HUF_MAX_CODE_LEN_ULTRA];
                const uint8_t* c2s;
                if (aux) {
                    c2s = aux->c2s_pool + aux->c2s_off[nid];
                } else {
                    int16_t stk_n[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
                    uint16_t stk_p[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
                    uint8_t stk_l[ZXC_HUF_MAX_CODE_LEN_ULTRA + 1];
                    int sp = 0;
                    stk_n[0] = (int16_t)nid;
                    stk_p[0] = 0;
                    stk_l[0] = 0;
                    while (sp >= 0) {
                        const int cn = stk_n[sp];
                        const uint32_t cp = stk_p[sp];
                        const int cl = stk_l[sp];
                        sp--;
                        if (t->nd[cn].sym >= 0) {
                            c2s_local[cp] = (uint8_t)t->nd[cn].sym;
                            continue;
                        }
                        sp++;
                        stk_n[sp] = t->nd[cn].child[0];
                        stk_p[sp] = (uint16_t)cp;
                        stk_l[sp] = (uint8_t)(cl + 1);
                        sp++;
                        stk_n[sp] = t->nd[cn].child[1];
                        stk_p[sp] = (uint16_t)(cp | (1U << cl));
                        stk_l[sp] = (uint8_t)(cl + 1);
                    }
                    c2s = c2s_local;
                }
                zxc_pivco_unpack_flat(buf_d + seq_off[nid], c, D, bit_ptr[nid], c2s);
            } else {
                const int ch0 = nd->child[0];
                const int ch1 = nd->child[1];
                if (ch0 >= 0 && ch1 >= 0 && t->nd[ch0].sym >= 0 && t->nd[ch1].sym >= 0) {
                    zxc_pivco_emit_leaf_pair(buf_d + seq_off[nid], c, (uint8_t)t->nd[ch0].sym,
                                             (uint8_t)t->nd[ch1].sym, bit_ptr[nid]);
                    continue;
                }
                const uint32_t nl = ch0 >= 0 ? count[ch0] : 0;
                const uint32_t src_off = ch0 >= 0 ? seq_off[ch0] : seq_off[nd->child[1]];
                zxc_pivco_merge(buf_d + seq_off[nid], buf_c + src_off, nl, c - nl, bit_ptr[nid]);
            }
        }
    }
    return ZXC_OK;
}

/**
 * @brief Decode a literal/token section (PivCo layout, inline lengths header).
 *
 * Unpacks and validates the 128-byte code-length header, then runs the
 * bottom-up merge decoder.
 *
 * @param[in]  payload      Section bytes (header + node runs).
 * @param[in]  payload_size Section size in bytes.
 * @param[out] dst          Receives the @p n decoded symbols; needs
 *                          ZXC_PAD_SIZE bytes of write slack past @p n.
 * @param[in]  n            Expected symbol count.
 * @param[in]  scratch      Level ping-pong buffer with at least
 *                          n + ZXC_PIVCO_SCRATCH_PAD bytes.
 * @return ZXC_OK or ZXC_ERROR_CORRUPT_DATA.
 */
int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n, uint8_t* RESTRICT scratch) {
    if (UNLIKELY(payload_size < (size_t)ZXC_HUF_TABLE_SIZE)) return ZXC_ERROR_CORRUPT_DATA;
    uint8_t code_len[ZXC_HUF_NUM_SYMBOLS];
    const int rc = zxc_huf_unpack_lengths(payload, code_len);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    zxc_pivco_tree_t t;
    if (UNLIKELY(zxc_pivco_tree_build(code_len, &t, NULL) != 0)) return ZXC_ERROR_CORRUPT_DATA;
    return zxc_pivco_decode_core(payload + ZXC_HUF_TABLE_SIZE, payload_size - ZXC_HUF_TABLE_SIZE,
                                 dst, n, &t, NULL, scratch);
}

/**
 * @brief Decode a shared-dictionary literal section (no inline header).
 *
 * Same as @ref zxc_huf_decode_section, but against the dictionary's PivCo
 * @p tree and decoder tables @p aux, prebuilt ONCE at attach time by
 * @ref zxc_huf_dict_tree_build (wire value enc_lit = 3) -- no per-block
 * unpack, tree rebuild, or c2s/skip rebuild.
 */
int zxc_huf_decode_section_dict(const uint8_t* RESTRICT payload, const size_t payload_size,
                                uint8_t* RESTRICT dst, const size_t n,
                                const zxc_pivco_tree_t* RESTRICT tree,
                                const zxc_pivco_decode_aux_t* RESTRICT aux,
                                uint8_t* RESTRICT scratch) {
    return zxc_pivco_decode_core(payload, payload_size, dst, n, tree, aux, scratch);
}
