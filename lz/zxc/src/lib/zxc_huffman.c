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
 * If ZXC_FUNCTION_SUFFIX is defined (e.g. _avx2, _neon), rename the public
 * entry points so each variant TU produces its own copy under a unique symbol
 * (e.g. zxc_huf_decode_section_avx2). The runtime dispatcher in
 * zxc_compress.c / zxc_decompress.c routes to the matching variant.
 *
 * The defines sit before zxc_internal.h so the header's prototypes are
 * rewritten with the same suffix as the definitions below.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_huf_build_code_lengths ZXC_CAT(zxc_huf_build_code_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_pack_lengths ZXC_CAT(zxc_huf_pack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_unpack_lengths ZXC_CAT(zxc_huf_unpack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size ZXC_CAT(zxc_huf_calc_size, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_calc_size_dict ZXC_CAT(zxc_huf_calc_size_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_dict_tree_build ZXC_CAT(zxc_huf_dict_tree_build, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section ZXC_CAT(zxc_huf_encode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section_dict ZXC_CAT(zxc_huf_encode_section_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section ZXC_CAT(zxc_huf_decode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section_dict ZXC_CAT(zxc_huf_decode_section_dict, ZXC_FUNCTION_SUFFIX)
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
        /* Degenerate single-symbol table: the format requires the lone symbol
         * to have code length exactly 1 (FORMAT.md, decoder validation
         * requirements); the encoder never emits anything else. Reject longer
         * unary chains. */
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
    /* Encodability is part of the estimate: a histogram symbol without a
     * code (dict table not covering this block) has no leaf, so the counts
     * below would silently ignore it and undercount. Such a candidate cannot
     * be emitted -- report it as unencodable instead. */
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

    /* Per-node bit cursors; emit MSB-first code bits while descending. At a
     * flat root, emit the symbol's packed D-bit residual (bit j = branch at
     * subtree level j) in one shot and stop descending.
     * Batching the per-bit RMW (read-modify-write) brings nothing:
     * per-node accumulators would still touch memory once per bit, and flat
     * roots already absorb the dense levels in one shot. */
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
    /* 64 outputs per step: the merge IS a pair of byte expands. Bytes of L
     * fill the 0-bit lanes of the control word, bytes of R the 1-bit lanes
     * (merge-masked into the first result). Masked loads keep the speculative
     * reads fault-safe without extra buffer slack. */
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
    /* 8 outputs per step: four-register VTBL over {L[0..7], -, R[0..7], -}.
     * The shared index tables address a 32-lane view (L lanes 0..15, R lanes
     * 16..31); an 8-output step only ever references lanes 0..7 and 16..23,
     * so the two unused d-registers stay undefined-but-harmless zeros. */
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
         * {byte_j, byte_j+1} per lane (byte_j = 3i>>3) via two VTBLs, then
         * extract bits [s,s+3) by multiply-by-2^(13-s) + >>13 (no per-lane u16
         * variable shift on ARMv7); s_j = 3j&7. 8 codes fit in 3 source bytes. */
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
        /* 16 codes from 4 bytes. SSE lacks per-lane byte shifts, so shift via a
         * per-lane multiply: (b << (6 - 2j)) >> 6 keeps bits [2j+1:2j]. Bytes
         * are spread to u16 lanes, multiplied by {64,16,4,1}, shifted, then the
         * two halves are packed back to 16 code bytes. */
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
        /* popcount of the c valid bits = right child's count. Count all but the
         * last byte fast, then the last byte with its padding masked off: the
         * 8-byte loop must not swallow the final byte unmasked when nbytes % 8 == 0
         * (else set padding bits inflate `ones` past the true child count). */
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
