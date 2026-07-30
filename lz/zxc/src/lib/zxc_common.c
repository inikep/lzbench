/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_common.c
 * @brief Shared library utilities: context management, header I/O,
 *        compress-bound calculation, and error-code name lookup.
 *
 * This translation unit contains the functions shared by both the buffer and
 * streaming APIs.  It is linked into every build of libzxc.
 */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * @brief Allocates memory aligned to the specified boundary.
 *
 * Uses `_aligned_malloc` on Windows and `posix_memalign` elsewhere.
 */
void* zxc_aligned_malloc(const size_t size, const size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#endif
}

/**
 * @brief Frees memory previously allocated by zxc_aligned_malloc().
 */
void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Returns @c sizeof(zxc_compress_opts_t) for ABI-safe allocation.
 *
 * Public API; see @c zxc_buffer.h. Lets callers (other languages, or a
 * different library version) size the options struct without knowing its layout.
 *
 * @return Size of @ref zxc_compress_opts_t in bytes.
 */
size_t zxc_compress_opts_size(void) { return sizeof(zxc_compress_opts_t); }

/**
 * @brief Returns @c sizeof(zxc_decompress_opts_t) for ABI-safe allocation.
 *
 * Public API; see @c zxc_buffer.h. Lets callers (other languages, or a
 * different library version) size the options struct without knowing its layout.
 *
 * @return Size of @ref zxc_decompress_opts_t in bytes.
 */
size_t zxc_decompress_opts_size(void) { return sizeof(zxc_decompress_opts_t); }

// Offset table of the persistent buffer carved by every cctx/dctx init. Both
// modes compute it identically, for the workspace sizer and the in-place init.
typedef struct {
    size_t total;
    // mode == 0 (decompress)
    size_t off_work;
    size_t off_lit_dctx;
    // mode == 0: scratch for a Huffman-coded GLO token section (enc_tok == HUFFMAN).
    size_t off_tok_dctx;
    size_t sz_tok_dctx;
    // mode == 0: PivCo decode level scratch (one chunk-sized ping-pong buffer).
    size_t off_pivco_dctx;
    size_t sz_pivco_dctx;
    // mode == 1 (compress)
    size_t off_hash_pos;
    size_t off_hash_tags;
    size_t off_chain;
    size_t off_seq_union;
    size_t off_extras;
    size_t off_lit_cctx;
    // meaningful only when sz_opt > 0 (level >= ZXC_LEVEL_DENSITY).
    size_t off_opt;
    // both modes: [dict | data] concat scratch, present only when dict_size > 0.
    size_t off_dict;
    // both modes: dict Huffman tree-at-attach state, present only when dict_size > 0.
    size_t off_dict_huf;
    // Sub-buffer sizes (re-used by the partitioning step + zero-init).
    size_t sz_hash_pos;
    size_t sz_hash_tags;
    size_t sz_opt;
    size_t sz_dict; /* 0 = no dictionary buffer. */
    size_t max_seq;
} zxc_cctx_layout_t;

/**
 * @brief Worst-case sequence count for one block. Shared by the compressor's
 *        buffer sizing and the decoder's token scratch: the decode side must
 *        accept exactly what the compress side can emit, so both derive from
 *        this single expression.
 */
static ZXC_ALWAYS_INLINE size_t zxc_cctx_max_seq(const size_t chunk_size) {
    return chunk_size / ZXC_LZ_MIN_MATCH_LEN + 16;
}

/**
 * @brief Decode-side entropy scratch sizes for one block: token scratch
 *        (worst-case sequence count + wild-read pad) and PivCo ping-pong
 *        scratch. Single definition shared by the layout (full provisioning:
 *        static workspaces) and the lazy heap allocator
 *        (@ref zxc_cctx_alloc_entropy_scratch), so the two can never drift.
 */
static void zxc_dctx_entropy_sizes(const size_t chunk_size, size_t* RESTRICT sz_tok,
                                   size_t* RESTRICT sz_pivco) {
    *sz_tok = zxc_cctx_max_seq(chunk_size) + ZXC_PAD_SIZE;
    *sz_pivco = chunk_size + ZXC_PIVCO_SCRATCH_PAD;
}

/**
 * @brief Computes the single-allocation memory layout for a compression /
 *        decompression context.
 *
 * Walks the same partition order used by @ref zxc_cctx_init_in_workspace and
 * records each sub-buffer's offset plus the running @c total, so the sizing
 * query and the partitioning step share one source of truth and can never
 * disagree.
 *
 * Decompress (@p mode == 0) reserves @c work_buf, @c lit_buffer (both padded
 * for wild-copy overshoot) and the token / PivCo decode scratch buffers.
 * Compress (@p mode == 1) reserves the LZ match-finder
 * tables (hash positions, tags, chain), the sequence / extras / literal buffers
 * and - only at @c level >= ZXC_LEVEL_DENSITY - the optimal-parser scratch. A
 * @p dict_size > 0 appends the [dict | data] concat scratch in both modes.
 *
 * Every offset is cache-line aligned via @c ZXC_ALIGN_CL.
 *
 * @param[in] chunk_size  Block size in bytes.
 * @param[in] mode        1 = compression, 0 = decompression.
 * @param[in] level       Compression level (only consulted when @p mode == 1).
 * @param[in] dict_size   Dictionary prefill size; when > 0 the layout includes
 *                        the [dict | data] concat buffer.
 * @param[in] defer_entropy_scratch  When non-zero, the decode-side token
 *                        and PivCo scratch are left out of the layout and
 *                        allocated lazily on the first entropy section.
 * @return Fully populated layout; @c .total is the required workspace size.
 */
static zxc_cctx_layout_t compute_cctx_layout(const size_t chunk_size, const int mode,
                                             const int level, const size_t dict_size,
                                             const int defer_entropy_scratch) {
    zxc_cctx_layout_t layout = {0};

    const size_t max_seq = zxc_cctx_max_seq(chunk_size);

    if (mode == 0) {
        // Decompress: work_buf + lit_buffer, padded for wild-copy overshoot and
        // sized worst-case. lit_buffer is provisioned at every level - the decoder
        // cannot predict a block's literal encoding (RAW / RLE / HUFFMAN).
        const size_t sz_work = chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
        const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

        layout.off_work = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_work);
        layout.off_lit_dctx = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_lit);
        // Token-section decode scratch (level-7 GLO Huffman-codes the tokens) +
        // PivCo ping-pong scratch. enc_lit/enc_tok are unpredictable, so static
        // workspaces provision both up front (no-alloc contract); heap contexts
        // defer them to the first entropy section, sparing L1-5 archives ~1.2x
        // chunk_size. See zxc_cctx_alloc_entropy_scratch.
        if (!defer_entropy_scratch) {
            size_t sz_tok = 0;
            size_t sz_pivco = 0;
            zxc_dctx_entropy_sizes(chunk_size, &sz_tok, &sz_pivco);
            layout.sz_tok_dctx = sz_tok;
            layout.off_tok_dctx = layout.total;
            layout.total += ZXC_ALIGN_CL(layout.sz_tok_dctx);
            layout.sz_pivco_dctx = sz_pivco;
            layout.off_pivco_dctx = layout.total;
            layout.total += ZXC_ALIGN_CL(layout.sz_pivco_dctx);
        }
    } else {
        // Compress: 6 partitions + optional opt_scratch at level >= ZXC_LEVEL_DENSITY.
        const uint32_t offset_bits = zxc_log2_u32((uint32_t)chunk_size);
        layout.max_seq = max_seq;
        layout.sz_hash_pos = ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
        layout.sz_hash_tags = ZXC_LZ_HASH_SIZE * sizeof(uint8_t);
        const size_t sz_chain = ZXC_LZ_WINDOW_SIZE * sizeof(uint16_t);
        // buf_sequences (GHI, level <= ZXC_LEVEL_FAST) aliases buf_offsets + buf_tokens (GLO,
        // level >= ZXC_LEVEL_DEFAULT). Mutually exclusive per block; sized for the larger.
        const size_t sz_seq_union = layout.max_seq * sizeof(uint32_t);
        const size_t vbyte_len = (offset_bits + 6) / 7;
        const size_t sz_extras = layout.max_seq * 2 * vbyte_len;
        const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

        // opt_scratch (level >= ZXC_LEVEL_DENSITY): the optimal parser's DP arrays,
        // reused transiently as package-merge scratch by the code-length builder,
        // so sized to the larger demand. Keep in sync with zxc_estimate_cctx_size()
        // and its consumer in zxc_compress.c.
        if (level >= ZXC_LEVEL_DENSITY) {
            size_t sz_dp;
            size_t sz_pl;
            size_t sz_po;
            size_t sz_bm;
            zxc_opt_dp_sizes(chunk_size, &sz_dp, &sz_pl, &sz_po, &sz_bm);
            const size_t dp_needed = sz_dp + sz_pl + sz_po + sz_bm;
            layout.sz_opt =
                (dp_needed > ZXC_HUF_BUILD_SCRATCH_SIZE) ? dp_needed : ZXC_HUF_BUILD_SCRATCH_SIZE;
        }

        layout.off_hash_pos = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_pos);
        layout.off_hash_tags = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_tags);
        layout.off_chain = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_chain);
        layout.off_seq_union = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_seq_union);
        layout.off_extras = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_extras);
        layout.off_lit_cctx = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_lit);
        // opt_scratch is appended last so it is absent for levels 1..5 (zero
        // waste on the common path) and only inflates the workspace at level 6.
        if (layout.sz_opt) {
            layout.off_opt = layout.total;
            layout.total += ZXC_ALIGN_CL(layout.sz_opt);
        }
    }

    // [dict | data] concat scratch (dict only). Compress chunk_size already
    // spans [dict | block]; decompress prepends dict to a (chunk + PAD) region.
    if (dict_size > 0) {
        layout.sz_dict = (mode == 1) ? (chunk_size + ZXC_DECOMPRESS_TAIL_PAD)
                                     : (dict_size + chunk_size + ZXC_DECOMPRESS_TAIL_PAD);
        layout.off_dict = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_dict);
        layout.off_dict_huf = layout.total;
        layout.total += ZXC_ALIGN_CL(sizeof(zxc_dict_huf_state_t));
    }
    return layout;
}

/**
 * @brief Returns the workspace byte count required for the given parameters.
 *
 * Public contract documented at the declaration in @c zxc_internal.h. Thin
 * wrapper that returns @c compute_cctx_layout(...).total, or 0 when
 * @p chunk_size is 0.
 */
size_t zxc_cctx_compute_workspace_size(const size_t chunk_size, const int mode, const int level,
                                       const size_t dict_size) {
    if (UNLIKELY(chunk_size == 0)) return 0;
    return compute_cctx_layout(chunk_size, mode, level, dict_size, 0).total;
}

/**
 * @brief Partitions a caller-supplied workspace into a ready-to-use context.
 *
 * Public contract (alignment, lifetime, return codes) documented at the
 * declaration in @c zxc_internal.h. Computes the layout via
 * @ref compute_cctx_layout, rejects an undersized @p workspace, then carves the
 * sub-buffers out of it. @c ctx->memory_block stays NULL so @ref zxc_cctx_free
 * leaves the caller-owned workspace untouched.
 */
int zxc_cctx_init_in_workspace(zxc_cctx_t* RESTRICT ctx, void* RESTRICT workspace,
                               const size_t workspace_size, const size_t chunk_size, const int mode,
                               const int level, const int checksum_enabled, const size_t dict_size,
                               const int defer_entropy_scratch) {
    if (UNLIKELY(!ctx || !workspace || chunk_size == 0)) return ZXC_ERROR_NULL_INPUT;

    const zxc_cctx_layout_t layout =
        compute_cctx_layout(chunk_size, mode, level, dict_size, defer_entropy_scratch);
    if (UNLIKELY(workspace_size < layout.total)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMSET(ctx, 0, sizeof(zxc_cctx_t));
    ctx->checksum_enabled = checksum_enabled;
    ctx->chunk_size = chunk_size;
    const uint32_t offset_bits = zxc_log2_u32((uint32_t)chunk_size);
    ctx->offset_bits = offset_bits;
    ctx->offset_mask = (uint32_t)((1ULL << offset_bits) - 1);
    ctx->max_epoch = (uint32_t)(1ULL << (32 - offset_bits));

    // memory_block stays NULL on the static-init path so zxc_cctx_free does
    // not try to free the caller's workspace.  Sub-buffer pointers carry the
    // partition; ownership is implicit (the caller owns @p workspace).
    uint8_t* const mem = (uint8_t*)workspace;

    // Dictionary concat scratch (both modes); init owns dict_size now so callers
    // no longer assign ctx->dict_size after init.
    ctx->dict_size = dict_size;
    if (dict_size > 0) {
        ctx->dict_buffer = mem + layout.off_dict;
        ctx->dict_buffer_cap = layout.sz_dict;
        ctx->dict_huf = (zxc_dict_huf_state_t*)(void*)(mem + layout.off_dict_huf);
    }

    if (mode == 0) {
        ctx->work_buf = mem + layout.off_work;
        ctx->work_buf_cap = chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
        ctx->lit_buffer = mem + layout.off_lit_dctx;
        ctx->lit_buffer_cap = chunk_size + ZXC_PAD_SIZE;
        if (layout.sz_pivco_dctx) {
            ctx->tok_buffer = mem + layout.off_tok_dctx;
            ctx->tok_buffer_cap = layout.sz_tok_dctx;
            ctx->pivco_scratch = mem + layout.off_pivco_dctx;
            ctx->pivco_scratch_cap = layout.sz_pivco_dctx;
        }
        return ZXC_OK;
    }

    ctx->hash_table = (uint32_t*)(mem + layout.off_hash_pos);
    ctx->hash_tags = mem + layout.off_hash_tags;
    ctx->chain_table = (uint16_t*)(mem + layout.off_chain);
    ctx->buf_sequences = (uint32_t*)(mem + layout.off_seq_union);
    ctx->buf_offsets = (uint16_t*)(mem + layout.off_seq_union);
    ctx->buf_tokens = mem + layout.off_seq_union + layout.max_seq * sizeof(uint16_t);
    ctx->buf_extras = mem + layout.off_extras;
    ctx->literals = mem + layout.off_lit_cctx;
    if (layout.sz_opt) {
        ctx->opt_scratch = mem + layout.off_opt;
        ctx->opt_scratch_cap = layout.sz_opt;
    }

    ctx->compression_level = level;
    ctx->epoch = 1;

    ZXC_MEMSET(ctx->hash_table, 0, layout.sz_hash_pos);
    ZXC_MEMSET(ctx->hash_tags, 0, layout.sz_hash_tags);
    return ZXC_OK;
}

/**
 * @brief Initialises a compression / decompression context, allocating the
 *        persistent buffer with @c ZXC_ALIGNED_MALLOC.
 *
 * Thin wrapper around zxc_cctx_init_in_workspace(): sizes the buffer via
 * @ref zxc_cctx_compute_workspace_size, allocates it, then partitions it.
 * The pointer is stored in @c ctx->memory_block so @ref zxc_cctx_free can
 * release it.  The static-cctx public API (see @c zxc_buffer.h) bypasses
 * this wrapper and partitions a caller-supplied workspace directly.
 */
int zxc_cctx_init(zxc_cctx_t* RESTRICT ctx, const size_t chunk_size, const int mode,
                  const int level, const int checksum_enabled, const size_t dict_size) {
    if (UNLIKELY(chunk_size == 0)) return ZXC_ERROR_NULL_INPUT;
    // Heap contexts defer the decode-side entropy scratch to the first entropy
    // section; static workspaces must have it already, they may never allocate.
    const int defer_entropy = (mode == 0);
    const size_t total =
        compute_cctx_layout(chunk_size, mode, level, dict_size, defer_entropy).total;
    if (UNLIKELY(total == 0)) return ZXC_ERROR_NULL_INPUT;

    uint8_t* const mem = (uint8_t*)ZXC_ALIGNED_MALLOC(total, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem)) return ZXC_ERROR_MEMORY;

    const int rc = zxc_cctx_init_in_workspace(ctx, mem, total, chunk_size, mode, level,
                                              checksum_enabled, dict_size, defer_entropy);
    if (UNLIKELY(rc != ZXC_OK)) {
        // LCOV_EXCL_START
        ZXC_ALIGNED_FREE(mem);
        return rc;
        // LCOV_EXCL_STOP
    }
    // Library-owned buffer: record the allocation so zxc_cctx_free frees it.
    ctx->memory_block = mem;
    return ZXC_OK;
}

/**
 * @brief Lazily allocates the deferred decode-side entropy scratch.
 *
 * Public contract at the declaration in @c zxc_internal.h. One aligned block
 * carries [tok_buffer | pivco_scratch], sized by @ref zxc_dctx_entropy_sizes
 * (the same source the full-provision layout uses), owned by the context and
 * released by @ref zxc_cctx_free.
 */
int zxc_cctx_alloc_entropy_scratch(zxc_cctx_t* ctx) {
    if (LIKELY(ctx->pivco_scratch != NULL)) return ZXC_OK;

    size_t sz_tok = 0;
    size_t sz_pivco = 0;
    zxc_dctx_entropy_sizes(ctx->chunk_size, &sz_tok, &sz_pivco);
    const size_t total = ZXC_ALIGN_CL(sz_tok) + ZXC_ALIGN_CL(sz_pivco);

    uint8_t* const mem = (uint8_t*)ZXC_ALIGNED_MALLOC(total, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE

    ctx->entropy_block = mem;
    ctx->tok_buffer = mem;
    ctx->tok_buffer_cap = sz_tok;
    ctx->pivco_scratch = mem + ZXC_ALIGN_CL(sz_tok);
    ctx->pivco_scratch_cap = sz_pivco;
    return ZXC_OK;
}

/**
 * @brief Releases all resources owned by a compression context.
 *
 * After this call every pointer inside @p ctx is @c NULL and the context
 * may be safely re-initialised with zxc_cctx_init().
 */
void zxc_cctx_free(zxc_cctx_t* ctx) {
    if (ctx->memory_block) {
        ZXC_ALIGNED_FREE(ctx->memory_block);
        ctx->memory_block = NULL;
    }
    if (ctx->entropy_block) {
        ZXC_ALIGNED_FREE(ctx->entropy_block);
        ctx->entropy_block = NULL;
    }

    ctx->lit_buffer = NULL;
    ctx->hash_table = NULL;
    ctx->hash_tags = NULL;
    ctx->chain_table = NULL;
    ctx->buf_sequences = NULL;
    ctx->buf_tokens = NULL;
    ctx->buf_offsets = NULL;
    ctx->buf_extras = NULL;
    ctx->literals = NULL;
    ctx->work_buf = NULL;
    ctx->tok_buffer = NULL;
    ctx->pivco_scratch = NULL;
    ctx->opt_scratch = NULL;
    ctx->dict_buffer = NULL;
    ctx->dict_huf = NULL;

    ctx->epoch = 0;
    ctx->lit_buffer_cap = 0;
    ctx->work_buf_cap = 0;
    ctx->tok_buffer_cap = 0;
    ctx->pivco_scratch_cap = 0;
    ctx->opt_scratch_cap = 0;
    ctx->dict_buffer_cap = 0;
    ctx->dict_size = 0;
    ctx->dict_huf_tree_ok = 0;
    ctx->lit_freq_acc = NULL;
}

/**
 * @brief Attach the shared dictionary literal Huffman table to a context.
 *
 * Validates the 128-byte packed code-lengths header and builds the dictionary's
 * PivCo tree, canonical codes and code lengths ONCE into the context
 * (tree-at-attach); per-block encode/estimate/decode reuse them. @p lengths need
 * only be valid during this call (the tree is a copy). A NULL @p lengths is a no-op.
 *
 * @param[in,out] ctx      Initialised context to attach the table to.
 * @param[in]     lengths  128-byte packed code lengths, or NULL for a no-op.
 * @return @ref ZXC_OK on success, @ref ZXC_ERROR_CORRUPT_DATA if @p lengths is
 *         structurally invalid (bad nibble, Kraft inequality).
 */
int zxc_cctx_attach_dict_huf(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT lengths) {
    if (UNLIKELY(!ctx)) return ZXC_ERROR_NULL_INPUT;
    ctx->dict_huf_tree_ok = 0;
    if (lengths == NULL) return ZXC_OK;

    // Empty (all-zero) table from a low-entropy corpus: treat it as "no shared table".
    int empty = 1;
    for (size_t i = 0; i < ZXC_HUF_TABLE_SIZE; i++) {
        if (lengths[i]) {
            empty = 0;
            break;
        }
    }
    if (UNLIKELY(empty || ctx->dict_huf == NULL)) return ZXC_OK;

    // Tree-at-attach: unpack + build the PivCo tree, codes and decoder tables
    // once here; the per-block encode/estimate/decode paths reuse them via
    // the context.
    const int rc = zxc_huf_dict_tree_build(lengths, &ctx->dict_huf->tree, ctx->dict_huf->codes,
                                           ctx->dict_huf->code_len, &ctx->dict_huf->dec);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    ctx->dict_huf_tree_ok = 1;
    return ZXC_OK;
}

// ============================================================================
// HEADER I/O
// ============================================================================

/**
 * @brief Serialises a ZXC file header into @p dst.
 *
 * Layout (16 bytes): Magic (4) | Version (1) | Chunk (1) | Flags (1) |
 * Reserved (7) | Checksum-16 (2).
 */
int zxc_write_file_header(uint8_t* RESTRICT dst, const size_t dst_capacity, const size_t chunk_size,
                          const int has_checksum, const uint32_t dict_id) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;

    // Block size stored as log2 exponent (e.g. 18 = 256 KB)
    dst[5] = (uint8_t)zxc_log2_u32((uint32_t)chunk_size);

    uint8_t flags = has_checksum ? (ZXC_FILE_FLAG_HAS_CHECKSUM | ZXC_CHECKSUM_RAPIDHASH) : 0;
    if (dict_id != 0) flags |= ZXC_FILE_FLAG_HAS_DICTIONARY;
    dst[6] = flags;

    // Bytes 7-13: Reserved / dict_id
    ZXC_MEMSET(dst + 7, 0, 7);
    if (dict_id != 0) zxc_store_le32(dst + 7, dict_id);

    // Bytes 14-15: checksum (16-bit)
    zxc_store_le16(dst + 14, 0);  // Zero out before hashing
    const uint16_t sum = zxc_hash16(dst);
    zxc_store_le16(dst + 14, sum);

    return ZXC_FILE_HEADER_SIZE;
}

/**
 * @brief Parses and validates a ZXC file header from @p src.
 *
 * Checks the magic word, format version, and 16-bit checksum.
 */
int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size,
                         size_t* RESTRICT out_block_size, int* RESTRICT out_has_checksum,
                         uint32_t* RESTRICT out_dict_id) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (UNLIKELY(zxc_le32(src) != ZXC_MAGIC_WORD)) return ZXC_ERROR_BAD_MAGIC;
    if (UNLIKELY(src[4] != ZXC_FILE_FORMAT_VERSION)) return ZXC_ERROR_BAD_VERSION;

    uint8_t temp[ZXC_FILE_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_FILE_HEADER_SIZE);
    // Zero out checksum bytes (14-15) before hash check
    temp[14] = 0;
    temp[15] = 0;
    // Header checksum (integrity), then the checksum-algorithm id in flags bits 0-3
    // (only 0 = RapidHash is defined). It is checked first via short-circuit.
    if (UNLIKELY(zxc_le16(src + 14) != zxc_hash16(temp) ||
                 (src[6] & ZXC_FILE_CHECKSUM_ALGO_MASK) != ZXC_CHECKSUM_RAPIDHASH))
        return ZXC_ERROR_BAD_HEADER;

    if (out_block_size) {
        const uint8_t code = src[5];
        if (UNLIKELY(code < ZXC_BLOCK_SIZE_MIN_LOG2 || code > ZXC_BLOCK_SIZE_MAX_LOG2))
            return ZXC_ERROR_BAD_BLOCK_SIZE;
        // Exponent encoding: block_size = 2^code  (4 KB - 2 MB)
        *out_block_size = (size_t)1U << code;
    }
    if (out_has_checksum) *out_has_checksum = (src[6] & ZXC_FILE_FLAG_HAS_CHECKSUM) ? 1 : 0;
    if (out_dict_id) *out_dict_id = (src[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? zxc_le32(src + 7) : 0;

    return ZXC_OK;
}

/**
 * @brief Serialises a block header (8 bytes) into @p dst.
 */
int zxc_write_block_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                           const zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(dst_capacity < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    dst[0] = bh->block_type;
    dst[1] = 0;  // Flags not used currently
    dst[2] = 0;  // Reserved
    zxc_store_le32(dst + 3, bh->comp_size);
    dst[7] = 0;               // Zero before hashing
    dst[7] = zxc_hash8(dst);  // Checksum at the end

    return ZXC_BLOCK_HEADER_SIZE;
}

/**
 * @brief Parses and validates a block header from @p src.
 *
 * Validates the 8-bit checksum embedded in the header.
 */
int zxc_read_block_header(const uint8_t* RESTRICT src, const size_t src_size,
                          zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(src_size < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    uint8_t temp[ZXC_BLOCK_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_BLOCK_HEADER_SIZE);
    temp[7] = 0;  // Zero out checksum byte before hashing
    if (UNLIKELY(src[7] != zxc_hash8(temp))) return ZXC_ERROR_BAD_HEADER;

    bh->block_type = src[0];
    bh->block_flags = 0;  // Flags not used currently
    bh->reserved = src[2];
    bh->comp_size = zxc_le32(src + 3);
    bh->header_checksum = src[7];

    return ZXC_OK;
}

/**
 * @brief Writes the 12-byte file footer (source size + global checksum).
 */
int zxc_write_file_footer(uint8_t* RESTRICT dst, const size_t dst_capacity, const uint64_t src_size,
                          const uint32_t global_hash, const int checksum_enabled) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_FOOTER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le64(dst, src_size);

    if (checksum_enabled) {
        zxc_store_le32(dst + sizeof(uint64_t), global_hash);
    } else {
        ZXC_MEMSET(dst + sizeof(uint64_t), 0, sizeof(uint32_t));
    }

    return ZXC_FILE_FOOTER_SIZE;
}

/**
 * @brief Writes the 12-byte GLO/GHI sub-header shared by both block types.
 *
 * @param[out] dst Destination buffer, at least 12 bytes.
 * @param[in]  gh  Populated header descriptor.
 */
static ZXC_ALWAYS_INLINE void zxc_write_gnr_header(uint8_t* RESTRICT dst,
                                                   const zxc_gnr_header_t* RESTRICT gh) {
    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_tok;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;
}

/**
 * @brief Reads the 12-byte GLO/GHI sub-header shared by both block types.
 *
 * @param[in]  src Source buffer, at least 12 bytes.
 * @param[out] gh  Receives the decoded header.
 */
static ZXC_ALWAYS_INLINE void zxc_read_gnr_header(const uint8_t* RESTRICT src,
                                                  zxc_gnr_header_t* RESTRICT gh) {
    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_tok = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];
}

/**
 * @brief Size of the GLO section descriptors, implied by the encoding fields.
 *
 * 0, 4 or 8 bytes - see @ref zxc_write_glo_header_and_desc for what they hold.
 */
static ZXC_ALWAYS_INLINE size_t zxc_glo_desc_size(const uint8_t enc_lit, const uint8_t enc_tok) {
    return ((enc_lit != ZXC_SECTION_ENCODING_RAW) ? sizeof(uint32_t) : 0) +
           ((enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) ? sizeof(uint32_t) : 0);
}

/**
 * @brief Serialises a GLO block header followed by its section descriptors.
 *
 * Only the two sizes the header cannot imply are stored: the literal section's
 * compressed size when it is RLE- or entropy-coded, and the token section's
 * when level 7 Huffman-codes it. The decoder derives the rest - literals raw
 * size from @c n_literals, offsets from @c n_sequences and @c enc_off, extras
 * from the payload residue - so those cannot be forged inconsistently.
 */
int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh, const uint32_t lit_comp,
                                  const uint32_t tok_comp) {
    const size_t desc_sz = zxc_glo_desc_size(gh->enc_lit, gh->enc_tok);
    const size_t needed = ZXC_GLO_HEADER_BINARY_SIZE + desc_sz;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_write_gnr_header(dst, gh);
    uint8_t* p = dst + ZXC_GLO_HEADER_BINARY_SIZE;

    if (gh->enc_lit != ZXC_SECTION_ENCODING_RAW) {
        zxc_store_le32(p, lit_comp);
        p += sizeof(uint32_t);
    }
    if (gh->enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) {
        zxc_store_le32(p, tok_comp);
    }

    return (int)needed;
}

/**
 * @brief Parses a GLO block header and its section descriptors from @p src.
 */
int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh, uint32_t* RESTRICT lit_comp,
                                 uint32_t* RESTRICT tok_comp) {
    if (UNLIKELY(len < ZXC_GLO_HEADER_BINARY_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    zxc_read_gnr_header(src, gh);

    const size_t desc_sz = zxc_glo_desc_size(gh->enc_lit, gh->enc_tok);
    const size_t needed = ZXC_GLO_HEADER_BINARY_SIZE + desc_sz;
    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t* p = src + ZXC_GLO_HEADER_BINARY_SIZE;

    if (gh->enc_lit != ZXC_SECTION_ENCODING_RAW) {
        *lit_comp = zxc_le32(p);
        p += sizeof(uint32_t);
    } else {
        *lit_comp = gh->n_literals;
    }
    *tok_comp = (gh->enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) ? zxc_le32(p) : gh->n_sequences;

    return (int)needed;
}

/**
 * @brief Serialises a GHI block header.
 *
 * GHI carries no section descriptors at all: its literals are always RAW
 * (`lit_comp == gh->n_literals`), its sequence stream is
 * `gh->n_sequences * 4` bytes wide, and its extras run from there to the
 * payload end.
 */
int zxc_write_ghi_header(uint8_t* RESTRICT dst, const size_t rem,
                         const zxc_gnr_header_t* RESTRICT gh) {
    if (UNLIKELY(rem < ZXC_GHI_HEADER_BINARY_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_write_gnr_header(dst, gh);
    return ZXC_GHI_HEADER_BINARY_SIZE;
}

/**
 * @brief Parses a GHI block header from @p src.
 */
int zxc_read_ghi_header(const uint8_t* RESTRICT src, const size_t len,
                        zxc_gnr_header_t* RESTRICT gh) {
    if (UNLIKELY(len < ZXC_GHI_HEADER_BINARY_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    zxc_read_gnr_header(src, gh);
    return ZXC_OK;
}

// ============================================================================
// COMPRESS BOUND CALCULATION
// ============================================================================
/**
 * @brief Returns the maximum compressed size for a given input size.
 *
 * The result accounts for the file header, per-block headers, block
 * checksums, worst-case expansion, EOF block, seekable overhead (SEK
 * block), and the file footer.
 *
 * The block count is derived from @ref ZXC_BLOCK_SIZE_MIN (4 KB) to
 * guarantee the bound holds for all valid block sizes and seekable mode.
 */
uint64_t zxc_compress_bound(const size_t input_size) {
    // Guard against uint64 overflow when summing per-block overhead
    // across very large inputs (input_size approaching SIZE_MAX).
    if (UNLIKELY(input_size > (SIZE_MAX - (SIZE_MAX >> 8)))) return 0;
    uint64_t n = ((uint64_t)input_size + ZXC_BLOCK_SIZE_MIN - 1) / ZXC_BLOCK_SIZE_MIN;
    if (n == 0) n = 1;
    return ZXC_FILE_HEADER_SIZE +
           (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + ZXC_BLOCK_FORMAT_OVERHEAD)) +
           (uint64_t)input_size + ZXC_BLOCK_HEADER_SIZE + /* EOF block */
           ZXC_BLOCK_HEADER_SIZE +                        /* SEK block header (seekable) */
           (n * ZXC_SEEK_ENTRY_SIZE) +                    /* SEK entries: 4 bytes per block */
           ZXC_FILE_FOOTER_SIZE;
}

/**
 * @brief Returns the maximum compressed size for a single block (no file framing).
 */
uint64_t zxc_compress_block_bound(const size_t input_size) {
    // Mirrors the Block API contract: outside [1, ZXC_BLOCK_SIZE_MAX] the call
    // would fail anyway, so the bound is undefined and 0 says "unusable". The cap
    // also makes the addition below trivially overflow-free.
    if (UNLIKELY(input_size == 0 || input_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    // Outer block header + payload (worst case: incompressible, raw bytes)
    // + inner format overhead + optional checksum.
    return (uint64_t)ZXC_BLOCK_HEADER_SIZE + (uint64_t)input_size + ZXC_BLOCK_FORMAT_OVERHEAD +
           ZXC_BLOCK_CHECKSUM_SIZE;
}

/**
 * @brief Returns the minimum dst_capacity required by zxc_decompress_block().
 *
 * The decoder uses speculative wild-copy writes on its fast path.
 * Sizing the destination to uncompressed_size + ZXC_PAD_SIZE*66 guarantees
 * the fast path is always reachable and that tail bounds checks never
 * spuriously reject the last literals of a valid block.
 *
 * Returns 0 if @p uncompressed_size exceeds ZXC_BLOCK_SIZE_MAX (the Block API
 * limit), or if the arithmetic would overflow.
 */
uint64_t zxc_decompress_block_bound(const size_t uncompressed_size) {
    if (UNLIKELY(uncompressed_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    return (uint64_t)uncompressed_size + ZXC_DECOMPRESS_TAIL_PAD;
}

/**
 * @brief Estimates the total buffer bytes allocated inside a cctx for a block.
 *
 * Thin wrapper around @ref zxc_cctx_compute_workspace_size for @c mode == 1
 * (compress), with @c src_size clamped up to a valid block size via
 * @ref zxc_block_size_ceil.  The opaque wrapper struct allocated by
 * @ref zxc_create_cctx adds a fixed overhead (< 128 B) that is negligible
 * next to the per-chunk buffers and is intentionally omitted.
 *
 * For @p level >= 6 the figure includes the optimal-parser scratch
 * (@c opt_scratch, ~8.125 bytes per chunk_size byte) used by the optimal
 * parser and reused as transient package-merge scratch for the Huffman
 * code-length builder.
 */
uint64_t zxc_estimate_cctx_size(const size_t src_size, const int level) {
    if (UNLIKELY(src_size == 0)) return 0;
    const size_t chunk_size = zxc_block_size_ceil(src_size);
    return (uint64_t)zxc_cctx_compute_workspace_size(chunk_size, 1, level, 0);
}

// ============================================================================
// ERROR CODE UTILITIES
// ============================================================================

/**
 * @brief Returns a human-readable string for the given error code.
 */
const char* zxc_error_name(const int code) {
    switch ((zxc_error_t)code) {
        case ZXC_OK:
            return "ZXC_OK";
        case ZXC_ERROR_MEMORY:
            return "ZXC_ERROR_MEMORY";
        case ZXC_ERROR_DST_TOO_SMALL:
            return "ZXC_ERROR_DST_TOO_SMALL";
        case ZXC_ERROR_SRC_TOO_SMALL:
            return "ZXC_ERROR_SRC_TOO_SMALL";
        case ZXC_ERROR_BAD_MAGIC:
            return "ZXC_ERROR_BAD_MAGIC";
        case ZXC_ERROR_BAD_VERSION:
            return "ZXC_ERROR_BAD_VERSION";
        case ZXC_ERROR_BAD_HEADER:
            return "ZXC_ERROR_BAD_HEADER";
        case ZXC_ERROR_BAD_CHECKSUM:
            return "ZXC_ERROR_BAD_CHECKSUM";
        case ZXC_ERROR_CORRUPT_DATA:
            return "ZXC_ERROR_CORRUPT_DATA";
        case ZXC_ERROR_BAD_OFFSET:
            return "ZXC_ERROR_BAD_OFFSET";
        case ZXC_ERROR_OVERFLOW:
            return "ZXC_ERROR_OVERFLOW";
        case ZXC_ERROR_IO:
            return "ZXC_ERROR_IO";
        case ZXC_ERROR_NULL_INPUT:
            return "ZXC_ERROR_NULL_INPUT";
        case ZXC_ERROR_BAD_BLOCK_TYPE:
            return "ZXC_ERROR_BAD_BLOCK_TYPE";
        case ZXC_ERROR_BAD_BLOCK_SIZE:
            return "ZXC_ERROR_BAD_BLOCK_SIZE";
        case ZXC_ERROR_DICT_REQUIRED:
            return "ZXC_ERROR_DICT_REQUIRED";
        case ZXC_ERROR_DICT_MISMATCH:
            return "ZXC_ERROR_DICT_MISMATCH";
        case ZXC_ERROR_DICT_TOO_LARGE:
            return "ZXC_ERROR_DICT_TOO_LARGE";
        case ZXC_ERROR_BAD_LEVEL:
            return "ZXC_ERROR_BAD_LEVEL";
        default:
            return "ZXC_UNKNOWN_ERROR";
    }
}

// ============================================================================
// LIBRARY INFORMATION
// ============================================================================

/**
 * @brief Returns the minimum supported compression level.
 *
 * Returns the value of ZXC_LEVEL_FASTEST (currently 1).
 * This allows integrators to discover the level range at runtime without relying on
 * compile-time macros alone.
 */
int zxc_min_level(void) { return ZXC_LEVEL_FASTEST; }

/**
 * @brief Returns the maximum supported compression level.
 *
 * Returns the value of ZXC_LEVEL_ULTRA (currently 7).
 */
int zxc_max_level(void) { return ZXC_LEVEL_ULTRA; }

/**
 * @brief Returns the default compression level.
 *
 * Returns the value of ZXC_LEVEL_DEFAULT (currently 3).
 */
int zxc_default_level(void) { return ZXC_LEVEL_DEFAULT; }

/**
 * @brief Returns the human-readable library version string.
 *
 * The returned pointer is a compile-time constant and must not be freed.
 * Format: "MAJOR.MINOR.PATCH" (e.g. "0.13.1").
 */
const char* zxc_version_string(void) { return ZXC_LIB_VERSION_STR; }
