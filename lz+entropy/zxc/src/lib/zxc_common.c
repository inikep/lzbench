// SPDX-License-Identifier: BSD-3-Clause
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
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
 * @brief Returns @c sizeof(zxc_frame_info_t) for ABI-safe mirroring.
 *
 * Public API; see @c zxc_buffer.h.
 */
// cppcheck-suppress unusedFunction
size_t zxc_frame_info_size(void) { return sizeof(zxc_frame_info_t); }

// Offset table of the persistent buffer carved by every cctx/dctx init. Both
// modes compute it identically, for the workspace sizer and the in-place init.
typedef struct {
    size_t total;
    // mode == 0 (decompress)
    size_t off_work;
    size_t off_lit_dctx;
    // mode == 0: scratch for a Huffman-coded SPL token section (enc_tok == HUFFMAN).
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
    size_t off_split_hist;
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
 * tables (hash positions, tags, chain), the sequence buffers, the match-split
 * histograms, the extras and literal buffers and - only at @c level >=
 * ZXC_LEVEL_DENSITY - the optimal-parser scratch. A
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
        // Token-section decode scratch (level-7 SPL Huffman-codes the tokens) +
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
        // Compress: 7 partitions + optional opt_scratch at level >= ZXC_LEVEL_DENSITY.
        const uint32_t offset_bits = zxc_log2_u32((uint32_t)chunk_size);
        layout.max_seq = max_seq;
        layout.sz_hash_pos = ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
        layout.sz_hash_tags = ZXC_LZ_HASH_SIZE * sizeof(uint8_t);
        const size_t sz_chain = ZXC_LZ_WINDOW_SIZE * sizeof(uint16_t);
        // buf_sequences (PAK, level <= ZXC_LEVEL_FAST) aliases buf_offsets, buf_tokens and
        // buf_split (SPL, level >= ZXC_LEVEL_DEFAULT): 4 bytes per sequence either way.
        const size_t sz_seq_union = layout.max_seq * sizeof(uint32_t);
        const size_t vbyte_len = (offset_bits + 6) / 7;
        const size_t sz_extras = layout.max_seq * 2 * vbyte_len;
        const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

        // opt_scratch (level >= ZXC_LEVEL_DENSITY): the optimal parser's DP arrays,
        // reused transiently by the code-length builder and its nudge, so sized to
        // the larger demand. Keep in sync with zxc_estimate_cctx_size() and its
        // consumer in zxc_compress.c.
        if (level >= ZXC_LEVEL_DENSITY) {
            size_t sz_dp;
            size_t sz_pl;
            size_t sz_po;
            size_t sz_bm;
            zxc_opt_dp_sizes(chunk_size, &sz_dp, &sz_pl, &sz_po, &sz_bm);
            const size_t dp_needed = sz_dp + sz_pl + sz_po + sz_bm;
            layout.sz_opt =
                (dp_needed > ZXC_HUF_NUDGE_SCRATCH_SIZE) ? dp_needed : ZXC_HUF_NUDGE_SCRATCH_SIZE;
        }

        layout.off_hash_pos = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_pos);
        layout.off_hash_tags = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_tags);
        layout.off_chain = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_chain);
        layout.off_seq_union = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_seq_union);
        layout.off_split_hist = layout.total;
        layout.total += ZXC_ALIGN_CL(sizeof(zxc_spl_match_split_hist_t));
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
    ctx->buf_split = ctx->buf_tokens + layout.max_seq;
    ctx->buf_split_hist = (zxc_spl_match_split_hist_t*)(void*)(mem + layout.off_split_hist);
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
    if (LIKELY(ctx->pivco_scratch)) return ZXC_OK;

    // Sized for the block the context was carved for, not chunk_size: a reader
    // over frames of several block sizes lowers chunk_size per block.
    size_t sz_tok = 0;
    size_t sz_pivco = 0;
    zxc_dctx_entropy_sizes(ctx->work_buf_cap - ZXC_DECOMPRESS_TAIL_PAD, &sz_tok, &sz_pivco);
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
    ctx->buf_split = NULL;
    ctx->buf_split_hist = NULL;
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
    if (!lengths) return ZXC_OK;

    // Empty (all-zero) table from a low-entropy corpus: treat it as "no shared table".
    int empty = 1;
    for (size_t i = 0; i < ZXC_HUF_TABLE_SIZE; i++) {
        if (lengths[i]) {
            empty = 0;
            break;
        }
    }
    if (UNLIKELY(empty || !ctx->dict_huf)) return ZXC_OK;

    // Tree-at-attach: built once, reused by every block.
    const int rc = zxc_huf_dict_tree_build(lengths, &ctx->dict_huf->tree, ctx->dict_huf->codes,
                                           ctx->dict_huf->code_len);
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
                          const int has_checksum, const uint32_t dict_id, const int has_seek) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;

    // Block size stored as log2 exponent (e.g. 18 = 256 KB)
    dst[5] = (uint8_t)zxc_log2_u32((uint32_t)chunk_size);

    uint8_t flags = has_checksum ? (ZXC_FILE_FLAG_HAS_CHECKSUM | ZXC_CHECKSUM_RAPIDHASH) : 0;
    if (dict_id != 0) flags |= ZXC_FILE_FLAG_HAS_DICTIONARY;
    if (has_seek) flags |= ZXC_FILE_FLAG_HAS_SEEK_TABLE;
    dst[6] = flags;

    // Bytes 7-13: Reserved / Dictionary ID
    ZXC_MEMSET(dst + 7, 0, 7);
    if (dict_id != 0) zxc_store_le32(dst + 7, dict_id);

    // Bytes 14-15: Header Checksum (16-bit)
    zxc_file_header_sign(dst);

    return ZXC_FILE_HEADER_SIZE;
}

/**
 * @brief Parses and validates a ZXC file header from @p src.
 *
 * Checks the magic word, format version, 16-bit checksum and reserved bits/bytes.
 */
int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size,
                         size_t* RESTRICT out_block_size, int* RESTRICT out_has_checksum,
                         uint32_t* RESTRICT out_dict_id, int* RESTRICT out_has_seek) {
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
    // Reserved: flag bit 4, bytes 7-13 minus the dict_id (7-10).
    if (UNLIKELY(src[6] & ZXC_FILE_FLAG_RESERVED)) return ZXC_ERROR_BAD_HEADER;
    uint8_t rsv = 0;
    for (size_t i = (src[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? 11 : 7; i < 14; i++) rsv |= src[i];
    if (UNLIKELY(rsv != 0)) return ZXC_ERROR_BAD_HEADER;

    if (out_block_size) {
        const uint8_t code = src[5];
        if (UNLIKELY(code < ZXC_BLOCK_SIZE_MIN_LOG2 || code > ZXC_BLOCK_SIZE_MAX_LOG2))
            return ZXC_ERROR_BAD_BLOCK_SIZE;
        // Exponent encoding: block_size = 2^code  (4 KB - 2 MB)
        *out_block_size = (size_t)1U << code;
    }
    if (out_has_checksum) *out_has_checksum = (src[6] & ZXC_FILE_FLAG_HAS_CHECKSUM) ? 1 : 0;
    if (out_dict_id) *out_dict_id = (src[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? zxc_le32(src + 7) : 0;
    if (out_has_seek) *out_has_seek = (src[6] & ZXC_FILE_FLAG_HAS_SEEK_TABLE) ? 1 : 0;

    return ZXC_OK;
}

/** @brief Reads a byte range of a scan source, refusing ranges past its end. */
static int zxc_scan_read(const zxc_scan_src_t* src, const uint64_t off, void* dst,
                         const size_t len) {
    if (UNLIKELY(off > src->size || (uint64_t)len > src->size - off))
        return ZXC_ERROR_SRC_TOO_SMALL;
    if (src->read_at) return src->read_at(src, off, dst, len);
    ZXC_MEMCPY(dst, src->data + off, len);
    return ZXC_OK;
}

int zxc_container_next(const zxc_scan_src_t* src, const uint64_t pos) {
    if (pos == src->size) return 0;
    uint8_t m[sizeof(uint32_t)];
    if (src->size - pos >= sizeof(m)) {
        const int rc = zxc_scan_read(src, pos, m, sizeof(m));
        if (UNLIKELY(rc != ZXC_OK)) return rc;
        if (zxc_magic_prefix_ok(m, sizeof(m))) return 1;
    }
    return pos == 0 ? ZXC_ERROR_BAD_MAGIC : ZXC_ERROR_CORRUPT_DATA;
}

int zxc_scan_frame(const zxc_scan_src_t* src, const uint64_t end, uint64_t* start,
                   zxc_frame_info_t* info, size_t* footer_len) {
    uint8_t h[ZXC_FILE_HEADER_SIZE];
    int rc;
    // Too short for a frame: what is left starts the input, its header speaks first.
    if (UNLIKELY(end < ZXC_FRAME_MIN_SIZE)) {
        const size_t n = end < sizeof(h) ? (size_t)end : sizeof(h);
        rc = zxc_scan_read(src, 0, h, n);
        return rc != ZXC_OK ? rc : zxc_short_input_verdict(h, n);
    }
    uint8_t tail[ZXC_FOOTER_MAX_SIZE_WITH_DIGEST];  // end > sizeof(tail)
    rc = zxc_scan_read(src, end - sizeof(tail), tail, sizeof(tail));
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    uint64_t dsize = 0;
    uint64_t frame = 0;
    size_t sizes = 0;
    rc = zxc_parse_file_footer(tail + sizeof(tail), sizeof(tail), &dsize, &frame, &sizes);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(frame < ZXC_FRAME_MIN_SIZE || frame > end)) return ZXC_ERROR_CORRUPT_DATA;

    const uint64_t at = end - frame;
    rc = zxc_scan_read(src, at, h, sizeof(h));
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    const size_t tl = zxc_frame_tail_len(frame);
    rc = zxc_read_frame_info(h, tail + sizeof(tail) - tl, tl, frame, info, footer_len);
    // Past the first frame, a wrong magic word is a broken container.
    if (UNLIKELY(rc == ZXC_ERROR_BAD_MAGIC && at != 0)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    *start = at;
    return ZXC_OK;
}

int zxc_scan_whole_frame(const zxc_scan_src_t* src, zxc_frame_info_t* info, size_t* footer_len) {
    uint8_t h[ZXC_FILE_HEADER_SIZE];
    const size_t n = src->size < sizeof(h) ? (size_t)src->size : sizeof(h);
    int rc = zxc_scan_read(src, 0, h, n);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    rc = zxc_short_input_verdict(h, n);  // the header speaks first
    if (UNLIKELY(rc != ZXC_ERROR_SRC_TOO_SMALL || src->size < ZXC_FRAME_MIN_SIZE)) return rc;
    uint64_t start = 0;
    rc = zxc_scan_frame(src, src->size, &start, info, footer_len);
    return rc == ZXC_OK && start != 0 ? ZXC_ERROR_CORRUPT_DATA : rc;
}

int zxc_scan_container(const zxc_scan_src_t* src, const size_t req_chunk,
                       zxc_container_info_t* info) {
    ZXC_MEMSET(info, 0, sizeof(*info));
    // The first header speaks first, as for a single frame: junk is a bad magic word.
    uint8_t h[ZXC_FILE_HEADER_SIZE];
    const size_t n = src->size < sizeof(h) ? (size_t)src->size : sizeof(h);
    size_t chunk = 0;
    int hrc = zxc_scan_read(src, 0, h, n);
    if (hrc == ZXC_OK) hrc = zxc_read_file_header(h, n, &chunk, NULL, NULL, NULL);
    if (UNLIKELY(hrc != ZXC_OK)) return hrc;

    uint64_t end = src->size;
    do {
        zxc_frame_info_t fi;
        uint64_t start = 0;
        const int rc = zxc_scan_frame(src, end, &start, &fi, NULL);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
        if (UNLIKELY(req_chunk && fi.block_size != req_chunk)) return ZXC_ERROR_BAD_BLOCK_SIZE;
        if (UNLIKELY(fi.decompressed_size > UINT64_MAX - info->dsize))
            return ZXC_ERROR_CORRUPT_DATA;
        info->dsize += fi.decompressed_size;
        info->frames++;
        info->first_dict_id = fi.dict_id;  // walking back, the last one seen is first
        if (fi.dict_id) info->uses_dict = 1;
        end = start;
    } while (end > 0);
    return ZXC_OK;
}

/**
 * @brief Validates a frame from its header and the end of its bytes.
 *
 * Header first, as every decoder does; then the footer, whose frame must span
 * @p total bytes and whose size the blocks could hold.
 */
int zxc_read_frame_info(const uint8_t* header, const uint8_t* tail, const size_t tail_len,
                        const uint64_t total, zxc_frame_info_t* info, size_t* footer_len) {
    size_t chunk = 0;
    int cs = 0;
    int seek = 0;
    uint32_t did = 0;
    const int hrc = zxc_read_file_header(header, ZXC_FILE_HEADER_SIZE, &chunk, &cs, &did, &seek);
    if (UNLIKELY(hrc != ZXC_OK)) return hrc;

    uint64_t stored = 0;
    uint64_t frame = 0;
    size_t sizes = 0;
    const int frc = zxc_parse_file_footer(tail + tail_len, tail_len, &stored, &frame, &sizes);
    if (UNLIKELY(frc != ZXC_OK)) return frc;
    const size_t flen = sizes + (cs ? (size_t)ZXC_FILE_DIGEST_SIZE : 0U);
    // All the frame's bytes are there: a footer longer than they allow is forged.
    if (UNLIKELY(flen > tail_len)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(frame != total || !zxc_footer_dsize_plausible(stored, chunk, total)))
        return ZXC_ERROR_CORRUPT_DATA;

    ZXC_MEMSET(info, 0, sizeof(*info));  // padding too: zxc_frame_info_copy() hands it out
    info->decompressed_size = stored;
    info->compressed_size = frame;
    info->digest = cs ? zxc_le64(tail + tail_len - flen) : 0;
    info->block_size = chunk;
    info->dict_id = did;
    info->format_version = header[4];
    info->has_checksum = (uint8_t)cs;
    info->has_seek_table = (uint8_t)seek;
    if (footer_len) *footer_len = flen;
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
 * Validates the 8-bit checksum, then requires Block Flags and Reserved to be 0.
 */
int zxc_read_block_header(const uint8_t* RESTRICT src, const size_t src_size,
                          zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(src_size < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    uint8_t temp[ZXC_BLOCK_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_BLOCK_HEADER_SIZE);
    temp[7] = 0;  // Zero out checksum byte before hashing
    if (UNLIKELY(src[7] != zxc_hash8(temp))) return ZXC_ERROR_BAD_HEADER;
    if (UNLIKELY((src[1] | src[2]) != 0)) return ZXC_ERROR_BAD_HEADER;

    bh->block_type = src[0];
    bh->block_flags = src[1];
    bh->reserved = src[2];
    bh->comp_size = zxc_le32(src + 3);
    bh->header_checksum = src[7];

    return ZXC_OK;
}

// =========================================================================
// SEEK TABLE WRITER (a frame block; zxc_seekable.c is the random-access reader)
// =========================================================================

/**
 * @brief Byte size of a seek table holding @p num_blocks blocks: header, then groups.
 *
 * Public API; sizes the destination of @ref zxc_write_seek_table. 0 when the table
 * does not fit size_t (32-bit hosts) or @p num_blocks is past what one can describe.
 */
size_t zxc_seek_table_size(const uint64_t num_blocks) {
    if (UNLIKELY(num_blocks > UINT64_MAX / ZXC_SEEK_ANCHOR_SIZE)) return 0;
    const uint64_t bytes = zxc_seek_table_bytes(num_blocks);
    if (UNLIKELY(bytes > SIZE_MAX - ZXC_BLOCK_HEADER_SIZE)) return 0;
    return ZXC_BLOCK_HEADER_SIZE + (size_t)bytes;
}

int zxc_seek_table_header(uint8_t* dst, const size_t dst_capacity, const uint64_t num_blocks) {
    // The field keeps the table size modulo 2^32; readers derive the count from the footer.
    const zxc_block_header_t bh = {
        .block_type = ZXC_BLOCK_SEK,
        .block_flags = 0,
        .reserved = 0,
        .comp_size = zxc_seek_size_field(zxc_seek_table_bytes(num_blocks))};
    return zxc_write_block_header(dst, dst_capacity, &bh);
}

int zxc_check_eof_header(const uint8_t* hdr) {
    const zxc_block_header_t eof = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    uint8_t want[ZXC_BLOCK_HEADER_SIZE];
    if (UNLIKELY(zxc_write_block_header(want, sizeof(want), &eof) < 0))
        return ZXC_ERROR_BAD_HEADER;  // LCOV_EXCL_LINE
    return memcmp(hdr, want, sizeof(want)) == 0 ? ZXC_OK : ZXC_ERROR_BAD_HEADER;
}

int zxc_check_seek_header(const uint8_t* hdr, const size_t avail, const uint64_t total_out,
                          const size_t block_size, uint64_t* sek_bytes) {
    const uint64_t n = zxc_seek_block_count(total_out, block_size);
    uint8_t want[ZXC_BLOCK_HEADER_SIZE];
    if (UNLIKELY(zxc_seek_table_header(want, sizeof(want), n) < 0))
        return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE
    const size_t m = avail < sizeof(want) ? avail : sizeof(want);
    if (memcmp(hdr, want, m) != 0) return ZXC_ERROR_CORRUPT_DATA;
    if (avail < sizeof(want)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (sek_bytes) *sek_bytes = zxc_seek_table_bytes(n);
    return ZXC_OK;
}

size_t zxc_seek_write_group(uint8_t* RESTRICT dst, uint64_t* RESTRICT anchor,
                            const uint32_t* RESTRICT sizes, const uint32_t cnt) {
    zxc_store_le64(dst, *anchor);
    uint8_t* p = dst + ZXC_SEEK_ANCHOR_SIZE;
    for (uint32_t k = 0; k < cnt; k++, p += ZXC_SEEK_SIZE_ENTRY) {
        zxc_store_le32(p, sizes[k]);
        *anchor += sizes[k];
    }
    return (size_t)(p - dst);
}

/**
 * @brief Serialises a seek table (a @c ZXC_BLOCK_SEK block) into @p dst.
 *
 * Public API; contract in @c zxc_seekable.h. Block header, then the groups.
 */
int64_t zxc_write_seek_table(uint8_t* dst, const size_t dst_capacity, const uint32_t* comp_sizes,
                             const uint64_t num_blocks) {
    const size_t total = zxc_seek_table_size(num_blocks);
    if (UNLIKELY(total == 0)) return ZXC_ERROR_OVERFLOW;
    if (UNLIKELY(dst_capacity < total)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(!dst || !comp_sizes)) return ZXC_ERROR_NULL_INPUT;

    const int hdr_res = zxc_seek_table_header(dst, dst_capacity, num_blocks);
    if (UNLIKELY(hdr_res < 0)) return hdr_res;
    uint8_t* p = dst + hdr_res;

    uint64_t anchor = ZXC_FILE_HEADER_SIZE;
    for (uint64_t i = 0; i < num_blocks; i += ZXC_SEEK_GROUP)
        p += zxc_seek_write_group(p, &anchor, comp_sizes + i,
                                  zxc_seek_group_len(num_blocks, i / ZXC_SEEK_GROUP));

    return (int64_t)(p - dst);
}

/** @brief Stores the low @p n bytes of @p v, little-endian. */
static void zxc_store_le_n(uint8_t* dst, uint64_t v, const size_t n) {
    for (size_t i = 0; i < n; i++, v >>= 8) dst[i] = (uint8_t)v;
}

/** @brief Loads @p n little-endian bytes. */
static uint64_t zxc_load_le_n(const uint8_t* src, const size_t n) {
    uint64_t v = 0;
    for (size_t i = n; i > 0; i--) v = (v << 8) | src[i - 1];
    return v;
}

/** @brief Encodes the @p l footer: digest, the two sizes, then L. */
static void zxc_encode_footer(uint8_t* dst, const zxc_footer_layout_t* l, const uint64_t src_size,
                              const uint64_t digest) {
    if (l->skip) zxc_store_le64(dst, digest);
    uint8_t* p = dst + l->skip;
    zxc_store_le_n(p, src_size, l->nd);
    p += l->nd;
    zxc_store_le_n(p, l->compressed_size, l->nf);
    p[l->nf] = (uint8_t)((l->nd - 1) | (l->nf - 1) << 4);
}

/**
 * @brief Writes the file footer: the archive digest when @p checksum_enabled,
 *        then the two sizes and their lengths byte.
 */
int zxc_write_file_footer(uint8_t* RESTRICT dst, const size_t dst_capacity, const uint64_t prefix,
                          const uint64_t src_size, const uint64_t digest,
                          const int checksum_enabled) {
    const zxc_footer_layout_t l = zxc_footer_layout(prefix, src_size, checksum_enabled);
    if (UNLIKELY(dst_capacity < l.len)) return ZXC_ERROR_DST_TOO_SMALL;
    zxc_encode_footer(dst, &l, src_size, digest);
    return (int)l.len;
}

/**
 * @brief Checks a footer against the one the frame implies.
 *
 * Compares the bytes present, digest aside, with the footer it would write: one
 * encoding per frame.
 */
int zxc_check_file_footer(const uint8_t* footer, const size_t avail, const zxc_footer_layout_t* l,
                          const uint64_t src_size, uint64_t* digest) {
    uint8_t want[ZXC_FOOTER_MAX_SIZE_WITH_DIGEST];
    zxc_encode_footer(want, l, src_size, 0);
    const size_t n = avail < l->len ? avail : l->len;
    if (n > l->skip && memcmp(footer + l->skip, want + l->skip, n - l->skip) != 0)
        return ZXC_ERROR_CORRUPT_DATA;
    if (avail < l->len) return ZXC_ERROR_SRC_TOO_SMALL;
    if (digest) *digest = l->skip ? zxc_le64(footer) : 0;
    return ZXC_OK;
}

/**
 * @brief Parses the two sizes back from the end of a frame.
 *
 * Bits 3 and 7 of L are reserved; a length longer than its value needs is
 * refused.
 */
int zxc_parse_file_footer(const uint8_t* end, const size_t avail, uint64_t* src_size,
                          uint64_t* compressed_size, size_t* sizes_len) {
    if (UNLIKELY(avail < ZXC_FILE_FOOTER_MIN_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;
    const uint8_t L = end[-1];
    if (UNLIKELY(L & 0x88U)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t nd = (size_t)(L & 0x07U) + 1;
    const size_t nf = (size_t)(L >> 4) + 1;
    const size_t need = nd + nf + 1;
    if (UNLIKELY(avail < need)) return ZXC_ERROR_SRC_TOO_SMALL;
    const uint64_t d = zxc_load_le_n(end - need, nd);
    const uint64_t f = zxc_load_le_n(end - 1 - nf, nf);
    // Minimal lengths: d on exactly nd bytes; f too, and nf the fewest that hold
    // the total, which one byte less would not (f counts its own bytes).
    if (UNLIKELY(zxc_uint_bytes(d) != nd || zxc_uint_bytes(f) != nf ||
                 (nf > 1 && zxc_uint_bytes(f - 1) < nf)))
        return ZXC_ERROR_CORRUPT_DATA;
    *src_size = d;
    *compressed_size = f;
    *sizes_len = need;
    return ZXC_OK;
}

/**
 * @brief Writes the 12-byte SPL/PAK sub-header shared by both block types.
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
 * @brief Reads the 12-byte SPL/PAK sub-header shared by both block types.
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
 * @brief Size of the SPL section descriptors, implied by the encoding fields.
 *
 * 0, 4 or 8 bytes (@ref ZXC_SPL_DESC_MAX_SIZE) - see @ref zxc_write_spl_header_and_desc
 * for what they hold.
 */
static ZXC_ALWAYS_INLINE size_t zxc_spl_desc_size(const uint8_t enc_lit, const uint8_t enc_tok) {
    return ((enc_lit != ZXC_SECTION_ENCODING_RAW) ? sizeof(uint32_t) : 0) +
           ((enc_tok == ZXC_SECTION_ENCODING_HUFFMAN) ? sizeof(uint32_t) : 0);
}

/**
 * @brief Serialises a SPL block header followed by its section descriptors.
 *
 * Only the two sizes the header cannot imply are stored: the literal section's
 * compressed size when it is RLE- or entropy-coded, and the token section's
 * when level 7 Huffman-codes it. The decoder derives the rest - literals raw
 * size from @c n_literals, offsets from @c n_sequences and @c enc_off, extras
 * from the payload residue - so those cannot be forged inconsistently.
 */
int zxc_write_spl_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh, const uint32_t lit_comp,
                                  const uint32_t tok_comp) {
    const size_t desc_sz = zxc_spl_desc_size(gh->enc_lit, gh->enc_tok);
    const size_t needed = ZXC_SPL_HEADER_BINARY_SIZE + desc_sz;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_write_gnr_header(dst, gh);
    uint8_t* p = dst + ZXC_SPL_HEADER_BINARY_SIZE;

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
 * @brief Parses a SPL block header and its section descriptors from @p src.
 */
int zxc_read_spl_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh, uint32_t* RESTRICT lit_comp,
                                 uint32_t* RESTRICT tok_comp) {
    if (UNLIKELY(len < ZXC_SPL_HEADER_BINARY_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    zxc_read_gnr_header(src, gh);

    const size_t desc_sz = zxc_spl_desc_size(gh->enc_lit, gh->enc_tok);
    const size_t needed = ZXC_SPL_HEADER_BINARY_SIZE + desc_sz;
    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t* p = src + ZXC_SPL_HEADER_BINARY_SIZE;

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
 * @brief Serialises a PAK block header.
 *
 * PAK carries no section descriptors at all: its literals are always RAW
 * (`lit_comp == gh->n_literals`), its sequence stream is
 * `gh->n_sequences * 4` bytes wide, and its extras run from there to the
 * payload end.
 */
int zxc_write_pak_header(uint8_t* RESTRICT dst, const size_t rem,
                         const zxc_gnr_header_t* RESTRICT gh) {
    if (UNLIKELY(rem < ZXC_PAK_HEADER_BINARY_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_write_gnr_header(dst, gh);
    return ZXC_PAK_HEADER_BINARY_SIZE;
}

/**
 * @brief Parses a PAK block header from @p src.
 */
int zxc_read_pak_header(const uint8_t* RESTRICT src, const size_t len,
                        zxc_gnr_header_t* RESTRICT gh) {
    if (UNLIKELY(len < ZXC_PAK_HEADER_BINARY_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

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
size_t zxc_compress_bound(const size_t input_size) {
    // n <= 2^52 even for a 64-bit size_t, so the overhead sum fits a uint64.
    uint64_t n = (uint64_t)input_size / ZXC_BLOCK_SIZE_MIN + (input_size % ZXC_BLOCK_SIZE_MIN != 0);
    if (n == 0) n = 1;
    const uint64_t overhead =
        ZXC_FILE_HEADER_SIZE +
        (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + ZXC_BLOCK_FORMAT_OVERHEAD)) +
        ZXC_BLOCK_HEADER_SIZE +          /* EOF block */
        ZXC_BLOCK_HEADER_SIZE +          /* SEK block header (seekable) */
        zxc_seek_table_bytes(n) +        /* SEK groups (seekable) */
        ZXC_FOOTER_MAX_SIZE_WITH_DIGEST; /* optional digest + footer */
    // The bound is a buffer size: input + overhead must fit a size_t.
    if (UNLIKELY(overhead > (uint64_t)(SIZE_MAX - input_size))) return 0;
    return input_size + (size_t)overhead;
}

/**
 * @brief Returns the maximum compressed size for a single block (no file framing).
 */
size_t zxc_compress_block_bound(const size_t input_size) {
    // Mirrors the Block API contract: outside [1, ZXC_BLOCK_SIZE_MAX] the call
    // would fail anyway, so the bound is undefined and 0 says "unusable". The cap
    // also makes the addition below trivially overflow-free.
    if (UNLIKELY(input_size == 0 || input_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    // Outer block header + payload (worst case: incompressible, raw bytes)
    // + inner format overhead + optional checksum.
    return ZXC_BLOCK_HEADER_SIZE + input_size + ZXC_BLOCK_FORMAT_OVERHEAD + ZXC_BLOCK_CHECKSUM_SIZE;
}

/**
 * @brief Returns the minimum dst_capacity required by zxc_decompress_block().
 *
 * The decoder uses speculative wild-copy writes on its fast path.
 * Sizing the destination to uncompressed_size + ZXC_DECOMPRESS_TAIL_PAD guarantees
 * the fast path is always reachable and that tail bounds checks never
 * spuriously reject the last literals of a valid block.
 *
 * Returns 0 if @p uncompressed_size exceeds ZXC_BLOCK_SIZE_MAX (the Block API
 * limit), or if the arithmetic would overflow.
 */
size_t zxc_decompress_block_bound(const size_t uncompressed_size) {
    if (UNLIKELY(uncompressed_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    return uncompressed_size + ZXC_DECOMPRESS_TAIL_PAD;
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
size_t zxc_estimate_cctx_size(const size_t src_size, const int level) {
    if (UNLIKELY(src_size == 0)) return 0;
    const size_t chunk_size = zxc_block_size_ceil(src_size);
    return zxc_cctx_compute_workspace_size(chunk_size, 1, level, 0);
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
        case ZXC_ERROR_DICT_UNSUPPORTED:
            return "ZXC_ERROR_DICT_UNSUPPORTED";
        case ZXC_ERROR_BAD_PARAM:
            return "ZXC_ERROR_BAD_PARAM";
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
 * Format: "MAJOR.MINOR.PATCH".
 */
const char* zxc_version_string(void) { return ZXC_LIB_VERSION_STR; }

unsigned zxc_version_number(void) { return ZXC_VERSION_NUMBER; }

const char* zxc_version_codename(void) { return ZXC_VERSION_CODENAME; }
