/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_seekable.c
 * @brief Seekable archive reader (random-access decompression) and seek table writer.
 *
 * The seek table is a standard ZXC block (type = ZXC_BLOCK_SEK) appended
 * between the EOF block and the file footer.  It records the compressed size
 * of every block (decompressed sizes are derived from the header's block_size),
 * enabling O(1) lookup + O(block_size) decompression for any byte range.
 *
 * On-disk layout of a SEK block:
 *
 *   [Block Header (8B)]   block_type=SEK, block_flags=0, comp_size=N*4
 *   [N x Entry (4B)]      comp_size(u32 LE) per block
 *
 * Detection from end of file:
 *   1. Read file header (first 16 bytes) => block_size
 *   2. Read file footer (last 12 bytes) => total_decompressed_size
 *   3. Derive num_blocks = ceil(total_decomp / block_size)
 *   4. Compute seek block size, read backward to the block header
 *   5. Validate block_type == ZXC_BLOCK_SEK
 */

#include "../../include/zxc_seekable.h"

#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"
#include "zxc_threads.h"

// =========================================================================
// Seek Table Writer
// =========================================================================

/**
 * @brief Byte size of a seek table holding @p num_blocks entries.
 *
 * Public API (declared in @c zxc_seekable.h): one block header plus
 * @p num_blocks fixed-size entries. Use it to size the destination buffer
 * before @ref zxc_write_seek_table.
 */
size_t zxc_seek_table_size(const uint32_t num_blocks) {
    return ZXC_BLOCK_HEADER_SIZE + (size_t)num_blocks * ZXC_SEEK_ENTRY_SIZE;
}

/**
 * @brief Serialises a seek table (a @c ZXC_BLOCK_SEK block) into @p dst.
 *
 * Public API; full contract in @c zxc_seekable.h. Emits the standard ZXC block
 * header followed by one little-endian @c u32 compressed-size entry per block.
 */
int64_t zxc_write_seek_table(uint8_t* dst, const size_t dst_capacity, const uint32_t* comp_sizes,
                             const uint32_t num_blocks) {
    if (UNLIKELY(num_blocks > UINT32_MAX / ZXC_SEEK_ENTRY_SIZE)) return ZXC_ERROR_OVERFLOW;

    const size_t total = zxc_seek_table_size(num_blocks);
    if (UNLIKELY(dst_capacity < total)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(!dst || !comp_sizes)) return ZXC_ERROR_NULL_INPUT;

    const uint32_t payload_size = num_blocks * ZXC_SEEK_ENTRY_SIZE;

    // Write standard ZXC block header
    const zxc_block_header_t bh = {
        .block_type = ZXC_BLOCK_SEK, .block_flags = 0, .reserved = 0, .comp_size = payload_size};
    const int hdr_res = zxc_write_block_header(dst, dst_capacity, &bh);
    if (UNLIKELY(hdr_res < 0)) return hdr_res;
    uint8_t* p = dst + hdr_res;

    // Write entries: comp_size(4) only
    for (uint32_t i = 0; i < num_blocks; i++) {
        zxc_store_le32(p, comp_sizes[i]);
        p += sizeof(uint32_t);
    }

    return (int64_t)(p - dst);
}

// =========================================================================
// Seekable Reader (Opaque Handle)
// =========================================================================

struct zxc_seekable_s {
    // Source - exactly one of {src, reader.read_at} is set. The FILE* variant
    // wraps pread() in its own reader ctx, indistinguishable from here.
    const uint8_t* src;
    uint64_t src_size;
    zxc_reader_t reader; /* user-supplied callback reader; read_at == NULL when unused */

    // Reader context owned by the handle and freed in zxc_seekable_free, set by
    // the thin wrappers. NULL when the caller owns reader.ctx itself.
    void* owned_reader_ctx;

    // Parsed seek table
    uint32_t num_blocks;
    uint32_t* comp_sizes;   /* array[num_blocks] */
    uint64_t* comp_offsets; /* prefix-sum: byte offset in compressed file per block */
    uint64_t total_decomp;  /* total decompressed size (from footer) */
    uint32_t max_comp_size; /* largest entry of comp_sizes, from the same walk */

    // File header info - block_size is always a power of 2 in [4KB, 2MB],
    // fits in 21 bits.
    uint32_t block_size;
    int file_has_checksums;
    uint32_t expected_dict_id; /* dict_id from the file header; 0 = no dictionary */

    // Reusable decompression context and compressed-block scratch. Both belong
    // to the single-threaded path, which is already not reentrant per handle;
    // the multi-threaded path gives each worker its own.
    zxc_cctx_t dctx;
    int dctx_initialized;
    uint8_t* read_buf;
    size_t read_buf_cap;

    // Dictionary (owned copy, freed in zxc_seekable_free).
    uint8_t* dict;
    size_t dict_size;
    // Shared literal Huffman table (owned copy; meaningful when has_dict_huf).
    uint8_t dict_huf[ZXC_HUF_TABLE_SIZE];
    int has_dict_huf;
};

/**
 * @struct zxc_seek_source_t
 * @brief Where the archive bytes come from during parsing.
 *
 * The two public entry points differ only in this: @ref zxc_seekable_open holds
 * the whole archive in memory, @ref zxc_seekable_open_reader reaches it through
 * a positioned callback. Everything after the first read is common, so the
 * parser below takes a source instead of being written twice.
 */
typedef struct {
    const uint8_t* data;     /* in-memory archive, NULL in callback mode */
    const zxc_reader_t* rdr; /* callback mode, NULL in buffer mode */
    uint64_t size;           /* archive size, both modes */
} zxc_seek_source_t;

/**
 * @brief Reads a byte range from the archive, whatever backs it.
 *
 * Returns 1 on success, 0 if the range falls outside the archive or the
 * caller's reader came up short: every bounds check on a parsed offset goes
 * through here.
 */
static int zxc_seek_source_read(const zxc_seek_source_t* src, void* dst, const size_t len,
                                const uint64_t off) {
    if (UNLIKELY(off > src->size || (uint64_t)len > src->size - off)) return 0;
    if (src->data) {
        ZXC_MEMCPY(dst, src->data + off, len);
        return 1;
    }
    return src->rdr->read_at(src->rdr->ctx, dst, len, off) == (int64_t)len;
}

/**
 * @brief Parses and validates the seek table at the end of the archive.
 *
 * Detection (backward from end):
 *   1. Read file header => block_size
 *   2. Read file footer => total_decomp_size
 *   3. Derive num_blocks = ceil(total_decomp_size / block_size)
 *   4. Compute expected seek block position, validate block_type == SEK
 *   5. Read comp_sizes, build the compressed-offset prefix sums, and check the
 *      layout lands exactly on the EOF block
 *
 * Returns a handle to free via @ref zxc_seekable_free, or NULL if the archive
 * is too small or the seek table is missing / malformed.
 */
static zxc_seekable* zxc_seekable_parse(const zxc_seek_source_t* src) {
    // Minimum: file_header(16) + eof_block(8) + seek_block_header(8)
    //          + file_footer(12) = 44
    const uint64_t MIN_SEEKABLE_SIZE =
        ZXC_FILE_HEADER_SIZE + ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE;
    if (UNLIKELY(src->size < MIN_SEEKABLE_SIZE)) return NULL;

    // Step 1: validate file header => block_size
    uint8_t header[ZXC_FILE_HEADER_SIZE];
    if (UNLIKELY(!zxc_seek_source_read(src, header, sizeof(header), 0))) return NULL;

    size_t block_size_sz = 0;
    int file_has_chk = 0;
    uint32_t header_dict_id = 0;
    if (UNLIKELY(zxc_read_file_header(header, sizeof(header), &block_size_sz, &file_has_chk,
                                      &header_dict_id) != ZXC_OK))
        return NULL;  // LCOV_EXCL_LINE
    const uint32_t block_size = (uint32_t)block_size_sz;
    if (UNLIKELY(block_size == 0)) return NULL;  // LCOV_EXCL_LINE

    // Step 2: read total decompressed size from the file footer
    uint8_t footer[ZXC_FILE_FOOTER_SIZE];
    if (UNLIKELY(
            !zxc_seek_source_read(src, footer, sizeof(footer), src->size - ZXC_FILE_FOOTER_SIZE)))
        return NULL;
    const uint64_t total_decomp = zxc_le64(footer);

    // A value of 0 means empty file - no seek table
    if (UNLIKELY(total_decomp == 0)) return NULL;

    // Step 3: derive num_blocks = ceil(total_decomp / block_size)
    const uint64_t num_blocks_64 = total_decomp / block_size + (total_decomp % block_size != 0);
    if (UNLIKELY(num_blocks_64 > UINT32_MAX)) return NULL;
    const uint32_t num_blocks = (uint32_t)num_blocks_64;

    // Step 4: locate and validate the seek block. Two headers of margin, not one:
    // the tail read below spans the EOF block, so tail_total could wrap on 32 bits.
    const uint64_t entries_total = num_blocks_64 * ZXC_SEEK_ENTRY_SIZE;
    if (UNLIKELY(entries_total > SIZE_MAX - 2 * ZXC_BLOCK_HEADER_SIZE)) return NULL;

    const size_t seek_block_total = ZXC_BLOCK_HEADER_SIZE + (size_t)entries_total;
    if (UNLIKELY((uint64_t)seek_block_total + ZXC_FILE_FOOTER_SIZE > src->size)) return NULL;

    const uint64_t seek_off = src->size - ZXC_FILE_FOOTER_SIZE - (uint64_t)seek_block_total;
    if (UNLIKELY(seek_off < ZXC_BLOCK_HEADER_SIZE)) return NULL;

    // The EOF block sits immediately before the seek block, so one read covers
    // both: [EOF 8][SEK header 8][entries]. Keeps a reader-backed open at three
    // reads (header, footer, tail) while validating the same layout the
    // in-memory path does.
    const size_t tail_total = ZXC_BLOCK_HEADER_SIZE + seek_block_total;
    const uint64_t tail_off = seek_off - ZXC_BLOCK_HEADER_SIZE;

    uint8_t* tail = NULL;
    const uint8_t* tail_view;
    zxc_seekable* s = NULL;
    if (src->data) {
        tail_view = src->data + tail_off;
    } else {
        tail = (uint8_t*)ZXC_MALLOC(tail_total);
        if (UNLIKELY(!tail)) return NULL;  // LCOV_EXCL_LINE
        if (UNLIKELY(!zxc_seek_source_read(src, tail, tail_total, tail_off))) goto fail;
        tail_view = tail;
    }

    const uint8_t* const eof_hdr = tail_view;
    const uint8_t* const seek_blk = tail_view + ZXC_BLOCK_HEADER_SIZE;

    zxc_block_header_t bh;
    // The SEK header stores the table size in a 32-bit field, so reject any larger value before
    // reading it.
    if (UNLIKELY(entries_total > UINT32_MAX)) goto fail;
    if (UNLIKELY(zxc_read_block_header(seek_blk, seek_block_total, &bh) != ZXC_OK ||
                 bh.block_type != ZXC_BLOCK_SEK || bh.comp_size != entries_total))
        goto fail;

    // Step 5: allocate the handle and parse the entries
    s = (zxc_seekable*)ZXC_CALLOC(1, sizeof(zxc_seekable));
    if (UNLIKELY(!s)) goto fail;  // LCOV_EXCL_LINE

    if (src->rdr) s->reader = *src->rdr;
    s->src = src->data;
    s->src_size = src->size;
    s->num_blocks = num_blocks;
    s->block_size = block_size;
    s->file_has_checksums = file_has_chk;
    s->expected_dict_id = header_dict_id;
    s->total_decomp = total_decomp;

    s->comp_sizes = (uint32_t*)ZXC_CALLOC(num_blocks, sizeof(uint32_t));
    s->comp_offsets = (uint64_t*)ZXC_CALLOC((size_t)num_blocks + 1, sizeof(uint64_t));
    if (UNLIKELY(!s->comp_sizes || !s->comp_offsets)) goto fail;  // LCOV_EXCL_LINE

    // Parse comp_sizes and build compressed prefix sums. Every entry is checked
    // against the archive size, so the prefix sum can neither overflow nor
    // point a later read out of bounds.
    {
        const uint8_t* ep = seek_blk + ZXC_BLOCK_HEADER_SIZE;
        uint64_t comp_acc = ZXC_FILE_HEADER_SIZE; /* blocks start after file header */
        for (uint32_t i = 0; i < num_blocks; i++) {
            s->comp_sizes[i] = zxc_le32(ep);
            ep += sizeof(uint32_t);

            // Reject entries below minimum (block header) or larger than the file
            if (UNLIKELY(s->comp_sizes[i] < ZXC_BLOCK_HEADER_SIZE || s->comp_sizes[i] > src->size))
                goto fail;
            if (s->comp_sizes[i] > s->max_comp_size) s->max_comp_size = s->comp_sizes[i];
            s->comp_offsets[i] = comp_acc;
            comp_acc += s->comp_sizes[i];
            // Reject if cumulative offset exceeds file size (inconsistent table)
            if (UNLIKELY(comp_acc > src->size)) goto fail;  // LCOV_EXCL_LINE
        }
        s->comp_offsets[num_blocks] = comp_acc;

        // Verify the prefix sum lands exactly on the EOF block, and that an EOF
        // block really sits there. Expected layout:
        // [header 16][data blocks][EOF 8][SEK block][footer 12]
        zxc_block_header_t eof_bh;
        if (UNLIKELY(comp_acc != seek_off - ZXC_BLOCK_HEADER_SIZE ||
                     zxc_read_block_header(eof_hdr, ZXC_BLOCK_HEADER_SIZE, &eof_bh) != ZXC_OK ||
                     eof_bh.block_type != ZXC_BLOCK_EOF))
            goto fail;
    }

    ZXC_FREE(tail);
    return s;

fail:
    ZXC_FREE(tail);
    zxc_seekable_free(s);
    return NULL;
}

/**
 * @brief Opens a seekable archive held entirely in a memory buffer.
 *
 * Public API; see @c zxc_seekable.h. Thin guard around
 * @ref zxc_seekable_parse, which detects and validates the trailing seek table.
 */
zxc_seekable* zxc_seekable_open(const void* src, const size_t src_size) {
    if (UNLIKELY(!src || src_size == 0)) return NULL;
    const zxc_seek_source_t source = {(const uint8_t*)src, NULL, (uint64_t)src_size};
    return zxc_seekable_parse(&source);
}

// zxc_seekable_open_file lives elsewhere: it builds a zxc_reader_t over pread()
// and delegates below, keeping this TU free of <stdio.h>.

/**
 * @brief Opens a seekable archive over a caller-supplied random-access reader.
 *
 * Public API; see @c zxc_seekable.h. Reads the file header, footer and seek
 * block through @p r->read_at (the FILE* variant wraps @c pread this way),
 * validates the SEK block, and builds the per-block compressed-offset prefix
 * sums. Unlike @ref zxc_seekable_open the archive is never mapped whole; only
 * the metadata is read up front.
 */
zxc_seekable* zxc_seekable_open_reader(const zxc_reader_t* r) {
    if (UNLIKELY(!r || !r->read_at || r->size == 0)) return NULL;
    const zxc_seek_source_t source = {NULL, r, r->size};
    return zxc_seekable_parse(&source);
}

/**
 * @brief Number of blocks in the archive.
 */
uint32_t zxc_seekable_get_num_blocks(const zxc_seekable* s) { return s ? s->num_blocks : 0; }

/**
 * @brief Total decompressed size of the archive.
 */
uint64_t zxc_seekable_get_decompressed_size(const zxc_seekable* s) {
    return s ? s->total_decomp : 0;
}

/**
 * @brief Compressed byte size of a given block.
 */
uint32_t zxc_seekable_get_block_comp_size(const zxc_seekable* s, const uint32_t block_idx) {
    if (UNLIKELY(!s || block_idx >= s->num_blocks)) return 0;
    return s->comp_sizes[block_idx];
}

/**
 * @brief Decompressed byte size of a given block.
 *
 * Every block decompresses to @c block_size except the last, which holds the
 * remainder of @c total_decomp.
 */
uint32_t zxc_seekable_get_block_decomp_size(const zxc_seekable* s, const uint32_t block_idx) {
    if (UNLIKELY(!s || block_idx >= s->num_blocks)) return 0;
    const uint64_t start = (uint64_t)block_idx * (uint64_t)s->block_size;
    const uint64_t remaining = s->total_decomp - start;
    return (remaining >= (uint64_t)s->block_size) ? s->block_size : (uint32_t)remaining;
}

// =========================================================================
// Random-Access Decompression
// =========================================================================

/**
 * @brief Maps a decompressed @p offset to its containing block index (O(1)).
 * @param[in] block_size  Fixed decompressed block size (a power of two).
 * @param[in] offset      Absolute decompressed byte offset.
 * @return Zero-based index of the block that holds @p offset.
 */
static uint32_t zxc_seek_find_block(const uint32_t block_size, const uint64_t offset) {
    return (uint32_t)(offset / (uint64_t)block_size);
}

/**
 * @brief Decompressed start offset of block @p idx (O(1)).
 * @param[in] block_size  Fixed decompressed block size.
 * @param[in] idx         Zero-based block index.
 * @return Absolute decompressed byte offset where block @p idx begins.
 */
static uint64_t zxc_seek_decomp_offset(const uint32_t block_size, const uint32_t idx) {
    return (uint64_t)idx * (uint64_t)block_size;
}

/**
 * @brief Decompressed size of block @p idx (O(1)).
 *
 * Returns @p block_size for every block except the last, which holds the
 * remainder of @p total_decomp.
 *
 * @param[in] block_size    Fixed decompressed block size.
 * @param[in] total_decomp  Total decompressed archive size.
 * @param[in] idx           Zero-based block index.
 * @return Decompressed byte size of block @p idx.
 */
static uint32_t zxc_seek_decomp_size(const uint32_t block_size, const uint64_t total_decomp,
                                     const uint32_t idx) {
    const uint64_t start = (uint64_t)idx * (uint64_t)block_size;
    const uint64_t remaining = total_decomp - start;
    return (remaining >= (uint64_t)block_size) ? block_size : (uint32_t)remaining;
}

/**
 * @brief Reads a compressed block into @p buf from the memory buffer or reader.
 *
 * Copies from @c s->src in buffer mode, otherwise calls @c s->reader.read_at
 * (which also backs the FILE* variant).
 *
 * @param[in]  s          Seekable handle.
 * @param[in]  block_idx  Zero-based block index to read.
 * @param[out] buf        Destination buffer.
 * @param[in]  buf_cap    Capacity of @p buf in bytes.
 * @return The block's compressed byte count on success, or a negative
 *         @ref zxc_error_t (@ref ZXC_ERROR_DST_TOO_SMALL,
 *         @ref ZXC_ERROR_SRC_TOO_SMALL, @ref ZXC_ERROR_IO).
 */
static int zxc_seek_read_block(const zxc_seekable* s, const uint32_t block_idx, uint8_t* buf,
                               const size_t buf_cap) {
    const uint64_t off = s->comp_offsets[block_idx];
    const uint32_t csz = s->comp_sizes[block_idx];
    if (UNLIKELY(csz > buf_cap)) return ZXC_ERROR_DST_TOO_SMALL;

    if (s->src) {
        // Buffer mode
        if (UNLIKELY(off + csz > s->src_size)) return ZXC_ERROR_SRC_TOO_SMALL;
        ZXC_MEMCPY(buf, s->src + off, csz);
    } else if (s->reader.read_at) {
        // Caller-supplied reader (also covers the FILE* variant, which
        // provides a pread-backed callback from zxc_seekable_file.c).
        const int64_t r = s->reader.read_at(s->reader.ctx, buf, csz, off);
        if (UNLIKELY(r != (int64_t)csz)) return (r < 0) ? (int)r : ZXC_ERROR_IO;
    } else {
        return ZXC_ERROR_NULL_INPUT;  // LCOV_EXCL_LINE
    }
    return (int)csz;
}

/**
 * @brief Decompresses the byte range [@p offset, @p offset + @p len) into @p dst.
 *
 * Public API; full contract in @c zxc_seekable.h. Maps the range to its block
 * span via O(1) division, decodes each covered block through a reusable,
 * lazily-initialised, dictionary-aware context, and copies out only the
 * requested sub-range. Single-threaded; see @ref zxc_seekable_decompress_range_mt
 * for the parallel variant.
 */
int64_t zxc_seekable_decompress_range(zxc_seekable* s, void* dst, const size_t dst_capacity,
                                      const uint64_t offset, const size_t len) {
    if (UNLIKELY(len == 0)) return 0;
    if (UNLIKELY(!s || !dst)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(dst_capacity < len)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(offset + len > s->total_decomp)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (UNLIKELY(s->expected_dict_id != 0 && (!s->dict || s->dict_size == 0)))
        return ZXC_ERROR_DICT_REQUIRED;

    // Initialize decompression context on first use
    if (!s->dctx_initialized) {
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&s->dctx, (size_t)s->block_size, 0, 0, 0, s->dict_size) !=
                     ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        if (UNLIKELY(zxc_cctx_attach_dict_huf(&s->dctx, s->has_dict_huf ? s->dict_huf : NULL) !=
                     ZXC_OK)) {
            // LCOV_EXCL_START
            zxc_cctx_free(&s->dctx);
            return ZXC_ERROR_CORRUPT_DATA;
            // LCOV_EXCL_STOP
        }
        s->dctx_initialized = 1;
        if (s->dict_size > 0) ZXC_MEMCPY(s->dctx.dict_buffer, s->dict, s->dict_size);
    }
    s->dctx.dict_size = s->dict_size;

    // work_buf is pre-sized to block_size + ZXC_DECOMPRESS_TAIL_PAD by the
    // matching zxc_cctx_init above.
    const size_t work_sz = (size_t)s->block_size + ZXC_DECOMPRESS_TAIL_PAD;

    // Find block range - O(1) division
    const uint32_t blk_start = zxc_seek_find_block(s->block_size, offset);
    const uint32_t blk_end = zxc_seek_find_block(s->block_size, offset + len - 1);

    uint8_t* out = (uint8_t*)dst;
    size_t remaining = len;

    // Compressed-block scratch, sized once for the largest block of the archive
    // and kept on the handle: a range read is often one of many.
    const size_t read_cap = (size_t)s->max_comp_size + ZXC_PAD_SIZE;
    if (s->read_buf_cap < read_cap) {
        uint8_t* const nb = (uint8_t*)ZXC_REALLOC(s->read_buf, read_cap);
        if (UNLIKELY(!nb)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        s->read_buf = nb;
        s->read_buf_cap = read_cap;
    }
    uint8_t* const read_buf = s->read_buf;

    for (uint32_t bi = blk_start; bi <= blk_end; bi++) {
        // Read compressed block data
        const int read_res = zxc_seek_read_block(s, bi, read_buf, read_cap);
        if (UNLIKELY(read_res < 0)) return read_res;  // LCOV_EXCL_LINE

        // Decompress the block: when a dictionary is active, decode into the
        // cctx-owned dict_buffer (which has dict content prepended) so that
        // match copies referencing dictionary bytes resolve naturally.
        uint8_t* dec_dst =
            s->dctx.dict_buffer ? s->dctx.dict_buffer + s->dict_size : s->dctx.work_buf;
        const int dec_res =
            zxc_decompress_chunk_wrapper(&s->dctx, read_buf, (size_t)read_res, dec_dst, work_sz);
        if (UNLIKELY(dec_res < 0)) return dec_res;  // LCOV_EXCL_LINE

        // Calculate which portion of this block's decompressed data we need
        const uint64_t blk_decomp_start = zxc_seek_decomp_offset(s->block_size, bi);
        const size_t skip = (offset > blk_decomp_start) ? (size_t)(offset - blk_decomp_start) : 0;
        if (UNLIKELY((size_t)dec_res < skip)) return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE
        const size_t avail = (size_t)dec_res - skip;
        const size_t copy = (avail < remaining) ? avail : remaining;

        ZXC_MEMCPY(out, dec_dst + skip, copy);
        out += copy;
        remaining -= copy;
    }

    return (int64_t)len;
}

// =========================================================================
// Multi-Threaded Random-Access Decompression (Fork-Join)
// =========================================================================

/**
 * @brief Per-block job descriptor for multi-threaded decompression.
 *
 * Each worker thread receives a pointer to one of these, performs the read +
 * decompress + memcpy sequence, and writes the result code into @c result.
 * The main thread inspects @c result after join.
 */
typedef struct {
    const zxc_seekable* s; /* shared handle (read-only) */
    uint32_t block_idx;    /* block to decompress */
    uint8_t* dst;          /* output pointer within caller's buffer */
    size_t skip;           /* bytes to skip at start of decompressed block */
    size_t copy_len;       /* bytes to copy into dst */
    int result;            /* 0 = OK, < 0 = error */
} zxc_seek_mt_job_t;

/**
 * @struct zxc_seek_mt_stripe_t
 * @brief Per-thread stripe descriptor for multi-threaded decompression.
 *
 * Each worker owns the job subset {first, first+stride, first+2*stride, ...}
 * of the shared @c jobs array and reuses one decompression context, one
 * dictionary copy and one read buffer across all of them, amortising what
 * would otherwise be per-block costs (context init, dict memcpy, malloc and
 * a thread spawn per block).  Stripes are pairwise disjoint, so workers
 * touch distinct jobs and distinct output ranges and need no
 * synchronisation beyond the final join.
 *
 * Written by @ref zxc_seekable_decompress_range_mt before the fork phase and
 * read-only for the worker (@ref zxc_seek_mt_worker); results travel through
 * the jobs themselves (@c zxc_seek_mt_job_t::result).
 *
 * @var zxc_seek_mt_stripe_t::jobs
 *      Job array shared by all workers; this worker only reads/writes the
 *      entries of its own stripe.
 * @var zxc_seek_mt_stripe_t::num_jobs
 *      Total number of jobs in @c jobs (stripe iteration bound).
 * @var zxc_seek_mt_stripe_t::first
 *      Index of this worker's first job (equals its worker index, in
 *      [0, @c stride)).
 * @var zxc_seek_mt_stripe_t::stride
 *      Stripe step between consecutive jobs of this worker; equals the
 *      worker-thread count.
 */
typedef struct {
    zxc_seek_mt_job_t* jobs;
    uint32_t num_jobs;
    uint32_t first;
    uint32_t stride;
} zxc_seek_mt_stripe_t;

/**
 * @brief Marks every job of a stripe with @p code (setup-failure path).
 *
 * @param[in,out] st   Stripe whose jobs to mark.
 * @param[in]     code Negative @ref zxc_error_t value.
 */
static void zxc_seek_mt_fail_stripe(zxc_seek_mt_stripe_t* st, const int code) {
    for (uint32_t i = st->first; i < st->num_jobs; i += st->stride) st->jobs[i].result = code;
}

/**
 * @brief Worker thread entry point for multi-threaded seekable decompression.
 *
 * Sets up its context, dictionary copy and read buffer once, then for each
 * block of its stripe: read (thread-safe pread), decompress, copy the
 * requested sub-range into the caller's output.  The dict prefix survives
 * across blocks because the decoder never writes below its dst.
 *
 * Each job's outcome goes into its @c result (read by the main thread after
 * join); on error the worker abandons the rest of its stripe.
 *
 * @param[in,out] arg  Pointer to this worker's `zxc_seek_mt_stripe_t`.
 * @return Always NULL (result codes are reported via the jobs).
 */
static void* zxc_seek_mt_worker(void* arg) {
    zxc_seek_mt_stripe_t* const st = (zxc_seek_mt_stripe_t*)arg;
    zxc_seek_mt_job_t* const jobs = st->jobs;
    const zxc_seekable* const s = jobs[st->first].s;

    // Thread-local decompression context (mode=0 for decompress-only)
    zxc_cctx_t dctx;
    if (UNLIKELY(zxc_cctx_init(&dctx, (size_t)s->block_size, 0, 0, 0, s->dict_size) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_seek_mt_fail_stripe(st, ZXC_ERROR_MEMORY);
        return NULL;
        // LCOV_EXCL_STOP
    }

    if (UNLIKELY(zxc_cctx_attach_dict_huf(&dctx, s->has_dict_huf ? s->dict_huf : NULL) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&dctx);
        zxc_seek_mt_fail_stripe(st, ZXC_ERROR_CORRUPT_DATA);
        return NULL;
        // LCOV_EXCL_STOP
    }
    const size_t work_sz = (size_t)s->block_size + ZXC_DECOMPRESS_TAIL_PAD;

    uint8_t* const dict_work = dctx.dict_buffer;
    if (dict_work) ZXC_MEMCPY(dict_work, s->dict, s->dict_size);

    // Read buffer sized for the largest compressed block of the stripe.
    size_t max_csz = 0;
    for (uint32_t i = st->first; i < st->num_jobs; i += st->stride) {
        const uint32_t csz = s->comp_sizes[jobs[i].block_idx];
        if (csz > max_csz) max_csz = csz;
    }
    uint8_t* const read_buf = (uint8_t*)ZXC_MALLOC(max_csz + ZXC_PAD_SIZE);
    if (UNLIKELY(!read_buf)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&dctx);
        zxc_seek_mt_fail_stripe(st, ZXC_ERROR_MEMORY);
        return NULL;
        // LCOV_EXCL_STOP
    }

    for (uint32_t i = st->first; i < st->num_jobs; i += st->stride) {
        zxc_seek_mt_job_t* const job = &jobs[i];

        const int read_res =
            zxc_seek_read_block(s, job->block_idx, read_buf, max_csz + ZXC_PAD_SIZE);
        if (UNLIKELY(read_res < 0)) {
            // LCOV_EXCL_START
            job->result = read_res;
            break;
            // LCOV_EXCL_STOP
        }

        // Decompress: use dict bounce buffer when dictionary is active
        uint8_t* dec_dst = dict_work ? dict_work + s->dict_size : dctx.work_buf;
        const int dec_res =
            zxc_decompress_chunk_wrapper(&dctx, read_buf, (size_t)read_res, dec_dst, work_sz);

        if (UNLIKELY(dec_res < 0)) {
            // LCOV_EXCL_START
            job->result = dec_res;
            break;
            // LCOV_EXCL_STOP
        }
        if (UNLIKELY((size_t)dec_res < job->skip + job->copy_len)) {
            // LCOV_EXCL_START
            job->result = ZXC_ERROR_CORRUPT_DATA;
            break;
            // LCOV_EXCL_STOP
        }

        // Copy the requested portion directly into the caller's output buffer
        ZXC_MEMCPY(job->dst, dec_dst + job->skip, job->copy_len);
        job->result = 0;
    }

    ZXC_FREE(read_buf);
    zxc_cctx_free(&dctx);
    return NULL;
}

/**
 * @brief Multi-threaded variant of @ref zxc_seekable_decompress_range.
 *
 * Public API; full contract in @c zxc_seekable.h. Plans one job per covered
 * block (each with its own thread-local context and read buffer) and runs them
 * fork-join in waves of up to @p n_threads. Falls back to the single-threaded
 * path for trivial spans. @p n_threads == 0 auto-detects the core count.
 */
int64_t zxc_seekable_decompress_range_mt(zxc_seekable* s, void* dst, const size_t dst_capacity,
                                         const uint64_t offset, const size_t len, int n_threads) {
    if (UNLIKELY(len == 0)) return 0;
    if (UNLIKELY(!s || !dst)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(dst_capacity < len)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(offset + len > s->total_decomp)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (UNLIKELY(s->expected_dict_id != 0 && (!s->dict || s->dict_size == 0)))
        return ZXC_ERROR_DICT_REQUIRED;

    // Find block range - O(1) division
    const uint32_t blk_start = zxc_seek_find_block(s->block_size, offset);
    const uint32_t blk_end = zxc_seek_find_block(s->block_size, offset + len - 1);
    const uint32_t num_jobs = blk_end - blk_start + 1;

    // Auto-detect thread count (0 = use all available cores)
    if (n_threads == 0) n_threads = zxc_num_procs();

    // Fallback to single-threaded path for trivial cases
    if (n_threads <= 1 || num_jobs <= 1) {
        return zxc_seekable_decompress_range(s, dst, dst_capacity, offset, len);
    }

    // Cap threads to number of blocks and max limit
    if ((uint32_t)n_threads > num_jobs) n_threads = (int)num_jobs;
    if (n_threads > ZXC_MAX_THREADS) n_threads = ZXC_MAX_THREADS;

    // Allocate job descriptors
    zxc_seek_mt_job_t* const jobs =
        (zxc_seek_mt_job_t*)ZXC_CALLOC(num_jobs, sizeof(zxc_seek_mt_job_t));
    if (UNLIKELY(!jobs)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE

    // Plan jobs: compute skip, copy_len, and dst pointer for each block
    uint8_t* out = (uint8_t*)dst;
    size_t remaining = len;
    for (uint32_t i = 0; i < num_jobs; i++) {
        const uint32_t bi = blk_start + i;
        const uint64_t blk_decomp_start = zxc_seek_decomp_offset(s->block_size, bi);
        const size_t skip = (offset > blk_decomp_start) ? (size_t)(offset - blk_decomp_start) : 0;
        const size_t blk_decomp_sz = zxc_seek_decomp_size(s->block_size, s->total_decomp, bi);
        if (UNLIKELY(blk_decomp_sz < skip)) {
            // LCOV_EXCL_START
            ZXC_FREE(jobs);
            return ZXC_ERROR_CORRUPT_DATA;
            // LCOV_EXCL_STOP
        }
        const size_t avail = blk_decomp_sz - skip;
        const size_t copy = (avail < remaining) ? avail : remaining;

        jobs[i].s = s;
        jobs[i].block_idx = bi;
        jobs[i].dst = out;
        jobs[i].skip = skip;
        jobs[i].copy_len = copy;
        jobs[i].result = 0;

        out += copy;
        remaining -= copy;
    }

    // Launch one persistent worker per thread
    pthread_t* const threads = (pthread_t*)ZXC_MALLOC((size_t)n_threads * sizeof(pthread_t));
    zxc_seek_mt_stripe_t* const stripes =
        (zxc_seek_mt_stripe_t*)ZXC_MALLOC((size_t)n_threads * sizeof(zxc_seek_mt_stripe_t));
    if (UNLIKELY(!threads || !stripes)) {
        // LCOV_EXCL_START
        ZXC_FREE(threads);
        ZXC_FREE(stripes);
        ZXC_FREE(jobs);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    int launched = 0;
    for (int t = 0; t < n_threads; t++) {
        stripes[t].jobs = jobs;
        stripes[t].num_jobs = num_jobs;
        stripes[t].first = (uint32_t)t;
        stripes[t].stride = (uint32_t)n_threads;
        if (UNLIKELY(pthread_create(&threads[t], NULL, zxc_seek_mt_worker, &stripes[t]) != 0)) {
            // LCOV_EXCL_START
            // Failed to create thread - mark its stripe as errored; already
            // launched workers keep running and are joined below.
            zxc_seek_mt_fail_stripe(&stripes[t], ZXC_ERROR_MEMORY);
            continue;
            // LCOV_EXCL_STOP
        }

        launched++;
        threads[launched - 1] = threads[t];
    }

    // Join phase
    for (int t = 0; t < launched; t++) pthread_join(threads[t], NULL);

    ZXC_FREE(threads);
    ZXC_FREE(stripes);

    // Report the first error in job order, if any.
    int64_t result = (int64_t)len;
    for (uint32_t i = 0; i < num_jobs; i++) {
        if (jobs[i].result < 0) {
            result = (int64_t)jobs[i].result;
            break;
        }
    }

    ZXC_FREE(jobs);
    return result;
}

/**
 * @brief Releases a seekable handle and every resource it owns.
 *
 * Public API; see @c zxc_seekable.h. Tears down the reusable context, the seek
 * arrays (comp sizes / offsets), the owned dictionary copy and any attached
 * reader context. NULL-safe.
 */
void zxc_seekable_free(zxc_seekable* s) {
    if (UNLIKELY(!s)) return;
    if (s->dctx_initialized) zxc_cctx_free(&s->dctx);
    ZXC_FREE(s->dict);
    ZXC_FREE(s->read_buf);
    ZXC_FREE(s->comp_sizes);
    ZXC_FREE(s->comp_offsets);
    ZXC_FREE(s->owned_reader_ctx);
    ZXC_FREE(s);
}

/**
 * @brief Installs the dictionary needed to decode a dict-compressed archive.
 *
 * Public API; full contract in @c zxc_seekable.h. Validates the dict_id against
 * the file header, then takes an owned copy of @p dict (and the optional shared
 * literal Huffman table @p dict_huf). Drops any context already built so the
 * [dict | decode] bounce buffer is re-carved on the next decompress.
 */
int zxc_seekable_set_dict(zxc_seekable* s, const void* dict, const size_t dict_size,
                          const void* dict_huf) {
    if (UNLIKELY(!s || !dict || dict_size == 0)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(s->expected_dict_id != 0 &&
                 zxc_dict_id(dict, dict_size, (const uint8_t*)dict_huf) != s->expected_dict_id))
        return ZXC_ERROR_DICT_MISMATCH;

    ZXC_FREE(s->dict);
    s->dict = NULL;
    s->dict_size = 0;
    s->has_dict_huf = 0;

    s->dict = (uint8_t*)ZXC_MALLOC(dict_size);
    if (UNLIKELY(!s->dict)) return ZXC_ERROR_MEMORY;
    ZXC_MEMCPY(s->dict, dict, dict_size);
    s->dict_size = dict_size;
    if (dict_huf) {
        ZXC_MEMCPY(s->dict_huf, dict_huf, ZXC_HUF_TABLE_SIZE);
        s->has_dict_huf = 1;
    }

    // The [dict | decode] bounce buffer is carved into the dctx workspace.
    // Drop any context built without it (or for a different dict size) so it is
    // re-carved with the new dict on the next decompress.
    if (s->dctx_initialized) {
        zxc_cctx_free(&s->dctx);
        s->dctx_initialized = 0;
    }
    return ZXC_OK;
}

/**
 * @brief Transfers ownership of a heap reader context to the handle.
 *
 * Cross-TU hook (declared in @c zxc_internal.h): @p ctx is released via
 * @c ZXC_FREE when @ref zxc_seekable_free runs. Used by
 * @ref zxc_seekable_open_file so its allocated reader state outlives the open
 * call. NULL-safe on @p s.
 */
void zxc_seekable_attach_owned_ctx(zxc_seekable* s, void* ctx) {
    if (s) s->owned_reader_ctx = ctx;
}
