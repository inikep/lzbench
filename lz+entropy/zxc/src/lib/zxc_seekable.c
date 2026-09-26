// SPDX-License-Identifier: BSD-3-Clause
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_seekable.c
 * @brief Seekable archive reader (random-access decompression) and seek table writer.
 *
 * The seek table is a standard ZXC block (type = ZXC_BLOCK_SEK) appended
 * between the EOF block and the file footer. It records where every block
 * starts, so a byte range costs one small table read plus the blocks it
 * covers; no table stays resident and opening costs two reads per frame, plus one
 * for the last frame's EOF and SEK headers.
 *
 * On-disk layout of a SEK block, for N blocks in G = ceil(N / 64) groups:
 *
 *   [Block Header (8B)]   block_type=SEK, block_flags=0, comp_size=(u32)(T ^ T >> 32),
 *                         T = G*8 + N*4 the table's byte size
 *   G x group:
 *     [Anchor (8B)]       byte offset (u64 LE) of the group's first block
 *     [<= 64 x Size (4B)] on-disk size (u32 LE) of each of its blocks
 *
 * Anchor + sizes lands on the next anchor (the EOF block for the last group); the
 * reader checks each group alone. N comes from the footer: the header's 32-bit
 * field does not bound it.
 *
 * Detection from the end, frame by frame:
 *   1. Parse the footer back from its last byte => total_decompressed_size and
 *      the frame's compressed size, so where its header starts
 *   2. The header: block_size, and HAS_SEEK_TABLE, without which the archive
 *      is not seekable
 *   3. Derive num_blocks = ceil(total_decomp / block_size)
 *   4. Groups are read and checked on access, with the EOF and SEK block
 *      headers ahead of them
 */

#include "../../include/zxc_seekable.h"

#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"
#include "zxc_threads.h"

// =========================================================================
// Seekable Reader (Opaque Handle)
// =========================================================================

/**
 * @struct zxc_seek_frame_t
 * @brief One frame of the archive: where it lies, in bytes and in blocks.
 */
typedef struct {
    uint64_t base;         /* archive offset of its file header */
    uint64_t decomp_base;  /* decompressed offset of its first byte */
    uint64_t block_base;   /* archive-wide index of its first block */
    uint64_t num_blocks;   /* ceil(total_decomp / block_size) */
    uint64_t total_decomp; /* from its footer */
    uint64_t table_off;    /* first table entry, absolute */
    uint64_t eof_off;      /* EOF block header, relative to base like the anchors */
    uint64_t entry_max;    /* largest legal block on disk: header + block_size + checksum */
    uint32_t block_size;   /* a power of 2 in [4 KB, 2 MB] */
    uint32_t dict_id;      /* 0 for none */
    int has_checksums;
} zxc_seek_frame_t;

/** Most frames a handle indexes: bounds the reads and memory of an open. */
#define ZXC_SEEK_MAX_FRAMES (1U << 20)

/** EOF and SEK block headers, ahead of a frame's table: checked when it is read. */
#define ZXC_SEEK_TAIL_HDRS (2 * ZXC_BLOCK_HEADER_SIZE)

/** @brief An installed dictionary, owned copy. */
typedef struct {
    uint8_t* data;
    size_t size;
    uint32_t id;
    int has_huf;
    uint8_t huf[ZXC_HUF_TABLE_SIZE]; /* shared literal table, when has_huf */
} zxc_seek_dict_t;

struct zxc_seekable_s {
    // Source - exactly one of {src, reader.read_at} is set. The FILE* variant
    // wraps pread() in its own reader ctx, indistinguishable from here.
    const uint8_t* src;
    uint64_t src_size;
    zxc_reader_t reader; /* user-supplied callback reader; read_at == NULL when unused */

    // Reader context owned by the handle and freed in zxc_seekable_free, set by
    // the thin wrappers. NULL when the caller owns reader.ctx itself.
    void* owned_reader_ctx;

    // Frames in archive order; their tables stay on disk.
    zxc_seek_frame_t* frames;
    size_t num_frames;
    uint64_t num_blocks;     /* over every frame */
    uint64_t total_decomp;   /* over every frame */
    uint32_t max_block_size; /* sizes the decoding contexts */

    int verify_checksums; /* caller's switch; needs the frame's checksums too */

    // Reusable decompression context and compressed-block scratch. Both belong
    // to the single-threaded path, which is already not reentrant per handle;
    // the multi-threaded path gives each worker its own.
    zxc_cctx_t dctx;
    int dctx_initialized;
    const zxc_seek_dict_t* dctx_dict; /* the dictionary dctx holds, NULL for none */
    uint8_t* read_buf;
    size_t read_buf_cap;

    // Dictionaries installed by zxc_seekable_set_dict, one per id.
    zxc_seek_dict_t* dicts;
    size_t num_dicts;
    size_t max_dict_size; /* sizes the contexts' [dict | decode] buffers */
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
 * @brief Reads a byte range from the archive, whatever backs it: the one bounds
 *        check every parsed offset goes through, written so it cannot wrap.
 * @return ZXC_OK; @ref ZXC_ERROR_SRC_TOO_SMALL outside the archive; the reader's
 *         code, or @ref ZXC_ERROR_IO, on a short read.
 */
static int zxc_seek_source_read(const zxc_seek_source_t* src, void* dst, const size_t len,
                                const uint64_t off) {
    if (UNLIKELY(off > src->size || (uint64_t)len > src->size - off))
        return ZXC_ERROR_SRC_TOO_SMALL;
    if (src->data) {
        ZXC_MEMCPY(dst, src->data + off, len);
        return ZXC_OK;
    }
    const int64_t r = src->rdr->read_at(src->rdr->ctx, dst, len, off);
    // A code outside int would truncate, possibly to ZXC_OK.
    if (UNLIKELY(r != (int64_t)len)) return (r < 0 && r >= INT_MIN) ? (int)r : ZXC_ERROR_IO;
    return ZXC_OK;
}

/**
 * @brief The handle's archive as a read source, for the per-block readers.
 */
static zxc_seek_source_t zxc_seek_source_of(const zxc_seekable* s) {
    const zxc_seek_source_t src = {s->src, s->src ? NULL : &s->reader, s->src_size};
    return src;
}

/** @brief @ref zxc_scan_src_t over a @c zxc_seek_source_t. */
static int zxc_seek_scan_read(const zxc_scan_src_t* scan, const uint64_t off, void* dst,
                              const size_t len) {
    return zxc_seek_source_read((const zxc_seek_source_t*)scan->ctx, dst, len, off);
}

/**
 * @brief Validates the seekable frame that ends at @p end, in two reads: header and
 *        footer as zxc_get_last_frame_info() does. Its EOF and SEK headers are
 *        checked with its table, on access (@ref zxc_seek_load_spans).
 *
 * @param[in]  src The archive.
 * @param[in]  end Offset just past the frame.
 * @param[out] f   The frame; its decompressed and block bases are left to the caller.
 * @return 1 for a seekable frame, 0 otherwise.
 */
static int zxc_seek_parse_frame(zxc_seek_source_t* src, const uint64_t end, zxc_seek_frame_t* f) {
    const zxc_scan_src_t scan = {zxc_seek_scan_read, NULL, src, end};
    zxc_frame_info_t fi;
    uint64_t start = 0;
    size_t footer_len = 0;
    if (UNLIKELY(zxc_scan_frame(&scan, end, &start, &fi, &footer_len) != ZXC_OK ||
                 !fi.has_seek_table))
        return 0;
    const uint64_t frame = end - start;
    const uint32_t block_size = (uint32_t)fi.block_size;
    const uint64_t num_blocks = zxc_seek_block_count(fi.decompressed_size, block_size);

    // Layout: [header 16][data blocks][EOF 8][SEK block][footer], in 64 bits:
    // the SEK header's field only holds the table size modulo 2^32.
    const uint64_t seek_block_total = ZXC_BLOCK_HEADER_SIZE + zxc_seek_table_bytes(num_blocks);
    if (UNLIKELY(seek_block_total + footer_len >
                 frame - ZXC_FILE_HEADER_SIZE - ZXC_BLOCK_HEADER_SIZE))
        return 0;
    const uint64_t eof_off = frame - footer_len - seek_block_total - ZXC_BLOCK_HEADER_SIZE;

    // Geometry only, no table read: num_blocks blocks of [header, entry_max] bytes
    // must fit the data area. Groups are checked on access.
    const uint64_t entry_max = (uint64_t)ZXC_BLOCK_HEADER_SIZE + block_size +
                               (fi.has_checksum ? ZXC_BLOCK_CHECKSUM_SIZE : 0U);
    const uint64_t data_area = eof_off - ZXC_FILE_HEADER_SIZE;
    const uint64_t min_span = num_blocks * ZXC_BLOCK_HEADER_SIZE;
    const uint64_t max_span =
        num_blocks > UINT64_MAX / entry_max ? UINT64_MAX : num_blocks * entry_max;
    if (UNLIKELY(data_area < min_span || data_area > max_span)) return 0;

    f->base = start;
    f->num_blocks = num_blocks;
    f->total_decomp = fi.decompressed_size;
    f->table_off = start + eof_off + 2 * ZXC_BLOCK_HEADER_SIZE;
    f->eof_off = eof_off;
    f->entry_max = entry_max;
    f->block_size = block_size;
    f->has_checksums = fi.has_checksum;
    f->dict_id = fi.dict_id;
    return 1;
}

/**
 * @brief Walks the frames back from the end through @ref zxc_seek_parse_frame,
 *        then numbers their bytes and blocks. NULL unless every frame is seekable.
 */
static zxc_seekable* zxc_seekable_parse(const zxc_seek_source_t* source) {
    zxc_seek_source_t src = *source;
    zxc_seek_frame_t* frames = NULL;
    size_t n = 0;
    size_t cap = 0;
    uint64_t end = src.size;
    do {
        if (UNLIKELY(n == ZXC_SEEK_MAX_FRAMES)) goto fail;
        if (n == cap) {
            cap = cap ? 2 * cap : 4;
            zxc_seek_frame_t* const grown =
                (zxc_seek_frame_t*)ZXC_REALLOC(frames, cap * sizeof(*frames));
            if (UNLIKELY(!grown)) goto fail;  // LCOV_EXCL_LINE
            frames = grown;
        }
        if (UNLIKELY(!zxc_seek_parse_frame(&src, end, &frames[n]))) goto fail;
        // The last frame's EOF/SEK headers (a plain archive's only frame) now: a
        // lying seek flag fails the open.
        if (n == 0) {
            uint8_t hdrs[ZXC_SEEK_TAIL_HDRS];
            const zxc_seek_frame_t* const f = &frames[0];
            if (UNLIKELY(zxc_seek_source_read(&src, hdrs, sizeof(hdrs),
                                              f->table_off - ZXC_SEEK_TAIL_HDRS) != ZXC_OK ||
                         zxc_check_eof_header(hdrs) != ZXC_OK ||
                         zxc_check_seek_header(hdrs + ZXC_BLOCK_HEADER_SIZE, ZXC_BLOCK_HEADER_SIZE,
                                               f->total_decomp, f->block_size, NULL) != ZXC_OK))
                goto fail;
        }
        end = frames[n++].base;
    } while (end > 0);

    // Walked back: archive order, then running bytes and blocks.
    uint64_t decomp = 0;
    uint64_t blocks = 0;
    uint32_t max_block_size = 0;
    for (size_t i = 0; i < n / 2; i++) {
        const zxc_seek_frame_t t = frames[i];
        frames[i] = frames[n - 1 - i];
        frames[n - 1 - i] = t;
    }
    for (size_t i = 0; i < n; i++) {
        zxc_seek_frame_t* const f = &frames[i];
        if (UNLIKELY(f->total_decomp > UINT64_MAX - decomp)) goto fail;
        f->decomp_base = decomp;
        f->block_base = blocks;
        decomp += f->total_decomp;
        blocks += f->num_blocks;  // no wrap: each frame holds at least 8 bytes per block
        if (f->block_size > max_block_size) max_block_size = f->block_size;
    }

    zxc_seekable* const s = (zxc_seekable*)ZXC_CALLOC(1, sizeof(zxc_seekable));
    if (UNLIKELY(!s)) goto fail;  // LCOV_EXCL_LINE
    if (src.rdr) s->reader = *src.rdr;
    s->src = src.data;
    s->src_size = src.size;
    s->frames = frames;
    s->num_frames = n;
    s->num_blocks = blocks;
    s->total_decomp = decomp;
    s->max_block_size = max_block_size;
    s->verify_checksums = 0; /* opt-in, see zxc_seekable_set_checksum */
    return s;

fail:
    ZXC_FREE(frames);
    return NULL;
}

/** @brief The frame holding archive-wide block @p idx, below the block count. */
static const zxc_seek_frame_t* zxc_seek_frame_of_block(const zxc_seekable* s, const uint64_t idx) {
    size_t lo = 0;
    size_t hi = s->num_frames;
    while (lo < hi) {  // the first frame starting past idx; the one before holds it
        const size_t mid = lo + (hi - lo) / 2;
        if (s->frames[mid].block_base <= idx)
            lo = mid + 1;
        else
            hi = mid;
    }
    return &s->frames[lo - 1];
}

/** @brief Archive-wide index of the block holding decompressed @p offset, below the size. */
static uint64_t zxc_seek_block_of_offset(const zxc_seekable* s, const uint64_t offset) {
    size_t lo = 0;
    size_t hi = s->num_frames;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (s->frames[mid].decomp_base <= offset)
            lo = mid + 1;
        else
            hi = mid;
    }
    const zxc_seek_frame_t* const f = &s->frames[lo - 1];
    return f->block_base + ((offset - f->decomp_base) >> zxc_ctz32(f->block_size));
}

/** @brief Scratch bound for @ref zxc_seek_load_spans(): the groups blocks [@p first,
 *  @p first + @p n) touch. */
static size_t zxc_seek_spans_raw_max(const uint64_t first, const uint32_t n) {
    const uint64_t groups = (first + n - 1) / ZXC_SEEK_GROUP - first / ZXC_SEEK_GROUP + 1;
    return ZXC_SEEK_TAIL_HDRS + (size_t)groups * ZXC_SEEK_GROUP_BYTES;
}

/**
 * @brief Loads where each block of [@p first, @p first + @p n) starts and its on-disk
 *        size, both taken from the block's own group.
 *
 * One read of the groups the range touches, plus the frame's EOF/SEK headers: in
 * the same read from group 0, in a 16-byte read of their own past it. Each group
 * is checked alone: anchor in the data area, sizes in [header, entry_max], end
 * within the data area and, for the last group, on the EOF block. The next anchor
 * is not read, so a bad one costs only its own group. A block's size is its own
 * entry, never the gap to the next anchor, so a range spanning groups gets the
 * verdict each group gives alone.
 *
 * @param[in]  s       Handle.
 * @param[in]  f       Frame of the blocks; @p first + @p n must not exceed its block count.
 * @param[in]  first   First block index within @p f.
 * @param[in]  n       Block count, at least 1.
 * @param[out] starts  Room for @p n archive offsets.
 * @param[out] sizes   Room for @p n sizes, each in [header, entry_max].
 * @param[out] raw     Scratch of @ref zxc_seek_spans_raw_max bytes.
 * @return ZXC_OK, @ref ZXC_ERROR_IO (short read), @ref ZXC_ERROR_CORRUPT_DATA.
 */
static int zxc_seek_load_spans(const zxc_seekable* s, const zxc_seek_frame_t* f,
                               const uint64_t first, const uint32_t n, uint64_t* RESTRICT starts,
                               uint32_t* RESTRICT sizes, uint8_t* RESTRICT raw) {
    const uint64_t last_group = zxc_seek_group_count(f->num_blocks) - 1;
    const uint64_t g0 = first / ZXC_SEEK_GROUP;
    const uint64_t g1 = (first + n - 1) / ZXC_SEEK_GROUP;

    size_t need = 0;
    for (uint64_t g = g0; g <= g1; g++)
        need += ZXC_SEEK_ANCHOR_SIZE +
                (size_t)zxc_seek_group_len(f->num_blocks, g) * ZXC_SEEK_SIZE_ENTRY;
    // The EOF/SEK headers, checked on each access rather than kept as state; read
    // with group 0 when it is wanted.
    const zxc_seek_source_t src = zxc_seek_source_of(s);
    const uint64_t at = f->table_off + g0 * ZXC_SEEK_GROUP_BYTES;
    int rc = g0 == 0 ? zxc_seek_source_read(&src, raw, ZXC_SEEK_TAIL_HDRS + need,
                                            at - ZXC_SEEK_TAIL_HDRS)
                     : zxc_seek_source_read(&src, raw, ZXC_SEEK_TAIL_HDRS,
                                            f->table_off - ZXC_SEEK_TAIL_HDRS);
    if (rc == ZXC_OK && g0 != 0)
        rc = zxc_seek_source_read(&src, raw + ZXC_SEEK_TAIL_HDRS, need, at);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(zxc_check_eof_header(raw) != ZXC_OK ||
                 zxc_check_seek_header(raw + ZXC_BLOCK_HEADER_SIZE, ZXC_BLOCK_HEADER_SIZE,
                                       f->total_decomp, f->block_size, NULL) != ZXC_OK))
        return ZXC_ERROR_CORRUPT_DATA;

    const uint8_t* p = raw + ZXC_SEEK_TAIL_HDRS;
    for (uint64_t g = g0; g <= g1; g++) {
        uint64_t pos = zxc_le64(p);
        p += ZXC_SEEK_ANCHOR_SIZE;
        if (UNLIKELY(g == 0 ? pos != ZXC_FILE_HEADER_SIZE
                            : pos < ZXC_FILE_HEADER_SIZE || pos > f->eof_off))
            return ZXC_ERROR_CORRUPT_DATA;

        const uint32_t cnt = zxc_seek_group_len(f->num_blocks, g);
        for (uint32_t k = 0; k < cnt; k++, p += ZXC_SEEK_SIZE_ENTRY) {
            const uint32_t sz = zxc_le32(p);
            if (UNLIKELY(sz < ZXC_BLOCK_HEADER_SIZE || sz > f->entry_max))
                return ZXC_ERROR_CORRUPT_DATA;
            const uint64_t idx = g * ZXC_SEEK_GROUP + k;
            if (idx >= first && idx < first + n) {
                starts[idx - first] = f->base + pos;
                sizes[idx - first] = sz;
            }
            pos += sz;  // no wrap: at most ZXC_SEEK_GROUP sizes of at most entry_max
        }
        // Within the data area; the last group, exactly on the EOF block.
        if (UNLIKELY(pos > f->eof_off || (g == last_group && pos != f->eof_off)))
            return ZXC_ERROR_CORRUPT_DATA;
    }
    return ZXC_OK;
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
 * Public API; see @c zxc_seekable.h. Reads the file header, footer and the
 * EOF/SEK block headers through @p r->read_at (the FILE* variant wraps @c pread
 * this way). The archive is never mapped whole; entries are read as blocks are
 * accessed.
 */
zxc_seekable* zxc_seekable_open_reader(const zxc_reader_t* r) {
    if (UNLIKELY(!r || !r->read_at || r->size == 0)) return NULL;
    const zxc_seek_source_t source = {NULL, r, r->size};
    return zxc_seekable_parse(&source);
}

/**
 * @brief Number of blocks in the archive.
 */
uint64_t zxc_seekable_get_num_blocks(const zxc_seekable* s) { return s ? s->num_blocks : 0; }

/**
 * @brief Total decompressed size of the archive.
 */
uint64_t zxc_seekable_get_decompressed_size(const zxc_seekable* s) {
    return s ? s->total_decomp : 0;
}

/**
 * @brief Compressed byte size of a given block, from its seek table entry.
 */
uint32_t zxc_seekable_get_block_comp_size(const zxc_seekable* s, const uint64_t block_idx) {
    if (UNLIKELY(!s || block_idx >= s->num_blocks)) return 0;
    const zxc_seek_frame_t* const f = zxc_seek_frame_of_block(s, block_idx);
    uint64_t start = 0;
    uint32_t size = 0;
    uint8_t raw[ZXC_SEEK_TAIL_HDRS + ZXC_SEEK_GROUP_BYTES];
    if (UNLIKELY(zxc_seek_load_spans(s, f, block_idx - f->block_base, 1, &start, &size, raw) !=
                 ZXC_OK))
        return 0;
    return size;
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
                                     const uint64_t idx) {
    const uint64_t start = idx * (uint64_t)block_size;
    const uint64_t remaining = total_decomp - start;
    return (remaining >= (uint64_t)block_size) ? block_size : (uint32_t)remaining;
}

/** @brief Decompressed byte size of a given block. */
uint32_t zxc_seekable_get_block_decomp_size(const zxc_seekable* s, const uint64_t block_idx) {
    if (UNLIKELY(!s || block_idx >= s->num_blocks)) return 0;
    const zxc_seek_frame_t* const f = zxc_seek_frame_of_block(s, block_idx);
    return zxc_seek_decomp_size(f->block_size, f->total_decomp, block_idx - f->block_base);
}

// =========================================================================
// Random-Access Decompression
// =========================================================================

/**
 * @brief Gives @p ctx frame @p f's dictionary, copied in unless @p *held, the one
 *        it holds, is already that one.
 *
 * @return ZXC_OK, or @ref ZXC_ERROR_DICT_REQUIRED when it is not installed.
 */
static int zxc_seek_use_dict(const zxc_seekable* s, zxc_cctx_t* ctx, const zxc_seek_dict_t** held,
                             const zxc_seek_frame_t* f) {
    const zxc_seek_dict_t* d = NULL;
    for (size_t i = 0; f->dict_id && i < s->num_dicts && !d; i++)
        if (s->dicts[i].id == f->dict_id) d = &s->dicts[i];
    if (UNLIKELY(f->dict_id && !d)) return ZXC_ERROR_DICT_REQUIRED;
    if (d != *held) {
        if (d) ZXC_MEMCPY(ctx->dict_buffer, d->data, d->size);
        if (UNLIKELY(zxc_cctx_attach_dict_huf(ctx, d && d->has_huf ? d->huf : NULL) != ZXC_OK))
            return ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE
        *held = d;
    }
    ctx->dict_size = d ? d->size : 0;
    return ZXC_OK;
}

/**
 * @brief Reads a compressed block into @p buf from the memory buffer or reader.
 *
 * @p off and @p csz come from @ref zxc_seek_load_spans. What @p off points at must parse
 * as a block header, checksum included, and agree with @p csz. An entry pointing into a
 * block is refused; one moved onto a block of the same size is not, and only the
 * position-seeded checksum catches that.
 *
 * @param[in]  s        Seekable handle.
 * @param[in]  off      Byte offset of the block in the archive.
 * @param[in]  csz      On-disk size of the block, at least a header.
 * @param[in]  has_cs   Its frame carries block checksums.
 * @param[out] buf      Destination buffer.
 * @param[in]  buf_cap  Capacity of @p buf in bytes.
 * @return @p csz, or a negative @ref zxc_error_t (@ref ZXC_ERROR_DST_TOO_SMALL,
 *         @ref ZXC_ERROR_SRC_TOO_SMALL, @ref ZXC_ERROR_IO, @ref ZXC_ERROR_BAD_HEADER,
 *         @ref ZXC_ERROR_CORRUPT_DATA).
 */
static int zxc_seek_read_block(const zxc_seekable* s, const uint64_t off, const uint32_t csz,
                               const int has_cs, uint8_t* buf, const size_t buf_cap) {
    if (UNLIKELY(csz > buf_cap)) return ZXC_ERROR_DST_TOO_SMALL;
    const zxc_seek_source_t src = zxc_seek_source_of(s);
    const int rc = zxc_seek_source_read(&src, buf, csz, off);
    if (UNLIKELY(rc != ZXC_OK)) return rc;

    zxc_block_header_t bh;
    const int hdr_res = zxc_read_block_header(buf, csz, &bh);
    if (UNLIKELY(hdr_res != ZXC_OK)) return hdr_res;
    const uint64_t on_disk =
        (uint64_t)ZXC_BLOCK_HEADER_SIZE + bh.comp_size + (has_cs ? ZXC_BLOCK_CHECKSUM_SIZE : 0U);
    if (UNLIKELY(on_disk != csz)) return ZXC_ERROR_CORRUPT_DATA;
    return (int)csz;
}

/**
 * @brief Decompresses the byte range [@p offset, @p offset + @p len) into @p dst.
 *
 * Public API; full contract in @c zxc_seekable.h. Maps the range to its blocks
 * (a frame lookup, then a division), decodes each through a reusable,
 * lazily-initialised, dictionary-aware context, and copies out only the
 * requested sub-range. Single-threaded; see @ref zxc_seekable_decompress_range_mt
 * for the parallel variant.
 */
int64_t zxc_seekable_decompress_range(zxc_seekable* s, void* dst, const size_t dst_capacity,
                                      const uint64_t offset, const size_t len) {
    if (UNLIKELY(len == 0)) return 0;
    if (UNLIKELY(!s || !dst)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(dst_capacity < len)) return ZXC_ERROR_DST_TOO_SMALL;
    if (UNLIKELY(offset > s->total_decomp || len > s->total_decomp - offset))
        return ZXC_ERROR_SRC_TOO_SMALL;

    // Initialize decompression context on first use.
    if (!s->dctx_initialized) {
        if (UNLIKELY(zxc_cctx_init(&s->dctx, (size_t)s->max_block_size, 0, 0, 0,
                                   s->max_dict_size) != ZXC_OK))
            return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        s->dctx_initialized = 1;
        s->dctx_dict = NULL;
    }

    const uint64_t blk_start = zxc_seek_block_of_offset(s, offset);
    const uint64_t blk_end = zxc_seek_block_of_offset(s, offset + len - 1);
    const zxc_seek_frame_t* f = zxc_seek_frame_of_block(s, blk_start);

    uint8_t* out = (uint8_t*)dst;
    size_t remaining = len;

    // One slice, and one read, per table group of a frame.
    uint64_t starts[ZXC_SEEK_GROUP] = {0};
    uint32_t sizes[ZXC_SEEK_GROUP] = {0};
    uint8_t raw[ZXC_SEEK_TAIL_HDRS + ZXC_SEEK_GROUP_BYTES];
    uint64_t bi = blk_start;
    while (bi <= blk_end) {
        while (bi >= f->block_base + f->num_blocks) f++;  // past this frame, empty ones too
        const uint64_t local = bi - f->block_base;
        const uint64_t left = blk_end - bi + 1;
        const uint64_t in_frame = f->num_blocks - local;
        const uint32_t to_group_end = ZXC_SEEK_GROUP - (uint32_t)(local % ZXC_SEEK_GROUP);
        uint32_t n = left < to_group_end ? (uint32_t)left : to_group_end;
        if (in_frame < n) n = (uint32_t)in_frame;
        const int span_res = zxc_seek_load_spans(s, f, local, n, starts, sizes, raw);
        if (UNLIKELY(span_res < 0)) return span_res;

        // Block scratch kept on the handle, sized to the slice's largest block, not entry_max.
        uint64_t need = 0;
        for (uint32_t k = 0; k < n; k++)
            if (sizes[k] > need) need = sizes[k];
        need += ZXC_PAD_SIZE;
        if (s->read_buf_cap < need) {
            uint8_t* const nb = (uint8_t*)ZXC_REALLOC(s->read_buf, (size_t)need);
            if (UNLIKELY(!nb)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
            s->read_buf = nb;
            s->read_buf_cap = (size_t)need;
        }
        uint8_t* const read_buf = s->read_buf;

        // The frame's geometry and dictionary: the context is sized for the largest.
        const int dict_res = zxc_seek_use_dict(s, &s->dctx, &s->dctx_dict, f);
        if (UNLIKELY(dict_res != ZXC_OK)) return dict_res;
        s->dctx.chunk_size = f->block_size;
        s->dctx.checksum_enabled = f->has_checksums && s->verify_checksums;
        const size_t work_sz = (size_t)f->block_size + ZXC_DECOMPRESS_TAIL_PAD;

        for (uint32_t k = 0; k < n; k++, bi++) {
            const uint64_t lb = local + k;  // position in the frame: the checksum seed
            const int read_res = zxc_seek_read_block(s, starts[k], sizes[k], f->has_checksums,
                                                     read_buf, s->read_buf_cap);
            if (UNLIKELY(read_res < 0)) return read_res;

            // Decompress the block: when a dictionary is active, decode into the
            // cctx-owned dict_buffer (which has dict content prepended) so that
            // match copies referencing dictionary bytes resolve naturally.
            uint8_t* dec_dst =
                s->dctx.dict_buffer ? s->dctx.dict_buffer + s->dctx.dict_size : s->dctx.work_buf;
            const int dec_res = zxc_decompress_chunk_wrapper(&s->dctx, read_buf, (size_t)read_res,
                                                             dec_dst, work_sz, lb);
            if (UNLIKELY(dec_res < 0)) return dec_res;
            if (UNLIKELY((uint32_t)dec_res !=
                         zxc_seek_decomp_size(f->block_size, f->total_decomp, lb)))
                return ZXC_ERROR_CORRUPT_DATA;

            // Calculate which portion of this block's decompressed data we need
            const uint64_t blk_decomp_start = f->decomp_base + lb * f->block_size;
            const size_t skip =
                (offset > blk_decomp_start) ? (size_t)(offset - blk_decomp_start) : 0;
            const size_t avail = (size_t)dec_res - skip;
            const size_t copy = (avail < remaining) ? avail : remaining;

            ZXC_MEMCPY(out, dec_dst + skip, copy);
            out += copy;
            remaining -= copy;
        }
    }

    if (UNLIKELY(remaining != 0)) return ZXC_ERROR_CORRUPT_DATA;
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
    uint64_t block_idx;  /* position in its frame: the checksum seed */
    uint64_t off;        /* where it starts in the archive (validated span) */
    size_t dst_off;      /* output offset from the caller's buffer start */
    size_t skip;         /* bytes to skip at start of decompressed block */
    size_t copy_len;     /* bytes to copy out */
    uint32_t csz;        /* its on-disk size */
    uint32_t block_size; /* its frame's */
    const zxc_seek_frame_t* frame;
    uint32_t decomp_sz; /* what it decodes to */
    int has_cs;         /* its frame carries block checksums */
    int result;         /* 0 = OK, < 0 = error; starts negative, see the launch loop */
} zxc_seek_mt_job_t;

/**
 * @struct zxc_seek_mt_shared_t
 * @brief State shared by every worker of one multi-threaded range read.
 *
 * Worker @c k owns the job subset {k, k+stride, k+2*stride, ...} and reuses one
 * decompression context, one dictionary copy and one read buffer across all of
 * them. Stripes are pairwise disjoint, so workers touch distinct jobs and
 * distinct output ranges and need no synchronisation beyond claiming their
 * index at start-up and the final join.
 *
 * Lives on the caller's stack: the caller's output pointer therefore never
 * reaches heap memory (jobs carry offsets into it), and there is no per-thread
 * array to size for @c ZXC_MAX_THREADS.
 */
typedef struct {
    const zxc_seekable* s;   /* handle being read (read-only) */
    zxc_seek_mt_job_t* jobs; /* job array; each worker touches only its stripe */
    uint8_t* dst_base;       /* start of the caller's output buffer */
    uint32_t num_jobs;       /* job count (stripe iteration bound) */
    uint32_t stride;         /* worker count attempted; stripes partition by it */
    uint32_t next;           /* next stripe index to hand out, under lock */
    pthread_mutex_t lock;    /* guards next */
} zxc_seek_mt_shared_t;

/**
 * @brief Marks every job of stripe @p first with @p code (setup-failure path).
 *
 * @param[in,out] sh    Shared state.
 * @param[in]     first Stripe index.
 * @param[in]     code  Negative @ref zxc_error_t value.
 */
static void zxc_seek_mt_fail_stripe(const zxc_seek_mt_shared_t* sh, const uint32_t first,
                                    const int code) {
    for (uint32_t i = first; i < sh->num_jobs; i += sh->stride) sh->jobs[i].result = code;
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
 * @param[in,out] arg  Pointer to the read's `zxc_seek_mt_shared_t`.
 * @return Always NULL (result codes are reported via the jobs).
 */
static void* zxc_seek_mt_worker(void* arg) {
    zxc_seek_mt_shared_t* const sh = (zxc_seek_mt_shared_t*)arg;
    zxc_seek_mt_job_t* const jobs = sh->jobs;
    const zxc_seekable* const s = sh->s;

    // Claim a stripe: the main thread hands every worker the same argument.
    pthread_mutex_lock(&sh->lock);
    const uint32_t first = sh->next++;
    pthread_mutex_unlock(&sh->lock);

    // Thread-local decompression context (mode=0 for decompress-only)
    zxc_cctx_t dctx;
    if (UNLIKELY(zxc_cctx_init(&dctx, (size_t)s->max_block_size, 0, 0, 0, s->max_dict_size) !=
                 ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_seek_mt_fail_stripe(sh, first, ZXC_ERROR_MEMORY);
        return NULL;
        // LCOV_EXCL_STOP
    }
    const zxc_seek_dict_t* held = NULL;

    // Read buffer sized for the largest compressed block of the stripe.
    size_t max_csz = 0;
    for (uint32_t i = first; i < sh->num_jobs; i += sh->stride) {
        if (jobs[i].csz > max_csz) max_csz = jobs[i].csz;
    }
    uint8_t* const read_buf = (uint8_t*)ZXC_MALLOC(max_csz + ZXC_PAD_SIZE);
    if (UNLIKELY(!read_buf)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&dctx);
        zxc_seek_mt_fail_stripe(sh, first, ZXC_ERROR_MEMORY);
        return NULL;
        // LCOV_EXCL_STOP
    }

    for (uint32_t i = first; i < sh->num_jobs; i += sh->stride) {
        zxc_seek_mt_job_t* const job = &jobs[i];

        const int read_res = zxc_seek_read_block(s, job->off, job->csz, job->has_cs, read_buf,
                                                 max_csz + ZXC_PAD_SIZE);
        if (UNLIKELY(read_res < 0)) {
            job->result = read_res;
            break;
        }

        const int dict_res = zxc_seek_use_dict(s, &dctx, &held, job->frame);
        if (UNLIKELY(dict_res != ZXC_OK)) {
            job->result = dict_res;
            break;
        }
        uint8_t* dec_dst = dctx.dict_buffer ? dctx.dict_buffer + dctx.dict_size : dctx.work_buf;
        dctx.chunk_size = job->block_size;
        dctx.checksum_enabled = job->has_cs && s->verify_checksums;
        const int dec_res = zxc_decompress_chunk_wrapper(
            &dctx, read_buf, (size_t)read_res, dec_dst,
            (size_t)job->block_size + ZXC_DECOMPRESS_TAIL_PAD, job->block_idx);

        if (UNLIKELY(dec_res < 0)) {
            job->result = dec_res;
            break;
        }
        if (UNLIKELY((uint32_t)dec_res != job->decomp_sz)) {
            job->result = ZXC_ERROR_CORRUPT_DATA;
            break;
        }

        // Copy the requested portion directly into the caller's output buffer
        ZXC_MEMCPY(sh->dst_base + job->dst_off, dec_dst + job->skip, job->copy_len);
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
    if (UNLIKELY(offset > s->total_decomp || len > s->total_decomp - offset))
        return ZXC_ERROR_SRC_TOO_SMALL;

    const uint64_t blk_start = zxc_seek_block_of_offset(s, offset);
    const uint64_t blk_end = zxc_seek_block_of_offset(s, offset + len - 1);
    const uint64_t jobs_64 = blk_end - blk_start + 1;

    // Auto-detect thread count (0 = use all available cores)
    if (n_threads == 0) n_threads = zxc_num_procs();

    // Fallback to single-threaded path for trivial cases, and for a range holding
    // more blocks than one job table can index (4 G blocks is 16 TiB of dst).
    if (n_threads <= 1 || jobs_64 <= 1 || jobs_64 > UINT32_MAX) {
        return zxc_seekable_decompress_range(s, dst, dst_capacity, offset, len);
    }
    const uint32_t num_jobs = (uint32_t)jobs_64;

    // Cap threads to number of blocks and max limit
    if ((uint32_t)n_threads > num_jobs) n_threads = (int)num_jobs;
    if (n_threads > ZXC_MAX_THREADS) n_threads = ZXC_MAX_THREADS;

    zxc_seek_mt_job_t* const jobs =
        (zxc_seek_mt_job_t*)ZXC_CALLOC(num_jobs, sizeof(zxc_seek_mt_job_t));
    uint64_t* const starts = (uint64_t*)ZXC_CALLOC(num_jobs, sizeof(uint64_t));
    uint32_t* const sizes = (uint32_t*)ZXC_CALLOC(num_jobs, sizeof(uint32_t));
    if (UNLIKELY(!jobs || !starts || !sizes)) {
        // LCOV_EXCL_START
        ZXC_FREE(jobs);
        ZXC_FREE(starts);
        ZXC_FREE(sizes);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    // Plan jobs frame by frame, each frame's table groups in one read.
    const zxc_seek_frame_t* f = zxc_seek_frame_of_block(s, blk_start);
    size_t out_off = 0;
    size_t remaining = len;
    int plan_res = ZXC_OK;
    for (uint32_t i = 0; i < num_jobs && plan_res == ZXC_OK;) {
        const uint64_t bi = blk_start + i;
        while (bi >= f->block_base + f->num_blocks) f++;  // past this frame, empty ones too
        const uint64_t local = bi - f->block_base;
        const uint64_t in_frame = f->num_blocks - local;
        const uint32_t n = in_frame < num_jobs - i ? (uint32_t)in_frame : num_jobs - i;
        uint8_t* const raw = (uint8_t*)ZXC_MALLOC(zxc_seek_spans_raw_max(local, n));
        plan_res = raw ? zxc_seek_load_spans(s, f, local, n, starts + i, sizes + i, raw)
                       : ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        ZXC_FREE(raw);
        for (uint32_t k = 0; k < n && plan_res == ZXC_OK; k++, i++) {
            const uint64_t lb = local + k;
            const uint64_t blk_decomp_start = f->decomp_base + lb * f->block_size;
            const size_t skip =
                (offset > blk_decomp_start) ? (size_t)(offset - blk_decomp_start) : 0;
            const uint32_t blk_decomp_sz = zxc_seek_decomp_size(f->block_size, f->total_decomp, lb);
            const size_t avail = blk_decomp_sz - skip;
            const size_t copy = (avail < remaining) ? avail : remaining;

            jobs[i].block_idx = lb;
            jobs[i].off = starts[i];
            jobs[i].csz = sizes[i];
            jobs[i].block_size = f->block_size;
            jobs[i].frame = f;
            jobs[i].decomp_sz = blk_decomp_sz;
            jobs[i].has_cs = f->has_checksums;
            jobs[i].dst_off = out_off;
            jobs[i].skip = skip;
            jobs[i].copy_len = copy;
            // Negative until a worker completes it: a stripe whose thread failed to
            // start is then reported without anyone having to know which one.
            jobs[i].result = ZXC_ERROR_MEMORY;

            out_off += copy;
            remaining -= copy;
        }
    }
    ZXC_FREE(starts);
    ZXC_FREE(sizes);
    if (UNLIKELY(plan_res != ZXC_OK)) {
        ZXC_FREE(jobs);
        return plan_res;
    }

    // Launch one persistent worker per thread
    pthread_t* const threads = (pthread_t*)ZXC_MALLOC((size_t)n_threads * sizeof(pthread_t));
    if (UNLIKELY(!threads)) {
        // LCOV_EXCL_START
        ZXC_FREE(jobs);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    zxc_seek_mt_shared_t sh;
    sh.s = s;
    sh.jobs = jobs;
    sh.dst_base = (uint8_t*)dst;
    sh.num_jobs = num_jobs;
    sh.stride = (uint32_t)n_threads;
    sh.next = 0;
    pthread_mutex_init(&sh.lock, NULL);

    int launched = 0;
    for (int t = 0; t < n_threads; t++) {
        // A failed start leaves one stripe unclaimed; its jobs keep their
        // negative result and the read reports it after the join.
        if (UNLIKELY(pthread_create(&threads[launched], NULL, zxc_seek_mt_worker, &sh) != 0))
            continue;  // LCOV_EXCL_LINE
        launched++;
    }

    // Join phase
    for (int t = 0; t < launched; t++) pthread_join(threads[t], NULL);

    pthread_mutex_destroy(&sh.lock);
    ZXC_FREE(threads);

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
 * Public API; see @c zxc_seekable.h. Tears down the reusable context, the
 * owned dictionary copy and any attached reader context. NULL-safe.
 */
void zxc_seekable_free(zxc_seekable* s) {
    if (UNLIKELY(!s)) return;
    if (s->dctx_initialized) zxc_cctx_free(&s->dctx);
    ZXC_FREE(s->frames);
    for (size_t i = 0; i < s->num_dicts; i++) ZXC_FREE(s->dicts[i].data);
    ZXC_FREE(s->dicts);
    ZXC_FREE(s->read_buf);
    ZXC_FREE(s->owned_reader_ctx);
    ZXC_FREE(s);
}

/**
 * @brief Turns per-block checksum verification on or off.
 *
 * Public API; see @c zxc_seekable.h. Only records the wish; the decode path
 * pushes it into the context on every call, like @c dict_size.
 */
int zxc_seekable_set_checksum(zxc_seekable* s, const int enabled) {
    if (UNLIKELY(!s)) return ZXC_ERROR_NULL_INPUT;
    s->verify_checksums = enabled ? 1 : 0;
    return ZXC_OK;
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
    const uint32_t id = zxc_dict_id(dict, dict_size, (const uint8_t*)dict_huf);
    int used = 0;
    int any = 0;
    for (size_t i = 0; i < s->num_frames && !used; i++) {
        used = s->frames[i].dict_id == id;
        any |= s->frames[i].dict_id != 0;
    }
    // No frame needs a dictionary: ignore it, as before. Otherwise one must use it.
    if (!any) return ZXC_OK;
    if (UNLIKELY(!used)) return ZXC_ERROR_DICT_MISMATCH;

    zxc_seek_dict_t* d = NULL;
    for (size_t i = 0; i < s->num_dicts && !d; i++)
        if (s->dicts[i].id == id) d = &s->dicts[i];
    if (!d) {
        zxc_seek_dict_t* const grown =
            (zxc_seek_dict_t*)ZXC_REALLOC(s->dicts, (s->num_dicts + 1) * sizeof(*s->dicts));
        if (UNLIKELY(!grown)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
        s->dicts = grown;
        d = &s->dicts[s->num_dicts];
        ZXC_MEMSET(d, 0, sizeof(*d));
    }
    uint8_t* const copy = (uint8_t*)ZXC_MALLOC(dict_size);
    if (UNLIKELY(!copy)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
    ZXC_MEMCPY(copy, dict, dict_size);
    if (d == &s->dicts[s->num_dicts]) s->num_dicts++;
    ZXC_FREE(d->data);
    d->data = copy;
    d->size = dict_size;
    d->id = id;
    d->has_huf = dict_huf != NULL;
    if (dict_huf) ZXC_MEMCPY(d->huf, dict_huf, ZXC_HUF_TABLE_SIZE);
    if (dict_size > s->max_dict_size) s->max_dict_size = dict_size;

    // The context's [dict | decode] buffer and the table it holds are re-carved
    // and reloaded on the next decompress.
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
