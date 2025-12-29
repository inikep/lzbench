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

/*
 * ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================
 */

void* zxc_aligned_malloc(size_t size, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#endif
}

void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

int zxc_cctx_init(zxc_cctx_t* ctx, size_t chunk_size, int mode, int level, int checksum_enabled) {
    ZXC_MEMSET(ctx, 0, sizeof(zxc_cctx_t));

    if (mode == 0) return 0;

    size_t max_seq = chunk_size / 4 + 256;
    size_t sz_hash = 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
    size_t sz_chain = chunk_size * sizeof(uint16_t);
    size_t sz_extras = max_seq * sizeof(uint32_t);
    size_t sz_offsets = max_seq * sizeof(uint16_t);
    size_t sz_tokens = max_seq * sizeof(uint8_t);
    size_t sz_lit = chunk_size;

    // Calculate sizes with alignment padding (64 bytes for cache line alignment)
    size_t total_size = 0;
    size_t off_hash = total_size;
    total_size += (sz_hash + 63) & ~63;
    size_t off_chain = total_size;
    total_size += (sz_chain + 63) & ~63;
    size_t off_extras = total_size;
    total_size += (sz_extras + 63) & ~63;
    size_t off_offsets = total_size;
    total_size += (sz_offsets + 63) & ~63;
    size_t off_tokens = total_size;
    total_size += (sz_tokens + 63) & ~63;
    size_t off_lit = total_size;
    total_size += (sz_lit + 63) & ~63;

    uint8_t* mem = (uint8_t*)zxc_aligned_malloc(total_size, 64);
    if (UNLIKELY(!mem)) return -1;

    ctx->memory_block = mem;
    ctx->hash_table = (uint32_t*)(mem + off_hash);
    ctx->chain_table = (uint16_t*)(mem + off_chain);
    ctx->buf_extras = (uint32_t*)(mem + off_extras);
    ctx->buf_offsets = (uint16_t*)(mem + off_offsets);
    ctx->buf_tokens = (uint8_t*)(mem + off_tokens);
    ctx->literals = (uint8_t*)(mem + off_lit);

    ctx->epoch = 1;
    ctx->compression_level = level;
    ctx->checksum_enabled = checksum_enabled;

    ZXC_MEMSET(ctx->hash_table, 0, sz_hash);
    return 0;
}

void zxc_cctx_free(zxc_cctx_t* ctx) {
    if (ctx->memory_block) {
        zxc_aligned_free(ctx->memory_block);
        ctx->memory_block = NULL;
    }

    if (ctx->lit_buffer) {
        free(ctx->lit_buffer);
        ctx->lit_buffer = NULL;
    }

    ctx->hash_table = NULL;
    ctx->chain_table = NULL;
    ctx->buf_extras = NULL;
    ctx->buf_offsets = NULL;
    ctx->buf_tokens = NULL;
    ctx->literals = NULL;

    ctx->lit_buffer_cap = 0;
}

/*
 * ============================================================================
 * CHECKSUM IMPLEMENTATION (XXH3)
 * ============================================================================
 * Uses XXH3 (64-bit) for extreme performance (> 30GB/s).
 */

#define XXH_INLINE_ALL
#include "../../include/xxhash.h"

uint64_t zxc_checksum(const void* data, size_t len) { return XXH3_64bits(data, len); }

/*
 * ============================================================================
 * HEADER I/O
 * ============================================================================
 * Serialization and deserialization of file and block headers.
 */

int zxc_write_file_header(uint8_t* dst, size_t dst_capacity) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return -1;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;
    dst[5] = 0;
    dst[6] = 0;
    dst[7] = 0;
    return ZXC_FILE_HEADER_SIZE;
}

int zxc_read_file_header(const uint8_t* src, size_t src_size) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE)) return -1;
    if (UNLIKELY(zxc_le32(src) != ZXC_MAGIC_WORD || src[4] != ZXC_FILE_FORMAT_VERSION)) return -1;
    return 0;
}

int zxc_write_block_header(uint8_t* dst, size_t dst_capacity, const zxc_block_header_t* bh) {
    if (UNLIKELY(dst_capacity < ZXC_BLOCK_HEADER_SIZE)) return -1;

    dst[0] = bh->block_type;
    dst[1] = bh->block_flags;
    zxc_store_le16(dst + 2, bh->reserved);
    zxc_store_le32(dst + 4, bh->comp_size);
    zxc_store_le32(dst + 8, bh->raw_size);
    return ZXC_BLOCK_HEADER_SIZE;
}

int zxc_read_block_header(const uint8_t* src, size_t src_size, zxc_block_header_t* bh) {
    if (UNLIKELY(src_size < ZXC_BLOCK_HEADER_SIZE)) return -1;

    bh->block_type = src[0];
    bh->block_flags = src[1];
    bh->reserved = zxc_le16(src + 2);
    bh->comp_size = zxc_le32(src + 4);
    bh->raw_size = zxc_le32(src + 8);
    return 0;
}

/*
 * ============================================================================
 * BITPACKING UTILITIES
 * ============================================================================
 */

void zxc_br_init(zxc_bit_reader_t* br, const uint8_t* src, size_t size) {
    br->ptr = src;
    br->end = src + size;
    br->accum = zxc_le64(br->ptr);
    br->ptr += 8;
    br->bits = 64;
}

// Packs an array of 32-bit integers into a bitstream using 'bits' bits per
// integer.

int zxc_bitpack_stream_32(const uint32_t* RESTRICT src, size_t count, uint8_t* RESTRICT dst,
                          size_t dst_cap, uint8_t bits) {
    size_t out_bytes = ((count * bits) + 7) / 8;

    if (UNLIKELY(dst_cap < out_bytes)) return -1;

    size_t bit_pos = 0;
    ZXC_MEMSET(dst, 0, out_bytes);

    for (size_t i = 0; i < count; i++) {
        uint64_t v = (uint64_t)src[i] << (bit_pos % 8);
        size_t byte_idx = bit_pos / 8;
        dst[byte_idx] |= (uint8_t)v;
        if (bits + (bit_pos % 8) > 8) dst[byte_idx + 1] |= (uint8_t)(v >> 8);
        if (bits + (bit_pos % 8) > 16) dst[byte_idx + 2] |= (uint8_t)(v >> 16);
        if (bits + (bit_pos % 8) > 24) dst[byte_idx + 3] |= (uint8_t)(v >> 24);
        if (bits + (bit_pos % 8) > 32) dst[byte_idx + 4] |= (uint8_t)(v >> 32);
        bit_pos += bits;
    }
    return (int)out_bytes;
}

int zxc_write_num_header(uint8_t* dst, size_t rem, const zxc_num_header_t* nh) {
    if (UNLIKELY(rem < ZXC_NUM_HEADER_BINARY_SIZE)) return -1;

    zxc_store_le64(dst, nh->n_values);
    zxc_store_le16(dst + 8, nh->frame_size);
    zxc_store_le16(dst + 10, 0);
    zxc_store_le32(dst + 12, 0);
    return ZXC_NUM_HEADER_BINARY_SIZE;
}

int zxc_read_num_header(const uint8_t* src, size_t src_size, zxc_num_header_t* nh) {
    if (UNLIKELY(src_size < ZXC_NUM_HEADER_BINARY_SIZE)) return -1;

    nh->n_values = zxc_le64(src);
    nh->frame_size = zxc_le16(src + 8);
    return 0;
}

int zxc_write_gnr_header_and_desc(uint8_t* dst, size_t rem, const zxc_gnr_header_t* gh,
                                  const zxc_section_desc_t desc[4]) {
    size_t needed = ZXC_GNR_HEADER_BINARY_SIZE + 4 * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return -1;

    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_litlen;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;

    zxc_store_le32(dst + 12, 0);
    uint8_t* p = dst + ZXC_GNR_HEADER_BINARY_SIZE;

    for (int i = 0; i < 4; i++) {
        zxc_store_le64(p, desc[i].sizes);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }

    return (int)needed;
}

int zxc_read_gnr_header_and_desc(const uint8_t* src, size_t len, zxc_gnr_header_t* gh,
                                 zxc_section_desc_t desc[4]) {
    size_t needed = ZXC_GNR_HEADER_BINARY_SIZE + 4 * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return -1;

    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_litlen = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];

    const uint8_t* p = src + ZXC_GNR_HEADER_BINARY_SIZE;

    for (int i = 0; i < 4; i++) {
        desc[i].sizes = zxc_le64(p);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }
    return 0;
}

size_t zxc_compress_bound(size_t input_size) {
    if (input_size > SIZE_MAX - (SIZE_MAX >> 10)) return 0;

    size_t n = (input_size + ZXC_CHUNK_SIZE - 1) / ZXC_CHUNK_SIZE;
    if (n == 0) n = 1;
    return ZXC_FILE_HEADER_SIZE + (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + 64)) +
           input_size;
}
