/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_sans_io.h"
#include "zxc_internal.h"

/*
 * ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================
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

void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

int zxc_cctx_init(zxc_cctx_t* RESTRICT ctx, const size_t chunk_size, const int mode,
                  const int level, const int checksum_enabled) {
    ZXC_MEMSET(ctx, 0, sizeof(zxc_cctx_t));

    ctx->compression_level = level;
    ctx->checksum_enabled = checksum_enabled;

    if (mode == 0) return 0;

    size_t max_seq = chunk_size / sizeof(uint32_t) + 256;
    size_t sz_hash = 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
    size_t sz_chain = chunk_size * sizeof(uint16_t);
    size_t sz_sequences = max_seq * sizeof(uint32_t);
    size_t sz_tokens = max_seq * sizeof(uint8_t);
    size_t sz_offsets = max_seq * sizeof(uint16_t);
    size_t sz_extras =
        max_seq * 2 *
        ZXC_VBYTE_ALLOC_LEN;  // Max 3 bytes per LL/ML VByte (sufficient for 256KB block)
    size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

    // Calculate sizes with alignment padding (64 bytes for cache line alignment)
    size_t total_size = 0;
    size_t off_hash = total_size;
    total_size += (sz_hash + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_chain = total_size;
    total_size += (sz_chain + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_sequences = total_size;
    total_size += (sz_sequences + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_tokens = total_size;
    total_size += (sz_tokens + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_offsets = total_size;
    total_size += (sz_offsets + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_extras = total_size;
    total_size += (sz_extras + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    size_t off_lit = total_size;
    total_size += (sz_lit + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    uint8_t* mem = (uint8_t*)zxc_aligned_malloc(total_size, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem)) return -1;

    ctx->memory_block = mem;
    ctx->hash_table = (uint32_t*)(mem + off_hash);
    ctx->chain_table = (uint16_t*)(mem + off_chain);
    ctx->buf_sequences = (uint32_t*)(mem + off_sequences);
    ctx->buf_tokens = (uint8_t*)(mem + off_tokens);
    ctx->buf_offsets = (uint16_t*)(mem + off_offsets);
    ctx->buf_extras = (uint8_t*)(mem + off_extras);
    ctx->literals = (uint8_t*)(mem + off_lit);

    ctx->epoch = 1;

    ZXC_MEMSET(ctx->hash_table, 0, sz_hash);
    return 0;
}

void zxc_cctx_free(zxc_cctx_t* ctx) {
    if (ctx->memory_block) {
        zxc_aligned_free(ctx->memory_block);
        ctx->memory_block = NULL;
    }

    if (ctx->lit_buffer) {
        free(ctx->lit_buffer);  // Use standard free (allocated with realloc)
        ctx->lit_buffer = NULL;
    }

    ctx->hash_table = NULL;
    ctx->chain_table = NULL;
    ctx->buf_sequences = NULL;
    ctx->buf_tokens = NULL;
    ctx->buf_offsets = NULL;
    ctx->buf_extras = NULL;
    ctx->literals = NULL;

    ctx->epoch = 0;
    ctx->lit_buffer_cap = 0;
}

/*
 * ============================================================================
 * HEADER I/O
 * ============================================================================
 */

int zxc_write_file_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                          const int has_checksum) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return -1;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;
    dst[5] = (uint8_t)(ZXC_BLOCK_SIZE / ZXC_BLOCK_UNIT);
    dst[6] = has_checksum ? (ZXC_FILE_FLAG_HAS_CHECKSUM | ZXC_CHECKSUM_RAPIDHASH) : 0;

    // Bytes 7-13: Reserved (must be 0)
    ZXC_MEMSET(dst + 7, 0, 7);

    // Bytes 14-15: CRC (16-bit)
    zxc_store_le16(dst + 14, 0);  // Zero out before hashing
    uint16_t crc = zxc_hash16(dst);
    zxc_store_le16(dst + 14, crc);

    return ZXC_FILE_HEADER_SIZE;
}

int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size,
                         size_t* RESTRICT out_block_size, int* RESTRICT out_has_checksum) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE || zxc_le32(src) != ZXC_MAGIC_WORD ||
                 src[4] != ZXC_FILE_FORMAT_VERSION))
        return -1;

    uint8_t temp[ZXC_FILE_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_FILE_HEADER_SIZE);
    // Zero out CRC bytes (14-15) before hash check
    temp[14] = 0;
    temp[15] = 0;
    if (UNLIKELY(zxc_le16(src + 14) != zxc_hash16(temp))) return -1;

    if (out_block_size) {
        const size_t units = src[5] ? src[5] : 64;  // Default to 64 block units (256KB)
        *out_block_size = units * ZXC_BLOCK_UNIT;
    }
    if (out_has_checksum) *out_has_checksum = (src[6] & ZXC_FILE_FLAG_HAS_CHECKSUM) ? 1 : 0;

    return 0;
}

int zxc_write_block_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                           const zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(dst_capacity < ZXC_BLOCK_HEADER_SIZE)) return -1;

    dst[0] = bh->block_type;
    dst[1] = 0;  // Flags not used currently
    dst[2] = 0;  // Reserved
    zxc_store_le32(dst + 3, bh->comp_size);
    dst[7] = 0;               // Zero before hashing
    dst[7] = zxc_hash8(dst);  // Checksum at the end

    return ZXC_BLOCK_HEADER_SIZE;
}

int zxc_read_block_header(const uint8_t* RESTRICT src, const size_t src_size,
                          zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(src_size < ZXC_BLOCK_HEADER_SIZE)) return -1;

    uint8_t temp[ZXC_BLOCK_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_BLOCK_HEADER_SIZE);
    temp[7] = 0;  // Zero out checksum byte before hashing
    if (UNLIKELY(src[7] != zxc_hash8(temp))) return -1;

    bh->block_type = src[0];
    bh->block_flags = 0;  // Flags not used currently
    bh->reserved = src[2];
    bh->comp_size = zxc_le32(src + 3);
    bh->header_crc = src[7];
    return 0;
}

int zxc_write_file_footer(uint8_t* RESTRICT dst, const size_t dst_capacity, const uint64_t src_size,
                          const uint32_t global_hash, const int checksum_enabled) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_FOOTER_SIZE)) return -1;

    zxc_store_le64(dst, src_size);

    if (checksum_enabled) {
        zxc_store_le32(dst + sizeof(uint64_t), global_hash);
    } else {
        ZXC_MEMSET(dst + sizeof(uint64_t), 0, sizeof(uint32_t));
    }

    return ZXC_FILE_FOOTER_SIZE;
}

int zxc_write_num_header(uint8_t* RESTRICT dst, const size_t rem,
                         const zxc_num_header_t* RESTRICT nh) {
    if (UNLIKELY(rem < ZXC_NUM_HEADER_BINARY_SIZE)) return -1;

    zxc_store_le64(dst, nh->n_values);
    zxc_store_le16(dst + 8, nh->frame_size);
    zxc_store_le16(dst + 10, 0);
    zxc_store_le32(dst + 12, 0);
    return ZXC_NUM_HEADER_BINARY_SIZE;
}

int zxc_read_num_header(const uint8_t* RESTRICT src, const size_t src_size,
                        zxc_num_header_t* RESTRICT nh) {
    if (UNLIKELY(src_size < ZXC_NUM_HEADER_BINARY_SIZE)) return -1;

    nh->n_values = zxc_le64(src);
    nh->frame_size = zxc_le16(src + 8);
    return 0;
}

int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    size_t needed = ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return -1;

    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_litlen;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;

    zxc_store_le32(dst + 12, 0);
    uint8_t* p = dst + ZXC_GLO_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GLO_SECTIONS; i++) {
        zxc_store_le64(p, desc[i].sizes);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }

    return (int)needed;
}

int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    size_t needed = ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return -1;

    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_litlen = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];

    const uint8_t* p = src + ZXC_GLO_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GLO_SECTIONS; i++) {
        desc[i].sizes = zxc_le64(p);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }
    return 0;
}

int zxc_write_ghi_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    size_t needed = ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return -1;

    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_litlen;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;

    zxc_store_le32(dst + 12, 0);
    uint8_t* p = dst + ZXC_GHI_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GHI_SECTIONS; i++) {
        zxc_store_le64(p, desc[i].sizes);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }

    return (int)needed;
}

int zxc_read_ghi_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    size_t needed = ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return -1;

    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_litlen = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];

    const uint8_t* p = src + ZXC_GHI_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GHI_SECTIONS; i++) {
        desc[i].sizes = zxc_le64(p);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }
    return 0;
}

/*
 * ============================================================================
 * BITPACKING UTILITIES
 * ============================================================================
 */

int zxc_bitpack_stream_32(const uint32_t* RESTRICT src, const size_t count, uint8_t* RESTRICT dst,
                          const size_t dst_cap, const uint8_t bits) {
    size_t out_bytes = ((count * bits) + ZXC_BITS_PER_BYTE - 1) / ZXC_BITS_PER_BYTE;

    if (UNLIKELY(dst_cap < out_bytes)) return -1;

    size_t bit_pos = 0;
    ZXC_MEMSET(dst, 0, out_bytes);

    // Create a mask for the input bits to prevent overflow
    // If bits is 32, the shift (1ULL << 32) is undefined behavior on 32-bit types,
    // but here we use uint64_t. (1ULL << 32) is fine on 64-bit.
    // However, if bits=64 (unlikely for a 32-bit packer), it would be an issue.
    // For 0 < bits <= 32:
    uint64_t val_mask =
        (bits == sizeof(uint32_t) * ZXC_BITS_PER_BYTE) ? UINT32_MAX : ((1ULL << bits) - 1);

    for (size_t i = 0; i < count; i++) {
        // Mask the input value to ensure we don't write garbage
        uint64_t v = ((uint64_t)src[i] & val_mask) << (bit_pos % ZXC_BITS_PER_BYTE);

        size_t byte_idx = bit_pos / ZXC_BITS_PER_BYTE;
        dst[byte_idx] |= (uint8_t)v;
        if (bits + (bit_pos % ZXC_BITS_PER_BYTE) > 1 * ZXC_BITS_PER_BYTE)
            dst[byte_idx + 1] |= (uint8_t)(v >> (1 * ZXC_BITS_PER_BYTE));
        if (bits + (bit_pos % ZXC_BITS_PER_BYTE) > 2 * ZXC_BITS_PER_BYTE)
            dst[byte_idx + 2] |= (uint8_t)(v >> (2 * ZXC_BITS_PER_BYTE));
        if (bits + (bit_pos % ZXC_BITS_PER_BYTE) > 3 * ZXC_BITS_PER_BYTE)
            dst[byte_idx + 3] |= (uint8_t)(v >> (3 * ZXC_BITS_PER_BYTE));
        if (bits + (bit_pos % ZXC_BITS_PER_BYTE) > 4 * ZXC_BITS_PER_BYTE)
            dst[byte_idx + 4] |= (uint8_t)(v >> (4 * ZXC_BITS_PER_BYTE));
        bit_pos += bits;
    }
    return (int)out_bytes;
}

/*
 * ============================================================================
 * COMPRESS BOUND CALCULATION
 * ============================================================================
 */
uint64_t zxc_compress_bound(const size_t input_size) {
    // Guard UINT64_MAX / SIZE_MAX would overflow.
    if (UNLIKELY(input_size > (SIZE_MAX - (SIZE_MAX >> 8)))) return 0;
    uint64_t n = ((uint64_t)input_size + ZXC_BLOCK_SIZE - 1) / ZXC_BLOCK_SIZE;
    if (n == 0) n = 1;
    return ZXC_FILE_HEADER_SIZE + (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + 64)) +
           (uint64_t)input_size + ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE;
}
