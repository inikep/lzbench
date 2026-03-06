/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * @file zxc_common.c
 * @brief Shared library utilities: context management, header I/O, bitpacking,
 *        compress-bound calculation, and error-code name lookup.
 *
 * This translation unit contains the functions shared by both the buffer and
 * streaming APIs.  It is linked into every build of libzxc.
 */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_error.h"
#include "../../include/zxc_sans_io.h"
#include "zxc_internal.h"

/*
 * ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================
 */

/*
 * @brief Allocates memory aligned to the specified boundary.
 *
 * Uses `_aligned_malloc` on Windows and `posix_memalign` elsewhere.
 *
 * @param[in] size      Number of bytes to allocate.
 * @param[in] alignment Required alignment (must be a power of two).
 * @return Pointer to the allocated block, or @c NULL on failure.
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

/*
 * @brief Frees memory previously allocated by zxc_aligned_malloc().
 *
 * @param[in] ptr Pointer returned by zxc_aligned_malloc() (may be @c NULL).
 */
void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*
 * @brief Initialises a compression context, allocating all internal buffers.
 *
 * A single cache-line-aligned allocation is carved into hash table, chain
 * table, sequence buffers, token buffers, offset buffers, extra-length
 * buffers, and a literal buffer.
 *
 * @param[out] ctx              Context to initialise (zeroed on entry).
 * @param[in]  chunk_size       Maximum uncompressed chunk size (bytes).
 * @param[in]  mode             0 = decompression (skip buffer alloc), 1 = compression.
 * @param[in]  level            Compression level (stored in ctx).
 * @param[in]  checksum_enabled Non-zero to enable checksum generation/verification.
 * @return @ref ZXC_OK on success, or @ref ZXC_ERROR_MEMORY on allocation failure.
 */
int zxc_cctx_init(zxc_cctx_t* RESTRICT ctx, const size_t chunk_size, const int mode,
                  const int level, const int checksum_enabled) {
    ZXC_MEMSET(ctx, 0, sizeof(zxc_cctx_t));

    ctx->compression_level = level;
    ctx->checksum_enabled = checksum_enabled;

    if (mode == 0) return ZXC_OK;

    const size_t max_seq = chunk_size / sizeof(uint32_t) + 256;
    const size_t sz_hash = 2 * ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
    const size_t sz_chain = chunk_size * sizeof(uint16_t);
    const size_t sz_sequences = max_seq * sizeof(uint32_t);
    const size_t sz_tokens = max_seq * sizeof(uint8_t);
    const size_t sz_offsets = max_seq * sizeof(uint16_t);
    const size_t sz_extras =
        max_seq * 2 *
        ZXC_VBYTE_ALLOC_LEN;  // Max 3 bytes per LL/ML VByte (sufficient for 256KB block)
    const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

    // Calculate sizes with alignment padding (64 bytes for cache line alignment)
    size_t total_size = 0;
    const size_t off_hash = total_size;
    total_size += (sz_hash + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_chain = total_size;
    total_size += (sz_chain + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_sequences = total_size;
    total_size += (sz_sequences + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_tokens = total_size;
    total_size += (sz_tokens + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_offsets = total_size;
    total_size += (sz_offsets + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_extras = total_size;
    total_size += (sz_extras + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    const size_t off_lit = total_size;
    total_size += (sz_lit + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    uint8_t* const mem = (uint8_t*)zxc_aligned_malloc(total_size, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem)) return ZXC_ERROR_MEMORY;

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
    return ZXC_OK;
}

/*
 * @brief Releases all resources owned by a compression context.
 *
 * After this call every pointer inside @p ctx is @c NULL and the context
 * may be safely re-initialised with zxc_cctx_init().
 *
 * @param[in,out] ctx Context to tear down.
 */
void zxc_cctx_free(zxc_cctx_t* ctx) {
    if (ctx->memory_block) {
        zxc_aligned_free(ctx->memory_block);
        ctx->memory_block = NULL;
    }

    if (ctx->lit_buffer) {
        free(ctx->lit_buffer);
        ctx->lit_buffer = NULL;
    }

    if (ctx->work_buf) {
        free(ctx->work_buf);
        ctx->work_buf = NULL;
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
    ctx->work_buf_cap = 0;
}

/*
 * ============================================================================
 * HEADER I/O
 * ============================================================================
 */

/*
 * @brief Serialises a ZXC file header into @p dst.
 *
 * Layout (16 bytes): Magic (4) | Version (1) | Chunk (1) | Flags (1) |
 * Reserved (7) | CRC-16 (2).
 *
 * @param[out] dst          Destination buffer (>= @ref ZXC_FILE_HEADER_SIZE bytes).
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  has_checksum Non-zero to set the checksum flag.
 * @return Number of bytes written (@ref ZXC_FILE_HEADER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
 */
int zxc_write_file_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                          const int has_checksum) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;

    // Dual-scale chunk size encoding
    // Large scale multiplier is 64 KB, fine scale is 4 KB (ratio: 64 / 4 = 16)
    const uint32_t units = (uint32_t)(ZXC_BLOCK_SIZE / ZXC_BLOCK_UNIT);
    dst[5] = units <= 127 ? (uint8_t)units : (uint8_t)((units / 16) | 0x80);

    // Flags are at offset 6
    dst[6] = has_checksum ? (ZXC_FILE_FLAG_HAS_CHECKSUM | ZXC_CHECKSUM_RAPIDHASH) : 0;

    // Bytes 7-13: Reserved (must be 0, 7 bytes)
    ZXC_MEMSET(dst + 7, 0, 7);

    // Bytes 14-15: CRC (16-bit)
    zxc_store_le16(dst + 14, 0);  // Zero out before hashing
    const uint16_t crc = zxc_hash16(dst);
    zxc_store_le16(dst + 14, crc);

    return ZXC_FILE_HEADER_SIZE;
}

/*
 * @brief Parses and validates a ZXC file header from @p src.
 *
 * Checks the magic word, format version, and CRC-16.
 *
 * @param[in]  src              Source buffer (>= @ref ZXC_FILE_HEADER_SIZE bytes).
 * @param[in]  src_size         Size of @p src.
 * @param[out] out_block_size   Receives the decoded block size (may be @c NULL).
 * @param[out] out_has_checksum Receives 1 if checksums are present, 0 otherwise
 *                              (may be @c NULL).
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size,
                         size_t* RESTRICT out_block_size, int* RESTRICT out_has_checksum) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (UNLIKELY(zxc_le32(src) != ZXC_MAGIC_WORD)) return ZXC_ERROR_BAD_MAGIC;
    if (UNLIKELY(src[4] != ZXC_FILE_FORMAT_VERSION)) return ZXC_ERROR_BAD_VERSION;

    uint8_t temp[ZXC_FILE_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_FILE_HEADER_SIZE);
    // Zero out CRC bytes (14-15) before hash check
    temp[14] = 0;
    temp[15] = 0;
    if (UNLIKELY(zxc_le16(src + 14) != zxc_hash16(temp))) return ZXC_ERROR_BAD_HEADER;

    if (out_block_size) {
        // Read Dual-Scale Chunk Size Code
        const uint8_t code = src[5];
        const size_t scale = (code & 0x80) ? (16 * ZXC_BLOCK_UNIT) : ZXC_BLOCK_UNIT;
        size_t value = code & 0x7F;
        if (UNLIKELY(value == 0)) value = 1;

        *out_block_size = value * scale;
    }
    // Flags are at offset 6
    if (out_has_checksum) *out_has_checksum = (src[6] & ZXC_FILE_FLAG_HAS_CHECKSUM) ? 1 : 0;

    return ZXC_OK;
}

/*
 * @brief Serialises a block header (8 bytes) into @p dst.
 *
 * @param[out] dst          Destination buffer (>= @ref ZXC_BLOCK_HEADER_SIZE bytes).
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  bh           Populated block header descriptor.
 * @return Number of bytes written (@ref ZXC_BLOCK_HEADER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
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

/*
 * @brief Parses and validates a block header from @p src.
 *
 * Validates the 8-bit CRC embedded in the header.
 *
 * @param[in]  src      Source buffer (>= @ref ZXC_BLOCK_HEADER_SIZE bytes).
 * @param[in]  src_size Size of @p src.
 * @param[out] bh       Receives the decoded block header fields.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
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
    bh->header_crc = src[7];
    return ZXC_OK;
}

/*
 * @brief Writes the 12-byte file footer (source size + global checksum).
 *
 * @param[out] dst              Destination buffer (>= @ref ZXC_FILE_FOOTER_SIZE bytes).
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  src_size         Original uncompressed size in bytes.
 * @param[in]  global_hash      Accumulated global checksum value.
 * @param[in]  checksum_enabled Non-zero to write the checksum; zero to zero-fill.
 * @return Number of bytes written (@ref ZXC_FILE_FOOTER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
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

/*
 * @brief Serialises a NUM block header (16 bytes).
 *
 * @param[out] dst Destination buffer (>= @ref ZXC_NUM_HEADER_BINARY_SIZE bytes).
 * @param[in]  rem Remaining capacity of @p dst.
 * @param[in]  nh  Populated NUM header descriptor.
 * @return Number of bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_write_num_header(uint8_t* RESTRICT dst, const size_t rem,
                         const zxc_num_header_t* RESTRICT nh) {
    if (UNLIKELY(rem < ZXC_NUM_HEADER_BINARY_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le64(dst, nh->n_values);
    zxc_store_le16(dst + 8, nh->frame_size);
    zxc_store_le16(dst + 10, 0);
    zxc_store_le32(dst + 12, 0);
    return ZXC_NUM_HEADER_BINARY_SIZE;
}

/*
 * @brief Parses a NUM block header from @p src.
 *
 * @param[in]  src      Source buffer (>= @ref ZXC_NUM_HEADER_BINARY_SIZE bytes).
 * @param[in]  src_size Size of @p src.
 * @param[out] nh       Receives the decoded NUM header fields.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_num_header(const uint8_t* RESTRICT src, const size_t src_size,
                        zxc_num_header_t* RESTRICT nh) {
    if (UNLIKELY(src_size < ZXC_NUM_HEADER_BINARY_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    nh->n_values = zxc_le64(src);
    nh->frame_size = zxc_le16(src + 8);
    return ZXC_OK;
}

/*
 * @brief Serialises a GLO block header followed by its section descriptors.
 *
 * @param[out] dst  Destination buffer.
 * @param[in]  rem  Remaining capacity of @p dst.
 * @param[in]  gh   Populated GLO header descriptor.
 * @param[in]  desc Array of @ref ZXC_GLO_SECTIONS section descriptors.
 * @return Total bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    const size_t needed =
        ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

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

/*
 * @brief Parses a GLO block header and its section descriptors from @p src.
 *
 * @param[in]  src  Source buffer.
 * @param[in]  len  Size of @p src.
 * @param[out] gh   Receives the decoded GLO header.
 * @param[out] desc Receives @ref ZXC_GLO_SECTIONS decoded section descriptors.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    const size_t needed =
        ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

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
    return ZXC_OK;
}

/*
 * @brief Serialises a GHI block header followed by its section descriptors.
 *
 * @param[out] dst  Destination buffer.
 * @param[in]  rem  Remaining capacity of @p dst.
 * @param[in]  gh   Populated GHI header descriptor.
 * @param[in]  desc Array of @ref ZXC_GHI_SECTIONS section descriptors.
 * @return Total bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_write_ghi_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    const size_t needed =
        ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

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

/*
 * @brief Parses a GHI block header and its section descriptors from @p src.
 *
 * @param[in]  src  Source buffer.
 * @param[in]  len  Size of @p src.
 * @param[out] gh   Receives the decoded GHI header.
 * @param[out] desc Receives @ref ZXC_GHI_SECTIONS decoded section descriptors.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_ghi_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    const size_t needed =
        ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

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
    return ZXC_OK;
}

/*
 * ============================================================================
 * BITPACKING UTILITIES
 * ============================================================================
 */

/*
 * @brief Bit-packs an array of 32-bit values into a compact byte stream.
 *
 * Each value is masked to @p bits width and packed contiguously.
 *
 * @param[in]  src     Source array of 32-bit integers.
 * @param[in]  count   Number of values to pack.
 * @param[out] dst     Destination byte buffer.
 * @param[in]  dst_cap Capacity of @p dst.
 * @param[in]  bits    Number of bits per value (0-32).
 * @return Number of bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_bitpack_stream_32(const uint32_t* RESTRICT src, const size_t count, uint8_t* RESTRICT dst,
                          const size_t dst_cap, const uint8_t bits) {
    const size_t out_bytes = ((count * bits) + ZXC_BITS_PER_BYTE - 1) / ZXC_BITS_PER_BYTE;

    if (UNLIKELY(dst_cap < out_bytes)) return ZXC_ERROR_DST_TOO_SMALL;

    size_t bit_pos = 0;
    ZXC_MEMSET(dst, 0, out_bytes);

    // Create a mask for the input bits to prevent overflow
    // If bits is 32, the shift (1ULL << 32) is undefined behavior on 32-bit types,
    // but here we use uint64_t. (1ULL << 32) is fine on 64-bit.
    // However, if bits=64 (unlikely for a 32-bit packer), it would be an issue.
    // For 0 < bits <= 32:
    const uint64_t val_mask =
        (bits == sizeof(uint32_t) * ZXC_BITS_PER_BYTE) ? UINT32_MAX : ((1ULL << bits) - 1);

    for (size_t i = 0; i < count; i++) {
        // Mask the input value to ensure we don't write garbage
        const uint64_t v = ((uint64_t)src[i] & val_mask) << (bit_pos % ZXC_BITS_PER_BYTE);

        const size_t byte_idx = bit_pos / ZXC_BITS_PER_BYTE;
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
/*
 * @brief Returns the maximum compressed size for a given input size.
 *
 * The result accounts for the file header, per-block headers, block
 * checksums, worst-case expansion, EOF block, and the file footer.
 *
 * @param[in] input_size Uncompressed input size in bytes.
 * @return Upper bound on compressed size, or 0 if @p input_size would overflow.
 */
uint64_t zxc_compress_bound(const size_t input_size) {
    // Guard UINT64_MAX / SIZE_MAX would overflow.
    if (UNLIKELY(input_size > (SIZE_MAX - (SIZE_MAX >> 8)))) return 0;
    uint64_t n = ((uint64_t)input_size + ZXC_BLOCK_SIZE - 1) / ZXC_BLOCK_SIZE;
    if (n == 0) n = 1;
    return ZXC_FILE_HEADER_SIZE + (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + 64)) +
           (uint64_t)input_size + ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE;
}

/*
 * ============================================================================
 * ERROR CODE UTILITIES
 * ============================================================================
 */

/*
 * @brief Returns a human-readable string for the given error code.
 *
 * @param[in] code An error code from @ref zxc_error_t (or @ref ZXC_OK).
 * @return A static string such as @c "ZXC_OK" or @c "ZXC_ERROR_MEMORY".
 *         Returns @c "ZXC_UNKNOWN_ERROR" for unrecognised codes.
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
        default:
            return "ZXC_UNKNOWN_ERROR";
    }
}
