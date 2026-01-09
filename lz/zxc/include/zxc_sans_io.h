/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef ZXC_SANS_IO_H
#define ZXC_SANS_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * ZXC Compression Library - Public Sans-IO API - To build your own driver
 * ============================================================================
 */

/**
 * @typedef zxc_cctx_t
 * @brief Compression Context structure.
 *
 * This structure holds the state and buffers required for the compression
 * process. It is designed to be reused across multiple blocks or calls to avoid
 * the overhead of repeated memory allocations.
 *
 * **Key Fields:**
 * - `hash_table`: Stores indices of 4-byte sequences. Size is `2 *
 * ZXC_LZ_HASH_SIZE` to reduce collisions (load factor < 0.5).
 * - `chain_table`: Handles collisions by storing the *previous* occurrence of a
 *   hash. This forms a linked list for each hash bucket, allowing us to
 * traverse history.
 * - `epoch`: Used for "Lazy Hash Table Invalidation". Instead of
 * `ZXC_MEMSET`ing the entire hash table (which is slow) for every block, we
 * store `(epoch << 16) | offset`. If the stored epoch doesn't match the current
 * `ctx->epoch`, the entry is considered invalid/empty.
 *
 * @field hash_table Pointer to the hash table used for LZ77 match finding.
 * @field chain_table Pointer to the chain table for collision resolution.
 * @field memory_block Pointer to the single allocation block containing all buffers.
 * @field epoch Current epoch counter for lazy hash table invalidation.
 * @field buf_extras Pointer to the buffer for extra lengths (LL >= 15 or ML >= 15).
 * @field buf_offsets Pointer to the buffer for offsets.
 * @field buf_tokens Pointer to the buffer for token sequences.
 * @field literals Pointer to the buffer for raw literal bytes.
 * @field lit_buffer Pointer to a scratch buffer for literal processing (e.g.,
 * RLE decoding).
 * @field lit_buffer_cap Current capacity of the literal scratch buffer.
 * @field checksum_enabled Flag indicating if checksums should be computed.
 * @field compression_level The configured compression level.
 */
typedef struct {
    // Hot zone: random access / high frequency
    // Kept at the start to ensure they reside in the first cache line (64 bytes).
    uint32_t* hash_table;   // Hash table for LZ77
    uint16_t* chain_table;  // Chain table for collision resolution
    void* memory_block;     // Single allocation block owner
    uint32_t epoch;         // Current epoch for hash table (checked per match)

    // Warm zone: sequential access per sequence
    uint32_t* buf_sequences;  // Buffer for sequence records (packed: LL (8) | ML (8) | Offset (16))
    uint8_t* buf_tokens;      // Buffer for token sequences
    uint16_t* buf_offsets;    // Buffer for offsets
    uint8_t* buf_extras;      // Buffer for extra lengths (vbytes for LL/ML)
    uint8_t* literals;        // Buffer for literal bytes

    // Cold zone: configuration / scratch / resizeable
    uint8_t* lit_buffer;    // Buffer scratch for literals (RLE)
    size_t lit_buffer_cap;  // Current capacity of this buffer
    int checksum_enabled;   // Checksum enabled flag
    int compression_level;  // Compression level
} zxc_cctx_t;

/**
 * @brief Initializes a ZXC compression context.
 *
 * Sets up the internal state required for compression operations, allocating
 * necessary buffers based on the chunk size and compression level.
 *
 * @param[out] ctx Pointer to the ZXC compression context structure to initialize.
 * @param[in] chunk_size The size of the data chunk to be compressed. This
 * determines the allocation size for various internal buffers.
 * @param[in] mode The operation mode (1 for compression, 0 for decompression).
 * @param[in] level The desired compression level to be stored in the context.
 * @param[in] checksum_enabled
 * @return 0 on success, or -1 if memory allocation fails for any of the
 * internal buffers.
 */
int zxc_cctx_init(zxc_cctx_t* ctx, size_t chunk_size, int mode, int level, int checksum_enabled);

/**
 * @brief Frees resources associated with a ZXC compression context.
 *
 * This function releases all internal buffers and tables associated with the
 * given ZXC compression context structure. It does not free the context pointer
 * itself, only its members.
 *
 * @param[in,out] ctx Pointer to the compression context to clean up.
 */
void zxc_cctx_free(zxc_cctx_t* ctx);

/**
 * @brief Writes the standard ZXC file header to a destination buffer.
 *
 * This function stores the magic word (little-endian) and the version number
 * into the provided buffer. It ensures the buffer has sufficient capacity
 * before writing.
 *
 * @param[out] dst The destination buffer where the header will be written.
 * @param[in] dst_capacity The total capacity of the destination buffer in bytes.
 * @return The number of bytes written (ZXC_FILE_HEADER_SIZE) on success,
 *         or -1 if the destination capacity is insufficient.
 */
int zxc_write_file_header(uint8_t* dst, size_t dst_capacity);

/**
 * @brief Validates and reads the ZXC file header from a source buffer.
 *
 * This function checks if the provided source buffer is large enough to contain
 * a ZXC file header and verifies that the magic word and version number match
 * the expected ZXC format specifications.
 *
 * @param[in] src Pointer to the source buffer containing the file data.
 * @param[in] src_size Size of the source buffer in bytes.
 * @param[out] out_block_size Optional pointer to receive the recommended block size
 * @return 0 if the header is valid, -1 otherwise (e.g., buffer too small,
 * invalid magic word, or incorrect version).
 */
int zxc_read_file_header(const uint8_t* src, size_t src_size, size_t* out_block_size);

/**
 * @struct zxc_block_header_t
 * @brief Represents the on-disk header structure for a ZXC block.
 *
 * This structure contains metadata required to parse and decompress a block.
 *
 * @var zxc_block_header_t::block_type
 * The type of the block (see zxc_block_type_t).
 * @var zxc_block_header_t::block_flags
 * Bit flags indicating properties like checksum presence.
 * @var zxc_block_header_t::reserved
 * Reserved bytes for future protocol extensions.
 * @var zxc_block_header_t::comp_size
 * The size of the compressed data payload in bytes (excluding this header).
 * @var zxc_block_header_t::raw_size
 * The size of the data after decompression.
 */
typedef struct {
    uint8_t block_type;   // Block type (e.g., RAW, GLO, GHI, NUM)
    uint8_t block_flags;  // Flags (e.g., checksum presence)
    uint16_t reserved;    // Reserved for future use
    uint32_t comp_size;   // Compressed size excluding header
    uint32_t raw_size;    // Decompressed size
} zxc_block_header_t;

/**
 * @brief Encodes a block header into the destination buffer.
 *
 * This function serializes the contents of a `zxc_block_header_t` structure
 * into a byte array in little-endian format. It ensures the destination buffer
 * has sufficient capacity before writing.
 *
 * @param[out] dst Pointer to the destination buffer where the header will be
 * written.
 * @param[in] dst_capacity The total size of the destination buffer in bytes.
 * @param[in] bh Pointer to the source block header structure containing the data to
 * write.
 *
 * @return The number of bytes written (ZXC_BLOCK_HEADER_SIZE) on success,
 *         or -1 if the destination buffer capacity is insufficient.
 */
int zxc_write_block_header(uint8_t* dst, size_t dst_capacity, const zxc_block_header_t* bh);

/**
 * @brief Read and parses a ZXC block header from a source buffer.
 *
 * This function extracts the block type, flags, reserved fields, compressed
 * size, and raw size from the first `ZXC_BLOCK_HEADER_SIZE` bytes of the source
 * buffer. It handles endianness conversion for multi-byte fields (Little
 * Endian).
 *
 * @param[in] src       Pointer to the source buffer containing the block data.
 * @param[in] src_size  The size of the source buffer in bytes.
 * @param[out] bh        Pointer to a `zxc_block_header_t` structure where the parsed
 *                  header information will be stored.
 *
 * @return 0 on success, or -1 if the source buffer is smaller than the
 *         required block header size.
 */
int zxc_read_block_header(const uint8_t* src, size_t src_size, zxc_block_header_t* bh);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_SANS_IO_H