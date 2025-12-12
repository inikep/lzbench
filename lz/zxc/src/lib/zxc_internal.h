/*
 * Copyright (c) 2025, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef ZXC_INTERNAL_H
#define ZXC_INTERNAL_H

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

/*
 * ============================================================================
 * SIMD INTRINSICS & COMPILER MACROS
 * ============================================================================
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#include <nmmintrin.h>
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define ZXC_USE_AVX512
#endif
#if defined(__AVX2__)
#define ZXC_USE_AVX2
#endif
#if defined(__SSE4_1__) || defined(__AVX__)
#define ZXC_USE_SSE41
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_acle.h>
#include <arm_neon.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#define ZXC_USE_NEON64
#else
#define ZXC_USE_NEON32
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) (__builtin_expect((!!(x)), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#define ZXC_PREFETCH_READ(ptr) __builtin_prefetch((const void*)(ptr), 0, 3)
#define ZXC_PREFETCH_WRITE(ptr) __builtin_prefetch((const void*)(ptr), 1, 3)
#define ZXC_MEMCPY(dst, src, size) __builtin_memcpy(dst, src, size)
#define ZXC_MEMSET(dst, val, size) __builtin_memset(dst, val, size)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define ZXC_PREFETCH_READ(ptr)
#define ZXC_PREFETCH_WRITE(ptr)
#define ZXC_MEMCPY(dst, src, size) memcpy(dst, src, size)
#define ZXC_MEMSET(dst, val, size) memset(dst, val, size)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ZXC_ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define ZXC_ALIGN(x) __declspec(align(x))
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
#define ZXC_ALIGN(x) _Alignas(x)
#else
#define ZXC_ALIGN(x) /* No alignment */
#endif

// Force inlining for critical paths
#if defined(__GNUC__) || defined(__clang__)
#define ZXC_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ZXC_ALWAYS_INLINE __forceinline
#else
#define ZXC_ALWAYS_INLINE inline
#endif

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif

/*
 * ============================================================================
 * CONSTANTS & FILE FORMAT
 * ============================================================================
 */

#define ZXC_MAGIC_WORD 0x0043585AU        // Magic signature "ZXC0" (Little Endian)
#define ZXC_FILE_FORMAT_VERSION 1         // Current file format version
#define ZXC_CHUNK_SIZE (256 * 1024)       // Size of data blocks processed by threads
#define ZXC_IO_BUFFER_SIZE (1024 * 1024)  // Size of stdio buffers
#define ZXC_PAD_SIZE 32                   // Padding size for buffer overruns

// Binary Header Sizes
#define ZXC_FILE_HEADER_SIZE 8  // Magic (4 bytes) + Version (1 byte) + Reserved (3 bytes)
#define ZXC_BLOCK_HEADER_SIZE \
    12  // Type (1) + Flags (1) + Reserved (2) + Comp Size (4) + Raw Size (4)
#define ZXC_NUM_HEADER_BINARY_SIZE 16  // Num Header: N Values (8) + Frame Size (2) + Reserved (6)
#define ZXC_GNR_HEADER_BINARY_SIZE \
    16  // GNR Header: N Sequences (4) + N Literals (4) + 4 x 1-byte Encoding Types
#define ZXC_SECTION_DESC_BINARY_SIZE 8  // Section Desc: Comp Size (4) + Raw Size (4)

// Block Flags
#define ZXC_BLOCK_FLAG_NONE 0U         // No flags
#define ZXC_BLOCK_FLAG_CHECKSUM 0x80U  // Block has a checksum (8 bytes after header)
#define ZXC_BLOCK_CHECKSUM_SIZE 8      // Size of checksum field in bytes

// Token Format Constants
#define ZXC_TOKEN_LIT_BITS 4    // Number of bits for Literal Length in token
#define ZXC_TOKEN_ML_MASK 0x0F  // Mask to extract Match Length from token

// LZ77 Constants
// The hash table uses 13 bits for addressing, resulting in 8192 (2^13) entries.
// The hash table uses 2x entries (load factor < 0.5) to reduce collisions.
// Each hash table entry stores: (epoch << 18) | offset.
// Total memory footprint: 64KB (8192 entries * 2 * 4 bytes each).
#define ZXC_LZ_HASH_BITS 13                       // (2*(2^13) * 4 bytes = 64KB)
#define ZXC_LZ_HASH_SIZE (1 << ZXC_LZ_HASH_BITS)  // Hash table size
#define ZXC_LZ_WINDOW_SIZE (1 << 16)              // 64KB sliding window
#define ZXC_LZ_MIN_MATCH 5                        // Minimum match length
#define ZXC_LZ_MAX_DIST (ZXC_LZ_WINDOW_SIZE - 1)  // Maximum offset distance

/**
 * @enum zxc_block_type_t
 * @brief Defines the different types of data blocks supported by the ZXC
 * format.
 *
 * This enumeration categorizes blocks based on the compression strategy
 * applied:
 * - `ZXC_BLOCK_RAW` (0): No compression. Used when data is incompressible (high
 * entropy) or when compression would expand the data size.
 * - `ZXC_BLOCK_GNR` (1): General-purpose compression (LZ77 + Bitpacking). This
 * is the default for most data (text, binaries, JSON, etc.).
 * - `ZXC_BLOCK_NUM` (2): Specialized compression for arrays of 32-bit integers.
 *   Uses Delta Encoding + ZigZag + Bitpacking.
 */
typedef enum { ZXC_BLOCK_RAW = 0, ZXC_BLOCK_GNR = 1, ZXC_BLOCK_NUM = 2 } zxc_block_type_t;

/**
 * @enum zxc_section_encoding_t
 * @brief Specifies the encoding methods used for internal data sections.
 *
 * These modes determine how specific components (like literals, match lengths,
 * or offsets) are stored within a block.
 * - `ZXC_SECTION_ENCODING_RAW`: Data is stored uncompressed.
 * - `ZXC_SECTION_ENCODING_RLE`: Run-Length Encoding.
 * - `ZXC_SECTION_ENCODING_BITPACK`: Bitpacking for integer values.
 * - `ZXC_SECTION_ENCODING_FSE`: Finite State Entropy (Reserved).
 * - `ZXC_SECTION_ENCODING_BITPACK_FSE`: Combined Bitpacking and FSE (Reserved).
 */
typedef enum {
    ZXC_SECTION_ENCODING_RAW = 0,
    ZXC_SECTION_ENCODING_RLE = 1,
    ZXC_SECTION_ENCODING_BITPACK = 2,
    ZXC_SECTION_ENCODING_FSE = 3,         // Reserved
    ZXC_SECTION_ENCODING_BITPACK_FSE = 4  // Reserved
} zxc_section_encoding_t;

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
    uint8_t block_type;   // Block type (e.g., RAW, GNR, NUM)
    uint8_t block_flags;  // Flags (e.g., checksum presence)
    uint16_t reserved;    // Reserved for future use
    uint32_t comp_size;   // Compressed size excluding header
    uint32_t raw_size;    // Decompressed size
} zxc_block_header_t;

/**
 * @struct zxc_gnr_header_t
 * @brief Header specific to General (LZ-based) compression blocks.
 *
 * This header follows the main block header when the block type is GNR. It
 * describes the layout of sequences and literals.
 *
 * @var zxc_gnr_header_t::n_sequences
 * The total count of LZ sequences in the block.
 * @var zxc_gnr_header_t::n_literals
 * The total count of literal bytes.
 * @var zxc_gnr_header_t::enc_lit
 * Encoding method used for the literal stream.
 * @var zxc_gnr_header_t::enc_litlen
 * Encoding method used for the literal lengths stream.
 * @var zxc_gnr_header_t::enc_mlen
 * Encoding method used for the match lengths stream.
 * @var zxc_gnr_header_t::enc_off
 * Encoding method used for the offset stream.
 */
typedef struct {
    uint32_t n_sequences;  // Number of sequences
    uint32_t n_literals;   // Number of literals
    uint8_t enc_lit;       // Literal encoding
    uint8_t enc_litlen;    // Literal lengths encoding
    uint8_t enc_mlen;      // Match lengths encoding
    uint8_t enc_off;       // Offset encoding (Unused in Token format, kept for alignment)
} zxc_gnr_header_t;

/**
 * @struct zxc_section_desc_t
 * @brief Describes the size attributes of a specific data section.
 *
 * Used to track the compressed and uncompressed sizes of sub-components
 * (e.g., a literal stream or offset stream) within a block.
 *
 * @var zxc_section_desc_t::comp_size
 * The size of the section on disk (compressed).
 * @var zxc_section_desc_t::raw_size
 * The size of the section in memory (decompressed).
 */
typedef struct {
    uint64_t sizes;  // comp_size (low 32) | raw_size (high 32)
} zxc_section_desc_t;

/**
 * @struct zxc_num_header_t
 * @brief Header specific to Numeric compression blocks.
 *
 * This header follows the main block header when the block type is NUM.
 *
 * @var zxc_num_header_t::n_values
 * The total number of numeric values encoded in the block.
 * @var zxc_num_header_t::frame_size
 * The size of the frame used for processing.
 */
typedef struct {
    uint64_t n_values;
    uint16_t frame_size;
} zxc_num_header_t;

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
 * @field buf_ll Pointer to the buffer for literal length codes.
 * @field buf_ml Pointer to the buffer for match length codes.
 * @field buf_off Pointer to the buffer for offset codes.
 * @field literals Pointer to the buffer for raw literal bytes.
 * @field epoch Current epoch counter for lazy hash table invalidation.
 * @field checksum_enabled Flag indicating if checksums should be computed.
 * @field compression_level The configured compression level.
 * @field lit_buffer Pointer to a scratch buffer for literal processing (e.g.,
 * RLE decoding).
 * @field lit_buffer_cap Current capacity of the literal scratch buffer.
 */
typedef struct {
    uint32_t* hash_table;   // Hash table for LZ77
    uint32_t* chain_table;  // Chain table for collision resolution
    uint32_t* buf_ll;       // Buffer for literal lengths
    uint32_t* buf_ml;       // Buffer for match lengths
    uint32_t* buf_off;      // Buffer for offsets
    uint8_t* literals;      // Buffer for literal bytes
    uint32_t epoch;         // Current epoch for hash table
    int checksum_enabled;   // Checksum enabled flag
    int compression_level;  // Compression level
    uint8_t* lit_buffer;    // Buffer scratch for literals (RLE)
    size_t lit_buffer_cap;  // Current capacity of this buffer
    void* memory_block;     // Single allocation block
} zxc_cctx_t;

/**
 * @typedef zxc_bit_reader_t
 * @brief Internal bit reader structure for ZXC compression/decompression.
 *
 * This structure maintains the state of the bit stream reading operation.
 * It buffers bits from the input byte stream into an accumulator to allow
 * reading variable-length bit sequences.
 *
 * @field ptr Pointer to the current position in the input byte stream. This
 * pointer advances as bytes are loaded into the accumulator.
 * @field end Pointer to the end of the input byte stream. Used to prevent
 * reading past the bounds of the input buffer.
 * @field accum Bit accumulator holding buffered bits. A 64-bit buffer that
 * holds the bits currently loaded from the stream but not yet consumed.
 * @field bits Number of valid bits currently in the accumulator. Indicates how
 * many bits are available in @c accum to be read.
 */
typedef struct {
    const uint8_t* ptr;
    const uint8_t* end;
    uint64_t accum;
    int bits;
} zxc_bit_reader_t;

/*
 * ============================================================================
 * MEMORY & ENDIANNESS HELPERS
 * ============================================================================
 * Functions to handle unaligned memory access and Little Endian conversion.
 */

/**
 * @brief Reads a 16-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given memory address as a
 * little-endian 16-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param p Pointer to the memory location to read from.
 * @return The 16-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_le16(const void* p) {
    uint16_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Reads a 32-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given pointer address as a
 * little-endian 32-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param p Pointer to the memory location to read from.
 * @return The 32-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_le32(const void* p) {
    uint32_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Reads a 64-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given memory address as a
 * little-endian 64-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param p Pointer to the memory location to read from.
 * @return The 64-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_le64(const void* p) {
    uint64_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Stores a 16-bit integer in memory using little-endian byte order.
 *
 * This function copies the value of a 16-bit unsigned integer to the specified
 * memory location. It uses memcpy to avoid strict aliasing violations and
 * potential unaligned access issues.
 *
 * @note This function assumes the system is little-endian or that the compiler
 * optimizes the memcpy to a store instruction that handles endianness if necessary
 * (though the implementation shown is a direct copy).
 *
 * @param p Pointer to the destination memory where the value will be stored.
 *          Must point to a valid memory region of at least 2 bytes.
 * @param v The 16-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le16(void* p, uint16_t v) { ZXC_MEMCPY(p, &v, sizeof(v)); }

/**
 * @brief Stores a 32-bit unsigned integer in little-endian format at the specified memory location.
 *
 * This function writes the 32-bit value `v` to the memory pointed to by `p`.
 * It uses `ZXC_MEMCPY` to ensure safe memory access, avoiding potential alignment issues
 * that could occur with direct pointer casting on some architectures.
 *
 * @note This function is marked as `ZXC_ALWAYS_INLINE` to minimize function call overhead.
 *
 * @param p Pointer to the destination memory where the value will be stored.
 * @param v The 32-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le32(void* p, uint32_t v) { ZXC_MEMCPY(p, &v, sizeof(v)); }

/**
 * @brief Stores a 64-bit unsigned integer in little-endian format at the specified memory location.
 *
 * This function copies the 64-bit value `v` to the memory pointed to by `p`.
 * It uses `ZXC_MEMCPY` to ensure safe memory access, avoiding potential alignment issues
 * that might occur with direct pointer dereferencing on some architectures.
 *
 * @note This function assumes the system is little-endian or that the compiler optimizes
 * the memcpy to a store instruction that handles endianness correctly if `ZXC_MEMCPY`
 * is defined appropriately.
 *
 * @param p Pointer to the destination memory where the value will be stored.
 * @param v The 64-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le64(void* p, uint64_t v) { ZXC_MEMCPY(p, &v, sizeof(v)); }

/**
 * @brief Copies 16 bytes from the source memory location to the destination memory location.
 *
 * This function is forced to be inlined and utilizes the internal ZXC_MEMCPY macro
 * to perform a fixed-size copy of 16 bytes. It is typically used for optimizing
 * small, fixed-size memory operations within the compression library.
 *
 * @param dst Pointer to the destination memory block.
 * @param src Pointer to the source memory block.
 */
static ZXC_ALWAYS_INLINE void zxc_copy16(void* dst, const void* src) { ZXC_MEMCPY(dst, src, 16); }

/**
 * @brief Counts trailing zeros in a 32-bit unsigned integer.
 *
 * This function returns the number of contiguous zero bits starting from the
 * least significant bit (LSB). If the input is 0, it returns 32.
 *
 * It utilizes compiler-specific built-ins for GCC/Clang (`__builtin_ctz`) and
 * MSVC (`_BitScanForward`) for optimal performance. If no supported compiler
 * is detected, it falls back to a portable De Bruijn sequence implementation.
 *
 * @param x The 32-bit unsigned integer to scan.
 * @return The number of trailing zeros (0-32).
 */
static ZXC_ALWAYS_INLINE int zxc_ctz32(uint32_t x) {
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long r;
    _BitScanForward(&r, x);
    return (int)r;
#else
    // Fallback De Bruijn (32 bits)
    static const int DeBruijn32[32] = {0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
                                       15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
                                       16, 7,  26, 12, 18, 6,  11, 5,  10, 9};
    return DeBruijn32[((uint32_t)((x & -((int)x)) * 0x077CB531U)) >> 27];
#endif
}

/**
 * @brief Counts the number of trailing zeros in a 64-bit unsigned integer.
 *
 * This function determines the number of zero bits following the least significant
 * one bit in the binary representation of `x`.
 *
 * @param x The 64-bit unsigned integer to scan.
 * @return The number of trailing zeros. Returns 64 if `x` is 0.
 *
 * @note This implementation uses compiler built-ins for GCC/Clang (`__builtin_ctzll`)
 *       and MSVC (`_BitScanForward64`) when available for optimal performance.
 *       It falls back to a De Bruijn sequence multiplication method for other compilers.
 */
static ZXC_ALWAYS_INLINE int zxc_ctz64(uint64_t x) {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    // Fallback De Bruijn
    static const int Debruijn64[64] = {
        0,  1,  48, 2,  57, 49, 28, 3,  61, 58, 50, 42, 38, 29, 17, 4,  62, 55, 59, 36, 53, 51,
        43, 22, 45, 39, 33, 30, 24, 18, 12, 5,  63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21,
        44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9,  13, 8,  7,  6};
    return Debruijn64[((x & -x) * 0x03F79D71B4CA8B09ULL) >> 58];
#endif
}

/**
 * @brief Computes a hash value for a given 32-bit integer.
 *
 * This internal function is marked as always inline to minimize function call overhead
 * during critical hashing operations. It takes a 32-bit integer input and transforms
 * it into a 32-bit hash value, typically used for hash table lookups or data distribution
 * within the compression algorithm.
 *
 * @param val The 32-bit integer value to be hashed.
 * @return uint32_t The computed 32-bit hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_func(uint32_t val) {
#if defined(ZXC_USE_SSE41) || defined(__SSE4_2__)
    return _mm_crc32_u32(0, val);
#elif defined(__ARM_FEATURE_CRC32)
    return __crc32cw(0, val);
#else
    uint32_t h = (val * 2654435761U);
    return (h >> 16) | (h << 16);
#endif
}

/**
 * @brief Calculates the index of the highest set bit (most significant bit) in a 32-bit integer.
 *
 * This function determines the position of the most significant bit that is set to 1.
 *
 * @param n The 32-bit unsigned integer to analyze.
 * @return The 0-based index of the highest set bit. If n is 0, the behavior is undefined.
 */
static ZXC_ALWAYS_INLINE uint8_t zxc_highbit32(uint32_t n) {
#ifdef _MSC_VER
    unsigned long index;
    return (n == 0) ? 0 : (_BitScanReverse(&index, n) ? (uint8_t)(index + 1) : 0);
#else
    return (n == 0) ? 0 : (32 - __builtin_clz(n));
#endif
}

/**
 * @brief Encodes a signed 32-bit integer using ZigZag encoding.
 *
 * ZigZag encoding maps signed integers to unsigned integers so that numbers with a small
 * absolute value (for instance, -1) have a small variant encoded value too. It does this
 * by "zig-zagging" back and forth through the positive and negative integers:
 *
 *  0 => 0
 * -1 => 1
 *  1 => 2
 * -2 => 3
 *  2 => 4
 *
 * This is particularly useful for variable-length encoding (varint) of signed integers,
 * as standard varint encoding is inefficient for negative numbers (which are interpreted
 * as very large unsigned integers).
 *
 * @param n The signed 32-bit integer to encode.
 * @return The ZigZag encoded unsigned 32-bit integer.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_zigzag_encode(int32_t n) {
    return ((uint32_t)n << 1) ^ (uint32_t)(-(int32_t)((uint32_t)n >> 31));
}

/**
 * @brief Decodes a 32-bit unsigned integer using ZigZag decoding.
 *
 * ZigZag encoding maps signed integers to unsigned integers so that numbers with a small
 * absolute value (for instance, -1) have a small variant encoded value too. It does this
 * by "zig-zagging" back and forth through the positive and negative integers:
 * 0 => 0, -1 => 1, 1 => 2, -2 => 3, 2 => 4, etc.
 *
 * This function reverses that process, converting the unsigned representation back into
 * the original signed 32-bit integer.
 *
 * @param n The unsigned 32-bit integer to decode.
 * @return The decoded signed 32-bit integer.
 */
static ZXC_ALWAYS_INLINE int32_t zxc_zigzag_decode(uint32_t n) {
    return (int32_t)(n >> 1) ^ -(int32_t)(n & 1);
}

/*
 * ============================================================================
 * COMPRESSION CONTEXT & STRUCTS
 * ============================================================================
 */

// Represents a found LZ77 sequence (Literal Length, Match Length, Offset)

/**
 * @typedef zxc_chunk_processor_t
 * @brief Function pointer type for processing a chunk of data.
 *
 * This type defines the signature for internal functions responsible for
 * processing (compressing or transforming) a specific chunk of input data.
 *
 * @param ctx     Pointer to the compression context containing state and
 * configuration.
 * @param in      Pointer to the input data buffer.
 * @param in_sz   Size of the input data in bytes.
 * @param out     Pointer to the output buffer where processed data will be
 * written.
 * @param out_cap Capacity of the output buffer in bytes.
 *
 * @return The number of bytes written to the output buffer on success, or a
 * negative error code on failure.
 */
typedef int (*zxc_chunk_processor_t)(zxc_cctx_t* ctx, const uint8_t* in, size_t in_sz, uint8_t* out,
                                     size_t out_cap);

/*
 * INTERNAL API
 * ------------
 */
/**
 * @brief Initializes a ZXC compression context.
 *
 * Sets up the internal state required for compression operations, allocating
 * necessary buffers based on the chunk size and compression level.
 *
 * @param ctx Pointer to the compression context structure to initialize.
 * @param chunk_size The size of the data chunks to process.
 * @param mode The compression mode (e.g., fast, high compression).
 * @param level The specific compression level (1-9).
 * @param checksum_enabled
 * @return 0 on success, or a negative error code on failure.
 */
int zxc_cctx_init(zxc_cctx_t* ctx, size_t chunk_size, int mode, int level, int checksum_enabled);

/**
 * @brief Frees resources associated with a ZXC compression context.
 *
 * Releases memory allocated during initialization and resets the context state.
 *
 * @param ctx Pointer to the compression context to free.
 */
void zxc_cctx_free(zxc_cctx_t* ctx);

/**
 * @brief Calculates a 64-bit XXH3checksum for a given input buffer.
 *
 * @param input Pointer to the data buffer.
 * @param len Length of the data in bytes.
 * @param seed Initial seed value for the hash calculation.
 * @return The calculated 64-bit hash value.
 */
uint64_t zxc_checksum(const void* RESTRICT input, size_t len);

/**
 * @brief Validates and reads the ZXC file header from a source buffer.
 *
 * Checks for the correct magic bytes and version number.
 *
 * @param src Pointer to the start of the file data.
 * @param src_size Size of the available source data (must be at least header
 * size).
 * @return The size of the header in bytes on success, or a negative error code.
 */
int zxc_read_file_header(const uint8_t* src, size_t src_size);

/**
 * @brief Writes the standard ZXC file header to a destination buffer.
 *
 * Writes the magic bytes and version information.
 *
 * @param dst Pointer to the destination buffer.
 * @param dst_capacity Maximum capacity of the destination buffer.
 * @return The number of bytes written on success, or a negative error code.
 */
int zxc_write_file_header(uint8_t* dst, size_t dst_capacity);

/**
 * @brief Parses a block header from the source stream.
 *
 * Decodes the block size, compression type, and checksum flags into the
 * provided block header structure.
 *
 * @param src Pointer to the current position in the source stream.
 * @param src_size Available bytes remaining in the source stream.
 * @param bh Pointer to a block header structure to populate.
 * @return The number of bytes read (header size) on success, or a negative
 * error code.
 */
int zxc_read_block_header(const uint8_t* src, size_t src_size, zxc_block_header_t* bh);

/**
 * @brief Encodes a block header into the destination buffer.
 *
 * Serializes the information contained in the block header structure (size,
 * flags, etc.) into the binary format expected by the decoder.
 *
 * @param dst Pointer to the destination buffer.
 * @param dst_capacity Maximum capacity of the destination buffer.
 * @param bh Pointer to the block header structure containing the metadata.
 * @return The number of bytes written on success, or a negative error code.
 */
int zxc_write_block_header(uint8_t* dst, size_t dst_capacity, const zxc_block_header_t* bh);

/**
 * @brief Initializes a bit reader structure.
 *
 * Sets up the internal state of the bit reader to read from the specified
 * source buffer.
 *
 * @param br Pointer to the bit reader structure to initialize.
 * @param src Pointer to the source buffer containing the data to read.
 * @param size The size of the source buffer in bytes.
 */
void zxc_br_init(zxc_bit_reader_t* br, const uint8_t* src, size_t size);

/**
 * @brief Bit-packs a stream of 32-bit integers into a destination buffer.
 *
 * Compresses an array of 32-bit integers by packing them using a specified
 * number of bits per integer.
 *
 * @param src Pointer to the source array of 32-bit integers.
 * @param count The number of integers to pack.
 * @param dst Pointer to the destination buffer where packed data will be
 * written.
 * @param dst_cap The capacity of the destination buffer in bytes.
 * @param bits The number of bits to use for each integer during packing.
 * @return The number of bytes written to the destination buffer, or a negative
 * error code on failure.
 */
int zxc_bitpack_stream_32(const uint32_t* RESTRICT src, size_t count, uint8_t* RESTRICT dst,
                          size_t dst_cap, uint8_t bits);

/**
 * @brief Writes a numeric header structure to a destination buffer.
 *
 * Serializes the `zxc_num_header_t` structure into the output stream.
 *
 * @param dst Pointer to the destination buffer.
 * @param rem The remaining space in the destination buffer.
 * @param nh Pointer to the numeric header structure to write.
 * @return The number of bytes written, or a negative error code if the buffer
 * is too small.
 */
int zxc_write_num_header(uint8_t* dst, size_t rem, const zxc_num_header_t* nh);

/**
 * @brief Reads a numeric header structure from a source buffer.
 *
 * Deserializes data from the input stream into a `zxc_num_header_t` structure.
 *
 * @param src Pointer to the source buffer.
 * @param src_size The size of the source buffer available for reading.
 * @param nh Pointer to the numeric header structure to populate.
 * @return The number of bytes read from the source, or a negative error code on
 * failure.
 */
int zxc_read_num_header(const uint8_t* src, size_t src_size, zxc_num_header_t* nh);

/**
 * @brief Writes a generic header and section descriptors to a destination
 * buffer.
 *
 * Serializes the `zxc_gnr_header_t` and an array of 4 section descriptors.
 *
 * @param dst Pointer to the destination buffer.
 * @param rem The remaining space in the destination buffer.
 * @param gh Pointer to the generic header structure to write.
 * @param desc Array of 4 section descriptors to write.
 * @return The number of bytes written, or a negative error code if the buffer
 * is too small.
 */
int zxc_write_gnr_header_and_desc(uint8_t* dst, size_t rem, const zxc_gnr_header_t* gh,
                                  const zxc_section_desc_t desc[4]);

/**
 * @brief Reads a generic header and section descriptors from a source buffer.
 *
 * Deserializes data into a `zxc_gnr_header_t` and an array of 4 section
 * descriptors.
 *
 * @param src Pointer to the source buffer.
 * @param len The length of the source buffer available for reading.
 * @param gh Pointer to the generic header structure to populate.
 * @param desc Array of 4 section descriptors to populate.
 * @return The number of bytes read from the source, or a negative error code on
 * failure.
 */
int zxc_read_gnr_header_and_desc(const uint8_t* src, size_t len, zxc_gnr_header_t* gh,
                                 zxc_section_desc_t desc[4]);

/**
 * @brief Runs the main compression/decompression stream engine.
 *
 * This function orchestrates the processing of data from an input stream to an
 * output stream, potentially utilizing multiple threads for parallel
 * processing. It handles the setup, execution, and teardown of the streaming
 * process based on the specified configuration.
 *
 * @param f_in Pointer to the input file stream (source data).
 * @param f_out Pointer to the output file stream (destination data).
 * @param n_threads The number of threads to use for processing. If 0 or 1,
 * processing may be sequential.
 * @param mode The operation mode (e.g., compression or decompression).
 * @param level The compression level to apply (relevant only for compression
 * mode).
 * @param checksum_enabled Flag indicating whether to calculate and verify checksums (1
 * for yes, 0 for no).
 * @param func The chunk processing callback function (`zxc_chunk_processor_t`)
 * responsible for handling individual data blocks.
 *
 * @return Returns 0 on success, or a non-zero error code on failure.
 */
int64_t zxc_stream_engine_run(FILE* f_in, FILE* f_out, int n_threads, int mode, int level,
                              int checksum_enabled, zxc_chunk_processor_t func);

/**
 * @brief Internal wrapper function to decompress a single chunk of data.
 *
 * This function handles the decompression of a specific chunk from the source
 * buffer into the destination buffer using the provided compression context. It
 * serves as an abstraction layer over the core decompression logic.
 *
 * @param ctx     Pointer to the ZXC compression context structure containing
 *                internal state and configuration.
 * @param src     Pointer to the source buffer containing compressed data.
 * @param src_sz  Size of the compressed data in the source buffer (in bytes).
 * @param dst     Pointer to the destination buffer where decompressed data will
 * be written.
 * @param dst_cap Capacity of the destination buffer (maximum bytes that can be
 * written).
 *
 * @return int    Returns 0 on success, or a negative error code on failure.
 *                Specific error codes depend on the underlying ZXC
 * implementation.
 */
int zxc_decompress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* src, size_t src_sz, uint8_t* dst,
                                 size_t dst_cap);

/**
 * @brief Wraps the internal chunk compression logic.
 *
 * This function acts as a wrapper to compress a single chunk of data using the
 * provided compression context. It handles the interaction with the underlying
 * compression algorithm for a specific block of memory.
 *
 * @param ctx   Pointer to the ZXC compression context containing configuration
 *              and state.
 * @param chunk Pointer to the source buffer containing the raw data to
 * compress.
 * @param src_sz    The size of the source chunk in bytes.
 * @param dst   Pointer to the destination buffer where compressed data will be
 * written.
 * @param dst_cap   The capacity of the destination buffer (maximum bytes to write).
 *
 * @return The number of bytes written to the destination buffer on success,
 *         or a negative error code on failure.
 */
int zxc_compress_chunk_wrapper(zxc_cctx_t* ctx, const uint8_t* chunk, size_t src_sz, uint8_t* dst,
                               size_t dst_cap);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_INTERNAL_H