/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ZXC_INTERNAL_H
#define ZXC_INTERNAL_H

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/rapidhash.h"
#include "../../include/zxc_buffer.h"
#include "../../include/zxc_sans_io.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__cplusplus) && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
    !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#define ZXC_ATOMIC _Atomic
#define ZXC_USE_C11_ATOMICS 1
#else
#define ZXC_ATOMIC volatile
#define ZXC_USE_C11_ATOMICS 0
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
#ifndef ZXC_USE_AVX512
#define ZXC_USE_AVX512
#endif
#endif
#if defined(__AVX2__)
#ifndef ZXC_USE_AVX2
#define ZXC_USE_AVX2
#endif
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64) || \
       defined(ZXC_USE_NEON32) || defined(ZXC_USE_NEON64))
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#ifndef ZXC_USE_NEON64
#define ZXC_USE_NEON64
#endif
#else
#ifndef ZXC_USE_NEON32
#define ZXC_USE_NEON32
#endif
#endif
#endif

/*
 * ============================================================================
 * LIKELY/UNLIKELY, PREFETCH, MEMCPY/MEMSET, ALIGN, ALWAYS_INLINE, ENDIANNESS
 * ============================================================================
 */

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#define RESTRICT __restrict__
#define ZXC_PREFETCH_READ(ptr) __builtin_prefetch((const void*)(ptr), 0, 3)
#define ZXC_PREFETCH_WRITE(ptr) __builtin_prefetch((const void*)(ptr), 1, 3)
#define ZXC_MEMCPY(dst, src, n) __builtin_memcpy(dst, src, n)
#define ZXC_MEMSET(dst, val, n) __builtin_memset(dst, val, n)
#define ZXC_ALIGN(x) __attribute__((aligned(x)))
#define ZXC_ALWAYS_INLINE inline __attribute__((always_inline))

#elif defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_IX86) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>
#define ZXC_PREFETCH_READ(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#define ZXC_PREFETCH_WRITE(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
#define ZXC_PREFETCH_READ(ptr) __prefetch((const void*)(ptr))
#define ZXC_PREFETCH_WRITE(ptr) __prefetch((const void*)(ptr))
#endif
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT __restrict
#pragma intrinsic(memcpy, memset)
#define ZXC_MEMCPY(dst, src, n) memcpy(dst, src, n)
#define ZXC_MEMSET(dst, val, n) memset(dst, val, n)
#define ZXC_ALIGN(x) __declspec(align(x))
#define ZXC_ALWAYS_INLINE __forceinline
#pragma intrinsic(_BitScanReverse)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#define ZXC_PREFETCH_READ(ptr)
#define ZXC_PREFETCH_WRITE(ptr)
#define ZXC_MEMCPY(dst, src, n) memcpy(dst, src, n)
#define ZXC_MEMSET(dst, val, n) memset(dst, val, n)
#define ZXC_ALWAYS_INLINE inline
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
#define ZXC_ALIGN(x) _Alignas(x)
#else
#define ZXC_ALIGN(x)
#endif
#endif

/*
 * Endianness detection
 */
#ifndef ZXC_LITTLE_ENDIAN
#if defined(_WIN32) || defined(__LITTLE_ENDIAN__) || \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define ZXC_LITTLE_ENDIAN
#elif defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define ZXC_BIG_ENDIAN
#else
#warning "Endianness not detected, defaulting to little-endian"
#define ZXC_LITTLE_ENDIAN
#endif
#endif

/*
 * Byte-swap helpers (used on big-endian only).
 */
#ifdef ZXC_BIG_ENDIAN
#if defined(__GNUC__) || defined(__clang__)
#define ZXC_BSWAP16(x) __builtin_bswap16(x)
#define ZXC_BSWAP32(x) __builtin_bswap32(x)
#define ZXC_BSWAP64(x) __builtin_bswap64(x)
#elif defined(_MSC_VER)
#define ZXC_BSWAP16(x) _byteswap_ushort(x)
#define ZXC_BSWAP32(x) _byteswap_ulong(x)
#define ZXC_BSWAP64(x) _byteswap_uint64(x)
#else
#define ZXC_BSWAP16(x) ((uint16_t)(((x) >> 8) | ((x) << 8)))
#define ZXC_BSWAP32(x) \
    ((uint32_t)(((x) >> 24) | (((x) >> 8) & 0xFF00) | (((x) << 8) & 0xFF0000) | ((x) << 24)))
#define ZXC_BSWAP64(x) \
    ((uint64_t)(((uint64_t)ZXC_BSWAP32((uint32_t)(x)) << 32) | ZXC_BSWAP32((uint32_t)((x) >> 32))))
#endif
#endif

/*
 * ============================================================================
 * CONSTANTS & FILE FORMAT
 * ============================================================================
 */

#define ZXC_MAGIC_WORD 0x9CB02EF5U            // Magic word identifying ZXC files
#define ZXC_FILE_FORMAT_VERSION 4             // Current file format version
#define ZXC_BLOCK_UNIT (4 * 1024)             // Block size unit (4KB)
#define ZXC_BLOCK_SIZE (64 * ZXC_BLOCK_UNIT)  // Size of data blocks processed by threads (256KB)
#define ZXC_IO_BUFFER_SIZE (1024 * 1024)      // Size of stdio buffers
#define ZXC_PAD_SIZE 32                       // Padding size for buffer overruns
#define ZXC_BITS_PER_BYTE 8                   // Number of bits per byte
#define ZXC_CACHE_LINE_SIZE 64                // Cache line size
#define ZXC_ALIGNMENT_MASK (ZXC_CACHE_LINE_SIZE - 1)  // Alignment mask
#define ZXC_VBYTE_MAX_LEN 5                           // Maximum length of variable byte encoding
#define ZXC_VBYTE_ALLOC_LEN 3  // Max length for allocation (sufficient for < 2MB blocks)

// File Header Parsing
#define ZXC_FILE_HEADER_SIZE \
    16  // Magic (4) + Version (1) + Chunk (1) + Flags (1) + Reserved (7) + CRC (2)
#define ZXC_FILE_FLAG_HAS_CHECKSUM 0x80U   // Flag in Flags byte (Bit 7)
#define ZXC_FILE_CHECKSUM_ALGO_MASK 0x0FU  // Algorithm ID (Bits 0-3)

#define ZXC_BLOCK_HEADER_SIZE 8    // Type (1) + Flags (1) + Reserved (1) + CRC (1) + Comp Size (4)
#define ZXC_BLOCK_CHECKSUM_SIZE 4  // Size of checksum field in bytes
#define ZXC_NUM_HEADER_BINARY_SIZE 16  // Num Header: N Values (8) + Frame Size (2) + Reserved (6)
#define ZXC_GLO_HEADER_BINARY_SIZE \
    16  // GLO Header: N Sequences (4) + N Literals (4) + 4 x 1-byte Encoding Types
#define ZXC_GHI_HEADER_BINARY_SIZE \
    16  // GHI Header: N Sequences (4) + N Literals (4) + 4 x 1-byte Encoding Types

// Section Descriptor Sizes
#define ZXC_SECTION_DESC_BINARY_SIZE 8     // Section Desc: Comp Size (4) + Raw Size (4)
#define ZXC_SECTION_SIZE_MASK 0xFFFFFFFFU  // Mask to extract 32-bit size from descriptor
#define ZXC_GLO_SECTIONS 4                 // Number of sections in GLO blocks
#define ZXC_GHI_SECTIONS 3                 // Number of sections in GHI blocks

// Checksum Algorithms
#define ZXC_CHECKSUM_RAPIDHASH 0x00U  // Default: rapidhash algorithm

#define ZXC_GLOBAL_CHECKSUM_SIZE 4  // Size of the global checksum (appended after EOF)
#define ZXC_FILE_FOOTER_SIZE 12     // Footer Size (8 bytes src size + 4 bytes global checksum)

// Token Format Constants
// Sequence Format Constants (GLO Token - 4-bit LL, 4-bit ML, 16-bit Offset)
#define ZXC_TOKEN_LIT_BITS 4  // Number of bits for Literal Length in token
#define ZXC_TOKEN_ML_BITS 4   // Number of bits for Match Length in token
#define ZXC_TOKEN_LL_MASK \
    ((1U << ZXC_TOKEN_LIT_BITS) - 1)  // Mask to extract Literal Length from token
#define ZXC_TOKEN_ML_MASK \
    ((1U << ZXC_TOKEN_ML_BITS) - 1)  // Mask to extract Match Length from token

// Sequence Format Constants (GHI Token - 8-bit LL, 8-bit ML, 16-bit Offset)
#define ZXC_SEQ_LL_BITS 8    // Number of bits for Literal Length in sequence
#define ZXC_SEQ_ML_BITS 8    // Number of bits for Match Length in sequence
#define ZXC_SEQ_OFF_BITS 16  // Number of bits for Offset in sequence
#define ZXC_SEQ_LL_MASK \
    ((1U << ZXC_SEQ_LL_BITS) - 1)  // Mask to extract Literal Length from sequence
#define ZXC_SEQ_ML_MASK ((1U << ZXC_SEQ_ML_BITS) - 1)  // Mask to extract Match Length from sequence
#define ZXC_SEQ_OFF_MASK ((1U << ZXC_SEQ_OFF_BITS) - 1)  // Mask to extract Offset from sequence

// Literal Stream Encoding Constants
#define ZXC_LIT_RLE_FLAG 0x80U  // Flag bit for RLE run in literal stream (128)
#define ZXC_LIT_LEN_MASK \
    (ZXC_LIT_RLE_FLAG - 1)  // Mask to extract length from RLE/Literal token (127)

// LZ77 Constants
// The hash table uses 13 bits for addressing, resulting in 8192 (2^13) entries.
// The hash table uses 2x entries (load factor < 0.5) to reduce collisions.
// Each hash table entry stores: (epoch << 18) | offset.
// Total memory footprint: 64KB (8192 entries * 2 * 4 bytes each).
#define ZXC_LZ_HASH_BITS 13                        // (2*(2^13) * 4 bytes = 64KB)
#define ZXC_LZ_HASH_SIZE (1U << ZXC_LZ_HASH_BITS)  // Hash table size
#define ZXC_LZ_WINDOW_SIZE (1U << 16)              // 64KB sliding window
// Note: sliding window of 64KB allows chain_table to use uint16_t for valid offsets (since any
// match > 64KB is invalid).
#define ZXC_LZ_MIN_MATCH_LEN 5                    // Minimum match length
#define ZXC_LZ_MAX_DIST (ZXC_LZ_WINDOW_SIZE - 1)  // Maximum offset distance

// Hash prime constants
#define ZXC_HASH_PRIME1 0x9E3779B1U
#define ZXC_HASH_PRIME2 0x85BA2D97U
#define ZXC_HASH_PRIME3 0xB0F57EE3U
#define ZXC_HASH_PRIME4 0x27D4EB2FU

/**
 * @struct zxc_lz77_params_t
 * @brief Search parameters for LZ77 compression levels.
 */
typedef struct {
    int search_depth;     // Max matches to check in hash chain
    int sufficient_len;   // Stop searching if match >= this length
    int use_lazy;         // Use lazy matching (check next position)
    int lazy_attempts;    // Max matches to check for lazy matching
    uint32_t step_base;   // Base step for literal advancement
    uint32_t step_shift;  // Shift for distance-based stepping
} zxc_lz77_params_t;

/**
 * @brief Retrieves LZ77 compression parameters based on the specified compression level.
 *
 * This inline function returns the appropriate LZ77 parameters configuration
 * for the given compression level.
 *
 * @param[in] level The compression level to use for determining LZ77 parameters.
 * @return zxc_lz77_params_t The LZ77 parameters structure corresponding to the specified level.
 */
static ZXC_ALWAYS_INLINE zxc_lz77_params_t zxc_get_lz77_params(const int level) {
    if (level >= 5) return (zxc_lz77_params_t){64, 256, 1, 16, 1, 31};
    // search_depth, sufficient_len, use_lazy, lazy_attempts, step_base, step_shift
    static const zxc_lz77_params_t table[5] = {
        {4, 16, 0, 0, 4, 4},  // fallback
        {4, 16, 0, 0, 4, 4},  // level 1
        {6, 24, 0, 0, 3, 6},  // level 2
        {4, 32, 1, 8, 1, 4},  // level 3
        {4, 32, 1, 8, 1, 5}   // level 4
    };
    return table[level < 1 ? 1 : level];
}

/**
 * @enum zxc_block_type_t
 * @brief Defines the different types of data blocks supported by the ZXC
 * format.
 *
 * This enumeration categorizes blocks based on the compression strategy
 * applied:
 * - `ZXC_BLOCK_RAW` (0): No compression. Used when data is incompressible (high
 * entropy) or when compression would expand the data size.
 * - `ZXC_BLOCK_GLO` (1): General-purpose compression (LZ77 + Bitpacking). This
 * is the default for most data (text, binaries, JSON, etc.). Includes 4 sections descriptors.
 * - `ZXC_BLOCK_NUM` (2): Specialized compression for arrays of 32-bit integers.
 *   Uses Delta Encoding + ZigZag + Bitpacking.
 * - `ZXC_BLOCK_GHI` (3): General-purpose high-velocity mode using LZ77 with advanced
 * techniques (lazy matching, step skipping) for maximum ratio. Includes 3 sections descriptors.
 * - `ZXC_BLOCK_EOF` (255): End of file marker.
 */
typedef enum {
    ZXC_BLOCK_RAW = 0,
    ZXC_BLOCK_GLO = 1,
    ZXC_BLOCK_NUM = 2,
    ZXC_BLOCK_GHI = 3,
    ZXC_BLOCK_EOF = 255
} zxc_block_type_t;

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
 * @struct zxc_gnr_header_t
 * @brief Header specific to General (LZ-based) compression blocks.
 *
 * This header follows the main block header when the block type is GLO/GHI. It
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

/**
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
 * @param[in] p Pointer to the memory location to read from.
 * @return The 16-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_le16(const void* p) {
    uint16_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP16(v);
#else
    return v;
#endif
}

/**
 * @brief Reads a 32-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given pointer address as a
 * little-endian 32-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param[in] p Pointer to the memory location to read from.
 * @return The 32-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_le32(const void* p) {
    uint32_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP32(v);
#else
    return v;
#endif
}

/**
 * @brief Reads a 64-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given memory address as a
 * little-endian 64-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param[in] p Pointer to the memory location to read from.
 * @return The 64-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_le64(const void* p) {
    uint64_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP64(v);
#else
    return v;
#endif
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
 * @param[out] p Pointer to the destination memory where the value will be stored.
 *          Must point to a valid memory region of at least 2 bytes.
 * @param[in] v The 16-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le16(void* p, const uint16_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint16_t s = ZXC_BSWAP16(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

/**
 * @brief Stores a 32-bit unsigned integer in little-endian format at the specified memory location.
 *
 * This function writes the 32-bit value `v` to the memory pointed to by `p`.
 * It uses `ZXC_MEMCPY` to ensure safe memory access, avoiding potential alignment issues
 * that could occur with direct pointer casting on some architectures.
 *
 * @note This function is marked as `ZXC_ALWAYS_INLINE` to minimize function call overhead.
 *
 * @param[out] p Pointer to the destination memory where the value will be stored.
 * @param[in] v The 32-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le32(void* p, const uint32_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint32_t s = ZXC_BSWAP32(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

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
 * @param[out] p Pointer to the destination memory where the value will be stored.
 * @param[in] v The 64-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le64(void* p, const uint64_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint64_t s = ZXC_BSWAP64(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

/**
 * @brief Computes the 1-byte checksum for block headers.
 *
 * @param[in] p Pointer to the input data to be hashed (8 bytes)
 * @return uint8_t The computed hash value.
 */
static ZXC_ALWAYS_INLINE uint8_t zxc_hash8(const uint8_t* p) {
    const uint64_t v = zxc_le64(p);
    uint64_t h = v * ZXC_HASH_PRIME1;
    h ^= (h >> 32);
    h *= ZXC_HASH_PRIME2;
    h ^= (h >> 32);
    return (uint8_t)h;
}

/**
 * @brief Computes the 2-byte checksum for file headers.
 *
 * This function generates a hash value by reading data from the given pointer.
 * The result is a 16-bit hash.
 *
 * @param[in] p Pointer to the input data to be hashed (16 bytes)
 * @return uint16_t The computed hash value.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_hash16(const uint8_t* p) {
    const uint32_t v0 = zxc_le32(p);
    const uint32_t v1 = zxc_le32(p + 4);
    const uint32_t v2 = zxc_le32(p + 8);
    const uint32_t v3 = zxc_le32(p + 12);

    uint32_t h = (v0 * ZXC_HASH_PRIME1);
    h ^= (v1 * ZXC_HASH_PRIME2);
    h ^= (v2 * ZXC_HASH_PRIME3);
    h ^= (v3 * ZXC_HASH_PRIME4);

    h = (h << 13) | (h >> 19);
    h *= ZXC_HASH_PRIME1;
    return (uint16_t)((h ^ (h >> 16)) & 0xFFFF);
}

/**
 * @brief Copies 16 bytes from the source memory location to the destination memory location.
 *
 * This function is forced to be inlined and uses SIMD intrinsics when available.
 * SSE2 on x86/x64, NEON on ARM, or memcpy as fallback.
 *
 * @param[out] dst Pointer to the destination memory block.
 * @param[in] src Pointer to the source memory block.
 */
static ZXC_ALWAYS_INLINE void zxc_copy16(void* dst, const void* src) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    // AVX2/AVX512: Single 128-bit unaligned load/store
    _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    vst1q_u8((uint8_t*)dst, vld1q_u8((const uint8_t*)src));
#else
    ZXC_MEMCPY(dst, src, 16);
#endif
}

/**
 * @brief Copies 32 bytes from source to destination using SIMD when available.
 *
 * Uses AVX2 on x86, NEON on ARM64/ARM32, or two 16-byte copies as fallback.
 *
 * @param[out] dst Pointer to the destination memory block.
 * @param[in] src Pointer to the source memory block.
 */
static ZXC_ALWAYS_INLINE void zxc_copy32(void* dst, const void* src) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    // AVX2/AVX512: Single 256-bit (32 byte) unaligned load/store
    _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    // NEON: Two 128-bit (16 byte) unaligned load/stores
    vst1q_u8((uint8_t*)dst, vld1q_u8((const uint8_t*)src));
    vst1q_u8((uint8_t*)dst + 16, vld1q_u8((const uint8_t*)src + 16));
#else
    ZXC_MEMCPY(dst, src, 32);
#endif
}

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
 * @param[in] x The 32-bit unsigned integer to scan.
 * @return The number of trailing zeros (0-32).
 */
static ZXC_ALWAYS_INLINE int zxc_ctz32(const uint32_t x) {
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
    return DeBruijn32[((uint32_t)((x & (0U - x)) * 0x077CB531U)) >> 27];
#endif
}

/**
 * @brief Counts the number of trailing zeros in a 64-bit unsigned integer.
 *
 * This function determines the number of zero bits following the least significant
 * one bit in the binary representation of `x`.
 *
 * @param[in] x The 64-bit unsigned integer to scan.
 * @return The number of trailing zeros. Returns 64 if `x` is 0.
 *
 * @note This implementation uses compiler built-ins for GCC/Clang (`__builtin_ctzll`)
 *       and MSVC (`_BitScanForward64`) when available for optimal performance.
 *       It falls back to a De Bruijn sequence multiplication method for other compilers.
 */
static ZXC_ALWAYS_INLINE int zxc_ctz64(const uint64_t x) {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    // Fallback De Bruijn
    static const int Debruijn64[64] = {
        0,  1,  48, 2,  57, 49, 28, 3,  61, 58, 50, 42, 38, 29, 17, 4,  62, 55, 59, 36, 53, 51,
        43, 22, 45, 39, 33, 30, 24, 18, 12, 5,  63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21,
        44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9,  13, 8,  7,  6};
    return Debruijn64[((x & (0ULL - x)) * 0x03F79D71B4CA8B09ULL) >> 58];
#endif
}

/**
 * @brief Calculates the index of the highest set bit (most significant bit) in a 32-bit integer.
 *
 * This function determines the position of the most significant bit that is set to 1.
 *
 * @param[in] n The 32-bit unsigned integer to analyze.
 * @return The 0-based index of the highest set bit. If n is 0, the behavior is undefined.
 */
static ZXC_ALWAYS_INLINE uint8_t zxc_highbit32(const uint32_t n) {
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
 * @param[in] n The signed 32-bit integer to encode.
 * @return The ZigZag encoded unsigned 32-bit integer.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_zigzag_encode(const int32_t n) {
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
 * @param[in] n The unsigned 32-bit integer to decode.
 * @return The decoded signed 32-bit integer.
 */
static ZXC_ALWAYS_INLINE int32_t zxc_zigzag_decode(const uint32_t n) {
    return (int32_t)(n >> 1) ^ -(int32_t)(n & 1);
}

/**
 * @brief Allocates aligned memory in a cross-platform manner.
 *
 * This function provides a unified interface for allocating memory with a specific
 * alignment requirement. It wraps `_aligned_malloc` for Windows
 * environments and `posix_memalign` for POSIX-compliant systems.
 *
 * @param[in] size The size of the memory block to allocate, in bytes.
 * @param[in] alignment The alignment value, which must be a power of two and a multiple
 *                  of `sizeof(void *)`.
 * @return A pointer to the allocated memory block, or NULL if the allocation fails.
 *         The returned pointer must be freed using the corresponding aligned free function.
 */
void* zxc_aligned_malloc(const size_t size, const size_t alignment);

/**
 * @brief Frees memory previously allocated with an aligned allocation function.
 *
 * This function provides a cross-platform wrapper for freeing aligned memory.
 * On Windows, it calls `_aligned_free`.
 * On other platforms, it falls back to the standard `free` function.
 *
 * @param[in] ptr A pointer to the memory block to be freed. If ptr is NULL, no operation is
 * performed.
 */
void zxc_aligned_free(void* ptr);

/*
 * ============================================================================
 * COMPRESSION CONTEXT & STRUCTS
 * ============================================================================
 */

/*
 * INTERNAL API
 * ------------
 */

/**
 * @brief Calculates a 32-bit hash for a given input buffer.
 * @param[in] input Pointer to the data buffer.
 * @param[in] len Length of the data in bytes.
 * @param[in] hash_method Checksum algorithm identifier (e.g., ZXC_CHECKSUM_RAPIDHASH).
 * @return The calculated 32-bit hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_checksum(const void* RESTRICT input, const size_t len,
                                               const uint8_t hash_method) {
    uint64_t hash;
    if (LIKELY(hash_method == ZXC_CHECKSUM_RAPIDHASH))
        hash = rapidhash(input, len);
    else
        // Default fallthrough to rapidhash for unknown types (safe default)
        hash = rapidhash(input, len);

    return (uint32_t)(hash ^ (hash >> (sizeof(uint32_t) * ZXC_BITS_PER_BYTE)));
}

/**
 * @brief Combines a running hash with a new block hash using rotate-left and XOR.
 *
 * This function updates a global checksum by rotating the current hash left by 1 bit
 * (with wraparound) and XORing with the new block hash. This provides a simple but
 * effective rolling hash that depends on the order of blocks.
 *
 * Formula: result = ((hash << 1) | (hash >> 31)) ^ block_hash
 *
 * @param[in] hash The current running hash value.
 * @param[in] block_hash The hash of the new block to combine.
 * @return The updated combined hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_combine_rotate(const uint32_t hash,
                                                          const uint32_t block_hash) {
    return ((hash << 1) | (hash >> 31)) ^ block_hash;
}

/**
 * @brief Loads up to 7 bytes from memory in little-endian order into a uint64_t.
 *
 * This is used for partial reads at stream boundaries where fewer than 8 bytes
 * remain. Unlike ZXC_MEMCPY into a uint64_t (which is endian-dependent), this
 * function always produces a value with byte 0 in the least-significant bits.
 *
 * @param[in] p Pointer to the source bytes.
 * @param[in] n Number of bytes to read (must be < 8).
 * @return The loaded value in native host order, with bytes arranged as if
 *         read from a little-endian stream.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_le_partial(const uint8_t* p, size_t n) {
#ifdef ZXC_BIG_ENDIAN
    uint64_t v = 0;
    for (size_t i = 0; i < n; i++) v |= (uint64_t)p[i] << (i * 8);
    return v;
#else
    uint64_t v = 0;
    n = n > sizeof(v) ? sizeof(v) : n;
    ZXC_MEMCPY(&v, p, n);
    return v;
#endif
}

/**
 * @brief Initializes a bit reader structure.
 *
 * Sets up the internal state of the bit reader to read from the specified
 * source buffer.
 *
 * @param[out] br Pointer to the bit reader structure to initialize.
 * @param[in] src Pointer to the source buffer containing the data to read.
 * @param[in] size The size of the source buffer in bytes.
 */
static ZXC_ALWAYS_INLINE void zxc_br_init(zxc_bit_reader_t* RESTRICT br,
                                          const uint8_t* RESTRICT src, const size_t size) {
    br->ptr = src;
    br->end = src + size;
    // Safety check: ensure we have at least 8 bytes to fill the accumulator
    if (UNLIKELY(size < sizeof(uint64_t))) {
        br->accum = zxc_le_partial(src, size);
        br->ptr += size;
        br->bits = (int)(size * 8);
    } else {
        br->accum = zxc_le64(br->ptr);
        br->ptr += sizeof(uint64_t);
        br->bits = sizeof(uint64_t) * 8;
    }
}

/**
 * @brief Ensures that the bit reader buffer contains at least the specified
 * number of bits.
 *
 * This function checks if the internal buffer of the bit reader has enough bits
 * available to satisfy a subsequent read operation of `needed` bits. If not, it
 * refills the buffer from the source.
 *
 * @param[in,out] br Pointer to the bit reader context.
 * @param[in] needed The number of bits required to be available in the buffer.
 */
static ZXC_ALWAYS_INLINE void zxc_br_ensure(zxc_bit_reader_t* RESTRICT br, const int needed) {
    if (UNLIKELY(br->bits < needed)) {
        int safe_bits = (br->bits < 0) ? 0 : br->bits;
        br->bits = safe_bits;

        // Mask out garbage bits (retain only valid existing bits)
#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
        br->accum = _bzhi_u64(br->accum, safe_bits);
#else
        br->accum &= ((1ULL << safe_bits) - 1);
#endif

        // Calculate how many bytes we can read
        // We want to fill up to the accumulation capability (64 bits for uint64_t)
        // Bytes needed = (capacity_bits - safe_bits) / 8
        int bytes_needed = ((int)(sizeof(uint64_t) * 8) - safe_bits) >> 3;

        // Bounds check: don't read past end
        size_t bytes_left = (size_t)(br->end - br->ptr);
        if (UNLIKELY(bytes_left < (size_t)bytes_needed)) {
            // Partial read (slow path / end of stream)
            uint64_t raw = zxc_le_partial(br->ptr, bytes_left);
            br->accum |= (raw << safe_bits);
            br->ptr += bytes_left;
            br->bits = safe_bits + (int)bytes_left * 8;
        } else {
            // Fast path: standard read
            uint64_t raw = zxc_le64(br->ptr);
            br->accum |= (raw << safe_bits);
            br->ptr += bytes_needed;
            br->bits = safe_bits + bytes_needed * 8;
        }
    }
}

/**
 * @brief Bit-packs a stream of 32-bit integers into a destination buffer.
 *
 * Compresses an array of 32-bit integers by packing them using a specified
 * number of bits per integer.
 *
 * @param[in] src Pointer to the source array of 32-bit integers.
 * @param[in] count The number of integers to pack.
 * @param[out] dst Pointer to the destination buffer where packed data will be
 * written.
 * @param[in] dst_cap The capacity of the destination buffer in bytes.
 * @param[in] bits The number of bits to use for each integer during packing.
 * @return int The number of bytes written to the destination buffer, or a negative
 * error code on failure.
 */
int zxc_bitpack_stream_32(const uint32_t* RESTRICT src, const size_t count, uint8_t* RESTRICT dst,
                          const size_t dst_cap, const uint8_t bits);

/**
 * @brief Writes a numeric header structure to a destination buffer.
 *
 * Serializes the `zxc_num_header_t` structure into the output stream.
 *
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] rem The remaining space in the destination buffer.
 * @param[in] nh Pointer to the numeric header structure to write.
 * @return int The number of bytes written, or a negative error code if the buffer
 * is too small.
 */
int zxc_write_num_header(uint8_t* RESTRICT dst, const size_t rem,
                         const zxc_num_header_t* RESTRICT nh);

/**
 * @brief Reads a numeric header structure from a source buffer.
 *
 * Deserializes data from the input stream into a `zxc_num_header_t` structure.
 *
 * @param[in] src Pointer to the source buffer.
 * @param[in] src_size The size of the source buffer available for reading.
 * @param[out] nh Pointer to the numeric header structure to populate.
 * @return int The number of bytes read from the source, or a negative error code on
 * failure.
 */
int zxc_read_num_header(const uint8_t* RESTRICT src, const size_t src_size,
                        zxc_num_header_t* RESTRICT nh);

/**
 * @brief Writes a generic header and section descriptors to a destination
 * buffer.
 *
 * Serializes the `zxc_gnr_header_t` and an array of 4 section descriptors.
 *
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] rem The remaining space in the destination buffer.
 * @param[in] gh Pointer to the generic header structure to write.
 * @param[in] desc Array of 4 section descriptors to write.
 * @return int The number of bytes written, or a negative error code if the buffer
 * is too small.
 */
int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GLO_SECTIONS]);

/**
 * @brief Reads a generic header and section descriptors from a source buffer.
 *
 * Deserializes data into a `zxc_gnr_header_t` and an array of 4 section
 * descriptors.
 *
 * @param[in] src Pointer to the source buffer.
 * @param[in] len The length of the source buffer available for reading.
 * @param[out] gh Pointer to the generic header structure to populate.
 * @param[out] desc Array of 4 section descriptors to populate.
 *
 * @return int Returns 0 on success, or a negative error code on failure.
 */
int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GLO_SECTIONS]);

/**
 * @brief Writes a record header and description to the destination buffer.
 *
 * @param dst Pointer to the destination buffer where the header and description will be written.
 * @param rem Remaining size available in the destination buffer.
 * @param gh Pointer to the GNR header structure containing header information.
 * @param desc Array of 3 section descriptors to be written along with the header.
 *
 * @return int Returns the number of bytes written on success, or a negative error code on failure.
 */
int zxc_write_ghi_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GHI_SECTIONS]);

/**
 * @brief Reads a record header and section descriptors from a buffer.
 *
 * This function parses the source buffer to extract a general header and
 * up to three section descriptors from a ZXC record.
 *
 * @param[in] src Pointer to the source buffer containing the record data.
 * @param[in] len Length of the source buffer in bytes.
 * @param[out] gh Pointer to a zxc_gnr_header_t structure to store the parsed header.
 * @param[out] desc Array of 3 zxc_section_desc_t structures to store the parsed section
 * descriptors.
 *
 * @return int Returns 0 on success, or a negative error code on failure.
 */
int zxc_read_ghi_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GHI_SECTIONS]);

/**
 * @brief Internal wrapper function to decompress a single chunk of data.
 *
 * This function handles the decompression of a specific chunk from the source
 * buffer into the destination buffer using the provided compression context. It
 * serves as an abstraction layer over the core decompression logic.
 *
 * @param[in,out] ctx     Pointer to the ZXC compression context structure containing
 *                internal state and configuration.
 * @param[in] src     Pointer to the source buffer containing compressed data.
 * @param[in] src_sz  Size of the compressed data in the source buffer (in bytes).
 * @param[out] dst     Pointer to the destination buffer where decompressed data will
 * be written.
 * @param[in] dst_cap Capacity of the destination buffer (maximum bytes that can be
 * written).
 *
 * @return int    Returns 0 on success, or a negative error code on failure.
 *                Specific error codes depend on the underlying ZXC
 * implementation.
 */
int zxc_decompress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap);

/**
 * @brief Wraps the internal chunk compression logic.
 *
 * This function acts as a wrapper to compress a single chunk of data using the
 * provided compression context. It handles the interaction with the underlying
 * compression algorithm for a specific block of memory.
 *
 * @param[in,out] ctx   Pointer to the ZXC compression context containing configuration
 *              and state.
 * @param[in] chunk Pointer to the source buffer containing the raw data to
 * compress.
 * @param[in] src_sz    The size of the source chunk in bytes.
 * @param[out] dst   Pointer to the destination buffer where compressed data will be
 * written.
 * @param[in] dst_cap   The capacity of the destination buffer (maximum bytes to write).
 *
 * @return int      The number of bytes written to the destination buffer on success,
 *                  or a negative error code on failure.
 */
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT chunk,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_INTERNAL_H