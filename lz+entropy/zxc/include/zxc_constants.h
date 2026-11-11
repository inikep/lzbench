/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_constants.h
 * @brief Public constants: library version, block sizes, compression levels.
 *
 * Include it to check the version at compile time, or to name the constants
 * the rest of the API expects.
 */

#ifndef ZXC_CONSTANTS_H
#define ZXC_CONSTANTS_H

/**
 * @defgroup version Library Version
 * @brief Compile-time version information.
 * @{
 */

/** @brief Major version number. */
#define ZXC_VERSION_MAJOR 1
/** @brief Minor version number. */
#define ZXC_VERSION_MINOR 0
/** @brief Patch version number. */
#define ZXC_VERSION_PATCH 0
/** @brief Release code name. */
#define ZXC_VERSION_CODENAME "Hydrogen"

/** @cond INTERNAL */
#define ZXC_STR_HELPER(x) #x
#define ZXC_STR(x) ZXC_STR_HELPER(x)
/** @endcond */

/**
 * @brief Human-readable version string in "MAJOR.MINOR.PATCH" form (e.g. "1.0.0").
 */
#define ZXC_LIB_VERSION_STR    \
    ZXC_STR(ZXC_VERSION_MAJOR) \
    "." ZXC_STR(ZXC_VERSION_MINOR) "." ZXC_STR(ZXC_VERSION_PATCH)

/**
 * @brief Version as one integer, MAJOR * 10000 + MINOR * 100 + PATCH (e.g. 10000 for 1.0.0).
 *
 * Orders like the version triple, so a header requirement reads
 * `#if ZXC_VERSION_NUMBER >= 10000`. zxc_version_number() returns the same
 * value for the library actually loaded, which may differ from this header.
 */
#define ZXC_VERSION_NUMBER (ZXC_VERSION_MAJOR * 10000 + ZXC_VERSION_MINOR * 100 + ZXC_VERSION_PATCH)

/** @} */ /* end of version */

/**
 * @defgroup block_size Block Size
 * @brief Block size constraints for compression.
 *
 * Block size must be a power of two in range
 * [@ref ZXC_BLOCK_SIZE_MIN, @ref ZXC_BLOCK_SIZE_MAX].
 * Pass 0 to any API to use @ref ZXC_BLOCK_SIZE_DEFAULT.
 * @{
 */
/** @brief log2(ZXC_BLOCK_SIZE_MIN) - exponent code for minimum block size. */
#define ZXC_BLOCK_SIZE_MIN_LOG2 12
/** @brief log2(ZXC_BLOCK_SIZE_MAX) - exponent code for maximum block size. */
#define ZXC_BLOCK_SIZE_MAX_LOG2 21
/** @brief Default block size (512 KB). */
#define ZXC_BLOCK_SIZE_DEFAULT (512 * 1024)
/** @brief Minimum allowed block size (4 KB = 2^12). */
#define ZXC_BLOCK_SIZE_MIN (1U << ZXC_BLOCK_SIZE_MIN_LOG2)
/** @brief Maximum allowed block size (2 MB = 2^21). */
#define ZXC_BLOCK_SIZE_MAX (1U << ZXC_BLOCK_SIZE_MAX_LOG2)
/** @} */ /* end of block_size */

/**
 * @defgroup dictionary Dictionary
 * @brief Constants for pre-trained dictionary support.
 * @{
 */
/** @brief Maximum dictionary content size in bytes (64 KB - 1).
 *
 * 16-bit for two reasons that happen to cap at the same 65535: the `.zxd`
 * header stores the content size in a 16-bit field, and LZ77 match offsets are
 * 16-bit (max distance 65535). */
#define ZXC_DICT_SIZE_MAX ((1U << 16) - 1U)
/** @brief Magic word opening a .zxd dictionary file, stored little-endian (C7 D1 B0 9C). */
#define ZXC_DICT_MAGIC 0x9CB0D1C7U
/** @brief Size of the .zxd dictionary file header in bytes. */
#define ZXC_DICT_HEADER_SIZE 16
/** @brief Size of a packed literal Huffman code-lengths table (256 symbols,
 *         4 bits each): the shared table in a .zxd file, and internally the
 *         per-block lengths header. See zxc_dict_train_huf() / zxc_dict_huf(). */
#define ZXC_HUF_TABLE_SIZE 128
/** @} */ /* end of dictionary */

/**
 * @defgroup threading Threading Limits
 * @brief Bounds on thread-count parameters accepted by the streaming APIs.
 * @{
 */
/** @brief Thread cap, a guard against memory exhaustion: each worker of the
 *  @c FILE* pipeline holds up to four jobs of two blocks. The @c THREADS
 *  setters refuse more; zxc_seekable_decompress_range_mt() clamps to it. */
#define ZXC_MAX_THREADS 512
/** @} */ /* end of threading */

/**
 * @defgroup file_format File Format Constants
 * @brief On-disk byte sizes of the archive header and footer.
 *
 * @{
 */
/** @brief Magic word opening every frame, stored little-endian (F5 2E B0 9C). */
#define ZXC_MAGIC_WORD 0x9CB02EF5U
/** @brief File header size: Magic(4) + Version(1) + Chunk(1) + Flags(1) + Reserved(7) +
 * Checksum(2). */
#define ZXC_FILE_HEADER_SIZE 16
/** @brief Smallest file footer: original_size(1) + compressed_size(1) + lengths(1). */
#define ZXC_FILE_FOOTER_MIN_SIZE 3
/** @brief Largest file footer, digest excluded: original_size(8) + compressed_size(8) + lengths(1).
 */
#define ZXC_FILE_FOOTER_MAX_SIZE 17
/** @brief Archive digest, opening the footer when the header sets HAS_CHECKSUM. */
#define ZXC_FILE_DIGEST_SIZE 8
/** @} */ /* end of file_format */

/**
 * @defgroup levels Compression Levels
 * @brief Predefined compression levels for the ZXC library.
 *
 * Higher levels spend more encoding time for a better ratio; decoding stays
 * fast at every level.
 * @{
 */

/**
 * @brief ZXC compression levels.
 *
 * Pass one as the @p level of zxc_compress() or zxc_stream_compress().
 */
typedef enum {
    ZXC_LEVEL_FASTEST = 1,  /**< Fastest compression; lowest ratio. Best for real-time use. */
    ZXC_LEVEL_FAST = 2,     /**< Fast compression; slightly better ratio than FASTEST. */
    ZXC_LEVEL_DEFAULT = 3,  /**< Recommended default: better ratio and decode speed than LZ4. */
    ZXC_LEVEL_BALANCED = 4, /**< Balanced trade-off between ratio and decode speed. */
    ZXC_LEVEL_COMPACT = 5,  /**< Denser encoding. Best for storage, firmware, and assets. */
    ZXC_LEVEL_DENSITY = 6,  /**< Higher density: adds Huffman-coded literals on top of COMPACT. */
    ZXC_LEVEL_ULTRA = 7     /**< Maximum density: Huffman-coded literals and sequence tokens. */
} zxc_compression_level_t;

/** @} */ /* end of levels */

#endif  // ZXC_CONSTANTS_H
