/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_COMPRESSION_OPERATION_HPP
#define QPL_COMPRESSION_OPERATION_HPP

#include "qpl/cpp_api/operations/operation.hpp"
#include "qpl/cpp_api/results/huffman_table.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @defgroup HL_COMPRESSION Compression API
 * @ingroup HL_PUBLIC
 * @{
 * @brief Compression entities and functions of high-level API
 */

/**
 * @brief Contains supported input formats for zero operations
 */
enum class zero_input_format {
    word_16_bit,    /**< For words series, where the word length is 16-bits */
    word_32_bit     /**< For words series, where the word length is 32-bits */
};

/**
 * @brief Contains supported CRC calculation types for many operations
 */
enum class crc_type {
    none,      /**< Do not calculate checksum */
    crc_32,    /**< To use 0x104c11db7 polynomial for crc calculation */
    crc_32c    /**< To use 0x11edc6f41 polynomial for crc calculation, which is the one used by iSCSI */
};

/**
 * @brief Contains supported compression levels
 */
enum compression_levels {
    /**
    * Comression levels for compression method
    */
    level_1 = 1,
    level_2 = 2,
    level_3 = 3,
    level_4 = 4,
    level_5 = 5,
    level_6 = 6,
    level_7 = 7,
    level_8 = 8,
    level_9 = 9,

    /**
     * Default compression level (good balance between compression ratio and speed)
     */
    default_level = level_1,

    /**
     * Max compression ratio
     */
    high_level = level_3
};

/**
 * @brief Contains supported compression modes
 */
enum compression_modes {
    /**
     * Compression with usage of standard (RFC 1951) Huffman table
     */
    fixed_mode,

    /**
     * Compression with custom pre-built Huffman table
     */
    static_mode,

    /**
     * Compression with custom Huffman table
     */
    dynamic_mode,

    /**
     * Compression with custom pre-built Huffman table, but no deflate header in output stream
     */
    canned_mode
};

/**
 * @brief Contains supported sizes of the mini-block
 */
enum mini_block_sizes : uint32_t {
    mini_block_size_none = 0u,    /**< No mini-blocks */
    mini_block_size_512  = 1u,    /**< Each 512 bytes are compressed independently */
    mini_block_size_1k   = 2u,    /**< Each 1 kb is compressed independently */
    mini_block_size_2k   = 3u,    /**< Each 2 kb are compressed independently */
    mini_block_size_4k   = 4u,    /**< Each 4 kb are compressed independently */
    mini_block_size_8k   = 5u,    /**< Each 8 kb are compressed independently */
    mini_block_size_16k  = 6u,    /**< Each 16 kb are compressed independently */
    mini_block_size_32k  = 7u     /**< Each 32 kb are compressed independently */
};

namespace internal {

/**
 * @addtogroup HL_PRIVATE
 * @{
 */

/**
 * @brief Struct that represents one instance of index used for mini-blocks feature
 */
struct index final {
    uint32_t bit_offset;    /**< Offset from the beginning to the start of mini-block in bits */
    uint32_t crc;           /**< CRC32 value of mini-block content */
};

/** @} */

} // namespace internal

namespace util {

/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * @brief Contains possible states of compressor while accessing compressed elements
 */
enum random_access_mode : uint32_t {
    disabled   = 0u,    /**< Simple decompression without anything else */
    mini_block = 1u,    /**< Reading mini-block content */
    header     = 2u     /**< Reading the header for further mini-block decompression */
};

/** @} */

} // namespace util

/**
 * @brief Common meta-data structure for compression operations
 */
struct compression_properties {
    bool              gzip_mode_        = false;                              /**< Is GZIP mode enabled */
    compression_modes compression_mode_ = compression_modes::dynamic_mode;    /**< Compression mode */
};

/**
 * @brief Compression specific meta-data
 */
struct deflate_properties : public compression_properties {
    /**
     * Compression level
     */
    compression_levels compression_level_ = compression_levels::default_level;

    /**
     * Size of one mini-block
     */
    mini_block_sizes mini_block_size_ = mini_block_sizes::mini_block_size_none;

    /**
     * Huffman table that should be used for compression
     */
    huffman_table<huffman_table_type::deflate> huffman_table_{};

    internal::index *index_array_     = nullptr;    /**< Pointer to indices array */
    uint32_t        index_array_size_ = 0;          /**< Size of indices array */
};

/**
 * @brief Compression specific meta-data
 */
struct inflate_properties : public compression_properties {
    /**
     * Huffman table that should be used for decompression
     */
    huffman_table<huffman_table_type::inflate> huffman_table_{};
};

/**
 * @brief Interface that contains compression operations specific properties
 */
class compression_operation : public operation {
public:
    /**
     * @brief Default simple constructor
     */
    constexpr explicit compression_operation() = default;

    /**
     * @brief Friend function that merges decompression operation with analytic ones
     */
    template <execution_path path,
            template <class> class allocator_t,
                             class input_iterator_t,
                             class output_iterator_t,
                             class operation_t>
    friend auto compose(compression_operation &decompress_operation,
                        operation_t &operation,
                        const input_iterator_t &source_begin,
                        const input_iterator_t &source_end,
                        const output_iterator_t &destination_begin,
                        const output_iterator_t &destination_end,
                        int32_t numa_id,
                        uint8_t *job_buffer,
                        uint32_t number_of_decompressed_elements) -> execution_result<uint32_t, sync>;

protected:
    /**
     * @brief Getter of currently set GZIP mode
     *
     * @return is GZIP mode enabled or not
     */
    virtual auto get_gzip_mode() -> bool = 0;
};

/**
 * @brief Interface that contains stateful compression specific methods
 */
class compression_stateful_operation : public operation {
public:
    /**
     * @brief Default simple constructor
     */
    constexpr explicit compression_stateful_operation() = default;

    /**
     * @brief Sets operation to the state where the next chunk is first
     */
    virtual void first_chunk(bool value) noexcept = 0;

    /**
     * @brief Sets operation to the state where the next chunk is last
     */
    virtual void last_chunk(bool value) noexcept = 0;

    /**
     * @brief Getter of currently processed bytes
     *
     * @return number of bytes that were processed
     */
    virtual auto get_processed_bytes() -> uint32_t = 0;

    bool is_first_chunk_ = false;    /**< Is next chunk first */
    bool is_last_chunk_  = false;    /**< Is next chunk last */
};

/**
 * @brief Interface that contains zero compress/decompress operations specific methods and fields
 */
class zero_operation : public operation {
public:
    /**
     * @brief Main constructor that accepts the zero input format
     *
     * @param  input_format  format used for zero operations
     */
    constexpr explicit zero_operation(zero_input_format input_format)
            : input_format_(input_format) {
        // Empty constructor
    };

protected:
    zero_input_format input_format_ = zero_input_format::word_16_bit;    /**< Zero input format */
    crc_type          crc_type_     = crc_type::none;                    /**< Crc type */
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_COMPRESSION_OPERATION_HPP
