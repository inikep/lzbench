/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_BLOCK_UTILS_HPP
#define QPL_DEFLATE_BLOCK_UTILS_HPP

#include "qpl/cpp_api/operations/compression/inflate_stateful_operation.hpp"
#include "constants.hpp"

namespace qpl::util {

/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * @brief Gives proper integer representation for the passed enum
 *
 * @param  mini_block_size  enum representation of mini-block size
 *
 * @return size in bytes associated with passed enum value
 */
static inline auto convert_mini_block_size(const mini_block_sizes mini_block_size) noexcept -> uint32_t {
    return 1u << (mini_block_size + minimal_mini_block_size_power);
}

/**
 * @brief Calculates number of mini-blocks with given size that are required to perform
 *        compression in mini-blocks mode
 *
 * @param  source_size      size of the source in bytes
 * @param  mini_block_size  size of one mini-block
 *
 * @return required number of mini-blocks
 */
static inline auto get_number_of_mini_blocks(const uint32_t source_size,
                                             const mini_block_sizes mini_block_size) noexcept -> uint32_t {
    const uint32_t power_of_mini_block_size = mini_block_size + minimal_mini_block_size_power;
    return (source_size + (1u << power_of_mini_block_size) - 1u) >> power_of_mini_block_size;
}

/**
 * @brief Calculates required number of indices according to given number of mini-blocks
 */
static inline auto get_index_array_size(const uint32_t number_of_mini_blocks) noexcept -> uint32_t {
    return number_of_mini_blocks + additional_block_indexes + additional_stream_indexes;
}

/**
 * @brief Calculates relative position of the required element in the compressed stream
 *        with given mini-blocks size
 *
 * @param  element_index    index of the element in uncompressed stream
 * @param  mini_block_size  size of one mini-block
 *
 * @return relative index of the element in mini-block
 */
static inline auto get_mini_block_index(const uint32_t element_index,
                                        mini_block_sizes mini_block_size) noexcept -> uint32_t {
    return element_index >> (mini_block_size + minimal_mini_block_size_power);
}

/**
 * @brief Parses deflate header and sets pass inflate operation into state where it's capable
 *        of decompressing mini-blocks
 *
 * @param  source            pointer to the source
 * @param  destination       pointer to the destination
 * @param  destination_size  size of the destination
 * @param  index_array       index_array pointer to indices array that should be used
 *                           for decompression
 * @param  operation         instance of inflate operation that should be used for parsing
 */
void read_header(const uint8_t *source,
                 const uint8_t *destination,
                 uint32_t destination_size,
                 internal::index *index_array,
                 internal::inflate_stateful_operation &operation);

/**
 * @brief Decompresses required mini-block into destination
 *
 * @param  source            pointer to the source
 * @param  destination       pointer to the destination
 * @param  destination_size  size of the destination
 * @param  mini_block_index  index of required mini-block
 * @param  index_array       pointer to indices array that should be used for decompression
 * @param  operation         instance of inflate operation that should be used for decompression
 */
void read_mini_block(const uint8_t *source,
                     const uint8_t *destination,
                     uint32_t destination_size,
                     uint32_t mini_block_index,
                     const internal::index *index_array,
                     internal::inflate_stateful_operation &operation);

/** @} */

} // namespace qpl::util

#endif // QPL_DEFLATE_BLOCK_UTILS_HPP
