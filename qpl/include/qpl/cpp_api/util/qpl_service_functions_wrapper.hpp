/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_SERVICE_FUNCTIONS_WRAPPER_HPP
#define QPL_SERVICE_FUNCTIONS_WRAPPER_HPP

#include <cstdint>
#include "qpl/cpp_api/common/definitions.hpp"

struct qpl_compression_huffman_table;
struct qpl_decompression_huffman_table;

namespace qpl::util {

/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * @brief Returns size of the buffer that is required to keep a job
 */
auto get_job_size() -> uint32_t;

/**
 * @brief Wrapper to access low-level API huffman table building function
 *
 * @note According to rfc1951 literals/lengths histogram has 280 elements,
 *       offsets histogram has 30 elements
 * @note For now, this function is SW path only (underlying low-level function is run on SW
 *       no matter what execution path parameter value is), this will be fixed
 *       in the following release
 *
 * @param  source_ptr                    pointer to a vector, which is used to collect statistics
 * @param  source_length                 length of the source vector
 * @param  literal_length_histogram_ptr  pointer to literals/lengths histogram,
 *                                       which should be updated
 * @param  offsets_histogram_ptr         pointer to offsets histogram, which should be updated
 * @param  level                         compression level of algorithm used to gather deflate
 *                                       statistics (value of this parameter corresponds
 *                                       low-level qpl_compression_levels enum)
 * @param  path                          execution path for this function (value of this parameter
 *                                       corresponds low-level qpl_path_t enum)
 */
void gather_deflate_statistics(const uint8_t *source_ptr,
                               uint32_t source_length,
                               uint32_t *literal_length_histogram_ptr,
                               uint32_t *offsets_histogram_ptr,
                               uint32_t level,
                               execution_path path);

/**
 * @brief Wrapper to access low-level API deflate statistics gathering function
 *
 * @note According to rfc1951 literals/lengths histogram has 280 elements,
 *       offsets histogram has 30 elements
 * @note For now, this function is SW path only (underlying low-level function is run on SW
 *       no matter what execution path parameter value is), this will be fixed
 *       in the following release
 *
 * @tparam  path  execution path
 *
 * @param  huffman_table_buffer_ptr      pointer to buffer, which will contain result huffman table
 * @param  literal_length_histogram_ptr  pointer to filled literals/lengths histogram
 * @param  offsets_histogram_ptr         pointer to filled offsets histogram
 */
template <execution_path path>
void build_huffman_table_from_statistics(qpl_compression_huffman_table *huffman_table_buffer_ptr,
                                         const uint32_t *literal_length_histogram_ptr,
                                         const uint32_t *offsets_histogram_ptr);

/**
 * @brief Wrapper to access low-level API function that converts compression table to the decompression table
 *
 * @tparam  path  execution path
 *
 * @param  compression_table_buffer_ptr    pointer to source compression table
 * @param  decompression_table_buffer_ptr  pointer to decompression table to be built
 */
template <execution_path path>
void build_decompression_huffman_table(qpl_compression_huffman_table *compression_table_buffer_ptr,
                                       qpl_decompression_huffman_table *decompression_table_buffer_ptr);

/** @} */

} // namespace qpl

#endif // QPL_SERVICE_FUNCTIONS_WRAPPER_HPP
