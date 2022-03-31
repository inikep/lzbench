/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_CONSTANTS_HPP
#define QPL_CONSTANTS_HPP

#include <cstdint>

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

constexpr uint32_t bit_bit_length   = 1;     /**< Number of bits in bit */
constexpr uint32_t byte_bit_length  = 8;     /**< Number of bits in byte */
constexpr uint32_t short_bit_length = 16;    /**< Number of bits in short integer */
constexpr uint32_t uint_bit_length  = 32;    /**< Number of bits in unsigned integer */

namespace util {
/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * Power of 2 that is the minimal size for a mini-block
 */
constexpr uint32_t minimal_mini_block_size_power = 8;

/**
 * Power of 2 that represents bit-length of one byte
 */
constexpr uint32_t byte_bit_length_power = 3;

/**
 * Number of service indices that are added to each deflate block in mini-blocks mode
 */
constexpr uint32_t additional_block_indexes = 2;

constexpr uint32_t additional_stream_indexes = 1; /**< @todo */

/**
 * Number of bytes needs to store stored-block's header
 */
constexpr uint32_t stored_block_header_byte_length = 5;

/** @} */

} // namespace util

namespace messages {
/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * @brief Messages for exceptions
 * @{
 */
constexpr const char *being_processed                = "operation_t or chain is still being processed";
constexpr const char *more_output_needed             = "Decompression operation filled output buffer before "
                                                       "finishing input. Need to submit a new operation with "
                                                       "the remaining input and a fresh output buffer";
constexpr const char *larger_output_needed           = "No progress was made because output buffer was too "
                                                       "small. Please resubmit with a larger output buffer";
constexpr const char *invalid_deflate_data           = "Invalid lookback distance found";
constexpr const char *invalid_parameter              = "Invalid parameter value in the job structure";
constexpr const char *buffer_exceeds_max_size        = "Buffer exceeds max size supported by hardware";
constexpr const char *library_internal_error         = "Unexpected internal error condition";
constexpr const char *verify_error                   = "CRC of decompressed verify output did not match CRC of input";
constexpr const char *invalid_index_generation       = "Parameters inappropriate for indexing usage";
constexpr const char *index_table_missed             = "Index table is not set";
constexpr const char *index_array_too_small          = "Indexing buffer too small";
constexpr const char *invalid_gzip_header            = "Invalid GZIP header";
constexpr const char *input_too_small                = "Buffer too small to hold GZIP header";
constexpr const char *invalid_block_size             = "Invalid block size used during indexing";
constexpr const char *invalid_huffman_table          = "Invalid Huffman table";
constexpr const char *null_ptr_error                 = "Null pointer error";
constexpr const char *incorrect_ignore_bits_value    = "Incorrect ignoreBits value (ignoreLowOrder + ignoreHighOrder "
                                                       "must be beyond 0..32)";
constexpr const char *incorrect_crc_64_polynomial    = "Incorrect polynomial value for CRC64";
constexpr const char *memory_allocation_error        = "Memory was not successfully allocated";
constexpr const char *incorrect_size                 = "Incorrect size. One of containers has not enough bytes";
constexpr const char *incorrect_prle_format          = "PRLE format is incorrect";
constexpr const char *output_overflow                = "Output index value is greater than max available for current "
                                                       "output data type";
constexpr const char *short_mask                     = "Mask buffer has less bytes than required "
                                                       "to process input elements";
constexpr const char *short_destination              = "Destination buffer has less bytes than required "
                                                       "to process input elements";
constexpr const char *distance_spans_mini_blocks     = "Distance spans mini-block boundary on indexing";
constexpr const char *length_spans_mini_blocks       = "Length spans mini-block boundary on indexing";
constexpr const char *verif_invalid_block_size       = "Invalid block size (not multiple of mini-block size)";
constexpr const char *incorrect_prle_bit_width       = "Bit width defined by PRLE stream was larger than 32";
constexpr const char *short_source                   = "Source buffer has less bytes than required "
                                                       "to process input elements";
constexpr const char *invalid_rle_counter            = "During RLE-burst, the cumulative count decreased(i.e. "
                                                       "count < prev count), or a count exceeded 2^16";
constexpr const char *invalid_zero_decompress_header = "Invalid header for ZeroDecompress functionality";
constexpr const char *no_any_exception_occurred      = "No any exception occurred";
/** @} */

/** @} */

} // namespace messages

/** @} */

} // namespace qpl

#endif // QPL_CONSTANTS_HPP
