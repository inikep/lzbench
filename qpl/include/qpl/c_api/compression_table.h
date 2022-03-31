/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  Job API (public C API)
 */

#ifndef QPL_HISTOGRAM_H_
#define QPL_HISTOGRAM_H_

#include <stdint.h>

#include "qpl/c_api/status.h"
#include "qpl/c_api/statistics.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup JOB_API_DEFINITIONS
 * @{
 */

/**
 * @name Deflate utility flags
 * @anchor deflate_utility_flags
 * @{
 */

/**
 * Flag which indicates whenever hardware representation of compression/decompression table should be used
 */
#define QPL_HW_REPRESENTATION            0x01u

/**
 * Flag which indicates whenever deflate header should be used
 */
#define QPL_DEFLATE_REPRESENTATION       0x04u

/**
 * Flag which indicates whenever software representation of compression/decompression table should be used
 */
#define QPL_SW_REPRESENTATION            0x08u

/**
 * Flag which indicates whenever huffman only representation of compression/decompression table should be used
 */
#define QPL_HUFFMAN_ONLY_REPRESENTATION  0x10u

/**
 * Combine all (software, hardware, deflate) representation flags to build the complete compression table
 */
#define QPL_COMPLETE_COMPRESSION_TABLE (QPL_HW_REPRESENTATION | QPL_DEFLATE_REPRESENTATION | QPL_SW_REPRESENTATION)
/** @} */

/**
 * @brief Internal structure that holds information for @ref qpl_op_compress operation
 */
typedef struct qpl_compression_huffman_table  qpl_compression_huffman_table;
extern const size_t QPL_COMPRESSION_TABLE_SIZE; /**< Size of the compression table in bytes*/

/**
 * @brief Internal structure that holds information for @ref qpl_op_decompress operation
 */
typedef struct qpl_decompression_huffman_table qpl_decompression_huffman_table;
extern const size_t QPL_DECOMPRESSION_TABLE_SIZE; /**< Size of the decompression table in bytes */

/**
 * @brief Structure for intermediate representation of Huffman token
 */
typedef struct {
    uint8_t  value;          /**< Encoded value */
    uint8_t  code_length;    /**< Length of Huffman code for given value */
    uint16_t code;           /**< Huffman code for given value */
} qpl_huffman_triplet;

/** @} */

/**
 * @addtogroup JOB_API_FUNCTIONS
 * @{
 */

/**
 * @brief Builds compression table out of deflate histogram
 *
 * @param[in]  histogram_ptr         Pointer to the histogram used in the compression table building
 * @param[out] table_ptr             Pointer to compression table that should be built
 * @param[in]  representation_flags  Flag that indicates which compression table representation should be built
 *
 * @note All representation flags are valid for this function (except for combination of huffman only and deflate representations)
 * @note Built table is guaranteed to be complete (in terms of that every deflate token is assigned a value)
 *
 * @return One of statuses presented in the @ref qpl_status
 */
QPL_API(qpl_status, qpl_build_compression_table, (const qpl_histogram *histogram_ptr,
                                                  qpl_compression_huffman_table *table_ptr,
                                                  uint32_t representation_flags)) ;

/**
 * @brief Builds decompression table out of intermediate representation
 *
 * @param[in]   triplets_ptr          Pointer to an array of triplets
 * @param[in]   triplets_count        Size of the array
 * @param[out]  table_ptr             Pointer to decompression table that should be built
 * @param[in]   representation_flags  Flags that specify what type of representation should be built
 *
 * @note Valid values for representation_flags are: QPL_HW_REPRESENTATION, QPL_SW_REPRESENTATION.
 *
 * @return One of statuses presented in the @ref qpl_status
 */
QPL_API(qpl_status, qpl_triplets_to_decompression_table, (const qpl_huffman_triplet *triplets_ptr,
                                                          size_t triplets_count,
                                                          qpl_decompression_huffman_table *table_ptr,
                                                          uint32_t representation_flags)) ;

/**
 * @brief Builds decompression table out of intermediate representation
 *
 * @param[in]   triplets_ptr          Pointer to an array of triplets
 * @param[in]   triplets_count        Size of the array
 * @param[out]  table_ptr             Pointer to compression table that should be built
 * @param[in]   representation_flags  Flag that indicates which compression table should be built
 *
 * @note Valid values for representation_flags are: QPL_HW_REPRESENTATION, QPL_SW_REPRESENTATION.
 *
 * @return One of statuses presented in the @ref qpl_status
 */
QPL_API(qpl_status, qpl_triplets_to_compression_table, (const qpl_huffman_triplet *triplets_ptr,
                                                        size_t triplets_count,
                                                        qpl_compression_huffman_table *table_ptr,
                                                        uint32_t representation_flags)) ;

/**
 * @brief Converts compression table to the decompression
 *
 * @param[in]  compression_table_ptr    Pointer to source compression table
 * @param[out] decompression_table_ptr  Pointer to decompression table to be built
 * @param[in]  representation_flags     Flag that indicates which decompression table representation should be built
 *
 * @note Valid values for representation_flags are: QPL_HW_REPRESENTATION, QPL_SW_REPRESENTATION, QPL_DEFLATE_REPRESENTATION.
 *
 * @return One of statuses presented in the @ref qpl_status
 */
QPL_API(qpl_status, qpl_comp_to_decompression_table, (const qpl_compression_huffman_table *compression_table_ptr,
                                                      qpl_decompression_huffman_table *decompression_table_ptr,
                                                      uint32_t representation_flags)) ;

/** @} */

#ifdef __cplusplus
}
#endif

#endif //QPL_HISTOGRAM_H_
