/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

/**
 * @defgroup HIGH_LEVEL_API Public API: High-level (C++)
 * @{
 * @brief Public Intel® Query Processing Library (Intel® QPL) C++ API (high-level API)
 * @}
 */

#ifndef QPL_INCLUDE_QPL_QPL_HPP_
#define QPL_INCLUDE_QPL_QPL_HPP_

// Chaining API
#include "cpp_api/chaining/operation_chain.hpp"
#include "cpp_api/chaining/merge_manipulator.hpp"

// Analytic Operations API
#include "cpp_api/operations/analytics/expand_operation.hpp"
#include "cpp_api/operations/analytics/extract_operation.hpp"
#include "cpp_api/operations/analytics/find_unique_operation.hpp"
#include "cpp_api/operations/analytics/scan_operation.hpp"
#include "cpp_api/operations/analytics/scan_range_operation.hpp"
#include "cpp_api/operations/analytics/select_operation.hpp"
#include "cpp_api/operations/analytics/set_membership_operation.hpp"
#include "cpp_api/operations/analytics/rle_burst_operation.hpp"

// Compression Operations API
#include "cpp_api/operations/compression/deflate_operation.hpp"
#include "cpp_api/operations/compression/inflate_operation.hpp"
#include "cpp_api/operations/compression/zero_compress_operation.hpp"
#include "cpp_api/operations/compression/zero_decompress_operation.hpp"
#include "cpp_api/results/deflate_block.hpp"
#include "cpp_api/results/deflate_stream.hpp"
#include "cpp_api/results/inflate_stream.hpp"

// Other Operations API
#include "cpp_api/operations/other/crc_operation.hpp"
#include "cpp_api/operations/other/copy_operation.hpp"

#endif //QPL_INCLUDE_QPL_QPL_HPP_
