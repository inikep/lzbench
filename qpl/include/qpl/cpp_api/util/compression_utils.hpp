/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_HIGH_LEVEL_API_COMPRESSION_UTILS_HPP
#define QPL_HIGH_LEVEL_API_COMPRESSION_UTILS_HPP

#include <type_traits>

#include "qpl/cpp_api/common/definitions.hpp"

namespace qpl::util {

template <class operation_t>
auto get_buffer_size() -> uint32_t;

} // namespace qpl::util

#endif // QPL_HIGH_LEVEL_API_COMPRESSION_UTILS_HPP
