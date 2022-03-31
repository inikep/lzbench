/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_STATUS_HANDLER_HPP
#define QPL_STATUS_HANDLER_HPP

#include <cstdint>

#include "qpl/cpp_api/util/exceptions.hpp"

namespace qpl::util {

/**
 * @brief Handles low-level status code using high-level rules
 *
 * @param  status_code  code that should be handled
 */
void handle_status(uint32_t status_code);

} // namespace qpl::util

#endif // QPL_STATUS_HANDLER_HPP
