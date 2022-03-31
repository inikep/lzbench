/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_HIGH_LEVEL_API_DEFINITIONS_HPP
#define QPL_HIGH_LEVEL_API_DEFINITIONS_HPP

#include <cstdint>

namespace qpl {

/**
 * @brief Contains supported execution paths
 */
enum execution_path : uint32_t {
    auto_detect,    /**< Tries to execute in hardware and executes in software in unsuccessful case */
    hardware,       /**< Tries to execute in hardware */
    software        /**< Executes in software */
};

}

#endif // QPL_HIGH_LEVEL_API_DEFINITIONS_HPP
