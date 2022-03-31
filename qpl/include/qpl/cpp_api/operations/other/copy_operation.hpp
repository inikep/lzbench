/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_COPY_OPERATION_HPP
#define QPL_COPY_OPERATION_HPP

#include "qpl/cpp_api/operations/operation.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HL_OTHER
 * @{
 */

/**
 * @brief operation_t that performs copy function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/copy_example.cpp QPL_HIGH_LEVEL_COPY_EXAMPLE
 */
class copy_operation final : public operation {
    template <execution_path path>
    friend auto internal::execute(copy_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

public:
    /**
     * @brief Simple default constructor
     */
    constexpr copy_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr copy_operation(const copy_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr copy_operation(copy_operation &&other) noexcept = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const copy_operation &other) -> copy_operation & = default;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif //QPL_COPY_OPERATION_HPP
