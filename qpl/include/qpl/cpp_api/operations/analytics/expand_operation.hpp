/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_EXPAND_OPERATION_HPP
#define QPL_EXPAND_OPERATION_HPP

#include "qpl/cpp_api/operations/analytic_operation.hpp"
#include "qpl/cpp_api/operations/common_operation.hpp"
#include "qpl/cpp_api/util/constants.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HL_ANALYTICS
 * @{
 */

namespace internal {
template <execution_path path>
auto validate_operation(expand_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs expand function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/expand_example.cpp QPL_HIGH_LEVEL_EXPAND_EXAMPLE
 */
class expand_operation final : public operation_with_mask {
    friend class internal::analytic_operation_builder<expand_operation>;

    friend class internal::operation_with_mask_builder<expand_operation>;

    template <execution_path path>
    friend auto internal::execute(expand_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(expand_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = internal::operation_with_mask_builder<expand_operation>;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr expand_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr expand_operation(const expand_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr expand_operation(expand_operation &&other) = default;

    /**
     * @brief Constructor that accepts bit mask for the operation
     *
     * @param  mask              pointer to mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    constexpr expand_operation(const uint8_t *mask, size_t mask_byte_length)
            : operation_with_mask(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const expand_operation &other) -> expand_operation & = default;

protected:
    void reset_mask(const uint8_t *mask, size_t mask_byte_length) noexcept override;

    void set_job_buffer(uint8_t *buffer) noexcept override;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_EXPAND_OPERATION_HPP
