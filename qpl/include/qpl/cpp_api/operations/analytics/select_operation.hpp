/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_SELECT_OPERATION_HPP
#define QPL_SELECT_OPERATION_HPP

#include "qpl/cpp_api/operations/analytic_operation.hpp"
#include "qpl/cpp_api/operations/common_operation.hpp"
#include "qpl/cpp_api/util/qpl_allocation_util.hpp"
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
auto validate_operation(select_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs select function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/select_example.cpp QPL_HIGH_LEVEL_SELECT_EXAMPLE
 */
class select_operation final : public operation_with_mask {
    friend class internal::analytic_operation_builder<select_operation>;

    friend class internal::operation_with_mask_builder<select_operation>;

    template <execution_path path>
    friend auto internal::execute(select_operation &operation,
                                  int32_t numa_id) -> execution_result<uint32_t, sync>;
    template <execution_path path>
    friend auto internal::validate_operation(select_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = internal::operation_with_mask_builder<select_operation>;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr select_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr select_operation(const select_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr select_operation(select_operation &&other) = default;

    /**
     * @brief Constructor that accepts bit mask for operation
     *
     * @param  mask              pointer to mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    constexpr select_operation(const uint8_t *mask, size_t mask_byte_length)
            : operation_with_mask(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const select_operation &other) -> select_operation & = default;

    void reset_mask(const uint8_t *mask, size_t mask_byte_length) noexcept override;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_SELECT_OPERATION_HPP
