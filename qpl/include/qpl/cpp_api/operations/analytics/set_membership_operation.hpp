/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_SET_MEMBERSHIP_OPERATION_HPP
#define QPL_SET_MEMBERSHIP_OPERATION_HPP

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
auto validate_operation(set_membership_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs set membership function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/set_membership_example.cpp QPL_HIGH_LEVEL_SET_MEMBERSHIP_EXAMPLE
 */
class set_membership_operation final : public operation_with_mask {
    class set_membership_operation_builder;

    friend class internal::analytic_operation_builder<set_membership_operation, set_membership_operation_builder>;

    friend class internal::operation_with_mask_builder<set_membership_operation, set_membership_operation_builder>;

    friend class internal::analytic_operation_builder<set_membership_operation>;

    friend class internal::operation_with_mask_builder<set_membership_operation>;

    template <execution_path path>
    friend auto internal::execute(set_membership_operation &operation,
                                  int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(set_membership_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = set_membership_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr set_membership_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr set_membership_operation(const set_membership_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr set_membership_operation(set_membership_operation &&other) = default;

    /**
     * @brief Constructor that accepts bitmask for the operation
     *
     * @param  mask              pointer to mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    constexpr set_membership_operation(const uint8_t *mask, size_t mask_byte_length)
            : operation_with_mask(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const set_membership_operation &other) -> set_membership_operation & = default;

    [[nodiscard]] auto get_output_vector_width() const noexcept -> uint32_t override;

protected:
    void reset_mask(const uint8_t *mask, size_t mask_byte_length) noexcept override;

    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    uint32_t number_low_order_bits_ignored_  = 0;
    uint32_t number_high_order_bits_ignored_ = 0;
};

/**
 * @brief Builder for @ref set_membership_operation (performs detailed configuration or re-configuration of the operation)
 */
class set_membership_operation::set_membership_operation_builder
        : public internal::operation_with_mask_builder<set_membership_operation, set_membership_operation_builder> {
    using parent_builder = operation_with_mask_builder<set_membership_operation, set_membership_operation_builder>;

public:
    /**
     * @brief Duplicates @ref set_membership_operation simple constructor
     */
    constexpr explicit set_membership_operation_builder() : parent_builder() {
        // Empty constructor
    }

    /**
     * @brief Duplicates @ref set_membership_operation mask constructor
     */
    constexpr explicit set_membership_operation_builder(const uint8_t *mask, size_t mask_byte_length)
            : parent_builder(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    constexpr explicit set_membership_operation_builder(set_membership_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets number of low order bits that should be ignored
     */
    [[nodiscard]] auto number_low_order_bits_to_ignore(uint32_t value) -> set_membership_operation_builder &;

    /**
     * @brief Sets number of high order bits that should be ignored
     */
    [[nodiscard]] auto number_high_order_bits_to_ignore(uint32_t value) -> set_membership_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_SET_MEMBERSHIP_OPERATION_HPP
