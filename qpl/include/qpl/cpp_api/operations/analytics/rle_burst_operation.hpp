/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_HIGH_LEVEL_API_OPERATIONS_ANALYTICS_RLE_BURST_OPERATION_HPP_
#define QPL_HIGH_LEVEL_API_OPERATIONS_ANALYTICS_RLE_BURST_OPERATION_HPP_

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
auto validate_operation(rle_burst_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs RLE-burst function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/rle_burst_example.cpp QPL_HIGH_LEVEL_RLE_BURST_EXAMPLE
 */
class rle_burst_operation final : public operation_with_mask {
    class rle_burst_operation_builder;

    friend class internal::analytic_operation_builder<rle_burst_operation, rle_burst_operation_builder>;

    friend class internal::operation_with_mask_builder<rle_burst_operation, rle_burst_operation_builder>;

    friend class internal::analytic_operation_builder<rle_burst_operation>;

    friend class internal::operation_with_mask_builder<rle_burst_operation>;

    template <execution_path path>
    friend auto internal::execute(rle_burst_operation &operation,
                                  uint8_t *source_buffer_ptr,
                                  size_t source_buffer_size,
                                  uint8_t *dest_buffer_ptr,
                                  size_t dest_buffer_size,
                                  uint8_t *mask_buffer_ptr,
                                  size_t mask_buffer_size,
                                  int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(rle_burst_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = rle_burst_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr rle_burst_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr rle_burst_operation(const rle_burst_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr rle_burst_operation(rle_burst_operation &&other) = default;

    /**
     * @brief Constructor that accepts bit mask for operation
     *
     * @param  mask              pointer to mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    constexpr rle_burst_operation(const uint8_t *mask, size_t mask_byte_length)
            : operation_with_mask(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const rle_burst_operation &other) -> rle_burst_operation & = default;

    void reset_mask(const uint8_t *mask, size_t mask_byte_length) noexcept override;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    uint32_t counter_bit_width_ = 0u; /**< Bit width of one counter in additional source */
};

/**
 * @brief Builder for @ref rle_burst_operation (performs detailed configuration or re-configuration of the operation)
 */
class rle_burst_operation::rle_burst_operation_builder
        : public internal::operation_with_mask_builder<rle_burst_operation, rle_burst_operation_builder> {
    using parent_builder = operation_with_mask_builder<rle_burst_operation, rle_burst_operation_builder>;

public:
    /**
     * @brief Duplicates @ref rle_burst_operation simple constructor
     */
    constexpr explicit rle_burst_operation_builder() : parent_builder() {
        // Empty constructor
    }

    /**
     * @brief Duplicates @ref rle_burst_operation mask constructor
     */
    constexpr explicit rle_burst_operation_builder(const uint8_t *mask, size_t mask_byte_length)
            : parent_builder(mask, mask_byte_length) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    constexpr explicit rle_burst_operation_builder(rle_burst_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets number of low order bits that should be ignored
     */
    [[nodiscard]] auto counter_bit_width(uint32_t value) -> rle_burst_operation_builder &;
};
/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif //QPL_HIGH_LEVEL_API_OPERATIONS_ANALYTICS_RLE_BURST_OPERATION_HPP_
