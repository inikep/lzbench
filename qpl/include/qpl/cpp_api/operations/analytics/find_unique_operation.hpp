/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_FIND_UNIQUE_OPERATION_HPP
#define QPL_FIND_UNIQUE_OPERATION_HPP

#include "qpl/cpp_api/operations/analytic_operation.hpp"
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
auto validate_operation(find_unique_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs find unique function
 */
class find_unique_operation final : public analytics_operation {
    class find_unique_operation_builder;

    friend class internal::analytic_operation_builder<find_unique_operation, find_unique_operation_builder>;

    template <execution_path path>
    friend auto internal::execute(find_unique_operation &operation,
                                  int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(find_unique_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = find_unique_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr find_unique_operation() : analytics_operation(false) {
        // Empty constructor
    }

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr find_unique_operation(const find_unique_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr find_unique_operation(find_unique_operation &&other) = default;

    /**
     * @brief Constructor that makes it possible to set number of bits to ignore
     *
     * @param  low_order_bits_to_ignore   number of low order bits that should be ignored
     * @param  high_order_bits_to_ignore  number of high order bits that should be ignored
     */
    constexpr find_unique_operation(uint32_t low_order_bits_to_ignore, uint32_t high_order_bits_to_ignore)
            : analytics_operation(false),
              number_low_order_bits_ignored_(low_order_bits_to_ignore),
              number_high_order_bits_ignored_(high_order_bits_to_ignore) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const find_unique_operation &other) -> find_unique_operation & = default;

    [[nodiscard]] auto get_output_vector_width() const noexcept -> uint32_t override;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    uint32_t number_low_order_bits_ignored_  = 0;
    uint32_t number_high_order_bits_ignored_ = 0;
};

/**
 * @brief Builder for @ref find_unique_operation (performs detailed configuration or
 *        re-configuration of the operation)
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/find_unique_example.cpp QPL_HIGH_LEVEL_FIND_UNIQUE_EXAMPLE
 */
class find_unique_operation::find_unique_operation_builder
        : public internal::analytic_operation_builder<find_unique_operation, find_unique_operation_builder> {
    using parent_builder = analytic_operation_builder<find_unique_operation, find_unique_operation_builder>;

public:
    /**
     * @brief Duplicates @ref find_unique_operation constructor
     */
    constexpr explicit find_unique_operation_builder(uint32_t low_order_bits_to_ignore = 0u,
                                                     uint32_t high_order_bits_to_ignore = 0u)
            : parent_builder(find_unique_operation(low_order_bits_to_ignore, high_order_bits_to_ignore)) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    constexpr explicit find_unique_operation_builder(find_unique_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets number of low order bits that should be ignored
     */
    [[nodiscard]] auto number_low_order_bits_to_ignore(uint32_t value) -> find_unique_operation_builder &;

    /**
     * @brief Sets number of high order bits that should be ignored
     */
    [[nodiscard]] auto number_high_order_bits_to_ignore(uint32_t value) -> find_unique_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_FIND_UNIQUE_OPERATION_HPP
