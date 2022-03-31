/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_EXTRACT_OPERATION_HPP
#define QPL_EXTRACT_OPERATION_HPP

#include "qpl/cpp_api/operations/analytic_operation.hpp"
#include "qpl/cpp_api/util/constants.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

namespace internal {
template <execution_path path>
auto validate_operation(extract_operation &operation) -> uint32_t;
}

/**
 * @addtogroup HL_ANALYTICS
 * @{
 */

/**
 * @brief operation_t that performs extract function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/extract_example.cpp QPL_HIGH_LEVEL_EXTRACT_EXAMPLE
 */
class extract_operation final : public analytics_operation {
    class extract_operation_builder;

    friend class internal::analytic_operation_builder<extract_operation, extract_operation_builder>;

    template <execution_path path>
    friend auto internal::validate_operation(extract_operation &operation) -> uint32_t;

    template <execution_path path>
    friend auto internal::execute(extract_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = extract_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr extract_operation() : analytics_operation(false) {
        // Empty constructor
    }

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr extract_operation(const extract_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr extract_operation(extract_operation &&other) = default;

    /**
     * @brief Main constructor that sets appropriate boundaries
     *
     * @param  lower_index  index of the lowest element that should be extracted
     * @param  upper_index  index of the highest element that should be extracted
     */
    constexpr extract_operation(uint32_t lower_index, uint32_t upper_index)
            : analytics_operation(false),
              lower_index_(lower_index),
              upper_index_(upper_index) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const extract_operation &other) -> extract_operation & = default;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    uint32_t lower_index_ = 0;
    uint32_t upper_index_ = 0;
};

/**
 * @brief Builder for @ref extract_operation (performs detailed configuration
 *        or re-configuration of the operation)
 */
class extract_operation::extract_operation_builder
        : public internal::analytic_operation_builder<extract_operation, extract_operation_builder> {
    using parent_builder = analytic_operation_builder<extract_operation, extract_operation_builder>;

public:
    /**
     * @brief Duplicates @ref extract_operation main constructor
     */
    constexpr explicit extract_operation_builder(uint32_t lower_index, uint32_t upper_index)
            : parent_builder(extract_operation(lower_index, upper_index)) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param operation instance of operation that should be re-constructed
     */
    constexpr explicit extract_operation_builder(extract_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets lower boundary for operation
     */
    [[nodiscard]] auto lower_index(uint32_t value) -> extract_operation_builder &;

    /**
     * @brief Sets upper boundary for operation
     */
    [[nodiscard]] auto upper_index(uint32_t value) -> extract_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_EXTRACT_OPERATION_HPP
