/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_SCAN_OPERATION_HPP
#define QPL_SCAN_OPERATION_HPP

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
auto validate_operation(scan_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs scan function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/scan_example.cpp QPL_HIGH_LEVEL_SCAN_EXAMPLE
 */
class scan_operation final : public analytics_operation {
    class scan_operation_builder;

    friend class internal::analytic_operation_builder<scan_operation, scan_operation_builder>;

    template <execution_path path>
    friend auto internal::execute(scan_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(scan_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = scan_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr scan_operation() : analytics_operation(false) {
        // Empty constructor
    }

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr scan_operation(const scan_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr scan_operation(scan_operation &&other) = default;

    /**
     * @brief Main constructor that accepts comparator and required boundary
     *
     * @param  comparator  specifies type of scan (>, <, ==, !=)
     * @param  boundary    value that should be used for comparing
     */
    constexpr scan_operation(comparators comparator, uint32_t boundary)
            : analytics_operation(false),
              boundary_(boundary),
              comparator_(comparator) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const scan_operation &other) -> scan_operation & = default;

    [[nodiscard]] auto get_output_vector_width() const noexcept -> uint32_t override;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    comparators comparator_ = comparators::equals;
    uint32_t    boundary_   = 0;
};

/**
 * @brief Builder for @ref scan_operation (performs detailed configuration or re-configuration of the operation)
 */
class scan_operation::scan_operation_builder
        : public internal::analytic_operation_builder<scan_operation, scan_operation_builder> {
    using parent_builder = analytic_operation_builder<scan_operation, scan_operation_builder>;

public:
    /**
     * @brief Duplicates @ref scan_operation main constructor
     */
    constexpr explicit scan_operation_builder(comparators comparator, uint32_t boundary)
            : parent_builder(scan_operation(comparator, boundary)) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    constexpr explicit scan_operation_builder(scan_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets comparator
     */
    [[nodiscard]] auto comparator(comparators value) -> scan_operation_builder &;

    /**
     * @brief Sets boundary for comparing
     */
    [[nodiscard]] auto boundary(uint32_t value) -> scan_operation_builder &;

    /**
     * @brief Sets inclusivity (to make operation >=, <=)
     */
    [[nodiscard]] auto is_inclusive(bool value) -> scan_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_SCAN_OPERATION_HPP
