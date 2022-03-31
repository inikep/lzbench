/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_SCAN_RANGE_OPERATION_HPP
#define QPL_SCAN_RANGE_OPERATION_HPP

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
auto validate_operation(scan_range_operation &operation) -> uint32_t;
}

/**
 * @brief operation_t that performs scan range function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/scan_range_example.cpp QPL_HIGH_LEVEL_SCAN_RANGE_EXAMPLE
 */
class scan_range_operation final : public analytics_operation {
    class scan_range_operation_builder;

    friend class internal::analytic_operation_builder<scan_range_operation, scan_range_operation_builder>;

    template <execution_path path>
    friend auto internal::execute(scan_range_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

    template <execution_path path>
    friend auto internal::validate_operation(scan_range_operation &operation) -> uint32_t;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = scan_range_operation_builder;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    constexpr scan_range_operation()
            : analytics_operation(true) {
        // Empty constructor
    }

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr scan_range_operation(const scan_range_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr scan_range_operation(scan_range_operation &&other) = default;

    /**
     * @brief Main constructor that sets appropriate boundaries
     *
     * @param  lower_boundary  lower boundary of the range
     * @param  upper_boundary  upper boundary of the range
     */
    constexpr scan_range_operation(uint32_t lower_boundary, uint32_t upper_boundary)
            : analytics_operation(true),
              lower_boundary_(lower_boundary),
              upper_boundary_(upper_boundary) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const scan_range_operation &other) -> scan_range_operation & = default;

    [[nodiscard]] auto get_output_vector_width() const noexcept -> uint32_t override;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

private:
    uint32_t lower_boundary_ = 0;
    uint32_t upper_boundary_ = std::numeric_limits<uint8_t>::max();
};

/**
 * @brief Builder for @ref scan_range_operation (performs detailed configuration
 * or re-configuration of the operation)
 */
class scan_range_operation::scan_range_operation_builder
        : public internal::analytic_operation_builder<scan_range_operation, scan_range_operation_builder> {
    using parent_builder = analytic_operation_builder<scan_range_operation, scan_range_operation_builder>;

public:
    /**
     * @brief Duplicates @ref scan_range_operation main constructor
     */
    constexpr explicit scan_range_operation_builder(uint32_t lower_boundary, uint32_t upper_boundary)
            : parent_builder(scan_range_operation(lower_boundary, upper_boundary)) {

        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    constexpr explicit scan_range_operation_builder(scan_range_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets lower boundary for operation
     */
    [[nodiscard]] auto lower_boundary(uint32_t value) -> scan_range_operation_builder &;

    /**
     * @brief Sets upper boundary for operation
     */
    [[nodiscard]] auto upper_boundary(uint32_t value) -> scan_range_operation_builder &;

    /**
     * @brief Sets inclusivity (allows to perform inclusive or exclusive scan range)
     */
    [[nodiscard]] auto is_inclusive(bool value) -> scan_range_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_SCAN_RANGE_OPERATION_HPP
