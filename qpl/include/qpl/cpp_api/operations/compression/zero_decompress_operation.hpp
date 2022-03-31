/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_ZERO_DECOMPRESS_OPERATION_HPP
#define QPL_ZERO_DECOMPRESS_OPERATION_HPP

#include "qpl/cpp_api/operations/compression_operation.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

/**
 * @brief operation_t that performs zero decompress function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/zero_compression_example.cpp QPL_HIGH_LEVEL_ZERO_COMPRESSION_EXAMPLE
 */
class zero_decompress_operation final : public zero_operation {
    template <execution_path path>
    friend auto internal::execute(zero_decompress_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

public:
    /**
     * @brief Main constructor that accepts the zero input format
     *
     * @param  input_format  format used for zero operations
     */
    constexpr explicit zero_decompress_operation(zero_input_format input_format)
            : zero_operation(input_format) {
        // Empty constructor
    };

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr zero_decompress_operation(const zero_decompress_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr zero_decompress_operation(zero_decompress_operation &&other) noexcept = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const zero_decompress_operation &other) -> zero_decompress_operation & = default;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif //QPL_ZERO_DECOMPRESS_OPERATION_HPP
