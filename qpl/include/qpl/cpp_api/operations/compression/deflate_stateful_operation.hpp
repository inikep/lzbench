/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_STATEFUL_OPERATION_HPP
#define QPL_DEFLATE_STATEFUL_OPERATION_HPP

#include "qpl/cpp_api/operations/compression_operation.hpp"
#include "qpl/cpp_api/operations/common_operation.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl::internal {

/**
 * @addtogroup HL_PRIVATE
 * @{
 */

/**
 * @brief Entity that represents stateful compression (aka compression by chunks)
 */
class deflate_stateful_operation final : public compression_stateful_operation, public common_operation {
public:
    deflate_stateful_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    deflate_stateful_operation(const deflate_stateful_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    deflate_stateful_operation(deflate_stateful_operation &&other) = default;

    /**
     * @brief Constructor that accepts existing compression properties
     *
     * @param  properties  object of @ref deflate_properties that contains meta-information about compression
     */
    explicit deflate_stateful_operation(deflate_properties properties)
            : compression_stateful_operation(),
              properties_(properties) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    auto operator=(const deflate_stateful_operation &other) -> deflate_stateful_operation & = default;

    void first_chunk(bool value) noexcept override;

    void last_chunk(bool value) noexcept override;

    void set_proper_flags() noexcept override;

    void set_buffers() noexcept override;

    void set_job_buffer(uint8_t *buffer) noexcept override;

    auto execute() -> std::pair<uint32_t, uint32_t> override;

    auto get_processed_bytes() -> uint32_t override;

private:
    deflate_properties properties_{}; /**< Meta-information about compression operation */
};

/** @} */

} // namespace qpl::internal

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_DEFLATE_STATEFUL_OPERATION_HPP
