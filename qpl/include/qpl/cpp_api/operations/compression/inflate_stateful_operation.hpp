/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_INFLATE_STATEFUL_OPERATION_HPP
#define QPL_INFLATE_STATEFUL_OPERATION_HPP

#include <utility>

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
 * @brief Entity that represents stateful decompression (aka decompression by chunks)
 */
class inflate_stateful_operation final : public compression_stateful_operation, public common_operation {
public:
    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    explicit inflate_stateful_operation() = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    inflate_stateful_operation(const inflate_stateful_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    inflate_stateful_operation(inflate_stateful_operation &&other) = default;

    /**
     * @brief Constructor that accepts existing decompression properties
     *
     * @param  properties  object of @ref compression_properties that contains meta-information
     *                     about compression
     */
    explicit inflate_stateful_operation(inflate_properties properties)
            : compression_stateful_operation(),
              properties_(std::move(properties)) {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    auto operator=(const inflate_stateful_operation &other) -> inflate_stateful_operation & = default;

    /**
     * @brief Sets operation to one of the specified states (Header, mini-blocks or none)
     */
    void enable_random_access(util::random_access_mode value) noexcept;

    /**
     * @brief Sets number of bits that should be ignored at the beginning of the stream
     */
    void set_start_bit_offset(uint32_t value) noexcept;

    /**
     * @brief Sets number of bits that should be ignored at the end of the stream
     */
    void set_end_bit_offset(uint32_t value) noexcept;

    void first_chunk(bool value) noexcept override;

    void last_chunk(bool value) noexcept override;

    void set_proper_flags() noexcept override;

    void set_buffers() noexcept override;

    void set_job_buffer(uint8_t *buffer) noexcept override;

    auto init_job(execution_path path) noexcept -> uint32_t override;

    auto get_processed_bytes() -> uint32_t override;

    /**
     * @todo Will be deprecated after ML introdusing
     */
    template <execution_path path>
    friend auto execute(inflate_stateful_operation &op,
                        uint8_t *source_begin,
                        uint8_t *source_end,
                        uint8_t *dest_begin,
                        uint8_t *dest_end) -> execution_result<uint32_t, sync>;

private:
    uint32_t start_bit_offset_ = 0u; /**< Number of bits that should be ignored at the beginning */
    uint32_t end_bit_offset_   = 0u; /**< Number of bits that should be ignored at the end */

    /**
     * Current decompression state
     */
    util::random_access_mode random_access_mode = util::disabled;

    /**
     * Meta-information about decompression operation
     */
    inflate_properties properties_{};
};

/** @} */

} // namespace qpl::internal

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_INFLATE_STATEFUL_OPERATION_HPP
