/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_CRC_OPERATION_HPP_
#define QPL_CRC_OPERATION_HPP_

#include "qpl/cpp_api/operations/operation.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HL_OTHER
 * @{
 */

/**
 * @brief operation_t that performs crc calculation function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/crc_example.cpp QPL_HIGH_LEVEL_CRC_EXAMPLE
 */
class crc_operation final : public operation {
    class crc_operation_builder;

    template <execution_path path>
    friend auto internal::execute(crc_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = crc_operation_builder;

    /**
     * @brief Main constructor that accepts the polynomial
     *
     * @param  polynomial  polynomial used for the operation
     */
    constexpr explicit crc_operation(uint64_t polynomial)
            : polynomial_(polynomial) {
        // Empty constructor
    };

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr crc_operation(const crc_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr crc_operation(crc_operation &&other) noexcept = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const crc_operation &other) -> crc_operation & = default;

protected:
    void set_job_buffer(uint8_t *buffer) noexcept override;

    parsers  bit_order_  = little_endian_packed_array; /**< Input format for crc calculation */
    bool     is_inverse_ = false; /**< Is crc inversion forward or inverse */
    uint64_t polynomial_ = 0; /**< Polynomial used for the crc calculation  */
};

/**
 * @brief Builder for @ref extract_operation (performs detailed configuration
 *        or re-configuration of the operation)
 */
class crc_operation::crc_operation_builder final : public internal::operation_builder<crc_operation> {
public:
    /**
     * @brief Duplicates @ref extract_operation main constructor
     */
    constexpr explicit crc_operation_builder(uint64_t polynomial)
            : operation_builder(crc_operation(polynomial)) {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param operation instance of operation that should be re-constructed
     */
    constexpr explicit crc_operation_builder(crc_operation operation)
            : operation_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets bit order for operation
     */
    template <parsers current_parser>
    [[nodiscard]] auto bit_order() -> crc_operation_builder & {
        static_assert((current_parser == parsers::big_endian_packed_array ||
                       current_parser == parsers::little_endian_packed_array
                      ), "Intel QPL doesn't support such bit order");

        operation_builder::operation_.bit_order_ = current_parser;

        return *this;
    }

    /**
     * @brief Sets inversion (allows to perform forward or inverse crc calculation)
     */
    [[nodiscard]] auto is_inverse(bool value) -> crc_operation_builder &;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif //QPL_CRC_OPERATION_HPP_
