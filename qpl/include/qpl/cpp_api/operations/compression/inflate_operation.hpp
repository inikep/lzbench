/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_INFLATE_OPERATION_HPP
#define QPL_INFLATE_OPERATION_HPP

#include "qpl/cpp_api/operations/compression_operation.hpp"
#include "qpl/cpp_api/operations/common_operation.hpp"
#include "qpl/cpp_api/util/qpl_service_functions_wrapper.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

namespace internal {

template <execution_path path>
auto call_inflate(inflate_operation &operation,
                  uint8_t *buffer_ptr,
                  size_t buffer_size) -> std::pair<uint32_t, uint32_t>;

}

/**
 * @brief operation_t that performs decompression function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/compression_example.cpp QPL_HIGH_LEVEL_COMPRESSION_EXAMPLE
 */
class inflate_operation final : public compression_operation {
    template <execution_path path>
    friend
    class inflate_stream;

    class inflate_operation_builder;

    friend class internal::compression_operation_builder<inflate_operation,
                                                         inflate_operation_builder>;

    template <execution_path path>
    friend auto internal::call_inflate(inflate_operation &operation,
                                       uint8_t *buffer_ptr,
                                       size_t buffer_size) -> std::pair<uint32_t, uint32_t>;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = inflate_operation_builder;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    inflate_operation(const inflate_operation &other) = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    inflate_operation(inflate_operation &&other) = default;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    explicit inflate_operation() : compression_operation() {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    auto operator=(const inflate_operation &other) -> inflate_operation & = default;

protected:
    /**
     * @brief Function for obtaining decompression properties
     *
     * @return @ref deflate_properties object that contains information about mode, level, etc
     */
    auto get_properties() -> inflate_properties;

    void set_job_buffer(uint8_t *buffer) noexcept override;

    auto get_gzip_mode() -> bool override;

private:
    inflate_properties properties_{}; /**< Meta-information about decompression operation */
};

/**
 * @brief Builder for @ref inflate_operation (performs detailed configuration
 *        or re-configuration of the operation)
 */
class inflate_operation::inflate_operation_builder
        : public internal::compression_operation_builder<inflate_operation, inflate_operation_builder> {
    using parent_builder = compression_operation_builder<inflate_operation, inflate_operation_builder>;

public:
    /**
     * @brief Duplicates @ref inflate_operation simple constructor
     */
    explicit inflate_operation_builder() : parent_builder() {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    explicit inflate_operation_builder(const inflate_operation &operation)
            : compression_operation_builder(operation) {
        // Empty constructor
    }

    /**
     * @brief Sets compression mode (canned_mode)
     *
     * @tparam  current_mode  mode that should be used while decompression
     *
     * @note canned mode require the user to specify a pre-built Huffman table
     */
    template <compression_modes current_mode,
            class = typename std::enable_if<current_mode == compression_modes::canned_mode>::type>
    [[nodiscard]] auto compression_mode(huffman_table <huffman_table_type::inflate> table) -> inflate_operation_builder & {
        parent_builder::operation_.properties_.compression_mode_ = current_mode;
        parent_builder::operation_.properties_.huffman_table_    = std::move(table);

        return *this;
    }
};

/**
 * @brief Makes inflate huffman table from deflate huffman table
 *
 * @tparam  path  execution path
 *
 * @param  deflate_table  source compression table, which is used to build inflate huffman table
 *
 * @return instance of @link qpl::huffman_table<qpl::huffman_table_type::inflate> @endlink
 */
template <execution_path path>
auto make_inflate_table(huffman_table <huffman_table_type::deflate> &deflate_table)
-> qpl::huffman_table<huffman_table_type::inflate> {
    huffman_table<huffman_table_type::inflate> inflate_table{};

    util::build_decompression_huffman_table<path>(deflate_table.get_table_data(),
                                                  inflate_table.get_table_data());

    return inflate_table;
}

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_INFLATE_OPERATION_HPP
