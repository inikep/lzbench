/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_HIGH_LEVEL_API_HUFFMAN_TABLE_HPP_
#define QPL_HIGH_LEVEL_API_HUFFMAN_TABLE_HPP_

#include <cstdint>

#include "qpl/cpp_api/util/pimpl.hpp"
#include "qpl/cpp_api/results/deflate_histogram.hpp"

struct qpl_compression_huffman_table;
struct qpl_decompression_huffman_table;

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

/**
 * @brief Contains supported Huffman table types
 */
enum class huffman_table_type {
    deflate,    /**< Represents compression table */
    inflate     /**< Represents decompression table */
};

/**
 * @brief Class specialization for representation of Huffman table
 *
 * @tparam  huffman_table_type  type of the kept value
 */
template <huffman_table_type table_type>
class huffman_table;

template <execution_path path = execution_path::software>
auto make_deflate_table(deflate_histogram &histogram) -> huffman_table<huffman_table_type::deflate>;

template <execution_path path = execution_path::software>
auto make_inflate_table(huffman_table<huffman_table_type::deflate> &deflate_table)
-> qpl::huffman_table<huffman_table_type::inflate>;

namespace internal {
class deflate_stateful_operation;

class inflate_stateful_operation;
}

/**
 * @brief Class that represents Huffman table for compression
 */
template <>
class huffman_table<huffman_table_type::deflate> {
    friend class deflate_operation;

    friend struct deflate_properties;

    friend class internal::deflate_stateful_operation;

    template <execution_path path>
    friend auto make_deflate_table(deflate_histogram &histogram) -> huffman_table<huffman_table_type::deflate>;

public:
    /**
     * @brief Default copy constructor
     */
    huffman_table(const huffman_table<huffman_table_type::deflate> &other);

    /**
     * @brief Default move constructor
     */
    huffman_table(huffman_table<huffman_table_type::deflate> &&other) noexcept;

    /**
     * @brief Default move assignment operator
     */
    auto operator=(huffman_table<huffman_table_type::deflate> &&other) noexcept -> huffman_table<huffman_table_type::deflate> &;

    /**
     * @brief Default copy assignment operator
     */
    auto operator=(const huffman_table<huffman_table_type::deflate> &other) -> huffman_table<huffman_table_type::deflate> &;

    ~huffman_table();

    /**
     * @brief Getter for the table data
     */
    auto get_table_data() -> qpl_compression_huffman_table *;

protected:
    /**
     * @brief Default constructor
     */
    explicit huffman_table();

private:
    static constexpr size_t impl_size  = 3904;
    static constexpr size_t impl_align = 64;

    util::pimpl<qpl_compression_huffman_table, impl_size, impl_align> table_;
};

/**
 * @brief Class that represents Huffman table for decompression
 */
template <>
class huffman_table<huffman_table_type::inflate> {
    friend class inflate_operation;

    friend struct inflate_properties;

    friend class internal::inflate_stateful_operation;

    template <execution_path path>
    friend auto make_inflate_table(huffman_table<huffman_table_type::deflate> &deflate_table)
    -> qpl::huffman_table<huffman_table_type::inflate>;

public:
    /**
     * @brief Default copy constructor
     */
    huffman_table(const huffman_table<huffman_table_type::inflate> &other);

    /**
     * @brief Default move constructor
     */
    huffman_table(huffman_table<huffman_table_type::inflate> &&other) noexcept;

    /**
     * @brief Default move assignment operator
     */
    auto operator=(huffman_table<huffman_table_type::inflate> &&other) noexcept -> huffman_table<huffman_table_type::inflate> &;

    /**
     * @brief Default copy assignment operator
     */
    auto operator=(const huffman_table<huffman_table_type::inflate> &other) -> huffman_table<huffman_table_type::inflate> &;

    ~huffman_table();

    /**
     * @brief Getter for the table data
     */
    auto get_table_data() -> qpl_decompression_huffman_table *;

protected:
    /**
     * @brief Default constructor
     */
    explicit huffman_table();

private:
    static constexpr size_t impl_size  = 27328;
    static constexpr size_t impl_align = 64;

    util::pimpl<qpl_decompression_huffman_table, impl_size, impl_align> table_;
};

/** @} */

} // namespace qpl

#endif // QPL_HIGH_LEVEL_API_HUFFMAN_TABLE_HPP_
