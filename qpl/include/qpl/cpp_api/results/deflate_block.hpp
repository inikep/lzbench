/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_BLOCK_HPP
#define QPL_DEFLATE_BLOCK_HPP

#include <cstring>
#include <memory>
#include <functional>

#include "qpl/cpp_api/util/deflate_block_utils.hpp"
#include "qpl/cpp_api/operations/compression/inflate_stateful_operation.hpp"

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

class deflate_operation;

template <template <class> class allocator_t>
class deflate_block;

namespace internal {
template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t>
auto build_deflate_block(deflate_operation &operation,
                         input_iterator_t source_begin,
                         input_iterator_t source_end,
                         mini_block_sizes mini_block_size) -> deflate_block<allocator_t>;
}

/**
 * @brief Class for accessing compressed stream elements without complete decompression
 *
 * @tparam  allocator_t  type of the allocator to be used for internal buffers allocations
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/deflate_block_example.cpp QPL_HIGH_LEVEL_DEFLATE_BLOCK_EXAMPLE
 */
template <template <class> class allocator_t = std::allocator>
class deflate_block final {
    template <execution_path path,
            template <class> class other_allocator_t,
                             class input_iterator_t>
    friend auto internal::build_deflate_block(deflate_operation &operation,
                                              input_iterator_t source_begin,
                                              input_iterator_t source_end,
                                              mini_block_sizes mini_block_size) -> deflate_block<other_allocator_t>;

    /**
     * @brief Type of deleter for internal job buffer
     */
    using unsigned_char_deleter = std::function<void(uint8_t *)>;

    /**
     * @brief Type of deleter for internal indices array
     */
    using index_deleter = std::function<void(internal::index *)>;

    /**
     * @brief Representation of one decompressed mini-block
     */
    struct mini_block_buffer final {
        /**
         * Decompressed content of the mini-block
         */
        std::unique_ptr<uint8_t[], unsigned_char_deleter> buffer;

        /**
         * Size of the mini-block
         */
        mini_block_sizes mini_block_size;

        /**
         * Size of the mini-block in bytes
         */
        uint32_t size = 0;

        /**
         * Index of stored mini-block
         */
        uint32_t stored_mini_block = 0;

        /**
         * Does mini-block contain decompressed data or not
         */
        bool is_empty = true;
    };

public:
    /**
     * @brief Deleted default constructor
     */
    deflate_block() = delete;

    /**
     * @brief Deleted copy constructor
     */
    deflate_block(const deflate_block &other) = delete;

    /**
     * @brief Default move constructor
     */
    deflate_block(deflate_block &&other) noexcept = default;

    /**
     * @brief Default destructor
     */
    ~deflate_block() = default;

    /**
     * @brief Overloaded subscript operator to access a compressed elements as if they were
     *        not compressed
     *
     * @param  index  position of desired element in decompressed stream
     *
     * @return element value
     */
    auto operator[](size_t index) -> uint8_t;

    /**
     * @brief Default assignment operator
     */
    auto operator=(const deflate_block &other) -> deflate_block & = delete;

    /**
     * @brief Getter for the size of the data after compression
     */
    [[nodiscard]] auto compressed_size() const noexcept -> size_t;

    /**
     * @brief Getter for the size of the data before compression
     */
    [[nodiscard]] auto size() const noexcept -> size_t;

protected:
    /**
     * @brief Simple constructor that is the only way to build a deflate block
     *
     * @param  mini_block_size   size of the mini-blocks inside the block
     * @param  index_array_size  number of indices that should be kept for proper work
     * @param  path              execution path that should be used
     */
    deflate_block(mini_block_sizes mini_block_size, uint32_t index_array_size, execution_path path);

    /**
     * @brief Assigns new source to the block
     *
     * @tparam  input_iterator_t  type of the source iterator
     *
     * @param   begin             pointer to the beginning of the stream
     * @param   end               pointer to the end of the stream
     */
    template <class input_iterator_t>
    void assign(input_iterator_t begin, input_iterator_t end);

    /**
     * @brief Sets the size of the compressed data
     */
    void set_compressed_size(uint32_t size);

    /**
     * Compressed data
     */
    std::unique_ptr<uint8_t[], unsigned_char_deleter> source_;

    /**
     * Buffer that keeps the job structure
     */
    std::unique_ptr<uint8_t[], unsigned_char_deleter> job_buffer_;

    /**
     * Array of indices for work with the block
     */
    std::unique_ptr<internal::index[], index_deleter> index_array_;

    /**
     * Instance of @ref internal.inflate_stateful_operation that is used for decompression
     */
    internal::inflate_stateful_operation operation_;

    /**
     * Something like cache for mini-blocks functionality
     */
    mini_block_buffer buffer_;

    /**
     * Size of compressed data
     */
    uint32_t source_size_ = 0;

    /**
     * Size of indices array
     */
    uint32_t index_array_size_ = 0;

    /**
     * Size of uncompressed data
     */
    uint32_t uncompressed_size_ = 0;
};

/** @} */

} // namespace qpl

#include "deflate_block.cxx"

#endif // QPL_DEFLATE_BLOCK_HPP
