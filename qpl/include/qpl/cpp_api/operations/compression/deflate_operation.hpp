/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_OPERATION_HPP
#define QPL_DEFLATE_OPERATION_HPP

#include "qpl/cpp_api/operations/compression_operation.hpp"
#include "qpl/cpp_api/operations/common_operation.hpp"
#include "qpl/cpp_api/util/qpl_service_functions_wrapper.hpp"
#include "qpl/cpp_api/results/deflate_block.hpp"
#include "qpl/cpp_api/results/stream.hpp"
#include "qpl/cpp_api/util/deflate_block_utils.hpp"
#include "qpl/cpp_api/util/status_handler.hpp"

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

template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t>
auto build_deflate_block(deflate_operation &operation,
                         input_iterator_t source_begin,
                         input_iterator_t source_end,
                         mini_block_sizes mini_block_size) -> deflate_block<allocator_t>;

template <execution_path path>
auto call_deflate(deflate_operation &operation,
                  uint8_t *buffer_ptr,
                  size_t buffer_size) -> std::pair<uint32_t, uint32_t>;

}

/**
 * @brief operation_t that performs compression function
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/compression_example.cpp QPL_HIGH_LEVEL_COMPRESSION_EXAMPLE
 */
class deflate_operation final : public compression_operation {
    template <class...>
    friend
    class operation_chain;

    template <execution_path path>
    friend
    class deflate_stream;

    class deflate_operation_builder;

    friend class internal::compression_operation_builder<deflate_operation, deflate_operation_builder>;

    template <execution_path path,
            template <class> class allocator_t,
                             class input_iterator_t>
    friend auto internal::build_deflate_block(deflate_operation &operation,
                                              input_iterator_t source_begin,
                                              input_iterator_t source_end,
                                              mini_block_sizes mini_block_size) -> deflate_block<allocator_t>;

    template <execution_path path>
    friend auto internal::call_deflate(deflate_operation &operation,
                                       uint8_t *buffer_ptr,
                                       size_t buffer_size) -> std::pair<uint32_t, uint32_t>;

public:
    /**
     * Builder type for operation detailed configuring
     */
    using builder = deflate_operation_builder;

    /**
     * @brief Copy constructor
     * @param  other  object that should be copied
     */
    deflate_operation(const deflate_operation &other) = default;

    /**
     * @brief Move constructor
     * @param  other  object that should be moved
     */
    deflate_operation(deflate_operation &&other) = default;

    /**
     * @brief Simple default constructor (required for using of @ref operation_chain)
     */
    explicit deflate_operation() : compression_operation() {
        // Empty constructor
    }

    /**
     * @brief Default assignment operator
     */
    auto operator=(const deflate_operation &other) -> deflate_operation & = default;

protected:
    /**
     * @brief Enables compression by mini-blocks
     *
     * @param  index_array       pointer to array of indices that should be completed
     * @param  index_array_size  size of indices array
     * @param  mini_block_size   size of one mini-block (512, 1k, 2k, 4k, 8k, 16k, 32k)
     */
    void enable_random_access(internal::index *index_array,
                              uint32_t index_array_size,
                              mini_block_sizes mini_block_size) noexcept;

    /**
     * @brief Disables compression by mini-blocks
     */
    void disable_random_access() noexcept;

    /**
     * @brief Function for obtaining compression properties
     *
     * @return @ref deflate_properties object that contains information about mode, level, headers, etc
     */
    auto get_properties() -> deflate_properties;

    void set_job_buffer(uint8_t *buffer) noexcept override;

    auto get_gzip_mode() -> bool override;

private:
    deflate_properties properties_{}; /**< Meta-information about compression operation */
};

/**
 * @brief Builder for @ref deflate_operation (performs detailed configuration
 *        or re-configuration of the operation)
 */
class deflate_operation::deflate_operation_builder
        : public internal::compression_operation_builder<deflate_operation, deflate_operation_builder> {
    using parent_builder = compression_operation_builder<deflate_operation, deflate_operation_builder>;

public:
    /**
     * @brief Duplicates @ref deflate_operation simple constructor
     */
    explicit deflate_operation_builder() : parent_builder() {
        // Empty constructor
    }

    /**
     * @brief Reconstructs already existing operation
     *
     * @param  operation  instance of operation that should be re-constructed
     */
    explicit deflate_operation_builder(deflate_operation operation)
            : parent_builder(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets compression level
     */
    [[nodiscard]] auto compression_level(compression_levels value) -> deflate_operation_builder &;

    /**
     * @brief Sets compression mode (fixed_mode or dynamic)
     *
     * @tparam  current_mode  mode that should be used while compression
     */
    template <compression_modes current_mode,
            class = typename std::enable_if<current_mode == compression_modes::dynamic_mode ||
                                            current_mode == compression_modes::fixed_mode>::type>
    [[nodiscard]] auto compression_mode() -> deflate_operation_builder & {
        parent_builder::operation_.properties_.compression_mode_ = current_mode;

        return *this;
    }

    /**
     * @brief Sets compression mode (static_mode or canned_mode)
     *
     * @tparam  current_mode  mode that should be used while compression
     *
     * @note unlike fixed and dynamic modes, static and canned require the user
     * to specify a pre-built Huffman table
     */
    template <compression_modes current_mode,
            class = typename std::enable_if<current_mode == compression_modes::static_mode ||
                                            current_mode == compression_modes::canned_mode>::type>
    [[nodiscard]] auto compression_mode(huffman_table<huffman_table_type::deflate> table) -> deflate_operation_builder & {
        parent_builder::operation_.properties_.compression_mode_ = current_mode;
        parent_builder::operation_.properties_.huffman_table_    = std::move(table);

        return *this;
    }
};

namespace internal {

template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t>
auto build_deflate_block(deflate_operation &operation,
                         const input_iterator_t source_begin,
                         const input_iterator_t source_end,
                         const mini_block_sizes mini_block_size) -> deflate_block<allocator_t> {
    const auto source_size           = static_cast<uint32_t>(std::distance(source_begin, source_end));
    const auto buffer_size           = source_size + util::stored_block_header_byte_length;
    const auto number_of_mini_blocks = util::get_number_of_mini_blocks(source_size, mini_block_size);
    const auto index_array_size      = util::get_index_array_size(number_of_mini_blocks);

    deflate_block<allocator_t> block(mini_block_size, index_array_size, path);

    operation.enable_random_access(block.index_array_.get(), index_array_size, mini_block_size);

    stream<allocator_t> temporary_buffer(buffer_size);

    auto result = internal::execute<path, allocator_t>(operation,
                                                       source_begin,
                                                       source_end,
                                                       temporary_buffer.begin(),
                                                       temporary_buffer.end(),
                                                       numa_auto_detect);

    uint32_t result_size = 0;
    result.handle([&result_size](uint32_t value) -> void {
                      result_size = value;
                  },
                  [](uint32_t status) -> void {
                      util::handle_status(status);
                  });

    auto destination_end_iterator = temporary_buffer.begin();
    destination_end_iterator += result_size;

    block.assign(temporary_buffer.begin(), destination_end_iterator);
    block.set_compressed_size(source_size);

    operation.disable_random_access();

    return block;
}

} // namespace internal

/**
 * @brief Performs building of the deflate block for random access to compressed data
 *
 * @tparam  path               path that should be used for block building
 * @tparam  allocator_t        type of the allocator to be used for memory management
 * @tparam  input_iterator_t   type of input iterator
 * @tparam  output_iterator_t  type of output iterator
 *
 * @param   operation          instance of configured @ref deflate_operation that should be used
 *                             for compression
 * @param   source_begin       iterator to the beginning of the source
 * @param   source_end         iterator to the end of the source
 * @param   mini_block_size    size of one mini-block (512, 1k, 2k, 4k, 8k, 16k, 32k)
 *
 * @return instance of @ref deflate_block
 */
template <execution_path path = qpl::software,
        template <class> class allocator_t = std::allocator,
                         class input_iterator_t>
auto build_deflate_block(deflate_operation &operation,
                         const input_iterator_t source_begin,
                         const input_iterator_t source_end,
                         const mini_block_sizes mini_block_size) -> deflate_block<allocator_t> {
    return internal::build_deflate_block<path, allocator_t>(operation,
                                                            source_begin,
                                                            source_end,
                                                            mini_block_size);
}

/**
 * @brief
 *
 * @tparam  path                path that should be used for block building
 * @tparam  allocator_t         type of the allocator to be used for memory management
 * @tparam  input_container_t   type of input container
 * @tparam  output_container_t  type of output container
 *
 * @param   operation           instance of configured @ref deflate_operation that should be used
 *                              for compression
 * @param   source              container for source
 * @param   mini_block_size     size of one mini-block (512, 1k, 2k, 4k, 8k, 16k, 32k)
 *
 * @return instance of @ref deflate_block
 */
template <execution_path path = qpl::software,
        template <class> class allocator_t = std::allocator,
                         class input_container_t>
auto build_deflate_block(deflate_operation &operation,
                         const input_container_t source,
                         const mini_block_sizes mini_block_size) -> deflate_block<allocator_t> {
    return qpl::build_deflate_block<path, allocator_t>(operation,
                                                       source.begin(),
                                                       source.end(),
                                                       mini_block_size);
}

/**
 * @brief Builds huffman table from @link deflate_histogram @endlink
 *
 * @tparam  execution_path  specifies execution path
 *
 * @param   histogram       deflate histogram, which is used to build huffman table
 *
 * @return instance of @link qpl::huffman_table @endlink
 */
template <execution_path path>
auto make_deflate_table(deflate_histogram &histogram) -> qpl::huffman_table<huffman_table_type::deflate> {
    huffman_table<huffman_table_type::deflate> table{};

    util::build_huffman_table_from_statistics<path>(table.get_table_data(),
                                                    histogram.get_literals_lengths(),
                                                    histogram.get_offsets());

    return table;
}

/**
 * @brief Builds huffman table from @link deflate_histogram  @endlink
 *
 * @note For now, this function supports only SW path execution, this will be fixed
 * in the following release.
 *
 * @tparam  execution_path    specifies execution path
 * @tparam  input_iterator_t  type of input iterator (random access + uint8_t value type)
 *
 * @param   source_begin      iterator that points to begin of the source
 * @param   source_end        iterator that points to end of the source
 * @param   histogram         deflate histogram to update
 * @param   level             compression level of algorithm used to gather deflate statistics
 */
template <execution_path path = software, class input_iterator_t>
void update_deflate_statistics(const input_iterator_t &source_begin,
                               const input_iterator_t &source_end,
                               deflate_histogram &histogram,
                               compression_levels level = compression_levels::default_level) {
    static_assert(std::is_same<typename std::iterator_traits<input_iterator_t>::iterator_category,
                               std::random_access_iterator_tag>::value,
                  "Passed input iterator doesn't support random access");

    static_assert(std::is_same<typename std::iterator_traits<input_iterator_t>::value_type,
                               uint8_t>::value,
                  "Passed input iterator value type should be uint8_t");

    auto source_length = static_cast<uint32_t>(std::distance(source_begin, source_end));

    util::gather_deflate_statistics(&(*source_begin),
                                    source_length,
                                    histogram.get_literals_lengths(),
                                    histogram.get_offsets(),
                                    level,
                                    path);
}

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_DEFLATE_OPERATION_HPP
