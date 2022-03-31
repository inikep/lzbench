/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_INFLATE_STREAM_HPP
#define QPL_INFLATE_STREAM_HPP

#include "qpl/cpp_api/results/compression_stream.hpp"
#include "qpl/cpp_api/operations/compression/inflate_operation.hpp"

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */
namespace internal {
class inflate_stateful_operation;
}

/**
 * @brief Class that performs decompression by chunks
 *
 * @tparam  path         execution path that should be used
 * @tparam  allocator_t  type of the allocator to be used
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/compression_stream_example.cpp QPL_HIGH_LEVEL_COMPRESSION_STREAM_EXAMPLE
 */
template <execution_path path = execution_path::software>
class inflate_stream : public compression_stream<path> {
public:
    /**
     * @brief Simple constructor that accepts instance of inflate operation
     *
     * @param  operation     instance of @ref inflate_operation that should be used  for decompression
     * @param  source_begin  pointer to the beginning of the source
     * @param  source_end    pointer to the end of the source
     */
    template <class input_iterator_t, template <class> class allocator_t = std::allocator>
    explicit inflate_stream(inflate_operation operation,
                            input_iterator_t source_begin,
                            input_iterator_t source_end)
            : compression_stream<path>(operation, std::distance(source_begin, source_end), allocator_t<uint8_t>()) {

        // TODO use allocator to allocate
        operation_ = std::make_unique<internal::inflate_stateful_operation>(operation.get_properties());

        const auto source_size = std::distance(source_begin, source_end);
        inflate_stream<path>::constructor_impl(&*source_begin, source_size );
    }

    /**
     * @brief Decompresses next chunk into output with specified boundaries
     *
     * @tparam  output_iterator_t  type of output iterator
     *
     * @param   destination_begin  pointer to the output beginning
     * @param   destination_end    pointer to the output end
     */
    template <class output_iterator_t>
    auto extract(const output_iterator_t &destination_begin,
                 const output_iterator_t &destination_end) -> inflate_stream & {
        // TODO check iterators requirements
        const auto destination_size = std::distance(destination_begin, destination_end);
        return inflate_stream::extract_impl(&*destination_begin, destination_size);
    }

protected:

    /**
     * @todo Method will be documented after ML introdusing
     */
    void constructor_impl(uint8_t *source_begin,
                          size_t source_size);

    /**
     * @todo Method will be documented after ML introdusing
     */
    auto extract_impl(uint8_t *destination_begin, size_t destination_size) -> inflate_stream &;

private:
    /**
     * Instance of inflate_stateful_operation that is used for decompression
     */
    std::unique_ptr<internal::inflate_stateful_operation> operation_;
};

/** @} */

} // namespace qpl

#endif // QPL_INFLATE_STREAM_HPP
