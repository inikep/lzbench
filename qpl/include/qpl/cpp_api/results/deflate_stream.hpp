/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_STREAM_HPP
#define QPL_DEFLATE_STREAM_HPP

#include "qpl/cpp_api/results/compression_stream.hpp"
#include "qpl/cpp_api/operations/compression/deflate_operation.hpp"
#include "qpl/cpp_api/operations/compression/deflate_stateful_operation.hpp"

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

/**
 * @brief Class that performs compression by chunks
 *
 * @tparam  path         execution path that should be used
 * @tparam  allocator_t  type of the allocator to be used
 *
 * Example of main usage:
 * @snippet high-level-api/simple-operations/compression_stream_example.cpp QPL_HIGH_LEVEL_COMPRESSION_STREAM_EXAMPLE
 */
template <execution_path path = execution_path::software>
class deflate_stream : public compression_stream<path> {
public:
    /**
     * @brief Simple constructor
     *
     * @param  operation         instance of @ref deflate_operation that should be used for compression
     * @param  destination_size  initial size of the destination buffer
     */
    template <template <class> class allocator_t = std::allocator>
    explicit deflate_stream(deflate_operation operation, const size_t destination_size)
            : compression_stream<path>(operation, destination_size, allocator_t<uint8_t>()) {
        operation_ = internal::deflate_stateful_operation(operation.get_properties());

        operation_.set_job_buffer(this->job_buffer_.get());
        operation_.init_job(path);
    }

    /**
     * @brief Performs compression of the new chunk with specified boundaries
     *
     * @tparam  input_iterator_t  type of input iterator
     *
     * @param   source_begin      pointer to the beginning of the source
     * @param   source_end        pointer to the end of the source
     */
    template <class input_iterator_t>
    auto push(const input_iterator_t &source_begin, const input_iterator_t &source_end) -> deflate_stream &;

    /**
         * @brief Method that changes the size of the buffer size to the specified
         *
         * @param  new_size  new destination buffer size
         */
    void resize(size_t new_size) noexcept;

    /**
     * @brief Performs compression of the new chunk considering it last and setting the stream to initial state
     *
     * @tparam  input_iterator_t  type of input iterator
     *
     * @param   source_begin      pointer to the beginning of the source
     * @param   source_end        pointer to the end of the source
     */
    template <class input_iterator_t>
    void flush(const input_iterator_t &source_begin, const input_iterator_t &source_end);

private:
    /**
     * @brief Internal implementation of operation execution
     *
     * @tparam  input_iterator_t  type of input iterator
     *
     * @param   source_begin      pointer to the beginning of the source
     * @param   source_end        pointer to the end of the source
     */
    template <class input_iterator_t>
    void submit_operation(const input_iterator_t &source_begin, const input_iterator_t &source_end);

    /**
     * Instance of operation that should be used for compression
     */
    internal::deflate_stateful_operation operation_;
};

/** @} */

} // namespace qpl

#include "qpl/cpp_api/results/deflate_stream.cxx"

#endif // QPL_DEFLATE_STREAM_HPP
