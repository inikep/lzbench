/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

namespace qpl {

template <template <class> class allocator_t>
deflate_block<allocator_t>::deflate_block(const mini_block_sizes mini_block_size,
                                          const uint32_t index_array_size,
                                          const execution_path path) {
    buffer_.mini_block_size = mini_block_size;
    buffer_.size            = util::convert_mini_block_size(buffer_.mini_block_size);
    index_array_size_ = index_array_size;

    auto job_buffer_size = qpl::util::get_job_size();

    buffer_.buffer = util::allocate_array<allocator_t, uint8_t>(buffer_.size);
    index_array_ = util::allocate_array<allocator_t, internal::index>(index_array_size_);
    job_buffer_  = util::allocate_array<allocator_t, uint8_t>(job_buffer_size);

    operation_.set_job_buffer(job_buffer_.get());
    operation_.init_job(path);
}

template <template <class> class allocator_t>
auto deflate_block<allocator_t>::compressed_size() const noexcept -> size_t {
    return static_cast<size_t>(source_size_);
}

template <template <class> class allocator_t>
auto deflate_block<allocator_t>::size() const noexcept -> size_t {
    return static_cast<size_t>(uncompressed_size_);
}

template <template <class> class allocator_t>
auto deflate_block<allocator_t>::operator[](const size_t index) -> uint8_t {
    auto mini_block_index = util::get_mini_block_index(static_cast<uint32_t>(index),
                                                       buffer_.mini_block_size);

    if (buffer_.is_empty || buffer_.stored_mini_block != mini_block_index) {
        util::read_mini_block(source_.get(),
                              buffer_.buffer.get(),
                              buffer_.size,
                              mini_block_index,
                              index_array_.get(),
                              operation_);

        buffer_.is_empty          = false;
        buffer_.stored_mini_block = mini_block_index;
    }

    return buffer_.buffer[static_cast<uint32_t>(index) % buffer_.size];
}

template <template <class> class allocator_t>
template <class input_iterator_t>
void deflate_block<allocator_t>::assign(input_iterator_t begin, input_iterator_t end) {
    source_size_ = static_cast<uint32_t>(std::distance(begin, end));
    source_      = util::allocate_array<allocator_t, uint8_t>(source_size_);

    std::memcpy(source_.get(), &*begin, source_size_);

    util::read_header(source_.get(),
                      buffer_.buffer.get(),
                      buffer_.size,
                      index_array_.get(),
                      operation_);
}

template <template <class> class allocator_t>
void deflate_block<allocator_t>::set_compressed_size(uint32_t size) {
    uncompressed_size_ = size;
}

} // namespace qpl
