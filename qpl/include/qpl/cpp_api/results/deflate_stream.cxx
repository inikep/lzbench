/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

namespace qpl {

template <execution_path path>
template <class input_iterator_t>
auto deflate_stream<path>::push(const input_iterator_t &source_begin,
                                const input_iterator_t &source_end) -> deflate_stream & {
    if (this->state_ == compression_stream_state::initial) {
        operation_.first_chunk(true);
        operation_.last_chunk(false);
        this->state_ = compression_stream_state::basic;
    } else {
        operation_.first_chunk(false);
        operation_.last_chunk(false);
    }

    if (this->buffer_current_ >= this->buffer_end_) {
        throw short_destination_exception(messages::short_destination);
    }

    submit_operation(source_begin, source_end);

    return *this;
}

template <execution_path path>
void deflate_stream<path>::resize(const size_t new_size) noexcept {
    size_t current_buffer_size = std::distance(this->destination_buffer_.get(), this->buffer_end_);

    if (new_size == current_buffer_size) {
        return;
    }

    // TODO use allocator as template parameter of this method
    auto temp_buffer = util::allocate_array<std::allocator,
                                            uint8_t>(static_cast<const uint32_t>(new_size));

    if (new_size < current_buffer_size) {
        std::copy(this->destination_buffer_.get(),
                  this->destination_buffer_.get() + new_size,
                  temp_buffer.get());
        current_buffer_size = new_size;
    } else {
        std::copy(this->destination_buffer_.get(), this->buffer_current_, temp_buffer.get());
    }

    this->destination_buffer_.reset(nullptr);

    this->destination_buffer_ = std::move(temp_buffer);
    this->buffer_current_     = this->destination_buffer_.get() + current_buffer_size;
    this->buffer_end_         = this->destination_buffer_.get() + new_size;
}

template <execution_path path>
template <class input_iterator_t>
void deflate_stream<path>::flush(const input_iterator_t &source_begin,
                                 const input_iterator_t &source_end) {
    if (this->state_ == compression_stream_state::initial) {
        operation_.first_chunk(true);
        operation_.last_chunk(true);
        this->state_ = compression_stream_state::basic;
    } else {
        operation_.first_chunk(false);
        operation_.last_chunk(true);
    }

    submit_operation(source_begin, source_end);

    this->state_ = initial;
}

template <execution_path path>
template <class input_iterator_t>
void deflate_stream<path>::submit_operation(const input_iterator_t &source_begin,
                                            const input_iterator_t &source_end) {
    operation_.set_proper_flags();

    // TODO remove usage of allocator as it's done for inflate
    auto result = internal::execute<path, std::allocator>(operation_,
                                                          source_begin,
                                                          source_end,
                                                          this->buffer_current_,
                                                          this->buffer_end_,
                                                          numa_auto_detect,
                                                          this->job_buffer_.get());

    auto shift = operation_.get_processed_bytes();
    this->buffer_current_ = this->destination_buffer_.get() + shift;

    result.if_absent([](uint32_t status) -> void {
        util::handle_status(status);
    });
}

} // namespace qpl
