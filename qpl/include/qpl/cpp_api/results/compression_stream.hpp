/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_COMPRESSION_STREAM_HPP
#define QPL_COMPRESSION_STREAM_HPP

#include <functional>

#include "qpl/cpp_api/operations/compression_operation.hpp"
#include "qpl/cpp_api/util/qpl_allocation_util.hpp"

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

/**
 * @brief Contains possible states of compression_stream
 */
enum compression_stream_state {
    initial,    /**< Initial state where header should be written first */
    basic       /**< Body and trailer state */
};

/**
 * @brief Entity for performing compression and decompression by chunks
 *
 * @tparam  path         execution path that should be used
 * @tparam  allocator_t  type of the allocator to be used
 */
template <execution_path path>
class compression_stream {
    /**
     * @brief Type of deleter for internal job buffer
     */
    using deleter_t = std::function<void(uint8_t *)>;

public:
    /**
     * @brief Deleted copy constructor
     */
    compression_stream(const compression_stream &other) = delete;

    /**
     * @brief Deleted assignment operator
     */
    auto operator=(const compression_stream &other) -> compression_stream & = delete;

    /**
     * @brief Default move constructor
     */
    compression_stream(compression_stream &&other) noexcept = default;

    /**
     * @brief Method that returns an iterator pointing to the first element in the buffer
     */
    auto begin() noexcept -> uint8_t * {
        return destination_buffer_.get();
    }

    /**
     * @brief Method that returns an iterator pointing to the past-the-end element in the buffer
     */
    auto end() noexcept -> uint8_t * {
        return buffer_end_;
    }

    /**
     * @brief Method that returns the number of elements in the buffer
     */
    auto size() noexcept -> size_t {
        return buffer_current_ - destination_buffer_.get();
    }

    /**
     * @brief Default constructor
     */
    template <class compression_operation_t, template <class> class allocator_t = std::allocator>
    explicit compression_stream(compression_operation_t operation, size_t buffer_size,
                                allocator_t<uint8_t> allocator = allocator_t<uint8_t>()) {
        static_assert(std::is_base_of<compression_operation, compression_operation_t>::value,
                      "Wrong type for operation");

        job_buffer_         = util::allocate_array<allocator_t, uint8_t>(qpl::util::get_job_size());
        destination_buffer_ = util::allocate_array<allocator_t,
                                                   uint8_t>(static_cast<const uint32_t>(buffer_size));
        buffer_current_     = destination_buffer_.get();
        buffer_end_         = destination_buffer_.get() + buffer_size;
    }

protected:
    /**
     * Current stream state
     */
    compression_stream_state state_ = initial;

    /**
     * Pointer to the last processed element
     */
    uint8_t *buffer_current_ = nullptr;

    /**
     * Pointer to the end of the source
     */
    uint8_t *buffer_end_ = nullptr;

    /**
     * Buffer that contains processed elements
     */
    std::unique_ptr<uint8_t[], deleter_t> destination_buffer_;

    /**
     * Buffer that contains allocated job buffer
     */
    std::unique_ptr<uint8_t[], deleter_t> job_buffer_;
};

/** @} */

} // namespace qpl

#endif // QPL_COMPRESSION_STREAM_HPP
