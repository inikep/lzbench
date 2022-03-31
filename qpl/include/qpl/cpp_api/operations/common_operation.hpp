/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_COMMON_OPERATION_HPP
#define QPL_COMMON_OPERATION_HPP

#include "qpl/cpp_api/util/qpl_allocation_util.hpp"
#include "qpl/cpp_api/results/execution_result.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @defgroup HL_PRIVATE Private API
 * @ingroup HIGH_LEVEL_API
 * @{
 * @brief Private entities and functions
 */
enum execution_path : uint32_t;

namespace internal {

template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t,
                         class output_iterator_t,
                         class operation_t>
auto execute(operation_t &operation,
             input_iterator_t source_begin,
             input_iterator_t source_end,
             output_iterator_t destination_begin,
             output_iterator_t destination_end,
             uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

} // namespace internal

/**
 * @brief Interface that contains common parts of any Intel QPL operation
 *
 * This class should be considered as common work with Intel QPL job structure
 */
class common_operation {
    template <execution_path path>
    friend
    class deflate_stream;

    template <execution_path path>
    friend
    class inflate_stream;

    template <template <class> class allocator_t>
    friend
    class deflate_block;

    template <execution_path path,
            template <class> class allocator_t,
                             class input_iterator_t,
                             class output_iterator_t,
                             class operation_t>
    friend auto internal::execute(operation_t &operation,
                                  input_iterator_t source_begin,
                                  input_iterator_t source_end,
                                  output_iterator_t destination_begin,
                                  output_iterator_t destination_end,
                                  int32_t numa_id,
                                  uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

public:
    /**
     * @brief Default simple constructor
     */
    constexpr common_operation() = default;

protected:
    /**
     * @brief Sets memory that should be used for Intel QPL job
     *
     * @param  buffer  pointer to memory for Intel QPL job
     */
    void reset_job_buffer(uint8_t *buffer) noexcept;

    /**
     * @brief Wrapper for common job initialization
     */
    virtual auto init_job(const execution_path path) noexcept -> uint32_t;

    /**
     * @brief Virtual method that is used by @link init_job() @endlink to configure Intel QPL job
     */
    virtual void set_proper_flags() noexcept = 0;

    /**
     * @brief Virtual method that is used by @link init_job() @endlink to set proper buffers
     */
    virtual void set_buffers() noexcept = 0;

    /**
     * @brief Virtual method that is used by @link execute() @endlink to get output elements count
     *
     * @return number of elements in output
     */
    virtual auto get_output_elements_count() noexcept -> uint32_t;

    /**
    * @brief Method that performs an execution of the operation
    *
    * @return pair consisting of status code and number of produced elements in output buffer
    */
    virtual auto execute() -> std::pair<uint32_t, uint32_t>;

    uint8_t *buffer_ = nullptr; /**< Pointer to buffer that should be used to keep job structure */
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_COMMON_OPERATION_HPP
