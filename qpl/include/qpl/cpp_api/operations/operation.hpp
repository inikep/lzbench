/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_OPERATION_HPP
#define QPL_OPERATION_HPP

#include <tuple>
#include <future>

#include "qpl/cpp_api/common/definitions.hpp"
#include "qpl/cpp_api/operations/operation_builder.hpp"
#include "qpl/cpp_api/results/execution_result.hpp"
#include "qpl/cpp_api/results/stream.hpp"
#include "qpl/cpp_api/util/constants.hpp"
#include "qpl/cpp_api/util/qpl_traits.hpp"
#include "qpl/cpp_api/util/qpl_service_functions_wrapper.hpp"
#include "qpl/cpp_api/util/compression_utils.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @addtogroup HIGH_LEVEL_API
 * @{
 */

/**
 * @defgroup HL_PUBLIC Public API
 * @ingroup HIGH_LEVEL_API
 * @{
 * @brief Contains public high-level API entities and functions
 */

/**
 * @defgroup HL_OTHER Other API
 * @ingroup HL_PUBLIC
 * @brief Other entities and functions of high-level API
 */

/**
 * @brief Contains scan supported comparators
 */
enum comparators {
    less,         /**< Represents < */
    greater,      /**< Represents > */
    equals,       /**< Represents == */
    not_equals    /**< Represents != */
};

constexpr const int32_t numa_auto_detect = -1; /**< Numa ID defined by default */

class extract_operation;

class scan_operation;

class scan_range_operation;

class find_unique_operation;

class set_membership_operation;

class expand_operation;

class select_operation;

class rle_burst_operation;

class copy_operation;

class crc_operation;

class zero_compress_operation;

class zero_decompress_operation;

class inflate_operation;

class deflate_operation;

namespace internal {

class inflate_stateful_operation;

template <class operation1_t, class operation2_t>
void connect_two(const operation1_t &a, operation2_t &b);

template <execution_path path>
auto execute(extract_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(scan_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(scan_range_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(find_unique_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(set_membership_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(expand_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(select_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(rle_burst_operation &operation,
             uint8_t *source_buffer_ptr,
             size_t source_buffer_size,
             uint8_t *dest_buffer_ptr,
             size_t dest_buffer_size,
             uint8_t *mask_buffer_ptr,
             size_t mask_buffer_size,
             int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(copy_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(crc_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(zero_compress_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(zero_decompress_operation &operation, int32_t numa_id) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(inflate_operation &operation,
             uint8_t *buffer_ptr,
             size_t buffer_size) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(inflate_operation &operation) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(deflate_operation &operation,
             uint8_t *buffer_ptr,
             size_t buffer_size) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(deflate_operation &operation) -> execution_result<uint32_t, sync>;

template <execution_path path>
auto execute(inflate_stateful_operation &operation,
             uint8_t *source_begin,
             uint8_t *source_end,
             uint8_t *dest_begin,
             uint8_t *dest_end) -> execution_result<uint32_t, sync>;

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
             int32_t numa_id,
             uint8_t *job_buffer = nullptr) -> execution_result<uint32_t, sync>;

} // namespace internal

template <execution_path path = execution_path::software,
        template <class> class allocator_t = std::allocator,
                         class mask_builder_t,
                         class mask_applier_t,
                         class input_iterator_t,
                         class output_iterator_t>
auto compose(mask_builder_t &mask_builder,
             mask_applier_t &mask_applier,
             const input_iterator_t &source_begin,
             const input_iterator_t &source_end,
             const output_iterator_t &destination_begin,
             const output_iterator_t &destination_end,
             int32_t numa_id,
             uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

class compression_stateful_operation;

/**
 * @brief Interface for every operation that can be executed by library
 */
class operation {
public:
    constexpr operation() = default;

    /**
     * @brief Method to set input and output buffers
     *
     * @tparam  input_iterator_t   type of input iterator (random access + uint8_t value type)
     * @tparam  output_iterator_t  type of output iterator (random access + uint8_t value type)
     *
     * @param   source_begin       iterator that points to begin of the source
     * @param   source_end         iterator that points to end of the source
     * @param   destination_begin  iterator that points to begin of the output
     * @param   destination_end    iterator that points to begin of the output
     */
    template <class input_iterator_t, class output_iterator_t>
    void set_buffers(input_iterator_t source_begin,
                     input_iterator_t source_end,
                     output_iterator_t destination_begin,
                     output_iterator_t destination_end) noexcept {
        // Check if we've got valid iterators
        static_assert(std::is_same<typename std::iterator_traits<input_iterator_t>::iterator_category,
                                   std::random_access_iterator_tag>::value,
                      "Passed input iterator doesn't support random access");
        static_assert(std::is_same<typename std::iterator_traits<input_iterator_t>::value_type, uint8_t>::value,
                      "Passed input iterator value type should be uint8_t");

        static_assert(std::is_same<typename std::iterator_traits<output_iterator_t>::iterator_category,
                                   std::random_access_iterator_tag>::value,
                      "Passed output iterator doesn't support random access");
        static_assert(std::is_same<typename std::iterator_traits<output_iterator_t>::value_type, uint8_t>::value,
                      "Passed output iterator value type should be uint8_t");

        // Assign the buffers
        source_      = &*source_begin;
        source_size_ = source_end - source_begin;

        destination_      = const_cast<uint8_t *>(&*destination_begin);
        destination_size_ = destination_end - destination_begin;
    }

    /**
     * @brief Friend function that executes the operation
     */
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

    /**
     * @brief Method to set input and output buffers with usage of containers
     *
     * This method is just a wrapper for the previous one
     *
     * @tparam  input_container_t   type of input container
     * @tparam  output_container_t  type of output container
     *
     * @param   input_container     container that owns memory that should be processed
     * @param   output_container    container that owns memory where result should be stored
     */
    template <class input_container_t, class output_container_t>
    void set_buffers(const input_container_t &input_container,
                     const output_container_t &output_container) noexcept {
        auto source_begin      = input_container.begin();
        auto source_end        = input_container.end();
        auto destination_begin = output_container.begin();
        auto destination_end   = output_container.end();

        operation::set_buffers(source_begin, source_end, destination_begin, destination_end);
    }

protected:
    const uint8_t *source_          = nullptr;    /**< Pointer to source */
    size_t        source_size_      = 0;          /**< Size of the source */
    uint8_t       *destination_     = nullptr;    /**< Pointer to output */
    size_t        destination_size_ = 0;          /**< Size of the output */

    /**
     * @brief Method to set memory for Intel QPL job (will be removed after ML introduction)
     *
     * @param  buffer  pointer to memory that should be used for Intel QPL job
     */
    virtual void set_job_buffer(uint8_t *buffer) noexcept = 0;
};

/**
 * @brief Base class for custom operations that should be implemented at the end-user side
 */
class custom_operation : public operation {
public:
    /**
     * @brief Virtual method that should be implemented in order to make custom operation
     *
     * @param  source              pointer to the source
     * @param  source_size         size of the source
     * @param  destination         pointer to the destination
     * @param  destination_size    size of the destination
     *
     * @return number of valid bytes in destination
     */
    [[nodiscard]] virtual uint32_t execute(const uint8_t *source,
                                           uint32_t source_size,
                                           uint8_t *destination,
                                           uint32_t destination_size) = 0;

private:
    /**
     * @brief Stub that will be removed later
     */
    void set_job_buffer(uint8_t *buffer) noexcept final {
        // No job buffer is required with custom operations
    }
};

namespace internal {
/**
 * @addtogroup HL_PRIVATE
 * @{
 */

/**
 * @brief This is an implementation of public wrapper (due to default allocator_t value problem)
 *
 * @tparam  path               execution path
 * @tparam  allocator_t        type of the allocator to be used for any internal allocation
 * @tparam  input_iterator_t   type of input iterator (same requirements as for @link operation @endlink)
 * @tparam  output_iterator_t  type of output iterator (same requirements as for @link operation @endlink)
 * @tparam  operation_t        type of operation that should be executed
 *
 * @param   op                 instance of operation that should be executed
 * @param   source_begin       iterator that points to begin of the source
 * @param   source_end         iterator that points to end of the source
 * @param   destination_begin  iterator that points to begin of the output
 * @param   destination_end    iterator that points to end of the output
 * @param   job_buffer         optional pointer to memory that should be used for Intel QPL job
 * @param   numa_id            Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t,
                         class output_iterator_t,
                         class operation_t>
auto execute(operation_t &op,
             const input_iterator_t source_begin,
             const input_iterator_t source_end,
             const output_iterator_t destination_begin,
             const output_iterator_t destination_end,
             int32_t numa_id,
             uint8_t *job_buffer) -> execution_result<uint32_t, sync> {
    if constexpr (traits::is_any<operation_t,
                                 extract_operation,
                                 scan_operation,
                                 scan_range_operation,
                                 find_unique_operation,
                                 set_membership_operation,
                                 expand_operation,
                                 select_operation,
                                 rle_burst_operation,
                                 copy_operation,
                                 crc_operation,
                                 zero_compress_operation,
                                 zero_decompress_operation,
                                 inflate_operation,
                                 deflate_operation>) {
        auto operation_helper = dynamic_cast<operation *>(&op);
        operation_helper->set_buffers(source_begin, source_end, destination_begin, destination_end);

        if constexpr (traits::is_any<operation_t,
                                     inflate_operation,
                                     deflate_operation>) {
            if constexpr (path == hardware) {
                return internal::execute<path>(op);
            }

            auto buf_size = util::get_buffer_size<operation_t>();

            stream<allocator_t> temp_buffer(buf_size);

            return internal::execute<path>(op, temp_buffer.data(), temp_buffer.size());
        } else {
            if constexpr (traits::is_any<operation_t, rle_burst_operation>) {
                // Maximum number of unpacked elements
                constexpr auto max_elements = 4096u;
                // Unpack buffer size, +1u - especially for RLE_Burst
                constexpr auto buf_size     = (max_elements + 1u) * sizeof(uint32_t);

                stream<allocator_t> source_buffer(buf_size);
                if constexpr (path != execution_path::hardware) {
                    std::fill(source_buffer.begin(), source_buffer.end(), 0);
                }

                stream<allocator_t> unpack_buffer(buf_size + 100);
                stream<allocator_t> mask_buffer(buf_size);
                stream<allocator_t> output_buffer(buf_size);

                return internal::execute<path>(op,
                                               unpack_buffer.data(),
                                               source_buffer.size(),
                                               output_buffer.data(),
                                               output_buffer.size(),
                                               mask_buffer.data(),
                                               mask_buffer.size(),
                                               numa_id);
            } else {
                return internal::execute<path>(op, numa_id);
            }
        }
    } else if constexpr (std::is_base_of<custom_operation, operation_t>::value) {
        auto operation_helper = dynamic_cast<custom_operation *>(&op);

        operation_helper->set_buffers(source_begin, source_end, destination_begin, destination_end);

        return execution_result<uint32_t, sync>(
                0,
                operation_helper->execute(operation_helper->source_,
                                          static_cast<uint32_t>(operation_helper->source_size_),
                                          operation_helper->destination_,
                                          static_cast<uint32_t>(operation_helper->destination_size_)));
    } else if constexpr (std::is_base_of<compression_stateful_operation, operation_t>::value) {
        auto operation_helper = dynamic_cast<operation *>(&op);
        operation_helper->set_buffers(source_begin, source_end, destination_begin, destination_end);

        op.set_buffers();
        auto result = op.execute();

        return execution_result<uint32_t, sync>(result.first, result.second);
    } else {
        auto operation_helper = dynamic_cast<operation *>(&op);
        operation_helper->set_buffers(source_begin, source_end, destination_begin, destination_end);

        if (job_buffer) {
            operation_helper->set_job_buffer(job_buffer);
            auto status = op.init_job(path);

            if (status) {
                return execution_result<uint32_t, sync>(status, 0);
            }

            auto result = op.execute();

            return execution_result<uint32_t, sync>(result.first, result.second);
        } else {
            allocator_t<uint8_t> allocator;
            job_buffer = allocator.allocate(qpl::util::get_job_size());

            operation_helper->set_job_buffer(job_buffer);
            auto status = op.init_job(path);

            if (status) {
                return execution_result<uint32_t, sync>(status, 0);
            }

            auto result = op.execute();

            allocator.deallocate(job_buffer, qpl::util::get_job_size());

            return execution_result<uint32_t, sync>(result.first, result.second);
        }
    }
}

/** @} */

} // namespace internal

/**
 * @brief Common method of any Intel QPL operation execution
 *
 * @tparam  path               execution path
 * @tparam  allocator_t        type of the allocator to be used for any internal allocation
 * @tparam  input_iterator_t   type of input iterator (same requirements as for @link operation @endlink)
 * @tparam  output_iterator_t  type of output iterator (same requirements as for @link operation @endlink)
 * @tparam  operation_t        type of operation that should be executed
 *
 * @param   operation          instance of operation that should be executed
 * @param   source_begin       iterator that points to begin of the source
 * @param   source_end         iterator that points to end of the source
 * @param   destination_begin  iterator that points to begin of the output
 * @param   destination_end    iterator that points to end of the output
 * @param   numa_id            Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path = execution_path::software,
        template <class> class allocator_t = std::allocator,
                         class input_iterator_t,
                         class output_iterator_t,
                         class operation_t>
auto execute(operation_t &operation,
             const input_iterator_t source_begin,
             const input_iterator_t source_end,
             const output_iterator_t destination_begin,
             const output_iterator_t destination_end,
             int32_t numa_id = numa_auto_detect) -> execution_result<uint32_t, sync> {
    return internal::execute<path, allocator_t>(operation,
                                                source_begin,
                                                source_end,
                                                destination_begin,
                                                destination_end,
                                                numa_id);
}

/**
 * @brief Just a wrapper for the previous function with usage of containers
 *
 * @tparam  path                execution path
 * @tparam  allocator_t         type of the allocator to be used for any internal allocation
 * @tparam  input_container_t   type of input container
 * @tparam  output_container_t  type of output container
 * @tparam  operation_t         type of operation that should be executed
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path = execution_path::software,
        template <class> class allocator_t = std::allocator,
                         class input_container_t,
                         class output_container_t,
                         class operation_t>
auto execute(operation_t &operation,
             const input_container_t &input,
             const output_container_t &output) -> execution_result<uint32_t, sync> {
    return qpl::execute<path, allocator_t>(operation,
                                           input.begin(),
                                           input.end(),
                                           output.begin(),
                                           output.end());
}

/**
 * @brief Performs asynchronous execution of simple operation
 *
 * @tparam  path               execution path
 * @tparam  allocator_t        type of the allocator to be used
 * @tparam  input_iterator_t   type of input iterator
 * @tparam  output_iterator_t  type of output iterator
 * @tparam  operation_t        type of executable operation
 *
 * @param   operation          instance of executable operation
 * @param   source_begin       pointer to the beginning of the source
 * @param   source_end         pointer to the end of the source
 * @param   destination_begin  pointer to the beginning of the destination
 * @param   destination_end    pointer to the end of the destination
 * @param   numa_id            Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path = qpl::software,
        template <class> class allocator_t = std::allocator,
                         class input_iterator_t,
                         class output_iterator_t,
                         class operation_t>
auto submit(operation_t &operation,
            const input_iterator_t source_begin,
            const input_iterator_t source_end,
            const output_iterator_t destination_begin,
            const output_iterator_t destination_end,
            int32_t numa_id = auto_detect) -> execution_result<uint32_t, sync> {
    return internal::execute<path, allocator_t>(operation,
                                                source_begin,
                                                source_end,
                                                destination_begin,
                                                destination_end,
                                                numa_id);
}

/**
 * @brief Performs asynchronous execution of simple operation
 *
 * @tparam  path                execution path
 * @tparam  allocator_t         type of the allocator to be used
 * @tparam  input_container_t   type of input container
 * @tparam  output_container_t  type of output container
 * @tparam  operation_t         type of executable operation
 *
 * @param   operation           instance of executable operation
 * @param   input               container for source
 * @param   output              container for the output
 * @param   numa_id             Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path = qpl::software,
        template <class> class allocator_t = std::allocator,
                         class input_container_t,
                         class output_container_t,
                         class operation_t>
auto submit(operation_t &operation,
            const input_container_t &input,
            const output_container_t &output,
            int32_t numa_id = auto_detect) -> execution_result<uint32_t, sync> {
    return qpl::submit<path, allocator_t>(operation,
                                          input.begin(),
                                          input.end(),
                                          output.begin(),
                                          output.end(),
                                          numa_id);
}

/**
 * @brief Returns a string with version of the library
 */
auto get_library_version() -> const char *;

/** @} End HL_PUBLIC_API group */

/** @} End HIGH_LEVEL_API group */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_OPERATION_HPP
