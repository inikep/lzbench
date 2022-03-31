/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_OPERATION_CHAIN_HPP
#define QPL_OPERATION_CHAIN_HPP

#include <tuple>
#include <cstring>

#include "qpl/cpp_api/operations/operation.hpp"
#include "qpl/cpp_api/operations/custom_operation.hpp"
#include "qpl/cpp_api/chaining/composition.hpp"
#include "qpl/cpp_api/results/stream.hpp"
#include "qpl/cpp_api/util/qpl_traits.hpp"
#include "qpl/cpp_api/util/qpl_internal.hpp"

namespace qpl {

/**
 * @addtogroup HIGH_LEVEL_API
 * @{
 */

/**
 * @brief Class that represents an operations pipeline
 *
 * @tparam  operations_t  list of types of operations that are included into one pipeline
 *
 * Example of main usage:
 * @snippet high-level-api/operation-chains/decompression_scan_select_example.cpp QPL_HIGH_LEVEL_CHAIN_EXAMPLE
 *
 * Example of set membership chaining:
 * @snippet high-level-api/operation-chains/set_membership_operation_chain_example.cpp QPL_HIGH_LEVEL_SET_MEMBERSHIP_CHAIN_EXAMPLE
 */
template <class... operations_t>
class operation_chain {
    template <class...>
    friend
    class operation_chain;

public:
    /**
     * @brief Simple constructor is deleted as there should be no possibility
     *        to create an empty chain
     */
    operation_chain() = delete;

    /**
     * @brief Constructor that accepts various number of operations
     *
     * @param  operations  list of operations that should be included into the chain
     */
    constexpr explicit operation_chain(operations_t... operations) {
        operations_ = std::make_tuple(operations...);
    }

    /**
     * @brief Constructor that extends an existing chain with additional operation
     *
     * @tparam  new_operation_t  type of new operation that should be added to the chain
     * @tparam  ops_t            types of already existing operations
     *
     * @param   chain            already existing operation chain
     * @param   operation        instance of operation that should be included in a new chain
     */
    template <class new_operation_t, class... ops_t>
    constexpr explicit operation_chain(operation_chain<ops_t...> chain,
                                       new_operation_t &&operation) {
        operations_ = std::tuple_cat(chain.operations_, std::make_tuple(operation));
    }

    /**
     * @brief Creates a new chain with newly specified operation
     *
     * @tparam  new_operation_t  type of new operation
     *
     * @param   new_operation    instance of new operation
     *
     * @return newly created chain object combined of the already existing one and new operation
     *
     * @note works only if new operation implements @ref operation interface
     */
    template <class new_operation_t,
              class = typename std::enable_if<traits::are_qpl_operations<new_operation_t>() ||
                                              std::is_same<merge, new_operation_t>::value>::type>
    constexpr auto operator|(new_operation_t new_operation)
    -> operation_chain<operations_t..., new_operation_t> {
        return operation_chain<operations_t..., new_operation_t>(*this, std::move(new_operation));
    }

    /**
     * @brief Creates a new chain with custom operation included
     *
     * @tparam  functor_t  type of custom operation
     *
     * @param   functor    instance of custom operation
     *
     * @return newly created chain object combined of the already existing one and custom operation
     *
     * @note custom operation should fit specific requirements
     */
    template <class functor_t,
              class = typename std::enable_if<custom_functor<functor_t>::is_correct_functor()>::type>
    constexpr auto operator|(functor_t functor)
    -> operation_chain<operations_t..., custom_functor<functor_t>> {
        return operation_chain<operations_t..., custom_functor<functor_t>>(*this,
                                                                           custom_functor(functor));
    }

    /**
     * @brief Friend function that executes the chain
     */
    template <execution_path path,
            template <class> class allocator_t,
                             uint32_t index,
                             class input_container_t,
                             class output_container_t,
                             class... ops_t>
    friend auto internal::execute(operation_chain<ops_t...> &chain,
                                  input_container_t &source,
                                  size_t source_size,
                                  output_container_t &destination,
                                  int32_t numa_id,
                                  uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

    /**
     * @brief Friend function that executes the chain
     */
    template <execution_path path,
            template <class> class allocator_t,
                             class input_container_t,
                             class output_container_t,
                             class... ops_t>
    friend auto execute(operation_chain<ops_t...> chain,
                        input_container_t &source,
                        output_container_t &destination,
                        int32_t numa_id) -> execution_result<uint32_t, sync>;

    /**
     * @brief Friend function that prepares the chain for execution
     */
    template <uint32_t index, class... ops_t>
    friend void internal::prepare_operations(operation_chain<ops_t...> &chain);

private:
    /**
     * List of operations that are included into the chain
     */
    std::tuple<operations_t...> operations_;
};

/**
 * @brief Performs an execution of the chain
 *
 * @note Returns internally allocated qpl::stream<> as result
 *
 * @tparam  allocator_t   type of the allocator to be used for internal allocations
 * @tparam  container_t   type of input container
 * @tparam  operations_t  list of operations that are included into pipeline (auto-deduced)
 *
 * @param   chain         instance of operation_chain<class...> that should be executed
 * @param   source        container that owns memory that should be processed with specified chain
 * @param   destination   container for destination
 * @param   numa_id       Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path = execution_path::software,
        template <class> class allocator_t = std::allocator,
                         class input_container_t,
                         class output_container_t,
                         class... operations_t>
auto execute(operation_chain<operations_t...> chain,
             input_container_t &source,
             output_container_t &destination,
             int32_t numa_id = numa_auto_detect) -> execution_result<uint32_t, sync> {
    static_assert(sizeof...(operations_t) > 1,
                  "Unable to execute chain with 1 operation, "
                  "use qpl::execute() for this operation");

    // Temporary buffers for processing
    const size_t buffer_size = std::max(destination.size(), source.size());

    stream<allocator_t> temporary_buffer1(buffer_size);
    stream<allocator_t> temporary_buffer2(buffer_size);

    // Job buffer for all operations
    stream<allocator_t> job_buffer(qpl::util::get_job_size());

    std::memcpy(temporary_buffer1.data(), source.data(), source.size());

    internal::prepare_operations(chain);
    auto result = internal::execute<path, allocator_t>(chain,
                                                       temporary_buffer1,
                                                       source.size(),
                                                       temporary_buffer2,
                                                       numa_id,
                                                       job_buffer.template as<uint8_t *>());

    result.if_present([&](uint32_t value) -> void {
        constexpr uint32_t number_of_operations = qpl::traits::actual_number_of_operations<operations_t...>();
        uint32_t           output_bit_width     = byte_bit_length;
        uint32_t           processed_bytes      = 0;

        if constexpr (traits::are_analytics_operations<typename std::tuple_element<sizeof...(operations_t) - 1,
                                                                                   decltype(chain.operations_)
                                                                                  >::type>()) {
            auto last_operation = std::get<sizeof...(operations_t) - 1>(chain.operations_);

            output_bit_width = dynamic_cast<const analytics_operation *>(&last_operation)->get_output_vector_width();
        }

        processed_bytes = (output_bit_width * value + 7u) >> 3u;

        if constexpr (number_of_operations % 2 == 0) {
            std::memcpy(destination.data(), temporary_buffer1.data(), processed_bytes);
        } else {
            std::memcpy(destination.data(), temporary_buffer2.data(), processed_bytes);
        }
    });

    return result;
}

/**
* @brief Performs an asyncronous execution of the chain
*
* @tparam  path                execution path that should be used
* @tparam  allocator_t         type of the allocator to be used
* @tparam  input_container_t   type of input container
* @tparam  output_container_t  type of output container
* @tparam  operations_t        list of operations types in the chain
*
* @param   chain               instance of the chain for execution
* @param   source              instance of source container
* @param   destination         instance of destination container
* @param   numa_id             Numa node identifier
*
* @return @ref execution_result consisting of status code and number of produced elements in output buffer
*/
template <execution_path path = qpl::software,
        template <class> class allocator_t = std::allocator,
                         class input_container_t,
                         class output_container_t,
                         class... operations_t>
auto submit(operation_chain<operations_t...> chain,
            input_container_t &source,
            output_container_t &destination,
            int32_t numa_id = numa_auto_detect) -> execution_result<uint32_t, sync> {
    return qpl::execute<path, allocator_t>(chain, source, destination, numa_id);
}

/**
 * @brief Overloaded operator | that describes how to unite two operations in chain
 *
 * @tparam  operation1_t  type of the first operation
 * @tparam  operation2_t  type of the second operation
 *
 * @param   operation1    first operation that should be united into chain
 * @param   operation2    second operation that should be united into chain
 *
 * @return instance of operation_chain<class...> that consists of two specified operations
 */
template <class operation1_t,
          class operation2_t,
          class = typename std::enable_if<traits::are_qpl_operations<operation1_t, operation2_t>()>::type>
constexpr auto operator|(operation1_t operation1, operation2_t operation2) -> operation_chain<operation1_t,
                                                                                              operation2_t> {
    return operation_chain<operation1_t, operation2_t>(std::move(operation1), std::move(operation2));
}

/**
 * @brief Overloaded operator | that describes how to unite operation
 *        and custom operation into the chain
 *
 * @tparam  operation_t  type of operation
 * @tparam  functor_t    type of custom operation
 *
 * @param   operation1   instance of operation
 * @param   operation2   instance of custom operation
 *
 * @return instance of @ref operation_chain that consists of two specified operations
 */
template <class operation_t,
          class functor_t,
          class = typename std::enable_if<traits::are_qpl_operations<operation_t>() &&
                                          custom_functor<functor_t>::is_correct_functor()>::type>
constexpr auto operator|(operation_t operation1, functor_t operation2) -> operation_chain<operation_t,
                                                                                          custom_functor<functor_t>> {
    return operation_chain<operation_t, custom_functor<functor_t>>(std::move(operation1), custom_functor(operation2));
}

/**
 * @brief Overloaded operator | that describes how to unite inflate operation
 *        and merge manipulator into chain
 *
 * @param  inflate_operation  inflate operation that should be united into chain
 * @param  merge_manipulator  merge manipulator that should be united into chain
 *
 * @return instance of operation_chain<class...> that consists of inflate operation and merge manipulator
 */
template <class operation_t,
          class = typename std::enable_if<std::is_same<inflate_operation,
                                                       operation_t>::value>::type>
constexpr auto operator|(operation_t inflate_operation, merge &&merge_manipulator) -> operation_chain<operation_t,
                                                                                                      merge> {
    return operation_chain<operation_t, merge>(std::move(inflate_operation), std::move(merge_manipulator));
}

/** @} */

} // namespace qpl

#endif // QPL_OPERATION_CHAIN_HPP
