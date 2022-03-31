/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_INTERNAL_HPP
#define QPL_INTERNAL_HPP

#include "qpl/cpp_api/util/qpl_traits.hpp"
#include "qpl/cpp_api/chaining/composition.hpp"

namespace qpl {

template <class...>
class operation_chain;

namespace internal {
/**
 * @addtogroup HL_PRIVATE
 * @{
 */

/**
 * @brief Performs execution of the chain by going along the operations and combining/executing them
 *
 * @tparam  path                execution path
 * @tparam  allocator_t         type of the allocator to be used
 * @tparam  index               index of currently executable operation
 * @tparam  input_container_t   type of input container
 * @tparam  output_container_t  type of output container
 * @tparam  operations_t        list of types of operations in the chain
 *
 * @param   chain               instance of @ref operation_chain
 * @param   source              container for the source
 * @param   source_size         size of the source
 * @param   destination         container for the destination
 * @param   job_buffer          pointer to the buffer that should be used as job
 * @param   numa_id             Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements in output buffer
 */
template <execution_path path,
        template <class> class allocator_t,
                         uint32_t index = 0,
                         class input_container_t,
                         class output_container_t,
                         class... operations_t>
auto execute(operation_chain<operations_t...> &chain,
             input_container_t &source,
             size_t source_size,
             output_container_t &destination,
             int32_t numa_id,
             uint8_t *job_buffer) -> execution_result <uint32_t, sync> {
    std::fill(destination.begin(), destination.end(), 0);

    if constexpr (sizeof...(operations_t) > (index + 1)) {
        if constexpr (traits::should_be_merged<typename std::tuple_element<index, decltype(chain.operations_)>::type,
                                               typename std::tuple_element<index + 1,
                                                                           decltype(chain.operations_)>::type>()) {
            auto result = qpl::compose<path, allocator_t>(std::get<index>(chain.operations_),
                                                          std::get<index + 1>(chain.operations_),
                                                          source.data(),
                                                          source.data() + source_size,
                                                          destination.begin(),
                                                          destination.end(),
                                                          numa_id,
                                                          job_buffer);

            if constexpr (sizeof...(operations_t) == index + 2) {
                return result;
            } else {
                auto return_value = result;
                result.if_present([&](uint32_t value) -> void {
                    auto next_operation = std::get<index + 2>(chain.operations_);
                    auto size           = value;

                    if constexpr (traits::need_number_of_elements<typename std::tuple_element<
                            index + 2,
                            decltype(chain.operations_)>::type>()) {
                        std::get<index + 2>(chain.operations_) =
                                typename decltype(next_operation)::builder(next_operation)
                                        .number_of_input_elements(value)
                                        .build();

                        size = (value * next_operation.get_input_vector_width() + 7) >> 3;
                    }

                    return_value = internal::execute<path, allocator_t, index + 2>(chain,
                                                                                   destination,
                                                                                   size,
                                                                                   source,
                                                                                   numa_id,
                                                                                   job_buffer);
                });

                return return_value;
            }
        } else {
            if constexpr (std::is_same<qpl::merge,
                                       typename std::tuple_element<index + 1,
                                                                   decltype(chain.operations_)>::type>::value) {
                auto result = compose<path, allocator_t>(
                        std::get<index>(chain.operations_),
                        std::get<index + 2>(chain.operations_),
                        source.data(),
                        source.data() + source_size,
                        destination.begin(),
                        destination.end(),
                        numa_id,
                        job_buffer,
                        std::get<index + 1>(chain.operations_).get_number_of_elements());

                if constexpr (sizeof...(operations_t) - 1 == index + 2) {
                    return result;
                } else {
                    auto return_value = result;

                    result.if_present([&](uint32_t value) -> void {
                        auto next_operation = std::get<index + 3>(chain.operations_);
                        auto size           = value;

                        if constexpr (traits::need_number_of_elements<
                                typename std::tuple_element<index + 3,
                                                            decltype(chain.operations_)>::type>()) {
                            std::get<index + 3>(chain.operations_) =
                                    typename decltype(next_operation)::builder(next_operation)
                                            .number_of_input_elements(value)
                                            .build();

                            size = (value * next_operation.get_input_vector_width() + 7) >> 3;
                        }

                        return_value = internal::execute<path, allocator_t, index + 3>(chain,
                                                                                       destination,
                                                                                       size,
                                                                                       source,
                                                                                       numa_id,
                                                                                       job_buffer);
                    });

                    return return_value;
                }
            } else {
                auto result = internal::execute<path, allocator_t>(std::get<index>(chain.operations_),
                                                                   source.data(),
                                                                   source.data() + source_size,
                                                                   destination.begin(),
                                                                   destination.end(),
                                                                   numa_id,
                                                                   job_buffer);

                if constexpr (sizeof...(operations_t) == (index + 1)) {
                    return result;
                } else {
                    auto return_value = result;

                    result.if_present([&](uint32_t value) -> void {
                        auto next_operation = std::get<index + 1>(chain.operations_);
                        auto size           = value;

                        if constexpr (traits::need_number_of_elements<
                                typename std::tuple_element<index + 1,
                                                            decltype(chain.operations_)>::type>()) {
                            std::get<index + 1>(chain.operations_) =
                                    typename decltype(next_operation)::builder(next_operation)
                                            .number_of_input_elements(value)
                                            .build();

                            size = (value * next_operation.get_input_vector_width() + 7) >> 3;
                        }

                        return_value = internal::execute<path, allocator_t, index + 1>(chain,
                                                                                       destination,
                                                                                       size,
                                                                                       source,
                                                                                       numa_id,
                                                                                       job_buffer);
                    });

                    return return_value;
                }
            }
        }
    } else {
        return internal::execute<path, allocator_t>(std::get<index>(chain.operations_),
                                                    source.data(),
                                                    source.data() + source_size,
                                                    destination.begin(),
                                                    destination.end(),
                                                    numa_id,
                                                    job_buffer);
    }
}

/**
 * @brief Configures two analytic operations to make their input and output bit-widths compatible
 *
 * @tparam  operation1_t                 type of the first analytic operation
 * @tparam  operation2_t                 type of the second analytic operation
 *
 * @param   first_analytic_operation     instance of the first analytic operation
 * @param   second_analytic_operation    instance of the second analytic operation
 */
template <class operation1_t, class operation2_t>
void connect_two(const operation1_t &first_analytic_operation,
                 operation2_t &second_analytic_operation) {
    if constexpr (traits::are_analytics_operations<operation1_t, operation2_t>()) {
        if constexpr (!traits::should_be_merged<operation1_t, operation2_t>()) {
            second_analytic_operation = typename operation2_t::builder(second_analytic_operation)
                    .input_vector_width(dynamic_cast<const analytics_operation *>(&first_analytic_operation)
                                                ->get_output_vector_width())
                    .build();
        }
        if constexpr (traits::should_be_merged<operation1_t, operation2_t>()) {
            second_analytic_operation = typename operation2_t::builder(second_analytic_operation)
                    .input_vector_width(first_analytic_operation.get_input_vector_width())
                    .build();
        }
    }
}

/**
 * @brief Walks along the @ref operation_chain and prepares included operations
 *        by making them compatible
 *
 * @tparam  index         index of current operation
 * @tparam  operations_t  list of types of operations that are included into the chain
 *
 * @param   chain         instance of @ref operation_chain
 */
template <uint32_t index = 1, class... operations_t>
void prepare_operations(operation_chain<operations_t...> &chain) {
    auto &previous_operation = std::get<index - 1>(chain.operations_);
    auto &operation          = std::get<index>(chain.operations_);

    internal::connect_two(previous_operation, operation);

    if constexpr (sizeof...(operations_t) != (index + 1)) {
        internal::prepare_operations<index + 1>(chain);
    }
}

/** @} */

} // namespace internal
} // namespace qpl

#endif // QPL_INTERNAL_HPP
