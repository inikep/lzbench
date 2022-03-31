/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_COMPOSITION_HPP
#define QPL_COMPOSITION_HPP

#include "qpl/cpp_api/operations/operation.hpp"
#include "qpl/cpp_api/operations/compression_operation.hpp"
#include "qpl/cpp_api/operations/analytic_operation.hpp"
#include "qpl/cpp_api/util/qpl_traits.hpp"

namespace qpl {

/**
 * @addtogroup HIGH_LEVEL_API
 * @{
 */

/**
 * @brief Composes two analytic operations that may be combined as "mask builder"
 *        and "mask applier"
 *
 * @tparam  path               execution path that should be used
 * @tparam  allocator_t        type of the allocator to be used
 * @tparam  mask_builder_t     type of operation that builds the mask
 * @tparam  mask_applier_t     type of operation that applies the mask
 * @tparam  input_iterator_t   type of input iterator
 * @tparam  output_iterator_t  type of output iterator
 *
 * @param   mask_builder       instance of operation that builds the mask
 * @param   mask_applier       instance of operation that applies the mask
 * @param   source_begin       beginning of the source
 * @param   source_end         end of the source
 * @param   destination_begin  beginning of the destination
 * @param   destination_end    end of the destination
 * @param   job_buffer         pointer to buffer that should be used for job structure
 * @param   numa_id            Numa node identifier
 *
 * @return @ref execution_result consisting of status code and number of produced elements
 *         in output buffer
 */
template <execution_path path,
        template <class> class allocator_t,
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
             uint8_t *job_buffer) -> execution_result<uint32_t, sync> {
    static_assert(std::is_base_of<analytics_operation, mask_builder_t>::value,
                  "Wrong type for mask builder");
    static_assert(std::is_base_of<operation_with_mask, mask_applier_t>::value,
                  "Wrong type for mask applier");

    auto mask_builder_operation = dynamic_cast<analytics_operation *>(&mask_builder);

    stream<allocator_t> mask((mask_builder_operation->number_of_input_elements_ + 7u) >> 3u);

    mask_builder = typename mask_builder_t::builder(mask_builder)
            .output_vector_width(1)
            .build();
    mask_applier = typename mask_applier_t::builder(mask_applier)
            .input_vector_width(mask_builder.get_input_vector_width())
            .mask(mask.template as<uint8_t *>(), mask.size())
            .build();

    switch (mask_builder_operation->parser_) {
        case big_endian_packed_array: {
            mask_applier = typename mask_applier_t::builder(mask_applier)
                    .template parser<big_endian_packed_array>(
                            mask_builder_operation->number_of_input_elements_)
                    .build();
            break;
        }
        case parquet_rle: {
            mask_applier = typename mask_applier_t::builder(mask_applier)
                    .template parser<parquet_rle>(
                            mask_builder_operation->number_of_input_elements_)
                    .build();
            break;
        }
        default: {
            mask_applier = typename mask_applier_t::builder(mask_applier)
                    .template parser<little_endian_packed_array>(
                            mask_builder_operation->number_of_input_elements_)
                    .build();
            break;
        }
    }

    internal::execute<path, allocator_t>(mask_builder,
                                         source_begin,
                                         source_end,
                                         mask.begin(),
                                         mask.end(),
                                         numa_id,
                                         job_buffer);

    auto result = internal::execute<path, allocator_t>(mask_applier,
                                                       source_begin,
                                                       source_end,
                                                       destination_begin,
                                                       destination_end,
                                                       numa_id,
                                                       job_buffer);

    mask_applier.reset_mask(nullptr, 0);

    return result;
}

/**
 * @brief This function composes two operations that are merged with manipulator
 *
 * @tparam  allocator_t                      type of the allocator to be used
 * @tparam  input_iterator_t                 type of input iterator (same requirements as
 *                                           for @link operation @endlink)
 * @tparam  output_iterator_t                type of output iterator (same requirements as
 *                                           for @link operation @endlink)
 * @tparam  operation_t                      operation that should be executed with
 *                                           enabled decompression
 *
 * @param   decompress_operation             instance of compression_operation that is used
 *                                           to get gzip flag
 * @param   operation                        operation that should be actually executed
 * @param   source_begin                     iterator that points to begin of the source
 * @param   source_end                       iterator that points to end of the source
 * @param   destination_begin                iterator that points to begin of the output
 * @param   destination_end                  iterator that points to end of the output
 * @param   job_buffer                       pointer to memory that should be used for Intel QPL job
 * @param   number_of_decompressed_elements  number of elements to process after decompression
 * @param   numa_id                          Numa node identifier
 *
 * According to merge functionality, decompress_operation should actually be Inflate,
 * this condition is controlled level above:
 * there's check for it in traits, that are used in chain execution.
 * Second operation type should be analytics_operation.
 *
 * @return @ref execution_result consisting of status code and number of produced elements
 *         in output buffer
 `*/
template <execution_path path,
        template <class> class allocator_t,
                         class input_iterator_t,
                         class output_iterator_t,
                         class operation_t>
auto compose(compression_operation &decompress_operation,
             operation_t &operation,
             const input_iterator_t &source_begin,
             const input_iterator_t &source_end,
             const output_iterator_t &destination_begin,
             const output_iterator_t &destination_end,
             int32_t numa_id,
             uint8_t *job_buffer,
             uint32_t number_of_decompressed_elements) -> execution_result<uint32_t, sync> {
    static_assert(std::is_base_of<analytics_operation, operation_t>::value,
                  "Second operation should be analytics");

    auto operation_helper = dynamic_cast<analytics_operation *>(&operation);
    operation_helper->enable_decompression(number_of_decompressed_elements,
                                           decompress_operation.get_gzip_mode());

    return internal::execute<path, allocator_t>(operation,
                                                source_begin,
                                                source_end,
                                                destination_begin,
                                                destination_end,
                                                numa_id,
                                                job_buffer);
}

/** @} */

} // namespace qpl

#endif // QPL_COMPOSITION_HPP
