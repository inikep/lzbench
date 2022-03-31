/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_MERGE_HPP
#define QPL_MERGE_HPP

#include "qpl/cpp_api/operations/operation.hpp"
#include "qpl/cpp_api/util/qpl_internal.hpp"
#include "operation_chain.hpp"

namespace qpl {

/**
 * @addtogroup HIGH_LEVEL_API
 * @{
 */

/**
 * @brief Control-flow entity that allows to merge analytic and decompression into one operation
 *
 * Example of main usage:
 * @snippet high-level-api/operation-chains/decompression_merged_with_analytics_example.cpp QPL_HIGH_LEVEL_MERGE_EXAMPLE
 */
class merge {
    template <class...>
    friend
    class operation_chain;

    template <execution_path path,
            template <class> class allocator_t,
            uint32_t index,
                             class input_container_t,
                             class output_container_t,
                             class... operations_t>
    friend auto internal::execute(operation_chain<operations_t...> &chain,
                                  input_container_t &source,
                                  size_t source_size,
                                  output_container_t &destination,
                                  int32_t numa_id,
                                  uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

public:
    /**
     * @brief Simple constructor that accepts number of elements that should be processed
     *        by analytic part
     *
     * @param  number_of_elements  number of elements that should be processed by analytic part
     *                             (or number of elements that were compressed)
     */
    constexpr explicit merge(uint32_t number_of_elements)
            : number_of_elements_(number_of_elements) {
        // Empty constructor
    }

    /**
     * @brief Simple constructor that is required by @ref operation_chain
     */
    constexpr merge() = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const merge &other) -> merge & = default;

    /**
     * @brief Move constructor
     *
     * @param  other  object that should be moved
     */
    constexpr merge(merge &&other) = default;

    /**
     * @brief Copy constructor
     *
     * @param  other  object that should be copied
     */
    constexpr merge(const merge &other) = default;

private:
    /**
     * @brief Simple getter of elements number
     */
    [[nodiscard]] auto get_number_of_elements() const -> uint32_t {
        return number_of_elements_;
    }

    /**
     * Number of elements that should be processed by the analytic part
     */
    uint32_t number_of_elements_ = 0;
};

/** @} */

} // namespace qpl

#endif // QPL_MERGE_HPP
