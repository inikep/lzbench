/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_DEFLATE_HISTOGRAM_HPP
#define QPL_DEFLATE_HISTOGRAM_HPP

#include <array>

#include "qpl/cpp_api/util/constants.hpp"
#include "qpl/cpp_api/operations/operation.hpp"

namespace qpl {

/**
 * @addtogroup HL_COMPRESSION
 * @{
 */

/**
 * @brief Class, that represents deflate histogram
 */
class deflate_histogram {
public:
    /**
     * Size of literals and match lengths alphabet
     */
    static constexpr auto literals_lengths_histogram_size = 286u;

    /**
     * Size of offsets alphabet
     */
    static constexpr auto offsets_histogram_size = 30u;

    /**
     * @brief Default copy constructor
     */
    constexpr deflate_histogram(const deflate_histogram &other) = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const deflate_histogram &other) -> deflate_histogram & = default;

    /**
     * @brief Default move constructor
     */
    constexpr deflate_histogram(deflate_histogram &&other) = default;

    /**
     * @brief Default constructor
     */
    constexpr explicit deflate_histogram() = default;

    /**
     * @brief Returns pointer to literals/lengths histogram
     */
    auto get_literals_lengths() noexcept -> uint32_t *;

    /**
     * @brief Returns pointer to offsets histogram
     */
    auto get_offsets() noexcept -> uint32_t *;

private:
    /**
     * Contains statistics (number of entrances) for literals and match lengths
     */
    std::array<uint32_t, literals_lengths_histogram_size> literalsLengths_buffer_ = {};

    /**
     * Contains statistics (number of entrances) for offsets
     */
    std::array<uint32_t, offsets_histogram_size> offsets_buffer_ = {};
};

/** @} */

} // namespace qpl

#endif // QPL_DEFLATE_HISTOGRAM_HPP
