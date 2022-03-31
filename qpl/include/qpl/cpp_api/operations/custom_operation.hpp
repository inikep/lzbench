/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_CUSTOM_OPERATION_HPP
#define QPL_CUSTOM_OPERATION_HPP

#include "qpl/cpp_api/operations/operation.hpp"

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

/**
 * @brief Helper class that processes custom functors to make it possible
 *        to modify @link operation_chain @endlink
 *
 * @tparam  functor_t  type of a functor that should be kept
 *
 * @note functor_t should fit specific requirements. This can be checked
 * with @link is_correct_functor() @endlink function
 */
template <class functor_t>
class custom_functor : public custom_operation {
public:
    /**
     * @brief Compile-time check if created functor is correct
     *
     * @return if the created functor is correct or not
     */
    static constexpr auto is_correct_functor() -> bool {
        return std::is_invocable<functor_t, const uint8_t *, uint32_t, uint8_t *, uint32_t>::value;
    }

    /**
     * @brief Type of functor that should be used for keeping a custom callable object
     */
    using functor_type = std::function<uint32_t(const uint8_t *, uint32_t, uint8_t *, uint32_t)>;

    /**
     * @brief Simple default constructor
     */
    constexpr custom_functor() = default;

    /**
     * @brief Basic constructor that accepts any callable object
     *
     * @param  functor  any callable object that satisfies
     *                  @link is_correct_functor() @endlink check
     */
    explicit custom_functor(functor_t functor)
            : functor_(std::move(functor)) {
        // Empty constructor
    }

    /**
     * @brief Executes created functor
     *
     * @param  source            pointer to source buffer
     * @param  source_size       size of source buffer
     * @param  destination       pointer to destination buffer
     * @param  destination_size  size of destination buffer
     *
     * @return number of bytes that were written into output
     */
    auto execute(const uint8_t *source,
                 uint32_t source_size,
                 uint8_t *destination,
                 uint32_t destination_size) -> uint32_t override {
        return functor_(source, source_size, destination, destination_size);
    }

private:
    functor_type functor_;    /**< Custom functor that is the implementation of custom operation */
};

/** @} */

} // namespace qpl

#endif // QPL_CUSTOM_OPERATION_HPP
