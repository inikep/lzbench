/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_ALLOCATION_UTIL_HPP
#define QPL_ALLOCATION_UTIL_HPP

#include <functional>
#include <memory>

namespace qpl::util {

/**
 * @addtogroup HL_UTIL
 * @{
 */

/**
 * array_t of the function that may be used as allocator
 */
using allocator_type = uint8_t *(*)(uint32_t);

/**
 * array_t of the function that may be used as deleter
 */
using deleter_t = void (*)(uint8_t *, uint32_t);

/**
 * @brief Creates anonymous function that acts as allocator
 *
 * @tparam  allocator_t  type of the allocator to be used
 */
template <template <class> class allocator_t>
constexpr auto get_allocator_function() -> allocator_type {
    return [](uint32_t size) -> uint8_t * {
        allocator_t <uint8_t> allocator;
        return allocator.allocate(size);
    };
}

/**
 * @brief Creates anonymous function that acts as deleter
 *
 * @tparam  allocator_t  type of the allocator to be used
 */
template <template <class> class allocator_t>
constexpr auto get_deleter_function() -> deleter_t {
    return [](uint8_t *pointer, uint32_t size) {
        allocator_t<uint8_t> allocator;
        allocator.deallocate(pointer, size * sizeof(uint8_t));
    };
}

/**
 * @brief Creates std::unique_ptr with custom deleter
 *
 * @tparam  allocator_t  type of the allocator to be used
 * @tparam  array_t      type of array element
 *
 * @param size number of elements in array
 */
template <template <class> class allocator_t, class array_t>
auto allocate_array(const uint32_t size) -> std::unique_ptr<array_t[], std::function<void(array_t * )>> {
    allocator_t<array_t> allocator;

    auto deleter = [size](array_t *ptr) {
        allocator_t<array_t> allocator;

        allocator.deallocate(ptr, size);
    };

    return std::unique_ptr<array_t[], std::function<void(array_t *)>>(allocator.allocate(size), deleter);
}

/** @} */

} // namespace qpl::util

#endif // QPL_ALLOCATION_UTIL_HPP
