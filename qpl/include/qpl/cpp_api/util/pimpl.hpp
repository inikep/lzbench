/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_HIGH_LEVEL_API_UTIL_PIMPL_HPP
#define QPL_HIGH_LEVEL_API_UTIL_PIMPL_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace qpl::util {

/**
 * @brief Class that represents pimpl idiom without dynamic memory allocation
 *
 * @tparam  T          native object type
 * @tparam  size       size of object
 * @tparam  alignment  alignment for object
 */
template <class T,
        size_t size,
        size_t alignment>
class pimpl {
public:
    /**
     * Main constructor
     */
    template <class... arguments>
    explicit pimpl(arguments &&... args) {
        new(data()) T(std::forward<arguments>(args)...);
    }

    /**
     * @brief Copy constructor
     */
    pimpl(const pimpl &other) {
        *data() = *other;
    }

    /**
     * @brief Move constructor
     */
    pimpl(pimpl &&other) noexcept {
        *data() = std::move(*other);
    };

    /**
     * @brief Copy assignment operator
     */
    auto operator=(const pimpl &other) -> pimpl & {
        *data() = *other;

        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    auto operator=(pimpl &&other) noexcept -> pimpl & {
        *data() = std::move(*other);

        return *this;
    }

    /**
     * @brief Member access operator
     */
    auto operator->() noexcept -> T * {
        return data();
    }

    /**
     * @brief Constant member access operator
     */
    auto operator->() const noexcept -> const T * {
        return data();
    }

    /**
     * @brief Dereference operator
     */
    auto operator*() noexcept -> T & {
        return *data();
    }

    /**
     * @brief Constant dereference operator
     */
    auto operator*() const noexcept -> const T & {
        return *data();
    }

    /**
     * @brief Class destructor with validation
     */
    ~pimpl() noexcept {
        validate<sizeof(T), alignof(T)>();
        data()->~T();
    }

    /**
     * @brief Returns a direct pointer to the data
     */
    auto data() noexcept -> T * {
        return reinterpret_cast<T *>(&data_);
    }

    /**
     * @brief Constant method that returns a direct pointer to the data
     */
    auto data() const noexcept -> const T * {
        return reinterpret_cast<const T *>(&data_);
    }

private:
    template <size_t actual_size,
            size_t actual_alignment>
    static void validate() noexcept {
        static_assert(size == actual_size, "Size and sizeof(T) mismatch");
        static_assert(alignment == actual_alignment, "Alignment and alignof(T) mismatch");
    }

    std::aligned_storage_t<size, alignment> data_;
};

}

#endif // QPL_HIGH_LEVEL_API_UTIL_PIMPL_HPP
