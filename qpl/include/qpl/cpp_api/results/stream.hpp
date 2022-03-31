/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_STREAM_HPP
#define QPL_STREAM_HPP

#include <cstdint>

namespace qpl {
/**
 * @addtogroup HL_PUBLIC
 * @{
 */

/**
 * @brief Class that represents the result of @link operation_chain @endlink execution
 *
 * @tparam  allocator_t  type of the allocator to be used
 */
template <template <class> class allocator_t = std::allocator>
class stream final {
public:
    /**
     * @brief Simple iterator for @link stream @endlink
     */
    class stream_iterator {
    public:
        /**
         * Type of the iterator
         */
        using iterator_type = stream_iterator;

        /**
         * Category of the iterator
         */
        using iterator_category = std::random_access_iterator_tag;

        /**
         * Type of iterator value
         */
        using value_type = uint8_t;

        /**
         * Type of distance between iterators
         */
        using difference_type = size_t;

        /**
         * Type of the pointer to value
         */
        using pointer = iterator_type *;

        /**
         * Type of the reference to value
         */
        using reference = iterator_type &;

        /**
         * @brief Simple constructor
         */
        constexpr explicit stream_iterator(value_type *ptr);

        /**
         * @brief Increments iterator to the next position
         */
        auto operator++() -> reference;

        /**
         * @brief Pre-increments iterator to the next position
         */
        auto operator++(int value) -> reference;

        /**
         * @brief Increments iterator to several positions ahead
         */
        auto operator+=(int value) -> reference;

        /**
         * @brief Calculates the distance between iterators
         */
        auto operator-(iterator_type other) const -> difference_type;

        /**
         * @brief Compares two iterators if given one is further than this
         */
        auto operator<(iterator_type other) const -> bool;

        /**
         * @brief Compares two iterators if this is further than given one
         */
        auto operator>(iterator_type other) const -> bool;

        /**
         * @brief Compares two iterators for inequality
         */
        auto operator!=(iterator_type other) const -> bool;

        /**
         * @brief Compares two iterators for equality
         */
        auto operator==(iterator_type other) const -> bool;

        /**
         * @brief Dereferences the iterator
         */
        auto operator*() const -> value_type &;

    private:
        value_type *pointer_;    /**< Pointer to current element for iterator */
    };

    /**
     * @brief Default constructor
     */
    stream() = default;

    /**
     * @brief Simple copy constructor
     */
    stream(const stream &other);

    /**
     * @brief Simple move constructor
     */
    stream(stream &&other) noexcept;

    /**
     * @brief Simple assignment operator
     */
    auto operator=(const stream &other) -> stream &;

    /**
     * @brief Constructor with default memory allocation
     *
     * @param  size   number of bytes that should be allocated
     * @param  align  value to align the size with
     */
    explicit stream(size_t size, uint32_t align = default_alignment);

    /**
     * @brief Simple destructor
     */
    ~stream();

    /**
     * @brief Method that allows to get the result as any desired type
     *
     * @tparam  return_t  type that should be returned
     *
     * @return new representation of the result
     */
    template <class return_t>
    auto as() -> return_t {
        return reinterpret_cast<return_t>(source_);
    }

    /**
     * @return Pointer to data
     */
    [[nodiscard]] auto data() const noexcept -> uint8_t *;

    /**
     * @return size in bytes of the stream
     */
    [[nodiscard]] auto size() const noexcept -> size_t;

    /**
     * @brief Accesses the element by the given index
     *
     * @param  index  of required element
     *
     * @return value of required element
     */
    [[nodiscard]] auto operator[](size_t index) const -> uint8_t;

    /**
     * @brief Returns iterator to the beginning of the stream
     */
    [[nodiscard]] auto begin() const noexcept -> stream_iterator;

    /**
     * @brief Returns iterator to the end of the stream
     */
    [[nodiscard]] auto end() const noexcept -> stream_iterator;

private:
    uint8_t              *source_ = nullptr;          /**< Pointer to the stream content */
    mutable size_t       size_    = 0;                /**< Size of the stream content */
    allocator_t<uint8_t> allocator_{};                /**< Instance of allocator */

    static constexpr uint32_t default_alignment = 64; /**< Default alignment of the stream size */

    /**
     * @brief Method for stream size alignment
     *
     * @param  size   value for alignment
     * @param  align  value to align with
     *
     * @return alignment value
     */
    [[nodiscard]] auto align_size(size_t size, uint32_t align) const noexcept -> size_t;
};

/** @} */

} // namespace qpl

#include "stream.cxx"

#endif // QPL_STREAM_HPP
