// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <type_traits>

#include "openzl/cpp/detail/Portability.hpp"
#include "openzl/cpp/poly/Byte.hpp"
#include "openzl/cpp/poly/Iterator.hpp"
#include "openzl/cpp/poly/TypeTraits.hpp"

#if ZL_CPP_HAS_SPAN
#    include <span>
#endif

namespace openzl {
namespace poly {
namespace impl {

template <typename R>
using DataValueType =
        std::remove_pointer<decltype(poly::data(std::declval<R>()))>;

template <typename R, typename T>
using IsConvertibleFrom =
        std::is_convertible<typename DataValueType<R>::type (*)[], T (*)[]>;

template <typename T, std::size_t Extent>
class span;

template <typename T>
struct IsArray : std::is_array<T> {};

template <typename T, std::size_t N>
struct IsArray<std::array<T, N>> : std::true_type {};

template <typename T>
struct IsSpan : std::false_type {};

template <typename T, std::size_t N>
struct IsSpan<span<T, N>> : std::true_type {};

template <typename T, typename = void>
struct IsContiguousSizedRange : std::false_type {};

template <typename T>
struct IsContiguousSizedRange<
        T,
        poly::void_t<
                decltype(poly::data(std::declval<T>())),
                decltype(poly::size(std::declval<T>()))>> : std::true_type {};

template <
        typename T,
        typename U = typename std::remove_cv<
                typename std::remove_reference<T>::type>::type>
using IsConstructibleFrom = std::conjunction<
        std::negation<IsArray<U>>,
        std::negation<IsSpan<U>>,
        IsContiguousSizedRange<T>>;

ZL_CPP_INLINE_VARIABLE constexpr std::size_t dynamic_extent =
        std::numeric_limits<std::size_t>::max();

template <typename T, std::size_t Extent = dynamic_extent>
class span {
   public:
    using element_type           = T;
    using value_type             = typename std::remove_cv<T>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using pointer                = T*;
    using const_pointer          = const T*;
    using reference              = T&;
    using const_reference        = const T&;
    using iterator               = pointer;
    using const_iterator         = const_pointer;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_assert(Extent == dynamic_extent, "Only dynamic_extent supported");

    span() noexcept = default;
    span(pointer first, size_type count) noexcept : data_(first), size_(count)
    {
    }
    span(pointer first, pointer last) noexcept
            : data_(first), size_(last - first)
    {
        assert(first <= last);
    }
    template <size_type N>
    span(element_type (&arr)[N]) noexcept : data_(arr), size_(N)
    {
    }
    template <
            typename U,
            size_type N,
            typename = typename std::enable_if<
                    IsConvertibleFrom<std::array<U, N>&, T>::value>::type>
    span(std::array<U, N>& arr) noexcept : data_(arr.data()), size_(N)
    {
    }
    template <
            typename U,
            size_type N,
            typename = typename std::enable_if<
                    IsConvertibleFrom<const std::array<U, N>&, T>::value>::type>
    span(const std::array<U, N>& arr) noexcept : data_(arr.data()), size_(N)
    {
    }

    template <
            typename R,
            typename = typename std::enable_if<
                    IsConstructibleFrom<R>::value
                    && IsConvertibleFrom<R, T>::value>::type>
    span(R&& r) noexcept : data_(poly::data(r)), size_(poly::size(r))
    {
    }
    template <
            typename U,
            size_type N,
            typename = typename std::enable_if<
                    IsConvertibleFrom<const span<U, N>&, T>::value>::type>
    span(const span<U, N>& other) noexcept
            : data_(other.data()), size_(other.size())
    {
    }
    span(const span& other) noexcept = default;

    span& operator=(const span& other) noexcept = default;

    iterator begin() const noexcept
    {
        return data_;
    }
    const_iterator cbegin() const noexcept
    {
        return data_;
    }

    iterator end() const noexcept
    {
        return data_ + size_;
    }
    const_iterator cend() const noexcept
    {
        return data_ + size_;
    }

    reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator{ end() };
    }
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator{ cend() };
    }

    reverse_iterator rend() const noexcept
    {
        return reverse_iterator{ begin() };
    }
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator{ cbegin() };
    }

    reference front() const
    {
        assert(size_ > 0);
        return data_[0];
    }

    reference back() const
    {
        assert(size_ > 0);
        return data_[size_ - 1];
    }

    reference operator[](size_type idx) const
    {
        assert(idx < size_);
        return data_[idx];
    }

    pointer data() const noexcept
    {
        return data_;
    }

    size_type size() const noexcept
    {
        return size_;
    }

    size_type size_bytes() const noexcept
    {
        return size_ * sizeof(element_type);
    }

    bool empty() const noexcept
    {
        return size_ == 0;
    }

    template <size_type Count>
    span<element_type, Count> first() const
    {
        static_assert(Count == dynamic_extent, "Only dynamic_extent supported");
    }

    span<element_type, dynamic_extent> first(size_type count) const
    {
        assert(count <= size_);
        return span(data(), count);
    }

    template <size_type Count>
    span<element_type, Count> last() const
    {
        static_assert(Count == dynamic_extent, "Only dynamic_extent supported");
    }

    span<element_type, dynamic_extent> last(size_type count) const
    {
        assert(count <= size_);
        return span(data() + size_ - count, count);
    }

    template <size_type Offset, size_type Count = dynamic_extent>
    span<element_type, Count> subspan() const
    {
        static_assert(Count == dynamic_extent, "Only dynamic_extent supported");
        return subspan(Offset);
    }

    span<element_type, dynamic_extent> subspan(
            size_type offset,
            size_type count = dynamic_extent) const
    {
        assert(offset <= size_);
        if (count == dynamic_extent) {
            count = size_ - offset;
        }
        assert(count <= (size_ - offset));
        return span(data() + offset, count);
    }

   private:
    pointer data_{};
    size_type size_{};
};

template <typename T, std::size_t N>
span<const poly::byte, N> as_bytes(span<T, N> s) noexcept
{
    static_assert(N == dynamic_extent, "Only dynamic_extent supported");
    return span<const poly::byte, N>(
            reinterpret_cast<const poly::byte*>(s.data()), s.size_bytes());
}

template <
        typename T,
        std::size_t N,
        typename = typename std::enable_if<!std::is_const<T>::value>::type>
span<poly::byte, N> as_writable_bytes(span<T, N> s) noexcept
{
    static_assert(N == dynamic_extent, "Only dynamic_extent supported");
    return span<poly::byte, N>(
            reinterpret_cast<poly::byte*>(s.data()), s.size_bytes());
}

} // namespace impl

#if ZL_CPP_HAS_SPAN
using std::as_bytes;
using std::as_writable_bytes;
using std::span;
#else
using impl::as_bytes;
using impl::as_writable_bytes;
using impl::span;
#endif

} // namespace poly
} // namespace openzl
