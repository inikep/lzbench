/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <cstring>
#include "qpl/cpp_api/results/stream.hpp"

namespace qpl {

template <template <class> class allocator_t>
stream<allocator_t>::stream(stream &&other) noexcept {
    source_ = other.source_;
    size_   = other.size_;

    other.source_ = nullptr;
    other.size_   = 0;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::operator=(const stream &other) -> stream & {
    source_ = allocator_.allocate(other.size_);
    size_   = other.size_;

    std::memcpy(source_, other.source_, size_);

    return *this;
}

template <template <class> class allocator_t>
stream<allocator_t>::stream(const stream &other) {
    source_ = allocator_.allocate(other.size_);
    size_   = other.size_;

    std::memcpy(source_, other.source_, size_);
}

template <template <class> class allocator_t>
stream<allocator_t>::stream(const size_t size, uint32_t align) {
    size_   = align_size(size, align);
    source_ = allocator_.allocate(size_);
}

template <template <class> class allocator_t>
stream<allocator_t>::~stream() {
    allocator_.deallocate(source_, size_);
}

template <template <class> class allocator_t>
auto stream<allocator_t>::data() const noexcept -> uint8_t * {
    return source_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::size() const noexcept -> size_t {
    return size_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::operator[](size_t index) const -> uint8_t {
    if (index > size_) {
        throw std::out_of_range("index passed to stream is out of range.");
    }

    return source_[index];
}

template <template <class> class allocator_t>
constexpr stream<allocator_t>::stream_iterator::stream_iterator(
        stream<allocator_t>::stream_iterator::value_type *ptr)
        : pointer_(ptr) {
}

template <template <class> class allocator_t>
auto stream<allocator_t>::begin() const noexcept -> stream<allocator_t>::stream_iterator {
    return stream<allocator_t>::stream_iterator(const_cast<uint8_t *>(source_));
}

template <template <class> class allocator_t>
auto stream<allocator_t>::end() const noexcept -> stream<allocator_t>::stream_iterator {
    return stream<allocator_t>::stream_iterator(const_cast<uint8_t *>(source_ + size_));
}

template <template <class> class allocator_t>
auto stream<allocator_t>::align_size(size_t size, uint32_t align) const noexcept -> size_t {
    return (((static_cast<uint32_t>(size)) + (align) - 1) & ~((align) - 1));
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator++()
-> stream<allocator_t>::stream_iterator::reference {
    pointer_++;
    return *this;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator++(int value)
-> stream<allocator_t>::stream_iterator::reference {
    pointer_ += value;
    return *this;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator+=(int value)
-> stream<allocator_t>::stream_iterator::iterator_type & {
    pointer_ += value;
    return *this;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator-(stream<allocator_t>::stream_iterator::iterator_type other) const
-> stream<allocator_t>::stream_iterator::difference_type {
    return pointer_ - other.pointer_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator<(stream<allocator_t>::stream_iterator::iterator_type other) const
-> bool {
    return pointer_ < other.pointer_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator>(stream<allocator_t>::stream_iterator::iterator_type other) const
-> bool {
    return pointer_ > other.pointer_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator!=(stream<allocator_t>::stream_iterator::iterator_type other) const
-> bool {
    return pointer_ != other.pointer_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator==(stream<allocator_t>::stream_iterator::iterator_type other) const
-> bool {
    return pointer_ == other.pointer_;
}

template <template <class> class allocator_t>
auto stream<allocator_t>::stream_iterator::operator*() const -> stream<allocator_t>::stream_iterator::value_type & {
    return *pointer_;
}

} // namespace qpl
