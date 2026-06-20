// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/cpp/poly/Span.hpp"

using namespace ::testing;

using openzl::poly::byte;

template <typename T>
using Span = openzl::poly::impl::span<T>;
using openzl::poly::impl::as_bytes;
using openzl::poly::impl::as_writable_bytes;

namespace openzl::tests {
class PolySpanTest : public Test {
   public:
    int carray_[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::array<int, 10> array_{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<int> data_{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
};

TEST_F(PolySpanTest, Construct1)
{
    Span<int> span;
    ASSERT_EQ(span.data(), nullptr);
    ASSERT_EQ(span.size(), 0);
}

TEST_F(PolySpanTest, Construct2)
{
    Span<int> span(data_.data(), data_.size());
    ASSERT_EQ(span.data(), data_.data());
    ASSERT_EQ(span.size(), data_.size());
}

TEST_F(PolySpanTest, Construct3)
{
    Span<int> span(data_.data(), data_.data() + data_.size());
    ASSERT_EQ(span.data(), data_.data());
    ASSERT_EQ(span.size(), data_.size());
}

TEST_F(PolySpanTest, Construct4)
{
    Span<int> span(carray_);
    ASSERT_EQ(span.data(), carray_);
    ASSERT_EQ(span.size(), sizeof(carray_) / sizeof(carray_[0]));
}

TEST_F(PolySpanTest, Construct5)
{
    Span<int> span(array_);
    ASSERT_EQ(span.data(), array_.data());
    ASSERT_EQ(span.size(), array_.size());

    Span<const int> span2(array_);
    ASSERT_EQ(span2.data(), array_.data());
    ASSERT_EQ(span2.size(), array_.size());
}

TEST_F(PolySpanTest, Construct6)
{
    const auto& array = array_;
    Span<const int> span(array);
    ASSERT_EQ(span.data(), array.data());
    ASSERT_EQ(span.size(), array.size());
}

TEST_F(PolySpanTest, Construct7)
{
    Span<int> span(data_);
    ASSERT_EQ(span.data(), data_.data());
    ASSERT_EQ(span.size(), data_.size());

    Span<const int> span2(data_);
    ASSERT_EQ(span2.data(), data_.data());
    ASSERT_EQ(span2.size(), data_.size());
}

TEST_F(PolySpanTest, Construct8)
{
    auto test = [](std::initializer_list<int> data) {
        Span<const int> span(data);
        ASSERT_NE(span.data(), nullptr);
        ASSERT_EQ(span.size(), 10);
        ASSERT_EQ(span[0], 1);
        ASSERT_EQ(span[9], 10);
    };
    test({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
}

TEST_F(PolySpanTest, Construct9)
{
    Span<int> span(data_);
    Span<const int> span2(span);
    ASSERT_EQ(span2.data(), data_.data());
    ASSERT_EQ(span2.size(), data_.size());
}

TEST_F(PolySpanTest, Construct10)
{
    Span<int> span(data_);
    Span<int> span2(span);
    ASSERT_EQ(span2.data(), data_.data());
    ASSERT_EQ(span2.size(), data_.size());
}

TEST_F(PolySpanTest, Iterators)
{
    Span<int> span(data_);
    ASSERT_EQ(span.begin(), data_.data());
    ASSERT_EQ(span.end(), data_.data() + data_.size());
    ASSERT_EQ(span.cbegin(), data_.data());
    ASSERT_EQ(span.cend(), data_.data() + data_.size());
    ASSERT_EQ(*span.rbegin(), *data_.rbegin());
    ASSERT_EQ(span.rend() - span.rbegin(), data_.size());
    ASSERT_EQ(*span.crbegin(), *data_.crbegin());
    ASSERT_EQ(span.crend() - span.crbegin(), data_.size());
}

TEST_F(PolySpanTest, ElementAccess)
{
    Span<int> span(data_);
    ASSERT_EQ(span.data(), data_.data());
    ASSERT_EQ(span.front(), data_.front());
    ASSERT_EQ(span.back(), data_.back());
    for (size_t i = 0; i < data_.size(); ++i) {
        ASSERT_EQ(span[i], data_[i]);
    }
}

TEST_F(PolySpanTest, Observers)
{
    Span<int> span(data_);
    ASSERT_EQ(span.size(), data_.size());
    ASSERT_EQ(span.size_bytes(), data_.size() * sizeof(int));
    ASSERT_EQ(span.empty(), data_.empty());
}

TEST_F(PolySpanTest, Subviews)
{
    Span<int> span(data_);
    auto first = span.first(5);
    ASSERT_EQ(first.size(), 5);
    ASSERT_EQ(first.data(), data_.data());

    auto last = span.last(5);
    ASSERT_EQ(last.size(), 5);
    ASSERT_EQ(last.data(), data_.data() + data_.size() - 5);

    auto sub = span.subspan(2, 5);
    ASSERT_EQ(sub.size(), 5);
    ASSERT_EQ(sub.data(), data_.data() + 2);

    auto sub2 = span.subspan(2);
    ASSERT_EQ(sub2.size(), data_.size() - 2);
    ASSERT_EQ(sub2.data(), data_.data() + 2);
}

TEST_F(PolySpanTest, AsBytes)
{
    Span<int> span(data_);
    auto bytes       = as_bytes(span);
    const byte* data = bytes.data();
    ASSERT_EQ(bytes.size(), data_.size() * sizeof(int));
    ASSERT_EQ(data, reinterpret_cast<const byte*>(data_.data()));
}

TEST_F(PolySpanTest, AsWritableBytes)
{
    Span<int> span(data_);
    auto bytes = as_writable_bytes(span);
    byte* data = bytes.data();
    ASSERT_EQ(bytes.size(), data_.size() * sizeof(int));
    ASSERT_EQ(data, reinterpret_cast<byte*>(data_.data()));
}

// TEST_F(PolySpanTest)

} // namespace openzl::tests
