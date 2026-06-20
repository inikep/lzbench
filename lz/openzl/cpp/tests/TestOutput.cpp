// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "openzl/openzl.hpp"

using namespace testing;

namespace openzl::tests {

TEST(TestOutput, get)
{
    Output output;
    ASSERT_NE(output.get(), nullptr);
}

TEST(TestOutput, defaultConstructor)
{
    Output output;
    ASSERT_THROW(output.type(), Exception);
    ASSERT_THROW(output.ptr(), Exception);
    ASSERT_THROW(output.eltWidth(), Exception);
    ASSERT_THROW(output.numElts(), Exception);
    ASSERT_THROW(output.contentSize(), Exception);
    ASSERT_THROW(output.stringLens(), Exception);
}

TEST(TestOutput, wrapSerial)
{
    char buffer[5];
    auto output = Output::wrapSerial({ buffer, sizeof(buffer) });
    ASSERT_EQ(output.type(), Type::Serial);
    ASSERT_EQ(output.eltWidth(), 1);
    ASSERT_EQ(output.ptr(), buffer);
    ASSERT_THROW(output.contentSize(), Exception);
    ASSERT_THROW(output.numElts(), Exception);
    ASSERT_THROW(output.stringLens(), Exception);

    output.commit(4);
    ASSERT_EQ(output.contentSize(), 4);
    ASSERT_EQ(output.numElts(), 4);
}

TEST(TestOutput, wrapStruct)
{
    std::array<int, 5> buffer{};
    auto output = Output::wrapStruct(poly::span<int>{ buffer });
    ASSERT_EQ(output.type(), Type::Struct);
    ASSERT_EQ(output.eltWidth(), 4);
    ASSERT_EQ(output.ptr(), buffer.data());
    ASSERT_THROW(output.contentSize(), Exception);
    ASSERT_THROW(output.numElts(), Exception);
    ASSERT_THROW(output.stringLens(), Exception);

    output.commit(4);
    ASSERT_EQ(output.contentSize(), 16);
    ASSERT_EQ(output.numElts(), 4);
}

TEST(TestOutput, wrapNumeric)
{
    std::array<int, 5> buffer{};
    auto output = Output::wrapNumeric(poly::span<int>{ buffer });
    ASSERT_EQ(output.type(), Type::Numeric);
    ASSERT_EQ(output.eltWidth(), 4);
    ASSERT_EQ(output.ptr(), buffer.data());
    ASSERT_THROW(output.contentSize(), Exception);
    ASSERT_THROW(output.numElts(), Exception);
    ASSERT_THROW(output.stringLens(), Exception);

    output.commit(4);
    ASSERT_EQ(output.contentSize(), 16);
    ASSERT_EQ(output.numElts(), 4);
}

TEST(TestOutput, setIntMetadata)
{
    Output output;
    ASSERT_EQ(output.getIntMetadata(0), poly::nullopt);
    output.setIntMetadata(0, 42);
    ASSERT_EQ(output.getIntMetadata(0), 42);
}

// TODO(terrelln): Test OutputRef functions once C++ custom codec support is
// implemented

} // namespace openzl::tests
