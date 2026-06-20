// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "tools/py/pybind_helpers.h"

#include <pybind11/pybind11.h>

using namespace ::testing;

namespace zstrong::pybind {
namespace py = pybind11;

template <typename T>
using fd = py::format_descriptor<T>;

TEST(PybindHelpersTest, getNativeIntegerSize)
{
    ASSERT_EQ(1, *getNativeIntegerSize("c"));
    ASSERT_EQ(1, *getNativeIntegerSize("@c"));
    ASSERT_EQ(sizeof(unsigned long), *getNativeIntegerSize("L"));

    ASSERT_EQ(std::nullopt, getNativeIntegerSize("!c"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("<c"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("=c"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize(">c"));

    ASSERT_EQ(std::nullopt, getNativeIntegerSize("@?"));

    ASSERT_EQ(std::nullopt, getNativeIntegerSize("?"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("e"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("f"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("d"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("s"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("p"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("P"));
    ASSERT_EQ(std::nullopt, getNativeIntegerSize("x"));

    ASSERT_EQ(std::nullopt, getNativeIntegerSize("LL"));
}

} // namespace zstrong::pybind
