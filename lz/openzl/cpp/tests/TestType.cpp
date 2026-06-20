// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/cpp/Type.hpp"

namespace openzl::tests {
TEST(TestType, TypeToCType)
{
    EXPECT_EQ(typeToCType(Type::Serial), ZL_Type_serial);
    EXPECT_EQ(typeToCType(Type::Struct), ZL_Type_struct);
    EXPECT_EQ(typeToCType(Type::Numeric), ZL_Type_numeric);
    EXPECT_EQ(typeToCType(Type::String), ZL_Type_string);
}

TEST(TestType, TypeMaskToCType)
{
    EXPECT_EQ(typeMaskToCType(TypeMask::Serial), ZL_Type_serial);
    EXPECT_EQ(typeMaskToCType(TypeMask::Struct), ZL_Type_struct);
    EXPECT_EQ(typeMaskToCType(TypeMask::Numeric), ZL_Type_numeric);
    EXPECT_EQ(typeMaskToCType(TypeMask::String), ZL_Type_string);
    EXPECT_EQ(typeMaskToCType(TypeMask::Any), ZL_Type_any);
    EXPECT_EQ(typeMaskToCType(TypeMask::None), ZL_Type_unassigned);
}
} // namespace openzl::tests
