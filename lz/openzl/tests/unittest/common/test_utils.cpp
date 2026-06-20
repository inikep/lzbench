// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/shared/utils.h"

namespace {

TEST(UtilsTest, max)
{
    ASSERT_EQ(ZL_MAX(0, 1), 1);
    ASSERT_EQ(ZL_MAX(-1, -2), -1);
}

TEST(UtilsTest, min)
{
    ASSERT_EQ(ZL_MIN(0, 1), 0);
    ASSERT_EQ(ZL_MIN(-1, -2), -2);
}

TEST(UtilsTest, isPow2)
{
    ASSERT_TRUE(ZL_isPow2(0));
    ASSERT_TRUE(ZL_isPow2(1));
    ASSERT_TRUE(ZL_isPow2(2));
    ASSERT_TRUE(ZL_isPow2(4));
    ASSERT_TRUE(ZL_isPow2(8));
    ASSERT_TRUE(ZL_isPow2(1ull << 63));
    ASSERT_FALSE(ZL_isPow2((uint64_t)-1));
    ASSERT_FALSE(ZL_isPow2(3));
    ASSERT_FALSE(ZL_isPow2(5));
    ASSERT_FALSE(ZL_isPow2(7));
    ASSERT_FALSE(ZL_isPow2(9));
}

TEST(UtilsTest, containerOf)
{
    struct Base {};
    struct Derived {
        int x;
        Base b;
    };

    Derived d;
    ASSERT_EQ(&d, ZL_CONTAINER_OF(&d.b, Derived, b));
    Base* p = nullptr;
    ASSERT_EQ((Derived*)nullptr, ZL_CONTAINER_OF(p, Derived, b));
}

TEST(UtilsTest, uintFits)
{
    ASSERT_TRUE(ZL_uintFits(0, 1));
    ASSERT_TRUE(ZL_uintFits(0xFF, 1));
    ASSERT_TRUE(ZL_uintFits(0xFFFF, 2));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFF, 3));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFFFF, 4));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFFFFFFULL, 5));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFFFFFFFFULL, 6));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFFFFFFFFFFULL, 7));
    ASSERT_TRUE(ZL_uintFits(0xFFFFFFFFFFFFFFFFULL, 8));
    ASSERT_FALSE(ZL_uintFits(0x100, 1));
    ASSERT_FALSE(ZL_uintFits(0x10000, 2));
    ASSERT_FALSE(ZL_uintFits(0x1000000, 3));
    ASSERT_FALSE(ZL_uintFits(0x100000000, 4));
    ASSERT_FALSE(ZL_uintFits(0x10000000000ULL, 5));
    ASSERT_FALSE(ZL_uintFits(0x1000000000000ULL, 6));
    ASSERT_FALSE(ZL_uintFits(0x100000000000000ULL, 7));
}

TEST(UtilsTest, SignComparisonIsAllowed)
{
    // GTest makes the flag extremely obnoxious because it doesn't allow type
    // unification of literals.
    unsigned x = 0;
    ASSERT_NE(x, 1) << "Tests should not be compiled with -Werror=sign-compare";
}

} // namespace
