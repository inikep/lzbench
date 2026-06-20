// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "openzl/codecs/common/copy.h"

namespace {
TEST(CopyTest, overlapCopy8Offset0)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data;
    ZS_overlapCopy8(&op, &ip, 0);
    ASSERT_EQ(op, (uint8_t*)data + 8);
    // We don't care what the data is or what ip is
    // Only that data <= ip <= op
    ASSERT_GE(ip, (uint8_t*)data);
    ASSERT_LE(ip, op);
}

TEST(CopyTest, overlapCopy8Offset1)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 1;
    ZS_overlapCopy8(&op, &ip, 1);
    ASSERT_EQ(std::string((const char*)data), "0000000009abcdef");
    ASSERT_EQ(op, (uint8_t*)data + 9);
    ASSERT_EQ(ip, (uint8_t*)data + 1);
}

TEST(CopyTest, overlapCopy8Offset2)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 2;
    ZS_overlapCopy8(&op, &ip, 2);
    ASSERT_EQ(std::string((const char*)data), "0101010101abcdef");
    ASSERT_EQ(op, (uint8_t*)data + 10);
    ASSERT_EQ(ip, (uint8_t*)data + 2);
}

TEST(CopyTest, overlapCopy8Offset3)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 3;
    ZS_overlapCopy8(&op, &ip, 3);
    ASSERT_EQ(std::string((const char*)data), "01201201201bcdef");
    ASSERT_EQ(op, (uint8_t*)data + 11);
    ASSERT_EQ(ip, (uint8_t*)data + 2);
}

TEST(CopyTest, overlapCopy8Offset4)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 4;
    ZS_overlapCopy8(&op, &ip, 4);
    ASSERT_EQ(std::string((const char*)data), "012301230123cdef");
    ASSERT_EQ(op, (uint8_t*)data + 12);
    ASSERT_EQ(ip, (uint8_t*)data + 4);
}

TEST(CopyTest, overlapCopy8Offset5)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 5;
    ZS_overlapCopy8(&op, &ip, 5);
    ASSERT_EQ(std::string((const char*)data), "0123401234012def");
    ASSERT_EQ(op, (uint8_t*)data + 13);
    ASSERT_EQ(ip, (uint8_t*)data + 3);
}

TEST(CopyTest, overlapCopy8Offset6)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 6;
    ZS_overlapCopy8(&op, &ip, 6);
    ASSERT_EQ(std::string((const char*)data), "01234501234501ef");
    ASSERT_EQ(op, (uint8_t*)data + 14);
    ASSERT_EQ(ip, (uint8_t*)data + 2);
}

TEST(CopyTest, overlapCopy8Offset7)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 7;
    ZS_overlapCopy8(&op, &ip, 7);
    ASSERT_EQ(std::string((const char*)data), "012345601234560f");
    ASSERT_EQ(op, (uint8_t*)data + 15);
    ASSERT_EQ(ip, (uint8_t*)data + 1);
}

TEST(CopyTest, overlapCopy8Offset8)
{
    uint8_t data[]    = "0123456789abcdef";
    uint8_t const* ip = (uint8_t const*)data;
    uint8_t* op       = (uint8_t*)data + 8;
    ZS_overlapCopy8(&op, &ip, 8);
    ASSERT_EQ(std::string((const char*)data), "0123456701234567");
    ASSERT_EQ(op, (uint8_t*)data + 16);
    ASSERT_EQ(ip, (uint8_t*)data + 8);
}

std::array<uint8_t, 3 * ZS_WILDCOPY_OVERLENGTH> getWildcopyArray()
{
    std::array<uint8_t, 3 * ZS_WILDCOPY_OVERLENGTH> data;
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t)i;
    }
    return data;
}

TEST(CopyTest, wildcopyTestAssumptions)
{
    // Tests assumes this.
    ASSERT_EQ(ZS_WILDCOPY_VECLEN * 2, ZS_WILDCOPY_OVERLENGTH);
}

TEST(CopyTest, wildcopyNoOverlapDstBeforeSrc)
{
    auto data = getWildcopyArray();
    // No overlap dst before src
    ZS_wildcopy(
            data.data(),
            data.data() + ZS_WILDCOPY_VECLEN,
            3 * ZS_WILDCOPY_VECLEN,
            ZS_wo_no_overlap);
    for (size_t i = 0; i < 3 * ZS_WILDCOPY_VECLEN; ++i) {
        ASSERT_EQ(data[i], ZS_WILDCOPY_VECLEN + i);
    }
}

TEST(CopyTest, wildcopyNoOverlapSrcBeforeDst)
{
    // No overlap src before dst
    auto data = getWildcopyArray();
    ZS_wildcopy(
            data.data() + ZS_WILDCOPY_VECLEN,
            data.data(),
            3 * ZS_WILDCOPY_VECLEN,
            ZS_wo_no_overlap);
    for (size_t i = 0; i < 3 * ZS_WILDCOPY_VECLEN; ++i) {
        ASSERT_EQ(data[ZS_WILDCOPY_VECLEN + i], i & 15);
    }
}

TEST(CopyTest, wildcopySrcBeforeDst)
{
    for (size_t offset = 1; offset <= 2 * ZS_WILDCOPY_VECLEN; ++offset) {
        auto data           = getWildcopyArray();
        auto check          = getWildcopyArray();
        size_t const length = data.size() - ZS_WILDCOPY_OVERLENGTH - offset;
        ZS_wildcopy(
                data.data() + offset,
                data.data(),
                (ptrdiff_t)length,
                ZS_wo_src_before_dst);
        for (size_t i = 0; i < length; ++i) {
            check[offset + i] = check[i];
        }
        for (size_t i = 0; i < offset + length; ++i) {
            ASSERT_EQ(data[i], check[i]);
        }
    }
}

TEST(CopyTest, wildcopyNegative)
{
    auto data = getWildcopyArray();
    // Negative length doesn't access out of bounds memory
    ZS_wildcopy(
            data.data() + data.size() - ZS_WILDCOPY_OVERLENGTH,
            data.data(),
            -1,
            ZS_wo_no_overlap);
    ZS_wildcopy(
            data.data(),
            data.data() + data.size() - ZS_WILDCOPY_OVERLENGTH,
            -1,
            ZS_wo_no_overlap);
    ZS_wildcopy(
            data.data() + data.size() - ZS_WILDCOPY_OVERLENGTH,
            data.data(),
            -1,
            ZS_wo_src_before_dst);
}

} // namespace
