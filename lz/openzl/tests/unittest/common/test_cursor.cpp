// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/common/cursor.h"
#include "tests/utils.h"

namespace {

void checkAvail(const ZL_RC& cursor, size_t available)
{
    EXPECT_TRUE(ZL_RC_has(&cursor, available));
    EXPECT_FALSE(ZL_RC_has(&cursor, available + 1));
}

void checkAvail(const ZL_WC& cursor, size_t available)
{
    EXPECT_TRUE(ZL_WC_has(&cursor, available));
    EXPECT_FALSE(ZL_WC_has(&cursor, available + 1));
}

std::string popStr(ZL_RC& cursor, size_t count)
{
    return std::string((const char*)ZL_RC_pull(&cursor, count), count);
}

TEST(WriteCursorTest, testWrap)
{
    constexpr size_t size = 100;
    uint8_t buf[size];

    auto wc = ZL_WC_wrap(buf, size);
    EXPECT_EQ(ZL_WC_begin(&wc), buf);
    EXPECT_EQ(ZL_WC_cbegin(&wc), buf);
    EXPECT_EQ(ZL_WC_ptr(&wc), buf);
    EXPECT_EQ(ZL_WC_cptr(&wc), buf);
    EXPECT_EQ(ZL_WC_size(&wc), 0u);
    EXPECT_EQ(ZL_WC_avail(&wc), size);
    EXPECT_EQ(ZL_WC_capacity(&wc), size);
    EXPECT_TRUE(ZL_WC_has(&wc, size));
    EXPECT_FALSE(ZL_WC_has(&wc, size + 1));
}

TEST(WriteCursorTest, testWrapFull)
{
    constexpr size_t size = 100;
    uint8_t buf[size];

    auto wc = ZL_WC_wrapFull(buf, size);
    EXPECT_EQ(ZL_WC_begin(&wc), buf);
    EXPECT_EQ(ZL_WC_cbegin(&wc), buf);
    EXPECT_EQ(ZL_WC_ptr(&wc), buf + size);
    EXPECT_EQ(ZL_WC_cptr(&wc), buf + size);
    EXPECT_EQ(ZL_WC_size(&wc), size);
    EXPECT_EQ(ZL_WC_avail(&wc), 0u);
    EXPECT_EQ(ZL_WC_capacity(&wc), size);
    EXPECT_TRUE(ZL_WC_has(&wc, 0));
    EXPECT_FALSE(ZL_WC_has(&wc, 1));
}

TEST(WriteCursorTest, testWrapPartial)
{
    constexpr size_t size = 100;
    constexpr size_t pos  = 33;
    uint8_t buf[size];

    auto wc = ZL_WC_wrapPartial(buf, pos, size);
    EXPECT_EQ(ZL_WC_begin(&wc), buf);
    EXPECT_EQ(ZL_WC_cbegin(&wc), buf);
    EXPECT_EQ(ZL_WC_ptr(&wc), buf + pos);
    EXPECT_EQ(ZL_WC_cptr(&wc), buf + pos);
    EXPECT_EQ(ZL_WC_size(&wc), pos);
    EXPECT_EQ(ZL_WC_avail(&wc), size - pos);
    EXPECT_EQ(ZL_WC_capacity(&wc), size);
    EXPECT_TRUE(ZL_WC_has(&wc, size - pos));
    EXPECT_FALSE(ZL_WC_has(&wc, size - pos + 1));
}

TEST(WriteCursorTest, testWrite)
{
    constexpr size_t size = 100;
    uint8_t buf[size];

    auto wc = ZL_WC_wrap(buf, size);
    checkAvail(wc, 100);
    ZL_WC_push(&wc, 0xFFu);
    checkAvail(wc, 99);
    ZL_WC_shove(&wc, (const uint8_t*)"\xFE\xFD\0\xFC", 4);
    checkAvail(wc, 95);

    auto rc = ZL_RC_wrapWC(&wc);
    checkAvail(rc, 5);
    EXPECT_EQ(popStr(rc, 2), std::string("\xFF\xFE", 2));
    EXPECT_EQ(popStr(rc, 3), std::string("\xFD\0\xFC", 3));
}

TEST(ReadCursorTest, testWrap)
{
    constexpr size_t size = 100;
    uint8_t buf[size]     = { 0 };

    auto rc = ZL_RC_wrap(buf, size);
    EXPECT_EQ(ZL_RC_ptr(&rc), buf);
    EXPECT_EQ(ZL_RC_avail(&rc), size);
    EXPECT_TRUE(ZL_RC_has(&rc, size));
    EXPECT_FALSE(ZL_RC_has(&rc, size + 1));
}

TEST(ReadCursorTest, testPrefix)
{
    constexpr size_t size = 100;
    uint8_t buf[size]     = { 0 };

    auto rc      = ZL_RC_wrap(buf, size);
    auto prefix1 = ZL_RC_prefix(&rc, 50);
    EXPECT_EQ(ZL_RC_ptr(&prefix1), ZL_RC_ptr(&rc));
    EXPECT_EQ(ZL_RC_avail(&prefix1), 50u);
    auto prefix2 = ZL_RC_prefix(&rc, 100);
    EXPECT_EQ(ZL_RC_avail(&prefix2), 100u);
}

TEST(ReadCursorTest, testSubtract)
{
    constexpr size_t size = 100;
    uint8_t buf[size]     = { 0 };

    auto rc = ZL_RC_wrap(buf, size);
    EXPECT_EQ(ZL_RC_ptr(&rc), buf);
    EXPECT_EQ(ZL_RC_avail(&rc), 100u);
    ZL_RC_subtract(&rc, 1);
    EXPECT_EQ(ZL_RC_ptr(&rc), buf);
    EXPECT_EQ(ZL_RC_avail(&rc), 99u);
    ZL_RC_subtract(&rc, 99);
    EXPECT_EQ(ZL_RC_avail(&rc), 0u);
}

TEST(ReadCursorTest, testRPop)
{
    constexpr size_t size = 2;
    uint8_t buf[size]     = { 0x00, 0x01 };

    auto rc = ZL_RC_wrap(buf, size);
    EXPECT_EQ(0x01, ZL_RC_rPop(&rc));
    EXPECT_EQ(0x00, ZL_RC_rPop(&rc));
    EXPECT_EQ(ZL_RC_avail(&rc), 0u);
}

TEST(ReadCursorTest, testRPull)
{
    constexpr size_t size = 100;
    uint8_t buf[size]     = { 0 };

    auto rc                  = ZL_RC_wrap(buf, size);
    uint8_t const* const ptr = ZL_RC_rPull(&rc, 10);
    EXPECT_EQ(ZL_RC_ptr(&rc), buf);
    EXPECT_EQ(ZL_RC_avail(&rc), 90u);
    EXPECT_EQ(ptr, buf + 90);
}

TEST(ReadCursorTest, testRPop32)
{
    std::vector<uint32_t> x = { 0x01234567, 0x12345678, 0x23456789 };

    auto rc            = ZL_RC_wrap((uint8_t const*)x.data(), 4 * x.size());
    uint32_t const xHE = ZL_RC_rPopHE32(&rc);
    uint32_t const xBE = ZL_RC_rPopBE32(&rc);
    uint32_t const xLE = ZL_RC_rPopLE32(&rc);
    EXPECT_EQ(x[2], xHE);
    if (ZL_isLittleEndian()) {
        EXPECT_EQ(x[1], ZL_swap32(xBE));
        EXPECT_EQ(x[0], xLE);
    } else {
        EXPECT_EQ(x[1], xBE);
        EXPECT_EQ(x[0], ZL_swap32(xLE));
    }
}

TEST(ReadCursorTest, testRead)
{
    std::string buf = "0123456789ABCDEF";

    auto rc = ZL_RC_wrap((const uint8_t*)buf.data(), buf.size());
    checkAvail(rc, buf.size());

    EXPECT_EQ(ZL_RC_pop(&rc), '0');
    EXPECT_EQ(ZL_RC_pop(&rc), '1');
    EXPECT_EQ(popStr(rc, 2), "23");
    EXPECT_EQ(popStr(rc, 4), "4567");
    EXPECT_EQ(std::string((const char*)ZL_RC_ptr(&rc)), "89ABCDEF");
    checkAvail(rc, 8);
    EXPECT_EQ(ZL_RC_pop(&rc), '8');
    EXPECT_EQ(popStr(rc, 7), "9ABCDEF");
}

TEST(ReadCursorTest, testRoundTripInts)
{
    std::string buf;
    buf.resize(100);

    auto wcobj = ZL_WC_wrap((uint8_t*)&buf[0], buf.size());
    auto wc    = &wcobj;

    const uint16_t uint16_val = 0xFEDCu;
    const uint32_t uint24_val = 0xFEDCBAu;
    const uint32_t uint32_val = 0xFEDCBA98u;
    const uint64_t uint64_val = 0xFEDCBA9876543210ull;

    const size_t size_val = ZL_64bits() ? uint64_val : uint32_val;

    ZL_WC_pushHE16(wc, uint16_val);
    ZL_WC_pushBE16(wc, uint16_val);
    ZL_WC_pushLE16(wc, uint16_val);

    ZL_WC_pushHE24(wc, uint24_val);
    ZL_WC_pushBE24(wc, uint24_val);
    ZL_WC_pushLE24(wc, uint24_val);

    ZL_WC_pushHE32(wc, uint32_val);
    ZL_WC_pushBE32(wc, uint32_val);
    ZL_WC_pushLE32(wc, uint32_val);

    ZL_WC_pushHE64(wc, uint64_val);
    ZL_WC_pushBE64(wc, uint64_val);
    ZL_WC_pushLE64(wc, uint64_val);

    const size_t expected_size = (2 + 3 + 4 + 8) * 3;
    EXPECT_EQ(ZL_WC_size(wc), expected_size);

    ZL_WC_pushHEST(wc, size_val);
    ZL_WC_pushBEST(wc, size_val);
    ZL_WC_pushLEST(wc, size_val);

    ZL_WC_pushVarint(wc, uint16_val);
    ZL_WC_pushVarint(wc, uint24_val);
    ZL_WC_pushVarint(wc, uint32_val);
    ZL_WC_pushVarint(wc, uint64_val);

    auto rcobj = ZL_RC_wrapWC(wc);
    auto rc    = &rcobj;

    EXPECT_EQ(ZL_RC_popHE16(rc), uint16_val);
    EXPECT_EQ(ZL_RC_popBE16(rc), uint16_val);
    EXPECT_EQ(ZL_RC_popLE16(rc), uint16_val);

    EXPECT_EQ(ZL_RC_popHE24(rc), uint24_val);
    EXPECT_EQ(ZL_RC_popBE24(rc), uint24_val);
    EXPECT_EQ(ZL_RC_popLE24(rc), uint24_val);

    EXPECT_EQ(ZL_RC_popHE32(rc), uint32_val);
    EXPECT_EQ(ZL_RC_popBE32(rc), uint32_val);
    EXPECT_EQ(ZL_RC_popLE32(rc), uint32_val);

    EXPECT_EQ(ZL_RC_popHE64(rc), uint64_val);
    EXPECT_EQ(ZL_RC_popBE64(rc), uint64_val);
    EXPECT_EQ(ZL_RC_popLE64(rc), uint64_val);

    EXPECT_EQ(ZL_RC_popHEST(rc), size_val);
    EXPECT_EQ(ZL_RC_popBEST(rc), size_val);
    EXPECT_EQ(ZL_RC_popLEST(rc), size_val);

    auto res = ZL_RC_popVarint(rc);
    EXPECT_ZS_VALID(res);
    EXPECT_EQ(ZL_RES_value(res), uint16_val);
    res = ZL_RC_popVarint(rc);
    EXPECT_ZS_VALID(res);
    EXPECT_EQ(ZL_RES_value(res), uint24_val);
    res = ZL_RC_popVarint(rc);
    EXPECT_ZS_VALID(res);
    EXPECT_EQ(ZL_RES_value(res), uint32_val);
    res = ZL_RC_popVarint(rc);
    EXPECT_ZS_VALID(res);
    EXPECT_EQ(ZL_RES_value(res), uint64_val);

    EXPECT_EQ(ZL_RC_avail(rc), 0u);
}

} // namespace
