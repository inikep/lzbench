// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <gtest/gtest.h>

#include "openzl/shared/mem.h"

namespace {

template <typename Int, typename Read>
void testRead(Read const& read)
{
    uint64_t const src64 = 0x0123456789abcdefull;
    Int const src        = Int(src64);
    Int const dst        = read(&src);
    ASSERT_EQ(src, dst);
}

// Once we've validated read we can use it to validate
// that write round trips.
template <typename Int, size_t C, typename Read, typename Write>
void testWrite(Read const& read, Write const& write)
{
    uint8_t const src64[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    uint8_t src[C];
    memcpy(src, src64, C);
    uint8_t dst[C];
    Int const r1 = read(src);
    write(dst, r1);
    Int const r2 = read(dst);
    ASSERT_EQ(r1, r2);
}

TEST(MemTest, read)
{
    testRead<uint16_t>(ZL_read16);
    testRead<uint32_t>(ZL_read32);
    testRead<uint64_t>(ZL_read64);
    testRead<size_t>(ZL_readST);
}

TEST(MemTest, write)
{
    testWrite<uint16_t, 2>(ZL_read16, ZL_write16);
    testWrite<uint32_t, 4>(ZL_read32, ZL_write32);
    testWrite<uint64_t, 8>(ZL_read64, ZS_write64);
}

TEST(MemTest, readLE)
{
    uint8_t const src[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    ASSERT_EQ(ZL_readLE16(src), 0x2301u);
    ASSERT_EQ(ZL_readLE24(src), 0x452301u);
    ASSERT_EQ(ZL_readLE32(src), 0x67452301u);
    ASSERT_EQ(ZL_readLE64(src), 0xefcdab8967452301ull);
    if (ZL_32bits()) {
        ASSERT_EQ(ZL_readLEST(src), ZL_readLE32(src));
    } else {
        ASSERT_EQ(ZL_readLEST(src), ZL_readLE64(src));
    }
}

TEST(MemTest, readBE)
{
    uint8_t const src[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    ASSERT_EQ(ZL_readBE16(src), 0x0123u);
    ASSERT_EQ(ZL_readBE24(src), 0x012345u);
    ASSERT_EQ(ZL_readBE32(src), 0x01234567u);
    ASSERT_EQ(ZL_readBE64(src), 0x0123456789abcdefull);
    if (ZL_32bits()) {
        ASSERT_EQ(ZL_readBEST(src), ZL_readBE32(src));
    } else {
        ASSERT_EQ(ZL_readBEST(src), ZL_readBE64(src));
    }
}

TEST(MemTest, writeLE)
{
    testWrite<uint16_t, 2>(ZL_readLE16, ZL_writeLE16);
    testWrite<uint32_t, 3>(ZL_readLE24, ZL_writeLE24);
    testWrite<uint32_t, 4>(ZL_readLE32, ZL_writeLE32);
    testWrite<uint64_t, 8>(ZL_readLE64, ZL_writeLE64);
    testWrite<uint64_t, sizeof(size_t)>(ZL_readLEST, ZL_writeLEST);
}

TEST(MemTest, writeBE)
{
    testWrite<uint16_t, 2>(ZL_readBE16, ZL_writeBE16);
    testWrite<uint32_t, 3>(ZL_readBE24, ZL_writeBE24);
    testWrite<uint32_t, 4>(ZL_readBE32, ZL_writeBE32);
    testWrite<uint64_t, 8>(ZL_readBE64, ZL_writeBE64);
    testWrite<uint64_t, sizeof(size_t)>(ZL_readBEST, ZL_writeBEST);
}

} // namespace
