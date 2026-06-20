// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/shared/hash.h"
#include "openzl/shared/mem.h"

namespace {

template <size_t Bytes, typename Int, typename Hash, typename HashPtr>
void testHash(Hash const& hash, HashPtr const& hashPtr)
{
    for (uint32_t bits = 1; bits < 32; ++bits) {
        uint32_t const mask = (1u << bits) - 1;
        uint8_t src[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
        uint64_t val   = ZL_readLE64(src);
        size_t const h = hash((Int)val, bits);
        // Check that all the hashes match
        ASSERT_EQ(h, hashPtr(src, bits));
        ASSERT_EQ(h, ZL_hashPtr(src, bits, Bytes));
        ASSERT_EQ(h & ~mask, 0u);
        // Check that we can zero bytes that we don't hash
        for (size_t b = Bytes; b < 8; ++b) {
            src[b] = 0;
            ASSERT_EQ(h, hashPtr(src, bits));
            ASSERT_EQ(h, ZL_hashPtr(src, bits, Bytes));
        }
        // Check that zeroing hashed bytes changes the hash
        if (bits > 10) {
            for (size_t b = 0; b < Bytes; ++b) {
                src[b] = 0;
                ASSERT_NE(h, hashPtr(src, bits));
                ASSERT_NE(h, ZL_hashPtr(src, bits, Bytes));
            }
        }
    }
}

TEST(HashTest, hash)
{
    testHash<3, uint32_t>(ZL_hash3, ZL_hash3Ptr);
    testHash<4, uint32_t>(ZL_hash4, ZL_hash4Ptr);
    testHash<5, uint64_t>(ZL_hash5, ZL_hash5Ptr);
    testHash<6, uint64_t>(ZL_hash6, ZL_hash6Ptr);
    testHash<7, uint64_t>(ZL_hash7, ZL_hash7Ptr);
    testHash<8, uint64_t>(ZL_hash8, ZL_hash8Ptr);
}

} // namespace
