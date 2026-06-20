// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/codecs/bitpack/common_bitpack_kernel.h"

namespace {

template <typename Int>
class Bitpack {
   private:
    int _nbBits;
    size_t _length;

   public:
    Bitpack(int nbBits, size_t length) : _nbBits(nbBits), _length(length) {}
    std::vector<uint8_t> encode(const std::vector<Int>& src)
    {
        std::vector<uint8_t> encoded(
                ZS_bitpackEncodeBound(src.size(), _nbBits));
        if (encoded.size() > 0) {
            memset(encoded.data(), 0, encoded.size() * sizeof(encoded[0]));
        }
        const size_t actual_size = ZS_bitpackEncode(
                encoded.data(),
                encoded.size(),
                src.data(),
                src.size(),
                sizeof(Int),
                _nbBits);
        EXPECT_EQ(actual_size, encoded.size());
        return encoded;
    }

    std::vector<Int> decode(const std::vector<uint8_t>& encoded)
    {
        std::vector<Int> decoded(_length);
        const size_t decodedBytes = ZS_bitpackDecode(
                decoded.data(),
                decoded.size(),
                sizeof(Int),
                encoded.data(),
                encoded.size(),
                _nbBits);
        EXPECT_EQ(decodedBytes, encoded.size());
        return decoded;
    }

    std::vector<Int> getData()
    {
        std::vector<Int> vec(_length);
        std::mt19937 mersenne_engine(10);
        std::uniform_int_distribution<Int> dist(
                0, (~(Int)0) >> (sizeof(Int) * 8 - (unsigned)_nbBits));
        auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
        };

        std::generate(vec.begin(), vec.end(), gen);
        return vec;
    }

    void test()
    {
        auto src     = getData();
        auto encoded = encode(src);
        auto decoded = decode(encoded);
        ASSERT_EQ(src, decoded);
    }
};

template <typename Int>
void testAllNbBits(size_t length)
{
    for (size_t nbBits = 1; nbBits <= sizeof(Int) * 8; nbBits++) {
        Bitpack<uint64_t>((int)nbBits, length).test();
    }
}

template <typename Int>
void testAllNbBitsAndLengths()
{
    testAllNbBits<Int>(0);
    testAllNbBits<Int>(1);
    testAllNbBits<Int>(5);
    testAllNbBits<Int>(20);
    testAllNbBits<Int>(100);
    testAllNbBits<Int>(1000);
}

TEST(BitpackTest, test8)
{
    testAllNbBitsAndLengths<uint8_t>();
}

TEST(BitpackTest, test16)
{
    testAllNbBitsAndLengths<uint16_t>();
}

TEST(BitpackTest, test32)
{
    testAllNbBitsAndLengths<uint32_t>();
}

TEST(BitpackTest, test64)
{
    testAllNbBitsAndLengths<uint64_t>();
}

} // namespace
