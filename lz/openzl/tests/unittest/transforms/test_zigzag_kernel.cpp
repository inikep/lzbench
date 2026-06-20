// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/codecs/zigzag/decode_zigzag_kernel.h"
#include "openzl/codecs/zigzag/encode_zigzag_kernel.h"

namespace {

template <class I, class U>
void roundTrip()
{
    EXPECT_EQ(sizeof(I), sizeof(U));

    I const minlim     = std::numeric_limits<I>::min();
    I const maxlim     = std::numeric_limits<I>::max();
    std::vector<I> src = { 0, -1, 1, -2, 2, minlim, maxlim };
    std::vector<U> dst(src.size());
    std::vector<I> recon(src.size());

    ZL_zigzagEncode(dst.data(), src.data(), src.size(), sizeof(U));

    ZL_zigzagDecode(recon.data(), dst.data(), dst.size(), sizeof(U));

    EXPECT_EQ(src, recon);
}

TEST(ZigzagKernelTest, roundTrip64)
{
    roundTrip<int64_t, uint64_t>();
}

TEST(ZigzagKernelTest, roundTrip32)
{
    roundTrip<int32_t, uint32_t>();
}

} // namespace
