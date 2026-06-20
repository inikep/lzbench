// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/common/limits.h"
#include "openzl/zl_version.h"

TEST(LimitsTest, TestVersionIncreaseForContrainerSize)
{
    const size_t container_size_limit = 1024 * 1024;
    ASSERT_GE((size_t)ZL_CONTAINER_SIZE_LIMIT, container_size_limit)
            << "ZL_CONTAINER_SIZE_LIMIT should never be decreased.";
    ASSERT_EQ((size_t)ZL_CONTAINER_SIZE_LIMIT, container_size_limit)
            << "ZL_CONTAINER_SIZE_LIMIT increases might result in encoder/decoder breakage. Please consider carefully and update ZL_MIN_FORMAT_VERSION if needed.";
}

TEST(LimitsTest, TestLimitsMonotonicallyIncrease)
{
    for (unsigned formatVersion = ZL_MIN_FORMAT_VERSION;
         formatVersion <= ZL_MAX_FORMAT_VERSION;
         ++formatVersion) {
        ASSERT_GE(
                ZL_runtimeStreamLimit(ZL_MAX_FORMAT_VERSION),
                ZL_runtimeStreamLimit(formatVersion));
        ASSERT_GE(
                ZL_runtimeNodeLimit(ZL_MAX_FORMAT_VERSION),
                ZL_runtimeNodeLimit(formatVersion));
        ASSERT_GE(
                ZL_transformOutStreamsLimit(ZL_MAX_FORMAT_VERSION),
                ZL_transformOutStreamsLimit(formatVersion));
    }
}
