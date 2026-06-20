// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <thread>

#include "tools/time/timefn.h"

namespace {

TIME_t now()
{
    return TIME_getTime();
}

TEST(TimeTest, ClockConsistency)
{
    auto const start = now();
    TIME_waitForNextTick();
    {
        auto const stop = now();

        ASSERT_GT(TIME_span_ns(start, stop), (Duration_ns)0);

        ASSERT_TRUE(TIME_span_ns(start, stop) == TIME_span_ns(start, stop));
    }
}

} // namespace
