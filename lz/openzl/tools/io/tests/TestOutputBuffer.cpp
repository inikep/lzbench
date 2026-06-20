// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <sstream>
#include "tools/io/OutputBuffer.h"

using namespace ::testing;

TEST(TestOutputBuffer, Basic)
{
    std::ostringstream buf;
    openzl::tools::io::OutputBuffer out{ buf };
    out.write("hello");
    out.write("world");
    EXPECT_EQ(buf.str(), "helloworld");
}

TEST(TestOutputBuffer, ToInput)
{
    std::ostringstream buf;
    openzl::tools::io::OutputBuffer out{ buf };
    out.write("hello");
    out.write("world");
    EXPECT_EQ(buf.str(), "helloworld");

    auto input = out.to_input();
    EXPECT_EQ(input->contents(), "helloworld");
}
