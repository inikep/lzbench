// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/zl_reflection.h"
#include "tests/codecs/test_codec.h"

using namespace ::testing;

namespace openzl {
namespace tests {

class ACEGraphTest : public CodecTest {
   public:
    void testACE(
            const Input& input,
            int minFormatVersion = ZL_MIN_FORMAT_VERSION)
    {
        for (int formatVersion = minFormatVersion;
             formatVersion <= ZL_MAX_FORMAT_VERSION;
             ++formatVersion) {
            compressor_.setParameter(CParam::FormatVersion, formatVersion);
            compressor_.selectStartingGraph(graphs::ACE{}(compressor_));
            testRoundTrip(input);
        }
    }
};

TEST_F(ACEGraphTest, Basic)
{
    testACE(Input::refSerial(
            "hello hello hello hello hello world hello world hello hello hello"));
    std::vector<uint32_t> data(1000, 42);
    data.push_back(350);
    testACE(Input::refStruct(poly::span<const uint32_t>(data)),
            ZL_TYPED_INPUT_VERSION_MIN);
    testACE(Input::refNumeric(poly::span<const uint32_t>(data)),
            ZL_TYPED_INPUT_VERSION_MIN);
    std::string content(300, 'a');
    std::vector<uint32_t> lengths = { 50, 100, 50, 10, 20, 30, 40 };
    testACE(Input::refString(content, lengths), ZL_TYPED_INPUT_VERSION_MIN);
}

TEST_F(ACEGraphTest, HasCorrectName)
{
    auto graph = graphs::ACE{}(compressor_);
    ASSERT_EQ(
            ZL_Compressor_Graph_getName(compressor_.get(), graph),
            std::string("zl.ace#0"));
}

} // namespace tests
} // namespace openzl
