// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cpp/tests/TestCustomCodec.hpp"

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"

using namespace testing;

namespace openzl::tests {

namespace {
class TestCustomCodec : public Test {
   public:
    void SetUp() override
    {
        compressor_ = Compressor();
        cctx_       = CCtx();
        dctx_       = DCtx();
    }

    void testRoundTrip(poly::span<const Input> inputs)
    {
        auto compressed   = cctx_.compress(inputs);
        auto decompressed = dctx_.decompress(compressed);
        EXPECT_EQ(decompressed.size(), inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            EXPECT_EQ(decompressed[i], inputs[i]);
        }
    }

    void testRoundTrip(const Input& input)
    {
        testRoundTrip({ &input, 1 });
    }

    Compressor compressor_;
    CCtx cctx_;
    DCtx dctx_;
};

const MultiInputCodecDescription kPlusOneDescription{
    .id                   = 1,
    .name                 = "!PlusOne",
    .inputTypes           = { Type::Numeric },
    .singletonOutputTypes = { Type::Numeric }
};
} // anonymous namespace

MultiInputCodecDescription PlusOneEncoder::multiInputDescription() const
{
    return kPlusOneDescription;
}

void PlusOneEncoder::encode(EncoderState& encoderState) const
{
    const auto& input = encoderState.inputs()[0];
    auto output =
            encoderState.createOutput(0, input.numElts(), input.eltWidth());
    assert(input.eltWidth() == 4);
    auto src = poly::span<const uint32_t>{ (const uint32_t*)input.ptr(),
                                           input.numElts() };
    auto dst = poly::span<uint32_t>{ (uint32_t*)output.ptr(), input.numElts() };
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i] + 1;
    }
    output.commit(dst.size());
}

MultiInputCodecDescription PlusOneDecoder::multiInputDescription() const
{
    return kPlusOneDescription;
}

void PlusOneDecoder::decode(DecoderState& decoderState) const
{
    const auto& input = decoderState.singletonInputs()[0];
    auto output =
            decoderState.createOutput(0, input.numElts(), input.eltWidth());
    assert(input.eltWidth() == 4);
    auto src = poly::span<const uint32_t>{ (const uint32_t*)input.ptr(),
                                           input.numElts() };
    auto dst = poly::span<uint32_t>{ (uint32_t*)output.ptr(), input.numElts() };
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i] - 1;
    }
    output.commit(dst.size());
}

TEST_F(TestCustomCodec, SimpleCodec)
{
    auto node = compressor_.registerCustomEncoder(
            std::make_shared<PlusOneEncoder>());
    auto graph = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor_.get(), node, ZL_GRAPH_ZSTD);
    compressor_.unwrap(
            ZL_Compressor_selectStartingGraphID(compressor_.get(), graph));
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    cctx_.refCompressor(compressor_);
    dctx_.registerCustomDecoder(std::make_shared<PlusOneDecoder>());

    std::vector<uint32_t> data(1000);
    data.push_back(1);
    data.push_back(2);
    auto input = Input::refNumeric(poly::span<const uint32_t>(data));
    testRoundTrip(input);
}
} // namespace openzl::tests
