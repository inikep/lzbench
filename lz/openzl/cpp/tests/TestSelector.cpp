// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "cpp/tests/TestUtils.hpp"
#include "openzl/openzl.hpp"

using namespace ::testing;

namespace openzl {

class TestSelector : public Test {
   public:
    void SetUp() override
    {
        compressor_ = Compressor();
    }

    std::string testRoundTrip(
            const Input& input,
            std::shared_ptr<Selector> selector)
    {
        compressor_.selectStartingGraph(
                compressor_.registerSelectorGraph(std::move(selector)));
        return openzl::testRoundTrip(compressor_, input);
    }

    Compressor compressor_;
};

namespace {
class PickNthSelector : public Selector {
   public:
    PickNthSelector(
            TypeMask inputTypeMask,
            std::vector<GraphID> customGraphs,
            size_t n)
            : inputTypeMask_(inputTypeMask),
              customGraphs_(std::move(customGraphs)),
              n_(n)
    {
    }

    SelectorDescription selectorDescription() const override
    {
        return {
            .inputTypeMask = inputTypeMask_,
            .customGraphs  = customGraphs_,
        };
    }

    GraphID select(SelectorState& state, const Input&) const override
    {
        if (n_ >= state.customGraphs().size()) {
            return graphs::Compress{}();
        } else {
            return state.customGraphs()[n_];
        }
    }

   private:
    TypeMask inputTypeMask_;
    std::vector<GraphID> customGraphs_;
    size_t n_;
};

class BruteForceSelector : public Selector {
   public:
    BruteForceSelector(
            TypeMask inputTypeMask,
            std::vector<GraphID> customGraphs)
            : inputTypeMask_(inputTypeMask),
              customGraphs_(std::move(customGraphs))
    {
    }

    SelectorDescription selectorDescription() const override
    {
        return {
            .inputTypeMask = inputTypeMask_,
            .customGraphs  = customGraphs_,
        };
    }

    GraphID select(SelectorState& state, const Input& input) const override
    {
        GraphID bestGraph = graphs::Store::graph;
        size_t bestSize   = input.contentSize();
        for (const auto graph : state.customGraphs()) {
            auto perf = state.tryGraph(input, graph);
            if (perf.has_value() && perf->compressedSize < bestSize) {
                bestGraph = graph;
                bestSize  = perf->compressedSize;
            }
        }
        return bestGraph;
    }

   private:
    TypeMask inputTypeMask_;
    std::vector<GraphID> customGraphs_;
};
} // namespace

TEST_F(TestSelector, basicSerial)
{
    auto selector = std::make_shared<PickNthSelector>(
            TypeMask::Serial,
            std::vector<GraphID>{ graphs::Store::graph,
                                  graphs::Zstd{ 5 }(compressor_) },
            1);
    testRoundTrip(
            Input::refSerial(
                    "hello world hello hello hello hello hello hello hello hello hello"),
            selector);
}

TEST_F(TestSelector, selectGraphNotInList)
{
    auto selector = std::make_shared<PickNthSelector>(
            TypeMask::Serial, std::vector<GraphID>{}, 0);
    testRoundTrip(
            Input::refSerial(
                    "hello world hello hello hello hello hello hello hello hello hello"),
            selector);
}

TEST_F(TestSelector, basicNumeric)
{
    auto selector = std::make_shared<PickNthSelector>(
            TypeMask::Numeric,
            std::vector<GraphID>{ graphs::Constant::graph,
                                  graphs::Compress::graph },
            0);
    std::vector<int64_t> data(1000, 0x42);
    testRoundTrip(Input::refNumeric(poly::span<const int64_t>(data)), selector);
}

TEST_F(TestSelector, inputTypeMaskMultipleTypes)
{
    auto selector = std::make_shared<PickNthSelector>(
            TypeMask::Numeric | TypeMask::Serial,
            std::vector<GraphID>{ graphs::Constant::graph,
                                  graphs::Compress{}() },
            0);
    std::vector<int64_t> data(1000, 0x42);
    testRoundTrip(Input::refNumeric(poly::span<const int64_t>(data)), selector);
}

TEST_F(TestSelector, tryGraphSerial)
{
    auto selector = std::make_shared<BruteForceSelector>(
            TypeMask::Serial,
            std::vector<GraphID>{ graphs::Constant::graph,
                                  graphs::Compress::graph,
                                  graphs::Bitpack::graph,
                                  graphs::FieldLz::graph });
    auto compressed = testRoundTrip(
            Input::refSerial(
                    "hellohellohellohellohello world hello hello hello hello hello hello hello hello hello hellohellohellohellohellohellohello"),
            selector);
    ASSERT_LE(compressed.size(), 60);
}

TEST_F(TestSelector, tryGraphNumericAndStruct)
{
    auto selector = std::make_shared<BruteForceSelector>(
            TypeMask::Struct | TypeMask::Numeric,
            std::vector<GraphID>{ graphs::Constant::graph,
                                  graphs::Compress::graph,
                                  graphs::Bitpack::graph,
                                  graphs::FieldLz::graph,
                                  graphs::Bitpack::graph,
                                  graphs::Entropy::graph });
    std::vector<int64_t> data(1000, 0x42);
    auto compressed = testRoundTrip(
            Input::refNumeric(poly::span<const int64_t>(data)), selector);
    ASSERT_LE(compressed.size(), 50);
    auto compressed2 = testRoundTrip(
            Input::refStruct(poly::span<const int64_t>(data)), selector);
    ASSERT_LE(compressed2.size(), 50);
    poly::span<const int32_t> data32((int32_t*)data.data(), data.size() * 2);
    auto compressed32 = testRoundTrip(Input::refNumeric(data32), selector);
    ASSERT_GT(compressed32.size(), compressed.size());
}

} // namespace openzl
