// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/dynamic.h>
#include <gtest/gtest.h>

#include "tools/zstrong_cpp.h"
#include "tools/zstrong_json.h"

using namespace ::testing;

namespace zstrong {

// standard declarations
folly::dynamic store()
{
    return folly::dynamic::object()(kNameKey, "store");
}

static folly::dynamic concat(
        const folly::dynamic& successor_sizes,
        const folly::dynamic& successor_concat)
{
    folly::dynamic graph = folly::dynamic::object();
    graph[kNameKey]      = "concat_serial";
    graph[kSuccessorsKey] =
            folly::dynamic::array(successor_sizes, successor_concat);
    return graph;
}

class ZstrongCppTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        size_t currIdx = 0;
        size_t nextIdx = 0;
        while ((nextIdx = data.find(' ', currIdx)) != std::string::npos) {
            stringSizes.push_back(nextIdx + 1 - currIdx);
            currIdx = nextIdx + 1;
        }
        stringSizes.push_back(data.size() - currIdx);
    }

    std::string_view data =
            "In einem Bächlein helle, "
            "Da schoß in froher Eil "
            "Die launische Forelle "
            "Vorüber wie ein Pfeil. "
            "Ich stand an dem Gestade "
            "Und sah in süßer Ruh "
            "Des muntern Fischleins Bade "
            "Im klaren Bächlein zu.";
    std::vector<size_t> stringSizes;
};

TEST_F(ZstrongCppTest, MultiInputTransform)
{
    auto graph = concat(store(), store());
    JsonGraph jsonGraph(graph);
    std::vector<std::string_view> dataVec{ data, data };
    auto compressed = compressMulti(dataVec, jsonGraph);
    // cannot use the single-output version with a multi-input compression
    EXPECT_THROW(
            { auto decompressedFail = decompress(compressed); },
            std::runtime_error);
    auto decompressed = decompressMulti(compressed, jsonGraph);

    ASSERT_EQ(dataVec.size(), decompressed.size());
    for (size_t i = 0; i < dataVec.size(); ++i) {
        ASSERT_EQ(dataVec[i], decompressed[i]);
    }
}

} // namespace zstrong
