// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <math.h>

#include <folly/dynamic.h>
#include <folly/json.h>

#include "openzl/common/assertion.h"
#include "tools/gbt_predictor/zstrong_gbt_predictor.h"

namespace zstrong::ml::gbt_predictor {

size_t GBTPredictor::initTreeFromJson(
        folly::dynamic const& json,
        std::unique_ptr<GBTPredictor_Node[]>& nodes)
{
    if (!json.isObject()) {
        throw std::runtime_error("Expected tree to be an object");
    }
    auto toIntVector = [](folly::dynamic const& json) -> std::vector<int> {
        std::vector<int> result;
        for (auto const& v : json) {
            result.push_back(v.asInt());
        }
        return result;
    };
    auto toFloatVector = [](folly::dynamic const& json) -> std::vector<float> {
        std::vector<float> result;
        for (auto const& v : json) {
            result.push_back(v.asDouble());
        }
        return result;
    };
    auto featureIdx    = toIntVector(json["featureIdx"]);
    auto leftChildIdx  = toIntVector(json["leftChildIdx"]);
    auto rightChildIdx = toIntVector(json["rightChildIdx"]);
    auto defaultLeft   = toIntVector(json["defaultLeft"]);
    auto values        = toFloatVector(json["value"]);

    const auto numNodes = featureIdx.size();
    if (numNodes != leftChildIdx.size() || numNodes != rightChildIdx.size()
        || numNodes != defaultLeft.size() || numNodes != values.size()) {
        throw std::runtime_error("Mismatched sizes in tree");
    }

    if (numNodes == 0) {
        throw std::runtime_error("Tree should have at least one node");
    }

    nodes = std::make_unique<GBTPredictor_Node[]>(numNodes);

    for (size_t i = 0; i < numNodes; ++i) {
        GBTPredictor_Node n;
        n.featureIdx    = featureIdx.at(i);
        n.leftChildIdx  = leftChildIdx.at(i);
        n.rightChildIdx = rightChildIdx.at(i);
        n.missingChildIdx =
                defaultLeft.at(i) ? n.leftChildIdx : n.rightChildIdx;
        n.value = values.at(i);

        // Verify that indices are valid
        auto verifyChildIdx = [&](int idx) {
            if (n.featureIdx == -1) {
                if (idx != -1) {
                    throw std::runtime_error("Invalid child index for a leaf");
                }
            } else if (idx <= (int)i || idx >= (int)numNodes) {
                // We always go forward and never beyond bound
                throw std::runtime_error(
                        "Invalid child index for an internal node");
            }
        };
        verifyChildIdx(n.leftChildIdx);
        verifyChildIdx(n.rightChildIdx);
        verifyChildIdx(n.missingChildIdx);

        // Verify that value is a valid number
        if (isnan(n.value) || isinf(n.value)) {
            throw std::runtime_error("Invalid value");
        }
        nodes[i] = n;
    }

    return numNodes;
}

GBTPredictor::GBTPredictor(folly::dynamic const& model)
{
    initFromJson(model);
}

GBTPredictor::GBTPredictor(std::string_view model)
{
    folly::dynamic const parsed = folly::parseJson(model);
    initFromJson(parsed);
}

void GBTPredictor::initFromJson(folly::dynamic const& model)
{
    if (!model.isArray()) {
        throw std::runtime_error("Cannot parse, expected array");
    }

    core_forests_ = std::make_unique<GBTPredictor_Forest[]>(model.size());

    int forest_ind = 0;
    for (auto const& forest : model) {
        // Forest is an array of trees, collect all trees for current forest
        // and add to forests_
        if (!forest.isArray()) {
            throw std::runtime_error("Expected forest to be an array");
        }

        std::unique_ptr<GBTPredictor_Tree[]> trees =
                std::make_unique<GBTPredictor_Tree[]>(forest.size());
        int tree_ind = 0;
        for (auto const& tree : forest) {
            std::unique_ptr<GBTPredictor_Node[]> nodes;
            size_t numNodes = initTreeFromJson(tree, nodes);
            core_nodes_.push_back(std::move(nodes));
            trees[tree_ind++] = { .numNodes = numNodes,
                                  .nodes    = core_nodes_.back().get() };
        }

        core_trees_.push_back(std::move(trees));

        core_forests_[forest_ind++] = { .numTrees = forest.size(),
                                        .trees    = core_trees_.back().get() };
    }

    core_predictor_ = std::make_shared<::GBTPredictor>((::GBTPredictor){
            .numForests = (size_t)forest_ind, .forests = core_forests_.get() });

    auto const report = GBTPredictor_validate(core_predictor_.get(), -1);
    if (ZL_isError(report)) {
        throw std::runtime_error(
                std::string("Invalid model: ")
                + ZL_ErrorCode_toString(ZL_errorCode(report)));
    }
}

size_t GBTPredictor::predict(const std::vector<float>& features) const
{
    return GBTPredictor_predict(
            core_predictor_.get(), features.data(), features.size());
}

size_t GBTPredictor::getNumClasses() const
{
    return GBTPredictor_getNumClasses(core_predictor_.get());
}

std::shared_ptr<::GBTPredictor> GBTPredictor::getCorePredictor() const
{
    return core_predictor_;
}

} // namespace zstrong::ml::gbt_predictor
