// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/utils/utils.h"
#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "tools/io/InputSetStatic.h"

namespace openzl::training {

CCtx refCCtxForTraining(const Compressor& compressor)
{
    openzl::CCtx cctx;
    cctx.setParameter(openzl::CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    cctx.setParameter(openzl::CParam::StickyParameters, ZL_MAX_FORMAT_VERSION);
    cctx.refCompressor(compressor);
    return cctx;
}

std::vector<MultiInput> inputSetToMultiInputs(tools::io::InputSet& inputs)
{
    // Convert the io inputs to MultiInputs
    std::vector<MultiInput> multiInputs;
    for (auto& input : inputs) {
        auto mi = openzl::training::MultiInput();
        mi.add(input);
        multiInputs.push_back(std::move(mi));
    }
    return multiInputs;
}

} // namespace openzl::training
