// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/compress/selectors/selector_brute_force.h"

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/zl_selector.h"

ZL_GraphID SI_selector_brute_force(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs)
{
    ZL_ASSERT_NN(selCtx);
    ZL_ASSERT_NN(inputStream);
    ZL_ASSERT_NN(customGraphs);
    ZL_ASSERT_GT(nbCustomGraphs, 0);

    const ZL_Type inputType = ZL_Input_type(inputStream);
    // make sure the input can be piped into the graph
    for (size_t i = 0; i < nbCustomGraphs; ++i) {
        ZL_GraphID gid = customGraphs[i];
        ZL_ASSERT(ZL_Selector_getInput0MaskForGraph(selCtx, gid) & inputType);
    }

    // brute force all graphs
    size_t bestSize = ZL_Input_contentSize(inputStream);
    if (inputType == ZL_Type_string) {
        bestSize += ZL_Input_numElts(inputStream) * sizeof(uint32_t);
    }
    int64_t bestIdx = -1;
    for (size_t i = 0; i < nbCustomGraphs; ++i) {
        ZL_GraphReport gr =
                ZL_Selector_tryGraph(selCtx, inputStream, customGraphs[i]);
        if (ZL_isError(gr.finalCompressedSize)) {
            continue;
        }
        size_t currSize = ZL_validResult(gr.finalCompressedSize);
        // printf("curr: %zu, best: %zu\n", currSize, bestSize);
        if (currSize < bestSize) {
            bestSize = currSize;
            bestIdx  = (int64_t)i;
        }
    }
    if (bestIdx == -1) {
        return ZL_GRAPH_STORE;
    }
    return customGraphs[bestIdx];
}

ZL_GraphID ZL_Compressor_registerBruteForceSelectorGraph(
        ZL_Compressor* cgraph,
        const ZL_GraphID* successors,
        size_t numSuccessors)
{
    const ZL_SelectorDesc desc = {
        .selector_f   = SI_selector_brute_force,
        .inStreamType = ZL_Type_serial | ZL_Type_numeric | ZL_Type_struct
                | ZL_Type_string,
        .customGraphs   = successors,
        .nbCustomGraphs = numSuccessors,
        .name           = "brute_force selector",
    };
    return ZL_Compressor_registerSelectorGraph(cgraph, &desc);
}
