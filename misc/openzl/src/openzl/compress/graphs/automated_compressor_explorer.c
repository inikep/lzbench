// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/zl_ace.h"
#include "openzl/zl_compressor.h"

ZL_GraphID ZL_Compressor_buildACEGraph(ZL_Compressor* compressor)
{
    return ZL_Compressor_buildACEGraphWithDefault(
            compressor, ZL_GRAPH_COMPRESS_GENERIC);
}

ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildACEGraph2(ZL_Compressor* compressor)
{
    return ZL_Compressor_buildACEGraphWithDefault2(
            compressor, ZL_GRAPH_COMPRESS_GENERIC);
}

ZL_GraphID ZL_Compressor_buildACEGraphWithDefault(
        ZL_Compressor* compressor,
        ZL_GraphID defaultGraph)
{
    ZL_RESULT_OF(ZL_GraphID)
    result = ZL_Compressor_buildACEGraphWithDefault2(compressor, defaultGraph);
    if (ZL_RES_isError(result)) {
        return ZL_GRAPH_ILLEGAL;
    }
    return ZL_RES_value(result);
}

/// @see ZL_Compressor_buildACEGraphWithDefault
ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildACEGraphWithDefault2(
        ZL_Compressor* compressor,
        ZL_GraphID defaultGraph)
{
    ZL_GraphParameters params = {
        .name = "zl.ace",
    };
    return ZL_Compressor_parameterizeGraph(compressor, defaultGraph, &params);
}
