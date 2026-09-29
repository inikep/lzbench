// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/compress/selectors/selector_genericLZ.h"
#include "openzl/zl_public_nodes.h"
// #include "openzl/common/assertion.h"

/* SI_selector_genericLZ():
 *
 * Note : this function is currently a bit too trivial,
 * and merely arbitrates between fastLZ and ROLZ depending on compression level
 * based on an arbitrary cut off value (<= 4.)
 * It will have to be updated in the future,
 * to better measure the cut off point,
 * take in consideration more parameters (such as decompression level)
 * and possibly integrate more choices (such as other LZ backends).
 */

ZL_GraphID SI_selector_genericLZ(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs)
{
    (void)selCtx;
    (void)inputStream;
    (void)customGraphs;
    (void)nbCustomGraphs;
    return ZL_GRAPH_ZSTD;
}
