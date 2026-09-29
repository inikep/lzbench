// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/entropy/encode_entropy_selector.h"

#include "openzl/codecs/constant/encode_constant_binding.h"
#include "openzl/common/assertion.h"
#include "openzl/compress/private_nodes.h"
#include "openzl/shared/data_stats.h"
#include "openzl/shared/utils.h"

/* EI_selector_entropy():
 *
 * The goal of this selector is to select the *best* entropy codec to run
 * given an input interpreted as a bunch of bytes (ZL_Type_serial).
 *
 * "best" however must be defined.
 * This is a combination of :
 * - allowed decompression speed
 * - time budget for analysis
 * - statistics of input
 *
 * Among the entropy codecs available, we can consider :
 * - FSE : the strongest for large inputs and squeezed statistics.
 *         Reasonably fast, but still slower than huffman.
 *         Worse header size.
 * - Huffman : the workhorse, dynamically adapts to real statistics.
 *         excellent compression ratio for "average" statistics,
 *         neither squeezed, nor noisy.
 *         average header size.
 * - Range : useful when values are present in range [0-X].
 *         more flexible than bitPack, as X can be any value.
 *         but also a bit slower. Tiny header.
 *         Likely useful when nb of values to encode is too small
 *         to make up for huffman's header.
 * - BitPack : trivial, great speed,
 *         Like range, but the X in [0,X] must be a power of 2.
 * - Constant : specific, only useful when all values are identical.
 *         fastest and simplest in this specific case.
 * - STORE : when data is basically incompressible,
 *         or not compressible enough given speed targets.
 *
 * Other entropy coder techniques can be added to this list later on,
 * featuring different speed / compression trade-off.
 */

ZL_GraphID EI_selector_entropy(ZL_Graph const* gctx, ZL_Edge const* sctx)
{
    ZL_Input const* inputStream = ZL_Edge_getData(sctx);
    ZL_ASSERT_EQ(ZL_Input_eltWidth(inputStream), 1);
    DataStatsU8 stats;
    DataStatsU8_init(
            &stats, ZL_Input_ptr(inputStream), ZL_Input_numElts(inputStream));

    size_t compressionSizes[]    = { stats.srcSize,
                                     DataStatsU8_getBitpackedSize(&stats),
                                     DataStatsU8_getFlatpackedSize(&stats),
                                     DataStatsU8_estimateHuffmanSizeFast(
                                          &stats, false),
                                     DataStatsU8_getConstantSize(&stats) };
    size_t const numCompressions = ZL_ARRAY_SIZE(compressionSizes);

    if (!ZL_Graph_isConstantSupported(gctx)) {
        compressionSizes[4] = (size_t)-1;
    }

    size_t minSizePos = 0;
    for (size_t i = 1; i < numCompressions; i++) {
        if (compressionSizes[i] < compressionSizes[minSizePos]) {
            minSizePos = i;
        }
    }

    ZL_GraphID const graphs[] = { ZL_GRAPH_STORE,
                                  ZL_GRAPH_BITPACK,
                                  ZL_GRAPH_FLATPACK,
                                  ZL_GRAPH_HUFFMAN,
                                  ZL_GRAPH_CONSTANT_SERIAL };

    ZL_ASSERT_EQ(ZL_ARRAY_SIZE(graphs), ZL_ARRAY_SIZE(compressionSizes));
    ZL_ASSERT_LT(minSizePos, ZL_ARRAY_SIZE(graphs));

    return graphs[minSizePos];
}
