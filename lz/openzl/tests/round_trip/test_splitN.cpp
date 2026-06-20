// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

// standard C
#include <stdio.h> // printf

// Zstrong
#include "openzl/common/debug.h" // ZL_REQUIRE
#include "openzl/compress/private_nodes.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_decompress.h" // ZL_decompress

namespace {

#if 0 // for debug only
void printHexa(const void* p, size_t size)
{
    const unsigned char* const b = (const unsigned char*)p;
    for (size_t n = 0; n < size; n++) {
        printf(" %02X ", b[n]);
    }
    printf("\n");
}
#endif

/* ------   create a custom nodes   -------- */

// None needed

/* ------   create custom parsers   -------- */

static unsigned const kTag = 0xABCDEF;

// Following ZL_SplitParserFn() signature
static ZL_SplitInstructions splitN_customParser(
        ZL_SplitState* s,
        const ZL_Input* in) noexcept
{
    assert(ZL_SplitState_getOpaquePtr(s) == (void const*)&kTag);
    assert(*(unsigned const*)ZL_SplitState_getOpaquePtr(s) == kTag);

    assert(in != nullptr);
    assert(ZL_Input_type(in) == ZL_Type_serial);
    size_t const srcSize = ZL_Input_numElts(in);
    // Let's arbitrarily split input into 4 segments
    size_t* const segSizes =
            (size_t*)ZL_SplitState_malloc(s, 4 * sizeof(size_t));
    if (segSizes == nullptr)
        return (ZL_SplitInstructions){ nullptr, 0 };
    segSizes[0] = srcSize / 5;
    segSizes[1] = srcSize / 4;
    segSizes[2] = srcSize / 3;
    segSizes[3] = srcSize - (segSizes[0] + segSizes[1] + segSizes[2]);
    return (ZL_SplitInstructions){ segSizes, 4 };
}

static ZL_SplitInstructions failingParser(ZL_SplitState* s, const ZL_Input* in)
{
    (void)in;
    (void)s;
    // failing on purpose, for tests
    return (ZL_SplitInstructions){ nullptr, 0 };
}

// This parser incorrectly provides an invalid size
static ZL_SplitInstructions splitN_wrongParser(
        ZL_SplitState* s,
        const ZL_Input* in) noexcept
{
    assert(in != nullptr);
    assert(ZL_Input_type(in) == ZL_Type_serial);
    size_t const srcSize = ZL_Input_numElts(in);
    // Let's arbitrarily split input into 3 segments,
    // sum length of these segments is intentionally too short.
    size_t* const segSizes =
            (size_t*)ZL_SplitState_malloc(s, 3 * sizeof(size_t));
    if (segSizes == nullptr)
        return (ZL_SplitInstructions){ nullptr, 0 };
    segSizes[0] = srcSize / 5;
    segSizes[1] = srcSize / 4;
    segSizes[2] = srcSize / 3;
    // Condition for this parser to be wrong
    assert((segSizes[0] + segSizes[1] + segSizes[2]) < srcSize);
    return (ZL_SplitInstructions){ segSizes, 3 };
}

/* ------   create custom graph   -------- */

// Note : This graph requires input Stream to be at least 50+ bytes long
static ZL_GraphID splitGraph_byParam(
        ZL_Compressor* cgraph,
        const size_t segmentSizes[],
        size_t nbSegments)
{
    printf("running splitGraph_byParam() \n");
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_NodeID splitByParams = ZL_Compressor_registerSplitNode_withParams(
            cgraph, ZL_Type_serial, segmentSizes, nbSegments);

    // Note : the operation generates multiple outputs (defined by parameters)
    //        nevertheless all these outputs share the same Outcome,
    //        i.e. have the same GraphID as successor.
    return ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, splitByParams, ZL_GRAPH_STORE);
}

/* This graph requires input to be at least 48+ bytes long */
static ZL_GraphID splitGraph_byParam_16_32_0(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph_byParam_16_32_0() \n");
    return splitGraph_byParam(cgraph, (const size_t[]){ 16, 32, 0 }, 3);
}

static ZL_GraphID splitGraph_byParam_0_0(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph_byParam_0_0() \n");
    /* This parameter creates 2 outputs, the first one is empty,
     * the second contains all input's content.
     * It's compatible with empty input. */
    return splitGraph_byParam(cgraph, (const size_t[]){ 0, 0 }, 2);
}

// This graph will necessarily fail at runtime
// because it received no splitting instructions (no parameter)
static ZL_GraphID splitGraph_noInstructions(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph with no Instructions \n");
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    return ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, { ZL_PrivateStandardNodeID_splitN }, ZL_GRAPH_STORE);
}

// Note : This graph requires input size to be >= sum(segmentSizes[]),
// assuming the last value is `0`,
// otherwise, it must be exactly == sum(segmentsSizes[]).
// Also : nbSegments must be <= maxNbSuccessors
static ZL_GraphID graph_splitByParam(
        ZL_Compressor* cgraph,
        const size_t segmentsSizes[],
        size_t nbSegments)
{
    printf("running graph_splitByParam() (%zu segments) \n", nbSegments);

    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_GraphID successors[] = {
        ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE,
        ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE,
    };
    size_t const maxNbSuccessors = sizeof(successors) / sizeof(*successors);
    assert(nbSegments <= maxNbSuccessors);
    (void)maxNbSuccessors;

    return ZL_Compressor_registerSplitGraph(
            cgraph, ZL_Type_serial, segmentsSizes, successors, nbSegments);
}

/* This graph requires input to be at least 4 bytes long */
static ZL_GraphID graph_splitByParam_2_2_0(ZL_Compressor* cgraph) noexcept
{
    printf("running graph_splitByParam_2_2_0() \n");
    return graph_splitByParam(cgraph, (const size_t[]){ 2, 2, 0 }, 3);
}

/* This graph is invalid : no segment defined */
static ZL_GraphID graph_splitByParam_NULL(ZL_Compressor* cgraph) noexcept
{
    printf("running graph_splitByParam_NULL() \n");
    return graph_splitByParam(cgraph, nullptr, 0);
}

static ZL_GraphID splitGraph_byExtParser(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph_byExtParser \n");
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_NodeID splitByExtParser =
            ZL_Compressor_registerSplitNode_withParser(
                    cgraph, ZL_Type_serial, splitN_customParser, &kTag);

    // Note : while this split operation will define a variable nb of outputs,
    //        all these outputs share the same Outcome,
    //        i.e. have the same GraphID as successor.
    return ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, splitByExtParser, ZL_GRAPH_STORE);
}

static ZL_GraphID splitGraph_withFailingParser(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph_withFailingParser \n");
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_NodeID splitByExtParser =
            ZL_Compressor_registerSplitNode_withParser(
                    cgraph, ZL_Type_serial, failingParser, nullptr);

    return ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, splitByExtParser, ZL_GRAPH_STORE);
}

static ZL_GraphID splitGraph_withWrongParser(ZL_Compressor* cgraph) noexcept
{
    printf("running splitGraph_withWrongParser \n");
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_NodeID splitByExtParser =
            ZL_Compressor_registerSplitNode_withParser(
                    cgraph, ZL_Type_serial, splitN_wrongParser, nullptr);

    return ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, splitByExtParser, ZL_GRAPH_STORE);
}

/* ------   compress, using provided graph function   -------- */

static size_t compress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        ZL_GraphFn graphf)
{
    ZL_REQUIRE_GE(dstCapacity, ZL_compressBound(srcSize));

    ZL_Report const r =
            ZL_compress_usingGraphFn(dst, dstCapacity, src, srcSize, graphf);
    EXPECT_EQ(ZL_isError(r), 0) << "compression failed \n";

    return ZL_validResult(r);
}

/* ------   decompress   -------- */

static size_t
decompress(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    // Check buffer size
    ZL_Report const dr = ZL_getDecompressedSize(src, srcSize);
    ZL_REQUIRE(!ZL_isError(dr));
    size_t const dstSize = ZL_validResult(dr);
    ZL_REQUIRE_GE(dstCapacity, dstSize);

    // Create decompression state, to store the custom decoder(s)
    ZL_DCtx* const dctx = ZL_DCtx_create();
    ZL_REQUIRE_NN(dctx);

    // Decompress, using custom decoder(s)
    ZL_Report const r =
            ZL_DCtx_decompress(dctx, dst, dstCapacity, src, srcSize);
    EXPECT_EQ(ZL_isError(r), 0) << "decompression failed \n";

    ZL_DCtx_free(dctx);
    return ZL_validResult(r);
}

/* ------   test internals   ------ */

static int roundTripTest(ZL_GraphFn graphf, const char* name, size_t arraySize)
{
    printf("\n=========================== \n");
    printf(" %s \n", name);
    printf("--------------------------- \n");
    // Generate test input
#define NB_INTS 78
    int input[NB_INTS] = {
        0,
    };
    assert(arraySize <= NB_INTS);
    for (size_t i = 0; i < arraySize; i++)
        input[i] = (int)i;
    size_t inputSize = arraySize * sizeof(int);

#define COMPRESSED_BOUND ZL_COMPRESSBOUND(sizeof(input))
    char compressed[COMPRESSED_BOUND] = { 0 };
    size_t cBoundSize                 = ZL_compressBound(inputSize);

    size_t const compressedSize =
            compress(compressed, cBoundSize, input, inputSize, graphf);
    printf("compressed %zu input bytes into %zu compressed bytes \n",
           inputSize,
           compressedSize);

    int decompressed[NB_INTS] = { 2, 28 };

    size_t const decompressedSize = decompress(
            decompressed, sizeof(decompressed), compressed, compressedSize);
    printf("decompressed %zu input bytes into %zu original bytes \n",
           compressedSize,
           decompressedSize);

    // round-trip check
    EXPECT_EQ(decompressedSize, inputSize)
            << "Error : decompressed size != original size \n";
    EXPECT_EQ(memcmp(input, decompressed, inputSize), 0)
            << "Error : decompressed content differs from original (corruption issue) !!!  \n";

    printf("round-trip success \n");
    return 0;
}

static int cFailTest(ZL_GraphFn graphf, const char* testName)
{
    printf("\n=========================== \n");
    printf(" %s \n", testName);
    printf("--------------------------- \n");
    // Generate test input => too short, will fail
    char input[40];
    for (int i = 0; i < 40; i++)
        input[i] = (char)i;

#define COMPRESSED_BOUND ZL_COMPRESSBOUND(sizeof(input))
    char compressed[COMPRESSED_BOUND] = { 0 };

    ZL_Report const r = ZL_compress_usingGraphFn(
            compressed, COMPRESSED_BOUND, input, sizeof(input), graphf);
    EXPECT_EQ(ZL_isError(r), 1) << "compression should have failed \n";

    printf("Compression failure observed as expected : %s \n",
           ZL_ErrorCode_toString(r._code));
    return 0;
}

/* ------   exposed tests   ------ */

static int test_splitGraph(void)
{
    return roundTripTest(
            splitGraph_byParam_16_32_0,
            "simple splitN-by-param round trip",
            NB_INTS);
}

static int test_failSplitNByParam(void)
{
    return cFailTest(
            splitGraph_byParam_16_32_0,
            "splitN-by-param on too small input => failure expected");
}

static int test_failSplitNoInstructions(void)
{
    return cFailTest(
            splitGraph_noInstructions,
            "split operation with no received instruction => failure expected");
}

static int test_failingParser(void)
{
    return cFailTest(
            splitGraph_withFailingParser,
            "split's parser will return failure {NULL, 0} => failure expected");
}

static int test_wrongParser(void)
{
    return cFailTest(
            splitGraph_withWrongParser,
            "split's parser will not map the entire input => failure expected");
}

static int test_splitByExtParser(void)
{
    return roundTripTest(
            splitGraph_byExtParser,
            "splitN using custom External parser",
            NB_INTS);
}

static int test_splitByExtParser_empty(void)
{
    return roundTripTest(
            splitGraph_byExtParser,
            "split an empty input using custom External parser",
            0);
}

static int test_splitGraph_0_0(void)
{
    return roundTripTest(
            splitGraph_byParam_0_0, "splitN-by-param {0 , 0}", NB_INTS);
}

static int test_split_0_0_empty(void)
{
    return roundTripTest(
            splitGraph_byParam_0_0,
            "split an empty input with param {0,0} (expected success)",
            0);
}

static int test_graph_splitByParam_2_2_0(void)
{
    return roundTripTest(
            graph_splitByParam_2_2_0,
            "createGraph_splitByParam{2, 2, 0}",
            NB_INTS);
}

static int test_graph_splitByParam_NULL(void)
{
    return cFailTest(
            graph_splitByParam_NULL,
            "split operation receives NULL array => failure expected");
}

TEST(SplitNTest, simpleSplitByParamTest)
{
    test_splitGraph();
}

TEST(SplitNTest, splitByParam_0_0)
{
    test_splitGraph_0_0();
}

TEST(SplitNTest, splitEmptyInput_withParam_0_0)
{
    test_split_0_0_empty();
}

TEST(SplitNTest, failSplitNByParam)
{
    test_failSplitNByParam();
}

TEST(SplitNTest, failSplitNoInstructions)
{
    test_failSplitNoInstructions();
}

TEST(SplitNTest, graph_splitByParam_2_2_0)
{
    test_graph_splitByParam_2_2_0();
}

TEST(SplitNTest, graph_splitByParam_NULL)
{
    test_graph_splitByParam_NULL();
}

TEST(SplitNTest, failingParser)
{
    test_failingParser();
}

TEST(SplitNTest, wrongParser)
{
    test_wrongParser();
}

TEST(SplitNTest, splitByExtParser)
{
    test_splitByExtParser();
}

TEST(SplitNTest, splitByExtParser_empty)
{
    test_splitByExtParser_empty();
}

} // namespace
