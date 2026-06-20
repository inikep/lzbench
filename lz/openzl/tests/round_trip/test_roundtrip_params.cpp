// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

// Test transforms with graph-time parameters

// standard C
#include <cstdio>  // printf
#include <cstdlib> // exit, rand
#include <cstring> // memcpy

// Zstrong
#include "openzl/common/debug.h" // ZL_REQUIRE
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h" // ZL_Compressor_registerSplitEncoder, ZL_Compressor_registerStaticGraph_fromNode
#include "openzl/zl_decompress.h" // ZL_decompress
#include "openzl/zl_selector.h"   // ZL_TypedEncoderDesc

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

/* ------   create custom typed selector   -------- */

enum { nb_local_int_params = 2 };
enum { paramId1 = 101, paramId2 = 202 };
enum { paramValue1 = 11, paramValue2 = 22 };
#define FLATBUFFER_STRING "string_parameter_from_stack"
enum { copyParamId1 = 901, nb_local_copy_params = 1 };
#define REFERENCED_STRING "stable_string_parameter"
enum { refParamId1 = 799, nb_local_ref_params = 1 };
#define OWNED_STRING "owned_string_parameter"
enum { clevel = 1, dlevel = 2 }; // Global parameters

static ZL_GraphID readParamsSelector(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* cfns,
        size_t nbCfns) noexcept
{
    ZL_REQUIRE_EQ(nbCfns, 0);
    (void)cfns;
    printf("Running readParams Selector: \n");
    ZL_Type const st = ZL_Input_type(inputStream);
    EXPECT_EQ(st, ZL_Type_serial);

    // Test Global parameters
    int clevelVar = ZL_Selector_getCParam(selCtx, ZL_CParam_compressionLevel);
    printf("test : query compression level : %i \n", clevel);
    EXPECT_EQ(clevelVar, clevel);
    int dlevelVar = ZL_Selector_getCParam(selCtx, ZL_CParam_decompressionLevel);
    printf("test : query decompression level : %i \n", dlevel);
    EXPECT_EQ(dlevelVar, dlevel);

    // Test Local Int parameters
    {
        ZL_LocalIntParams const lip = ZL_Selector_getLocalIntParams(selCtx);
        size_t const nbIntParams    = lip.nbIntParams;
        printf("test : query local int parameters (%zu) \n", nbIntParams);
        EXPECT_EQ((int)nbIntParams, nb_local_int_params);
        int const param0 = lip.intParams[0].paramId;
        int const value0 = lip.intParams[0].paramValue;
        printf("param0 (%i)  => %i (value0) \n", param0, value0);
        EXPECT_EQ(param0, paramId1);
        EXPECT_EQ(value0, paramValue1);
        int const param1 = lip.intParams[1].paramId;
        int const value1 = lip.intParams[1].paramValue;
        printf("param1 (%i)  => %i (value1) \n", param1, value1);
        EXPECT_EQ(param1, paramId2);
        EXPECT_EQ(value1, paramValue2);
    }

    // Test Local Copy parameters
    {
        ZL_CopyParam const cp =
                ZL_Selector_getLocalCopyParam(selCtx, copyParamId1);
        printf("test : query local gen parameters \n");
        int const paramId      = cp.paramId;
        const char* paramPtr   = (const char*)cp.paramPtr;
        size_t const paramSize = cp.paramSize;
        printf("paramId=%i  => value=%s \n", paramId, paramPtr);
        EXPECT_EQ(paramId, copyParamId1);
        EXPECT_STREQ(paramPtr, FLATBUFFER_STRING);
        EXPECT_EQ((int)paramSize, (int)strlen(paramPtr) + 1);
    }

    // Test Local Ref parameters
    {
        ZL_RefParam const rp = ZL_Selector_getLocalParam(selCtx, refParamId1);
        printf("test : query local ref parameters \n");
        int const paramId          = rp.paramId;
        const char* const paramPtr = (const char*)rp.paramRef;
        printf("paramId=%i  => value=%s \n", paramId, paramPtr);
        EXPECT_EQ(paramId, refParamId1);
        EXPECT_STREQ(paramPtr, REFERENCED_STRING);
        EXPECT_EQ(rp.paramSize, strlen(paramPtr) + 1);
    }

    // Test Ref parameter passed as copy parameter
    {
        ZL_RefParam const rp = ZL_Selector_getLocalParam(selCtx, copyParamId1);
        printf("test : query copy parameter as a ref parameter \n");
        int const paramId          = rp.paramId;
        const char* const paramPtr = (const char*)rp.paramRef;
        printf("paramId=%i  => value=%s \n", paramId, paramPtr);
        EXPECT_EQ(paramId, copyParamId1);
        EXPECT_STREQ(paramPtr, FLATBUFFER_STRING);
        EXPECT_EQ(rp.paramSize, strlen(paramPtr) + 1);
    }

    return ZL_GRAPH_STORE;
}

/* ------   create custom graph   -------- */

// This graph function follows the ZL_GraphFn definition.
static ZL_GraphID typedSelectorWithParamsGraph(ZL_Compressor* cgraph) noexcept
{
    printf("running typedSelectorWithParamsGraph() \n");

    // Global parameters
    ZL_REQUIRE_SUCCESS(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION));
    ZL_REQUIRE_SUCCESS(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_compressionLevel, clevel));
    ZL_REQUIRE_SUCCESS(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_decompressionLevel, dlevel));

    // Local Integer parameters
    ZL_IntParam ip[nb_local_int_params] = { { paramId1, paramValue1 },
                                            { paramId2, paramValue2 } };
    ZL_LocalIntParams lip               = { ip, nb_local_int_params };

    // Copy parameters (will be copied into @cgraph)
    char stackString[sizeof(FLATBUFFER_STRING)] = FLATBUFFER_STRING;
    ZL_CopyParam cp[nb_local_copy_params]       = {
        { copyParamId1, stackString, sizeof(stackString) }
    };
    ZL_LocalCopyParams lcp = { cp, nb_local_copy_params };

    // This string will be referenced, so it must remain stable in memory
    static const char stableString[sizeof(REFERENCED_STRING)] =
            REFERENCED_STRING;
    ZL_RefParam rp[nb_local_ref_params] = {
        { refParamId1, stableString, sizeof(REFERENCED_STRING) }
    };
    ZL_LocalRefParams lrp = { rp, nb_local_ref_params };

    // Assemble all params
    ZL_LocalParams lp       = { lip, lcp, lrp };
    ZL_SelectorDesc tselect = {
        .selector_f     = readParamsSelector,
        .inStreamType   = ZL_Type_serial,
        .nbCustomGraphs = 0,
        .localParams    = lp,
    };

    // Graph is just a selector node
    ZL_GraphID const fn = ZL_Compressor_registerSelectorGraph(cgraph, &tselect);
    memset(stackString, 0, sizeof(stackString));
    // Note : it's necessary to print stackString address,
    // otherwise, the compiler will simplify it and make it a constant
    // thus missing the memset() operation, which tests that string data was
    // effectively copied.
    printf("stackString = %p \n", (const void*)stackString);
    return fn;
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

    // Create a single decompression state, to store the custom decoder(s)
    // The decompression state will be re-employed
    static ZL_DCtx* const dctx = ZL_DCtx_create();
    ZL_REQUIRE_NN(dctx);

    // Decompress, using custom decoder(s)
    ZL_Report const r =
            ZL_DCtx_decompress(dctx, dst, dstCapacity, src, srcSize);
    EXPECT_EQ(ZL_isError(r), 0) << "decompression failed \n";

    return ZL_validResult(r);
}

/* ------   round trip test   ------ */

static int roundTripTest(ZL_GraphFn graphf, const char* name)
{
    printf("\n=========================== \n");
    printf(" Node with parameters : %s \n", name);
    printf("--------------------------- \n");
    // Generate test input
#define NB_CHAR ((size_t)77)
    char input[NB_CHAR];
    for (size_t i = 0; i < NB_CHAR; i++)
        input[i] = (char)i;

#define COMPRESSED_BOUND ZL_COMPRESSBOUND(sizeof(input))
    char compressed[COMPRESSED_BOUND] = { 0 };

    size_t const compressedSize = compress(
            compressed, COMPRESSED_BOUND, input, sizeof(input), graphf);
    printf("compressed %zu input bytes into %zu compressed bytes \n",
           sizeof(input),
           compressedSize);

    char decompressed[NB_CHAR] = { 2 };

    size_t const decompressedSize = decompress(
            decompressed, sizeof(decompressed), compressed, compressedSize);
    printf("decompressed %zu input bytes into %zu original bytes \n",
           compressedSize,
           decompressedSize);

    // round-trip check
    EXPECT_EQ(decompressedSize, sizeof(input))
            << "Error : decompressed size != original size \n";
    EXPECT_EQ(memcmp(input, decompressed, sizeof(input)), 0)
            << "Error : decompressed content differs from original (corruption issue) !!!  \n";

    printf("round-trip success \n");
    return 0;
}

TEST(ParamsTest, typedSelector)
{
    roundTripTest(
            typedSelectorWithParamsGraph,
            "Typed Selector requests global and local parameters");
}

} // namespace
