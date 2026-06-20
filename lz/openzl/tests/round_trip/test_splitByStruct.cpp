// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

// standard C
#include <stdio.h> // printf

// Zstrong
#include "openzl/common/debug.h" // ZL_REQUIRE
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

// None needed

/* ------   create custom graph   -------- */

// Note : This graph requires input size to be
// an exact multiple of structSize = sum(fieldSizes).
// Also, limited to nbFields <= maxNbSuccessors.
static ZL_GraphID graph_splitByStruct(
        ZL_Compressor* cgraph,
        const size_t fieldSizes[],
        size_t nbFields)
{
    printf("running graph_splitByStruct() (%zu fields) \n", nbFields);

    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));

    const ZL_GraphID successors[] = {
        ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE,
        ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE, ZL_GRAPH_STORE,
    };
    size_t const maxNbSuccessors = sizeof(successors) / sizeof(*successors);
    assert(nbFields <= maxNbSuccessors);
    (void)maxNbSuccessors;

    return ZL_Compressor_registerSplitByStructGraph(
            cgraph, fieldSizes, successors, nbFields);
}

/* structure size : 12 */
static ZL_GraphID splitGraph_struct_4_4_4(ZL_Compressor* cgraph) noexcept
{
    return graph_splitByStruct(cgraph, (const size_t[]){ 4, 4, 4 }, 3);
}

/* structure size : 33 */
static ZL_GraphID splitGraph_struct_8_1_4_2_3_15(ZL_Compressor* cgraph) noexcept
{
    return graph_splitByStruct(
            cgraph, (const size_t[]){ 8, 1, 4, 2, 3, 15 }, 6);
}

static ZL_GraphID splitGraph_struct_0_0(ZL_Compressor* cgraph) noexcept
{
    return graph_splitByStruct(cgraph, (const size_t[]){ 0, 0 }, 2);
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
#define NB_INTS 150
    int input[NB_INTS];
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

TEST(SplitByStruct, test_splitByStruct_12)
{
    roundTripTest(
            splitGraph_struct_4_4_4,
            "splitByStruct test, structure is 3 fields of 4 bytes",
            30);
}

TEST(SplitByStruct, test_splitByStruct_33)
{
    roundTripTest(
            splitGraph_struct_8_1_4_2_3_15,
            "splitByStruct test, structure size is 33 bytes",
            33 * 4);
}

TEST(SplitByStruct, test_splitByStruct_invalidInputSize)
{
    cFailTest(
            splitGraph_struct_8_1_4_2_3_15,
            "splitByStruct on input which is not a multiple of structure size => failure expected");
}

TEST(SplitByStruct, test_splitByStruct_0)
{
    cFailTest(
            splitGraph_struct_0_0,
            "splitByStruct with a structure of size 0 => failure expected");
}

} // namespace
