// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \zs2_pipeline.c
 *
 * Trivial example of a pipeline graph
 *
 * The example is a simple series of increasing numbers in binary format
 * (32-bit). The series is interpreted as Little Endian 32-bit numbers, then
 * delta-transformed, then the result of this transform is compressed.
 * The produced executable will only work on files which have a size that is a
 * multiple of 4.
 */

#undef NDEBUG       // We always want assertions
#include <assert.h> // note : this is user code
#include <stdio.h>  // printf
#include <stdlib.h> // exit
#include <string.h> // memcmp

#include "openzl/common/logging.h" // ZL_REQUIRE
#include "openzl/zl_compress.h"    // ZL_compress_usingGraphFn
#include "openzl/zl_compressor.h" // ZL_Compressor_registerStaticGraph_fromPipelineNodes1o
#include "openzl/zl_decompress.h" // ZL_decompress

#include "tools/fileio/fileio.h" // FIO_*

/* ------   create custom transforms   -------- */

// None in this example

/* ------   create custom graph for array of 32-bit integers  -------- */

// This simple graph is just a pipeline of 1->1 transforms.
// The graph function follows the ZL_GraphFn definition
// to be passed as parameter to ZL_compress_usingGraphFn()
static ZL_GraphID multiStages_pipeline(ZL_Compressor* cgraph)
{
    ZL_REQUIRE(!ZL_isError(ZL_Compressor_setParameter(
            cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));
    return ZL_Compressor_registerStaticGraph_fromPipelineNodes1o(
            cgraph,
            ZL_NODELIST(
                    ZL_NODE_INTERPRET_AS_LE32,
                    ZL_NODE_DELTA_INT,
                    ZL_NODE_ZIGZAG,
                    ZL_NODE_CONVERT_SERIAL_TO_TOKEN4),
            ZL_GRAPH_FIELD_LZ);
}

/* ------   compress, using the custom graph   -------- */

// This optional layer of ZL_GraphFn function
// is just there to add global parameters
// on top of multiStages_pipeline().
static ZL_GraphID graph_and_parameters(ZL_Compressor* cgraph)
{
    ZL_GraphID const gid = multiStages_pipeline(cgraph);

    // Place to setup Global parameters, passed as part of @cgraph
    // Note: it's preferable to setup global parameters after the graph function
    // so that it overrides any global parameter that *might* have been setup as
    // part of the graph function (which is discouraged practice).
    ZL_Report const r =
            ZL_Compressor_setParameter(cgraph, ZL_CParam_compressionLevel, 3);
    assert(!ZL_isError(r));

    return gid;
}

static size_t
compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    assert(dstCapacity >= ZL_compressBound(srcSize));

    ZL_Report const r = ZL_compress_usingGraphFn(
            dst, dstCapacity, src, srcSize, graph_and_parameters);
    assert(!ZL_isError(r));

    return ZL_validResult(r);
}

/* ------   decompress   -------- */

// Register custom transforms, and decode
static size_t
decompress(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    // Check buffer size
    ZL_Report const dr = ZL_getDecompressedSize(src, srcSize);
    assert(!ZL_isError(dr));
    size_t const dstSize = ZL_validResult(dr);
    assert(dstCapacity >= dstSize);

    // Note : no need to register custom transforms in this case

    // Decompress, using only standard decoders
    ZL_Report const r = ZL_decompress(dst, dstCapacity, src, srcSize);
    assert(!ZL_isError(r));
    return ZL_validResult(r);
}

/* ------   ======================   ------ */
/* ------   simple round trip test   ------ */
/* ------   ======================   ------ */

// Note: in order to work, this command line program must receive as input
// an array of 32-bit integers, in Little Endian format.

static void usage(char const* program)
{
    fprintf(stderr, "USAGE: %s INPUT [OUTPUT]\n", program);
}

int main(int argc, char const** argv)
{
    // Reduce log level to warnings and above
    ZL_g_logLevel = ZL_LOG_LVL_WARN;

    // Extremely simple usage / help.
    if (argc < 2 || argc > 3 || !strcmp(argv[1], "--help")
        || !strcmp(argv[1], "-h")) {
        usage(argv[0]);
        return 0;
    }

    char const* const inputFile  = argv[1];
    char const* const outputFile = argc == 3 ? argv[2] : NULL;

    size_t const inputSize = FIO_sizeof_file(inputFile);

    void* const input = malloc(inputSize);
    ZL_REQUIRE_NN(input);

    {
        FILE* const f = fopen(inputFile, "r");
        ZL_REQUIRE_EQ(inputSize, fread(input, 1, inputSize, f));
        fclose(f);
    }

    size_t const compressBound = ZL_compressBound(inputSize);
    void* const compressed     = malloc(compressBound);
    ZL_REQUIRE_NN(compressed);

    size_t const compressedSize =
            compress(compressed, compressBound, input, inputSize);
    fprintf(stderr,
            "compressed %zu input bytes into %zu compressed bytes \n",
            inputSize,
            compressedSize);

    void* const decompressed = malloc(inputSize);
    ZL_REQUIRE_NN(decompressed);

    size_t const decompressedSize =
            decompress(decompressed, inputSize, compressed, compressedSize);
    fprintf(stderr,
            "decompressed %zu input bytes into %zu original bytes \n",
            compressedSize,
            decompressedSize);

    // round-trip check
    if (decompressedSize != inputSize) {
        fprintf(stderr,
                "Error : decompressed size (%zu) != original size (%zu) \n",
                decompressedSize,
                inputSize);
        return 1;
    }
    if (memcmp(input, decompressed, inputSize) != 0) {
        fprintf(stderr,
                "Error : decompressed content differs from original (corruption issue) !!!  \n");
        return 1;
    }

    if (outputFile != NULL) {
        FILE* f = fopen(outputFile, "w");
        ZL_REQUIRE_EQ(compressedSize, fwrite(compressed, 1, compressedSize, f));
        fclose(f);
    }

    fprintf(stderr, "round-trip success \n");

    free(input);
    free(compressed);
    free(decompressed);

    return 0;
}
