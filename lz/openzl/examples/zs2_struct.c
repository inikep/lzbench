// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \zs2_struct.c
 *
 * Example of compression for an array of Structures
 *
 * This example is illustrates by the SAO format within the Silesia corpus,
 * which is array of 28 bytes structures,
 * preceded by a 28 bytes header.
 * The generated example will only work on files which size is a multiple of 28,
 * though it's specifically designed to compress the SAO format,
 * and will work poorly in other cases.
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

/* ------   create custom graph specialized for SAO format   -------- */

// Goal of this graph :
// stronger compression ratio than cmix on sao (3726989)
// at fastest compression speed possible
static ZL_GraphID sao_graph_v1(ZL_Compressor* cgraph)
{
    if (ZL_isError(ZL_Compressor_setParameter(
                cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION))) {
        abort();
    }
    /* The SAO format consists of a header,
     * which is 28 bytes for the dirSilesia/sao sample specifically,
     * followed by an array of structures, each one describing a star.
     *
     * For the record, here is the Header format (it's currently ignored):
     *
     * Integer*4 STAR0=0   Subtract from star number to get sequence number
     * Integer*4 STAR1=1   First star number in file
     * Integer*4 STARN=258996  Number of stars in file (pos 8)
     * Integer*4 STNUM=1   0 if no star i.d. numbers are present
     *                     1 if star i.d. numbers are in catalog file
     *                     2 if star i.d. numbers are  in file
     * Logical*4 MPROP=t   True if proper motion is included
     *                     False if no proper motion is included
     * Integer*4 NMAG=1    Number of magnitudes present
     * Integer*4 NBENT=32  Number of bytes per star entry
     * Total : 28 bytes
     */
    size_t const headerSize = 28;

    /* Star record : 28 bytes for the dirSilesia/sao sample specifically
     * Real*4 XNO       Catalog number of star (not present, since stnum==0)
     * Real*8 SRA0      B1950 Right Ascension (radians)
     * Real*8 SDEC0     B1950 Declination (radians)
     * Character*2 IS   Spectral type (2 characters)
     * Integer*2 MAG    V Magnitude * 100
     * Real*4 XRPM      R.A. proper motion (radians per year)
     * Real*4 XDPM      Dec. proper motion (radians per year)
     */
    ZL_GraphID sra0 = ZL_Compressor_registerStaticGraph_fromPipelineNodes1o(
            cgraph,
            ZL_NODELIST(ZL_NODE_INTERPRET_TOKEN_AS_LE, ZL_NODE_DELTA_INT),
            ZL_GRAPH_FIELD_LZ);
    ZL_GraphID sdec0 = ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, ZL_NODE_TRANSPOSE_SPLIT, ZL_GRAPH_ZSTD);
    ZL_GraphID token_compress = ZL_Compressor_registerTokenizeGraph(
            cgraph,
            ZL_Type_struct,
            false,
            ZL_GRAPH_FIELD_LZ,
            ZL_GRAPH_FIELD_LZ);
    ZL_GraphID num_huffman = ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph,
            ZL_NODE_INTERPRET_TOKEN_AS_LE,
            ZL_Compressor_registerTokenizeGraph(
                    cgraph,
                    ZL_Type_numeric,
                    false,
                    ZL_GRAPH_HUFFMAN,
                    ZL_GRAPH_HUFFMAN));
    ZL_GraphID is             = num_huffman;
    ZL_GraphID mag            = num_huffman;
    ZL_GraphID xrpm           = token_compress;
    ZL_GraphID xdpm           = token_compress;
    ZL_GraphID splitStructure = ZL_Compressor_registerSplitByStructGraph(
            cgraph,
            (const size_t[]){ 8, 8, 2, 2, 4, 4 },
            (const ZL_GraphID[]){ sra0, sdec0, is, mag, xrpm, xdpm },
            6);

    return ZL_Compressor_registerSplitGraph(
            cgraph,
            ZL_Type_serial,
            (const size_t[]){ headerSize, 0 },
            (const ZL_GraphID[]){ ZL_GRAPH_STORE, splitStructure },
            2);
}

/* ------   compress, using the custom graph   -------- */

// The rest of this file is essentially the same as zs2_pipeline.c

// This optional layer of ZL_GraphFn function
// is just there to add global parameters
// on top of sao_graph_v1().
static ZL_GraphID graph_and_parameters(ZL_Compressor* cgraph)
{
    ZL_GraphID const gid = sao_graph_v1(cgraph);

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
