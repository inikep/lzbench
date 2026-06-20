// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "openzl/codecs/zl_generic.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_graph_api.h"
#include "openzl/zl_opaque_types.h"
#include "openzl/zl_version.h"

#include "../decode_ycocg_binding.h"
#include "../encode_ycocg_binding.h"

static ZL_Compressor* create_trivial_ycocg_compressor(void)
{
    ZL_Compressor* zlc = ZL_Compressor_create();
    assert(zlc != NULL);

    // Register the custom codec
    ZL_NodeID ycocg_node = ZL_Compressor_registerTypedEncoder(
            zlc, &YCOCG_encoder_registration_structure);
    assert(ZL_NodeID_isValid(ycocg_node));

    // Use it to create a (trivial) custom Graph
    ZL_GraphID ycocg_graph = ZL_Compressor_registerStaticGraph_fromNode(
            zlc,
            ycocg_node,
            ZL_GRAPHLIST(
                    ZL_GRAPH_COMPRESS_GENERIC,
                    ZL_GRAPH_COMPRESS_GENERIC,
                    ZL_GRAPH_COMPRESS_GENERIC));
    assert(ZL_GraphID_isValid(ycocg_graph));

    // The last registered Graph is the default starting Graph,
    // so the Compressor is set
    (void)ycocg_graph;
    return zlc;
}

static void* create_input(size_t size)
{
    void* buffer = malloc(size);
    assert(buffer != NULL);
    for (size_t n = 0; n < size; n++) {
        ((char*)buffer)[n] = (char)rand();
    }
    return buffer;
}

static void test_roundtrip(void)
{
    // Compression requires a ZL_CCtx state
    // This is the place where compression parameters are stored
    // A ZL_Compressor is considered a compression parameter
    ZL_CCtx* cctx = ZL_CCtx_create();
    assert(cctx != NULL);

    // Create the ZL_Compressor* object,
    // this one registers a custom node and creates a trivial graph
    printf("registration and insertion of ycocg_node successful \n");
    ZL_Compressor* ycocg = create_trivial_ycocg_compressor();

    // Generate input & output buffers
    size_t nbPixels  = (size_t)rand() % 999999;
    size_t inputSize = nbPixels * 3;
    printf("generating input (%zu bytes)\n", inputSize);
    void* input = create_input(inputSize);
    assert(input != NULL);
    size_t dstCapacity = ZL_compressBound(inputSize);
    void* compressed   = malloc(dstCapacity);
    assert(compressed != NULL);

    // Set compression parameters
    ZL_Report rcs = ZL_CCtx_refCompressor(cctx, ycocg);
    assert(!ZL_isError(rcs));
    // @note: currently, it's mandatory to explicitly set a format version
    ZL_Report sffs = ZL_CCtx_setParameter(
            cctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION);
    assert(!ZL_isError(sffs));

    printf("starting compression: ");
    fflush(NULL);
    ZL_Report cReport =
            ZL_CCtx_compress(cctx, compressed, dstCapacity, input, inputSize);

    // Results are encapsulted into a ZL_Report sum type (cSize | error)
    // which must be controlled and then unwrapped if valid.
    if (ZL_isError(cReport)) {
        const char* s = ZL_CCtx_getErrorContextString(cctx, cReport);
        printf("error: %s \n", s);
        exit(1);
    }
    size_t cSize = ZL_validResult(cReport);
    assert(cSize <= dstCapacity);
    printf("completed successfully \n");

    printf("starting decompression\n");
    void* decompressed = malloc(inputSize);
    assert(decompressed != NULL);

    // Similar to Compression, Decompression requires a ZL_DCtx state
    ZL_DCtx* dctx = ZL_DCtx_create();
    assert(dctx != NULL);

    // Since the graph uses a custom codec,
    // a corresponding custom decoder must be declared.
    // It doesn't matter if or where it's employed in the graph.
    ZL_Report rtds = ZL_DCtx_registerTypedDecoder(
            dctx, &YCOCG_decoder_registration_structure);
    assert(!ZL_isError(rtds));

    ZL_Report ds = ZL_DCtx_decompress(
            dctx, decompressed, inputSize, compressed, cSize);
    if (ZL_isError(ds)) {
        const char* s = ZL_DCtx_getErrorContextString(dctx, ds);
        printf("error: %s \n", s);
        exit(1);
    }
    size_t decompressedSize = ZL_validResult(ds);
    // ensure input and decompressed are identical
    assert(decompressedSize == inputSize);
    assert(!memcmp(decompressed, input, inputSize));
    printf("roundtrip completed and checked successfully\n");

    ZL_DCtx_free(dctx);
    free(decompressed);
    free(compressed);
    free(input);
    ZL_Compressor_free(ycocg);
    ZL_CCtx_free(cctx);
}

int main()
{
    test_roundtrip();
    return 0;
}
