// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>

#include "openzl/common/assertion.h"
#include "openzl/common/logging.h"
#include "openzl/zl_data.h"
#include "openzl/zl_reflection.h"
#include "tools/fileio/fileio.h"
#include "tools/streamdump/stream_dump2.h"

typedef struct {
    const char* srcFileName;
    const char* dstFilePrefix;

} stream_dump_args_t;

static void usage(int argc, char* argv[])
{
    (void)argc;
    const char* progname = argv[0];
    ZL_RLOG(ALWAYS, "%s:\n", progname);
    ZL_RLOG(ALWAYS, "    Decompress zstrong files and dump stream contents.\n");
    ZL_RLOG(ALWAYS, "\n");
    ZL_RLOG(ALWAYS, "Usage:\n");
    ZL_RLOG(ALWAYS, "    %s input_file\n", progname);
    ZL_RLOG(ALWAYS, "\n");
    ZL_RLOG(ALWAYS,
            "This program takes an input that is a zstrong-compressed frame, decompresses\n"
            "it, and writes out a file for each stream in the frame. The stream files are\n"
            "the input file name suffixed with a period and the stream number.\n");
    // ZL_RLOG(ALWAYS, "\n");
    // ZL_RLOG(ALWAYS, "Options:\n");
    // ZL_RLOG(ALWAYS, "    \n");
    exit(1);
}

static stream_dump_args_t parse_args(int argc, char* argv[])
{
    if (argc != 2) {
        usage(argc, argv);
    }
    stream_dump_args_t args;
    args.srcFileName   = argv[1];
    args.dstFilePrefix = argv[1];
    return args;
}

static void write_stream_to_file(
        const char* prefix,
        size_t nbStreams,
        ZL_IDType strid,
        const ZL_DataInfo* strm)
{
    ZL_RC contents = ZL_RC_wrap(
            ZL_DataInfo_getDataPtr(strm), ZL_DataInfo_getContentSize(strm));
    ZL_LOG(V, "Stream %u has %lu bytes.", strid, ZL_RC_avail(&contents));
    char* outFileName;
    {
        const size_t prefixLen           = strlen(prefix);
        const size_t outFileNameCapacity = prefixLen + strlen(".streams.")
                + 10 /* max uint32_t len */ + 1 /* '\0' */;
        outFileName = malloc(outFileNameCapacity);
        ZL_REQUIRE_NN(outFileName);
        int streamWidth = 0;
        for (ZL_IDType i = (ZL_IDType)nbStreams; i; i /= 10) {
            streamWidth++;
        }
        const size_t outFileNameLen = (size_t)snprintf(
                outFileName,
                outFileNameCapacity,
                "%s.streams.%0*u",
                prefix,
                streamWidth,
                strid);
        ZL_REQUIRE_LE(outFileNameLen, outFileNameCapacity);
    }

    FIO_writeFile(contents, outFileName);
    free(outFileName);
}

static size_t
fill_csize(const ZL_ReflectionCtx* rctx, size_t* csize, size_t stream)
{
    if (csize[stream] != (size_t)-1) {
        // Already filled
        return csize[stream];
    }
    const ZL_DataInfo* info =
            ZL_ReflectionCtx_getStream_lastChunk(rctx, stream);
    ZL_REQUIRE_NN(info);
    const ZL_CodecInfo* consumer = ZL_DataInfo_getConsumerCodec(info);
    if (consumer == NULL) {
        // Stored stream
        csize[stream] = ZL_DataInfo_getContentSize(info);
        return csize[stream];
    }

    ZL_REQUIRE_EQ(
            ZL_CodecInfo_getNumInputs(consumer),
            1,
            "Only valid for single input transforms");
    size_t const nbSuccessors = ZL_CodecInfo_getNumOutputs(consumer);
    csize[stream]             = ZL_CodecInfo_getHeaderSize(consumer);
    for (size_t i = 0; i < nbSuccessors; ++i) {
        const ZL_DataInfo* successor = ZL_CodecInfo_getOutput(consumer, i);
        const size_t successorIndex  = ZL_DataInfo_getIndex(successor);
        csize[stream] += fill_csize(rctx, csize, successorIndex);
    }
    return csize[stream];
}

static void write_stream_graph_dot_file(
        const char* prefix,
        const ZL_ReflectionCtx* rctx,
        const size_t compressedSize)
{
    char* outFileName;
    {
        const size_t prefixLen = strlen(prefix);
        const size_t outFileNameCapacity =
                prefixLen + strlen(".streams.dot") + 1 /* '\0' */;
        outFileName = malloc(outFileNameCapacity);
        ZL_REQUIRE_NN(outFileName);
        const size_t outFileNameLen = (size_t)snprintf(
                outFileName, outFileNameCapacity, "%s.streams.dot", prefix);
        ZL_REQUIRE_LE(outFileNameLen, outFileNameCapacity);
    }

    FILE* const f = fopen(outFileName, "wb");
    ZL_REQUIRE_NN(f);

    free(outFileName);

    const size_t nbStreams  = ZL_ReflectionCtx_getNumStreams_lastChunk(rctx);
    const size_t nbDecoders = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);

    fprintf(f, "digraph stream_topo {\n");

    // Build cSize, which is the total compressed size of each stream
    size_t* cSize = malloc(nbStreams * sizeof(size_t));
    ZL_REQUIRE_NN(cSize);
    memset(cSize, -1, nbStreams * sizeof(size_t));
    for (size_t strid = 0; strid < nbStreams; ++strid) {
        fill_csize(rctx, cSize, strid);

        const ZL_DataInfo* info =
                ZL_ReflectionCtx_getStream_lastChunk(rctx, strid);
        const ZL_Type stype   = ZL_DataInfo_getType(info);
        const size_t eltWidth = ZL_DataInfo_getEltWidth(info);
        const size_t nbElts   = ZL_DataInfo_getNumElts(info);

        const char* typename = "Unknown";
        const char* strmname = NULL; // TODO: plumb through

        switch (stype) {
            case ZL_Type_serial:
                typename = "Serialized";
                break;
            case ZL_Type_struct:
                typename = "Fixed_Width";
                break;
            case ZL_Type_numeric:
                typename = "Numeric";
                break;
            case ZL_Type_string:
                typename = "Variable_Size";
                break;
            default:
                typename = "custom type mask";
        }

        fprintf(f,
                "S%u [shape=record,label=\"Stream: %u\\n",
                (unsigned)strid,
                (unsigned)strid);

        if (strmname != NULL) {
            fprintf(f, "Name: %s\\n", strmname);
        }

        fprintf(f,
                "Type: %s\\nEltWidth: %zu\\n#Elts: %zu",
                typename,
                eltWidth,
                nbElts);

        // TODO only count stored streams or something?
        {
            double pct = (double)cSize[strid] / (double)compressedSize * 100;
            fprintf(f, "\\nCSize: %u", (unsigned)cSize[strid]);
            fprintf(f, "\\nShare: %5.2lf%%", pct);
        }

        fprintf(f, "\"];\n");
    }

    fprintf(f, "\n");

    free(cSize);

    for (ZL_IDType did = 0; did < nbDecoders; did++) {
        const ZL_CodecInfo* info =
                ZL_ReflectionCtx_getCodec_lastChunk(rctx, did);
        const size_t nbInputStreams  = ZL_CodecInfo_getNumInputs(info);
        const size_t nbOutputStreams = ZL_CodecInfo_getNumOutputs(info);

        const char* trtypeStr =
                ZL_CodecInfo_isStandardCodec(info) ? "Standard" : "Custom";
        const char* trname = ZL_CodecInfo_getName(info);
        ZL_REQUIRE_NN(trname);

        fprintf(f,
                "T%u [shape=Mrecord,label=\"%s (ID: %u)\\n %s transform %u\\nHeader size: %u",
                did,
                trname,
                ZL_CodecInfo_getCodecID(info),
                trtypeStr,
                did,
                (unsigned)ZL_CodecInfo_getHeaderSize(info));
        fprintf(f, "\"];\n");

        for (size_t i = 0; i < nbOutputStreams; ++i) {
            const ZL_DataInfo* output = ZL_CodecInfo_getOutput(info, i);
            const unsigned outputIdx  = (unsigned)ZL_DataInfo_getIndex(output);
            fprintf(f,
                    "T%u -> S%u [label=\"#%u\"];\n",
                    did,
                    outputIdx,
                    (unsigned)(nbOutputStreams - 1 - i));
        }

        for (size_t i = 0; i < nbInputStreams; i++) {
            const ZL_DataInfo* input = ZL_CodecInfo_getInput(info, i);
            const unsigned inputIdx  = (unsigned)ZL_DataInfo_getIndex(input);
            fprintf(f,
                    "S%u -> T%u [label=\"#%u\"];\n",
                    inputIdx,
                    did,
                    (unsigned)i);
        }
    }

    fprintf(f, "}\n");

    ZL_REQUIRE_EQ(fclose(f), 0);
}

int main(int argc, char* argv[])
{
    const stream_dump_args_t args = parse_args(argc, argv);

    ZL_Buffer input = FIO_createBuffer_fromFilename(args.srcFileName);
    ZL_RC inputRC   = ZL_B_getRC(&input);

    ZL_ReflectionCtx* const rctx = ZL_ReflectionCtx_create();

    stream_dump_register_decoders(ZL_ReflectionCtx_getDCtx(rctx));

    ZL_REQUIRE_SUCCESS(ZL_ReflectionCtx_setCompressedFrame(
            rctx, ZL_RC_ptr(&inputRC), ZL_RC_avail(&inputRC)));

    const size_t nbStreams = ZL_ReflectionCtx_getNumStreams_lastChunk(rctx);
    for (size_t strid = 0; strid < nbStreams; strid++) {
        const ZL_DataInfo* const strm =
                ZL_ReflectionCtx_getStream_lastChunk(rctx, strid);
        write_stream_to_file(
                args.dstFilePrefix, nbStreams, (ZL_IDType)strid, strm);
    }

    write_stream_graph_dot_file(args.dstFilePrefix, rctx, ZL_B_size(&input));

    ZL_ReflectionCtx_free(rctx);
    ZL_B_destroy(&input);

    return 0;
}
