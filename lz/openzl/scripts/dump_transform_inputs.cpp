// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * CLI tool to dump the input of every instance of a transform in a Zstrong
 * graph. Dumps each input to the output directory named:
 *
 *   <output-dir>/<filename(zstrong-file)>.<transform-instance>.<input-index>
 *
 * Usage:
 *   dump_transform_inputs <transform-id> <output-dir> [<zstrong-file> ...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "openzl/common/debug.h"
#include "openzl/zl_reflection.h"

namespace {
namespace fs = std::filesystem;

int usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s <transform-id> <output-dir> [<zstrong-file> ...]\n",
            prog);
    return 1;
}

std::string readFile(const fs::path& path)
{
    std::ifstream file(path);
    return std::string{ std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>() };
}

void writeFile(const fs::path& path, std::string_view data)
{
    FILE* f         = fopen(path.c_str(), "wb");
    const auto size = fwrite(data.data(), 1, data.size(), f);
    ZL_REQUIRE_EQ(size, data.size());
    fclose(f);
}

void handleStream(const ZL_DataInfo* stream, const fs::path& outPrefix)
{
    const auto type = ZL_DataInfo_getType(stream);

    auto writeContent = [&outPrefix, stream](const std::string& suffix) {
        const auto outPath = outPrefix.string() + suffix;
        const auto data    = ZL_DataInfo_getDataPtr(stream);
        const auto size    = ZL_DataInfo_getContentSize(stream);
        writeFile(outPath, { (const char*)data, size });
    };

    switch (type) {
        case ZL_Type_serial:
            writeContent(".serial");
            break;
        case ZL_Type_struct:
            writeContent(
                    ".struct."
                    + std::to_string(ZL_DataInfo_getEltWidth(stream)));
            break;
        case ZL_Type_numeric:
            writeContent(
                    ".numeric."
                    + std::to_string(ZL_DataInfo_getEltWidth(stream)));
            break;
        case ZL_Type_string: {
            writeContent(".string.content");
            const auto lensPath = outPrefix.string() + ".string.lengths";
            const auto lens     = ZL_DataInfo_getLengthsPtr(stream);
            const auto nbElts   = ZL_DataInfo_getNumElts(stream);
            writeFile(lensPath, { (const char*)lens, nbElts * sizeof(*lens) });
            break;
        }
        default:
            ZL_REQUIRE_FAIL("Unknown stream type: %d", type);
    }
}

void handleTransform(
        const ZL_CodecInfo* transform,
        const fs::path& outPrefix,
        size_t transformIdx)
{
    const auto nbInputs = ZL_CodecInfo_getNumInputs(transform);
    for (size_t i = 0; i < nbInputs; ++i) {
        const auto input   = ZL_CodecInfo_getInput(transform, i);
        const auto outPath = outPrefix.string() + "."
                + std::to_string(transformIdx) + "." + std::to_string(i);
        handleStream(input, outPath);
    }
}

void handleFile(
        const fs::path& inFile,
        const fs::path& outPrefix,
        int transformId)
{
    const auto data        = readFile(inFile);
    ZL_ReflectionCtx* rctx = ZL_ReflectionCtx_create();
    // TODO: Register custom transforms
    ZL_REQUIRE_SUCCESS(ZL_ReflectionCtx_setCompressedFrame(
            rctx, data.data(), data.size()));

    const auto nbTransforms = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);
    for (size_t i = 0, outIndex = 0; i < nbTransforms; ++i) {
        const auto transform = ZL_ReflectionCtx_getCodec_lastChunk(rctx, i);
        if (ZL_CodecInfo_getCodecID(transform) != transformId) {
            continue;
        }
        handleTransform(transform, outPrefix, outIndex++);
    }

    ZL_ReflectionCtx_free(rctx);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        return usage(argv[0]);
    }
    const int transformId = atoi(argv[1]);
    if (transformId == 0 && !(argv[1][0] == '0' && argv[1][1] == '\0')) {
        fprintf(stderr, "Error: invalid transform ID: %s\n", argv[1]);
        return usage(argv[0]);
    }

    const fs::path outDir = argv[2];
    fs::create_directories(outDir);

    for (int i = 3; i < argc; ++i) {
        const fs::path inFile    = argv[i];
        const fs::path outPrefix = outDir / inFile.filename();
        handleFile(inFile, outPrefix, transformId);
    }
}
