// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <chrono>

#include <folly/FileUtil.h>
#include <folly/dynamic.h>
#include <folly/json.h>

#include "tools/zstrong_cpp.h"
#include "tools/zstrong_json.h"

/**
 * The best way to generate a JSON graph is using the Python bindings.
 *
 * ```py
 * import zstrong_json as zs
 * import json
 *
 * graph = zs.transforms.interpret_as_le32(zs.graph.field_lz())
 * print(json.dumps(graph))
 * ```
 */

namespace {
constexpr size_t kRepeats = 5;
void usage()
{
    fprintf(stderr, "Usage: compress JSON INPUT [OUTPUT]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "json: Either a JSON file or a JSON object.\n");
    fprintf(stderr, "input: Input file to compress.\n");
    fprintf(stderr,
            "output: Optionally output file to write compressed data to.\n"
            "        Defaults to INPUT.zs\n");
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        usage();
        return 1;
    }

    folly::dynamic json;
    std::string jsonStr;
    try {
        if (argv[1][0] == '{') {
            jsonStr = argv[1];
        } else {
            if (!folly::readFile(argv[1], jsonStr)) {
                fprintf(stderr, "Failed to read JSON file: %s\n", argv[1]);
                usage();
                return 1;
            }
        }
        json = folly::parseJson(jsonStr);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to parse JSON: %s\n", e.what());
        fprintf(stderr, "JSON: %s\n", jsonStr.c_str());
        usage();
        return 1;
    }

    std::string src;
    if (!folly::readFile(argv[2], src)) {
        fprintf(stderr, "Failed to read input file: %s\n", argv[2]);
        usage();
        return 1;
    }

    zstrong::CCtx cctx;
    zstrong::CGraph cgraph;
    zstrong::DCtx dctx;

    std::string compressed;
    try {
        auto graph = zstrong::JsonGraph(std::move(json));
        cgraph.unwrap(ZL_Compressor_selectStartingGraphID(
                cgraph.get(), graph.registerGraph(*cgraph)));

        compressed = zstrong::compress(cctx, src, cgraph);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to compress: %s\n", e.what());
        return 1;
    }

    std::string roundTripped;
    try {
        roundTripped = zstrong::decompress(dctx, compressed);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to decompress: %s\n", e.what());
        return 1;
    }

    if (src != roundTripped) {
        fprintf(stderr, "Round-trip failed\n");
        return 1;
    }

    std::string outFile = argc == 4 ? argv[3] : (std::string(argv[2]) + ".zs");
    if (!folly::writeFile(compressed, outFile.c_str())) {
        fprintf(stderr, "Failed to write output file: %s\n", outFile.c_str());
        return 1;
    }

    std::chrono::nanoseconds compressNs{ 0 };
    std::chrono::nanoseconds decompressNs{ 0 };
    size_t uncompressedBytes = 0;

    for (size_t i = 0; i < kRepeats; ++i) {
        auto start = std::chrono::steady_clock::now();
        zstrong::compress(cctx, &compressed, src, cgraph);
        auto stop = std::chrono::steady_clock::now();
        compressNs += (stop - start);

        start        = std::chrono::steady_clock::now();
        roundTripped = zstrong::decompress(dctx, compressed);
        stop         = std::chrono::steady_clock::now();
        decompressNs += (stop - start);

        uncompressedBytes += src.size();
    }

    const auto compressionSpeed =
            double(uncompressedBytes) / compressNs.count() * 1000.0;
    const auto decompressionSpeed =
            double(uncompressedBytes) / decompressNs.count() * 1000.0;

    fprintf(stderr, "Original size      : %zu\n", src.size());
    fprintf(stderr, "Compressed size    : %zu\n", compressed.size());
    fprintf(stderr,
            "Compression ratio  : %.2f\n",
            double(src.size()) / double(compressed.size()));
    fprintf(stderr, "Compression speed  : %.2f\n", compressionSpeed);
    fprintf(stderr, "Decompression speed: %.2f\n", decompressionSpeed);

    return 0;
}
