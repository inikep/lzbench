// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Round trips the source file, validates the data round trips, and prints the
 * compressed size.
 */

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "openzl/codecs/rolz/decode_rolz_kernel.h"
#include "openzl/codecs/rolz/encode_rolz_kernel.h"
#include "tools/fileio/fileio.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_decompress.h"

namespace {
enum class Mode {
    LZ,
    ROLZ,
    ZS_FIELD_LZ,
};

Mode getMode(std::string mode)
{
    if (mode == "lz")
        return Mode::LZ;
    if (mode == "rolz")
        return Mode::ROLZ;
    if (mode == "field")
        return Mode::ZS_FIELD_LZ;
    throw std::runtime_error("Unsupported mode!");
}

size_t compressBound(size_t inputSize, Mode mode)
{
    switch (mode) {
        case Mode::LZ:
            return ZS_fastLzCompressBound(inputSize);
        case Mode::ROLZ:
            return ZS_rolzCompressBound(inputSize);
        case Mode::ZS_FIELD_LZ:
            return ZL_compressBound(inputSize);
    }
    throw std::runtime_error("Unsupported mode!");
}

ZL_Report compress(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize,
        uint32_t fieldSize,
        Mode mode)
{
    switch (mode) {
        case Mode::LZ: {
            auto const ret = ZS_fastLzCompress(dst, dstCapacity, src, srcSize);
            ZL_REQUIRE(!ZL_isError(ret));
            return ret;
        }
        case Mode::ROLZ: {
            auto const ret = ZS_rolzCompress(dst, dstCapacity, src, srcSize);
            ZL_REQUIRE(!ZL_isError(ret));
            return ret;
        }
        case Mode::ZS_FIELD_LZ: {
            ZL_Compressor* cgraph = ZL_Compressor_create();
            ZL_REQUIRE_NN(cgraph);
            ZL_GraphID graph = ZL_Compressor_registerFieldLZGraph(cgraph);
            switch (fieldSize) {
                case 1:
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_CONVERT_NUM_TO_TOKEN, graph);
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_INTERPRET_AS_LE8, graph);
                    break;
                case 2:
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_CONVERT_NUM_TO_TOKEN, graph);
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_INTERPRET_AS_LE16, graph);
                    break;
                case 4:
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_CONVERT_SERIAL_TO_TOKEN4, graph);
                    break;
                case 8:
                    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                            cgraph, ZL_NODE_CONVERT_SERIAL_TO_TOKEN8, graph);
                    break;
                default:
                    ZL_REQUIRE_FAIL("Unsupported field size");
            }
            ZL_REQUIRE_SUCCESS(
                    ZL_Compressor_selectStartingGraphID(cgraph, graph));
            auto const ret = ZL_compress_usingCompressor(
                    dst, dstCapacity, src, srcSize, cgraph);
            ZL_Compressor_free(cgraph);
            return ret;
        }
    }
    throw std::runtime_error("Unsupported mode!");
}

ZL_Report decompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        Mode mode)
{
    switch (mode) {
        case Mode::LZ: {
            auto const ret =
                    ZS_fastLzDecompress(dst, dstCapacity, src, srcSize);
            ZL_REQUIRE(!ZL_isError(ret));
            return ret;
        }
        case Mode::ROLZ: {
            auto const ret = ZL_rolzDecompress(dst, dstCapacity, src, srcSize);
            ZL_REQUIRE(!ZL_isError(ret));
            return ret;
        }
        case Mode::ZS_FIELD_LZ:
            return ZL_decompress(dst, dstCapacity, src, srcSize);
    }
    throw std::runtime_error("Unsupported mode!");
}

} // namespace

int main(int argc, char** argv)
{
    ZL_g_logLevel = ZL_LOG_LVL_WARN;

    if (argc < 3) {
        fprintf(stderr,
                "USAGE: %s (lz|rolz|int) INPUT [FIELD_SIZE] [FIZED_OFFSET]\n",
                argv[0]);
        return 1;
    }
    auto const mode       = getMode(argv[1]);
    char const* inputFile = argv[2];

    uint32_t fieldSize = 0;
    if (mode == Mode::ZS_FIELD_LZ) {
        if (argc != 4) {
            throw std::runtime_error("Wrong # of arguments");
        }
        fieldSize = (uint32_t)atoi(argv[3]);
    } else if (argc != 3) {
        throw std::runtime_error("Too many arguments!");
    }

    const size_t inputSize = FIO_sizeof_file(inputFile);
    std::string input;
    input.resize(inputSize);
    FILE* f = fopen(inputFile, "rb");
    if (f == nullptr) {
        return 2;
    }
    {
        size_t const s = fread(&input[0], 1, input.size(), f);
        if (s != input.size()) {
            return 3;
        }
        input.resize(s);
    }

    std::string compressed;
    compressed.resize(compressBound(input.size(), mode));
    std::string roundTripped;
    roundTripped.resize(input.size() + 16);
    {
        auto const start = std::chrono::steady_clock::now();
        auto const ret   = compress(
                &compressed[0],
                compressed.size(),
                input.data(),
                input.size(),
                fieldSize,
                mode);
        if (ZL_isError(ret)) {
            return 4;
        }
        auto const stop = std::chrono::steady_clock::now();
        auto const us   = std::chrono::duration_cast<std::chrono::microseconds>(
                                stop - start)
                                .count();
        auto const sec = double(us) / 1000000;
        compressed.resize(ZL_validResult(ret));
        fprintf(stderr,
                "TENTATIVE: %zu -> %zu (%.2f) in %.2fs @ %.2f MB/s\n",
                input.size(),
                compressed.size(),
                double(input.size()) / double(compressed.size()),
                sec,
                (double(input.size()) / (1024 * 1024)) / sec);
    }
    if (mode == Mode::ZS_FIELD_LZ) {
        std::string outputFile{ inputFile };
        outputFile += ".zs";
        FILE* output = fopen(outputFile.c_str(), "wb");
        ZL_REQUIRE_NN(output);
        ZL_REQUIRE_EQ(
                compressed.size(),
                fwrite(compressed.data(), 1, compressed.size(), output));
        fclose(output);
    }
    {
        int const repeats = 5;
        size_t size;
        std::chrono::nanoseconds minNs = std::chrono::hours(24);
        for (int i = 0; i < repeats; ++i) {
            auto const start = std::chrono::steady_clock::now();
            auto const ret   = decompress(
                    &roundTripped[0],
                    roundTripped.size(),
                    compressed.data(),
                    compressed.size(),
                    mode);
            if (ZL_isError(ret)) {
                return 5;
            }
            size            = ZL_validResult(ret);
            auto const stop = std::chrono::steady_clock::now();
            auto const ns   = stop - start;
            minNs           = std::min(ns, minNs);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        roundTripped.resize(size);
        if (size != input.size()) {
            fprintf(stderr,
                    "ERROR: Round tripped size wrong. Expected %zu and got %zu\n",
                    input.size(),
                    size);
            return 6;
        }
        if (input != roundTripped) {
            size_t badPos = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                if (input[i] != roundTripped[i]) {
                    badPos = i;
                    break;
                }
            }
            fprintf(stderr, "ERROR: Round trip failed (pos=%zu)!\n", badPos);
            return 7;
        }
        auto const ns  = minNs.count();
        auto const sec = double(ns) / 1e+9;
        fprintf(stderr,
                "DECOMPRESS: %.4fs @ %.2f MB/s\n",
                sec,
                (double(input.size()) / (1024 * 1024)) / sec);
    }
    fprintf(stderr,
            "SUCCESS: %zu -> %zu (%.2f)\n",
            input.size(),
            compressed.size(),
            double(input.size()) / double(compressed.size()));
    fclose(f);
    return 0;
}
