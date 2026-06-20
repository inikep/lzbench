// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <cstdio>
#include <filesystem>

#include <folly/FileUtil.h>
#include <folly/init/Init.h>

#include "custom_transforms/tulip_v2/encode_tulip_v2.h"
#include "openzl/compress/cctx.h"
#include "openzl/decompress/dctx2.h"
#include "tools/zstrong_cpp.h"

namespace {
namespace fs = std::filesystem;

size_t constexpr kEncodeRepeats = 1;
size_t constexpr kDecodeRepeats = 1;

struct State {
    std::chrono::nanoseconds cTime{ 0 };
    std::chrono::nanoseconds dTime{ 0 };
    size_t cBytes{ 0 };
    size_t dBytes{ 0 };
    size_t cStreamMemory{ 0 };
    size_t dStreamMemory{ 0 };

    void print()
    {
        double const cMBps = double(kEncodeRepeats) * double(dBytes * 1000)
                / double(cTime.count());
        double const dMBps = double(kDecodeRepeats) * double(dBytes * 1000)
                / double(dTime.count());
        double const ratio = double(dBytes) / double(cBytes);
        fprintf(stderr,
                "C MB/s = %.2f - D MB/s = %.2f - Ratio = %.2f - C Size = %zu - D Size = %zu C Mem = %zu - D Mem = %zu\n",
                cMBps,
                dMBps,
                ratio,
                cBytes,
                dBytes,
                cStreamMemory,
                dStreamMemory);
    }
};

std::string decompress(State& state, std::string const& compressed)
{
    std::string decompressed;
    size_t const decompressedSize = zstrong::unwrap(
            ZL_getDecompressedSize(compressed.data(), compressed.size()));
    decompressed.resize(decompressedSize);

    zstrong::DCtx dctx;
    dctx.unwrap(zstrong::tulip_v2::registerCustomTransforms(dctx.get(), 0, 10));
    size_t const size = dctx.unwrap(ZL_DCtx_decompress(
            dctx.get(),
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            compressed.size()));
    decompressed.resize(size);
    state.dStreamMemory = DCTX_streamMemory(dctx.get());
    return decompressed;
}

std::string
compress(State& state, std::string const& data, zstrong::CGraph const& cgraph)
{
    zstrong::CCtx cctx;
    auto compressed     = zstrong::compress(cctx, data, cgraph);
    state.cStreamMemory = CCTX_streamMemory(cctx.get());
    return compressed;
}

std::string compress(State& state, std::string const& data)
{
    zstrong::CGraph cgraph;
    cgraph.unwrap(ZL_Compressor_setParameter(
            cgraph.get(), ZL_CParam_compressionLevel, 1));
    cgraph.unwrap(ZL_Compressor_selectStartingGraphID(
            cgraph.get(),
            zstrong::tulip_v2::createTulipV2Graph(cgraph.get(), {}, 0, 10)));
    auto start      = std::chrono::steady_clock::now();
    auto compressed = compress(state, data, cgraph);
    for (size_t i = 1; i < kEncodeRepeats; ++i)
        compressed = compress(state, data, cgraph);
    auto stop = std::chrono::steady_clock::now();
    state.cTime += (stop - start);

    start             = std::chrono::steady_clock::now();
    auto decompressed = decompress(state, compressed);
    for (size_t i = 1; i < kDecodeRepeats; ++i)
        decompressed = decompress(state, compressed);
    stop = std::chrono::steady_clock::now();
    state.dTime += (stop - start);

    state.cBytes += compressed.size();
    state.dBytes += data.size();

    if (decompressed != data) {
        throw std::runtime_error("Failed to round trip!");
    }
    return compressed;
}

void handleFile(State& state, std::string const& inputFile)
{
    std::string data;
    if (!folly::readFile(inputFile.c_str(), data)) {
        throw std::runtime_error("failed to read file: " + inputFile);
    }

    std::string compressed = compress(state, data);

    auto outputFile = inputFile + ".zs";
    if (!folly::writeFile(compressed, outputFile.c_str())) {
        throw std::runtime_error("failed to write file: " + outputFile);
    }
}
} // namespace

#if 1
extern "C" {
extern size_t varintLength[11];
extern size_t varintLengthO1[3][3];
}
#endif
int main(int argc, char** argv)
{
    folly::Init init(&argc, &argv);

    // ZL_g_logLevel = ZL_LOG_LVL_V4;

    std::vector<std::string> files;
    for (int arg = 1; arg < argc; ++arg) {
        if (fs::is_regular_file(argv[arg])) {
            files.emplace_back(argv[arg]);
        } else if (fs::is_directory(argv[arg])) {
            for (auto const& entry :
                 fs::recursive_directory_iterator(argv[arg])) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                files.push_back(entry.path().native());
            }
        } else {
            throw std::runtime_error(
                    std::string("Arg is not a file or directory: ")
                    + argv[arg]);
        }
    }

    State state;
    for (auto const& file : files) {
        if (file.ends_with(".zs")) {
            continue;
        }
        handleFile(state, file);
    }

    state.print();

#if 0
    size_t total = 0;
    for (size_t i = 1; i <= 10; ++i) {
        total += varintLength[i];
    }

    for (size_t i = 0; i <= 10; ++i) {
        fprintf(stderr,
                "varint[%zu] = %.1f\n",
                i,
                (double)varintLength[i] * 100.0 / (double)total);
    }
    for (int i = 0; i < 3; ++i) {
        size_t t = 0;
        for (int j = 0; j < 3; ++j) {
            t += varintLengthO1[i][j];
        }
        for (int j = 0; j < 3; ++j) {
            fprintf(stderr,
                    "varintO1[%d][%d] = %.1f\n",
                    i,
                    j,
                    (double)varintLengthO1[i][j] * 100. / (double)t);
        }
    }
#endif
}
