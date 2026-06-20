// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "openzl/common/assertion.h"
#include "openzl/common/stream.h"
#include "openzl/openzl.hpp"

using namespace testing;

namespace openzl::tests {

TEST(TestInput, get)
{
    auto input = Input::refSerial("hello world");
    ASSERT_NE(input.get(), nullptr);
}

TEST(TestInput, refSerial)
{
    const char* ptr = "hello";
    auto input      = Input::refSerial(ptr);
    ASSERT_EQ(input.type(), Type::Serial);
    ASSERT_EQ(input.eltWidth(), 1);
    ASSERT_EQ(input.numElts(), 5);
    ASSERT_EQ(input.contentSize(), 5);
    ASSERT_EQ(input.ptr(), ptr);
    ASSERT_THROW(input.stringLens(), Exception);
}

TEST(TestInput, refStruct)
{
    std::array<int, 3> data = { 0, 1, 2 };
    auto input              = Input::refStruct(poly::span<const int>{ data });
    ASSERT_EQ(input.type(), Type::Struct);
    ASSERT_EQ(input.eltWidth(), 4);
    ASSERT_EQ(input.numElts(), 3);
    ASSERT_EQ(input.contentSize(), 12);
    ASSERT_EQ(input.ptr(), data.data());
    ASSERT_THROW(input.stringLens(), Exception);
}

TEST(TestInput, refNumeric)
{
    std::array<int, 3> data = { 0, 1, 2 };
    auto input              = Input::refNumeric(poly::span<const int>{ data });
    ASSERT_EQ(input.type(), Type::Numeric);
    ASSERT_EQ(input.eltWidth(), 4);
    ASSERT_EQ(input.numElts(), 3);
    ASSERT_EQ(input.contentSize(), 12);
    ASSERT_EQ(input.ptr(), data.data());
    ASSERT_THROW(input.stringLens(), Exception);
}

TEST(TestInput, refString)
{
    const char* content             = "hello world i am string";
    std::array<uint32_t, 5> lengths = { 6, 6, 2, 3, 6 };
    auto input                      = Input::refString(content, lengths);
    ASSERT_EQ(input.type(), Type::String);
    ASSERT_EQ(input.eltWidth(), 0);
    ASSERT_EQ(input.numElts(), 5);
    ASSERT_EQ(input.contentSize(), 23);
    ASSERT_EQ(input.ptr(), content);
    ASSERT_EQ(input.stringLens(), lengths.data());

    ASSERT_THROW(
            Input::refString(content, 1, lengths.data(), lengths.size()),
            Exception);
}

TEST(TestInput, setIntMetadata)
{
    auto input = Input::refSerial("hello world");
    ASSERT_EQ(input.getIntMetadata(42), poly::nullopt);
    input.setIntMetadata(42, 350);
    ASSERT_EQ(input.getIntMetadata(42), 350);
}

TEST(TestInput, useConcurrentlySerial)
{
    std::array<Input, 2> inputs = { Input::refSerial("hello world"),
                                    Input::refSerial(
                                            "hello world hello hello") };
    std::vector<std::thread> threads;
    Compressor compressor;
    compressor.selectStartingGraph(ZL_GRAPH_COMPRESS_GENERIC);
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&inputs, &compressor]() {
            for (size_t j = 0; j < 100; ++j) {
                CCtx cctx;
                cctx.refCompressor(compressor);
                auto roundTripped = DCtx().decompress(cctx.compress(inputs));
                ASSERT_EQ(roundTripped.size(), inputs.size());
                for (size_t k = 0; k < inputs.size(); ++k) {
                    ASSERT_EQ(roundTripped[k], inputs[k]);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

TEST(TestInput, useConcurrentlyString)
{
    const char* content             = "hello world i am string";
    std::array<uint32_t, 5> lengths = { 6, 6, 2, 3, 6 };
    auto input                      = Input::refString(content, lengths);
    std::vector<std::thread> threads;
    Compressor compressor;
    compressor.selectStartingGraph(ZL_GRAPH_COMPRESS_GENERIC);
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&input, &compressor]() {
            for (size_t j = 0; j < 100; ++j) {
                CCtx cctx;
                cctx.refCompressor(compressor);
                auto roundTripped =
                        DCtx().decompressOne(cctx.compressOne(input));
                ASSERT_EQ(roundTripped, input);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

TEST(TestInput, useConcurrentlyWithRefcount)
{
    // Create an Input object that is refcounted.
    // This is not possible with the public API, but it happens in training
    // introspection.
    ZL_Data* data = STREAM_create({ 0 });
    ZL_REQUIRE(!ZL_isError(STREAM_reserve(data, ZL_Type_serial, 1, 11)));
    memcpy(STREAM_getWBuffer(data).start, "hello world", 11);
    ZL_REQUIRE(!ZL_isError(ZL_Data_commit(data, 11)));
    auto input = InputRef(ZL_codemodMutDataAsInput(data));

    std::vector<std::thread> threads;
    Compressor compressor;
    compressor.selectStartingGraph(ZL_GRAPH_COMPRESS_GENERIC);
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&input, &compressor]() {
            for (size_t j = 0; j < 100; ++j) {
                CCtx cctx;
                cctx.refCompressor(compressor);
                auto roundTripped =
                        DCtx().decompressOne(cctx.compressOne(input));
                ASSERT_EQ(roundTripped, input);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    STREAM_free(data);
}
} // namespace openzl::tests
