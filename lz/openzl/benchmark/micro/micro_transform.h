// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>
#include "benchmark/benchmark_data.h"

namespace zstrong::bench::micro {

/**
 * ZstrongTransform:
 * A interface for implementation of benchmarks for Zstrong Transfoms.
 * Allows us to have a unified non-parameterized interface for transfoms
 * benchmarksing.
 **/
class ZstrongTransform {
   public:
    ZstrongTransform()          = default;
    virtual ~ZstrongTransform() = default;

    /**
     * benchEncoding:
     * Benchmarks encoding using given benchmarking state.
     * `state` shouldn't be used outside of this function.
     * Adds the following metrics:
     * - bytes_per_second - encoding speed
     */
    virtual void benchEncoding(
            benchmark::State& state,
            const std::string_view& src) = 0;

    /**
     * benchDecoding:
     * Benchmarks decode using given benchmarking state.
     * `state` shouldn't be used outside of this function.
     * Adds the following metrics:
     * - bytes_per_second - decoding speed
     */
    virtual void benchDecoding(
            benchmark::State& state,
            const std::string_view& src) = 0;
    virtual std::string name()           = 0;
};

/**
 * ParameterizedZstrongTransform:
 * A base class that eases the definitions of Zstrong transforms by supporting
 * type-parameterized abstractions that can fit typed inputs and multiple typed
 * outputs.
 * Most transform benchmarks should inherit this class, provide input/output
 * types in template and implement encode, decode and name.
 * `Tsrc` is the input type. `Tdsts` is one or more output types.
 * Note that roundtrip testing is performed as part of the benchmarking
 * operation.
 **/
template <typename Tsrc, typename... Tdsts>
class ParameterizedZstrongTransform : public ZstrongTransform {
    /**
     * toTypedSrc:
     * Converts a byte based src into a `Tsrc` typed vector.
     * Rounds down to the closest number of complete elements.
     **/
    std::vector<Tsrc> toTypedSrc(const std::string_view& src)
    {
        std::vector<Tsrc> typedSrc;
        typedSrc.resize(src.size() / sizeof(Tsrc));
        memcpy(typedSrc.data(), src.data(), typedSrc.size() * sizeof(Tsrc));
        return typedSrc;
    }

   public:
    virtual ~ParameterizedZstrongTransform() = default;

    /**
     * encode:
     * The encoding operation of the transform.
     * The encoder should make sure the output vectors are of the correct size.
     * Note: always take a reference into output to avoid costly copy
     * operations.
     **/
    virtual void encode(
            const std::vector<Tsrc>& src,
            std::tuple<std::vector<Tdsts>...>& output) = 0;

    /**
     * decode:
     * The decoding operation of the transform.
     * The decoder should make sure the output vector is of the correct size.
     * Note: always take a reference into output to avoid costly copy
     * operations.
     **/
    virtual void decode(
            const std::tuple<std::vector<Tdsts>...>& src,
            std::vector<Tsrc>& output) = 0;

    /**
     * roundtrip:
     * Checks that decoding and the encoded data results back in the src.
     * Raise exception on failure.
     **/
    void roundtrip(const std::vector<Tsrc>& src)
    {
        std::tuple<std::vector<Tdsts>...> encoded;
        std::vector<Tsrc> decoded;
        encode(src, encoded);
        decode(encoded, decoded);
        if (src != decoded) {
            throw std::runtime_error{ "Failed roundtrip testing" };
        }
    }

    void benchEncoding(benchmark::State& state, const std::vector<Tsrc>& src)
    {
        std::tuple<std::vector<Tdsts>...> encoded;

        // Verify correct operation
        roundtrip(src);

        // Encode once to "prime" the encoded buffers .
        encode(src, encoded);

        for (auto _ : state) {
            encode(src, encoded);
            benchmark::DoNotOptimize(encoded);
            benchmark::ClobberMemory();
        }
        state.SetBytesProcessed(
                (int64_t)(src.size() * sizeof(Tsrc) * state.iterations()));
    }

    void benchEncoding(benchmark::State& state, const std::string_view& src)
            override
    {
        benchEncoding(state, toTypedSrc(src));
    }

    void benchDecoding(benchmark::State& state, const std::vector<Tsrc>& src)
    {
        std::tuple<std::vector<Tdsts>...> encoded;
        std::vector<Tsrc> decoded;

        // Verify correct operation
        roundtrip(src);

        // Prep endoded data
        encode(src, encoded);

        // Decode once to "price" the decoded buffer.
        decode(encoded, decoded);

        for (auto _ : state) {
            decode(encoded, decoded);
            benchmark::DoNotOptimize(decoded);
            benchmark::ClobberMemory();
        }
        state.SetBytesProcessed(
                (int64_t)(src.size() * sizeof(Tsrc) * state.iterations()));
    }
    void benchDecoding(benchmark::State& state, const std::string_view& src)
            override
    {
        benchDecoding(state, toTypedSrc(src));
    }
};

} // namespace zstrong::bench::micro
