// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include "benchmark/benchmark_testcase.h"
#include "benchmark/micro/micro_bench.h"
#include "openzl/codecs/prefix/decode_prefix_kernel.h"
#include "openzl/codecs/prefix/encode_prefix_kernel.h"

namespace zstrong::bench::micro {

template <typename eltType>
class PrefixTransform
        : public ParameterizedZstrongTransform<eltType, uint8_t, uint32_t> {
   private:
    static_assert(std::is_integral<eltType>::value);
    using encoderType = void (*)(
            uint8_t* const,
            uint32_t* const,
            uint32_t* const,
            const uint8_t* const,
            const size_t,
            const uint32_t* const,
            const size_t);
    using decoderType = ZL_Report (*)(
            uint8_t* const,
            uint32_t* const,
            const uint8_t* const,
            const size_t,
            const uint32_t* const,
            const uint32_t* const);
    const encoderType encoderFn_;
    const decoderType decoderFn_;
    const std::string transformName_;
    const size_t nbBytes_;
    ZL_SetStringLensInstructions instructs_;
    std::vector<uint32_t> suffixSizes;

   public:
    PrefixTransform(
            encoderType encoderFn,
            decoderType decoderFn,
            std::string transformName,
            const size_t nbBytes,
            ZL_SetStringLensInstructions instructs)
            : encoderFn_(encoderFn),
              decoderFn_(decoderFn),
              transformName_(transformName),
              nbBytes_(nbBytes),
              instructs_(instructs)
    {
    }

    void encode(
            const std::vector<eltType>& src,
            std::tuple<std::vector<uint8_t>, std::vector<uint32_t>>& output)
            override
    {
        auto& [v1, v2] = output;
        v1.resize(nbBytes_);
        v2.resize(nbBytes_);
        suffixSizes.resize(instructs_.nbStrings);
        encoderFn_(
                v1.data(),
                suffixSizes.data(),
                v2.data(),
                src.data(),
                instructs_.nbStrings,
                instructs_.stringLens,
                nbBytes_);
    }

    void decode(
            const std::tuple<std::vector<uint8_t>, std::vector<uint32_t>>& src,
            std::vector<eltType>& output) override
    {
        auto& [v1, v2] = src;
        output.resize(nbBytes_);
        std::vector<uint32_t> fieldSizes(instructs_.nbStrings);
        (void)decoderFn_(
                output.data(),
                fieldSizes.data(),
                v1.data(),
                instructs_.nbStrings,
                suffixSizes.data(),
                v2.data());
    }

    std::string name() override
    {
        return transformName_;
    }
};
} // namespace zstrong::bench::micro
