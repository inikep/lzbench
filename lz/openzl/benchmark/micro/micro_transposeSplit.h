// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include "benchmark/benchmark_testcase.h"
#include "benchmark/micro/micro_bench.h"
#include "openzl/codecs/transpose/decode_transpose_kernel.h"
#include "openzl/codecs/transpose/encode_transpose_kernel.h"

namespace zstrong::bench::micro {

// TODO (@timothyoei): combine tsplits with template magic
template <typename eltType>
class TransposeSplit4Transform : public ParameterizedZstrongTransform<
                                         eltType,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t> {
   private:
    static_assert(std::is_integral<eltType>::value);
    using encoderType = void (*)(uint8_t**, const void*, size_t, size_t);
    using decoderType = void (*)(void*, const uint8_t**, size_t, size_t);
    const encoderType encoderFn_;
    const decoderType decoderFn_;
    const std::string transformName_;
    const size_t eltWidth_ = 4;
    size_t nbElts_;

   public:
    TransposeSplit4Transform(
            encoderType encoderFn,
            decoderType decoderFn,
            std::string transformName)
            : encoderFn_(encoderFn),
              decoderFn_(decoderFn),
              transformName_(transformName)
    {
    }

    void encode(
            const std::vector<eltType>& src,
            std::tuple<
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>>& output) override
    {
        nbElts_ = src.size() / 4;
        std::vector<uint8_t*> dst;
        dst.reserve(4);
        auto& [v1, v2, v3, v4] = output;
        v1.resize(nbElts_);
        v2.resize(nbElts_);
        v3.resize(nbElts_);
        v4.resize(nbElts_);
        dst.push_back(v1.data());
        dst.push_back(v2.data());
        dst.push_back(v3.data());
        dst.push_back(v4.data());
        encoderFn_(dst.data(), src.data(), nbElts_, eltWidth_);
    }

    void decode(
            const std::tuple<
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>>& src,
            std::vector<eltType>& output) override
    {
        output.resize(nbElts_ * eltWidth_);
        std::vector<const uint8_t*> in;
        in.reserve(4);
        auto& [v1, v2, v3, v4] = src;
        in.push_back(v1.data());
        in.push_back(v2.data());
        in.push_back(v3.data());
        in.push_back(v4.data());
        decoderFn_(output.data(), in.data(), nbElts_, eltWidth_);
    }

    std::string name() override
    {
        return transformName_;
    }
};

template <typename eltType>
class TransposeSplit8Transform : public ParameterizedZstrongTransform<
                                         eltType,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t,
                                         uint8_t> {
   private:
    static_assert(std::is_integral<eltType>::value);
    using encoderType = void (*)(uint8_t**, const void*, size_t, size_t);
    using decoderType = void (*)(void*, const uint8_t**, size_t, size_t);
    const encoderType encoderFn_;
    const decoderType decoderFn_;
    const std::string transformName_;
    const size_t eltWidth_ = 8;
    size_t nbElts_;

   public:
    TransposeSplit8Transform(
            encoderType encoderFn,
            decoderType decoderFn,
            std::string transformName)
            : encoderFn_(encoderFn),
              decoderFn_(decoderFn),
              transformName_(transformName)
    {
    }

    void encode(
            const std::vector<eltType>& src,
            std::tuple<
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>>& output) override
    {
        nbElts_ = src.size() / eltWidth_;
        std::vector<uint8_t*> dst;
        dst.reserve(8);
        auto& [v1, v2, v3, v4, v5, v6, v7, v8] = output;
        v1.resize(nbElts_);
        v2.resize(nbElts_);
        v3.resize(nbElts_);
        v4.resize(nbElts_);
        v5.resize(nbElts_);
        v6.resize(nbElts_);
        v7.resize(nbElts_);
        v8.resize(nbElts_);
        dst.push_back(v1.data());
        dst.push_back(v2.data());
        dst.push_back(v3.data());
        dst.push_back(v4.data());
        dst.push_back(v5.data());
        dst.push_back(v6.data());
        dst.push_back(v7.data());
        dst.push_back(v8.data());
        encoderFn_(dst.data(), src.data(), nbElts_, eltWidth_);
    }

    void decode(
            const std::tuple<
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>,
                    std::vector<uint8_t>>& src,
            std::vector<eltType>& output) override
    {
        output.resize(nbElts_ * eltWidth_);
        std::vector<const uint8_t*> in;
        in.reserve(8);
        auto& [v1, v2, v3, v4, v5, v6, v7, v8] = src;
        in.push_back(v1.data());
        in.push_back(v2.data());
        in.push_back(v3.data());
        in.push_back(v4.data());
        in.push_back(v5.data());
        in.push_back(v6.data());
        in.push_back(v7.data());
        in.push_back(v8.data());
        decoderFn_(output.data(), in.data(), nbElts_, eltWidth_);
    }

    std::string name() override
    {
        return transformName_;
    }
};
} // namespace zstrong::bench::micro
