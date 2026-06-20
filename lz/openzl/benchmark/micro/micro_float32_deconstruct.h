// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include "benchmark/benchmark_testcase.h"
#include "benchmark/micro/micro_bench.h"
#include "openzl/codecs/float_deconstruct/decode_float_deconstruct_kernel.h"
#include "openzl/codecs/float_deconstruct/encode_float_deconstruct_kernel.h"

namespace zstrong::bench::micro {

template <typename eltType>
class FloatDeconstructTransform
        : public ParameterizedZstrongTransform<eltType, uint8_t, uint8_t> {
   private:
    static_assert(std::is_integral<eltType>::value);
    using encoderType = void (*)(const eltType*, uint8_t*, uint8_t*, size_t);
    using decoderType =
            void (*)(eltType*, const uint8_t*, const uint8_t*, size_t);
    const size_t _exponentBytes;
    const size_t _signFracBytes;
    const encoderType _encoderFunc;
    const decoderType _decoderFunc;
    const std::string _transformName;

   public:
    FloatDeconstructTransform(
            size_t exponentBytes,
            size_t signFracBytes,
            encoderType encoderFunc,
            decoderType decoderFunc,
            const std::string& transformName)
            : _exponentBytes(exponentBytes),
              _signFracBytes(signFracBytes),
              _encoderFunc(encoderFunc),
              _decoderFunc(decoderFunc),
              _transformName(transformName)
    {
    }

    void encode(
            const std::vector<eltType>& src,
            std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>& output)
            override
    {
        auto& [exponent, signFrac] = output;
        exponent.resize(src.size() * _exponentBytes);
        signFrac.resize(src.size() * _signFracBytes);
        _encoderFunc(src.data(), exponent.data(), signFrac.data(), src.size());
    }

    void decode(
            const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>& src,
            std::vector<eltType>& output) override
    {
        auto& [exponent, signFrac] = src;
        output.resize(exponent.size() / _exponentBytes);
        _decoderFunc(
                output.data(), exponent.data(), signFrac.data(), output.size());
    }

    std::string name() override
    {
        return _transformName;
    }
};

} // namespace zstrong::bench::micro
