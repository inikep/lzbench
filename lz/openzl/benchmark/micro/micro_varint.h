// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include "benchmark/benchmark_testcase.h"
#include "benchmark/micro/micro_bench.h"
#include "openzl/shared/varint.h"

namespace zstrong::bench::micro {

template <typename Int>
using VarintEncoderFn = size_t (*)(Int val, uint8_t* dst);
using VarintDecoderFn =
        ZL_RESULT_OF(uint64_t) (*)(const uint8_t** src, const uint8_t* end);

template <
        typename Int,
        VarintEncoderFn<Int> kEncodeFn,
        VarintDecoderFn kDecodeFn>
class VarintTransform : public ParameterizedZstrongTransform<Int, uint8_t> {
    static_assert(std::is_integral<Int>::value);

   public:
    VarintTransform(std::string transformName)
            : transformName_(std::move(transformName))
    {
    }

    void encode(
            const std::vector<Int>& src,
            std::tuple<std::vector<uint8_t>>& output) override
    {
        auto& [encoded] = output;
        encoded.resize(src.size() * ZL_VARINT_LENGTH_64);
        uint8_t* dst         = encoded.data();
        auto rawSrc          = src.data();
        const auto rawSrcEnd = src.data() + src.size();
#if defined(__GNUC__) && defined(__x86_64__)
        // Benchmark is unstable without alignment on Intel Skylake, this is a
        // temporary fix and we will need a better one
        __asm__(".p2align 6");
#endif
        while (rawSrc < rawSrcEnd) {
            dst += kEncodeFn(*rawSrc++, dst);
        }
        numValues_   = src.size();
        encodedSize_ = ((size_t)(dst - encoded.data()));
    }

    void decode(
            const std::tuple<std::vector<uint8_t>>& src,
            std::vector<Int>& output) override
    {
        auto& [encoded] = src;
        ZL_REQUIRE_LE(encodedSize_, encoded.size());

        output.resize(numValues_);
        uint8_t const* ptr       = encoded.data();
        uint8_t const* const end = encoded.data() + encodedSize_;
        Int* out                 = output.data();
#if defined(__GNUC__) && defined(__x86_64__)
        // Benchmark is unstable without alignment on Intel Skylake, this is a
        // temporary fix and we will need a better one
        __asm__(".p2align 6");
#endif
        while (ptr < end) {
            auto result = kDecodeFn(&ptr, end);
            ZL_REQUIRE_SUCCESS(result);
            *out++ = (Int)ZL_RES_value(result);
        }
        ZL_REQUIRE_EQ(out, output.data() + output.size());
    }

    std::string name() override
    {
        return transformName_;
    }

   private:
    std::string transformName_;

    size_t numValues_;
    size_t encodedSize_;
};

} // namespace zstrong::bench::micro
