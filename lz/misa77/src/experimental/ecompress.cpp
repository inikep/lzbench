// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#include "experimental/ecompress_dispatch.h"
#include "experimental/params.h"
#include "misa77/experimental.h"
#include "misa77/misa77.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

namespace misa77
{
    namespace experimental
    {
        uint64_t compress_tuned(const uint8_t* __restrict src,
                                uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap,
                                const param& given)
        {
#if defined(__x86_64__)
            if (__builtin_cpu_supports("avx2"))
                return compress_tuned_avx2(src, src_size, dst, dst_cap, given);
            return compress_tuned_sse2(src, src_size, dst, dst_cap, given);
#else
            return compress_tuned_portable(src, src_size, dst, dst_cap, given);
#endif
        }

        namespace
        {
            // number of times we decode a sample in suggest_homogenous
            constexpr uint64_t iters = 25;
        } // namespace

        param suggest_homogeneous(const uint8_t* __restrict src,
                                  uint64_t src_size,
                                  uint8_t* __restrict dst,
                                  uint64_t dst_cap,
                                  uint8_t option)
        {
            auto fn = &compress_tuned_portable;
#if defined(__x86_64__)
            if (__builtin_cpu_supports("avx2"))
                fn = &compress_tuned_avx2;
            else
                fn = &compress_tuned_sse2;
#endif
            std::vector<uint8_t> decode_buf(decompressed_buffer_bound(src_size));

            // min-of-iters decode time in ns (of the compressed stream currently in `dst`)
            auto decode_time = [&](uint64_t csz) -> uint64_t
            {
                uint64_t here_time = std::numeric_limits<uint64_t>::max();
                for (uint64_t i = 0; i < iters; i++)
                {
                    auto start = std::chrono::steady_clock::now();
                    decompress(dst, csz, decode_buf.data(), decode_buf.size());
                    auto end = std::chrono::steady_clock::now();
                    here_time = std::min(
                        here_time,
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                                .count()));
                }
                return here_time;
            };

            // Ratio baseline
            uint64_t base_csz = fn(src, src_size, dst, dst_cap, params::p_default);

            // Largest acceptable compressed size on the sample (pray that it generalizes).
            uint64_t limit = (option == 1) ? base_csz + base_csz / 20
                                           : base_csz + std::max(base_csz / 5, src_size / 10);

            param suggestion = params::p_default;
            uint64_t best_time = decode_time(base_csz);

            for (const param& p : params::all_candidates)
            {
                uint64_t csz = fn(src, src_size, dst, dst_cap, p);
                if (csz == 0 or csz > limit)
                    continue;
                uint64_t here_time = decode_time(csz);
                if (here_time < best_time)
                    suggestion = p, best_time = here_time;
            }

            return suggestion;
        }

        uint64_t adaptive_compress(const uint8_t* __restrict src,
                                   uint64_t src_size,
                                   uint8_t* __restrict dst,
                                   uint64_t dst_cap,
                                   uint8_t option)
        {
            constexpr uint64_t sample_size = 1
                                             << 20; // 1 MB for now, I think 512 KB will suffice too

            // We just use the prefix for now, look into making this more sophisticated later
            param opt =
                suggest_homogeneous(src, std::min(src_size, sample_size), dst, dst_cap, option);

#if defined(__x86_64__)
            if (__builtin_cpu_supports("avx2"))
                return compress_tuned_avx2(src, src_size, dst, dst_cap, opt);
            return compress_tuned_sse2(src, src_size, dst, dst_cap, opt);
#else
            return compress_tuned_portable(src, src_size, dst, dst_cap, opt);
#endif
        }

        uint64_t yolo_compress(const uint8_t* __restrict src,
                               uint64_t src_size,
                               uint8_t* __restrict dst,
                               uint64_t dst_cap)
        {
            return compress_tuned(src, src_size, dst, dst_cap, params::loose_champion);
        }

    } // namespace experimental
} // namespace misa77