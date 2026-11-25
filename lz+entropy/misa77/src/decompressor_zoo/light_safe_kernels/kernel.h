// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

#if defined(MISA77_ASM_KERNELS) && defined(__x86_64__)
// Sequence loops in x86_64.inc. Return the final `out`, or null on a rejected stream.
// Disable C++ name mangling.
extern "C"
{
    uint8_t* misa77_light_safe_kernel_avx2(const uint8_t* src,
                                           uint8_t* dst,
                                           const uint8_t* literals,
                                           uint8_t* out_limit) noexcept;
    uint8_t* misa77_light_safe_kernel_sse2(const uint8_t* src,
                                           uint8_t* dst,
                                           const uint8_t* literals,
                                           uint8_t* out_limit) noexcept;
}
#elif defined(MISA77_ASM_KERNELS) && defined(__aarch64__)
// Sequence loop in aarch64.inc. Returns the final `out`, or null on a rejected stream.
extern "C"
{
    uint8_t* misa77_light_safe_kernel_neon(const uint8_t* src,
                                           uint8_t* dst,
                                           const uint8_t* literals,
                                           uint8_t* out_limit) noexcept;
}
#endif
