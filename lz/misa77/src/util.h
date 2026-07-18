// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>
#include <cstring>

// ISA-agnostic helpers.
// See src/isa for ISA-specialized helpers

namespace misa77
{
    [[gnu::always_inline]]
    inline uint16_t loadu2(const uint8_t* ptr)
    {
        uint16_t v;
        memcpy(&v, ptr, 2);
        return v;
    }

    [[gnu::always_inline]]
    inline void storeu2(uint8_t* ptr, const uint16_t val)
    {
        memcpy(ptr, &val, 2);
    }

    [[gnu::always_inline]]
    inline uint32_t loadu4(const uint8_t* ptr)
    {
        uint32_t v;
        memcpy(&v, ptr, 4);
        return v;
    }

    [[gnu::always_inline]]
    inline void storeu4(uint8_t* ptr, const uint32_t val)
    {
        memcpy(ptr, &val, 4);
    }

    [[gnu::always_inline]]
    inline uint64_t loadu8(const uint8_t* ptr)
    {
        uint64_t v;
        memcpy(&v, ptr, 8);
        return v;
    }

    [[gnu::always_inline]]
    inline void storeu8(uint8_t* ptr, const uint64_t val)
    {
        memcpy(ptr, &val, 8);
    }
} // namespace misa77