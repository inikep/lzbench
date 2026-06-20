// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_TRANSFORMS_JSON_EXTRACT_COMMON_JSON_EXTRACT_H
#define CUSTOM_TRANSFORMS_JSON_EXTRACT_COMMON_JSON_EXTRACT_H

#include <cassert>
#include <string_view>

#include "openzl/shared/bits.h"
#include "openzl/shared/portability.h"

namespace zstrong {
/// Start with 'A' because it allows 15 symbols without changing the upper
/// nibble.
enum class Token : char {
    START = 'A',

    INT   = 'A',
    FLOAT = 'B',
    STR   = 'C',
    TRUE  = 'D',
    FALSE = 'E',
    NUL   = 'F',

    END = 'F',
};

size_t constexpr kBlockSize   = 4 * 1024;
size_t constexpr kBitmaskSize = kBlockSize / 64;

/// Set the bit at @p pos in @p bitmask.
inline void setBit(uint64_t* bitmask, size_t pos)
{
    assert(pos < kBlockSize);
    size_t const idx = pos / 64;
    size_t const bit = pos % 64;
    bitmask[idx] |= uint64_t(1) << bit;
}

/// Get the bit at @p pos in @p bitmask.
inline bool getBit(uint64_t const* bitmask, size_t pos)
{
    assert(pos < kBlockSize);
    size_t const idx = pos / 64;
    size_t const bit = pos % 64;
    return (bitmask[idx] & ((uint64_t)1 << bit)) != 0;
}

/**
 * Generic fallback for building a bitmask when vector instructions are
 * unavailable, or near the end of the block.
 *
 * @param bitmask The bitmask we are filling, assumed to be set to zero and
 * pre-filled up to @p offset. Of size @p kBitmaskSize.
 * @param isInSet predicate that tells whether a character in @p src should have
 * the corresponding bit set.
 * @param offset Positions [0, offset) are already set in @p bitmaskA
 * @param extendBlock If set, and the last bit in the bitmask is set, consume
 * bytes in the input until isInSet returns false. This avoids duplicate
 * symbols during encoding.
 *
 * @returns The block of source data that the bitmap covers.
 */
template <typename Pred>
std::string_view buildBitmaskFallback(
        uint64_t* bitmask,
        std::string_view& src,
        Pred&& isInSet,
        size_t offset,
        bool extendBlock = false)
{
    ZL_ASSERT_LE(offset, src.size());
    ZL_ASSERT_LE(offset, kBlockSize);
    size_t blockSize = std::min(src.size(), kBlockSize);
    for (size_t i = offset; i < blockSize; ++i) {
        if (isInSet(src[i])) {
            setBit(bitmask, i);
        }
    }
    if (extendBlock && blockSize > 0 && getBit(bitmask, blockSize - 1)) {
        while (blockSize < src.size() && isInSet(src[blockSize])) {
            ++blockSize;
        }
    }
    auto const block = src.substr(0, blockSize);
    src              = src.substr(blockSize);
    return block;
}

/// Count the number of leading zero bits in a 64-bit chunk of a bitmask.
template <typename Mask>
uint32_t countFirstZeros(Mask mask)
{
    if (ZL_isLittleEndian()) {
        return ZL_ctz64((uint64_t)mask);
    } else {
        return ZL_clz64((uint64_t)mask);
    }
}

/// @returns A mask that skips the first @p n bits in @p mask.
template <typename Mask>
Mask skipN(Mask mask, uint32_t n)
{
    if (ZL_isLittleEndian()) {
        return mask >> n;
    } else {
        return mask << n;
    }
}

template <typename Int>
size_t alignDown(Int val, size_t align)
{
    return (val / align) * align;
}

template <typename Int>
size_t alignUp(Int val, size_t align)
{
    return alignDown(val + align - 1, align);
}
} // namespace zstrong

#endif
