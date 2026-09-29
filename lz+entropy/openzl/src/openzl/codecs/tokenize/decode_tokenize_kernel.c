// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/tokenize/decode_tokenize_kernel.h"

#include <string.h>

#include "openzl/codecs/common/copy.h"
#include "openzl/common/assertion.h"

ZL_FORCE_INLINE uint64_t
readIndexAt(void const* indices, size_t i, size_t idxWidth)
{
    if (idxWidth == 1) {
        return *((uint8_t const*)indices + i);
    }
    if (idxWidth == 2) {
        return *((uint16_t const*)indices + i);
    }
    if (idxWidth == 4) {
        return *((uint32_t const*)indices + i);
    }
    if (idxWidth == 8) {
        return *((uint64_t const*)indices + i);
    }
    ZL_ASSERT_FAIL("Impossible");
    return 0;
}

ZL_FORCE_INLINE bool tokenizeValidateIndicesImpl(
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t idxWidth)
{
    bool bad = false;
    for (size_t i = 0; i < nbElts; ++i) {
        bad |= readIndexAt(indices, i, idxWidth) >= alphabetSize;
    }
    return !bad;
}

#define GEN_TOKENIZE_VALIDATE_INDICES(idxWidth)                      \
    ZL_FORCE_NOINLINE bool tokenizeValidateIndices##idxWidth(        \
            size_t alphabetSize, void const* indices, size_t nbElts) \
    {                                                                \
        return tokenizeValidateIndicesImpl(                          \
                alphabetSize, indices, nbElts, idxWidth);            \
    }

GEN_TOKENIZE_VALIDATE_INDICES(1)
GEN_TOKENIZE_VALIDATE_INDICES(2)
GEN_TOKENIZE_VALIDATE_INDICES(4)
GEN_TOKENIZE_VALIDATE_INDICES(8)

bool ZS_tokenizeValidateIndices(
        size_t alphabetSize,
        const void* indices,
        size_t nbElts,
        size_t idxWidth)
{
    switch (idxWidth) {
        case 1:
            return tokenizeValidateIndices1(alphabetSize, indices, nbElts);
        case 2:
            return tokenizeValidateIndices2(alphabetSize, indices, nbElts);
        case 4:
            return tokenizeValidateIndices4(alphabetSize, indices, nbElts);
        case 8:
            return tokenizeValidateIndices8(alphabetSize, indices, nbElts);
        default:
            ZL_ASSERT_FAIL("Bad idxWidth = %u", (unsigned)idxWidth);
            return false;
    }
}

ZL_FORCE_INLINE void writeSymbolAt(
        void* dst,
        size_t dstIdx,
        void const* alphabet,
        size_t idx,
        size_t eltWidth)
{
    void* const dstPtr       = (uint8_t*)dst + dstIdx * eltWidth;
    void const* const srcPtr = (uint8_t const*)alphabet + idx * eltWidth;
    memcpy(dstPtr, srcPtr, eltWidth);
}

ZL_FORCE_INLINE bool tokenizeDecodeEltWidthIdxWidth(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    for (size_t i = 0; i < nbElts; ++i) {
        uint64_t index = readIndexAt(indices, i, idxWidth);
        // For speed don't return an error if an out-of-bounds index is
        // detected, simply replace it with the 0 symbol. We've already handled
        // the case where alphabetSize == 0.
        ZL_ASSERT_NE(alphabetSize, 0, "Already validated");
        index = ZL_UNLIKELY(index >= alphabetSize) ? 0 : index;
        writeSymbolAt(dst, i, alphabet, index, eltWidth);
    }
    return true;
}

ZL_FORCE_INLINE bool tokenizeDecodeEltWidth(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    if (idxWidth == 1) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 1);
    }
    if (idxWidth == 2) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 2);
    }
    if (idxWidth == 4) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 4);
    }
    if (idxWidth == 8) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 8);
    }
    ZL_ASSERT_FAIL("Impossible");
    return false;
}

#define GEN_TOKENIZE_DECODE(eltWidth)                \
    ZL_FORCE_NOINLINE bool tokenizeDecode##eltWidth( \
            void* dst,                               \
            void const* alphabet,                    \
            size_t alphabetSize,                     \
            void const* indices,                     \
            size_t nbElts,                           \
            size_t idxWidth)                         \
    {                                                \
        return tokenizeDecodeEltWidth(               \
                dst,                                 \
                alphabet,                            \
                alphabetSize,                        \
                indices,                             \
                nbElts,                              \
                eltWidth,                            \
                idxWidth);                           \
    }

GEN_TOKENIZE_DECODE(1)
GEN_TOKENIZE_DECODE(2)
GEN_TOKENIZE_DECODE(4)
GEN_TOKENIZE_DECODE(8)

ZL_FORCE_NOINLINE bool tokenizeDecodeGeneric(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    return tokenizeDecodeEltWidth(
            dst, alphabet, alphabetSize, indices, nbElts, eltWidth, idxWidth);
}

bool ZS_tokenizeDecode(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    if (alphabetSize == 0) {
        return nbElts == 0;
    }
    switch (eltWidth) {
        case 1:
            return tokenizeDecode1(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 2:
            return tokenizeDecode2(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 4:
            return tokenizeDecode4(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 8:
            return tokenizeDecode8(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        default:
            return tokenizeDecodeGeneric(
                    dst,
                    alphabet,
                    alphabetSize,
                    indices,
                    nbElts,
                    eltWidth,
                    idxWidth);
    }
}

size_t ZS_tokenizeComputeVSFContentSize(
        const uint8_t* const indices,
        size_t const idxWidth,
        size_t const nbElts,
        const uint32_t* const alphabetFieldSizes,
        size_t alphabetSize)
{
    ZL_ASSERT(ZS_tokenizeValidateIndices(
            alphabetSize, indices, nbElts, idxWidth));
    ZL_ASSERT_EQ(sizeof(size_t), 8);
    size_t totalSize = 0;
    for (size_t i = 0; i < nbElts; ++i) {
        totalSize += alphabetFieldSizes[readIndexAt(indices, i, idxWidth)];
    }
    return totalSize;
}

size_t ZS_tokenizeVSFDecodeWorkspaceSize(
        size_t const alphabetSize,
        size_t const alphabetFieldSizesSum)
{
    size_t alphabetStartsSize = sizeof(uint8_t*) * alphabetSize;
    size_t alphabetWildcopyBufferSize =
            alphabetFieldSizesSum + ZS_WILDCOPY_OVERLENGTH;
    return alphabetStartsSize + alphabetWildcopyBufferSize;
}

void ZS_tokenizeVSFDecode(
        const uint8_t* const alphabet,
        size_t const alphabetSize,
        const uint8_t* const indices,
        const uint32_t* const alphabetFieldSizes,
        size_t const alphabetFieldSizesSum,
        uint8_t* const out,
        uint32_t* const dstFieldSizes,
        size_t const dstNbElts,
        size_t const dstNbBytes,
        size_t const idxWidth,
        void* workspace)
{
    ZL_ASSERT(ZS_tokenizeValidateIndices(
            alphabetSize, indices, dstNbElts, idxWidth));

    // Move alphabet into a larger buffer to guarantee wildcopy read safety
    uint8_t* const alphabetBuffer =
            (uint8_t*)workspace + sizeof(uint8_t*) * alphabetSize;
    memcpy(alphabetBuffer, alphabet, alphabetFieldSizesSum);

    // Store the start of each token start for quick access
    const uint8_t** const alphabetStarts = (const uint8_t**)workspace;
    const uint8_t* nextSrcEltPtr         = alphabetBuffer;
    for (size_t i = 0; i < alphabetSize; ++i) {
        alphabetStarts[i] = nextSrcEltPtr;
        nextSrcEltPtr += alphabetFieldSizes[i];
    }

    // Find until when it's safe to wildcopy
    size_t wildcopySafeIndexLimit = dstNbElts > 0 ? dstNbElts - 1 : 0;
    {
        size_t tailSum = 0;
        for (; wildcopySafeIndexLimit > 0; --wildcopySafeIndexLimit) {
            uint64_t const idx =
                    readIndexAt(indices, wildcopySafeIndexLimit, idxWidth);
            ZL_ASSERT_LT(idx, alphabetSize);
            uint32_t const eltWidth = alphabetFieldSizes[idx];
            tailSum += eltWidth;
            if (tailSum > ZS_WILDCOPY_OVERLENGTH) {
                break;
            }
        }
    }
    // Copy tokens to output, use wildcopy until we can't do so safely
    uint8_t* nextDstEltPtr = out;
#if defined(__GNUC__) && defined(__x86_64__)
    // Set alignment to maintain stability in benchmarks
    __asm__(".p2align 6");
#endif
    for (size_t i = 0; i < wildcopySafeIndexLimit; ++i) {
        uint64_t const idx      = readIndexAt(indices, i, idxWidth);
        uint32_t const eltWidth = alphabetFieldSizes[idx];
        ZL_ASSERT_LE(
                nextDstEltPtr + eltWidth + ZS_WILDCOPY_OVERLENGTH,
                out + dstNbBytes);
        ZS_wildcopy(
                nextDstEltPtr, alphabetStarts[idx], eltWidth, ZS_wo_no_overlap);
        nextDstEltPtr += eltWidth;
        dstFieldSizes[i] = eltWidth;
    }
    for (size_t i = wildcopySafeIndexLimit; i < dstNbElts; ++i) {
        uint64_t const idx      = readIndexAt(indices, i, idxWidth);
        uint32_t const eltWidth = alphabetFieldSizes[idx];
        memcpy(nextDstEltPtr, alphabetStarts[idx], eltWidth);
        nextDstEltPtr += eltWidth;
        dstFieldSizes[i] = eltWidth;
    }
}
