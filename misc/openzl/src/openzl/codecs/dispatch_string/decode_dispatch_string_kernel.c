// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/dispatch_string/decode_dispatch_string_kernel.h"

#include <assert.h>
#include <string.h> // memset

#include "openzl/codecs/dispatch_string/common_dispatch_string.h"

void ZL_DispatchString_decode(
        void* restrict dst,
        uint32_t dstStrLens[],
        size_t dstNbStrs,
        const uint8_t nbSrcs,
        const char* const* const restrict srcBuffers,
        const uint32_t* const* const restrict srcStrLens,
        const size_t srcNbStrs[],
        const uint8_t inputIndices[])
{
    const char* srcPtrs[ZL_DISPATCH_STRING_MAX_DISPATCHES_V20];
    for (size_t i = 0; i < nbSrcs; ++i) {
        srcPtrs[i] = srcBuffers[i];
    }

    size_t currSrc[ZL_DISPATCH_STRING_MAX_DISPATCHES_V20]              = { 0 };
    size_t firstNonblockCopyIdx[ZL_DISPATCH_STRING_MAX_DISPATCHES_V20] = { 0 };

    assert(srcNbStrs != NULL);
    for (size_t i = 0; i < nbSrcs; ++i) {
        assert(srcStrLens[i] != NULL);
        size_t cumLen = 0;
        if (srcNbStrs[i] == 0) { // special case of empty src
            continue;
        }
        for (size_t idx = srcNbStrs[i] - 1; idx > 0; --idx) {
            cumLen += srcStrLens[i][idx];
            if (cumLen >= ZL_DISPATCH_STRING_BLK_SIZE) {
                firstNonblockCopyIdx[i] =
                        idx + 1; // idx is the last block copy, so idx + 1 is
                                 // the first non-block copy
                break;
            }
        }
    }

    for (size_t i = 0; i < dstNbStrs; ++i) {
        const uint8_t srcIndex = inputIndices[i];
        assert(srcIndex < nbSrcs);
        const size_t currIdx  = currSrc[srcIndex];
        const uint32_t strLen = srcStrLens[srcIndex][currIdx];
        if (strLen <= ZL_DISPATCH_STRING_BLK_SIZE
            && currIdx < firstNonblockCopyIdx[srcIndex]) {
            memcpy(dst, srcPtrs[srcIndex], ZL_DISPATCH_STRING_BLK_SIZE);
        } else {
            memcpy(dst, srcPtrs[srcIndex], strLen);
        }
        dstStrLens[i] = strLen;
        dst           = (char*)dst + strLen;
        ++currSrc[srcIndex];
        srcPtrs[srcIndex] += strLen;
    }
}

void ZL_DispatchString_decode16(
        void* restrict dst,
        uint32_t dstStrLens[],
        size_t dstNbStrs,
        const uint16_t nbSrcs,
        const char* const* const restrict srcBuffers,
        const uint32_t* const* const restrict srcStrLens,
        const size_t srcNbStrs[],
        const uint16_t inputIndices[])
{
    const char* srcPtrs[ZL_DISPATCH_STRING_MAX_DISPATCHES];
    for (size_t i = 0; i < nbSrcs; ++i) {
        srcPtrs[i] = srcBuffers[i];
    }

    size_t currSrc[ZL_DISPATCH_STRING_MAX_DISPATCHES]              = { 0 };
    size_t firstNonblockCopyIdx[ZL_DISPATCH_STRING_MAX_DISPATCHES] = { 0 };

    assert(srcNbStrs != NULL);
    for (size_t i = 0; i < nbSrcs; ++i) {
        assert(srcStrLens[i] != NULL);
        size_t cumLen = 0;
        if (srcNbStrs[i] == 0) { // special case of empty src
            continue;
        }
        for (size_t idx = srcNbStrs[i] - 1; idx > 0; --idx) {
            cumLen += srcStrLens[i][idx];
            if (cumLen >= ZL_DISPATCH_STRING_BLK_SIZE) {
                firstNonblockCopyIdx[i] =
                        idx + 1; // idx is the last block copy, so idx + 1 is
                                 // the first non-block copy
                break;
            }
        }
    }

    for (size_t i = 0; i < dstNbStrs; ++i) {
        const uint16_t srcIndex = inputIndices[i];
        assert(srcIndex < nbSrcs);
        const size_t currIdx  = currSrc[srcIndex];
        const uint32_t strLen = srcStrLens[srcIndex][currIdx];
        if (strLen <= ZL_DISPATCH_STRING_BLK_SIZE
            && currIdx < firstNonblockCopyIdx[srcIndex]) {
            memcpy(dst, srcPtrs[srcIndex], ZL_DISPATCH_STRING_BLK_SIZE);
        } else {
            memcpy(dst, srcPtrs[srcIndex], strLen);
        }
        dstStrLens[i] = strLen;
        dst           = (char*)dst + strLen;
        ++currSrc[srcIndex];
        srcPtrs[srcIndex] += strLen;
    }
}
