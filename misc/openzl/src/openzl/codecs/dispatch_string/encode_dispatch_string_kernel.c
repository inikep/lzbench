// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/dispatch_string/encode_dispatch_string_kernel.h"

#include <assert.h>
#include <string.h> // memset

#include "openzl/codecs/dispatch_string/common_dispatch_string.h"

void ZL_DispatchString_encode(
        uint8_t nbDsts,
        void** restrict dstBuffers,
        uint32_t** restrict dstStrLens,
        size_t dstSizes[],
        const void* restrict src,
        const uint32_t srcStrLens[],
        const size_t nbStrs,
        const uint8_t outputIndices[])
{
    for (size_t i = 0; i < nbDsts; ++i) {
        assert(dstBuffers[i] != NULL);
        assert(dstStrLens[i] != NULL);
    }

    // check for the degenerate case of empty input & output
    if (nbDsts != 0) {
        memset(dstSizes, 0, nbDsts * sizeof(size_t));
    }

    const char* srcPtr = src;
    void* dstPtrs[ZL_DISPATCH_STRING_MAX_DISPATCHES_V20];
    for (size_t i = 0; i < nbDsts; ++i) {
        dstPtrs[i] = dstBuffers[i];
    }

    // find the first idx where we aren't guaranteed to be able to read
    // ZL_DISPATCH_STRING_BLK_SIZE bytes without overflowing
    size_t firstNonBlkIdx = 0;
    size_t endSum         = 0;
    if (nbStrs != 0) { // check for degenerate case of empty input & output
        for (size_t i = nbStrs - 1; i > 0; --i) {
            endSum += srcStrLens[i];
            if (endSum >= ZL_DISPATCH_STRING_BLK_SIZE) {
                firstNonBlkIdx = i + 1;
                break;
            }
        }
    }

    for (size_t i = 0; i < firstNonBlkIdx; ++i) {
        const uint8_t dstIdx = outputIndices[i];
        assert(dstIdx < nbDsts);
        const size_t currN        = dstSizes[dstIdx];
        const uint32_t currStrLen = srcStrLens[i];
        dstStrLens[dstIdx][currN] = currStrLen;
        if (currStrLen <= ZL_DISPATCH_STRING_BLK_SIZE) {
            memcpy(dstPtrs[dstIdx], srcPtr, ZL_DISPATCH_STRING_BLK_SIZE);
        } else {
            memcpy(dstPtrs[dstIdx], srcPtr, currStrLen);
        }
        srcPtr += currStrLen;
        dstPtrs[dstIdx] = (char*)dstPtrs[dstIdx] + currStrLen;
        ++dstSizes[dstIdx];
    }
    // end condition
    for (size_t i = firstNonBlkIdx; i < nbStrs; ++i) {
        const uint8_t dstIdx = outputIndices[i];
        assert(dstIdx < nbDsts);
        const size_t currN        = dstSizes[dstIdx];
        const uint32_t currStrLen = srcStrLens[i];
        dstStrLens[dstIdx][currN] = currStrLen;
        memcpy(dstPtrs[dstIdx], srcPtr, currStrLen);
        srcPtr += currStrLen;
        dstPtrs[dstIdx] = (char*)dstPtrs[dstIdx] + currStrLen;
        ++dstSizes[dstIdx];
    }
    return;
}

void ZL_DispatchString_encode16(
        uint16_t nbDsts,
        void** restrict dstBuffers,
        uint32_t** restrict dstStrLens,
        size_t dstSizes[],
        const void* restrict src,
        const uint32_t srcStrLens[],
        const size_t nbStrs,
        const uint16_t outputIndices[])
{
    for (size_t i = 0; i < nbDsts; ++i) {
        assert(dstBuffers[i] != NULL);
        assert(dstStrLens[i] != NULL);
    }

    // check for the degenerate case of empty input & output
    if (nbDsts != 0) {
        memset(dstSizes, 0, nbDsts * sizeof(size_t));
    }

    const char* srcPtr = src;
    void* dstPtrs[ZL_DISPATCH_STRING_MAX_DISPATCHES];
    for (size_t i = 0; i < nbDsts; ++i) {
        dstPtrs[i] = dstBuffers[i];
    }

    // find the first idx where we aren't guaranteed to be able to read
    // ZL_DISPATCH_STRING_BLK_SIZE bytes without overflowing
    size_t firstNonBlkIdx = 0;
    size_t endSum         = 0;
    if (nbStrs != 0) { // check for degenerate case of empty input & output
        for (size_t i = nbStrs - 1; i > 0; --i) {
            endSum += srcStrLens[i];
            if (endSum >= ZL_DISPATCH_STRING_BLK_SIZE) {
                firstNonBlkIdx = i + 1;
                break;
            }
        }
    }

    for (size_t i = 0; i < firstNonBlkIdx; ++i) {
        const uint16_t dstIdx = outputIndices[i];
        assert(dstIdx < nbDsts);
        const size_t currN        = dstSizes[dstIdx];
        const uint32_t currStrLen = srcStrLens[i];
        dstStrLens[dstIdx][currN] = currStrLen;
        if (currStrLen <= ZL_DISPATCH_STRING_BLK_SIZE) {
            memcpy(dstPtrs[dstIdx], srcPtr, ZL_DISPATCH_STRING_BLK_SIZE);
        } else {
            memcpy(dstPtrs[dstIdx], srcPtr, currStrLen);
        }
        srcPtr += currStrLen;
        dstPtrs[dstIdx] = (char*)dstPtrs[dstIdx] + currStrLen;
        ++dstSizes[dstIdx];
    }
    // end condition
    for (size_t i = firstNonBlkIdx; i < nbStrs; ++i) {
        const uint16_t dstIdx = outputIndices[i];
        assert(dstIdx < nbDsts);
        const size_t currN        = dstSizes[dstIdx];
        const uint32_t currStrLen = srcStrLens[i];
        dstStrLens[dstIdx][currN] = currStrLen;
        memcpy(dstPtrs[dstIdx], srcPtr, currStrLen);
        srcPtr += currStrLen;
        dstPtrs[dstIdx] = (char*)dstPtrs[dstIdx] + currStrLen;
        ++dstSizes[dstIdx];
    }
    return;
}
