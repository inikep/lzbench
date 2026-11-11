// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <string.h> // memcpy

#include "encode_dispatch_by_tag_kernel.h"

// indexBuffer is valid and contains at least nbElts bytes
// All values within indexBuffers are presumed < nbDstBuffers
// All buffers in dstBuffers are presumed allocated, and have enough capacity
// (not checked) Note : the function doesn't return how many elts were written
// into each dstBuffer,
//        because it's presumed already known from indexBuffer's histogram.
//        but it modifies the values of dstBufferPtrs
//        which point at the end of their filled buffer on return.
// @return : 0 if success, non-0 if error
static inline void ZS_DispatchByTag_encode_kernel(
        void* restrict dstBufferPtrs[],
        size_t nbDstBuffers,
        const void* restrict srcBuffer,
        size_t nbElts,
        size_t eltSize,
        const uint8_t* restrict indexBuffer)
{
    for (size_t n = 0; n < nbDstBuffers; n++) {
        assert(dstBufferPtrs[n] != NULL);
    }
    assert(indexBuffer != NULL);

    const char* srcPtr = srcBuffer;

    for (size_t n = 0; n < nbElts; n++) {
        size_t const idx = indexBuffer[n];
        assert(idx < nbDstBuffers);
        memcpy(dstBufferPtrs[idx], srcPtr, eltSize);
        dstBufferPtrs[idx] = (char*)dstBufferPtrs[idx] + eltSize;
        srcPtr += eltSize;
    }
}

void ZS_DispatchByTag_encode(
        void* restrict dstBuffers[],
        size_t nbDstBuffers,
        const void* restrict src,
        size_t nbElts,
        size_t eltSize,
        const uint8_t* restrict indexBuffer)
{
    switch (eltSize) {
        /* specialized variants, for faster speed
         * on my laptop : splitBy4 : 3.2 GB/s
         *                splitBy8 : 5.5 GB/s
         * vs generic splitBy4 : 1.1 GB/s
         *    generic splitBy8 : 3.1 GB/s*/
        case 4:
            ZS_DispatchByTag_encode_kernel(
                    dstBuffers, nbDstBuffers, src, nbElts, 4, indexBuffer);
            return;
        case 8:
            ZS_DispatchByTag_encode_kernel(
                    dstBuffers, nbDstBuffers, src, nbElts, 8, indexBuffer);
            return;
        /* generic variant, any eltSize (slower) */
        default:
            ZS_DispatchByTag_encode_kernel(
                    dstBuffers,
                    nbDstBuffers,
                    src,
                    nbElts,
                    eltSize,
                    indexBuffer);
    }
}
