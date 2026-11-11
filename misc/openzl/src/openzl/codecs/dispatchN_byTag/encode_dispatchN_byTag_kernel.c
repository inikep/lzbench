// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation note :
// raw transform == minimal dependency
#include "openzl/codecs/dispatchN_byTag/encode_dispatchN_byTag_kernel.h"
#include <assert.h>
#include <string.h> // memcpy

// sumArrayST is only used in assert() statements, which are disabled in release
// builds (NDEBUG). To avoid unused function warnings in release builds, we
// conditionally compile it.
#ifndef NDEBUG
static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}
#endif

/* ZL_dispatchN_byTag():
 *
 * It's a simple kernel,
 * which loops through the array of size_t @segmentSizes,
 * and copy each segment into its target buffer, decided by @tags.
 *
 * This is a generic variant, using memcpy().
 * If need be, in the future, one could add specialized variants
 * focusing on copying fixed size, for example if all segments sizes are <= 8.
 *
 * Note however that this transform is designed primarily to dispatch
 * a few (several dozens) large segments,
 * in contrast to many thousands small fields,
 * which would deserve a different abstraction (new Variable-size token type).
 *
 * All input parameters are presumed valid.
 * In which case, the transform is necessarily successful.
 * See interface descrition in *.h.
 */
void ZL_dispatchN_byTag(
        void* restrict dstBuffers[],
        const size_t* restrict segmentSizes,
        const unsigned* restrict tags,
        size_t nbSegments,
        const void* src,
        size_t srcSize)
{
    assert(sumArrayST(segmentSizes, nbSegments) == srcSize);
    (void)srcSize;
    if (srcSize) {
        assert(src != NULL);
        assert(tags != NULL);
        assert(segmentSizes != NULL);
        assert(dstBuffers != NULL);
        assert(nbSegments != 0);
    }
    for (size_t n = 0; n < nbSegments; n++) {
        size_t const ssize = segmentSizes[n];
        assert(dstBuffers[tags[n]] != NULL);
        memcpy(dstBuffers[tags[n]], src, ssize);
        src                 = (const char*)src + ssize;
        dstBuffers[tags[n]] = (char*)dstBuffers[tags[n]] + ssize;
    }
}
