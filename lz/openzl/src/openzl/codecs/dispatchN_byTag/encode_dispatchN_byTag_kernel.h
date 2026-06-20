// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCHN_BYTAG_ENCODE_DISPATCHN_BYTAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCHN_BYTAG_ENCODE_DISPATCHN_BYTAG_KERNEL_H

#include <stddef.h> // size_t

/* ZL_dispatchN_byTag():
 *
 * dispatch input @src
 * into segments of variable size, decided by @segmentSizes[],
 * which are grouped into outputs of same id, decided by @tags[].
 *
 * **All parameters are presumed valid**, in particular :
 * - Input @src is presumed valid and correctly sized,
 * - sum(@segmentSizes[]) == @srcSize (necessarily)
 * - @segmentSizes[] and @tags[] have same nb of elts, aka @nbSegments
 * - all values in @tags[] reference existing index in @dstBuffers
 * - Output buffers, referenced in @dstBuffers[], are valid and correctly sized.
 *
 * Presuming that all conditions are respected (see above)
 * this function will necessarily succeed.
 */

void ZL_dispatchN_byTag(
        void* restrict dstBuffers[],
        const size_t* restrict segmentSizes,
        const unsigned* restrict tags,
        size_t nbSegments,
        const void* src,
        size_t srcSize);

#endif
