// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCH_BY_TAG_ENCODE_DISPATCH_BY_TAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCH_BY_TAG_ENCODE_DISPATCH_BY_TAG_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t

/**
 * Dispatch input @src
 * containing @nbElts of fixed size @eltSize
 * into @nbDstBuffers non-overlapping buffers,
 * which starting positions are stored into array @dstBuffers
 * which must be valid and large enough to receive all their elts.
 *
 * The dispatch is controlled by indexBuffer
 * which is presumed valid and contain at least @nbElts bytes.
 * Note that, by design, this dispatch transform cannot split into more than 256
 * dstBuffers. All values within indexBuffers are presumed < nbDstBuffers.
 *
 * Presuming that all conditions are respected (see above)
 * this function can never fail.
 *
 * On return, @dstBuffers pointers are updated to their end position.
 */
void ZS_DispatchByTag_encode(
        void* restrict dstBuffers[],
        size_t nbDstBuffers,
        const void* restrict src,
        size_t nbElts,
        size_t eltSize,
        const uint8_t* restrict indexBuffer);

#endif
