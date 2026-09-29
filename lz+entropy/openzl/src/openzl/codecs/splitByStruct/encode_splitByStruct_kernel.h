// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_SPLITBYSTRUCT_ENCODE_SPLIT_BY_STRUCT_KERNEL_H
#define ZSTRONG_TRANSFORMS_SPLITBYSTRUCT_ENCODE_SPLIT_BY_STRUCT_KERNEL_H

#include <stddef.h> // size_t

/* ZS_dispatchArrayFixedSizeStruct():
 *
 * dispatch input @src
 * into @nbStructMembers non-overlapping buffers,
 * which starting positions are stored in array @dstBuffers
 * which must contain @nbStructMembers pointers.
 *
 * The fixed size of each member must be provided in array @structMemberSizes,
 * which must contain @nbStructMembers elements.
 *
 * All provided arrays are presumed correctly sized.
 * Input @src is presumed valid and correctly sized,
 * which implies that @srcSize is an exact multiple of structure size.
 * All destination buffers are also presumed correctly sized.
 * The size of each buffer is known before hand,
 * since it's the size of each struct members multiplied by nb of structures.
 *
 * Presuming that all conditions are respected (see above)
 * this function can never fail.
 *
 * On return, @dstBuffers pointers are updated to their end position.
 */

void ZS_dispatchArrayFixedSizeStruct(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t srcSize,
        const size_t* structMemberSizes);

#endif
