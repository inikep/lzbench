// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_CONVERSION_COMMON_ENDIANNESS_KERNEL_H
#define ZSTRONG_TRANSFORMS_CONVERSION_COMMON_ENDIANNESS_KERNEL_H

#include "openzl/common/cursor.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

ZL_INLINE void ZS_Endianness_transform(
        ZL_WC* dst,
        ZL_RC* src,
        const ZL_Endianness dstEndianness,
        const ZL_Endianness srcEndianness,
        size_t eltSize)
{
    const size_t srcSize = ZL_RC_avail(src);

    ZL_ASSERT(eltSize == 1 || eltSize == 2 || eltSize == 4 || eltSize == 8);
    ZL_ASSERT_EQ((size_t)(srcSize % eltSize), 0);
    ZL_WC_ASSERT_HAS(dst, ZL_RC_avail(src));

    if (dstEndianness == srcEndianness || eltSize == 1) {
        ZL_WC_moveAll(dst, src);
        return;
    }

    const uint8_t* srcPtr = ZL_RC_ptr(src);
    uint8_t* dstPtr       = ZL_WC_ptr(dst);

    switch (eltSize) {
        case 2: {
            const size_t nbElts = srcSize / eltSize;
            for (size_t i = 0; i < nbElts; i++) {
                ZL_write16(dstPtr, ZL_swap16(ZL_read16(srcPtr)));
                srcPtr += eltSize;
                dstPtr += eltSize;
            }
            break;
        }
        case 4: {
            const size_t nbElts = srcSize / eltSize;
            for (size_t i = 0; i < nbElts; i++) {
                ZL_write32(dstPtr, ZL_swap32(ZL_read32(srcPtr)));
                srcPtr += eltSize;
                dstPtr += eltSize;
            }
            break;
        }
        case 8: {
            const size_t nbElts = srcSize / eltSize;
            for (size_t i = 0; i < nbElts; i++) {
                ZS_write64(dstPtr, ZL_swap64(ZL_read64(srcPtr)));
                srcPtr += eltSize;
                dstPtr += eltSize;
            }
            break;
        }
        default:
            ZL_ASSERT_FAIL("Illegal element size for endianness conversion.");
            break;
    }

    ZL_RC_advance(src, srcSize);
    ZL_WC_advance(dst, srcSize);
}

ZL_END_C_DECLS

#endif
