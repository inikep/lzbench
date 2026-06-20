// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMMON_PDQSORT_H
#define ZSTRONG_COMMON_PDQSORT_H

#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

void pdqsort1(uint8_t* data, size_t nbElts);
void pdqsort2(uint16_t* data, size_t nbElts);
void pdqsort4(uint32_t* data, size_t nbElts);
void pdqsort8(uint64_t* data, size_t nbElts);

ZL_INLINE void pdqsort(void* data, size_t nbElts, size_t eltSize)
{
    switch (eltSize) {
        case 1:
            pdqsort1((uint8_t*)data, nbElts);
            break;
        case 2:
            pdqsort2((uint16_t*)data, nbElts);
            break;
        case 4:
            pdqsort4((uint32_t*)data, nbElts);
            break;
        case 8:
            pdqsort8((uint64_t*)data, nbElts);
            break;
        default:
            ZL_ASSERT_FAIL("Unsupported element size");
            break;
    }
}

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_PDQSORT_H
