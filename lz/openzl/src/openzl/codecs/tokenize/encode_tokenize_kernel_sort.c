// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/tokenize/encode_tokenize_kernel_sort.h"

static int vsfComparator(const void* lhs, const void* rhs);

#define T VSFKey
#define COMP(a, b) (vsfComparator(a, b) < 0)
#include "openzl/shared/detail/pdqsort-inl.h"

ZL_FORCE_INLINE int vsfComparator(const void* lhs, const void* rhs)
{
    const VSFKey* const lhsPtr = (const VSFKey*)lhs;
    const VSFKey* const rhsPtr = (const VSFKey*)rhs;
    size_t const lhsSize       = lhsPtr->fieldSize;
    size_t const rhsSize       = rhsPtr->fieldSize;

    if (lhsSize == rhsSize) {
        return memcmp(lhsPtr->fieldStart, rhsPtr->fieldStart, lhsSize);
    } else {
        size_t const maxCmpLen = lhsSize < rhsSize ? lhsSize : rhsSize;
        int const cmpRes =
                memcmp(lhsPtr->fieldStart, rhsPtr->fieldStart, maxCmpLen);

        if (cmpRes == 0) {
            return lhsSize < rhsSize ? -1 : 1;
        }
        return cmpRes;
    }
}

void pqdsortVsf(VSFKey* data, size_t nbElts)
{
    pdqsort_branchless(data, data + nbElts);
}
