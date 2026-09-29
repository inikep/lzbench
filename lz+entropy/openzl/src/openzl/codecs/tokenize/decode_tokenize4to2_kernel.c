// Copyright (c) Meta Platforms, Inc. and affiliates.

/* Note : using stdc assert() presumes that disabling is controlled by defining
 * NDEBUG this may have to be changed if there is some other form of centralized
 * debug control. On the other hand, note that a design goal of this transform
 * is to _NOT_ depend on anything from zstrong core.
 */
#include <assert.h>

#include "openzl/codecs/tokenize/decode_tokenize4to2_kernel.h"

size_t ZS_tokenize4to2_decode(
        uint32_t* dst32,
        size_t dst32Capacity,
        const uint16_t* srcIndex16,
        size_t srcIndex16Size,
        const uint32_t* alphabet32,
        size_t alphabet32Size)
{
    if (srcIndex16Size == 0) {
        return 0;
    }
    (void)alphabet32Size;
    (void)dst32Capacity;
    assert(dst32Capacity >= srcIndex16Size);
    assert(dst32 != NULL);
    assert(srcIndex16 != NULL);
    assert(alphabet32 != NULL);

    for (size_t n = 0; n < srcIndex16Size; n++) {
        assert(srcIndex16[n] < alphabet32Size);
        dst32[n] = alphabet32[srcIndex16[n]];
    }

    return srcIndex16Size;
}
