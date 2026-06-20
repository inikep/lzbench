// Copyright (c) Meta Platforms, Inc. and affiliates.

/* Note : using stdc assert() presumes that disabling is controlled by defining
 * NDEBUG this may have to be changed if there is some other form of centralized
 * debug control. On the other hand, note that a design goal of this transform
 * is to _NOT_ depend on anything from zstrong core.
 */
#include <assert.h>

#include "openzl/codecs/tokenize/decode_tokenize2to1_kernel.h"
#include "openzl/common/debug.h"

size_t ZS_tokenize2to1_decode(
        uint16_t* dst16,
        size_t dst16Capacity,
        const uint8_t* srcIndex8,
        size_t srcIndex8Size,
        const uint16_t* alphabet16,
        size_t alphabet16Size)
{
    (void)dst16Capacity;
    (void)alphabet16Size;
    assert(dst16Capacity >= srcIndex8Size);
    assert(dst16 != NULL);
    assert(srcIndex8 != NULL);
    assert(alphabet16 != NULL);

    for (size_t n = 0; n < srcIndex8Size; n++) {
        uint8_t index = srcIndex8[n];
        assert(index < alphabet16Size);
        dst16[n] = alphabet16[index];
    }

    return srcIndex8Size;
}
