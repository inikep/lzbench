// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/range_pack/encode_range_pack_kernel.h"

#include "openzl/common/assertion.h"
#include "openzl/shared/mem.h"

#define GEN_RANGE_PACK(SrcInt, DstInt)                                      \
    static void rangePack_##SrcInt##_##DstInt(                              \
            DstInt* dst, const SrcInt* src, size_t nbElts, SrcInt minValue) \
    {                                                                       \
        if (minValue) {                                                     \
            for (size_t i = 0; i < nbElts; i++) {                           \
                ZL_ASSERT_GE(src[i], minValue);                             \
                ZL_ASSERT_LE(src[i] - minValue, (SrcInt)((DstInt) - 1));    \
                dst[i] = (DstInt)(src[i] - minValue);                       \
            }                                                               \
        } else {                                                            \
            if (sizeof(SrcInt) == sizeof(DstInt)) {                         \
                memcpy(dst, src, sizeof(SrcInt) * nbElts);                  \
            } else {                                                        \
                for (size_t i = 0; i < nbElts; i++) {                       \
                    ZL_ASSERT_LE(src[i], (SrcInt)((DstInt) - 1));           \
                    dst[i] = (DstInt)src[i];                                \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

GEN_RANGE_PACK(uint64_t, uint64_t)
GEN_RANGE_PACK(uint64_t, uint32_t)
GEN_RANGE_PACK(uint64_t, uint16_t)
GEN_RANGE_PACK(uint64_t, uint8_t)
GEN_RANGE_PACK(uint32_t, uint32_t)
GEN_RANGE_PACK(uint32_t, uint16_t)
GEN_RANGE_PACK(uint32_t, uint8_t)
GEN_RANGE_PACK(uint16_t, uint16_t)
GEN_RANGE_PACK(uint16_t, uint8_t)
GEN_RANGE_PACK(uint8_t, uint8_t)

#undef GEN_RANGE_PACK

void rangePackEncode(
        void* dst,
        size_t dstWidth,
        const void* src,
        size_t srcWidth,
        size_t nbElts,
        size_t srcMinValue)
{
    ZL_ASSERT_GE(srcWidth, dstWidth);

    // Use a macro to define the different cases
#define RANGE_PACK_ENCODE_CASE(SrcInt, DstInt)                      \
    if (srcWidth == sizeof(SrcInt) && dstWidth == sizeof(DstInt)) { \
        rangePack_##SrcInt##_##DstInt(                              \
                (DstInt*)dst,                                       \
                (const SrcInt*)src,                                 \
                nbElts,                                             \
                (SrcInt)srcMinValue);                               \
        return;                                                     \
    }
    RANGE_PACK_ENCODE_CASE(uint64_t, uint64_t)
    RANGE_PACK_ENCODE_CASE(uint64_t, uint32_t)
    RANGE_PACK_ENCODE_CASE(uint64_t, uint16_t)
    RANGE_PACK_ENCODE_CASE(uint64_t, uint8_t)
    RANGE_PACK_ENCODE_CASE(uint32_t, uint32_t)
    RANGE_PACK_ENCODE_CASE(uint32_t, uint16_t)
    RANGE_PACK_ENCODE_CASE(uint32_t, uint8_t)
    RANGE_PACK_ENCODE_CASE(uint16_t, uint16_t)
    RANGE_PACK_ENCODE_CASE(uint16_t, uint8_t)
    RANGE_PACK_ENCODE_CASE(uint8_t, uint8_t)
#undef RANGE_PACK_ENCODE_CASE
    ZL_ASSERT_FAIL(
            "Executing rangePackEncode with mismatched widths - %zu bytes to %zu bytes",
            srcWidth,
            dstWidth);
}
