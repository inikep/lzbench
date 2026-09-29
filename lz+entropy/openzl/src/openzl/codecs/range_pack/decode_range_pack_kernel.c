// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/range_pack/decode_range_pack_kernel.h"

#include "openzl/common/debug.h"
#include "openzl/common/logging.h"
#include "openzl/shared/mem.h"

#define GEN_RANGE_UNPACK(DstInt, SrcInt)                                    \
    static void rangeUnpack_##SrcInt##_##DstInt(                            \
            DstInt* dst, const SrcInt* src, size_t nbElts, DstInt minValue) \
    {                                                                       \
        if (minValue) {                                                     \
            for (size_t i = 0; i < nbElts; i++) {                           \
                dst[i] = (DstInt)((DstInt)src[i] + minValue);               \
            }                                                               \
        } else {                                                            \
            if (sizeof(SrcInt) == sizeof(DstInt)) {                         \
                memcpy(dst, src, sizeof(SrcInt) * nbElts);                  \
            } else {                                                        \
                for (size_t i = 0; i < nbElts; i++) {                       \
                    ZL_ASSERT_LE((DstInt)src[i], (DstInt) - 1);             \
                    dst[i] = (DstInt)src[i];                                \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

GEN_RANGE_UNPACK(uint64_t, uint64_t)
GEN_RANGE_UNPACK(uint64_t, uint32_t)
GEN_RANGE_UNPACK(uint64_t, uint16_t)
GEN_RANGE_UNPACK(uint64_t, uint8_t)
GEN_RANGE_UNPACK(uint32_t, uint32_t)
GEN_RANGE_UNPACK(uint32_t, uint16_t)
GEN_RANGE_UNPACK(uint32_t, uint8_t)
GEN_RANGE_UNPACK(uint16_t, uint16_t)
GEN_RANGE_UNPACK(uint16_t, uint8_t)
GEN_RANGE_UNPACK(uint8_t, uint8_t)

#undef GEN_RANGE_UNPACK

void rangePackDecode(
        void* dst,
        size_t dstWidth,
        const void* src,
        size_t srcWidth,
        size_t nbElts,
        size_t dstMinValue)
{
    ZL_ASSERT_LE(srcWidth, dstWidth);

    // Use a macro to define the different cases
#define RANGE_PACK_DECODE_CASE(DstInt, SrcInt)                      \
    if (srcWidth == sizeof(SrcInt) && dstWidth == sizeof(DstInt)) { \
        rangeUnpack_##SrcInt##_##DstInt(                            \
                (DstInt*)dst,                                       \
                (const SrcInt*)src,                                 \
                nbElts,                                             \
                (DstInt)dstMinValue);                               \
        return;                                                     \
    }
    RANGE_PACK_DECODE_CASE(uint64_t, uint64_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint32_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint32_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint16_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint16_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint8_t, uint8_t)
#undef RANGE_PACK_DECODE_CASE
    ZL_ASSERT_FAIL(
            "Executing rangePackDecode with mismatched widths - %zu bytes to %zu bytes",
            srcWidth,
            dstWidth);
}
