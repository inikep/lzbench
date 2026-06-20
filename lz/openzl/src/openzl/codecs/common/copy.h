// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_COPY_H
#define ZS_COMMON_COPY_H

#include <string.h>

#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

ZL_INLINE void ZS_copy4(void* dst, const void* src)
{
    memcpy(dst, src, 4);
}
ZL_INLINE void ZS_copy8(void* dst, const void* src)
{
    memcpy(dst, src, 8);
}
ZL_INLINE void ZS_copy16(void* dst, const void* src)
{
    memcpy(dst, src, 16);
}

#define ZS_COPY(dst, src, n)  \
    do {                      \
        ZS_copy##n(dst, src); \
        dst += n;             \
        src += n;             \
    } while (0)

typedef enum {
    ZS_wo_no_overlap,
    ZS_wo_src_before_dst,
    /*  ZS_wo_dst_before_src, */
} ZS_wildcopyOverlap;

#define ZS_WILDCOPY_OVERLENGTH 32
#define ZS_WILDCOPY_VECLEN 16

/*! ZS_overlapCopy8() :
 *  Copies 8 bytes from ip to op and updates op and ip where ip <= op.
 *  If the offset is < 8 then the offset is spread to at least 8 bytes.
 *
 *  Precondition: *ip <= *op
 *  Postcondition: *op - *op >= 8
 */
ZL_FORCE_INLINE void
ZS_overlapCopy8(uint8_t** op, uint8_t const** ip, size_t offset)
{
    ZL_ASSERT(*ip <= *op);
    if (offset < 8) {
        /* close range match, overlap */
        static const int dec32table[] = { 0, 1, 2, 1, 4, 4, 4, 4 }; /* added */
        static const int dec64table[] = {
            8, 8, 8, 7, 8, 9, 10, 11
        }; /* subtracted */
        int const sub2 = dec64table[offset];
        (*op)[0]       = (*ip)[0];
        (*op)[1]       = (*ip)[1];
        (*op)[2]       = (*ip)[2];
        (*op)[3]       = (*ip)[3];
        *ip += dec32table[offset];
        ZS_copy4(*op + 4, *ip);
        *ip -= sub2;
    } else {
        ZS_copy8(*op, *ip);
    }
    *ip += 8;
    *op += 8;
    ZL_ASSERT(*op - *ip >= 8);
}

/*! ZS_wildcopy() :
 *  Custom version of memcpy(), can over read/write up to ZS_WILDCOPY_OVERLENGTH
 *  bytes (if length==0)
 *  @param ovtype controls the overlap detection
 *         - ZS_no_overlap: The source and destination are guaranteed to be at
 *           least ZS_WILDCOPY_VECLEN bytes apart.
 *         - ZS_overlap_src_before_dst: The src and dst may overlap. The src
 *           buffer must be before the dst buffer.
 */
ZL_FORCE_INLINE ZL_FLATTEN_ATTR void ZS_wildcopy(
        void* dst,
        const void* src,
        ptrdiff_t length,
        ZS_wildcopyOverlap const ovtype)
{
    ptrdiff_t diff      = (uint8_t*)dst - (const uint8_t*)src;
    const uint8_t* ip   = (const uint8_t*)src;
    uint8_t* op         = (uint8_t*)dst;
    uint8_t* const oend = op + length;

    if (ovtype == ZS_wo_src_before_dst && diff < ZS_WILDCOPY_VECLEN) {
        /* Handle short offset copies. */
        ZL_ASSERT(diff >= 0);
        ZS_overlapCopy8(&op, &ip, (size_t)diff);
        if (op >= oend)
            return;
        do {
            ZS_COPY(op, ip, 8);
            ZS_COPY(op, ip, 8);
        } while (op < oend);
    } else {
        ZL_ASSERT(diff >= ZS_WILDCOPY_VECLEN || diff <= -ZS_WILDCOPY_VECLEN);
        /* Separate the first two COPY16() call because the copy length is
         * almost certain to be short, so the branches have different
         * probabilities.
         */
        ZS_COPY(op, ip, 16);
        if (op >= oend)
            return;
        do {
            ZS_COPY(op, ip, 16);
            ZS_COPY(op, ip, 16);
        } while (op < oend);
    }
}

/// A safe version of ZS_wildcopy which preserves all the same semantics,
/// except that it doesn't over-copy.
/// It is intended to be used in the "end loop" that handles the last few
/// sequences. So it is optimized to be fast for long copies.
ZL_FORCE_INLINE ZL_FLATTEN_ATTR void ZS_safecopy(
        void* dst,
        const void* src,
        ptrdiff_t length,
        ZS_wildcopyOverlap const ovtype)
{
    const uint8_t* ip   = (const uint8_t*)src;
    uint8_t* op         = (uint8_t*)dst;
    uint8_t* const oend = op + length;

    if (length > ZS_WILDCOPY_OVERLENGTH) {
        ptrdiff_t const wildlen = length - ZS_WILDCOPY_OVERLENGTH;
        ZS_wildcopy(op, ip, wildlen, ovtype);
        op += wildlen;
        ip += wildlen;
    }
    while (op < oend)
        *op++ = *ip++;
}

ZL_END_C_DECLS

#endif // ZS_COMMON_COPY_H
