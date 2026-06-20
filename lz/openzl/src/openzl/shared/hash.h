// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_HASH_H
#define ZS_COMMON_HASH_H

#include "openzl/common/debug.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

#define ZL_HASH_READ_SIZE 8

static const uint32_t ZL_prime2bytes = 506832829U;
ZL_INLINE size_t ZL_hash2(uint32_t u, uint32_t h)
{
    return ((u << (32 - 16)) * ZL_prime2bytes) >> (32 - h);
}
ZL_INLINE size_t ZL_hash2Ptr(void const* ptr, uint32_t h)
{
    return ZL_hash2(ZL_readLE16(ptr), h);
}

static const uint32_t ZL_prime3bytes = 506832829U;
ZL_INLINE size_t ZL_hash3(uint32_t u, uint32_t h)
{
    return ((u << (32 - 24)) * ZL_prime3bytes) >> (32 - h);
}
ZL_INLINE size_t ZL_hash3Ptr(void const* ptr, uint32_t h)
{
    return ZL_hash3(ZL_readLE32(ptr), h);
}

static const uint32_t ZL_prime4bytes = 2654435761U;
ZL_INLINE size_t ZL_hash4(uint32_t u, uint32_t h)
{
    return (u * ZL_prime4bytes) >> (32 - h);
}
ZL_INLINE size_t ZL_hash4Ptr(void const* ptr, uint32_t h)
{
    return ZL_hash4(ZL_read32(ptr), h);
}

static const uint64_t ZL_prime5bytes = 889523592379ULL;
ZL_INLINE size_t ZL_hash5(uint64_t u, uint32_t h)
{
    return (size_t)(((u << (64 - 40)) * ZL_prime5bytes) >> (64 - h));
}
ZL_INLINE size_t ZL_hash5Ptr(void const* p, uint32_t h)
{
    return ZL_hash5(ZL_readLE64(p), h);
}

static const uint64_t ZL_prime6bytes = 227718039650203ULL;
ZL_INLINE size_t ZL_hash6(uint64_t u, uint32_t h)
{
    return (size_t)(((u << (64 - 48)) * ZL_prime6bytes) >> (64 - h));
}
ZL_INLINE size_t ZL_hash6Ptr(void const* p, uint32_t h)
{
    return ZL_hash6(ZL_readLE64(p), h);
}

static const uint64_t ZL_prime7bytes = 58295818150454627ULL;
ZL_INLINE size_t ZL_hash7(uint64_t u, uint32_t h)
{
    return (size_t)(((u << (64 - 56)) * ZL_prime7bytes) >> (64 - h));
}
ZL_INLINE size_t ZL_hash7Ptr(void const* p, uint32_t h)
{
    return ZL_hash7(ZL_readLE64(p), h);
}

static const uint64_t ZL_prime8bytes = 0xCF1BBCDCB7A56463ULL;
ZL_INLINE size_t ZL_hash8(uint64_t u, uint32_t h)
{
    return (size_t)(((u)*ZL_prime8bytes) >> (64 - h));
}
ZL_INLINE size_t ZL_hash8Ptr(void const* p, uint32_t h)
{
    return ZL_hash8(ZL_readLE64(p), h);
}
ZL_INLINE size_t ZL_hash12Ptr(void const* p, uint32_t h)
{
    return ZL_hash8(ZL_readLE64(p) ^ ZL_readLE32((uint8_t const*)p + 8), h);
}
ZL_INLINE size_t ZL_hash16Ptr(void const* p, uint32_t h)
{
    return ZL_hash8(ZL_readLE64(p) ^ ZL_readLE64((uint8_t const*)p + 8), h);
}

ZL_INLINE size_t ZL_hash(uint64_t u, uint32_t h, uint32_t l)
{
    switch (l) {
        case 1:
            ZL_ASSERT_FAIL("Hash length %u too small", l);
            return h;
        case 2:
            return ZL_hash2((uint16_t)u, h);
        case 3:
            return ZL_hash3((uint32_t)u, h);
        case 4:
            return ZL_hash4((uint32_t)u, h);
        case 5:
            return ZL_hash5(u, h);
        case 6:
            return ZL_hash6(u, h);
        case 7:
            return ZL_hash7(u, h);
        case 8:
            return ZL_hash8(u, h);
        default:
            ZL_ASSERT_FAIL("Hash length %u too large", l);
            return ZL_hash8(u, h);
    }
}

ZL_INLINE size_t ZL_hashPtr(void const* p, uint32_t h, uint32_t l)
{
    switch (l) {
        case 1:
            ZL_ASSERT_FAIL("Hash length %u too small", l);
            return h;
        case 2:
            return ZL_hash2Ptr(p, h);
        case 3:
            return ZL_hash3Ptr(p, h);
        case 4:
            return ZL_hash4Ptr(p, h);
        case 5:
            return ZL_hash5Ptr(p, h);
        case 6:
            return ZL_hash6Ptr(p, h);
        case 7:
            return ZL_hash7Ptr(p, h);
        case 8:
            return ZL_hash8Ptr(p, h);
        case 12:
            return ZL_hash12Ptr(p, h);
        case 16:
            return ZL_hash16Ptr(p, h);
        default:
            ZL_ASSERT_FAIL("Hash length %u too large", l);
            return ZL_hash8Ptr(p, h);
    }
}

ZL_END_C_DECLS

#endif // ZS_COMMON_HASH_H
