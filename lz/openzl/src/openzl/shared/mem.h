// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_MEM_H
#define ZS_COMMON_MEM_H

#include <stdalign.h> // alignof
#include <string.h>   // memcpy

#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h" // ZL_isLittleEndian
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

// Just a redirector to memcpy
// Future version will allow redirecting away from <string.h> if needed
ZL_INLINE void ZL_memcpy(void* dst, const void* src, size_t size)
{
    memcpy(dst, src, size);
}

// Just a redirector to memset
// Future version will allow redirecting away from <string.h> if needed
ZL_INLINE void ZL_memset(void* src, char val, size_t size)
{
    memset(src, val, size);
}

/* Note :
 * These memory access methods
 * are designed to handle endianess transparently.
 * They expect their parameter pointer to be valid,
 * aka they point to a valid buffer with enough space for intended value */

ZL_INLINE uint8_t ZL_read8(const void* memPtr)
{
    uint8_t val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

ZL_INLINE uint16_t ZL_read16(const void* memPtr)
{
    uint16_t val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

/* Note (@Cyan) : which meaning carries this function ?
 * 24-bit is not a native format.
 * It follows that reading a 3-bytes values is always a serialized operation,
 * hence with a defined endianness.
 */
ZL_INLINE uint32_t ZL_read24(const void* memPtr)
{
    uint32_t val = 0;
    if (ZL_isLittleEndian()) {
        memcpy(&val, memPtr, 3);
    } else {
        memcpy((uint8_t*)&val + 1, memPtr, 3);
    }
    return val;
}

ZL_INLINE uint32_t ZL_read32(const void* memPtr)
{
    uint32_t val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

ZL_INLINE uint64_t ZL_read64(const void* memPtr)
{
    uint64_t val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

ZL_INLINE size_t ZL_readST(const void* memPtr)
{
    size_t val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

ZL_INLINE uint64_t ZL_readN(const void* memPtr, size_t kWidth)
{
    switch (kWidth) {
        case 1:
            return ZL_read8(memPtr);
        case 2:
            return ZL_read16(memPtr);
        case 4:
            return ZL_read32(memPtr);
        case 8:
            return ZL_read64(memPtr);
        default:
            ZL_ASSERT(false);
            return 0;
    }
}

ZL_INLINE void ZL_write8(void* memPtr, uint8_t value)
{
    memcpy(memPtr, &value, sizeof(value));
}

ZL_INLINE void ZL_write16(void* memPtr, uint16_t value)
{
    memcpy(memPtr, &value, sizeof(value));
}

ZL_INLINE void ZL_write24(void* memPtr, uint32_t value)
{
    if (ZL_isLittleEndian())
        memcpy(memPtr, &value, 3);
    else
        memcpy(memPtr, (uint8_t*)&value + 1, 3);
}

ZL_INLINE void ZL_write32(void* memPtr, uint32_t value)
{
    memcpy(memPtr, &value, sizeof(value));
}

ZL_INLINE void ZS_write64(void* memPtr, uint64_t value)
{
    memcpy(memPtr, &value, sizeof(value));
}

ZL_INLINE void ZL_writeST(void* memPtr, size_t value)
{
    memcpy(memPtr, &value, sizeof(value));
}

ZL_INLINE void ZL_writeN(void* memPtr, uint64_t val, size_t kWidth)
{
    switch (kWidth) {
        case 1:
            ZL_write8(memPtr, (uint8_t)val);
            break;
        case 2:
            ZL_write16(memPtr, (uint16_t)val);
            break;
        case 4:
            ZL_write32(memPtr, (uint32_t)val);
            break;
        case 8:
            ZS_write64(memPtr, (uint64_t)val);
            break;
        default:
            ZL_ASSERT(false);
            break;
    }
}

/*=== Little endian r/w ===*/

ZL_INLINE uint16_t ZL_readLE16(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_read16(memPtr);
    else {
        const uint8_t* p = (const uint8_t*)memPtr;
        return (uint16_t)(p[0] + (p[1] << 8));
    }
}

ZL_INLINE void ZL_writeLE16(void* memPtr, uint16_t val)
{
    if (ZL_isLittleEndian()) {
        ZL_write16(memPtr, val);
    } else {
        ZL_write16(memPtr, ZL_swap16(val));
    }
}

ZL_INLINE uint32_t ZL_readLE24(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_read24(memPtr);
    else
        return ZL_swap32(ZL_read24(memPtr)) >> 8;
}

ZL_INLINE void ZL_writeLE24(void* memPtr, uint32_t val)
{
    if (ZL_isLittleEndian())
        ZL_write24(memPtr, val);
    else
        ZL_write24(memPtr, ZL_swap32(val) >> 8);
}

ZL_INLINE uint32_t ZL_readLE32(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_read32(memPtr);
    else
        return ZL_swap32(ZL_read32(memPtr));
}

ZL_INLINE void ZL_writeLE32(void* memPtr, uint32_t val32)
{
    if (ZL_isLittleEndian())
        ZL_write32(memPtr, val32);
    else
        ZL_write32(memPtr, ZL_swap32(val32));
}

ZL_INLINE uint64_t ZL_readLE64(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_read64(memPtr);
    else
        return ZL_swap64(ZL_read64(memPtr));
}

/* This variant is employed to read a variable < 8 bytes,
 * typically when reaching close to buffer boundaries.
 * CONDITION :
 * only works for @nbBytes <= 8
 */
ZL_INLINE uint64_t ZL_readLE64_N(const void* memPtr, size_t nbBytes)
{
    ZL_ASSERT_LE(nbBytes, 8);
    uint64_t r                = 0;
    const uint8_t* const ptr8 = (const uint8_t*)
            memPtr; // Note : this cast shouldn't be necessary, but apparently
                    // this header is included in some C++ file ...
    for (size_t n = 0; n < nbBytes; n++) {
        r += ((uint64_t)ptr8[n] << (8 * n));
    }
    return r;
}

ZL_INLINE void ZL_writeLE64(void* memPtr, uint64_t val64)
{
    if (ZL_isLittleEndian())
        ZS_write64(memPtr, val64);
    else
        ZS_write64(memPtr, ZL_swap64(val64));
}

/* This variant is employed to write a variable <= 8 bytes,
 * typically when reaching close to buffer boundaries.
 * CONDITION :
 * only works for @nbBytes <= 8
 */
ZL_INLINE void ZL_writeLE64_N(void* memPtr, uint64_t val64, size_t nbBytes)
{
    ZL_ASSERT_LE(nbBytes, 8);
    uint8_t* const ptr8 = (uint8_t*)memPtr;
    for (size_t n = 0; n < nbBytes; ++n) {
        ptr8[n] = (uint8_t)(val64 >> (8 * n));
    }
}

ZL_INLINE size_t ZL_readLEST(const void* memPtr)
{
    if (ZL_32bits())
        return (size_t)ZL_readLE32(memPtr);
    else
        return (size_t)ZL_readLE64(memPtr);
}

ZL_INLINE void ZL_writeLEST(void* memPtr, size_t val)
{
    if (ZL_32bits())
        ZL_writeLE32(memPtr, (uint32_t)val);
    else
        ZL_writeLE64(memPtr, (uint64_t)val);
}

/*=== Big endian r/w ===*/

ZL_INLINE uint16_t ZL_readBE16(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_swap16(ZL_read16(memPtr));
    else
        return ZL_read16(memPtr);
}

ZL_INLINE void ZL_writeBE16(void* memPtr, uint16_t val16)
{
    if (ZL_isLittleEndian())
        ZL_write16(memPtr, ZL_swap16(val16));
    else
        ZL_write16(memPtr, val16);
}

ZL_INLINE uint32_t ZL_readBE24(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_swap32(ZL_read24(memPtr)) >> 8;
    else
        return ZL_read24(memPtr);
}

ZL_INLINE void ZL_writeBE24(void* memPtr, uint32_t val)
{
    if (ZL_isLittleEndian())
        ZL_write24(memPtr, ZL_swap32(val) >> 8);
    else
        ZL_write24(memPtr, val);
}

ZL_INLINE uint32_t ZL_readBE32(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_swap32(ZL_read32(memPtr));
    else
        return ZL_read32(memPtr);
}

ZL_INLINE void ZL_writeBE32(void* memPtr, uint32_t val32)
{
    if (ZL_isLittleEndian())
        ZL_write32(memPtr, ZL_swap32(val32));
    else
        ZL_write32(memPtr, val32);
}

ZL_INLINE uint64_t ZL_readBE64(const void* memPtr)
{
    if (ZL_isLittleEndian())
        return ZL_swap64(ZL_read64(memPtr));
    else
        return ZL_read64(memPtr);
}

ZL_INLINE void ZL_writeBE64(void* memPtr, uint64_t val64)
{
    if (ZL_isLittleEndian())
        ZS_write64(memPtr, ZL_swap64(val64));
    else
        ZS_write64(memPtr, val64);
}

ZL_INLINE size_t ZL_readBEST(const void* memPtr)
{
    if (ZL_32bits())
        return (size_t)ZL_readBE32(memPtr);
    else
        return (size_t)ZL_readBE64(memPtr);
}

ZL_INLINE void ZL_writeBEST(void* memPtr, size_t val)
{
    if (ZL_32bits())
        ZL_writeBE32(memPtr, (uint32_t)val);
    else
        ZL_writeBE64(memPtr, (uint64_t)val);
}

#if ZL_CANONICAL_ENDIANNESS_IS_LITTLE
#    define ZL_readCE16 ZL_readLE16
#    define ZL_readCE24 ZL_readLE24
#    define ZL_readCE32 ZL_readLE32
#    define ZL_readCE64 ZL_readLE64
#    define ZL_readCEST ZL_readLEST
#    define ZL_writeCE16 ZL_writeLE16
#    define ZL_writeCE24 ZL_writeLE24
#    define ZL_writeCE32 ZL_writeLE32
#    define ZL_writeCE64 ZL_writeLE64
#    define ZL_writeCEST ZL_writeLEST
#else
#    define ZL_readCE16 ZL_readBE16
#    define ZL_readCE24 ZL_readBE24
#    define ZL_readCE32 ZL_readBE32
#    define ZL_readCE64 ZL_readBE64
#    define ZL_readCEST ZL_readBEST
#    define ZL_writeCE16 ZL_writeBE16
#    define ZL_writeCE24 ZL_writeBE24
#    define ZL_writeCE32 ZL_writeBE32
#    define ZL_writeCE64 ZL_writeBE64
#    define ZL_writeCEST ZL_writeBEST
#endif

/* ZS_consume*() & ZS_push*():
 * Below variants automatically update the provided pointer
 * to point after the value being consumed or pushed.
 * Note : another variant could use a cursor instead,
 *        with the advantage that it could check boundary
 *        but then, it would have to provide an error code,
 *        that the user would have to check, and react to,
 *        making both interface and caller's code more complex.
 */
ZL_INLINE uint32_t ZL_consumeLE32(const void** memPtr)
{
    uint32_t const r = ZL_readLE32(*memPtr);
    *memPtr          = (const char*)(*memPtr) + sizeof(uint32_t);
    return r;
}

ZL_INLINE void ZL_pushLE32(void** memPtr, uint32_t val32)
{
    ZL_writeLE32(*memPtr, val32);
    *memPtr = (char*)(*memPtr) + sizeof(uint32_t);
}

/* *****   Alignment   ***** */

ZL_INLINE size_t MEM_alignmentForNumericWidth(size_t width)
{
    switch (width) {
        case 1:
            return 1;
        case 2:
            return alignof(uint16_t);
        case 4:
            return alignof(uint32_t);
        case 8:
            return alignof(uint64_t);
        default:
            ZL_ASSERT_FAIL("invalid numeric width (%zu)", width);
            return 1;
    }
}

// Test that @_ptr is properly aligned on @_N bytes boundaries
// @_N is a fixed size value, and not directly related to any type.
// @_N must be a clean power of 2 value (1, 2, 4, 8, 16, 32, etc.)
// Note : it would be better to assert() this condition,
//        but this unit doesn't #include <assert.h>
#define MEM_IS_ALIGNED_N(_ptr, _N) (((size_t)(_ptr) & ((_N) - 1)) == 0)

// Test that @_ptr is properly aligned for @_type
// This is the proper variant for any @_type,
// as it ensures adjustment to local ABI
// Note : _Alignof is C11
#if defined(__cplusplus)
#    define MEM_IS_ALIGNED(_ptr, _type) MEM_IS_ALIGNED_N(_ptr, alignof(_type))
#else
#    define MEM_IS_ALIGNED(_ptr, _type) MEM_IS_ALIGNED_N(_ptr, _Alignof(_type))
#endif

ZL_INLINE int MEM_isAlignedForNumericWidth(const void* p, size_t width)
{
    switch (width) {
        case 1:
            return 1;
        case 2:
            return MEM_IS_ALIGNED(p, uint16_t);
        case 4:
            return MEM_IS_ALIGNED(p, uint32_t);
        case 8:
            return MEM_IS_ALIGNED(p, uint64_t);
        default:
            ZL_ASSERT_FAIL("invalid numeric width (%zu)", width);
            return 0;
    }
}

// Helper, providing the distance between 2 ptrs in bytes,
// whatever their base type.
// Condition 1 : all ptrs are valid
// Condition 2 : low ptr is presumed provided first.
ZL_INLINE size_t MEM_ptrDistance(const void* low, const void* up)
{
    ZL_ASSERT_NN(low);
    ZL_ASSERT_NN(up);
    ZL_ASSERT((low) <= (up)); // note : ZL_ASSERT_LE() currently fails with
                              // void* pointers
    return (size_t)((const char*)up - (const char*)low);
}

ZL_END_C_DECLS

#endif // ZS_COMMON_MEM_H
