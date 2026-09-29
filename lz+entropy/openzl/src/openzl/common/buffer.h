// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines a memory management abstraction which provides tools to
 * allocate and interact with memory buffers.
 *
 * TODO: build tooling to permit sharing ownership?
 */

#ifndef ZSTRONG_COMMON_BUFFER_H
#define ZSTRONG_COMMON_BUFFER_H

#include "openzl/common/cursor.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * The ZL_Buffer struct owns and manages a memory buffer.
 *
 * You should not interact with its internals directly. Instead, use the methods
 * defined below.
 *
 * The struct is not trivially copyable.
 */
typedef struct {
    ZL_WC _wc;
} ZL_Buffer;
#define ZL_B ZL_Buffer

ZL_INLINE ZL_B ZL_B_createNull(void)
{
    return (ZL_B){ ._wc = ZL_WC_makeEmpty() };
}

ZL_INLINE ZL_B ZL_B_create(size_t capacity)
{
    uint8_t* buf = (uint8_t*)malloc(capacity);
    ZL_ASSERT(!capacity || buf);
    return (ZL_B){ ._wc = ZL_WC_wrap(buf, capacity) };
}

ZL_INLINE ZL_B ZL_B_move(ZL_B* b)
{
    ZL_B copy = *b;
    *b        = ZL_B_createNull();
    return copy;
}

ZL_INLINE void ZL_B_destroy(ZL_B* b)
{
    ZL_ASSERT_NN(b);
    free(ZL_WC_begin(&b->_wc));
}

ZL_INLINE bool ZL_B_isNull(const ZL_B* b)
{
    return ZL_WC_cbegin(&b->_wc) == NULL;
}

ZL_INLINE size_t ZL_B_size(const ZL_B* b)
{
    return ZL_WC_size(&b->_wc);
}

ZL_INLINE size_t ZL_B_capacity(const ZL_B* b)
{
    return ZL_WC_capacity(&b->_wc);
}

ZL_INLINE void ZL_B_resize(ZL_B* b, size_t newcapacity)
{
    ZL_ASSERT_NN(b);
    size_t const used = ZL_WC_size(&b->_wc);
    ZL_ASSERT_GE(newcapacity, used);
    uint8_t* buf = ZL_WC_begin(&b->_wc);
    buf          = (uint8_t*)realloc(buf, newcapacity);
    ZL_ASSERT(!newcapacity || buf);
    b->_wc = ZL_WC_wrapPartial(buf, used, newcapacity);
}

/**
 * Ensures that the requested capacity is available in addition to the space
 * currently in use. Allocates just the requested space.
 */
ZL_INLINE void ZL_B_reserve(ZL_B* b, size_t additional_capacity)
{
    ZL_ASSERT_NN(b);
    size_t total_needed = ZL_B_size(b) + additional_capacity;
    if (ZL_B_capacity(b) < total_needed) {
        ZL_B_resize(b, total_needed);
    }
}

/**
 * Ensures that the requested capacity is available in addition to the space
 * currently in use. If not enough space is available, it at least doubles the
 * allocated space.
 */
ZL_INLINE void ZL_B_reserve2(ZL_B* b, size_t additional_capacity)
{
    ZL_ASSERT_NN(b);
    size_t total_needed = ZL_B_size(b) + additional_capacity;
    if (ZL_B_capacity(b) >= total_needed) {
        return;
    }
    if (ZL_B_capacity(b) * 2 > total_needed) {
        total_needed = ZL_B_capacity(b) * 2;
    }
    ZL_B_resize(b, total_needed);
}

ZL_INLINE ZL_RC ZL_B_getRC(const ZL_B* b)
{
    ZL_ASSERT_NN(b);
    return ZL_RC_wrapWC(&b->_wc);
}

ZL_INLINE ZL_WC* ZL_B_getWC(ZL_B* b)
{
    return &b->_wc;
}

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_BUFFER_H
