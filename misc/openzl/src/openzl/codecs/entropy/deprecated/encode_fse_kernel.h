// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_ENCODE_FSE_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_ENCODE_FSE_KERNEL_H

#include <stddef.h> // size_t

#include "openzl/common/cursor.h"
#include "openzl/common/zstrong_internal.h"
#include "openzl/shared/clustering.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/* Experimental FSE context encoder.
 * For normal FSE encoding use zstrong/common/entropy.h
 */

ZL_Report ZS_fseContextEncode(
        ZL_WC* dst,
        ZL_RC* src,
        ZL_RC* context,
        ZL_ContextClustering const* clustering);
ZL_Report
ZS_fseO1Encode(ZL_WC* dst, ZL_RC* src, ZL_ContextClustering const* clustering);

/// The first byte of context is ignored.
/// This function may not be super useful, because you need to have the
/// clustering. But it is currently provided because ZS_fseContextO1Decode() can
/// be useful.
ZL_Report ZS_fseContextO1Encode(
        ZL_WC* dst,
        ZL_RC* src,
        ZL_RC* context,
        uint8_t (*mix)(void* opaque, uint8_t ctx, uint8_t o1),
        void* opaque,
        ZL_ContextClustering const* clustering);

ZL_END_C_DECLS

#endif
