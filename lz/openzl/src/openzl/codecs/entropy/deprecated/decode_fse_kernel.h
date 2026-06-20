// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_DECODE_FSE_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_DECODE_FSE_KERNEL_H

#include "openzl/common/base_types.h" // ZL_Report
#include "openzl/common/cursor.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/* Experimental FSE context encoder.
 * For normal FSE encoding use zstrong/common/entropy.h
 */

ZL_Report ZS_fseContextDecode(ZL_WC* dst, ZL_RC* src, ZL_RC* context);
ZL_Report ZS_fseO1Decode(ZL_WC* dst, ZL_RC* src);

/// The first byte of context is ignored.
ZL_Report ZS_fseContextO1Decode(
        ZL_WC* dst,
        ZL_RC* src,
        ZL_RC* context,
        uint8_t (*mix)(void* opaque, uint8_t ctx, uint8_t o1),
        void* opaque);

ZL_END_C_DECLS

#endif
