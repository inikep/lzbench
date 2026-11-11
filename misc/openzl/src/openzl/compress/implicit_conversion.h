// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_IMPLICIT_CONVERSION_H
#define ZSTRONG_IMPLICIT_CONVERSION_H

#include "openzl/shared/portability.h"
#include "openzl/zl_data.h"
#include "openzl/zl_opaque_types.h"

ZL_BEGIN_C_DECLS

/* Tell if origType is supported by dstTypes,
 * either directly, or via an implicit conversion.
 * Both origType and dstTypes can be bitmaps
 * with multiple types activated. */
int ICONV_isCompatible(const ZL_Type origType, const ZL_Type dstTypes);

ZL_NodeID ICONV_implicitConversionNodeID(
        const ZL_Type srcType,
        const ZL_Type dstType);

ZL_END_C_DECLS

#endif // ZSTRONG_IMPLICIT_CONVERSION_H
