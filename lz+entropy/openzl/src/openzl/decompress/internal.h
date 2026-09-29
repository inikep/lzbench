// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_DECOMPRESS_INTERNAL_H
#define ZS_DECOMPRESS_INTERNAL_H

#include "openzl/common/base_types.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

ZL_Report ZL_rolzDecompress(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize);

ZL_END_C_DECLS

#endif // ZS_DECOMPRESS_INTERNAL_H
