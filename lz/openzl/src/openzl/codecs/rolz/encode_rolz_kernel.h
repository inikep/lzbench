// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_ROLZ_H
#define ZSTRONG_COMPRESS_ROLZ_H

#include "openzl/common/base_types.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

size_t ZS_rolzCompressBound(size_t srcSize);
size_t ZS_fastLzCompressBound(size_t srcSize);

ZL_Report
ZS_rolzCompress(void* dst, size_t dstCapacity, void const* src, size_t srcSize);
ZL_Report ZS_fastLzCompress(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize);

ZL_END_C_DECLS

#endif // ZSTRONG_COMPRESS_ROLZ_H
