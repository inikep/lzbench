// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_COMMON_HUF_AVX2_H
#define ZSTRONG_COMMON_HUF_AVX2_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

size_t ZS_HufAvx2_encodeBound(size_t srcSize);
size_t ZS_HufAvx2_encode(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize);
size_t
ZS_HufAvx2_decode(void* dst, size_t dstSize, void const* src, size_t srcSize);

size_t ZS_Huf16Avx2_encodeBound(size_t srcSize);
size_t ZS_Huf16Avx2_encode(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize);
size_t
ZS_Huf16Avx2_decode(void* dst, size_t dstSize, void const* src, size_t srcSize);

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_HUF_AVX2_H
