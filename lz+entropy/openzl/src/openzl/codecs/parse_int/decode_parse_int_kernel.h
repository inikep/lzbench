// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_DECODE_PARSE_INT_KERNEL_H
#define ZSTRONG_DECODE_PARSE_INT_KERNEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

size_t ZL_DecodeParseInt_fillFieldSizes(
        uint32_t* fieldSizes,
        size_t nbElts,
        int64_t const* nums);

void ZL_DecodeParseInt_fillContent(
        char* dst,
        size_t const dstSize,
        const size_t nbElts,
        int64_t const* nums,
        uint32_t const* const fieldSizes);

ZL_END_C_DECLS

#endif
