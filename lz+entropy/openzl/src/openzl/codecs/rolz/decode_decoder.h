// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_DECOMPRESS_DECODER_H
#define ZS_DECOMPRESS_DECODER_H

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

typedef struct {
    // Empty structs are disallowed
    int _;
} ZS_decoderCtx;

typedef struct {
    char const* name;
    ZS_decoderCtx* (*ctx_create)(void);
    void (*ctx_release)(ZS_decoderCtx* ctx);
    void (*ctx_reset)(ZS_decoderCtx* ctx);
    ZL_Report (*decompress)(
            ZS_decoderCtx* ctx,
            uint8_t* dst,
            size_t capacity,
            uint8_t const* src,
            size_t size);
} ZS_decoder;

extern const ZS_decoder ZS_experimentalDecoder;
extern const ZS_decoder ZS_fastDecoder;
static ZS_decoder const* const ZS_rolzDecoder   = &ZS_experimentalDecoder;
static ZS_decoder const* const ZS_fastLzDecoder = &ZS_fastDecoder;

ZL_END_C_DECLS

#endif // ZS_DECOMPRESS_DECODER_H
