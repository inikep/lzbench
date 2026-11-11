// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMPRESS_ENCODER_H
#define ZS_COMPRESS_ENCODER_H

#include "openzl/codecs/rolz/encode_rolz_sequences.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

typedef enum {
    ZS_LiteralEncoding_o0,
    ZS_LiteralEncoding_o1,
} ZS_LiteralEncoding_e;

// The sum of all possible encoder parameters.
// Not every encoder must use every parameter.
typedef struct {
    uint32_t rolzContextDepth;
    uint32_t rolzContextLog;
    uint32_t rolzRowLog;
    uint32_t rolzMinLength;
    bool rolzPredictMatchLength;

    uint32_t lzMinLength;

    uint32_t repMinLength;

    uint32_t fieldSize;
    uint32_t fixedOffset;

    ZS_LiteralEncoding_e literalEncoding;
    bool zstdCompressLiterals;
} ZS_EncoderParameters;

typedef struct {
    // Empty structs are disallowed
    int _;
} ZS_encoderCtx;

typedef struct {
    char const* name;
    ZS_encoderCtx* (*ctx_create)(ZS_EncoderParameters const* params);
    void (*ctx_release)(ZS_encoderCtx* ctx);
    void (*ctx_reset)(ZS_encoderCtx* ctx);
    size_t (*compressBound)(size_t numLiterals, size_t numSequences);
    size_t (*compress)(
            ZS_encoderCtx* ctx,
            uint8_t* dst,
            size_t capacity,
            ZS_RolzSeqStore const* seqStore);
} ZS_encoder;

extern const ZS_encoder ZS_experimentalEncoder;
extern const ZS_encoder ZS_fastEncoder;
static ZS_encoder const* const ZS_rolzEncoder   = &ZS_experimentalEncoder;
static ZS_encoder const* const ZS_fastLzEncoder = &ZS_fastEncoder;

ZL_END_C_DECLS

#endif // ZS_COMPRESS_ENCODER_H
