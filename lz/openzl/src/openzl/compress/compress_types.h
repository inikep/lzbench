// Copyright (c) Meta Platforms, Inc. and affiliates.

// Defines types used in several units within zstrong/compress

#ifndef ZS_COMPRESS_TYPES
#define ZS_COMPRESS_TYPES

#include "openzl/common/logging.h" // STR_REPLACE_NULL
#include "openzl/shared/portability.h"
#include "openzl/zl_ctransform.h" // ZL_MIEncoderDesc

ZL_BEGIN_C_DECLS

// Internal transform's Declaration structure
typedef struct {
    ZL_MIEncoderDesc publicDesc;
    const void* privateParam;
    size_t ppSize; // if == 0, privateParam won't be moved,
                   // in which case, @privateParam's content must remain stable.
} InternalTransform_Desc;

ZL_INLINE const char* CT_getTrName(const InternalTransform_Desc* itd)
{
    return STR_REPLACE_NULL(itd->publicDesc.name);
}

typedef enum {
    node_illegal = 0,
    node_internalTransform,
} NodeType_e;

typedef struct {
    unsigned minFormatVersion;
    unsigned maxFormatVersion;
} FormatLimits;

ZL_END_C_DECLS

#endif // ZS_COMPRESS_TYPES
