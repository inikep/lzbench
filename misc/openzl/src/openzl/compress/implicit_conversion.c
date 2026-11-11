// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/compress/implicit_conversion.h"
#include "openzl/zl_ctransform.h" // ZL_NodeID_isValid

int ICONV_isCompatible(const ZL_Type origType, const ZL_Type dstTypes)
{
    // origType is supported by dstTypes
    // if they have one bit (type) in common.
    if (origType & dstTypes) {
        return 1;
    }
    // If not, origType may still be compatible with dstTypes
    // if there is at least one implicit conversion available.
    return ZL_NodeID_isValid(
            ICONV_implicitConversionNodeID(origType, dstTypes));
}

ZL_NodeID ICONV_implicitConversionNodeID(
        const ZL_Type srcType,
        const ZL_Type dstType)
{
    // When multiple implicit conversions are possible,
    // i.e. numeric could be converted into fixed_size or serialized,
    // prefer numeric->fixed_size.
    if ((srcType & ZL_Type_numeric) && (dstType & ZL_Type_struct)) {
        return ZL_NODE_CONVERT_NUM_TO_TOKEN;
    } else if ((srcType & ZL_Type_struct) && (dstType & ZL_Type_serial)) {
        return ZL_NODE_CONVERT_TOKEN_TO_SERIAL;
    } else if ((srcType & ZL_Type_numeric) && (dstType & ZL_Type_serial)) {
        return ZL_NODE_CONVERT_NUM_TO_SERIAL;
    }
    return ZL_NODE_ILLEGAL;
}
