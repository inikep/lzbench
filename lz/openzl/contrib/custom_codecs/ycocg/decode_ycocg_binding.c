// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "decode_ycocg_binding.h"
#include "decode_ycocg_kernel.h"
#include "openzl/zl_data.h"
#include "openzl/zl_dtransform.h" // ZL_Decoder
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

ZL_Report YCOCG_decode_serial(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    // guaranteed by engine: nb of Inputs (from codec signature)
    assert(ins != NULL);
    const ZL_Input* y  = ins[0];
    const ZL_Input* co = ins[1];
    const ZL_Input* cg = ins[2];
    assert(y != NULL && co != NULL && cg != NULL);

    // guaranteed by engine + codec signature: types
    assert(ZL_Input_type(y) == ZL_Type_numeric);
    assert(ZL_Input_type(co) == ZL_Type_numeric);
    assert(ZL_Input_type(cg) == ZL_Type_numeric);

    // NOT guaranteed by engine + codec signature:
    // eltWidth, and numElts equality
    if (ZL_Input_eltWidth(y) != 1)
        return ZL_returnError(ZL_ErrorCode_corruption);
    if (ZL_Input_eltWidth(co) != 2)
        return ZL_returnError(ZL_ErrorCode_corruption);
    if (ZL_Input_eltWidth(cg) != 2)
        return ZL_returnError(ZL_ErrorCode_corruption);

    size_t nbPixels = ZL_Input_contentSize(y);
    if (ZL_Input_numElts(co) != nbPixels)
        return ZL_returnError(ZL_ErrorCode_corruption);
    if (ZL_Input_numElts(cg) != nbPixels)
        return ZL_returnError(ZL_ErrorCode_corruption);

    // Output creation
    // Note: allocation is controlled by the engine
    ZL_Output* rgb = ZL_Decoder_create1OutStream(dictx, nbPixels * 3, 1);
    if (rgb == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    // All conditions validated:
    // let's invoke the decoder kernel
    const void* yptr  = ZL_Input_ptr(y);
    const void* coptr = ZL_Input_ptr(co);
    const void* cgptr = ZL_Input_ptr(cg);
    void* rgbptr      = ZL_Output_ptr(rgb);

    YCOCG_decodeArray_RGB24(rgbptr, yptr, coptr, cgptr, nbPixels);

    // Explicitly commit the nb of elements produced into the Output stream
    if (ZL_isError(ZL_Output_commit(rgb, 3 * nbPixels)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);

    return ZL_returnSuccess();
}
