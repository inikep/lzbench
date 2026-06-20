// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "encode_ycocg_binding.h"
#include "encode_ycocg_kernel.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder
#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

ZL_Report YCOCG_encode_serial(ZL_Encoder* eictx, const ZL_Input* in)
{
    /* Guaranteed by engine and codec signature: input type */
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_serial);
    assert(ZL_Input_contentSize(in) % 3 == 0);
    size_t nbPixels = ZL_Input_contentSize(in) / 3;

    // Output creations
    // Note: allocation is controlled by the engine
    ZL_Output* y  = ZL_Encoder_createTypedStream(eictx, 0, nbPixels, 1);
    ZL_Output* co = ZL_Encoder_createTypedStream(eictx, 1, nbPixels, 2);
    ZL_Output* cg = ZL_Encoder_createTypedStream(eictx, 2, nbPixels, 2);
    if (y == NULL || co == NULL || cg == NULL) {
        // No need to free: engine will take care of it
        return ZL_returnError(ZL_ErrorCode_allocation);
    }

    // All conditions validated:
    // let's invoke the encoder kernel
    void* yptr         = ZL_Output_ptr(y);
    void* coptr        = ZL_Output_ptr(co);
    void* cgptr        = ZL_Output_ptr(cg);
    const void* rgbptr = ZL_Input_ptr(in);

    YCOCG_encodeArray_RGB24(yptr, coptr, cgptr, rgbptr, nbPixels);

    // Explicitly commit the nb of elements produced into each Output stream
    if (ZL_isError(ZL_Output_commit(y, nbPixels)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    if (ZL_isError(ZL_Output_commit(co, nbPixels)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    if (ZL_isError(ZL_Output_commit(cg, nbPixels)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);

    return ZL_returnSuccess();
}
