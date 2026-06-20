// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CONTRIB_CUSTOM_CODECS_YCOCG_ENCODE_BINDING_H
#define CONTRIB_CUSTOM_CODECS_YCOCG_ENCODE_BINDING_H

#include "graph_ycocg.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder

ZL_Report YCOCG_encode_serial(ZL_Encoder* eictx, const ZL_Input* in);

/* Registration Structure for the YCoCg custom codec
 * Use ZL_Compressor_registerTypedEncoder()
 *
 * The codec accepts as Input a single Serial stream,
 * input size must be a multiple of 3.
 * It's expected to represent RGB 24-bit format.
 *
 * The codec will produce 3 numeric streams as outputs,
 * in order: y (8-bit), Co (16-bit signed) and Cg (16-bit signed).
 */
static const ZL_TypedEncoderDesc YCOCG_encoder_registration_structure = {
    .gd          = YCOCG_GRAPH,
    .transform_f = YCOCG_encode_serial,
    .name        = "YCOCG_encode_serial"
};

#endif
