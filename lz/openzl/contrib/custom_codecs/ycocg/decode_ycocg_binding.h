// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_BITPACK_DECODE_BITPACK_BINDING_H
#define ZSTRONG_TRANSFORMS_BITPACK_DECODE_BITPACK_BINDING_H

#include "graph_ycocg.h"
#include "openzl/zl_dtransform.h" // ZL_Decoder

ZL_Report YCOCG_decode_serial(ZL_Decoder* decoder, const ZL_Input* ins[]);

/* Registration Structure for the YCoCg custom decoder.
 * Use ZL_DCtx_registerTypedDecoder()
 *
 * Registering the decoder is enough, there is nothing else to do.
 */
static const ZL_TypedDecoderDesc YCOCG_decoder_registration_structure = {
    .gd          = YCOCG_GRAPH,
    .transform_f = YCOCG_decode_serial,
    .name        = "YCOCG_decode_serial",
};

#endif
