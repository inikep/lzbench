// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_ZL_QUANTIZE_H
#define OPENZL_CODECS_ZL_QUANTIZE_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * TODO: We should expose a more generic quantization codec that allows
 * configuring the quantization scheme. Then merge these two into that
 * new codec.
 */

/**
 * Quantizes 32-bit integers into codewords and extra bits.
 * The codeword is log2(value), and extra bits contain log2(value) bits per
 * value. This codec is typically used for the offsets of an LZ codec.
 *
 *
 * @warning The value 0 is not supported and will fail compression!
 *
 * Input: Numeric 32-bit unsigned integers
 * Output 0: Numeric 8-bit unsigned codes
 * Output 1: Serial extra-bits
 */
#define ZL_NODE_QUANTIZE_OFFSETS           \
    (ZL_NodeID)                            \
    {                                      \
        ZL_StandardNodeID_quantize_offsets \
    }

/**
 * Quantizes 32-bit integers into codewords and extra bits.
 * Values <= 32 each get their own codeword, after that it transitions to a
 * power-of-2 quantization scheme. This codec is typically used for the lengths
 * of an LZ codec. It will perform well when there isn't higher-order
 * correlation to capture, and most values are <= 32.
 *
 * Input: Numeric 32-bit unsigned integers
 * Output 0: Numeric 8-bit unsigned codes
 * Output 1: Serial extra-bits
 */
#define ZL_NODE_QUANTIZE_LENGTHS           \
    (ZL_NodeID)                            \
    {                                      \
        ZL_StandardNodeID_quantize_lengths \
    }

#if defined(__cplusplus)
}
#endif

#endif
