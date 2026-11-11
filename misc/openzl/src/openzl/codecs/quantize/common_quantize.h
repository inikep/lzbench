// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_QUANTIZE_COMMON_QUANTIZE_H
#define ZSTRONG_TRANSFORMS_QUANTIZE_COMMON_QUANTIZE_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Parameters to control the quantization operation.
 * The encoding and decoding algorithms are as follows:
 *
 *   fn quantizeEncode(value, params) -> (code, bits):
 *     if (value < params.maxPow2)
 *       code = params.valueToCode[value]
 *     else
 *       code = params.delta + floor(log2(value))
 *     num_bits = params.bits[code]
 *     bits = value & ((1 << num_bits) - 1)
 *     return (code, bits)
 *
 *   fn quantizeDecode(code, bitstream) -> value:
 *     bits = bitstream.read(bits[code])
 *     value = base[code] + bits
 *     return value
 */
typedef struct {
    /// The total number of codes.
    size_t nbCodes;
    /// The mapping of value to code for values < maxPow2.
    uint8_t const* valueToCode;
    /// The offset for the first of the power of 2 codes.
    /// Must be: #(codes for values < maxPow2) - log2(maxPow2).
    uint32_t delta;
    /// The maximum power of 2 that uses valueToCode. Values larger
    /// than this get assigned the code: delta + floor(log2(value)).
    uint32_t maxPow2;
    /// The number of extra bits that each code has. bits[code] may be
    /// zero. Must be one value for every possible code.
    /// bits MUST be increasing, though this restriction could be removed.
    uint8_t const* bits;
    /// The base value for each code for decoding. The extra bits are
    /// added to the base to get the decoded value.
    uint32_t const* base;
} ZL_Quantize32Params;

extern const ZL_Quantize32Params ZL_quantizeOffsetsParams;
extern const ZL_Quantize32Params ZL_quantizeLengthsParams;

ZL_END_C_DECLS

#endif
