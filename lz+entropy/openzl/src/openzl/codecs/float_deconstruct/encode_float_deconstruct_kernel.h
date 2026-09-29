// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_FLOAT_DECONSTRUCT_ENCODE_FLOAT_DECONSTRUCT_KERNEL_H
#define ZSTRONG_TRANSFORMS_FLOAT_DECONSTRUCT_ENCODE_FLOAT_DECONSTRUCT_KERNEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/* Encodes each float32 element from @src32 by splitting its exponent bits
 * into one stream and its sign/fraction bits into another. After encoding,
 *   - byte elements of @exponent contain the exponent (bits 23-30)
 *     of corresponding elements of @src32.
 *   - 3-byte elements of @signFrac contain the sign (bit 31) and
 *     fraction (bits 0-22) of corresponding elements of @src32.
 *     Sign bits are stored in the LSB position of @signFrac elements.
 *
 * IMPORTANT: It is crucial to provide sufficiently large @exponent and
 * @signFrac buffers. For n float32 elements,
 *   - @exponent requires at least n bytes.
 *   - @signFrac requires at least 3n bytes.
 */
void FLTDECON_float32_deconstruct_encode(
        uint32_t const* __restrict src32,
        uint8_t* __restrict exponent,
        uint8_t* __restrict signFrac,
        size_t nbElts);

/* Encodes each bfloat16 element from @src16 by splitting its exponent bits
 * into one stream and its sign/fraction bits into another. After encoding,
 *   - byte elements of @exponent contain the exponent (bits 7-14)
 *     of corresponding elements of @src16.
 *   - byte elements of @signFrac contain the sign (bit 15) and
 *     fraction (bits 0-6) of corresponding elements of @src16.
 *     Sign bits are stored in the LSB position of @signFrac elements.
 *
 * IMPORTANT: It is crucial to provide sufficiently large @exponent and
 * @signFrac buffers. For n bfloat16 elements,
 *   - @exponent requires at least n bytes.
 *   - @signFrac requires at least n bytes.
 */
void FLTDECON_bfloat16_deconstruct_encode(
        uint16_t const* __restrict src16,
        uint8_t* __restrict exponent,
        uint8_t* __restrict signFrac,
        size_t nbElts);

/* Encodes each float16 element from @src16 by splitting its exponent bits
 * into one stream and its sign/fraction bits into another. After encoding,
 *   - byte elements of @exponent contain the exponent (bits 10-14)
 *     of corresponding elements of @src16. Exponent bits are stored in
 *     the lower 5 bits of each byte element of @exponent.
 *   - two-byte (word) elements of @signFrac contain the sign (bit 15) and
 *     fraction (bits 0-9) of corresponding elements of @src16.
 *     Sign bits are stored in the LSB position of @signFrac elements.
 *     Fraction bits are stored in bits 1-10 of @signFrac elements.
 *
 * IMPORTANT: It is crucial to provide sufficiently large @exponent and
 * @signFrac buffers. For n float16 elements,
 *   - @exponent requires at least n bytes.
 *   - @signFrac requires at least 2n bytes.
 */
void FLTDECON_float16_deconstruct_encode(
        uint16_t const* __restrict src16,
        uint8_t* __restrict exponent,
        uint8_t* __restrict signFrac,
        size_t nbElts);

ZL_END_C_DECLS

#endif
