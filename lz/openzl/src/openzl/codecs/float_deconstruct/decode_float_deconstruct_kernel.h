// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_FLOAT_DECONSTRUCT_DECODE_FLOAT_DECONSTRUCT_KERNEL_H
#define ZSTRONG_TRANSFORMS_FLOAT_DECONSTRUCT_DECODE_FLOAT_DECONSTRUCT_KERNEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/* Given the output of FLTDECON_float32_deconstruct_encode,
 * reconstructs the original buffer used to produce that output.
 *
 * @param dst32 output buffer with capacity for @nbElts 32-bit elements.
 * @param exponent buffer containing @nbElts exponent bytes pulled from
 * the original source buffer.
 * @param signFrac buffer containing 3 * @nbElts signFrac bytes pulled from
 * the original source buffer.
 */
void FLTDECON_float32_deconstruct_decode(
        uint32_t* __restrict dst32,
        uint8_t const* __restrict exponent,
        uint8_t const* __restrict signFrac,
        size_t nbElts);

/* Given the output of FLTDECON_bfloat16_deconstruct_encode,
 * reconstructs the original buffer used to produce that output.
 *
 * @param dst16 output buffer with capacity for @nbElts 16-bit elements.
 * @param exponent buffer containing @nbElts exponent bytes pulled from
 * the original source buffer.
 * @param signFrac buffer containing @nbElts signFrac bytes pulled from
 * the original source buffer.
 */
void FLTDECON_bfloat16_deconstruct_decode(
        uint16_t* __restrict dst16,
        uint8_t const* __restrict exponent,
        uint8_t const* __restrict signFrac,
        size_t nbElts);

/* Given the output of FLTDECON_float16_deconstruct_encode,
 * reconstructs the original buffer used to produce that output.
 *
 * @param dst16 output buffer with capacity for @nbElts 16-bit elements.
 * @param exponent buffer containing @nbElts exponent bytes constructed from
 * the original source buffer.
 * @param signFrac buffer containing 2 * @nbElts signFrac bytes constructed from
 * the original source buffer.
 */
void FLTDECON_float16_deconstruct_decode(
        uint16_t* __restrict dst16,
        uint8_t const* __restrict exponent,
        uint8_t const* __restrict signFrac,
        size_t nbElts);

ZL_END_C_DECLS

#endif
