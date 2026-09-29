// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/float_deconstruct/decode_float_deconstruct_kernel.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

static void float32_deconstruct_decode_scalar(
        uint32_t* __restrict const dst32,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    for (size_t i = 0; i < nbElts; ++i) {
        uint32_t f32 = 0;
        f32 |= (uint32_t)exponent[i] << 23;

        // TODO: measure if 4-byte copy is faster,
        // like on the encoding side
        uint32_t const sf = ZL_readCE24(signFrac + 3 * i);

        f32 |= (sf << 31) | (sf >> 1);
        dst32[i] = f32;
    }
}

static void bfloat16_deconstruct_decode_scalar(
        uint16_t* __restrict const dst16,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    for (size_t i = 0; i < nbElts; ++i) {
        uint8_t const sf = signFrac[i];
        uint16_t frac    = (uint16_t)(sf >> 1);
        uint16_t sign    = (uint16_t)(sf << 15);
        uint16_t expnt   = (uint16_t)(exponent[i] << 7);
        dst16[i]         = sign | expnt | frac;
    }
}

static void float16_deconstruct_decode_scalar(
        uint16_t* __restrict const dst16,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    for (size_t i = 0; i < nbElts; ++i) {
        uint16_t const sf = ZL_readCE16(signFrac + 2 * i);
        uint16_t frac     = (uint16_t)(sf >> 1);
        uint16_t sign     = (uint16_t)(sf << 15);
        uint16_t expnt    = (uint16_t)(exponent[i] << 10);
        dst16[i]          = sign | expnt | frac;
    }
}

void FLTDECON_float32_deconstruct_decode(
        uint32_t* __restrict const dst32,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    // TODO(embg) Add an AVX2 variant
    float32_deconstruct_decode_scalar(dst32, exponent, signFrac, nbElts);
}

void FLTDECON_bfloat16_deconstruct_decode(
        uint16_t* __restrict const dst16,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    // TODO(embg) Add an AVX2 variant
    bfloat16_deconstruct_decode_scalar(dst16, exponent, signFrac, nbElts);
}

void FLTDECON_float16_deconstruct_decode(
        uint16_t* __restrict const dst16,
        uint8_t const* __restrict const exponent,
        uint8_t const* __restrict const signFrac,
        size_t const nbElts)
{
    // Clang 14 does a great job auto-vectorizing this.
    // For now, seems like we don't need a hand-optimized AVX2 kernel.
    // Godbolt: https://godbolt.org/z/TWT3TfWjn
    float16_deconstruct_decode_scalar(dst16, exponent, signFrac, nbElts);
}
