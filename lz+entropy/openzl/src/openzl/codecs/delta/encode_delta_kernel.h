// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DELTA_ENCODE_DELTA_KERNEL_H
#define ZSTRONG_TRANSFORMS_DELTA_ENCODE_DELTA_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#include "openzl/zl_portability.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Delta encodes the elements in @p src, stores the first element to @p first,
 * and the deltas to @p deltas.
 *
 * @param src The source array. Contains @p nbElts elements.
 * @param nbElts the nb of elts in @p src.
 * @param first Output for the first element of @p src.
 * Not filled when @p nbElts == 0.
 * @param dst Output for the deltas of the last @p nbElts - 1 elements.
 *
 * NOTE: Presuming all assumptions above are correctly verified,
 * these functions are always successful.
 */
void ZS_deltaEncode8(
        uint8_t* first,
        uint8_t* deltas,
        uint8_t const* src,
        size_t nbElts);
void ZS_deltaEncode16(
        uint16_t* first,
        uint16_t* deltas,
        uint16_t const* src,
        size_t nbElts);
void ZS_deltaEncode32(
        uint32_t* first,
        uint32_t* deltas,
        uint32_t const* src,
        size_t nbElts);
void ZS_deltaEncode64(
        uint64_t* first,
        uint64_t* deltas,
        uint64_t const* src,
        size_t nbElts);

/// NOTE: @p first doesn't have to be aligned to @p deltas does.
void ZS_deltaEncode(
        void* first,
        void* deltas,
        void const* src,
        size_t nbElts,
        size_t eltWidth);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_TRANSFORMS_DELTA_ENCODE_DELTA_KERNEL_H
