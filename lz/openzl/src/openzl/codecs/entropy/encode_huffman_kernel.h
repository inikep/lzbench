// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_ENCODE_HUFFMAN_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_ENCODE_HUFFMAN_KERNEL_H

#include <stddef.h> // size_t

#include "openzl/common/cursor.h"
#include "openzl/common/zstrong_internal.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/// Deprecated
/// Encodes data using LA Huffman
/// @param maxSymbolValue The maximum possible symbol, can be any value but
/// smaller is faster
/// @param maxTableLog The max table log, must be <= 20. Set to 0 for default.
ZL_Report ZS_largeHuffmanEncode(
        ZL_WC* dst,
        uint16_t const* src,
        size_t size,
        uint16_t maxSymbolValue,
        int maxTableLog);

typedef struct {
    uint16_t symbol;
    uint16_t nbBits;
} ZS_Huf16CElt;

/// Builds a Huffman table using the given histogram.
/// @param maxSymbolValue The maximum possible symbol, can be any value but
/// smaller is faster
/// @param maxTableLog The max table log, must be <= 20. Set to 0 for default.
ZL_Report ZS_largeHuffmanBuildCTable(
        ZS_Huf16CElt* ctable,
        unsigned const* histogram,
        uint16_t maxSymbolValue,
        int maxNbBits);

/// Deprecated
/// Encodes the Huffman table
ZL_Report ZS_largeHuffmanWriteCTable(
        ZL_WC* dst,
        ZS_Huf16CElt const* ctable,
        uint16_t maxSymbolValue,
        int maxNbBits);

/// Encodes data using the given Huffman table with 1 stream
/// @returns Success or an error code.
ZL_Report ZS_largeHuffmanEncodeUsingCTable(
        ZL_WC* dst,
        uint16_t const* src,
        size_t size,
        ZS_Huf16CElt const* ctable,
        int maxNbBits);

/// Encodes data using the given Huffman table with 4 streams
/// @returns Success or an error code.
ZL_Report ZS_largeHuffmanEncodeUsingCTableX4(
        ZL_WC* dst,
        uint16_t const* src,
        size_t size,
        ZS_Huf16CElt const* ctable,
        int maxNbBits);

ZL_END_C_DECLS

#endif
