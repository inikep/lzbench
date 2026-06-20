// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_DECODE_HUFFMAN_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_DECODE_HUFFMAN_KERNEL_H

#include "openzl/common/base_types.h" // ZL_Report
#include "openzl/common/cursor.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

#define ZS_kLargeHuffmanMaxNbBits 20

/// Deprecated
/// @returns the number of elements decoded
ZL_Report ZS_largeHuffmanDecode(uint16_t* dst, size_t capacity, ZL_RC* src);

typedef struct {
    uint16_t symbol;
    uint16_t nbBits;
} ZS_Huf16DElt;

/// Deprecated
/// Reads the encoded Huffman table from source.
/// @returns The Huffman table, must be freed
ZS_Huf16DElt* ZS_largeHuffmanCreateDTable(ZL_RC* src, int* tableLogPtr);

/// @returns true iff the Huffman weights are valid and @ref
/// ZS_largeHuffmanBuildDTable can be called.
///
/// Checks 3 conditions:
/// 1. The tableLog is not larger than ZS_kLargeHuffmanMaxNbBits
/// 2. The weights ((1 << w) >> 1) add up to 1 << tableLog
/// 3. There are at least 2 non-zero weights
bool ZS_largeHuffmanValidWeights(
        const uint8_t* weights,
        size_t nbWeights,
        int tableLog);

/// Builds the Huffman table from weights.
/// @pre ZS_largeHuffmanValidWeights(weights, nbWeights, tableLog)
/// @param dtable Must have 2^tableLog elements
void ZS_largeHuffmanBuildDTable(
        ZS_Huf16DElt* dtable,
        uint8_t const* weights,
        size_t nbWeights,
        int tableLog);

/// Decodes Huffman given the DTable using 4 streams
/// @returns The number of decoded elements
ZL_Report ZS_largeHuffmanDecodeUsingDTableX4(
        uint16_t* dst,
        size_t capacity,
        ZL_RC* src,
        ZS_Huf16DElt const* dtable,
        int tableLog);

/// Decodes Huffman given the DTable using 1 stream
/// @returns The number of decoded elements
ZL_Report ZS_largeHuffmanDecodeUsingDTable(
        uint16_t* dst,
        size_t capacity,
        ZL_RC* src,
        ZS_Huf16DElt const* dtable,
        int tableLog);

ZL_END_C_DECLS

#endif
