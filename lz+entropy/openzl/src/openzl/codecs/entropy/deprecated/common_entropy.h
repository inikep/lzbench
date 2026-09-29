// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_COMMON_ENTROPY_H
#define ZSTRONG_COMMON_ENTROPY_H

#include "openzl/common/assertion.h"
#include "openzl/common/cursor.h"
#include "openzl/common/speed.h"
#include "openzl/shared/histogram.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Entropy:
 *
 * This library provides generic O0 entropy compression for ZStrong.
 * It provides an common and efficient header format for the supported
 * entropy compression methods in ZStrong.
 *
 * Sharing a common format the supports multiple entropy codecs gives
 * users the flexibility to select different [de]compression speed and
 * ratio tradeoffs, without changing the decoder. E.g. if Literal encoding
 * normally uses Huffman, but one file has a particularly uneven distribution,
 * it can change to FSE. Or if a file is Base-64 encoded random data, the
 * BIT format can be used.
 *
 * This API is flexible on the element size because much of the code for
 * headers, selecting encoding types, block splitting, etc. is independent of
 * element size. Underneath it will dispatch to the right {en,de}coder based on
 * the element size. Not all sizes and cardinalities must be supported by all
 * types, just what we need in practice.
 *
 * The header format is designed to be as small as possible for small data and
 * repeated tables. This is important for dictionary compression, because we
 * don't want header costs to dominate.
 *
 * The HUF/FSE methods allow repeated tables. The repeated tables are injected
 * using the ZS_Entropy_TableManager_t. This allows different use cases to
 * select how they handle their repeated tables. E.g. they could not support
 * repeated tables, or they could have a static set of pre-built tables, or they
 * could use LRU. The entropy compressor shouldn't care, and should work for all
 * of these cases, so it is injected to support them all. The only constraint is
 * that only at most ZS_ENTROPY_MAX_TABLE_MANAGER_SIZE repeated tables are
 * supported.
 *
 * The user controls which methods are supported with the allowedTypes
 * parameter. Additionally, they can constrain the {en,de}coding speed and the
 * decoding speed vs. ratio tradeoff.
 *
 * The supported methods are listed below, and there is space for two more
 * formats in the header without any extra cost:
 * * Huf: Huffman encoding - 2-bytes element size max.
 * * Fse: FSE encoding - 1-byte element size max currently.
 * * Constant: Constant encoding - 1-8 byte element size.
 * * Raw: Raw encoding - 1-8 byte element size.
 * * Bit: Bit-packing  - 1-8 byte element size.
 * format?
 * * Multi: Recursive entropy compression - block splitting.
 */

/**
 * Format:
 *
 * Brief format description, likely belongs in its own doc eventually, but
 * here for now.
 *
 * Shared Header:
 * bits [0, 3) - ZS_Entropy_Type_e in the low 3-bits
 *
 * Huf & Fse:
 * bits [3, 5) - table mode {0-2 = repeat-index, 3 = inline-table}
 * bits [5, 6) - format flag (e.g. AVX2 Huf). TODO(terrelln): Remove.
 * bits [6, 7) - large-size {0 = sizes fit in header, 1 = extra varints}
 * bits [7, 12) - Low 5 bits of decoded size
 * bits [12, 16) - Low 4 bits of encoded size
 * If large-size: Decoded size varint.
 * If large-size: Encoded size varint.
 * If inline-table: Table.
 * Encoded data.
 *
 * Raw & Constant Header:
 * bits [3, 8) - Decoded size varint (high bit set means more varint-bytes)
 * If high bit set: Decoded size varint.
 * If Constant: Single element.
 * If Raw: Decoded size elements.
 *
 * Bit header:
 * bits [3-8) - # bits
 * Varint - decoded size
 *
 * Multi header:
 * bits [3, 8) - Num sub-blocks varint (high bit set means more varint-bytes)
 * If high bit set: Num sub-blocks varint.
 * Sub-blocks: Each follow the entropy format.
 */

#define ZS_ENTROPY_MAX_TABLE_MANAGER_SIZE 3

typedef enum {
    ZS_Entropy_Type_huf       = 0,
    ZS_Entropy_Type_fse       = 1,
    ZS_Entropy_Type_constant  = 2,
    ZS_Entropy_Type_raw       = 3,
    ZS_Entropy_Type_bit       = 4,
    ZS_Entropy_Type_multi     = 5,
    ZS_Entropy_Type_reserved0 = 6,
    ZS_Entropy_Type_reserved1 = 7,
} ZS_Entropy_Type_e;

typedef enum {
    ZS_Entropy_TypeMask_huf      = 1 << ZS_Entropy_Type_huf,
    ZS_Entropy_TypeMask_fse      = 1 << ZS_Entropy_Type_fse,
    ZS_Entropy_TypeMask_constant = 1 << ZS_Entropy_Type_constant,
    ZS_Entropy_TypeMask_raw      = 1 << ZS_Entropy_Type_raw,
    ZS_Entropy_TypeMask_bit      = 1 << ZS_Entropy_Type_bit,
    ZS_Entropy_TypeMask_multi    = 1 << ZS_Entropy_Type_multi,
    ZS_Entropy_TypeMask_all      = -1,
} ZS_Entropy_TypeMask_e;

/// TODO: This interface isn't 100% finished. This is about what I want,
/// but we need experience to figure out if this is right.
/// E.g. do we really want to have void*, or a more specific type.
typedef struct ZS_Entropy_TableManager_s {
    /// Returns the table at index or NULL if none.
    void const* (*getTable)(
            const struct ZS_Entropy_TableManager_s* manager,
            ZS_Entropy_Type_e type,
            size_t index);
    /// Tell the table manager that we used the table at index.
    void (*useTable)(
            struct ZS_Entropy_TableManager_s* manager,
            ZS_Entropy_Type_e type,
            size_t index);
    /// Tell the table manager that we are using a new table.
    /// Passes the table to the table manager to manage.
    void (*newTable)(
            struct ZS_Entropy_TableManager_s* manager,
            void const* table,
            ZS_Entropy_Type_e type);
} ZS_Entropy_TableManager;

/// A set of block splits to enforce.
/// E.g. the splits [A, B, C]. Split the block of length N into:
/// [0, A), [A, B), [B, C), [C, N).
typedef struct {
    /// The element that we should split at.
    /// Each split must be > 0 and less than the input size.
    /// And the splits must be strictly increasing.
    size_t const* splits;
    /// The total number of splits.
    size_t nbSplits;
} ZS_Entropy_BlockSplits;

typedef struct {
    /// The encoding types that the encoder is allowed to use.
    /// The decoder must support all the types the encoder supports.
    ZS_Entropy_TypeMask_e allowedTypes;
    /// How fast the entropy coder needs to work. Might constrain
    /// it to RAW/Constant/BIT only. Or might limit the number of options
    /// tried.
    ZL_EncodeSpeed encodeSpeed;
    /// How fast the decoder needs to run. Also can allow specifying
    /// a ratio/decompression speed tradeoff.
    ZL_DecodeSpeed decodeSpeed;
    /// A pre-computed histogram or NULL.
    ZL_Histogram const* precomputedHistogram;
    /// An estimate of the cardinality (not max element) or 0.
    uint64_t cardinalityEstimate;
    /// The maximum value estimate value or 0.
    uint64_t maxValueUpperBound;
    /// The maximum allowed tableLog or 0 for default.
    uint32_t maxTableLog;
    /// Use AVX2 Huffman?
    bool allowAvx2Huffman;
    // Number of parallel FSE states to use, must be the same in encoder and
    // decoder
    uint8_t fseNbStates;
    /// Optionally a list of block splits to be used.
    /// Ignored if allowedTypes does not contain ZS_Entropy_TypeMask_multi.
    ZS_Entropy_BlockSplits const* blockSplits;
    /// The table manager for repeated tables or NULL.
    /// The decoder must have the same table manager.
    ZS_Entropy_TableManager* tableManager;
} ZS_Entropy_EncodeParameters;

/// Returns default encoder parameters with the given allowedTypes.
ZS_Entropy_EncodeParameters ZS_Entropy_EncodeParameters_fromAllowedTypes(
        ZS_Entropy_TypeMask_e allowedTypes);

typedef struct {
    /// Control decoding speed by only allowing faster modes
    /// at the expense of less flexibility in the future.
    ZS_Entropy_TypeMask_e allowedTypes;
    /// The table manager for repeated tables or NULL.
    ZS_Entropy_TableManager* tableManager;
    // Number of parallel FSE states to use, must be the same in encoder and
    // decoder
    uint8_t fseNbStates;
} ZS_Entropy_DecodeParameters;

/// Returns default decoder parameters that allow decoding any type.
ZS_Entropy_DecodeParameters ZS_Entropy_DecodeParameters_default(void);

/// @returns an upper bound on the encoded size.
size_t ZS_Entropy_encodedSizeBound(size_t srcSize, size_t elementSize);

/// Encodes the source using the given parameters.
/// A ZL_WC is used for the output buffer because it is a byte-container
/// that we are only writing the prefix of.
/// @returns Okay or an error.
ZL_Report ZS_Entropy_encode(
        ZL_WC* dst,
        void const* src,
        size_t srcSize,
        size_t elementSize,
        ZS_Entropy_EncodeParameters const* params);

/// Simplified API which encodes using Huf | Constant | Raw | Multi.
/// Calls ZS_Entropy_encode() under the hood.
ZL_Report ZS_Entropy_encodeHuf(
        ZL_WC* dst,
        void const* src,
        size_t srcSize,
        size_t elementSize);

/// Simplified API which encodes using FSE | Constant | Raw | Multi.
/// Calls ZS_Entropy_encode() under the hood.
/// nbStates controls the number of parallel FSE states, using 0 will
/// result in using default.
ZL_Report ZS_Entropy_encodeFse(
        ZL_WC* dst,
        void const* src,
        size_t srcSize,
        size_t elementSize,
        uint8_t nbStates);

/// Reads the encoded size from the entropy frame.
/// @returns the encoded size or an error.
ZL_Report
ZS_Entropy_getEncodedSize(void const* src, size_t srcSize, size_t elementSize);

/// Reads the decoded size from the entropy frame.
/// @returns the decoded size or an error.
ZL_Report
ZS_Entropy_getDecodedSize(void const* src, size_t srcSize, size_t elementSize);

/// Reads the encoding type from the frame.
/// @returns the ZS_Entropy_Type_e encoding type or an error.
ZL_Report ZS_Entropy_getType(void const* src, size_t srcSize);

/// Decodes an entropy compressed frame.
/// A ZL_RC is used for the input buffer because it is a byte-container that
/// we are only peeling the prefix off of.
/// @returns The number of elements written or an error.
ZL_Report ZS_Entropy_decode(
        void* dst,
        size_t dstCapacity,
        ZL_RC* src,
        size_t elementSize,
        ZS_Entropy_DecodeParameters const* params);

/// Calls ZS_Entropy_decode() with the default parameters.
/// All encoding types are supported.
ZL_Report ZS_Entropy_decodeDefault(
        void* dst,
        size_t dstCapacity,
        ZL_RC* src,
        size_t elementSize);

/// Returns the header size.
/// This can be used to determine where the RAW data begins to avoid a memcpy.
ZL_Report ZS_Entropy_getHeaderSize(void const* src, size_t srcSize);

ZS_Entropy_TableManager* ZS_Entropy_LRUTableManager_create(size_t maxTables);
void ZS_Entropy_LRUTableManager_destroy(ZS_Entropy_TableManager* manager);

/**
 * Low level API:
 *
 * This API is what is used to {en,de}code each of the individual formats
 * without the header. That means that the srcSize and dstSize must be
 * exact.
 */

// /// @returns Okay or an error.
// ZL_Report ZS_Fse_encode(
//   ZL_WC* dst,
//   void const* src,
//   size_t srcSize,
//   size_t elementSize,
//   ZS_Entropy_DecodeParameters_t const* params);

// /// @returns Okay or an error.
// ZL_Report ZS_Huf_encode(
//   ZL_WC* dst,
//   void const* src,
//   size_t srcSize,
//   size_t elementSize,
//   ZS_Entropy_DecodeParameters_t const* params);

/// @returns Okay or an error.
ZL_Report ZS_Constant_encode(ZL_WC* dst, void const* src, size_t elementSize);

/// @returns Okay or an error.
ZL_Report
ZS_Raw_encode(ZL_WC* dst, void const* src, size_t srcSize, size_t elementSize);

ZL_Report ZS_Fse_decode(
        void* dst,
        size_t dstSize,
        void const* src,
        size_t srcSize,
        size_t elementSize,
        ZS_Entropy_DecodeParameters const* params);

ZL_Report ZS_Huf_decode(
        void* dst,
        size_t dstSize,
        void const* src,
        size_t srcSize,
        size_t elementSize,
        ZS_Entropy_DecodeParameters const* params);

ZL_Report ZS_Raw_decode(
        void* const dst,
        size_t const dstSize,
        void const* const src,
        size_t const srcSize,
        size_t const elementSize);

ZL_Report ZS_Constant_decode(
        void* dst,
        size_t dstSize,
        void const* src,
        size_t srcSize,
        size_t elementSize);

ZL_Report ZS_Bit_decode(
        void* dst,
        size_t dstSize,
        void const* src,
        size_t srcSize,
        size_t elementSize,
        size_t numBits);

// TODO: FSE/HUF build encoding tables and decoding tables
//       for the table manager & dictionaries & pre-defined

/**
 * Implementation details
 */

#define ZS_HUF_MAX_BLOCK_SIZE (1u << 17)
#define ZS_HUF16_MAX_TABLE_LOG (13)

#define ZS_ENTROPY_AVX2_HUFFMAN_DEFAULT false

#define ZS_ENTROPY_DEFAULT_FSE_NBSTATES (2)

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_ENTROPY_H
