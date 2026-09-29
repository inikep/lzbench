// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_TOKENIZE_DECODE_TOKENIZE_KERNEL_H
#define ZSTRONG_TRANSFORMS_TOKENIZE_DECODE_TOKENIZE_KERNEL_H

#include "openzl/shared/portability.h"
#include "openzl/zl_data.h"

ZL_BEGIN_C_DECLS

/**
 * Validates that all indices are less than @p alphabetSize.
 *
 * @returns true iff the indices are valid.
 */
bool ZS_tokenizeValidateIndices(
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t idxWidth);

/**
 * Decodes a tokenization transform. This function guarantees safety on
 * all inputs, including corrupted inputs.
 *
 * @param dst Destination buffer with capacity @p nbElts of width @p eltWidth.
 * @param alphabet The alphabet symbols of width @p eltWidth.
 * @param alphabetSize The number of symbols in the @p alphabet.
 * @param indices The indices into the alphabet of width @p idxWidth.
 * @param eltWidth The width of the symbols in @p dst and @p alphabet.
 * @param idxWidth The width of the @p indices.
 *
 * @returns true if decoding succeeded, and @p dst is filled with @p nbElts
 * symbols. Otherwise decoding failed, and @p dst may not be filled. This
 * function may silently succeed on corrupted inputs, e.g. on out of bounds
 * indices. If you want to detect out of bounds indicies, call
 * @ref ZS_tokenizeValidateIndices.
 */
bool ZS_tokenizeDecode(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth);

/**
 * Calculates the size of the original input variable-size field stream
 *
 * @returns the number of bytes used by the unencoded variable-size field stream
 *          (i.e., `ZL_Input_contentSize`)
 */
size_t ZS_tokenizeComputeVSFContentSize(
        const uint8_t* indices,
        size_t idxWidth,
        size_t nbElts,
        const uint32_t* alphabetFieldSizes,
        size_t alphabetSize);

/**
 * Calcualtes the size of the workspace needed for ZS_tokenizeVSFDecode.
 *
 * @returns the number of bytes needed by the workspace for ZS_tokenizeVSFDecode
 */
size_t ZS_tokenizeVSFDecodeWorkspaceSize(
        size_t const alphabetSize,
        size_t const alphabetFieldSizesSum);

/**
 * Decodes a variable-size field tokenization transform. This function
 * guarantees safety on all inputs, including corrupted inputs
 *
 * @param alphabet The stream buffer of unique tokens that comprise the
 * alphabet
 * @param alphabetSize The number of tokens in the @p alphabet
 * @param indices The stream buffer of indices that replace elements in
 * the original source stream with their corresponding indices
 * @param alphabetFieldSizes The width of each element in @p alphabet
 * @param alphabetFieldSizesSum The sum of each element in @p alphabetFieldSizes
 * @param out The stream buffer into which elements in the indices
 * stream are replaced by their coresponding element in the alphabet
 * stream
 * @param dstFieldSizes The width of each element of @p out that will be
 * set if decoding succeeds
 * @param dstNbElts The number of elements in @p out
 * @param idxWidth The minimum number of bytes used to encode each index
 * @param workspace Workspace needed for ZS_tokenizeVSFDecode, must be allocated
 * to be at least as large as ZS_tokenizeVSFDecodeWorkspaceSize indicates.
 * @note This function assumes all inputs are validated beforehand
 * (i.e., null, stream type, and corruption checks). The only check provided by
 * this function is that all indices are less than @p alphabetSize
 */
void ZS_tokenizeVSFDecode(
        const uint8_t* alphabet,
        size_t alphabetSize,
        const uint8_t* indices,
        const uint32_t* alphabetFieldSizes,
        size_t alphabetFieldSizesSum,
        uint8_t* out,
        uint32_t* dstFieldSizes,
        size_t dstNbElts,
        size_t dstNbBytes,
        size_t idxWidth,
        void* workspace);

ZL_END_C_DECLS

#endif
