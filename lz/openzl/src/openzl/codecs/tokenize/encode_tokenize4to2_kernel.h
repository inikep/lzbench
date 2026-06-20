// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_TOKENIZE4TO2_H
#define ZSTRONG_COMPRESS_TOKENIZE4TO2_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uintx

typedef enum {
    ZS_tam_unsorted = 0,
    ZS_tam_sorted   = 1,
} ZS_TokenizeAlphabetMode;

/* ZS_tokenize4to2_encode():
 * accepts as input an array of (fixed size) 4-bytes symbols.
 *
 * @return : alphabetSize (nb of different 4-bytes symbols)
 *
 * @dstAlphabet : list of unique symbols present in srcSymbols
 *    + alphabetCapacity is expressed in nb of 4-bytes symbols
 *
 * @dstIndex : the list of indexes, using 2 bytes per index.
 *    + There are necessarily nbSymbols indexes written into dstIndex
 *    + indexCapacity must be >= nbSymbols
 *
 * Conditions : the nb of different symbols (alphabetSize) MUST be <= 65536
 *              and alphabetCapacity MUST be >= alphabetSize
 *              all pointers are presumed valid and non NULL
 *
 * Opened topics :
 *
 * - The function requires a workspace for the hashset.
 *   This workspace is currently allocated directly, from heap,
 *   but it would be preferable to not allocate in the transform.
 *
 * - Avoiding any dynamic allocation within the transform requires
 *   to correctly size the hashset at its creation.

 * - This function will therefore require a targetCardinalityLog parameter,
 *   so that the hashset can be sized directly to an appropriate size.
 *
 * - This requires a cardinality estimator, like hyperloglog,
 *   which could be provided through a dedicated function.
 *   It would help both for allocation of the workspace,
 *   and for proper sizing of alphabetCapacity.
 *
 * - The cardinality estimator can also be useful
 *   just to evaluate the benefit of the tokenization transform
 *   before deciding to trigger it (dynamic decision mode)
 *
 * - sorted list of symbols in the alphabet:
 *   In contrast with (current) 2to1 tokenizer,
 *   4to2 doesn't sort symbols in the dictionary by default.
 *   Symbols can be sorted, but it costs a non negligible amount of time,
 *   requiring 2 passes, and applying a sort function
 *   (in contrast with faster single-pass mode).
 *   This mode is enabled with alphabetMode == ZS_tam_sorted.
 */

size_t ZS_tokenize4to2_encode(
        uint16_t* dstIndex,
        size_t indexCapacity,
        uint32_t* dstAlphabet,
        size_t alphabetCapacity,
        const uint32_t* srcSymbols,
        size_t nbSymbols,
        ZS_TokenizeAlphabetMode alphabetMode);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_COMPRESS_TOKENIZE4TO2_H
