// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_TOKENIZEVARTO4_H
#define ZSTRONG_COMPRESS_TOKENIZEVARTO4_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uintx

/* ZS_tokenizeVarto4_encode():
 * accepts as input a buffer @srcBuffer of size @srcBufferSize
 * divided into @nbTokens elts described by their sizes in @tokenSizes.
 * The sum of all tokenSizes must be == @srcBufferSize.
 *
 * @return : a structure ZS_TokVar_result with 2 fields
 *      + @alphabetSize (nb of unique symbols)
 *        which tells how many values are present in @symbolSizes
 *        (necessairly <= @ssaCapacity)
 *      + @dstSize: amount of data written into @dstBuffer.
 *        they are the content of all unique symbols concatenated back to back.
 *        The sum of all values in @symbolSizes is == @dstSize.
 *
 * @dstIndex : the list of indexes, using 4 bytes per index.
 *    + There are necessarily @nbTokens indexes written into @dstIndex
 *    + @indexCapacity must be >= @nbTokens
 *    + @indexCapacity is expressed as nb of 4 bytes indexes
 *
 * @dstBuffer : concatenation of all _unique_ elements
 *    + To cover worst case scenarios, @dstCapacity should be >= @srcBufferSize
 *
 * @symbolSizes: size of each unique symbol concatenated into @dstBuffer
 *    + To cover worst case scenarios, @ssaCapacity should be >= @nbTokens
 *
 * Conditions : all pointers are presumed valid and non NULL
 *    + @workspace is a scratch buffer
 *      its size @wkspSize must be >=
 * ZS_tokenizeVarto4_encode_wkspSize(cardinalityEstimation)
 *
 *
 * Opened topics :
 *
 * - Avoiding dynamic re-allocation *within* the transform's hot loop
 *   requires to correctly size the hashMap at its creation.
 *   Therefore, this function requests a @cardinalityEstimation parameter,
 *   so that the hashMap can be sized directly to an appropriate size.
 *   @cardinalityEstimation doesn't have to be precise,
 *   but it should be "about right" (within <= ~30%).
 *   When in doubt, provide a sure over-estimate:
 *   this will result in over-allocation (and corresponding initialization)
 *   but at least, the algorithm will work properly.
 *   Under-evaluation, in contrast, can lead to infinite loop.
 *
 * - This requires a cardinality estimator, like hyperloglog,
 *   which can be provided through another dedicated function.
 *   Cardinality estimation is helpful both for workspace's allocation,
 *   for proper sizing of alphabetCapacity,
 *   and also to evaluate the potential benefits of tokenization transform
 *   before deciding to trigger it (dynamic decision mode).
 *
 * - Order of symbols in the alphabet:
 *   This tokenizer orders IDs in token appearance order in the source.
 *   Symbols could be sorted differently, using another rule,
 *   for example using lexicographically order,
 *   but it would require 2 passes, applying a sort function,
 *   which would cost a non trivial additional amount of cpu time,
 *   (in contrast with current single-pass design)
 *
 * - token sizes : size_t or uint32_t ?
 */

typedef struct {
    size_t dstSize;
    size_t alphabetSize;
} ZS_TokVar_result;

ZS_TokVar_result ZS_tokenizeVarto4_encode(
        uint32_t* dstIndex,
        size_t indexCapacity,
        void* dstBuffer,
        size_t dstCapacity,
        size_t* symbolSizes,
        size_t ssaCapacity,
        const void* srcBuffer,
        size_t srcBufferSize,
        const size_t* tokenSizes,
        size_t nbTokens,
        uint32_t cardinalityEstimation,
        void* wksp,
        size_t wkspSize);

// public symbol, for library
size_t ZS_tokenizeVarto4_encode_wkspSize(uint32_t cardinalityEstimation);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_COMPRESS_TOKENIZE4TO2_H
