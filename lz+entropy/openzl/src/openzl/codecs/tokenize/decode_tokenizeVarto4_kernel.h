// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_DECOMPRESS_TOKENIZEVARTO4_H
#define ZSTRONG_DECOMPRESS_TOKENIZEVARTO4_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uintx

/* ZS_tokenizeVarto4_decode():
 *
 * write into buffer @dst
 * from an array of 4-byte indexes stored into @indexes
 * translated using the @alphabetBuffer + @symbolSizes map.
 *
 * @return : size of data written into @dstBuffer (necessarily <= @dstCapacity)
 *
 * @tokenSizes: will contain the size of each token concatenated into
 * @dstbuffer. There will be necessarily @nbTokens sizes written. Therefore,
 * @tsaCapacity must be >= @nbTokens
 *
 * Conditions : all buffers must exist (!=NULL) and be valid
 *              @dstCapacity is presumed large enough to contain the decoded
 * content
 *              @workspace must be at least as large as @symbolSizes array
 *                        (see ZS_tokenizeVarto4_decode_wkspSize() below)
 *                         and aligned on 8-bytes boundaries.
 *              @alphabetBuffer must be larger than its content by a 32-bytes
 * guard band. its content has a size == sum of all @symbolSizes.
 *
 * Note : input is assumed to be correct;
 *        this function is not protected vs malicious inputs (beyond some
 * assert())
 *
 * Note 2 : @alphabetBuffer **requires** an end guard band (32-bytes)
 *          to ensure overcpy() doesn't read beyond its boundaries.
 *
 * Opened questions :
 *
 * - token sizes : size_t or uint32_t ?
 *   size_t feels more "natural" for an object's size,
 *   but uint32_t would be more compact (on 64-bit systems, which is our primary
 * target)
 */

size_t ZS_tokenizeVarto4_decode(
        void* dstBuffer,
        size_t dstCapacity,
        size_t* tokenSizes,
        size_t tsaCapacity,
        const uint32_t* indexes,
        size_t nbTokens,
        const void* alphabetBuffer,
        size_t alphabetBufferSize,
        const size_t* symbolSizes,
        size_t alphabetSize,
        void* workspace,
        size_t wkspSize);

// public symbol, for library
size_t ZS_tokenizeVarto4_decode_wkspSize(size_t alphabetSize);

// define, for static allocation
#define ZS_TOKENIZEVARTO4_DECODE_WKSPSIZE(as) ((as) * sizeof(uint64_t))

/* ZS_tokenizeVarto4_decode_kernel():
 * Inner function of ZS_tokenizeVarto4_decode(), for debugging purposes.
 * This function requires @sTable to be pre-calculated.
 * @maxSymbolSize, aka maximum value present in @symbolSizes array
 * must also be provided.
 */
typedef struct {
    uint32_t pos;
    uint32_t len;
} SymbolDesc;

size_t ZS_tokenizeVarto4_decode_kernel(
        void* dstBuffer,
        size_t dstCapacity,
        size_t* tokenSizes,
        const uint32_t* indexes,
        size_t nbTokens,
        const void* alphabetBuffer,
        size_t alphabetBufferSize,
        const SymbolDesc* sTable,
        size_t alphabetSize,
        size_t maxSymbolSize);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_DECOMPRESS_TOKENIZE4TO2_H
