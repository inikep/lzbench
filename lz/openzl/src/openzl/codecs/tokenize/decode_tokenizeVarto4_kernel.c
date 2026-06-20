// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string.h> // memcpy
/* Note : using stdc assert() presumes that disabling is controlled by defining
 * NDEBUG this may have to be changed if there is some other form of centralized
 * debug control. On the other hand, keep in mind that it should be _possible_
 * for a transform to be fully external to the project, hence it should be
 * possible to _NOT_ depend on anything from zstrong core.
 */
#include <assert.h>

#include "openzl/codecs/tokenize/decode_tokenizeVarto4_kernel.h"

/* Naming convention :
 * Public symbols (invoked from outside this unit)
 * use prefix ZS_tokenizeVarto4_decode_*,
 * which is a bit long, but unique, and explicit.
 * Private static symbols use the shortened prefix ZS_tv4d_* for brevity.
 * It means the same thing, and is still unique.
 */

size_t ZS_tokenizeVarto4_decode_wkspSize(size_t alphabetSize)
{
    return ZS_TOKENIZEVARTO4_DECODE_WKSPSIZE(alphabetSize);
}

/* Note : it's likely that below memory copy functions
 *        will later be defined in a shared header file.
 *        When that will be the case, the shared header will be included,
 *        and below local copy will be removed.
 */

static inline void
overcpy_by(void* dst, const void* src, size_t len, size_t slabSize)
{
    memcpy(dst, src, slabSize);
    if (len > slabSize) {
        memcpy((char*)dst + slabSize, (const char*)src + slabSize, slabSize);
        if (len > 2 * slabSize) {
            size_t written = 2 * slabSize;
            do {
                memcpy((char*)dst + written,
                       (const char*)src + written,
                       slabSize);
                written += slabSize;
            } while (written < len);
        }
    }
}

typedef void(mem_f)(void* dst, const void* src, size_t size);

static void overcpy_by16(void* d, const void* s, size_t l)
{
    overcpy_by(d, s, l, 16);
}
static void overcpy_by32(void* d, const void* s, size_t l)
{
    overcpy_by(d, s, l, 32);
}
static void memcpy_generic(void* d, const void* s, size_t l)
{
    memcpy(d, s, l);
}

typedef enum { mce_by16, mce_by32, mce_generic } memcpy_e;

// Core transform loop
// Note : this design works well for "small" enough alphabets
// that fit into L1 or L2 cache.
// For larger alphabets, it will be preferable to employ prefetching.
// But prefetching is detrimental to small alphabets
// which is our main target for the tokenization transform.
static inline size_t ZS_tv4d_kernel_loop(
        void* dstBuffer,
        size_t dstCapacity,
        size_t* tokenSizes,
        const uint32_t* indexes,
        size_t nbTokens,
        const void* alphabetBuffer,
        size_t alphabetBufferSize,
        const SymbolDesc* sTable,
        size_t alphabetSize,
        mem_f cpyf)
{
    const char* const srcBuffer = alphabetBuffer;
    char* const dstPtr          = dstBuffer;
    size_t dstPos               = 0;
    for (size_t n = 0; n < nbTokens; n++) {
        uint32_t const index = indexes[n];
        assert(index < alphabetSize);
        (void)alphabetSize;
        size_t const len    = sTable[index].len;
        size_t const srcPos = sTable[index].pos;
        assert(srcPos + len <= alphabetBufferSize);
        (void)alphabetBufferSize;
        if (cpyf == overcpy_by16) {
            assert(srcPos + ((len - 1) / 16) + 16 <= alphabetBufferSize);
        }
        if (cpyf == overcpy_by32) {
            assert(srcPos + ((len - 1) / 32) + 32 <= alphabetBufferSize);
        }
        assert(dstPos + len <= dstCapacity);
        (void)dstCapacity;
        cpyf(dstPtr + dstPos, srcBuffer + srcPos, len);
        tokenSizes[n] = len;
        dstPos += len;
    }

    return dstPos;
}

static size_t ZS_tv4d_kernel_memSelector(
        void* dstBuffer,
        size_t dstCapacity,
        size_t* tokenSizes,
        const uint32_t* indexes,
        size_t nbTokens,
        const void* alphabetBuffer,
        size_t alphabetBufferSize,
        const SymbolDesc* sTable,
        size_t alphabetSize,
        memcpy_e memfid)
{
    switch (memfid) {
        case mce_by16:
            return ZS_tv4d_kernel_loop(
                    dstBuffer,
                    dstCapacity,
                    tokenSizes,
                    indexes,
                    nbTokens,
                    alphabetBuffer,
                    alphabetBufferSize,
                    sTable,
                    alphabetSize,
                    overcpy_by16);
        case mce_by32:
            return ZS_tv4d_kernel_loop(
                    dstBuffer,
                    dstCapacity,
                    tokenSizes,
                    indexes,
                    nbTokens,
                    alphabetBuffer,
                    alphabetBufferSize,
                    sTable,
                    alphabetSize,
                    overcpy_by32);
        case mce_generic:
            return ZS_tv4d_kernel_loop(
                    dstBuffer,
                    dstCapacity,
                    tokenSizes,
                    indexes,
                    nbTokens,
                    alphabetBuffer,
                    alphabetBufferSize,
                    sTable,
                    alphabetSize,
                    memcpy_generic);
        default:
            assert(0);
            return 0; /* unreachable */
    }
}

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
        size_t maxSymbolSize)

{
    // select memcpy function
    memcpy_e const memfid =
            (maxSymbolSize >= 18) ? mce_by32 : mce_by16; // heuristic threshold
    // overcpy() functions write more than "just" the exact symbol size.
    // ensure there is enough space before end of buffer to use overcpy
    // note : dstCapacity could also be _larger_ than regenerated size
    //        in which case, overcpy() could be used up to the last symbol!
    //        However, at this tage in the code, we don't know that (yet)
    //        Use worst case hypothesis,
    //        aka dstCapacity is _exactly_ the size to regenerate.
    size_t const trailingSpace = (maxSymbolSize >= 18) ? 32 : 16;

    size_t nbTrailingSymbols = 0;
    size_t countTS           = 0;
    while (countTS < trailingSpace) {
        nbTrailingSymbols++;
        countTS += sTable[alphabetSize - nbTrailingSymbols].len;
    }

    size_t dstSize = ZS_tv4d_kernel_memSelector(
            dstBuffer,
            dstCapacity,
            tokenSizes,
            indexes,
            nbTokens - nbTrailingSymbols,
            alphabetBuffer,
            alphabetBufferSize,
            sTable,
            alphabetSize,
            memfid);
    // finalize using safer memcpy()
    dstSize += ZS_tv4d_kernel_memSelector(
            (char*)dstBuffer + dstSize,
            dstCapacity,
            tokenSizes + nbTokens - nbTrailingSymbols,
            indexes + nbTokens - nbTrailingSymbols,
            nbTrailingSymbols,
            alphabetBuffer,
            alphabetBufferSize,
            sTable,
            alphabetSize,
            mce_generic); // writes exactly the nb of bytes of each symbol,
                          // hence no overwrite at buffer boundary
    return dstSize;
}

/* @symbolStarts and @symbolSizes are both arrays of @alphabetSize elts
 * @return : maxSymbolSize, largest value present in @symbolSizes
 *
 * note : it's possible that maxSymbolSize might be already known from metadata
 *        not recalculating here is an optimization that could be done later.
 *
 * note 2: if it wasn't for the search of max value,
 *         this function would be similar to delta_decode
 */
static inline size_t ZS_tv4d_buildSymbolTable(
        SymbolDesc* sTable,
        const size_t* symbolSizes,
        size_t alphabetSize)
{
    size_t max    = 0;
    sTable[0].pos = 0;
    max = sTable[0].len = (uint32_t)symbolSizes[0];
    for (size_t n = 1; n < alphabetSize; n++) {
        sTable[n].pos = sTable[n - 1].pos + sTable[n - 1].len;
        assert(symbolSizes[n] < UINT32_MAX);
        sTable[n].len = (uint32_t)symbolSizes[n];
        if (symbolSizes[n] > max) {
            max = symbolSizes[n];
        }
    }
    return max;
}

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
        size_t wkspSize)
{
    assert(tsaCapacity >= nbTokens);
    (void)tsaCapacity;

    // build symbolStarts array
    assert(wkspSize >= ZS_tokenizeVarto4_decode_wkspSize(alphabetSize));
    (void)wkspSize;
    assert((size_t)workspace % sizeof(uint64_t) == 0); // proper alignment
    SymbolDesc* const sTable = workspace;
    size_t const maxSymbolSize =
            ZS_tv4d_buildSymbolTable(sTable, symbolSizes, alphabetSize);

    return ZS_tokenizeVarto4_decode_kernel(
            dstBuffer,
            dstCapacity,
            tokenSizes,
            indexes,
            nbTokens,
            alphabetBuffer,
            alphabetBufferSize,
            sTable,
            alphabetSize,
            maxSymbolSize);
}
