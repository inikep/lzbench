// Copyright (c) Meta Platforms, Inc. and affiliates.

/// MinGW: Use the ANSI stdio functions (e.g. to get correct printf for 64-bits)
#undef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1

/* ===   Dependencies   === */
#include "benchmark/unitBench/scenarios/codecs/tokenize.h"

#include <stdio.h>  // printf, fflush
#include <stdlib.h> // malloc, free, exit, abort

#include "openzl/codecs/tokenize/decode_tokenize2to1_kernel.h"
#include "openzl/codecs/tokenize/decode_tokenizeVarto4_kernel.h"
#include "openzl/codecs/tokenize/encode_tokenize2to1_kernel.h"
#include "openzl/codecs/tokenize/encode_tokenize4to2_kernel.h"
#include "openzl/codecs/tokenize/encode_tokenizeVarto4_kernel.h"

size_t tokenize2to1Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    assert(dstCapacity > srcSize + 512);
    (void)dstCapacity;

    const uint16_t* const src16 = src;
    size_t const src16Size      = srcSize / sizeof(uint16_t);
    uint8_t* dst8               = dst;
    size_t const dst8Size       = src16Size;
    uint16_t* alphabet16        = (uint16_t*)dst + dst8Size;
    size_t const alphabetSize   = 256;

    return ZS_tokenize2to1_encode(
            dst8, dst8Size, alphabet16, alphabetSize, src16, src16Size);
}

size_t tokenize2to1Decode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    assert(srcSize >= 512); // fake alphabet map
    assert(dstCapacity >= 2 * srcSize);

    uint16_t* const dst16            = dst;
    size_t const dst16Capacity       = dstCapacity / sizeof(uint16_t);
    const uint16_t* const alphabet16 = src;
    size_t const alphabetSize        = 256;

    return ZS_tokenize2to1_decode(
            dst16, dst16Capacity, src, srcSize, alphabet16, alphabetSize);
}
void tokenize2to1Decode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    double const sec           = rt.nanoSecPerRun / 1000000000.;
    double const nbRunsPerSec  = 1. / sec;
    double const nbBytesPerSec = nbRunsPerSec * (double)(srcSize * 2);

    printf("%s: decode %zu 8-bit indexes into %zu 16-bit tokens (using a fake map) in %.2f ms  ==> %.1f MB/s \n",
           fname,
           srcSize,
           rt.sumOfReturn,
           sec * 1000.,
           nbBytesPerSec / (1 << 20));
    (void)srcname;
    fflush(NULL);
}

size_t tokenize4to2Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    assert(dstCapacity > srcSize + 65536 * 4);
    (void)dstCapacity;

    const uint32_t* const src32 = src;
    size_t const src32Size      = srcSize / sizeof(uint32_t);
    uint16_t* dst16             = dst;
    size_t const dst16Size      = src32Size;
    uint32_t* alphabet32        = (uint32_t*)dst + dst16Size;
    size_t const alphabetSize   = 65536;

    return ZS_tokenize4to2_encode(
            dst16,
            dst16Size,
            alphabet32,
            alphabetSize,
            src32,
            src32Size,
            ZS_tam_unsorted);
}

// TokenizeVarto4 test :
size_t tokenizeVarto4Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    // In this scenario, input is presumed prepared.
    // First, input presents an array of token lengths
    // (array size, then lengths) as 4 bytes integers.
    // Then the rest of the input contains the tokens.
    // The sum of length must be == size of input (so last token makes up for
    // it). Input is also altered to ensure that alphabet remains under control.
    // Effectively, tokens are repeated.
    // dstCapacity must be >= 4 x srcSize to guarantee non overlapping output

    // size_t    alphabetSize
    // size_t    nbTokens
    // size_t*   tokenSizes
    // void*     srcContent

    (void)customPayload;

    const void* const srcEnd             = (const char*)src + srcSize;
    uint32_t const cardinalityEstimation = (uint32_t)((const size_t*)src)[0];
    size_t const nbTokens                = ((const size_t*)src)[1];
    const size_t* const tokenSizes       = ((const size_t*)src) + 2;
    const size_t* const tsaEnd           = tokenSizes + nbTokens;
    const void* const srcContent         = tsaEnd;
    assert(srcEnd >= srcContent);
    size_t const srcBufferSize =
            (size_t)((const char*)srcEnd - (const char*)srcContent);

    uint32_t* const dstIndex   = dst;
    size_t const indexCapacity = nbTokens;
    void* const indexEnd       = dstIndex + nbTokens;
    size_t* const symbolSizes  = indexEnd;
    size_t const ssaCapacity   = nbTokens;
    void* const ssaEnd         = symbolSizes + ssaCapacity;
    void* const alphabetPtr    = ssaEnd;
    void* const dstEnd         = (char*)dst + dstCapacity;
    assert(dstEnd >= alphabetPtr);
    size_t const alphabetCapacity =
            (size_t)((char*)dstEnd - (char*)alphabetPtr);

    size_t const wkspSize =
            ZS_tokenizeVarto4_encode_wkspSize(cardinalityEstimation);
    void* const workspace = malloc(wkspSize);
    assert(workspace != NULL);

    ZS_TokVar_result const r = ZS_tokenizeVarto4_encode(
            dstIndex,
            indexCapacity,
            alphabetPtr,
            alphabetCapacity,
            symbolSizes,
            ssaCapacity,
            srcContent,
            srcBufferSize,
            tokenSizes,
            nbTokens,
            cardinalityEstimation,
            workspace,
            wkspSize);

    free(workspace);
    return r.dstSize;
}

// modify input so that it feature the format expected by the benched function
// size_t    alphabetSize
// size_t    nbTokens
// size_t*   tokenSizes
// void*     srcContent
// ensure alphabetSize <= 256
static size_t
tokenizeVar_preparation(void* src, size_t srcSize, size_t alphabetSize)
{
    size_t const srcSizeMin = 3 * sizeof(size_t) + 1;
    if (srcSize <= srcSizeMin) {
        printf("srcSize (%zu) is too small (< %zu) \n", srcSize, srcSizeMin);
        exit(1);
    }

    // temporary storage for original @src content
    void* const srcCopy = malloc(srcSize);
    assert(srcCopy != NULL);
    assert(src != NULL);
    memcpy(srcCopy, src, srcSize);

    size_t const nbCandidateTokens = alphabetSize;
    typedef struct {
        size_t pos;
        size_t len;
    } TokenDesc;
    TokenDesc* const tokenDesc = malloc(nbCandidateTokens * sizeof(*tokenDesc));
    assert(tokenDesc != NULL);

    // generate list of candidate tokens
#define MAX_TOKVAR_LEN 16
    for (size_t n = 0; n < nbCandidateTokens; n++) {
        int const len    = (rand() % MAX_TOKVAR_LEN) + 1;
        size_t const pos = (size_t)rand() % (srcSize - (size_t)len);
        tokenDesc[n]     = (TokenDesc){ .len = (size_t)len, .pos = pos };
        // printf("token%5zu: '%*.*s' \n", n, len, len, (char*)srcCopy + pos);
    }

    // Prepare space for header
    size_t* const alphabetSizePtr = src;
    size_t* const nbTokensPtr     = alphabetSizePtr + 1;
    size_t* const sizeArray       = nbTokensPtr + 1;
    size_t sizeArrayEnd           = 3 * sizeof(size_t);
    size_t contentBegin           = srcSize;

    size_t actualNbTokens = 0;
    size_t* sizeArrayPtr  = sizeArray;

    // write tokens into @src backward, to ensure enough space at beginning
    while (sizeArrayEnd + MAX_TOKVAR_LEN < contentBegin) {
        TokenDesc const newToken =
                tokenDesc[(size_t)rand() % nbCandidateTokens];
        size_t const len = newToken.len;
        memcpy((char*)src + contentBegin - len,
               (char*)srcCopy + newToken.pos,
               len);
        *sizeArrayPtr++ = len;
        sizeArrayEnd += sizeof(size_t);
        contentBegin -= len;
        actualNbTokens++;
        // printf("write token '%*.*s' (len %zu)\n", (int)len, (int)len,
        // (char*)src + newToken.pos, len);
    }
    // last token : take everything remaining
    assert(contentBegin > sizeArrayEnd);
    size_t const lastLen = contentBegin - sizeArrayEnd;
    memcpy((char*)src + sizeArrayEnd, srcCopy, lastLen);
    *sizeArrayPtr = lastLen;
    actualNbTokens++;
    // printf("writing '%*.*s' (len %zu)\n", (int)lastLen, (int)lastLen, dst,
    // lastLen);
    *alphabetSizePtr = alphabetSize;
    *nbTokensPtr     = actualNbTokens;

    // revert token lengths (since they were written backward)
    size_t const nbSwaps = actualNbTokens / 2;
    for (size_t n = 0; n < nbSwaps; n++) {
        size_t const tmp                  = sizeArray[n];
        sizeArray[n]                      = sizeArray[actualNbTokens - 1 - n];
        sizeArray[actualNbTokens - 1 - n] = tmp;
    }

    printf("Generated %zu tokens (max %zu different) \n",
           actualNbTokens,
           nbCandidateTokens);
    free(tokenDesc);
    free(srcCopy);

    return srcSize;
}

size_t tokenizeVarto4_preparation(void* s, size_t ss, const BenchPayload* bp)
{
    int const alphabetSizeMin = 0;
    int const alphabetSizeMax = 1 << 28;
    int const i               = bp->intParam;
    if ((i < alphabetSizeMin) || (i > alphabetSizeMax)) {
        printf("Parameter AlphabetSize (%i) is out of bound [%i, %i] \n",
               i,
               alphabetSizeMin,
               alphabetSizeMax);
        exit(1);
    }
    size_t const alphabetSize = i ? (size_t)i : ss / MAX_TOKVAR_LEN / 3;
    printf("Preparing tokenizeVarto4Decode with an alphabet size of %zu \n",
           alphabetSize);
    return tokenizeVar_preparation(s, ss, alphabetSize);
}

size_t tokenizeVarto4Decode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    // In this scenario, input is presumed prepared.

    // size_t    alphabetSize
    // size_t    nbTokens
    // size_t    alphabetContentSize
    // size_t    symbolStarts[alphabetSize]
    // size_t    symbolSizes[alphabetSize]
    // uint32_t  tokenIndexes[nbTokens]
    // uint8_t   alphabetContent[alphabetContentSize]
    (void)customPayload;

    const size_t* srcT               = src;
    size_t const alphabetSize        = srcT[0];
    size_t const nbTokens            = srcT[1];
    size_t const alphabetContentSize = srcT[2];
    const size_t* const symbolStarts = (const void*)(srcT + 3);
    const size_t* const symbolSizes =
            (const void*)(symbolStarts + alphabetSize);
    const uint32_t* const tokenIndexes =
            (const void*)(symbolSizes + alphabetSize);
    const uint8_t* const alphabetContent =
            (const void*)(tokenIndexes + nbTokens);

    const void* const srcEnd        = (const char*)src + srcSize;
    const void* const srcContentEnd = alphabetContent + alphabetContentSize;
    assert(srcContentEnd <= srcEnd);
    (void)srcEnd;
    (void)srcContentEnd;
    size_t const alphabetBufferSize =
            (size_t)((const char*)srcEnd - (const char*)src);

    size_t* const tokenSizes    = dst;
    size_t const tokenArraySize = nbTokens * sizeof(tokenSizes);
    assert(dstCapacity > tokenArraySize);
    void* const wksp      = (char*)dst + tokenArraySize;
    size_t const wkspSize = ZS_tokenizeVarto4_decode_wkspSize(alphabetSize);
    assert(dstCapacity > wkspSize + tokenArraySize);
    void* const dstContent = (char*)wksp + wkspSize;
    void* const dstEnd     = (char*)dst + dstCapacity;
    assert(dstEnd > dstContent);
    size_t const dstContentCapacity =
            (size_t)((char*)dstEnd - (char*)dstContent);

#if 0
    size_t const dstSize = ZS_tokenizeVarto4_decode_kernel(
                                    // write
                                    dstContent, dstContentCapacity,
                                    tokenSizes,
                                    // read
                                    tokenIndexes, nbTokens,
                                    alphabetContent, alphabetBufferSize,
                                    symbolStarts, symbolSizes,
                                    alphabetSize, MAX_TOKVAR_LEN
                                    );
#else
    size_t const dstSize = ZS_tokenizeVarto4_decode(
            // write
            dstContent,
            dstContentCapacity,
            tokenSizes,
            nbTokens,
            // read
            tokenIndexes,
            nbTokens,
            alphabetContent,
            alphabetBufferSize,
            symbolSizes,
            alphabetSize,
            wksp,
            wkspSize);
#endif
    return dstSize;
}

// modify input so that it feature the format expected by the benched function
// size_t    alphabetSize
// size_t    nbTokens
// size_t    alphabetContentSize
// size_t    symbolStarts[alphabetSize]
// size_t    symbolSizes[alphabetSize]
// uint32_t  tokenIndexes[nbTokens]
// uint8_t   alphabetContent[alphabetContentSize]
static size_t
tokVarDecode_preparation(void* src, size_t srcSize, size_t alphabetSize)
{
    assert(srcSize
           > 8 * sizeof(size_t) + 40); // minimum for format to be meaningful

    assert(src != NULL);

    size_t* const srcT           = src;
    srcT[0]                      = alphabetSize;
    size_t* const nbTokensPtr    = srcT + 1;
    size_t* const alConSizePtr   = srcT + 2;
    size_t* const symbolStarts   = srcT + 3;
    size_t* const symbolSizes    = symbolStarts + alphabetSize;
    void* const tiVoid           = symbolSizes + alphabetSize;
    uint32_t* const tokenIndexes = (uint32_t*)tiVoid;

    if ((char*)tokenIndexes >= (char*)src + srcSize + 40) {
        printf("srcSize (%zu) too small for this alphabet\n", srcSize);
        fflush(NULL);
        abort();
    }
    symbolStarts[0]  = 0;
    symbolSizes[0]   = (size_t)((rand() % MAX_TOKVAR_LEN) + 1);
    size_t alConSize = symbolSizes[0];
    for (size_t n = 1; n < alphabetSize; n++) {
        symbolSizes[n]  = (size_t)((rand() % MAX_TOKVAR_LEN) + 1);
        symbolStarts[n] = symbolStarts[n - 1] + symbolSizes[n - 1];
        alConSize += symbolSizes[n];
        // printf("symbol %3zu has a size of %3zu \n", n, symbolSizes[n]);
    }
    alConSizePtr[0] = alConSize;

    if ((char*)tokenIndexes + alConSize + 32 >= (char*)src + srcSize) {
        printf("srcSize (%zu) too small for this alphabet\n", srcSize);
        fflush(NULL);
        abort();
    }
    size_t const nbTokens = (size_t)(((char*)src + srcSize)
                                     - ((char*)tokenIndexes + alConSize + 32))
            / sizeof(uint32_t);
    assert(nbTokens > 1);
    nbTokensPtr[0] = nbTokens;

    for (size_t n = 0; n < nbTokens; n++) {
        tokenIndexes[n] = (uint32_t)((size_t)rand() % alphabetSize);
    }
    return srcSize;
}

size_t tokVarDecode_prep(void* s, size_t ss, const BenchPayload* bp)
{
    int const alphabetSizeMin = 0;
    int const alphabetSizeMax = 1 << 28;
    int const i               = bp->intParam;
    if ((i < alphabetSizeMin) || (i > alphabetSizeMax)) {
        printf("Parameter AlphabetSize (%i) is out of bound [%i, %i] \n",
               i,
               alphabetSizeMin,
               alphabetSizeMax);
        exit(1);
    }
    size_t const alphabetSize = i ? (size_t)i : ss / MAX_TOKVAR_LEN / 10;
    printf("Preparing tokenizeVarto4 with an alphabet size of %zu \n",
           alphabetSize);
    return tokVarDecode_preparation(s, ss, alphabetSize);
}

size_t tokVarDecode_outSize(const void* src, size_t srcSize)
{
    // read input, evaluate output size
    // size_t    alphabetSize
    // size_t    nbTokens
    // size_t    alphabetContentSize
    // size_t    symbolStarts[alphabetSize]
    // size_t    symbolSizes[alphabetSize]
    // uint32_t  tokenIndexes[nbTokens]
    // uint8_t   alphabetContent[alphabetContentSize]
    const size_t* const srcT           = src;
    size_t const alphabetSize          = srcT[0];
    size_t const nbTokens              = srcT[1];
    const size_t* const symbolSizes    = (srcT + 3) + alphabetSize;
    const void* const tiVoid           = symbolSizes + alphabetSize;
    const uint32_t* const tokenIndexes = (const uint32_t*)tiVoid;
    assert((const char*)(tokenIndexes + nbTokens) < (const char*)src + srcSize);
    (void)srcSize;

    size_t decodedContentSize = 0;

    for (size_t n = 0; n < nbTokens; n++) {
        decodedContentSize += symbolSizes[tokenIndexes[n]];
    }

    size_t const tokSizesArraySize = nbTokens * sizeof(size_t);
    size_t const wkspSize = ZS_tokenizeVarto4_decode_wkspSize(alphabetSize);
    printf("will decode %zu tokens, from an alphabet of %zu symbols \n",
           nbTokens,
           alphabetSize);
    // printf("tokSizesArraySize=%zu, decodedContentSize=%zu \n",
    // tokSizesArraySize, decodedContentSize);
    return decodedContentSize + tokSizesArraySize + wkspSize;
}
