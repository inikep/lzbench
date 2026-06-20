/* ******************************************************************
 * FSE : Finite State Entropy decoder
 * Copyright (c) Yann Collet, Facebook, Inc.
 *
 *  You can contact the author at :
 *  - FSE source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *  - Public forum : https://groups.google.com/forum/#!forum/lz4c
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
****************************************************************** */


/* **************************************************************
*  Includes
****************************************************************/
#include "../common/debug.h"      /* assert */
#include "../bitstream.h"
#include "../common/compiler.h"
#define FSE_STATIC_LINKING_ONLY
#include "../fse.h"
#include "../common/error_private.h"
#define ZSTD_DEPS_NEED_MALLOC
#include "../common/zstd_deps.h"
#include "../common/bits.h"       /* ZSTD_highbit32 */


/* **************************************************************
*  Error Management
****************************************************************/
#undef FSE_isError
#define FSE_isError ERR_isError
#define FSE_STATIC_ASSERT(c) DEBUG_STATIC_ASSERT(c)   /* use only *after* variable declarations */


/* **************************************************************
*  Templates
****************************************************************/
/*
  designed to be included
  for type-specific functions (template emulation in C)
  Objective is to write these functions only once, for improved maintenance
*/

/* safety checks */
#ifndef FSE_FUNCTION_EXTENSION
#  error "FSE_FUNCTION_EXTENSION must be defined"
#endif
#ifndef FSE_FUNCTION_TYPE
#  error "FSE_FUNCTION_TYPE must be defined"
#endif

/* Function names */
#define FSE_CAT(X,Y) X##Y
#define FSE_FUNCTION_NAME(X,Y) FSE_CAT(X,Y)
#define FSE_TYPE_NAME(X,Y) FSE_CAT(X,Y)


/* Function templates */
FSE_DTable* FSE_createDTable (unsigned tableLog)
{
    if (tableLog > FSE_TABLELOG_ABSOLUTE_MAX) tableLog = FSE_TABLELOG_ABSOLUTE_MAX;
    return (FSE_DTable*)ZSTD_malloc( FSE_DTABLE_SIZE_U32(tableLog) * sizeof (U32) );
}

void FSE_freeDTable (FSE_DTable* dt)
{
    ZSTD_free(dt);
}

static size_t FSE_buildDTable_internal(FSE_DTable* dt, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog, void* workSpace, size_t wkspSize)
{
    void* const tdPtr = dt+1;   /* because *dt is unsigned, 32-bits aligned on 32-bits */
    FSE_DECODE_TYPE* const tableDecode = (FSE_DECODE_TYPE*) (tdPtr);
    U16* symbolNext = (U16*)workSpace;
    BYTE* spread = (BYTE*)(symbolNext + maxSymbolValue + 1);

    U32 const maxSV1 = maxSymbolValue + 1;
    U32 const tableSize = 1u << tableLog;
    U32 highThreshold = tableSize-1;

    /* Sanity Checks */
    if (FSE_BUILD_DTABLE_WKSP_SIZE(tableLog, maxSymbolValue) > wkspSize) return ERROR(maxSymbolValue_tooLarge);
    if (maxSymbolValue > FSE_MAX_SYMBOL_VALUE) return ERROR(maxSymbolValue_tooLarge);
    if (tableLog > FSE_MAX_TABLELOG) return ERROR(tableLog_tooLarge);

    /* Init, lay down lowprob symbols */
    {   FSE_DTableHeader DTableH;
        DTableH.tableLog = (U16)tableLog;
        DTableH.fastMode = 1;
        {   S16 const largeLimit= (S16)(1 << (tableLog-1));
            U32 s;
            for (s=0; s<maxSV1; s++) {
                if (normalizedCounter[s]==-1) {
                    tableDecode[highThreshold--].symbol = (FSE_FUNCTION_TYPE)s;
                    symbolNext[s] = 1;
                } else {
                    if (normalizedCounter[s] >= largeLimit) DTableH.fastMode=0;
                    symbolNext[s] = (U16)normalizedCounter[s];
        }   }   }
        ZSTD_memcpy(dt, &DTableH, sizeof(DTableH));
    }

    /* Spread symbols */
    if (highThreshold == tableSize - 1) {
        size_t const tableMask = tableSize-1;
        size_t const step = FSE_TABLESTEP(tableSize);
        /* First lay down the symbols in order.
         * We use a uint64_t to lay down 8 bytes at a time. This reduces branch
         * misses since small blocks generally have small table logs, so nearly
         * all symbols have counts <= 8. We ensure we have 8 bytes at the end of
         * our buffer to handle the over-write.
         */
        {
            U64 const add = 0x0101010101010101ull;
            size_t pos = 0;
            U64 sv = 0;
            U32 s;
            for (s=0; s<maxSV1; ++s, sv += add) {
                int i;
                int const n = normalizedCounter[s];
                MEM_write64(spread + pos, sv);
                for (i = 8; i < n; i += 8) {
                    MEM_write64(spread + pos + i, sv);
                }
                pos += (size_t)n;
            }
        }
        /* Now we spread those positions across the table.
         * The benefit of doing it in two stages is that we avoid the
         * variable size inner loop, which caused lots of branch misses.
         * Now we can run through all the positions without any branch misses.
         * We unroll the loop twice, since that is what empirically worked best.
         */
        {
            size_t position = 0;
            size_t s;
            size_t const unroll = 2;
            assert(tableSize % unroll == 0); /* FSE_MIN_TABLELOG is 5 */
            for (s = 0; s < (size_t)tableSize; s += unroll) {
                size_t u;
                for (u = 0; u < unroll; ++u) {
                    size_t const uPosition = (position + (u * step)) & tableMask;
                    tableDecode[uPosition].symbol = spread[s + u];
                }
                position = (position + (unroll * step)) & tableMask;
            }
            assert(position == 0);
        }
    } else {
        U32 const tableMask = tableSize-1;
        U32 const step = FSE_TABLESTEP(tableSize);
        U32 s, position = 0;
        for (s=0; s<maxSV1; s++) {
            int i;
            for (i=0; i<normalizedCounter[s]; i++) {
                tableDecode[position].symbol = (FSE_FUNCTION_TYPE)s;
                position = (position + step) & tableMask;
                while (position > highThreshold) position = (position + step) & tableMask;   /* lowprob area */
        }   }
        if (position!=0) return ERROR(GENERIC);   /* position must reach all cells once, otherwise normalizedCounter is incorrect */
    }

    /* Build Decoding table */
    {   U32 u;
        for (u=0; u<tableSize; u++) {
            FSE_FUNCTION_TYPE const symbol = (FSE_FUNCTION_TYPE)(tableDecode[u].symbol);
            U32 const nextState = symbolNext[symbol]++;
            tableDecode[u].nbBits = (BYTE) (tableLog - ZSTD_highbit32(nextState) );
            tableDecode[u].newState = (U16) ( (nextState << tableDecode[u].nbBits) - tableSize);
    }   }

    return 0;
}

size_t FSE_buildDTable_wksp(FSE_DTable* dt, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog, void* workSpace, size_t wkspSize)
{
    return FSE_buildDTable_internal(dt, normalizedCounter, maxSymbolValue, tableLog, workSpace, wkspSize);
}


#ifndef FSE_COMMONDEFS_ONLY

/*-*******************************************************
*  Decompression (Byte symbols)
*********************************************************/
size_t FSE_buildDTable_constant (FSE_DTable* dt, BYTE symbolValue)
{
    void* ptr = dt;
    FSE_DTableHeader* const DTableH = (FSE_DTableHeader*)ptr;
    void* dPtr = dt + 1;
    FSE_decode_t* const cell = (FSE_decode_t*)dPtr;

    DTableH->tableLog = 0;
    DTableH->fastMode = 0;

    cell->newState = 0;
    cell->symbol = symbolValue;
    cell->nbBits = 0;

    return 0;
}


size_t FSE_buildDTable_raw (FSE_DTable* dt, unsigned nbBits)
{
    void* ptr = dt;
    FSE_DTableHeader* const DTableH = (FSE_DTableHeader*)ptr;
    void* dPtr = dt + 1;
    FSE_decode_t* const dinfo = (FSE_decode_t*)dPtr;
    const unsigned tableSize = 1u << nbBits;
    const unsigned tableMask = tableSize - 1;
    const unsigned maxSV1 = tableMask+1;
    unsigned s;

    /* Sanity checks */
    if (nbBits < 1) return ERROR(GENERIC);         /* min size */

    /* Build Decoding Table */
    DTableH->tableLog = (U16)nbBits;
    DTableH->fastMode = 1;
    for (s=0; s<maxSV1; s++) {
        dinfo[s].newState = 0;
        dinfo[s].symbol = (BYTE)s;
        dinfo[s].nbBits = (BYTE)nbBits;
    }

    return 0;
}

FORCE_INLINE_TEMPLATE BYTE FSE_getSymbol(FSE_DState_t* state, BIT_DStream_t* bitD, const unsigned fast) {
    if(fast) {
        return FSE_decodeSymbolFast(state, bitD);
    }
    return FSE_decodeSymbol(state, bitD);
}

// Reloads bitD bitstream and decodes 4 symbols from bitStream.
// If bitD stream doesn't contain enough bits to safely decode 4 symbols returns 1.
// Returns 0 on success.
FORCE_INLINE_TEMPLATE int FSE_decompress_usingDTable_internal_x4(BYTE *op, FSE_DState_t* state1, FSE_DState_t* state2,
                FSE_DState_t* state3, FSE_DState_t* state4, BIT_DStream_t* bitD, const unsigned fast) {
    if(BIT_reloadDStreamFast(bitD) != BIT_DStream_unfinished) {
        return 1;
    }
    op[0] = FSE_getSymbol(state1, bitD, fast);
    op[1] = FSE_getSymbol(state2, bitD, fast);
    op[2] = FSE_getSymbol(state3, bitD, fast);
    op[3] = FSE_getSymbol(state4, bitD, fast);
    return 0;
}

FORCE_INLINE_TEMPLATE size_t FSE_decompress_usingDTable_generic(
          void* dst, size_t maxDstSize,
    const void* cSrc, size_t cSrcSize,
    const FSE_DTable* dt, const unsigned fast,
    const unsigned nbStates)
{
    // Would prefer a static_assert but having a problem with compile flags
    assert(nbStates == 2 || nbStates == 4);

    BYTE* const ostart = (BYTE*) dst;
    BYTE* op = ostart;
    BYTE* const omax = op + maxDstSize;
    BYTE* const olimit = op + ((maxDstSize>>2)<<2);

    BIT_DStream_t bitD;
    FSE_DState_t state1, state2, state3, state4;
    FSE_DState_t* states[4] = {&state1, &state2, &state1, &state2};

    /* Init */
    CHECK_F(BIT_initDStream(&bitD, cSrc, cSrcSize));

    FSE_initDState(&state1, &bitD, dt);
    FSE_initDState(&state2, &bitD, dt);

    if(nbStates == 4) {
        FSE_initDState(&state3, &bitD, dt);
        FSE_initDState(&state4, &bitD, dt);
        states[2] = &state3;
        states[3] = &state4;
    }


    /* 4 symbols per loop */
    for ( ; op<olimit ; op+=4) {
        if(FSE_decompress_usingDTable_internal_x4(op, states[0], states[1], states[2], states[3], &bitD, fast)) {
            break;
        }
    }

    /* Process the tail -
     * We follow these steps:
     * 1. Decode as many symbols as we can while still reading from the bitstream
     * 2. We continue decoding symbols as long as their decoding doesn't require
     *    more bits from the bitstream (using `FSE_decodeTailSymbol`).
     *    We stop when we hit a state that requires more bits to advance to the next
     *    state.
     * 3. We decode symbols from the remaining nbStates-1 states.
     *    Explanation -
     *    The last decoded state for each of our nbStates requires at least one bit
     *    beyond the end of the bitstream. We've already decoded one state that requires
     *    bits beyond the end of the bitstream (in the previous step).
     *    So now we know we should have reached the last symbol for each of our states,
     *    and we decode the rest of them.
     */
    int i = 0;
    while(!BIT_endOfDStreamPassed(&bitD) && op < omax) {
        BIT_reloadDStream(&bitD);
        *op++ = FSE_getSymbol(states[(i++)%4], &bitD, fast);
    }
    if(!BIT_endOfDStream(&bitD)) {
        if(BIT_endOfDStreamPassed(&bitD)) {
            // We are past the end of the stream which shouldn't happen, we should've reached the exact end
            // This should happen only if there's a problem such as a corruption
            return ERROR(corruption_detected);
        }
        // We haven't reached the end which means that destination buffer is too small
        return ERROR(dstSize_tooSmall);
    }
    // Then continue reading symbols until states need bits to progress
    while(op < omax) {
        if(!FSE_decodeTailSymbol(states[(i++)%4], op++))
            break;
    }
    if(omax-op < nbStates-1) {
        return ERROR(dstSize_tooSmall);
    }
    // Flush last symbol in each remaining state
    for(size_t j=0; j < nbStates-1; j++) {
        *op++ = FSE_getSymbol(states[(i++)%4], &bitD, fast);
    }


    return (size_t)(op-ostart);
}

static size_t FSE_decompress_usingDTable_slow(
    void* dst, size_t maxDstSize,
    const void* cSrc, size_t cSrcSize,
    const FSE_DTable* dt, unsigned nbStates)
{
    if(nbStates == 2) {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 0, 2);
    } else if(nbStates == 4) {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 0, 4);
    } else {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 0, nbStates);
    }
}

static size_t FSE_decompress_usingDTable_fast(
          void* dst, size_t maxDstSize,
    const void* cSrc, size_t cSrcSize,
    const FSE_DTable* dt, unsigned nbStates)
{
    if(nbStates == 2) {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 1, 2);
    } else if(nbStates == 4) {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 1, 4);
    } else {
        return FSE_decompress_usingDTable_generic(dst, maxDstSize, cSrc, cSrcSize, dt, 1, nbStates);
    }
}


size_t FSE_decompress_usingDTable(void* dst, size_t originalSize,
                            const void* cSrc, size_t cSrcSize,
                            const FSE_DTable* dt, unsigned nbStates)
{
    const void* ptr = dt;
    const FSE_DTableHeader* DTableH = (const FSE_DTableHeader*)ptr;
    const U32 fastMode = DTableH->fastMode;

    /* select fast mode (static) */
    if (fastMode) return FSE_decompress_usingDTable_fast(dst, originalSize, cSrc, cSrcSize, dt, nbStates);
    return FSE_decompress_usingDTable_slow(dst, originalSize, cSrc, cSrcSize, dt, nbStates);
}


size_t FSE_decompress_wksp(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize, unsigned maxLog, unsigned nbStates, void* workSpace, size_t wkspSize)
{
    return FSE_decompress_wksp_bmi2(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, workSpace, wkspSize, /* bmi2 */ 0);
}

typedef struct {
    short ncount[FSE_MAX_SYMBOL_VALUE + 1];
    FSE_DTable dtable[1]; /* Dynamically sized */
} FSE_DecompressWksp;


FORCE_INLINE_TEMPLATE size_t FSE_decompress_wksp_body(
        void* dst, size_t dstCapacity,
        const void* cSrc, size_t cSrcSize,
        unsigned maxLog, unsigned nbStates,
        void* workSpace, size_t wkspSize,
        int bmi2)
{
    const BYTE* const istart = (const BYTE*)cSrc;
    const BYTE* ip = istart;
    unsigned tableLog;
    unsigned maxSymbolValue = FSE_MAX_SYMBOL_VALUE;
    FSE_DecompressWksp* const wksp = (FSE_DecompressWksp*)workSpace;

    DEBUG_STATIC_ASSERT((FSE_MAX_SYMBOL_VALUE + 1) % 2 == 0);
    if (wkspSize < sizeof(*wksp)) return ERROR(GENERIC);

    /* normal FSE decoding mode */
    {
        size_t const NCountLength = FSE_readNCount_bmi2(wksp->ncount, &maxSymbolValue, &tableLog, istart, cSrcSize, bmi2);
        if (FSE_isError(NCountLength)) return NCountLength;
        if (tableLog > maxLog) return ERROR(tableLog_tooLarge);
        assert(NCountLength <= cSrcSize);
        ip += NCountLength;
        cSrcSize -= NCountLength;
    }

    if (FSE_DECOMPRESS_WKSP_SIZE(tableLog, maxSymbolValue) > wkspSize) return ERROR(tableLog_tooLarge);
    assert(sizeof(*wksp) + FSE_DTABLE_SIZE(tableLog) <= wkspSize);
    workSpace = (BYTE*)workSpace + sizeof(*wksp) + FSE_DTABLE_SIZE(tableLog);
    wkspSize -= sizeof(*wksp) + FSE_DTABLE_SIZE(tableLog);

    CHECK_F( FSE_buildDTable_internal(wksp->dtable, wksp->ncount, maxSymbolValue, tableLog, workSpace, wkspSize) );

    {
        const void* ptr = wksp->dtable;
        const FSE_DTableHeader* DTableH = (const FSE_DTableHeader*)ptr;
        const U32 fastMode = DTableH->fastMode;

        /* select fast mode (static) */
        if (fastMode) return FSE_decompress_usingDTable_fast(dst, dstCapacity, ip, cSrcSize, wksp->dtable, nbStates);
        return FSE_decompress_usingDTable_slow(dst, dstCapacity, ip, cSrcSize, wksp->dtable, nbStates);
    }
}

/* Avoids the FORCE_INLINE of the _body() function. */
static size_t FSE_decompress_wksp_body_default(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize, unsigned maxLog,
    unsigned nbStates, void* workSpace, size_t wkspSize)
{
    return FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, workSpace, wkspSize, 0);
}

#if DYNAMIC_BMI2
BMI2_TARGET_ATTRIBUTE static size_t FSE_decompress_wksp_body_bmi2(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize, unsigned maxLog,
    unsigned nbStates, void* workSpace, size_t wkspSize)
{
    return FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, workSpace, wkspSize, 1);
}
#endif

size_t FSE_decompress_wksp_bmi2(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize, unsigned maxLog,
    unsigned nbStates, void* workSpace, size_t wkspSize, int bmi2)
{
#if DYNAMIC_BMI2
    if (bmi2) {
        return FSE_decompress_wksp_body_bmi2(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, workSpace, wkspSize);
    }
#endif
    (void)bmi2;
    return FSE_decompress_wksp_body_default(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, workSpace, wkspSize);
}


typedef FSE_DTable DTable_max_t[FSE_DTABLE_SIZE_U32(FSE_MAX_TABLELOG)];

#ifndef ZSTD_NO_UNUSED_FUNCTIONS
size_t FSE_buildDTable(FSE_DTable* dt, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog) {
    U32 wksp[FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE)];
    return FSE_buildDTable_wksp(dt, normalizedCounter, maxSymbolValue, tableLog, wksp, sizeof(wksp));
}

size_t FSE_decompress(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize)
{
    return FSE_decompress2(dst, dstCapacity, cSrc, cSrcSize, FSE_MAX_TABLELOG, FSE_DEFAULT_NBSTATES);
}

size_t FSE_decompress2(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize, unsigned maxLog, unsigned nbStates)
{
    /* Static analyzer seems unable to understand this table will be properly initialized later */
    U32 wksp[FSE_DECOMPRESS_WKSP_SIZE_U32(FSE_MAX_TABLELOG, FSE_MAX_SYMBOL_VALUE)];
    maxLog = maxLog == 0 ? FSE_MAX_TABLELOG : maxLog;
    nbStates = nbStates == 0 ? FSE_DEFAULT_NBSTATES : nbStates;
    return FSE_decompress_wksp(dst, dstCapacity, cSrc, cSrcSize, maxLog, nbStates, wksp, sizeof(wksp));
}


#endif


#endif   /* FSE_COMMONDEFS_ONLY */
