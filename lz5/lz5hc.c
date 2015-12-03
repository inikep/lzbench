/*
    LZ5 HC - High Compression Mode of LZ5
    Copyright (C) 2011-2015, Yann Collet.
    Copyright (C) 2015, Przemyslaw Skibinski <inikep@gmail.com>

    BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the
    distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    You can contact the author at :
       - LZ5 source repository : https://github.com/inikep/lz5
       - LZ5 public forum : https://groups.google.com/forum/#!forum/lz5c
*/




/* *************************************
*  Includes
***************************************/
#include "lz5common.h"
#include "lz5.h"
#include "lz5hc.h"
#include <stdio.h>


/**************************************
*  HC Compression
**************************************/


int LZ5_alloc_mem_HC(LZ5HC_Data_Structure* ctx, int compressionLevel)
{
    ctx->compressionLevel = compressionLevel;  
    if (compressionLevel > g_maxCompressionLevel) ctx->compressionLevel = g_maxCompressionLevel;
    if (compressionLevel < 1) ctx->compressionLevel = LZ5HC_compressionLevel_default;

    ctx->params = LZ5HC_defaultParameters[ctx->compressionLevel];

    ctx->hashTable = (U32*) ALLOCATOR(1, sizeof(U32)*((1 << ctx->params.hashLog3)+(1 << ctx->params.hashLog)));
    if (!ctx->hashTable)
        return 0;

    ctx->hashTable3 = ctx->hashTable + (1 << ctx->params.hashLog);

    ctx->chainTable = (U32*) ALLOCATOR(1, sizeof(U32)*(1 << ctx->params.contentLog));
    if (!ctx->chainTable)
    {
        FREEMEM(ctx->hashTable);
        ctx->hashTable = NULL;
        return 0;
    }

    return 1;
}

void LZ5_free_mem_HC(LZ5HC_Data_Structure* ctx)
{
    if (ctx->chainTable) FREEMEM(ctx->chainTable);
    if (ctx->hashTable) FREEMEM(ctx->hashTable);    
}


static void LZ5HC_init (LZ5HC_Data_Structure* ctx, const BYTE* start)
{
    MEM_INIT((void*)ctx->hashTable, 0, sizeof(U32)*((1 << ctx->params.hashLog) + (1 << ctx->params.hashLog3)));
    MEM_INIT(ctx->chainTable, 0xFF, sizeof(U32)*(1 << ctx->params.contentLog));

    ctx->nextToUpdate = (1 << ctx->params.windowLog);
    ctx->base = start - (1 << ctx->params.windowLog);
    ctx->end = start;
    ctx->dictBase = start - (1 << ctx->params.windowLog);
    ctx->dictLimit = (1 << ctx->params.windowLog);
    ctx->lowLimit = (1 << ctx->params.windowLog);
    ctx->last_off = 1;
}


/* Update chains up to ip (excluded) */
FORCE_INLINE void LZ5HC_Insert (LZ5HC_Data_Structure* ctx, const BYTE* ip)
{
    U32* chainTable = ctx->chainTable;
    U32* HashTable  = ctx->hashTable;
#if MINMATCH == 3
    U32* HashTable3  = ctx->hashTable3;
#endif 
    const BYTE* const base = ctx->base;
    const U32 target = (U32)(ip - base);
    const U32 contentMask = (1 << ctx->params.contentLog) - 1;
    U32 idx = ctx->nextToUpdate;

    while(idx < target)
    {
        U32 h = LZ5HC_hashPtr(base+idx, ctx->params.hashLog, ctx->params.searchLength);
        chainTable[idx & contentMask] = (U32)(idx - HashTable[h]);
        HashTable[h] = idx;
#if MINMATCH == 3
        HashTable3[LZ5HC_hash3Ptr(base+idx, ctx->params.hashLog3)] = idx;
#endif 
       idx++;
    }

    ctx->nextToUpdate = target;
}

    
FORCE_INLINE int LZ5HC_FindBestMatch (LZ5HC_Data_Structure* ctx,   /* Index table will be updated */
                                               const BYTE* ip, const BYTE* const iLimit,
                                               const BYTE** matchpos)
{
    U32* const chainTable = ctx->chainTable;
    U32* const HashTable = ctx->hashTable;
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const U32 dictLimit = ctx->dictLimit;
    const U32 maxDistance = (1 << ctx->params.windowLog);     
    const U32 lowLimit = (ctx->lowLimit + maxDistance > (U32)(ip-base)) ? ctx->lowLimit : (U32)(ip - base) - (maxDistance - 1);
    const U32 contentMask = (1 << ctx->params.contentLog) - 1;
    U32 matchIndex;
    const BYTE* match;
    int nbAttempts=ctx->params.searchNum;
    size_t ml=0, mlt;

    matchIndex = HashTable[LZ5HC_hashPtr(ip, ctx->params.hashLog, ctx->params.searchLength)];

    match = ip - ctx->last_off;
    if (MEM_read24(match) == MEM_read24(ip))
    {
        ml = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
        *matchpos = match;
        return (int)ml;
    }

#if MINMATCH == 3
    size_t offset = ip - base - ctx->hashTable3[LZ5HC_hash3Ptr(ip, ctx->params.hashLog3)];
    if (offset > 0 && offset < LZ5_SHORT_OFFSET_DISTANCE)
    {
        match = ip - offset;
        if (match > base && MEM_read24(ip) == MEM_read24(match))
        {
            ml = 3;//MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
            *matchpos = match;
        }
    }
#endif

    while ((matchIndex>=lowLimit) && (nbAttempts))
    {
        nbAttempts--;
        if (matchIndex >= dictLimit)
        {
            match = base + matchIndex;
            if (match < ip && *(match+ml) == *(ip+ml) && (MEM_read32(match) == MEM_read32(ip)))
            {
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
                if (!ml || (mlt > ml && LZ5HC_better_price(ip - *matchpos, ml, ip - match, mlt, ctx->last_off)))
//                if (mlt > ml && (LZ5_NORMAL_MATCH_COST(mlt - MINMATCH, (ip - match == ctx->last_off) ? 0 : (ip - match)) < LZ5_NORMAL_MATCH_COST(ml - MINMATCH, (ip - *matchpos == ctx->last_off) ? 0 : (ip - *matchpos)) + (LZ5_NORMAL_LIT_COST(mlt - ml))))
                { ml = mlt; *matchpos = match; }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iLimit) vLimit = iLimit;
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iLimit))
                    mlt += MEM_count(ip+mlt, base+dictLimit, iLimit);
                if (!ml || (mlt > ml && LZ5HC_better_price(ip - *matchpos, ml, ip - match, mlt, ctx->last_off)))
             //   if (mlt > ml && (LZ5_NORMAL_MATCH_COST(mlt - MINMATCH, (ip - match == ctx->last_off) ? 0 : (ip - match)) < LZ5_NORMAL_MATCH_COST(ml - MINMATCH, (ip - *matchpos == ctx->last_off) ? 0 : (ip - *matchpos)) + (LZ5_NORMAL_LIT_COST(mlt - ml))))
                { ml = mlt; *matchpos = base + matchIndex; }   /* virtual matchpos */
            }
        }
        matchIndex -= chainTable[matchIndex & contentMask];
    }

    return (int)ml;
}


FORCE_INLINE int LZ5HC_FindBestMatchFast (LZ5HC_Data_Structure* ctx, U32 matchIndex, U32 matchIndex3, /* Index table will be updated */
                                               const BYTE* ip, const BYTE* const iLimit,
                                               const BYTE** matchpos)
{
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const U32 dictLimit = ctx->dictLimit;
    const U32 maxDistance = (1 << ctx->params.windowLog);     
    const U32 lowLimit = (ctx->lowLimit + maxDistance > (U32)(ip-base)) ? ctx->lowLimit : (U32)(ip - base) - (maxDistance - 1);
    const BYTE* match;
    size_t ml=0, mlt;

    match = ip - ctx->last_off;
    if (MEM_read24(match) == MEM_read24(ip))
    {
        ml = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
        *matchpos = match;
        return (int)ml;
    }

#if MINMATCH == 3
    size_t offset = ip - base - matchIndex3;
    if (offset > 0 && offset < LZ5_SHORT_OFFSET_DISTANCE)
    {
        match = ip - offset;
        if (match > base && MEM_read24(ip) == MEM_read24(match))
        {
            ml = 3;//MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
            *matchpos = match;
        }
    }
#endif

    if (matchIndex>=lowLimit)
    {
        if (matchIndex >= dictLimit)
        {
            match = base + matchIndex;
            if (match < ip && *(match+ml) == *(ip+ml) && (MEM_read32(match) == MEM_read32(ip)))
            {
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
                if (!ml || (mlt > ml && LZ5HC_better_price(ip - *matchpos, ml, ip - match, mlt, ctx->last_off)))
         //       if (ml==0 || ((mlt > ml) && LZ5_NORMAL_MATCH_COST(mlt - MINMATCH, (ip - match == ctx->last_off) ? 0 : (ip - match)) < LZ5_NORMAL_MATCH_COST(ml - MINMATCH, (ip - *matchpos == ctx->last_off) ? 0 : (ip - *matchpos)) + (LZ5_NORMAL_LIT_COST(mlt - ml))))
                { ml = mlt; *matchpos = match; }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iLimit) vLimit = iLimit;
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iLimit))
                    mlt += MEM_count(ip+mlt, base+dictLimit, iLimit);
                if (!ml || (mlt > ml && LZ5HC_better_price(ip - *matchpos, ml, ip - match, mlt, ctx->last_off)))
//                if (ml==0 || ((mlt > ml) && LZ5_NORMAL_MATCH_COST(mlt - MINMATCH, (ip - match == ctx->last_off) ? 0 : (ip - match)) < LZ5_NORMAL_MATCH_COST(ml - MINMATCH, (ip - *matchpos == ctx->last_off) ? 0 : (ip - *matchpos)) + (LZ5_NORMAL_LIT_COST(mlt - ml))))
                { ml = mlt; *matchpos = base + matchIndex; }   /* virtual matchpos */
            }
        }
    }
    
    return (int)ml;
}


FORCE_INLINE int LZ5HC_FindBestMatchFaster (LZ5HC_Data_Structure* ctx, U32 matchIndex,  /* Index table will be updated */
                                               const BYTE* ip, const BYTE* const iLimit,
                                               const BYTE** matchpos)
{
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const U32 dictLimit = ctx->dictLimit;
    const U32 maxDistance = (1 << ctx->params.windowLog);     
    const U32 lowLimit = (ctx->lowLimit + maxDistance > (U32)(ip-base)) ? ctx->lowLimit : (U32)(ip - base) - (maxDistance - 1);
    const BYTE* match;
    size_t ml=0, mlt;

    match = ip - ctx->last_off;
    if (MEM_read24(match) == MEM_read24(ip))
    {
        ml = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
        *matchpos = match;
        return (int)ml;
    }

    if (matchIndex>=lowLimit)
    {
        if (matchIndex >= dictLimit)
        {
            match = base + matchIndex;
            if (match < ip && *(match+ml) == *(ip+ml) && (MEM_read32(match) == MEM_read32(ip)))
            {
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
                if (mlt > ml) { ml = mlt; *matchpos = match; }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iLimit) vLimit = iLimit;
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iLimit))
                    mlt += MEM_count(ip+mlt, base+dictLimit, iLimit);
                if (mlt > ml) { ml = mlt; *matchpos = base + matchIndex; }   /* virtual matchpos */
            }
        }
    }
    
    return (int)ml;
}


FORCE_INLINE int LZ5HC_FindBestMatchFastest (LZ5HC_Data_Structure* ctx, U32 matchIndex,  /* Index table will be updated */
                                               const BYTE* ip, const BYTE* const iLimit,
                                               const BYTE** matchpos)
{
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const U32 dictLimit = ctx->dictLimit;
    const U32 maxDistance = (1 << ctx->params.windowLog);     
    const U32 lowLimit = (ctx->lowLimit + maxDistance > (U32)(ip-base)) ? ctx->lowLimit : (U32)(ip - base) - (maxDistance - 1);
    const BYTE* match;
    size_t ml=0, mlt;

    if (matchIndex>=lowLimit)
    {
        if (matchIndex >= dictLimit)
        {
            match = base + matchIndex;
            if (match < ip && *(match+ml) == *(ip+ml) && (MEM_read32(match) == MEM_read32(ip)))
            {
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
                if (mlt > ml) { ml = mlt; *matchpos = match; }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iLimit) vLimit = iLimit;
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iLimit))
                    mlt += MEM_count(ip+mlt, base+dictLimit, iLimit);
                if (mlt > ml) { ml = mlt; *matchpos = base + matchIndex; }   /* virtual matchpos */
            }
        }
    }
    
    return (int)ml;
}


FORCE_INLINE int LZ5HC_GetWiderMatch (
    LZ5HC_Data_Structure* ctx,
    const BYTE* const ip,
    const BYTE* const iLowLimit,
    const BYTE* const iHighLimit,
    int longest,
    const BYTE** matchpos,
    const BYTE** startpos)
{
    U32* const chainTable = ctx->chainTable;
    U32* const HashTable = ctx->hashTable;
    const BYTE* const base = ctx->base;
    const U32 dictLimit = ctx->dictLimit;
    const BYTE* const lowPrefixPtr = base + dictLimit;
    const U32 maxDistance = (1 << ctx->params.windowLog);
    const U32 lowLimit = (ctx->lowLimit + maxDistance > (U32)(ip-base)) ? ctx->lowLimit : (U32)(ip - base) - (maxDistance - 1);
    const U32 contentMask = (1 << ctx->params.contentLog) - 1;
    const BYTE* const dictBase = ctx->dictBase;
    const BYTE* match;
    U32   matchIndex;
    int nbAttempts = ctx->params.searchNum;


    /* First Match */
    matchIndex = HashTable[LZ5HC_hashPtr(ip, ctx->params.hashLog, ctx->params.searchLength)];

    match = ip - ctx->last_off;
    if (MEM_read24(match) == MEM_read24(ip))
    {
        int mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iHighLimit) + MINMATCH;
        
        int back = 0;
        while ((ip+back>iLowLimit) && (match+back > lowPrefixPtr) && (ip[back-1] == match[back-1])) back--;
        mlt -= back;

        if (mlt > longest)
        {
            *matchpos = match+back;
            *startpos = ip+back;
            longest = (int)mlt;
        }
    }


#if MINMATCH == 3
    size_t offset = ip - base - ctx->hashTable3[LZ5HC_hash3Ptr(ip, ctx->params.hashLog3)];
    if (offset > 0 && offset < LZ5_SHORT_OFFSET_DISTANCE)
    {
        match = ip - offset;
        if (match > base && MEM_read24(ip) == MEM_read24(match))
        {
            int mlt = MEM_count(ip+MINMATCH, match+MINMATCH, iHighLimit) + MINMATCH;

            int back = 0;
            while ((ip+back>iLowLimit) && (match+back > lowPrefixPtr) && (ip[back-1] == match[back-1])) back--;
            mlt -= back;

            if (!longest || (mlt > longest && LZ5HC_better_price(ip+back - *matchpos, longest, ip - match, mlt, ctx->last_off)))
//          if (!longest || (mlt > longest && LZ5_NORMAL_MATCH_COST(mlt - MINMATCH, (ip - match == ctx->last_off) ? 0 : (ip - match)) < LZ5_NORMAL_MATCH_COST(longest - MINMATCH, (ip+back - *matchpos == ctx->last_off) ? 0 : (ip+back - *matchpos)) + LZ5_NORMAL_LIT_COST(mlt - longest)))
            {
                *matchpos = match+back;
                *startpos = ip+back;
                longest = (int)mlt;
            }
        }
    }
#endif

    while ((matchIndex>=lowLimit) && (nbAttempts))
    {
        nbAttempts--;
        if (matchIndex >= dictLimit)
        {
            match = base + matchIndex;

       //   if (*(ip + longest) == *(matchPtr + longest))
            if (match < ip && MEM_read32(match) == MEM_read32(ip))
            {
                int mlt = MINMATCH + MEM_count(ip+MINMATCH, match+MINMATCH, iHighLimit);
                int back = 0;

                while ((ip+back>iLowLimit)
                       && (match+back > lowPrefixPtr)
                       && (ip[back-1] == match[back-1]))
                        back--;

                mlt -= back;

                if (!longest || (mlt > longest && LZ5HC_better_price(ip+back - *matchpos, longest, ip - match, mlt, ctx->last_off)))
                {
                    longest = (int)mlt;
                    *matchpos = match+back;
                    *startpos = ip+back;
                }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                size_t mlt;
                int back=0;
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iHighLimit) vLimit = iHighLimit;
                mlt = MEM_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iHighLimit))
                    mlt += MEM_count(ip+mlt, base+dictLimit, iHighLimit);
                while ((ip+back > iLowLimit) && (matchIndex+back > lowLimit) && (ip[back-1] == match[back-1])) back--;
                mlt -= back;
                if ((int)mlt > longest) { longest = (int)mlt; *matchpos = base + matchIndex + back; *startpos = ip+back; }
            }
        }
        matchIndex -= chainTable[matchIndex & contentMask];
    }


    return longest;
}



typedef enum { noLimit = 0, limitedOutput = 1 } limitedOutput_directive;

/*
LZ5 uses 3 types of codewords from 2 to 4 bytes long:
- 1_OO_LL_MMM OOOOOOOO - 10-bit offset, 3-bit match length, 2-bit literal length
- 00_LLL_MMM OOOOOOOO OOOOOOOO - 16-bit offset, 3-bit match length, 3-bit literal length
- 010_LL_MMM OOOOOOOO OOOOOOOO OOOOOOOO - 24-bit offset, 3-bit match length, 2-bit literal length 
- 011_LL_MMM - last offset, 3-bit match length, 2-bit literal length
*/

FORCE_INLINE int LZ5HC_encodeSequence (
    LZ5HC_Data_Structure* ctx,
    const BYTE** ip,
    BYTE** op,
    const BYTE** anchor,
    int matchLength,
    const BYTE* const match,
    limitedOutput_directive limitedOutputBuffer,
    BYTE* oend)
{
    int length;
    BYTE* token;

    /* Encode Literal length */
    length = (int)(*ip - *anchor);
    token = (*op)++;
    if ((limitedOutputBuffer) && ((*op + (length>>8) + length + (2 + 1 + LASTLITERALS)) > oend)) return 1;   /* Check output limit */

    if (*ip-match >= LZ5_SHORT_OFFSET_DISTANCE && *ip-match < LZ5_MID_OFFSET_DISTANCE && (U32)(*ip-match) != ctx->last_off)
    {
        if (length>=(int)RUN_MASK) { int len; *token=(RUN_MASK<<ML_BITS); len = length-RUN_MASK; for(; len > 254 ; len-=255) *(*op)++ = 255;  *(*op)++ = (BYTE)len; }
        else *token = (BYTE)(length<<ML_BITS);
    }
    else
    {
        if (length>=(int)RUN_MASK2) { int len; *token=(RUN_MASK2<<ML_BITS); len = length-RUN_MASK2; for(; len > 254 ; len-=255) *(*op)++ = 255;  *(*op)++ = (BYTE)len; }
        else *token = (BYTE)(length<<ML_BITS);
        
    }

    /* Copy Literals */
    MEM_wildCopy(*op, *anchor, (*op) + length);
    *op += length;

    /* Encode Offset */
    if ((U32)(*ip-match) == ctx->last_off)
    {
        *token+=(3<<ML_RUN_BITS2);
//            printf("2last_off=%d *token=%d\n", last_off, *token);
    }
    else
	if (*ip-match < LZ5_SHORT_OFFSET_DISTANCE)
	{
		*token+=((4+((*ip-match)>>8))<<ML_RUN_BITS2);
		**op=*ip-match; (*op)++;
	}
	else
	if (*ip-match < LZ5_MID_OFFSET_DISTANCE)
	{
		MEM_writeLE16(*op, (U16)(*ip-match)); *op+=2;
	}
	else
	{
		*token+=(2<<ML_RUN_BITS2);
		MEM_writeLE24(*op, (U32)(*ip-match)); *op+=3;
	}
    ctx->last_off = *ip-match;

    /* Encode MatchLength */
    length = (int)(matchLength-MINMATCH);
    if ((limitedOutputBuffer) && (*op + (length>>8) + (1 + LASTLITERALS) > oend)) return 1;   /* Check output limit */
    if (length>=(int)ML_MASK) { *token+=ML_MASK; length-=ML_MASK; for(; length > 509 ; length-=510) { *(*op)++ = 255; *(*op)++ = 255; } if (length > 254) { length-=255; *(*op)++ = 255; } *(*op)++ = (BYTE)length; }
    else *token += (BYTE)(length);

    LZ5HC_DEBUG("%u: ENCODE literals=%u off=%u mlen=%u out=%u\n", (U32)(*ip - ctx->inputBuffer), (U32)(*ip - *anchor), (U32)(*ip-match), (U32)matchLength, 2+(U32)(*op - ctx->outputBuffer));

    /* Prepare next loop */
    *ip += matchLength;
    *anchor = *ip;

    return 0;
}


static int LZ5HC_compress_lowest_price (
    LZ5HC_Data_Structure* ctx,
    const char* source,
    char* dest,
    int inputSize,
    int maxOutputSize,
    limitedOutput_directive limit
    )
{
    ctx->inputBuffer = (BYTE*) source;
    ctx->outputBuffer = (BYTE*) dest;
    const BYTE* ip = (const BYTE*) source;
    const BYTE* anchor = ip;
    const BYTE* const iend = ip + inputSize;
    const BYTE* const mflimit = iend - MFLIMIT;
    const BYTE* const matchlimit = (iend - LASTLITERALS);

    BYTE* op = (BYTE*) dest;
    BYTE* const oend = op + maxOutputSize;

    int   ml, ml2, ml0;
    const BYTE* ref=NULL;
    const BYTE* start2=NULL;
    const BYTE* ref2=NULL;
    const BYTE* start0;
    const BYTE* ref0;
    const BYTE* lowPrefixPtr = ctx->base + ctx->dictLimit;

    /* init */
    ctx->end += inputSize;

    ip++;

    /* Main Loop */
    while (ip < mflimit)
    {
        LZ5HC_Insert(ctx, ip);
        ml = LZ5HC_FindBestMatch (ctx, ip, matchlimit, (&ref));
        if (!ml) { ip++; continue; }

        int back = 0;
        while ((ip+back>anchor) && (ref+back > lowPrefixPtr) && (ip[back-1] == ref[back-1])) back--;
        ml -= back;
        ip += back;
        ref += back;

        /* saved, in case we would skip too much */
        start0 = ip;
        ref0 = ref;
        ml0 = ml;

_Search:
        if (ip+ml >= mflimit) goto _Encode;

        LZ5HC_Insert(ctx, ip);
        ml2 = LZ5HC_GetWiderMatch(ctx, ip + ml - 2, anchor, matchlimit, 0, &ref2, &start2);
        if (ml2 == 0) goto _Encode;

        {
        int price, best_price;
        U32 off0=0, off1=0;
        uint8_t *pos, *best_pos;

    //	find the lowest price for encoding ml bytes
        best_pos = (uint8_t*)ip;
        best_price = 1<<30;
        off0 = (uint8_t*)ip - ref;
        off1 = start2 - ref2;

        for (pos = (uint8_t*)ip + ml; pos >= start2; pos--)
        {
            int common0 = pos - ip;
            if (common0 >= MINMATCH)
            {
                price = LZ5_CODEWORD_COST(ip - anchor, (off0 == ctx->last_off) ? 0 : off0, common0 - MINMATCH);
                
                int common1 = start2 + ml2 - pos;
                if (common1 >= MINMATCH)
                    price += LZ5_CODEWORD_COST(0, (off1 == off0) ? 0 : (off1), common1 - MINMATCH);
                else
                    price += LZ5_LIT_ONLY_COST(common1);

                if (price < best_price)
                {
                    best_price = price;
                    best_pos = pos;
                }
            }
            else
            {
                price = LZ5_CODEWORD_COST(start2 - anchor, (off1 == ctx->last_off) ? 0 : off1, ml2 - MINMATCH);

                if (price < best_price)
                {
                    best_price = price;
                    best_pos = pos;
                }

                break;
            }
        }
    //    LZ5HC_DEBUG("%u: TRY last_off=%d literals=%u off=%u mlen=%u literals2=%u off2=%u mlen2=%u best=%d\n", (U32)(ip - ctx->inputBuffer), ctx->last_off, (U32)(ip - anchor), off0, (U32)ml,  (U32)(start2 - anchor), off1, ml2, (U32)(best_pos - ip));
        ml = best_pos - ip;
        }


        if (ml < MINMATCH)
        {
            ip = start2;
            ref = ref2;
            ml = ml2;
            goto _Search;
        }
        
_Encode:

        if (start0 < ip)
        {
            if (LZ5HC_more_profitable(ip - ref, ml, start0 - ref0, ml0, ref0 - ref, ctx->last_off))
            {
                ip = start0;
                ref = ref0;
                ml = ml0;
            }
        }

        if (LZ5HC_encodeSequence(ctx, &ip, &op, &anchor, ml, ref, limit, oend)) return 0;
    }

    /* Encode Last Literals */
    {
        int lastRun = (int)(iend - anchor);
        if ((limit) && (((char*)op - dest) + lastRun + 1 + ((lastRun+255-RUN_MASK)/255) > (U32)maxOutputSize)) return 0;  /* Check output limit */
        if (lastRun>=(int)RUN_MASK) { *op++=(RUN_MASK<<ML_BITS); lastRun-=RUN_MASK; for(; lastRun > 254 ; lastRun-=255) *op++ = 255; *op++ = (BYTE) lastRun; }
        else *op++ = (BYTE)(lastRun<<ML_BITS);
        memcpy(op, anchor, iend - anchor);
        op += iend-anchor;
    }

    /* End */
    return (int) (((char*)op)-dest);
}



static int LZ5HC_compress_price_fast (
    LZ5HC_Data_Structure* ctx,
    const char* source,
    char* dest,
    int inputSize,
    int maxOutputSize,
    limitedOutput_directive limit
    )
{
    ctx->inputBuffer = (BYTE*) source;
    ctx->outputBuffer = (BYTE*) dest;
    const BYTE* ip = (const BYTE*) source;
    const BYTE* anchor = ip;
    const BYTE* const iend = ip + inputSize;
    const BYTE* const mflimit = iend - MFLIMIT;
    const BYTE* const matchlimit = (iend - LASTLITERALS);

    BYTE* op = (BYTE*) dest;
    BYTE* const oend = op + maxOutputSize;

    int   ml, ml2=0;
    const BYTE* ref=NULL;
    const BYTE* start2=NULL;
    const BYTE* ref2=NULL;
    const BYTE* lowPrefixPtr = ctx->base + ctx->dictLimit;
    U32* HashTable  = ctx->hashTable;
#if MINMATCH == 3
    U32* HashTable3  = ctx->hashTable3;
#endif 
    const BYTE* const base = ctx->base;
    U32* HashPos, *HashPos3;

    /* init */
    ctx->end += inputSize;

    ip++;

    /* Main Loop */
    while (ip < mflimit)
    {
        HashPos = &HashTable[LZ5HC_hashPtr(ip, ctx->params.hashLog, ctx->params.searchLength)];
        HashPos3 = &HashTable3[LZ5HC_hash3Ptr(ip, ctx->params.hashLog3)];
        ml = LZ5HC_FindBestMatchFast (ctx, *HashPos, *HashPos3, ip, matchlimit, (&ref));
        *HashPos =  (U32)(ip - base);
#if MINMATCH == 3
        *HashPos3 = (U32)(ip - base);
#endif 
        if (!ml) { ip++; continue; }

        if ((U32)(ip - ref) == ctx->last_off) { ml2=0; goto _Encode; }
        
        {
        int back = 0;
        while ((ip+back>anchor) && (ref+back > lowPrefixPtr) && (ip[back-1] == ref[back-1])) back--;
        ml -= back;
        ip += back;
        ref += back;
        }
        
_Search:
        if (ip+ml >= mflimit) goto _Encode;

        start2 = ip + ml - 2;
        HashPos = &HashTable[LZ5HC_hashPtr(start2, ctx->params.hashLog, ctx->params.searchLength)];
        ml2 = LZ5HC_FindBestMatchFaster(ctx, *HashPos, start2, matchlimit, (&ref2));      
        *HashPos = (U32)(start2 - base);
        if (!ml2) goto _Encode;

        {
        int back = 0;
        while ((start2+back>ip) && (ref2+back > lowPrefixPtr) && (start2[back-1] == ref2[back-1])) back--;
        ml2 -= back;
        start2 += back;
        ref2 += back;
        }

    //    LZ5HC_DEBUG("%u: TRY last_off=%d literals=%u off=%u mlen=%u literals2=%u off2=%u mlen2=%u best=%d\n", (U32)(ip - ctx->inputBuffer), ctx->last_off, (U32)(ip - anchor), off0, (U32)ml,  (U32)(start2 - anchor), off1, ml2, (U32)(best_pos - ip));

        if (ml2 <= ml) { ml2 = 0; goto _Encode; }

        if (start2 <= ip)
        {
            ip = start2; ref = ref2; ml = ml2;
            ml2 = 0;
            goto _Encode;
        }

        if (start2 - ip < 3) 
        { 
            ip = start2; ref = ref2; ml = ml2;
            ml2 = 0; 
            goto _Search; 
        }


        if (start2 < ip + ml) 
        {
            int correction = ml - (int)(start2 - ip);
            start2 += correction;
            ref2 += correction;
            ml2 -= correction;
            if (ml2 < 3) { ml2 = 0; }
        }
        
_Encode:
        if (LZ5HC_encodeSequence(ctx, &ip, &op, &anchor, ml, ref, limit, oend)) return 0;

        if (ml2)
        {
            ip = start2; ref = ref2; ml = ml2;
            ml2 = 0;
            goto _Search;
        }
    }

    /* Encode Last Literals */
    {
        int lastRun = (int)(iend - anchor);
        if ((limit) && (((char*)op - dest) + lastRun + 1 + ((lastRun+255-RUN_MASK)/255) > (U32)maxOutputSize)) return 0;  /* Check output limit */
        if (lastRun>=(int)RUN_MASK) { *op++=(RUN_MASK<<ML_BITS); lastRun-=RUN_MASK; for(; lastRun > 254 ; lastRun-=255) *op++ = 255; *op++ = (BYTE) lastRun; }
        else *op++ = (BYTE)(lastRun<<ML_BITS);
        memcpy(op, anchor, iend - anchor);
        op += iend-anchor;
    }

    /* End */
    return (int) (((char*)op)-dest);
}



static int LZ5HC_compress_fast (
    LZ5HC_Data_Structure* ctx,
    const char* source,
    char* dest,
    int inputSize,
    int maxOutputSize,
    limitedOutput_directive limit
    )
{
    ctx->inputBuffer = (BYTE*) source;
    ctx->outputBuffer = (BYTE*) dest;
    const BYTE* ip = (const BYTE*) source;
    const BYTE* anchor = ip;
    const BYTE* const iend = ip + inputSize;
    const BYTE* const mflimit = iend - MFLIMIT;
    const BYTE* const matchlimit = (iend - LASTLITERALS);

    BYTE* op = (BYTE*) dest;
    BYTE* const oend = op + maxOutputSize;

    int   ml;
    const BYTE* ref=NULL;
    const BYTE* lowPrefixPtr = ctx->base + ctx->dictLimit;
    const BYTE* const base = ctx->base;
    U32* HashPos;
    U32* HashTable  = ctx->hashTable;
	const int accel = (ctx->params.searchNum>0)?ctx->params.searchNum:1;
    
    /* init */
    ctx->end += inputSize;

    ip++;

    /* Main Loop */
    while (ip < mflimit)
    {
        HashPos = &HashTable[LZ5HC_hashPtr(ip, ctx->params.hashLog, ctx->params.searchLength)];
        ml = LZ5HC_FindBestMatchFastest (ctx, *HashPos, ip, matchlimit, (&ref));
        *HashPos =  (U32)(ip - base);
        if (!ml) { ip+=accel; continue; }

        int back = 0;
        while ((ip+back>anchor) && (ref+back > lowPrefixPtr) && (ip[back-1] == ref[back-1])) back--;
        ml -= back;
        ip += back;
        ref += back;

        if (LZ5HC_encodeSequence(ctx, &ip, &op, &anchor, ml, ref, limit, oend)) return 0;

    }

    /* Encode Last Literals */
    {
        int lastRun = (int)(iend - anchor);
        if ((limit) && (((char*)op - dest) + lastRun + 1 + ((lastRun+255-RUN_MASK)/255) > (U32)maxOutputSize)) return 0;  /* Check output limit */
        if (lastRun>=(int)RUN_MASK) { *op++=(RUN_MASK<<ML_BITS); lastRun-=RUN_MASK; for(; lastRun > 254 ; lastRun-=255) *op++ = 255; *op++ = (BYTE) lastRun; }
        else *op++ = (BYTE)(lastRun<<ML_BITS);
        memcpy(op, anchor, iend - anchor);
        op += iend-anchor;
    }

    /* End */
    return (int) (((char*)op)-dest);
}



static int LZ5HC_compress_generic (void* ctxvoid, const char* source, char* dest, int inputSize, int maxOutputSize, limitedOutput_directive limit)
{
    LZ5HC_Data_Structure* ctx = (LZ5HC_Data_Structure*) ctxvoid;

    switch(ctx->params.strategy)
    {
    default:
    case LZ5HC_fast:
        return LZ5HC_compress_fast(ctx, source, dest, inputSize, maxOutputSize, limit);
    case LZ5HC_price_fast:
        return LZ5HC_compress_price_fast(ctx, source, dest, inputSize, maxOutputSize, limit);
    case LZ5HC_lowest_price:
        return LZ5HC_compress_lowest_price(ctx, source, dest, inputSize, maxOutputSize, limit);
    }

    return 0;
}


int LZ5_sizeofStateHC(void) { return sizeof(LZ5HC_Data_Structure); }

int LZ5_compress_HC_extStateHC (void* state, const char* src, char* dst, int srcSize, int maxDstSize)
{
    if (((size_t)(state)&(sizeof(void*)-1)) != 0) return 0;   /* Error : state is not aligned for pointers (32 or 64 bits) */
    LZ5HC_init ((LZ5HC_Data_Structure*)state, (const BYTE*)src);
    if (maxDstSize < LZ5_compressBound(srcSize))
        return LZ5HC_compress_generic (state, src, dst, srcSize, maxDstSize, limitedOutput);
    else
        return LZ5HC_compress_generic (state, src, dst, srcSize, maxDstSize, noLimit);
}


int LZ5_compress_HC(const char* src, char* dst, int srcSize, int maxDstSize, int compressionLevel)
{
#if LZ5HC_HEAPMODE==1
    LZ5HC_Data_Structure* statePtr = malloc(sizeof(LZ5HC_Data_Structure));
#else
    LZ5HC_Data_Structure state;
    LZ5HC_Data_Structure* const statePtr = &state;
#endif

    int cSize = 0;
    
    if (!LZ5_alloc_mem_HC(statePtr, compressionLevel))
        return 0;
        
    cSize = LZ5_compress_HC_extStateHC(statePtr, src, dst, srcSize, maxDstSize);

    LZ5_free_mem_HC(statePtr);

#if LZ5HC_HEAPMODE==1
    free(statePtr);
#endif
    return cSize;
}



/**************************************
*  Streaming Functions
**************************************/
/* allocation */
LZ5_streamHC_t* LZ5_createStreamHC(int compressionLevel) 
{ 
    LZ5HC_Data_Structure* statePtr = (LZ5HC_Data_Structure*)malloc(sizeof(LZ5_streamHC_t));
    if (!statePtr)
        return NULL;

    if (!LZ5_alloc_mem_HC(statePtr, compressionLevel))
    {
        FREEMEM(statePtr);
        return NULL;
    }
    return (LZ5_streamHC_t*) statePtr; 
}

int LZ5_freeStreamHC (LZ5_streamHC_t* LZ5_streamHCPtr)
{
    LZ5HC_Data_Structure* statePtr = (LZ5HC_Data_Structure*)LZ5_streamHCPtr;
    if (statePtr)
    {
        LZ5_free_mem_HC(statePtr);
        free(LZ5_streamHCPtr); 
    }
    return 0; 
}


/* initialization */
void LZ5_resetStreamHC (LZ5_streamHC_t* LZ5_streamHCPtr)
{
    LZ5_STATIC_ASSERT(sizeof(LZ5HC_Data_Structure) <= sizeof(LZ5_streamHC_t));   /* if compilation fails here, LZ5_STREAMHCSIZE must be increased */
    ((LZ5HC_Data_Structure*)LZ5_streamHCPtr)->base = NULL;
}

int LZ5_loadDictHC (LZ5_streamHC_t* LZ5_streamHCPtr, const char* dictionary, int dictSize)
{
    LZ5HC_Data_Structure* ctxPtr = (LZ5HC_Data_Structure*) LZ5_streamHCPtr;
    if (dictSize > LZ5_DICT_SIZE)
    {
        dictionary += dictSize - LZ5_DICT_SIZE;
        dictSize = LZ5_DICT_SIZE;
    }
    LZ5HC_init (ctxPtr, (const BYTE*)dictionary);
    if (dictSize >= 4) LZ5HC_Insert (ctxPtr, (const BYTE*)dictionary +(dictSize-3));
    ctxPtr->end = (const BYTE*)dictionary + dictSize;
    return dictSize;
}


/* compression */

static void LZ5HC_setExternalDict(LZ5HC_Data_Structure* ctxPtr, const BYTE* newBlock)
{
    if (ctxPtr->end >= ctxPtr->base + 4)
        LZ5HC_Insert (ctxPtr, ctxPtr->end-3);   /* Referencing remaining dictionary content */
    /* Only one memory segment for extDict, so any previous extDict is lost at this stage */
    ctxPtr->lowLimit  = ctxPtr->dictLimit;
    ctxPtr->dictLimit = (U32)(ctxPtr->end - ctxPtr->base);
    ctxPtr->dictBase  = ctxPtr->base;
    ctxPtr->base = newBlock - ctxPtr->dictLimit;
    ctxPtr->end  = newBlock;
    ctxPtr->nextToUpdate = ctxPtr->dictLimit;   /* match referencing will resume from there */
}

static int LZ5_compressHC_continue_generic (LZ5HC_Data_Structure* ctxPtr,
                                            const char* source, char* dest,
                                            int inputSize, int maxOutputSize, limitedOutput_directive limit)
{
    /* auto-init if forgotten */
    if (ctxPtr->base == NULL)
        LZ5HC_init (ctxPtr, (const BYTE*) source);

    /* Check overflow */
    if ((size_t)(ctxPtr->end - ctxPtr->base) > 2 GB)
    {
        size_t dictSize = (size_t)(ctxPtr->end - ctxPtr->base) - ctxPtr->dictLimit;
        if (dictSize > LZ5_DICT_SIZE) dictSize = LZ5_DICT_SIZE;

        LZ5_loadDictHC((LZ5_streamHC_t*)ctxPtr, (const char*)(ctxPtr->end) - dictSize, (int)dictSize);
    }

    /* Check if blocks follow each other */
    if ((const BYTE*)source != ctxPtr->end)
        LZ5HC_setExternalDict(ctxPtr, (const BYTE*)source);

    /* Check overlapping input/dictionary space */
    {
        const BYTE* sourceEnd = (const BYTE*) source + inputSize;
        const BYTE* dictBegin = ctxPtr->dictBase + ctxPtr->lowLimit;
        const BYTE* dictEnd   = ctxPtr->dictBase + ctxPtr->dictLimit;
        if ((sourceEnd > dictBegin) && ((const BYTE*)source < dictEnd))
        {
            if (sourceEnd > dictEnd) sourceEnd = dictEnd;
            ctxPtr->lowLimit = (U32)(sourceEnd - ctxPtr->dictBase);
            if (ctxPtr->dictLimit - ctxPtr->lowLimit < 4) ctxPtr->lowLimit = ctxPtr->dictLimit;
        }
    }

    return LZ5HC_compress_generic (ctxPtr, source, dest, inputSize, maxOutputSize, limit);
}

int LZ5_compress_HC_continue (LZ5_streamHC_t* LZ5_streamHCPtr, const char* source, char* dest, int inputSize, int maxOutputSize)
{
    if (maxOutputSize < LZ5_compressBound(inputSize))
        return LZ5_compressHC_continue_generic ((LZ5HC_Data_Structure*)LZ5_streamHCPtr, source, dest, inputSize, maxOutputSize, limitedOutput);
    else
        return LZ5_compressHC_continue_generic ((LZ5HC_Data_Structure*)LZ5_streamHCPtr, source, dest, inputSize, maxOutputSize, noLimit);
}


/* dictionary saving */

int LZ5_saveDictHC (LZ5_streamHC_t* LZ5_streamHCPtr, char* safeBuffer, int dictSize)
{
    LZ5HC_Data_Structure* streamPtr = (LZ5HC_Data_Structure*)LZ5_streamHCPtr;
    int prefixSize = (int)(streamPtr->end - (streamPtr->base + streamPtr->dictLimit));
    if (dictSize > LZ5_DICT_SIZE) dictSize = LZ5_DICT_SIZE;
    if (dictSize < 4) dictSize = 0;
    if (dictSize > prefixSize) dictSize = prefixSize;
    memmove(safeBuffer, streamPtr->end - dictSize, dictSize);
    {
        U32 endIndex = (U32)(streamPtr->end - streamPtr->base);
        streamPtr->end = (const BYTE*)safeBuffer + dictSize;
        streamPtr->base = streamPtr->end - endIndex;
        streamPtr->dictLimit = endIndex - dictSize;
        streamPtr->lowLimit = endIndex - dictSize;
        if (streamPtr->nextToUpdate < streamPtr->dictLimit) streamPtr->nextToUpdate = streamPtr->dictLimit;
    }
    return dictSize;
}

/***********************************
*  Deprecated Functions
***********************************/
/* Deprecated compression functions */
/* These functions are planned to start generate warnings by r132 approximately */
int LZ5_compressHC(const char* src, char* dst, int srcSize) { return LZ5_compress_HC (src, dst, srcSize, LZ5_compressBound(srcSize), 0); }
int LZ5_compressHC_limitedOutput(const char* src, char* dst, int srcSize, int maxDstSize) { return LZ5_compress_HC(src, dst, srcSize, maxDstSize, 0); }
int LZ5_compressHC_continue (LZ5_streamHC_t* ctx, const char* src, char* dst, int srcSize) { return LZ5_compress_HC_continue (ctx, src, dst, srcSize, LZ5_compressBound(srcSize)); }
int LZ5_compressHC_limitedOutput_continue (LZ5_streamHC_t* ctx, const char* src, char* dst, int srcSize, int maxDstSize) { return LZ5_compress_HC_continue (ctx, src, dst, srcSize, maxDstSize); } 
int LZ5_compressHC_withStateHC (void* state, const char* src, char* dst, int srcSize) { return LZ5_compress_HC_extStateHC (state, src, dst, srcSize, LZ5_compressBound(srcSize)); }
int LZ5_compressHC_limitedOutput_withStateHC (void* state, const char* src, char* dst, int srcSize, int maxDstSize) { return LZ5_compress_HC_extStateHC (state, src, dst, srcSize, maxDstSize); } 
