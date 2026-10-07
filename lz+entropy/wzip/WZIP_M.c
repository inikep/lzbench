/*
 * WZIP_M - WZIP for inputs under 32 KB
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

/*This code assume the source data size is up to 1<<15.
  It allows to set Hash Tables and Hash Chains in compact Sint16.	
  It deploys two-level hash tables, namely, over 3 and 4 literals.
*/
#include "Memry.h"
#include "BitStream_Huffman.h"
#include "WZIP.h"
#include <math.h>
/* extension of a dictionary candidate stops at the dictionary end; input candidates are unlimited */
#define DICT_LIMIT(p) ((dictSize && (const Uint8*)(p) >= (const Uint8*)dictEnd - dictSize && (const Uint8*)(p) < (const Uint8*)dictEnd) ? dictLastMatch : NULL)
#include <stdio.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

//#define WZIP_DEBUG

/* CapHufLitBits may not exceed 14, so that 4 branches can be read in parallel before flushing */
#define   CapHufLitBits        MAX_HufWeight     
#define   CapHufLitRunBits     9
#define   CapHufMchLenBits     9
#define   CapHufMchOffBits     9

#define   MinMatchLen          3
#define   MaxMatchLen          32767                 /* the whole of a 32K block */
#define   MaxHashLen           4

#define   WLZ_Hash0(stream)    Hash_3B(stream)     /* corresponding to MinMatchLen = 3 */
#define   WLZ_Hash1(stream)    Hash_4B(stream) 



/* The number of Huffman elements may not exceed 256, so that each entry is conveniently expressed by a byte */
#define   N_HufLits            256    
#define   N_HufLitRun          32                    /* literal runs up to 2^16 - 1 */
#define   N_HufMchLen          28                    /* match lengths 3..MaxMatchLen (2^15 - 1): value symbols 3..30 */

#define   OffWidth3            13                                    /* window width associated with match length of 3, if exceeding 8, then LZ packing must be re-done */
#define   OffWidth4            15                                    /* window width associated with match length of 4 */
#define   OffWidth5            15                                    /* window width associated with match length of 5 */
#define   OffWidth6            15                                    /* window width associated with match length of 6 */
#define   OffWidth7            15                                    /* window width associated with match length of 7 */
#define   OffWidth             15

#define   OffCasheSize          4                                   /* cashe size for the latest matching offsets */ 
#define   MchOffGroup           (MaxHashLen-MinMatchLen+1)           /* Number of groups rendering different Huffman encoding */
#define   N_HufMchOffMax       (2 * OffWidth)

#define   WINDOW(w)            ( (1<<w) -OffCasheSize +1 )
/* Predefined offset window size with respect to each matching length (up to 8) */
static const Uint32 OffWindowTable[9] = { 0, 0, 0,  WINDOW(OffWidth3),   WINDOW(OffWidth4),   WINDOW(OffWidth5),
						 WINDOW(OffWidth6),  WINDOW(OffWidth7),  WINDOW(OffWidth)  };
static const unsigned N_HufMchOff[] = {2*OffWidth3,   2*OffWidth4,  2 * OffWidth5,  2 * OffWidth6,  2 * OffWidth7,  2 * OffWidth};

/* Literal runs and match lengths: values 0..7 have their own symbols; larger ones a range symbol plus raw low bits
   (four ranges each for msb 3..5, two for msb 6..7, one per msb from 8 up), as in WZIP_S */
static const Uint8 RangeBase[16]  = { 0, 0, 0, 8, 12, 16, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31 };
static const Uint8 RangeShift[16] = { 0, 0, 0, 1,  2,  3,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15 };
static const Uint32 ValueBase[32] = { 0, 1, 2, 3, 4, 5, 6, 7,   8, 10, 12, 14,   16, 20, 24, 28,   32, 40, 48, 56,
                                      64, 96,   128, 192,   256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
static const Uint8 ValueBits[32]  = { 0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1,   2, 2, 2, 2,   3, 3, 3, 3,
                                      5, 5,   6, 6,   8, 9, 10, 11, 12, 13, 14, 15 };

ForceInlineTemplate Uint32 Value_Code(Uint32 value)
{
	if (value < 8) return value;
	const Uint32 msb = High_Bit32(value);
	return RangeBase[msb] + ((value - (1u << msb)) >> RangeShift[msb]);
}

static ExtHuffman_Lit const ExtHufMchOff[] = {
	{0, 0},  {1, 0},  {2, 0}, {3, 0},           {2, 1}, {3, 1},  {2, 2},  {3, 2},    
	{2, 3},  {3, 3},  {2, 4}, {3, 4},           {2, 5}, {3, 5},  {2, 6},  {3, 6},
	{2, 7},  {3, 7},  {2, 8},  {3, 8},          {2, 9}, {3, 9},  {2, 10}, {3, 10},
	{2, 11},  {3, 11}, {2, 12}, {3, 12},        {2, 13}, {3, 13}, {2, 14}, {3, 14},
	{2, 15},  {3, 15}, {2, 16}, {3, 16},        {2, 17}, {3, 17}, {2, 18}, {3, 18},
	{2, 19},  {3, 19}, {2, 20},  {3, 20},
};

typedef struct {
	Huffman_Str litRunHuf[N_HufLitRun];
	Huffman_Str mchLenHuf[N_HufMchLen];
	Huffman_Str mchOffHuf[MchOffGroup][N_HufMchOffMax];
} WLZ_Huffman_Set;

typedef struct {
	HufCode_Str litRun[N_HufLitRun];
	HufCode_Str mchLen[N_HufMchLen];
	HufCode_Str mchOff[MchOffGroup][N_HufMchOffMax];
} WLZ_HufCode_Set;

typedef struct {
	Uint32 maxLitRunHufWt;
	Uint32 maxMchLenHufWt;
	Uint32 maxMchOffHufWt[MchOffGroup];
	Uint8 litRunHufWt[N_HufLitRun];
	Uint8 mchLenHufWt[N_HufMchLen];
	Uint8 mchOffHufWt[MchOffGroup][N_HufMchOffMax];
} WLZ_HufWt_Set;

typedef struct {
	Uint32 litRun;
	Uint32 mchLen;
	Uint32 mchOff;
} WLZ_Set;

ForceInlineTemplate Uint32 WLZ_Match_Count(const Uint8* srcPtr, const Uint8* matchPtr, const Uint8* const srcLimit, const Uint8* const matchLimit)
{
	Uint32 matchLen = 0;
	reg_t matchDiff;

	matchDiff = MemReadARCH(matchPtr) ^ MemReadARCH(srcPtr);
	while ( 0 == matchDiff && srcPtr < srcLimit && (matchLimit == NULL || matchPtr < matchLimit) ) {
		srcPtr += REG_SIZE;
		matchPtr += REG_SIZE;
		matchLen += REG_SIZE;
		matchDiff = MemReadARCH(matchPtr) ^ MemReadARCH(srcPtr);
	}

	matchLen += matchDiff == 0 ? REG_SIZE : N_ZeroBytes(matchDiff);

	return matchLen;
}

ForceInlineTemplate int Offset_Huffman_Index(int offset, int offMsb)
{
	if (offMsb < 2) return offset;
	return 2 * (offMsb - 1) + (offset >> (offMsb - 1));
}

/* OffCasheSize may not exceed 4, otherwise, program will crash.
   The cache is kept most recent first: a hit moves its offset to the front, a new offset pushes out the oldest */
ForceInlineTemplate Uint32 Offset_Cashe(Uint32* lastOffset, Uint32 matchOffset)
{
	const Uint32 casheIdx = (matchOffset == lastOffset[0]) + (matchOffset == lastOffset[1]) * 2 + (matchOffset == lastOffset[2]) * 3 + (matchOffset == lastOffset[3]) * 4;

	if (casheIdx) {
		const Uint32 k = casheIdx - 1;
		if (k >= 3) lastOffset[3] = lastOffset[2];
		if (k >= 2) lastOffset[2] = lastOffset[1];
		if (k >= 1) lastOffset[1] = lastOffset[0];
		lastOffset[0] = matchOffset;
		return k;
	}
	else {
		lastOffset[3] = lastOffset[2];
		lastOffset[2] = lastOffset[1];
		lastOffset[1] = lastOffset[0];
		lastOffset[0] = matchOffset;
		return matchOffset + OffCasheSize -	1;
	}

}

ForceInlineTemplate Uint32 WLZ2_Compress_Fast(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	WLZ_Set* wlzSeq,
	Uint8** zipLitStream,
	WLZ_Huffman_Set *huffmanSet)
{
	Uint32 i;
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	const int dictSize = wzipStr->dictSize;
	const Uint8*  dictEnd = wzipStr->dictEnd;
	Uint32 nLzLits;

	Huffman_Str litHuf[N_HufLits];
	WLZ_Set* destPtr = wlzSeq;
	Uint8* zipLitPtr = *zipLitStream;
	zipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32  zipLitBlkSize;
	Uint8* lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* lzLitEnd = lzLitBuffer + HUF_BlockSize;
	Uint32 hashV0, hashV1;
	int  match0Idx, match1Idx;
	Uint32  lazyMatchLen, lazyMatchOffset, lazyMatchFail;
	Uint32 matchLen, matchOffset;
	Uint32 matchLen2, matchOffset2;
	Uint32 lastOffset[OffCasheSize];
	Uint32 litRun;
	Sint16* hash0Table = (Sint16 *)wzipStr->hash0Table;
	Sint16* hash1Table = (Sint16 *)wzipStr->hash1Table;
	const Uint32 offWindow = OffWindowTable[8];
	int litRunHufIdx, mchLenHufIdx, offsetMsb, offsetHufIdx;

	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	reg_t currPattern, diffPattern;

	const Uint8* matchPtr;

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "WLZ2_Compress_Index.txt", "w");
	fprintf(fptr, "WLZ2_Compress_Fast: srcSize=%i\n", srcSize);
#endif

	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	nLzLits = 0;
	Uint32 srcIdx = 0;
	for (i = 0; i < OffCasheSize; i++ )
		lastOffset[i] = 0xFFFFFFFFu;            /* unset: above every offset, so never a hit */

	while (1) {

		while (1) {
			if (srcIdx == 10) {
				srcIdx += 0;
			}

			if (unlikely(srcPtr >= srcLastMatch)) goto _last_literals;

			hashV0 = WLZ_Hash0(srcPtr) & wzipStr->hash0Mask;
			hashV1 = WLZ_Hash1(srcPtr) & wzipStr->hash1Mask;
						
			match0Idx = hash0Table[hashV0];
			hash0Table[hashV0] = srcIdx;

			match1Idx = hash1Table[hashV1];
			hash1Table[hashV1] = srcIdx;

			matchLen = matchLen2 = matchOffset2 = 0;
			currPattern = MemReadARCH(srcPtr);
			matchOffset = srcIdx - match1Idx;
			if ( match1Idx>=-dictSize && matchOffset > 0 && matchOffset < offWindow ) {
				matchPtr = (dictSize && match1Idx < 0) ? dictEnd + match1Idx : srcPtr - matchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				if (0 == diffPattern) {
					matchLen = REG_SIZE + WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, DICT_LIMIT(matchPtr));
					break;
				}
				matchLen = N_ZeroBytes(diffPattern);
				if ( matchLen <= MaxHashLen && matchOffset >= OffWindowTable[matchLen] )
					matchLen = 0;
				if (matchLen > matchLen2) {
					matchLen2 = matchLen;
					matchOffset2 = matchOffset;
				}
			}

			matchOffset = srcIdx - match0Idx;
			if ( match0Idx >= -dictSize && matchOffset > 0 && matchOffset < offWindow ) {
				matchPtr = (dictSize && match0Idx < 0) ? dictEnd + match0Idx : srcPtr - matchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				matchLen = diffPattern? N_ZeroBytes(diffPattern) : REG_SIZE;
				if (matchLen <= MaxHashLen && matchOffset >= OffWindowTable[matchLen])
					matchLen = 0;
			}

			if (matchLen2 > matchLen) {
				matchLen = matchLen2;
				matchOffset = matchOffset2;
			}
			if (matchLen >= MinMatchLen)
				break;

			*lzLitPtr++ = *srcPtr;
			litHuf[*srcPtr].freq++;
			srcPtr++;
			srcIdx++;
			if (lzLitPtr == lzLitEnd) {
				zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
				zipLitPtr += zipLitBlkSize;
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
				nLzLits += HUF_BlockSize;
			}
		}

		lazyMatchFail = 1;
		srcPtr++;
		srcIdx++;
		currPattern = MemReadARCH(srcPtr);

		hashV0 = WLZ_Hash0(srcPtr) & wzipStr->hash0Mask;
		hashV1 = WLZ_Hash1(srcPtr) & wzipStr->hash1Mask;
		match0Idx = hash0Table[hashV0];
		hash0Table[hashV0] = srcIdx;
		match1Idx = hash1Table[hashV1];
		hash1Table[hashV1] = srcIdx;
		
		lazyMatchOffset = srcIdx - match1Idx;
		if ( match1Idx>=-dictSize && lazyMatchOffset>0 && lazyMatchOffset < offWindow ) {
			matchPtr = (dictSize && match1Idx < 0) ? dictEnd + match1Idx : srcPtr - lazyMatchOffset;
			diffPattern = currPattern ^ MemReadARCH(matchPtr);

			if (0 == diffPattern) {
				lazyMatchLen = REG_SIZE + WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, DICT_LIMIT(matchPtr));
			}
			else {
				lazyMatchLen = N_ZeroBytes(diffPattern);
				if (lazyMatchOffset >= OffWindowTable[lazyMatchLen] ) lazyMatchLen = 0;
			}

			if (lazyMatchLen > matchLen) {
				matchLen = lazyMatchLen;
				matchOffset = lazyMatchOffset;
				lazyMatchFail = 0;
			}
		}

		if (lazyMatchFail) {
			srcPtr--;
			srcIdx--;
		}
		else {
			*lzLitPtr++ = *(srcPtr - 1);
			litHuf[*(srcPtr-1)].freq++;
			if (lzLitPtr == lzLitEnd) {
				zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
				zipLitPtr += zipLitBlkSize;
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
				nLzLits += HUF_BlockSize;
			}
		}

		matchLen = min(matchLen, MaxMatchLen);

		/* index the match's later positions; those from srcLastMatch on are never searched (and a hash there would
		   read past the input when the match ends near it) */
		for (i = 2; i < matchLen && srcPtr + i < srcLastMatch; i++) {
			hashV0 = WLZ_Hash0(srcPtr + i) & wzipStr->hash0Mask;
			hashV1 = WLZ_Hash1(srcPtr + i) & wzipStr->hash1Mask;
			hash0Table[hashV0] = srcIdx + i;
			hash1Table[hashV1] = srcIdx + i;
		}

		litRun = (Uint32)(srcPtr - anchor);

		/* special backward match to mitigate spurious hashing match. The match source (a negative history
		   position lies in the dictionary) extends backward within its own segment only: the decoder copies
		   a match from either the dictionary or the output, never across the two */
		while (litRun > 0 && lzLitPtr > lzLitBuffer && matchLen < MaxMatchLen &&
			((int)srcIdx - (int)matchOffset > 0 || ((int)srcIdx - (int)matchOffset < 0 && (int)srcIdx - (int)matchOffset > -dictSize))) {
			const int histIdx = (int)srcIdx - (int)matchOffset - 1;
			if (*(srcPtr - 1) != (histIdx >= 0 ? source[histIdx] : dictEnd[histIdx])) break;
			srcIdx--;
			srcPtr--;
			lzLitPtr--;
			matchLen++;
			litRun--;
			litHuf[*srcPtr].freq--;
		}

#ifdef WZIP_DEBUG
		fprintf(fptr, "srcIdx=%d, litRun=%d,  matchLen=%d, matchOffset=%d,   ",
			(int)(anchor - (const Uint8*)source), litRun, matchLen, matchOffset);
#endif
		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Encode Literal Run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		litRunHufIdx = Value_Code(litRun);
		destPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ValueBits[litRunHufIdx]]) << 8;
		huffmanSet->litRunHuf[litRunHufIdx].freq++;

		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Fast Encode Match Pair  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		srcIdx += matchLen;
		srcPtr += matchLen;

		matchOffset = Offset_Cashe(lastOffset, matchOffset);
#ifdef WZIP_DEBUG
		fprintf(fptr, "-->  matchLen=%d,  matchOff=%d\n", matchLen, matchOffset);
		fflush(fptr);
#endif
		{
			const Uint32 lenSym = Value_Code(matchLen);
			mchLenHufIdx = lenSym - MinMatchLen;
			destPtr->mchLen = mchLenHufIdx ^ (matchLen & BitMask[ValueBits[lenSym]]) << 8;
			huffmanSet->mchLenHuf[mchLenHufIdx].freq++;
		}
		
		if (matchOffset < 4) {
			destPtr->mchOff = matchOffset;
			huffmanSet->mchOffHuf[min(MchOffGroup - 1, mchLenHufIdx)][matchOffset].freq++;
		}
		else {
			offsetMsb = High_Bit32(matchOffset);
			offsetHufIdx = Offset_Huffman_Index(matchOffset, offsetMsb);
			destPtr->mchOff = offsetHufIdx ^ (matchOffset & BitMask[ExtHufMchOff[offsetHufIdx].lsBits]) <<8;
			huffmanSet->mchOffHuf[min(MchOffGroup-1, mchLenHufIdx)][offsetHufIdx].freq++;
		}

		destPtr++;
		anchor = srcPtr;
		matchLen = 0;
	}

_last_literals:
	/* Encode Last Literals */
	litRun = (int)(srcEnd - anchor);

	int lastLits = (int)(srcEnd - srcPtr);
	if (lzLitPtr + lastLits > lzLitEnd) {
		while (lzLitPtr < lzLitEnd) {
			litHuf[*srcPtr].freq++;
			*lzLitPtr++ = *srcPtr++;
		}
		zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
		zipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);     /* remaining number literals in the buffer to be flushed */
	zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, lastBufLits, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
	zipLitPtr += zipLitBlkSize;
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(*zipLitStream, nLzLits);       /* record the number of LZ literals at the beginning of zip stream */
	else               MemWriteLE2(*zipLitStream, (Uint16)nLzLits);
	*zipLitStream = zipLitPtr;

#ifdef WZIP_DEBUG
	fprintf(fptr, "srcIdx=%d, litRun=%d\n", (int)(anchor - (const Uint8*)source), litRun);
	fclose(fptr);
#endif

	litRunHufIdx = Value_Code(litRun);
		destPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ValueBits[litRunHufIdx]]) << 8;
		huffmanSet->litRunHuf[litRunHufIdx].freq++;

	free(lzLitBuffer);
	return (Uint32)(destPtr+1-wlzSeq);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************Hash - Chain Compression Functions * *******************************************************************
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
typedef struct wlz_match {
	Uint32 len;                                        // LZ match length
	Uint32 off;                                        // LZ match offset/distance  
} WLZ_Match;

ForceInlineTemplate void WZIP_Search_Hash1Chain(WZIP_State_Str* const wzipStr, const Uint8* const source, Uint32 currIdx,
	const Uint8* srcLastMatch, const Uint8* dictLastMatch, WLZ_Match* matchStr, int chainSearchCnt)
{
	Uint16* chainTable = (Uint16*)wzipStr->chain1Table;
	const Uint32 chainMask = wzipStr->chain1Mask;
	Sint16* hashTable = (Sint16*)wzipStr->hash1Table;
	int preIdx = wzipStr->curr1Idx;
	wzipStr->curr1Idx = max(preIdx, currIdx);
	Uint8* matchPtr, * srcPtr;
	Uint32 hashV, dist, matchDist;
	int matchLen, matchIdx;
	const int dictSize = wzipStr->dictSize;
	Uint8* const dictEnd = wzipStr->dictEnd;
	const Uint32 offWindow = OffWindowTable[8];
	

	srcPtr = (Uint8*)source + preIdx;
	while (preIdx <= currIdx) {
		hashV = WLZ_Hash1(srcPtr++) & wzipStr->hash1Mask;
		dist = preIdx - hashTable[hashV];
		chainTable[preIdx & chainMask] = (dist > 0 && dist < chainMask && hashTable[hashV] >= -dictSize) ? dist : chainMask;
		hashTable[hashV] = preIdx++;
	}

	srcPtr = (Uint8*)source + currIdx;
	Uint32 currPattern = MemRead4(srcPtr);
	matchDist = chainTable[currIdx & chainMask];
	matchIdx = currIdx - matchDist;
	while (matchIdx >= -dictSize && matchDist < offWindow && chainSearchCnt) {		
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, DICT_LIMIT(matchPtr));
			if (matchLen > matchStr->len && matchDist < OffWindowTable[min(8, matchLen)]) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
			}
		}
		chainSearchCnt--;
		matchDist += chainTable[matchIdx & chainMask];
		matchIdx = currIdx - matchDist;
	}
}


ForceInlineTemplate int WZIP_Search_Hash1Chain_2D(WZIP_State_Str* const wzipStr, const Uint8* const source, Uint32 currIdx, 
	int maxBack, int nextMatchLen, const Uint8* srcLastMatch, const Uint8* dictLastMatch, WLZ_Match* matchStr, int chainSearchCnt)
{
	Uint16* chain1Table = (Uint16*)wzipStr->chain1Table;
	Sint16* hash1Table = (Sint16*)wzipStr->hash1Table;
	const Uint32 chain1Mask = wzipStr->chain1Mask;
	Uint8* matchPtr, * srcPtr;
	Uint32 hashV, dist, matchDist;
	int matchLen, matchIdx;
	int back, score, optBack = maxBack;
	Uint8 backByte[8];
	int backIdx, backTable[8] = { 0, 1, 0, 2,  0, 1, 0, 3 };
	Uint32 currPattern;
	const int dictSize = wzipStr->dictSize;
	Uint8* const dictEnd = wzipStr->dictEnd;
	const Uint32 offWindow = OffWindowTable[8];   // note offWindow <= chain1Mask, nearly equal 

	srcPtr = (Uint8*)source + wzipStr->curr1Idx;
	while (wzipStr->curr1Idx <= currIdx) {
		hashV = WLZ_Hash1(srcPtr++) & wzipStr->hash1Mask;
		dist = wzipStr->curr1Idx - hash1Table[hashV];
		chain1Table[wzipStr->curr1Idx & chain1Mask] = (dist >0 && dist < chain1Mask && hash1Table[hashV] >= -dictSize) ? dist : chain1Mask;
		hash1Table[hashV] = wzipStr->curr1Idx++;
	}

	srcPtr = (Uint8*)source + currIdx;
	int searchCnt = chainSearchCnt * 3 / 4 + 4;
	currPattern = MemRead4(srcPtr);
	backByte[0] = *(srcPtr - 1);
	backByte[1] = *(srcPtr - 2);
	backByte[2] = *(srcPtr - 3);
	//backByte[3] = *(srcPtr - 4);
	matchDist = chain1Table[currIdx & chain1Mask];
	matchIdx = currIdx - matchDist;
	while (matchIdx>= -dictSize && matchDist < offWindow && searchCnt) {		
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, DICT_LIMIT(matchPtr));

			//backIdx = (backByte[0] == *(matchPtr - 1));
			backIdx = (matchIdx >= 2 || (matchIdx < 0 && matchIdx >= 2 - (int)dictSize)) ? (backByte[0] == *(matchPtr - 1)) ^ ((backByte[1] == *(matchPtr - 2)) << 1) : 0;   /* never extend before the history start */
			//backIdx = (backByte[0] == *(matchPtr - 1)) ^ ((backByte[1] == *(matchPtr - 2)) << 1) ^ ((backByte[2] == *(matchPtr - 3)) << 2);
			back = backTable[backIdx];
			matchLen += back;
			score = matchLen - matchStr->len + (back - optBack) * nextMatchLen / maxBack;
			if ((score > 0 || (back < optBack && score == 0)) && matchDist < OffWindowTable[min(8, matchLen)]) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
				optBack = back;
			}
		}
		searchCnt--;
		matchDist += chain1Table[matchIdx & chain1Mask];
		matchIdx = currIdx - matchDist;
	}
	//if (maxBack - optBack > 0) return maxBack - optBack;

	searchCnt = 2 + chainSearchCnt / 4;
	back = maxBack - 1;
	currIdx -= back;
	srcPtr -= back;
	currPattern = MemRead4(srcPtr);
	matchDist = chain1Table[currIdx & chain1Mask];
	matchIdx = currIdx - matchDist;
	while ( matchIdx >= -dictSize && matchDist < offWindow && searchCnt ) {
		
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, DICT_LIMIT(matchPtr));
			score = matchLen - matchStr->len + (back - optBack) * nextMatchLen / maxBack;
			if (score >= 0 && matchDist < OffWindowTable[min(8, matchLen)] ) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
				optBack = back;
			}
		}
		searchCnt--;
		matchDist += chain1Table[matchIdx & chain1Mask];
		matchIdx = currIdx - matchDist;
	}
	return maxBack - optBack;
}


/** forced inline, to ensure branches are decided at compilation time **/
ForceInlineTemplate Uint32 WLZ2_Compress(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	WLZ_Set* wlzSeq,
	Uint8** zipLitStream,
	WLZ_Huffman_Set* huffmanSet,
	int maxSearchCnt)
{
	Sint16* hash0Table = (Sint16*)wzipStr->hash0Table;
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	const int dictSize = wzipStr->dictSize;
	const Uint8* dictEnd = wzipStr->dictEnd;
	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	Uint32 nLzLits;

	Huffman_Str litHuf[N_HufLits];
	WLZ_Set* destPtr = wlzSeq;
	Uint8* zipLitPtr = *zipLitStream;
	zipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32  zipLitBlkSize;
	Uint8* lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* lzLitEnd = lzLitBuffer + HUF_BlockSize;
	Uint32 lastOffset[OffCasheSize];
	int litRun, litRunHufIdx, mchLenHufIdx, offsetMsb, offsetHufIdx;
	
	WLZ_Match matchStr, nextMatchStr = { 0, 0 };

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "WLZ2_Compress_Index.txt", "w");
	fprintf(fptr, "WZIP2_Compress_Kernel: srcSize=%i\n", srcSize);
#endif
	wzipStr->curr1Idx = 0;

	int curr0Idx = 0;
	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	memset(lastOffset, 0xFF, OffCasheSize * sizeof(int));     /* unset: above every offset, so never a hit */
	nLzLits = 0;
	Uint32 srcIdx = 0, hashV;
	int nextMatchDone = 0;
	while (1) {

		while (1) {
			if (srcIdx == 31296) {
				srcIdx += 0;
			}

			if (unlikely(srcPtr >= srcLastMatch)) goto _last_literals;

			if (nextMatchDone) {
				matchStr = nextMatchStr;
				nextMatchDone = 0;
			}
			else {
				matchStr.len = 0;

				WZIP_Search_Hash1Chain(wzipStr, source, srcIdx, srcLastMatch, dictLastMatch, &matchStr, maxSearchCnt);
				if (!matchStr.len) {
					Uint8* src_ptr = (Uint8*)source + curr0Idx;
					while (src_ptr < srcPtr) {
						hashV = WLZ_Hash0(src_ptr++) & wzipStr->hash0Mask;
						hash0Table[hashV] = curr0Idx++;
					}
					hashV = WLZ_Hash0(srcPtr) & wzipStr->hash0Mask;
					int match0Idx = hash0Table[hashV];
					int offset = srcIdx - match0Idx;   // note curr0Idx == srcIdx
					hash0Table[hashV] = curr0Idx++;
					if (match0Idx >= -dictSize && offset > 0 && offset < OffWindowTable[MaxHashLen]) {
						const Uint8* matchPtr = (dictSize && match0Idx < 0) ? dictEnd + match0Idx : srcPtr - offset;
						reg_t diff = MemReadARCH(srcPtr) ^ MemReadARCH(matchPtr);
						matchStr.len = diff? N_ZeroBytes(diff) : REG_SIZE;
						matchStr.off = offset;
						if (matchStr.len <= MaxHashLen && offset >= OffWindowTable[matchStr.len])
							matchStr.len = 0;
					}
				}
			}
			if (matchStr.len >= MinMatchLen) break;

			*lzLitPtr++ = *srcPtr;
			litHuf[*srcPtr].freq++;
			srcPtr++;
			srcIdx++;
			if (lzLitPtr == lzLitEnd) {
				zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
				zipLitPtr += zipLitBlkSize;
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
				nLzLits += HUF_BlockSize;
			}
		}
		if (likely(srcPtr + matchStr.len < srcLastMatch && srcPtr + MaxHashLen < srcLastMatch) && matchStr.len<=MaxMatchLen ) {

			nextMatchStr.len = 2;  /*nextMatchStr.off = WLZ_MAX_DIST;*/
			WZIP_Search_Hash1Chain(wzipStr, source, srcIdx + matchStr.len, srcLastMatch, dictLastMatch, &nextMatchStr, maxSearchCnt/2);
			int lazyForward = WZIP_Search_Hash1Chain_2D(wzipStr, source, srcIdx + 3, 3, nextMatchStr.len, srcLastMatch, dictLastMatch, &matchStr, 1 + maxSearchCnt / 4);

			nextMatchDone = (0 == lazyForward);

			const Uint8* srcPtrEnd = srcPtr + lazyForward;
			while (srcPtr < srcPtrEnd) {
				*lzLitPtr++ = *srcPtr;
				litHuf[*srcPtr].freq++;
				srcPtr++;
				if (lzLitPtr == lzLitEnd) {
					zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
					zipLitPtr += zipLitBlkSize;
					lzLitPtr = lzLitBuffer;
					memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
					nLzLits += HUF_BlockSize;
				}
			}
			srcIdx += lazyForward;
		}

		if (matchStr.len > MaxMatchLen) {
			matchStr.len = MaxMatchLen;
			nextMatchDone = 0;
		}

		litRun = (Uint32)(srcPtr - anchor);
		

#ifdef WZIP_DEBUG
		fprintf(fptr, "srcIdx=%d, litRun=%d,  matchLen=%d, matchOffset=%d,   ",
			(int)(anchor - (const Uint8*)source), litRun, matchStr.len, matchStr.off);
#endif
		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Encode Literal Run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		litRunHufIdx = Value_Code(litRun);
		destPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ValueBits[litRunHufIdx]]) << 8;
		huffmanSet->litRunHuf[litRunHufIdx].freq++;

		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Fast Encode Match Pair  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		srcIdx += matchStr.len;
		srcPtr += matchStr.len;

		matchStr.off = Offset_Cashe(lastOffset, matchStr.off);
#ifdef WZIP_DEBUG
		fprintf(fptr, "-->  matchLen=%d,  matchOff=%d\n", matchStr.len, matchStr.off);
		fflush(fptr);
#endif
		{
			const Uint32 lenSym = Value_Code(matchStr.len);
			mchLenHufIdx = lenSym - MinMatchLen;
			destPtr->mchLen = mchLenHufIdx ^ (matchStr.len & BitMask[ValueBits[lenSym]]) << 8;
			huffmanSet->mchLenHuf[mchLenHufIdx].freq++;
		}

		if (matchStr.off < 4) {
			destPtr->mchOff = matchStr.off;
			huffmanSet->mchOffHuf[min(MchOffGroup - 1, mchLenHufIdx)][matchStr.off].freq++;
		}
		else {
			offsetMsb = High_Bit32(matchStr.off);
			offsetHufIdx = Offset_Huffman_Index(matchStr.off, offsetMsb);
			destPtr->mchOff = offsetHufIdx ^ (matchStr.off& BitMask[ExtHufMchOff[offsetHufIdx].lsBits])<<8;
			huffmanSet->mchOffHuf[min(MchOffGroup - 1, mchLenHufIdx)][offsetHufIdx].freq++;
		}

		destPtr++;
		anchor = srcPtr;
		matchStr.len = 0;
	}

_last_literals:
	/* Encode Last Literals */
	litRun = (int)(srcEnd - anchor);

	int lastLits = (int)(srcEnd - srcPtr);
	if (lzLitPtr + lastLits > lzLitEnd) {
		while (lzLitPtr < lzLitEnd) {
			litHuf[*srcPtr].freq++;
			*lzLitPtr++ = *srcPtr++;
		}
		zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
		zipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);     /* remaining number literals in the buffer to be flushed */
	zipLitBlkSize = Huffman_Compress_Block(lzLitBuffer, lastBufLits, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
	zipLitPtr += zipLitBlkSize;
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(*zipLitStream, nLzLits);       /* record the number of LZ literals at the beginning of zip stream */
	else               MemWriteLE2(*zipLitStream, (Uint16)nLzLits);
	*zipLitStream = zipLitPtr;

#ifdef WZIP_DEBUG
	fprintf(fptr, "srcIdx=%d, litRun=%d\n", (int)(anchor - (const Uint8*)source), litRun);
	fclose(fptr);
#endif

	litRunHufIdx = Value_Code(litRun);
		destPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ValueBits[litRunHufIdx]]) << 8;
		huffmanSet->litRunHuf[litRunHufIdx].freq++;

	free(lzLitBuffer);
	return (Uint32)(destPtr+1 - wlzSeq);
}


ForceInlineTemplate Uint32 Huffman_Compress_Seq_Body(WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd, Uint8* zipBuffer, WLZ_HufCode_Set* hufCodeSet)
{
	Uint32 litRunHufIdx, mchLenHufIdx, offHufIdx, offGroup;
	/* even sequences go to stream A (forward, here), odd ones to stream B, stored byte-reversed after A */
	const Uint32 capB = (Uint32)(wlzSeqEnd - wlzSeq) * 8 + 16;
	Uint8* const bufB = (Uint8*)malloc(capB);
	Bit_Stream streams[2] = { { 0, 0, zipBuffer }, { 0, 0, bufB } };
	

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "Huffman_Compress_Index.txt", "w");
#endif

	for(WLZ_Set* wlzSeqPtr = wlzSeq; wlzSeqPtr<wlzSeqEnd; wlzSeqPtr++) {
		Bit_Stream bitStream = streams[(wlzSeqPtr - wlzSeq) & 1];
		litRunHufIdx = wlzSeqPtr->litRun &255;
#ifdef WZIP_DEBUG 
		fprintf(fptr, "lzIdx=%d,  litRunIdx=%d,  ", (Uint32)(lzBufPtr - lzBuffer - 1), litRunHufIdx);
#endif
		BITStream_Write(bitStream, hufCodeSet->litRun[litRunHufIdx].code, hufCodeSet->litRun[litRunHufIdx].nbits);
		if (ValueBits[litRunHufIdx]) {
			BITStream_Write(bitStream, wlzSeqPtr->litRun >> 8, ValueBits[litRunHufIdx]);
			BITStream_Write_Flush(bitStream);
		}

#ifdef WZIP_DEBUG 
		if (ValueBits[litRunHufIdx])
			fprintf(fptr, "lsBits=%d;    ", ValueBits[litRunHufIdx]);
#endif

		if (wlzSeqPtr == wlzSeqEnd - 1) {  /* terminal record carries only the last literal run */
			streams[(wlzSeqPtr - wlzSeq) & 1] = bitStream;
			break;
		}

		mchLenHufIdx = wlzSeqPtr->mchLen &255;
		BITStream_Write(bitStream, hufCodeSet->mchLen[mchLenHufIdx].code, hufCodeSet->mchLen[mchLenHufIdx].nbits);
#ifdef WZIP_DEBUG 
		fprintf(fptr, "matchLenIdx=%d,  ", mchLenHufIdx);
		if (ValueBits[mchLenHufIdx + MinMatchLen])
			fprintf(fptr, "lsBits=%d;      ", ValueBits[mchLenHufIdx + MinMatchLen]);
#endif

		offGroup = min(MchOffGroup - 1, mchLenHufIdx);
		offHufIdx = wlzSeqPtr->mchOff &255;
		BITStream_Write(bitStream, hufCodeSet->mchOff[offGroup][offHufIdx].code, hufCodeSet->mchOff[offGroup][offHufIdx].nbits);
		if (offHufIdx >= 4) {                      /* append tail bits */
			BITStream_Write(bitStream, wlzSeqPtr->mchOff>>8, ExtHufMchOff[offHufIdx].lsBits);
		}
		BITStream_Write_Flush(bitStream);

		if (ValueBits[mchLenHufIdx + MinMatchLen]) {
			BITStream_Write(bitStream, wlzSeqPtr->mchLen >> 8, ValueBits[mchLenHufIdx + MinMatchLen]);
		}
		streams[(wlzSeqPtr - wlzSeq) & 1] = bitStream;


#ifdef WZIP_DEBUG 
		fprintf(fptr, "OffGrp=%d,   matchOffHufIdx=%d,  ", offGroup, offHufIdx);
		if (offHufIdx >= 4)
			fprintf(fptr, "lsBits=%d,  lsValue=%d\n", ExtHufMchOff[offHufIdx].lsBits, lsValue);
		else fprintf(fptr, "\n");

		fflush(fptr);
#endif

}
	BITStream_Write_FlushEnd(streams[0]);
	BITStream_Write_FlushEnd(streams[1]);
	Uint8* zipPtr = streams[0].streamPtr;
	for (Uint8* q = streams[1].streamPtr; q > bufB; )      /* stream B byte-reversed: the decoder reads it backward */
		*zipPtr++ = *--q;
	free(bufB);

#ifdef WZIP_DEBUG
	fclose(fptr);
#endif
	return (Uint32)(zipPtr - zipBuffer);
}


static Uint32 Huffman_Compress_Seq_Kernel(WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd, Uint8* zipBuffer, WLZ_HufCode_Set* hufCodeSet)
{
	return Huffman_Compress_Seq_Body(wlzSeq, wlzSeqEnd, zipBuffer, hufCodeSet);
}


/* Second Pass:  Apply Huffman encoding on top of WLZ compression */
ForceInlineTemplate Uint32 Huffman_Compress_WLZ(WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd,
	Uint8* zipBuffer, Uint32 zipBufSize,
	WLZ_Huffman_Set* huffmanSet)
{
	int i;
	Uint8* zipBufPtr;
	Bit_Stream bitStream = { 0, 0, zipBuffer };


	WLZ_HufCode_Set hufCodeSet;
	Huffman_Str hufWtHuf[MAX_HufWeight + 3] = { 0 };
	HufCode_Str hufWtHufCode[MAX_HufWeight + 3];
	Uint8 hufWtSet[MAX_HufSize * 2 + N_HufLitRun+N_HufMchLen + N_HufMchOffMax*MchOffGroup];
	Uint32 hufWtSetSize = 0;

	Build_Huffman_Table(huffmanSet->litRunHuf, N_HufLitRun, CapHufLitRunBits, hufCodeSet.litRun);
	Build_Huffman_Table(huffmanSet->mchLenHuf, N_HufMchLen, CapHufMchLenBits, hufCodeSet.mchLen);
	for (i = 0; i < MchOffGroup; i++) {
		Build_Huffman_Table(huffmanSet->mchOffHuf[i], N_HufMchOff[i], CapHufMchOffBits, hufCodeSet.mchOff[i]);
	}

#ifdef WZIP_DEBUG 
	FILE* fptr = NULL;
	fopen_s(&fptr, "WZIP_Huffman_Trees.txt", "w");
	fprintf(fptr, "\nLiteral Run Huffman Tree\n");
	for (i = 0; i < N_HufLitRun; i++) {
		fprintf(fptr, "(%4d,  %2d),  ", huffmanSet->litRunHuf[i].freq, huffmanSet->litRunHuf[i].freq>0? hufCodeSet.litRun[i].nbits: 0);
		if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\nMatch Length Huffman Tree\n");
	for (i = 0; i < N_HufMchLen; i++) {
		fprintf(fptr, "(%4d,  %2d),  ", huffmanSet->mchLenHuf[i].freq, huffmanSet->mchLenHuf[i].freq>0? hufCodeSet.mchLen[i].nbits: 0);
		if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\nMatch Offset Huffman Trees\n");
	for(int n=0; n<MchOffGroup; n++) {
		for (i = 0; i < (int)N_HufMchOff[n]; i++) {
			fprintf(fptr, "(%4d,  %4d),  ", huffmanSet->mchOffHuf[n][i].freq, huffmanSet->mchOffHuf[n][i].freq>0? hufCodeSet.mchOff[n][i].nbits: 0);
			if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n\n");
	}

	fclose(fptr);
#endif

	hufWtSetSize = Count_Huffman_Weight_Frequency(hufCodeSet.litRun, N_HufLitRun, hufWtHuf, hufWtSet + hufWtSetSize);
	hufWtSetSize += Count_Huffman_Weight_Frequency(hufCodeSet.mchLen, N_HufMchLen, hufWtHuf, hufWtSet + hufWtSetSize);
	for (i = 0; i < MchOffGroup; i++)
		hufWtSetSize += Count_Huffman_Weight_Frequency(hufCodeSet.mchOff[i], N_HufMchOff[i], hufWtHuf, hufWtSet + hufWtSetSize);

	Build_Huffman_Table(hufWtHuf, MAX_HufWeight + 3, MAX_HufHufWt, hufWtHufCode);
	Write_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, hufWtHufCode);
	Write_Huffman_Header_byHuffman(&bitStream, hufWtHufCode, hufWtSet, hufWtSetSize);

	BITStream_Write_FlushEnd(bitStream);
	zipBufPtr = bitStream.streamPtr;

	Uint32 resLen = Huffman_Compress_Seq_Kernel(wlzSeq, wlzSeqEnd, zipBufPtr, &hufCodeSet);
	zipBufPtr += resLen;

	Uint32 zipLzSeqSize = (Uint32)(zipBufPtr - zipBuffer);
	if (zipLzSeqSize <= zipBufSize)
		return zipLzSeqSize;
	else {
		return 0;             /* overflow: the caller gives up */
	}
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimal parsing (levels 10-12) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A shortest-path parse, priced in bits from symbol statistics with the encoder's own symbols: literals, literal-run and
   match-length codes with their extra bits, and the offset code of the match's length group with its raw bits. A raw
   offset must lie in the window of its length; an offset in the repeat cache (tracked per path) costs only its slot code
   and has no window limit. The block is parsed twice: the first pass learns the statistics that price the second.
   Match finder, nearest first: a 3-byte hash chain for length 3 (window of length 3), and a 4-byte hash chain for
   lengths 4+ (the whole window). */
#define   OPT_Num              4096
#define   OPT_Unit             256                     /* prices in 1/256 bit */
#define   OPT_Inf              0x3FFFFFFF
#define   OPT_UpdateBytes      (1 << 12)               /* prices are refreshed from the statistics every so many bytes */
#ifndef OPT_Depth10
#define   OPT_Depth10          64                      /* levels 10, 11, 12: chain search depth, and the match length */
#define   OPT_Depth11          256                     /* that ends a segment */
#define   OPT_Depth12          4096
#define   OPT_Sufficient10     64
#define   OPT_Sufficient11     256
#define   OPT_Sufficient12     1024
#endif
#define   OPT_Chain_Size       (1 << OffWidth)         /* covers the whole window */

typedef struct {
	int price;                                         /* cost of the path up to here, pending literal-run code included */
	int litLen;                                        /* literals since the last match on the path */
	Uint32 mLen, mOff;                                 /* match ending here (mLen 0: a literal ends here), raw offset */
	Uint32 rep[OffCasheSize];                          /* offset cache after the path */
} Opt_Node;

typedef struct {
	Uint32 len, off;
} Opt_Cand;

typedef struct {
	Uint32 start, len, off;
} Opt_Path;

typedef struct {
	Uint32 litFreq[N_HufLits], litRunFreq[N_HufLitRun], mchLenFreq[N_HufMchLen], mchOffFreq[MchOffGroup][N_HufMchOffMax];
	int litPrice[N_HufLits], litRunPrice[N_HufLitRun], mchLenPrice[N_HufMchLen], mchOffPrice[MchOffGroup][N_HufMchOffMax];
} Opt_Stats;

typedef struct {
	int headA[1 << 14], headB[1 << 15];                /* 3-byte and 4-byte hash heads */
	Uint32 chainA[OPT_Chain_Size], chainB[OPT_Chain_Size];
	Uint32 next;                                       /* next input position to insert */
} Opt_Finder;

ForceInlineTemplate int LitRun_Symbol(Uint32 litRun, int* extraBits)
{
	const Uint32 sym = Value_Code(litRun);
	*extraBits = ValueBits[sym];
	return (int)sym;
}

ForceInlineTemplate int MchLen_Symbol(Uint32 len, int* extraBits)
{
	const Uint32 sym = Value_Code(len);
	*extraBits = ValueBits[sym];
	return (int)sym - MinMatchLen;
}

/* v: cache slot (0-3) or offset + OffCasheSize - 1 */
ForceInlineTemplate int Offset_Symbol(Uint32 v, int* extraBits)
{
	if (v < OffCasheSize) { *extraBits = 0; return (int)v; }
	const int msb = High_Bit32(v);
	*extraBits = msb - 1;
	return Offset_Huffman_Index(v, msb);
}

static void Opt_Set_Prices(const Uint32* freq, int n, int* price)
{
	Uint64 total = 0;
	for (int i = 0; i < n; i++) total += freq[i] + 1;
	for (int i = 0; i < n; i++) price[i] = (int)(OPT_Unit * log2((double)total / (freq[i] + 1)));
}

static void Opt_Update_Prices(Opt_Stats* st)
{
	Opt_Set_Prices(st->litFreq, N_HufLits, st->litPrice);
	Opt_Set_Prices(st->litRunFreq, N_HufLitRun, st->litRunPrice);
	Opt_Set_Prices(st->mchLenFreq, N_HufMchLen, st->mchLenPrice);
	for (int g = 0; g < MchOffGroup; g++)
		Opt_Set_Prices(st->mchOffFreq[g], N_HufMchOff[g], st->mchOffPrice[g]);
}

static void Opt_Init_Stats(Opt_Stats* st, const Uint8* source, Uint32 srcSize)
{
	memset(st, 0, sizeof(*st));
	for (Uint32 i = 0; i < srcSize; i++) st->litFreq[source[i]]++;
	for (int i = 0; i < N_HufLitRun; i++) st->litRunFreq[i] = 64 >> min(i, 6);
	/* a weak prior over lengths, halving every 8 symbols: a steeper one makes short lengths cheap enough that the
	   parser splits long matches, and the counts of that parse keep them cheap */
	for (int i = 0; i < N_HufMchLen; i++) st->mchLenFreq[i] = 64 >> min(i / 8, 6);
	for (int g = 0; g < MchOffGroup; g++) {
		for (int i = 0; i < (int)N_HufMchOff[g]; i++) st->mchOffFreq[g][i] = 2;
		st->mchOffFreq[g][0] = 16; st->mchOffFreq[g][1] = 8; st->mchOffFreq[g][2] = 4; st->mchOffFreq[g][3] = 4;
	}
	Opt_Update_Prices(st);
}

ForceInlineTemplate int LitRun_Price(const Opt_Stats* st, Uint32 litRun)
{
	int extra;
	const int sym = LitRun_Symbol(litRun, &extra);
	return st->litRunPrice[sym] + extra * OPT_Unit;
}

/* price of a match of length len with raw offset off from a path whose offset cache is rep; OPT_Inf if not codable */
ForceInlineTemplate int Match_Price(const Opt_Stats* st, Uint32 len, Uint32 off, const Uint32* rep)
{
	int mlExtra, offExtra;
	const int mlSym = MchLen_Symbol(len, &mlExtra);
	const int group = min(MchOffGroup - 1, mlSym);
	Uint32 v;
	if (off == rep[0]) v = 0;
	else if (off == rep[1]) v = 1;
	else if (off == rep[2]) v = 2;
	else if (off == rep[3]) v = 3;
	else {
		if (off >= OffWindowTable[min(8, len)]) return OPT_Inf;
		v = off + OffCasheSize - 1;
	}
	const int offSym = Offset_Symbol(v, &offExtra);
	return st->mchLenPrice[mlSym] + mlExtra * OPT_Unit + st->mchOffPrice[group][offSym] + offExtra * OPT_Unit;
}

/* Length of the match at distance `offset` from srcIdx, 0 when out of history. Dictionary positions after -16 are not
   used, so that 8-byte reads and match extension stay inside the dictionary. */
ForceInlineTemplate int Repeat_Match_Len(const Uint8* const source, Uint32 srcIdx, Uint32 offset, const int dictSize, const Uint8* const dictEnd,
	const Uint8* const srcLastMatch, const Uint8* const dictLastMatch)
{
	if (offset == 0 || offset > srcIdx + (Uint32)dictSize) return 0;
	const int histIdx = (int)srcIdx - (int)offset;
	if (histIdx < 0 && histIdx > -16) return 0;
	const Uint8* const srcPtr = source + srcIdx;
	const Uint8* const matchPtr = histIdx < 0 ? dictEnd + histIdx : srcPtr - offset;
	const reg_t diff = MemReadARCH(srcPtr) ^ MemReadARCH(matchPtr);
	if (diff) return (int)N_ZeroBytes(diff);
	return REG_SIZE + (int)WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, DICT_LIMIT(matchPtr));
}

#define OPT_CHAIN_INSERT(head, chain, h, i)                                                                            \
	{   const int prev_ = (head)[h], d_ = (i) - prev_;                                                                  \
		(chain)[(Uint32)(i) & (OPT_Chain_Size - 1)] = (prev_ >= -dictSize && d_ > 0 && d_ < OPT_Chain_Size) ? (Uint32)d_ : OPT_Chain_Size; \
		(head)[h] = (i);                                                                                             \
	}

static void Opt_Finder_Init(Opt_Finder* f, const Uint8* dictEnd, const int dictSize)
{
	memset(f->headA, 0x80, sizeof(f->headA));         /* 0x80808080: before any history */
	memset(f->headB, 0x80, sizeof(f->headB));
	f->next = 0;
	/* dictionary positions up to -16 (8-byte reads and match extension stay inside it) */
	for (int i = -dictSize; i <= -16; i++) {
		OPT_CHAIN_INSERT(f->headA, f->chainA, Hash_3B(dictEnd + i) & ((1 << 14) - 1), i);
		OPT_CHAIN_INSERT(f->headB, f->chainB, Hash_4B(dictEnd + i) & ((1 << 15) - 1), i);
	}
}

/* Collects match candidates at currIdx: for each length, the nearest offset found for it (lengths and offsets both
   increase along the list), limited to offsets that the window of the length admits. */
ForceInlineTemplate int Opt_Candidates(Opt_Finder* const f, const Uint8* const source, Uint32 currIdx, const int dictSize,
	const Uint8* const dictEnd, const Uint8* const srcLastMatch, const Uint8* const dictLastMatch, int searchCnt, Opt_Cand* const cand)
{
	const Uint8* const srcPtr = source + currIdx;
	const Uint8* matchPtr;
	int n = 0;
	for (; f->next <= currIdx; f->next++) {
		OPT_CHAIN_INSERT(f->headA, f->chainA, Hash_3B(source + f->next) & ((1 << 14) - 1), (int)f->next);
		OPT_CHAIN_INSERT(f->headB, f->chainB, Hash_4B(source + f->next) & ((1 << 15) - 1), (int)f->next);
	}

	/* length 3: the 3-byte chain within the window of length 3, until a longer match */
	{
		const reg_t currPattern = MemReadARCH(srcPtr);
		Uint32 matchDist = f->chainA[currIdx & (OPT_Chain_Size - 1)];
		int cnt = searchCnt;
		while (matchDist < OffWindowTable[3] && cnt--) {
			const int matchIdx = (int)currIdx - (int)matchDist;
			if (matchIdx < -dictSize) break;
			matchPtr = matchIdx < 0 ? dictEnd + matchIdx : srcPtr - matchDist;
			const reg_t diff = currPattern ^ MemReadARCH(matchPtr);
			const int len = diff ? (int)N_ZeroBytes(diff) : REG_SIZE;
			if (len >= MinMatchLen) {
				if (len == MinMatchLen) { cand[n].len = 3; cand[n].off = matchDist; n++; }
				break;                                         /* a longer match is the 4-byte chain's */
			}
			if (matchIdx < 0) break;
			matchDist += f->chainA[(Uint32)matchIdx & (OPT_Chain_Size - 1)];
		}
	}

	/* lengths 4+: the 4-byte chain, nearest first, recording each longer match */
	{
		const Uint32 currPattern = MemRead4(srcPtr);
		Uint32 matchDist = f->chainB[currIdx & (OPT_Chain_Size - 1)];
		int bestLen = MinMatchLen, cnt = searchCnt;
		while (matchDist < OffWindowTable[8] && cnt--) {
			const int matchIdx = (int)currIdx - (int)matchDist;
			if (matchIdx < -dictSize) break;
			matchPtr = matchIdx < 0 ? dictEnd + matchIdx : srcPtr - matchDist;
			if (currPattern == MemRead4(matchPtr)) {
				const int len = 4 + (int)WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, DICT_LIMIT(matchPtr));
				if (len > bestLen) {
					bestLen = len;
					cand[n].len = min(len, MaxMatchLen); cand[n].off = matchDist; n++;
					if (len >= MaxMatchLen) break;
				}
			}
			if (matchIdx < 0) break;
			matchDist += f->chainB[(Uint32)matchIdx & (OPT_Chain_Size - 1)];
		}
	}
	return n;
}

ForceInlineTemplate void Opt_Relax(Opt_Node* const opt, int* const lastPos, int from, Uint32 len, Uint32 off, int price)
{
	const int to = from + (int)len;
	while (*lastPos < to) opt[++*lastPos].price = OPT_Inf;
	if (price < opt[to].price) {
		Opt_Node* const node = opt + to;
		node->price = price;
		node->litLen = 0;
		node->mLen = len;
		node->mOff = off;
		const Uint32* const rep = opt[from].rep;
		{                                          /* the offset moves to the front; a new one pushes out the oldest */
			const int k = off == rep[0] ? 0 : off == rep[1] ? 1 : off == rep[2] ? 2 : off == rep[3] ? 3 : 4;
			node->rep[3] = k >= 3 ? rep[2] : rep[3];
			node->rep[2] = k >= 2 ? rep[1] : rep[2];
			node->rep[1] = k >= 1 ? rep[0] : rep[1];
			node->rep[0] = off;
		}
	}
}

/* adds a chosen sequence to the statistics; lastOffset follows the encoder's offset cache */
static void Opt_Count(Opt_Stats* st, const Uint8* source, Uint32 anchor, const Opt_Path* m, Uint32* lastOffset)
{
	int extra;
	for (Uint32 q = anchor; q < m->start; q++) st->litFreq[source[q]]++;
	st->litRunFreq[LitRun_Symbol(m->start - anchor, &extra)]++;
	const int mlSym = MchLen_Symbol(m->len, &extra);
	st->mchLenFreq[mlSym]++;
	st->mchOffFreq[min(MchOffGroup - 1, mlSym)][Offset_Symbol(Offset_Cashe(lastOffset, m->off), &extra)]++;
}

/* one parse of the whole block; returns the number of matches written to path (in order) */
static int Opt_Parse(const Uint8* const source, const Uint32 srcSize, const int dictSize, const Uint8* const dictEnd,
	Opt_Stats* const st, int searchCnt, int sufficientLen, Opt_Finder* const finder, Opt_Node* const opt,
	Opt_Cand* const cand, Opt_Path* const segPath, Opt_Path* const path)
{
	const Uint8* const srcLastMatch = source + srcSize - REG_SIZE * 2;
	const Uint32 lastMatchIdx = srcSize > REG_SIZE * 2 ? srcSize - REG_SIZE * 2 : 0;
	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	Uint32 lastOffset[OffCasheSize], countOffset[OffCasheSize];
	memset(lastOffset, 0xFF, sizeof(lastOffset));            /* unset: above every offset */
	memcpy(countOffset, lastOffset, sizeof(lastOffset));
	Opt_Finder_Init(finder, dictEnd, dictSize);
	Uint32 anchor = 0, pos = 0, nextUpdate = OPT_UpdateBytes;
	int nPath = 0;

	while (pos < lastMatchIdx) {
		if (pos >= nextUpdate) { Opt_Update_Prices(st); nextUpdate = pos + OPT_UpdateBytes; }

		/* ---- shortest path over the segment starting at pos */
		int lastPos = 0, endCur = -1;
		opt[0].litLen = (int)(pos - anchor);
		opt[0].price = LitRun_Price(st, opt[0].litLen);
		opt[0].mLen = 0;
		memcpy(opt[0].rep, lastOffset, sizeof(lastOffset));
		const int newRunPrice = LitRun_Price(st, 0);

		for (int cur = 0; ; cur++) {
			if (cur > 0) {                                           /* literal step into cur */
				const Opt_Node* const prev = opt + cur - 1;
				const int litLen = prev->litLen + 1;
				const int price = prev->price + st->litPrice[source[pos + cur - 1]] + LitRun_Price(st, litLen) - LitRun_Price(st, litLen - 1);
				if (cur > lastPos) opt[cur].price = OPT_Inf, lastPos = cur;
				if (price < opt[cur].price) {
					opt[cur].price = price;
					opt[cur].litLen = litLen;
					opt[cur].mLen = 0;
					memcpy(opt[cur].rep, prev->rep, sizeof(prev->rep));
				}
			}
			const Uint32 idx = pos + cur;
			if ((cur > 0 && cur >= lastPos) || cur >= OPT_Num || idx >= lastMatchIdx) { endCur = cur; break; }

			const Opt_Node* const node = opt + cur;
			const int base = node->price + newRunPrice;
			Uint32 longest = 0, longestOff = 0;

			for (int k = 0; k < OffCasheSize; k++) {                 /* repeat offsets */
				const Uint32 off = node->rep[k];
				if (k && (off == node->rep[0] || (k > 1 && off == node->rep[1]) || (k > 2 && off == node->rep[2]))) continue;
				Uint32 len = (Uint32)Repeat_Match_Len(source, idx, off, dictSize, dictEnd, srcLastMatch, dictLastMatch);
				if (len < MinMatchLen) continue;
				len = min(len, MaxMatchLen);
				if (len > longest) { longest = len; longestOff = off; }
				if (len >= (Uint32)sufficientLen) continue;
				for (Uint32 l = MinMatchLen; l <= len; l++)
					Opt_Relax(opt, &lastPos, cur, l, off, base + Match_Price(st, l, off, node->rep));
			}

			const int nCand = Opt_Candidates(finder, source, idx, dictSize, dictEnd, srcLastMatch, dictLastMatch, searchCnt, cand);
			if (nCand && cand[nCand - 1].len > longest) { longest = cand[nCand - 1].len; longestOff = cand[nCand - 1].off; }
			if (longest >= (Uint32)sufficientLen) {                  /* take it and end the segment */
				while (lastPos > cur) opt[lastPos--].price = OPT_Inf;
				lastPos = cur;
				Opt_Relax(opt, &lastPos, cur, longest, longestOff, 0);
				endCur = lastPos;
				break;
			}
			Uint32 l = MinMatchLen;
			for (int c = 0; c < nCand; c++)
				for (; l <= cand[c].len; l++) {
					const int mp = Match_Price(st, l, cand[c].off, node->rep);
					if (mp < OPT_Inf) Opt_Relax(opt, &lastPos, cur, l, cand[c].off, base + mp);
				}
		}

		/* ---- back-trace the segment's matches, then append them in order */
		int nSeg = 0;
		for (int cur = endCur; cur > 0; ) {
			if (opt[cur].mLen) {
				segPath[nSeg].start = pos + (Uint32)cur - opt[cur].mLen;
				segPath[nSeg].len = opt[cur].mLen;
				segPath[nSeg].off = opt[cur].mOff;
				nSeg++;
				cur -= (int)opt[cur].mLen;
			}
			else cur--;
		}
		while (nSeg--) {
			path[nPath] = segPath[nSeg];
			Offset_Cashe(lastOffset, path[nPath].off);
			Opt_Count(st, source, anchor, path + nPath, countOffset);
			anchor = path[nPath].start + path[nPath].len;
			nPath++;
		}
		pos += endCur > 0 ? (Uint32)endCur : 1;
	}
	return nPath;
}

#define   OPT_NoMemory         0xFFFFFFFFu             /* the optimal parser could not allocate its structures */

/* Optimal parse and encoding of the block: fills wlzSeq and huffmanSet like WLZ2_Compress, and writes the literals */
static Uint32 WLZ2_Compress_Opt(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	WLZ_Set* wlzSeq,
	Uint8** zipLitStream,
	WLZ_Huffman_Set* huffmanSet,
	int maxSearchCnt,
	int sufficientLen)
{
	const int dictSize = wzipStr->dictSize;
	const Uint8* const dictEnd = wzipStr->dictEnd;
	Opt_Stats* const st = (Opt_Stats*)malloc(sizeof(Opt_Stats));
	Opt_Finder* const finder = (Opt_Finder*)malloc(sizeof(Opt_Finder));
	Opt_Node* const opt = (Opt_Node*)malloc((OPT_Num + MaxMatchLen + 2) * sizeof(Opt_Node));
	Opt_Cand* const cand = (Opt_Cand*)malloc((maxSearchCnt + 8) * sizeof(Opt_Cand));
	const int maxPath = (int)(srcSize / MinMatchLen) + 2;
	Opt_Path* const segPath = (Opt_Path*)malloc(((OPT_Num + MaxMatchLen) / MinMatchLen + 2) * sizeof(Opt_Path));
	Opt_Path* const path = (Opt_Path*)malloc(maxPath * sizeof(Opt_Path));
	Uint8* const lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);
	if (!st || !finder || !opt || !cand || !segPath || !path || !lzLitBuffer) {
		free(st); free(finder); free(opt); free(cand); free(segPath); free(path); free(lzLitBuffer);
		return OPT_NoMemory;
	}

	/* two passes: the first learns the statistics that price the second */
	Opt_Init_Stats(st, source, srcSize);
	Opt_Parse(source, srcSize, dictSize, dictEnd, st, maxSearchCnt, sufficientLen, finder, opt, cand, segPath, path);
	Opt_Update_Prices(st);
	const int nPath = Opt_Parse(source, srcSize, dictSize, dictEnd, st, maxSearchCnt, sufficientLen, finder, opt, cand, segPath, path);

	/* ---- encode, exactly as WLZ2_Compress does */
	Huffman_Str litHuf[N_HufLits];
	memset(litHuf, 0, sizeof(litHuf));
	Uint8* zipLitPtr = *zipLitStream + ((srcSize >> 16) ? 4 : 2);    /* reserved for the number of literals */
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* const lzLitEnd = lzLitBuffer + HUF_BlockSize;
	Uint32 nLzLits = 0, anchor = 0, lastOffset[OffCasheSize];
	memset(lastOffset, 0xFF, sizeof(lastOffset));            /* unset: above every offset */
	WLZ_Set* destPtr = wlzSeq;
	int extra;
	for (int i = 0; i <= nPath; i++) {
		const Uint32 start = i < nPath ? path[i].start : srcSize;
		for (Uint32 q = anchor; q < start; q++) {
			*lzLitPtr++ = source[q];
			litHuf[source[q]].freq++;
			if (lzLitPtr == lzLitEnd) {
				zipLitPtr += Huffman_Compress_Block(lzLitBuffer, HUF_BlockSize, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, sizeof(litHuf));
				nLzLits += HUF_BlockSize;
			}
		}
		const Uint32 litRun = start - anchor;
		const int litRunHufIdx = LitRun_Symbol(litRun, &extra);
		destPtr->litRun = litRunHufIdx ^ (litRun & BitMask[extra]) << 8;
		huffmanSet->litRunHuf[litRunHufIdx].freq++;
		if (i == nPath) break;                                       /* terminal record: the last literal run only */

		const Uint32 len = path[i].len;
		const Uint32 v = Offset_Cashe(lastOffset, path[i].off);
		const int mchLenHufIdx = MchLen_Symbol(len, &extra);
		destPtr->mchLen = mchLenHufIdx ^ (len & BitMask[extra]) << 8;
		huffmanSet->mchLenHuf[mchLenHufIdx].freq++;
		const int offsetHufIdx = Offset_Symbol(v, &extra);
		destPtr->mchOff = offsetHufIdx ^ (v & BitMask[extra]) << 8;
		huffmanSet->mchOffHuf[min(MchOffGroup - 1, mchLenHufIdx)][offsetHufIdx].freq++;
		destPtr++;
		anchor = start + len;
	}
	const Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);
	zipLitPtr += Huffman_Compress_Block(lzLitBuffer, lastBufLits, zipLitPtr, litHuf, N_HufLits, CapHufLitBits);
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(*zipLitStream, nLzLits);
	else               MemWriteLE2(*zipLitStream, (Uint16)nLzLits);
	*zipLitStream = zipLitPtr;

	free(st); free(finder); free(opt); free(cand); free(segPath); free(path); free(lzLitBuffer);
	return (Uint32)(destPtr + 1 - wlzSeq);
}

int WZIP_Compress_M(WZIP_State_Str* wzipStr, const void* const source, int srcSize, void* const wzipStream, int wzipCapSize)
{
	Uint8* wzipStreamPtr = (Uint8*)wzipStream;
	Uint32 wlzSeqLen=0;

	WLZ_Huffman_Set huffmanSet;
	memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));

	Uint32 wlzSeqMax = srcSize / 3 + 2;    /* every match covers at least three bytes; plus the terminal record */
	WLZ_Set* const wlzSeq = (WLZ_Set*)malloc(wlzSeqMax * sizeof(WLZ_Set));
	if (NULL == wlzSeq) return 0;
	
	static const int sufficientLen[3] = { OPT_Sufficient10, OPT_Sufficient11, OPT_Sufficient12 };
	if (0 == wzipStr->compressLevel)
		wlzSeqLen = WLZ2_Compress_Fast(wzipStr, (Uint8 *)source, srcSize, wlzSeq, &wzipStreamPtr, &huffmanSet);
	else if (wzipStr->compressLevel >= 10)
		wlzSeqLen = WLZ2_Compress_Opt(wzipStr, (Uint8*)source, srcSize, wlzSeq, &wzipStreamPtr, &huffmanSet, wzipStr->maxSearchCnt, sufficientLen[wzipStr->compressLevel - 10]);
	else
		wlzSeqLen = WLZ2_Compress(wzipStr, (Uint8*)source, srcSize, wlzSeq, &wzipStreamPtr, &huffmanSet, wzipStr->maxSearchCnt);
	if (OPT_NoMemory == wlzSeqLen) { free(wlzSeq); return 0; }

	int wzipSize = (int)(wzipStreamPtr - (Uint8*)wzipStream);               /* length of Huffman literal compression */
	wzipCapSize -= wzipSize;	
	const Uint32 seqSize = wzipCapSize > 0 ? Huffman_Compress_WLZ(wlzSeq, wlzSeq + wlzSeqLen, wzipStreamPtr, wzipCapSize, &huffmanSet) : 0;
	free(wlzSeq);
	return seqSize ? wzipSize + (int)seqSize : 0;       /* 0: the output buffer was too small */
}

WZIP_State_Str* WZIP_New_State_M(int level, const void* dict, int dictSize)
{
	WZIP_State_Str* const wzipStr = (WZIP_State_Str*)calloc(1, sizeof(WZIP_State_Str));
	if (NULL == wzipStr) return NULL;
	wzipStr->compressLevel = level;
	wzipStr->dictSize = dictSize;
	wzipStr->dictEnd = dict ? (Uint8*)dict + dictSize : NULL;
	wzipStr->hash0Mask = BitMask[14];
	wzipStr->hash1Mask = BitMask[15];
	wzipStr->hash2Mask = 0;
	wzipStr->chain2Mask = 0;
	if (0 == level) {		
		wzipStr->chain1Mask = 0;		
	}
	else if (level <= 12) {
		static const int optDepth[3] = { OPT_Depth10, OPT_Depth11, OPT_Depth12 };
		if (level >= 10)
			wzipStr->maxSearchCnt = optDepth[level - 10];         /* optimal parsing */
		else if(level<=4)
			wzipStr->maxSearchCnt = 1<<(2*level);
		else wzipStr->maxSearchCnt = 1 << (4+level);
		wzipStr->chain1Mask = BitMask[OffWidth4];
	}
	else {
		fprintf(stderr, "compression level must be in [0, 12]\n");
		WZIP_Free_State(wzipStr);
		return NULL;
	}

	/* zeroed: an entry not yet written then names position 0, always in the history (a stale one from reused memory
	   could name a position near the dictionary's end, whose compare reads past it) */
	wzipStr->hash0Table = calloc((size_t)wzipStr->hash0Mask + 1, sizeof(Sint16));
	wzipStr->hash1Table = calloc((size_t)wzipStr->hash1Mask + 1, sizeof(Sint16));
	wzipStr->hash2Table = NULL;

	if (0 == wzipStr->chain1Mask) wzipStr->chain1Table = NULL;
	else wzipStr->chain1Table = calloc((size_t)wzipStr->chain1Mask + 1, sizeof(Uint16));
	
	if (NULL == wzipStr->hash0Table || NULL == wzipStr->hash1Table || (wzipStr->chain1Mask && NULL == wzipStr->chain1Table)) {
		WZIP_Free_State(wzipStr);
		return NULL;
	}
	if (dictSize == 0 || dict == NULL)
		return wzipStr;

	// pre-build dictionary 
	Sint16* hash0Table = (Sint16*)wzipStr->hash0Table;
	Sint16* hash1Table = (Sint16*)wzipStr->hash1Table;
	Uint16* chain1Table = (Uint16*)wzipStr->chain1Table;
	const Uint32 chain1Mask = wzipStr->chain1Mask;

	int hashV, dist, matchIdx;
	const Uint8* dictPtr = (Uint8 *)dict;
	/* positions up to -16 only: 8-byte compares and match extension then stay inside the dictionary */
	for (int i = -dictSize; i <= -16; i++, dictPtr++) {
		hashV = WLZ_Hash0(dictPtr) & wzipStr->hash0Mask;
		matchIdx = hash0Table[hashV];
		hash0Table[hashV] = i;

		hashV = WLZ_Hash1(dictPtr) & wzipStr->hash1Mask;
		matchIdx = hash1Table[hashV];
		hash1Table[hashV] = i;
		if (chain1Mask) {
			dist = i - matchIdx;
			chain1Table[(Uint32)i & chain1Mask] = (dist > 0 && dist < chain1Mask) ? dist : chain1Mask;
		}
	}
	return wzipStr;
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ WZIP Decoompression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* a match that starts in the dictionary, `produced` bytes into the output; a corrupt one may run on into the output */
static void Copy_Dict_Match(Uint8* destPtr, const Uint8* dest, const Uint32 produced, const Uint32 offset, const Uint32 len, const Uint8* dictEnd)
{
	const Uint32 inDict = min(len, offset - produced);
	memcpy(destPtr, dictEnd - (offset - produced), inDict);
	for (Uint32 k = inDict; k < len; k++) destPtr[k] = dest[k - inDict];
}

/* Executes one sequence: litRun literals, then a match; returns the new end of the output. The caller has checked
   the literal run, the offset and the length against the literals, the history and the output. */
ForceInlineTemplate Uint8* WLZ_Execute(Uint8* destPtr, const Uint8* litPtr, const Uint32 litRun, const Uint32 matchLen, const Uint32 matchOffset,
	Uint8* const dest, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize)
{
	static const unsigned inc4table[8] = { 0, 0, 0,  1,  0,  4, 4, 4 };     /* 4 % matchOffset */
	static const unsigned inc8table[8] = { 0, 0, 0,  2,  0,  3, 2, 1 };     /* 8 % matchOffset */
	Uint8* destPtrEnd = destPtr + litRun;
	Uint8* matchPtr;

	if (unlikely(matchLen + 16 > (Uint32)(destEnd - destPtrEnd))) {  /* near the end: wild copies would write past it */
		memcpy(destPtr, litPtr, litRun);
		destPtr = destPtrEnd;
		destPtrEnd += matchLen;
		if (dictSize && (Uint32)(destPtr - dest) < matchOffset)
			Copy_Dict_Match(destPtr, dest, (Uint32)(destPtr - dest), matchOffset, matchLen, dictEnd);
		else
			for (Uint8* q = destPtr; q < destPtrEnd; q++) *q = *(q - matchOffset);
		return destPtrEnd;
	}
	MemWildCopy(destPtr, litPtr, destPtrEnd);
	destPtr = destPtrEnd;
	destPtrEnd += matchLen;
	if (dictSize && (Uint32)(destPtr - dest) < matchOffset) {
		/* the match starts in the dictionary, which the compressor never lets it run past: copy exactly, as the
		   dictionary may end at the end of its buffer */
		Copy_Dict_Match(destPtr, dest, (Uint32)(destPtr - dest), matchOffset, matchLen, dictEnd);
	}
	else if (likely(matchOffset >= 16)) {
		matchPtr = destPtr - matchOffset;
		MemWildCopy(destPtr, matchPtr, destPtrEnd);
	}
	else {
		matchPtr = destPtr - matchOffset;
		if (likely(matchOffset < 8)) {
			destPtr[0] = matchPtr[0];
			destPtr[1] = matchPtr[1];
			destPtr[2] = matchPtr[2];
			destPtr[3] = matchPtr[3];
			memcpy(destPtr + 4, matchPtr + inc4table[matchOffset], 4);   /* inc4table equivalent to 4 % matchOffset */
			matchPtr += inc8table[matchOffset];                         /* equivalent to 8 % matchOffset */
		}
		else {
			memcpy(destPtr, matchPtr, 8);
			matchPtr += 8;
		}
		MemWildCopy_Overlap(destPtr + 8, matchPtr, destPtrEnd);
	}
	return destPtrEnd;
}

/* Stream B lies byte-reversed at the end of the block: a little-endian read of the 8 bytes below its read position
   gives the same container a forward big-endian read would */
#define BITStream_Read_FlushBack(bitStream)   {                       \
	bitStream.streamPtr -= (Uint32)(bitStream.nUsedBits >> 3);        \
	bitStream.container = MemReadLE8(bitStream.streamPtr);            \
	bitStream.nUsedBits = bitStream.nUsedBits & 7;                    \
}

/* Builds a sequence decoding table and returns the shift that indexes it. The two-stream decoder reads the match fields
   of a terminal record before it knows the record is terminal, and then drops them, so every table must decode any bits
   harmlessly: an empty table (no symbol used in this block) becomes a one-bit table, and the entries of a one-bit table
   that a one-symbol code leaves unset decode as symbol 0 with no bits */
static Uint32 Build_Safe_DecTableX1(const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Uint8* hufCodeBits, Huffman_DemapX1* hufDemapX1)
{
	hufDemapX1[0].lit = hufDemapX1[1].lit = 0;
	hufDemapX1[0].nbits = hufDemapX1[1].nbits = 0;
	Build_Huffman_DecTableX1(hufCodeSize, maxHufCodeBits, hufCodeBits, hufDemapX1);
	return 64 - (maxHufCodeBits ? maxHufCodeBits : 1);
}

/* stream bounds: the input sits in a buffer with M_PadFront zero bytes before it and M_PadBack after it; each round
   checks that both streams are still within M_StreamSlack of the input, and a round reads fewer than 32 bytes more */
#define M_PadFront      128
#define M_PadBack       1024       /* also covers the code tables, read before any check */
#define M_StreamSlack   32

/* Decodes the sequences; returns the decoded size, or -1 if the stream is corrupt. The literal run and match length
   of every sequence are checked against the output, and its offset against the bytes decoded and the dictionary; the
   streams are kept within the padded input. The literal buffer spans the output: each literal read becomes an output
   byte, so it needs no check. */
ForceInlineTemplate int Decompress_WLZ_Sequence_Body(Uint8* wzipSeqStart, Uint8* lzLitBuffer, Uint8* dest, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize, WLZ_HufWt_Set* hufWtSet, const Uint8* const seqEnd,
	const Uint8* const bufStart)
{
	register Uint32 i, n, lsValue, mchLenHufIdx;
	register Uint32 litRun, matchLen, matchOffset;
	Uint32 offsetLast[OffCasheSize];
	Uint8* destPtr = (Uint8*)dest;


	register Bit_Stream bitStream, bitStreamB;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(wzipSeqStart);
	bitStream.streamPtr = wzipSeqStart;
	bitStreamB.nUsedBits = 0;
	bitStreamB.streamPtr = (Uint8*)seqEnd - 8;
	bitStreamB.container = MemReadLE8(bitStreamB.streamPtr);
	Uint8* lzLitBufPtr = lzLitBuffer;

	Huffman_DemapX1 litRunHufDemapX1[1 << CapHufLitRunBits];
	const Uint32 remMaxLitRunHufWt = Build_Safe_DecTableX1(N_HufLitRun, hufWtSet->maxLitRunHufWt, hufWtSet->litRunHufWt, litRunHufDemapX1);

	Huffman_DemapX1 mchLenHufDemapX1[1 << CapHufMchLenBits];
	const Uint32 remMaxMchLenHufWt = Build_Safe_DecTableX1(N_HufMchLen, hufWtSet->maxMchLenHufWt, hufWtSet->mchLenHufWt, mchLenHufDemapX1);

	Huffman_DemapX1 mchOffHufDemapX1[MchOffGroup][(1 << CapHufMchOffBits)];
	Uint32 remMaxMchOffHufWt[MchOffGroup];

	for (i = 0; i < MchOffGroup; i++) {
		remMaxMchOffHufWt[i] = Build_Safe_DecTableX1(N_HufMchOff[i], hufWtSet->maxMchOffHufWt[i], hufWtSet->mchOffHufWt[i], mchOffHufDemapX1[i]);
	}

#ifdef WZIP_DEBUG
	FILE* fptr;
	fopen_s(&fptr, "WZIP_Seq_Decompress.txt", "w");
#endif
	memset(offsetLast, 0xFF, OffCasheSize * sizeof(int));
	/* two sequences per round, A from the forward stream and B from the backward one: their symbol reads are independent
	   chains of table lookups, which the processor overlaps; the offset cache and the copies still go in order */
	while (1) {
		Uint32 litRunB, mchLenHufIdxB, nB, rawA = 0, rawB = 0, matchLenB;
		if (unlikely(bitStream.streamPtr > seqEnd + M_StreamSlack || bitStreamB.streamPtr < bufStart - M_StreamSlack)) return -1;

		BITStream_Read_ExtHufX1(bitStream, remMaxLitRunHufWt, litRunHufDemapX1, litRun);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxLitRunHufWt, litRunHufDemapX1, litRunB);
		if (unlikely(ValueBits[litRun])) {               /* range symbol: base plus raw low bits */
			BITStream_Read(bitStream, ValueBits[litRun], lsValue);
			litRun = ValueBase[litRun] + lsValue;
			BITStream_Read_Flush(bitStream);
		}
		if (unlikely(ValueBits[litRunB])) {
			BITStream_Read(bitStreamB, ValueBits[litRunB], lsValue);
			litRunB = ValueBase[litRunB] + lsValue;
			BITStream_Read_FlushBack(bitStreamB);
		}

		if (unlikely(litRun >= (Uint32)(destEnd - destPtr))) {   /* A is the terminal record */
			if (litRun > (Uint32)(destEnd - destPtr)) return -1;
			memcpy(destPtr, lzLitBufPtr, litRun);            /* exact: the output buffer may end right here */
			destPtr += litRun;
			break;
		}

		/* B's symbols are read even when B turns out to be the terminal record: they are then unused */
		BITStream_Read_ExtHufX1(bitStream, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdx);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdxB);
		const Uint32 offGroup = min(MchOffGroup - 1, mchLenHufIdx), offGroupB = min(MchOffGroup - 1, mchLenHufIdxB);
		BITStream_Read_ExtHufX1(bitStream, remMaxMchOffHufWt[offGroup], mchOffHufDemapX1[offGroup], n);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxMchOffHufWt[offGroupB], mchOffHufDemapX1[offGroupB], nB);
		if (likely(n >= OffCasheSize)) {
			i = (n >> 1) - 1;
			BITStream_Read(bitStream, i, lsValue);
			rawA = ((2 ^ (n & 1)) << i ^ lsValue) - (OffCasheSize - 1);
		}
		if (likely(nB >= OffCasheSize)) {
			i = (nB >> 1) - 1;
			BITStream_Read(bitStreamB, i, lsValue);
			rawB = ((2 ^ (nB & 1)) << i ^ lsValue) - (OffCasheSize - 1);
		}
		matchLen = mchLenHufIdx + MinMatchLen;
		if (unlikely(ValueBits[matchLen])) {
			BITStream_Read(bitStream, ValueBits[matchLen], lsValue);
			matchLen = ValueBase[matchLen] + lsValue;
		}
		matchLenB = mchLenHufIdxB + MinMatchLen;
		if (unlikely(ValueBits[matchLenB])) {
			BITStream_Read(bitStreamB, ValueBits[matchLenB], lsValue);
			matchLenB = ValueBase[matchLenB] + lsValue;
		}
		BITStream_Read_Flush(bitStream);
		BITStream_Read_FlushBack(bitStreamB);

		/* A */
		if (likely(n >= OffCasheSize)) {
			matchOffset = rawA;
			offsetLast[3] = offsetLast[2];                       /* explicit shifts: a memmove() here becomes a library call */
			offsetLast[2] = offsetLast[1];
			offsetLast[1] = offsetLast[0];
			offsetLast[0] = matchOffset;
		}
		else {                                           /* a hit moves to the front of the cache */
			matchOffset = offsetLast[n];
			if (n) {
				offsetLast[3] = n >= 3 ? offsetLast[2] : offsetLast[3];
				offsetLast[2] = n >= 2 ? offsetLast[1] : offsetLast[2];
				offsetLast[1] = offsetLast[0];
				offsetLast[0] = matchOffset;
			}
		}
		if (unlikely(matchOffset - 1 >= (Uint32)(destPtr - dest) + litRun + (Uint32)dictSize ||
		             matchLen > (Uint32)(destEnd - destPtr) - litRun)) return -1;    /* corrupt: before the history, past the end */
		destPtr = WLZ_Execute(destPtr, lzLitBufPtr, litRun, matchLen, matchOffset, dest, destEnd, dictEnd, dictSize);
		lzLitBufPtr += litRun;

		/* B */
		if (unlikely(litRunB >= (Uint32)(destEnd - destPtr))) {  /* B is the terminal record */
			if (litRunB > (Uint32)(destEnd - destPtr)) return -1;
			memcpy(destPtr, lzLitBufPtr, litRunB);
			destPtr += litRunB;
			break;
		}
		if (likely(nB >= OffCasheSize)) {
			matchOffset = rawB;
			offsetLast[3] = offsetLast[2];
			offsetLast[2] = offsetLast[1];
			offsetLast[1] = offsetLast[0];
			offsetLast[0] = matchOffset;
		}
		else {                                           /* a hit moves to the front of the cache */
			matchOffset = offsetLast[nB];
			if (nB) {
				offsetLast[3] = nB >= 3 ? offsetLast[2] : offsetLast[3];
				offsetLast[2] = nB >= 2 ? offsetLast[1] : offsetLast[2];
				offsetLast[1] = offsetLast[0];
				offsetLast[0] = matchOffset;
			}
		}
		if (unlikely(matchOffset - 1 >= (Uint32)(destPtr - dest) + litRunB + (Uint32)dictSize ||
		             matchLenB > (Uint32)(destEnd - destPtr) - litRunB)) return -1;
		destPtr = WLZ_Execute(destPtr, lzLitBufPtr, litRunB, matchLenB, matchOffset, dest, destEnd, dictEnd, dictSize);
		lzLitBufPtr += litRunB;
	}

#ifdef WZIP_DEBUG
	fclose(fptr);
#endif

	return (int)(destPtr - (Uint8*)dest);
}

#define SEQ_DEC_PARAMS  Uint8* wzipSeqStart, Uint8* lzLitBuffer, Uint8* dest, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize,  \
    WLZ_HufWt_Set* hufWtSet, const Uint8* const seqEnd, const Uint8* const bufStart
#define SEQ_DEC_ARGS    wzipSeqStart, lzLitBuffer, dest, destEnd, dictEnd, dictSize, hufWtSet, seqEnd, bufStart

#define DECOMPRESS_SEQUENCE_GEN(fun)                                                                                                                  \
    static int fun(SEQ_DEC_PARAMS)                                                                                                                  \
    {                                                                                                                                               \
        return fun##_Body(SEQ_DEC_ARGS);                                                                                                            \
    }


DECOMPRESS_SEQUENCE_GEN(Decompress_WLZ_Sequence)

/* The same decoder compiled for BMI2, whose shifts by a register count take one micro-op and leave the flags alone;
   chosen at run time, as zstd does. */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#  define WZIP_DYNAMIC_BMI2 1
static __attribute__((target("bmi,bmi2,lzcnt")))
int Decompress_WLZ_Sequence_Bmi2(SEQ_DEC_PARAMS)
{
	return Decompress_WLZ_Sequence_Body(SEQ_DEC_ARGS);
}
static int CPU_Has_Bmi2(void)
{
	static int has = -1;
	if (has < 0) {
		__builtin_cpu_init();
		has = __builtin_cpu_supports("bmi2") != 0;
	}
	return has;
}
#else
#  define WZIP_DYNAMIC_BMI2 0
#endif


/* srcSize must be exact: the second sequence stream is read backward from the end of the block. Returns destSize, or
   0 if the stream is corrupt; no read or write leaves source[0, srcSize), dest[0, destSize) or the dictionary. */
int WZIP_Decompress_M(
	const void* const source, int const srcSize,
	void* const dest, int const destSize,
	void* const dict, int const dictSize)
{
	int i, maxBits;
	Uint32 nLzLits;
	const int histSize = dict && dictSize > 0 ? dictSize : 0;
	Uint8* const dictEnd = histSize ? (Uint8*)dict + dictSize : NULL;
	const int nLenBytes = (destSize >> 16) ? 4 : 2;
	if (destSize <= 0 || srcSize < nLenBytes + 8) return 0;

	/* the input, small by design (under 32 KB decoded), in a zero-padded copy: see M_PadFront */
	Uint8* const buf = (Uint8*)malloc(M_PadFront + (size_t)srcSize + M_PadBack);
	if (NULL == buf) return 0;
	Uint8* const body = buf + M_PadFront;
	const Uint8* const bodyEnd = body + srcSize;
	memset(buf, 0, M_PadFront);
	memcpy(body, source, srcSize);
	memset(body + srcSize, 0, M_PadBack);
	Uint8* srcPtr = body;

	nLzLits = nLenBytes == 4 ? MemReadLE4(srcPtr) : MemReadLE2(srcPtr);   /* the length of the LZ literal sequence */
	srcPtr += nLenBytes;
	/* sized to the output, zero beyond the literals: every literal consumed becomes an output byte, so no sequence can
	   read past it, even a corrupt one (and 256 more bytes for MemWildCopy) */
	Uint8* lzLitBuffer = nLzLits <= (Uint32)destSize ? (Uint8*)calloc((size_t)destSize + 256, 1) : NULL;
	if (NULL == lzLitBuffer) { free(buf); return 0; }
	int decSize = -1;
	const int zipLitSize = Huffman_Decompress(srcPtr, (int)(bodyEnd - srcPtr), lzLitBuffer, nLzLits, N_HufLits);
	if (zipLitSize < 0) goto _end;
	srcPtr += zipLitSize;

	Bit_Stream bitStream;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(srcPtr);
	bitStream.streamPtr = srcPtr;

	Uint8 wtHufWt[MAX_HufWeight + 3];                /* Second-level Huffman Weight table on the Huffman weights */
	Huffman_DemapX1 wtHufDemapX1[1 << MAX_HufHufWt];     /* Second-level Huffman demapper for Huffman weights */
	WLZ_HufWt_Set hufWtSet;

	const int maxWtHufWt = Huffman_Read_Code(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, wtHufWt);
	if (maxWtHufWt <= 0) goto _end;
	Build_Huffman_DecTableX1(MAX_HufWeight + 3, (Uint32)maxWtHufWt, wtHufWt, wtHufDemapX1);
	if ((maxBits = Huffman_Read_Code_byHuffman(&bitStream, (Uint32)maxWtHufWt, wtHufDemapX1, N_HufLitRun, CapHufLitRunBits, hufWtSet.litRunHufWt)) < 0) goto _end;
	hufWtSet.maxLitRunHufWt = (Uint32)maxBits;
	if ((maxBits = Huffman_Read_Code_byHuffman(&bitStream, (Uint32)maxWtHufWt, wtHufDemapX1, N_HufMchLen, CapHufMchLenBits, hufWtSet.mchLenHufWt)) < 0) goto _end;
	hufWtSet.maxMchLenHufWt = (Uint32)maxBits;
	for (i = 0; i < MchOffGroup; i++) {
		if ((maxBits = Huffman_Read_Code_byHuffman(&bitStream, (Uint32)maxWtHufWt, wtHufDemapX1, N_HufMchOff[i], CapHufMchOffBits, hufWtSet.mchOffHufWt[i])) < 0) goto _end;
		hufWtSet.maxMchOffHufWt[i] = (Uint32)maxBits;
	}
	BITStream_Read_FlushEnd(bitStream);
	if (bitStream.streamPtr > bodyEnd) goto _end;

	Uint8* const destEnd = (Uint8*)dest + destSize;
#if WZIP_DYNAMIC_BMI2
	if (CPU_Has_Bmi2())
		decSize = Decompress_WLZ_Sequence_Bmi2(bitStream.streamPtr, lzLitBuffer, dest, destEnd, dictEnd, histSize, &hufWtSet, bodyEnd, body);
	else
#endif
	decSize = Decompress_WLZ_Sequence(bitStream.streamPtr, lzLitBuffer, dest, destEnd, dictEnd, histSize, &hufWtSet, bodyEnd, body);

_end:
	free(lzLitBuffer);
	free(buf);
	if (destSize == decSize) return decSize;
	else return 0;
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trusted mode (opt-in) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* The decoder the paper measured, unchanged: no checks, for input known to come unmodified from WZIP's encoder;
   the input must stay readable WZIP_TRUSTED_SRC_PAD bytes past its end. A damaged stream can make it read or write
   out of bounds. */

ForceInlineTemplate int Decompress_WLZ_Sequence_Trusted_Body(Uint8* wzipSeqStart, Uint8* lzLitBuffer, Uint8* dest, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize, WLZ_HufWt_Set* hufWtSet, const Uint8* const seqEnd)
{
	register Uint32 i, n, lsValue, mchLenHufIdx;
	register Uint32 litRun, matchLen, matchOffset;
	Uint32 offsetLast[OffCasheSize];
	Uint8* destPtr = (Uint8*)dest;


	register Bit_Stream bitStream, bitStreamB;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(wzipSeqStart);
	bitStream.streamPtr = wzipSeqStart;
	bitStreamB.nUsedBits = 0;
	bitStreamB.streamPtr = (Uint8*)seqEnd - 8;
	bitStreamB.container = MemReadLE8(bitStreamB.streamPtr);
	Uint8* lzLitBufPtr = lzLitBuffer;

	Huffman_DemapX1 litRunHufDemapX1[1 << CapHufLitRunBits];
	const Uint32 remMaxLitRunHufWt = Build_Safe_DecTableX1(N_HufLitRun, hufWtSet->maxLitRunHufWt, hufWtSet->litRunHufWt, litRunHufDemapX1);

	Huffman_DemapX1 mchLenHufDemapX1[1 << CapHufMchLenBits];
	const Uint32 remMaxMchLenHufWt = Build_Safe_DecTableX1(N_HufMchLen, hufWtSet->maxMchLenHufWt, hufWtSet->mchLenHufWt, mchLenHufDemapX1);

	Huffman_DemapX1 mchOffHufDemapX1[MchOffGroup][(1 << CapHufMchOffBits)];
	Uint32 remMaxMchOffHufWt[MchOffGroup];

	for (i = 0; i < MchOffGroup; i++) {
		remMaxMchOffHufWt[i] = Build_Safe_DecTableX1(N_HufMchOff[i], hufWtSet->maxMchOffHufWt[i], hufWtSet->mchOffHufWt[i], mchOffHufDemapX1[i]);
	}

#ifdef WZIP_DEBUG
	FILE* fptr;
	fopen_s(&fptr, "WZIP_Seq_Decompress.txt", "w");
#endif
	memset(offsetLast, 0xFF, OffCasheSize * sizeof(int));
	/* two sequences per round, A from the forward stream and B from the backward one: their symbol reads are independent
	   chains of table lookups, which the processor overlaps; the offset cache and the copies still go in order */
	while (1) {
		Uint32 litRunB, mchLenHufIdxB, nB, rawA = 0, rawB = 0, matchLenB;

		BITStream_Read_ExtHufX1(bitStream, remMaxLitRunHufWt, litRunHufDemapX1, litRun);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxLitRunHufWt, litRunHufDemapX1, litRunB);
		if (unlikely(ValueBits[litRun])) {               /* range symbol: base plus raw low bits */
			BITStream_Read(bitStream, ValueBits[litRun], lsValue);
			litRun = ValueBase[litRun] + lsValue;
			BITStream_Read_Flush(bitStream);
		}
		if (unlikely(ValueBits[litRunB])) {
			BITStream_Read(bitStreamB, ValueBits[litRunB], lsValue);
			litRunB = ValueBase[litRunB] + lsValue;
			BITStream_Read_FlushBack(bitStreamB);
		}

		if (unlikely(destPtr + litRun >= destEnd)) {     /* A is the terminal record */
			memcpy(destPtr, lzLitBufPtr, litRun);            /* exact: the output buffer may end right here */
			destPtr += litRun;
			break;
		}

		/* B's symbols are read even when B turns out to be the terminal record: they are then unused */
		BITStream_Read_ExtHufX1(bitStream, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdx);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdxB);
		const Uint32 offGroup = min(MchOffGroup - 1, mchLenHufIdx), offGroupB = min(MchOffGroup - 1, mchLenHufIdxB);
		BITStream_Read_ExtHufX1(bitStream, remMaxMchOffHufWt[offGroup], mchOffHufDemapX1[offGroup], n);
		BITStream_Read_ExtHufX1(bitStreamB, remMaxMchOffHufWt[offGroupB], mchOffHufDemapX1[offGroupB], nB);
		if (likely(n >= OffCasheSize)) {
			i = (n >> 1) - 1;
			BITStream_Read(bitStream, i, lsValue);
			rawA = ((2 ^ (n & 1)) << i ^ lsValue) - (OffCasheSize - 1);
		}
		if (likely(nB >= OffCasheSize)) {
			i = (nB >> 1) - 1;
			BITStream_Read(bitStreamB, i, lsValue);
			rawB = ((2 ^ (nB & 1)) << i ^ lsValue) - (OffCasheSize - 1);
		}
		matchLen = mchLenHufIdx + MinMatchLen;
		if (unlikely(ValueBits[matchLen])) {
			BITStream_Read(bitStream, ValueBits[matchLen], lsValue);
			matchLen = ValueBase[matchLen] + lsValue;
		}
		matchLenB = mchLenHufIdxB + MinMatchLen;
		if (unlikely(ValueBits[matchLenB])) {
			BITStream_Read(bitStreamB, ValueBits[matchLenB], lsValue);
			matchLenB = ValueBase[matchLenB] + lsValue;
		}
		BITStream_Read_Flush(bitStream);
		BITStream_Read_FlushBack(bitStreamB);

		/* A */
		if (likely(n >= OffCasheSize)) {
			matchOffset = rawA;
			offsetLast[3] = offsetLast[2];                       /* explicit shifts: a memmove() here becomes a library call */
			offsetLast[2] = offsetLast[1];
			offsetLast[1] = offsetLast[0];
			offsetLast[0] = matchOffset;
		}
		else {                                           /* a hit moves to the front of the cache */
			matchOffset = offsetLast[n];
			if (n) {
				offsetLast[3] = n >= 3 ? offsetLast[2] : offsetLast[3];
				offsetLast[2] = n >= 2 ? offsetLast[1] : offsetLast[2];
				offsetLast[1] = offsetLast[0];
				offsetLast[0] = matchOffset;
			}
		}
		destPtr = WLZ_Execute(destPtr, lzLitBufPtr, litRun, matchLen, matchOffset, dest, destEnd, dictEnd, dictSize);
		lzLitBufPtr += litRun;

		/* B */
		if (unlikely(destPtr + litRunB >= destEnd)) {    /* B is the terminal record */
			memcpy(destPtr, lzLitBufPtr, litRunB);
			destPtr += litRunB;
			break;
		}
		if (likely(nB >= OffCasheSize)) {
			matchOffset = rawB;
			offsetLast[3] = offsetLast[2];
			offsetLast[2] = offsetLast[1];
			offsetLast[1] = offsetLast[0];
			offsetLast[0] = matchOffset;
		}
		else {                                           /* a hit moves to the front of the cache */
			matchOffset = offsetLast[nB];
			if (nB) {
				offsetLast[3] = nB >= 3 ? offsetLast[2] : offsetLast[3];
				offsetLast[2] = nB >= 2 ? offsetLast[1] : offsetLast[2];
				offsetLast[1] = offsetLast[0];
				offsetLast[0] = matchOffset;
			}
		}
		destPtr = WLZ_Execute(destPtr, lzLitBufPtr, litRunB, matchLenB, matchOffset, dest, destEnd, dictEnd, dictSize);
		lzLitBufPtr += litRunB;
	}

#ifdef WZIP_DEBUG
	fclose(fptr);
#endif

	return (Uint32)(destPtr - (Uint8*)dest);
}


#define SEQ_TRUSTED_PARAMS  Uint8* wzipSeqStart, Uint8* lzLitBuffer, Uint8* dest, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize, \
    WLZ_HufWt_Set* hufWtSet, const Uint8* const seqEnd
static int Decompress_WLZ_Sequence_Trusted(SEQ_TRUSTED_PARAMS)
{
	return Decompress_WLZ_Sequence_Trusted_Body(wzipSeqStart, lzLitBuffer, dest, destEnd, dictEnd, dictSize, hufWtSet, seqEnd);
}
#if WZIP_DYNAMIC_BMI2
static __attribute__((target("bmi,bmi2,lzcnt")))
int Decompress_WLZ_Sequence_Trusted_Bmi2(SEQ_TRUSTED_PARAMS)
{
	return Decompress_WLZ_Sequence_Trusted_Body(wzipSeqStart, lzLitBuffer, dest, destEnd, dictEnd, dictSize, hufWtSet, seqEnd);
}
#endif

/* srcSize must be exact: the second sequence stream is read backward from the end of the block */
int WZIP_Decompress_M_Trusted(
	const void* const source, int const srcSize,
	void* const dest, int const destSize,
	void* const dict, int const dictSize)
{
	int i;
	Uint32 nLzLits, zipLitSize;
	Uint8* srcPtr = (Uint8*)source;
	Uint8* const dictEnd = dict ? (Uint8*)dict + dictSize : NULL;

	if ( destSize >> 16 ) {                            /* Read the length of LZ literal sequence */
		nLzLits = MemReadLE4(srcPtr);
		srcPtr += 4;
	}
	else {
		nLzLits = MemReadLE2(srcPtr);
		srcPtr += 2;
	}
	Uint8* lzLitBuffer = (Uint8*)malloc(nLzLits + 256);                 /* leave margin to allow for overwrite by MemWildCopy */
	zipLitSize = Huffman_Decompress_Trusted(srcPtr, lzLitBuffer, nLzLits, N_HufLits);
	srcPtr += zipLitSize;

	Bit_Stream bitStream;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(srcPtr);
	bitStream.streamPtr = srcPtr;

	Uint8 wtHufWt[MAX_HufWeight + 3];                /* Second-level Huffman Weight table on the Huffman weights */
	Uint32 maxWtHufWt;
	Huffman_DemapX1 wtHufDemapX1[1 << MAX_HufHufWt];     /* Second-level Huffman demapper for Huffman weights */
	WLZ_HufWt_Set hufWtSet;

	maxWtHufWt = Read_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, wtHufWt);
	Build_Huffman_DecTableX1(MAX_HufWeight + 3, maxWtHufWt, wtHufWt, wtHufDemapX1);
	hufWtSet.maxLitRunHufWt = Read_Huffman_Header_byHuffman(&bitStream, maxWtHufWt, wtHufDemapX1, N_HufLitRun, hufWtSet.litRunHufWt);
	hufWtSet.maxMchLenHufWt = Read_Huffman_Header_byHuffman(&bitStream, maxWtHufWt, wtHufDemapX1, N_HufMchLen, hufWtSet.mchLenHufWt);
	for (i = 0; i < MchOffGroup; i++)
		hufWtSet.maxMchOffHufWt[i] = Read_Huffman_Header_byHuffman(&bitStream, maxWtHufWt, wtHufDemapX1, N_HufMchOff[i], hufWtSet.mchOffHufWt[i]);
	BITStream_Read_FlushEnd(bitStream);

	Uint8* const destEnd = (Uint8*)dest + destSize;
	int decSize;
#if WZIP_DYNAMIC_BMI2
	if (CPU_Has_Bmi2())
		decSize = Decompress_WLZ_Sequence_Trusted_Bmi2(bitStream.streamPtr, lzLitBuffer, dest, destEnd, dictEnd, dictSize, &hufWtSet, (const Uint8*)source + srcSize);
	else
#endif
	decSize = Decompress_WLZ_Sequence_Trusted(bitStream.streamPtr, lzLitBuffer, dest, destEnd, dictEnd, dictSize, &hufWtSet, (const Uint8*)source + srcSize);

	free(lzLitBuffer);

	if (destSize == decSize) return decSize;
	else return 0;
}
