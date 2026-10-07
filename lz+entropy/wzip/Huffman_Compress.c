/*
 * WZIP - Huffman encoding of literal blocks and sequence streams
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

/*MSB first packing of a multi-bit symbol. Herein we assume symBits is less than 32.
 sym: Symbol to be packed
 symBits: number of bits representing symbol
 The function packs MSB first so that Huffman Codes can retain their prefix property
*/

#include <string.h>     /* memcpy, memset */
#include <stdio.h>      /* fprintf (debug) */

#include "BitStream_Huffman.h"
#include "Memry.h"

int struct_cmp(const void* a, const void* b)
{
	Huffman_Str* ia = (Huffman_Str*)a;
	Huffman_Str* ib = (Huffman_Str*)b;
	return ia->freq - ib->freq;
}

/* Compute Huffman minimum reduancy and the associated bit lengths */
void Calculate_Minimum_Redundancy(Huffman_Str* A, int nLits)
{
	int root, leaf, next, avbl, used, dpth;

	A[0].freq += A[1].freq; root = 0; leaf = 2;
	for (next = 1; next < nLits - 1; next++)
	{
		if (leaf >= nLits || A[root].freq < A[leaf].freq) { A[next].freq = A[root].freq; A[root++].freq = next; }
		else A[next].freq = A[leaf++].freq;

		if (leaf >= nLits || (root < next && A[root].freq < A[leaf].freq)) { A[next].freq = A[next].freq + A[root].freq; A[root++].freq = next; }
		else A[next].freq = (A[next].freq + A[leaf++].freq);
	}
	A[nLits - 2].freq = 0; for (next = nLits - 3; next >= 0; next--) A[next].freq = A[A[next].freq].freq + 1;
	avbl = 1; used = dpth = 0; root = nLits - 2; next = nLits - 1;
	while (avbl > 0)
	{
		while (root >= 0 && A[root].freq == (Uint32)dpth) { used++; root--; }
		while (avbl > used) { A[next--].freq = dpth; avbl--; }
		avbl = 2 * used; dpth++; used = 0;
	}
}

static int Enforce_Max_Weight(Huffman_Str* sortHufStr, const int nEffLits, int capLitBits)
{
	assert(nEffLits > 0);
	int maxLitBits = sortHufStr[0].nbits;  /* Note sortHufStr is under descreasing order of nbits */
	//return maxLitBits;

	if (maxLitBits <= capLitBits ) {
		if ( maxLitBits <=9 || sortHufStr[2].nbits == maxLitBits) return maxLitBits;
		else capLitBits = maxLitBits - 1;    /* With small cost, reduce max bit length by 1 to simplify decompression */
	}

	int totalCost = 0;
	const int baseCost = 1 << (maxLitBits - capLitBits);
	int n = 0;

	while (sortHufStr[n].nbits > capLitBits) {
		totalCost += baseCost - (1 << (maxLitBits - sortHufStr[n].nbits));
		sortHufStr[n].nbits = (Uint16)capLitBits;
		n++;
	}  /* n stops at huffNode[n].nbBits <= maxNbBits */
	while (sortHufStr[n].nbits == capLitBits) n++;   /* n end at index of largest symbol using < maxNbBits */

	/* renormalize totalCost */
	totalCost >>= (maxLitBits - capLitBits);  /* note : totalCost is necessarily a multiple of baseCost */

	/* repay normalized cost */
	Uint32 const noSymbol = 0xF0F0F0F0;
	Uint32 rankFirst[MAX_HufWeight + 2];

	/* Get pos of last (smallest) symbol per rank */
	memset(rankFirst, 0xF0, sizeof(rankFirst));
	Uint32 currentNbBits = capLitBits;
	int pos;
	for (pos = n; pos < nEffLits; pos++) {
		if (sortHufStr[pos].nbits >= currentNbBits) continue;
		currentNbBits = sortHufStr[pos].nbits;   /* < capLitBits */
		rankFirst[capLitBits - currentNbBits] = (Uint32)pos;      /* first position of the kind */
	} 

	while (totalCost > 0) {
		int nBitsToDecrease = High_Bit32((Uint32)totalCost) + 1;
		for (; nBitsToDecrease > 1; nBitsToDecrease--) {
			Uint32 const highPos = rankFirst[nBitsToDecrease];
			Uint32 const lowPos = rankFirst[nBitsToDecrease - 1];
			if (highPos == noSymbol) continue;
			if (lowPos == noSymbol) break;
			{   Uint32 const highTotal = sortHufStr[highPos].freq;
			Uint32 const lowTotal = 2 * sortHufStr[lowPos].freq;
			if (highTotal <= lowTotal) break;
			}
		}

		/* only triggered when no more rank 1 symbol left => find closest one (note : there is necessarily at least one !) */
		while ((nBitsToDecrease <= MAX_HufWeight) && (rankFirst[nBitsToDecrease] == noSymbol))
			nBitsToDecrease++;
		totalCost -= 1 << (nBitsToDecrease - 1);
		if (rankFirst[nBitsToDecrease - 1] == noSymbol)
			rankFirst[nBitsToDecrease - 1] = rankFirst[nBitsToDecrease];   /* this rank is no longer empty */
		sortHufStr[rankFirst[nBitsToDecrease]].nbits++;
		if ( (int)rankFirst[nBitsToDecrease] == nEffLits - 1)    /* special case, reached largest symbol */
			rankFirst[nBitsToDecrease] = noSymbol;
		else {
			rankFirst[nBitsToDecrease]++;
			if (sortHufStr[rankFirst[nBitsToDecrease]].nbits != capLitBits - nBitsToDecrease)
				rankFirst[nBitsToDecrease] = noSymbol;   /* this rank is now empty */
		}
	}   /* while (totalCost > 0) */

	if (totalCost < 0) {  /* Sometimes, cost correction overshoot */
		if (rankFirst[1] == noSymbol) {  /* special case : no rank 1 symbol (using capLitBits-1); let's create one from largest rank 0 (using capLitBits) */
			while (sortHufStr[n].nbits == capLitBits) n++;
			sortHufStr[n - 1].nbits--;
			assert(n < nEffLits);
			rankFirst[1] = (Uint32)(n - 1);
			totalCost++;
		}
		while (totalCost < 0) {
			rankFirst[1]--;
			sortHufStr[rankFirst[1]].nbits--;
			totalCost++;
		}
	}    

	return capLitBits;
}
	

/*Construct canonical Huffman codes based on the given bit widths
  This function is the code construction for Huffman Tree and length enforcement
  litHuf->nbits: The huffman bit width representation of literal
  litHuf->size: Total number of literals to be constructed
  litHuf->capBits: Maximum length of Huffman Tree to be enforced (i.e. no longer than 7 bits)
  litHuf->code: Huffman code tree
  If we do not enforce Maximum huffman bit width we may need around 10 bytes for header
  Enforcing Header also simplifies tree creation hardware since tree is limited to 7 bits

  After Replacing value Set current position to zero then compare with following
  position to see if it is the natural position
  if not compare with next value until natural position
  is found, then insert value ( first compare 7 with 5, shift 5 by 1 position
  then compare with 6 and shift by 1 position, finally compare with 8, then
  insert in current position for natural order

  Bits     1    2    3    4    5    7    7    7    7
  Map      4    3    0    1    2    5    7    6    8

  Bits     1    2    3    4    5    7    7    7    7
  Map      4    3    0    1    2    5    6    7    8
  Ovrflw   64 + 32 + 16 + 8  + 4  + 1  + 1  + 1  + 1 = 128

  Code Construction
  Map represents Address into Table storing the HuffCode
  Bits     1    2    3    4       5        7       7        7        7
  Map      4    3    0    1       2        5       6        7        8
 Code     0    10   110  1110    11110    1111100 1111101  1111110  1111111*/

/*It constructs Huffman tree subject to a maximum tree depth
  It returns the number of total compressed bytes through Huffman encoding (excluding Huffman tree)
 This is the wrapper function to call the previous functions to generate the Huffman Tree */
int Build_Huffman_Table(Huffman_Str* hufStr, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str *litHuf)
{
	Uint16 lit;
	int i, n, nEffLits;      /* number of effective literals */
	Uint32 nbits, minIdx;
	Huffman_Str sortHufStr[1024];                /* alphabets up to 1024 symbols (WZIP_L's joint symbol) */
	memset(litHuf, 0, hufCodeSize * sizeof(HufCode_Str));

	nEffLits = 0;                    /* eliminate zero entries */
	for (i = 0; i < (int)hufCodeSize; i++) {
		if (hufStr[i].freq) {
			sortHufStr[nEffLits].freq = hufStr[i].freq;
			sortHufStr[nEffLits++].lit = (Uint16)i;
		}
	}
	switch (nEffLits) {
	case 0: 
		return 0;
	case 1: 
		litHuf[sortHufStr[0].lit].nbits = 1;
		litHuf[sortHufStr[0].lit].code = 0;
		return 1;
	case 2: 
		litHuf[sortHufStr[0].lit].nbits = 1;
		litHuf[sortHufStr[0].lit].code = 0;
		litHuf[sortHufStr[1].lit].nbits = 1;
		litHuf[sortHufStr[1].lit].code = 1;
		return 1;
	}

	qsort(sortHufStr, nEffLits, sizeof(Huffman_Str), struct_cmp);     /* sort the effective literals in increasing order of frequency */
	Calculate_Minimum_Redundancy(sortHufStr, nEffLits);
	for (i = 0; i < nEffLits; i++) {
		sortHufStr[i].nbits = (Uint8)sortHufStr[i].freq;
		sortHufStr[i].freq = hufStr[sortHufStr[i].lit].freq;    /* recover the original frequency */
	}
	
	Enforce_Max_Weight(sortHufStr, nEffLits, hufCodeCapBits);

	/* Sequential Bubble sorting for literals among equal nbits */
	for (i = nEffLits - 1; i > 0; i--) {  
		nbits = sortHufStr[i].nbits;
		minIdx = i;
		for (n = i - 1; n >= 0 && sortHufStr[n].nbits == nbits; n--)  
			if (sortHufStr[n].lit < sortHufStr[minIdx].lit) {
				minIdx = n;
			}
		lit = sortHufStr[minIdx].lit;
		sortHufStr[minIdx].lit = sortHufStr[i].lit;
		sortHufStr[i].lit = lit;
	}

	memset(litHuf, 0, hufCodeSize*sizeof(HufCode_Str));
	Uint32 hufCode = 0;
	Uint64 totHufBits = 0;
	for (i = nEffLits-1; i>0; i--) {
		assert(sortHufStr[i-1].nbits >= sortHufStr[i].nbits);
		litHuf[sortHufStr[i].lit].code = (Uint16)hufCode;
		litHuf[sortHufStr[i].lit].nbits = sortHufStr[i].nbits;
		totHufBits += sortHufStr[i].nbits * sortHufStr[i].freq;
		hufCode = (hufCode + 1) << (sortHufStr[i-1].nbits - sortHufStr[i].nbits);
	}
	litHuf[sortHufStr[0].lit].code = (Uint16)hufCode;
	litHuf[sortHufStr[0].lit].nbits = sortHufStr[0].nbits;
	totHufBits += sortHufStr[0].nbits * sortHufStr[0].freq;

	return (int)( (totHufBits+7)>>3 );
}


/* For dynamic Huffman encoding, the head of sequential Huffman bit widths must be packed in front of the Huffman coded data,
   so that cononical Huffman tree can be reconstructed at the decompressor
   litHuf->nbits: The huffman bit width representation of literal
   litHuf->size: Total number of literals to be constructed
   litHuf->capBits: Maximum length of Huffman Tree to be enforced
   */

void Write_Huffman_Header(Bit_Stream* bitStr, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str* hufStr)
{
	Uint32 i, repLen, nBits;
	const Uint32 hufLenBits = N_Bits(hufCodeCapBits);
	register Bit_Stream bitStream = *bitStr;

	BITStream_Write(bitStream, hufStr[0].nbits, hufLenBits);

	for (i = 1; i < hufCodeSize; i++) {
		nBits = hufStr[i].nbits;
		BITStream_Write(bitStream, nBits, hufLenBits);
		if (hufStr[i - 1].nbits == nBits) {
			for (repLen = 0; i < hufCodeSize - 1 && nBits == hufStr[i + 1].nbits; i++)
				repLen++;
			if (repLen < 3) {
				BITStream_Write(bitStream, repLen, 2);
			}
			else {
				BITStream_Write(bitStream, 3, 2);
				repLen -= 3;
				if (repLen < 15) {
					BITStream_Write(bitStream, repLen, 4);
				}
				else {
					BITStream_Write(bitStream, 15, 4);
					repLen -= 15;
					BITStream_Write(bitStream, repLen, 8);
				}
			}
		}
		BITStream_Write_Flush(bitStream);
	}
	*bitStr = bitStream;
}

int Count_Huffman_Weight_Frequency(HufCode_Str* litHuf, const Uint32 hufCodeSize, Huffman_Str* hufWtHufStr, Uint8* hufWtSeq)
{
	Uint32 i, nBits, repZero, repLen;
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	Uint8* hufWtSeqPtr = hufWtSeq;
	if (litHuf[0].nbits) {
		*hufWtSeqPtr++ = (Uint8)litHuf[0].nbits;
		hufWtHufStr[litHuf[0].nbits].freq++;
		i = 1;
	}
	else { i = 0; }

	while(i < hufCodeSize) {
		nBits = litHuf[i].nbits;
		if (0 == nBits) {
			for (repZero = 0; i < hufCodeSize && 0 == litHuf[i].nbits && repZero < 256; i++)    /* longer runs: in chunks */
				repZero++;
			if (repZero == 1) {
				*hufWtSeqPtr++ = 0;
				hufWtHufStr[0].freq++;
			}
			else {
				*hufWtSeqPtr++ = (Uint8)repZeroSym;
				*hufWtSeqPtr++ = (Uint8)(repZero-1); /* this allow repZero=256 to represented by a byte */
				hufWtHufStr[repZeroSym].freq++;
			}
		}
		else if (nBits == litHuf[i - 1].nbits) {
			for (repLen = 1; i < hufCodeSize - 1 && nBits == litHuf[i + 1].nbits && repLen < 255; i++)    /* in chunks */
				repLen++;
			i++;
			if (repLen == 1) {
				*hufWtSeqPtr++ = (Uint8)nBits;
				hufWtHufStr[nBits].freq++;
			}
			else {
				*hufWtSeqPtr++ = (Uint8)repLenSym;
				*hufWtSeqPtr++ = (Uint8)repLen;
				hufWtHufStr[repLenSym].freq++;
			}
		}
		else {
			*hufWtSeqPtr++ = (Uint8)nBits;
			hufWtHufStr[nBits].freq++;
			i++;
		}
	}

	return (Uint32)(hufWtSeqPtr - hufWtSeq);
}

/* the bits Write_Huffman_Header_byHuffman writes for the weight sequence hufWtSeq[0, seqSize) */
Uint32 Huffman_Header_Bits(const HufCode_Str* hufLenHufStr, const Uint8* hufWtSeq, int seqSize)
{
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	const Uint8* p = hufWtSeq;
	const Uint8* const end = hufWtSeq + seqSize;
	Uint32 bits = 0;
	while (p < end) {
		const Uint32 nBits = *p++;
		bits += hufLenHufStr[nBits].nbits;
		if (repZeroSym == nBits) {
			const Uint32 repZero = 1 + *p++;
			bits += 3 + (repZero >= 9 ? 6 + (repZero - 9 >= 63 ? 8 : 0) : 0);
		}
		else if (repLenSym == nBits) {
			const Uint32 repLen = *p++ - 2u;
			bits += 2 + (repLen >= 3 ? 4 + (repLen - 3 >= 15 ? 8 : 0) : 0);
		}
	}
	return bits;
}

/*Write Huffman header which is also Huffman coded.
  It contains two special elements, one is the number of repeated zeros (at least 2).
  The other is the number of repeated non-zero elements (at least 2). */
 void Write_Huffman_Header_byHuffman(Bit_Stream* bitStr, HufCode_Str *hufLenHufStr, Uint8* hufWtSeq, int seqSize)
{
	Uint32 nBits, repZero, repLen;
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	Uint8* hufWtSeqPtr = hufWtSeq;
	const Uint8* hufWtSeqEnd = hufWtSeq + seqSize;
	register Bit_Stream bitStream = *bitStr;
	
	while( hufWtSeqPtr<hufWtSeqEnd ) {
		/*fprintf(stderr, "hufWtSeq[%d] = %d\n", (int)(hufWtSeqPtr - hufWtSeq), *hufWtSeqPtr);
		if (195 == (int)(hufWtSeqPtr - hufWtSeq)) {
			nBits += 0;
		}*/

		nBits = *hufWtSeqPtr++;
		if( nBits<repLenSym ) {
			BITStream_Write(bitStream, hufLenHufStr[nBits].code, hufLenHufStr[nBits].nbits);
		}	
		else if (repZeroSym == nBits) {
			BITStream_Write(bitStream, hufLenHufStr[repZeroSym].code, hufLenHufStr[repZeroSym].nbits);
			repZero = 1 + *hufWtSeqPtr++;
			if (repZero < 9) {
				BITStream_Write(bitStream, repZero - 2, 3);
			}
			else {
				BITStream_Write(bitStream, 7, 3);
				repZero -= 9;
				if (repZero < 63) {
					BITStream_Write(bitStream, repZero, 6);
				}
				else {
					BITStream_Write(bitStream, 63, 6);
					repZero -= 63;
					BITStream_Write(bitStream, repZero, 8);
				}
			}
		}
		else {   /*if (repLenSym == nBits) */
			repLen = *hufWtSeqPtr++;
			repLen -= 2;
			BITStream_Write(bitStream, hufLenHufStr[repLenSym].code, hufLenHufStr[repLenSym].nbits);
			if (repLen < 3) {
				BITStream_Write(bitStream, repLen, 2);
			}
			else {
				BITStream_Write(bitStream, 3, 2);
				repLen -= 3;
				if (repLen >= 15) {
					BITStream_Write(bitStream, 15, 4);
					repLen -= 15;
					BITStream_Write(bitStream, repLen, 8);
				}
				else BITStream_Write(bitStream, repLen, 4);
			}
		}		
		BITStream_Write_Flush(bitStream);
	}
	*bitStr = bitStream;
}

ForceInlineTemplate Uint32 Huffman_Compress1X_Body(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	Uint8* srcPtr = (Uint8*)srcStart;
	Bit_Stream bitStream = { 0, 0, (Uint8*)dest };
	Uint8* srcEnd;
	Uint32 srcModSize;

	srcModSize = srcSize & ~3;
	srcEnd = srcPtr + srcModSize;
	while (srcPtr < srcEnd) {
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write_Flush(bitStream);
	}

	switch (srcSize & 3) {
	case 3:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
	case 2:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
	case 1:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);
	default:
		BITStream_Write_FlushEnd(bitStream);
	}

	return (Uint32)(bitStream.streamPtr - (Uint8*)dest);
}

Uint32 Huffman_Compress1X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	return Huffman_Compress1X_Body(srcStart, srcSize, dest, litHuf);
}


/* It assumes the source literals may be arranged in either direction, 1 being normal starting, -1 being reverse */
Uint32  Huffman_Compress4X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	Uint8* srcPtr = (Uint8*)srcStart;
	Uint8* destPtr = (Uint8*)dest + 6;                      /* the first 6 bytes are reserved to record 3 segment starting positions */

	int srcSegSize = (srcSize>>4) <<2 ;                     /* divide into 4 segments wherein the first three divides 4 */
	Uint32 resSegSize[4];

	resSegSize[0] = Huffman_Compress1X_Kernel(srcPtr,                  srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[0];

	resSegSize[1] = Huffman_Compress1X_Kernel(srcPtr + srcSegSize,      srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[1];
	resSegSize[1] += resSegSize[0];           /* accumulation */

	resSegSize[2] = Huffman_Compress1X_Kernel(srcPtr + 2* srcSegSize,  srcSegSize,  destPtr, litHuf);
	destPtr += resSegSize[2];
	resSegSize[2] += resSegSize[1];

	resSegSize[3] = Huffman_Compress1X_Kernel(srcPtr + 3* srcSegSize,  srcSize-3*srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[3];
	resSegSize[3] += resSegSize[2];

	/* Record segment starting locations */
	destPtr = (Uint8*)dest;
	MemWriteLE2(destPtr, (Uint16)resSegSize[0]);    
	MemWriteLE2(destPtr+2, (Uint16)resSegSize[1]);
	MemWriteLE2(destPtr+4, (Uint16)resSegSize[2]);

	return resSegSize[3] + 6;
}

/* bits of the symbols counted in h under code; all ones if code lacks one of them */
Uint64 Huffman_Code_Bits(const Huffman_Str* h, const HufCode_Str* code, const Uint32 n)
{
	Uint64 bits = 0;
	for (Uint32 k = 0; k < n; k++)
		if (h[k].freq) {
			if (0 == code[k].nbits) return ((Uint64)-1);
			bits += (Uint64)h[k].freq * code[k].nbits;
		}
	return bits;
}

/* Codes a block of literals: stored (type 0), with a code of its own whose lengths the block carries (type 1), or
   with the code of the last type-1 block of the stream, prev (type 2), whichever is smallest; a type-1 block becomes
   prev. litHuf: the block's literal counts. prev may be NULL (no type 2). */
Uint32 Huffman_Compress_Block_Rep(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits,
	Uint32 litHufCapBits, Huffman_Prev* prev)
{
	HufCode_Str litHufCode[MAX_HufSize];
	Huffman_Str hufHufStr[MAX_HufWeight + 3] = { 0 };
	HufCode_Str hufHufCode[MAX_HufWeight + 3];
	Uint8 hufWtSeq[MAX_HufSize];
	Uint8 header[HUF_HeaderBound];
	Uint8* destPtr = (Uint8*)dest;
	Uint32 comprSize;

	const Uint32 estSize = (Uint32)Build_Huffman_Table(litHuf, nLits, litHufCapBits, litHufCode);    /* coded size in bytes, no header */
	const Uint64 prevBits = prev && prev->valid ? Huffman_Code_Bits(litHuf, prev->code, nLits) : ((Uint64)-1);
	Uint32 headerSize = 0;
	if (estSize < srcSize && srcSize >= 64) {            /* the header of a code of its own */
		const Uint32 seqSize = Count_Huffman_Weight_Frequency(litHufCode, nLits, hufHufStr, hufWtSeq);
		Build_Huffman_Table(hufHufStr, MAX_HufWeight + 3, MAX_HufHufWt, hufHufCode);
		Bit_Stream bitStream = { 0, 0, header };
		Write_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, hufHufCode);
		Write_Huffman_Header_byHuffman(&bitStream, hufHufCode, hufWtSeq, seqSize);
		BITStream_Write_FlushEnd(bitStream);
		headerSize = (Uint32)(bitStream.streamPtr - header);
	}
	/* sizes past the type byte and body size: the previous code's, a new one's; the previous code also where a
	   new one would not pay for its header (as on a short last block) */
	const Uint64 repSize = prevBits == ((Uint64)-1) ? ((Uint64)-1) : (prevBits + 7) / 8;
	const Uint64 newSize = headerSize ? (Uint64)headerSize + estSize : ((Uint64)-1);
	const int useRep = repSize <= newSize;
	if ((useRep ? repSize : newSize) >= srcSize) {       /* incompressible scenario */
		*destPtr++ = 0;
		MemWildCopy( destPtr, srcStart, destPtr + srcSize );
		return srcSize + 1;
	}

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Huffman Compression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	*destPtr++ = useRep ? 2 : 1;
	destPtr += 2;                                        /* the body's size */
	if (!useRep) {
		memcpy(destPtr, header, headerSize);
		destPtr += headerSize;
	}
	const HufCode_Str* const code = useRep ? prev->code : litHufCode;
	if (srcSize >= MinStream4XSize) {
		comprSize = Huffman_Compress4X_Kernel(srcStart, srcSize, destPtr, (HufCode_Str*)code);
	}
	else {
		comprSize = Huffman_Compress1X_Kernel(srcStart, srcSize, destPtr, (HufCode_Str*)code);
	}

	if ( comprSize >= (Uint32)(0.99 * srcSize) ) {    /* nearly incompressible, then abandon compression */
		destPtr = (Uint8*)dest;
		*destPtr++ = 0;
		MemWildCopy(destPtr, srcStart, destPtr + srcSize);
		return srcSize + 1;
	}

	assert(comprSize < (1u << 16));                      /* below 0.99 of a block of at most 64K literals */
	MemWriteLE2((Uint8*)dest+1, (Uint16)comprSize);            /* Record the compressed literal size, excluding Huffman header */
	destPtr += comprSize;                              /* it is used to determine the decompressor type  */
	if (!useRep && prev) {
		memcpy(prev->code, litHufCode, nLits * sizeof(HufCode_Str));
		prev->valid = 1;
	}

	return (Uint32)(destPtr - (Uint8*)dest);            /* The second part is the size of the Huffman header */
}

Uint32  Huffman_Compress_Block(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits, Uint32 litHufCapBits)
{
	return Huffman_Compress_Block_Rep(srcStart, srcSize, dest, litHuf, nLits, litHufCapBits, NULL);
}

Uint32 Huffman_Compress(const void* srcStart, Uint32 srcSize, void* dest, Uint32 nLits, Uint32 litHufCapBits)
{
	Huffman_Str litHuf[MAX_HufSize];

	Uint32 i, n;
	Uint32 nFullBlocks = srcSize / HUF_BlockSize;
	Uint32 lastBlockSize = srcSize - HUF_BlockSize * nFullBlocks;
	Uint8* srcPtr = (Uint8*)srcStart;
	Uint8* destPtr = (Uint8*)dest;
	Uint32 blockComprSize = 0;

	for (n = 0; n < nFullBlocks; n++) {
		memset(litHuf, 0, MAX_HufSize * sizeof(Huffman_Str));
		for (i = 0; i < HUF_BlockSize; i++)
			litHuf[*srcPtr++].freq++;
		blockComprSize = Huffman_Compress_Block((Uint8*)srcStart + n * HUF_BlockSize, HUF_BlockSize, destPtr, litHuf, nLits, litHufCapBits);
		destPtr += blockComprSize;
	}
	if (lastBlockSize) {
		memset(litHuf, 0, MAX_HufSize * sizeof(Huffman_Str));
		for (i = 0; i < lastBlockSize; i++)
			litHuf[*srcPtr++].freq++;
		blockComprSize = Huffman_Compress_Block((Uint8*)srcStart + n * HUF_BlockSize, lastBlockSize, destPtr, litHuf, nLits, litHufCapBits);
		destPtr += blockComprSize;
	}

	return (Uint32)(destPtr - (Uint8*)dest);
}