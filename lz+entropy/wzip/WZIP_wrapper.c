/*
 * WZIP - one-call interface (wzip_compress, wzip_decompress) and the window schedule
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

#if defined (__cplusplus)
extern "C" {
#endif

#ifndef WZIP_COMMON_198382716213
#define WZIP_COMMON_198382716213

#include <stddef.h>   /* size_t */
#include <stdio.h>
#include "Memry.h"
#include "BitStream_Huffman.h"
#include "WZIP.h"



void WZIP_State_Load_Dict(WZIP_State_Str* dictStr, WZIP_State_Str* dstStr) {
	if (NULL == dictStr->hash0Table || NULL == dstStr->hash0Table) return;     /* levels 7-13 keep no hash tables */
	int elmSize = dstStr->hash1Mask >>15 ? 4 : 2;
	memcpy(dstStr->hash0Table, dictStr->hash0Table, (1 + dstStr->hash0Mask) * elmSize);
	memcpy(dstStr->hash1Table, dictStr->hash1Table, (1 + dstStr->hash1Mask) * elmSize);
	if (dstStr->hash2Mask)
		memcpy(dstStr->hash2Table, dictStr->hash2Table, (1 + dstStr->hash2Mask) * elmSize);

	if (dstStr->chain1Mask)
		memcpy(dstStr->chain1Table, dictStr->chain1Table, (1 + dstStr->chain1Mask) * sizeof(Uint16));
	

	if (dstStr->chain2Mask) {
		elmSize = dstStr->chain2Mask >>16 ? 4 : 2;
		memcpy(dstStr->chain2Table, dictStr->chain2Table, (1 + dstStr->chain2Mask) * elmSize);
	}
}

/* frees a state of WZIP_New_State_L or WZIP_New_State_M, and its tables */
void WZIP_Free_State(WZIP_State_Str* wzipStr)
{
	if (NULL == wzipStr) return;
	free(wzipStr->hash0Table);
	free(wzipStr->hash1Table);
	free(wzipStr->hash2Table);
	free(wzipStr->chain1Table);
	free(wzipStr->chain2Table);
	free(wzipStr->sched);
	free(wzipStr);
}

unsigned wzip_versionNumber(void) { return WZIP_VERSION_NUMBER; }
const char* wzip_versionString(void) { return WZIP_VERSION_STRING; }

//It returns a safe compression buffer size without overflowing, even if the input data is uncompressible.
int WZIP_Cap_CmprSize(int srcSize) {
	return srcSize + WZIP_MEM_OVERHEAD;
}

void WZIP_Set_OffWidth(int srcSize, int* offWidth) {
	memset(offWidth, 0, 9 * sizeof(int));
	/* from 64 MB on, length 7 stays below the widest window, which covers the input (on enwik8: +0.08% at level 11) */
	if (srcSize >> 28) {
		offWidth[3] = 10;
		offWidth[4] = 15;
		offWidth[5] = 20;
		offWidth[6] = 24;
		offWidth[7] = 25;
		offWidth[8] = 26;
	} else if (srcSize >> 26) {
		offWidth[3] = 11;
		offWidth[4] = 15;
		offWidth[5] = 19;
		offWidth[6] = 23;
		offWidth[7] = 25;
		offWidth[8] = 26;
	} else if (srcSize >> 25) {
		offWidth[3] = 11;
		offWidth[4] = 15;
		offWidth[5] = 19;
		offWidth[6] = 22;
		offWidth[7] = 24;
		offWidth[8] = 24;
	} else if (srcSize >> 24) {
		offWidth[3] = 12;
		offWidth[4] = 16;
		offWidth[5] = 20;
		offWidth[6] = 22;
		offWidth[7] = 23;
		offWidth[8] = 23;
	} else if (srcSize >> 23) {
		offWidth[3] = 12;
		offWidth[4] = 16;
		offWidth[5] = 19;
		offWidth[6] = 22;
		offWidth[7] = 22;
		offWidth[8] = 22;
	} else if (srcSize >> 22) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 18;
		offWidth[6] = 21;
		offWidth[7] = 21;
		offWidth[8] = 21;
	} else if (srcSize >> 21) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 17;
		offWidth[6] = 20;
		offWidth[7] = 20;
		offWidth[8] = 20;
	} else if (srcSize >> 20) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 17;
		offWidth[6] = 19;
		offWidth[7] = 19;
		offWidth[8] = 19;
	} else if (srcSize >> 19) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 17;
		offWidth[6] = 18;
		offWidth[7] = 18;
		offWidth[8] = 18;
	} else if (srcSize >> 18) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 17;
		offWidth[6] = 17;
		offWidth[7] = 17;
		offWidth[8] = 17;
	} else if (srcSize >> 17) {
		offWidth[3] = 12;
		offWidth[4] = 15;
		offWidth[5] = 16;
		offWidth[6] = 16;
		offWidth[7] = 16;
		offWidth[8] = 16;
	} else if (srcSize >> 15) {
		offWidth[3] = 13;
		offWidth[4] = 14;
		offWidth[5] = 15;
		offWidth[6] = 15;
		offWidth[7] = 15;
		offWidth[8] = 15;
	}

	/* The longest match lengths reach back over the whole input (up to 2^27, the widest offset code), and lengths 3-7
	   up to 2^26. A one-shot decoder holds the whole output anyway, so this costs it no memory. */
	if (offWidth[8] > 0) {
		const int top = offWidth[8];
		int w = top;
		while (w < WZIP_MAX_OFF_WIDTH && (1 << w) - 3 < srcSize) w++;         /* (1 << w) - 3: the window of width w */
		for (int i = 3; i <= 8; i++)
			if (offWidth[i] == top) offWidth[i] = i < 8 ? min(w, WZIP_SHORT_OFF_WIDTH) : w;
	}

#ifdef WZIP_TEST_MAX_OFF_WIDTH    /* tests only, another format: narrower windows, so that small inputs wrap the indexes */
	for (int i = 3; i <= 8; i++)
		if (offWidth[i] > WZIP_TEST_MAX_OFF_WIDTH) offWidth[i] = WZIP_TEST_MAX_OFF_WIDTH;
#endif
	for (int i = 8; i > 3; i--)
		assert(offWidth[i] >= offWidth[i - 1]);     /* windows never narrow as the length grows */
}

/* Read the decoded size, stored ahead of the compressed text: 2 bytes below 32 KB, else 4 (the first two carry the
   flag in the top bit and the low 15 bits, the next two the high bits); 0: the data follows uncompressed.
   *srcSize loses the header's length (the payload follows it). */
int WZIP_Read_DecSize(const void* const source, int* srcSize)
{
	const Uint8* const src = (const Uint8*)source;
	if (*srcSize < 2) { *srcSize = 0; return 0; }
	int decSize = MemReadLE2(src);
	*srcSize -= 2;
	if (decSize >> 15) {
		if (*srcSize < 2) { *srcSize = 0; return 0; }
		decSize = (decSize & BitMask[15]) | (MemReadLE2(src + 2) << 15);
		*srcSize -= 2;
	}
	return decSize;
}

/* Stores the input uncompressed: a zero size field, then the bytes. */
static int WZIP_Store(const void* source, int srcSize, Uint8* dst, int dstCap)
{
	if (dstCap < srcSize + 2) return 0;
	MemWriteLE2(dst, 0);
	memcpy(dst + 2, source, srcSize);
	return srcSize + 2;
}

/* Compresses into dst, which holds at least WZIP_Cap_CmprSize(srcSize) bytes; a WZIP_L stream (32 KB and more) with
   the dictionary, if any (a WZIP_M stream never uses one). */
static int WZIP_Compress_Bounded(const void* source, int srcSize, Uint8* dst, int dstCap, int level, int nbWorkers,
	const void* dict, int dictSize)
{
	if (srcSize < 32)                                   /* not worth compressing */
		return WZIP_Store(source, srcSize, dst, dstCap);

	/* the decoded size goes ahead of the compressed text: its length is known before compressing */
	const int hdrSize = (srcSize >> 15) ? 4 : 2;
	WZIP_State_Str* wzipStr;
	int cmprSize = 0;
	if (srcSize >> 15) {
		if (NULL == (wzipStr = WZIP_New_State_L(level, srcSize, dictSize > 0 ? dict : NULL, dictSize > 0 ? dictSize : 0))) return 0;
		WZIP_Set_Workers(wzipStr, nbWorkers);
		cmprSize = WZIP_Compress_L(wzipStr, source, srcSize, dst + hdrSize, dstCap - hdrSize);
		WZIP_Free_State(wzipStr);
	}
	else {
		/* WZIP_M checks its sequence stream only after writing it: give it room for any input, then copy */
		const int mBound = 5 * srcSize + 65536;
		Uint8* const mBuf = (Uint8*)malloc(mBound);
		if (NULL == mBuf) return 0;
		if (NULL == (wzipStr = WZIP_New_State_M(min(level, 12), NULL, 0))) { free(mBuf); return 0; }     /* levels 0-12 */
		cmprSize = WZIP_Compress_M(wzipStr, source, srcSize, mBuf, mBound);
		WZIP_Free_State(wzipStr);
		if (cmprSize > dstCap - hdrSize) cmprSize = 0;
		if (cmprSize) memcpy(dst + hdrSize, mBuf, cmprSize);
		free(mBuf);
	}

	if (0 == cmprSize || srcSize - cmprSize < 32)       /* nothing gained (or the encoder ran out of room): store */
		return WZIP_Store(source, srcSize, dst, dstCap);

	if (srcSize >> 15) {
		MemWriteLE2(dst, (Uint16)((1 << 15) | (srcSize & BitMask[15])));
		MemWriteLE2(dst + 2, (Uint16)(srcSize >> 15));
	}
	else MemWriteLE2(dst, (Uint16)srcSize);
	return cmprSize + hdrSize;
}

/* Compresses srcSize bytes at level 0-13 into wzipStream, whose capacity is *wzipCapSize. Returns the compressed size,
   at most srcSize + 2, or 0 if it does not fit or an argument is invalid. The caller's buffer is never reallocated;
   with less than WZIP_Cap_CmprSize(srcSize) bytes, the stream is built in a temporary buffer and copied if it fits. */
int wzip_compress(const void* source, int srcSize, void* wzipStream, int *wzipCapSize, int level) {
	return wzip_compress_mt(source, srcSize, wzipStream, wzipCapSize, level, 1);
}

/* wzip_compress with up to nbWorkers threads (WZIP.h); the stream is the same */
int wzip_compress_mt(const void* source, int srcSize, void* wzipStream, int *wzipCapSize, int level, int nbWorkers) {
	return wzip_compress_usingDict(source, srcSize, wzipStream, wzipCapSize, level, nbWorkers, NULL, 0);
}

/* wzip_compress_mt with a dictionary for a WZIP_L stream (WZIP.h) */
int wzip_compress_usingDict(const void* source, int srcSize, void* wzipStream, int *wzipCapSize, int level, int nbWorkers,
	const void* dict, int dictSize) {
	if (NULL == source || NULL == wzipStream || NULL == wzipCapSize || srcSize < 0 || (Uint32)srcSize > WZIP_MAX_INPUT_SIZE
	    || level < 0 || level > 13 || dictSize < 0 || (dictSize && NULL == dict))
		return 0;
	const int cap = *wzipCapSize, bound = WZIP_Cap_CmprSize(srcSize);
	if (cap >= bound)
		return WZIP_Compress_Bounded(source, srcSize, (Uint8*)wzipStream, cap, level, nbWorkers, dict, dictSize);

	Uint8* const tmp = (Uint8*)malloc(bound);
	if (NULL == tmp) return 0;
	int cmprSize = WZIP_Compress_Bounded(source, srcSize, tmp, bound, level, nbWorkers, dict, dictSize);
	if (cmprSize > cap) cmprSize = 0;
	if (cmprSize) memcpy(wzipStream, tmp, cmprSize);
	free(tmp);
	return cmprSize;
}

/* Decompresses a stream made by wzip_compress into decmp, whose capacity *decCapSize must hold the decoded size
   (WZIP_Read_DecSize reports it). Returns the decoded size, or 0 if the buffer is too small or the stream is corrupt. */
int wzip_decompress(const void* source, int srcSize, void* decmp, int *decCapSize)
{
	return wzip_decompress_usingDict(source, srcSize, decmp, decCapSize, NULL, 0);
}

/* wzip_decompress of a stream made by wzip_compress_usingDict, with the same dictionary (WZIP.h) */
int wzip_decompress_usingDict(const void* source, int srcSize, void* decmp, int *decCapSize, const void* dict, int dictSize)
{
	if (NULL == source || NULL == decmp || NULL == decCapSize || srcSize < 2 || dictSize < 0 || (dictSize && NULL == dict)) return 0;
	const int totalSize = srcSize;
	const int decSize = WZIP_Read_DecSize(source, &srcSize);
	const Uint8* const srcPtr = (const Uint8*)source + (totalSize - srcSize);     /* past the size header */
	if (0 == decSize) {                                 /* stored uncompressed */
		if (*decCapSize < srcSize) return 0;
		memcpy(decmp, srcPtr, srcSize);
		return srcSize;
	}
	if (*decCapSize < decSize) return 0;

	if (decSize >> 15)
		return WZIP_Decompress_L(srcPtr, srcSize, decmp, decSize, (void*)dict, dictSize);
	else
		return WZIP_Decompress_M(srcPtr, srcSize, decmp, decSize, NULL, 0);
}

/* Trusted mode (WZIP.h): the same routing, to the unchecked decoders */
int wzip_decompress_trusted(const void* source, int srcSize, void* decmp, int *decCapSize)
{
	if (NULL == source || NULL == decmp || NULL == decCapSize || srcSize < 2) return 0;
	const int totalSize = srcSize;
	const int decSize = WZIP_Read_DecSize(source, &srcSize);
	const Uint8* const srcPtr = (const Uint8*)source + (totalSize - srcSize);     /* past the size header */
	if (0 == decSize) {                                 /* stored uncompressed */
		if (*decCapSize < srcSize) return 0;
		memcpy(decmp, srcPtr, srcSize);
		return srcSize;
	}
	if (*decCapSize < decSize) return 0;

	if (decSize >> 15)
		return WZIP_Decompress_L_Trusted(srcPtr, srcSize, decmp, decSize, NULL, 0);
	else
		return WZIP_Decompress_M_Trusted(srcPtr, srcSize, decmp, decSize, NULL, 0);
}

#endif

#if defined (__cplusplus)
}
#endif