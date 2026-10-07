/*
 * WLZ4 - multi-window, LZ4-class compression
 * Copyright (c) 2019-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 * Portions derived from LZ4, Copyright (c) 2011-present Yann Collet (BSD 2-Clause); see NOTICE.
 */

/*
 * ACCELERATION_DEFAULT :
 * Select "acceleration" for WLZ_compress_fast() when parameter value <= 0
 */
#define ACCELERATION_DEFAULT 1
#define WLZ_GAIN_THRESHOLD   -4

#define WLZ_HASH1_MASK   ((1<<WLZ_HASH1BITS)-1) 
#define WLZ_HASH2_MASK   ((1<<WLZ_HASH2BITS)-1) 
#define WLZhc_HASH1_MASK   ((1<<WLZhc_HASH1BITS)-1) 
#define WLZhc_HASH2_MASK   ((1<<WLZhc_HASH2BITS)-1) 

#define WLZ_MATCH1_WINDOW  (1<<8)
#define WLZ_MATCH2_WINDOW  (1<<16)                     /* the hash chains (offsets from 32K on take three bytes) */
#define WLZ_KERNEL_FARLEN  6                           /* the greedy and lazy parsers take three-byte offsets from this length */

 /*-************************************
 *  Error detection
 **************************************/

//#define WLZ_DEBUG

#if defined(WLZ_DEBUG)
#  include <assert.h>
#else
#  ifndef assert
#    define assert(condition) ((void)0)
#  endif
#endif

#define WLZ_STATIC_ASSERT(c)   { enum { WLZ_static_assert = 1/(int)(!!(c)) }; }   /* use after variable declarations */

/*-************************************
*  CPU Feature Detection
**************************************/
/* WLZ_MEMORY_ACCESS
 * By default, access to unaligned memory is controlled by `memcpy()`, which is safe and portable.
 * Unfortunately, on some target/compiler combinations, the generated assembly is sub-optimal.
 * The below switch allow to select different access method for improved performance.
 * Method 0 (default) : use `memcpy()`. Safe and portable.
 * Method 1 : `__packed` statement. It depends on compiler extension (ie, not portable).
 *            This method is safe if your compiler supports it, and *generally* as fast or faster than `memcpy`.
 * Method 2 : direct access. This method is portable but violate C standard.
 *            It can generate buggy code on targets which assembly generation depends on alignment.
 *            But in some circumstances, it's the only known way to get the most performance (ie GCC + ARMv6)
 * See https://fastcompression.blogspot.fr/2015/08/accessing-unaligned-memory.html for details.
 * Prefer these methods in priority order (0 > 1 > 2)
 */
#ifndef WLZ_MEMORY_ACCESS   /* can be defined externally */
#  if defined(__GNUC__) && \
  ( defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) \
  || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6T2__) )
#    define WLZ_MEMORY_ACCESS 2
#  elif (defined(__INTEL_COMPILER) && !defined(_WIN32)) || defined(__GNUC__)
#    define WLZ_MEMORY_ACCESS 1
#  endif
#endif

/*
 * FORCE_SW_BITCOUNT
 * Define this parameter if your target system or compiler does not support hardware bit count
 */
#if defined(_MSC_VER) && defined(_WIN32_WCE)   /* Visual Studio for WinCE doesn't support Hardware bit count */
#  define FORCE_SW_BITCOUNT
#endif



/*-************************************
*  Dependency
**************************************/
/*
 * WLZ_SRC_INCLUDED:
 * Amalgamation flag, whether WLZ.c is included
 */
#ifndef WLZ_SRC_INCLUDED
#  define WLZ_SRC_INCLUDED 1
#endif

#ifndef WLZ_STATIC_LINKING_ONLY
#define WLZ_STATIC_LINKING_ONLY
#endif

#include "WLZ4.h"
#include "Memry.h"

/* a prefetch for writing, where the compiler offers one */
#if defined(__GNUC__) || defined(__clang__)
#  define WLZ_PREFETCH_W(p)   __builtin_prefetch((p), 1)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <xmmintrin.h>
#  define WLZ_PREFETCH_W(p)   _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
#  define WLZ_PREFETCH_W(p)   ((void)(p))
#endif

/*-************************************
*  Compiler Options
**************************************/
#ifdef _MSC_VER    /* Visual Studio */
#  include <intrin.h>
#  pragma warning(disable : 4127)        /* disable: C4127: conditional expression is constant */
#  pragma warning(disable : 4293)        /* disable: C4293: too large shift (32-bits) */
#endif  /* _MSC_VER */


#include <stdio.h>
#include <stdlib.h>   /* malloc, calloc, free */
#include <string.h>   /* memset, memcpy */

#undef MIN
#define MIN(a,b)    ( (a) < (b) ? (a) : (b) )
#ifndef max
#  define max(a,b)  ( (a) > (b) ? (a) : (b) )
#endif
#ifndef min
#  define min(a,b)  ( (a) < (b) ? (a) : (b) )
#endif



/* WLZ_FAST_DEC_LOOP: LZ4-derived copy helpers, unused by the decoder below, which need symbols Memry.h no longer
   provides; off unless asked for */
#ifndef WLZ_FAST_DEC_LOOP
#  define WLZ_FAST_DEC_LOOP 0
#endif

#if WLZ_FAST_DEC_LOOP

WLZ_O2_INLINE_GCC_PPC64LE void
WLZ_memcpy_using_offset_base(Uint8* destPtr, const Uint8* srcPtr, Uint8* dstEnd, const int offset)
{
    if (offset < 8) {
        destPtr[0] = srcPtr[0];
        destPtr[1] = srcPtr[1];
        destPtr[2] = srcPtr[2];
        destPtr[3] = srcPtr[3];
        srcPtr += inc32table[offset];
        memcpy(destPtr+4, srcPtr, 4);
        srcPtr -= dec64table[offset];
        destPtr += 8;
    } else {
        memcpy(destPtr, srcPtr, 8);
        destPtr += 8;
        srcPtr += 8;
    }

    MemWildCpy8(destPtr, srcPtr, dstEnd);
}

/* customized variant of memcpy, which can overwrite up to 32 bytes beyond dstEnd
 * this version copies two times 16 bytes (instead of one time 32 bytes)
 * because it must be compatible with offsets >= 16. */
WLZ_O2_INLINE_GCC_PPC64LE void
MemWildCpy4(void* destPtr, const void* srcPtr, void* dstEnd)
{
    Uint8* d = (Uint8*)destPtr;
    const Uint8* s = (const Uint8*)srcPtr;
    Uint8* const e = (Uint8*)dstEnd;

    do { memcpy(d,s,16); memcpy(d+16,s+16,16); d+=32; s+=32; } while (d<e);
}

WLZ_O2_INLINE_GCC_PPC64LE void
WLZ_memcpy_using_offset(Uint8* destPtr, const Uint8* srcPtr, Uint8* dstEnd, const int offset)
{
    Uint8 v[8];
    switch(offset) {
    case 1:
        memset(v, *srcPtr, 8);
        goto copy_loop;
    case 2:
        memcpy(v, srcPtr, 2);
        memcpy(&v[2], srcPtr, 2);
        memcpy(&v[4], &v[0], 4);
        goto copy_loop;
    case 4:
        memcpy(v, srcPtr, 4);
        memcpy(&v[4], srcPtr, 4);
        goto copy_loop;
    default:
        WLZ_memcpy_using_offset_base(destPtr, srcPtr, dstEnd, offset);
        return;
    }

 copy_loop:
    memcpy(destPtr, v, 8);
    destPtr += 8;
    while (destPtr < dstEnd) {
        memcpy(destPtr, v, 8);
        destPtr += 8;
    }
}
#endif


/*-************************************
*  Common Constants
**************************************/

#define MIN_MATCH_LEN  3
#define MAX_HASH_LEN   5  


#define WILDCOPYLENGTH 8
#define MATCH_SAFEGUARD_DISTANCE  ((2*WILDCOPYLENGTH) - MIN_MATCH_LEN)   /* ensure it's possible to write 2 x wildcopyLength without overflowing output buffer */
#define FASTLOOP_SAFE_DISTANCE 64


#ifndef WLZ_MAX_DIST   /* can be user - defined at compile time */
#  define WLZ_MAX_DIST     (WLZ_MATCH2_WINDOW-1)
#endif

#define ML_BITS  4
#define ML_MASK  ((1U<<ML_BITS)-1)
#define RUN_BITS (8-ML_BITS)
#define RUN_MASK ((1U<<RUN_BITS)-1)


/*-************************************
*  Common functions
**************************************/



#ifndef WLZ_COMMONDEFS_ONLY
/*-************************************
*  Local Constants
**************************************/
static const Uint32 WLZ_skipTrigger = 6;  /* Increase this value ==> compression run slower on incompressible data */


/*-************************************
*  Local Utils
**************************************/
int WLZ_versionNumber(void) { return WLZ_VERSION_NUMBER; }
const char* WLZ_versionString(void) { return WLZ_VERSION_STRING; }

/*-******************************
*  Compression functions
********************************/



/* multiplicative hashes of the 3 (Hash1) and 5 (Hash2) bytes at stream, reduced to 'bits': the bytes sit at the
   top of a 64-bit word, so the top bits of the product depend on all of them; one multiply and one shift */
#define WLZ_HASH_PRIME   0x9E3779B185EBCA87ULL

ForceInlineTemplate Uint32 WLZ_Hash1(const Uint8* const stream, const Uint32 bits)
{
	return (Uint32)((((Uint64)MemReadLE4(stream)) << 40) * WLZ_HASH_PRIME >> (64 - bits));
}

ForceInlineTemplate Uint32 WLZ_Hash2(const Uint8* const stream, const Uint32 bits)
{
	return (Uint32)((MemReadLE8(stream) << 24) * WLZ_HASH_PRIME >> (64 - bits));
}


#define WLZ_WRITE_ExtraLength(destPtr, len) {         \
	if (likely (len <252) ) *destPtr++ = (Uint8)len;  \
	else {                                            \
		len -= 252;                                   \
		int n = 1 + (len > ByteMask[1]) + (len > ByteMask[2]) + (len > ByteMask[3]); \
		*destPtr++ = (Uint8)(251 + n);                \
		MemWriteLE4(destPtr, len);           \
		destPtr += n;                                 \
	}                                                 \
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  Match codes (the low nibble of the token) and offsets: code c is length 3 + c, and code 15 length 18 + extension.
 *     0     : length 3, a 1-byte offset (window 256)
 *     1..2  : lengths 4..5, a flagged offset of 1 or 2 bytes
 *     3..15 : lengths 6 and up, a flagged offset of 2 or 3 bytes
 *  A flagged offset is little-endian with its low bit the flag, 0 for the short form and 1 for one more byte; the
 *  offset is the other bits: 7 (window 128), 15 (32K) or 23 (8M). One rule gives every size: code 0 takes 1 byte,
 *  any other code 1 + (code > 2) + flag. A far match needs no length extension below length 18, and the decoder takes
 *  the offset size from the flag after the 4-byte read it does anyway. A block ends with code 0 and a zero offset.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define WLZ_NEAR_WINDOW     (1u << 8)               /* length 3: one byte */
#define WLZ_TINY_WINDOW     (1u << 7)               /* lengths 4-5: one flagged byte */
#define WLZ_SHORT_WINDOW    (1u << 15)              /* two flagged bytes */
#define WLZ_MID_WINDOW      (1u << 16)              /* the reach of the 64K chains */
#define WLZ_FAR_WINDOW      (1u << 23)              /* three flagged bytes, lengths 6 and up */
#define WLZ_SHORT_MAXLEN    5                       /* lengths 4-5 take one or two flagged bytes */
#define WLZ_FAR_MINLEN      6                       /* the shortest length with a three-byte offset */
#define WLZ_CODE_LONG       15                      /* length 18 + extension */
/* by match code, for the decoder: flagged offset, and offset bytes less one before the flag; table lookups keep
   the per-sequence instruction count of the old format (two compares cost a tenth of the decoding speed) */
static const Uint8 WLZ_CodeFlag[16] = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const Uint8 WLZ_CodeBase[16] = { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const Uint32 WLZ_OffMask[4] = { 0xFF, 0x7F, 0x7FFF, 0x7FFFFF };   /* by (bytes + flagged - 1): one plain byte; one,
                                                                            two or three flagged bytes (flag shifted out) */

/* bytes of a length extension of value v */
ForceInlineTemplate int WLZ_Ext_Size(const Uint32 v)
{
	return v < 252 ? 1 : 2 + (v - 252 > 0xFF) + (v - 252 > 0xFFFF) + (v - 252 > 0xFFFFFF);
}

/* bytes of the match part of a sequence (offset and length extension; not the token), or 0 if not codable */
ForceInlineTemplate int WLZ_Match_Size(const Uint32 len, const Uint32 off)
{
	if (len < MIN_MATCH_LEN || off == 0) return 0;
	if (len == MIN_MATCH_LEN) return off < WLZ_NEAR_WINDOW ? 1 : 0;
	if (len <= WLZ_SHORT_MAXLEN) return off < WLZ_TINY_WINDOW ? 1 : off < WLZ_SHORT_WINDOW ? 2 : 0;
	if (off >= WLZ_FAR_WINDOW) return 0;
	return (off < WLZ_SHORT_WINDOW ? 2 : 3) + (len >= 18 ? WLZ_Ext_Size(len - 18) : 0);
}

/* bytes of a sequence: token, literal run, match; a match the format cannot code counts as too large to pay, so a
   kernel that proposes one keeps its bytes as literals */
ForceInlineTemplate int WLZ_Seq_Size(const Uint32 litLen, const Uint32 matchLen, const Uint32 off)
{
	const int m = WLZ_Match_Size(matchLen, off);
	return m ? 1 + (int)litLen + (litLen >= RUN_MASK ? WLZ_Ext_Size(litLen - RUN_MASK) : 0) + m : 1 << 28;
}

/* writes one sequence: the literals from anchor, then the match (which must be codable) */
ForceInlineTemplate Uint8* WLZ_Encode_Sequence(Uint8* destPtr, const Uint8* anchor, const Uint32 litLen, const Uint32 matchLen, const Uint32 off)
{
	Uint8* const token = destPtr++;
	if (litLen >= RUN_MASK) {
		Uint32 extraLitLen = litLen - RUN_MASK;
		*token = (RUN_MASK << ML_BITS);
		WLZ_WRITE_ExtraLength(destPtr, extraLitLen);
		/* only past 16 literals: the wild copy always moves one chunk, and for a run of 15 or 16 before a match
		   16 bytes from the input's end it would read past it (found by fuzzing) */
		if (litLen > 16) MemWildCopy(destPtr + 16, anchor + 16, destPtr + litLen);
	}
	else *token = (Uint8)(litLen << ML_BITS);
	memcpy(destPtr, anchor, 16);
	destPtr += litLen;

	if (matchLen == MIN_MATCH_LEN) {                     /* code 0: a one-byte offset */
		*destPtr++ = (Uint8)off;
		return destPtr;
	}
	/* a flagged offset: 1 + (length >= 6) bytes, or one more when the flag is set */
	const Uint32 longer = matchLen > WLZ_SHORT_MAXLEN;
	const Uint32 flag = off >= (longer ? WLZ_SHORT_WINDOW : WLZ_TINY_WINDOW);
	MemWriteLE4(destPtr, off << 1 | flag);
	destPtr += 1 + longer + flag;
	if (matchLen >= MIN_MATCH_LEN + WLZ_CODE_LONG) {
		Uint32 ext = matchLen - (MIN_MATCH_LEN + WLZ_CODE_LONG);
		*token += WLZ_CODE_LONG;
		WLZ_WRITE_ExtraLength(destPtr, ext);
	}
	else *token += (Uint8)(matchLen - MIN_MATCH_LEN);
	return destPtr;
}

/* a sequence is written only if the output stays within -WLZ_GAIN_THRESHOLD bytes of the input covered: in
   incompressible data a short match after a long literal run costs more than it saves, and its bytes stay literals.
   This bounds the output (input size + a few bytes) without giving up on the rest of the input. */
#define WLZ_SEQ_PAYS(outBytes, seqBytes, inBytes)   ((int)(outBytes) + (int)(seqBytes) + WLZ_GAIN_THRESHOLD <= (int)(inBytes))

ForceInlineTemplate int WLZ_Match_Count(const Uint8* srcPtr, const Uint8* matchPtr, const Uint8* const srcLimit, const Uint8* const matchLimit)
{
	int matchLen = 0;
	reg_t matchDiff;

	matchDiff = MemReadARCH(matchPtr) ^ MemReadARCH(srcPtr);
	while (0 == matchDiff && srcPtr < srcLimit && (matchLimit == NULL || matchPtr < matchLimit) ) {
		srcPtr += REG_SIZE;
		matchPtr += REG_SIZE;
		matchLen += REG_SIZE;
		matchDiff = MemReadARCH(matchPtr) ^ MemReadARCH(srcPtr);
	} 

	matchLen += matchDiff==0? REG_SIZE : N_ZeroBytes(matchDiff);

	return matchLen;
}




/****************************************************************************************************************************************************************************/

/** forced inline, to ensure constant branches are decided at compilation time **/
/* the decoded size, ahead of the block: 2 bytes below 32 KB, else 4 (the first two carry the flag in the top bit and
   the low 15 bits, the next two the high bits), so a decoder can size its output from the first bytes */
ForceInlineTemplate Uint8* WLZ_Write_Size(Uint8* destPtr, const Uint32 srcSize)
{
	if (srcSize >> 15) {
		MemWriteLE2(destPtr, (Uint16)((1 << 15) | (srcSize & ((1 << 15) - 1))));
		MemWriteLE2(destPtr + 2, (Uint16)(srcSize >> 15));
		return destPtr + 4;
	}
	MemWriteLE2(destPtr, (Uint16)srcSize);
	return destPtr + 2;
}

ForceInlineTemplate Uint32 WLZ_Compress_Kernel(
                 WLZ_State_Str* const wlzStr,
                 const char* const source,
                 char* const destiny,
				 const int srcSize,
                 int acceleration)
{
	unsigned i;
    const Uint8* srcPtr = (const Uint8*) source;
    const Uint8* anchor = (const Uint8*) source;
    const Uint8* const srcEnd = (const Uint8*)source + srcSize;
    const Uint8* const srcLastMatch = srcEnd - REG_SIZE*2;
    Uint8* destPtr = (Uint8*) destiny;
	Uint32 hashV1, hashV2;
	int  match1Idx, match2Idx;
	Uint32  lazyMatchLen, lazyMatchOffset, lazyMatchFail;
	Uint32 matchLen, matchOffset;
	Uint32 litLen, extraLitLen;
	int *hash1Table = wlzStr->hash1Table;
	int *hash2Table = wlzStr->hash2Table;
	const Uint32 dictSize = wlzStr->dictSize;
	const Uint8* const dictEnd = wlzStr->dictEnd;
	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	reg_t currPattern, diffPattern;

	const Uint8 *matchPtr;
	const Uint32 isLazyMatch = (acceleration == 0);         // acceleration=0 indicates lazy match

#ifdef WLZ_DEBUG
	FILE *fptr = fopen("WLZ_Compress_Index.txt", "w");
	fprintf(fptr, "WLZ_Compress_Kernel: srcSize=%i\n", srcSize);
#endif
    
    /* If init conditions are not met, we don't have to mark stream
     * as having dirty context, since no action was taken yet */
    if ((Uint32)srcSize > (Uint32)WLZ_MAX_INPUT_SIZE) return 0;           /* Unsupported srcSize, too large (or negative) */
	destPtr = WLZ_Write_Size(destPtr, (Uint32)srcSize);
	Uint8* const payload = destPtr;                         /* the guard counts the payload only */
	acceleration = max(acceleration, 1);

    if ( srcSize <= MAX_HASH_LEN ) goto _last_literals;        /* Input too small, no compression (all literals) */

	Uint32 lastOffset = 1;
	Uint32 srcIdx = 1;
	srcPtr++;
	while( 1 ) {
        
        int step = 1; 
        int searchMatchNb = acceleration << WLZ_skipTrigger;
        while( 1 )  {
			/*if (srcIdx == 4194292) {
				srcIdx += 0;
			}*/

            if ( unlikely(srcPtr >= srcLastMatch) ) goto _last_literals;
			
			hashV1 = WLZ_Hash1(srcPtr, WLZ_HASH1BITS);  
			hashV2 = WLZ_Hash2(srcPtr, WLZ_HASH2BITS);
			match2Idx = hash2Table[ hashV2 ];
			hash2Table[ hashV2 ] = srcIdx;
			match1Idx = hash1Table[ hashV1 ];
			hash1Table[ hashV1 ] = srcIdx;		
			currPattern = MemReadARCH(srcPtr);
			matchLen = 0;
			matchOffset = srcIdx - match2Idx;
			if ( match2Idx && matchOffset - 1 < WLZ_FAR_WINDOW - 1 ) {         /* 1 <= offset < window */			
				matchPtr = (dictSize && match2Idx < 0) ? dictEnd+match2Idx : srcPtr - matchOffset;
				diffPattern = currPattern^ MemReadARCH(matchPtr);
				if (0==diffPattern) {
					matchLen = REG_SIZE + WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, dictLastMatch);
					break;
				}

				matchLen = N_ZeroBytes(diffPattern);
				if (matchLen >= (matchOffset < WLZ_SHORT_WINDOW ? MAX_HASH_LEN : WLZ_KERNEL_FARLEN))   /* three offset bytes need a longer match */
					break;
			}
			/*if (matchOffset != lastOffset) {
				matchOffset = lastOffset;
				match2Idx = srcIdx - matchOffset;
				matchPtr = (dictSize && match2Idx < 0) ? dictEnd + match2Idx : srcPtr - matchOffset;
				if (MemRead4(srcPtr) == MemRead4(matchPtr)) {
					matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, dictLastMatch);
					if (matchLen >= MAX_HASH_LEN || matchOffset < 256) 
						break;
				}
			}*/

			matchOffset = srcIdx - match1Idx;
			if (match1Idx && matchOffset - 1 < WLZ_FAR_WINDOW - 1 ) {
				matchPtr = (dictSize && match1Idx < 0) ? dictEnd+match1Idx : srcPtr - matchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				if (0 == diffPattern) {
					matchLen = REG_SIZE + WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, dictLastMatch);
					break;
				}
				matchLen = N_ZeroBytes(diffPattern);
				if (matchLen >= (matchOffset < WLZ_SHORT_WINDOW ? MAX_HASH_LEN : WLZ_KERNEL_FARLEN) || ( matchOffset < WLZ_MATCH1_WINDOW && matchLen>= MIN_MATCH_LEN) )
					break;
			}
			
			srcPtr += step;
			srcIdx += step;
			step = (searchMatchNb++ >> WLZ_skipTrigger);
        } 
				
		if (isLazyMatch) {
			lazyMatchLen = 0;
			lazyMatchFail = 1;
			srcPtr++;
			srcIdx++;

			lazyMatchOffset = lastOffset;
			match2Idx = srcIdx - lazyMatchOffset;
			matchPtr = (dictSize && match2Idx < 0) ? dictEnd + match2Idx : srcPtr - lazyMatchOffset;
			if (MemRead4(srcPtr) == MemRead4(matchPtr)) {
				lazyMatchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, dictLastMatch);
				if (lazyMatchLen > matchLen && (lazyMatchLen >= MAX_HASH_LEN || lazyMatchOffset < 256)
				    && WLZ_Match_Size(lazyMatchLen, lazyMatchOffset)) {
					matchLen = lazyMatchLen;
					matchOffset = lazyMatchOffset;
					lazyMatchFail = 0;
				}
			}

			hashV1 = WLZ_Hash1(srcPtr, WLZ_HASH1BITS);
			hashV2 = WLZ_Hash2(srcPtr, WLZ_HASH2BITS);
			match2Idx = hash2Table[hashV2];
			hash2Table[hashV2] = srcIdx;
			hash1Table[hashV1] = srcIdx;
			currPattern = MemReadARCH(srcPtr);
			lazyMatchOffset = srcIdx - match2Idx;
			if (match2Idx && lazyMatchOffset - 1 < WLZ_FAR_WINDOW - 1 && lazyMatchOffset != lastOffset ) {
				const Uint32 need = lazyMatchOffset < WLZ_SHORT_WINDOW ? MAX_HASH_LEN : WLZ_KERNEL_FARLEN;
				matchPtr = (dictSize && match2Idx < 0) ? dictEnd + match2Idx : srcPtr - lazyMatchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				
				if (0 == diffPattern) {
					lazyMatchLen = REG_SIZE + WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, dictLastMatch);
				}
				else lazyMatchLen = N_ZeroBytes(diffPattern);

				if ( (matchLen!= MAX_HASH_LEN-1 && lazyMatchLen > matchLen&& lazyMatchLen >= need) ||  (matchLen== MAX_HASH_LEN-1 && lazyMatchLen>= max(need, MAX_HASH_LEN +1)) ) {
					matchLen = lazyMatchLen;
					matchOffset = lazyMatchOffset;
					lazyMatchFail = 0;
				}
			}

			srcIdx -= lazyMatchFail;
			srcPtr -= lazyMatchFail;
		}
				
		if (matchOffset > 1) {
			i = 1 + isLazyMatch;
			hashV1 = WLZ_Hash1(srcPtr + i, WLZ_HASH1BITS);
			hashV2 = WLZ_Hash2(srcPtr + i, WLZ_HASH2BITS);
			hash1Table[ hashV1 ] = srcIdx + i;
			hash2Table[ hashV2 ] = srcIdx + i;

			for (++i; i < matchLen-1 && srcPtr + i + 1 <= srcLastMatch; i+=2) {   /* 8-byte hash reads stay inside the input */
				hashV1 = WLZ_Hash1(srcPtr + i, WLZ_HASH1BITS);
				hashV2 = WLZ_Hash2(srcPtr + i, WLZ_HASH2BITS);
				hash1Table[hashV1] = srcIdx + i;
				hash2Table[hashV2] = srcIdx + i;
				hashV2 = WLZ_Hash2(srcPtr + i+1, WLZ_HASH2BITS);
								hash2Table[hashV2] = srcIdx + i+1;
			}
		}

		litLen = (Uint32)(srcPtr - anchor);
		if (!WLZ_SEQ_PAYS(destPtr - payload, WLZ_Seq_Size(litLen, matchLen, matchOffset), srcPtr + matchLen - (const Uint8*)source)) {
			srcPtr++;                                           /* costs more than it saves: a literal, and search on */
			srcIdx++;
			matchLen = 0;
			continue;
		}

#ifdef WLZ_DEBUG
		fprintf(fptr, "srcIdx=%d, destIdx=%d,  litLen=%d,  matchLen=%d, matchOffset=%d\n",
			(int)(anchor - (const Uint8*)source), (int)(destPtr - (Uint8*)destiny), litLen, matchLen, matchOffset);
#endif
		destPtr = WLZ_Encode_Sequence(destPtr, anchor, litLen, matchLen, matchOffset);
		srcIdx += matchLen;
		srcPtr += matchLen;
		lastOffset = matchOffset;
		anchor = srcPtr;
		matchLen = 0;
    }

_last_literals:
    /* Encode Last Literals */
    litLen = (int)(srcEnd - anchor);
        
#ifdef WLZ_DEBUG
	fprintf(fptr, "srcIdx=%d, destIdx=%d,  litLen=%d\n",
		(int)(anchor - (const Uint8*)source), (int)(destPtr - (Uint8*)destiny), litLen);
#endif

    if (litLen >= RUN_MASK) {
        extraLitLen = litLen - RUN_MASK;
        *destPtr++ = RUN_MASK << ML_BITS;
		WLZ_WRITE_ExtraLength(destPtr, extraLitLen);
    } else {
        *destPtr++ = (Uint8)(litLen <<ML_BITS );
    }
	memcpy(destPtr, anchor, litLen);                      /* exact: the input may end right here */
    destPtr += litLen;
	*destPtr++ = 0;                                       /* a zero offset ends the block */

#ifdef WLZ_DEBUG
    fprintf(fptr, "Compressed %i bytes into %i bytes\n", srcSize, (Uint32)(((char*)destPtr) - destiny));
	fclose(fptr);
#endif
    return (Uint32)(((char*)destPtr) - destiny);
}


unsigned WLZ_Compress(WLZ_State_Str *wlzStr, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize)
{
	
	unsigned destSizeBound = WLZ_COMPRESSBOUND(srcSize);
	if (destCapSize < destSizeBound ) {
		return 0;
	}

	WLZ_Init_State(wlzStr);
    return WLZ_Compress_Kernel(wlzStr, source, destiny, srcSize, 0);
}

unsigned WLZ_Compress_Fast(WLZ_State_Str *wlzStr, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int acceleration)
{
	if (destCapSize < WLZ_COMPRESSBOUND(srcSize)) {
		return 0;
	}
	
	WLZ_Init_State(wlzStr);
	return WLZ_Compress_Kernel(wlzStr, source, destiny, srcSize, acceleration);
}


/* hidden WLZ_DEBUG function */
/* strangely enough, gcc generates faster code when this function is uncommented, even if unused */
unsigned WLZ_Compress_wDict(const char *dictionary, unsigned dictSize, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int acceleration)
{
	if (destCapSize < WLZ_COMPRESSBOUND(srcSize)) {
		return 0;
	}
	unsigned result;
	WLZ_State_Str* const wlzStr = WLZ_New_State();       /* a local struct had no table: Init_State wrote through a wild pointer */
	if (wlzStr == NULL) return 0;

	WLZ_Load_Dictionary(wlzStr, dictionary, dictSize);

	result = WLZ_Compress_Kernel(wlzStr, source, destiny, srcSize, acceleration);

	WLZ_Free_State(wlzStr);
	return result;
}

unsigned WLZ_Compress_wDictStr(WLZ_State_Str dictStr, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int acceleration)
{
	if (destCapSize < WLZ_COMPRESSBOUND(srcSize)) {
		return 0;
	}

	unsigned result;
	WLZ_State_Str* const wlzStr = WLZ_New_State();
	if (wlzStr == NULL) return 0;

	WLZ_Attach_Dictionary(wlzStr, &dictStr);

	result = WLZ_Compress_Kernel(wlzStr, source, destiny, srcSize, acceleration);

	WLZ_Free_State(wlzStr);
	return result;
}

/*-******************************
*  Streaming functions
********************************/

#ifndef _MSC_VER  /* for some reason, Visual fails the aligment test on 32-bit x86 :
                     it reports an aligment of 8-bytes,
                     while actually aligning WLZ_State_Str on 4 bytes. */
#endif

WLZ_State_Str *WLZ_New_State()
{
	WLZ_State_Str* const wlzStr = (WLZ_State_Str*)calloc(1, sizeof(WLZ_State_Str));
	if (wlzStr == NULL) return NULL;
	wlzStr->hash2Table = (int *)calloc((1 << WLZ_HASH2BITS), sizeof(int));
	if (wlzStr->hash2Table == NULL) { free(wlzStr); return NULL; }
	return wlzStr;
}

void WLZ_Init_State(WLZ_State_Str *wlzStr)
{
	memset(wlzStr->hash1Table, 0, (1 << WLZ_HASH1BITS) * sizeof(int));
	memset(wlzStr->hash2Table, 0, (1 << WLZ_HASH2BITS) * sizeof(int));

	wlzStr->dictSize = 0;
	wlzStr->dictEnd = NULL;
}

void WLZ_Free_State (WLZ_State_Str *wlzStr)
{
	if (wlzStr == NULL) return;
	free(wlzStr->hash2Table);                             /* the dictionary is the caller's: only referenced */
	free(wlzStr);
}



unsigned WLZ_Load_Dictionary (WLZ_State_Str * wlzStr, const char* dictionary, unsigned dictSize)
{
	const Uint8* dictPtr;
	Uint32 hashV[2];
	int  dictIdx;
	wlzStr->dictSize = dictSize;
	wlzStr->dictEnd = (const Uint8*)dictionary + dictSize;
	int hashUnit = 8;     // max of reg_t and max-hash

	memset(wlzStr->hash1Table, 0, (1 << WLZ_HASH1BITS) * sizeof(int));
	if (wlzStr->hash2Table == NULL) wlzStr->hash2Table = (int *)calloc((1 << WLZ_HASH2BITS), sizeof(int));
	else memset(wlzStr->hash2Table, 0, (1 << WLZ_HASH2BITS) * sizeof(int));

   
    if (dictSize < hashUnit) {
        return 0;
    }

	for (dictPtr = wlzStr->dictEnd - MIN(dictSize, WLZ_MATCH2_WINDOW); dictPtr <= wlzStr->dictEnd - hashUnit; dictPtr++) {
		hashV[0] = WLZ_Hash1(dictPtr, WLZ_HASH1BITS);
		hashV[1] = WLZ_Hash2(dictPtr, WLZ_HASH2BITS);
		dictIdx = (int)(dictPtr - wlzStr->dictEnd);
		wlzStr->hash1Table[ hashV[0] ] = dictIdx;
		wlzStr->hash2Table[ hashV[1] ] = dictIdx;
    }

    return dictSize;
}

void WLZ_Attach_Dictionary(WLZ_State_Str *workStr, const WLZ_State_Str *dictStr) 
{
	if (dictStr == NULL) return;

	memcpy(workStr->hash1Table, dictStr->hash1Table, (1 << WLZ_HASH1BITS) * sizeof(int));
	memcpy(workStr->hash2Table, dictStr->hash2Table, (1 << WLZ_HASH2BITS) * sizeof(int));
    
	workStr->dictSize = dictStr->dictSize;
	workStr->dictEnd = dictStr->dictEnd;
}

/*! WLZ_Save_Dictionary() :
 *  If previously compressed data block is not guaranteed to remain available at its memory location,
 *  save it into a safer place (char* safeBuffer).
 *  Note : you don't need to call WLZ_loadDict() afterwards,
 *         dictionary is immediately usable, you can therefore call WLZ_compress_fast_continue().
 *  Return : saved dictionary size in bytes (necessarily <= dictSize), or 0 if error.
 */
unsigned WLZ_Save_Dictionary(WLZ_State_Str* dictStr, WLZ_State_Str* workStr)
{
	int i;
	for (i = 0; !(i >>WLZ_HASH1BITS); i++)
		dictStr->hash1Table[i] = workStr->hash1Table[i] == 0 ? 0 : workStr->hash1Table[i] - workStr->dictSize;

	for (i = 0; !(i >>WLZ_HASH2BITS); i++)
		dictStr->hash2Table[i] = workStr->hash2Table[i] == 0 ? 0 : workStr->hash2Table[i] - workStr->dictSize;

	dictStr->dictEnd = workStr->dictEnd;
	dictStr->dictSize = workStr->dictSize;
	return dictStr->dictSize;
}












/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ******************************************************************** Hash-Chain Compression Functions  ********************************************************************
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
static const int Search_Level_Map[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 1000 };

typedef struct wlz_match {
	int len;                                        // LZ match length
	int off;                                        // LZ match offset/distance
} WLZ_Match;

/* the net gain of a match: its length less its offset bytes beyond two (a one-byte offset gains one); a match the
   format cannot code gains nothing. Below the shortest match (a search's starting bar) the gain is the length. */
ForceInlineTemplate int WLZ_Gain(const int len, const Uint32 off)
{
	if (len < MIN_MATCH_LEN) return len;
	const int m = WLZ_Match_Size((Uint32)len, off);
	return m ? len - (m - 2 - (len >= MIN_MATCH_LEN + WLZ_CODE_LONG ? WLZ_Ext_Size((Uint32)len - (MIN_MATCH_LEN + WLZ_CODE_LONG)) : 0)) : -(1 << 20);
}
#define WLZ_GAIN(len, off)  WLZ_Gain((int)(len), (Uint32)(off))

ForceInlineTemplate Uint32 WLZ_Hash8(const Uint8* const p, const Uint32 bits)
{
	return (Uint32)(MemReadLE8(p) * WLZ_HASH_PRIME >> (64 - bits));
}

/* inserts the positions up to currIdx: the 5-byte hash chain (distances capped at 64K) and, for far matches, each
   position's uncapped distance to the previous position of its 5-byte hash and its previous position of the 8-byte hash */
ForceInlineTemplate void WLZhc_Insert(WLZhc_State_Str* const wlzStr, const Uint8* const source, const int currIdx)
{
	int* const hash2Table = wlzStr->hash2Table;
	const Uint32 far8Bits = wlzStr->far8Bits;
	const Uint8* srcPtr = source + wlzStr->currIdx;
	while (wlzStr->currIdx < currIdx) {
		const int idx = ++wlzStr->currIdx;
		srcPtr++;
		const Uint32 h5 = WLZ_Hash2(srcPtr, WLZhc_HASH2BITS), h8 = WLZ_Hash8(srcPtr, far8Bits);
		WLZ_PREFETCH_W(&wlzStr->far8Head[WLZ_Hash8(srcPtr + 8, far8Bits)]);   /* the far table slot 8 positions on (a likely miss) */
		const int prev = hash2Table[h5];
		const Uint32 dist = prev != 0 ? (Uint32)(idx - prev) : WLZ_MAX_DIST;
		hash2Table[h5] = idx;
		wlzStr->chain2Table[(Uint16)idx] = (Uint16)MIN(WLZ_MAX_DIST, dist);
		wlzStr->farLink[(Uint16)idx] = prev != 0 ? dist : 0;
		wlzStr->far8Prev[(Uint16)idx] = wlzStr->far8Head[h8];
		wlzStr->far8Head[h8] = (Uint32)idx;
	}
}

ForceInlineTemplate void WLZhc_Search_Hash1Table(WLZhc_State_Str* const wlzStr, const Uint8 *source, int currIdx, WLZ_Match *matchStr)
{
	int *hash1Table = wlzStr->hash1Table;
	const Uint8 *matchPtr, *srcPtr;
	Uint32 hashV0, matchDist;
	int matchIdx;
	reg_t diffPattern;

	srcPtr = source + wlzStr->curr1Idx;
	while(wlzStr->curr1Idx < currIdx-1) {
		hashV0 = WLZ_Hash1(++srcPtr, WLZhc_HASH1BITS);
		hash1Table[hashV0] = ++wlzStr->curr1Idx;
	}

	matchStr->len = 0;
	srcPtr = source + currIdx;
	reg_t currPattern = MemReadARCH(srcPtr);
	hashV0 = WLZ_Hash1(srcPtr, WLZhc_HASH1BITS);
	matchDist = currIdx - hash1Table[hashV0];
	hash1Table[hashV0] = currIdx;
	if (matchDist < WLZ_MATCH1_WINDOW) {
		matchIdx = currIdx - matchDist;
		matchPtr = (wlzStr->dictSize && matchIdx < 0) ? wlzStr->dictEnd + matchIdx : srcPtr - matchDist;
		diffPattern = currPattern ^ MemReadARCH(matchPtr);
		matchStr->len = diffPattern == 0 ? REG_SIZE : N_ZeroBytes(diffPattern);
		matchStr->off = matchDist;
	}
}

ForceInlineTemplate void WLZhc_Search_HashChain(WLZhc_State_Str* const wlzStr, const Uint8 *source, int currIdx, const Uint8 *srcLastMatch, const Uint8 *dictLastMatch, WLZ_Match *matchStr, int chainSearchCnt)
{
	Uint16 *chain2Table = wlzStr->chain2Table;
	const Uint8 *matchPtr, *srcPtr;
	Uint32 matchDist;
	int matchLen, matchIdx;
	const Uint32 dictSize = wlzStr->dictSize;
	const Uint8* const dictEnd = wlzStr->dictEnd;

	WLZhc_Insert(wlzStr, source, currIdx);
	srcPtr = source + currIdx;

	Uint32 currPattern = MemRead4(srcPtr);
	matchDist = chain2Table[(Uint16)currIdx];
	if (matchDist >= WLZ_MAX_DIST) {                                /* nothing within 64K: the previous position, if in the far window */
		const Uint32 farDist = wlzStr->farLink[(Uint16)currIdx];
		if (farDist >= WLZ_MAX_DIST && farDist < wlzStr->farWindow && (int)farDist <= currIdx && currPattern == MemRead4(srcPtr - farDist)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, srcPtr - farDist + 4, srcLastMatch, NULL);
			if (matchLen >= WLZ_KERNEL_FARLEN && WLZ_GAIN(matchLen, farDist) > WLZ_GAIN(matchStr->len, matchStr->off)) {
				matchStr->len = matchLen;
				matchStr->off = farDist;
			}
		}
	}
	while (matchDist < WLZ_MAX_DIST && (int)matchDist < currIdx + (int)dictSize && chainSearchCnt) {
		matchIdx = currIdx - matchDist;
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, dictLastMatch);
			if (WLZ_GAIN(matchLen, matchDist) > WLZ_GAIN(matchStr->len, matchStr->off)) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
			}
		}
		chainSearchCnt--;
		matchDist += chain2Table[(Uint16)matchIdx];
	}
	{   /* the previous position of the 8 bytes here, when beyond the two-byte window */
		const Uint32 p8 = wlzStr->far8Prev[(Uint16)currIdx], farDist = (Uint32)currIdx - p8;
		if (p8 && farDist >= WLZ_SHORT_WINDOW && farDist < wlzStr->farWindow && MemReadLE8(srcPtr) == MemReadLE8(srcPtr - farDist)) {
			matchLen = 8 + WLZ_Match_Count(srcPtr + 8, srcPtr - farDist + 8, srcLastMatch, NULL);
			if (WLZ_GAIN(matchLen, farDist) > WLZ_GAIN(matchStr->len, matchStr->off)) {
				matchStr->len = matchLen;
				matchStr->off = farDist;
			}
		}
	}
}

ForceInlineTemplate int WLZhc_Search_HashChain_2D(WLZhc_State_Str* const wlzStr, const Uint8 *source, int currIdx, int maxBack, int nextMatchLen, const Uint8 *srcLastMatch, const Uint8 *dictLastMatch, WLZ_Match *matchStr, int chainSearchCnt)
{
	Uint16 *chain2Table = wlzStr->chain2Table;
	const Uint8 *matchPtr, *srcPtr;
	Uint32 matchDist;
	int matchLen, matchIdx;
	int back, score, optBack = maxBack;
	Uint8 backByte[MAX_HASH_LEN + 1];
	int backIdx, backTable[8] = { 0, 1, 0, 2,  0, 1, 0, 3 };
	int vSearchCnt;
	Uint32 currPattern;
	const Uint32 dictSize = wlzStr->dictSize;
	const Uint8* const dictEnd = wlzStr->dictEnd;

	WLZhc_Insert(wlzStr, source, currIdx);
	srcPtr = source + currIdx;

	vSearchCnt = chainSearchCnt;
	currPattern = MemRead4(srcPtr);
	backByte[0] = *(srcPtr - 1);
	backByte[1] = *(srcPtr - 2);
	backByte[2] = *(srcPtr - 3);
	//backByte[3] = *(srcPtr - 4);
	matchDist = chain2Table[(Uint16)currIdx];
	while (matchDist < WLZ_MAX_DIST && (int)matchDist < currIdx + (int)dictSize && vSearchCnt) {
		matchIdx = currIdx - matchDist;
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4+ WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, dictLastMatch);

			//backIdx = (backByte[0] == *(matchPtr - 1));
			backIdx = (matchIdx >= 2 || (matchIdx < 0 && matchIdx >= 2 - (int)dictSize)) ? (backByte[0] == *(matchPtr - 1)) ^ ((backByte[1] == *(matchPtr - 2)) << 1) : 0;   /* never extend before the history start */
			//backIdx = (backByte[0] == *(matchPtr - 1)) ^ ((backByte[1] == *(matchPtr - 2)) << 1) ^ ((backByte[2] == *(matchPtr - 3)) << 2);
			back = backTable[ backIdx ];
			matchLen += back;
			score = WLZ_GAIN(matchLen, matchDist) - WLZ_GAIN(matchStr->len, matchStr->off) + (back-optBack) * nextMatchLen / maxBack ;
			if ( score>0 || (back<optBack && score==0) ) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
				optBack = back;
			}
			vSearchCnt--;
		}
		matchDist += chain2Table[(Uint16)matchIdx];
	}
	//if (maxBack - optBack > 0) return maxBack - optBack;

	vSearchCnt = 1 + chainSearchCnt/8;
	back = maxBack - 1;
	currIdx -= back;
	srcPtr -= back;
	currPattern = MemRead4(srcPtr);
	matchDist = chain2Table[(Uint16)currIdx];
	while (matchDist < WLZ_MAX_DIST && (int)matchDist < currIdx + (int)dictSize && vSearchCnt) {
		matchIdx = currIdx - matchDist;
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			matchLen = 4 + WLZ_Match_Count(srcPtr + 4, matchPtr + 4, srcLastMatch, dictLastMatch);
			score = WLZ_GAIN(matchLen, matchDist) - WLZ_GAIN(matchStr->len, matchStr->off) + (back - optBack) * nextMatchLen / maxBack;
			if (score >=0) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
				optBack = back; 
			}
			vSearchCnt--;
		}
		matchDist += chain2Table[(Uint16)matchIdx];
	}
	return maxBack - optBack;
}


 /** forced inline, to ensure branches are decided at compilation time **/
ForceInlineTemplate Uint32 WLZhc_Compress_Kernel(
	WLZhc_State_Str* const wlzStr,
	const char* const source,
	char* const destiny,
	const int srcSize,
	int validSearchLimit)
{
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	Uint8* destPtr = (Uint8*)destiny;
	WLZ_Match matchStr, nextMatchStr = { 0, 0 };
	int lazyForward;
	Uint32 litLen, extraLitLen;
	const Uint32 dictSize = wlzStr->dictSize;
	const Uint8* const dictLastMatch = dictSize ? wlzStr->dictEnd - REG_SIZE * 2 : NULL;

#ifdef WLZ_DEBUG
	FILE *fptr = fopen("WLZhc_Compress_Index.txt", "w");
	fprintf(fptr, "WLZ_Compress_Kernel: srcSize=%i\n", srcSize);
#endif

	/* If init conditions are not met, we don't have to mark stream
	 * as having dirty context, since no action was taken yet */
	if ((Uint32)srcSize > (Uint32)WLZ_MAX_INPUT_SIZE) return 0;           /* Unsupported srcSize, too large (or negative) */
	destPtr = WLZ_Write_Size(destPtr, (Uint32)srcSize);
	Uint8* const payload = destPtr;                         /* the guard counts the payload only */

	if (srcSize <= MAX_HASH_LEN) goto _last_literals;        /* Input too small, no compression (all literals) */

	int srcIdx = 1;
	int nextMatchDone = 0;
	srcPtr++;	
	while (1) {

		while (1) {
			/*if (srcIdx == 11184) {
				srcIdx += 0;
			}*/

			if (unlikely(srcPtr >= srcLastMatch)) goto _last_literals;
			
			if (nextMatchDone) {
				matchStr = nextMatchStr;
				nextMatchDone = 0;
			}
			else {
				matchStr.len = 0; matchStr.off = 0;
				WLZhc_Search_HashChain(wlzStr, (const Uint8*)source, srcIdx, srcLastMatch, dictLastMatch, &matchStr, validSearchLimit);
			}
			if (matchStr.len >= ((Uint32)matchStr.off < WLZ_SHORT_WINDOW ? MAX_HASH_LEN : WLZ_KERNEL_FARLEN)) break;   /* three offset bytes need a longer match */

			WLZhc_Search_Hash1Table(wlzStr, (const Uint8*)source, srcIdx, &matchStr);
			if (matchStr.len >= MIN_MATCH_LEN) break;

			srcPtr ++;
			srcIdx ++;
		}
		if (likely(srcPtr + matchStr.len < srcLastMatch && srcPtr + MAX_HASH_LEN < srcLastMatch)) {
			
			if ( matchStr.off==1 ) {        // cut short repetative hash chain
				wlzStr->currIdx = max(wlzStr->currIdx, srcIdx + matchStr.len - MAX_HASH_LEN);
				wlzStr->curr1Idx = max(wlzStr->curr1Idx, srcIdx + matchStr.len - MAX_HASH_LEN);
			}

			nextMatchStr.len = 2; nextMatchStr.off = 0;
			WLZhc_Search_HashChain(wlzStr, (const Uint8*)source, srcIdx + matchStr.len, srcLastMatch, dictLastMatch, &nextMatchStr, validSearchLimit);

			lazyForward = WLZhc_Search_HashChain_2D(wlzStr, (const Uint8*)source, srcIdx + 3, 3, nextMatchStr.len, srcLastMatch, dictLastMatch, &matchStr, 1+validSearchLimit/2 );
			
			
			nextMatchDone = (0==lazyForward);
			srcPtr += lazyForward;
			srcIdx += lazyForward;
		}

		litLen = (Uint32)(srcPtr - anchor);
		if (!WLZ_SEQ_PAYS(destPtr - payload, WLZ_Seq_Size(litLen, matchStr.len, matchStr.off), srcPtr + matchStr.len - (const Uint8*)source)) {
			srcPtr++;                                           /* costs more than it saves: a literal, and search on */
			srcIdx++;
			nextMatchDone = 0;
			continue;
		}

#ifdef WLZ_DEBUG
		fprintf(fptr, "srcIdx=%d, destIdx=%d,  litLen=%d,  matchLen=%d, matchOffset=%d\n",
			(int)(anchor - (const Uint8*)source), (int)(destPtr - (Uint8*)destiny), litLen, matchStr.len, matchStr.off);
#endif
		destPtr = WLZ_Encode_Sequence(destPtr, anchor, litLen, matchStr.len, matchStr.off);
		srcIdx += matchStr.len;
		srcPtr += matchStr.len;
		anchor = srcPtr;
	}

_last_literals:
	/* Encode Last Literals */
	litLen = (int)(srcEnd - anchor);

#ifdef WLZ_DEBUG
	fprintf(fptr, "srcIdx=%d, destIdx=%d,  litLen=%d\n",
		(int)(anchor - (const Uint8*)source), (int)(destPtr - (Uint8*)destiny), litLen);
#endif

	if (litLen >= RUN_MASK) {
		extraLitLen = litLen - RUN_MASK;
		*destPtr++ = RUN_MASK << ML_BITS;
		WLZ_WRITE_ExtraLength(destPtr, extraLitLen);
	}
	else {
		*destPtr++ = (Uint8)(litLen << ML_BITS);
	}
	memcpy(destPtr, anchor, litLen);                      /* exact: the input may end right here */
	destPtr += litLen;
	*destPtr++ = 0;                                       /* a zero offset ends the block */

#ifdef WLZ_DEBUG
	fprintf(fptr, "Compressed %i bytes into %i bytes\n", srcSize, (Uint32)(((char*)destPtr) - destiny));
	fclose(fptr);
#endif
	return (Uint32)(((char*)destPtr) - destiny);
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  Optimal parsing (levels 8-12): the parse of least compressed size, a shortest path over byte-exact prices.
 *  At each position five candidates cover the codable matches: the longest match within 128 and within 256 (lengths
 *  3-5: one byte for length 3 within 256 and for lengths 4-5 within 128), within 32K (two flagged bytes), within 64K
 *  and beyond 64K within 8M (three flagged bytes, lengths 6 and up); each length takes the cheapest candidate that
 *  reaches it. As in LZ4HC's optimal parser, each position keeps its cheapest path,
 *  with the length of its pending literal run.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define WLZ_OPT_NUM         (1 << 12)
#define WLZ_OPT_TRAILING    3
#define WLZ_OPT_INF         (1 << 30)
#define WLZ_FAR_HASHBITS    20

typedef struct {
	int price;                                      /* bytes to code the input up to here */
	int mlen;                                       /* match ending here, or 1: a literal */
	int off;
	int litlen;                                     /* literals ending here (after the last match) */
} WLZ_Opt_Node;

typedef struct {
	int nbSearches;                                 /* chain steps in the 64K window */
	int nbShort;                                    /* chain steps in the 256 window */
	int nbFar;                                      /* chain steps in the far window (each a likely cache miss) */
	int sufficientLen;                              /* a match this long is taken at once */
	int fullUpdate;                                 /* search every position whose price may still improve */
} WLZ_Opt_Params;

#define WLZhc_OPT_LEVEL_MIN 8
#define WLZhc_LEVEL_MAX     12

/* from level 8 on, optimal parsing is both smaller and faster than lazy parsing with deeper chains */
static const WLZ_Opt_Params WLZ_Opt_Level[WLZhc_LEVEL_MAX - WLZhc_OPT_LEVEL_MIN + 1] = {
	{    16,  8,   8,   32, 1 },                    /*  8 */
	{    32,  8,  16,   64, 1 },                    /*  9 */
	{    64, 16,  32,   64, 1 },                    /* 10 */
	{   256, 32,  64,  128, 1 },                    /* 11 */
	{  4096, 64, 256, WLZ_OPT_NUM, 1 },             /* 12 */
};

ForceInlineTemplate int WLZ_Lit_Price(int litLen)
{
	return litLen + (litLen >= (int)RUN_MASK ? WLZ_Ext_Size((Uint32)(litLen - (int)RUN_MASK)) : 0);
}

/* a sequence: token, its literal run, the match (offset and length extension) */
ForceInlineTemplate int WLZ_Seq_Price(int litLen, int matchLen, Uint32 off)
{
	return 1 + WLZ_Lit_Price(litLen) + WLZ_Match_Size((Uint32)matchLen, off);
}

/* the far window: 6-byte hashes, links of 32 bits */
typedef struct {
	int* head;
	Uint32* link;                                   /* link[pos & (window - 1)]: distance to the previous position of the hash */
	Uint32 window;                                  /* the level's far window (a power of two): the links are a ring over it */
} WLZ_Far_Finder;

ForceInlineTemplate Uint32 WLZ_Hash6(const Uint8* const p)
{
	return (Uint32)((MemReadLE8(p) << 16) * WLZ_HASH_PRIME >> (64 - WLZ_FAR_HASHBITS));
}

ForceInlineTemplate Uint32 WLZ_Hash4(const Uint8* const p, const Uint32 bits)
{
	return (Uint32)((((Uint64)MemReadLE4(p)) << 32) * WLZ_HASH_PRIME >> (64 - bits));
}

/* inserts the positions up to target: the 4-byte chain over 64K, the 3-byte chain over 256 (chain1, by the low byte),
   the 6-byte chain over the far window */
ForceInlineTemplate void WLZ_Opt_Insert(WLZhc_State_Str* const s, const Uint8* const src, const int target, Uint8* const chain1, WLZ_Far_Finder* const far)
{
	int idx = s->currIdx;
	while (idx < target) {
		idx++;
		const Uint32 h2 = WLZ_Hash4(src + idx, WLZhc_HASH2BITS);
		const Uint32 d2 = s->hash2Table[h2] ? (Uint32)(idx - s->hash2Table[h2]) : WLZ_MAX_DIST;
		s->hash2Table[h2] = idx;
		s->chain2Table[(Uint16)idx] = (Uint16)MIN(WLZ_MAX_DIST, d2);
		const Uint32 h1 = WLZ_Hash1(src + idx, WLZhc_HASH1BITS);
		const Uint32 d1 = s->hash1Table[h1] ? (Uint32)(idx - s->hash1Table[h1]) : WLZ_NEAR_WINDOW;
		s->hash1Table[h1] = idx;
		chain1[(Uint8)idx] = (Uint8)(d1 < WLZ_NEAR_WINDOW ? d1 : 0);
		if (far->head) {
			WLZ_PREFETCH_W(&far->head[WLZ_Hash6(src + idx + 8)]);   /* the head slot 8 positions on (a likely miss) */
			const Uint32 h6 = WLZ_Hash6(src + idx);
			const int prev = far->head[h6];
			far->link[idx & (far->window - 1)] = prev ? (Uint32)(idx - prev) : far->window;
			far->head[h6] = idx;
		}
	}
	s->currIdx = idx;
}

/* the longest match at idx within 128 and within 256 (lengths 3-5: one byte for length 3 within 256, for lengths 4-5
   within 128), within 32K and 64K (4 and up), and beyond 64K within the far window (6 and up) */
ForceInlineTemplate void WLZ_Opt_Find(WLZhc_State_Str* const s, const Uint8* const src, const int idx, const Uint8* const srcLastMatch,
	const Uint8* const chain1, const WLZ_Far_Finder* const far, const WLZ_Opt_Params* const par,
	WLZ_Match* const tinyM, WLZ_Match* const nearM, WLZ_Match* const shortM, WLZ_Match* const midM, WLZ_Match* const farM)
{
	const Uint8* const ip = src + idx;
	const Uint32 pattern4 = MemRead4(ip);
	tinyM->len = 0; tinyM->off = 0;
	nearM->len = 0; nearM->off = 0;
	midM->len = 0; midM->off = 0;
	shortM->len = 0; shortM->off = 0;
	farM->len = 0; farM->off = 0;

	/* 256 window, by the 3-byte chain, nearest first: the best within 128 is a snapshot on the way */
	{   Uint32 dist = chain1[(Uint8)idx];
		int n = par->nbShort, tinyDone = 0;
		while (dist && dist < WLZ_NEAR_WINDOW && n--) {
			if (!tinyDone && dist >= WLZ_TINY_WINDOW) { *tinyM = *nearM; tinyDone = 1; }
			const Uint8* const mp = ip - dist;
			const reg_t diff = MemReadARCH(ip) ^ MemReadARCH(mp);
			int len = diff ? (int)N_ZeroBytes(diff) : REG_SIZE + WLZ_Match_Count((Uint8*)ip + REG_SIZE, (Uint8*)mp + REG_SIZE, srcLastMatch, NULL);
			if (len > nearM->len) {
				nearM->len = len; nearM->off = (int)dist;
				if (len > WLZ_SHORT_MAXLEN) break;         /* longer lengths take two offset bytes: the 64K search covers them */
			}
			const Uint32 step = chain1[(Uint8)(idx - dist)];
			if (!step) break;
			dist += step;
		}
		if (!tinyDone && (Uint32)nearM->off < WLZ_TINY_WINDOW) *tinyM = *nearM;
	}
	/* 64K window, by the 4-byte chain */
	{   Uint32 dist = s->chain2Table[(Uint16)idx];
		int n = par->nbSearches;
		int bestLen = 3;
		while (dist < WLZ_MAX_DIST && (int)dist <= idx && n--) {
			const Uint8* const mp = ip - dist;
			if (MemRead4(mp) == pattern4 && mp[bestLen] == ip[bestLen]) {
				const int len = 4 + WLZ_Match_Count((Uint8*)ip + 4, (Uint8*)mp + 4, srcLastMatch, NULL);
				if (len > bestLen) {
					bestLen = len; midM->len = len; midM->off = (int)dist;
					if (dist < WLZ_SHORT_WINDOW) *shortM = *midM;   /* nearest first: the best within 32K */
					if (len >= par->sufficientLen) break;
				}
			}
			dist += s->chain2Table[(Uint16)(idx - dist)];
		}
	}
	if (nearM->len >= WLZ_FAR_MINLEN && nearM->len > midM->len) *midM = *nearM;   /* the 3-byte chain found the longer one */
	if (nearM->len >= WLZ_FAR_MINLEN && nearM->len > shortM->len) *shortM = *nearM;
	if (nearM->len > WLZ_SHORT_MAXLEN) nearM->len = WLZ_SHORT_MAXLEN;
	if (tinyM->len > WLZ_SHORT_MAXLEN) tinyM->len = WLZ_SHORT_MAXLEN;
	{   const WLZ_Match* const m = midM;                 /* a near long match: its short prefixes */
		const int pl = m->len < WLZ_SHORT_MAXLEN ? m->len : WLZ_SHORT_MAXLEN;
		if (pl > nearM->len && (Uint32)m->off < WLZ_NEAR_WINDOW) { nearM->len = pl; nearM->off = m->off; }
		if (pl > tinyM->len && (Uint32)m->off < WLZ_TINY_WINDOW) { tinyM->len = pl; tinyM->off = m->off; }
	}
	if (nearM->len < MIN_MATCH_LEN) nearM->len = 0;
	if (tinyM->len < MIN_MATCH_LEN) tinyM->len = 0;
	if (midM->len <= WLZ_SHORT_MAXLEN && (Uint32)midM->off >= WLZ_SHORT_WINDOW) midM->len = 0;   /* not codable */

	/* the far window beyond 64K, by the 6-byte chain: only worth it past what the 64K window reaches */
	if (far->head && midM->len < par->sufficientLen) {
		Uint32 dist = far->link[idx & (far->window - 1)];
		int n = par->nbFar;
		int bestLen = max(midM->len, WLZ_FAR_MINLEN - 1);
		while (dist < far->window && (int)dist <= idx && n--) {
			const Uint8* const mp = ip - dist;
			if (dist >= WLZ_MID_WINDOW && mp[bestLen] == ip[bestLen] && MemRead4(mp) == pattern4) {
				const int len = 4 + WLZ_Match_Count((Uint8*)ip + 4, (Uint8*)mp + 4, srcLastMatch, NULL);
				if (len > bestLen) {
					bestLen = len; farM->len = len; farM->off = (int)dist;
					if (len >= par->sufficientLen) break;
				}
			}
			dist += far->link[(idx - dist) & (far->window - 1)];
		}
	}
}

/* the cheapest codable candidate reaching length ml (tiny, near, short, mid, far); returns its offset, or 0 if none */
ForceInlineTemplate Uint32 WLZ_Opt_Offset(const int ml, const WLZ_Match* tinyM, const WLZ_Match* nearM, const WLZ_Match* shortM, const WLZ_Match* midM, const WLZ_Match* farM)
{
	Uint32 best = 0;
	int bestSize = 1 << 30, sz;
	if (ml <= tinyM->len && (sz = WLZ_Match_Size((Uint32)ml, (Uint32)tinyM->off)) && sz < bestSize) { bestSize = sz; best = (Uint32)tinyM->off; }
	if (ml <= nearM->len && (sz = WLZ_Match_Size((Uint32)ml, (Uint32)nearM->off)) && sz < bestSize) { bestSize = sz; best = (Uint32)nearM->off; }
	if (ml <= shortM->len && (sz = WLZ_Match_Size((Uint32)ml, (Uint32)shortM->off)) && sz < bestSize) { bestSize = sz; best = (Uint32)shortM->off; }
	if (ml <= midM->len && (sz = WLZ_Match_Size((Uint32)ml, (Uint32)midM->off)) && sz < bestSize) { bestSize = sz; best = (Uint32)midM->off; }
	if (ml <= farM->len && (sz = WLZ_Match_Size((Uint32)ml, (Uint32)farM->off)) && sz < bestSize) { bestSize = sz; best = (Uint32)farM->off; }
	return best;
}

static Uint32 WLZhc_Compress_Optimal(WLZhc_State_Str* const wlzStr, const char* const source, char* const destiny, const int srcSize, const WLZ_Opt_Params* const par)
{
	const Uint8* const src = (const Uint8*)source;
	const Uint8* ip = src + 1;
	const Uint8* anchor = src;
	const Uint8* const srcEnd = src + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	Uint8* destPtr = (Uint8*)destiny;
	Uint8 chain1[WLZ_NEAR_WINDOW] = { 0 };
	WLZ_Opt_Node* const opt = (WLZ_Opt_Node*)malloc((WLZ_OPT_NUM + WLZ_OPT_TRAILING + 64) * sizeof(WLZ_Opt_Node));
	WLZ_Far_Finder far = { NULL, NULL, wlzStr->farWindow };
	WLZ_Match tinyM, nearM, shortM, midM, farM;
	const int sufficientLen = min(par->sufficientLen, WLZ_OPT_NUM - 1);
	Uint32 litLen, extraLitLen;

	if ((Uint32)srcSize > (Uint32)WLZ_MAX_INPUT_SIZE || opt == NULL) { free(opt); return 0; }
	if (srcSize > (int)WLZ_MID_WINDOW && far.window > WLZ_MID_WINDOW && par->nbFar) {   /* the far window only when the input reaches past 64K */
		far.head = (int*)calloc((size_t)1 << WLZ_FAR_HASHBITS, sizeof(int));
		far.link = (Uint32*)malloc((size_t)min((Uint32)srcSize, far.window) * sizeof(Uint32));
		if (far.head == NULL || far.link == NULL) { free(far.head); free(far.link); far.head = NULL; far.link = NULL; }
	}
	destPtr = WLZ_Write_Size(destPtr, (Uint32)srcSize);
	Uint8* const payload = destPtr;                         /* the guard counts the payload only */
	if (srcSize <= 5 * 4) goto _last_literals;

	while (ip < srcLastMatch) {
		const int llen = (int)(ip - anchor);
		int cur, lastPos, bestLen, bestOff, known;           /* known: positions up to here are initialized */

		WLZ_Opt_Insert(wlzStr, src, (int)(ip - src), chain1, &far);
		WLZ_Opt_Find(wlzStr, src, (int)(ip - src), srcLastMatch, chain1, &far, par, &tinyM, &nearM, &shortM, &midM, &farM);
		if (!nearM.len && !midM.len && !farM.len) { ip++; continue; }

		{   const WLZ_Match* const lm = farM.len > midM.len ? &farM : &midM;
			if (lm->len >= sufficientLen) {                  /* good enough: coded at once */
				if (WLZ_SEQ_PAYS(destPtr - payload, WLZ_Seq_Size((Uint32)llen, (Uint32)lm->len, (Uint32)lm->off), ip + lm->len - src)) {
					destPtr = WLZ_Encode_Sequence(destPtr, anchor, (Uint32)llen, (Uint32)lm->len, (Uint32)lm->off);
					anchor = ip + lm->len;
				}
				ip += lm->len;
				continue;
			}
		}

		/* the first position: literals, then its matches */
		for (int r = 0; r < MIN_MATCH_LEN; r++) {
			opt[r].price = WLZ_Lit_Price(llen + r); opt[r].mlen = 1; opt[r].off = 0; opt[r].litlen = llen + r;
		}
		lastPos = max(max(nearM.len, midM.len), farM.len);
		for (int r = MIN_MATCH_LEN; r <= lastPos + WLZ_OPT_TRAILING; r++) opt[r].price = WLZ_OPT_INF;
		for (int ml = MIN_MATCH_LEN; ml <= lastPos; ml++) {
			const Uint32 off = WLZ_Opt_Offset(ml, &tinyM, &nearM, &shortM, &midM, &farM);
			if (!off) continue;
			opt[ml].price = WLZ_Seq_Price(llen, ml, off); opt[ml].mlen = ml; opt[ml].off = (int)off; opt[ml].litlen = llen;
		}
		for (int a = 1; a <= WLZ_OPT_TRAILING; a++) {
			opt[lastPos + a].price = opt[lastPos].price + WLZ_Lit_Price(a); opt[lastPos + a].mlen = 1; opt[lastPos + a].off = 0; opt[lastPos + a].litlen = a;
		}
		known = lastPos + WLZ_OPT_TRAILING;

		/* the following positions */
		for (cur = 1; cur < lastPos; cur++) {
			const Uint8* const curPtr = ip + cur;
			if (curPtr >= srcLastMatch) break;
			if (opt[cur].price >= WLZ_OPT_INF) continue;      /* not reachable (no match of that length) */
			if (par->fullUpdate) {
				/* nothing to gain where the next position is no dearer, unless a short match could still pay off */
				if (opt[cur + 1].price <= opt[cur].price && opt[cur + MIN_MATCH_LEN].price < opt[cur].price + 2) continue;
			}
			else if (opt[cur + 1].price <= opt[cur].price) continue;

			WLZ_Opt_Insert(wlzStr, src, (int)(curPtr - src), chain1, &far);
			WLZ_Opt_Find(wlzStr, src, (int)(curPtr - src), srcLastMatch, chain1, &far, par, &tinyM, &nearM, &shortM, &midM, &farM);
			if (!nearM.len && !midM.len && !farM.len) continue;

			{   const WLZ_Match* const lm = farM.len > midM.len ? &farM : &midM;
				if (lm->len >= sufficientLen || cur + lm->len >= WLZ_OPT_NUM) {    /* coded at once */
					bestLen = lm->len; bestOff = lm->off;
					lastPos = cur + 1;
					goto _encode;
				}
			}

			/* literals after cur */
			{   const int baseLit = opt[cur].litlen;
				for (int l = 1; l < MIN_MATCH_LEN; l++) {
					const int price = opt[cur].price - WLZ_Lit_Price(baseLit) + WLZ_Lit_Price(baseLit + l);
					if (price < opt[cur + l].price) {
						opt[cur + l].price = price; opt[cur + l].mlen = 1; opt[cur + l].off = 0; opt[cur + l].litlen = baseLit + l;
					}
				}
			}
			/* matches from cur */
			{   const int ll = opt[cur].mlen == 1 ? opt[cur].litlen : 0;
				const int base = opt[cur].mlen == 1 ? (cur > ll ? opt[cur - ll].price : 0) : opt[cur].price;
				const int maxLen = max(max(nearM.len, midM.len), farM.len);
				for (int ml = MIN_MATCH_LEN; ml <= maxLen; ml++) {
					const Uint32 off = WLZ_Opt_Offset(ml, &tinyM, &nearM, &shortM, &midM, &farM);
					if (!off) continue;
					const int pos = cur + ml;
					const int price = base + WLZ_Seq_Price(ll, ml, off);
					while (known < pos) opt[++known].price = WLZ_OPT_INF;     /* fresh positions */
					if (price <= opt[pos].price) {
						if (ml == maxLen && lastPos < pos) lastPos = pos;
						opt[pos].price = price; opt[pos].mlen = ml; opt[pos].off = (int)off; opt[pos].litlen = ll;
					}
				}
			}
			for (int a = 1; a <= WLZ_OPT_TRAILING; a++) {
				opt[lastPos + a].price = opt[lastPos].price + WLZ_Lit_Price(a); opt[lastPos + a].mlen = 1; opt[lastPos + a].off = 0; opt[lastPos + a].litlen = a;
			}
			if (known < lastPos + WLZ_OPT_TRAILING) known = lastPos + WLZ_OPT_TRAILING;
		}

		bestLen = opt[lastPos].mlen;
		bestOff = opt[lastPos].off;
		cur = lastPos - bestLen;

	_encode:    /* back-trace from the last match, then code the sequences in order */
		{   int candidate = cur, selLen = bestLen, selOff = bestOff;
			while (1) {
				const int nextLen = opt[candidate].mlen, nextOff = opt[candidate].off;
				opt[candidate].mlen = selLen; opt[candidate].off = selOff;
				selLen = nextLen; selOff = nextOff;
				if (nextLen > candidate) break;
				candidate -= nextLen;
			}
		}
		{   int r = 0;
			while (r < lastPos) {
				const int ml = opt[r].mlen, off = opt[r].off;
				if (ml == 1) { ip++; r++; continue; }
				r += ml;
				if (WLZ_SEQ_PAYS(destPtr - payload, WLZ_Seq_Size((Uint32)(ip - anchor), (Uint32)ml, (Uint32)off), ip + ml - src)) {
					destPtr = WLZ_Encode_Sequence(destPtr, anchor, (Uint32)(ip - anchor), (Uint32)ml, (Uint32)off);
					anchor = ip + ml;                   /* else it costs more than it saves: its bytes stay literals */
				}
				ip += ml;
			}
		}
	}

_last_literals:
	free(opt); free(far.head); free(far.link);
	litLen = (Uint32)(srcEnd - anchor);
	if (litLen >= RUN_MASK) {
		extraLitLen = litLen - RUN_MASK;
		*destPtr++ = RUN_MASK << ML_BITS;
		WLZ_WRITE_ExtraLength(destPtr, extraLitLen);
	}
	else *destPtr++ = (Uint8)(litLen << ML_BITS);
	memcpy(destPtr, anchor, litLen);                    /* exact: the input may end right here */
	destPtr += litLen;
	*destPtr++ = 0;                                     /* a zero offset ends the block */
	return (Uint32)(destPtr - (Uint8*)destiny);
}

WLZhc_State_Str *WLZhc_New_State()
{
	WLZhc_State_Str* const wlzStr = (WLZhc_State_Str*)calloc(1, sizeof(WLZhc_State_Str));
	if (wlzStr == NULL) return NULL;
	wlzStr->hash2Table = (int *)malloc((1 << WLZhc_HASH2BITS) * sizeof(int));
	wlzStr->chain2Table = (Uint16 *)malloc(WLZ_MATCH2_WINDOW * sizeof(short));
	wlzStr->farLink = (Uint32 *)calloc(WLZ_MATCH2_WINDOW, sizeof(Uint32));
	wlzStr->far8Prev = (Uint32 *)calloc(WLZ_MATCH2_WINDOW, sizeof(Uint32));
	wlzStr->far8Head = (Uint32 *)calloc((size_t)1 << WLZhc_FAR8BITS, sizeof(Uint32));
	wlzStr->far8Bits = WLZhc_FAR8BITS;
	wlzStr->farWindow = WLZ_FAR_WINDOW;
	if (wlzStr->hash2Table == NULL || wlzStr->chain2Table == NULL || wlzStr->farLink == NULL || wlzStr->far8Prev == NULL || wlzStr->far8Head == NULL) {
		WLZhc_Free_State(wlzStr);
		return NULL;
	}
	return wlzStr;
}
void WLZhc_Init_State(WLZhc_State_Str *wlzStr)
{

	wlzStr->currIdx = wlzStr->curr1Idx = 1;
	memset(wlzStr->hash1Table, 0, (1 << WLZhc_HASH1BITS) * sizeof(int));

	memset(wlzStr->hash2Table, 0, (1 << WLZhc_HASH2BITS) * sizeof(int));
	memset(wlzStr->chain2Table, 0xFF, WLZ_MATCH2_WINDOW * sizeof(short));
	memset(wlzStr->far8Head, 0, sizeof(Uint32) << wlzStr->far8Bits);   /* the far links are written before they are read */

	wlzStr->dictSize = 0;
	wlzStr->dictEnd = NULL;
}

void WLZhc_Free_State(WLZhc_State_Str *wlzStr)
{
	if (wlzStr == NULL) return;
	free(wlzStr->hash2Table);
	free(wlzStr->chain2Table);                            /* the dictionary is the caller's: only referenced */
	free(wlzStr->farLink);
	free(wlzStr->far8Prev);
	free(wlzStr->far8Head);
	free(wlzStr);
}

unsigned WLZhc_Load_Dictionary(WLZhc_State_Str * wlzStr, const char* dictionary, unsigned dictSize)
{
	const Uint8* dictPtr;
	Uint32 hashV1, hashV2;
	int  dictIdx;

	WLZhc_Init_State(wlzStr);
	wlzStr->dictSize = dictSize;
	wlzStr->dictEnd = (const Uint8*)dictionary + dictSize;
	unsigned hashUnit = 8;     // max of reg_t and max-hash

	if (dictSize < hashUnit) {
		return 0;
	}

	for (dictPtr = wlzStr->dictEnd - MIN(dictSize, WLZ_MATCH2_WINDOW); dictPtr <= wlzStr->dictEnd - hashUnit; dictPtr++) {
		hashV1 = WLZ_Hash1(dictPtr, WLZhc_HASH1BITS);
		hashV2 = WLZ_Hash2(dictPtr, WLZhc_HASH2BITS);
		dictIdx = (int)(dictPtr - wlzStr->dictEnd);
		wlzStr->chain2Table[(Uint16)(dictIdx + WLZ_MATCH2_WINDOW )] = (Uint16)(dictIdx - wlzStr->hash2Table[hashV2]);
		wlzStr->hash1Table[ hashV1 ] = dictIdx;
		wlzStr->hash2Table[ hashV2 ] = dictIdx;
	}

	return dictSize;
}

void WLZhc_Attach_Dictionary(WLZhc_State_Str *workStr, const WLZhc_State_Str *dictStr)
{
	if (dictStr == NULL) return;

	memcpy(workStr->hash1Table, dictStr->hash1Table, (1 << WLZhc_HASH1BITS) * sizeof(int));
	memcpy(workStr->hash2Table, dictStr->hash2Table, (1 << WLZhc_HASH2BITS) * sizeof(int));
	memcpy(workStr->chain2Table, dictStr->chain2Table, WLZ_MATCH2_WINDOW * sizeof(short));

	workStr->dictSize = dictStr->dictSize;
	workStr->dictEnd = dictStr->dictEnd;
}

int WLZhc_Save_Dictionary(WLZhc_State_Str* dictStr, WLZhc_State_Str* workStr)
{
	int i;
	for (i = 0; !(i >> WLZhc_HASH1BITS); i++)
		dictStr->hash1Table[i] = workStr->hash1Table[i] == 0 ? 0 : workStr->hash1Table[i] - workStr->dictSize;

	for (i = 0; !(i >> WLZhc_HASH2BITS); i++)
		dictStr->hash2Table[i] = workStr->hash2Table[i] == 0 ? 0 : workStr->hash2Table[i] - workStr->dictSize;
	memcpy(dictStr->chain2Table, workStr->chain2Table, WLZ_MATCH2_WINDOW * sizeof(short));

	dictStr->dictEnd = workStr->dictEnd;
	dictStr->dictSize = workStr->dictSize;
	return dictStr->dictSize;
}

unsigned WLZhc_Compress(WLZhc_State_Str *wlzStr, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int level)
{

	unsigned destSizeBound = WLZ_COMPRESSBOUND(srcSize);
	if (destCapSize < destSizeBound) {
		return 0;
	}
	
	/* levels 0-7: lazy parsing, 8-12: optimal parsing; the far window by the level (WLZhc_LEVEL_WINDOW_LOG) */
	if (level < 0) level = 0;
	if (level > WLZhc_LEVEL_MAX) level = WLZhc_LEVEL_MAX;
	wlzStr->farWindow = 1u << WLZhc_LEVEL_WINDOW_LOG(level);
	/* the 8-byte far table of the lazy levels: one entry per 4 bytes of the input within the window, from 2^10 to
	   2^WLZhc_FAR8BITS */
	wlzStr->far8Bits = min(WLZhc_FAR8BITS, max(10, (int)High_Bit32(min(srcSize, wlzStr->farWindow) | 1) - 2));
	WLZhc_Init_State(wlzStr);

	if (level >= WLZhc_OPT_LEVEL_MIN) return WLZhc_Compress_Optimal(wlzStr, source, destiny, (int)srcSize, &WLZ_Opt_Level[level - WLZhc_OPT_LEVEL_MIN]);
	return WLZhc_Compress_Kernel(wlzStr, source, destiny, srcSize, Search_Level_Map[level]);
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ************************************************************************  Decompression Functions  ************************************************************************
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* the decoded size, from the block header (srcSize: the compressed size, for a length check only) */
unsigned WLZ_Read_DecSize(const char *source, unsigned srcSize)
{
	if (srcSize < 2) return 0;
	const unsigned lo = MemReadLE2(source);
	if (!(lo >> 15)) return lo;
	if (srcSize < 4) return 0;
	return (lo & ((1 << 15) - 1)) | ((unsigned)MemReadLE2(source + 2) << 15);
}

/* bytes of the block header */
static unsigned WLZ_Size_Bytes(const char *source)
{
	return (MemReadLE2(source) >> 15) ? 4 : 2;
}


#define WLZ_READ_ExtraLength(srcPtr, extLen)   {                \
	extLen = *srcPtr++;                                         \
	if ( unlikely(extLen > 251) ) {                             \
		token = extLen - 251;                                   \
		extLen = (Uint32)(252 + (MemReadLE4(srcPtr) & ByteMask[token]));    \
        srcPtr += token;                                        \
    }                                                           \
}

/* match copies of the decoder; each may write up to 31 bytes past dstEnd (WLZ_MEM_OVERHEAD) */
ForceInlineTemplate void WLZ_WildCopy32(Uint8* d, const Uint8* s, Uint8* const dstEnd)   /* offset >= 16 */
{
	do { memcpy(d, s, 16); memcpy(d + 16, s + 16, 16); d += 32; s += 32; } while (d < dstEnd);
}
ForceInlineTemplate void WLZ_WildCopy8(Uint8* d, const Uint8* s, Uint8* const dstEnd)    /* offset >= 8 */
{
	do { memcpy(d, s, 8); d += 8; s += 8; } while (d < dstEnd);
}
/* offsets 1-7: offsets 1, 2 and 4 repeat an 8-byte pattern; the others copy the first 8 bytes piecewise, after which
   the source trails by at least 8 */
ForceInlineTemplate void WLZ_Copy_Short_Offset(Uint8* d, const Uint8* s, Uint8* const dstEnd, const unsigned offset)
{
	static const int inc3table[8] = { 0, -2, -3,  -2, -3,  1, 1, 1 };
	static const int dec8table[8] = { 0, 0, 0, -1, 0,  1, 2, 3 };
	Uint8 v[8];
	if (offset == 1) memset(v, *s, 8);
	else if (offset == 2) { memcpy(v, s, 2); memcpy(v + 2, s, 2); memcpy(v + 4, v, 4); }
	else if (offset == 4) { memcpy(v, s, 4); memcpy(v + 4, s, 4); }
	else {
		d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
		s += 3 + inc3table[offset];
		memcpy(d + 4, s, 4);
		s -= dec8table[offset];
		WLZ_WildCopy8(d + 8, s, dstEnd);
		return;
	}
	do { memcpy(d, v, 8); d += 8; } while (d < dstEnd);
}

/* a match of offset <= produced + dictionary size: from the dictionary (a corrupt one may run on into the output), or
   from the output; the wild copies write up to 31 bytes past the match (WLZ_MEM_OVERHEAD) */
ForceInlineTemplate void WLZ_Copy_Match(Uint8* destPtr, const Uint8* const dest, const size_t produced, const unsigned offset,
	const unsigned matchLen, const Uint8* const dictEnd)
{
	if (offset > produced) {
		const size_t inDict = matchLen < offset - produced ? matchLen : offset - produced;
		memcpy(destPtr, dictEnd - (offset - produced), inDict);
		for (size_t k = inDict; k < matchLen; k++) destPtr[k] = dest[k - inDict];
		return;
	}
	const Uint8* const match = destPtr - offset;
	if (offset >= 16) WLZ_WildCopy32(destPtr, match, destPtr + matchLen);
	else if (offset >= 8) WLZ_WildCopy8(destPtr, match, destPtr + matchLen);
	else WLZ_Copy_Short_Offset(destPtr, match, destPtr + matchLen, offset);
}

/* the same copy, writing nothing past the match: for the end of the output */
ForceInlineTemplate void WLZ_Copy_Match_Exact(Uint8* destPtr, const Uint8* const dest, const size_t produced, const unsigned offset,
	const unsigned matchLen, const Uint8* const dictEnd)
{
	if (offset > produced) {
		WLZ_Copy_Match(destPtr, dest, produced, offset, matchLen, dictEnd);   /* its dictionary path is exact */
		return;
	}
	const Uint8* const match = destPtr - offset;
	if (offset >= matchLen) memcpy(destPtr, match, matchLen);
	else for (unsigned k = 0; k < matchLen; k++) destPtr[k] = match[k];
}

/* a length extension, reading nothing past srcEnd; returns 0 if it does not fit */
static int WLZ_Read_Ext_Exact(const Uint8** srcRef, const Uint8* const srcEnd, size_t* value)
{
	const Uint8* s = *srcRef;
	if (s >= srcEnd) return 0;
	size_t v = *s++;
	if (v > 251) {
		const unsigned n = (unsigned)v - 251;
		if ((size_t)(srcEnd - s) < n) return 0;
		Uint32 x = 0;
		for (unsigned k = 0; k < n; k++) x |= (Uint32)s[k] << (8 * k);
		v = 252 + (size_t)x;
		s += n;
	}
	*srcRef = s;
	*value = v;
	return 1;
}

#define WLZ_LIT_UNKNOWN   ((size_t)-1)

/* The last sequences, within WLZ_SRC_MARGIN bytes of the end of the input: every field is read exactly, as it may end
   the input. litLen: the literal run of the current sequence if its extension is already read, else WLZ_LIT_UNKNOWN.
   Returns the decoded size at the end marker, or 0 if the input is corrupt. */
static unsigned WLZ_Decode_Tail(const Uint8* srcPtr, const Uint8* const srcEnd, unsigned token, size_t litLen,
	Uint8* destPtr, Uint8* const dest, Uint8* const destEnd, const Uint8* const dictEnd, const size_t dictSize)
{
	while (1) {
		const unsigned code = token & ML_MASK;
		size_t ext, matchLen;
		unsigned offset;
		if (litLen == WLZ_LIT_UNKNOWN) {
			litLen = token >> ML_BITS;
			if (litLen == RUN_MASK) {
				if (!WLZ_Read_Ext_Exact(&srcPtr, srcEnd, &ext)) return 0;
				litLen += ext;
			}
		}
		if (litLen > (size_t)(srcEnd - srcPtr) || litLen > (size_t)(destEnd - destPtr)) return 0;
		memcpy(destPtr, srcPtr, litLen);
		srcPtr += litLen;
		destPtr += litLen;

		if (srcPtr >= srcEnd) return 0;
		if (code == 0) {                                 /* length 3: a one-byte offset */
			offset = *srcPtr++;
			matchLen = MIN_MATCH_LEN;
		}
		else {                                           /* a flagged offset: 1 + (code > 2) bytes, or one more */
			const size_t nOff = 1 + (code > 2) + (srcPtr[0] & 1);
			if ((size_t)(srcEnd - srcPtr) < nOff) return 0;
			offset = ((Uint32)srcPtr[0] | (nOff >= 2 ? (Uint32)srcPtr[1] << 8 : 0) | (nOff == 3 ? (Uint32)srcPtr[2] << 16 : 0)) >> 1;
			srcPtr += nOff;
			matchLen = MIN_MATCH_LEN + code;
			if (code == WLZ_CODE_LONG) {
				if (!WLZ_Read_Ext_Exact(&srcPtr, srcEnd, &ext)) return 0;
				matchLen += ext;
			}
		}
		const size_t produced = (size_t)(destPtr - dest);
		if (offset == 0 || offset > produced + dictSize)
			return offset == 0 && code != WLZ_CODE_LONG ? (unsigned)produced : 0;   /* the end marker, or corrupt */
		if (matchLen > (size_t)(destEnd - destPtr)) return 0;
		WLZ_Copy_Match_Exact(destPtr, dest, produced, offset, (unsigned)matchLen, dictEnd);
		destPtr += matchLen;
		if (srcPtr >= srcEnd) return 0;                  /* a block ends with the end marker */
		token = *srcPtr++;
		litLen = WLZ_LIT_UNKNOWN;
	}
}

#define WLZ_SRC_MARGIN    64      /* the main loop reads at most this far past a sequence's literal run */
#define WLZ_DST_MARGIN    40      /* a short sequence (up to 14 literals, a 17-byte match copied in 24) writes less */

/* The decoder checks every sequence, as LZ4's safe decoder does, at little cost to the common short sequence. The main
   loop runs while WLZ_SRC_MARGIN bytes of input and WLZ_DST_MARGIN bytes of output remain: the input margin covers
   every read past a literal run (16-byte copies, the offset word, a length extension, the next token), and the output
   margin a short sequence, so only a long literal run or a long match checks its length. Every offset is checked
   against the bytes decoded (and the dictionary). Near either end, WLZ_Decode_Tail finishes with exact checks.
   Nothing is written past dest + decSize: a long match within 32 bytes of the end is copied exactly. Returns the
   decoded size at the end marker, or 0 if the input is corrupt. Forced inline, so that a dictionary size of 0 removes
   its branches. */
ForceInlineTemplate unsigned
WLZ_Decompress_Kernel(
                 const Uint8* const src,
                 const Uint8* const srcEnd,
                 Uint8* const dest,
                 const unsigned decSize,
                 const Uint8* const dictEnd,
                 const size_t dictSize)
{
    const Uint8* srcPtr = src;
    Uint8* destPtr = dest;
    Uint8* const destEnd = dest + decSize;
    const Uint8* match;
    register unsigned token;
	register unsigned litLen, matchLen, offset, code, nb, adv;
	Uint32 word;

	if (srcPtr >= srcEnd) return 0;
	token = *srcPtr++;
	if ((size_t)(srcEnd - srcPtr) <= WLZ_SRC_MARGIN || decSize <= WLZ_DST_MARGIN)     /* short: all exact */
		return WLZ_Decode_Tail(srcPtr, srcEnd, token, WLZ_LIT_UNKNOWN, destPtr, dest, destEnd, dictEnd, dictSize);
	const Uint8* const srcFast = srcEnd - WLZ_SRC_MARGIN;
	Uint8* const destFast = destEnd - WLZ_DST_MARGIN;
    while (1) {
		if (unlikely(srcPtr > srcFast || destPtr > destFast))
			return WLZ_Decode_Tail(srcPtr, srcEnd, token, WLZ_LIT_UNKNOWN, destPtr, dest, destEnd, dictEnd, dictSize);
        litLen = token >> ML_BITS;    /* literal length */
		code = token & ML_MASK;       /* match code */

		if (unlikely(litLen == RUN_MASK)) {
			WLZ_READ_ExtraLength(srcPtr, litLen);
			litLen += RUN_MASK;
			/* a run that reaches within the margins of either end is finished exactly (the macro reuses token) */
			if (unlikely((size_t)litLen + WLZ_SRC_MARGIN > (size_t)(srcEnd - srcPtr) || litLen > (size_t)(destFast - destPtr)))
				return WLZ_Decode_Tail(srcPtr, srcEnd, code, litLen, destPtr, dest, destEnd, dictEnd, dictSize);
			MemWildCopy(destPtr + 16, srcPtr + 16, destPtr + litLen);
		}
		MemCopy16(destPtr, srcPtr);
		srcPtr += litLen;
		destPtr += litLen;

		/* the offset: one byte for code 0, else 1 + (code > 2) bytes or, by its flag (the low bit), one more. The 4
		   bytes read for it also hold the next token, unless a length extension follows: taking it from there keeps a
		   second load off the chain that each sequence waits on */
		/* adv: the offset bytes less one, (code > 2) + flag. It is on the chain to the next token, so it stays one add:
		   a three-term sum here compiles to a slow three-operand lea and costs a tenth of the decoding speed */
		nb = WLZ_CodeFlag[code];                         /* flagged */
		adv = WLZ_CodeBase[code];
		word = MemReadLE4(srcPtr);
		adv += word & nb;
		offset = (word >> nb) & WLZ_OffMask[adv + nb];
		token = (word >> (8 * adv + 8)) & 0xFF;
		srcPtr += adv + 2;
		matchLen = MIN_MATCH_LEN + code;
		if (unlikely(code == WLZ_CODE_LONG)) {           /* the extension, then the next token */
			srcPtr--;
			WLZ_READ_ExtraLength(srcPtr, litLen);
			matchLen += litLen;
			token = *srcPtr++;
			if (unlikely(matchLen > (size_t)(destEnd - destPtr))) return 0;         /* corrupt: past the end */
		}

		if (dictSize) {                                  /* the history includes the dictionary */
			if (unlikely((size_t)offset - 1 >= (size_t)(destPtr - dest) + dictSize))     /* before the history: */
				return offset == 0 && code != WLZ_CODE_LONG ? (unsigned)(destPtr - dest) : 0;   /* the end marker, or corrupt */
			if ((size_t)(destPtr - dest) < offset) {     /* starts in the dictionary */
				WLZ_Copy_Match(destPtr, dest, (size_t)(destPtr - dest), offset, matchLen, dictEnd);
				destPtr += matchLen;
				continue;
			}
		}
		else if (unlikely((uintptr_t)destPtr - (uintptr_t)dest < offset)) return 0;     /* corrupt: before the output */
		match = destPtr - offset;

		if (likely(code != WLZ_CODE_LONG)) {             /* at most 17 bytes */
			if (likely(offset >= 8)) {                   /* 8-byte copies, each from bytes already written */
				memcpy(destPtr, match, 8);
				memcpy(destPtr + 8, match + 8, 8);
				if (unlikely(matchLen > 16)) memcpy(destPtr + 16, match + 16, 8);   /* length 17 */
				destPtr += matchLen;
				continue;
			}
			if (unlikely(!offset)) return (unsigned)(destPtr - dest);                   /* the end marker */
			WLZ_Copy_Short_Offset(destPtr, match, destPtr + matchLen, offset);
			destPtr += matchLen;
			continue;
		}
		if (unlikely(!offset)) return 0;                 /* corrupt: a long match cannot end the block */
		if (unlikely(matchLen + 32 > (size_t)(destEnd - destPtr)))    /* the wild copies would pass the end */
			WLZ_Copy_Match_Exact(destPtr, dest, (size_t)(destPtr - dest), offset, matchLen, dictEnd);
		else if (offset >= 16) WLZ_WildCopy32(destPtr, match, destPtr + matchLen);
		else if (offset >= 8) WLZ_WildCopy8(destPtr, match, destPtr + matchLen);
		else WLZ_Copy_Short_Offset(destPtr, match, destPtr + matchLen, offset);
		destPtr += matchLen;
    }
}



/*===== Instantiate the API decoding functions. =====*/

FORCE_O2_GCC_PPC64LE
unsigned WLZ_Decompress(const char* source, char* destiny, unsigned compressedSize, unsigned decCapSize)
{
	if (source == NULL || destiny == NULL || compressedSize < 4) return 0;   /* header, token and the ending zero offset at least */
	const unsigned decSize = WLZ_Read_DecSize(source, compressedSize);
	if (decCapSize < decSize) return 0;
	const unsigned d = WLZ_Decompress_Kernel((const Uint8*)source + WLZ_Size_Bytes(source), (const Uint8*)source + compressedSize,
		(Uint8*)destiny, decSize, NULL, 0);
	return d == decSize ? d : 0;
}

FORCE_O2_GCC_PPC64LE
unsigned WLZ_Decompress_wDict(const char* source, char* destiny, unsigned compressedSize, unsigned decCapSize,
                                     const char* dictionary, unsigned dictSize)
{
	if (source == NULL || destiny == NULL || compressedSize < 4) return 0;
	const unsigned decSize = WLZ_Read_DecSize(source, compressedSize);
	if (decCapSize < decSize) return 0;
	if (dictionary == NULL) dictSize = 0;
	const unsigned d = WLZ_Decompress_Kernel((const Uint8*)source + WLZ_Size_Bytes(source), (const Uint8*)source + compressedSize,
		(Uint8*)destiny, decSize, (const Uint8*)dictionary + dictSize, dictSize);
	return d == decSize ? d : 0;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trusted mode (opt-in) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* The decoder the paper measured, unchanged: no checks at all, for input known to come unmodified from WLZ4's
   encoder. The stored size sizes the output (plus WLZ_MEM_OVERHEAD for wild copies); the input must stay readable
   WLZ_TRUSTED_SRC_PAD bytes past its end. A damaged stream can make it read or write out of bounds. */
ForceInlineTemplate unsigned
WLZ_Decompress_Kernel_Trusted(
                 const char* const src,
                 char* const dest,
                 const Uint8* const dictEnd,  
                 const unsigned dictSize)
{
	if (src == NULL) {
		return 0;
	}

    const Uint8* srcPtr = (const Uint8*) src;
    Uint8* destPtr = (Uint8*) dest;
    const Uint8* match;
    register unsigned token = *srcPtr++;
	register unsigned litLen, matchLen, offset, code, nb, adv;
	Uint32 word;

    while (1) {
        litLen = token >> ML_BITS;    /* literal length */
		code = token & ML_MASK;       /* match code */

		if (unlikely(litLen == RUN_MASK)) {
			WLZ_READ_ExtraLength(srcPtr, litLen);
			litLen += RUN_MASK;
			MemWildCopy(destPtr + 16, srcPtr + 16, destPtr + litLen);
		}
		MemCopy16(destPtr, srcPtr);
		srcPtr += litLen;
		destPtr += litLen;

		/* the offset: one byte for code 0, else 1 + (code > 2) bytes or, by its flag (the low bit), one more. The 4
		   bytes read for it also hold the next token, unless a length extension follows: taking it from there keeps a
		   second load off the chain that each sequence waits on */
		/* adv: the offset bytes less one, kept to one add on the chain to the next token (see the checked decoder) */
		nb = WLZ_CodeFlag[code];                         /* flagged */
		adv = WLZ_CodeBase[code];
		word = MemReadLE4(srcPtr);
		adv += word & nb;
		offset = (word >> nb) & WLZ_OffMask[adv + nb];
		token = (word >> (8 * adv + 8)) & 0xFF;
		srcPtr += adv + 2;
		matchLen = MIN_MATCH_LEN + code;

		if (dictSize && (ptrdiff_t)(destPtr - (const Uint8*)dest) < (ptrdiff_t)offset) {   /* in the dictionary, whole */
			match = dictEnd + ((ptrdiff_t)(destPtr - (const Uint8*)dest) - (ptrdiff_t)offset);
			if (unlikely(code == WLZ_CODE_LONG)) {
				srcPtr--;
				WLZ_READ_ExtraLength(srcPtr, litLen);
				matchLen += litLen;
				token = *srcPtr++;
			}
			memcpy(destPtr, match, matchLen);
			destPtr += matchLen;
			continue;
		}
		match = destPtr - offset;

		if (likely(code != WLZ_CODE_LONG)) {             /* at most 17 bytes */
			if (likely(offset >= 8)) {                   /* 8-byte copies, each from bytes already written */
				memcpy(destPtr, match, 8);
				memcpy(destPtr + 8, match + 8, 8);
				if (unlikely(matchLen > 16)) memcpy(destPtr + 16, match + 16, 8);   /* length 17 */
				destPtr += matchLen;
				continue;
			}
			if (unlikely(!offset)) return (unsigned)(destPtr - (const Uint8*)dest);
			WLZ_Copy_Short_Offset(destPtr, match, destPtr + matchLen, offset);
			destPtr += matchLen;
			continue;
		}
		srcPtr--;                                        /* the extension, then the next token */
		WLZ_READ_ExtraLength(srcPtr, litLen);
		matchLen += litLen;
		token = *srcPtr++;
		if (offset >= 16) WLZ_WildCopy32(destPtr, match, destPtr + matchLen);
		else if (offset >= 8) WLZ_WildCopy8(destPtr, match, destPtr + matchLen);
		else WLZ_Copy_Short_Offset(destPtr, match, destPtr + matchLen, offset);
		destPtr += matchLen;
    }

    /* end of decoding */
    return (unsigned) (destPtr- (const Uint8*)dest);     /* Nb of output bytes decoded */
}




FORCE_O2_GCC_PPC64LE
unsigned WLZ_Decompress_Trusted(const char* source, char* destiny, unsigned compressedSize, unsigned decCapSize)
{
	if (decCapSize < WLZ_Read_DecSize(source, compressedSize) + WLZ_MEM_OVERHEAD) {
		return 0;
	}

    	if (compressedSize < 4) return 0;                     /* header, token and the ending zero offset at least */
	return WLZ_Decompress_Kernel_Trusted(source + WLZ_Size_Bytes(source), destiny, NULL, 0);
}

FORCE_O2_GCC_PPC64LE
unsigned WLZ_Decompress_wDict_Trusted(const char* source, char* destiny, unsigned compressedSize, unsigned decCapSize,
                                     const char* dictionary, unsigned dictSize)
{
	if (decCapSize < WLZ_Read_DecSize(source, compressedSize) + WLZ_MEM_OVERHEAD) {
		return 0; 
	}

	const Uint8* const dictEnd = (const Uint8*)dictionary + dictSize;

		if (compressedSize < 4) return 0;
	return WLZ_Decompress_Kernel_Trusted(source + WLZ_Size_Bytes(source), destiny, dictEnd, dictSize);
}



#endif   /* WLZ_COMMONDEFS_ONLY */
