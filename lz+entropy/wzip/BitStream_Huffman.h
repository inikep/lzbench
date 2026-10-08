/*
 * WZIP - bit streams and Huffman table interface
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

#if defined (__cplusplus)
extern "C" {
#endif

#ifndef __BITSTREAM__HUFFMAN__1983827168213
#define __BITSTREAM__HUFFMAN__1983827168213

#include <stdlib.h>   /* malloc, free, qsort */
#include <assert.h>
#include "Memry.h"

#define Max(a,b) (((a) > (b)) ? (a) : (b))
#define Min(a,b) (((a) < (b)) ? (a) : (b))
#ifndef min
#  define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#  define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

static const unsigned BitMask[32] = {
	0,          1,         3,         7,         0xF,       0x1F,
	0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
	0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
	0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
	0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
	0x3FFFFFFF, 0x7FFFFFFF }; /* up to 31 bits */


#define MAX_HufSize  256                     /* Maximum number of Huffman states */
#define MAX_HufWeight  12                    /* Max (forced) Huffman weight */
#define MAX_HufHufWt   7                     /* Max second-level Huffman weight */   
#define MinStream4XSize  512
#define HUF_BlockSize    (1<<15)

/* The container holds 64 bits on every platform (BIT_CONTAINER_BITS), whatever the machine word, so that 32-bit
   builds write and read the same streams. */
#define BIT_CONTAINER_BITS  64

/* Bit position starts from highest to lowest, i.e., 63 downward to 0. 
   This setup is convenient for Huffman decoding wherein highest bits are used for one-shot Huffman decoding */
typedef struct {
	Uint64 container;
	Uint32 nUsedBits;
	Uint8* streamPtr;
} Bit_Stream;

typedef struct {
	Uint32 freq;                                       /* frequencies, later converted into Huffman bit length */
	Uint16 lit;                                        /* To make the overall size multiple of 4 */
	Uint16 nbits;
} Huffman_Str;

typedef struct {
	Uint16 code;         /* Huffman prefix code representation */
	Uint16 nbits;        /* bit width of each code,  it is inflated to 2-byte to optimize element alignment */
} HufCode_Str; 

typedef struct {
	Uint16 code;         /* Huffman prefix code representation */
	Uint8 nbits;        /* bit width of each code */
	Uint8 lsBits;       /* uncoded least significant bits */
} ExtHufCode_Str;

typedef struct {
	Uint8 lit;
	Uint8  nbits;
} Huffman_DemapX1;

typedef struct {
	Uint8  lit[2];
	Uint8  nbits;
	Uint8  length;
} Huffman_DemapX2;

typedef struct {
	Uint8 msValue;     /* value in  most significant bits used for Huffman coding*/
	Uint8 lsBits;      /* number of uncoded least significant bits */
} ExtHuffman_Lit;      /* literal representation for extended Huffman coding */

typedef struct {
	Uint8 hufIdx;
	Uint8 lsBits;
} ExtHuffman_Idx;

typedef struct {
	ExtHuffman_Lit extLit;
	Uint16  nbits;
} ExtHuffman_DemapX1;


//This sorting algorithm takes short cycles but is relatively complex in hardware
static inline Uint32 Fast_Sort_Width(Uint8* hufCodeBits, const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Huffman_DemapX1* sortHufCode, Uint32 *cumFreq)
{
	Uint32 i, k;
	Uint32 hufWtFreq[MAX_HufWeight + 1] = { 0 };   /* frequency of hufCodeBits */

	for (i = 0; i < hufCodeSize; i++)
		hufWtFreq[hufCodeBits[i]]++;
	hufWtFreq[0] = 0;   // skip zero-count

	cumFreq[1] = 0;
	for (i = 2; i <= maxHufCodeBits+1; i++) {
		cumFreq[i] = cumFreq[i-1] + hufWtFreq[i-1];
	}
	cumFreq[0] = cumFreq[maxHufCodeBits + 1];    // to place zero-freqency literals at the end

	/* radix sorting literals in increasing order of litHufBits, while maintaining the original order for ones with equal litHufBits*/
	for (i = 0; i < hufCodeSize; i++) {
		k = cumFreq[ hufCodeBits[i] ]++;
		sortHufCode[k].lit = (Uint8)i;
		sortHufCode[k].nbits = hufCodeBits[i];
	}

	return cumFreq[maxHufCodeBits + 1];   // return the effective number of literals
}

/*It must guarantee sym is under nbits bits, and nbits is less than bitPos */
#define BITStream_Write(bitStream, sym, nbits)   {                                   \
	assert((sym) < (Uint32)(1 << (nbits)) );                                         \
	assert((nbits) <= BIT_CONTAINER_BITS - bitStream.nUsedBits);                     \
	bitStream.nUsedBits += (nbits);                                                  \
	bitStream.container ^= (Uint64)(sym) << (BIT_CONTAINER_BITS - bitStream.nUsedBits); \
}

#define BITStream_Write_Flush(bitStream)  {                  \
	assert(bitStream.nUsedBits <= BIT_CONTAINER_BITS);       \
	Uint32 nBytes = bitStream.nUsedBits >> 3;                \
	MemWriteBE8(bitStream.streamPtr, bitStream.container);   \
	bitStream.streamPtr += nBytes;                           \
	bitStream.container <<= (Uint32)(nBytes * 8);            \
	bitStream.nUsedBits &=  7;                               \
}

/* closing of bit-stream write */
#define BITStream_Write_FlushEnd(bitStream)  {               \
	int nBytes = (7+bitStream.nUsedBits) >> 3;               \
	MemWriteBE8(bitStream.streamPtr, bitStream.container);  \
	bitStream.streamPtr += nBytes;                           \
}


#define BITStream_Read(bitStream, nbits, sym)   {                                               \
	assert(BIT_CONTAINER_BITS - bitStream.nUsedBits >= (nbits));                                 \
	sym = (Uint32)( bitStream.container<< bitStream.nUsedBits >> (BIT_CONTAINER_BITS - (nbits)) ); \
	bitStream.nUsedBits += (nbits);                                                             \
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Huffman Coding/Decoding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Note: remMaxHufBits = BIT_CONTAINER_BITS - maxHufCodeBits */
#define BITStream_Read_HufX0(bitStream, remMaxHufBits, hufCodeBits, hufCodeDemap, symPtr)   {       \
	assert(remMaxHufBits >= bitStream.nUsedBits);                                                \
	Uint32 const r =  (Uint32)( bitStream.container << bitStream.nUsedBits >> remMaxHufBits );   \
	*symPtr = hufCodeDemap[r];                                                                       \
	bitStream.nUsedBits += hufCodeBits[*symPtr++];                                                       \
}

#define BITStream_Read_HufX1(bitStream, remMaxHufBits, hufCodeDemapX1, symPtr)   {                    \
	assert( remMaxHufBits >= bitStream.nUsedBits );                                               \
	Uint32 const r = (Uint32)( (bitStream.container << bitStream.nUsedBits) >> remMaxHufBits );    \
	*symPtr++ = hufCodeDemapX1[r].lit;                                                                   \
	bitStream.nUsedBits += hufCodeDemapX1[r].nbits;                                                \
}

#define BITStream_Read_ExtHufX1(bitStream, remMaxHufBits, hufCodeDemapX1, sym)   {                 \
	assert(remMaxHufBits >= bitStream.nUsedBits);                                                  \
	Uint32 const r = (Uint32)((bitStream.container << bitStream.nUsedBits) >> remMaxHufBits);      \
	sym = hufCodeDemapX1[r].lit;                                                                \
	bitStream.nUsedBits += hufCodeDemapX1[r].nbits;                                                \
}

/* Note it returns the length of literals, i.e., 1 or 2, which differs from X1 look-up which returns decompressed symbol */
#define BITStream_Read_HufX2(destPtr, bitStream, remHufDemapBitsX2, hufCodeDemapX2)    {             \
	assert( remHufDemapBitsX2 >= bitStream.nUsedBits );                                              \
	Uint32 const r = (Uint32)( (bitStream.container << bitStream.nUsedBits) >> remHufDemapBitsX2 );  \
	memcpy(destPtr, hufCodeDemapX2[r].lit, 2);                                                      \
	bitStream.nUsedBits += hufCodeDemapX2[r].nbits;                                                  \
	destPtr += hufCodeDemapX2[r].length;                                                             \
}

/* Read single byte at the end to avoid buffer overflow (alternatively, data overwrite) */
#define BITStream_Read_HufEndX2(destPtr, bitStream, remHufDemapBitsX2, hufCodeDemapX2)  {            \
	assert(remHufDemapBitsX2 >= bitStream.nUsedBits);                                                \
	Uint32 const r = (Uint32)((bitStream.container << bitStream.nUsedBits) >> remHufDemapBitsX2);    \
	*destPtr++ = hufCodeDemapX2[r].lit[0];                                                           \
	bitStream.nUsedBits += hufCodeDemapX2[r].nbits;                                                  \
}


#define BITStream_Read_Flush(bitStream)   {                           \
	bitStream.streamPtr += (Uint32)(bitStream.nUsedBits >> 3);        \
	bitStream.container = MemReadBE8(bitStream.streamPtr);            \
	bitStream.nUsedBits = bitStream.nUsedBits & 7;                    \
}

/* Closing of Bit-Stream read */
#define BITStream_Read_FlushEnd(bitStream)                  \
	bitStream.streamPtr += (7+bitStream.nUsedBits) >> 3



int Build_Huffman_Table(Huffman_Str* hufStr, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str* hufCode);
void Write_Huffman_Header(Bit_Stream* bitStream, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str* hufStr);
int Count_Huffman_Weight_Frequency(HufCode_Str* hufStr, const Uint32 hufCodeSize, Huffman_Str* hufWtHufStr, Uint8* hufWtSeq);
void Write_Huffman_Header_byHuffman(Bit_Stream* bitStream, HufCode_Str* hufLenHufStr, Uint8* hufWtSeq, int seqSize);

Uint32 Huffman_Compress1X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf);
Uint32 Huffman_Compress4X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf);
Uint32 Huffman_Compress_Block(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits, Uint32 litHufCapBits);
Uint32 Huffman_Compress(const void* srcStart, Uint32 srcSize, void* dest, Uint32 nLits, Uint32 litHufCapBits);

#define HUF_CODE_CORRUPT  0xFFu           /* returned by the header readers for a repeat run past the table's end */
Uint32 Read_Huffman_Header(Bit_Stream* bitStream, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, Uint8* hufCodeBits);
Uint32 Read_Huffman_Header_byHuffman(Bit_Stream* bitStream, Uint32 maxHufWtHufCodeBits, Huffman_DemapX1* hufWtHufCodeDemapX1, const Uint32 hufCodeSize, Uint8* hufCodeBits);
int Huffman_Check_Code(const Uint8* hufCodeBits, const Uint32 nSym, const Uint32 capBits);
int Huffman_Read_Code(Bit_Stream* bitStream, const Uint32 nSym, const Uint32 capBits, Uint8* hufCodeBits);
int Huffman_Read_Code_byHuffman(Bit_Stream* bitStream, const Uint32 maxWtBits, Huffman_DemapX1* wtDemap, const Uint32 nSym,
	const Uint32 capBits, Uint8* hufCodeBits);
Uint32 Huffman_Build_SafeX1(const Uint32 nSym, const int maxBits, Uint8* hufCodeBits, Huffman_DemapX1* table);

void Build_Huffman_DecTableX0(const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Uint8* hufCodeBits, Uint8* hufCodeDemap);
void Build_Huffman_DecTableX1(const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Uint8* hufCodeBits, Huffman_DemapX1* hufDemapX1);
void Build_Huffman_DecTableX2(const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Uint8* hufCodeBits, Huffman_DemapX2* hufDecTableX2, const Uint32 hufTableBits);
void Build_ExtHuffman_DecTableX1(const Uint32 hufCodeSize, const Uint32 maxHufCodeBits, Uint8* hufCodeBits, const ExtHuffman_Lit* extHufLit, ExtHuffman_DemapX1* litHufDemapX1);

/* A literal block is stored (type 0), carries the lengths of its own code (type 1), or reuses the code of the
   stream's last type-1 block (type 2). The encoder keeps that code in a Huffman_Prev, the decoder in a
   Huffman_DecState, with the decoding table built from it, which blocks of type 2 then use as it is. */
#define HUF_HeaderBound  512                 /* a literal block's code lengths take fewer bytes than this */
typedef struct {
	HufCode_Str code[MAX_HufSize];
	int valid;                               /* a type-1 block has set it */
} Huffman_Prev;
typedef struct {
	Uint8 bits[MAX_HufSize];                 /* the code's lengths */
	Uint32 maxBits;
	int valid;                               /* a type-1 block has set them */
	int table;                               /* the table built from them: 0 none yet, 1 x1, 2 x2 */
	Uint32 tableBits;                        /* x2's lookup width */
	Huffman_DemapX1 x1[1 << MAX_HufWeight];
	Huffman_DemapX2 x2[4 + (1 << MAX_HufWeight)];
} Huffman_DecState;
Uint64 Huffman_Code_Bits(const Huffman_Str* h, const HufCode_Str* code, const Uint32 n);
int Log2_Price(const Uint64 total, const Uint64 d);
Uint32 Huffman_Header_Bits(const HufCode_Str* hufLenHufStr, const Uint8* hufWtSeq, int seqSize);
Uint32 Huffman_Compress_Block_Rep(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits,
	Uint32 litHufCapBits, Huffman_Prev* prev);

int Huffman_Select_Decompressor(Uint32 cmprSize, Uint32 srcSize, Uint32 maxHufBits, Uint32* decTabBitsX2);
int Huffman_Decompress(const void* source, const int srcSize, void* dest, Uint32 destSize, int nLits);
Uint32 Huffman_Decompress_Trusted(const void* source, void* dest, Uint32 destSize, int nLits);   /* no checks: trusted mode */
int Huffman_Skip(const void* source, const int srcSize, Uint32 destSize, int nLits);
Uint32 Huffman_Skip_Trusted(const void* source, Uint32 destSize, int nLits);
int Huffman_Decompress_Next(const void* src, const void* srcEnd, void* dest, Uint32 destSize, int nLits, Huffman_DecState* st);
Uint32 Huffman_Decompress_Next_Trusted(const void* src, void* dest, Uint32 destSize, int nLits, Huffman_DecState* st);
void Huffman_Decompress_Block_Body(Uint8* litHufCodeBits, int nLits, Uint32 maxLitHufBits, int algId, Uint8* cmprBuffer, Uint32 cmprSize, Uint8* decBuffer, const Uint32 decSize);

void Huffman_Tester(); 
int WZIP_fullSpeedBench(const int compressAlg);

#endif

#if defined (__cplusplus)
}
#endif