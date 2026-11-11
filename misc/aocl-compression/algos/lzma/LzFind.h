/* LzFind.h -- Match finder for LZ algorithms
2021-07-13 : Igor Pavlov : Public domain */

/**
* Modifications Copyright (C) 2022-23, Advanced Micro Devices. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from this
* software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __LZ_FIND_H
#define __LZ_FIND_H

#include "7zTypes.h"
/* 
    In Release mode assert statements are disabled by CMake,
    by passing NDEBUG flag, which disables assert statement.
    In Debug mode assert statements are automatically enabled.
 */
#include <assert.h>

#ifdef AOCL_LZMA_OPT
#include "LzHash.h"

#define kHashGuarentee (1 << 16) // Minimum number of blocks required in dictionary for cache efficient hash chains

/* Hash chain blocks are implemented using circular buffers. Format is as follows:    
* [hcHead | hcChain................]
*           <----HASH_CHAIN_MAX--->
*  <-------HASH_CHAIN_SLOT_SZ----->
hcChain part is a circular buffer */
#define HASH_CHAIN_MAX_8 7 // max length of hash-chain in a block
#define HASH_CHAIN_SLOT_SZ_8 (HASH_CHAIN_MAX_8+1) // head_ptr and hash-chain
#define HASH_CHAIN_16_LEVEL 2 //for levels < this use HASH_CHAIN_SLOT_SZ_8, rest HASH_CHAIN_SLOT_SZ_16

#define HASH_CHAIN_MAX_16 15 // max length of hash-chain in a block
#define HASH_CHAIN_SLOT_SZ_16 (HASH_CHAIN_MAX_16+1) // head_ptr and hash-chain
#endif

/* Default strategy to enable cache efficient hash chain implementation:
* input_size < MAX_SIZE_FOR_CE_HC_OFF : disabled
* MAX_SIZE_FOR_CE_HC_OFF <= input_size < MIN_SIZE_FOR_CE_HC_ON : enabled/disabled based on other settings
* input_size >= MIN_SIZE_FOR_CE_HC_ON : enabled */
#define MAX_SIZE_FOR_CE_HC_OFF (32 * 1024) // 32KB
#define MIN_SIZE_FOR_CE_HC_ON (kHashGuarentee * HASH_CHAIN_SLOT_SZ_8) //512KB


// Condition to use cache efficient hash chains
#define USE_CACHE_EFFICIENT_HASH_CHAIN (!p->btMode && p->cacheEfficientSearch)

EXTERN_C_BEGIN

typedef UInt32 CLzRef;

/* descriptions for hash table parameters below are based on
 * 4 - byte associated hash currently being used in source code*/
typedef struct _CMatchFinder
{
  Byte *buffer; // pointer to input data stream
  UInt32 pos;
  UInt32 posLimit;
  UInt32 streamPos;  /* wrap over Zero is allowed (streamPos < pos). Use (UInt32)(streamPos - pos) */
  UInt32 lenLimit;

  UInt32 cyclicBufferPos;  // these 'pos' values are saved in hash. range [0, cyclicBufferSize)
  UInt32 cyclicBufferSize; /* it must be = (historySize + 1) */

  Byte streamEndWasReached;
  Byte btMode;
  Byte bigHash;
  Byte directInput;

  UInt32 matchMaxLen;
  /* hash and son point to contiguous memory locations
* hash : hash tables
* son  : binary tree or hash chain
* hash                                          son
* ---------------------------------------------------------------------------
* | 2-byte table | 3-byte table | 4-byte table |
* ---------------------------------------------------------------------------
* <------fixedHashSize--------->
* <---------------hashSizeSum------------------><-------numSons-------------->
* numSons = cyclicBufferSize [for HC] and 2*cyclicBufferSize [for BT] algos
* numSons can be of fixed size, as max number of nodes that can be
* present in the dictionary at any given point = historySize */
  CLzRef *hash; // table with heads of hash-chains or roots of BSTs in the dictionary
  CLzRef *son; // dictionary
  UInt32 hashMask; // determines the max number of bits in the hash
  UInt32 cutValue; // hard limit on number of nodes to look at in BT/HC when searching for matches
#ifdef AOCL_LZMA_OPT
  UInt16 level; // 0 <= level <= 9
  /* Value of cacheEfficientSearch determines the function used for dictionary search.
   * -----------------------------------------------------------------------|--------------------------|------------------------------------------------|
   * Conditions when method is called                                       |Method                    | Description                                    |
   * -----------------------------------------------------------------------|--------------------------|------------------------------------------------|
   * cacheEfficientSearch = 1, algo = !btMode, level  < HASH_CHAIN_16_LEVEL |AOCL_Hc_GetMatchesSpec_8  | Cache efficient hash chains with block size 8  |
   * cacheEfficientSearch = 1, algo = !btMode, level >= HASH_CHAIN_16_LEVEL |AOCL_Hc_GetMatchesSpec_16 | Cache efficient hash chains with block size 16 |
   * cacheEfficientSearch = 0, algo = !btMode,                              |AOCL_Hc_GetMatchesSpec    | Reference style intervowen hash chains         | */
  UInt16 cacheEfficientSearch; // 0: disabled, 1: cache efficient hash chains
#endif
  Byte *bufferBase;
  ISeqInStream *stream;
  
  UInt32 blockSize;
  UInt32 keepSizeBefore;
  UInt32 keepSizeAfter;

  UInt32 numHashBytes; // default 4 
  size_t directInputRem;
  UInt32 historySize; // size of search buffer aka dictionary
  UInt32 fixedHashSize; // size for all 2-byte, 3-byte associated hash tables
  UInt32 hashSizeSum; // size for all hash tables combined (2-byte, 3-byte, 4-byte)
  SRes result;
  UInt32 crc[256]; // crc used in hash calculation
  size_t numRefs; // total size of memory allocated for all hash tables and dictionary

  UInt64 expectedDataSize;
} CMatchFinder;

#define Inline_MatchFinder_GetPointerToCurrentPos(p) ((const Byte *)(p)->buffer)

#define Inline_MatchFinder_GetNumAvailableBytes(p) ((UInt32)((p)->streamPos - (p)->pos))

/*
#define Inline_MatchFinder_IsFinishedOK(p) \
    ((p)->streamEndWasReached \
        && (p)->streamPos == (p)->pos \
        && (!(p)->directInput || (p)->directInputRem == 0))
*/
      
int MatchFinder_NeedMove(CMatchFinder *p);
/* Byte *MatchFinder_GetPointerToCurrentPos(CMatchFinder *p); */
void MatchFinder_MoveBlock(CMatchFinder *p);
void MatchFinder_ReadIfRequired(CMatchFinder *p);

void MatchFinder_Construct(CMatchFinder *p);

/* Conditions:
     historySize <= 3 GB
     keepAddBufferBefore + matchMaxLen + keepAddBufferAfter < 511MB
*/
int MatchFinder_Create(CMatchFinder *p, UInt32 historySize,
    UInt32 keepAddBufferBefore, UInt32 matchMaxLen, UInt32 keepAddBufferAfter,
    ISzAllocPtr alloc);
void MatchFinder_Free(CMatchFinder *p, ISzAllocPtr alloc);
void MatchFinder_Normalize3(UInt32 subValue, CLzRef *items, size_t numItems);
// void MatchFinder_ReduceOffsets(CMatchFinder *p, UInt32 subValue);

/*
#define Inline_MatchFinder_InitPos(p, val) \
    (p)->pos = (val); \
    (p)->streamPos = (val);
*/

#define Inline_MatchFinder_ReduceOffsets(p, subValue) \
    (p)->pos -= (subValue); \
    (p)->streamPos -= (subValue);


UInt32 * GetMatchesSpec1(UInt32 lenLimit, UInt32 curMatch, UInt32 pos, const Byte *buffer, CLzRef *son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 _cutValue,
    UInt32 *distances, UInt32 maxLen);

/*
Conditions:
  Mf_GetNumAvailableBytes_Func must be called before each Mf_GetMatchLen_Func.
  Mf_GetPointerToCurrentPos_Func's result must be used only before any other function
*/

typedef void (*Mf_Init_Func)(void *object);
typedef UInt32 (*Mf_GetNumAvailableBytes_Func)(void *object);
typedef const Byte * (*Mf_GetPointerToCurrentPos_Func)(void *object);
typedef UInt32 * (*Mf_GetMatches_Func)(void *object, UInt32 *distances);
typedef void (*Mf_Skip_Func)(void *object, UInt32);

typedef struct _IMatchFinder
{
  Mf_Init_Func Init;
  Mf_GetNumAvailableBytes_Func GetNumAvailableBytes;
  Mf_GetPointerToCurrentPos_Func GetPointerToCurrentPos;
  Mf_GetMatches_Func GetMatches;
  Mf_Skip_Func Skip;
} IMatchFinder2;

void MatchFinder_CreateVTable(CMatchFinder *p, IMatchFinder2 *vTable);

void MatchFinder_Init_LowHash(CMatchFinder *p);
void MatchFinder_Init_HighHash(CMatchFinder *p);
void MatchFinder_Init_4(CMatchFinder *p);
void MatchFinder_Init(CMatchFinder *p);

UInt32* Bt3Zip_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances);
UInt32* Hc3Zip_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances);

void Bt3Zip_MatchFinder_Skip(CMatchFinder *p, UInt32 num);
void Hc3Zip_MatchFinder_Skip(CMatchFinder *p, UInt32 num);

#ifdef AOCL_LZMA_OPT
/* Conditions:
     historySize <= 3 GB
     keepAddBufferBefore + matchMaxLen + keepAddBufferAfter < 511MB
*/
int AOCL_MatchFinder_Create(CMatchFinder* p, UInt32 historySize,
  UInt32 keepAddBufferBefore, UInt32 matchMaxLen, UInt32 keepAddBufferAfter,
  ISzAllocPtr alloc);
void AOCL_MatchFinder_Free(CMatchFinder* p, ISzAllocPtr alloc);
void AOCL_MatchFinder_Init(CMatchFinder* p);
void AOCL_MatchFinder_CreateVTable(CMatchFinder* p, IMatchFinder2* vTable);
#endif
EXTERN_C_END

#define GetUi16(p) (*(const UInt16 *)(const void *)(p))
#define GetUi32(p) (*(const UInt32 *)(const void *)(p))
#define SetUi16(p, v) { *(UInt16 *)(void *)(p) = (v); }
#define SetUi32(p, v) { *(UInt32 *)(void *)(p) = (v); }

#ifdef AOCL_UNIT_TEST
#ifdef AOCL_LZMA_OPT
EXTERN_C_BEGIN
LZMALIB_API void Test_HC_MatchFinder_Normalize3(UInt32 subValue, CLzRef* hash, CLzRef* son,
    Byte btMode, UInt32 fixedHashSize, UInt32 cyclicBufferSize, UInt32 hashSizeSum,
    size_t numRefs, UInt32 hashMask, int level);

LZMALIB_API void Test_AOCL_HC_MatchFinder_Normalize3(UInt32 subValue, CLzRef* hash, CLzRef* son,
    Byte btMode, UInt32 fixedHashSize, UInt32 cyclicBufferSize, UInt32 hashSizeSum,
    size_t numRefs, UInt32 hashMask, int level);

LZMALIB_API int Test_AOCL_Find_Matching_Bytes_Len(int startLen, Byte* data1, Byte* data2, int limit);

LZMALIB_API UInt32 Test_Compute_Hash_Mask(UInt32 sz, UInt32 block_cnt);

LZMALIB_API UInt32 Test_Compute_Hash(Byte* cur, CMatchFinder* p);

LZMALIB_API UInt32 Test_Circular_Inc(UInt32 hcHead, UInt32 HASH_CHAIN_SLOT_SZ, UInt32 HASH_CHAIN_MAX);

LZMALIB_API UInt32 Test_Circular_Dec(UInt32 hcHead, UInt32 HASH_CHAIN_SLOT_SZ, UInt32 HASH_CHAIN_MAX);

LZMALIB_API UInt32 Test_Hc_GetMatchesSpec(size_t lenLimit, UInt32 hcHead, UInt32 hv, UInt32 pos,
    const Byte* cur, CLzRef* son, size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 cutValue, UInt32* d, unsigned maxLen, int blockSz);
EXTERN_C_END
#endif /* AOCL_LZMA_OPT */
#endif /* AOCL_UNIT_TEST */
#endif
