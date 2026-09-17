/* LzFind.c -- Match finder for LZ algorithms
2021-11-29 : Igor Pavlov : Public domain */

/**
* Modifications Copyright (C) 2022-2024, Advanced Micro Devices. All rights reserved.
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

/*------------------------------------------------------------------------
* Summary of all variants of AOCL dictionary search mechanisms implemented
* ------------------------------------------------------------------------
* Method                    | Description                                    | Default for levels    | Conditions when method is called                                                    |
* --------------------------|------------------------------------------------|-----------------------|-------------------------------------------------------------------------------------|
* AOCL_Hc_GetMatchesSpec_8  | Cache efficient hash chains with block size 8  | 0-1, large input      | algo = !btMode, level  < HASH_CHAIN_16_LEVEL, cacheEfficientSearch = 1              |
* AOCL_Hc_GetMatchesSpec_16 | Cache efficient hash chains with block size 16 | 2-4, large input      | algo = !btMode, level >= HASH_CHAIN_16_LEVEL, cacheEfficientSearch = 1              |
* AOCL_Hc_GetMatchesSpec    | Reference style intervowen hash chains         | 0-4, small/mid input  | algo = !btMode,                               cacheEfficientSearch = 0              |
* AOCL_GetMatchesSpec1      | Reference style intervowen binary search trees | 5-9                   | algo =  btMode                                                                      |
* AOCL_SkipMatchesSpec      | Reference style intervowen binary search trees | 5-9                   | algo =  btMode, skip operation. Add node and re-balance tree without searching dict |
* 
* Non-default settings: If numHashBytes is set to 4 and level is < 5
* AOCL_Hc_GetMatchesSpec_8  | Cache efficient hash chains with block size 8  | 0-1, large/mid input  | algo = !btMode, level  < HASH_CHAIN_16_LEVEL, cacheEfficientSearch = 1              |
* AOCL_Hc_GetMatchesSpec_16 | Cache efficient hash chains with block size 16 | 2-4, large/mid input  | algo = !btMode, level >= HASH_CHAIN_16_LEVEL, cacheEfficientSearch = 1              |
* AOCL_Hc_GetMatchesSpec    | Reference style intervowen hash chains         | 0-4, small input      | algo = !btMode,                               cacheEfficientSearch = 0              |
* ------------------------------------------------------------------------*/

#include "Precomp.h"

#include <string.h>

#include "LzFind.h"
#include "LzHash.h"
#include "algos/common/aoclHashChain.h"
#include "utils/utils.h"

#define kBlockMoveAlign       (1 << 7)    // alignment for memmove()
#define kBlockSizeAlign       (1 << 16)   // alignment for block allocation
#define kBlockSizeReserveMin  (1 << 24)   // it's 1/256 from 4 GB dictinary

#define kEmptyHashValue 0

#ifdef AOCL_EXTEND_CODE_COVERAGE
    /* Set smaller limit for pos roll over. If not it happens only at UINT_MAX.
    This allows testing of Normalize functions which are not hit otherwise */
    #define kMaxValForNormalize ((UInt32)(1 << 20) + 0xFFF) // for debug
#else
    #define kMaxValForNormalize ((UInt32)0) // UInt32 pos overflows beyond capacity (upon UINT_MAX++)
#endif

// #define kNormalizeAlign (1 << 7) // alignment for speculated accesses

#define GET_AVAIL_BYTES(p) \
  Inline_MatchFinder_GetNumAvailableBytes(p)


// #define kFix5HashSize (kHash2Size + kHash3Size + kHash4Size)
#define kFix5HashSize kFix4HashSize

/*
 HASH2_CALC:
   if (hv) match, then cur[0] and cur[1] also match
*/
#define HASH2_CALC hv = GetUi16(cur);

// (crc[0 ... 255] & 0xFF) provides one-to-one correspondence to [0 ... 255]

/*
 HASH3_CALC:
   if (cur[0]) and (h2) match, then cur[1]            also match
   if (cur[0]) and (hv) match, then cur[1] and cur[2] also match
*/
#define HASH3_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  hv = (temp ^ ((UInt32)cur[2] << 8)) & p->hashMask; }

#define HASH4_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  temp ^= ((UInt32)cur[2] << 8); \
  h3 = temp & (kHash3Size - 1); \
  hv = (temp ^ (p->crc[cur[3]] << kLzHash_CrcShift_1)) & p->hashMask; }

#ifdef AOCL_LZMA_OPT
#define kEmptySonValue 0

// Change wrt HASH4_CALC: 3-byte fixed table not used. h3 not computed.
#define AOCL_HASH4_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  temp ^= ((UInt32)cur[2] << 8); \
  hv = (temp ^ (p->crc[cur[3]] << kLzHash_CrcShift_1)) & p->hashMask; }

// Change wrt HASH5_CALC 3-byte fixed table not used. h3 not computed.
#define AOCL_HASH5_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  temp ^= ((UInt32)cur[2] << 8); \
  temp ^= (p->crc[cur[3]] << kLzHash_CrcShift_1); \
  hv = (temp ^ (p->crc[cur[4]] << kLzHash_CrcShift_2)) & p->hashMask; }
#endif

#define HASH5_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  temp ^= ((UInt32)cur[2] << 8); \
  h3 = temp & (kHash3Size - 1); \
  temp ^= (p->crc[cur[3]] << kLzHash_CrcShift_1); \
  /* h4 = temp & p->hash4Mask; */ /* (kHash4Size - 1); */ \
  hv = (temp ^ (p->crc[cur[4]] << kLzHash_CrcShift_2)) & p->hashMask; }

#define HASH_ZIP_CALC hv = ((cur[2] | ((UInt32)cur[0] << 8)) ^ p->crc[cur[1]]) & 0xFFFF;

static void LzInWindow_Free(CMatchFinder *p, ISzAllocPtr alloc)
{
  if (!p->directInput)
  {
    ISzAlloc_Free(alloc, p->bufferBase);
    p->bufferBase = NULL;
  }
}


static int LzInWindow_Create2(CMatchFinder *p, UInt32 blockSize, ISzAllocPtr alloc)
{
  if (blockSize == 0)
    return 0;
  if (!p->bufferBase || p->blockSize != blockSize)
  {
    // size_t blockSizeT;
    LzInWindow_Free(p, alloc);
    p->blockSize = blockSize;
    // blockSizeT = blockSize;
    
    // printf("\nblockSize = 0x%x\n", blockSize);
    /*
    #if defined _WIN64
    // we can allocate 4GiB, but still use UInt32 for (p->blockSize)
    // we use UInt32 type for (p->blockSize), because
    // we don't want to wrap over 4 GiB,
    // when we use (p->streamPos - p->pos) that is UInt32.
    if (blockSize >= (UInt32)0 - (UInt32)kBlockSizeAlign)
    {
      blockSizeT = ((size_t)1 << 32);
      printf("\nchanged to blockSizeT = 4GiB\n");
    }
    #endif
    */
    
    p->bufferBase = (Byte *)ISzAlloc_Alloc(alloc, blockSize);
    // printf("\nbufferBase = %p\n", p->bufferBase);
    // return 0; // for debug
  }
  return (p->bufferBase != NULL);
}

static const Byte *MatchFinder_GetPointerToCurrentPos(CMatchFinder *p) { return p->buffer; }

static UInt32 MatchFinder_GetNumAvailableBytes(CMatchFinder *p) { return GET_AVAIL_BYTES(p); }


MY_NO_INLINE
static void MatchFinder_ReadBlock(CMatchFinder *p)
{
  if (p->streamEndWasReached || p->result != SZ_OK)
    return;

  /* We use (p->streamPos - p->pos) value.
     (p->streamPos < p->pos) is allowed. */

  if (p->directInput)
  {
    UInt32 curSize = 0xFFFFFFFF - GET_AVAIL_BYTES(p);
    if (curSize > p->directInputRem)
      curSize = (UInt32)p->directInputRem;
    p->directInputRem -= curSize;
    p->streamPos += curSize;
    if (p->directInputRem == 0)
      p->streamEndWasReached = 1;
    return;
  }
  
  for (;;)
  {
    Byte *dest = p->buffer + GET_AVAIL_BYTES(p);
    size_t size = (size_t)(p->bufferBase + p->blockSize - dest);
    if (size == 0)
    {
      /* we call ReadBlock() after NeedMove() and MoveBlock().
         NeedMove() and MoveBlock() povide more than (keepSizeAfter)
         to the end of (blockSize).
         So we don't execute this branch in normal code flow.
         We can go here, if we will call ReadBlock() before NeedMove(), MoveBlock().
      */
      // p->result = SZ_ERROR_FAIL; // we can show error here
      return;
    }

    // #define kRead 3
    // if (size > kRead) size = kRead; // for debug

    p->result = ISeqInStream_Read(p->stream, dest, &size);
    if (p->result != SZ_OK)
      return;
    if (size == 0)
    {
      p->streamEndWasReached = 1;
      return;
    }
    p->streamPos += (UInt32)size;
    if (GET_AVAIL_BYTES(p) > p->keepSizeAfter)
      return;
    /* here and in another (p->keepSizeAfter) checks we keep on 1 byte more than was requested by Create() function
         (GET_AVAIL_BYTES(p) >= p->keepSizeAfter) - minimal required size */
  }

  // on exit: (p->result != SZ_OK || p->streamEndWasReached || GET_AVAIL_BYTES(p) > p->keepSizeAfter)
}



MY_NO_INLINE
void MatchFinder_MoveBlock(CMatchFinder *p)
{
  const size_t offset = (size_t)(p->buffer - p->bufferBase) - p->keepSizeBefore;
  const size_t keepBefore = (offset & (kBlockMoveAlign - 1)) + p->keepSizeBefore;
  p->buffer = p->bufferBase + keepBefore;
  memmove(p->bufferBase,
      p->bufferBase + (offset & ~((size_t)kBlockMoveAlign - 1)),
      keepBefore + (size_t)GET_AVAIL_BYTES(p));
}

/* We call MoveBlock() before ReadBlock().
   So MoveBlock() can be wasteful operation, if the whole input data
   can fit in current block even without calling MoveBlock().
   in important case where (dataSize <= historySize)
     condition (p->blockSize > dataSize + p->keepSizeAfter) is met
     So there is no MoveBlock() in that case case.
*/

int MatchFinder_NeedMove(CMatchFinder *p)
{
  if (p->directInput)
    return 0;
  if (p->streamEndWasReached || p->result != SZ_OK)
    return 0;
  return ((size_t)(p->bufferBase + p->blockSize - p->buffer) <= p->keepSizeAfter);
}

void MatchFinder_ReadIfRequired(CMatchFinder *p)
{
  if (p->keepSizeAfter >= GET_AVAIL_BYTES(p))
    MatchFinder_ReadBlock(p);
}



static void MatchFinder_SetDefaultSettings(CMatchFinder *p)
{
  p->cutValue = 32;
  p->btMode = 1;
  p->numHashBytes = 4;
  p->bigHash = 0;
}

#define kCrcPoly 0xEDB88320

void MatchFinder_Construct(CMatchFinder *p)
{
  unsigned i;
  p->bufferBase = NULL;
  p->directInput = 0;
  p->hash = NULL;
  p->expectedDataSize = (UInt64)(Int64)-1;
  MatchFinder_SetDefaultSettings(p);

  for (i = 0; i < 256; i++)
  {
    UInt32 r = (UInt32)i;
    unsigned j;
    for (j = 0; j < 8; j++)
      r = (r >> 1) ^ (kCrcPoly & ((UInt32)0 - (r & 1)));
    p->crc[i] = r;
  }
}

static void MatchFinder_FreeThisClassMemory(CMatchFinder *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->hash);
  p->hash = NULL;
}

void MatchFinder_Free(CMatchFinder *p, ISzAllocPtr alloc)
{
  MatchFinder_FreeThisClassMemory(p, alloc);
  LzInWindow_Free(p, alloc);
}

#ifdef AOCL_LZMA_OPT
void AOCL_MatchFinder_Free(CMatchFinder* p, ISzAllocPtr alloc) {
  MatchFinder_FreeThisClassMemory(p, alloc);
  LzInWindow_Free(p, alloc);
}
#endif

static CLzRef* AllocRefs(size_t num, ISzAllocPtr alloc)
{
  size_t sizeInBytes = (size_t)num * sizeof(CLzRef);
  if (sizeInBytes / sizeof(CLzRef) != num)
    return NULL;
  return (CLzRef *)ISzAlloc_Alloc(alloc, sizeInBytes);
}

#if (kBlockSizeReserveMin < kBlockSizeAlign * 2)
  #error Stop_Compiling_Bad_Reserve
#endif



static UInt32 GetBlockSize(CMatchFinder *p, UInt32 historySize)
{
  UInt32 blockSize = (p->keepSizeBefore + p->keepSizeAfter);
  /*
  if (historySize > kMaxHistorySize)
    return 0;
  */
  // printf("\nhistorySize == 0x%x\n", historySize);
  
  if (p->keepSizeBefore < historySize || blockSize < p->keepSizeBefore)  // if 32-bit overflow
    return 0;
  
  {
    const UInt32 kBlockSizeMax = (UInt32)0 - (UInt32)kBlockSizeAlign;
    const UInt32 rem = kBlockSizeMax - blockSize;
    const UInt32 reserve = (blockSize >> (blockSize < ((UInt32)1 << 30) ? 1 : 2))
        + (1 << 12) + kBlockMoveAlign + kBlockSizeAlign; // do not overflow 32-bit here
    if (blockSize >= kBlockSizeMax
        || rem < kBlockSizeReserveMin) // we reject settings that will be slow
      return 0;
    if (reserve >= rem)
      blockSize = kBlockSizeMax;
    else
    {
      blockSize += reserve;
      blockSize &= ~(UInt32)(kBlockSizeAlign - 1);
    }
  }
  // printf("\n LzFind_blockSize = %x\n", blockSize);
  // printf("\n LzFind_blockSize = %d\n", blockSize >> 20);
  return blockSize;
}


int MatchFinder_Create(CMatchFinder *p, UInt32 historySize,
    UInt32 keepAddBufferBefore, UInt32 matchMaxLen, UInt32 keepAddBufferAfter,
    ISzAllocPtr alloc)
{
  /* we need one additional byte in (p->keepSizeBefore),
     since we use MoveBlock() after (p->pos++) and before dictionary using */
  // keepAddBufferBefore = (UInt32)0xFFFFFFFF - (1 << 22); // for debug
  p->keepSizeBefore = historySize + keepAddBufferBefore + 1;

  keepAddBufferAfter += matchMaxLen;
  /* we need (p->keepSizeAfter >= p->numHashBytes) */
  if (keepAddBufferAfter < p->numHashBytes)
    keepAddBufferAfter = p->numHashBytes;
  // keepAddBufferAfter -= 2; // for debug
  p->keepSizeAfter = keepAddBufferAfter;

  if (p->directInput)
    p->blockSize = 0;
  if (p->directInput || LzInWindow_Create2(p, GetBlockSize(p, historySize), alloc))
  {
    const UInt32 newCyclicBufferSize = historySize + 1; // do not change it
    UInt32 hs;
    p->matchMaxLen = matchMaxLen;
    {
      // UInt32 hs4;
      p->fixedHashSize = 0;
      hs = (1 << 16) - 1;
      if (p->numHashBytes != 2)
      {
        hs = historySize;
        if (hs > p->expectedDataSize)
          hs = (UInt32)p->expectedDataSize;
        if (hs != 0)
          hs--;
        hs |= (hs >> 1);
        hs |= (hs >> 2);
        hs |= (hs >> 4);
        hs |= (hs >> 8);
        // we propagated 16 bits in (hs). Low 16 bits must be set later
        hs >>= 1;
        if (hs >= (1 << 24))
        {
          if (p->numHashBytes == 3)
            hs = (1 << 24) - 1;
          else
            hs >>= 1;
          /* if (bigHash) mode, GetHeads4b() in LzFindMt.c needs (hs >= ((1 << 24) - 1))) */
        }
        
        // hs = ((UInt32)1 << 25) - 1; // for test
        
        // (hash_size >= (1 << 16)) : Required for (numHashBytes > 2)
        hs |= (1 << 16) - 1; /* don't change it! */
        
        // bt5: we adjust the size with recommended minimum size
        if (p->numHashBytes >= 5)
          hs |= (256 << kLzHash_CrcShift_2) - 1;
      }
      /* p->hashMask is set based on dictionary size
      * which is inturn set based on level and input data size
      * This determines the max number of bits up to which a value can be stored in the hash
      * E.g: historySize (4MB)  : hashMask (0x1F FFFF)
      * E.g: historySize (16MB) : hashMask (0x7F FFFF)
      * E.g: historySize (64MB) : hashMask (0xFF FFFF)
      */
      p->hashMask = hs;
      hs++; // size of hash table required

      /*
      hs4 = (1 << 20);
      if (hs4 > hs)
        hs4 = hs;
      // hs4 = (1 << 16); // for test
      p->hash4Mask = hs4 - 1;
      */

      if (p->numHashBytes > 2) p->fixedHashSize += kHash2Size; // space for additional hash tables : 2-byte hash
      if (p->numHashBytes > 3) p->fixedHashSize += kHash3Size; // space for additional hash tables : 3-byte hash
      // if (p->numHashBytes > 4) p->fixedHashSize += hs4; // kHash4Size;
      hs += p->fixedHashSize; // space for all hash tables combined
    }

    {
      size_t newSize;
      size_t numSons;
      p->historySize = historySize;
      p->hashSizeSum = hs;
      p->cyclicBufferSize = newCyclicBufferSize; // it must be = (historySize + 1)
      
      numSons = newCyclicBufferSize;
      if (p->btMode)
        numSons <<= 1; //for bt, son is : <leftNode0> <rightNode0> <leftNode1> <rightNode1>... Hence 2x size.
      newSize = hs + numSons;

      // aligned size is not required here, but it can be better for some loops
      #define NUM_REFS_ALIGN_MASK 0xF
      newSize = (newSize + NUM_REFS_ALIGN_MASK) & ~(size_t)NUM_REFS_ALIGN_MASK;

      if (p->hash && p->numRefs == newSize)
        return 1;
      
      MatchFinder_FreeThisClassMemory(p, alloc);
      p->numRefs = newSize;
      p->hash = AllocRefs(newSize, alloc);
      
      if (p->hash)
      {
        p->son = p->hash + p->hashSizeSum;
        return 1;
      }
    }
  }

  MatchFinder_Free(p, alloc);
  return 0;
}

#ifdef AOCL_LZMA_OPT
#define AOCL_HC_COMPUTE_HASH_MASK(hs) \
/* Hash mask computation for use with cache efficient hash chain dictionary.
* Hc4 and Hc5 are the only configurations supported for hash chain mode.
* Hence, p->numHashBytes = 2,3 are not accounted for here. 
* 2-byte hash table uses 10-bits: minimum mask required for this = (1 << 10) - 1.
* 4 and 5 byte hashes guarantee byte-1 and byte-2 are the same, provided
* mask of (1 << 16) - 1 or higher is used. This assumption is used in
* *_GetMatches_* functions. So, we need to guarantee this. */ \
if (hs != 0) \
    hs--; \
hs |= (hs >> 1); \
hs |= (hs >> 2); \
hs |= (hs >> 4); \
hs |= (hs >> 8); \
/* lower 16-bits accounted for in (|= kHashGuarentee - 1) */ \
/* When block_cnt is not a power of 2, hs can be > block_cnt-1. 
* If not hash can produce values > (block_cnt - 1), which is out of range. 
* Adjust this, as we want mask to be the largest 2^N - 1 not greater than block_cnt-1.*/ \
while (hs > (block_cnt - 1)) \
    hs >>= 1; \
 \
hs |= kHashGuarentee - 1; /* don't remove this. Guarentees minimum mask size. */

#define AOCL_SET_NUM_SONS(numSons) \
if (p->level < HASH_CHAIN_16_LEVEL) \
    numSons = (size_t)(p->hashMask + 1) * HASH_CHAIN_SLOT_SZ_8; \
else \
    numSons = (size_t)(p->hashMask + 1) * HASH_CHAIN_SLOT_SZ_16; \

#define AOCL_SET_NUM_SONS_SZ(numSons, HASH_CHAIN_SLOT_SZ) \
numSons = (size_t)(p->hashMask + 1) * HASH_CHAIN_SLOT_SZ; \

/*
* @brief: Allocate all data structures for hash tables, dictionary, etc
* 
* Changes wrt MatchFinder_Create:
* + For hc-mode: hash table size derived from historySize.
    Block sizes set based on level. 3-byte fixed hash table not used.
* + For bt-mode: 3-byte fixed hash table not used.
*/
int AOCL_MatchFinder_Create(CMatchFinder* p, UInt32 historySize,
    UInt32 keepAddBufferBefore, UInt32 matchMaxLen, UInt32 keepAddBufferAfter,
    ISzAllocPtr alloc) {
    /* we need one additional byte in (p->keepSizeBefore),
       since we use MoveBlock() after (p->pos++) and before dictionary using */
       // keepAddBufferBefore = (UInt32)0xFFFFFFFF - (1 << 22); // for debug
    p->keepSizeBefore = historySize + keepAddBufferBefore + 1;

    keepAddBufferAfter += matchMaxLen;
    /* we need (p->keepSizeAfter >= p->numHashBytes) */
    if (keepAddBufferAfter < p->numHashBytes)
        keepAddBufferAfter = p->numHashBytes;
    // keepAddBufferAfter -= 2; // for debug
    p->keepSizeAfter = keepAddBufferAfter;

    if (p->directInput)
        p->blockSize = 0;
    if (p->directInput || LzInWindow_Create2(p, GetBlockSize(p, historySize), alloc))
    {
        const UInt32 newCyclicBufferSize = historySize + 1; // do not change it
        UInt32 hs;
        size_t newSize;
        size_t numSons;
        p->matchMaxLen = matchMaxLen;

        if (USE_CACHE_EFFICIENT_HASH_CHAIN) { // set sizes to support cache efficient dictionary (hash-chain mode only)
            UInt32 block_cnt; // size of hash table required
            hs = historySize;
            if (hs > p->expectedDataSize)
                hs = (UInt32)p->expectedDataSize;
            if (p->level < HASH_CHAIN_16_LEVEL) {
                block_cnt = hs = hs / HASH_CHAIN_SLOT_SZ_8;
            }
            else {
                block_cnt = hs = hs / HASH_CHAIN_SLOT_SZ_16;
            }

            AOCL_HC_COMPUTE_HASH_MASK(hs)
            p->hashMask = hs;

            assert((p->hashMask + 1) >= kHashGuarentee);

            /* Fixed tables to use based on p->numHashBytes
            * If numHashBytes = 3/4/5: 2-byte fixed table only */
            p->fixedHashSize = 0;
            if (p->numHashBytes > 2) p->fixedHashSize += kHash2Size; // space for additional hash tables : 2-byte hash
            // if (p->numHashBytes > 3) p->fixedHashSize += kHash3Size;
            // if (p->numHashBytes > 4) p->fixedHashSize += hs4; // kHash4Size;

            p->historySize = historySize;
            p->cyclicBufferSize = newCyclicBufferSize; // it must be = (historySize + 1). This limit goes into compressed stream header.
            p->hashSizeSum = p->fixedHashSize; // fixed tables only. Hash heads for numHashBytes-table are stored as part of dictionary.
            AOCL_SET_NUM_SONS(numSons)
            newSize = p->hashSizeSum + numSons;
        }
        else // same as reference except when numHashBytes==4, 3-byte fixed table is not used
        {
            // UInt32 hs4;
            p->fixedHashSize = 0;
            hs = (1 << 16) - 1;
            if (p->numHashBytes != 2)
            {
                hs = historySize;
                if (hs > p->expectedDataSize)
                    hs = (UInt32)p->expectedDataSize;
                if (hs != 0)
                    hs--;
                hs |= (hs >> 1);
                hs |= (hs >> 2);
                hs |= (hs >> 4);
                hs |= (hs >> 8);
                // we propagated 16 bits in (hs). Low 16 bits must be set later
                hs >>= 1;
                if (hs >= (1 << 24))
                {
                    if (p->numHashBytes == 3)
                        hs = (1 << 24) - 1;
                    else
                        hs >>= 1;
                    /* if (bigHash) mode, GetHeads4b() in LzFindMt.c needs (hs >= ((1 << 24) - 1))) */
                }

                // hs = ((UInt32)1 << 25) - 1; // for test

                // (hash_size >= (1 << 16)) : Required for (numHashBytes > 2)
                hs |= (1 << 16) - 1; /* don't change it! */

                // bt5: we adjust the size with recommended minimum size
                if (p->numHashBytes >= 5)
                    hs |= (256 << kLzHash_CrcShift_2) - 1;
            }
            /* p->hashMask is set based on dictionary size
            * which is inturn set based on level and input data size
            * This determines the max number of bits up to which a value can be stored in the hash
            * E.g: historySize (4MB)  : hashMask (0x1F FFFF)
            * E.g: historySize (16MB) : hashMask (0x7F FFFF)
            * E.g: historySize (64MB) : hashMask (0xFF FFFF)
            */
            p->hashMask = hs;
            hs++; // size of hash table required

            /*
            hs4 = (1 << 20);
            if (hs4 > hs)
              hs4 = hs;
            // hs4 = (1 << 16); // for test
            p->hash4Mask = hs4 - 1;
            */

            /* Fixed tables to use based on p->numHashBytes
            * If numHashBytes = 3: 2-byte table
            * If numHashBytes = 4: 2-byte table
            * If numHashBytes = 5: 2-byte, 3-byte tables
            */
            if (p->numHashBytes > 2) p->fixedHashSize += kHash2Size; // space for additional hash tables : 2-byte hash
            if (p->numHashBytes > 4) p->fixedHashSize += kHash3Size; // space for additional hash tables : 3-byte hash
            hs += p->fixedHashSize; // space for all hash tables combined

            p->historySize = historySize;
            p->hashSizeSum = hs;
            p->cyclicBufferSize = newCyclicBufferSize;  // it must be = (historySize + 1). This limit goes into compressed stream header.

            numSons = newCyclicBufferSize;
            if (p->btMode)
                numSons <<= 1; //for bt, son is : <leftNode0> <rightNode0> <leftNode1> <rightNode1>... Hence 2x size.
            newSize = hs + numSons;
        }
        LOG_FORMATTED(DEBUG, logCtx, "Match finder created with historySize = %u, "
        "cyclicBufferSize = %u, fixedHashSize = %u", p->historySize,
         p->cyclicBufferSize, p->fixedHashSize);

        {
            // aligned size is not required here, but it can be better for some loops
#define NUM_REFS_ALIGN_MASK 0xF
            newSize = (newSize + NUM_REFS_ALIGN_MASK) & ~(size_t)NUM_REFS_ALIGN_MASK;

            if (p->hash && p->numRefs == newSize)
                return 1;

            MatchFinder_FreeThisClassMemory(p, alloc);
            p->numRefs = newSize;
            p->hash = AllocRefs(newSize, alloc);
            LOG_FORMATTED(DEBUG, logCtx, "Total number of nodes in all hash tables = %zu", newSize);

            if (p->hash)
            {
                p->son = p->hash + p->hashSizeSum;
                return 1;
            }
        }
    }

    AOCL_MatchFinder_Free(p, alloc);
    return 0;
}
#endif

static void MatchFinder_SetLimits(CMatchFinder *p)
{
  UInt32 k;
  UInt32 n = kMaxValForNormalize - p->pos;
  if (n == 0)
    n = (UInt32)(Int32)-1;  // we allow (pos == 0) at start even with (kMaxValForNormalize == 0)
  
  k = p->cyclicBufferSize - p->cyclicBufferPos;
  if (k < n)
    n = k;

  k = GET_AVAIL_BYTES(p);
  {
    const UInt32 ksa = p->keepSizeAfter;
    UInt32 mm = p->matchMaxLen;
    if (k > ksa)
      k -= ksa; // we must limit exactly to keepSizeAfter for ReadBlock
    else if (k >= mm)
    {
      // the limitation for (p->lenLimit) update
      k -= mm;   // optimization : to reduce the number of checks
      k++;
      // k = 1; // non-optimized version : for debug
    }
    else
    {
      mm = k;
      if (k != 0)
        k = 1;
    }
    p->lenLimit = mm;
  }
  if (k < n)
    n = k;
  
  p->posLimit = p->pos + n;
}


void MatchFinder_Init_LowHash(CMatchFinder *p)
{
  size_t i;
  CLzRef *items = p->hash;
  const size_t numItems = p->fixedHashSize;
  for (i = 0; i < numItems; i++)
    items[i] = kEmptyHashValue;
}


void MatchFinder_Init_HighHash(CMatchFinder *p)
{
  size_t i;
  CLzRef *items = p->hash + p->fixedHashSize;
  const size_t numItems = (size_t)p->hashMask + 1;
  for (i = 0; i < numItems; i++)
    items[i] = kEmptyHashValue;
}


void MatchFinder_Init_4(CMatchFinder *p)
{
  p->buffer = p->bufferBase;
  {
    /* kEmptyHashValue = 0 (Zero) is used in hash tables as NO-VALUE marker.
       the code in CMatchFinderMt expects (pos = 1) */
    p->pos =
    p->streamPos =
        1; // it's smallest optimal value. do not change it
        // 0; // for debug
  }
  p->result = SZ_OK;
  p->streamEndWasReached = 0;
}


// (CYC_TO_POS_OFFSET == 0) is expected by some optimized code
#define CYC_TO_POS_OFFSET 0
// #define CYC_TO_POS_OFFSET 1 // for debug

void MatchFinder_Init(CMatchFinder *p)
{
  MatchFinder_Init_HighHash(p); //Init hash head table
  MatchFinder_Init_LowHash(p); //Init fixed hash tables
  MatchFinder_Init_4(p);
  // if (readData)
  MatchFinder_ReadBlock(p);

  /* if we init (cyclicBufferPos = pos), then we can use one variable
     instead of both (cyclicBufferPos) and (pos) : only before (cyclicBufferPos) wrapping */
  p->cyclicBufferPos = (p->pos - CYC_TO_POS_OFFSET); // init with relation to (pos)
  // p->cyclicBufferPos = 0; // smallest value
  // p->son[0] = p->son[1] = 0; // unused: we can init skipped record for speculated accesses.
  MatchFinder_SetLimits(p);
}

#ifdef AOCL_LZMA_OPT
/* 
* @brief: Initialize dictionary and hash tables
*
* Changes wrt MatchFinder_Init:
* + For hc-mode, initialize hash-chains and chain head pointers.
* + For bt-mode, call reference */
void AOCL_MatchFinder_Init(CMatchFinder* p)
{
    if (USE_CACHE_EFFICIENT_HASH_CHAIN) { // for hash-chain mode, init to support cache efficient dictionary
        // Init son: hash-chains and chain head pointers
        size_t numSons;
        AOCL_SET_NUM_SONS(numSons)
        memset(p->son, 0, sizeof(CLzRef) * numSons); //kEmptySonValue
        LOG_FORMATTED(DEBUG, logCtx, "Initialized cehc data structure of %zu nodes", numSons);

        MatchFinder_Init_LowHash(p); //Init fixed hash tables
        MatchFinder_Init_4(p);
        // if (readData)
        MatchFinder_ReadBlock(p);

        /* if we init (cyclicBufferPos = pos), then we can use one variable
           instead of both (cyclicBufferPos) and (pos) : only before (cyclicBufferPos) wrapping */
        p->cyclicBufferPos = (p->pos - CYC_TO_POS_OFFSET); // init with relation to (pos)
        // p->cyclicBufferPos = 0; // smallest value
        // p->son[0] = p->son[1] = 0; // unused: we can init skipped record for speculated accesses.
        MatchFinder_SetLimits(p);
    }
    else { // same as reference
        MatchFinder_Init(p);
    }
}
#endif

// #define _SHOW_HW_STATUS

#ifdef _SHOW_HW_STATUS
#include <stdio.h>
#define _PRF(x) x
_PRF(;)
#else
#define _PRF(x)
#endif


typedef void (MY_FAST_CALL *LZFIND_SATUR_SUB_CODE_FUNC)(
    UInt32 subValue, CLzRef *items, const CLzRef *lim);
#if defined(USE_SATUR_SUB_128) && !defined(FORCE_SATUR_SUB_128)
static LZFIND_SATUR_SUB_CODE_FUNC g_LzFind_SaturSub;
#endif
// kEmptyHashValue must be zero
// #define SASUB_32(i) v = items[i];  m = v - subValue;  if (v < subValue) m = kEmptyHashValue;  items[i] = m;
#define SASUB_32(i) v = items[i];  if (v < subValue) v = subValue; items[i] = v - subValue;

#define DEFAULT_SaturSub LzFind_SaturSub_32

MY_NO_INLINE
static
void
MY_FAST_CALL
LzFind_SaturSub_32(UInt32 subValue, CLzRef *items, const CLzRef *lim)
{
  do
  {
    UInt32 v;
    SASUB_32(0)
    SASUB_32(1)
    SASUB_32(2)
    SASUB_32(3)
    SASUB_32(4)
    SASUB_32(5)
    SASUB_32(6)
    SASUB_32(7)
    items += 8;
  }
  while (items != lim);
}


MY_NO_INLINE
void MatchFinder_Normalize3(UInt32 subValue, CLzRef *items, size_t numItems)
{
  #define K_NORM_ALIGN_BLOCK_SIZE (1 << 6)
  
  CLzRef *lim;

  for (; numItems != 0 && ((unsigned)(ptrdiff_t)items & (K_NORM_ALIGN_BLOCK_SIZE - 1)) != 0; numItems--)
  {
    UInt32 v;
    SASUB_32(0);
    items++;
  }

  {
    #define K_NORM_ALIGN_MASK (K_NORM_ALIGN_BLOCK_SIZE / 4 - 1)
    lim = items + (numItems & ~(size_t)K_NORM_ALIGN_MASK);
    numItems &= K_NORM_ALIGN_MASK;
    if (items != lim)
    {
      #if defined(USE_SATUR_SUB_128) && !defined(FORCE_SATUR_SUB_128)
        if (g_LzFind_SaturSub)
          g_LzFind_SaturSub(subValue, items, lim);
        else
      #endif
          DEFAULT_SaturSub(subValue, items, lim);
    }
    items = lim;
  }


  for (; numItems != 0; numItems--)
  {
    UInt32 v;
    SASUB_32(0);
    items++;
  }
}

#ifdef AOCL_LZMA_OPT
/* Reduce the pointer values for all nodes in hash-chains. 
* kEmptySonValue must be 0 */
#define AOCL_SASUB_32(i) \
v = items[i]; \
if (v == kEmptySonValue || v < subValue) \
    v = subValue; /* empty nodes and out of bound nodes are reset to 0 */ \
items[i] = v - subValue;

#define AOCL_NORMALIZE_HASH_CHAIN_TABLE(HASH_CHAIN_SLOT_SZ) { \
CLzRef* items = p->son; \
size_t numSons; \
AOCL_SET_NUM_SONS_SZ(numSons, HASH_CHAIN_SLOT_SZ) \
for (size_t i = 0; i < numSons; i += HASH_CHAIN_SLOT_SZ, items += HASH_CHAIN_SLOT_SZ) \
{ \
    UInt32 v; \
    /* First node of each block in son contains the hash head reference.
     * Do not modify this.*/ \
    v = items[0]; \
    if (v == kEmptySonValue) continue; /* if hash head is empty, there is no hash-chain */ \
    \
    /* For nodes 1 to N, normalize */ \
    for (int j = 1; j < HASH_CHAIN_SLOT_SZ; ++j) { \
        AOCL_SASUB_32(j) \
    } \
} \
}

/*
* @brief Normalize nodes in hash table and hash chains.
* 
* This functions gets called only when p->pos rolls over from UINT_MAX.
* Since this is only possible when input stream is >4GB, aocl_compression_bench
* will never call this (as input buffer is limited to 1GB).
*
* For functionality testing, enable AOCL_EXTEND_CODE_COVERAGE flag. This
* will force Normalize calls before pos reaches UINT_MAX
* 
* Changes wrt MatchFinder_Normalize3
* + If USE_CACHE_EFFICIENT_HASH_CHAIN, modified to suit cache efficient data structures
* + Else calls MatchFinder_Normalize3 with same parameters as reference 
*/
MY_NO_INLINE
void AOCL_MatchFinder_Normalize3(UInt32 subValue, CMatchFinder* p)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    if (USE_CACHE_EFFICIENT_HASH_CHAIN) { // normalize cache efficient data structures
        {
            // normalize fixed hash tables
            CLzRef* items = p->hash;
            MatchFinder_Normalize3(subValue, items, (size_t)p->fixedHashSize);
            LOG_FORMATTED(DEBUG, logCtx, "Normalized fixed hash table of size = %zu", (size_t)p->fixedHashSize);
        }

        // normalize hash chain table
        if (p->level < HASH_CHAIN_16_LEVEL)
        {
            AOCL_NORMALIZE_HASH_CHAIN_TABLE(HASH_CHAIN_SLOT_SZ_8)
            LOG_FORMATTED(DEBUG, logCtx, "Normalized chain table of size = %zu",
             (size_t)(p->hashMask + 1) * HASH_CHAIN_SLOT_SZ_8);
        }
        else {
            AOCL_NORMALIZE_HASH_CHAIN_TABLE(HASH_CHAIN_SLOT_SZ_16)
            LOG_FORMATTED(DEBUG, logCtx, "Normalized chain table of size = %zu",
             (size_t)(p->hashMask + 1) * HASH_CHAIN_SLOT_SZ_16);
        }
    }
    else { // normalize reference data structures
        size_t numSonRefs = p->cyclicBufferSize;
        if (p->btMode)
            numSonRefs <<= 1; //bt-mode, 2 values per node <l,r>
        MatchFinder_Normalize3(subValue, p->hash, (size_t)p->hashSizeSum + numSonRefs);
        LOG_FORMATTED(DEBUG, logCtx, "Normalized all hash tables of size = %zu",
             (size_t)p->hashSizeSum + numSonRefs);
    }
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
}

/*
* @brief Handles overflow checks
* 
* call AOCL_MatchFinder_CheckLimits() only after (p->pos++) update. 
* It handles the case where p->pos rolls over to 0 from UINT_MAX
* 
* Changes wrt MatchFinder_CheckLimits:
* + AOCL_MatchFinder_Normalize3() called in place of MatchFinder_Normalize3()
* + numSonRefs obtained inside AOCL_MatchFinder_Normalize3()
*/
MY_NO_INLINE
static void AOCL_MatchFinder_CheckLimits(CMatchFinder* p)
{
    if (// !p->streamEndWasReached && p->result == SZ_OK &&
        p->keepSizeAfter == GET_AVAIL_BYTES(p))
    {
        // we try to read only in exact state (p->keepSizeAfter == GET_AVAIL_BYTES(p))
        if (MatchFinder_NeedMove(p))
            MatchFinder_MoveBlock(p);
        MatchFinder_ReadBlock(p);
    }

    if (p->pos == kMaxValForNormalize)
        if (GET_AVAIL_BYTES(p) >= p->numHashBytes) // optional optimization for last bytes of data.
            /*
               if we disable normalization for last bytes of data, and
               if (data_size == 4 GiB), we don't call wastfull normalization,
               but (pos) will be wrapped over Zero (0) in that case.
               And we cannot resume later to normal operation
            */
        {
            // MatchFinder_Normalize(p);
            /* after normalization we need (p->pos >= p->historySize + 1); */
            /* we can reduce subValue to aligned value, if want to keep alignment
               of (p->pos) and (p->buffer) for speculated accesses. */
            const UInt32 subValue = (p->pos - p->historySize - 1) /* & ~(UInt32)(kNormalizeAlign - 1) */;
            // const UInt32 subValue = (1 << 15); // for debug
            // printf("\nMatchFinder_Normalize() subValue == 0x%x\n", subValue);
            Inline_MatchFinder_ReduceOffsets(p, subValue);
            AOCL_MatchFinder_Normalize3(subValue, p);
        }

    if (p->cyclicBufferPos == p->cyclicBufferSize)
        p->cyclicBufferPos = 0;
    MatchFinder_SetLimits(p);
}
#endif

// call MatchFinder_CheckLimits() only after (p->pos++) update
MY_NO_INLINE
static void MatchFinder_CheckLimits(CMatchFinder *p)
{
  if (// !p->streamEndWasReached && p->result == SZ_OK &&
      p->keepSizeAfter == GET_AVAIL_BYTES(p))
  {
    // we try to read only in exact state (p->keepSizeAfter == GET_AVAIL_BYTES(p))
    if (MatchFinder_NeedMove(p))
      MatchFinder_MoveBlock(p);
    MatchFinder_ReadBlock(p);
  }

  if (p->pos == kMaxValForNormalize)
  if (GET_AVAIL_BYTES(p) >= p->numHashBytes) // optional optimization for last bytes of data.
    /*
       if we disable normalization for last bytes of data, and
       if (data_size == 4 GiB), we don't call wastfull normalization,
       but (pos) will be wrapped over Zero (0) in that case.
       And we cannot resume later to normal operation
    */
  {
    // MatchFinder_Normalize(p);
    /* after normalization we need (p->pos >= p->historySize + 1); */
    /* we can reduce subValue to aligned value, if want to keep alignment
       of (p->pos) and (p->buffer) for speculated accesses. */
    const UInt32 subValue = (p->pos - p->historySize - 1) /* & ~(UInt32)(kNormalizeAlign - 1) */;
    // const UInt32 subValue = (1 << 15); // for debug
    // printf("\nMatchFinder_Normalize() subValue == 0x%x\n", subValue);
    size_t numSonRefs = p->cyclicBufferSize;
    if (p->btMode)
      numSonRefs <<= 1;
    Inline_MatchFinder_ReduceOffsets(p, subValue);
    MatchFinder_Normalize3(subValue, p->hash, (size_t)p->hashSizeSum + numSonRefs);
  }

  if (p->cyclicBufferPos == p->cyclicBufferSize)
    p->cyclicBufferPos = 0;  
  MatchFinder_SetLimits(p);
}

/*
  (lenLimit > maxLen)
*/
/*
* This function is used for finding matches when hashChain mode
* algorithm is selected. It navigates through the linked list based
* hash chain dictionary and looks for matches at these positions.
*
* Finds <len,dist> pairs for matches found in dictionary
* for input stream in cur[0...lenLimit] and saves them in 'distances'
* @output: distances
*
* @inputs:
* lenLimit: max length match we are looking for
* curMatch: position value from hash table for Hash(cur[0-4])
*           we expect to find matching bytes at this offset from cur
* pos : position value of cur
* *cur: pointer to current byte in stream
* *son: pointer to dictionary
* _cyclicBufferPos: current position in cyclic buffer
* _cyclicBufferSize: size of cyclic buffer
* cutValue: hard limit on number of nodes to look at when searching for matches
* *distances: output array to store <len,dist> pairs that are found
* maxLen: minimum length of match we should be able to find in
* dictionary (maxLen is previously set by searching for match in UPDATE_maxLen)
*
* 'son' holds the dictionary
* The dictionary is made up of multiple linked lists (hash chains)
* son[x] either has 0, if no link exists, or a link to next ()
* item in the linked list (pointers themselves are not stored in the
* dictionary. Instead UInt32 values that map to position are stored)
* E.g:
* abcdef...1234....abcdexy...123...abcd....
* <----------------<---------------
*          <-----------------
* root node of each of hash chains is an entry from hash table (curMatch)
* Every call to Hc_GetMatchesSpec, adds curMatch to the dictionary
*/
MY_FORCE_INLINE
static UInt32 * Hc_GetMatchesSpec(size_t lenLimit, UInt32 curMatch, UInt32 pos, const Byte *cur, CLzRef *son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue,
    UInt32 *d, unsigned maxLen)
{
  /*
  son[_cyclicBufferPos] = curMatch;
  for (;;)
  {
    UInt32 delta = pos - curMatch;
    if (cutValue-- == 0 || delta >= _cyclicBufferSize)
      return d;
    {
      const Byte *pb = cur - delta;
      curMatch = son[_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)];
      if (pb[maxLen] == cur[maxLen] && *pb == *cur)
      {
        UInt32 len = 0;
        while (++len != lenLimit)
          if (pb[len] != cur[len])
            break;
        if (maxLen < len)
        {
          maxLen = len;
          *d++ = len;
          *d++ = delta - 1;
          if (len == lenLimit)
            return d;
        }
      }
    }
  }
  */

  const Byte *lim = cur + lenLimit; // do not check for length beyond this
  son[_cyclicBufferPos] = curMatch; // add curMatch to dictionary
  do
  {
    UInt32 delta;

    if (curMatch == 0)
      break;
    // if (curMatch2 >= curMatch) return NULL;
    delta = pos - curMatch;
    if (delta >= _cyclicBufferSize) // Return if delta is beyond search buffer
      break;
    {
      ptrdiff_t diff;
      curMatch = son[_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)];  // offset where you expect to find a match
      diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
      /* cur[0:maxlen - 1] are already matched by this iteration
      *  Just need to check from cur[maxLen] onwards */
      if (cur[maxLen] == cur[(ptrdiff_t)maxLen + diff])
      {
        const Byte *c = cur;
        while (*c == c[diff]) // compare bytes cur wrt previous match
        {
          if (++c == lim) // length of match reached lenlimit, no need to match further
          {
            d[0] = (UInt32)(lim - cur);
            d[1] = delta - 1;
            return d + 2; // return <len,dist> pairs found
          }
        }
        {
          const unsigned len = (unsigned)(c - cur);
          if (maxLen < len) // found a match longer than maxLen. save <len,dist> pair in distances
          {
            maxLen = len;
            d[0] = (UInt32)len;
            d[1] = delta - 1;
            d += 2;
          }
        }
      }
    }
  }
  while (--cutValue); // limit number of nodes in hash chain to look at for matches
  
  return d;
}

/*
* This function is used for finding matches when binTree mode
* algorithm is selected. It navigates through the binary tree based
* dictionary and looks for matches at these positions.
*
* Finds <len,dist> pairs for matches found in dictionary
* for input stream in cur[0...lenLimit] and saves them in 'distances'
* @output: distances
*
* @inputs:
* lenLimit: max length match we are looking for
* curMatch: position value from hash table for Hash(cur[0-4])
*           we expect to find matching bytes at this offset from cur
* pos : position value of cur
* *cur: pointer to current byte in stream
* *son: pointer to dictionary
* _cyclicBufferPos: current position in cyclic buffer
* _cyclicBufferSize: size of cyclic buffer
* cutValue: hard limit on number of nodes to look at when searching for matches
* *distances: output array to store <len,dist> pairs that are found
* maxLen: minimum length of match we should be able to find in
* dictionary (maxLen is previously set by searching for match in UPDATE_maxLen)
*
* 'son' holds the dictionary
* The dictionary is made up of multiple binary trees
* ptr0 points to right-child node
* ptr1 points to left-child node
* root node of each of the binary trees is an entry from hash table (curMatch)
*
* Pointers themselves are not stored in the dictionary. Instead UInt32 values
* that indicate position are used. 'pos', 'curMatch' contain such position values.
* E.g:          [abcd]nodeP=450
*                  |
*         ------------------
*         |                |
* [abcc]nodeL=102   [abce]nodeR=205
* At offset of (450-102) from current position [abcd], you have a match [abcc]
*
* Every call to GetMatchesSpec1, adds curMatch to the dictionary (unless a
* match longer than lenLimit is found)
*/
MY_FORCE_INLINE
UInt32 * GetMatchesSpec1(UInt32 lenLimit, UInt32 curMatch, UInt32 pos, const Byte *cur, CLzRef *son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue,
    UInt32 *d, UInt32 maxLen)
{
  CLzRef *ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1; // right child node
  CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1); // left child node
  unsigned len0 = 0, len1 = 0; // len of match found in right and left branch respectively

  UInt32 cmCheck;

  // if (curMatch >= pos) { *ptr0 = *ptr1 = kEmptyHashValue; return NULL; }

  /* range of pos [_cyclicBufferSize, _cyclicBufferSize+N)
  * when no entry exits for a particular leaf node in tree, it will have value 0
  * by default. When we reach such a node, curMatch become 0
  * delta becomes >= _cyclicBufferSize
  * return at this point. search completed.*/
  cmCheck = (UInt32)(pos - _cyclicBufferSize);
  if ((UInt32)pos <= _cyclicBufferSize)
    cmCheck = 0;

  /* Look through binary tree associated with curMatch if one exists
  * Find matches and save <len, dist> pairs in distances */
  if (cmCheck < curMatch)
  do
  {
    const UInt32 delta = pos - curMatch; // offset to match wrt cur
    {
      // pair of <left,right> node at offset delta within circular buffer
      CLzRef *pair = son + ((size_t)(_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)) << 1);
      const Byte *pb = cur - delta; // previous bytes in stream to compare cur with
      /* We can start comparing from len onwards as pb[0:len-1] == cur[0:len-1]
      until this iteration. Nodes further down the tree will satisfy this condition
      and hopefully match some more bytes. Why select min(len0, len1) though? */
      unsigned len = (len0 < len1 ? len0 : len1);
      const UInt32 pair0 = pair[0];
      if (pb[len] == cur[len]) // bytes matched
      {
        if (++len != lenLimit && pb[len] == cur[len])
          while (++len != lenLimit) // keep matching until you get a mismatch byte or reach lenLimit
            if (pb[len] != cur[len])
              break;
        if (maxLen < len) // found a match longer than maxLen. save <len,dist> pair in distances
        {
          maxLen = (UInt32)len;
          *d++ = (UInt32)len; // save len of match
          *d++ = delta - 1; // save dist of match
          if (len == lenLimit) // match of lenLimit found. No need to check further. return.
          {
            *ptr1 = pair0;
            *ptr0 = pair[1];
            return d;
          }
        }
      }
      /* pb[0 to len-1] == cur[0 to len-1]
      * pb[len] != cur[len]. Mismatch on this byte. */
      if (pb[len] < cur[len]) // previous byte is smaller. set curMatch to left child node 
      {
        *ptr1 = curMatch;
        // const UInt32 curMatch2 = pair[1];
        // if (curMatch2 >= curMatch) { *ptr0 = *ptr1 = kEmptyHashValue;  return NULL; }
        // curMatch = curMatch2;
        curMatch = pair[1]; // propagate matched position
        ptr1 = pair + 1; // move to matched right child node
        len1 = len;
      }
      else // previous byte is larger. set curMatch to right child node 
      {
        *ptr0 = curMatch;
        curMatch = pair[0];// propagate matched position
        ptr0 = pair; // move to matched left child node
        len0 = len;
      }
    }
  }
  while(--cutValue && cmCheck < curMatch); // search completed return

  *ptr0 = *ptr1 = kEmptyHashValue;
  return d;
}

#ifdef AOCL_LZMA_OPT
// circular buffer pos inc. hcHead range must be: hcBase+1 <= hcHead <= hcBase+HASH_CHAIN_MAX.
#define CIRC_INC_HEAD(hcHead, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) \
((((hcHead + 1) % HASH_CHAIN_SLOT_SZ) != 0) ? (hcHead + 1) : (hcHead + 1 - HASH_CHAIN_MAX))

// circular buffer pos dec. hcHead range must be: hcBase+1 <= hcHead <= hcBase+HASH_CHAIN_MAX.
#define CIRC_DEC_HEAD(hcHead, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) \
((((hcHead - 1) % HASH_CHAIN_SLOT_SZ) != 0) ? (hcHead - 1) : (hcHead + HASH_CHAIN_MAX - 1))

// Compare bytes in data2 and data1 using UInt32 ptrs and __builtin_ctz
#define AOCL_FIND_MATCHING_BYTES_LEN(len, limit, data1, data2, exit_point) \
    if (likely(limit >= 4)) { \
        UInt32 D = 0; \
        UInt32 lenLimit4 = limit - 4; \
        while (len <= lenLimit4) { \
            UInt32 C1 = *(UInt32*)&data2[len]; \
            UInt32 C2 = *(UInt32*)&data1[len]; \
            D = C1 ^ C2; \
            if (D) { \
                int trail = __builtin_ctz(D); \
                len += (trail >> 3); \
                goto exit_point; \
            } \
            len += 4; \
        } \
    } \
    while (len < limit) { \
    if (data2[len] != data1[len]) \
        break; \
    len++; \
    } \
exit_point:

#define AOCL_HC_GETMATCHES_SPEC(HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) { \
    UInt32 curMatch, hcCur; \
    hcCur = hcHeadPos; /* set current node to head of chain */ \
    AOCL_COMMON_CEHCFIX_GET(son, 0, hcCur, 0, curMatch, \
        HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) \
    UInt32 delta = (pos - curMatch); \
    unsigned checkLen = maxLen - 1; /* to ensure cur does not access beyond lenLimit */ \
    \
    if (curMatch > 0 && delta < _cyclicBufferSize) { \
        __builtin_prefetch(cur, 0, 2); \
        do \
        { \
            { \
                ptrdiff_t diff; \
                diff = (ptrdiff_t)0 - delta; \
                /* We need a match of at least maxLen bytes.
                * Speed up comparison by checking 2 bytes at maxLen,
                * followed by first 4 bytes. Proceed to match rest of
                * the bytes only if above two match.
                *
                * Checking 2 bytes instead of 1 at maxLen results in
                * slight increase in compression ratio */ \
                if (GetUi16(cur + checkLen) == GetUi16(cur + checkLen + diff) && \
                    GetUi32(cur) == GetUi32(cur + diff)) \
                { \
                    const Byte* c = cur; \
                    const Byte* cd = c + diff; \
                    unsigned len = 4; /* first 4 bytes already matched */ \
                    \
                    AOCL_FIND_MATCHING_BYTES_LEN(len, lenLimit, c, cd, exit_point_check_len) \
                    \
                    { \
                        if (maxLen < len) /* found a match longer than maxLen. Save <len, dist> pair in distances */ \
                        { \
                            d[0] = (UInt32)len; \
                            d[1] = delta - 1; \
                            d += 2; \
                            if (len == lenLimit) /* length of match reached lenlimit, no need to match further */ \
                            { \
                                AOCL_COMMON_CEHCFIX_INSERT(son, 0, hcHeadPos, 0, pos, hv, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) /* add pos to beginning of hash chain */ \
                                return d; /* return <len, dist> pairs found */ \
                            } \
                            maxLen = len; \
                            checkLen = maxLen; \
                            if(unlikely(maxLen >= (lenLimit - 1))) \
                                checkLen = (maxLen - 1); \
                        } \
                    } \
                } \
            } \
            \
            AOCL_COMMON_CEHCFIX_MOVE_TO_NEXT(son, hcCur, curMatch, \
                HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX, kEmptySonValue, hcHeadPos) \
            delta = pos - curMatch; \
            if (delta >= _cyclicBufferSize) \
                break;  /* match offset further than max dictionary size allowed */ \
            \
        } while (--cutValue); /* limit number of nodes in hash chain to check */ \
    } \
    AOCL_COMMON_CEHCFIX_INSERT(son, 0, hcHeadPos, 0, pos, hv, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) /* add pos to beginning of hash chain */ \
    return d; /* return <len, dist> pairs found */ \
}

/*
* This function is used for finding matches when hashChain mode
* algorithm is selected. It navigates through the linked list based
* hash chain dictionary and looks for matches at these positions.
*
* Finds <len,dist> pairs for matches found in dictionary
* for input stream in cur[0...lenLimit] and saves them in 'distances'
* @output: distances
*
* @inputs:
* lenLimit: max length match we are looking for
* hcHead: position within block where current head of the chain exists for respective hash
* pos : position value of cur
* *cur: pointer to current byte in stream
* *son: pointer to dictionary
* _cyclicBufferSize: size of cyclic buffer
* cutValue: hard limit on number of nodes to look at when searching for matches
* *d: output array to store <len,dist> pairs that are found
* maxLen: minimum length of match we are looking to find in
* dictionary (maxLen is previously set by searching for match in UPDATE_maxLen)
*
* 'son' holds the dictionary
* The dictionary is made up of multiple blocks.
* Each block contains [hcHead | hcChain ....]
* hcChain is a circular linked lists
* hcHead indicates where in hcChain hash head exists
* 
* Each element in the block, son[x] either has kEmptySonValue, if no link exists,
* or a link to some point in the input stream where other colliding matches exist
* (pointers themselves are not stored in the dictionary. 
* Instead UInt32 values that map to position are stored)
* 
* Every call to AOCL_Hc_GetMatchesSpec, adds pos to the dictionary
* 
* Changes wrt Hc_GetMatchesSpec: 
* + if (curMatch > 0 && delta < _cyclicBufferSize) check before entering loop
* + cache efficient hash-chain blocks
*/
//MY_FORCE_INLINE
static UInt32* AOCL_Hc_GetMatchesSpec_8(size_t lenLimit, UInt32 hcHeadPos, UInt32 hv, UInt32 pos,
    const Byte* cur, CLzRef* son, size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 cutValue, UInt32* d, unsigned maxLen)
{
    AOCL_HC_GETMATCHES_SPEC(HASH_CHAIN_SLOT_SZ_8, HASH_CHAIN_MAX_8)
}

//MY_FORCE_INLINE
static UInt32* AOCL_Hc_GetMatchesSpec_16(size_t lenLimit, UInt32 hcHeadPos, UInt32 hv, UInt32 pos,
    const Byte* cur, CLzRef* son, size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 cutValue, UInt32* d, unsigned maxLen)
{
    AOCL_HC_GETMATCHES_SPEC(HASH_CHAIN_SLOT_SZ_16, HASH_CHAIN_MAX_16)
}

/*
* This function is used for finding matches when hashChain mode
* algorithm is selected. It navigates through the linked list based
* hash chain dictionary and looks for matches at these positions.
* The hash chains are intervowen within a single buffer.
* 
* Finds <len,dist> pairs for matches found in dictionary
* for input stream in cur[0...lenLimit] and saves them in 'distances'
* @output: distances
*
* @inputs:
* lenLimit: max length match we are looking for
* curMatch: position value from hash table for Hash(cur[0-4])
*           we expect to find matching bytes at this offset from cur
* pos : position value of cur
* *cur: pointer to current byte in stream
* *son: pointer to dictionary
* _cyclicBufferPos: current position in cyclic buffer
* _cyclicBufferSize: size of cyclic buffer
* cutValue: hard limit on number of nodes to look at when searching for matches
* *distances: output array to store <len,dist> pairs that are found
* maxLen: minimum length of match we should be able to find in
* dictionary (maxLen is previously set by searching for match in UPDATE_maxLen)
*
* 'son' holds the dictionary
* The dictionary is made up of multiple linked lists (hash chains)
* son[x] either has 0, if no link exists, or a link to next ()
* item in the linked list (pointers themselves are not stored in the
* dictionary. Instead UInt32 values that map to position are stored)
* E.g:
* abcdef...1234....abcdexy...123...abcd....
* <----------------<---------------
*          <-----------------
* root node of each of hash chains is an entry from hash table (curMatch)
* Every call to Hc_GetMatchesSpec, adds curMatch to the dictionary
*
* Changes wrt Hc_GetMatchesSpec:
* + if (curMatch > 0 && delta < _cyclicBufferSize) check before entering loop
* + __builtin_prefetch(cur)
*/
MY_FORCE_INLINE
static UInt32* AOCL_Hc_GetMatchesSpec(size_t lenLimit, UInt32 curMatch, UInt32 pos, const Byte* cur, CLzRef* son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue,
    UInt32* d, unsigned maxLen)
{
    const Byte* lim = cur + lenLimit; // do not check for length beyond this
    son[_cyclicBufferPos] = curMatch; // add curMatch to dictionary

    UInt32 delta = (pos - curMatch);
    if (curMatch > 0 && delta < _cyclicBufferSize) {
        __builtin_prefetch(cur, 0, 2);
        do
        {
            {
                ptrdiff_t diff;
                curMatch = son[_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)];  // offset where you expect to find a match
                diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
                /* We have match of length maxlen already. We want to find a longer match than this.
                Check at cur[maxLen]. If it doesn't match, no need to compare entire string.*/
                if (cur[maxLen] == cur[(ptrdiff_t)maxLen + diff])
                {
                    const Byte* c = cur;
                    while (*c == c[diff]) // compare bytes cur wrt previous match
                    {
                        if (++c == lim) // length of match reached lenlimit, no need to match further
                        {
                            d[0] = (UInt32)(lim - cur);
                            d[1] = delta - 1;
                            return d + 2; // return <len,dist> pairs found
                        }
                    }
                    {
                        const unsigned len = (unsigned)(c - cur);
                        if (maxLen < len) // found a match longer than maxLen. save <len,dist> pair in distances
                        {
                            maxLen = len;
                            d[0] = (UInt32)len;
                            d[1] = delta - 1;
                            d += 2;
                        }
                    }
                }
            }


            if (curMatch == 0)
                break;
            // if (curMatch2 >= curMatch) return NULL;
            delta = pos - curMatch;
            /* range of pos [_cyclicBufferSize, _cyclicBufferSize+N)
            * when no link exists at a particular position, it will have value 0
            * by default. When we reach such a node, curMatch become 0
            * delta becomes >= _cyclicBufferSize
            * return at this point. search completed.*/
            if (delta >= _cyclicBufferSize)
                break;

        } while (--cutValue); // limit number of nodes in hash chain to look at for matches
    }
    return d;
}

/*
* This function is used for finding matches when binTree mode
* algorithm is selected. It navigates through the binary tree based
* dictionary and looks for matches at these positions.
*
* Finds <len,dist> pairs for matches found in dictionary
* for input stream in cur[0...lenLimit] and saves them in 'distances'
* @output: distances
*
* @inputs:
* lenLimit: max length match we are looking for
* curMatch: position value from hash table for Hash(cur[0-4])
*           we expect to find matching bytes at this offset from cur
* pos : position value of cur
* *cur: pointer to current byte in stream
* *son: pointer to dictionary
* _cyclicBufferPos: current position in cyclic buffer
* _cyclicBufferSize: size of cyclic buffer
* cutValue: hard limit on number of nodes to look at when searching for matches
* *distances: output array to store <len,dist> pairs that are found
* maxLen: minimum length of match we should be able to find in
* dictionary (maxLen is previously set by searching for match in UPDATE_maxLen)
*
* 'son' holds the dictionary
* The dictionary is made up of multiple binary trees
* ptr0 points to right-child node
* ptr1 points to left-child node
* root node of each of the binary trees is an entry from hash table (curMatch)
*
* Pointers themselves are not stored in the dictionary. Instead UInt32 values
* that indicate position are used. 'pos', 'curMatch' contain such position values.
* E.g:          [abcd]nodeP=450
*                  |
*         ------------------
*         |                |
* [abcc]nodeL=102   [abce]nodeR=205
* At offset of (450-102) from current position [abcd], you have a match [abcc]
*
* Every call to AOCL_GetMatchesSpec1, adds curMatch to the dictionary (unless a
* match longer than lenLimit is found)
* 
* Changes wrt GetMatchesSpec1:
* + AOCL_FIND_MATCHING_BYTES_LEN used for matching
*/
MY_FORCE_INLINE
static UInt32* AOCL_GetMatchesSpec1(UInt32 lenLimit, UInt32 curMatch, UInt32 pos, const Byte* cur, CLzRef* son,
  size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue,
  UInt32* d, UInt32 maxLen)
{
  CLzRef* ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1; // right child node
  CLzRef* ptr1 = son + ((size_t)_cyclicBufferPos << 1); // left child node
  unsigned len0 = 0, len1 = 0; // len of match found in right and left branch respectively

  UInt32 cmCheck;

  /* range of pos [_cyclicBufferSize, _cyclicBufferSize+N)
  * when no entry exits for a particular leaf node in tree, it will have value 0
  * by default. When we reach such a node, curMatch become 0
  * delta becomes >= _cyclicBufferSize
  * return at this point. search completed.*/
  cmCheck = (UInt32)(pos - _cyclicBufferSize);
  if ((UInt32)pos <= _cyclicBufferSize)
    cmCheck = 0;

  /* Look through binary tree associated with curMatch if one exists
  * Find matches and save <len, dist> pairs in distances */
  if (cmCheck < curMatch) {
      //__builtin_prefetch(cur);
    do
    {
      const UInt32 delta = pos - curMatch; // offset to match wrt cur
      {
        // pair of <left,right> node at offset delta within circular buffer
        CLzRef* pair = son + ((size_t)(_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)) << 1);
        const Byte* pb = cur - delta; // previous bytes in stream to compare cur with
        /* We can start comparing from len onwards as pb[0:len-1] == cur[0:len-1]
        until this iteration. Nodes further down the tree will satisfy this condition
        and hopefully match some more bytes. Why select min(len0, len1) though? */
        unsigned len = (len0 < len1 ? len0 : len1);
        const UInt32 pair0 = pair[0];
        if (pb[len] == cur[len]) // bytes matched
        {
          ++len;
          AOCL_FIND_MATCHING_BYTES_LEN(len, lenLimit, cur, pb, aocl_bt_gms_len_update)
          if (maxLen < len) // found a match longer than maxLen. save <len,dist> pair in distances
          {
            maxLen = (UInt32)len;
            *d++ = (UInt32)len; // save len of match
            *d++ = delta - 1; // save dist of match
            if (len == lenLimit) // match of lenLimit found. No need to check further. return.
            {
              *ptr1 = pair0;
              *ptr0 = pair[1];
              return d;
            }
          }
        }
        /* pb[0 to len-1] == cur[0 to len-1]
        * pb[len] != cur[len]. Mismatch on this byte. */
        if (pb[len] < cur[len]) // previous byte is smaller. set curMatch to left child node 
        {
          *ptr1 = curMatch;
          // const UInt32 curMatch2 = pair[1];
          // if (curMatch2 >= curMatch) { *ptr0 = *ptr1 = kEmptyHashValue;  return NULL; }
          // curMatch = curMatch2;
          curMatch = pair[1]; // propagate matched position
          ptr1 = pair + 1; // move to matched right child node
          len1 = len;
        }
        else // previous byte is larger. set curMatch to right child node 
        {
          *ptr0 = curMatch;
          curMatch = pair[0];// propagate matched position
          ptr0 = pair; // move to matched left child node
          len0 = len;
        }
      }
    } while (--cutValue && cmCheck < curMatch); // search completed return
  }

  *ptr0 = *ptr1 = kEmptyHashValue;
  return d;
}

// Exact same code as AOCL_GetMatchesSpec1, except this does not update distances
static void AOCL_SkipMatchesSpec(UInt32 lenLimit, UInt32 curMatch, UInt32 pos, const Byte* cur, CLzRef* son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue)
{
    CLzRef* ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1;
    CLzRef* ptr1 = son + ((size_t)_cyclicBufferPos << 1);
    unsigned len0 = 0, len1 = 0;

    UInt32 cmCheck;

    cmCheck = (UInt32)(pos - _cyclicBufferSize);
    if ((UInt32)pos <= _cyclicBufferSize)
        cmCheck = 0;

    if (// curMatch >= pos ||  // failure
        cmCheck < curMatch)
        {
        do
        {
            const UInt32 delta = pos - curMatch;
            {
                CLzRef* pair = son + ((size_t)(_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)) << 1);
                const Byte* pb = cur - delta;
                unsigned len = (len0 < len1 ? len0 : len1);
                if (pb[len] == cur[len])
                {
                    ++len;
                    AOCL_FIND_MATCHING_BYTES_LEN(len, lenLimit, cur, pb, aocl_bt_sms_len_update)
                    {
                        if (len == lenLimit)
                        {
                            *ptr1 = pair[0];
                            *ptr0 = pair[1];
                            return;
                        }
                    }
                }
                if (pb[len] < cur[len])
                {
                    *ptr1 = curMatch;
                    curMatch = pair[1];
                    ptr1 = pair + 1;
                    len1 = len;
                }
                else
                {
                    *ptr0 = curMatch;
                    curMatch = pair[0];
                    ptr0 = pair;
                    len0 = len;
                }
            }
        } while (--cutValue && cmCheck < curMatch);
        }

        *ptr0 = *ptr1 = kEmptyHashValue;
        return;
}
#endif

// Exact same code as GetMatchesSpec1, except this does not update distances
static void SkipMatchesSpec(UInt32 lenLimit, UInt32 curMatch, UInt32 pos, const Byte *cur, CLzRef *son,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize, UInt32 cutValue)
{
  CLzRef *ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1;
  CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1);
  unsigned len0 = 0, len1 = 0;

  UInt32 cmCheck;

  cmCheck = (UInt32)(pos - _cyclicBufferSize);
  if ((UInt32)pos <= _cyclicBufferSize)
    cmCheck = 0;

  if (// curMatch >= pos ||  // failure
      cmCheck < curMatch)
  do
  {
    const UInt32 delta = pos - curMatch;
    {
      CLzRef *pair = son + ((size_t)(_cyclicBufferPos - delta + ((delta > _cyclicBufferPos) ? _cyclicBufferSize : 0)) << 1);
      const Byte *pb = cur - delta;
      unsigned len = (len0 < len1 ? len0 : len1);
      if (pb[len] == cur[len])
      {
        while (++len != lenLimit)
          if (pb[len] != cur[len])
            break;
        {
          if (len == lenLimit)
          {
            *ptr1 = pair[0];
            *ptr0 = pair[1];
            return;
          }
        }
      }
      if (pb[len] < cur[len])
      {
        *ptr1 = curMatch;
        curMatch = pair[1];
        ptr1 = pair + 1;
        len1 = len;
      }
      else
      {
        *ptr0 = curMatch;
        curMatch = pair[0];
        ptr0 = pair;
        len0 = len;
      }
    }
  }
  while(--cutValue && cmCheck < curMatch);
  
  *ptr0 = *ptr1 = kEmptyHashValue;
  return;
}


#define MOVE_POS \
  ++p->cyclicBufferPos; \
  p->buffer++; \
  { const UInt32 pos1 = p->pos + 1; p->pos = pos1; if (pos1 == p->posLimit) MatchFinder_CheckLimits(p); }

#define MOVE_POS_RET MOVE_POS return distances;

MY_NO_INLINE
static void MatchFinder_MovePos(CMatchFinder *p)
{
  /* we go here at the end of stream data, when (avail < num_hash_bytes)
     We don't update sons[cyclicBufferPos << btMode].
     So (sons) record will contain junk. And we cannot resume match searching
     to normal operation, even if we will provide more input data in buffer.
     p->sons[p->cyclicBufferPos << p->btMode] = 0;  // kEmptyHashValue
     if (p->btMode)
        p->sons[(p->cyclicBufferPos << p->btMode) + 1] = 0;  // kEmptyHashValue
  */
  MOVE_POS;
}

#define GET_MATCHES_HEADER2(minLen, ret_op) \
  unsigned lenLimit; UInt32 hv; Byte *cur; UInt32 curMatch; \
  lenLimit = (unsigned)p->lenLimit; { if (lenLimit < minLen) { MatchFinder_MovePos(p); ret_op; }} \
  cur = p->buffer;

#define GET_MATCHES_HEADER(minLen) GET_MATCHES_HEADER2(minLen, return distances)
#define SKIP_HEADER(minLen)   do { GET_MATCHES_HEADER2(minLen, continue)

#define MF_PARAMS(p)  lenLimit, curMatch, p->pos, p->buffer, p->son, p->cyclicBufferPos, p->cyclicBufferSize, p->cutValue

#define AOCL_MF_PARAMS_CEHC(p)  lenLimit, curMatch, hv, p->pos, p->buffer, p->son, p->cyclicBufferPos, p->cyclicBufferSize, p->cutValue

#define SKIP_FOOTER  SkipMatchesSpec(MF_PARAMS(p)); MOVE_POS; } while (--num);

#define GET_MATCHES_FOOTER_BASE(_maxLen_, func) \
  distances = func(MF_PARAMS(p), \
  distances, (UInt32)_maxLen_); MOVE_POS_RET;

#define GET_MATCHES_FOOTER_BT(_maxLen_) \
  GET_MATCHES_FOOTER_BASE(_maxLen_, GetMatchesSpec1)

#define GET_MATCHES_FOOTER_HC(_maxLen_) \
  GET_MATCHES_FOOTER_BASE(_maxLen_, Hc_GetMatchesSpec)



#define UPDATE_maxLen { \
    const ptrdiff_t diff = (ptrdiff_t)0 - (ptrdiff_t)d2; \
    const Byte *c = cur + maxLen; \
    const Byte *lim = cur + lenLimit; \
    for (; c != lim; c++) if (*(c + diff) != *c) break; \
    maxLen = (unsigned)(c - cur); }

#ifdef AOCL_LZMA_OPT
#define AOCL_MOVE_POS \
  ++p->cyclicBufferPos; \
  p->buffer++; \
  { const UInt32 pos1 = p->pos + 1; p->pos = pos1; if (pos1 == p->posLimit) AOCL_MatchFinder_CheckLimits(p); }

#define AOCL_MOVE_POS_RET AOCL_MOVE_POS return distances;

MY_NO_INLINE
static void AOCL_MatchFinder_MovePos(CMatchFinder* p)
{
    /* we go here at the end of stream data, when (avail < num_hash_bytes)
       We don't update sons[cyclicBufferPos << btMode].
       So (sons) record will contain junk. And we cannot resume match searching
       to normal operation, even if we will provide more input data in buffer.
       p->sons[p->cyclicBufferPos << p->btMode] = 0;  // kEmptyHashValue
       if (p->btMode)
          p->sons[(p->cyclicBufferPos << p->btMode) + 1] = 0;  // kEmptyHashValue
    */
    AOCL_MOVE_POS;
}

#define AOCL_GET_MATCHES_HEADER2(minLen, ret_op) \
  unsigned lenLimit; UInt32 hv; Byte *cur; UInt32 curMatch; \
  lenLimit = (unsigned)p->lenLimit; { if (lenLimit < minLen) { AOCL_MatchFinder_MovePos(p); ret_op; }} \
  cur = p->buffer;

#define AOCL_GET_MATCHES_HEADER(minLen) AOCL_GET_MATCHES_HEADER2(minLen, return distances)
#define AOCL_SKIP_HEADER(minLen)   do { AOCL_GET_MATCHES_HEADER2(minLen, continue)

#define AOCL_SKIP_FOOTER  AOCL_SkipMatchesSpec(MF_PARAMS(p)); AOCL_MOVE_POS; } while (--num);

#define AOCL_GET_MATCHES_FOOTER_BASE(_maxLen_, func) \
  distances = func(MF_PARAMS(p), \
  distances, (UInt32)_maxLen_); AOCL_MOVE_POS_RET;

#define AOCL_GET_MATCHES_FOOTER_BASE_CEHC(_maxLen_, func) \
  distances = func(AOCL_MF_PARAMS_CEHC(p), \
  distances, (UInt32)_maxLen_); AOCL_MOVE_POS_RET;

#define AOCL_GET_MATCHES_FOOTER_BT(_maxLen_) \
  AOCL_GET_MATCHES_FOOTER_BASE(_maxLen_, AOCL_GetMatchesSpec1)

#define AOCL_GET_MATCHES_FOOTER_HC_8(_maxLen_) \
  AOCL_GET_MATCHES_FOOTER_BASE_CEHC(_maxLen_, AOCL_Hc_GetMatchesSpec_8)

#define AOCL_GET_MATCHES_FOOTER_HC_16(_maxLen_) \
  AOCL_GET_MATCHES_FOOTER_BASE_CEHC(_maxLen_, AOCL_Hc_GetMatchesSpec_16)

#define AOCL_GET_MATCHES_FOOTER_HC(_maxLen_) \
  AOCL_GET_MATCHES_FOOTER_BASE(_maxLen_, AOCL_Hc_GetMatchesSpec)
#endif

static UInt32* Bt2_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  GET_MATCHES_HEADER(2)
  HASH2_CALC;
  curMatch = p->hash[hv];
  p->hash[hv] = p->pos;
  GET_MATCHES_FOOTER_BT(1)
}

UInt32* Bt3Zip_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  GET_MATCHES_HEADER(3)
  HASH_ZIP_CALC;
  curMatch = p->hash[hv];
  p->hash[hv] = p->pos;
  GET_MATCHES_FOOTER_BT(2)
}


#define SET_mmm  \
  mmm = p->cyclicBufferSize; \
  if (pos < mmm) \
    mmm = pos;


static UInt32* Bt3_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  UInt32 mmm;
  UInt32 h2, d2, pos;
  unsigned maxLen;
  UInt32 *hash;
  GET_MATCHES_HEADER(3)

  HASH3_CALC;

  hash = p->hash;
  pos = p->pos;

  d2 = pos - hash[h2];

  curMatch = (hash + kFix3HashSize)[hv];
  
  hash[h2] = pos;
  (hash + kFix3HashSize)[hv] = pos;

  SET_mmm

  maxLen = 2;

  if (d2 < mmm && *(cur - d2) == *cur)
  {
    UPDATE_maxLen
    distances[0] = (UInt32)maxLen;
    distances[1] = d2 - 1;
    distances += 2;
    if (maxLen == lenLimit)
    {
      SkipMatchesSpec(MF_PARAMS(p));
      MOVE_POS_RET; // process and return distances
    }
  }
  
  GET_MATCHES_FOOTER_BT(maxLen) // process and return distances
}

/* Compute hash, locate node in binTree and find matches at these positions.
* Hash is computed on 4 bytes: Hash(p->buffer[0-4]).
* GetMatchesSpec1() is called to find matches in binTree.
*
* @outputs:
* return: number of matched pairs
* distances: matches saved as <len,dist> pairs
*
* @inputs
* p: match finder struct
* p->buffer: pointer to current byte in input stream */
static UInt32* Bt4_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  UInt32 mmm;
  UInt32 h2, h3, d2, d3, pos;
  unsigned maxLen;
  UInt32 *hash;
  GET_MATCHES_HEADER(4)

    /* read 4 bytes from input stream p->buffer and compute hash for it
  * cur = p->buffer
  * cur[0-4] are used to compute hash
  *
  * compute hashes: h2, h3, hv
  * 3 hash tables are created and stored in 'hash'
  * H2 (10-bit hash) created using c[0-1] stored from [0, kFix3HashSize)
  * H3 (16-bit hash) created using c[0-2] stored from [kFix3HashSize, kFix4HashSize)
  * HV (hashMask   ) created using c[0-3] stored from [kFix4HashSize, max hash size) */
  HASH4_CALC;

  hash = p->hash;
  pos = p->pos;

  /* Eg:
*                 curMatch                                                   pos
*              ------------------------------------------------------------------
* Data stream: ...abcd..............abcw.................abxy................abcd
*                                                        <---------d2------->
*                                   <-------------------d3------------------>
*                 <---------------------------delta------------------------->
*
* How does hash guarantee 2/3/4 bytes are same?
* Can't we have 2 sequences e.g. abcd, uvwx map to the same hash? */
  d2 = pos - hash                  [h2]; // dist to last 2 byte match
  d3 = pos - (hash + kFix3HashSize)[h3]; // dist to last 3 byte match
  
  curMatch = (hash + kFix4HashSize)[hv]; // dist to last 4 byte match

  hash                  [h2] = pos;
  (hash + kFix3HashSize)[h3] = pos;
  (hash + kFix4HashSize)[hv] = pos;

  SET_mmm

  maxLen = 3;
  
  for (;;)
  {
    if (d2 < mmm && *(cur - d2) == *cur)
    {
      distances[0] = 2;
      distances[1] = d2 - 1;
      distances += 2;
      if (*(cur - d2 + 2) == cur[2])
      {
        // distances[-2] = 3;
      }
      else if (d3 < mmm && *(cur - d3) == *cur)
      {
        d2 = d3;
        distances[1] = d3 - 1;
        distances += 2;
      }
      else
        break;
    }
    else if (d3 < mmm && *(cur - d3) == *cur)
    {
      d2 = d3;
      distances[1] = d3 - 1;
      distances += 2;
    }
    else
      break;

    /* 2 or 3 bytes are already matched at this point (maxLen)
  * starting from maxLen+1, find how many bytes match at offset d2
  * update maxLen to this length
  * Data stream: ....ab****..............ab$$$$
  *                  <--------d2-------->
  *                  match *s and $s */
    UPDATE_maxLen
    distances[-2] = (UInt32)maxLen;

    /* if already reached lenLimit, there is no need to check for better 
    * matches at d3 and curMatch offsets, as d2 will have the smallest offset */ 
    if (maxLen == lenLimit)
    {
      SkipMatchesSpec(MF_PARAMS(p));
      MOVE_POS_RET
    }
    break;
  }
  
  GET_MATCHES_FOOTER_BT(maxLen)
}


static UInt32* Bt5_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  UInt32 mmm;
  UInt32 h2, h3, d2, d3, maxLen, pos;
  UInt32 *hash;
  GET_MATCHES_HEADER(5)

  HASH5_CALC;

  hash = p->hash;
  pos = p->pos;

  d2 = pos - hash                  [h2];
  d3 = pos - (hash + kFix3HashSize)[h3];
  // d4 = pos - (hash + kFix4HashSize)[h4];

  curMatch = (hash + kFix5HashSize)[hv];

  hash                  [h2] = pos;
  (hash + kFix3HashSize)[h3] = pos;
  // (hash + kFix4HashSize)[h4] = pos;
  (hash + kFix5HashSize)[hv] = pos;

  SET_mmm

  maxLen = 4;

  for (;;)
  {
    if (d2 < mmm && *(cur - d2) == *cur)
    {
      distances[0] = 2;
      distances[1] = d2 - 1;
      distances += 2;
      if (*(cur - d2 + 2) == cur[2])
      {
      }
      else if (d3 < mmm && *(cur - d3) == *cur)
      {
        distances[1] = d3 - 1;
        distances += 2;
        d2 = d3;
      }
      else
        break;
    }
    else if (d3 < mmm && *(cur - d3) == *cur)
    {
      distances[1] = d3 - 1;
      distances += 2;
      d2 = d3;
    }
    else
      break;

    distances[-2] = 3;
    if (*(cur - d2 + 3) != cur[3])
      break;
    UPDATE_maxLen
    distances[-2] = (UInt32)maxLen;
    if (maxLen == lenLimit)
    {
      SkipMatchesSpec(MF_PARAMS(p));
      MOVE_POS_RET;
    }
    break;
  }
  
  GET_MATCHES_FOOTER_BT(maxLen)
}


/* Compute hash, locate node in hash chain and find matches at these positions.
* Hash is computed on 4 bytes: Hash(p->buffer[0-4]).
* Hc_GetMatchesSpec() is called to find matches in hashChain.
*
* @outputs:
* return: distances: matches saved as <len,dist> pairs
*
* @inputs
* p: match finder struct
* p->buffer: pointer to current byte in input stream */
static UInt32* Hc4_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  UInt32 mmm;
  UInt32 h2, h3, d2, d3, pos;
  unsigned maxLen;
  UInt32 *hash;
  GET_MATCHES_HEADER(4)

  HASH4_CALC;

  hash = p->hash;
  pos = p->pos;
  
  d2 = pos - hash                  [h2];
  d3 = pos - (hash + kFix3HashSize)[h3];
  curMatch = (hash + kFix4HashSize)[hv];

  hash                  [h2] = pos;
  (hash + kFix3HashSize)[h3] = pos;
  (hash + kFix4HashSize)[hv] = pos;

  SET_mmm

  maxLen = 3;

  for (;;)
  {
    if (d2 < mmm && *(cur - d2) == *cur)
    {
      distances[0] = 2;
      distances[1] = d2 - 1;
      distances += 2;
      if (*(cur - d2 + 2) == cur[2])
      {
        // distances[-2] = 3;
      }
      else if (d3 < mmm && *(cur - d3) == *cur)
      {
        d2 = d3;
        distances[1] = d3 - 1;
        distances += 2;
      }
      else
        break;
    }
    else if (d3 < mmm && *(cur - d3) == *cur)
    {
      d2 = d3;
      distances[1] = d3 - 1;
      distances += 2;
    }
    else
      break;

    UPDATE_maxLen
    distances[-2] = (UInt32)maxLen;
    if (maxLen == lenLimit)
    {
      p->son[p->cyclicBufferPos] = curMatch;
      MOVE_POS_RET;
    }
    break;
  }
  
  GET_MATCHES_FOOTER_HC(maxLen);
}


static UInt32 * Hc5_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  UInt32 mmm;
  UInt32 h2, h3, d2, d3, maxLen, pos;
  UInt32 *hash;
  GET_MATCHES_HEADER(5)

  HASH5_CALC;

  hash = p->hash;
  pos = p->pos;

  d2 = pos - hash                  [h2];
  d3 = pos - (hash + kFix3HashSize)[h3];
  // d4 = pos - (hash + kFix4HashSize)[h4];

  curMatch = (hash + kFix5HashSize)[hv];

  hash                  [h2] = pos;
  (hash + kFix3HashSize)[h3] = pos;
  // (hash + kFix4HashSize)[h4] = pos;
  (hash + kFix5HashSize)[hv] = pos;

  SET_mmm
  
  maxLen = 4;

  for (;;)
  {
    if (d2 < mmm && *(cur - d2) == *cur)
    {
      distances[0] = 2;
      distances[1] = d2 - 1;
      distances += 2;
      if (*(cur - d2 + 2) == cur[2])
      {
      }
      else if (d3 < mmm && *(cur - d3) == *cur)
      {
        distances[1] = d3 - 1;
        distances += 2;
        d2 = d3;
      }
      else
        break;
    }
    else if (d3 < mmm && *(cur - d3) == *cur)
    {
      distances[1] = d3 - 1;
      distances += 2;
      d2 = d3;
    }
    else
      break;

    distances[-2] = 3;
    if (*(cur - d2 + 3) != cur[3])
      break;
    UPDATE_maxLen
    distances[-2] = maxLen;
    if (maxLen == lenLimit)
    {
      p->son[p->cyclicBufferPos] = curMatch;
      MOVE_POS_RET;
    }
    break;
  }
  
  GET_MATCHES_FOOTER_HC(maxLen);
}


UInt32* Hc3Zip_MatchFinder_GetMatches(CMatchFinder *p, UInt32 *distances)
{
  GET_MATCHES_HEADER(3)
  HASH_ZIP_CALC;
  curMatch = p->hash[hv];
  p->hash[hv] = p->pos;
  GET_MATCHES_FOOTER_HC(2)
}


static void Bt2_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  SKIP_HEADER(2)
  {
    HASH2_CALC;
    curMatch = p->hash[hv];
    p->hash[hv] = p->pos;
  }
  SKIP_FOOTER
}

void Bt3Zip_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  SKIP_HEADER(3)
  {
    HASH_ZIP_CALC;
    curMatch = p->hash[hv];
    p->hash[hv] = p->pos;
  }
  SKIP_FOOTER
}

static void Bt3_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  SKIP_HEADER(3)
  {
    UInt32 h2;
    UInt32 *hash;
    HASH3_CALC;
    hash = p->hash;
    curMatch = (hash + kFix3HashSize)[hv];
    hash[h2] =
    (hash + kFix3HashSize)[hv] = p->pos;
  }
  SKIP_FOOTER
}

static void Bt4_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  SKIP_HEADER(4)
  {
    UInt32 h2, h3;
    UInt32 *hash;
    HASH4_CALC;
    hash = p->hash;
    curMatch = (hash + kFix4HashSize)[hv];
    hash                  [h2] =
    (hash + kFix3HashSize)[h3] =
    (hash + kFix4HashSize)[hv] = p->pos;
  }
  SKIP_FOOTER
}

static void Bt5_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  SKIP_HEADER(5)
  {
    UInt32 h2, h3;
    UInt32 *hash;
    HASH5_CALC;
    hash = p->hash;
    curMatch = (hash + kFix5HashSize)[hv];
    hash                  [h2] =
    (hash + kFix3HashSize)[h3] =
    // (hash + kFix4HashSize)[h4] =
    (hash + kFix5HashSize)[hv] = p->pos;
  }
  SKIP_FOOTER
}


#define HC_SKIP_HEADER(minLen) \
    do { if (p->lenLimit < minLen) { MatchFinder_MovePos(p); num--; continue; } { \
    Byte *cur; \
    UInt32 *hash; \
    UInt32 *son; \
    UInt32 pos = p->pos; \
    UInt32 num2 = num; \
    /* (p->pos == p->posLimit) is not allowed here !!! */ \
    { const UInt32 rem = p->posLimit - pos; if (num2 > rem) num2 = rem; } \
    num -= num2; \
    { const UInt32 cycPos = p->cyclicBufferPos; \
      son = p->son + cycPos; \
      p->cyclicBufferPos = cycPos + num2; } \
    cur = p->buffer; \
    hash = p->hash; \
    do { \
    UInt32 curMatch; \
    UInt32 hv;


#define HC_SKIP_FOOTER \
    cur++;  pos++;  *son++ = curMatch; \
    } while (--num2); \
    p->buffer = cur; \
    p->pos = pos; \
    if (pos == p->posLimit) MatchFinder_CheckLimits(p); \
    }} while(num); \


static void Hc4_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  HC_SKIP_HEADER(4)

    UInt32 h2, h3;
    HASH4_CALC;
    curMatch = (hash + kFix4HashSize)[hv];
    hash                  [h2] =
    (hash + kFix3HashSize)[h3] =
    (hash + kFix4HashSize)[hv] = pos;
  
  HC_SKIP_FOOTER
}


static void Hc5_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  HC_SKIP_HEADER(5)
  
    UInt32 h2, h3;
    HASH5_CALC
    curMatch = (hash + kFix5HashSize)[hv];
    hash                  [h2] =
    (hash + kFix3HashSize)[h3] =
    // (hash + kFix4HashSize)[h4] =
    (hash + kFix5HashSize)[hv] = pos;
  
  HC_SKIP_FOOTER
}


void Hc3Zip_MatchFinder_Skip(CMatchFinder *p, UInt32 num)
{
  HC_SKIP_HEADER(3)

    HASH_ZIP_CALC;
    curMatch = hash[hv];
    hash[hv] = pos;

  HC_SKIP_FOOTER
}


void MatchFinder_CreateVTable(CMatchFinder *p, IMatchFinder2 *vTable)
{
  vTable->Init = (Mf_Init_Func)MatchFinder_Init;
  vTable->GetNumAvailableBytes = (Mf_GetNumAvailableBytes_Func)MatchFinder_GetNumAvailableBytes;
  vTable->GetPointerToCurrentPos = (Mf_GetPointerToCurrentPos_Func)MatchFinder_GetPointerToCurrentPos;
  if (!p->btMode)
  {
    if (p->numHashBytes <= 4)
    {
      vTable->GetMatches = (Mf_GetMatches_Func)Hc4_MatchFinder_GetMatches;
      vTable->Skip = (Mf_Skip_Func)Hc4_MatchFinder_Skip;
    }
    else
    {
      vTable->GetMatches = (Mf_GetMatches_Func)Hc5_MatchFinder_GetMatches;
      vTable->Skip = (Mf_Skip_Func)Hc5_MatchFinder_Skip;
    }
  }
  else if (p->numHashBytes == 2)
  {
    vTable->GetMatches = (Mf_GetMatches_Func)Bt2_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt2_MatchFinder_Skip;
  }
  else if (p->numHashBytes == 3)
  {
    vTable->GetMatches = (Mf_GetMatches_Func)Bt3_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt3_MatchFinder_Skip;
  }
  else if (p->numHashBytes == 4)  /* if (p->numHashBytes == 4). Default setting. */
  {
    vTable->GetMatches = (Mf_GetMatches_Func)Bt4_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt4_MatchFinder_Skip;
  }
 else
  {
    vTable->GetMatches = (Mf_GetMatches_Func)Bt5_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt5_MatchFinder_Skip;
  }
}

#ifdef AOCL_LZMA_OPT
#define AOCL_HC_SKIP_HEADER(minLen) \
    do { if (unlikely(p->lenLimit < minLen)) { AOCL_MatchFinder_MovePos(p); num--; continue; } { \
    Byte *cur; \
    UInt32 *hash; \
    UInt32 *son; \
    UInt32 pos = p->pos; \
    UInt32 num2 = num; \
    /* (p->pos == p->posLimit) is not allowed here !!! */ \
    { const UInt32 rem = p->posLimit - pos; if (num2 > rem) num2 = rem; } \
    num -= num2; \
    p->cyclicBufferPos += num2; \
    cur = p->buffer; \
    hash = p->hash; \
    son = p->son; \
    do { \
    UInt32 curMatch; \
    UInt32 hv;

#define AOCL_HC_SKIP_FOOTER \
    cur++;  pos++; \
    } while (--num2); \
    p->buffer = cur; \
    p->pos = pos; \
    if (pos == p->posLimit) AOCL_MatchFinder_CheckLimits(p); \
    }} while(num);

/* Called when: algo = !btMode, cacheEfficientSearch = 1
* HASH_CHAIN_SLOT_SZ: HASH_CHAIN_SLOT_SZ_8 (level < HASH_CHAIN_16_LEVEL), HASH_CHAIN_SLOT_SZ_16 (level >= HASH_CHAIN_16_LEVEL)
* AOCL_HASH_CALC: AOCL_HASH4_CALC (numHashBytes <= 4), AOCL_HASH5_CALC (numHashBytes > 4) */
#define AOCL_HC_MF_GETMATCHES(N, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX, AOCL_HASH_CALC, AOCL_GET_MATCHES_FOOTER_HC) \
    UInt32 mmm; \
    UInt32 h2, d2, pos; \
    unsigned maxLen; \
    UInt32* hash, * son; \
    AOCL_GET_MATCHES_HEADER(N) \
    \
    AOCL_HASH_CALC; \
    \
    hash = p->hash; \
    pos = p->pos; \
    son = p->son; \
    \
    d2 = pos - hash[h2]; \
    AOCL_COMMON_CEHCFIX_GET_HEAD(son, 0, curMatch, 0, hv, \
        HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX, kEmptySonValue) /* update head position pointer */ \
    hash[h2] = pos; \
    \
    SET_mmm \
    \
    maxLen = 3; \
    \
    for (;;) \
    { \
        if (d2 < mmm && *(cur - d2) == *cur) \
        { \
            distances[0] = 2; \
            distances[1] = d2 - 1; \
            distances += 2; \
            if (*(cur - d2 + 2) != cur[2]) \
                break; \
        } \
        else \
            break; \
        \
        UPDATE_maxLen \
        distances[-2] = (UInt32)maxLen; \
        if (maxLen == lenLimit) \
        { \
            AOCL_COMMON_CEHCFIX_INSERT(son, 0, curMatch, 0, pos, hv, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) /* set pos at current head */ \
            AOCL_MOVE_POS_RET; \
        } \
        break; \
    } \
    \
    AOCL_GET_MATCHES_FOOTER_HC(maxLen); \

/* Called when: algo = !btMode, cacheEfficientSearch = 1
* HASH_CHAIN_SLOT_SZ: HASH_CHAIN_SLOT_SZ_8 (level < HASH_CHAIN_16_LEVEL), HASH_CHAIN_SLOT_SZ_16 (level >= HASH_CHAIN_16_LEVEL)
* AOCL_HASH_CALC: AOCL_HASH4_CALC (numHashBytes <= 4), AOCL_HASH5_CALC (numHashBytes > 4) */
#define AOCL_HC_MF_SKIP(N, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX, AOCL_HASH_CALC) \
    AOCL_HC_SKIP_HEADER(N) \
    \
    UInt32 h2; \
    AOCL_HASH_CALC; \
    AOCL_COMMON_CEHCFIX_GET_HEAD(son, 0, curMatch, 0, hv, \
        HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX, kEmptySonValue) /* update head position pointer */ \
    hash[h2] = pos; \
    AOCL_COMMON_CEHCFIX_INSERT(son, 0, curMatch, 0, pos, hv, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX) /* set pos at current head */ \
    \
    AOCL_HC_SKIP_FOOTER

/* Compute hash, locate node in hash chain and find matches at these positions.
* Hash is computed on 4 bytes: Hash(p->buffer[0-4]).
* AOCL_Hc_GetMatchesSpec() is called to find matches in hashChain.
*
* @outputs:
* return: distances: matches saved as <len,dist> pairs
*
* @inputs
* p: match finder struct
* p->buffer: pointer to current byte in input stream 
*
* Changes wrt Hc4_MatchFinder_GetMatches:
* + h3 hash table not used. Simplified if conditions to suit this.
* 
* Called when: algo = !btMode, level < HASH_CHAIN_16_LEVEL, numHashBytes <= 4, cacheEfficientSearch = 1
* Calls: AOCL_Hc_GetMatchesSpec_8: Cache efficient hash chain with block size 8 */
static UInt32* AOCL_Hc4_MatchFinder_GetMatches_Cehc8(CMatchFinder* p, UInt32* distances)
{
    AOCL_HC_MF_GETMATCHES(4, HASH_CHAIN_SLOT_SZ_8, HASH_CHAIN_MAX_8,
        AOCL_HASH4_CALC, AOCL_GET_MATCHES_FOOTER_HC_8)
}

/* 
* @brief: Skip is called after a match of certain length is found. 
*         Skip bytes based on this. While skipping,
*         nodes need to be added to the dictionary
* 
* Changes wrt Hc4_MatchFinder_Skip:
* + h3 hash table not used.*/
static void AOCL_Hc4_MatchFinder_Skip_Cehc8(CMatchFinder* p, UInt32 num)
{
    AOCL_HC_MF_SKIP(4, HASH_CHAIN_SLOT_SZ_8, HASH_CHAIN_MAX_8, AOCL_HASH4_CALC)
}

/* 16-node block equivalent of AOCL_Hc4_MatchFinder_GetMatches_Cehc8 
*
* Called when: algo = !btMode, level >= HASH_CHAIN_16_LEVEL, numHashBytes <= 4, cacheEfficientSearch = 1
* Calls: AOCL_Hc_GetMatchesSpec_16: Cache efficient hash chain with block size 16 */
static UInt32* AOCL_Hc4_MatchFinder_GetMatches_Cehc16(CMatchFinder* p, UInt32* distances)
{
    AOCL_HC_MF_GETMATCHES(4, HASH_CHAIN_SLOT_SZ_16, HASH_CHAIN_MAX_16,
        AOCL_HASH4_CALC, AOCL_GET_MATCHES_FOOTER_HC_16)
}

/* 16-node block equivalent of AOCL_Hc4_MatchFinder_Skip_Cehc8 */
static void AOCL_Hc4_MatchFinder_Skip_Cehc16(CMatchFinder* p, UInt32 num)
{
    AOCL_HC_MF_SKIP(4, HASH_CHAIN_SLOT_SZ_16, HASH_CHAIN_MAX_16, AOCL_HASH4_CALC)
}

/* @brief: Compute hash, locate node in hash chain and find matches at these positions.
* Hash is computed on 4 bytes: Hash(p->buffer[0-4]).
* AOCL_Hc_GetMatchesSpec() is called to find matches in hashChain.
*
* @outputs:
* return: distances: matches saved as <len,dist> pairs
*
* @inputs
* p: match finder struct
* p->buffer: pointer to current byte in input stream
*
* Changes wrt Hc4_MatchFinder_GetMatches:
* + h3 hash table not used. Simplified if conditions to suit this.
* + calls AOCL_Hc_GetMatchesSpec() to find matches 
*
* Called when: algo = !btMode, numHashBytes <= 4, cacheEfficientSearch = 0
* Calls: AOCL_Hc_GetMatchesSpec: Reference style hash chain */
static UInt32* AOCL_Hc4_MatchFinder_GetMatches(CMatchFinder* p, UInt32* distances)
{
    UInt32 mmm;
    UInt32 h2, d2, pos;
    unsigned maxLen;
    UInt32* hash;
    GET_MATCHES_HEADER(4)

    AOCL_HASH4_CALC;

    hash = p->hash;
    pos = p->pos;

    d2 = pos - hash[h2];
    curMatch = (hash + kFix3HashSize)[hv];

    hash[h2] = pos;
    (hash + kFix3HashSize)[hv] = pos;

    SET_mmm

    maxLen = 3;

    for (;;)
    {
        if (d2 < mmm && *(cur - d2) == *cur)
        {
            distances[0] = 2;
            distances[1] = d2 - 1;
            distances += 2;
            if (*(cur - d2 + 2) != cur[2])
                break;
        }
        else
            break;

        UPDATE_maxLen
        distances[-2] = (UInt32)maxLen;
        if (maxLen == lenLimit)
        {
            p->son[p->cyclicBufferPos] = curMatch;
            MOVE_POS_RET;
        }
        break;
    }

    AOCL_GET_MATCHES_FOOTER_HC(maxLen);
}

/* @brief: Move buffer pointers while adding new entries to dictionary.
* Changes wrt Hc4_MatchFinder_Skip:
* + h3 hash table not used.*/
static void AOCL_Hc4_MatchFinder_Skip(CMatchFinder* p, UInt32 num)
{
    HC_SKIP_HEADER(4)

    UInt32 h2;
    AOCL_HASH4_CALC;
    curMatch = (hash + kFix3HashSize)[hv];
    hash[h2] = (hash + kFix3HashSize)[hv] = pos;

    HC_SKIP_FOOTER
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_GetMatches_Cehc8 
*
* Called when: algo = !btMode, level < HASH_CHAIN_16_LEVEL, numHashBytes > 4, cacheEfficientSearch = 1
* Calls: AOCL_Hc_GetMatchesSpec_8: Cache efficient hash chain with block size 8 */
static UInt32* AOCL_Hc5_MatchFinder_GetMatches_Cehc8(CMatchFinder* p, UInt32* distances)
{
    AOCL_HC_MF_GETMATCHES(5, HASH_CHAIN_SLOT_SZ_8, HASH_CHAIN_MAX_8,
        AOCL_HASH5_CALC, AOCL_GET_MATCHES_FOOTER_HC_8)
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_Skip_Cehc8 */
static void AOCL_Hc5_MatchFinder_Skip_Cehc8(CMatchFinder* p, UInt32 num)
{
    AOCL_HC_MF_SKIP(5, HASH_CHAIN_SLOT_SZ_8, HASH_CHAIN_MAX_8, AOCL_HASH5_CALC)
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_GetMatches_Cehc16
*
* Called when: algo = !btMode, level >= HASH_CHAIN_16_LEVEL, numHashBytes > 4, cacheEfficientSearch = 1
* Calls: AOCL_Hc_GetMatchesSpec_16: Cache efficient hash chain with block size 16 */
static UInt32* AOCL_Hc5_MatchFinder_GetMatches_Cehc16(CMatchFinder* p, UInt32* distances)
{
    AOCL_HC_MF_GETMATCHES(5, HASH_CHAIN_SLOT_SZ_16, HASH_CHAIN_MAX_16,
        AOCL_HASH5_CALC, AOCL_GET_MATCHES_FOOTER_HC_16)
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_Skip_Cehc16 */
static void AOCL_Hc5_MatchFinder_Skip_Cehc16(CMatchFinder* p, UInt32 num)
{
    AOCL_HC_MF_SKIP(5, HASH_CHAIN_SLOT_SZ_16, HASH_CHAIN_MAX_16, AOCL_HASH5_CALC)
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_GetMatches 
*
* Called when: algo = !btMode, numHashBytes > 4, cacheEfficientSearch = 0
* Calls: AOCL_Hc_GetMatchesSpec: Reference style hash chain */
static UInt32* AOCL_Hc5_MatchFinder_GetMatches(CMatchFinder* p, UInt32* distances)
{
    UInt32 mmm;
    UInt32 h2, d2, pos;
    unsigned maxLen;
    UInt32* hash;
    GET_MATCHES_HEADER(5)

    AOCL_HASH5_CALC;

    hash = p->hash;
    pos = p->pos;

    d2 = pos - hash[h2];
    curMatch = (hash + kFix3HashSize)[hv];

    hash[h2] = pos;
    (hash + kFix3HashSize)[hv] = pos;

    SET_mmm

    maxLen = 3;

    for (;;)
    {
        if (d2 < mmm && *(cur - d2) == *cur)
        {
            distances[0] = 2;
            distances[1] = d2 - 1;
            distances += 2;
            if (*(cur - d2 + 2) != cur[2])
                break;
        }
        else
            break;

        UPDATE_maxLen
        distances[-2] = (UInt32)maxLen;
        if (maxLen == lenLimit)
        {
            p->son[p->cyclicBufferPos] = curMatch;
            MOVE_POS_RET;
        }
        break;
    }

    AOCL_GET_MATCHES_FOOTER_HC(maxLen);
}

/* 5-byte hash equivalent of AOCL_Hc4_MatchFinder_Skip */
static void AOCL_Hc5_MatchFinder_Skip(CMatchFinder* p, UInt32 num)
{
    HC_SKIP_HEADER(5)

    UInt32 h2;
    AOCL_HASH5_CALC;
    curMatch = (hash + kFix3HashSize)[hv];
    hash[h2] = (hash + kFix3HashSize)[hv] = pos;

    HC_SKIP_FOOTER
}

/* Compute hash, locate node in binTree and find matches at these positions.
* Hash is computed on 4 bytes: Hash(p->buffer[0-4]).
* GetMatchesSpec1() is called to find matches in binTree.
*
* @outputs:
* return: number of matched pairs
* distances: matches saved as <len,dist> pairs
*
* @inputs
* p: match finder struct
* p->buffer: pointer to current byte in input stream
* 
* Changes wrt Bt4_MatchFinder_GetMatches:
* + h3 hash table not used. Simplified if conditions to suit this.
*
* Called when: algo = btMode, numHashBytes = 4
* Calls: AOCL_GetMatchesSpec1: Reference style BST */
static UInt32* AOCL_Bt4_MatchFinder_GetMatches(CMatchFinder* p, UInt32* distances)
{
  UInt32 mmm;
  UInt32 h2, d2, pos;
  unsigned maxLen;
  UInt32* hash;
  AOCL_GET_MATCHES_HEADER(4)

    /* read 4 bytes from input stream p->buffer and compute hash for it
    * cur = p->buffer
    * cur[0-4] are used to compute hash
    *
    * compute hashes: h2, hv
    * 2 hash tables are created and stored in 'hash'
    * H2 (10-bit hash) created using c[0-1] stored from [0, kFix3HashSize)
    * HV (hashMask   ) created using c[0-3] stored from [kFix3HashSize, max hash size) */
    AOCL_HASH4_CALC;

  hash = p->hash;
  pos = p->pos;

  /* Eg:
*                 curMatch                                                   pos
*              ------------------------------------------------------------------
* Data stream: ...abcd..............abcw.................abxy................abcd
*                                                        <---------d2------->
*                                   <-------------------d3------------------>
*                 <---------------------------delta------------------------->
*
* How does hash guarantee 2/3/4 bytes are same?
* Can't we have 2 sequences e.g. abcd, uvwx map to the same hash? */
  d2 = pos - hash[h2]; // dist to last 2 byte match
  curMatch = (hash + kFix3HashSize)[hv]; // dist to last 4 byte match

  hash[h2] = pos;
  (hash + kFix3HashSize)[hv] = pos;

  SET_mmm

    maxLen = 3;

  for (;;) //Used only to provide exit points via break. Runs only once.
  {
    if (d2 < mmm && *(cur - d2) == *cur)
    {
      distances[0] = 2;
      distances[1] = d2 - 1;
      distances += 2;
      if (*(cur - d2 + 2) != cur[2])
        break; //no need to look for matches below min length
    }
    else
      break;

    /* 2 or 3 bytes are already matched at this point (maxLen)
  * starting from maxLen+1, find how many bytes match at offset d2
  * update maxLen to this length
  * Data stream: ....ab****..............ab$$$$
  *                  <--------d2-------->
  *                  match *s and $s */
    UPDATE_maxLen
      distances[-2] = (UInt32)maxLen;

    /* if already reached lenLimit, there is no need to check for better
    * matches at d3 and curMatch offsets, as d2 will have the smallest offset */
    if (maxLen == lenLimit)
    {
        AOCL_SkipMatchesSpec(MF_PARAMS(p));
        AOCL_MOVE_POS_RET
    }
    break;
  }

  AOCL_GET_MATCHES_FOOTER_BT(maxLen)
}

/* Changes wrt Bt4_MatchFinder_Skip:
* + h3 hash table not used.*/
static void AOCL_Bt4_MatchFinder_Skip(CMatchFinder* p, UInt32 num)
{
    AOCL_SKIP_HEADER(4)
  {
    UInt32 h2;
    UInt32* hash;
    AOCL_HASH4_CALC;
    hash = p->hash;
    curMatch = (hash + kFix3HashSize)[hv];
    hash[h2] =
      (hash + kFix3HashSize)[hv] = p->pos;
  }
    AOCL_SKIP_FOOTER
}

#endif

#ifdef AOCL_LZMA_OPT
/* Changes wrt MatchFinder_CreateVTable:
* + GetMatches and Skip function pointers directed to AOCL_ optimized
*   routines for p->numHashBytes = 4, 5 [Default hash bytes used based on level].
.*/
void AOCL_MatchFinder_CreateVTable(CMatchFinder* p, IMatchFinder2* vTable) {
  vTable->Init = (Mf_Init_Func)AOCL_MatchFinder_Init;
  vTable->GetNumAvailableBytes = (Mf_GetNumAvailableBytes_Func)MatchFinder_GetNumAvailableBytes;
  vTable->GetPointerToCurrentPos = (Mf_GetPointerToCurrentPos_Func)MatchFinder_GetPointerToCurrentPos;
  if (!p->btMode)
  {
      if (!p->cacheEfficientSearch) {
          if (p->numHashBytes <= 4) {
              LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc4_MatchFinder without cehc");
              vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc4_MatchFinder_GetMatches;
              vTable->Skip = (Mf_Skip_Func)AOCL_Hc4_MatchFinder_Skip;
          }
          else {
            LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc5_MatchFinder without cehc");
              vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc5_MatchFinder_GetMatches;
              vTable->Skip = (Mf_Skip_Func)AOCL_Hc5_MatchFinder_Skip;
          }
      }
      else {
          if (p->numHashBytes <= 4)
          {
              if (p->level < HASH_CHAIN_16_LEVEL) {
                  LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc4_MatchFinder with cehc of block size 8");
                  vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc4_MatchFinder_GetMatches_Cehc8;
                  vTable->Skip = (Mf_Skip_Func)AOCL_Hc4_MatchFinder_Skip_Cehc8;
              }
              else {
                LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc4_MatchFinder with cehc of block size 16");
                  vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc4_MatchFinder_GetMatches_Cehc16;
                  vTable->Skip = (Mf_Skip_Func)AOCL_Hc4_MatchFinder_Skip_Cehc16;
              }
          }
          else
          {
              if (p->level < HASH_CHAIN_16_LEVEL) {
                LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc5_MatchFinder with cehc of block size 8");
                  vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc5_MatchFinder_GetMatches_Cehc8;
                  vTable->Skip = (Mf_Skip_Func)AOCL_Hc5_MatchFinder_Skip_Cehc8;
              }
              else {
                LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Hc5_MatchFinder with cehc of block size 16");
                  vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Hc5_MatchFinder_GetMatches_Cehc16;
                  vTable->Skip = (Mf_Skip_Func)AOCL_Hc5_MatchFinder_Skip_Cehc16;
              }
          }
      }
  }
  else if (p->numHashBytes == 2)
  {
    LOG_UNFORMATTED(DEBUG, logCtx, "Using Bt2_MatchFinder");
    vTable->GetMatches = (Mf_GetMatches_Func)Bt2_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt2_MatchFinder_Skip;
  }
  else if (p->numHashBytes == 3)
  {
    LOG_UNFORMATTED(DEBUG, logCtx, "Using Bt3_MatchFinder");
    vTable->GetMatches = (Mf_GetMatches_Func)Bt3_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt3_MatchFinder_Skip;
  }
  else if (p->numHashBytes == 4)  /* Default setting. */
  {
    LOG_UNFORMATTED(DEBUG, logCtx, "Using AOCL_Bt4_MatchFinder");
    vTable->GetMatches = (Mf_GetMatches_Func)AOCL_Bt4_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)AOCL_Bt4_MatchFinder_Skip;
  }
  else
  {
    LOG_UNFORMATTED(DEBUG, logCtx, "Using Bt5_MatchFinder");
    vTable->GetMatches = (Mf_GetMatches_Func)Bt5_MatchFinder_GetMatches;
    vTable->Skip = (Mf_Skip_Func)Bt5_MatchFinder_Skip;
  }
}
#endif

#ifdef AOCL_UNIT_TEST
#ifdef AOCL_LZMA_OPT
/* Move these APIs within the scope of gtest once the framework is ready */
void Test_HC_MatchFinder_Normalize3(UInt32 subValue, CLzRef* hash, CLzRef* son,
    Byte btMode, UInt32 fixedHashSize, UInt32 cyclicBufferSize, UInt32 hashSizeSum,
    size_t numRefs, UInt32 hashMask, int level) {
    CMatchFinder p;
    p.hash = hash;
    p.son = son;
    p.btMode = btMode;
    p.fixedHashSize = fixedHashSize;
    p.cyclicBufferSize = cyclicBufferSize;
    p.hashSizeSum = hashSizeSum;
    p.numRefs = numRefs;
    p.hashMask = hashMask;
    p.level = level;
    p.expectedDataSize = MIN_SIZE_FOR_CE_HC_ON;
    p.cacheEfficientSearch = 1;

    size_t numSonRefs = p.cyclicBufferSize;
    if (p.btMode)
        numSonRefs <<= 1;
    MatchFinder_Normalize3(subValue, p.hash, (size_t)p.hashSizeSum + numSonRefs);
}

void Test_AOCL_HC_MatchFinder_Normalize3(UInt32 subValue, CLzRef* hash, CLzRef* son,
    Byte btMode, UInt32 fixedHashSize, UInt32 cyclicBufferSize, UInt32 hashSizeSum,
    size_t numRefs, UInt32 hashMask, int level) {
    CMatchFinder p;
    p.hash = hash;
    p.son = son;
    p.btMode = btMode;
    p.fixedHashSize = fixedHashSize;
    p.cyclicBufferSize = cyclicBufferSize;
    p.hashSizeSum = hashSizeSum;
    p.numRefs = numRefs;
    p.hashMask = hashMask;
    p.level = level;
    p.expectedDataSize = MIN_SIZE_FOR_CE_HC_ON;
    p.cacheEfficientSearch = 1;

    AOCL_MatchFinder_Normalize3(subValue, &p);
}

int Test_AOCL_Find_Matching_Bytes_Len(int startLen, Byte* data1, Byte* data2, int limit) {
    int len = startLen;
    AOCL_FIND_MATCHING_BYTES_LEN(len, limit, data1, data2, exit_point_test)
    return len;
}

UInt32 Test_Compute_Hash_Mask(UInt32 sz, UInt32 block_cnt) {
    UInt32 hs = sz;
    AOCL_HC_COMPUTE_HASH_MASK(hs)
    return hs;
}

UInt32 Test_Compute_Hash(Byte* cur, CMatchFinder* p) {
    UInt32 h2, hv;
    AOCL_HASH5_CALC;
    h2 -= h2; hv += h2; //dummy operations to suppress unused variable warning
    return hv;
}

UInt32 Test_Circular_Inc(UInt32 hcHead, UInt32 HASH_CHAIN_SLOT_SZ, UInt32 HASH_CHAIN_MAX) {
    hcHead = CIRC_INC_HEAD(hcHead, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX);
    return hcHead;
}

UInt32 Test_Circular_Dec(UInt32 hcHead, UInt32 HASH_CHAIN_SLOT_SZ, UInt32 HASH_CHAIN_MAX) {
    hcHead = CIRC_DEC_HEAD(hcHead, HASH_CHAIN_SLOT_SZ, HASH_CHAIN_MAX);
    return hcHead;
}

UInt32 Test_Hc_GetMatchesSpec(size_t lenLimit, UInt32 hcHead, UInt32 hv, UInt32 pos,
    const Byte* cur, CLzRef* son, size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 cutValue, UInt32* d, unsigned maxLen, int blockSz) {
    if (blockSz == 8) {
        UInt32* du = AOCL_Hc_GetMatchesSpec_8(lenLimit, hcHead, hv, pos, cur, son,
            _cyclicBufferPos, _cyclicBufferSize, cutValue, d, maxLen);
        return du - d;
    }
    else {
        UInt32* du = AOCL_Hc_GetMatchesSpec_16(lenLimit, hcHead, hv, pos, cur, son,
            _cyclicBufferPos, _cyclicBufferSize, cutValue, d, maxLen);
        return du - d;
    }
}
#endif
#endif
