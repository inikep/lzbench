/* LzmaEnc.c -- LZMA Encoder
2022-07-15: Igor Pavlov : Public domain */

/**
* Modifications Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
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

#include "Precomp.h"

#include <string.h>

/* #define SHOW_STAT */
/* #define SHOW_STAT2 */

#if defined(SHOW_STAT) || defined(SHOW_STAT2)
#include <stdio.h>
#endif

#include "utils/utils.h"
#include "utils/dispatcher.h"
#include "algos/common/aoclAlgoLog.h"
#include "aocl_lzma_fmv_utils.h"
#include "aocl_lzma_dispatch_variants.h"
/* Shared FMV selection helper and variant entry layouts are centralized in
 * aocl_lzma_fmv_utils.h and aocl_lzma_dispatch_variants.h. */

#include "LzmaEnc.h"

#include "utils/utils.h"

#include "LzFind.h"
#define _7ZIP_ST
#ifndef _7ZIP_ST
#include "LzFindMt.h"
#endif

#include <limits.h>
/* the following LzmaEnc_* declarations is internal LZMA interface for LZMA2 encoder */

#ifdef AOCL_ENABLE_THREADS
#include "threads/threads.h"
#endif

SRes LzmaEnc_PrepareForLzma2(CLzmaEncHandle pp, ISeqInStream *inStream, UInt32 keepWindowSize,
    ISzAllocPtr alloc, ISzAllocPtr allocBig);
SRes LzmaEnc_MemPrepare(CLzmaEncHandle pp, const Byte *src, SizeT srcLen,
    UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig);
SRes LzmaEnc_CodeOneMemBlock(CLzmaEncHandle pp, BoolInt reInit,
    Byte *dest, size_t *destLen, UInt32 desiredPackSize, UInt32 *unpackSize);
const Byte *LzmaEnc_GetCurBuf(CLzmaEncHandle pp);
void LzmaEnc_Finish(CLzmaEncHandle pp);
void LzmaEnc_SaveState(CLzmaEncHandle pp);
void LzmaEnc_RestoreState(CLzmaEncHandle pp);

#ifdef SHOW_STAT
static unsigned g_STAT_OFFSET = 0;
#endif

// #define kLzmaMaxHistorySize ((UInt32)3 << 29) //1.5 GB
// #define kLzmaMaxHistorySize ((UInt32)7 << 29)
/* for good normalization speed we still reserve 256 MB before 4 GB range */
#define kLzmaMaxHistorySize ((UInt32)15 << 28)

#define kNumTopBits 24
#define kTopValue ((UInt32)1 << kNumTopBits)

#define kNumBitModelTotalBits 11
#define kBitModelTotal (1 << kNumBitModelTotalBits)
#define kNumMoveBits 5
#define kProbInitValue (kBitModelTotal >> 1)

#define kNumMoveReducingBits 4
#define kNumBitPriceShiftBits 4
// #define kBitPrice (1 << kNumBitPriceShiftBits)

#define REP_LEN_COUNT 64

#ifdef AOCL_LZMA_OPT
/* Dynamic dispatcher setup function for native APIs.
 * All native APIs that call aocl optimized functions within their call stack,
 * must call AOCL_SETUP_NATIVE() at the start of the function. This sets up 
 * appropriate code paths to take based on user defined environment variables,
 * as well as cpu instruction set supported by the runtime machine. */
static void aocl_setup_native(void);
#define AOCL_SETUP_NATIVE() aocl_setup_native()
#else
#define AOCL_SETUP_NATIVE()
#endif

static int setup_ok_lzma_encode = 0; // flag to indicate status of dynamic dispatcher setup
#if !defined(AOCL_ENABLE_THREADS) || defined(AOCL_USE_TBB)
static atomic_flag setup_lzmaenc = ATOMIC_FLAG_INIT;
#endif

//Forward declarations to allow default pointer initializations
// Function pointers for optimization overloads
void (*LzmaEncProps_Normalize_fp)(CLzmaEncProps* p) = LzmaEncProps_Normalize;

void LzmaEncProps_Init(CLzmaEncProps *p)
{
  if(p==NULL){
    LOG_UNFORMATTED(ERR, logCtx, "Invalid CLzmaEncProps");
    return;
  }
  LOG_UNFORMATTED(TRACE, logCtx, "Enter");
  p->level = 5;
  p->dictSize = p->mc = 0;
  p->reduceSize = (UInt64)(Int64)-1;
  p->lc = p->lp = p->pb = p->algo = p->fb = p->btMode = p->numHashBytes = p->numThreads = -1;
  p->writeEndMark = 0;
  p->affinity = 0;
#ifdef AOCL_LZMA_OPT
  p->srcLen = 0;
  p->cacheEfficientStrategy = -1;
#endif
  LOG_UNFORMATTED(TRACE, logCtx, "Exit");
}

// Default settings as per LZMA SDK 22.01
void LzmaEncProps_Normalize(CLzmaEncProps *p)
{
  if(p == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid CLzmaEncProps");
    return;
  }

  int level = p->level;
  if (level < 0) level = 5;
  p->level = level;
  
  if (p->dictSize == 0)
    p->dictSize =
      ( level <= 3 ? ((UInt32)1 << (level * 2 + 16)) :
      ( level <= 6 ? ((UInt32)1 << (level + 19)) :
      ( level <= 7 ? ((UInt32)1 << 25) : ((UInt32)1 << 26)
      )));
  if (p->dictSize > p->reduceSize)
  {
    UInt32 v = (UInt32)p->reduceSize;
    const UInt32 kReduceMin = ((UInt32)1 << 12);
    if (v < kReduceMin)
      v = kReduceMin;
    if (p->dictSize > v)
      p->dictSize = v;
  }

  if (p->lc < 0) p->lc = 3;
  if (p->lp < 0) p->lp = 0;
  if (p->pb < 0) p->pb = 2;

  if (p->algo < 0) p->algo = (level < 5 ? 0 : 1);
  if (p->fb < 0) p->fb = (level < 7 ? 32 : 64);
  if (p->btMode < 0) p->btMode = (p->algo == 0 ? 0 : 1);
  if (p->numHashBytes < 0) p->numHashBytes = (p->btMode ? 4 : 5);
  if (p->mc == 0) p->mc = (16 + ((unsigned)p->fb >> 1)) >> (p->btMode ? 0 : 1);
  
  if (p->numThreads < 0)
    p->numThreads =
      #ifndef _7ZIP_ST
      ((p->btMode && p->algo) ? 2 : 1);
      #else
      1;
      #endif
}

#ifdef AOCL_LZMA_OPT
/*
* @brief: Set parameters based on user inputs and level
* 
* Changes wrt LzmaEncProps_Normalize:
* + Dictionary sizes to suit cache efficient hash chains for lower levels
*/
void AOCL_LzmaEncProps_Normalize(CLzmaEncProps* p)
{
  if(p == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid CLzmaEncProps");
    return;
  }

  int level = p->level;
  if (level < 0) level = 5;
  p->level = level;

  if (p->lc < 0) p->lc = 3;
  if (p->lp < 0) p->lp = 0;
  if (p->pb < 0) p->pb = 2;

  if (p->algo < 0) p->algo = (level < 5 ? 0 : 1);
  if (p->fb < 0) p->fb = (level < 7 ? 32 : 64);
  if (p->btMode < 0) p->btMode = (p->algo == 0 ? 0 : 1);
  if (p->numHashBytes < 0) p->numHashBytes = (p->btMode ? 4 : 5);
  if (p->mc == 0) p->mc = (16 + ((unsigned)p->fb >> 1)) >> (p->btMode ? 0 : 1);

  if (!p->btMode) { // cache efficient implementation is available only for hash chain mode
      if (p->cacheEfficientStrategy < 0) { // if not explicitly set by user
          if (p->srcLen > 0) {
              if (p->srcLen < MAX_SIZE_FOR_CE_HC_OFF) { // file size too small to benefit from cehc
                  p->cacheEfficientStrategy = 0;
              }
              else if (p->srcLen >= MAX_SIZE_FOR_CE_HC_OFF && p->srcLen < MIN_SIZE_FOR_CE_HC_ON) { // for mid size files, set based on numHashBytes
                  if (p->numHashBytes < 5) {
                      // Lower numHashBytes result in more collisions. This produces longer hash chains.
                      // Cehc benefits are primarily seen when more time is spent searching through the
                      // dictionary, which happens when chains are longer.
                      p->cacheEfficientStrategy = 1;
                  }
                  else {
                      p->cacheEfficientStrategy = 0;
                  }
              }
              else { // p->srcLen >= MIN_SIZE_FOR_CE_HC_ON
                  p->cacheEfficientStrategy = 1; // file lize large enought to benefit from cehc
              }
          }
          else {
              p->cacheEfficientStrategy = 0;
          }
      }
  }
  else {
      p->cacheEfficientStrategy = 0;
  }

  if (p->cacheEfficientStrategy) { // use cache efficient settings
      /* Using larger dictionaries to compensate for compression drop
       * due to unused slots in cache efficient dictionary blocks */
      if (p->dictSize == 0) 
          p->dictSize =
          ( level <= 2 ? ((UInt32)1 << (level + 19)) :
          ( level <= 4 ? ((UInt32)1 << (level + 20)) :
          ( level <= 6 ? ((UInt32)1 << (level + 19)) :
          ( level <= 7 ? ((UInt32)1 << 25) : ((UInt32)1 << 26)
          ))));
  }
  else { // use reference settings
      if (p->dictSize == 0)
          p->dictSize =
              ( level <= 3 ? ((UInt32)1 << (level * 2 + 16)) :
              ( level <= 6 ? ((UInt32)1 << (level + 19)) :
              ( level <= 7 ? ((UInt32)1 << 25) : ((UInt32)1 << 26)
              )));
  }

  if (p->dictSize > p->reduceSize)
  {
    UInt32 v = (UInt32)p->reduceSize;
    const UInt32 kReduceMin = ((UInt32)1 << 12);
    if (v < kReduceMin)
      v = kReduceMin;
    if (p->dictSize > v)
      p->dictSize = v;
  }

  if (p->cacheEfficientStrategy) {
      /* For hash chain algo, cache efficient hash chains with direct mapping
      * from hash to blocks are used. Size of the hash table is derived from dictSize.
      * Due to certain assumptions on the hashes, a minimum hash table size of 
      * 'kHashGuarentee' is required. Hence, we need to impose a minimum dictSize  
      * to ensure derived hash table sizes are valid. */
      if (level < HASH_CHAIN_16_LEVEL) {
          if ((p->dictSize / HASH_CHAIN_SLOT_SZ_8) < kHashGuarentee)
              p->dictSize = kHashGuarentee * HASH_CHAIN_SLOT_SZ_8;

      }
      else {
          if ((p->dictSize / HASH_CHAIN_SLOT_SZ_16) < kHashGuarentee)
              p->dictSize = kHashGuarentee * HASH_CHAIN_SLOT_SZ_16;
      }
  }

  LOG_FORMATTED(DEBUG, logCtx, "Parameters set : dictSize = %u, lc = %d, lp = %d, pb = %d,"
  " algo = %d, fb = %d, btMode = %d, numHashBytes = %d, mc = %u, cacheEfficientStrategy = %d",
      p->dictSize, p->lc, p->lp, p->pb, p->algo, p->fb, p->btMode, p->numHashBytes,
      p->mc, p->cacheEfficientStrategy);

  if (p->numThreads < 0)
    p->numThreads =
      #ifndef _7ZIP_ST
      ((p->btMode && p->algo) ? 2 : 1);
      #else
      1;
      #endif
}
#endif

UInt32 LzmaEncProps_GetDictSize(const CLzmaEncProps *props2)
{
  if(props2 == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid CLzmaEncProps");
    return 0;
  }

  CLzmaEncProps props = *props2;
#ifdef AOCL_LZMA_OPT
  LzmaEncProps_Normalize_fp(&props);
#else
  LzmaEncProps_Normalize(&props);
#endif
  return props.dictSize;
}


/*
x86/x64:

BSR:
  IF (SRC == 0) ZF = 1, DEST is undefined;
                  AMD : DEST is unchanged;
  IF (SRC != 0) ZF = 0; DEST is index of top non-zero bit
  BSR is slow in some processors

LZCNT:
  IF (SRC  == 0) CF = 1, DEST is size_in_bits_of_register(src) (32 or 64)
  IF (SRC  != 0) CF = 0, DEST = num_lead_zero_bits
  IF (DEST == 0) ZF = 1;

LZCNT works only in new processors starting from Haswell.
if LZCNT is not supported by processor, then it's executed as BSR.
LZCNT can be faster than BSR, if supported.
*/

// #define LZMA_LOG_BSR

#if defined(MY_CPU_ARM_OR_ARM64) /* || defined(MY_CPU_X86_OR_AMD64) */

  #if (defined(__clang__) && (__clang_major__ >= 6)) \
      || (defined(__GNUC__) && (__GNUC__ >= 6))
      #define LZMA_LOG_BSR
  #elif defined(_MSC_VER) && (_MSC_VER >= 1300)
    // #if defined(MY_CPU_ARM_OR_ARM64)
      #define LZMA_LOG_BSR
    // #endif
  #endif
#endif

// #include <intrin.h>

#ifdef LZMA_LOG_BSR

#if defined(__clang__) \
    || defined(__GNUC__)

/*
  C code:                  : (30 - __builtin_clz(x))
    gcc9/gcc10 for x64 /x86  : 30 - (bsr(x) xor 31)
    clang10 for x64          : 31 + (bsr(x) xor -32)
*/

  #define MY_clz(x)  ((unsigned)__builtin_clz(x))
  // __lzcnt32
  // __builtin_ia32_lzcnt_u32

#else  // #if defined(_MSC_VER)

  #ifdef MY_CPU_ARM_OR_ARM64

    #define MY_clz  _CountLeadingZeros

  #else // if defined(MY_CPU_X86_OR_AMD64)

    // #define MY_clz  __lzcnt  // we can use lzcnt (unsupported by old CPU)
    // _BitScanReverse code is not optimal for some MSVC compilers
    #define BSR2_RET(pos, res) { unsigned long zz; _BitScanReverse(&zz, (pos)); zz--; \
      res = (zz + zz) + (pos >> zz); }

  #endif // MY_CPU_X86_OR_AMD64

#endif // _MSC_VER


#ifndef BSR2_RET

    #define BSR2_RET(pos, res) { unsigned zz = 30 - MY_clz(pos); \
      res = (zz + zz) + (pos >> zz); }

#endif


unsigned GetPosSlot1(UInt32 pos);
unsigned GetPosSlot1(UInt32 pos)
{
  unsigned res;
  BSR2_RET(pos, res);
  return res;
}
#define GetPosSlot2(pos, res) { BSR2_RET(pos, res); }
#define GetPosSlot(pos, res) { if (pos < 2) res = pos; else BSR2_RET(pos, res); }


#else // ! LZMA_LOG_BSR

#define kNumLogBits (11 + sizeof(size_t) / 8 * 3)

#define kDicLogSizeMaxCompress ((kNumLogBits - 1) * 2 + 7)

static void LzmaEnc_FastPosInit(Byte *g_FastPos)
{
  unsigned slot;
  g_FastPos[0] = 0;
  g_FastPos[1] = 1;
  g_FastPos += 2;
  
  for (slot = 2; slot < kNumLogBits * 2; slot++)
  {
    size_t k = ((size_t)1 << ((slot >> 1) - 1));
    size_t j;
    for (j = 0; j < k; j++)
      g_FastPos[j] = (Byte)slot;
    g_FastPos += k;
  }
}

/* we can use ((limit - pos) >> 31) only if (pos < ((UInt32)1 << 31)) */
/*
#define BSR2_RET(pos, res) { unsigned zz = 6 + ((kNumLogBits - 1) & \
  (0 - (((((UInt32)1 << (kNumLogBits + 6)) - 1) - pos) >> 31))); \
  res = p->g_FastPos[pos >> zz] + (zz * 2); }
*/

/*
#define BSR2_RET(pos, res) { unsigned zz = 6 + ((kNumLogBits - 1) & \
  (0 - (((((UInt32)1 << (kNumLogBits)) - 1) - (pos >> 6)) >> 31))); \
  res = p->g_FastPos[pos >> zz] + (zz * 2); }
*/

#define BSR2_RET(pos, res) { unsigned zz = (pos < (1 << (kNumLogBits + 6))) ? 6 : 6 + kNumLogBits - 1; \
  res = p->g_FastPos[pos >> zz] + (zz * 2); }

/*
#define BSR2_RET(pos, res) { res = (pos < (1 << (kNumLogBits + 6))) ? \
  p->g_FastPos[pos >> 6] + 12 : \
  p->g_FastPos[pos >> (6 + kNumLogBits - 1)] + (6 + (kNumLogBits - 1)) * 2; }
*/

#define GetPosSlot1(pos) p->g_FastPos[pos]
#define GetPosSlot2(pos, res) { BSR2_RET(pos, res); }
#define GetPosSlot(pos, res) { if (pos < kNumFullDistances) res = p->g_FastPos[pos & (kNumFullDistances - 1)]; else BSR2_RET(pos, res); }

#endif // LZMA_LOG_BSR


#define LZMA_NUM_REPS 4

typedef UInt16 CState;
typedef UInt16 CExtra;

/* Cost and state of system if a particular choice is made for
* encoding this packet. It is used to store the most optimal
* choice (lowest price) for a given len */
typedef struct
{
  UInt32 price; // cost of making this choice
  CState state; // state of system. One of kNumStates.
  CExtra extra;
      // 0   : normal
      // 1   : LIT : MATCH
      // > 1 : MATCH (extra-1) : LIT : REP0 (len)
  UInt32 len; // length of match
  UInt32 dist; // distance at which match is found
  UInt32 reps[LZMA_NUM_REPS]; // rep distances until this point
} COptimal;


// 18.06
#define kNumOpts (1 << 11)
#define kPackReserve (kNumOpts * 8)
// #define kNumOpts (1 << 12)
// #define kPackReserve (1 + kNumOpts * 2)

#define kNumLenToPosStates 4
#define kNumPosSlotBits 6
// #define kDicLogSizeMin 0
#define kDicLogSizeMax 32
#define kDistTableSizeMax (kDicLogSizeMax * 2)

#define kNumAlignBits 4
#define kAlignTableSize (1 << kNumAlignBits)
#define kAlignMask (kAlignTableSize - 1)

#define kStartPosModelIndex 4
#define kEndPosModelIndex 14
#define kNumFullDistances (1 << (kEndPosModelIndex >> 1))

typedef
#ifdef _LZMA_PROB32
  UInt32
#else
  UInt16
#endif
  CLzmaProb;

#define LZMA_PB_MAX 4
#define LZMA_LC_MAX 8
#define LZMA_LP_MAX 4

#define LZMA_NUM_PB_STATES_MAX (1 << LZMA_PB_MAX)

#define kLenNumLowBits 3
#define kLenNumLowSymbols (1 << kLenNumLowBits)
#define kLenNumHighBits 8
#define kLenNumHighSymbols (1 << kLenNumHighBits)
#define kLenNumSymbolsTotal (kLenNumLowSymbols * 2 + kLenNumHighSymbols)

#define LZMA_MATCH_LEN_MIN 2 // min match len 2
#define LZMA_MATCH_LEN_MAX (LZMA_MATCH_LEN_MIN + kLenNumSymbolsTotal - 1) // max match len 273

#define kNumStates 12


typedef struct
{
  CLzmaProb low[LZMA_NUM_PB_STATES_MAX << (kLenNumLowBits + 1)];
  CLzmaProb high[kLenNumHighSymbols];
} CLenEnc;


typedef struct
{
  unsigned tableSize;
  UInt32 prices[LZMA_NUM_PB_STATES_MAX][kLenNumSymbolsTotal];
  // UInt32 prices1[LZMA_NUM_PB_STATES_MAX][kLenNumLowSymbols * 2];
  // UInt32 prices2[kLenNumSymbolsTotal];
} CLenPriceEnc;

#define GET_PRICE_LEN(p, posState, len) \
    ((p)->prices[posState][(size_t)(len) - LZMA_MATCH_LEN_MIN])

/*
#define GET_PRICE_LEN(p, posState, len) \
    ((p)->prices2[(size_t)(len) - 2] + ((p)->prices1[posState][((len) - 2) & (kLenNumLowSymbols * 2 - 1)] & (((len) - 2 - kLenNumLowSymbols * 2) >> 9)))
*/

// Range encoder struct
typedef struct
{
  UInt32 range; // dynamically scaled integer value that provides the bound for current range
  unsigned cache;
  UInt64 low; // encoded value that exists within range. Compressed bytes are extracted from this.
  UInt64 cacheSize;
  Byte *buf; // buffer to hold generated compressed bytes
  Byte *bufLim; // byte limit upon which buf is flushed to file 
  Byte *bufBase;
  ISeqOutStream *outStream;
  UInt64 processed;
  SRes res;
  #ifdef AOCL_ENABLE_THREADS
  BoolInt flushData;
  #endif
} CRangeEnc;


typedef struct
{
  CLzmaProb *litProbs;

  unsigned state;
  UInt32 reps[LZMA_NUM_REPS];

  CLzmaProb posAlignEncoder[1 << kNumAlignBits];
  CLzmaProb isRep[kNumStates];
  CLzmaProb isRepG0[kNumStates];
  CLzmaProb isRepG1[kNumStates];
  CLzmaProb isRepG2[kNumStates];
  CLzmaProb isMatch[kNumStates][LZMA_NUM_PB_STATES_MAX];
  CLzmaProb isRep0Long[kNumStates][LZMA_NUM_PB_STATES_MAX];

  CLzmaProb posSlotEncoder[kNumLenToPosStates][1 << kNumPosSlotBits];
  CLzmaProb posEncoders[kNumFullDistances];
  
  CLenEnc lenProbs;
  CLenEnc repLenProbs;

} CSaveState;


typedef UInt32 CProbPrice;


typedef struct
{
  void *matchFinderObj;
  IMatchFinder2 matchFinder; // set to hashchain or bintree based dictionary module used to find matches

  unsigned optCur;
  unsigned optEnd;

  unsigned longestMatchLen;
  unsigned numPairs;
  UInt32 numAvail;

  unsigned state;
  unsigned numFastBytes; // limit on len upon which we exit optimal parsing
  unsigned additionalOffset; // offset to keep count of bytes that have already been processed via ReadMatchDistances()
  UInt32 reps[LZMA_NUM_REPS];
  unsigned lpMask, pbMask;
  CLzmaProb *litProbs; // probability context models for literal encoding
  CRangeEnc rc;

  UInt32 backRes; // final rep id or distance for current match

  unsigned lc, lp, pb;
  unsigned lclp;

  BoolInt fastMode;
  BoolInt writeEndMark;
  BoolInt finished;
  BoolInt multiThread;
  BoolInt needInit;
  // BoolInt _maxMode;

  UInt64 nowPos64;
  
  unsigned matchPriceCount;
  // unsigned alignPriceCount;
  int repLenEncCounter;

  unsigned distTableSize;

  UInt32 dictSize;
  SRes result;

  #ifndef _7ZIP_ST
  BoolInt mtMode;
  // begin of CMatchFinderMt is used in LZ thread
  CMatchFinderMt matchFinderMt;
  // end of CMatchFinderMt is used in BT and HASH threads
  // #else
  // CMatchFinder matchFinderBase;
  #endif
  CMatchFinder matchFinderBase;

  
  // we suppose that we have 8-bytes alignment after CMatchFinder
 
  #ifndef _7ZIP_ST
  Byte pad[128];
  #endif
  
  // LZ thread
  CProbPrice ProbPrices[kBitModelTotal >> kNumMoveReducingBits]; // price for different probability values

  /* We want to find matches in dictionary for every item in input stream (after hashing)
  * For every such cycle, we need a data structure to hold matches found
  * Match is represented as a <len,dist> pair
  * For a given 'len', the best match is the one with the shortest 'dist'
  * Hence, it is sufficient to store only 1 entry per 'len' (one with shortest 'dist')
  * Maximum match len supported is LZMA_MATCH_LEN_MAX = 273 as per lzma standard
  * 'matches' size is allocated to LZMA_MATCH_LEN_MAX * 2 to hold <len,dist> pairs
  * for all allowed 'len'
  * matches[0]=len1, matches[1]=dist1, matches[2]=len2, matches[3]=dist2,... */
  // we want {len , dist} pairs to be 8-bytes aligned in matches array
  UInt32 matches[LZMA_MATCH_LEN_MAX * 2 + 2];

  // we want 8-bytes alignment here
  UInt32 alignPrices[kAlignTableSize]; // price for different align values used to encode distance
  UInt32 posSlotPrices[kNumLenToPosStates][kDistTableSizeMax]; // price for different slot values used to encode distance
  UInt32 distancesPrices[kNumLenToPosStates][kNumFullDistances];  // price for different direct bits used to encode distance

  /* Depending on type of packet and bit position within the packet, different
  * contexts are used by the range encoder.
  * Following probability contexts exist:
  * posAlignEncoder[16] : 4 align bits in long distances
  * isRep[12], isRepG0[12], isRepG1[12], isRepG2[12] : states = 12
  * isMatch[12][16], isRep0Long[12][16]. 4 bits from dictionary pos used as context (2^4).
  * posSlotEncoder[4][64] : 4 len values used in distance context * 2^6 slots 
  * posEncoders[128] : distance direct bits */
  CLzmaProb posAlignEncoder[1 << kNumAlignBits];
  CLzmaProb isRep[kNumStates];
  CLzmaProb isRepG0[kNumStates];
  CLzmaProb isRepG1[kNumStates];
  CLzmaProb isRepG2[kNumStates];
  CLzmaProb isMatch[kNumStates][LZMA_NUM_PB_STATES_MAX];
  CLzmaProb isRep0Long[kNumStates][LZMA_NUM_PB_STATES_MAX];
  CLzmaProb posSlotEncoder[kNumLenToPosStates][1 << kNumPosSlotBits];
  CLzmaProb posEncoders[kNumFullDistances];
  
  CLenEnc lenProbs;
  CLenEnc repLenProbs;

  #ifndef LZMA_LOG_BSR
  Byte g_FastPos[1 << kNumLogBits];
  #endif

  CLenPriceEnc lenEnc;
  CLenPriceEnc repLenEnc;

  /* Record optimal prices for different len. In each GetOptimum() call,
  * the type of packet we decide to encode as well as len and distance
  * impacts the price. Also, multiple iterations of ReadMatchDistances() 
  * are made which results in more price values. The optimal price for
  * each len is maintained in opt[1 to 273] */
  COptimal opt[kNumOpts];

  CSaveState saveState;

  // BoolInt mf_Failure;
  #ifndef _7ZIP_ST
  Byte pad2[128];
  #endif
} CLzmaEnc;

//Forward declarations to allow default pointer initializations
static unsigned GetOptimum(CLzmaEnc* p, UInt32 position);

// Function pointers for optimization overloads
 void (*MatchFinder_CreateVTable_fp)(CMatchFinder* p, IMatchFinder2* vTable) = MatchFinder_CreateVTable;
 int (*MatchFinder_Create_fp)(CMatchFinder* p, UInt32 historySize,
  UInt32 keepAddBufferBefore, UInt32 matchMaxLen, UInt32 keepAddBufferAfter,
  ISzAllocPtr alloc) = MatchFinder_Create;
 void (*MatchFinder_Free_fp)(CMatchFinder* p, ISzAllocPtr alloc) = MatchFinder_Free;
 unsigned (*GetOptimum_fp)(CLzmaEnc* p, UInt32 position) = GetOptimum;
 SRes(*LzmaEnc_SetProps_fp)(CLzmaEncHandle pp, const CLzmaEncProps* props2) = LzmaEnc_SetProps;

#define MFB (p->matchFinderBase)
/*
#ifndef _7ZIP_ST
#define MFB (p->matchFinderMt.MatchFinder)
#endif
*/

#define COPY_ARR(dest, src, arr) memcpy(dest->arr, src->arr, sizeof(src->arr));

void LzmaEnc_SaveState(CLzmaEncHandle pp)
{
  if (pp == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "LzmaEnc_SaveState: Invalid CLzmaEncHandle");
    return;
  }
  CLzmaEnc *p = (CLzmaEnc *)pp;
  CSaveState *dest = &p->saveState;
  
  dest->state = p->state;
  
  dest->lenProbs = p->lenProbs;
  dest->repLenProbs = p->repLenProbs;

  COPY_ARR(dest, p, reps);

  COPY_ARR(dest, p, posAlignEncoder);
  COPY_ARR(dest, p, isRep);
  COPY_ARR(dest, p, isRepG0);
  COPY_ARR(dest, p, isRepG1);
  COPY_ARR(dest, p, isRepG2);
  COPY_ARR(dest, p, isMatch);
  COPY_ARR(dest, p, isRep0Long);
  COPY_ARR(dest, p, posSlotEncoder);
  COPY_ARR(dest, p, posEncoders);

  memcpy(dest->litProbs, p->litProbs, ((UInt32)0x300 << p->lclp) * sizeof(CLzmaProb));
}


void LzmaEnc_RestoreState(CLzmaEncHandle pp)
{
  if (pp == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "LzmaEnc_RestoreState: Invalid CLzmaEncHandle");
    return;
  }
  CLzmaEnc *dest = (CLzmaEnc *)pp;
  const CSaveState *p = &dest->saveState;

  dest->state = p->state;

  dest->lenProbs = p->lenProbs;
  dest->repLenProbs = p->repLenProbs;
  
  COPY_ARR(dest, p, reps);
  
  COPY_ARR(dest, p, posAlignEncoder);
  COPY_ARR(dest, p, isRep);
  COPY_ARR(dest, p, isRepG0);
  COPY_ARR(dest, p, isRepG1);
  COPY_ARR(dest, p, isRepG2);
  COPY_ARR(dest, p, isMatch);
  COPY_ARR(dest, p, isRep0Long);
  COPY_ARR(dest, p, posSlotEncoder);
  COPY_ARR(dest, p, posEncoders);

  memcpy(dest->litProbs, p->litProbs, ((UInt32)0x300 << dest->lclp) * sizeof(CLzmaProb));
}



SRes LzmaEnc_SetProps(CLzmaEncHandle pp, const CLzmaEncProps *props2)
{
  if(props2 == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid props2");
    return SZ_ERROR_PARAM;
  }

  CLzmaEnc *p = (CLzmaEnc *)pp;
  CLzmaEncProps props = *props2;
  LzmaEncProps_Normalize(&props);

  if (props.lc > LZMA_LC_MAX
      || props.lp > LZMA_LP_MAX
      || props.pb > LZMA_PB_MAX)
    return SZ_ERROR_PARAM;


  if (props.dictSize > kLzmaMaxHistorySize)
    props.dictSize = kLzmaMaxHistorySize;

  #ifndef LZMA_LOG_BSR
  {
    const UInt64 dict64 = props.dictSize;
    if (dict64 > ((UInt64)1 << kDicLogSizeMaxCompress))
      return SZ_ERROR_PARAM;
  }
  #endif

  p->dictSize = props.dictSize;
  {
    unsigned fb = (unsigned)props.fb;
    if (fb < 5)
      fb = 5;
    if (fb > LZMA_MATCH_LEN_MAX)
      fb = LZMA_MATCH_LEN_MAX;
    p->numFastBytes = fb;
  }
  p->lc = (unsigned)props.lc;
  p->lp = (unsigned)props.lp;
  p->pb = (unsigned)props.pb;
  p->fastMode = (props.algo == 0);

  #ifdef AOCL_ENABLE_THREADS
  p->rc.flushData = True;
  #endif
  // p->_maxMode = True;
  MFB.btMode = (Byte)(props.btMode ? 1 : 0);
  {
    unsigned numHashBytes = 4;
    if (props.btMode)
    {
           if (props.numHashBytes <  2) numHashBytes = 2;
      else if (props.numHashBytes <  4) numHashBytes = (unsigned)props.numHashBytes;
    }
    if (props.numHashBytes >= 5) numHashBytes = 5;

    MFB.numHashBytes = numHashBytes;
  }

  MFB.cutValue = props.mc;

  p->writeEndMark = (BoolInt)props.writeEndMark;

  #ifndef _7ZIP_ST
  /*
  if (newMultiThread != _multiThread)
  {
    ReleaseMatchFinder();
    _multiThread = newMultiThread;
  }
  */
  p->multiThread = (props.numThreads > 1);
  p->matchFinderMt.btSync.affinity =
  p->matchFinderMt.hashSync.affinity = props.affinity;
  #endif

  return SZ_OK;
}

#ifdef AOCL_LZMA_OPT
/*
* @brief: Update encode handle with user settings
* 
* Changes wrt LzmaEnc_SetProps:
* + pass level information to MFB 
* + call AOCL_LzmaEncProps_Normalize 
*/
SRes AOCL_LzmaEnc_SetProps(CLzmaEncHandle pp, const CLzmaEncProps* props2)
{
    if(props2 == NULL)
    {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid props2");
    return SZ_ERROR_PARAM;
    }

    CLzmaEnc* p = (CLzmaEnc*)pp;
    CLzmaEncProps props = *props2;
    AOCL_LzmaEncProps_Normalize(&props);

    if (props.lc > LZMA_LC_MAX
        || props.lp > LZMA_LP_MAX
        || props.pb > LZMA_PB_MAX) {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid props2");
        return SZ_ERROR_PARAM;
    }


    if (props.dictSize > kLzmaMaxHistorySize)
        props.dictSize = kLzmaMaxHistorySize;

#ifndef LZMA_LOG_BSR
    {
        const UInt64 dict64 = props.dictSize;
        if (dict64 > ((UInt64)1 << kDicLogSizeMaxCompress)) {
            LOG_UNFORMATTED(ERR, logCtx, "Invalid dict size");
            return SZ_ERROR_PARAM;
        }
    }
#endif

    p->dictSize = props.dictSize;
    {
        unsigned fb = (unsigned)props.fb;
        if (fb < 5)
            fb = 5;
        if (fb > LZMA_MATCH_LEN_MAX)
            fb = LZMA_MATCH_LEN_MAX;
        p->numFastBytes = fb;
    }
    p->lc = (unsigned)props.lc;
    p->lp = (unsigned)props.lp;
    p->pb = (unsigned)props.pb;
    p->fastMode = (props.algo == 0);
    #ifdef AOCL_ENABLE_THREADS
    p->rc.flushData = True;
    #endif
    // p->_maxMode = True;
    MFB.btMode = (Byte)(props.btMode ? 1 : 0);
    {
        unsigned numHashBytes = 4;
        if (props.btMode)
        {
            if (props.numHashBytes < 2) numHashBytes = 2;
            else if (props.numHashBytes < 4) numHashBytes = (unsigned)props.numHashBytes;
        }
        if (props.numHashBytes >= 5) numHashBytes = 5;

        MFB.numHashBytes = numHashBytes;
    }

    MFB.cutValue = props.mc;
    MFB.level = props.level; // pass level to MFB
    if (!props.btMode && props.cacheEfficientStrategy) {
        MFB.cacheEfficientSearch = 1; // Search using cache efficient hash chains
    }
    else {
        MFB.cacheEfficientSearch = 0; // Cache efficient search disabled
    }

    p->writeEndMark = (BoolInt)props.writeEndMark;

#ifndef _7ZIP_ST
    /*
    if (newMultiThread != _multiThread)
    {
      ReleaseMatchFinder();
      _multiThread = newMultiThread;
    }
    */
    p->multiThread = (props.numThreads > 1);
    p->matchFinderMt.btSync.affinity =
        p->matchFinderMt.hashSync.affinity = props.affinity;
#endif

    return SZ_OK;
}
#endif


void LzmaEnc_SetDataSize(CLzmaEncHandle pp, UInt64 expectedDataSiize)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  MFB.expectedDataSize = expectedDataSiize;
}


#define kState_Start 0
#define kState_LitAfterMatch 4
#define kState_LitAfterRep   5
#define kState_MatchAfterLit 7
#define kState_RepAfterLit   8

static const Byte kLiteralNextStates[kNumStates] = {0, 0, 0, 0, 1, 2, 3, 4,  5,  6,   4, 5};
static const Byte kMatchNextStates[kNumStates]   = {7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10};
static const Byte kRepNextStates[kNumStates]     = {8, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11};
static const Byte kShortRepNextStates[kNumStates]= {9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11};

#define IsLitState(s) ((s) < 7)
#define GetLenToPosState2(len) (((len) < kNumLenToPosStates - 1) ? (len) : kNumLenToPosStates - 1)
#define GetLenToPosState(len) (((len) < kNumLenToPosStates + 1) ? (len) - 2 : kNumLenToPosStates - 1)

#define kInfinityPrice (1 << 30)

static void RangeEnc_Construct(CRangeEnc *p)
{
  p->outStream = NULL;
  p->bufBase = NULL;
}

#define RangeEnc_GetProcessed(p)       (        (p)->processed + (size_t)((p)->buf - (p)->bufBase) +         (p)->cacheSize)
#define RangeEnc_GetProcessed_sizet(p) ((size_t)(p)->processed + (size_t)((p)->buf - (p)->bufBase) + (size_t)(p)->cacheSize)

#define RC_BUF_SIZE (1 << 16)

static int RangeEnc_Alloc(CRangeEnc *p, ISzAllocPtr alloc)
{
  if (!p->bufBase)
  {
    p->bufBase = (Byte *)ISzAlloc_Alloc(alloc, RC_BUF_SIZE);
    if (!p->bufBase)
      return 0;
    p->bufLim = p->bufBase + RC_BUF_SIZE;
  }
  return 1;
}

static void RangeEnc_Free(CRangeEnc *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->bufBase);
  p->bufBase = NULL;
}

static void RangeEnc_Init(CRangeEnc *p)
{
  p->range = 0xFFFFFFFF;
  p->cache = 0;
  p->low = 0;
  p->cacheSize = 0;

  p->buf = p->bufBase;

  p->processed = 0;
  p->res = SZ_OK;
}

MY_NO_INLINE static void RangeEnc_FlushStream(CRangeEnc *p)
{
  const size_t num = (size_t)(p->buf - p->bufBase);
  if (p->res == SZ_OK
#ifdef AOCL_ENABLE_THREADS
    && p->flushData
#endif /* AOCL_ENABLE_THREADS */
  )
  {
    if (num != ISeqOutStream_Write(p->outStream, p->bufBase, num))
      p->res = SZ_ERROR_WRITE;
  }
  p->processed += num;
  p->buf = p->bufBase;
}

MY_NO_INLINE static void MY_FAST_CALL RangeEnc_ShiftLow(CRangeEnc *p)
{
#ifdef AOCL_ENABLE_THREADS
  if (p->flushData){
#endif
  UInt32 low = (UInt32)p->low;
  unsigned high = (unsigned)(p->low >> 32);
  p->low = (UInt32)(low << 8);
  if (low < (UInt32)0xFF000000 || high != 0)
  {
    {
      Byte *buf = p->buf;
      *buf++ = (Byte)(p->cache + high); // next compressed byte is available. Save it.
      p->cache = (unsigned)(low >> 24);
      p->buf = buf;
      if (buf == p->bufLim)
        RangeEnc_FlushStream(p); // flush to output file
      if (p->cacheSize == 0)
        return;
    }
    high += 0xFF;
    for (;;)
    {
      Byte *buf = p->buf;
      *buf++ = (Byte)(high); // next compressed byte is available. Save it.
      p->buf = buf;
      if (buf == p->bufLim)
        RangeEnc_FlushStream(p); // flush to output file
      if (--p->cacheSize == 0)
        return;
    }
  }
  p->cacheSize++;
#ifdef AOCL_ENABLE_THREADS
  }
#endif
}

static void RangeEnc_FlushData(CRangeEnc *p)
{
  int i;
  for (i = 0; i < 5; i++)
    RangeEnc_ShiftLow(p);
}

/* if range is small, i.e. < 2^24, normalize it. Also check if next 
* compressed byte is ready to be saved and add it to p->buf */
#define RC_NORM(p) if (range < kTopValue) { range <<= 8; RangeEnc_ShiftLow(p); }

// Compute bound within range based on prob
#define RC_BIT_PRE(p, prob) \
  ttt = *(prob); \
  newBound = (range >> kNumBitModelTotalBits) * ttt;

// #define _LZMA_ENC_USE_BRANCH

#ifdef _LZMA_ENC_USE_BRANCH

#define RC_BIT(p, prob, bit) { \
  RC_BIT_PRE(p, prob) \
  if (bit == 0) { range = newBound; ttt += (kBitModelTotal - ttt) >> kNumMoveBits; } \
  else { (p)->low += newBound; range -= newBound; ttt -= ttt >> kNumMoveBits; } \
  *(prob) = (CLzmaProb)ttt; \
  RC_NORM(p) \
  }

#else

#define RC_BIT(p, prob, bit) { \
  UInt32 mask; \
  RC_BIT_PRE(p, prob) \
  mask = 0 - (UInt32)bit; \
  range &= mask; \
  mask &= newBound; \
  range -= mask; \
  (p)->low += mask; \
  mask = (UInt32)bit - 1; \
  range += newBound & mask; \
  mask &= (kBitModelTotal - ((1 << kNumMoveBits) - 1)); \
  mask += ((1 << kNumMoveBits) - 1); \
  ttt += (UInt32)((Int32)(mask - ttt) >> kNumMoveBits); \
  *(prob) = (CLzmaProb)ttt; \
  RC_NORM(p) \
  }

#endif




#define RC_BIT_0_BASE(p, prob) \
  range = newBound; *(prob) = (CLzmaProb)(ttt + ((kBitModelTotal - ttt) >> kNumMoveBits));

#define RC_BIT_1_BASE(p, prob) \
  range -= newBound; (p)->low += newBound; *(prob) = (CLzmaProb)(ttt - (ttt >> kNumMoveBits)); \

/* Encode 0. range for next iteration becomes [0, bound). i.e. size = bound
* Update probability model: prob + delta [Prob of 0 increased] 
* RC_BIT_PRE must be called before this to set newBound */
#define RC_BIT_0(p, prob) \
  RC_BIT_0_BASE(p, prob) \
  RC_NORM(p)

/* Encode 1, range for next iteration becomes [bound, range). i.e. size = range-bound
* Update probability model: prob - delta [Prob of 1 increased]
* RC_BIT_PRE must be called before this to set newBound */
#define RC_BIT_1(p, prob) \
  RC_BIT_1_BASE(p, prob) \
  RC_NORM(p)

static void RangeEnc_EncodeBit_0(CRangeEnc *p, CLzmaProb *prob)
{
  UInt32 range, ttt, newBound;
  range = p->range;
  RC_BIT_PRE(p, prob)
  RC_BIT_0(p, prob)
  p->range = range;
}

static void LitEnc_Encode(CRangeEnc *p, CLzmaProb *probs, UInt32 sym)
{
  UInt32 range = p->range;
  sym |= 0x100;
  do
  {
    UInt32 ttt, newBound;
    // RangeEnc_EncodeBit(p, probs + (sym >> 8), (sym >> 7) & 1);
    CLzmaProb *prob = probs + (sym >> 8);
    UInt32 bit = (sym >> 7) & 1;
    sym <<= 1;
    RC_BIT(p, prob, bit);
  }
  while (sym < 0x10000);
  p->range = range;
}

static void LitEnc_EncodeMatched(CRangeEnc *p, CLzmaProb *probs, UInt32 sym, UInt32 matchByte)
{
  UInt32 range = p->range;
  UInt32 offs = 0x100;
  sym |= 0x100;
  do
  {
    UInt32 ttt, newBound;
    CLzmaProb *prob;
    UInt32 bit;
    matchByte <<= 1;
    // RangeEnc_EncodeBit(p, probs + (offs + (matchByte & offs) + (sym >> 8)), (sym >> 7) & 1);
    prob = probs + (offs + (matchByte & offs) + (sym >> 8));
    bit = (sym >> 7) & 1;
    sym <<= 1;
    offs &= ~(matchByte ^ sym);
    RC_BIT(p, prob, bit);
  }
  while (sym < 0x10000);
  p->range = range;
}



static void LzmaEnc_InitPriceTables(CProbPrice *ProbPrices)
{
  UInt32 i;
  for (i = 0; i < (kBitModelTotal >> kNumMoveReducingBits); i++)
  {
    const unsigned kCyclesBits = kNumBitPriceShiftBits;
    UInt32 w = (i << kNumMoveReducingBits) + (1 << (kNumMoveReducingBits - 1));
    unsigned bitCount = 0;
    unsigned j;
    for (j = 0; j < kCyclesBits; j++)
    {
      w = w * w;
      bitCount <<= 1;
      while (w >= ((UInt32)1 << 16))
      {
        w >>= 1;
        bitCount++;
      }
    }
    ProbPrices[i] = (CProbPrice)(((unsigned)kNumBitModelTotalBits << kCyclesBits) - 15 - bitCount);
    // printf("\n%3d: %5d", i, ProbPrices[i]);
  }
}


#define GET_PRICE(prob, bit) \
  p->ProbPrices[((prob) ^ (unsigned)(((-(int)(bit))) & (kBitModelTotal - 1))) >> kNumMoveReducingBits];

#define GET_PRICEa(prob, bit) \
     ProbPrices[((prob) ^ (unsigned)((-((int)(bit))) & (kBitModelTotal - 1))) >> kNumMoveReducingBits];

#define GET_PRICE_0(prob) p->ProbPrices[(prob) >> kNumMoveReducingBits]
#define GET_PRICE_1(prob) p->ProbPrices[((prob) ^ (kBitModelTotal - 1)) >> kNumMoveReducingBits]

#define GET_PRICEa_0(prob) ProbPrices[(prob) >> kNumMoveReducingBits]
#define GET_PRICEa_1(prob) ProbPrices[((prob) ^ (kBitModelTotal - 1)) >> kNumMoveReducingBits]


static UInt32 LitEnc_GetPrice(const CLzmaProb *probs, UInt32 sym, const CProbPrice *ProbPrices)
{
  UInt32 price = 0;
  sym |= 0x100;
  do
  {
    unsigned bit = sym & 1;
    sym >>= 1;
    price += GET_PRICEa(probs[sym], bit);
  }
  while (sym >= 2);
  return price;
}


static UInt32 LitEnc_Matched_GetPrice(const CLzmaProb *probs, UInt32 sym, UInt32 matchByte, const CProbPrice *ProbPrices)
{
  UInt32 price = 0;
  UInt32 offs = 0x100;
  sym |= 0x100;
  do
  {
    matchByte <<= 1;
    price += GET_PRICEa(probs[offs + (matchByte & offs) + (sym >> 8)], (sym >> 7) & 1);
    sym <<= 1;
    offs &= ~(matchByte ^ sym);
  }
  while (sym < 0x10000);
  return price;
}


static void RcTree_ReverseEncode(CRangeEnc *rc, CLzmaProb *probs, unsigned numBits, unsigned sym)
{
  UInt32 range = rc->range;
  unsigned m = 1;
  do
  {
    UInt32 ttt, newBound;
    unsigned bit = sym & 1;
    // RangeEnc_EncodeBit(rc, probs + m, bit);
    sym >>= 1;
    RC_BIT(rc, probs + m, bit);
    m = (m << 1) | bit;
  }
  while (--numBits);
  rc->range = range;
}



static void LenEnc_Init(CLenEnc *p)
{
  unsigned i;
  for (i = 0; i < (LZMA_NUM_PB_STATES_MAX << (kLenNumLowBits + 1)); i++)
    p->low[i] = kProbInitValue;
  for (i = 0; i < kLenNumHighSymbols; i++)
    p->high[i] = kProbInitValue;
}

static void LenEnc_Encode(CLenEnc *p, CRangeEnc *rc, unsigned sym, unsigned posState)
{
  UInt32 range, ttt, newBound;
  CLzmaProb *probs = p->low;
  range = rc->range;
  RC_BIT_PRE(rc, probs);
  if (sym >= kLenNumLowSymbols)
  {
    RC_BIT_1(rc, probs);
    probs += kLenNumLowSymbols;
    RC_BIT_PRE(rc, probs);
    if (sym >= kLenNumLowSymbols * 2)
    {
      RC_BIT_1(rc, probs);
      rc->range = range;
      // RcTree_Encode(rc, p->high, kLenNumHighBits, sym - kLenNumLowSymbols * 2);
      LitEnc_Encode(rc, p->high, sym - kLenNumLowSymbols * 2);
      return;
    }
    sym -= kLenNumLowSymbols;
  }

  // RcTree_Encode(rc, probs + (posState << kLenNumLowBits), kLenNumLowBits, sym);
  {
    unsigned m;
    unsigned bit;
    RC_BIT_0(rc, probs);
    probs += (posState << (1 + kLenNumLowBits));
    bit = (sym >> 2)    ; RC_BIT(rc, probs + 1, bit); m = (1 << 1) + bit;
    bit = (sym >> 1) & 1; RC_BIT(rc, probs + m, bit); m = (m << 1) + bit;
    bit =  sym       & 1; RC_BIT(rc, probs + m, bit);
    rc->range = range;
  }
}

static void SetPrices_3(const CLzmaProb *probs, UInt32 startPrice, UInt32 *prices, const CProbPrice *ProbPrices)
{
  unsigned i;
  for (i = 0; i < 8; i += 2)
  {
    UInt32 price = startPrice;
    UInt32 prob;
    price += GET_PRICEa(probs[1           ], (i >> 2));
    price += GET_PRICEa(probs[2 + (i >> 2)], (i >> 1) & 1);
    prob = probs[4 + (i >> 1)];
    prices[i    ] = price + GET_PRICEa_0(prob);
    prices[i + 1] = price + GET_PRICEa_1(prob);
  }
}


MY_NO_INLINE static void MY_FAST_CALL LenPriceEnc_UpdateTables(
    CLenPriceEnc *p,
    unsigned numPosStates,
    const CLenEnc *enc,
    const CProbPrice *ProbPrices)
{
  UInt32 b;
 
  {
    unsigned prob = enc->low[0];
    UInt32 a, c;
    unsigned posState;
    b = GET_PRICEa_1(prob);
    a = GET_PRICEa_0(prob);
    c = b + GET_PRICEa_0(enc->low[kLenNumLowSymbols]);
    for (posState = 0; posState < numPosStates; posState++)
    {
      UInt32 *prices = p->prices[posState];
      const CLzmaProb *probs = enc->low + (posState << (1 + kLenNumLowBits));
      SetPrices_3(probs, a, prices, ProbPrices);
      SetPrices_3(probs + kLenNumLowSymbols, c, prices + kLenNumLowSymbols, ProbPrices);
    }
  }

  /*
  {
    unsigned i;
    UInt32 b;
    a = GET_PRICEa_0(enc->low[0]);
    for (i = 0; i < kLenNumLowSymbols; i++)
      p->prices2[i] = a;
    a = GET_PRICEa_1(enc->low[0]);
    b = a + GET_PRICEa_0(enc->low[kLenNumLowSymbols]);
    for (i = kLenNumLowSymbols; i < kLenNumLowSymbols * 2; i++)
      p->prices2[i] = b;
    a += GET_PRICEa_1(enc->low[kLenNumLowSymbols]);
  }
  */
 
  // p->counter = numSymbols;
  // p->counter = 64;

  {
    unsigned i = p->tableSize;
    
    if (i > kLenNumLowSymbols * 2)
    {
      const CLzmaProb *probs = enc->high;
      UInt32 *prices = p->prices[0] + kLenNumLowSymbols * 2;
      i -= kLenNumLowSymbols * 2 - 1;
      i >>= 1;
      b += GET_PRICEa_1(enc->low[kLenNumLowSymbols]);
      do
      {
        /*
        p->prices2[i] = a +
        // RcTree_GetPrice(enc->high, kLenNumHighBits, i - kLenNumLowSymbols * 2, ProbPrices);
        LitEnc_GetPrice(probs, i - kLenNumLowSymbols * 2, ProbPrices);
        */
        // UInt32 price = a + RcTree_GetPrice(probs, kLenNumHighBits - 1, sym, ProbPrices);
        unsigned sym = --i + (1 << (kLenNumHighBits - 1));
        UInt32 price = b;
        do
        {
          unsigned bit = sym & 1;
          sym >>= 1;
          price += GET_PRICEa(probs[sym], bit);
        }
        while (sym >= 2);

        {
          unsigned prob = probs[(size_t)i + (1 << (kLenNumHighBits - 1))];
          prices[(size_t)i * 2    ] = price + GET_PRICEa_0(prob);
          prices[(size_t)i * 2 + 1] = price + GET_PRICEa_1(prob);
        }
      }
      while (i);

      {
        unsigned posState;
        size_t num = (p->tableSize - kLenNumLowSymbols * 2) * sizeof(p->prices[0][0]);
        for (posState = 1; posState < numPosStates; posState++)
          memcpy(p->prices[posState] + kLenNumLowSymbols * 2, p->prices[0] + kLenNumLowSymbols * 2, num);
      }
    }
  }
}

/*
  #ifdef SHOW_STAT
  g_STAT_OFFSET += num;
  printf("\n MovePos %u", num);
  #endif
*/
  
// Skip moves data and dictionary pointers by num
#define MOVE_POS(p, num) { \
    p->additionalOffset += (num); \
    p->matchFinder.Skip(p->matchFinderObj, (UInt32)(num)); }


static unsigned ReadMatchDistances(CLzmaEnc *p, unsigned *numPairsRes)
{
  unsigned numPairs;
  
  p->additionalOffset++;
  p->numAvail = p->matchFinder.GetNumAvailableBytes(p->matchFinderObj);
  {
    /* Find best matches in dictionary and record those <len,dist> pairs
  * Matches are stored in p->matches
  * numPairs is the total number of valid match 'len's found
  * Values in 'matches' after numPairs are ignored */
    const UInt32 *d = p->matchFinder.GetMatches(p->matchFinderObj, p->matches);
    // if (!d) { p->mf_Failure = True; *numPairsRes = 0;  return 0; }
    numPairs = (unsigned)(d - p->matches);
  }
  *numPairsRes = numPairs;
  
  #ifdef SHOW_STAT
  printf("\n i = %u numPairs = %u    ", g_STAT_OFFSET, numPairs / 2);
  g_STAT_OFFSET++;
  {
    unsigned i;
    for (i = 0; i < numPairs; i += 2)
      printf("%2u %6u   | ", p->matches[i], p->matches[i + 1]);
  }
  #endif
  
  if (numPairs == 0)
    return 0;
  {
    const unsigned len = p->matches[(size_t)numPairs - 2]; // longest matched length
    if (len != p->numFastBytes)
      return len;
    {
      UInt32 numAvail = p->numAvail;
      if (numAvail > LZMA_MATCH_LEN_MAX)
        numAvail = LZMA_MATCH_LEN_MAX; // max len limit allowed by lzma format
      {
        const Byte *p1 = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
        const Byte *p2 = p1 + len; // cur end
        const ptrdiff_t dif = (ptrdiff_t)-1 - (ptrdiff_t)p->matches[(size_t)numPairs - 1];
        /* GetMatches() has a limit on length of matches it looks for. This is < LZMA_MATCH_LEN_MAX,
        * E.g. 32 or 64. Longer lengths are not searched for, to reduce search time. Since the format allows
        * lengths upto LZMA_MATCH_LEN_MAX, for the longest matched pair we have found in p->matches,
        * we continue to check if further bytes also match (limited to total len LZMA_MATCH_LEN_MAX)
        * Return this longer len. */
        const Byte *lim = p1 + numAvail;
        for (; p2 != lim && *p2 == p2[dif]; p2++)
        {}
        return (unsigned)(p2 - p1);
      }
    }
  }
}

#define MARK_LIT ((UInt32)(Int32)-1)

#define MakeAs_Lit(p)       { (p)->dist = MARK_LIT; (p)->extra = 0; } // mark COptimal as literal
#define MakeAs_ShortRep(p)  { (p)->dist = 0; (p)->extra = 0; } // mark COptimal as ShortRep
#define IsShortRep(p)       ((p)->dist == 0)


#define GetPrice_ShortRep(p, state, posState) \
  ( GET_PRICE_0(p->isRepG0[state]) + GET_PRICE_0(p->isRep0Long[state][posState]))

#define GetPrice_Rep_0(p, state, posState) ( \
    GET_PRICE_1(p->isMatch[state][posState]) \
  + GET_PRICE_1(p->isRep0Long[state][posState])) \
  + GET_PRICE_1(p->isRep[state]) \
  + GET_PRICE_0(p->isRepG0[state])

MY_FORCE_INLINE
static UInt32 GetPrice_PureRep(const CLzmaEnc *p, unsigned repIndex, size_t state, size_t posState)
{
  UInt32 price;
  UInt32 prob = p->isRepG0[state];
  if (repIndex == 0)
  {
    price = GET_PRICE_0(prob);
    price += GET_PRICE_1(p->isRep0Long[state][posState]);
  }
  else
  {
    price = GET_PRICE_1(prob);
    prob = p->isRepG1[state];
    if (repIndex == 1)
      price += GET_PRICE_0(prob);
    else
    {
      price += GET_PRICE_1(prob);
      price += GET_PRICE(p->isRepG2[state], repIndex - 2);
    }
  }
  return price;
}


// Decide final <len,distance> from values in p->opt[i]
static unsigned Backward(CLzmaEnc *p, unsigned cur)
{
  unsigned wr = cur + 1;
  p->optEnd = wr;

  for (;;)
  {
    UInt32 dist = p->opt[cur].dist;
    unsigned len = (unsigned)p->opt[cur].len;
    unsigned extra = (unsigned)p->opt[cur].extra;
    cur -= len;

    if (extra)
    {
      wr--;
      p->opt[wr].len = (UInt32)len;
      cur -= extra;
      len = extra;
      if (extra == 1)
      {
        p->opt[wr].dist = dist;
        dist = MARK_LIT;
      }
      else
      {
        p->opt[wr].dist = 0;
        len--;
        wr--;
        p->opt[wr].dist = MARK_LIT;
        p->opt[wr].len = 1;
      }
    }

    if (cur == 0)
    {
      p->backRes = dist;
      p->optCur = wr;
      return len;
    }
    
    wr--;
    p->opt[wr].dist = dist;
    p->opt[wr].len = (UInt32)len;
  }
}



#define LIT_PROBS(pos, prevByte) \
  (p->litProbs + (UInt32)3 * (((((pos) << 8) + (prevByte)) & p->lpMask) << p->lc))

// Optimal parsing implementation
static unsigned GetOptimum(CLzmaEnc *p, UInt32 position)
{
  unsigned last, cur;
  UInt32 reps[LZMA_NUM_REPS];
  unsigned repLens[LZMA_NUM_REPS];
  UInt32 *matches;

  {
    UInt32 numAvail;
    unsigned numPairs, mainLen, repMaxIndex, i, posState;
    UInt32 matchPrice, repMatchPrice;
    const Byte *data;
    Byte curByte, matchByte;
    
    p->optCur = p->optEnd = 0;
    
    /* if additionalOffset is > 0, parsing for cur byte has already been completed
    * in one of the previous ReadMatchDistances() calls inside optimal parsing loop. 
    * Results are already available in p-> and can be used
    * without having to find matches again. */
    if (p->additionalOffset == 0)
      mainLen = ReadMatchDistances(p, &numPairs);
    else
    {
      mainLen = p->longestMatchLen;
      numPairs = p->numPairs;
    }
    
    numAvail = p->numAvail;
    if (numAvail < 2)
    {
      p->backRes = MARK_LIT;
      return 1;
    }
    if (numAvail > LZMA_MATCH_LEN_MAX)
      numAvail = LZMA_MATCH_LEN_MAX;
    
    data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
    repMaxIndex = 0; //rep with max len match
    
    // Get match lengths at distances in each of the reps for cur data
    for (i = 0; i < LZMA_NUM_REPS; i++)
    {
      unsigned len;
      const Byte *data2;
      reps[i] = p->reps[i];
      data2 = data - reps[i];
      if (data[0] != data2[0] || data[1] != data2[1])
      {
        repLens[i] = 0;
        continue;
      }
      for (len = 2; len < numAvail && data[len] == data2[len]; len++)
      {}
      repLens[i] = len;
      if (len > repLens[repMaxIndex])
        repMaxIndex = i;
      if (len == LZMA_MATCH_LEN_MAX) // 21.03 : optimization
        break;
    }
    
    /* If long enough match is found in a rep, it can be selected
    * instead of <mainLen, mainDist> as we save on encoding mainDist */
    if (repLens[repMaxIndex] >= p->numFastBytes)
    {
      unsigned len;
      p->backRes = (UInt32)repMaxIndex;
      len = repLens[repMaxIndex];
      MOVE_POS(p, len - 1)
      return len;
    }
    
    matches = p->matches;
    #define MATCHES  matches
    // #define MATCHES  p->matches
    
    if (mainLen >= p->numFastBytes)  // if len long enough, choose mainLen
    {
      p->backRes = MATCHES[(size_t)numPairs - 1] + LZMA_NUM_REPS;
      MOVE_POS(p, mainLen - 1)
      return mainLen;
    }
    
    curByte = *data;
    matchByte = *(data - reps[0]);

    last = repLens[repMaxIndex]; // last = longest matched length among reps and mainLen
    if (last <= mainLen)
      last = mainLen;
    
    if (last < 2 && curByte != matchByte)
    {
      p->backRes = MARK_LIT;
      return 1;
    }
    
    p->opt[0].state = (CState)p->state;
    
    posState = (position & p->pbMask);
    
    /* From this point onwards, cost for taking different decisions 'price'
    * is calculated. Every decision, choosing to encode as a literal,
    * srep, rep or match has a corresponding price. */
    {
      const CLzmaProb *probs = LIT_PROBS(position, *(data - 1));
      p->opt[1].price = GET_PRICE_0(p->isMatch[p->state][posState]) +
        (!IsLitState(p->state) ?
          LitEnc_Matched_GetPrice(probs, curByte, matchByte, p->ProbPrices) :
          LitEnc_GetPrice(probs, curByte, p->ProbPrices));
    }

    MakeAs_Lit(&p->opt[1]);
    
    matchPrice = GET_PRICE_1(p->isMatch[p->state][posState]);
    repMatchPrice = matchPrice + GET_PRICE_1(p->isRep[p->state]);
    
    // 18.06
    if (matchByte == curByte && repLens[0] == 0)
    {
      UInt32 shortRepPrice = repMatchPrice + GetPrice_ShortRep(p, p->state, posState);
      if (shortRepPrice < p->opt[1].price)
      {
        p->opt[1].price = shortRepPrice;
        MakeAs_ShortRep(&p->opt[1]);
      }
      if (last < 2)
      {
        p->backRes = p->opt[1].dist;
        return 1;
      }
    }
   
    p->opt[1].len = 1;
    
    p->opt[0].reps[0] = reps[0];
    p->opt[0].reps[1] = reps[1];
    p->opt[0].reps[2] = reps[2];
    p->opt[0].reps[3] = reps[3];
    
    // ---------- REP ----------
    
    for (i = 0; i < LZMA_NUM_REPS; i++)
    {
      unsigned repLen = repLens[i];
      UInt32 price;
      if (repLen < 2)
        continue;
      price = repMatchPrice + GetPrice_PureRep(p, i, p->state, posState);
      do
      {
        UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, repLen);
        COptimal *opt = &p->opt[repLen];
        if (price2 < opt->price)
        {
          opt->price = price2;
          opt->len = (UInt32)repLen;
          opt->dist = (UInt32)i;
          opt->extra = 0;
        }
      }
      while (--repLen >= 2);
    }
    
    
    // ---------- MATCH ----------
    {
      unsigned len = repLens[0] + 1;
      if (len <= mainLen)
      {
        unsigned offs = 0;
        UInt32 normalMatchPrice = matchPrice + GET_PRICE_0(p->isRep[p->state]);

        if (len < 2)
          len = 2;
        else
          while (len > MATCHES[offs])
            offs += 2;
    
        for (; ; len++)
        {
          COptimal *opt;
          UInt32 dist = MATCHES[(size_t)offs + 1];
          UInt32 price = normalMatchPrice + GET_PRICE_LEN(&p->lenEnc, posState, len);
          unsigned lenToPosState = GetLenToPosState(len);
       
          if (dist < kNumFullDistances)
            price += p->distancesPrices[lenToPosState][dist & (kNumFullDistances - 1)];
          else
          {
            unsigned slot;
            GetPosSlot2(dist, slot);
            price += p->alignPrices[dist & kAlignMask];
            price += p->posSlotPrices[lenToPosState][slot];
          }
          
          opt = &p->opt[len];
          
          if (price < opt->price)
          {
            opt->price = price;
            opt->len = (UInt32)len;
            opt->dist = dist + LZMA_NUM_REPS;
            opt->extra = 0;
          }
          
          if (len == MATCHES[offs])
          {
            offs += 2;
            if (offs == numPairs)
              break;
          }
        }
      }
    }
    

    cur = 0;

    #ifdef SHOW_STAT2
    /* if (position >= 0) */
    {
      unsigned i;
      printf("\n pos = %4X", position);
      for (i = cur; i <= last; i++)
      printf("\nprice[%4X] = %u", position - cur + i, p->opt[i].price);
    }
    #endif
  }


  
  // ---------- Optimal Parsing ----------
  /* last : longest matched length among reps and mainLen
  * p->opt[i] : base price and system states until now
  * cur : counter starting at 0
  * 
  * For cur in range [0, last) we check if skipping these many bytes (L) and
  * coding them using the strategy in p->opt[L] is worth it, if we can 
  * find a better matching sequence in the future that gives a better price overall.
  * Checking for last and beyond isn't necessary as we might as well encode the
  * current match using the best price strategy we have so far and leave the rest
  * of the sequence for the next GetOptimum() call. 
  * 
  * p->opt[L] also contains state and reps values if said strategy was used. This
  * will act as previous system state when evaluating price for byte sequence 
  * starting at offset L.
  * 
  * Run ReadMatchDistances() at offsets [0, last) from current position.
  * Update p->opt s (p->opt[i].price += newPrice, etc) for each p->opt[i]. */
  for (;;)
  {
    unsigned numAvail;
    UInt32 numAvailFull;
    unsigned newLen, numPairs, prev, state, posState, startLen;
    UInt32 litPrice, matchPrice, repMatchPrice;
    BoolInt nextIsLit;
    Byte curByte, matchByte;
    const Byte *data;
    COptimal *curOpt, *nextOpt;

    if (++cur == last)
      break; // no need to check beyond last. This can be handled in the next GetOptimum() call.
    
    // 18.06
    if (cur >= kNumOpts - 64)
    {
      unsigned j, best;
      UInt32 price = p->opt[cur].price;
      best = cur;
      for (j = cur + 1; j <= last; j++)
      {
        UInt32 price2 = p->opt[j].price;
        if (price >= price2)
        {
          price = price2;
          best = j;
        }
      }
      {
        unsigned delta = best - cur;
        if (delta != 0)
        {
          MOVE_POS(p, delta);
        }
      }
      cur = best;
      break;
    }

    newLen = ReadMatchDistances(p, &numPairs); //check matches for byte sequence starting at offset cur
    
    if (newLen >= p->numFastBytes) // break out of parsing cycle once match longer than numFastBytes is found
    {
      p->numPairs = numPairs;
      p->longestMatchLen = newLen;
      break;
    }
    
    curOpt = &p->opt[cur]; // state of system if p->opt[cur] was selected as strategy above

    position++;

    // we need that check here, if skip_items in p->opt are possible
    /*
    if (curOpt->price >= kInfinityPrice)
      continue;
    */

    prev = cur - curOpt->len;

    if (curOpt->len == 1)
    {
      state = (unsigned)p->opt[prev].state;
      if (IsShortRep(curOpt))
        state = kShortRepNextStates[state];
      else
        state = kLiteralNextStates[state];
    }
    else
    {
      const COptimal *prevOpt;
      UInt32 b0;
      UInt32 dist = curOpt->dist;

      if (curOpt->extra)
      {
        prev -= (unsigned)curOpt->extra;
        state = kState_RepAfterLit;
        if (curOpt->extra == 1)
          state = (dist < LZMA_NUM_REPS ? kState_RepAfterLit : kState_MatchAfterLit);
      }
      else
      {
        state = (unsigned)p->opt[prev].state;
        if (dist < LZMA_NUM_REPS)
          state = kRepNextStates[state];
        else
          state = kMatchNextStates[state];
      }

      prevOpt = &p->opt[prev];
      b0 = prevOpt->reps[0];

      if (dist < LZMA_NUM_REPS) //update reps as per dist in curOpt
      {
        if (dist == 0)
        {
          reps[0] = b0;
          reps[1] = prevOpt->reps[1];
          reps[2] = prevOpt->reps[2];
          reps[3] = prevOpt->reps[3];
        }
        else
        {
          reps[1] = b0;
          b0 = prevOpt->reps[1];
          if (dist == 1)
          {
            reps[0] = b0;
            reps[2] = prevOpt->reps[2];
            reps[3] = prevOpt->reps[3];
          }
          else
          {
            reps[2] = b0;
            reps[0] = prevOpt->reps[dist];
            reps[3] = prevOpt->reps[dist ^ 1];
          }
        }
      }
      else
      {
        reps[0] = (dist - LZMA_NUM_REPS + 1);
        reps[1] = b0;
        reps[2] = prevOpt->reps[1];
        reps[3] = prevOpt->reps[2];
      }
    }
    
    curOpt->state = (CState)state;
    curOpt->reps[0] = reps[0];
    curOpt->reps[1] = reps[1];
    curOpt->reps[2] = reps[2];
    curOpt->reps[3] = reps[3];

    data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
    curByte = *data;
    matchByte = *(data - reps[0]);

    posState = (position & p->pbMask);

    /*
    The order of Price checks:
       <  LIT
       <= SHORT_REP
       <  LIT : REP_0
       <  REP    [ : LIT : REP_0 ]
       <  MATCH  [ : LIT : REP_0 ]
    */

    {
      // start with base price and add to it based on strategy for new match sequence
      UInt32 curPrice = curOpt->price;
      unsigned prob = p->isMatch[state][posState];
      matchPrice = curPrice + GET_PRICE_1(prob);
      litPrice = curPrice + GET_PRICE_0(prob);
    }

    nextOpt = &p->opt[(size_t)cur + 1];
    nextIsLit = False;

    // here we can allow skip_items in p->opt, if we don't check (nextOpt->price < kInfinityPrice)
    // 18.new.06
    if ((nextOpt->price < kInfinityPrice
        // && !IsLitState(state)
        && matchByte == curByte)
        || litPrice > nextOpt->price
        )
      litPrice = 0;
    else
    {
      const CLzmaProb *probs = LIT_PROBS(position, *(data - 1));
      litPrice += (!IsLitState(state) ?
          LitEnc_Matched_GetPrice(probs, curByte, matchByte, p->ProbPrices) :
          LitEnc_GetPrice(probs, curByte, p->ProbPrices));
      
      if (litPrice < nextOpt->price)
      {
        nextOpt->price = litPrice;
        nextOpt->len = 1;
        MakeAs_Lit(nextOpt);
        nextIsLit = True;
      }
    }

    repMatchPrice = matchPrice + GET_PRICE_1(p->isRep[state]);
    
    numAvailFull = p->numAvail;
    {
      unsigned temp = kNumOpts - 1 - cur;
      if (numAvailFull > temp)
        numAvailFull = (UInt32)temp;
    }

    // 18.06
    // ---------- SHORT_REP ----------
    if (IsLitState(state)) // 18.new
    if (matchByte == curByte)
    if (repMatchPrice < nextOpt->price) // 18.new
    // if (numAvailFull < 2 || data[1] != *(data - reps[0] + 1))
    if (
        // nextOpt->price >= kInfinityPrice ||
        nextOpt->len < 2   // we can check nextOpt->len, if skip items are not allowed in p->opt
        || (nextOpt->dist != 0
            // && nextOpt->extra <= 1 // 17.old
            )
        )
    {
      UInt32 shortRepPrice = repMatchPrice + GetPrice_ShortRep(p, state, posState);
      // if (shortRepPrice <= nextOpt->price) // 17.old
      if (shortRepPrice < nextOpt->price)  // 18.new
      {
        nextOpt->price = shortRepPrice;
        nextOpt->len = 1;
        MakeAs_ShortRep(nextOpt);
        nextIsLit = False;
      }
    }
    
    if (numAvailFull < 2)
      continue;
    numAvail = (numAvailFull <= p->numFastBytes ? numAvailFull : p->numFastBytes);

    // numAvail <= p->numFastBytes

    // ---------- LIT : REP_0 ----------

    if (!nextIsLit
        && litPrice != 0 // 18.new
        && matchByte != curByte
        && numAvailFull > 2)
    {
      const Byte *data2 = data - reps[0];
      if (data[1] == data2[1] && data[2] == data2[2])
      {
        unsigned len;
        unsigned limit = p->numFastBytes + 1;
        if (limit > numAvailFull)
          limit = numAvailFull;
        for (len = 3; len < limit && data[len] == data2[len]; len++)
        {}
        
        {
          unsigned state2 = kLiteralNextStates[state];
          unsigned posState2 = (position + 1) & p->pbMask;
          UInt32 price = litPrice + GetPrice_Rep_0(p, state2, posState2);
          {
            unsigned offset = cur + len;

            if (last < offset)
              last = offset;
          
            // do
            {
              UInt32 price2;
              COptimal *opt;
              len--;
              // price2 = price + GetPrice_Len_Rep_0(p, len, state2, posState2);
              price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len);

              opt = &p->opt[offset];
              // offset--;
              if (price2 < opt->price)
              {
                opt->price = price2;
                opt->len = (UInt32)len;
                opt->dist = 0;
                opt->extra = 1;
              }
            }
            // while (len >= 3);
          }
        }
      }
    }
    
    startLen = 2; /* speed optimization */

    {
      // ---------- REP ----------
      unsigned repIndex = 0; // 17.old
      // unsigned repIndex = IsLitState(state) ? 0 : 1; // 18.notused
      for (; repIndex < LZMA_NUM_REPS; repIndex++)
      {
        unsigned len;
        UInt32 price;
        const Byte *data2 = data - reps[repIndex];
        if (data[0] != data2[0] || data[1] != data2[1])
          continue;
        
        for (len = 2; len < numAvail && data[len] == data2[len]; len++)
        {}
        
        // if (len < startLen) continue; // 18.new: speed optimization

        {
          unsigned offset = cur + len;
          if (last < offset)
            last = offset;
        }
        {
          unsigned len2 = len;
          price = repMatchPrice + GetPrice_PureRep(p, repIndex, state, posState);
          do
          {
            UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, len2);
            COptimal *opt = &p->opt[cur + len2];
            if (price2 < opt->price)
            {
              opt->price = price2;
              opt->len = (UInt32)len2;
              opt->dist = (UInt32)repIndex;
              opt->extra = 0;
            }
          }
          while (--len2 >= 2);
        }
        
        if (repIndex == 0) startLen = len + 1;  // 17.old
        // startLen = len + 1; // 18.new

        /* if (_maxMode) */
        {
          // ---------- REP : LIT : REP_0 ----------
          // numFastBytes + 1 + numFastBytes

          unsigned len2 = len + 1;
          unsigned limit = len2 + p->numFastBytes;
          if (limit > numAvailFull)
            limit = numAvailFull;
          
          len2 += 2;
          if (len2 <= limit)
          if (data[len2 - 2] == data2[len2 - 2])
          if (data[len2 - 1] == data2[len2 - 1])
          {
            unsigned state2 = kRepNextStates[state];
            unsigned posState2 = (position + len) & p->pbMask;
            price += GET_PRICE_LEN(&p->repLenEnc, posState, len)
                + GET_PRICE_0(p->isMatch[state2][posState2])
                + LitEnc_Matched_GetPrice(LIT_PROBS(position + len, data[(size_t)len - 1]),
                    data[len], data2[len], p->ProbPrices);
            
            // state2 = kLiteralNextStates[state2];
            state2 = kState_LitAfterRep;
            posState2 = (posState2 + 1) & p->pbMask;


            price += GetPrice_Rep_0(p, state2, posState2);

          for (; len2 < limit && data[len2] == data2[len2]; len2++)
          {}
          
          len2 -= len;
          // if (len2 >= 3)
          {
            {
              unsigned offset = cur + len + len2;

              if (last < offset)
                last = offset;
              // do
              {
                UInt32 price2;
                COptimal *opt;
                len2--;
                // price2 = price + GetPrice_Len_Rep_0(p, len2, state2, posState2);
                price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len2);

                opt = &p->opt[offset];
                // offset--;
                if (price2 < opt->price)
                {
                  opt->price = price2;
                  opt->len = (UInt32)len2;
                  opt->extra = (CExtra)(len + 1);
                  opt->dist = (UInt32)repIndex;
                }
              }
              // while (len2 >= 3);
            }
          }
          }
        }
      }
    }


    // ---------- MATCH ----------
    /* for (unsigned len = 2; len <= newLen; len++) */
    if (newLen > numAvail)
    {
      newLen = numAvail;
      for (numPairs = 0; newLen > MATCHES[numPairs]; numPairs += 2);
      MATCHES[numPairs] = (UInt32)newLen;
      numPairs += 2;
    }
    
    // startLen = 2; /* speed optimization */

    if (newLen >= startLen)
    {
      UInt32 normalMatchPrice = matchPrice + GET_PRICE_0(p->isRep[state]);
      UInt32 dist;
      unsigned offs, posSlot, len;
      
      {
        unsigned offset = cur + newLen;
        if (last < offset)
          last = offset;
      }

      offs = 0;
      while (startLen > MATCHES[offs])
        offs += 2;
      dist = MATCHES[(size_t)offs + 1];
      
      // if (dist >= kNumFullDistances)
      GetPosSlot2(dist, posSlot);
      
      for (len = /*2*/ startLen; ; len++)
      {
        UInt32 price = normalMatchPrice + GET_PRICE_LEN(&p->lenEnc, posState, len);
        {
          COptimal *opt;
          unsigned lenNorm = len - 2;
          lenNorm = GetLenToPosState2(lenNorm);
          if (dist < kNumFullDistances)
            price += p->distancesPrices[lenNorm][dist & (kNumFullDistances - 1)];
          else
            price += p->posSlotPrices[lenNorm][posSlot] + p->alignPrices[dist & kAlignMask];
          
          opt = &p->opt[cur + len];
          if (price < opt->price)
          {
            opt->price = price;
            opt->len = (UInt32)len;
            opt->dist = dist + LZMA_NUM_REPS;
            opt->extra = 0;
          }
        }

        if (len == MATCHES[offs])
        {
          // if (p->_maxMode) {
          // MATCH : LIT : REP_0

          const Byte *data2 = data - dist - 1;
          unsigned len2 = len + 1;
          unsigned limit = len2 + p->numFastBytes;
          if (limit > numAvailFull)
            limit = numAvailFull;
          
          len2 += 2;
          if (len2 <= limit)
          if (data[len2 - 2] == data2[len2 - 2])
          if (data[len2 - 1] == data2[len2 - 1])
          {
          for (; len2 < limit && data[len2] == data2[len2]; len2++)
          {}
          
          len2 -= len;
          
          // if (len2 >= 3)
          {
            unsigned state2 = kMatchNextStates[state];
            unsigned posState2 = (position + len) & p->pbMask;
            unsigned offset;
            price += GET_PRICE_0(p->isMatch[state2][posState2]);
            price += LitEnc_Matched_GetPrice(LIT_PROBS(position + len, data[(size_t)len - 1]),
                    data[len], data2[len], p->ProbPrices);

            // state2 = kLiteralNextStates[state2];
            state2 = kState_LitAfterMatch;

            posState2 = (posState2 + 1) & p->pbMask;
            price += GetPrice_Rep_0(p, state2, posState2);

            offset = cur + len + len2;

            if (last < offset)
              last = offset;
            // do
            {
              UInt32 price2;
              COptimal *opt;
              len2--;
              // price2 = price + GetPrice_Len_Rep_0(p, len2, state2, posState2);
              price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len2);
              opt = &p->opt[offset];
              // offset--;
              if (price2 < opt->price)
              {
                opt->price = price2;
                opt->len = (UInt32)len2;
                opt->extra = (CExtra)(len + 1);
                opt->dist = dist + LZMA_NUM_REPS;
              }
            }
            // while (len2 >= 3);
          }

          }
        
          offs += 2;
          if (offs == numPairs)
            break;
          dist = MATCHES[(size_t)offs + 1];
          // if (dist >= kNumFullDistances)
            GetPosSlot2(dist, posSlot);
        }
      }
    }
  }

  do
    p->opt[last].price = kInfinityPrice; //reset prices to max
  while (--last);

  return Backward(p, cur); // select best <len, distance> pair from values computed in p->opt[i]
}

#ifdef AOCL_LZMA_OPT
#define AOCL_OPT_PARSE_REP(repIndex) /* if (_maxMode) */ \
{ \
/*---------- REP : LIT : REP_0 ----------*/  \
/* numFastBytes + 1 + numFastBytes*/ \
unsigned len2 = len + 1; \
unsigned limit = len2 + p->numFastBytes; \
if (limit > numAvailFull) \
    limit = numAvailFull; \
 \
len2 += 2; \
if (len2 <= limit) \
    if (data[len2 - 2] == data2[len2 - 2]) \
        if (data[len2 - 1] == data2[len2 - 1]) \
        { \
            unsigned state2 = kRepNextStates[state]; \
            unsigned posState2 = (position + len) & p->pbMask; \
            price += GET_PRICE_LEN(&p->repLenEnc, posState, len) \
                + GET_PRICE_0(p->isMatch[state2][posState2]) \
                + LitEnc_Matched_GetPrice(LIT_PROBS(position + len, data[(size_t)len - 1]), \
                    data[len], data2[len], p->ProbPrices); \
 \
            /* state2 = kLiteralNextStates[state2];*/  \
            state2 = kState_LitAfterRep; \
            posState2 = (posState2 + 1) & p->pbMask; \
 \
            price += GetPrice_Rep_0(p, state2, posState2); \
 \
            for (; len2 < limit && data[len2] == data2[len2]; len2++) \
            {}  \
 \
            len2 -= len; \
            /*if (len2 >= 3)*/ \
            { \
                { \
                    unsigned offset = cur + len + len2; \
 \
                    if (last < offset) \
                        last = offset; \
                    /*do*/ \
                    { \
                        UInt32 price2; \
                        COptimal* opt; \
                        len2--; \
                        /* price2 = price + GetPrice_Len_Rep_0(p, len2, state2, posState2);*/ \
                        price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len2); \
                         \
                        opt = &p->opt[offset]; \
                        /* offset--;*/ \
                        if (price2 < opt->price) \
                        { \
                            opt->price = price2; \
                            opt->len = (UInt32)len2; \
                            opt->extra = (CExtra)(len + 1); \
                            opt->dist = (UInt32)repIndex; \
                        } \
                    } \
                    /* while (len2 >= 3); */ \
                } \
            } \
        } \
}

// Compare bytes in data2 and data1 using UInt32 ptrs and __builtin_ctz
#define AOCL_FIND_MATCHING_BYTES_LEN(len, limit, data1, data2) { \
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
            break; \
        } \
        len += 4; \
    } \
} \
while (len < limit) { \
    if (data2[len] != data1[len]) \
        break; \
    len++; \
} \
}

MY_FORCE_INLINE
static UInt32 AOCL_GetPrice_PureRep_Non0(const CLzmaEnc* p, unsigned repIndex, size_t state, size_t posState)
{
    UInt32 prob = p->isRepG0[state];
    UInt32 price = GET_PRICE_1(prob);
    prob = p->isRepG1[state];
    if (repIndex == 1)
        price += GET_PRICE_0(prob);
    else
    {
        price += GET_PRICE_1(prob);
        price += GET_PRICE(p->isRepG2[state], repIndex - 2);
    }
    return price;
}

MY_FORCE_INLINE
static UInt32 AOCL_GetPrice_PureRep_0(const CLzmaEnc* p, size_t state, size_t posState)
{
    UInt32 prob = p->isRepG0[state];
    UInt32 price = GET_PRICE_0(prob);
    price += GET_PRICE_1(p->isRep0Long[state][posState]);
    return price;
}

/* @brief: Use optimal parsing to find best matches 
*  Changes:
*   - rep 0 loop unrolling in : for (i = 0; i < LZMA_NUM_REPS; i++)
*   - Byte matching using AOCL_FIND_MATCHING_BYTES_LEN()
*   - if (data[0] != data2[0] || data[1] != data2[1]) TO if (GetUi16(data) != GetUi16(data2))
*   - p->opt[0].reps[0] = reps[0]... TO memcpy */
static unsigned AOCL_GetOptimum(CLzmaEnc* p, UInt32 position)
{
    unsigned last, cur;
    UInt32 reps[LZMA_NUM_REPS];
    unsigned repLens[LZMA_NUM_REPS];
    UInt32* matches;

    {
        UInt32 numAvail;
        unsigned numPairs, mainLen, repMaxIndex, i, posState;
        UInt32 matchPrice, repMatchPrice;
        const Byte* data;
        Byte curByte, matchByte;

        p->optCur = p->optEnd = 0;

        /* if additionalOffset is > 0, parsing for cur byte has already been completed
        * in one of the previous ReadMatchDistances() calls inside optimal parsing loop.
        * Results are already available in p-> and can be used
        * without having to find matches again. */
        if (p->additionalOffset == 0)
            mainLen = ReadMatchDistances(p, &numPairs);
        else
        {
            mainLen = p->longestMatchLen;
            numPairs = p->numPairs;
        }

        numAvail = p->numAvail;
        if (numAvail < 2)
        {
            p->backRes = MARK_LIT;
            return 1;
        }
        if (numAvail > LZMA_MATCH_LEN_MAX)
            numAvail = LZMA_MATCH_LEN_MAX;

        data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
        repMaxIndex = 0; //rep with max len match

        // Get match lengths at distances in each of the reps for cur data
        for (i = 0; i < LZMA_NUM_REPS; i++)
        {
            unsigned len;
            const Byte* data2;
            reps[i] = p->reps[i];
#ifdef AOCL_ENABLE_THREADS
            if(p->reps[i] == -1)
            {
              repLens[i] = 0;
              continue;
            }
#endif /* AOCL_ENABLE_THREADS */
            data2 = data - reps[i];
            if (GetUi16(data) != GetUi16(data2)) //(data[0] != data2[0] || data[1] != data2[1])
            {
                repLens[i] = 0;
                continue;
            }

            len = 2;
            AOCL_FIND_MATCHING_BYTES_LEN(len, numAvail, data, data2) //for (len = 2; len < numAvail && data[len] == data2[len]; len++){}
            
                repLens[i] = len;
            if (len > repLens[repMaxIndex])
                repMaxIndex = i;
            if (len == LZMA_MATCH_LEN_MAX) // 21.03 : optimization
                break;
        }

        /* If long enough match is found in a rep, it can be selected
        * instead of <mainLen, mainDist> as we save on encoding mainDist */
        if (repLens[repMaxIndex] >= p->numFastBytes)
        {
            unsigned len;
            p->backRes = (UInt32)repMaxIndex;
            len = repLens[repMaxIndex];
            MOVE_POS(p, len - 1)
                return len;
        }

        matches = p->matches;
#define MATCHES  matches
        // #define MATCHES  p->matches

        if (mainLen >= p->numFastBytes)  // if len long enough, choose mainLen
        {
            p->backRes = MATCHES[(size_t)numPairs - 1] + LZMA_NUM_REPS;
            MOVE_POS(p, mainLen - 1)
                return mainLen;
        }

        last = repLens[repMaxIndex]; // last = longest matched length among reps and mainLen
        if (last <= mainLen)
            last = mainLen;
#ifdef AOCL_ENABLE_THREADS
        if(reps[0] == -1)
        {
          p->backRes = MARK_LIT;
          return 1;
        }
#endif /* AOCL_ENABLE_THREADS */
        
        curByte = *data;
        matchByte = *(data - reps[0]); 

        if (last < 2 && curByte != matchByte)
        {
            p->backRes = MARK_LIT;
            return 1;
        }

        p->opt[0].state = (CState)p->state;

        posState = (position & p->pbMask);

        /* From this point onwards, cost for taking different decisions 'price'
        * is calculated. Every decision, choosing to encode as a literal,
        * srep, rep or match has a corresponding price. */
        {
            const CLzmaProb* probs = LIT_PROBS(position, *(data - 1));
            p->opt[1].price = GET_PRICE_0(p->isMatch[p->state][posState]) +
                (!IsLitState(p->state) ?
                    LitEnc_Matched_GetPrice(probs, curByte, matchByte, p->ProbPrices) :
                    LitEnc_GetPrice(probs, curByte, p->ProbPrices));
        }

        MakeAs_Lit(&p->opt[1]);

        matchPrice = GET_PRICE_1(p->isMatch[p->state][posState]);
        repMatchPrice = matchPrice + GET_PRICE_1(p->isRep[p->state]);

        // 18.06
        if (matchByte == curByte && repLens[0] == 0)
        {
            UInt32 shortRepPrice = repMatchPrice + GetPrice_ShortRep(p, p->state, posState);
            if (shortRepPrice < p->opt[1].price)
            {
                p->opt[1].price = shortRepPrice;
                MakeAs_ShortRep(&p->opt[1]);
            }
            if (last < 2)
            {
                p->backRes = p->opt[1].dist;
                return 1;
            }
        }

        p->opt[1].len = 1;

        /*p->opt[0].reps[0] = reps[0];
        p->opt[0].reps[1] = reps[1];
        p->opt[0].reps[2] = reps[2];
        p->opt[0].reps[3] = reps[3];*/
        memcpy(p->opt[0].reps, reps, 4 * sizeof(UInt32));

        // ---------- REP ----------
        {
            // REP 0
            unsigned repLen = repLens[0];
            UInt32 price;
            if (repLen >= 2) {
                price = repMatchPrice + AOCL_GetPrice_PureRep_0(p, p->state, posState);
                do
                {
                    UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, repLen);
                    COptimal* opt = &p->opt[repLen];
                    if (price2 < opt->price)
                    {
                        opt->price = price2;
                        opt->len = (UInt32)repLen;
                        opt->dist = (UInt32)0;
                        opt->extra = 0;
                    }
                } while (--repLen >= 2);
            }
        }

        // REP > 0
        for (i = 1; i < LZMA_NUM_REPS; i++)
        {
            unsigned repLen = repLens[i];
            UInt32 price;
            if (repLen < 2)
                continue;
            price = repMatchPrice + AOCL_GetPrice_PureRep_Non0(p, i, p->state, posState);
            do
            {
                UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, repLen);
                COptimal* opt = &p->opt[repLen];
                if (price2 < opt->price)
                {
                    opt->price = price2;
                    opt->len = (UInt32)repLen;
                    opt->dist = (UInt32)i;
                    opt->extra = 0;
                }
            } while (--repLen >= 2);
        }


        // ---------- MATCH ----------
        {
            unsigned len = repLens[0] + 1;
            if (len <= mainLen)
            {
                unsigned offs = 0;
                UInt32 normalMatchPrice = matchPrice + GET_PRICE_0(p->isRep[p->state]);

                if (len < 2)
                    len = 2;
                else
                    while (len > MATCHES[offs])
                        offs += 2;

                for (; ; len++)
                {
                    COptimal* opt;
                    UInt32 dist = MATCHES[(size_t)offs + 1];
                    UInt32 price = normalMatchPrice + GET_PRICE_LEN(&p->lenEnc, posState, len);
                    unsigned lenToPosState = GetLenToPosState(len);

                    if (dist < kNumFullDistances)
                        price += p->distancesPrices[lenToPosState][dist & (kNumFullDistances - 1)];
                    else
                    {
                        unsigned slot;
                        GetPosSlot2(dist, slot);
                        price += p->alignPrices[dist & kAlignMask];
                        price += p->posSlotPrices[lenToPosState][slot];
                    }

                    opt = &p->opt[len];

                    if (price < opt->price)
                    {
                        opt->price = price;
                        opt->len = (UInt32)len;
                        opt->dist = dist + LZMA_NUM_REPS;
                        opt->extra = 0;
                    }

                    if (len == MATCHES[offs])
                    {
                        offs += 2;
                        if (offs == numPairs)
                            break;
                    }
                }
            }
        }


        cur = 0;

#ifdef SHOW_STAT2
        /* if (position >= 0) */
        {
            unsigned i;
            printf("\n pos = %4X", position);
            for (i = cur; i <= last; i++)
                printf("\nprice[%4X] = %u", position - cur + i, p->opt[i].price);
        }
#endif
    }



    // ---------- Optimal Parsing ----------
    /* last : longest matched length among reps and mainLen
    * p->opt[i] : base price and system states until now
    * cur : counter starting at 0
    *
    * For cur in range [0, last) we check if skipping these many bytes (L) and
    * coding them using the strategy in p->opt[L] is worth it, if we can
    * find a better matching sequence in the future that gives a better price overall.
    * Checking for last and beyond isn't necessary as we might as well encode the
    * current match using the best price strategy we have so far and leave the rest
    * of the sequence for the next GetOptimum() call.
    *
    * p->opt[L] also contains state and reps values if said strategy was used. This
    * will act as previous system state when evaluating price for byte sequence
    * starting at offset L.
    *
    * Run ReadMatchDistances() at offsets [0, last) from current position.
    * Update p->opt s (p->opt[i].price += newPrice, etc) for each p->opt[i]. */
    for (;;)
    {
        unsigned numAvail;
        UInt32 numAvailFull;
        unsigned newLen, numPairs, prev, state, posState, startLen;
        UInt32 litPrice, matchPrice, repMatchPrice;
        BoolInt nextIsLit;
        Byte curByte, matchByte;
        const Byte* data;
        COptimal* curOpt, * nextOpt;

        if (++cur == last)
            break; // no need to check beyond last. This can be handled in the next GetOptimum() call.

        // 18.06
        if (cur >= kNumOpts - 64)
        {
            unsigned j, best;
            UInt32 price = p->opt[cur].price;
            best = cur;
            for (j = cur + 1; j <= last; j++)
            {
                UInt32 price2 = p->opt[j].price;
                if (price >= price2)
                {
                    price = price2;
                    best = j;
                }
            }
            {
                unsigned delta = best - cur;
                if (delta != 0)
                {
                    MOVE_POS(p, delta);
                }
            }
            cur = best;
            break;
        }

        newLen = ReadMatchDistances(p, &numPairs); //check matches for byte sequence starting at offset cur

        if (newLen >= p->numFastBytes) // break out of parsing cycle once match longer than numFastBytes is found
        {
            p->numPairs = numPairs;
            p->longestMatchLen = newLen;
            break;
        }

        curOpt = &p->opt[cur]; // state of system if p->opt[cur] was selected as strategy above

        position++;

        // we need that check here, if skip_items in p->opt are possible
        /*
        if (curOpt->price >= kInfinityPrice)
          continue;
        */

        prev = cur - curOpt->len;

        if (curOpt->len == 1)
        {
            state = (unsigned)p->opt[prev].state;
            if (IsShortRep(curOpt))
                state = kShortRepNextStates[state];
            else
                state = kLiteralNextStates[state];
        }
        else
        {
            const COptimal* prevOpt;
            UInt32 b0;
            UInt32 dist = curOpt->dist;

            if (curOpt->extra)
            {
                prev -= (unsigned)curOpt->extra;
                state = kState_RepAfterLit;
                if (curOpt->extra == 1)
                    state = (dist < LZMA_NUM_REPS ? kState_RepAfterLit : kState_MatchAfterLit);
            }
            else
            {
                state = (unsigned)p->opt[prev].state;
                if (dist < LZMA_NUM_REPS)
                    state = kRepNextStates[state];
                else
                    state = kMatchNextStates[state];
            }

            prevOpt = &p->opt[prev];
            b0 = prevOpt->reps[0];

            if (dist < LZMA_NUM_REPS) //update reps as per dist in curOpt
            {
                if (dist == 0)
                {
                    reps[0] = b0;
                    reps[1] = prevOpt->reps[1];
                    reps[2] = prevOpt->reps[2];
                    reps[3] = prevOpt->reps[3];
                }
                else
                {
                    reps[1] = b0;
                    b0 = prevOpt->reps[1];
                    if (dist == 1)
                    {
                        reps[0] = b0;
                        reps[2] = prevOpt->reps[2];
                        reps[3] = prevOpt->reps[3];
                    }
                    else
                    {
                        reps[2] = b0;
                        reps[0] = prevOpt->reps[dist];
                        reps[3] = prevOpt->reps[dist ^ 1];
                    }
                }
            }
            else
            {
                reps[0] = (dist - LZMA_NUM_REPS + 1);
                reps[1] = b0;
                reps[2] = prevOpt->reps[1];
                reps[3] = prevOpt->reps[2];
            }
        }

        curOpt->state = (CState)state;
        memcpy(curOpt->reps, reps, 4 * sizeof(UInt32));
        /*curOpt->reps[0] = reps[0];
        curOpt->reps[1] = reps[1];
        curOpt->reps[2] = reps[2];
        curOpt->reps[3] = reps[3];*/

        data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
        curByte = *data;
        matchByte = *(data - reps[0]);

        posState = (position & p->pbMask);

        /*
        The order of Price checks:
           <  LIT
           <= SHORT_REP
           <  LIT : REP_0
           <  REP    [ : LIT : REP_0 ]
           <  MATCH  [ : LIT : REP_0 ]
        */

        {
            // start with base price and add to it based on strategy for new match sequence
            UInt32 curPrice = curOpt->price;
            unsigned prob = p->isMatch[state][posState];
            matchPrice = curPrice + GET_PRICE_1(prob);
            litPrice = curPrice + GET_PRICE_0(prob);
        }

        nextOpt = &p->opt[(size_t)cur + 1];
        nextIsLit = False;

        // here we can allow skip_items in p->opt, if we don't check (nextOpt->price < kInfinityPrice)
        // 18.new.06
        if ((nextOpt->price < kInfinityPrice
            // && !IsLitState(state)
            && matchByte == curByte)
            || litPrice > nextOpt->price
            )
            litPrice = 0;
        else
        {
            const CLzmaProb* probs = LIT_PROBS(position, *(data - 1));
            litPrice += (!IsLitState(state) ?
                LitEnc_Matched_GetPrice(probs, curByte, matchByte, p->ProbPrices) :
                LitEnc_GetPrice(probs, curByte, p->ProbPrices));

            if (litPrice < nextOpt->price)
            {
                nextOpt->price = litPrice;
                nextOpt->len = 1;
                MakeAs_Lit(nextOpt);
                nextIsLit = True;
            }
        }

        repMatchPrice = matchPrice + GET_PRICE_1(p->isRep[state]);

        numAvailFull = p->numAvail;
        {
            unsigned temp = kNumOpts - 1 - cur;
            if (numAvailFull > temp)
                numAvailFull = (UInt32)temp;
        }

        // 18.06
        // ---------- SHORT_REP ----------
        if (IsLitState(state)) // 18.new
            if (matchByte == curByte)
                if (repMatchPrice < nextOpt->price) // 18.new
                    // if (numAvailFull < 2 || data[1] != *(data - reps[0] + 1))
                    if (
                        // nextOpt->price >= kInfinityPrice ||
                        nextOpt->len < 2   // we can check nextOpt->len, if skip items are not allowed in p->opt
                        || (nextOpt->dist != 0
                            // && nextOpt->extra <= 1 // 17.old
                            )
                        )
                    {
                        UInt32 shortRepPrice = repMatchPrice + GetPrice_ShortRep(p, state, posState);
                        // if (shortRepPrice <= nextOpt->price) // 17.old
                        if (shortRepPrice < nextOpt->price)  // 18.new
                        {
                            nextOpt->price = shortRepPrice;
                            nextOpt->len = 1;
                            MakeAs_ShortRep(nextOpt);
                            nextIsLit = False;
                        }
                    }

        if (numAvailFull < 2)
            continue;
        numAvail = (numAvailFull <= p->numFastBytes ? numAvailFull : p->numFastBytes);

        // numAvail <= p->numFastBytes

        // ---------- LIT : REP_0 ----------

        if (!nextIsLit
            && litPrice != 0 // 18.new
            && matchByte != curByte
            && numAvailFull > 2
#ifdef AOCL_ENABLE_THREADS
            && reps[0] != -1
#endif /* AOCL_ENABLE_THREADS */
            )
        {
            const Byte* data2 = data - reps[0];
            if (GetUi16(data + 1) == GetUi16(data2 + 1)) // if (data[1] == data2[1] && data[2] == data2[2])
            {
                unsigned len;
                unsigned limit = p->numFastBytes + 1;
                if (limit > numAvailFull)
                    limit = numAvailFull;
                len = 3;
                AOCL_FIND_MATCHING_BYTES_LEN(len, limit, data, data2); //for (len = 3; len < limit && data[len] == data2[len]; len++){}

                {
                    unsigned state2 = kLiteralNextStates[state];
                    unsigned posState2 = (position + 1) & p->pbMask;
                    UInt32 price = litPrice + GetPrice_Rep_0(p, state2, posState2);
                    {
                        unsigned offset = cur + len;

                        if (last < offset)
                            last = offset;

                        // do
                        {
                            UInt32 price2;
                            COptimal* opt;
                            len--;
                            // price2 = price + GetPrice_Len_Rep_0(p, len, state2, posState2);
                            price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len);

                            opt = &p->opt[offset];
                            // offset--;
                            if (price2 < opt->price)
                            {
                                opt->price = price2;
                                opt->len = (UInt32)len;
                                opt->dist = 0;
                                opt->extra = 1;
                            }
                        }
                        // while (len >= 3);
                    }
                }
            }
        }

        startLen = 2; /* speed optimization */

        {
            // ---------- REP 0 ----------
            unsigned len;
            UInt32 price;
#ifdef AOCL_ENABLE_THREADS
            if(reps[0] != -1)
#endif /* AOCL_ENABLE_THREADS */
            {
              const Byte* data2 = data - reps[0];
              if (GetUi16(data) == GetUi16(data2)) //if (data[0] == data2[0] && data[1] == data2[1]) {
              {
                  len = 2;
                  AOCL_FIND_MATCHING_BYTES_LEN(len, numAvail, data, data2) // for (len = 2; len < numAvail && data[len] == data2[len]; len++) {}
                  // if (len < startLen) continue; // 18.new: speed optimization
                  {
                      unsigned offset = cur + len;
                      if (last < offset)
                          last = offset;
                  }
                  {
                      unsigned len2 = len;
                      price = repMatchPrice + AOCL_GetPrice_PureRep_0(p, state, posState);
                      do
                      {
                          UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, len2);
                          COptimal* opt = &p->opt[cur + len2];
                          if (price2 < opt->price)
                          {
                              opt->price = price2;
                              opt->len = (UInt32)len2;
                              opt->dist = (UInt32)0;
                              opt->extra = 0;
                          }
                      } while (--len2 >= 2);
                  }

                  startLen = len + 1;  // 17.old

                  AOCL_OPT_PARSE_REP(0)
              }
            }
        }

        {
            // ---------- REP > 0 ----------
            // unsigned repIndex = IsLitState(state) ? 0 : 1; // 18.notused
            for (unsigned repIndex = 1; repIndex < LZMA_NUM_REPS; repIndex++)
            {
                unsigned len;
                UInt32 price;
#ifdef AOCL_ENABLE_THREADS
                if(reps[repIndex] == -1) continue;
#endif /* AOCL_ENABLE_THREADS */
                const Byte* data2 = data - reps[repIndex];
                if (GetUi16(data) != GetUi16(data2)) //if (data[0] != data2[0] || data[1] != data2[1])
                    continue;

                len = 2;
                AOCL_FIND_MATCHING_BYTES_LEN(len, numAvail, data, data2) // for (len = 2; len < numAvail && data[len] == data2[len]; len++) {}
                // if (len < startLen) continue; // 18.new: speed optimization
                {
                    unsigned offset = cur + len;
                    if (last < offset)
                        last = offset;
                }
                {
                    unsigned len2 = len;
                    price = repMatchPrice + AOCL_GetPrice_PureRep_Non0(p, repIndex, state, posState);
                    do
                    {
                        UInt32 price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState, len2);
                        COptimal* opt = &p->opt[cur + len2];
                        if (price2 < opt->price)
                        {
                            opt->price = price2;
                            opt->len = (UInt32)len2;
                            opt->dist = (UInt32)repIndex;
                            opt->extra = 0;
                        }
                    } while (--len2 >= 2);
                }

                // startLen = len + 1; // 18.new

                AOCL_OPT_PARSE_REP(repIndex)
            }
        }


        // ---------- MATCH ----------
        /* for (unsigned len = 2; len <= newLen; len++) */
        if (newLen > numAvail)
        {
            newLen = numAvail;
            for (numPairs = 0; newLen > MATCHES[numPairs]; numPairs += 2);
            MATCHES[numPairs] = (UInt32)newLen;
            numPairs += 2;
        }

        // startLen = 2; /* speed optimization */

        if (newLen >= startLen)
        {
            UInt32 normalMatchPrice = matchPrice + GET_PRICE_0(p->isRep[state]);
            UInt32 dist;
            unsigned offs, posSlot, len;

            {
                unsigned offset = cur + newLen;
                if (last < offset)
                    last = offset;
            }

            offs = 0;
            while (startLen > MATCHES[offs])
                offs += 2;
            dist = MATCHES[(size_t)offs + 1];

            // if (dist >= kNumFullDistances)
            GetPosSlot2(dist, posSlot);

            for (len = /*2*/ startLen; ; len++)
            {
                UInt32 price = normalMatchPrice + GET_PRICE_LEN(&p->lenEnc, posState, len);
                {
                    COptimal* opt;
                    unsigned lenNorm = len - 2;
                    lenNorm = GetLenToPosState2(lenNorm);
                    if (dist < kNumFullDistances)
                        price += p->distancesPrices[lenNorm][dist & (kNumFullDistances - 1)];
                    else
                        price += p->posSlotPrices[lenNorm][posSlot] + p->alignPrices[dist & kAlignMask];

                    opt = &p->opt[cur + len];
                    if (price < opt->price)
                    {
                        opt->price = price;
                        opt->len = (UInt32)len;
                        opt->dist = dist + LZMA_NUM_REPS;
                        opt->extra = 0;
                    }
                }

                if (len == MATCHES[offs])
                {
                    // if (p->_maxMode) {
                    // MATCH : LIT : REP_0

                    const Byte* data2 = data - dist - 1;
                    unsigned len2 = len + 1;
                    unsigned limit = len2 + p->numFastBytes;
                    if (limit > numAvailFull)
                        limit = numAvailFull;

                    len2 += 2;
                    if (len2 <= limit)
                        if (GetUi16(data + len2 - 2) == GetUi16(data2 + len2 - 2))
                            //if (data[len2 - 2] == data2[len2 - 2]) //if (data[len2 - 1] == data2[len2 - 1])
                        {
                            AOCL_FIND_MATCHING_BYTES_LEN(len2, limit, data, data2) //for (; len2 < limit && data[len2] == data2[len2]; len2++){}

                            len2 -= len;

                            // if (len2 >= 3)
                            {
                                unsigned state2 = kMatchNextStates[state];
                                unsigned posState2 = (position + len) & p->pbMask;
                                unsigned offset;
                                price += GET_PRICE_0(p->isMatch[state2][posState2]);
                                price += LitEnc_Matched_GetPrice(LIT_PROBS(position + len, data[(size_t)len - 1]),
                                    data[len], data2[len], p->ProbPrices);

                                // state2 = kLiteralNextStates[state2];
                                state2 = kState_LitAfterMatch;

                                posState2 = (posState2 + 1) & p->pbMask;
                                price += GetPrice_Rep_0(p, state2, posState2);

                                offset = cur + len + len2;

                                if (last < offset)
                                    last = offset;
                                // do
                                {
                                    UInt32 price2;
                                    COptimal* opt;
                                    len2--;
                                    // price2 = price + GetPrice_Len_Rep_0(p, len2, state2, posState2);
                                    price2 = price + GET_PRICE_LEN(&p->repLenEnc, posState2, len2);
                                    opt = &p->opt[offset];
                                    // offset--;
                                    if (price2 < opt->price)
                                    {
                                        opt->price = price2;
                                        opt->len = (UInt32)len2;
                                        opt->extra = (CExtra)(len + 1);
                                        opt->dist = dist + LZMA_NUM_REPS;
                                    }
                                }
                                // while (len2 >= 3);
                            }

                        }

                    offs += 2;
                    if (offs == numPairs)
                        break;
                    dist = MATCHES[(size_t)offs + 1];
                    // if (dist >= kNumFullDistances)
                    GetPosSlot2(dist, posSlot);
                }
            }
        }
    }

    do
        p->opt[last].price = kInfinityPrice; //reset prices to max
    while (--last);

    return Backward(p, cur); // select best <len, distance> pair from values computed in p->opt[i]
}
#endif

#define ChangePair(smallDist, bigDist) (((bigDist) >> 7) > (smallDist))


/* Hack on optimal parsing strategy with some heuristics to provide
* faster matches albeit not optimal */ 
static unsigned GetOptimumFast(CLzmaEnc *p)
{
  UInt32 numAvail, mainDist;
  unsigned mainLen, numPairs, repIndex, repLen, i;
  const Byte *data;

  /* if additionalOffset is > 0, parsing for cur byte has already been completed 
  * in one of the previous ReadMatchDistances() calls used to check matches for
  * future sequences. Results are already available in p-> and can be used 
  * without having to find matches again. */
  if (p->additionalOffset == 0)
    mainLen = ReadMatchDistances(p, &numPairs); // find matches in dictionary. Return longest matched length.
  else
  {
    mainLen = p->longestMatchLen;
    numPairs = p->numPairs;
  }

  numAvail = p->numAvail;
  p->backRes = MARK_LIT;
  if (numAvail < 2)
    return 1;
  // if (mainLen < 2 && p->state == 0) return 1; // 18.06.notused
  if (numAvail > LZMA_MATCH_LEN_MAX)
    numAvail = LZMA_MATCH_LEN_MAX; // max len allowed by lzma format
  data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
  repLen = repIndex = 0;
  
  /* Find longest match at distances in any of the reps for 'data'.
  * If long enough match is found at any of the reps, it can be used
  * instead of <mainLen, mainDist> as we save on encoding mainDist */
  for (i = 0; i < LZMA_NUM_REPS; i++)
  {
    unsigned len;
    const Byte *data2 = data - p->reps[i];
    if (data[0] != data2[0] || data[1] != data2[1])
      continue;
    for (len = 2; len < numAvail && data[len] == data2[len]; len++)
    {}
    if (len >= p->numFastBytes)
    {
      p->backRes = (UInt32)i;
      MOVE_POS(p, len - 1)
      return len;
    }
    if (len > repLen)
    {
      repIndex = i; // rep with longest match
      repLen = len; // longest match len
    }
  }

  if (mainLen >= p->numFastBytes) //if len long enough, just choose this
  {
    p->backRes = p->matches[(size_t)numPairs - 1] + LZMA_NUM_REPS;
    MOVE_POS(p, mainLen - 1)
    return mainLen; // choose mainLen as final match
  }

  mainDist = 0; /* for GCC */
  
  if (mainLen >= 2)
  {
    /* mainLen and mainDist are the last 2 of the valid elements in p->matches
    * mainLen: p->matches[numPairs - 2], mainDist: p->matches[numPairs - 1]
    * As we go up the list, we will have matches with lesser len (which will also
    * have smaller distances). Start comparing with previous entries in matches.
    * LZ77 always chooses the largest 'len' match irrespective of the distance.
    * In lzma, choosing a smaller <len, distance> pair can be justified if a smaller 
    * 'len' match is available at a shorter distance.
    * Heuristic of difference of len being > 1 or difference in distance not being
    * large enough is used to justify which pair to choose. */
    mainDist = p->matches[(size_t)numPairs - 1];
    while (numPairs > 2)
    {
      UInt32 dist2;
      if (mainLen != p->matches[(size_t)numPairs - 4] + 1) // compare with previous matches len
        break; // difference in len > 1 break
      dist2 = p->matches[(size_t)numPairs - 3];
      if (!ChangePair(dist2, mainDist))  // compare with previous matches distance
        break; // distance not large enough : dist2 > mainDist/7 break
      numPairs -= 2;
      mainLen--;
      mainDist = dist2;
    }
    if (mainLen == 2 && mainDist >= 0x80)
      mainLen = 1;
  }

  /* If any of the rep matches has repLen close to or larger than
  * the new mainLen, then it is better to stick to that repLen as
  * you will save on encoding distance */
  if (repLen >= 2)
    if (    repLen + 1 >= mainLen
        || (repLen + 2 >= mainLen && mainDist >= (1 << 9))
        || (repLen + 3 >= mainLen && mainDist >= (1 << 15)))
  {
    p->backRes = (UInt32)repIndex;
    MOVE_POS(p, repLen - 1)
    return repLen; // choose repLen as final match
  }
  
  if (mainLen < 2 || numAvail <= 2)
    return 1;

  /* Check if skipping a byte and coding it as a literal is worth it, i.e.
  * if match for sequence starting at next byte is better overall */
  {
    unsigned len1 = ReadMatchDistances(p, &p->numPairs); // Find matches by skipping a byte
    p->longestMatchLen = len1;
  
    if (len1 >= 2)
    {
      UInt32 newDist = p->matches[(size_t)p->numPairs - 1];
      // Criterion used to justify if we can skip current byte
      if (   (len1 >= mainLen && newDist < mainDist)
          || (len1 == mainLen + 1 && !ChangePair(mainDist, newDist))
          || (len1 >  mainLen + 1)
          || (len1 + 1 >= mainLen && mainLen >= 3 && ChangePair(newDist, mainDist)))
        return 1; // code current byte as literal
    }
  }
  
  data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
  
  /* Compare with reps for new sequence. This might still justify skipping
  * a byte, even if criterion above is not satisfied */
  for (i = 0; i < LZMA_NUM_REPS; i++)
  {
    unsigned len, limit;
    const Byte *data2 = data - p->reps[i];
    if (data[0] != data2[0] || data[1] != data2[1])
      continue;
    limit = mainLen - 1;
    for (len = 2;; len++)
    {
      if (len >= limit)
        return 1; // new sequence provides long enough match at this rep. Worth skipping a byte.
      if (data[len] != data2[len])
        break;
    }
  }
  
  /* Not worth skipping byte. Return longest mainLen and mainDist from
  * the first ReadMatchDistances call */
  p->backRes = mainDist + LZMA_NUM_REPS;
  if (mainLen != 2)
  {
    MOVE_POS(p, mainLen - 2)
  }
  return mainLen;
}




static void WriteEndMarker(CLzmaEnc *p, unsigned posState)
{
  UInt32 range;
  range = p->rc.range;
  {
    UInt32 ttt, newBound;
    CLzmaProb *prob = &p->isMatch[p->state][posState];
    RC_BIT_PRE(&p->rc, prob)
    RC_BIT_1(&p->rc, prob)
    prob = &p->isRep[p->state];
    RC_BIT_PRE(&p->rc, prob)
    RC_BIT_0(&p->rc, prob)
  }
  p->state = kMatchNextStates[p->state];
  
  p->rc.range = range;
  LenEnc_Encode(&p->lenProbs, &p->rc, 0, posState);
  range = p->rc.range;

  {
    // RcTree_Encode_PosSlot(&p->rc, p->posSlotEncoder[0], (1 << kNumPosSlotBits) - 1);
    CLzmaProb *probs = p->posSlotEncoder[0];
    unsigned m = 1;
    do
    {
      UInt32 ttt, newBound;
      RC_BIT_PRE(p, probs + m)
      RC_BIT_1(&p->rc, probs + m);
      m = (m << 1) + 1;
    }
    while (m < (1 << kNumPosSlotBits));
  }
  {
    // RangeEnc_EncodeDirectBits(&p->rc, ((UInt32)1 << (30 - kNumAlignBits)) - 1, 30 - kNumAlignBits);    UInt32 range = p->range;
    unsigned numBits = 30 - kNumAlignBits;
    do
    {
      range >>= 1;
      p->rc.low += range;
      RC_NORM(&p->rc)
    }
    while (--numBits);
  }
   
  {
    // RcTree_ReverseEncode(&p->rc, p->posAlignEncoder, kNumAlignBits, kAlignMask);
    CLzmaProb *probs = p->posAlignEncoder;
    unsigned m = 1;
    do
    {
      UInt32 ttt, newBound;
      RC_BIT_PRE(p, probs + m)
      RC_BIT_1(&p->rc, probs + m);
      m = (m << 1) + 1;
    }
    while (m < kAlignTableSize);
  }
  p->rc.range = range;
}


static SRes CheckErrors(CLzmaEnc *p)
{
  if (p->result != SZ_OK)
    return p->result;
  if (p->rc.res != SZ_OK)
    p->result = SZ_ERROR_WRITE;
  if (MFB.result != SZ_OK)
    p->result = SZ_ERROR_READ;
  if (p->result != SZ_OK)
    p->finished = True;
  return p->result;
}


MY_NO_INLINE static SRes Flush(CLzmaEnc *p, UInt32 nowPos)
{
  /* ReleaseMFStream(); */
  p->finished = True;
  if (p->writeEndMark)
    WriteEndMarker(p, nowPos & p->pbMask);
  RangeEnc_FlushData(&p->rc);
  RangeEnc_FlushStream(&p->rc);
  return CheckErrors(p);
}


MY_NO_INLINE static void FillAlignPrices(CLzmaEnc *p)
{
  unsigned i;
  const CProbPrice *ProbPrices = p->ProbPrices;
  const CLzmaProb *probs = p->posAlignEncoder;
  // p->alignPriceCount = 0;
  for (i = 0; i < kAlignTableSize / 2; i++)
  {
    UInt32 price = 0;
    unsigned sym = i;
    unsigned m = 1;
    unsigned bit;
    UInt32 prob;
    bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[m], bit); m = (m << 1) + bit;
    bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[m], bit); m = (m << 1) + bit;
    bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[m], bit); m = (m << 1) + bit;
    prob = probs[m];
    p->alignPrices[i    ] = price + GET_PRICEa_0(prob);
    p->alignPrices[i + 8] = price + GET_PRICEa_1(prob);
    // p->alignPrices[i] = RcTree_ReverseGetPrice(p->posAlignEncoder, kNumAlignBits, i, p->ProbPrices);
  }
}


MY_NO_INLINE static void FillDistancesPrices(CLzmaEnc *p)
{
  // int y; for (y = 0; y < 100; y++) {

  UInt32 tempPrices[kNumFullDistances];
  unsigned i, lps;

  const CProbPrice *ProbPrices = p->ProbPrices;
  p->matchPriceCount = 0;

  for (i = kStartPosModelIndex / 2; i < kNumFullDistances / 2; i++)
  {
    unsigned posSlot = GetPosSlot1(i);
    unsigned footerBits = (posSlot >> 1) - 1;
    unsigned base = ((2 | (posSlot & 1)) << footerBits);
    const CLzmaProb *probs = p->posEncoders + (size_t)base * 2;
    // tempPrices[i] = RcTree_ReverseGetPrice(p->posEncoders + base, footerBits, i - base, p->ProbPrices);
    UInt32 price = 0;
    unsigned m = 1;
    unsigned sym = i;
    unsigned offset = (unsigned)1 << footerBits;
    base += i;
    
    if (footerBits)
    do
    {
      unsigned bit = sym & 1;
      sym >>= 1;
      price += GET_PRICEa(probs[m], bit);
      m = (m << 1) + bit;
    }
    while (--footerBits);

    {
      unsigned prob = probs[m];
      tempPrices[base         ] = price + GET_PRICEa_0(prob);
      tempPrices[base + offset] = price + GET_PRICEa_1(prob);
    }
  }

  for (lps = 0; lps < kNumLenToPosStates; lps++)
  {
    unsigned slot;
    unsigned distTableSize2 = (p->distTableSize + 1) >> 1;
    UInt32 *posSlotPrices = p->posSlotPrices[lps];
    const CLzmaProb *probs = p->posSlotEncoder[lps];
    
    for (slot = 0; slot < distTableSize2; slot++)
    {
      // posSlotPrices[slot] = RcTree_GetPrice(encoder, kNumPosSlotBits, slot, p->ProbPrices);
      UInt32 price;
      unsigned bit;
      unsigned sym = slot + (1 << (kNumPosSlotBits - 1));
      unsigned prob;
      bit = sym & 1; sym >>= 1; price  = GET_PRICEa(probs[sym], bit);
      bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[sym], bit);
      bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[sym], bit);
      bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[sym], bit);
      bit = sym & 1; sym >>= 1; price += GET_PRICEa(probs[sym], bit);
      prob = probs[(size_t)slot + (1 << (kNumPosSlotBits - 1))];
      posSlotPrices[(size_t)slot * 2    ] = price + GET_PRICEa_0(prob);
      posSlotPrices[(size_t)slot * 2 + 1] = price + GET_PRICEa_1(prob);
    }
    
    {
      UInt32 delta = ((UInt32)((kEndPosModelIndex / 2 - 1) - kNumAlignBits) << kNumBitPriceShiftBits);
      for (slot = kEndPosModelIndex / 2; slot < distTableSize2; slot++)
      {
        posSlotPrices[(size_t)slot * 2    ] += delta;
        posSlotPrices[(size_t)slot * 2 + 1] += delta;
        delta += ((UInt32)1 << kNumBitPriceShiftBits);
      }
    }

    {
      UInt32 *dp = p->distancesPrices[lps];
      
      dp[0] = posSlotPrices[0];
      dp[1] = posSlotPrices[1];
      dp[2] = posSlotPrices[2];
      dp[3] = posSlotPrices[3];

      for (i = 4; i < kNumFullDistances; i += 2)
      {
        UInt32 slotPrice = posSlotPrices[GetPosSlot1(i)];
        dp[i    ] = slotPrice + tempPrices[i];
        dp[i + 1] = slotPrice + tempPrices[i + 1];
      }
    }
  }
  // }
}



static void LzmaEnc_Construct(CLzmaEnc *p)
{
  RangeEnc_Construct(&p->rc);
  MatchFinder_Construct(&MFB);
  
  #ifndef _7ZIP_ST
  p->matchFinderMt.MatchFinder = &MFB;
  MatchFinderMt_Construct(&p->matchFinderMt);
  #endif

  {
    CLzmaEncProps props;
    LzmaEncProps_Init(&props);
#ifdef AOCL_LZMA_OPT
    LzmaEnc_SetProps_fp(p, &props);
#else
    LzmaEnc_SetProps(p, &props);
#endif
  }

  #ifndef LZMA_LOG_BSR
  LzmaEnc_FastPosInit(p->g_FastPos);
  #endif

  LzmaEnc_InitPriceTables(p->ProbPrices);
  p->litProbs = NULL;
  p->saveState.litProbs = NULL;
}

CLzmaEncHandle LzmaEnc_Create(ISzAllocPtr alloc)
{
  AOCL_SETUP_NATIVE();
  void *p = NULL;
  if (alloc == NULL) return p;
  p = ISzAlloc_Alloc(alloc, sizeof(CLzmaEnc));
  if (p)
    LzmaEnc_Construct((CLzmaEnc *)p);
  return p;
}

static void LzmaEnc_FreeLits(CLzmaEnc *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->litProbs);
  ISzAlloc_Free(alloc, p->saveState.litProbs);
  p->litProbs = NULL;
  p->saveState.litProbs = NULL;
}

static void LzmaEnc_Destruct(CLzmaEnc *p, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  #ifndef _7ZIP_ST
  MatchFinderMt_Destruct(&p->matchFinderMt, allocBig);
  #endif
  
#ifdef AOCL_LZMA_OPT
  MatchFinder_Free_fp(&MFB, allocBig);
#else
  MatchFinder_Free(&MFB, allocBig);
#endif
  LzmaEnc_FreeLits(p, alloc);
  RangeEnc_Free(&p->rc, alloc);
}

void LzmaEnc_Destroy(CLzmaEncHandle p, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  AOCL_SETUP_NATIVE();
  /* Guard against NULL so a destroy on an unset/NULL handle is a no-op
   * instead of dereferencing/freeing an invalid pointer (CPUPL-8846). This
   * only covers the NULL case; it cannot detect an already-freed (dangling,
   * non-NULL) handle. Callers must still not reuse or re-destroy a handle
   * after LzmaEnc_Destroy(): the handle is invalid once destroyed. */
  if (p == NULL)
  {
    /* Documented no-op path (see LzmaEnc.h): keep it quiet at TRACE rather
     * than ERR so a valid NULL call does not emit spurious error logs. */
    LOG_UNFORMATTED(TRACE, logCtx, "LzmaEnc_Destroy called with NULL handle; no-op");
    return;
  }
  LzmaEnc_Destruct((CLzmaEnc *)p, alloc, allocBig);
  ISzAlloc_Free(alloc, p);
}


MY_NO_INLINE
static SRes LzmaEnc_CodeOneBlock(CLzmaEnc *p, UInt32 maxPackSize, UInt32 maxUnpackSize)
{
  UInt32 nowPos32, startPos32;
  if (p->needInit)
  {
    p->matchFinder.Init(p->matchFinderObj);
    p->needInit = 0;
  }

  if (p->finished)
    return p->result;
  RINOK(CheckErrors(p));

  nowPos32 = (UInt32)p->nowPos64;
  startPos32 = nowPos32;

  if (p->nowPos64 == 0) // first byte in stream
  {
    unsigned numPairs;
    Byte curByte;
    if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
      return Flush(p, nowPos32);
    ReadMatchDistances(p, &numPairs);
    RangeEnc_EncodeBit_0(&p->rc, &p->isMatch[kState_Start][0]);
    // p->state = kLiteralNextStates[p->state];
    curByte = *(p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset);
    LitEnc_Encode(&p->rc, p->litProbs, curByte);
    p->additionalOffset--;
    nowPos32++;
  }

  if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) != 0)
  
  for (;;)
  {
    UInt32 dist;
    unsigned len, posState;
    UInt32 range, ttt, newBound;
    CLzmaProb *probs;
  
    if (p->fastMode) // hash-chain algo
      len = GetOptimumFast(p);
    else // bin-tree algo
    {
      unsigned oci = p->optCur;
      if (p->optEnd == oci)
#ifdef AOCL_LZMA_OPT
        len = GetOptimum_fp(p, nowPos32);
#else
        len = GetOptimum(p, nowPos32);
#endif
      else
      {
        const COptimal *opt = &p->opt[oci];
        len = opt->len;
        p->backRes = opt->dist;
        p->optCur = oci + 1;
      }
    }

    posState = (unsigned)nowPos32 & p->pbMask;
    range = p->rc.range;
    probs = &p->isMatch[p->state][posState];
    
    RC_BIT_PRE(&p->rc, probs)
    
    dist = p->backRes;

    #ifdef SHOW_STAT2
    printf("\n pos = %6X, len = %3u  pos = %6u", nowPos32, len, dist);
    #endif

    /* Depending on the type of data to be saved: literal, srep, 
    * rep or match, corresponding code is encoded. This is followed by 
    * optionally encoding len and/or distance values */
    if (dist == MARK_LIT) // literal
    {
      Byte curByte;
      const Byte *data;
      unsigned state;

      RC_BIT_0(&p->rc, probs); // code:0+*
      p->rc.range = range;
      data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset;
      probs = LIT_PROBS(nowPos32, *(data - 1)); // get appropriate context model for literal encoding 
      curByte = *data;
      state = p->state;
      p->state = kLiteralNextStates[state];
      if (IsLitState(state))
        LitEnc_Encode(&p->rc, probs, curByte);
      else
        LitEnc_EncodeMatched(&p->rc, probs, curByte, *(data - p->reps[0]));
    }
    else // rep, srep or match
    {
      RC_BIT_1(&p->rc, probs); // code:1+*
      probs = &p->isRep[p->state];
      RC_BIT_PRE(&p->rc, probs)
      
      if (dist < LZMA_NUM_REPS) // rep or srep
      {
        RC_BIT_1(&p->rc, probs); // code:11+*
        probs = &p->isRepG0[p->state];
        RC_BIT_PRE(&p->rc, probs)
        if (dist == 0) // srep or rep0
        {
          RC_BIT_0(&p->rc, probs); // code:110+*
          probs = &p->isRep0Long[p->state][posState];
          RC_BIT_PRE(&p->rc, probs)
          if (len != 1)
          {
            RC_BIT_1_BASE(&p->rc, probs); //code:1101
          }
          else // srep
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1100
            p->state = kShortRepNextStates[p->state];
          }
        }
        else // rep1-3
        {
          RC_BIT_1(&p->rc, probs); // code:111+*
          probs = &p->isRepG1[p->state];
          RC_BIT_PRE(&p->rc, probs)
          if (dist == 1) // rep1
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1110
            dist = p->reps[1];
          }
          else // rep2-3
          {
            RC_BIT_1(&p->rc, probs); // code:1111+*
            probs = &p->isRepG2[p->state];
            RC_BIT_PRE(&p->rc, probs)
            if (dist == 2) // rep2
            {
              RC_BIT_0_BASE(&p->rc, probs); // code:11110
              dist = p->reps[2];
            }
            else // rep3
            {
              RC_BIT_1_BASE(&p->rc, probs); // code:11111
              dist = p->reps[3];
              p->reps[3] = p->reps[2]; // move reps queue dist->0->1->2->3
            }
            p->reps[2] = p->reps[1]; // move reps queue dist->0->1->2->3
          }
          p->reps[1] = p->reps[0]; // move reps queue dist->0->1->2->3
          p->reps[0] = dist; // move reps queue dist->0->1->2->3
        }

        RC_NORM(&p->rc)

        p->rc.range = range;

        if (len != 1) // code saved, now encode len
        {
          LenEnc_Encode(&p->repLenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState);
          --p->repLenEncCounter;
          p->state = kRepNextStates[p->state];
        }
      }
      else // match
      {
        unsigned posSlot;
        RC_BIT_0(&p->rc, probs);
        p->rc.range = range;
        p->state = kMatchNextStates[p->state];

        LenEnc_Encode(&p->lenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState); // encode len
        // --p->lenEnc.counter;

        dist -= LZMA_NUM_REPS;
        p->reps[3] = p->reps[2];
        p->reps[2] = p->reps[1];
        p->reps[1] = p->reps[0];
        p->reps[0] = dist + 1;
        
        p->matchPriceCount++;
        GetPosSlot(dist, posSlot);
        // RcTree_Encode_PosSlot(&p->rc, p->posSlotEncoder[GetLenToPosState(len)], posSlot);
        {
          UInt32 sym = (UInt32)posSlot + (1 << kNumPosSlotBits);
          range = p->rc.range;
          probs = p->posSlotEncoder[GetLenToPosState(len)];
          do // Range code 'slot' bitwise
          {
            CLzmaProb *prob = probs + (sym >> kNumPosSlotBits);
            UInt32 bit = (sym >> (kNumPosSlotBits - 1)) & 1;
            sym <<= 1;
            RC_BIT(&p->rc, prob, bit);
          }
          while (sym < (1 << kNumPosSlotBits * 2));
          p->rc.range = range;
        }
        
        if (dist >= kStartPosModelIndex) // dist > 3 (dist 0:3 no direct_bits)
        {
          unsigned footerBits = ((posSlot >> 1) - 1);

          if (dist < kNumFullDistances) // dis: 4-127, 
          {
            unsigned base = ((2 | (posSlot & 1)) << footerBits);
            RcTree_ReverseEncode(&p->rc, p->posEncoders + base, footerBits, (unsigned)(dist /* - base */));
          }
          else // dist >= 128
          {
            UInt32 pos2 = (dist | 0xF) << (32 - footerBits); // direct_bits-4 part
            range = p->rc.range;
            // RangeEnc_EncodeDirectBits(&p->rc, posReduced >> kNumAlignBits, footerBits - kNumAlignBits);
            /*
            do
            {
              range >>= 1;
              p->rc.low += range & (0 - ((dist >> --footerBits) & 1));
              RC_NORM(&p->rc)
            }
            while (footerBits > kNumAlignBits);
            */
            do // fixed 0.5 prob model
            {
              range >>= 1;
              p->rc.low += range & (0 - (pos2 >> 31));
              pos2 += pos2;
              RC_NORM(&p->rc)
            }
            while (pos2 != 0xF0000000);


            // RcTree_ReverseEncode(&p->rc, p->posAlignEncoder, kNumAlignBits, posReduced & kAlignMask);

            { // remaining 4 align bits LSB to MSB
              unsigned m = 1;
              unsigned bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1;             RC_BIT(&p->rc, p->posAlignEncoder + m, bit);
              p->rc.range = range;
              // p->alignPriceCount++;
            }
          }
        }
      }
    }

    nowPos32 += (UInt32)len;
    p->additionalOffset -= len;
    
    if (p->additionalOffset == 0)
    {
      UInt32 processed;

      if (!p->fastMode)
      {
        /*
        if (p->alignPriceCount >= 16) // kAlignTableSize
          FillAlignPrices(p);
        if (p->matchPriceCount >= 128)
          FillDistancesPrices(p);
        if (p->lenEnc.counter <= 0)
          LenPriceEnc_UpdateTables(&p->lenEnc, 1 << p->pb, &p->lenProbs, p->ProbPrices);
        */
        if (p->matchPriceCount >= 64)
        {
          FillAlignPrices(p);
          // { int y; for (y = 0; y < 100; y++) {
          FillDistancesPrices(p);
          // }}
          LenPriceEnc_UpdateTables(&p->lenEnc, (unsigned)1 << p->pb, &p->lenProbs, p->ProbPrices);
        }
        if (p->repLenEncCounter <= 0)
        {
          p->repLenEncCounter = REP_LEN_COUNT;
          LenPriceEnc_UpdateTables(&p->repLenEnc, (unsigned)1 << p->pb, &p->repLenProbs, p->ProbPrices);
        }
      }
    
      if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
        break;
      processed = nowPos32 - startPos32;
      
      if (maxPackSize)
      {
        if (processed + kNumOpts + 300 >= maxUnpackSize
            || RangeEnc_GetProcessed_sizet(&p->rc) + kPackReserve >= maxPackSize)
          break;
      }
      else if (processed >= (1 << 17))
      {
        p->nowPos64 += nowPos32 - startPos32;
        return CheckErrors(p);
      }
    }
  }

  p->nowPos64 += nowPos32 - startPos32;
  return Flush(p, nowPos32);
}



#define kBigHashDicLimit ((UInt32)1 << 24)

static SRes LzmaEnc_Alloc(CLzmaEnc *p, UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  UInt32 beforeSize = kNumOpts;
  UInt32 dictSize;

  if (!RangeEnc_Alloc(&p->rc, alloc))
    return SZ_ERROR_MEM;

  #ifndef _7ZIP_ST
  p->mtMode = (p->multiThread && !p->fastMode && (MFB.btMode != 0));
  #endif

  {
    unsigned lclp = p->lc + p->lp;
    if (!p->litProbs || !p->saveState.litProbs || p->lclp != lclp)
    {
      LzmaEnc_FreeLits(p, alloc);
      p->litProbs = (CLzmaProb *)ISzAlloc_Alloc(alloc, ((UInt32)0x300 << lclp) * sizeof(CLzmaProb));
      p->saveState.litProbs = (CLzmaProb *)ISzAlloc_Alloc(alloc, ((UInt32)0x300 << lclp) * sizeof(CLzmaProb));
      if (!p->litProbs || !p->saveState.litProbs)
      {
        LzmaEnc_FreeLits(p, alloc);
        return SZ_ERROR_MEM;
      }
      p->lclp = lclp;
    }
  }

  MFB.bigHash = (Byte)(p->dictSize > kBigHashDicLimit ? 1 : 0);


  dictSize = p->dictSize;
  if (dictSize == ((UInt32)2 << 30) ||
      dictSize == ((UInt32)3 << 30))
  {
    /* 21.03 : here we reduce the dictionary for 2 reasons:
       1) we don't want 32-bit back_distance matches in decoder for 2 GB dictionary.
       2) we want to elimate useless last MatchFinder_Normalize3() for corner cases,
          where data size is aligned for 1 GB: 5/6/8 GB.
          That reducing must be >= 1 for such corner cases. */
    dictSize -= 1;
  }

  if (beforeSize + dictSize < keepWindowSize)
    beforeSize = keepWindowSize - dictSize;

  /* in worst case we can look ahead for
        max(LZMA_MATCH_LEN_MAX, numFastBytes + 1 + numFastBytes) bytes.
     we send larger value for (keepAfter) to MantchFinder_Create():
        (numFastBytes + LZMA_MATCH_LEN_MAX + 1)
  */

  #ifndef _7ZIP_ST
  if (p->mtMode)
  {
    RINOK(MatchFinderMt_Create(&p->matchFinderMt, dictSize, beforeSize,
        p->numFastBytes, LZMA_MATCH_LEN_MAX + 1 /* 18.04 */
        , allocBig));
    p->matchFinderObj = &p->matchFinderMt;
    MFB.bigHash = (Byte)(
        (p->dictSize > kBigHashDicLimit && MFB.hashMask >= 0xFFFFFF) ? 1 : 0);
    MatchFinderMt_CreateVTable(&p->matchFinderMt, &p->matchFinder);
  }
  else
  #endif
  {
#ifdef AOCL_LZMA_OPT
    if (!MatchFinder_Create_fp(&MFB, dictSize, beforeSize,
        p->numFastBytes, LZMA_MATCH_LEN_MAX + 1 /* 21.03 */
        , allocBig))
#else
      if (!MatchFinder_Create(&MFB, dictSize, beforeSize,
          p->numFastBytes, LZMA_MATCH_LEN_MAX + 1 /* 21.03 */
          , allocBig))
#endif
      return SZ_ERROR_MEM;
    p->matchFinderObj = &MFB;
#ifdef AOCL_LZMA_OPT
    MatchFinder_CreateVTable_fp(&MFB, &p->matchFinder);
#else
    MatchFinder_CreateVTable(&MFB, &p->matchFinder);
#endif
  }
  
  return SZ_OK;
}

static void LzmaEnc_Init(CLzmaEnc *p)
{
  unsigned i;
  p->state = 0;
  p->reps[0] =
  p->reps[1] =
  p->reps[2] =
  p->reps[3] = 1;

  RangeEnc_Init(&p->rc);

  for (i = 0; i < (1 << kNumAlignBits); i++)
    p->posAlignEncoder[i] = kProbInitValue;

  for (i = 0; i < kNumStates; i++)
  {
    unsigned j;
    for (j = 0; j < LZMA_NUM_PB_STATES_MAX; j++)
    {
      p->isMatch[i][j] = kProbInitValue;
      p->isRep0Long[i][j] = kProbInitValue;
    }
    p->isRep[i] = kProbInitValue;
    p->isRepG0[i] = kProbInitValue;
    p->isRepG1[i] = kProbInitValue;
    p->isRepG2[i] = kProbInitValue;
  }

  {
    for (i = 0; i < kNumLenToPosStates; i++)
    {
      CLzmaProb *probs = p->posSlotEncoder[i];
      unsigned j;
      for (j = 0; j < (1 << kNumPosSlotBits); j++)
        probs[j] = kProbInitValue;
    }
  }
  {
    for (i = 0; i < kNumFullDistances; i++)
      p->posEncoders[i] = kProbInitValue;
  }

  {
    UInt32 num = (UInt32)0x300 << (p->lp + p->lc);
    UInt32 k;
    CLzmaProb *probs = p->litProbs;
    for (k = 0; k < num; k++)
      probs[k] = kProbInitValue;
  }


  LenEnc_Init(&p->lenProbs);
  LenEnc_Init(&p->repLenProbs);

  p->optEnd = 0;
  p->optCur = 0;

  {
    for (i = 0; i < kNumOpts; i++)
      p->opt[i].price = kInfinityPrice;
  }

  p->additionalOffset = 0;

  p->pbMask = ((unsigned)1 << p->pb) - 1;
  p->lpMask = ((UInt32)0x100 << p->lp) - ((unsigned)0x100 >> p->lc);

  // p->mf_Failure = False;
}


static void LzmaEnc_InitPrices(CLzmaEnc *p)
{
  if (!p->fastMode)
  {
    FillDistancesPrices(p);
    FillAlignPrices(p);
  }

  p->lenEnc.tableSize =
  p->repLenEnc.tableSize =
      p->numFastBytes + 1 - LZMA_MATCH_LEN_MIN;

  p->repLenEncCounter = REP_LEN_COUNT;

  LenPriceEnc_UpdateTables(&p->lenEnc, (unsigned)1 << p->pb, &p->lenProbs, p->ProbPrices);
  LenPriceEnc_UpdateTables(&p->repLenEnc, (unsigned)1 << p->pb, &p->repLenProbs, p->ProbPrices);
}

static SRes LzmaEnc_AllocAndInit(CLzmaEnc *p, UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  unsigned i;
  for (i = kEndPosModelIndex / 2; i < kDicLogSizeMax; i++)
    if (p->dictSize <= ((UInt32)1 << i))
      break;
  p->distTableSize = i * 2;

  p->finished = False;
  p->result = SZ_OK;
  RINOK(LzmaEnc_Alloc(p, keepWindowSize, alloc, allocBig));
  LzmaEnc_Init(p);
  LzmaEnc_InitPrices(p);
  p->nowPos64 = 0;
  return SZ_OK;
}

static SRes LzmaEnc_Prepare(CLzmaEncHandle pp, ISeqOutStream *outStream, ISeqInStream *inStream,
    ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  MFB.stream = inStream;
  p->needInit = 1;
  p->rc.outStream = outStream;
  return LzmaEnc_AllocAndInit(p, 0, alloc, allocBig);
}

SRes LzmaEnc_PrepareForLzma2(CLzmaEncHandle pp,
    ISeqInStream *inStream, UInt32 keepWindowSize,
    ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  MFB.stream = inStream;
  p->needInit = 1;
  return LzmaEnc_AllocAndInit(p, keepWindowSize, alloc, allocBig);
}

static void LzmaEnc_SetInputBuf(CLzmaEnc *p, const Byte *src, SizeT srcLen)
{
  MFB.directInput = 1;
  MFB.bufferBase = (Byte *)src;
  MFB.directInputRem = srcLen;
}

SRes LzmaEnc_MemPrepare(CLzmaEncHandle pp, const Byte *src, SizeT srcLen,
    UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  LzmaEnc_SetInputBuf(p, src, srcLen);
  p->needInit = 1;

  LzmaEnc_SetDataSize(pp, srcLen);
  return LzmaEnc_AllocAndInit(p, keepWindowSize, alloc, allocBig);
}

void LzmaEnc_Finish(CLzmaEncHandle pp)
{
  #ifndef _7ZIP_ST
  CLzmaEnc *p = (CLzmaEnc *)pp;
  if (p->mtMode)
    MatchFinderMt_ReleaseStream(&p->matchFinderMt);
  #else
  UNUSED_VAR(pp);
  #endif
}


typedef struct
{
  ISeqOutStream vt;
  Byte *data;
  SizeT rem;
  BoolInt overflow;
} CLzmaEnc_SeqOutStreamBuf;

static size_t SeqOutStreamBuf_Write(const ISeqOutStream *pp, const void *data, size_t size)
{
  CLzmaEnc_SeqOutStreamBuf *p = CONTAINER_FROM_VTBL(pp, CLzmaEnc_SeqOutStreamBuf, vt);
  if (p->rem < size)
  {
    size = p->rem;
    p->overflow = True;
  }
  if (size != 0)
  {
    memcpy(p->data, data, size);
    p->rem -= size;
    p->data += size;
  }
  return size;
}


/*
UInt32 LzmaEnc_GetNumAvailableBytes(CLzmaEncHandle pp)
{
  const CLzmaEnc *p = (CLzmaEnc *)pp;
  return p->matchFinder.GetNumAvailableBytes(p->matchFinderObj);
}
*/

const Byte *LzmaEnc_GetCurBuf(CLzmaEncHandle pp)
{
  if (pp == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "LzmaEnc_GetCurBuf: Invalid CLzmaEncHandle");
    return NULL;
  }
  const CLzmaEnc *p = (CLzmaEnc *)pp;
  return p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset;
}


// (desiredPackSize == 0) is not allowed
SRes LzmaEnc_CodeOneMemBlock(CLzmaEncHandle pp, BoolInt reInit,
    Byte *dest, size_t *destLen, UInt32 desiredPackSize, UInt32 *unpackSize)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  UInt64 nowPos64;
  SRes res;
  CLzmaEnc_SeqOutStreamBuf outStream;

  outStream.vt.Write = SeqOutStreamBuf_Write;
  outStream.data = dest;
  outStream.rem = *destLen;
  outStream.overflow = False;

  p->writeEndMark = False;
  p->finished = False;
  p->result = SZ_OK;

  if (reInit)
    LzmaEnc_Init(p);
  LzmaEnc_InitPrices(p);
  RangeEnc_Init(&p->rc);
  p->rc.outStream = &outStream.vt;
  nowPos64 = p->nowPos64;
  
  res = LzmaEnc_CodeOneBlock(p, desiredPackSize, *unpackSize);
  
  *unpackSize = (UInt32)(p->nowPos64 - nowPos64);
  *destLen -= outStream.rem;
  if (outStream.overflow) {
      LOG_UNFORMATTED(ERR, logCtx, "Out stream overflow");
      return SZ_ERROR_OUTPUT_EOF;
  }

  return res;
}


MY_NO_INLINE
static SRes LzmaEnc_Encode2(CLzmaEnc *p, ICompressProgress *progress)
{
  SRes res = SZ_OK;

  #ifndef _7ZIP_ST
  Byte allocaDummy[0x300];
  allocaDummy[0] = 0;
  allocaDummy[1] = allocaDummy[0];
  #endif

  for (;;)
  {
    res = LzmaEnc_CodeOneBlock(p, 0, 0);
    if (res != SZ_OK || p->finished)
      break;
    if (progress)
    {
      res = ICompressProgress_Progress(progress, p->nowPos64, RangeEnc_GetProcessed(&p->rc));
      if (res != SZ_OK)
      {
        res = SZ_ERROR_PROGRESS;
        break;
      }
    }
  }
  
  LzmaEnc_Finish(p);

  /*
  if (res == SZ_OK && !Inline_MatchFinder_IsFinishedOK(&MFB))
    res = SZ_ERROR_FAIL;
  }
  */

  return res;
}


SRes LzmaEnc_Encode(CLzmaEncHandle pp, ISeqOutStream *outStream, ISeqInStream *inStream, ICompressProgress *progress,
    ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  if(outStream == NULL || inStream == NULL || alloc == NULL || allocBig == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
    return SZ_ERROR_PARAM;
  }

  AOCL_SETUP_NATIVE();
  if(outStream == NULL || inStream == NULL || alloc == NULL || allocBig == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
    return SZ_ERROR_PARAM;
  }

  RINOK(LzmaEnc_Prepare(pp, outStream, inStream, alloc, allocBig));
  return LzmaEnc_Encode2((CLzmaEnc *)pp, progress);
}


SRes LzmaEnc_WriteProperties(CLzmaEncHandle pp, Byte *props, SizeT *size)
{
    if (pp == NULL || props == NULL || size == NULL || *size < LZMA_PROPS_SIZE) {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
        return SZ_ERROR_PARAM;
    }
  *size = LZMA_PROPS_SIZE;
  {
    const CLzmaEnc *p = (const CLzmaEnc *)pp;
    const UInt32 dictSize = p->dictSize;
    UInt32 v;
    props[0] = (Byte)((p->pb * 5 + p->lp) * 9 + p->lc);
    
    // we write aligned dictionary value to properties for lzma decoder
    if (dictSize >= ((UInt32)1 << 21))
    {
      const UInt32 kDictMask = ((UInt32)1 << 20) - 1;
      v = (dictSize + kDictMask) & ~kDictMask;
      if (v < dictSize)
        v = dictSize;
    }
    else
    {
      unsigned i = 11 * 2;
      do
      {
        v = (UInt32)(2 + (i & 1)) << (i >> 1);
        i++;
      }
      while (v < dictSize);
    }

    SetUi32(props + 1, v);
    return SZ_OK;
  }
}


unsigned LzmaEnc_IsWriteEndMark(CLzmaEncHandle pp)
{
  return (unsigned)((CLzmaEnc *)pp)->writeEndMark;
}


SRes LzmaEnc_MemEncode(CLzmaEncHandle pp, Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    int writeEndMark, ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  AOCL_SETUP_NATIVE();
  if (pp == NULL || src == NULL || srcLen == 0 || dest == NULL || destLen == NULL || alloc == NULL || allocBig == NULL) {
      LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
      return SZ_ERROR_PARAM;
  }

  SRes res;
  CLzmaEnc *p = (CLzmaEnc *)pp;

  CLzmaEnc_SeqOutStreamBuf outStream;

  outStream.vt.Write = SeqOutStreamBuf_Write;
  outStream.data = dest;
  outStream.rem = *destLen;
  outStream.overflow = False;

  p->writeEndMark = writeEndMark;
  p->rc.outStream = &outStream.vt;

  res = LzmaEnc_MemPrepare(pp, src, srcLen, 0, alloc, allocBig);
  
  if (res == SZ_OK)
  {
    res = LzmaEnc_Encode2(p, progress);
    if (res == SZ_OK && p->nowPos64 != srcLen) {
        LOG_UNFORMATTED(ERR, logCtx, "Not all src bytes processed");
        res = SZ_ERROR_FAIL;
    }
  }

  *destLen -= outStream.rem;
  if (outStream.overflow) {
      LOG_UNFORMATTED(ERR, logCtx, "Out stream overflow");
      return SZ_ERROR_OUTPUT_EOF;
  }
  return res;
}

/* Ensure user settings passed are within range.
* Range values are regulated based on ranges indicated in LzmaEnc.h */
SRes ValidateParams(const CLzmaEncProps* props) {
    //If <0 uses default
    if (props->level > 9 ||
        props->lc > 8 ||
        props->lp > 4 ||
        props->pb > 4 ||
        props->algo > 1 ||
        ((props->fb >= 0) && (props->fb < 5 || props->fb > 273)) ||
        props->btMode > 1 ||
        props->numHashBytes > 5 ||
        props->mc > ((UInt32)1 << 30) ||
        props->writeEndMark > 1) {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid props");
        return SZ_ERROR_PARAM;
    }
    else
        return SZ_OK;
}

//Minimum compressed buffer size
#define MIN_PAD_SIZE (16*1024)
size_t Lzma_compressBound(size_t insize)
{
    size_t outSize = (insize + (insize / 6) + MIN_PAD_SIZE);
    return outSize;
}

SRes LzmaEncode_ST(Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark,
    ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
  if (src == NULL || srcLen == 0 || dest == NULL || propsEncoded == NULL ||
      props == NULL || propsSize == NULL || destLen == NULL ||
      *destLen > (ULLONG_MAX - LZMA_PROPS_SIZE)) // handles case when dest size is < LZMA_PROPS_SIZE, resulting in destLen rolling over in calling APIs
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return SZ_ERROR_PARAM;
  }

  if (ValidateParams(props) != SZ_OK)
  {
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return SZ_ERROR_PARAM;
  }

  CLzmaEnc *p = (CLzmaEnc *)LzmaEnc_Create(alloc);
  SRes res;
  if (!p)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Context creation failed");
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return SZ_ERROR_MEM;
  }

#ifdef AOCL_LZMA_OPT
  CLzmaEncProps props_cur = *props;
  props_cur.srcLen = srcLen; //same srcLen value must be set here and passed to LzmaEnc_MemEncode()
  res = LzmaEnc_SetProps_fp(p, &props_cur);
#else
  res = LzmaEnc_SetProps(p, props);
#endif

  if (res == SZ_OK)
  {
    res = LzmaEnc_WriteProperties(p, propsEncoded, propsSize);
    if (res == SZ_OK)
      res = LzmaEnc_MemEncode(p, dest, destLen, src, srcLen,
          writeEndMark, progress, alloc, allocBig);
  }

  LzmaEnc_Destroy(p, alloc, allocBig);
  
  AOCL_LOG_API_SUMMARY(props->level, srcLen, *destLen);
  LOG_UNFORMATTED(TRACE, logCtx, "Exit");
  return res;
}

#ifdef AOCL_ENABLE_THREADS
static SRes (*LzmaEncode_mt_fp) (Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark,
    ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig) = LzmaEncode_ST;
#endif

SRes LzmaEncode(Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark,
    ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
    SRes res;
#ifdef AOCL_ENABLE_THREADS
    res = LzmaEncode_mt_fp(dest, destLen, src, srcLen, props, propsEncoded, propsSize, writeEndMark, progress, alloc, allocBig);
#else
    res = LzmaEncode_ST(dest, destLen, src, srcLen, props, propsEncoded, propsSize, writeEndMark, progress, alloc, allocBig);
#endif /* AOCL_ENABLE_THREADS */
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return res;
}

#ifdef AOCL_ENABLE_THREADS

/**************************************
 * Helper functions for MT - Start
 *************************************/
#define LZMA_GET_WINDOW_FACTOR 1

typedef struct SequenceEntry {
  unsigned match_len;          // length of match (or 1 if no match)
  unsigned dist;               // distance (MARK_LIT if no match, else it represents the offset of match)
  unsigned additionalOffset;   // 
  struct SequenceEntry *next;
} SequenceEntry;

int addEntry(SequenceEntry **head, SequenceEntry **tail, unsigned len, unsigned dist, unsigned additionalOffset)
{
  SequenceEntry *entry = (SequenceEntry *)malloc(sizeof(SequenceEntry));
  if (!entry){
    return -1; // memory allocation failed
  }
  entry->match_len = len;
  entry->dist = dist;
  entry->additionalOffset=additionalOffset;
  entry->next = NULL;

  if (*head == NULL)
    *head = *tail = entry;
  else
  {
    (*tail)->next = entry;
    *tail = entry;
  }
  return 0;
}

void freeSequenceEntry(SequenceEntry *head)
{
  SequenceEntry *current = head;
  while (current != NULL)
  {
    SequenceEntry *next = current->next;
    free(current);
    current = next;
  } 
}

/**************************************
 * Helper functions for MT - End
 *************************************/

/**************************************
 * Init functions for MT - Start
 *************************************/
/**
 * LzmaEnc_Init_mt(): Same as LzmaEnc_Init(), but for MT mode.
 * Initializes p->reps[i] (i = 0..3) to -1 instead of 1.
 */
static void LzmaEnc_Init_mt(CLzmaEnc *p)
{
  unsigned i;
  p->state = 0;
  p->reps[0] =
  p->reps[1] =
  p->reps[2] =
  p->reps[3] = -1; // Modified for MT implementation

  RangeEnc_Init(&p->rc);

  for (i = 0; i < (1 << kNumAlignBits); i++)
    p->posAlignEncoder[i] = kProbInitValue;

  for (i = 0; i < kNumStates; i++)
  {
    unsigned j;
    for (j = 0; j < LZMA_NUM_PB_STATES_MAX; j++)
    {
      p->isMatch[i][j] = kProbInitValue;
      p->isRep0Long[i][j] = kProbInitValue;
    }
    p->isRep[i] = kProbInitValue;
    p->isRepG0[i] = kProbInitValue;
    p->isRepG1[i] = kProbInitValue;
    p->isRepG2[i] = kProbInitValue;
  }

  {
    for (i = 0; i < kNumLenToPosStates; i++)
    {
      CLzmaProb *probs = p->posSlotEncoder[i];
      unsigned j;
      for (j = 0; j < (1 << kNumPosSlotBits); j++)
        probs[j] = kProbInitValue;
    }
  }
  {
    for (i = 0; i < kNumFullDistances; i++)
      p->posEncoders[i] = kProbInitValue;
  }

  {
    UInt32 num = (UInt32)0x300 << (p->lp + p->lc);
    UInt32 k;
    CLzmaProb *probs = p->litProbs;
    for (k = 0; k < num; k++)
      probs[k] = kProbInitValue;
  }


  LenEnc_Init(&p->lenProbs);
  LenEnc_Init(&p->repLenProbs);

  p->optEnd = 0;
  p->optCur = 0;

  {
    for (i = 0; i < kNumOpts; i++)
      p->opt[i].price = kInfinityPrice;
  }

  p->additionalOffset = 0;

  p->pbMask = ((unsigned)1 << p->pb) - 1;
  p->lpMask = ((UInt32)0x100 << p->lp) - ((unsigned)0x100 >> p->lc);

  // p->mf_Failure = False;
}

/**
 * LzmaEnc_AllocAndInit_mt(): Same as LzmaEnc_AllocAndInit(), but for MT mode.
 * Calls LzmaEnc_Init_mt() instead of LzmaEnc_Init().
 */
static SRes LzmaEnc_AllocAndInit_mt(CLzmaEnc *p, UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  unsigned i;
  for (i = kEndPosModelIndex / 2; i < kDicLogSizeMax; i++)
    if (p->dictSize <= ((UInt32)1 << i))
      break;
  p->distTableSize = i * 2;

  p->finished = False;
  p->result = SZ_OK;
  RINOK(LzmaEnc_Alloc(p, keepWindowSize, alloc, allocBig));
  LzmaEnc_Init_mt(p);
  LzmaEnc_InitPrices(p);
  p->nowPos64 = 0;
  return SZ_OK;
}

/**
 * LzmaEnc_MemPrepare_mt(): Same as LzmaEnc_MemPrepare(), but for MT mode.
 * Calls LzmaEnc_AllocAndInit_mt instead of LzmaEnc_AllocAndInit().
 */
SRes LzmaEnc_MemPrepare_mt(CLzmaEncHandle pp, const Byte *src, SizeT srcLen,
  UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CLzmaEnc *p = (CLzmaEnc *)pp;
  LzmaEnc_SetInputBuf(p, src, srcLen);
  p->needInit = 1;

  LzmaEnc_SetDataSize(pp, srcLen);
  return LzmaEnc_AllocAndInit_mt(p, keepWindowSize, alloc, allocBig);
}

/**************************************
 * Init functions for MT - End
 *************************************/
/**************************************
 * MT 1st Pass functions - Start
 *************************************/

/* Hack on optimal parsing strategy with some heuristics to provide
* faster matches albeit not optimal */ 
/**
 * AOCL_GetOptimumFast_mt():
 * Same as GetOptimumFast() but for multi-threaded implementation
 * where p->reps[i] is not checked if it is -1.
 */
static unsigned AOCL_GetOptimumFast_mt(CLzmaEnc *p)
{
  UInt32 numAvail, mainDist;
  unsigned mainLen, numPairs, repIndex, repLen, i;
  const Byte *data;

  /* if additionalOffset is > 0, parsing for cur byte has already been completed 
  * in one of the previous ReadMatchDistances() calls used to check matches for
  * future sequences. Results are already available in p-> and can be used 
  * without having to find matches again. */
  if (p->additionalOffset == 0)
    mainLen = ReadMatchDistances(p, &numPairs); // find matches in dictionary. Return longest matched length.
  else
  {
    mainLen = p->longestMatchLen;
    numPairs = p->numPairs;
  }

  numAvail = p->numAvail;
  p->backRes = MARK_LIT;
  if (numAvail < 2)
    return 1;
  // if (mainLen < 2 && p->state == 0) return 1; // 18.06.notused
  if (numAvail > LZMA_MATCH_LEN_MAX)
    numAvail = LZMA_MATCH_LEN_MAX; // max len allowed by lzma format
  data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
  repLen = repIndex = 0;
  
  /* Find longest match at distances in any of the reps for 'data'.
  * If long enough match is found at any of the reps, it can be used
  * instead of <mainLen, mainDist> as we save on encoding mainDist */
  for (i = 0; i < LZMA_NUM_REPS; i++)
  {
    unsigned len;
    if(p->reps[i] == -1) /* Modified for MT implementation */
      continue;
    const Byte *data2 = data - p->reps[i];
    if (data[0] != data2[0] || data[1] != data2[1])
      continue;
    for (len = 2; len < numAvail && data[len] == data2[len]; len++)
    {}
    if (len >= p->numFastBytes)
    {
      p->backRes = (UInt32)i;
      MOVE_POS(p, len - 1)
      return len;
    }
    if (len > repLen)
    {
      repIndex = i; // rep with longest match
      repLen = len; // longest match len
    }
  }

  if (mainLen >= p->numFastBytes) //if len long enough, just choose this
  {
    p->backRes = p->matches[(size_t)numPairs - 1] + LZMA_NUM_REPS;
    MOVE_POS(p, mainLen - 1)
    return mainLen; // choose mainLen as final match
  }

  mainDist = 0; /* for GCC */
  
  if (mainLen >= 2)
  {
    /* mainLen and mainDist are the last 2 of the valid elements in p->matches
    * mainLen: p->matches[numPairs - 2], mainDist: p->matches[numPairs - 1]
    * As we go up the list, we will have matches with lesser len (which will also
    * have smaller distances). Start comparing with previous entries in matches.
    * LZ77 always chooses the largest 'len' match irrespective of the distance.
    * In lzma, choosing a smaller <len, distance> pair can be justified if a smaller 
    * 'len' match is available at a shorter distance.
    * Heuristic of difference of len being > 1 or difference in distance not being
    * large enough is used to justify which pair to choose. */
    mainDist = p->matches[(size_t)numPairs - 1];
    while (numPairs > 2)
    {
      UInt32 dist2;
      if (mainLen != p->matches[(size_t)numPairs - 4] + 1) // compare with previous matches len
        break; // difference in len > 1 break
      dist2 = p->matches[(size_t)numPairs - 3];
      if (!ChangePair(dist2, mainDist))  // compare with previous matches distance
        break; // distance not large enough : dist2 > mainDist/7 break
      numPairs -= 2;
      mainLen--;
      mainDist = dist2;
    }
    if (mainLen == 2 && mainDist >= 0x80)
      mainLen = 1;
  }

  /* If any of the rep matches has repLen close to or larger than
  * the new mainLen, then it is better to stick to that repLen as
  * you will save on encoding distance */
  if (repLen >= 2)
    if (    repLen + 1 >= mainLen
        || (repLen + 2 >= mainLen && mainDist >= (1 << 9))
        || (repLen + 3 >= mainLen && mainDist >= (1 << 15)))
  {
    p->backRes = (UInt32)repIndex;
    MOVE_POS(p, repLen - 1)
    return repLen; // choose repLen as final match
  }
  
  if (mainLen < 2 || numAvail <= 2)
    return 1;

  /* Check if skipping a byte and coding it as a literal is worth it, i.e.
  * if match for sequence starting at next byte is better overall */
  {
    unsigned len1 = ReadMatchDistances(p, &p->numPairs); // Find matches by skipping a byte
    p->longestMatchLen = len1;
  
    if (len1 >= 2)
    {
      UInt32 newDist = p->matches[(size_t)p->numPairs - 1];
      // Criterion used to justify if we can skip current byte
      if (   (len1 >= mainLen && newDist < mainDist)
          || (len1 == mainLen + 1 && !ChangePair(mainDist, newDist))
          || (len1 >  mainLen + 1)
          || (len1 + 1 >= mainLen && mainLen >= 3 && ChangePair(newDist, mainDist)))
        return 1; // code current byte as literal
    }
  }
  
  data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - 1;
  
  /* Compare with reps for new sequence. This might still justify skipping
  * a byte, even if criterion above is not satisfied */
  for (i = 0; i < LZMA_NUM_REPS; i++)
  {
    unsigned len, limit;
    if(p->reps[i] == -1) /* Modified for MT implementation */
      continue;
    const Byte *data2 = data - p->reps[i];
    if (data[0] != data2[0] || data[1] != data2[1])
      continue;
    limit = mainLen - 1;
    for (len = 2;; len++)
    {
      if (len >= limit)
        return 1; // new sequence provides long enough match at this rep. Worth skipping a byte.
      if (data[len] != data2[len])
        break;
    }
  }
  
  /* Not worth skipping byte. Return longest mainLen and mainDist from
  * the first ReadMatchDistances call */
  p->backRes = mainDist + LZMA_NUM_REPS;
  if (mainLen != 2)
  {
    MOVE_POS(p, mainLen - 2)
  }
  return mainLen;
}

/**
 * LzmaEnc_CodeOneBlock_mt_1st_pass(): Same as LzmaEnc_CodeOneBlock(), but for MT mode.
 * Uses SequenceEntry list to store the sequence of matches and literals.
 * This is used to encode the data in the 2nd pass.
 * Does not flush the data to the output stream.
 */
MY_NO_INLINE
static SRes LzmaEnc_CodeOneBlock_mt_1st_pass(CLzmaEnc *p, UInt32 maxPackSize, UInt32 maxUnpackSize, SequenceEntry** head, SequenceEntry** tail)
{
  UInt32 nowPos32, startPos32;
  if (p->needInit)
  {
    p->matchFinder.Init(p->matchFinderObj);
    p->needInit = 0;
  }

  if (p->finished)
    return p->result;
  RINOK(CheckErrors(p));

  nowPos32 = (UInt32)p->nowPos64;
  startPos32 = nowPos32;


  if (p->nowPos64 == 0) // first byte in stream
  {
    unsigned numPairs;
    Byte curByte;
    if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
      return Flush(p, nowPos32);
    ReadMatchDistances(p, &numPairs);
    RangeEnc_EncodeBit_0(&p->rc, &p->isMatch[kState_Start][0]);
    // p->state = kLiteralNextStates[p->state];
    curByte = *(p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset);
    LitEnc_Encode(&p->rc, p->litProbs, curByte);

    // Modified for MT implementation
    addEntry(head, tail, 1 /* len */, MARK_LIT /* dist */, p->additionalOffset);

    p->additionalOffset--;
    nowPos32++;
  }

  if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) != 0)
  
  for (;;)
  {
    UInt32 dist;
    unsigned len, posState;
    UInt32 range, ttt, newBound;
    CLzmaProb *probs;
  
    if (p->fastMode) // hash-chain algo
      len = AOCL_GetOptimumFast_mt(p);
    else // bin-tree algo
    {
      unsigned oci = p->optCur;
      if (p->optEnd == oci)
        len = AOCL_GetOptimum(p, nowPos32);
      else
      {
        const COptimal *opt = &p->opt[oci];
        len = opt->len;
        p->backRes = opt->dist;
        p->optCur = oci + 1;
      }
    }

    posState = (unsigned)nowPos32 & p->pbMask;
    range = p->rc.range;
    probs = &p->isMatch[p->state][posState];
    
    RC_BIT_PRE(&p->rc, probs)
    
    dist = p->backRes;

    #ifdef SHOW_STAT2
    printf("\n pos = %6X, len = %3u  pos = %6u", nowPos32, len, dist);
    #endif

    /* Depending on the type of data to be saved: literal, srep, 
    * rep or match, corresponding code is encoded. This is followed by 
    * optionally encoding len and/or distance values */
    if (dist == MARK_LIT) // literal
    {
      Byte curByte;
      const Byte *data;
      unsigned state;

      RC_BIT_0(&p->rc, probs); // code:0+*
      p->rc.range = range;
      data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset;
      probs = LIT_PROBS(nowPos32, *(data - 1)); // get appropriate context model for literal encoding 
      curByte = *data;
      state = p->state;
      p->state = kLiteralNextStates[state];

      // Modified for MT implementation
      addEntry(head, tail, 1 /* len */, MARK_LIT /* dist */, p->additionalOffset);

      if (IsLitState(state))
        LitEnc_Encode(&p->rc, probs, curByte);
      else
        LitEnc_EncodeMatched(&p->rc, probs, curByte, *(data - p->reps[0]));

    }
    else // rep, srep or match
    {
      // Modified for MT implementation
      unsigned match_len = len;
      unsigned match_offset = dist;

      RC_BIT_1(&p->rc, probs); // code:1+*
      probs = &p->isRep[p->state];
      RC_BIT_PRE(&p->rc, probs)
      
      if (dist < LZMA_NUM_REPS) // rep or srep
      {
        RC_BIT_1(&p->rc, probs); // code:11+*
        probs = &p->isRepG0[p->state];
        RC_BIT_PRE(&p->rc, probs)
        if (dist == 0) // srep or rep0
        {
          RC_BIT_0(&p->rc, probs); // code:110+*
          probs = &p->isRep0Long[p->state][posState];
          RC_BIT_PRE(&p->rc, probs)
          if (len != 1)
          {
            RC_BIT_1_BASE(&p->rc, probs); //code:1101
          }
          else // srep
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1100
            p->state = kShortRepNextStates[p->state];
          }
        }
        else // rep1-3
        {
          RC_BIT_1(&p->rc, probs); // code:111+*
          probs = &p->isRepG1[p->state];
          RC_BIT_PRE(&p->rc, probs)
          if (dist == 1) // rep1
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1110
            dist = p->reps[1];
          }
          else // rep2-3
          {
            RC_BIT_1(&p->rc, probs); // code:1111+*
            probs = &p->isRepG2[p->state];
            RC_BIT_PRE(&p->rc, probs)
            if (dist == 2) // rep2
            {
              RC_BIT_0_BASE(&p->rc, probs); // code:11110
              dist = p->reps[2];
            }
            else // rep3
            {
              RC_BIT_1_BASE(&p->rc, probs); // code:11111
              dist = p->reps[3];
              p->reps[3] = p->reps[2]; // move reps queue dist->0->1->2->3
            }
            p->reps[2] = p->reps[1]; // move reps queue dist->0->1->2->3
          }
          p->reps[1] = p->reps[0]; // move reps queue dist->0->1->2->3
          p->reps[0] = dist; // move reps queue dist->0->1->2->3
        }

        RC_NORM(&p->rc)

        p->rc.range = range;

        if (len != 1) // code saved, now encode len
        {
          LenEnc_Encode(&p->repLenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState);
          --p->repLenEncCounter;
          p->state = kRepNextStates[p->state];
        }
      }
      else // match
      {
        unsigned posSlot;
        RC_BIT_0(&p->rc, probs);
        p->rc.range = range;
        p->state = kMatchNextStates[p->state];

        LenEnc_Encode(&p->lenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState); // encode len
        // --p->lenEnc.counter;

        dist -= LZMA_NUM_REPS;
        p->reps[3] = p->reps[2];
        p->reps[2] = p->reps[1];
        p->reps[1] = p->reps[0];
        p->reps[0] = dist + 1;
        
        p->matchPriceCount++;
        GetPosSlot(dist, posSlot);
        // RcTree_Encode_PosSlot(&p->rc, p->posSlotEncoder[GetLenToPosState(len)], posSlot);
        {
          UInt32 sym = (UInt32)posSlot + (1 << kNumPosSlotBits);
          range = p->rc.range;
          probs = p->posSlotEncoder[GetLenToPosState(len)];
          do // Range code 'slot' bitwise
          {
            CLzmaProb *prob = probs + (sym >> kNumPosSlotBits);
            UInt32 bit = (sym >> (kNumPosSlotBits - 1)) & 1;
            sym <<= 1;
            RC_BIT(&p->rc, prob, bit);
          }
          while (sym < (1 << kNumPosSlotBits * 2));
          p->rc.range = range;
        }
        
        if (dist >= kStartPosModelIndex) // dist > 3 (dist 0:3 no direct_bits)
        {
          unsigned footerBits = ((posSlot >> 1) - 1);

          if (dist < kNumFullDistances) // dis: 4-127, 
          {
            unsigned base = ((2 | (posSlot & 1)) << footerBits);
            RcTree_ReverseEncode(&p->rc, p->posEncoders + base, footerBits, (unsigned)(dist /* - base */));
          }
          else // dist >= 128
          {
            UInt32 pos2 = (dist | 0xF) << (32 - footerBits); // direct_bits-4 part
            range = p->rc.range;
            // RangeEnc_EncodeDirectBits(&p->rc, posReduced >> kNumAlignBits, footerBits - kNumAlignBits);
            /*
            do
            {
              range >>= 1;
              p->rc.low += range & (0 - ((dist >> --footerBits) & 1));
              RC_NORM(&p->rc)
            }
            while (footerBits > kNumAlignBits);
            */
            do // fixed 0.5 prob model
            {
              range >>= 1;
              p->rc.low += range & (0 - (pos2 >> 31));
              pos2 += pos2;
              RC_NORM(&p->rc)
            }
            while (pos2 != 0xF0000000);


            // RcTree_ReverseEncode(&p->rc, p->posAlignEncoder, kNumAlignBits, posReduced & kAlignMask);

            { // remaining 4 align bits LSB to MSB
              unsigned m = 1;
              unsigned bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1;             RC_BIT(&p->rc, p->posAlignEncoder + m, bit);
              p->rc.range = range;
              // p->alignPriceCount++;
            }
          }
        }
      }
      // Modified for MT implementation
      addEntry(head, tail, match_len /* len */, match_offset /* dist */, p->additionalOffset);
    }

    nowPos32 += (UInt32)len;
    p->additionalOffset -= len;
        
    if (p->additionalOffset == 0)
    {
      UInt32 processed;

      if (!p->fastMode)
      {
        /*
        if (p->alignPriceCount >= 16) // kAlignTableSize
          FillAlignPrices(p);
        if (p->matchPriceCount >= 128)
          FillDistancesPrices(p);
        if (p->lenEnc.counter <= 0)
          LenPriceEnc_UpdateTables(&p->lenEnc, 1 << p->pb, &p->lenProbs, p->ProbPrices);
        */
        if (p->matchPriceCount >= 64)
        {
          FillAlignPrices(p);
          // { int y; for (y = 0; y < 100; y++) {
          FillDistancesPrices(p);
          // }}
          LenPriceEnc_UpdateTables(&p->lenEnc, (unsigned)1 << p->pb, &p->lenProbs, p->ProbPrices);
        }
        if (p->repLenEncCounter <= 0)
        {
          p->repLenEncCounter = REP_LEN_COUNT;
          LenPriceEnc_UpdateTables(&p->repLenEnc, (unsigned)1 << p->pb, &p->repLenProbs, p->ProbPrices);
        }
      }
    
      if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
        break;
      processed = nowPos32 - startPos32;
      
      if (maxPackSize)
      {
        if (processed + kNumOpts + 300 >= maxUnpackSize
            || RangeEnc_GetProcessed_sizet(&p->rc) + kPackReserve >= maxPackSize)
          break;
      }
      else if (processed >= (1 << 17))
      {
        p->nowPos64 += nowPos32 - startPos32;
        return CheckErrors(p);
      }
    }
  }

  p->nowPos64 += nowPos32 - startPos32;
  return Flush(p, nowPos32);
}

/**
 * LzmaEnc_Encode2_mt_1st_pass(): Same as LzmaEnc_Encode2(), but for MT mode.
 * Calls LzmaEnc_CodeOneBlock_mt_1st_pass() in place of LzmaEnc_CodeOneBlock()
 * in a loop until all data is processed.
 */
MY_NO_INLINE
static SRes LzmaEnc_Encode2_mt_1st_pass(CLzmaEnc *p, ICompressProgress *progress, SequenceEntry** head, SequenceEntry** tail)
{
  SRes res = SZ_OK;

  #ifndef _7ZIP_ST
  Byte allocaDummy[0x300];
  allocaDummy[0] = 0;
  allocaDummy[1] = allocaDummy[0];
  #endif

  for (;;)
  {
    res = LzmaEnc_CodeOneBlock_mt_1st_pass(p, 0, 0, head, tail);
    if (res != SZ_OK || p->finished)
      break;
    if (progress)
    {
      res = ICompressProgress_Progress(progress, p->nowPos64, RangeEnc_GetProcessed(&p->rc));
      if (res != SZ_OK)
      {
        res = SZ_ERROR_PROGRESS;
        break;
      }
    }
  }
  
  LzmaEnc_Finish(p);

  /*
  if (res == SZ_OK && !Inline_MatchFinder_IsFinishedOK(&MFB))
    res = SZ_ERROR_FAIL;
  }
  */

  return res;
}

/**
 * LzmaEnc_MemEncode_mt_1st_pass(): Same as LzmaEnc_MemEncode(), but for MT mode.
 * calls LzmaEnc_MemPrepare_mt() to prepare the encoder and LzmaEnc_Encode2_mt_1st_pass() to encode
 * in place of LzmaEnc_MemPrepare() and LzmaEnc_Encode2() respectively.
 */
SRes LzmaEnc_MemEncode_mt_1st_pass(CLzmaEncHandle pp, Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    int writeEndMark, ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig, SequenceEntry** head, SequenceEntry** tail)
{
  AOCL_SETUP_NATIVE();
  if (pp == NULL || src == NULL || srcLen == 0 || dest == NULL || destLen == NULL || alloc == NULL || allocBig == NULL) {
      LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
      return SZ_ERROR_PARAM;
  }

  SRes res;
  CLzmaEnc *p = (CLzmaEnc *)pp;

  CLzmaEnc_SeqOutStreamBuf outStream;

  outStream.vt.Write = SeqOutStreamBuf_Write;
  outStream.data = dest;
  outStream.rem = *destLen;
  outStream.overflow = False;

  p->writeEndMark = writeEndMark;
  p->rc.outStream = &outStream.vt;

  res = LzmaEnc_MemPrepare_mt(pp, src, srcLen, 0, alloc, allocBig);
 
  if (res == SZ_OK)
  {
    res = LzmaEnc_Encode2_mt_1st_pass(p, progress, head, tail);
    if (res == SZ_OK && p->nowPos64 != srcLen) {
        LOG_UNFORMATTED(ERR, logCtx, "Not all src bytes processed");
        res = SZ_ERROR_FAIL;
    }
  }

  *destLen -= outStream.rem;
  if (outStream.overflow) {
      LOG_UNFORMATTED(ERR, logCtx, "Out stream overflow");
      return SZ_ERROR_OUTPUT_EOF;
  }
  return res;
}
/**************************************
 * MT 1st Pass functions - End
 *************************************/
/**************************************
 * MT 2nd Pass functions - Start
 *************************************/
/**
 * LzmaEnc_CodeOneBlock_st_2nd_pass(): Same as LzmaEnc_CodeOneBlock(), but used in MT mode 
 * for the second pass that operates in ST mode. Iterates through the SequenceEntry list
 * and encodes each entry as a literal, match or rep.
 * Calls Flush() at the end to flush the encoded data to the output stream only if it is
 * the last thread.
 */
MY_NO_INLINE
static SRes LzmaEnc_CodeOneBlock_st_2nd_pass(SequenceEntry **head, CLzmaEnc *p, UInt32 maxPackSize, UInt32 maxUnpackSize, int last_thread)
{
  SequenceEntry *current = *head;
  UInt32 nowPos32, startPos32;
  if (p->needInit)
  {
    p->matchFinder.Init(p->matchFinderObj);
    p->needInit = 0;
  }

  if (p->finished)
    return p->result;
  RINOK(CheckErrors(p));

  nowPos32 = (UInt32)p->nowPos64;
  startPos32 = nowPos32;

  if (p->nowPos64 == 0) // first byte in  stream
  {
    // unsigned numPairs;   // edit - commented
    Byte curByte;
    if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
      return Flush(p, nowPos32);
    // ReadMatchDistances(p, &numPairs);     // edit - commented
    RangeEnc_EncodeBit_0(&p->rc, &p->isMatch[kState_Start][0]);
    // p->state = kLiteralNextStates[p->state];
    // curByte = *(p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset); // edit - commented

    /* Modified for MT implementation - Start */
    curByte = *(p->matchFinderBase.bufferBase + nowPos32); 
    p->additionalOffset = current->additionalOffset;
    current = current->next;
    /* Modified for MT implementation - End */

    LitEnc_Encode(&p->rc, p->litProbs, curByte);
    p->additionalOffset--;
    nowPos32++;

  }

  // if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) != 0) // edit - commented
  
  /* Modified for MT implementation - Start */
  while(current != NULL)
  {
    UInt32 dist;
    unsigned len, posState;
    UInt32 range, ttt, newBound;
    CLzmaProb *probs;
    Byte* data;
    
    if(current->dist == MARK_LIT){
      data = (p->matchFinderBase.bufferBase + nowPos32); // literal data
    }
    len = current->match_len;                         // match length; 1 if it is literal
    dist = current->dist;                             // distance
    p->additionalOffset = current->additionalOffset;  // additional offset

    current = current->next;                          // move to next entry
    
    p->backRes = dist;

    /* Modified for MT implementation - End */

    posState = (unsigned)nowPos32 & p->pbMask;
    range = p->rc.range;
    probs = &p->isMatch[p->state][posState];
    
    RC_BIT_PRE(&p->rc, probs)
    

    #ifdef SHOW_STAT2
    printf("\n pos = %6X, len = %3u  pos = %6u", nowPos32, len, dist);
    #endif
    
    /* Depending on the type of data to be saved: literal, srep, 
    * rep or match, corresponding code is encoded. This is followed by 
    * optionally encoding len and/or distance values */
    if (dist == MARK_LIT) // literal
    {
      Byte curByte;
      // const Byte *data;  // edit - commented
      unsigned state;

      RC_BIT_0(&p->rc, probs); // code:0+*
      p->rc.range = range;
      // data = p->matchFinder.GetPointerToCurrentPos(p->matchFinderObj) - p->additionalOffset; // edit - commented
      probs = LIT_PROBS(nowPos32, *(data - 1)); // get appropriate context model for literal encoding 
      curByte = *data;
      state = p->state;
      p->state = kLiteralNextStates[state];
      if (IsLitState(state))
        LitEnc_Encode(&p->rc, probs, curByte);
      else
        LitEnc_EncodeMatched(&p->rc, probs, curByte, *(data - p->reps[0]));
    }
    else // rep, srep or match
    {
      RC_BIT_1(&p->rc, probs); // code:1+*
      probs = &p->isRep[p->state];
      RC_BIT_PRE(&p->rc, probs)
      
      if (dist < LZMA_NUM_REPS) // rep or srep
      {
        RC_BIT_1(&p->rc, probs); // code:11+*
        probs = &p->isRepG0[p->state];
        RC_BIT_PRE(&p->rc, probs)
        if (dist == 0) // srep or rep0
        {
          RC_BIT_0(&p->rc, probs); // code:110+*
          probs = &p->isRep0Long[p->state][posState];
          RC_BIT_PRE(&p->rc, probs)
          if (len != 1)
          {
            RC_BIT_1_BASE(&p->rc, probs); //code:1101
          }
          else // srep
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1100
            p->state = kShortRepNextStates[p->state];
          }
        }
        else // rep1-3
        {
          RC_BIT_1(&p->rc, probs); // code:111+*
          probs = &p->isRepG1[p->state];
          RC_BIT_PRE(&p->rc, probs)
          if (dist == 1) // rep1
          {
            RC_BIT_0_BASE(&p->rc, probs); // code:1110
            dist = p->reps[1];
          }
          else // rep2-3
          {
            RC_BIT_1(&p->rc, probs); // code:1111+*
            probs = &p->isRepG2[p->state];
            RC_BIT_PRE(&p->rc, probs)
            if (dist == 2) // rep2
            {
              RC_BIT_0_BASE(&p->rc, probs); // code:11110
              dist = p->reps[2];
            }
            else // rep3
            {
              RC_BIT_1_BASE(&p->rc, probs); // code:11111
              dist = p->reps[3];
              p->reps[3] = p->reps[2]; // move reps queue dist->0->1->2->3
            }
            p->reps[2] = p->reps[1]; // move reps queue dist->0->1->2->3
          }
          p->reps[1] = p->reps[0]; // move reps queue dist->0->1->2->3
          p->reps[0] = dist; // move reps queue dist->0->1->2->3
        }

        RC_NORM(&p->rc)

        p->rc.range = range;

        if (len != 1) // code saved, now encode len
        {
          LenEnc_Encode(&p->repLenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState);
          --p->repLenEncCounter;
          p->state = kRepNextStates[p->state];
        }
      }
      else // match
      {
        unsigned posSlot;
        RC_BIT_0(&p->rc, probs);
        p->rc.range = range;
        p->state = kMatchNextStates[p->state];

        LenEnc_Encode(&p->lenProbs, &p->rc, len - LZMA_MATCH_LEN_MIN, posState); // encode len
        // --p->lenEnc.counter;

        dist -= LZMA_NUM_REPS;
        p->reps[3] = p->reps[2];
        p->reps[2] = p->reps[1];
        p->reps[1] = p->reps[0];
        p->reps[0] = dist + 1;
        
        p->matchPriceCount++;
        GetPosSlot(dist, posSlot);
        // RcTree_Encode_PosSlot(&p->rc, p->posSlotEncoder[GetLenToPosState(len)], posSlot);
        {
          UInt32 sym = (UInt32)posSlot + (1 << kNumPosSlotBits);
          range = p->rc.range;
          probs = p->posSlotEncoder[GetLenToPosState(len)];
          do // Range code 'slot' bitwise
          {
            CLzmaProb *prob = probs + (sym >> kNumPosSlotBits);
            UInt32 bit = (sym >> (kNumPosSlotBits - 1)) & 1;
            sym <<= 1;
            RC_BIT(&p->rc, prob, bit);
          }
          while (sym < (1 << kNumPosSlotBits * 2));
          p->rc.range = range;
        }
        
        if (dist >= kStartPosModelIndex) // dist > 3 (dist 0:3 no direct_bits)
        {
          unsigned footerBits = ((posSlot >> 1) - 1);

          if (dist < kNumFullDistances) // dis: 4-127, 
          {
            unsigned base = ((2 | (posSlot & 1)) << footerBits);
            RcTree_ReverseEncode(&p->rc, p->posEncoders + base, footerBits, (unsigned)(dist /* - base */));
          }
          else // dist >= 128
          {
            UInt32 pos2 = (dist | 0xF) << (32 - footerBits); // direct_bits-4 part
            range = p->rc.range;
            // RangeEnc_EncodeDirectBits(&p->rc, posReduced >> kNumAlignBits, footerBits - kNumAlignBits);
            /*
            do
            {
              range >>= 1;
              p->rc.low += range & (0 - ((dist >> --footerBits) & 1));
              RC_NORM(&p->rc)
            }
            while (footerBits > kNumAlignBits);
            */
            do // fixed 0.5 prob model
            {
              range >>= 1;
              p->rc.low += range & (0 - (pos2 >> 31));
              pos2 += pos2;
              RC_NORM(&p->rc)
            }
            while (pos2 != 0xF0000000);


            // RcTree_ReverseEncode(&p->rc, p->posAlignEncoder, kNumAlignBits, posReduced & kAlignMask);

            { // remaining 4 align bits LSB to MSB
              unsigned m = 1;
              unsigned bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1; dist >>= 1; RC_BIT(&p->rc, p->posAlignEncoder + m, bit); m = (m << 1) + bit;
              bit = dist & 1;             RC_BIT(&p->rc, p->posAlignEncoder + m, bit);
              p->rc.range = range;
              // p->alignPriceCount++;
            }
          }
        }
      }
    }

    nowPos32 += (UInt32)len;
    p->additionalOffset -= len;
    
    if (p->additionalOffset == 0)
    {
      UInt32 processed;

      if (!p->fastMode)
      {
        /*
        if (p->alignPriceCount >= 16) // kAlignTableSize
          FillAlignPrices(p);
        if (p->matchPriceCount >= 128)
          FillDistancesPrices(p);
        if (p->lenEnc.counter <= 0)
          LenPriceEnc_UpdateTables(&p->lenEnc, 1 << p->pb, &p->lenProbs, p->ProbPrices);
        */
        if (p->matchPriceCount >= 64)
        {
          FillAlignPrices(p);
          // { int y; for (y = 0; y < 100; y++) {
          FillDistancesPrices(p);
          // }}
          LenPriceEnc_UpdateTables(&p->lenEnc, (unsigned)1 << p->pb, &p->lenProbs, p->ProbPrices);
        }
        if (p->repLenEncCounter <= 0)
        {
          p->repLenEncCounter = REP_LEN_COUNT;
          LenPriceEnc_UpdateTables(&p->repLenEnc, (unsigned)1 << p->pb, &p->repLenProbs, p->ProbPrices);
        }
      }
    
      // edit - commented
      // if (p->matchFinder.GetNumAvailableBytes(p->matchFinderObj) == 0)
      //   break;
      processed = nowPos32 - startPos32;
      
      if (maxPackSize)
      {
        if (processed + kNumOpts + 300 >= maxUnpackSize
            || RangeEnc_GetProcessed_sizet(&p->rc) + kPackReserve >= maxPackSize)
          break;
      }
      else if (processed >= (1 << 17))      
      {
        p->nowPos64 += nowPos32 - startPos32;
        *head = current; // Modified for MT implementation
        return CheckErrors(p);
      }
    }
  }

  /* Modified for MT implementation - Start */
  p->nowPos64 += nowPos32 - startPos32;
  if (last_thread) {
    return Flush(p, nowPos32);
  }

  p->finished = True;
  return CheckErrors(p);
  /* Modified for MT implementation - End */
}

/**
 * LzmaEnc_Encode2_st_2nd_pass(): Same as LzmaEnc_Encode2(), but used in MT mode 
 * for the second pass that operates in ST mode. Calls LzmaEnc_CodeOneBlock_st_2nd_pass()
 * in place of LzmaEnc_CodeOneBlock() in a loop until all data is processed.
 */
MY_NO_INLINE
static SRes LzmaEnc_Encode2_st_2nd_pass(CLzmaEnc *p, ICompressProgress *progress, SequenceEntry** head, int last_thread)
{
  SRes res = SZ_OK;

  #ifndef _7ZIP_ST
  Byte allocaDummy[0x300];
  allocaDummy[0] = 0;
  allocaDummy[1] = allocaDummy[0];
  #endif

  for (;;)
  {
    res = LzmaEnc_CodeOneBlock_st_2nd_pass(head, p, 0, 0, last_thread);
    if (res != SZ_OK || p->finished)
      break;
    if (progress)
    {
      res = ICompressProgress_Progress(progress, p->nowPos64, RangeEnc_GetProcessed(&p->rc));
      if (res != SZ_OK)
      {
        res = SZ_ERROR_PROGRESS;
        break;
      }
    }
  }
  
  LzmaEnc_Finish(p);

  /*
  if (res == SZ_OK && !Inline_MatchFinder_IsFinishedOK(&MFB))
    res = SZ_ERROR_FAIL;
  }
  */

  return res;
}

/**
 * LzmaEnc_MemEncode_st_2nd_pass(): Same as LzmaEnc_MemEncode(), but used in MT mode
 * for second pass that operates in ST mode. Calls LzmaEnc_MemPrepare_mt() to prepare the encoder and
 * LzmaEnc_Encode2_st_2nd_pass() in a loop for each thread to encode the data.
 * Frees the SequenceEntry list for each thread after encoding is done.
 */
SRes LzmaEnc_MemEncode_st_2nd_pass(CLzmaEncHandle pp, Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    int writeEndMark, ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig, aocl_thread_group_t thread_group_handle)
{
  AOCL_SETUP_NATIVE();
  if (pp == NULL || src == NULL || srcLen == 0 || dest == NULL || destLen == NULL)
      return SZ_ERROR_PARAM;

  SRes res;
  CLzmaEnc *p = (CLzmaEnc *)pp;

  CLzmaEnc_SeqOutStreamBuf outStream;

  outStream.vt.Write = SeqOutStreamBuf_Write;
  outStream.data = dest;
  outStream.rem = *destLen;
  outStream.overflow = False;

  p->writeEndMark = writeEndMark;
  p->rc.outStream = &outStream.vt;

  res = LzmaEnc_MemPrepare_mt(pp, src, srcLen, 0, alloc, allocBig);

  aocl_thread_info_t cur_thread_info;
  AOCL_UINT32 thread_cnt = 0;
  if (res == SZ_OK)
  {
    /* Modified for MT implementation - Start */
    for (thread_cnt = 0; thread_cnt < thread_group_handle.num_threads; thread_cnt++)
    {
      cur_thread_info = thread_group_handle.threads_info_list[thread_cnt];
      p->finished =  0; // reset finished state for next thread
      SequenceEntry *seqEntryList = cur_thread_info.additional_state_info;

      res = LzmaEnc_Encode2_st_2nd_pass(p, progress, (SequenceEntry **) &cur_thread_info.additional_state_info, thread_cnt==(thread_group_handle.num_threads - 1) ? 1 : 0);

      // free SequenceEntry list for this thread
      freeSequenceEntry(seqEntryList);
    }
    /* Modified for MT implementation - End */
    if (res == SZ_OK && p->nowPos64 != srcLen)
      res = SZ_ERROR_FAIL;
  }

  *destLen -= outStream.rem;
  if (outStream.overflow)
    return SZ_ERROR_OUTPUT_EOF;
  return res;
}

/**************************************
 * MT Pass 2 functions - End
 *************************************/

typedef struct
{
  aocl_thread_group_t *thread_group_handle;
  int writeEndMark;
  ISzAllocPtr alloc;
  ISzAllocPtr allocBig;
  const CLzmaEncProps *props;
} AOCL_LzmaEncode_MT_parallel_ctx;

static void AOCL_LzmaEncode_MT_parallel_worker(AOCL_UINT32 thread_id, void *context)
{
  AOCL_LzmaEncode_MT_parallel_ctx *ctx = (AOCL_LzmaEncode_MT_parallel_ctx *)context;
  aocl_thread_group_t *thread_group_handle = ctx->thread_group_handle;
  aocl_thread_info_t cur_thread_info;
  AOCL_UINT32 cmpr_bound_pad = 0;
  AOCL_UINT32 is_error = 1;
  SizeT local_result = 0;
  SequenceEntry *head = NULL;
  SequenceEntry *tail = NULL;

  if (aocl_do_partition_compress_mt(thread_group_handle, &cur_thread_info, cmpr_bound_pad, thread_id) == 0)
  {
    CLzmaEnc *p = (CLzmaEnc *)LzmaEnc_Create(ctx->alloc);
    SRes res;
    if (p)
    {
      ICompressProgress *curProgress = NULL;
      CLzmaEncProps props_cur = *ctx->props;
      props_cur.srcLen = cur_thread_info.partition_src_size;
      res = LzmaEnc_SetProps_fp(p, &props_cur);
      if (res == SZ_OK)
      {
        local_result = cur_thread_info.dst_trap_size;
        p->rc.flushData = False;
        res = LzmaEnc_MemEncode_mt_1st_pass(p, (Byte *)cur_thread_info.dst_trap, &local_result,
            (const Byte *)cur_thread_info.partition_src, cur_thread_info.partition_src_size,
            ctx->writeEndMark, curProgress, ctx->alloc, ctx->allocBig, &head, &tail);
        is_error = res;
      }
    }
    LzmaEnc_Destroy(p, ctx->alloc, ctx->allocBig);
  }

  thread_group_handle->threads_info_list[thread_id].partition_src = cur_thread_info.partition_src;
  thread_group_handle->threads_info_list[thread_id].dst_trap = cur_thread_info.dst_trap;
  thread_group_handle->threads_info_list[thread_id].additional_state_info = head;
  thread_group_handle->threads_info_list[thread_id].dst_trap_size = local_result;
  thread_group_handle->threads_info_list[thread_id].partition_src_size = cur_thread_info.partition_src_size;
  thread_group_handle->threads_info_list[thread_id].last_bytes_len = 0;
  thread_group_handle->threads_info_list[thread_id].is_error = is_error;
  thread_group_handle->threads_info_list[thread_id].num_child_threads = 0;
}

 SRes AOCL_LzmaEncode_MT(Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
  const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark,
  ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  if (src == NULL || srcLen == 0 || dest == NULL || propsEncoded == NULL ||
    props == NULL || propsSize == NULL || destLen == NULL ||
    *destLen > (ULLONG_MAX - LZMA_PROPS_SIZE)) // handles case when dest size is < LZMA_PROPS_SIZE, resulting in destLen rolling over in calling APIs
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return SZ_ERROR_PARAM;
  }

  if (ValidateParams(props) != SZ_OK)
  {
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return SZ_ERROR_PARAM;
  }
  aocl_thread_group_t thread_group_handle;
  aocl_thread_info_t cur_thread_info;
  AOCL_INT32 rap_frame_len = -1;
  SRes result;

  size_t window_len = 0;
  {
    // write header
    CLzmaEnc* p = (CLzmaEnc*)LzmaEnc_Create(alloc);
    SRes res;
    if (!p)
    {
        return SZ_ERROR_MEM;
    }

    CLzmaEncProps props_cur = *props;
    props_cur.srcLen = srcLen; //same srcLen value must be set here and passed to LzmaEnc_MemEncode()
    res = LzmaEnc_SetProps_fp(p, &props_cur);
    window_len = p->dictSize;
    if (res != SZ_OK) return res;
    
    res = LzmaEnc_WriteProperties(p, propsEncoded, propsSize);
    if (res != SZ_OK) return res;

    LzmaEnc_Destroy(p, alloc, allocBig);
  }

  AOCL_UINT32 window_factor = LZMA_GET_WINDOW_FACTOR;
  rap_frame_len = aocl_setup_parallel_compress_mt(&thread_group_handle, (char*)src,
                        (char*)dest, srcLen, *destLen, window_len, window_factor);

  if (rap_frame_len < 0) {
    return SZ_ERROR_PARAM;
  }

  if (thread_group_handle.num_threads == 1)
  {
    LOG_UNFORMATTED(INFO, logCtx, "Running single threaded compress");
    result = LzmaEncode_ST(dest, destLen, src, srcLen, props, propsEncoded, propsSize, writeEndMark,
                            progress, alloc, allocBig);
  }
  else
  {
    AOCL_LzmaEncode_MT_parallel_ctx parallel_ctx;
    parallel_ctx.thread_group_handle = &thread_group_handle;
    parallel_ctx.writeEndMark = writeEndMark;
    parallel_ctx.alloc = alloc;
    parallel_ctx.allocBig = allocBig;
    parallel_ctx.props = props;
    aocl_parallel_for(thread_group_handle.num_threads, AOCL_LzmaEncode_MT_parallel_worker, &parallel_ctx);

    /* Post processing in single - threaded mode : */

    for (AOCL_UINT32 thread_cnt = 0; thread_cnt < thread_group_handle.num_threads; thread_cnt++)
    {
        cur_thread_info = thread_group_handle.threads_info_list[thread_cnt];
        //In case of any thread partitioning or alloc errors, exit the compression process with error
        if (cur_thread_info.is_error != SZ_OK)
        {
          aocl_destroy_parallel_compress_mt(&thread_group_handle);
          return cur_thread_info.is_error;
        }
      }

    // TODO: Add RAP frame to the destination buffer to support MT decompression

    // 2nd pass: Extract LZ77 output from each thread and run entropy coder on it
    CLzmaEnc* p = (CLzmaEnc*)LzmaEnc_Create(alloc);
    SRes res;
    if (!p)
    {
      LOG_UNFORMATTED(INFO, logCtx, "Exit");
      return SZ_ERROR_MEM;
    }
    CLzmaEncProps props_cur = *props;
    props_cur.srcLen = srcLen; //same srcLen value must be set here and passed to LzmaEnc_MemEncode()
    res = LzmaEnc_SetProps_fp(p, &props_cur);

    if (res == SZ_OK){
      res = LzmaEnc_MemEncode_st_2nd_pass(p, (Byte *)thread_group_handle.dst, destLen, src, srcLen,
        writeEndMark, progress, alloc, allocBig, thread_group_handle);
    }
    
    LzmaEnc_Destroy(p, alloc, allocBig);
    aocl_destroy_parallel_compress_mt(&thread_group_handle);

    result = SZ_OK;
  }
  return result;
}
#endif /* AOCL_ENABLE_THREADS */

static void aocl_register_lzma_encode_fmv(int optOff, CpuFeatures cpuFeatures)
{
    if (optOff)
    {
        /* C baseline profile */
        MatchFinder_CreateVTable_fp = MatchFinder_CreateVTable;
        MatchFinder_Create_fp       = MatchFinder_Create;
        MatchFinder_Free_fp         = MatchFinder_Free;
        GetOptimum_fp               = GetOptimum;
        LzmaEncProps_Normalize_fp   = LzmaEncProps_Normalize;
        LzmaEnc_SetProps_fp         = LzmaEnc_SetProps;
#ifdef AOCL_ENABLE_THREADS
        LzmaEncode_mt_fp            = LzmaEncode_ST;
#endif
        return;
    }

    /* FMV variant table (highest priority first). */
    static const AoclLzmaEncodeDispatchVariant variants[] = {
#ifdef AOCL_LZMA_OPT
        { 0, AOCL_LZMA_ENCODE_PROFILE_OPT }
#else
        { 0, AOCL_LZMA_ENCODE_PROFILE_BASELINE }
#endif
    };

    /* Select first compatible FMV variant for detected CPU features. */
    size_t variant_index = aocl_lzma_select_fmv_variant(
        variants,
        AOCL_LZMA_ARRAY_SIZE(variants),
        sizeof(variants[0]),
        offsetof(AoclLzmaEncodeDispatchVariant, required_features),
        cpuFeatures);

    if (variant_index >= AOCL_LZMA_ARRAY_SIZE(variants)) {
        return;
    }

    switch (variants[variant_index].profile) {
        case AOCL_LZMA_ENCODE_PROFILE_OPT:
#ifdef AOCL_LZMA_OPT
            MatchFinder_CreateVTable_fp = AOCL_MatchFinder_CreateVTable;
            MatchFinder_Create_fp       = AOCL_MatchFinder_Create;
            MatchFinder_Free_fp         = AOCL_MatchFinder_Free;
            GetOptimum_fp               = AOCL_GetOptimum;
            LzmaEncProps_Normalize_fp   = AOCL_LzmaEncProps_Normalize;
            LzmaEnc_SetProps_fp         = AOCL_LzmaEnc_SetProps;
#ifdef AOCL_ENABLE_THREADS
            LzmaEncode_mt_fp            = AOCL_LzmaEncode_MT;
#endif
#endif
            break;

        case AOCL_LZMA_ENCODE_PROFILE_BASELINE:
        default:
            MatchFinder_CreateVTable_fp = MatchFinder_CreateVTable;
            MatchFinder_Create_fp       = MatchFinder_Create;
            MatchFinder_Free_fp         = MatchFinder_Free;
            GetOptimum_fp               = GetOptimum;
            LzmaEncProps_Normalize_fp   = LzmaEncProps_Normalize;
            LzmaEnc_SetProps_fp         = LzmaEnc_SetProps;
#ifdef AOCL_ENABLE_THREADS
            LzmaEncode_mt_fp            = LzmaEncode_ST;
#endif
            break;
    }
}

void aocl_setup_lzma_encode(int optOff, int optLevel, size_t insize,
    size_t level, size_t windowLog)
{
    AOCL_ENTER_CRITICAL(setup_lzmaenc)
    if (!setup_ok_lzma_encode) {
        CpuFeatures cpuFeatures = Dispatcher_GetSupportedFeaturesForLevel(Dispatcher_IntToLevel((int)optLevel));
        optOff = optOff ? 1 : get_disable_opt_flags(0);
        aocl_register_lzma_encode_fmv(optOff, cpuFeatures);
        setup_ok_lzma_encode = 1;
    }
    AOCL_EXIT_CRITICAL(setup_lzmaenc)
}

#ifdef AOCL_LZMA_OPT
static void aocl_setup_native(void) {
    AOCL_ENTER_CRITICAL(setup_lzmaenc)
    if (!setup_ok_lzma_encode) {
        CpuFeatures cpuFeatures = Dispatcher_GetFeaturesFromEnv();
        int optOff = get_disable_opt_flags(0);
        aocl_register_lzma_encode_fmv(optOff, cpuFeatures);
        setup_ok_lzma_encode = 1;
    }
    AOCL_EXIT_CRITICAL(setup_lzmaenc)
}
#endif

void aocl_destroy_lzma_encode(void){
    AOCL_ENTER_CRITICAL(setup_lzmaenc)
    setup_ok_lzma_encode = 0;
    AOCL_EXIT_CRITICAL(setup_lzmaenc)
}

#ifdef AOCL_UNIT_TEST
#ifdef AOCL_LZMA_OPT
void Test_LzmaEncProps_Normalize_Dyn(CLzmaEncProps* p) {
#ifdef AOCL_LZMA_OPT
    LzmaEncProps_Normalize_fp(p);
#else
    LzmaEncProps_Normalize(p);
#endif
}

UInt64 Test_SetDataSize(CLzmaEncHandle pp, UInt64 expectedDataSize) {
    LzmaEnc_SetDataSize(pp, expectedDataSize);
    CLzmaEnc* p = (CLzmaEnc*)pp;
    return p->matchFinderBase.expectedDataSize;
}

SRes Test_WriteProperties(CLzmaEncHandle pp, Byte* props, SizeT* size, UInt32 dictSize) {
    CLzmaEnc* p = (CLzmaEnc*)pp;
    p->dictSize = dictSize;
    return LzmaEnc_WriteProperties(pp, props, size);
}

unsigned Test_IsWriteEndMark(CLzmaEncHandle pp, unsigned wem) {
    CLzmaEnc* p = (CLzmaEnc*)pp;
    p->writeEndMark = wem;
    return LzmaEnc_IsWriteEndMark(p);
}

SRes Test_SetProps_Dyn(CLzmaEncHandle pp, const CLzmaEncProps* props) {
    CLzmaEnc* p = (CLzmaEnc*)pp;
#ifdef AOCL_LZMA_OPT
    SRes res = LzmaEnc_SetProps_fp(p, props);
#else
    SRes res = LzmaEnc_SetProps(p, props);
#endif
    return res;
}

TestCLzmaEnc Get_CLzmaEnc_Params(CLzmaEncHandle pp) {
    CLzmaEnc* p = (CLzmaEnc*)pp;
    TestCLzmaEnc params;
    params.numFastBytes         = p->numFastBytes;
    params.lc                   = p->lc;
    params.lp                   = p->lp;
    params.pb                   = p->pb;
    params.fastMode             = p->fastMode;
    params.writeEndMark         = p->writeEndMark;
    params.dictSize             = p->dictSize;
    params.btMode               = p->matchFinderBase.btMode;
    params.cutValue             = p->matchFinderBase.cutValue;
    params.level                = p->matchFinderBase.level;
    params.cacheEfficientSearch = p->matchFinderBase.cacheEfficientSearch;
    params.numHashBytes         = p->matchFinderBase.numHashBytes;
    return params;
}
#endif /* AOCL_LZMA_OPT */
#endif /* AOCL_UNIT_TEST */
