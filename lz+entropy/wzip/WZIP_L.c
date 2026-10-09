/*
 * WZIP_L - WZIP for inputs of 32 KB and more
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

/*This code assumes the source data size is at least 1<<15, up to 1<<31.
  It deploys elastic sliding windows whose sizes are dynamically set upon the source size. 
  It assumes the source size is included in the compressed stream so that the decoder can pre-determine the elastic sliding windows.
  Huffman coding is applied over every unit size of HUF_BlockSize literals, as well as every unit size of SEQ_BlockSize WLZ triples.
*/
#include "Memry.h"
#include "BitStream_Huffman.h"
#include "WZIP.h"
#ifndef WZIP_MULTITHREAD
#  define WZIP_MULTITHREAD 0                       /* 1: WZIP_Set_Workers may run match finding in threads (pthreads or Win32) */
#endif
#include <stdio.h>
#include <math.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

//#define WZIP_DEBUG

/* CapHufLitBits may not exceed 14, so that 4 branches can be read in parallel before flushing */
#define   CapHufLitBits        MAX_HufWeight     
#define   CapHufLitRunBits     11
#define   CapHufMchLenBits     11
#define   CapHufMchOffBits     10

#define   MinMatchLen          3
#define   MaxMatchLen          4095
#define   MaxLitRunMsb         11


#define   WLZ_Hash0(stream)    Hash_3B(stream)     /* corresponding to MinMatchLen = 3 */
#define	  WLZ_Hash1(stream)     ( hash1Len == 4 ? Hash_4B(stream) : hash1Len == 5 ? Hash_5B(stream) : Hash_6B(stream) )
#define   WLZ_Hash2(stream)     ( hash2Len == 5 ? Hash_5B(stream) : hash2Len == 6 ? Hash_6B(stream) : hash2Len == 7 ? Hash_7B(stream) : Hash_8B(stream) )

/* The number of Huffman elements may not exceed 256, so that each entry is conveniently expressed by a byte */
#define   N_HufLits            256    
#define   N_HufLitRun          71
#define   N_HufMchLen          68                    /* lengths 3..MaxMatchLen (symbols 0-66), and the run symbol */
#define   RunSym               (N_HufMchLen - 1)     /* a run of copies of the preceding byte, its count in the offset field */
/* the match-length table codes a joint symbol: literal-run class (0: no literals, 1: one literal, 2: two or more,
   followed by a literal-run symbol) times the match-length symbol */
#define   N_LitClass           3
#define   N_SlotSel            5                     /* cache slot 0-3, or 4: a new offset (or a run count) from the offset table */
#define   N_HufJoint           (N_SlotSel * N_LitClass * N_HufMchLen)
#define   JointIdx(slotSel, litClass, mlSym)   (((slotSel) * N_LitClass + (litClass)) * N_HufMchLen + (mlSym))
#define   N_HufJointClassic    (N_LitClass * N_HufMchLen)   /* the joint symbol of a classic block: no cache slot */
#define   LitClass(litRunSym)  ((litRunSym) < 2 ? (litRunSym) : 2)
#define   LitRunDirect         32
#define   n_hufMachOff(offwidth) (2*offWidth)
#define   N_HufMchOffMax       (27*2)                              /* offsets of up to 27 bits */
#define   SEQ_BlockBound       (SEQ_BlockSize * 12 + 8192)         /* a block of sequences writes at most this: 93 bits a sequence, tables */
/* The sequence blocks wait at the end of the output buffer while the literal stream, which the stream puts first, grows
   from its start; at the end they move up behind the literals. Each block is coded in a scratch buffer of
   SEQ_BlockBound bytes (wlzStream) and stacked downward from the end of the capacity, so that no buffer of the
   input's size holds the sequences. Room runs out exactly when the literals and the sequences together would exceed
   the capacity, as with a separate buffer: the caller then stores the input. A literal block that may reach the
   stacked sequences is coded in a scratch buffer too (litScratch) and copied if it fits. */
typedef struct {
	Uint8* end;                                        /* the end of the output's capacity */
	Uint8* tail;                                       /* the lowest stacked byte */
	Uint32* sizes;                                     /* the blocks' sizes, in order */
	size_t n, cap;
} Seq_Stack;

static int Seq_Stack_Init(Seq_Stack* q, Uint8* end)
{
	q->end = q->tail = end;
	q->n = 0; q->cap = 64;
	q->sizes = (Uint32*)malloc(q->cap * sizeof(Uint32));
	return q->sizes != NULL;
}

/* stacks a block; 0 if the literals and the sequences would exceed the capacity, or memory runs out */
static int Seq_Stack_Push(Seq_Stack* q, const Uint8* block, Uint32 len, const Uint8* litEnd)
{
	if ((size_t)(q->tail - litEnd) < len) return 0;
	if (q->n == q->cap) {
		Uint32* const p_ = (Uint32*)realloc(q->sizes, 2 * q->cap * sizeof(Uint32));
		if (NULL == p_) return 0;
		q->sizes = p_; q->cap *= 2;
	}
	q->tail -= len;
	memcpy(q->tail, block, len);
	q->sizes[q->n++] = len;
	return 1;
}

static void Reverse_Bytes(Uint8* p, size_t n)
{
	for (Uint8* q = p + n - 1; p < q; p++, q--) { const Uint8 t = *p; *p = *q; *q = t; }
}

/* moves the stacked blocks, in their order, to dst (the end of the literals); returns their size */
static size_t Seq_Stack_Place(Seq_Stack* q, Uint8* dst)
{
	const size_t total = (size_t)(q->end - q->tail);
	if (dst + total <= q->tail) {                      /* apart: each block straight to its place */
		const Uint8* src = q->end;
		for (size_t k = 0; k < q->n; k++) { src -= q->sizes[k]; memcpy(dst, src, q->sizes[k]); dst += q->sizes[k]; }
	}
	else {                                             /* overlapping: down as a whole (last block first), then turned */
		memmove(dst, q->tail, total);
		Reverse_Bytes(dst, total);
		for (size_t k = 0; k < q->n; k++) { Reverse_Bytes(dst, q->sizes[k]); dst += q->sizes[k]; }
	}
	return total;
}

/* codes the sequences wlzSeq..wlzSeqPtr as a block and stacks it */
#define   SEQ_PUT_BLOCK()      { const Uint32 n_ = Huffman_Compress_WLZ(S_, wlzSeq, wlzSeqPtr, wlzStream, SEQ_BlockBound, &huffmanSet, &seqPrev); \
		if (!Seq_Stack_Push(&seqStack, wlzStream, n_, wzipLitPtr)) goto _lit_overflow; }
/* codes nLits_ literals of lzLitBuffer at wzipLitPtr (zipLitBlkSize: their size), below the stacked sequences */
#define   LIT_PUT_BLOCK(nLits_) { if (wzipLitEnd - wzipLitPtr < (ptrdiff_t)(nLits_) + LIT_BlockSlack) goto _lit_overflow;   \
		if (seqStack.tail - wzipLitPtr >= (ptrdiff_t)(nLits_) + LIT_BlockSlack)                                         \
			zipLitBlkSize = Huffman_Compress_Block_Rep(lzLitBuffer, (nLits_), wzipLitPtr, litHuf, N_HufLits, CapHufLitBits, &litPrev); \
		else {                                                                                                          \
			zipLitBlkSize = Huffman_Compress_Block_Rep(lzLitBuffer, (nLits_), litScratch, litHuf, N_HufLits, CapHufLitBits, &litPrev); \
			if (seqStack.tail - wzipLitPtr < (ptrdiff_t)zipLitBlkSize) goto _lit_overflow;                              \
			memcpy(wzipLitPtr, litScratch, zipLitBlkSize);                                                              \
		} }
#define   LIT_BlockSlack       512                                 /* a literal block writes at most its size plus this */
#define   OFF_SymBits          6                                   /* a sequence packs its offset symbol in 6 bits, raw bits above */
#define   MaxMchOffGroup       8
#define   SEQ_BlockSize        (1<<14)                   /*unit size of WLZ sequence to be Huffman coded */

#define   OffCasheSize          4                                   /* cashe size for the latest matching offsets */
#define   WINDOW(w)            ( (1<<w) -OffCasheSize +1 )
/* the format's two count limits: a literal run takes at most 24 raw bits (symbol 70), and a run's count is an offset
   value of the widest window. Longer runs are split; a longer literal run (16 MiB without a match) makes the
   compressor fail, so that the caller stores the input */
#define   MaxRunCount          ((Uint32)WINDOW(OffWidth[8]) - 1)
#define   LIT_RUN_70(litRun)   (LitRunTooLong |= (litRun) >> 24 != 0, N_HufLitRun - 1)

/* chain insertion runs L_InsertAhead positions ahead of the search, prefetching each inserted position's first
   candidate; hash slots are prefetched L_HashAhead positions ahead of insertion. A position's chain link is fixed
   when it is inserted, but a chain table of 2^w slots walked to a window of 2^w - 3 lets a position inserted ahead
   overwrite the slot of one still in the window: a walk can then follow a stale link to another position of the
   history, a candidate compared like any other. The walks stop at one in the dictionary's last 15 bytes, which hold
   no entries and whose 8-byte compares would read past it. Search positions lie before srcEnd - 16, so hashing
   L_InsertAhead (<= 8) positions ahead stays inside the input. */
#define   L_InsertAhead        8
#define   L_HashAhead          24
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <mmintrin.h>
#  define PREFETCH_L1(p)       _mm_prefetch((const char*)(p), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#  define PREFETCH_L1(p)       __builtin_prefetch((p), 0, 3)
#else
#  define PREFETCH_L1(p)       ((void)(p))
#endif
/* level 0 (fast mode): one hash table of 2^L0F_HashLog entries on L0F_MinMatch bytes, a window of 2^L0F_WindowLog,
   and a probe step that grows by one every 2^L0F_SkipLog literals */
#define   L0F_HashLog          15
#define   L0F_MinMatch         6
#define   L0F_SkipLog          6
#define   L0F_WindowLog        20
/* level 1: hash tables of at most 2^L0_HashLog entries; in a literal run, the probe step grows by one every
   2^L0_SkipLog literals */
#define   L0_HashLog           18
#define   L0_SkipLog           8
/* in levels 0 and 1, the probe step stops growing at 1 + L_MaxProbeStep, so that they find matches again soon after a
   long stretch without any; a literal run of L_RunBreakAt bytes (half the format's limit of 2^24) ends at the first
   short match L_Run_Break finds, and only without one is the input stored */
#ifndef L_MaxProbeStep
#define   L_MaxProbeStep       1024
#endif
#define   L_RunBreakAt         (1u << 23)
#define   L_PROBE_STEP(run, skipLog)   (min((Uint32)((run) >> (skipLog)), (Uint32)L_MaxProbeStep) + 1)

/* a candidate is only taken if it matches at least `need` bytes: test byte need-1 before counting, when both
   reads are in bounds. It rejects only candidates that the full count would reject. */
#define   CANNOT_REACH(need)   ((need) > 4 && srcPtr + (need) <= srcLastMatch && \
                                (!(dictSize && matchIdx < 0) || matchPtr + (need) <= dictLastMatch) && \
                                srcPtr[(need) - 1] != matchPtr[(need) - 1])


/* The window schedule and offset groups of one input, and the encoder's literal-run flag. Each compression state and
   each decoding call holds its own (so that WZIP_L runs in several threads at once); the functions reach it through a
   pointer S_, under the names below. */
typedef struct {
	int   offWidth[9];                                 /* the windows of the input (its offset codes), as decoders derive them */
	int   srchWidth[9];                                /* the encoder's: offWidth capped by the level's window */
	int   nHufMchOff[MaxMchOffGroup];
	int   mchOffGroup;
	int   offGroupsFine;                               /* 1: the eight-group layout (lengths 3, 4, 5, 6, 7, 8-9, 10-15, 16+) */
	Uint8 offGroupOf[N_HufMchLen];                     /* length symbol -> offset group */
	int   litRunTooLong;                               /* set by the encoder: a literal run the format cannot code */
} WZL_Sched;
#define   OffWidth             (S_->offWidth)
#define   SrchWidth            (S_->srchWidth)
#define   N_HufMchOff          (S_->nHufMchOff)
#define   MchOffGroup          (S_->mchOffGroup)
#define   OffGroupsFine        (S_->offGroupsFine)
#define   OffGroupOf           (S_->offGroupOf)
#define   LitRunTooLong        (S_->litRunTooLong)
#define   SCHED(wzipStr)       WZL_Sched* const S_ = (WZL_Sched*)(wzipStr)->sched

/* Offset groups, contiguous ranges of length symbols: by default one per length below the widest window (natural) and
   one for the rest; the fine layout splits the long lengths further. Each group's offset alphabet covers the widest
   window among its lengths. The decoder computes the group as min(m, cap) + (m >= 7) + (m >= 13), the last two for
   the fine layout only. */
static void Set_Offset_Groups(WZL_Sched* const S_, int natural, int fine)
{
	static const int fineStart[8] = { 0, 1, 2, 3, 4, 5, 7, 13 };
	int start[MaxMchOffGroup], k = 0;
	if (fine)
		for (k = 0; k < 8; k++) start[k] = fineStart[k];
	else
		for (k = 0; k < natural && k < MaxMchOffGroup; k++) start[k] = k;
	MchOffGroup = k;
	for (int g = 0; g < k; g++) {
		const int end = g + 1 < k ? start[g + 1] : N_HufMchLen;
		for (int m = start[g]; m < end; m++) OffGroupOf[m] = (Uint8)g;
		const int lastLen = end - 1 + MinMatchLen < LitRunDirect ? end - 1 + MinMatchLen : 8;
		N_HufMchOff[g] = 2 * OffWidth[min(8, lastLen)];
	}
}

/* The window schedule follows the history: with a dictionary of D bytes before an input of n, the windows are those
   of an input of n + D bytes (as for the end of one stream that long), so that a match can reach as far into the
   dictionary as one stream's could; encoder and decoders derive them alike. */
static int WZL_History(const int n, const int dictSize)
{
	const long long h = (long long)n + (dictSize > 0 ? dictSize : 0);
	return h > 0x7FFFFFFF ? 0x7FFFFFFF : (int)h;
}

/* We combine 8-bit Huffman index and up-to 24 appended bits into Uint32, Therefore, expanding to more than 24 appended bits will break the code.
*/
static ExtHuffman_Lit const ExtHufLitRun[] = {
	{0, 0},   {1, 0},  {2, 0},  {3, 0},     {4, 0},  {5, 0},  {6, 0},  {7, 0},      {8, 0},  {9, 0},  {10, 0}, {11, 0},      {12, 0}, {13, 0}, {14, 0}, {15, 0},
	{16, 0},  {17, 0}, {18, 0}, {19, 0},    {20, 0}, {21, 0}, {22, 0}, {23, 0},     {24, 0}, {25, 0}, {26, 0}, {27, 0},      {28, 0}, {29, 0}, {30, 0}, {31, 0},  
	{16, 1},  {17, 1}, {18, 1}, {19, 1},    {20, 1}, {21, 1}, {22, 1}, {23, 1},     {24, 1}, {25, 1}, {26, 1}, {27, 1},      {28, 1}, {29, 1}, {30, 1}, {31, 1},      /* 32-47:  32 - 63 */
	{8, 3},  {9, 3},  {10, 3},  {11, 3},    {12, 3},  {13, 3}, {14, 3}, {15, 3},             /* 48-55:  64 - 127 */
	{4, 5},  {5, 5}, {6, 5}, {7, 5},                                                         /* 56-59:  128 - 255 */
	{4, 6},  {5, 6}, {6, 6}, {7, 6},                                                         /* 60-63:  256 - 511 */
	{2, 8},  {3, 8},                                                                         /* 64-65:  512 - 1023 */
	{2, 9},  {3, 9}, 																		 /* 66-67:  1024 - 2047 */
	{2, 10}, {3, 10},																		 /* 68-69:  2048 - 4095 */
	{0, 24},                                                                                 /* 70:     2048 - 16M  */
};

static const ExtHuffman_Idx LitRunHufMap[] = {
	{1, 0},  {2, 0},  {4, 0},  {8, 0},       {16, 0}, {32, 1},  {48, 3},  {56, 5},     {60, 6}, {64, 8}, {66, 9},  {68, 10},  {70, 24},
};

static ExtHuffman_Lit const ExtHufMchLen[] = {
	                             {3, 0},      {4, 0}, {5, 0},  {6, 0}, {7, 0},      {8, 0}, {9, 0}, {10, 0}, {11, 0},       {12, 0}, {13, 0}, {14, 0}, {15, 0},
	{16, 0},  {17, 0}, {18, 0}, {19, 0},    {20, 0}, {21, 0}, {22, 0}, {23, 0},     {24, 0}, {25, 0}, {26, 0}, {27, 0},     {28, 0}, {29, 0}, {30, 0}, {31, 0},
	{16, 1},  {17, 1}, {18, 1}, {19, 1},    {20, 1}, {21, 1}, {22, 1}, {23, 1},     {24, 1}, {25, 1}, {26, 1}, {27, 1},      {28, 1}, {29, 1}, {30, 1}, {31, 1},      /* 32-47:  32 - 63 */
	{8, 3},  {9, 3},  {10, 3},  {11, 3},    {12, 3},  {13, 3}, {14, 3}, {15, 3},             /* 48-55:  64 - 127 */
	{4, 5},  {5, 5}, {6, 5}, {7, 5},                                                         /* 56-59:  128 - 255 */
	{4, 6},  {5, 6}, {6, 6}, {7, 6},                                                         /* 60-63:  256 - 511 */
	{2, 8},  {3, 8},                                                                         /* 64-65:  512 - 1023 */
	{2, 9},  {3, 9}, 																		 /* 66-67:  1024 - 2047 */
	{2, 10}, {3, 10},																		 /* 68-69:  2048 - 4095 */
};

static ExtHuffman_Lit const ExtHufMchOff[] = {
	{0, 0},  {1, 0},  {2, 0},  {3, 0},          {2, 1}, {3, 1},  {2, 2},  {3, 2},    
	{2, 3},  {3, 3},  {2, 4},  {3, 4},          {2, 5}, {3, 5},  {2, 6},  {3, 6},
	{2, 7},  {3, 7},  {2, 8},  {3, 8},          {2, 9}, {3, 9},  {2, 10}, {3, 10},
	{2, 11},  {3, 11}, {2, 12}, {3, 12},        {2, 13}, {3, 13}, {2, 14}, {3, 14},
	{2, 15},  {3, 15}, {2, 16}, {3, 16},        {2, 17}, {3, 17}, {2, 18}, {3, 18},
	{2, 19},  {3, 19}, {2, 20}, {3, 20},        {2, 21}, {3, 21}, {2, 22}, {3, 22},
	{2, 23},  {3, 23}, {2, 24}, {3, 24},        {2, 25}, {3, 25},
};

typedef struct {
	Huffman_Str litRunHuf[N_HufLitRun];
	Huffman_Str mchLenHuf[N_HufJoint];                /* joint literal-run class and match length */
	Huffman_Str mchOffHuf[MaxMchOffGroup][N_HufMchOffMax];
} WLZ_Huffman_Set;

typedef struct {
	HufCode_Str litRun[N_HufLitRun];
	HufCode_Str mchLen[N_HufJoint];
	HufCode_Str mchOff[MaxMchOffGroup][N_HufMchOffMax];
} WLZ_HufCode_Set;

typedef struct {
	Uint32 maxLitRunHufWt;
	Uint32 maxMchLenHufWt;
	Uint32 slotJoint;                                  /* the block's joint symbol carries the cache slot */
	Uint32 maxMchOffHufWt[MaxMchOffGroup];
	Uint8 litRunHufWt[N_HufLitRun];
	Uint8 mchLenHufWt[N_HufJoint];
	Uint8 mchOffHufWt[MaxMchOffGroup][N_HufMchOffMax];
} WLZ_HufWt_Set;


/* The codes a stream's sequence blocks last sent for each table (literal run, joint symbol, the offsets of each length
   group), which a later block may reuse instead of sending its own (section 5.4 of doc/WZIP_format.md) */
typedef struct {
	WLZ_HufCode_Set code;
	int haveLitRun, haveMchLen, haveOff[MaxMchOffGroup];
	int mchLenSlotJoint;                               /* the coding (slot-joint or classic) of the joint code */
} Seq_Prev;

static void Seq_Prev_Init(Seq_Prev* const prev)
{
	prev->haveLitRun = prev->haveMchLen = 0;
	memset(prev->haveOff, 0, sizeof(prev->haveOff));
}

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

/* Bytes that match at srcPtr against history starting in the dictionary at matchPtr. The dictionary is followed by the
   input (positions -D..-1, then 0..), so a match that reaches the dictionary's end goes on from the input's start, as
   both decoders copy it. srcLimit bounds the input side as in WLZ_Match_Count. */
ForceInlineTemplate Uint32 Hist_Match_Count(const Uint8* srcPtr, const Uint8* matchPtr, const Uint8* const srcLimit,
	const Uint8* const dictEnd, const Uint8* const source)
{
	Uint32 len = 0;
	while (matchPtr + REG_SIZE <= dictEnd && srcPtr < srcLimit) {      /* whole words inside the dictionary */
		const reg_t diff = MemReadARCH(srcPtr) ^ MemReadARCH(matchPtr);
		if (diff) return len + N_ZeroBytes(diff);
		srcPtr += REG_SIZE;
		matchPtr += REG_SIZE;
		len += REG_SIZE;
	}
	if (srcPtr >= srcLimit) return len;
	for (; matchPtr < dictEnd; srcPtr++, matchPtr++, len++)            /* its last bytes */
		if (*srcPtr != *matchPtr) return len;
	return len + WLZ_Match_Count(srcPtr, source, srcLimit, NULL);       /* on from the input's start */
}

/* the length of the match at srcPtr with history at matchPtr, in the dictionary or the input (dictSize, dictEnd and
   source in scope). A dictionary that ends where the input starts (a prefix in the same buffer) is counted through as
   one piece of memory. */
#define HIST_COUNT(srcPtr, matchPtr, srcLimit)  ((dictSize && (const Uint8*)dictEnd != (const Uint8*)source              \
		&& (const Uint8*)(matchPtr) >= (const Uint8*)dictEnd - dictSize && (const Uint8*)(matchPtr) < (const Uint8*)dictEnd) \
	? Hist_Match_Count((srcPtr), (matchPtr), (srcLimit), (const Uint8*)dictEnd, (const Uint8*)source)                     \
	: WLZ_Match_Count((srcPtr), (matchPtr), (srcLimit), NULL))

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

/* A run of `count` copies of the preceding byte (a distance-1 match not bound by the match-length cap): the run symbol,
   with count + OffCasheSize - 1 coded as an offset of the widest window. Runs leave the offset cache alone; a run
   straight after another (no literals between them) extends it, up to MaxRunCount (count <= MaxRunCount: the callers
   cap it). Returns the next free sequence. */
ForceInlineTemplate WLZ_Set* Store_Run(WZL_Sched* const S_, WLZ_Set* seq, const WLZ_Set* const seqStart, WLZ_Huffman_Set* const hs, Uint32 litRun, Uint32 count)
{
	const int group = OffGroupOf[RunSym];
	Uint32 prev = 0;
	if (litRun == 0 && seq > seqStart && (seq - 1)->mchLen == RunSym) {
		const Uint32 idx = (seq - 1)->mchOff & BitMask[OFF_SymBits], nb = ExtHufMchOff[idx].lsBits;
		prev = (((Uint32)ExtHufMchOff[idx].msValue << nb) | ((seq - 1)->mchOff >> OFF_SymBits)) - (OffCasheSize - 1);
	}
	if (prev && count <= MaxRunCount - prev) {
		seq--;
		count += prev;
		hs->mchOffHuf[group][seq->mchOff & BitMask[OFF_SymBits]].freq--;
	}
	else if (litRun < LitRunDirect)                      /* a new run: its literal run as the other sequences code it */
		seq->litRun = litRun;
	else {
		const int msb = High_Bit32(litRun);
		const int lr = msb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[msb].hufIdx + ((litRun ^ 1 << msb) >> LitRunHufMap[msb].lsBits);
		seq->litRun = lr ^ (litRun & BitMask[ExtHufLitRun[lr].lsBits]) << 8;
	}
	const Uint32 v = count + OffCasheSize - 1;
	const int idx = Offset_Huffman_Index((int)v, High_Bit32(v));
	seq->mchLen = RunSym;
	seq->mchOff = idx ^ (v & BitMask[ExtHufMchOff[idx].lsBits]) << OFF_SymBits;
	hs->mchOffHuf[group][idx].freq++;
	return seq + 1;
}

/* Codes the sequences into two bit streams, alternately: A (the block's sequences 0, 2, 4, ...) at zipBuffer and B
   (1, 3, 5, ...) at zipBufferB, so that a decoder decodes two at once; returns the size of A, *sizeB that of B */
ForceInlineTemplate Uint32 Huffman_Compress_Seq_Body(WZL_Sched* const S_, WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd, Uint8* zipBuffer,
	Uint8* zipBufferB, Uint32* sizeB, WLZ_HufCode_Set* hufCodeSet, const int slotJoint)
{
	Uint32 litRunHufIdx, mchLenHufIdx, offHufIdx, offGroup, nDone = 0;
	register Bit_Stream bitStream = { 0, 0, zipBuffer };      /* the stream of the next sequence */
	Bit_Stream otherStream = { 0, 0, zipBufferB };


#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "Huffman_Compress_Index.txt", "w");
#endif

	for (WLZ_Set* wlzSeqPtr = wlzSeq; wlzSeqPtr < wlzSeqEnd; wlzSeqPtr++) {
		litRunHufIdx = wlzSeqPtr->litRun &255;
		const int isLast = wlzSeqPtr->mchLen == 255;          /* protocal for ending: the last literal run only */
		mchLenHufIdx = isLast ? 0 : wlzSeqPtr->mchLen & 255;
		offHufIdx = wlzSeqPtr->mchOff & BitMask[OFF_SymBits];
		const Uint32 slotSel = !slotJoint || isLast || mchLenHufIdx == RunSym || offHufIdx >= OffCasheSize ? OffCasheSize : offHufIdx;
		const Uint32 jointIdx = slotJoint ? JointIdx(slotSel, LitClass(litRunHufIdx), mchLenHufIdx) : LitClass(litRunHufIdx) * N_HufMchLen + mchLenHufIdx;
		BITStream_Write(bitStream, hufCodeSet->mchLen[jointIdx].code, hufCodeSet->mchLen[jointIdx].nbits);
		if (litRunHufIdx >= 2) {
			BITStream_Write_Flush(bitStream);
			BITStream_Write(bitStream, hufCodeSet->litRun[litRunHufIdx].code, hufCodeSet->litRun[litRunHufIdx].nbits);
			if (litRunHufIdx >= LitRunDirect)
				BITStream_Write(bitStream, wlzSeqPtr->litRun >> 8, ExtHufLitRun[litRunHufIdx].lsBits);
			BITStream_Write_Flush(bitStream);
		}

#ifdef WZIP_DEBUG 
		if (litRunHufIdx >= LitRunDirect)
			fprintf(fptr, "lsBits=%d,  lsValue=%d;    ", ExtHufLitRun[litRunHufIdx].lsBits, lsValue);
#endif

		if (isLast)
			break;

#ifdef WZIP_DEBUG 
		fprintf(fptr, "matchLenIdx=%d,  ", mchLenHufIdx);
		if (mchLenHufIdx >= LitRunDirect - MinMatchLen)
			fprintf(fptr, "lsBits=%d,  lsValue=%d;      ", ExtHufMchLen[mchLenHufIdx].lsBits, mchLenLsValue);
#endif

		if (slotSel == OffCasheSize) {             /* the offset symbol, unless a slot-joint block has the slot in the joint */
			offGroup = OffGroupOf[mchLenHufIdx];
			BITStream_Write(bitStream, hufCodeSet->mchOff[offGroup][offHufIdx].code, hufCodeSet->mchOff[offGroup][offHufIdx].nbits);
			if (offHufIdx >= OffCasheSize)         /* append tail bits */
				BITStream_Write(bitStream, wlzSeqPtr->mchOff >> OFF_SymBits, ExtHufMchOff[offHufIdx].lsBits);
		}
		BITStream_Write_Flush(bitStream);

		if (mchLenHufIdx >= LitRunDirect - MinMatchLen && mchLenHufIdx != RunSym) {
			BITStream_Write(bitStream, wlzSeqPtr->mchLen >> 8, ExtHufMchLen[mchLenHufIdx].lsBits);
		}
		{ const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* the next sequence: the other stream */
		nDone++;


#ifdef WZIP_DEBUG 
		fprintf(fptr, "OffGrp=%d,   matchOffHufIdx=%d,  ", offGroup, offHufIdx);
		if (offHufIdx >= 4)
			fprintf(fptr, "lsBits=%d,  lsValue=%d\n", ExtHufMchOff[offHufIdx].lsBits, lsValue);
		else fprintf(fptr, "\n");

		fflush(fptr);
#endif

	}
	if (nDone & 1) { const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* bitStream: A */
	BITStream_Write_FlushEnd(bitStream);
	BITStream_Write_FlushEnd(otherStream);
	*sizeB = (Uint32)(otherStream.streamPtr - zipBufferB);

#ifdef WZIP_DEBUG 
	fclose(fptr);
#endif
	return (Uint32)(bitStream.streamPtr - zipBuffer);
}


static Uint32 Huffman_Compress_Seq_Kernel(WZL_Sched* const S_, WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd, Uint8* zipBuffer,
	Uint8* zipBufferB, Uint32* sizeB, WLZ_HufCode_Set* hufCodeSet, const int slotJoint)
{
	return slotJoint ? Huffman_Compress_Seq_Body(S_, wlzSeq, wlzSeqEnd, zipBuffer, zipBufferB, sizeB, hufCodeSet, 1)
	                 : Huffman_Compress_Seq_Body(S_, wlzSeq, wlzSeqEnd, zipBuffer, zipBufferB, sizeB, hufCodeSet, 0);
}

/* coded size of a table's symbols in bits, with about 4 bits of table header per used symbol */
static Uint64 Table_Bits(const Huffman_Str* h, const Uint32 n, const Uint32 cap)
{
	HufCode_Str code[N_HufJoint];
	Build_Huffman_Table((Huffman_Str*)h, n, cap, code);
	Uint64 bits = 0;
	for (Uint32 k = 0; k < n; k++)
		if (h[k].freq) bits += (Uint64)h[k].freq * code[k].nbits + 4;
	return bits;
}

/* Second Pass:  Apply Huffman encoding on top of WLZ compression */
ForceInlineTemplate Uint32 Huffman_Compress_WLZ(WZL_Sched* const S_, WLZ_Set* wlzSeq, WLZ_Set* const wlzSeqEnd,
	Uint8* wzipBuffer, Uint32 wzipBufSize,
	WLZ_Huffman_Set* huffmanSet, Seq_Prev* const prev)
{
	int i;
	Uint8* wzipBufPtr;
	Bit_Stream bitStream = { 0, 0, wzipBuffer };

	WLZ_HufCode_Set hufCodeSet;
	Huffman_Str hufWtHuf[MAX_HufWeight + 3] = { 0 };
	HufCode_Str hufWtHufCode[MAX_HufWeight + 3];
	Uint8 hufWtSet[MAX_HufSize * 2 + N_HufLitRun + N_HufJoint + N_HufMchOffMax * MaxMchOffGroup];
	Uint32 hufWtSetSize = 0;
	/* joint (literal-run class, match length) and literal-run (runs of two or more) frequencies of this block */
	memset(huffmanSet->litRunHuf, 0, sizeof(huffmanSet->litRunHuf));
	memset(huffmanSet->mchLenHuf, 0, sizeof(huffmanSet->mchLenHuf));
	memset(huffmanSet->mchOffHuf, 0, sizeof(huffmanSet->mchOffHuf));
	/* counts of both codings: slot-joint, and classic (joint = class x length, cache slots in the offset tables) */
	Huffman_Str jointClassic[N_HufJointClassic] = { 0 };
	Uint32 slotCnt[MaxMchOffGroup][OffCasheSize] = { 0 };
	for (WLZ_Set* q = wlzSeq; q < wlzSeqEnd; q++) {
		const Uint32 l = q->litRun & 255, last = q->mchLen == 255, m = last ? 0 : q->mchLen & 255, o = q->mchOff & BitMask[OFF_SymBits];
		const Uint32 slotSel = last || m == RunSym || o >= OffCasheSize ? OffCasheSize : o;
		huffmanSet->mchLenHuf[JointIdx(slotSel, LitClass(l), m)].freq++;
		jointClassic[LitClass(l) * N_HufMchLen + m].freq++;
		if (!last && slotSel == OffCasheSize) huffmanSet->mchOffHuf[OffGroupOf[m]][o].freq++;
		if (!last && slotSel < OffCasheSize) slotCnt[OffGroupOf[m]][slotSel]++;
		if (l >= 2) huffmanSet->litRunHuf[l].freq++;
	}
	for (i = 0; i < 2; i++) {                                /* two used symbols at least, for well-formed tables */
		if (!huffmanSet->litRunHuf[2 + i].freq) huffmanSet->litRunHuf[2 + i].freq = 1;
		if (!huffmanSet->mchLenHuf[i].freq) huffmanSet->mchLenHuf[i].freq = 1;
		if (!jointClassic[i].freq) jointClassic[i].freq = 1;
	}
	/* slot-joint only when it is smaller by more than 1/64: on blocks where it gains little, the classic decoder is
	   faster (a smaller joint table to read and build per block) */
	Uint64 sjBits = Table_Bits(huffmanSet->mchLenHuf, N_HufJoint, CapHufMchLenBits);
	Uint64 clBits = Table_Bits(jointClassic, N_HufJointClassic, CapHufMchLenBits);
	for (i = 0; i < MchOffGroup; i++) {
		sjBits += Table_Bits(huffmanSet->mchOffHuf[i], N_HufMchOff[i], CapHufMchOffBits);
		for (int k = 0; k < OffCasheSize; k++) huffmanSet->mchOffHuf[i][k].freq += slotCnt[i][k];
		clBits += Table_Bits(huffmanSet->mchOffHuf[i], N_HufMchOff[i], CapHufMchOffBits);
	}
	const int slotJoint = sjBits + (clBits >> 6) < clBits;
	if (slotJoint) {
		for (i = 0; i < MchOffGroup; i++)
			for (int k = 0; k < OffCasheSize; k++) huffmanSet->mchOffHuf[i][k].freq -= slotCnt[i][k];
	}
	else memcpy(huffmanSet->mchLenHuf, jointClassic, sizeof(jointClassic));
	const Uint32 nJoint = slotJoint ? N_HufJoint : N_HufJointClassic;
	Build_Huffman_Table(huffmanSet->litRunHuf, N_HufLitRun, CapHufLitRunBits, hufCodeSet.litRun);
	Build_Huffman_Table(huffmanSet->mchLenHuf, nJoint, CapHufMchLenBits, hufCodeSet.mchLen);
	for (i = 0; i < MchOffGroup; i++) {
		Build_Huffman_Table(huffmanSet->mchOffHuf[i], N_HufMchOff[i], CapHufMchOffBits, hufCodeSet.mchOff[i]);
	}

#ifdef WZIP_DEBUG 
	FILE* fptr = NULL;
	fopen_s(&fptr, "WZIP_Huffman_Trees.txt", "w");
	fprintf(fptr, "\nLiteral Run Huffman Tree\n");
	for (i = 0; i < N_HufLitRun; i++) {
		fprintf(fptr, "(%4d,  %2d),  ", huffmanSet->litRunHuf[i].freq, huffmanSet->litRunHuf[i].freq > 0 ? hufCodeSet.litRun[i].nbits : 0);
		if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\nMatch Length Huffman Tree\n");
	for (i = 0; i < N_HufMchLen; i++) {
		fprintf(fptr, "(%4d,  %2d),  ", huffmanSet->mchLenHuf[i].freq, huffmanSet->mchLenHuf[i].freq > 0 ? hufCodeSet.mchLen[i].nbits : 0);
		if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\nMatch Offset Huffman Trees\n");
	for (int n = 0; n < MchOffGroup; n++) {
		for (i = 0; i < (int)N_HufMchOff[n]; i++) {
			fprintf(fptr, "(%4d,  %4d),  ", huffmanSet->mchOffHuf[n][i].freq, huffmanSet->mchOffHuf[n][i].freq > 0 ? hufCodeSet.mchOff[n][i].nbits : 0);
			if (0 == ((i + 1) & 7)) fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n\n");
	}

	fclose(fptr);
#endif

	/* Each table either sends its code's lengths or reuses the stream's last code of the table (prev), whichever takes
	   fewer bits: the block's symbols under the previous code, against its symbols under its own code plus the
	   lengths of that code (under the weight code of all the tables). Blocks that reuse a code thus also spare the
	   decoder building its table. */
	const Uint32 nTab = 2 + MchOffGroup;                    /* literal run, joint, offset groups */
	const Huffman_Str* tabFreq[2 + MaxMchOffGroup];
	HufCode_Str* tabCode[2 + MaxMchOffGroup];
	HufCode_Str* tabPrev[2 + MaxMchOffGroup];
	Uint32 tabSize[2 + MaxMchOffGroup], segStart[2 + MaxMchOffGroup + 1], reuse[2 + MaxMchOffGroup];
	int tabHave[2 + MaxMchOffGroup];
	tabFreq[0] = huffmanSet->litRunHuf; tabCode[0] = hufCodeSet.litRun; tabPrev[0] = prev->code.litRun;
	tabSize[0] = N_HufLitRun; tabHave[0] = prev->haveLitRun;
	tabFreq[1] = huffmanSet->mchLenHuf; tabCode[1] = hufCodeSet.mchLen; tabPrev[1] = prev->code.mchLen;
	tabSize[1] = nJoint; tabHave[1] = prev->haveMchLen && prev->mchLenSlotJoint == slotJoint;
	for (i = 0; i < MchOffGroup; i++) {
		tabFreq[2 + i] = huffmanSet->mchOffHuf[i]; tabCode[2 + i] = hufCodeSet.mchOff[i]; tabPrev[2 + i] = prev->code.mchOff[i];
		tabSize[2 + i] = N_HufMchOff[i]; tabHave[2 + i] = prev->haveOff[i];
	}
	segStart[0] = 0;
	for (Uint32 t = 0; t < nTab; t++)
		segStart[t + 1] = segStart[t] + Count_Huffman_Weight_Frequency(tabCode[t], tabSize[t], hufWtHuf, hufWtSet + segStart[t]);
	Build_Huffman_Table(hufWtHuf, MAX_HufWeight + 3, MAX_HufHufWt, hufWtHufCode);
	Uint64 bodyBits = 0;
	int nSent = 0;
	for (Uint32 t = 0; t < nTab; t++) {
		const Uint64 own = Huffman_Code_Bits(tabFreq[t], tabCode[t], tabSize[t])
		                 + Huffman_Header_Bits(hufWtHufCode, hufWtSet + segStart[t], (int)(segStart[t + 1] - segStart[t]));
		const Uint64 old = tabHave[t] ? Huffman_Code_Bits(tabFreq[t], tabPrev[t], tabSize[t]) : ((Uint64)-1);
		reuse[t] = old <= own;
		bodyBits += reuse[t] ? old : own;
		nSent += !reuse[t];
	}

	BITStream_Write(bitStream, (Uint32)slotJoint, 1);          /* the block's coding */
	for (Uint32 t = 0; t < nTab; t++)
		BITStream_Write(bitStream, reuse[t], 1);              /* each table: its own code, or the last one */
	BITStream_Write_Flush(bitStream);
	if (nSent) {                                         /* the weight code of the tables sent, and their lengths */
		memset(hufWtHuf, 0, sizeof(hufWtHuf));
		hufWtSetSize = 0;
		for (Uint32 t = 0; t < nTab; t++)
			if (!reuse[t]) hufWtSetSize += Count_Huffman_Weight_Frequency(tabCode[t], tabSize[t], hufWtHuf, hufWtSet + hufWtSetSize);
		Build_Huffman_Table(hufWtHuf, MAX_HufWeight + 3, MAX_HufHufWt, hufWtHufCode);
		Write_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, hufWtHufCode);
		Write_Huffman_Header_byHuffman(&bitStream, hufWtHufCode, hufWtSet, hufWtSetSize);
	}
	BITStream_Write_FlushEnd(bitStream);
	wzipBufPtr = bitStream.streamPtr;
	for (Uint32 t = 0; t < nTab; t++) {                  /* the codes the block uses; those sent become the last */
		if (reuse[t]) memcpy(tabCode[t], tabPrev[t], tabSize[t] * sizeof(HufCode_Str));
		else memcpy(tabPrev[t], tabCode[t], tabSize[t] * sizeof(HufCode_Str));
	}
	if (!reuse[0]) prev->haveLitRun = 1;
	if (!reuse[1]) { prev->haveMchLen = 1; prev->mchLenSlotJoint = slotJoint; }
	for (i = 0; i < MchOffGroup; i++)
		if (!reuse[2 + i]) prev->haveOff[i] = 1;
	assert((Uint64)(wzipBufPtr - wzipBuffer) + 3 + bodyBits / 8 <= wzipBufSize);   /* the callers give it a whole block of room (SEQ_BlockBound), twice */

	/* the size of stream A (u24), A, then B, coded behind the buffer's first SEQ_BlockBound bytes and moved up */
	Uint32 sizeB;
	Uint8* const scratchB = wzipBuffer + wzipBufSize;
	const Uint32 sizeA = Huffman_Compress_Seq_Kernel(S_, wlzSeq, wlzSeqEnd, wzipBufPtr + 3, scratchB, &sizeB, &hufCodeSet, slotJoint);
	wzipBufPtr[0] = (Uint8)sizeA; wzipBufPtr[1] = (Uint8)(sizeA >> 8); wzipBufPtr[2] = (Uint8)(sizeA >> 16);
	wzipBufPtr += 3 + sizeA;
	memmove(wzipBufPtr, scratchB, sizeB);
	wzipBufPtr += sizeB;

	return (Uint32)(wzipBufPtr - wzipBuffer);
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fast compression without using hash-chain  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

typedef struct wlz_match {
	Uint32 len;                                        // LZ match length
	Uint32 off;                                        // LZ match offset/distance
} WLZ_Match;

/* Parsing by gain = G_Byte * length - offset cost. The offset cost is log2(offset) bits, or 0 for an offset in the
   repeat cache, which is coded by its cache slot. Starting a match 1, 2 or 3 bytes later costs G_Delay[] for the
   literals in between. */
#define   G_Byte               4
static const int G_Delay[4] = { 0, 6, 12, 18 };

ForceInlineTemplate int Offset_Cost(Uint32 offset, const Uint32* const lastOffset)
{
	if (offset == lastOffset[0] || offset == lastOffset[1] || offset == lastOffset[2] || offset == lastOffset[3]) return 0;
	return (int)High_Bit32(offset);
}

ForceInlineTemplate int Match_Gain(const WLZ_Match* const m, const Uint32* const lastOffset)
{
	return G_Byte * (int)m->len - Offset_Cost(m->off, lastOffset);
}

/* chain candidates come in increasing distance: a longer one is taken only if its extra length pays for its extra
   offset bits */
ForceInlineTemplate int Pays_Off(int matchLen, Uint32 matchDist, const WLZ_Match* const best, const Uint32* const lastOffset)
{
	return (int)best->len < MinMatchLen ||
		G_Byte * (matchLen - (int)best->len) > Offset_Cost(matchDist, lastOffset) - Offset_Cost(best->off, lastOffset);
}

/* Length of the match at distance `offset` from srcIdx, 0 when out of history. A match in the dictionary may run on
   into the input. */
ForceInlineTemplate int Repeat_Match_Len(const Uint8* const source, Uint32 srcIdx, Uint32 offset, const int dictSize, const Uint8* const dictEnd,
	const Uint8* const srcLastMatch, const Uint8* const dictLastMatch)
{
	(void)dictLastMatch;
	if (offset == 0 || offset > srcIdx + (Uint32)dictSize) return 0;
	const int histIdx = (int)srcIdx - (int)offset;
	const Uint8* const srcPtr = source + srcIdx;
	if (histIdx < 0 && dictEnd != source) return (int)Hist_Match_Count(srcPtr, dictEnd + histIdx, srcLastMatch, dictEnd, source);
	const Uint8* const matchPtr = srcPtr - offset;                /* in the input, or in a dictionary just before it */
	const reg_t diff = MemReadARCH(srcPtr) ^ MemReadARCH(matchPtr);
	if (diff) return (int)N_ZeroBytes(diff);
	return REG_SIZE + (int)WLZ_Match_Count(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch, NULL);
}

/* A match of whole words read at once (diff 0): on 64-bit targets 8 bytes or more, in the widest window, which the
   candidates were taken from; with the 4-byte words of 32-bit targets it may be shorter than hash2Len, whose window
   is narrower (constant true on 64-bit targets) */
#define WORD_MATCH_FITS(len, off)   (REG_SIZE >= 8 || (len) > hash2Len || (off) < WINDOW(SrchWidth[len]))

/* The first position from p on (before end) whose next MinMatchLen bytes or more repeat at an offset below `window`
   (the window of the shortest matches, so the match is valid at any length): it ends a literal run that the fast
   levels' probes have not ended before the run outgrows the format. Each position is looked up in a table of the last
   position per hash of its first 3 bytes, then entered. NULL if there is none. */
static const Uint8* L_Run_Break(const Uint8* const source, const Uint8* p, const Uint8* const end, const Uint32 window,
	Uint32* const len, Uint32* const off)
{
	Uint32 last[1 << 12];
	memset(last, 0xFF, sizeof(last));
	for (; p < end; p++) {
		const Uint32 pos = (Uint32)(p - source);
		const Uint32 h = (((Uint32)p[0] | (Uint32)p[1] << 8 | (Uint32)p[2] << 16) * 2654435761u) >> 20;
		const Uint32 cand = last[h];
		last[h] = pos;
		if (cand != 0xFFFFFFFF && pos - cand < window && 0 == memcmp(source + cand, p, MinMatchLen)) {
			*off = pos - cand;
			*len = MinMatchLen + WLZ_Match_Count(p + MinMatchLen, source + cand + MinMatchLen, end, NULL);
			return p;
		}
	}
	return NULL;
}

ForceInlineTemplate Uint32 WLZ2_Compress_Fast(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	Uint8* wzipStream,
	int wzipCapSize)
{
	SCHED(wzipStr);
	Uint32 i;
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	const int dictSize = wzipStr->dictSize;
	const Uint8*  dictEnd = wzipStr->dictEnd;
	Uint32 nLzLits;

	Huffman_Str litHuf[N_HufLits];
	Uint8* wzipLitPtr = wzipStream;
	const Uint8* const wzipLitEnd = wzipLitPtr + wzipCapSize;
	wzipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32  zipLitBlkSize;
	Uint8* lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);    
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* lzLitEnd = lzLitBuffer + HUF_BlockSize;
	Uint32 hashV1, hashV2;
	int  match1Idx, match2Idx;
	int  lazyMatchLen, lazyMatchOffset, lazyMatchFail;
	int matchLen, matchOffset;
	int matchLen2, matchOffset2;
	int lastOffset[OffCasheSize];
	int litRun;
	int* hash1Table = (int *)wzipStr->hash1Table;
	int* hash2Table = (int *)wzipStr->hash2Table;
	const Uint32 offWindow = WINDOW(SrchWidth[8]);
	int litRunMsb, litRunHufIdx, matchLenMsb, mchLenHufIdx, offsetMsb, offsetHufIdx;
	const int hash1Len = wzipStr->hash1Len;
	const int hash2Len = wzipStr->hash2Len;

	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	reg_t currPattern, diffPattern;

	const Uint8* matchPtr;
	WLZ_Huffman_Set huffmanSet;
	Huffman_Prev litPrev;                              /* the literal code that later blocks may reuse */
	litPrev.valid = 0;
	Seq_Prev seqPrev;                                  /* the sequence codes that later blocks may reuse */
	Seq_Prev_Init(&seqPrev);
	memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));

	WLZ_Set* const wlzSeq = (WLZ_Set*)malloc(SEQ_BlockSize*sizeof(WLZ_Set));
	WLZ_Set* wlzSeqPtr = wlzSeq;
	WLZ_Set* const wlzSeqEnd = wlzSeq + SEQ_BlockSize;

	Uint8* wlzStream = (Uint8*)malloc(2 * SEQ_BlockBound + 64);         /* one coded block of sequences, and room for its stream B */
	Uint8* const litScratch = (Uint8*)malloc(HUF_BlockSize + LIT_BlockSlack + 64);
	Seq_Stack seqStack;
	const int stackOk = Seq_Stack_Init(&seqStack, wzipStream + wzipCapSize);
	if (NULL == lzLitBuffer || NULL == wlzSeq || NULL == wlzStream || NULL == litScratch || !stackOk) goto _lit_overflow;

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "WLZ2_Compress_Index.txt", "w");
	fprintf(fptr, "WLZ2_Compress_Fast: srcSize=%i\n", srcSize);
#endif

	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	nLzLits = 0;
	Uint32 srcIdx = 0;
	for (i = 0; i < OffCasheSize; i++ )
		lastOffset[i] = -1;                     /* unset (0xFFFFFFFF): above every offset, so never a hit */

	Uint32 runBreakAt = L_RunBreakAt;
	while (1) {

		while (1) {
			if (unlikely(srcPtr >= srcLastMatch)) goto _last_literals;
			if (unlikely((Uint32)(srcPtr - anchor) >= runBreakAt)) {     /* a literal run nearing the format's limit */
				Uint32 bLen, bOff;
				const Uint8* const b = L_Run_Break(source, srcPtr, srcLastMatch, WINDOW(SrchWidth[MinMatchLen]), &bLen, &bOff);
				if (NULL == b) runBreakAt = 0xFFFFFFFF;     /* none to the end: the run grows too long, and the input is stored */
				else {
					while (srcPtr < b) {
						*lzLitPtr++ = *srcPtr;
						litHuf[*srcPtr].freq++;
						srcPtr++;
						srcIdx++;
						if (lzLitPtr == lzLitEnd) {
							LIT_PUT_BLOCK(HUF_BlockSize);
							wzipLitPtr += zipLitBlkSize;
							lzLitPtr = lzLitBuffer;
							memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
							nLzLits += HUF_BlockSize;
						}
					}
					matchLen = (int)bLen;
					matchOffset = (int)bOff;
					break;
				}
			}

			hashV1 = WLZ_Hash1(srcPtr) & wzipStr->hash1Mask;
			hashV2 = WLZ_Hash2(srcPtr) & wzipStr->hash2Mask;

			match1Idx = hash1Table[hashV1];
			hash1Table[hashV1] = srcIdx;

			match2Idx = hash2Table[hashV2];
			hash2Table[hashV2] = srcIdx;

			matchLen = matchLen2 = matchOffset2 = 0;
			currPattern = MemReadARCH(srcPtr);
			matchOffset = srcIdx - match2Idx;
			if ( match2Idx >= -dictSize && matchOffset>0 && matchOffset < offWindow ) {
				matchPtr = (dictSize && match2Idx < 0) ? dictEnd + match2Idx : srcPtr - matchOffset;
				
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				if (0 == diffPattern) {
					matchLen = REG_SIZE + HIST_COUNT(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch);
					if (WORD_MATCH_FITS(matchLen, matchOffset)) break;
					matchLen = 0;
				}
				else {
					matchLen = N_ZeroBytes(diffPattern);
					if ( matchLen <= hash2Len && matchOffset>= WINDOW(SrchWidth[matchLen]) )
						matchLen = 0;
				}
				matchLen2 = matchLen;
				matchOffset2 = matchOffset;
			}

			matchOffset = srcIdx - match1Idx;
			if ( match1Idx>=-dictSize && matchOffset > 0 && matchOffset < offWindow ) {
				matchPtr = (dictSize && match1Idx < 0) ? dictEnd + match1Idx : srcPtr - matchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);
				if (0 == diffPattern) {
					matchLen = REG_SIZE + HIST_COUNT(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch);
					if (WORD_MATCH_FITS(matchLen, matchOffset)) break;
					matchLen = 0;
				}
				else {
					matchLen = N_ZeroBytes(diffPattern);
					if ( matchLen <= hash2Len && matchOffset >= WINDOW(SrchWidth[matchLen]) )
						matchLen = 0;
				}
				if (matchLen > matchLen2) {
					matchLen2 = matchLen;
					matchOffset2 = matchOffset;
				}
			}

			matchLen = matchLen2;
			matchOffset = matchOffset2;
			{   /* the most recent offset: no offset bits, and not bound by the length windows */
				const int repLen = Repeat_Match_Len(source, srcIdx, (Uint32)lastOffset[0], dictSize, dictEnd, srcLastMatch, dictLastMatch);
				if (repLen >= MinMatchLen && (matchLen < MinMatchLen || G_Byte * repLen > G_Byte * matchLen - Offset_Cost((Uint32)matchOffset, (const Uint32*)lastOffset))) {
					matchLen = repLen;
					matchOffset = lastOffset[0];
				}
			}
			if (matchLen >= MinMatchLen)
				break;

			/* no match: emit literals up to the next probe, which moves further apart in long literal runs */
			const Uint8* const nextProbe = srcPtr + L_PROBE_STEP(srcPtr - anchor, L0_SkipLog);
			do {
				*lzLitPtr++ = *srcPtr;
				litHuf[*srcPtr].freq++;
				srcPtr++;
				srcIdx++;
				if (lzLitPtr == lzLitEnd) {
					LIT_PUT_BLOCK(HUF_BlockSize);
					wzipLitPtr += zipLitBlkSize;
					lzLitPtr = lzLitBuffer;
					memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
					nLzLits += HUF_BlockSize;
				}
			} while (srcPtr < nextProbe && srcPtr < srcLastMatch);
		}

		lazyMatchFail = 1;
		srcPtr++;
		srcIdx++;
		currPattern = MemReadARCH(srcPtr);

		hashV1 = WLZ_Hash1(srcPtr) & wzipStr->hash1Mask;
		hashV2 = WLZ_Hash2(srcPtr) & wzipStr->hash2Mask;
		match1Idx = hash1Table[hashV1];
		hash1Table[hashV1] = srcIdx;
		match2Idx = hash2Table[hashV2];
		hash2Table[hashV2] = srcIdx;
		
		lazyMatchOffset = srcIdx - match2Idx;
		if ( match2Idx>=-dictSize && lazyMatchOffset>0 && lazyMatchOffset < offWindow ) {
			matchPtr = (dictSize && match2Idx < 0) ? dictEnd + match2Idx : srcPtr - lazyMatchOffset;
			diffPattern = currPattern ^ MemReadARCH(matchPtr);

			if (0 == diffPattern) {
				lazyMatchLen = REG_SIZE + HIST_COUNT(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch);
				if (!WORD_MATCH_FITS(lazyMatchLen, lazyMatchOffset)) lazyMatchLen = 0;
			}
			else {
				lazyMatchLen = N_ZeroBytes(diffPattern);
				if (lazyMatchOffset >= WINDOW(SrchWidth[lazyMatchLen]) ) lazyMatchLen = 0;
			}

			if (lazyMatchLen > matchLen) {
				matchLen = lazyMatchLen;
				matchOffset = lazyMatchOffset;
				lazyMatchFail = 0;
			}
		}
		if (lazyMatchFail) {
			lazyMatchOffset = srcIdx - match1Idx;
			if (match1Idx >= -dictSize && lazyMatchOffset > 0 && lazyMatchOffset < offWindow) {
				matchPtr = (dictSize && match1Idx < 0) ? dictEnd + match1Idx : srcPtr - lazyMatchOffset;
				diffPattern = currPattern ^ MemReadARCH(matchPtr);

				if (0 == diffPattern) {
					lazyMatchLen = REG_SIZE + HIST_COUNT(srcPtr + REG_SIZE, matchPtr + REG_SIZE, srcLastMatch);
					if (!WORD_MATCH_FITS(lazyMatchLen, lazyMatchOffset)) lazyMatchLen = 0;
				}
				else {
					lazyMatchLen = N_ZeroBytes(diffPattern);
					if (lazyMatchOffset >= WINDOW(SrchWidth[lazyMatchLen]) ) lazyMatchLen = 0;
				}

				if (lazyMatchLen > matchLen) {
					matchLen = lazyMatchLen;
					matchOffset = lazyMatchOffset;
					lazyMatchFail = 0;
				}
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
				LIT_PUT_BLOCK(HUF_BlockSize);
				wzipLitPtr += zipLitBlkSize;
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
				nLzLits += HUF_BlockSize;
			}
		}

		matchLen = min(matchLen, MaxMatchLen);
		
		/* index the start and the last two positions of the match (the hashes read 8 bytes: stop 16 before the end) */
		for (i = 2; i < matchLen && srcPtr + i <= srcLastMatch; i = (i + 3 < matchLen) ? matchLen - 2 : i + 1) {
			hashV1 = WLZ_Hash1(srcPtr + i) & wzipStr->hash1Mask;
			hashV2 = WLZ_Hash2(srcPtr + i) & wzipStr->hash2Mask;
			hash1Table[hashV1] = srcIdx + i;
			hash2Table[hashV2] = srcIdx + i;
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
		if (litRun < LitRunDirect) {
			wlzSeqPtr->litRun = litRun;
			huffmanSet.litRunHuf[litRun].freq++;
		}
		else {
			litRunMsb = High_Bit32(litRun);
			litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
			wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits])<<8;
			huffmanSet.litRunHuf[litRunHufIdx].freq++;
		}

		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Fast Encode Match Pair  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		if (matchOffset == 1) {                  /* a run is not bound by the match-length cap */
			while (srcPtr + matchLen < srcLastMatch && srcPtr[matchLen] == srcPtr[matchLen - 1]) matchLen++;
			if (matchLen > MaxRunCount) matchLen = MaxRunCount;
		}
		srcIdx += matchLen;
		srcPtr += matchLen;

		if (matchOffset == 1) wlzSeqPtr = Store_Run(S_, wlzSeqPtr, wlzSeq, &huffmanSet, litRun, matchLen);
		else {
		matchOffset = Offset_Cashe((Uint32*)lastOffset, matchOffset);
#ifdef WZIP_DEBUG
		fprintf(fptr, "-->  matchLen=%d,  matchOff=%d\n", matchLen, matchOffset);
		fflush(fptr);
#endif
		if (matchLen < LitRunDirect ) {
			mchLenHufIdx = matchLen - MinMatchLen;
			wlzSeqPtr->mchLen = mchLenHufIdx;
			huffmanSet.mchLenHuf[mchLenHufIdx].freq++;
		}
		else {
			matchLenMsb = High_Bit32(matchLen);
			mchLenHufIdx = LitRunHufMap[matchLenMsb].hufIdx + ((matchLen ^ 1 << matchLenMsb) >> LitRunHufMap[matchLenMsb].lsBits) - MinMatchLen;
			wlzSeqPtr->mchLen = mchLenHufIdx ^ (matchLen & BitMask[LitRunHufMap[matchLenMsb].lsBits])<<8;
			huffmanSet.mchLenHuf[mchLenHufIdx].freq++;
		}
		
		if (matchOffset < 4) {
			wlzSeqPtr->mchOff = matchOffset;
			huffmanSet.mchOffHuf[OffGroupOf[mchLenHufIdx]][matchOffset].freq++;
		}
		else {
			offsetMsb = High_Bit32(matchOffset);
			offsetHufIdx = Offset_Huffman_Index(matchOffset, offsetMsb);
			wlzSeqPtr->mchOff = offsetHufIdx ^ (matchOffset & BitMask[ExtHufMchOff[offsetHufIdx].lsBits]) << OFF_SymBits;
			//huffmanSet.mchOffHuf[OffsetGroupTable[min(15, mchLenHufIdx)]][offsetHufIdx].freq++;
			huffmanSet.mchOffHuf[OffGroupOf[mchLenHufIdx]][offsetHufIdx].freq++;
		}

		wlzSeqPtr++;
		}
		anchor = srcPtr;
		matchLen = 0;

		if (wlzSeqPtr == wlzSeqEnd) {
			SEQ_PUT_BLOCK();
			memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));
			wlzSeqPtr = wlzSeq;
		}
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
		LIT_PUT_BLOCK(HUF_BlockSize);
		wzipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);     /* remaining number literals in the buffer to be flushed */
	if (lastBufLits) {                                   /* none when the literals filled their last block exactly */
		LIT_PUT_BLOCK(lastBufLits);
		wzipLitPtr += zipLitBlkSize;
	}
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(wzipStream, nLzLits);       /* record the number of LZ literals at the beginning of zip stream */
	else               MemWriteLE2(wzipStream, (Uint16)nLzLits);

#ifdef WZIP_DEBUG
	fprintf(fptr, "srcIdx=%d, litRun=%d\n", (int)(anchor - (const Uint8*)source), litRun);
	fclose(fptr);
#endif

	if (litRun < LitRunDirect) {
		wlzSeqPtr->litRun = litRun;
		huffmanSet.litRunHuf[litRun].freq++;
	}
	else {
		litRunMsb = High_Bit32(litRun);
		litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
		wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits])<<8;
		huffmanSet.litRunHuf[litRunHufIdx].freq++;
	}
	wlzSeqPtr->mchLen = 255;   /* protocal for ending */
	wlzSeqPtr++;

	SEQ_PUT_BLOCK();
	int cmprSize = (int)((wzipLitPtr - wzipStream) + Seq_Stack_Place(&seqStack, wzipLitPtr));

	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return cmprSize;
_lit_overflow:                 /* the output buffer is full: the caller stores the input raw */
	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return 0;
}

/* Literal frequencies of a block, counted in four tables so that repeated bytes do not wait on each other */
static void Literal_Histogram(const Uint8* buf, Uint32 n, Huffman_Str* litHuf)
{
	Uint32 c[4][256] = { { 0 } };
	Uint32 k = 0;
	for (; k + 4 <= n; k += 4) {
		c[0][buf[k]]++; c[1][buf[k + 1]]++; c[2][buf[k + 2]]++; c[3][buf[k + 3]]++;
	}
	for (; k < n; k++) c[0][buf[k]]++;
	for (k = 0; k < 256; k++) litHuf[k].freq = c[0][k] + c[1][k] + c[2][k] + c[3][k];
}

/* A sequence of level 0: literal-run, match-length and offset symbols with their raw low bits, and their counts */
ForceInlineTemplate void L0_Store_Sequence(WZL_Sched* const S_, WLZ_Set* const seq, WLZ_Huffman_Set* const hs, const Uint32 litRun, const Uint32 matchLen, const Uint32 matchOffset)
{
	if (litRun < LitRunDirect) {
		seq->litRun = litRun;
	}
	else {
		const int litRunMsb = High_Bit32(litRun);
		const int litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
		seq->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits]) << 8;
	}
	int mchLenHufIdx;
	if (matchLen < LitRunDirect) {
		mchLenHufIdx = matchLen - MinMatchLen;
		seq->mchLen = mchLenHufIdx;
	}
	else {
		const int matchLenMsb = High_Bit32(matchLen);
		mchLenHufIdx = LitRunHufMap[matchLenMsb].hufIdx + ((matchLen ^ 1 << matchLenMsb) >> LitRunHufMap[matchLenMsb].lsBits) - MinMatchLen;
		seq->mchLen = mchLenHufIdx ^ (matchLen & BitMask[LitRunHufMap[matchLenMsb].lsBits]) << 8;
	}
	const int offGroup = OffGroupOf[mchLenHufIdx];
	if (matchOffset < OffCasheSize) {
		seq->mchOff = matchOffset;
		hs->mchOffHuf[offGroup][matchOffset].freq++;
	}
	else {
		const int offsetMsb = High_Bit32(matchOffset);
		const int offsetHufIdx = Offset_Huffman_Index(matchOffset, offsetMsb);
		seq->mchOff = offsetHufIdx ^ (matchOffset & BitMask[ExtHufMchOff[offsetHufIdx].lsBits]) << OFF_SymBits;
		hs->mchOffHuf[offGroup][offsetHufIdx].freq++;
	}
}

/* Level 0, the fast mode, in the manner of zstd's fast strategy: each probed position looks up one small hash table
   on L0F_MinMatch bytes; the most recent offset is tried one byte ahead first. In a literal run the probe step grows
   by one every 2^L0F_SkipLog literals. Literals are copied when their match is emitted and counted per Huffman block.
   After a match, two positions in it are indexed, and the second most recent offset is tried right at its end. */
static Uint32 WLZ2_Compress_Fast1(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	Uint8* wzipStream,
	int wzipCapSize)
{
	SCHED(wzipStr);
	Uint32 i;
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	Uint32 nLzLits;

	Huffman_Str litHuf[N_HufLits];
	Uint8* wzipLitPtr = wzipStream;
	const Uint8* const wzipLitEnd = wzipLitPtr + wzipCapSize;
	wzipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32  zipLitBlkSize;
	Uint8* lzLitBuffer = (Uint8*)malloc(HUF_BlockSize + 32);   /* slack for 16-byte literal copies */    
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* lzLitEnd = lzLitBuffer + HUF_BlockSize;
	int lastOffset[OffCasheSize];
	int litRun;
	int* hash1Table = (int *)wzipStr->hash1Table;
	int litRunMsb, litRunHufIdx;

	WLZ_Huffman_Set huffmanSet;
	Huffman_Prev litPrev;                              /* the literal code that later blocks may reuse */
	litPrev.valid = 0;
	Seq_Prev seqPrev;                                  /* the sequence codes that later blocks may reuse */
	Seq_Prev_Init(&seqPrev);
	memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));

	WLZ_Set* const wlzSeq = (WLZ_Set*)malloc(SEQ_BlockSize*sizeof(WLZ_Set));
	WLZ_Set* wlzSeqPtr = wlzSeq;
	WLZ_Set* const wlzSeqEnd = wlzSeq + SEQ_BlockSize;

	Uint8* wlzStream = (Uint8*)malloc(2 * SEQ_BlockBound + 64);         /* one coded block of sequences, and room for its stream B */
	Uint8* const litScratch = (Uint8*)malloc(HUF_BlockSize + LIT_BlockSlack + 64);
	Seq_Stack seqStack;
	const int stackOk = Seq_Stack_Init(&seqStack, wzipStream + wzipCapSize);
	if (NULL == lzLitBuffer || NULL == wlzSeq || NULL == wlzStream || NULL == litScratch || !stackOk) goto _lit_overflow;

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "WLZ2_Compress_Index.txt", "w");
	fprintf(fptr, "WLZ2_Compress_Fast: srcSize=%i\n", srcSize);
#endif

	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	nLzLits = 0;
	for (i = 0; i < OffCasheSize; i++ )
		lastOffset[i] = -1;                     /* unset (0xFFFFFFFF): above every offset, so never a hit */

#define L0_EMIT_LITERALS(from, to)   {                                                                    \
		const Uint8* p_ = (from);                                                                        \
		while (p_ < (to)) {                                                                              \
			const Uint32 n_ = (Uint32)min((to) - p_, lzLitEnd - lzLitPtr);                               \
			if (n_ <= 16 && p_ + 16 <= srcEnd) memcpy(lzLitPtr, p_, 16);   /* short runs: one fixed copy */ \
			else if (p_ + n_ + 16 <= srcEnd) MemWildCopy(lzLitPtr, p_, lzLitPtr + n_);                       \
			else memcpy(lzLitPtr, p_, n_);                                                               \
			lzLitPtr += n_; p_ += n_;                                                                    \
			if (lzLitPtr == lzLitEnd) {                                                                  \
				Literal_Histogram(lzLitBuffer, HUF_BlockSize, litHuf);                                  \
				LIT_PUT_BLOCK(HUF_BlockSize); \
				wzipLitPtr += zipLitBlkSize;                                                             \
				lzLitPtr = lzLitBuffer;                                                                  \
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));                                      \
				nLzLits += HUF_BlockSize;                                                                \
			}                                                                                            \
		}                                                                                                \
	}
#define L0_STORE_SEQUENCE(litLen, mLen_, offset)   {                                                          \
		if ((offset) == 1) wlzSeqPtr = Store_Run(S_, wlzSeqPtr, wlzSeq, &huffmanSet, (litLen), (mLen_));      \
		else L0_Store_Sequence(S_, wlzSeqPtr++, &huffmanSet, (litLen), (mLen_), Offset_Cashe((Uint32*)lastOffset, (offset))); \
		if (wlzSeqPtr == wlzSeqEnd) {                                                                    \
			SEQ_PUT_BLOCK(); \
			memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));                                             \
			wlzSeqPtr = wlzSeq;                                                                          \
		}                                                                                                \
	}

	int fHashLog = L0F_HashLog;
	const int fMls = L0F_MinMatch, fStep = L0F_SkipLog, fWinLog = L0F_WindowLog;
	fHashLog = min(fHashLog, (int)High_Bit32(wzipStr->hash1Mask + 1));           /* within the allocated table */
	const Uint32 fWindow = min(WINDOW(fWinLog), WINDOW(SrchWidth[fMls]));        /* every match of fMls bytes or more is in its window */
	const int fShift = 64 - 8 * fMls, fHashShift = 64 - fHashLog;
	int* const fTable = hash1Table;
	memset(fTable, 0xFF, ((size_t)1 << fHashLog) * sizeof(int));
#define F_HASH(p)   ((Uint32)(((MemRead8(p) << fShift) * 0xCF1BBCDCB7A56463ULL) >> fHashShift))

	const Uint8* ip = srcPtr;
	while (ip < srcLastMatch) {
		const Uint8* mStart;
		Uint32 mLen, mOff;
		if (unlikely((Uint32)(ip - anchor) >= L_RunBreakAt)) {     /* a literal run nearing the format's limit */
			mStart = L_Run_Break(source, ip, srcLastMatch, WINDOW(SrchWidth[MinMatchLen]), &mLen, &mOff);
			if (NULL == mStart) break;                   /* no short match to the end: the run is too long, the input stored */
		}
		else {
			const Uint32 cur = (Uint32)(ip - source);
			const Uint32 h = F_HASH(ip);
			const int m = fTable[h];
			fTable[h] = (int)cur;

			const Uint32 rep0 = (Uint32)lastOffset[0];
			if (rep0 <= cur && MemRead4(ip + 1) == MemRead4(ip + 1 - rep0)) {   /* the most recent offset, one byte ahead */
				mStart = ip + 1;
				mOff = rep0;
				mLen = 4 + WLZ_Match_Count(ip + 5, ip + 5 - rep0, srcLastMatch, NULL);
			}
			else {
				if (m < 0 || (Uint32)m >= cur || cur - (Uint32)m >= fWindow) {
					ip += L_PROBE_STEP(ip - anchor, fStep);
					continue;
				}
				const Uint8* const mp = source + m;
				const reg_t diff = MemReadARCH(ip) ^ MemReadARCH(mp);
				const Uint32 len = diff ? N_ZeroBytes(diff) : REG_SIZE + WLZ_Match_Count(ip + REG_SIZE, mp + REG_SIZE, srcLastMatch, NULL);
				if (len < (Uint32)fMls) {
					ip += L_PROBE_STEP(ip - anchor, fStep);
					continue;
				}
				mStart = ip; mOff = cur - (Uint32)m; mLen = len;
				while (mStart > anchor && (Uint32)(mStart - source) > mOff && mStart[-1] == mStart[-1 - (int)mOff]) {
					mStart--;
					mLen++;
				}
			}
		}
		if (mOff == 1) {                         /* a run is not bound by the match-length cap */
			while (mStart + mLen < srcLastMatch && mStart[mLen] == mStart[mLen - 1]) mLen++;
			if (mLen > MaxRunCount) mLen = MaxRunCount;
		}
		else if (mLen > MaxMatchLen) mLen = MaxMatchLen;
		L0_EMIT_LITERALS(anchor, mStart);
		L0_STORE_SEQUENCE((Uint32)(mStart - anchor), mLen, mOff);
		ip = mStart + mLen;
		anchor = ip;
		if (ip >= srcLastMatch) break;
		fTable[F_HASH(mStart + 2)] = (int)(mStart + 2 - source);            /* index the match */
		fTable[F_HASH(ip - 2)] = (int)(ip - 2 - source);
		while (ip < srcLastMatch) {        /* the second most recent offset right at the end of the match */
			const Uint32 rep1 = (Uint32)lastOffset[1];
			if (rep1 > (Uint32)(ip - source) || MemRead4(ip) != MemRead4(ip - rep1)) break;
			Uint32 len = 4 + WLZ_Match_Count(ip + 4, ip + 4 - rep1, srcLastMatch, NULL);
			if (len > MaxMatchLen) len = MaxMatchLen;
			L0_STORE_SEQUENCE(0, len, rep1);
			fTable[F_HASH(ip)] = (int)(ip - source);
			ip += len;
			anchor = ip;
		}
	}
	L0_EMIT_LITERALS(anchor, srcEnd);       /* the last literals, all copied here */
	srcPtr = srcEnd;
#undef F_HASH
#undef L0_EMIT_LITERALS
#undef L0_STORE_SEQUENCE

	/* Encode Last Literals */
	litRun = (int)(srcEnd - anchor);

	int lastLits = (int)(srcEnd - srcPtr);
	if (lzLitPtr + lastLits > lzLitEnd) {
		while (lzLitPtr < lzLitEnd) {
			litHuf[*srcPtr].freq++;
			*lzLitPtr++ = *srcPtr++;
		}
		LIT_PUT_BLOCK(HUF_BlockSize);
		wzipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);     /* remaining number literals in the buffer to be flushed */
	if (lastBufLits) {                                   /* none when the literals filled their last block exactly */
		Literal_Histogram(lzLitBuffer, lastBufLits, litHuf);     /* the fast loop counts literals per block */
		LIT_PUT_BLOCK(lastBufLits);
		wzipLitPtr += zipLitBlkSize;
	}
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(wzipStream, nLzLits);       /* record the number of LZ literals at the beginning of zip stream */
	else               MemWriteLE2(wzipStream, (Uint16)nLzLits);

#ifdef WZIP_DEBUG
	fprintf(fptr, "srcIdx=%d, litRun=%d\n", (int)(anchor - (const Uint8*)source), litRun);
	fclose(fptr);
#endif

	if (litRun < LitRunDirect) {
		wlzSeqPtr->litRun = litRun;
		huffmanSet.litRunHuf[litRun].freq++;
	}
	else {
		litRunMsb = High_Bit32(litRun);
		litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
		wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits])<<8;
		huffmanSet.litRunHuf[litRunHufIdx].freq++;
	}
	wlzSeqPtr->mchLen = 255;   /* protocal for ending */
	wlzSeqPtr++;

	SEQ_PUT_BLOCK();
	int cmprSize = (int)((wzipLitPtr - wzipStream) + Seq_Stack_Place(&seqStack, wzipLitPtr));

	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return cmprSize;
_lit_overflow:                 /* the output buffer is full: the caller stores the input raw */
	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return 0;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************Hash - Chain Compression Functions * *******************************************************************
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

ForceInlineTemplate void WZIP_Search_Hash1Chain(WZIP_State_Str* const wzipStr, const Uint8* const source, Uint32 currIdx,
	                                            const Uint8* srcLastMatch, const Uint32* lastOffset, WLZ_Match* matchStr, int chainSearchCnt)
{
	SCHED(wzipStr);
	Uint32* chain1Table = (Uint32 *)wzipStr->chain1Table;
	int* hash1Table = (int *)wzipStr->hash1Table;
	const Uint32 chain1Mask = wzipStr->chain1Mask;
	const Uint32 maxMatchLen = wzipStr->hash2Len - 1;
	Uint8* matchPtr, *srcPtr;
	int matchLen, matchIdx;
	const int dictSize = wzipStr->dictSize;
	Uint8* const dictEnd = wzipStr->dictEnd;
	const Uint32 off1Window = WINDOW(SrchWidth[maxMatchLen]);  
	const int hash1Len = wzipStr->hash1Len;
	
	srcPtr = (Uint8*)source + wzipStr->curr1Idx;
	while (wzipStr->curr1Idx <= currIdx + L_InsertAhead) {
		if (srcPtr + L_HashAhead <= srcLastMatch)
			PREFETCH_L1(hash1Table + (WLZ_Hash1(srcPtr + L_HashAhead) & wzipStr->hash1Mask));
		Uint32 hashV = WLZ_Hash1(srcPtr++) & wzipStr->hash1Mask;
		const int prevIdx = hash1Table[hashV];
		if (prevIdx >= -dictSize) PREFETCH_L1(prevIdx < 0 ? dictEnd + prevIdx : source + prevIdx);
		PREFETCH_L1(chain1Table + ((Uint32)prevIdx & chain1Mask));
		int dist = wzipStr->curr1Idx - hash1Table[hashV];
		chain1Table[wzipStr->curr1Idx & chain1Mask] = (dist>0 && dist< chain1Mask && hash1Table[hashV]>=-dictSize)? dist: chain1Mask;
		hash1Table[hashV] = wzipStr->curr1Idx++;
	}

	srcPtr = (Uint8*)source + currIdx;
	Uint64 diffPattern, currPattern = MemReadARCH(srcPtr);
	int matchDist = chain1Table[currIdx & chain1Mask];
	matchIdx = currIdx - matchDist;
	while (matchIdx >= -dictSize && matchDist < off1Window && chainSearchCnt) {
		if (matchIdx < 0 && matchIdx > -16 && dictEnd != source) break;   /* a stale link (see L_InsertAhead) */
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		diffPattern = currPattern ^ MemReadARCH(matchPtr);
		matchLen = diffPattern ? N_ZeroBytes(diffPattern) : REG_SIZE;
		if (matchLen > matchStr->len && matchDist< WINDOW(SrchWidth[matchLen]) && Pays_Off(matchLen, matchDist, matchStr, lastOffset)) {
			matchStr->len = matchLen;
			matchStr->off = matchDist;
			if (matchLen >= maxMatchLen) break;
		}
		
		chainSearchCnt--;
		matchDist += chain1Table[matchIdx & chain1Mask];
		matchIdx = currIdx - matchDist;
	}
}
ForceInlineTemplate void WZIP_Search_Hash2Chain(WZIP_State_Str* const wzipStr, const Uint8* const source, Uint32 currIdx,
	                       const Uint8* srcLastMatch, const Uint8* dictLastMatch, const Uint32* lastOffset, WLZ_Match* matchStr, int chainSearchCnt)
{
	SCHED(wzipStr);
	Uint32* chain2Table = (Uint32*)wzipStr->chain2Table;
	const Uint32 chain2Mask = wzipStr->chain2Mask;
	int* hash2Table = (int*)wzipStr->hash2Table;
	Uint8* matchPtr, * srcPtr;
	int matchLen, matchIdx;
	const int dictSize = wzipStr->dictSize;
	Uint8* const dictEnd = wzipStr->dictEnd;
	const Uint32 off2Window = WINDOW(SrchWidth[8]);
	const int hash2Len = wzipStr->hash2Len;

	srcPtr = (Uint8*)source + wzipStr->curr2Idx;
	while (wzipStr->curr2Idx <= currIdx + L_InsertAhead) {
		if (srcPtr + L_HashAhead <= srcLastMatch)
			PREFETCH_L1(hash2Table + (WLZ_Hash2(srcPtr + L_HashAhead) & wzipStr->hash2Mask));
		Uint32 hashV = WLZ_Hash2(srcPtr++) & wzipStr->hash2Mask;
		const int prevIdx = hash2Table[hashV];
		if (prevIdx >= -dictSize) PREFETCH_L1(prevIdx < 0 ? dictEnd + prevIdx : source + prevIdx);
		PREFETCH_L1(chain2Table + ((Uint32)prevIdx & chain2Mask));
		int dist = wzipStr->curr2Idx - hash2Table[hashV];
		chain2Table[wzipStr->curr2Idx & chain2Mask] = (dist>0 && dist< chain2Mask && hash2Table[hashV] >= -dictSize)? dist : chain2Mask;
		hash2Table[hashV] = wzipStr->curr2Idx++;

	}

	srcPtr = (Uint8*)source + currIdx;
	Uint32 currPattern = MemRead4(srcPtr);
	int matchDist = chain2Table[currIdx & chain2Mask];
	matchIdx = currIdx - matchDist;
	while (matchIdx >= -dictSize && matchDist < off2Window && chainSearchCnt) {
		if (matchIdx < 0 && matchIdx > -16 && dictEnd != source) break;   /* a stale link (see L_InsertAhead) */
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr) && !CANNOT_REACH((int)matchStr->len + 1)) {
			matchLen = 4 + HIST_COUNT(srcPtr + 4, matchPtr + 4, srcLastMatch);
			if (matchLen > matchStr->len && matchDist < WINDOW(SrchWidth[min(8, matchLen)]) && Pays_Off(matchLen, matchDist, matchStr, lastOffset)) {
				matchStr->len = matchLen;
				matchStr->off = matchDist;
			}
		}
		chainSearchCnt--;
		matchDist += chain2Table[matchIdx &chain2Mask];
		matchIdx = currIdx - matchDist;
	}
}

/* Lazy evaluation: looks for a better match starting 1 to maxBack bytes after the current one. Candidates are taken
   from the chain at currIdx (= current start + maxBack), extended backward by up to 2 bytes, and at
   currIdx - (maxBack - 1). A candidate replaces the current match when its gain, less the cost of the literals it
   delays, is larger. Returns the number of bytes by which the match start moves. */
ForceInlineTemplate int WZIP_Search_Hash2Chain_2D(WZIP_State_Str* const wzipStr, const Uint8* const source, Uint32 currIdx,
	int maxBack, const Uint32* lastOffset, const Uint8* srcLastMatch, const Uint8* dictLastMatch, WLZ_Match* matchStr, int chainSearchCnt)
{
	SCHED(wzipStr);
	Uint32* chain2Table = (Uint32*)wzipStr->chain2Table;
	int* hash2Table = (int*)wzipStr->hash2Table;
	const Uint32 chain2Mask = wzipStr->chain2Mask;
	Uint8* matchPtr, *srcPtr;
	int matchLen, matchIdx, back, optBack = maxBack;
	static const int backTable[4] = { 0, 1, 0, 2 };
	const int dictSize = wzipStr->dictSize;
	Uint8* const dictEnd = wzipStr->dictEnd;
	const Uint32 off2Window = WINDOW(SrchWidth[8]);
	const int hash2Len = wzipStr->hash2Len;
	int bestGain = Match_Gain(matchStr, lastOffset);

	srcPtr = (Uint8*)source + wzipStr->curr2Idx;
	while (wzipStr->curr2Idx <= currIdx + L_InsertAhead) {
		if (srcPtr + L_HashAhead <= srcLastMatch)
			PREFETCH_L1(hash2Table + (WLZ_Hash2(srcPtr + L_HashAhead) & wzipStr->hash2Mask));
		Uint32 hashV = WLZ_Hash2(srcPtr++) & wzipStr->hash2Mask;
		const int prevIdx = hash2Table[hashV];
		if (prevIdx >= -dictSize) PREFETCH_L1(prevIdx < 0 ? dictEnd + prevIdx : source + prevIdx);
		PREFETCH_L1(chain2Table + ((Uint32)prevIdx & chain2Mask));
		int dist = wzipStr->curr2Idx - hash2Table[hashV];
		chain2Table[wzipStr->curr2Idx & chain2Mask] = (dist > 0 && dist < chain2Mask && hash2Table[hashV] >= -dictSize) ? dist : chain2Mask;
		hash2Table[hashV] = wzipStr->curr2Idx++;
	}

	/* candidates at currIdx, extended backward by up to 2 bytes */
	srcPtr = (Uint8*)source + currIdx;
	Uint32 currPattern = MemRead4(srcPtr);
	const Uint8 back0 = srcPtr[-1], back1 = srcPtr[-2];
	int matchDist = chain2Table[currIdx & chain2Mask];
	int searchCnt = chainSearchCnt * 3 / 4 + 4;
	while (matchDist < off2Window && matchDist <= currIdx + dictSize && searchCnt) {
		matchIdx = currIdx - matchDist;
		if (matchIdx < 0 && matchIdx > -16 && dictEnd != source) break;   /* a stale link (see L_InsertAhead) */
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			const int backIdx = (matchIdx >= 2 || (matchIdx < 0 && matchIdx >= 2 - (int)dictSize)) ? (back0 == *(matchPtr - 1)) ^ ((back1 == *(matchPtr - 2)) << 1) : 0;   /* never extend before the history start */
			back = backTable[backIdx];
			const int cost = Offset_Cost(matchDist, lastOffset) + G_Delay[maxBack - back];
			const int need = bestGain + cost;               /* the candidate must reach G_Byte * length > need */
			if (need < 0 || !CANNOT_REACH(need / G_Byte + 1 - back)) {
				matchLen = 4 + HIST_COUNT(srcPtr + 4, matchPtr + 4, srcLastMatch) + back;
				if (G_Byte * matchLen - cost > bestGain && matchDist < WINDOW(SrchWidth[min(8, matchLen)])) {
					matchStr->len = matchLen;
					matchStr->off = matchDist;
					optBack = back;
					bestGain = G_Byte * matchLen - cost;
				}
			}
		}
		searchCnt--;
		matchDist += chain2Table[matchIdx & chain2Mask];
	}

	/* candidates at currIdx - (maxBack - 1) */
	back = maxBack - 1;
	currIdx -= back;
	srcPtr -= back;
	currPattern = MemRead4(srcPtr);
	matchDist = chain2Table[currIdx & chain2Mask];
	searchCnt = 2 + chainSearchCnt / 4;
	while (matchDist < off2Window && matchDist <= currIdx + dictSize && searchCnt) {
		matchIdx = currIdx - matchDist;
		if (matchIdx < 0 && matchIdx > -16 && dictEnd != source) break;   /* a stale link (see L_InsertAhead) */
		matchPtr = (dictSize && matchIdx < 0) ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr)) {
			const int cost = Offset_Cost(matchDist, lastOffset) + G_Delay[maxBack - back];
			const int need = bestGain + cost;
			if (need < 0 || !CANNOT_REACH(need / G_Byte + 1)) {
				matchLen = 4 + HIST_COUNT(srcPtr + 4, matchPtr + 4, srcLastMatch);
				if (G_Byte * matchLen - cost > bestGain && matchDist < WINDOW(SrchWidth[min(8, matchLen)])) {
					matchStr->len = matchLen;
					matchStr->off = matchDist;
					optBack = back;
					bestGain = G_Byte * matchLen - cost;
				}
			}
		}
		searchCnt--;
		matchDist += chain2Table[matchIdx & chain2Mask];
	}
	return maxBack - optBack;
}


/* Replaces *matchStr (starting `delay` bytes later than srcIdx, with gain bestGain) by a repeat-offset match starting at
   srcIdx + 1 .. srcIdx + maxDelay when that gains more. A repeat offset is coded by its cache slot, so it is not bound by
   the length windows. Returns the delay of the chosen match. */
ForceInlineTemplate int Pick_Repeat_Offset(const Uint8* const source, Uint32 srcIdx, int minDelay, int maxDelay, int delay, int bestGain,
	const Uint32* const lastOffset, const int dictSize, const Uint8* const dictEnd, const Uint8* const srcLastMatch,
	const Uint8* const dictLastMatch, WLZ_Match* const matchStr)
{
	for (int d = minDelay; d <= maxDelay; d++)
		for (int k = 0; k < OffCasheSize; k++) {
			const int len = Repeat_Match_Len(source, srcIdx + d, lastOffset[k], dictSize, dictEnd, srcLastMatch, dictLastMatch);
			const int gain = G_Byte * len - k - G_Delay[d];              /* slot 0 is the cheapest to code */
			if (len >= MinMatchLen && gain > bestGain) {
				bestGain = gain;
				matchStr->len = len;
				matchStr->off = lastOffset[k];
				delay = d;
			}
		}
	return delay;
}
#define PICK_REPEAT_HERE(ms) Pick_Repeat_Offset(source, srcIdx, 0, 0, 0, (ms).len >= MinMatchLen ? Match_Gain(&(ms), lastOffset) : -(1 << 30), \
                                                lastOffset, dictSize, dictEnd, srcLastMatch, dictLastMatch, &(ms))

/** forced inline, to ensure branches are decided at compilation time **/
ForceInlineTemplate Uint32 WLZ2_Compress(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	Uint8* wzipStream,
	int wzipCapSize,
	int maxSearchCnt)
{
	SCHED(wzipStr);
	int* hash0Table = (int*)wzipStr->hash0Table;
	const Uint8* srcPtr = (const Uint8*)source;
	const Uint8* anchor = (const Uint8*)source;
	const Uint8* const srcEnd = (const Uint8*)source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	const int dictSize = wzipStr->dictSize;
	const Uint8* dictEnd = wzipStr->dictEnd;
	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	Uint32 nLzLits;
	Huffman_Str litHuf[N_HufLits];
	Uint8* wzipLitPtr = wzipStream;
	const Uint8* const wzipLitEnd = wzipLitPtr + wzipCapSize;
	wzipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32  zipLitBlkSize;
	Uint32 lastOffset[OffCasheSize];
	int litRun, litRunMsb, litRunHufIdx, matchLenMsb, mchLenHufIdx, offsetMsb, offsetHufIdx;
	const int hash2Len = wzipStr->hash2Len;
	
	WLZ_Match matchStr = { 0, 0 }, nextMatchStr = { 0, 0 };

	WLZ_Huffman_Set huffmanSet;
	Huffman_Prev litPrev;                              /* the literal code that later blocks may reuse */
	litPrev.valid = 0;
	Seq_Prev seqPrev;                                  /* the sequence codes that later blocks may reuse */
	Seq_Prev_Init(&seqPrev);
	memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));

	Uint8* const lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* lzLitEnd = lzLitBuffer + HUF_BlockSize;

	WLZ_Set* const wlzSeq = (WLZ_Set*)malloc(SEQ_BlockSize * sizeof(WLZ_Set));
	WLZ_Set* wlzSeqPtr = wlzSeq;
	WLZ_Set* const wlzSeqEnd = wlzSeq + SEQ_BlockSize;

	Uint8* wlzStream = (Uint8*)malloc(2 * SEQ_BlockBound + 64);         /* one coded block of sequences, and room for its stream B */
	Uint8* const litScratch = (Uint8*)malloc(HUF_BlockSize + LIT_BlockSlack + 64);
	Seq_Stack seqStack;
	const int stackOk = Seq_Stack_Init(&seqStack, wzipStream + wzipCapSize);
	if (NULL == lzLitBuffer || NULL == wlzSeq || NULL == wlzStream || NULL == litScratch || !stackOk) goto _lit_overflow;

#ifdef WZIP_DEBUG 
	FILE* fptr;
	fopen_s(&fptr, "WLZ2_Compress_Index.txt", "w");
	fprintf(fptr, "WZIP2_Compress_Kernel: srcSize=%i\n", srcSize);
	int wlzStats[MaxMatchLen + 1] = { 0 };
#endif
	wzipStr->curr1Idx = 0;
	wzipStr->curr2Idx = 0;

	int curr0Idx = 0;
	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	memset(lastOffset, 0xFF, OffCasheSize * sizeof(int));     /* unset: above every offset, so never a hit */
	nLzLits = 0;
	Uint32 srcIdx = 0, hashV;
	int nextMatchDone = 0;
	while (1) {

		while (1) {
			/*if (srcIdx == 1926247) {
				srcIdx += 0;
			}*/

			if (unlikely(srcPtr >= srcLastMatch)) goto _last_literals;

			if (nextMatchDone) {
				matchStr = nextMatchStr;
				nextMatchDone = 0;
				PICK_REPEAT_HERE(matchStr);
			}
			else {
				matchStr.len = 0;
				WZIP_Search_Hash2Chain(wzipStr, source, srcIdx, srcLastMatch, dictLastMatch, lastOffset, &matchStr, maxSearchCnt);
				if (matchStr.len >= hash2Len) {
					PICK_REPEAT_HERE(matchStr);
					break;
				}

				/* levels 2-3 use the 3-byte table and the long hash chain only; the 4/5-byte chain serves the lazy levels */
				if (wzipStr->compressLevel > 3)
					WZIP_Search_Hash1Chain(wzipStr, source, srcIdx, srcLastMatch, lastOffset, &matchStr, maxSearchCnt);
				if (!matchStr.len) {
					Uint8* src_ptr = (Uint8*)source + curr0Idx;
					while (src_ptr < srcPtr) {
						hashV = WLZ_Hash0(src_ptr++) & wzipStr->hash0Mask;
						hash0Table[hashV] = curr0Idx++;
					}
					hashV = WLZ_Hash0(srcPtr) & wzipStr->hash0Mask;
					int match0Idx = hash0Table[hashV];
					int offset = srcIdx - match0Idx;   // note curr0Idx=srcIdx
					hash0Table[hashV] = curr0Idx++; 
					if ( match0Idx >= -dictSize && offset > 0 && offset < WINDOW(SrchWidth[hash2Len]) ) {
						const Uint8* matchPtr = (dictSize && match0Idx < 0) ? dictEnd + match0Idx : srcPtr - offset;
						reg_t diffPattern = MemReadARCH(srcPtr) ^ MemReadARCH(matchPtr);
						matchStr.len = diffPattern? N_ZeroBytes(diffPattern) : REG_SIZE;
						matchStr.off = offset;
						if (matchStr.len <= hash2Len && offset >= WINDOW(SrchWidth[matchStr.len]) )
							matchStr.len = 0;
					}
				}
				PICK_REPEAT_HERE(matchStr);
			}
			if (matchStr.len >= MinMatchLen) break;

			*lzLitPtr++ = *srcPtr;
			litHuf[*srcPtr].freq++;
			srcPtr++;
			srcIdx++;
			if (lzLitPtr == lzLitEnd) {
				LIT_PUT_BLOCK(HUF_BlockSize);
				wzipLitPtr += zipLitBlkSize;
				lzLitPtr = lzLitBuffer;
				memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
				nLzLits += HUF_BlockSize;
			}
		}

		if (wzipStr->compressLevel > 3 && srcPtr + matchStr.len < srcLastMatch && srcPtr + hash2Len < srcLastMatch && matchStr.len <= MaxMatchLen) {

			nextMatchStr.len = 2;  //nextMatchStr.off = WLZ_MAX_DIST;
			WZIP_Search_Hash2Chain(wzipStr, source, srcIdx + matchStr.len, srcLastMatch, dictLastMatch, lastOffset, &nextMatchStr, maxSearchCnt);
			if ( nextMatchStr.len < hash2Len) {
				WZIP_Search_Hash1Chain(wzipStr, source, srcIdx + matchStr.len, srcLastMatch, lastOffset, &nextMatchStr, maxSearchCnt/4);
			}
			
			int lazyForward = WZIP_Search_Hash2Chain_2D(wzipStr, source, srcIdx + 3, 3, lastOffset, srcLastMatch, dictLastMatch, &matchStr, maxSearchCnt / 2);
			/* repeat offsets at the next two positions compete as well */
			lazyForward = Pick_Repeat_Offset(source, srcIdx, 1, 2, lazyForward, Match_Gain(&matchStr, lastOffset) - G_Delay[lazyForward],
			                                 lastOffset, dictSize, dictEnd, srcLastMatch, dictLastMatch, &matchStr);

			nextMatchDone = (0 == lazyForward);

			const Uint8* srcPtrEnd = srcPtr + lazyForward;
			while (srcPtr < srcPtrEnd) {
				*lzLitPtr++ = *srcPtr;
				litHuf[*srcPtr].freq++;
				srcPtr++;
				if (lzLitPtr == lzLitEnd) {
					LIT_PUT_BLOCK(HUF_BlockSize);
					wzipLitPtr += zipLitBlkSize;
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
		fprintf(fptr, "srcIdx=%d, litRun=%d,  matLen=%d, offset=%d ",
			(int)(anchor - (const Uint8*)source), litRun, matchStr.len, matchStr.off);
#endif
		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Encode Literal Run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		if (litRun < LitRunDirect) {
			wlzSeqPtr->litRun = litRun;
			huffmanSet.litRunHuf[litRun].freq++;
		} else {
			litRunMsb = High_Bit32(litRun);
			litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
			wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits])<<8;
			huffmanSet.litRunHuf[litRunHufIdx].freq++;
		}

		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Fast Encode Match Pair  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		if (matchStr.off == 1) {                 /* a run is not bound by the match-length cap */
			const Uint32 len0 = matchStr.len;
			while (srcPtr + matchStr.len < srcLastMatch && srcPtr[matchStr.len] == srcPtr[matchStr.len - 1]) matchStr.len++;
			if (matchStr.len > MaxRunCount) matchStr.len = MaxRunCount;
			if (matchStr.len != len0) nextMatchDone = 0;    /* the looked-ahead match started inside the run */
		}
		srcIdx += matchStr.len;
		srcPtr += matchStr.len;

		if (matchStr.off == 1) wlzSeqPtr = Store_Run(S_, wlzSeqPtr, wlzSeq, &huffmanSet, litRun, matchStr.len) - 1;
		else {
		matchStr.off = Offset_Cashe(lastOffset, matchStr.off);
#ifdef WZIP_DEBUG
		fprintf(fptr, "-> %d,  str=", matchStr.off);
		for (int i = -matchStr.len; i < 0; i++)
			fprintf(fptr, "%c", *(srcPtr + i));
		fprintf(fptr, "\n");
		fflush(fptr);
		wlzStats[matchStr.len]++;
#endif
		if (matchStr.len < LitRunDirect) {
			mchLenHufIdx = matchStr.len - MinMatchLen;
			wlzSeqPtr->mchLen = mchLenHufIdx;
			huffmanSet.mchLenHuf[mchLenHufIdx].freq++;
		}
		else {
			matchLenMsb = High_Bit32(matchStr.len);
			mchLenHufIdx = LitRunHufMap[matchLenMsb].hufIdx + ((matchStr.len ^ 1 << matchLenMsb) >> LitRunHufMap[matchLenMsb].lsBits) - MinMatchLen;
			wlzSeqPtr->mchLen = mchLenHufIdx ^ (matchStr.len & BitMask[LitRunHufMap[matchLenMsb].lsBits])<<8;
			huffmanSet.mchLenHuf[mchLenHufIdx].freq++;
		}

		if (matchStr.off < 4) {
			wlzSeqPtr->mchOff = (Uint8)matchStr.off;
			huffmanSet.mchOffHuf[OffGroupOf[mchLenHufIdx]][matchStr.off].freq++;
		}
		else {
			offsetMsb = High_Bit32(matchStr.off);
			offsetHufIdx = Offset_Huffman_Index(matchStr.off, offsetMsb);
			wlzSeqPtr->mchOff = offsetHufIdx ^ (matchStr.off & BitMask[ExtHufMchOff[offsetHufIdx].lsBits]) << OFF_SymBits;
			//huffmanSet.mchOffHuf[OffsetGroupTable[min(15, mchLenHufIdx)]][offsetHufIdx].freq++;
			huffmanSet.mchOffHuf[OffGroupOf[mchLenHufIdx]][offsetHufIdx].freq++;
		}
		}

		anchor = srcPtr;
		matchStr.len = 0;

		if (++wlzSeqPtr==wlzSeqEnd) {
			SEQ_PUT_BLOCK();
			memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));		
			wlzSeqPtr = wlzSeq;
		}
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
		LIT_PUT_BLOCK(HUF_BlockSize);
		wzipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);     /* remaining number literals in the buffer to be flushed */
	if (lastBufLits) {                                   /* none when the literals filled their last block exactly */
		LIT_PUT_BLOCK(lastBufLits);
		wzipLitPtr += zipLitBlkSize;
	}
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(wzipStream, nLzLits);       /* record the number of LZ literals at the beginning of zip stream */
	else               MemWriteLE2(wzipStream, (Uint16)nLzLits);

#ifdef WZIP_DEBUG
	fprintf(fptr, "srcIdx=%d, litRun=%d\n", (int)(anchor - (const Uint8*)source), litRun);
	fclose(fptr);

	if (NULL == (fptr = fopen("wlz_stats.txt", "w"))) {
		fprintf(stderr, "unable to write file wlz_stats.txt\n");
		exit(1);
	}
	fprintf(fptr, "wlz match statistics\n");
	fprintf(fptr, "   1: %d  uncompressed literals\n", nLzLits);
	fprintf(fptr, "match-len   count\n");
	for (int i = MinMatchLen; i <= MaxMatchLen; i++)
		if( wlzStats[i] )
			fprintf(fptr, "%8d: %d\n", i, wlzStats[i]);
	fclose(fptr);
#endif

	if (litRun < LitRunDirect) {
		wlzSeqPtr->litRun = (Uint8)litRun;
		huffmanSet.litRunHuf[litRun].freq++;
	}
	else {
		litRunMsb = High_Bit32(litRun);
		litRunHufIdx = litRunMsb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[litRunMsb].hufIdx + ((litRun ^ 1 << litRunMsb) >> LitRunHufMap[litRunMsb].lsBits);
		wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[ExtHufLitRun[litRunHufIdx].lsBits])<<8;
		huffmanSet.litRunHuf[litRunHufIdx].freq++;
	}
	wlzSeqPtr->mchLen = 255;    /* protocal for ending */
	wlzSeqPtr++;

	SEQ_PUT_BLOCK();
	int cmprSize = (int)((wzipLitPtr - wzipStream) + Seq_Stack_Place(&seqStack, wzipLitPtr));

	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return cmprSize;
_lit_overflow:                 /* the output buffer is full: the caller stores the input raw */
	free(lzLitBuffer);
	free(wlzSeq);
	free(wlzStream);
	free(litScratch);
	free(seqStack.sizes);
	return 0;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimal parsing (levels 7-13) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A shortest-path parse over segments of up to OPT_Num positions. Every position is priced in bits from running symbol
   statistics, using the same symbols as the encoder: literals, literal-run and match-length codes with their extra bits,
   and the offset code of the match's length group with its raw bits. A raw offset must lie in the window of its length;
   an offset in the repeat cache (tracked per path) costs only its slot code and has no window limit. A match longer than
   the level's sufficient length ends the segment at once. */
#define   OPT_Num              4096
#define   OPT_Unit             256                     /* prices in 1/256 bit */
#define   OPT_Inf              0x3FFFFFFF
#define   OPT_UpdateBytes      (1 << 15)               /* prices are refreshed from the statistics every so many bytes */
/* levels 7 to 13: the search depth (tree steps), the match length that ends a segment, passes and states. Levels 7-10:
   shallow searches and short segments (each level between its neighbours in both ratio and speed, and ahead of the
   lazy parser at equal speed). From level 11 the tree searches deep and segments run long: with the chains capped
   (OPT_ChainMax) this costs little, and gains more than the extra states and passes, which levels 12 and 13 add */
static const int OPT_LevelDepth[7] = { 4, 4, 8, 16, 256, 256, 256 };
static const int OPT_LevelSufficient[7] = { 16, 32, 32, 32, 1024, 1024, 1024 };
static const int OPT_LevelPasses[7] = { 1, 1, 1, 1, 1, 1, 2 };
static const int OPT_LevelStates[7] = { 1, 1, 1, 1, 1, 3, 3 };
/* chains A and B stop after this many steps: the tree finds the long matches, and deeper chain walks found almost
   nothing (level 13: -0.02% at 1.85x the speed with the chains at 4x and 1x the depth of 256) */
#define   OPT_ChainMax         16
/* a pass before the last only gathers the symbol counts that price the next one: a shallow one-state search does
   nearly as well (level 13: -0.02% at 1.3x the speed of a full first pass) */
#define   OPT_StatsDepth       16
#define   OPT_StatsSufficient  64

typedef struct {
	int price;                                         /* cost of the path up to here, pending literal-run code included */
	int litLen;                                        /* literals since the last match on the path */
	Uint32 mLen, mOff;                                 /* match ending here (mLen 0: a literal ends here), raw offset */
	Uint32 rep[OffCasheSize];                          /* offset cache after the path */
	int prevC;                                         /* state (literal-run class) this one comes from */
} Opt_Node;

/* Up to three states per position, one per literal-run class (0, 1, 2+ literals since the last match): a match's joint
   symbol depends on that class, so the best arrival of each class makes the shortest path exact with respect to it.
   With one state, the cheapest arrival is kept whatever its class. */
#define   OPT_C                3
#define   OPT_Class(litLen)    ((litLen) < 2 ? (litLen) : 2)
/* a later pass prices each region of 2^OPT_RegionLog bytes from the symbol counts the previous pass found in it */
#define   OPT_RegionLog        17

typedef struct {
	Uint32 len, off;
} Opt_Cand;

typedef struct {
	Uint32 start, len, off;
} Opt_Path;

typedef struct {
	Uint32 litFreq[N_HufLits], litRunFreq[N_HufLitRun], mchLenFreq[N_HufJoint], mchOffFreq[MaxMchOffGroup][N_HufMchOffMax];
	int litPrice[N_HufLits], litRunPrice[N_HufLitRun], mchLenPrice[N_HufJoint], mchOffPrice[MaxMchOffGroup][N_HufMchOffMax];
} Opt_Stats;

ForceInlineTemplate int LitRun_Symbol(WZL_Sched* const S_, Uint32 litRun, int* extraBits)
{
	if (litRun < LitRunDirect) { *extraBits = 0; return (int)litRun; }
	const int msb = High_Bit32(litRun);
	const int sym = msb >= MaxLitRunMsb ? LIT_RUN_70(litRun) : LitRunHufMap[msb].hufIdx + ((litRun ^ 1 << msb) >> LitRunHufMap[msb].lsBits);
	*extraBits = ExtHufLitRun[sym].lsBits;
	return sym;
}

ForceInlineTemplate int MchLen_Symbol(Uint32 len, int* extraBits)
{
	if (len < LitRunDirect) { *extraBits = 0; return (int)len - MinMatchLen; }
	const int msb = High_Bit32(len);
	*extraBits = LitRunHufMap[msb].lsBits;
	return LitRunHufMap[msb].hufIdx + ((len ^ 1 << msb) >> LitRunHufMap[msb].lsBits) - MinMatchLen;
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
	for (int i = 0; i < n; i++) price[i] = Log2_Price(total, (Uint64)freq[i] + 1);     /* in 1/OPT_Unit bit */
}

static void Opt_Update_Prices(WZL_Sched* const S_, Opt_Stats* st)
{
	Opt_Set_Prices(st->litFreq, N_HufLits, st->litPrice);
	Opt_Set_Prices(st->litRunFreq, N_HufLitRun, st->litRunPrice);
	Opt_Set_Prices(st->mchLenFreq, N_HufJoint, st->mchLenPrice);
	for (int g = 0; g < MchOffGroup; g++)
		Opt_Set_Prices(st->mchOffFreq[g], N_HufMchOff[g], st->mchOffPrice[g]);
}

/* halves the counts so that the prices follow the data */
static void Opt_Age_Stats(WZL_Sched* const S_, Opt_Stats* st)
{
	for (int i = 0; i < N_HufLits; i++) st->litFreq[i] >>= 1;
	for (int i = 0; i < N_HufLitRun; i++) st->litRunFreq[i] >>= 1;
	for (int i = 0; i < N_HufJoint; i++) st->mchLenFreq[i] >>= 1;
	for (int g = 0; g < MchOffGroup; g++)
		for (int i = 0; i < N_HufMchOffMax; i++) st->mchOffFreq[g][i] >>= 1;
}

static void Opt_Init_Stats(WZL_Sched* const S_, Opt_Stats* st, const Uint8* source, Uint32 srcSize)
{
	memset(st, 0, sizeof(*st));
	Uint32 hist[256] = { 0 };
	for (Uint32 i = 0; i < srcSize; i++) hist[source[i]]++;
	for (int i = 0; i < 256; i++) st->litFreq[i] = (Uint32)(((Uint64)hist[i] << 12) / (srcSize + 1));
	for (int i = 0; i < N_HufLitRun; i++) st->litRunFreq[i] = 256 >> min(i, 8);
	/* a weak prior over lengths, halving every 8 symbols: a steeper one makes short lengths cheap enough that the
	   parser splits long matches, and the counts of that parse keep them cheap */
	{   static const Uint32 slotW[N_SlotSel] = { 4, 2, 1, 1, 12 };    /* cache slots 0-3 and new offsets */
		for (int i = 0; i < N_HufJoint; i++)
			st->mchLenFreq[i] = ((256u >> min(i % N_HufMchLen / 8, 8)) >> (i / N_HufMchLen % N_LitClass)) * slotW[i / (N_HufMchLen * N_LitClass)] >> 2;
	}
	for (int g = 0; g < MchOffGroup; g++) {
		for (int i = 0; i < N_HufMchOff[g]; i++) st->mchOffFreq[g][i] = 4;
		st->mchOffFreq[g][0] = st->mchOffFreq[g][1] = st->mchOffFreq[g][2] = st->mchOffFreq[g][3] = 0;   /* slots: in the joint symbol */
	}
	Opt_Update_Prices(S_, st);
}

/* the literal-run code: none for runs of 0 and 1 (their class is in the joint symbol) */
ForceInlineTemplate int LitRun_Price(WZL_Sched* const S_, const Opt_Stats* st, Uint32 litRun)
{
	int extra;
	if (litRun < 2) return 0;
	const int sym = LitRun_Symbol(S_, litRun, &extra);
	return st->litRunPrice[sym] + extra * OPT_Unit;
}

/* price of a match of length len with raw offset off from a path whose offset cache is rep; OPT_Inf if not codable */
ForceInlineTemplate int Match_Price(WZL_Sched* const S_, const Opt_Stats* st, Uint32 len, Uint32 off, const Uint32* rep, Uint32 litLen)
{
	if (off == 1) {       /* a run: the run symbol, and its count as an offset of the widest window */
		const Uint32 v = len + OffCasheSize - 1;
		const int msb = High_Bit32(v);
		return st->mchLenPrice[JointIdx(OffCasheSize, LitClass(min(litLen, 2)), RunSym)] + st->mchOffPrice[OffGroupOf[RunSym]][Offset_Huffman_Index((int)v, msb)] + (msb - 1) * OPT_Unit;
	}
	int mlExtra, offExtra;
	const int mlSym = MchLen_Symbol(len, &mlExtra);
	const int group = OffGroupOf[mlSym];
	Uint32 v;
	if (off == rep[0]) v = 0;
	else if (off == rep[1]) v = 1;
	else if (off == rep[2]) v = 2;
	else if (off == rep[3]) v = 3;
	else {
		if (off >= (Uint32)WINDOW(OffWidth[min(8, len)])) return OPT_Inf;
		v = off + OffCasheSize - 1;
	}
	if (v < OffCasheSize) return st->mchLenPrice[JointIdx(v, LitClass(min(litLen, 2)), mlSym)] + mlExtra * OPT_Unit;
	const int offSym = Offset_Symbol(v, &offExtra);
	return st->mchLenPrice[JointIdx(OffCasheSize, LitClass(min(litLen, 2)), mlSym)] + mlExtra * OPT_Unit + st->mchOffPrice[group][offSym] + offExtra * OPT_Unit;
}

/* the length symbol and the price of the extra bits of every match length, for the parser's inner loop */
typedef struct {
	Uint8 sym[MaxMatchLen + 1];
	int extra[MaxMatchLen + 1];
} Opt_LenTab;

static void Opt_Init_LenTab(Opt_LenTab* const t)
{
	for (Uint32 l = MinMatchLen; l <= MaxMatchLen; l++) {
		int extra;
		t->sym[l] = (Uint8)MchLen_Symbol(l, &extra);
		t->extra[l] = extra * OPT_Unit;
	}
}

/* Match finder of the optimal parser: three levels, each searched to the window of its longest length.
     A: lengths 3-4, 3-byte hash chain, to the window of length 4 (nearly exhaustive)
     B: lengths 5-6, 5-byte hash chain, to the window of length 6
     C: lengths 7+,  7-byte hash binary tree over the full window
   Chains are walked nearest first, so each length gets its nearest offset; each walk stops at the first match long
   enough for the next level, which finds that match (or a nearer one) itself. The tree covers long matches cheaply.
   A dictionary takes the positions before the input (-D..-1, the input following it): its positions within each
   level's window (up to -16) are inserted before the input, so that every level finds matches in it. */
#define   OPT_Nil              0x80000000u                       /* below any position, dictionary included */
#define   OPT_TreeLead         (8 << 12)                         /* > the most tree threads run apart (Opt_MT) */

typedef struct {
	int* headA, *headB, *headC;
	Uint32* chainA, *chainB, *bt;
	Uint32 hMaskA, hMaskB, hMaskC, maskA, maskB, maskC;
	int winA, winB, winC;
	Uint32 nextA, nextB, nextC;                        /* next input position to insert */
} Opt_Finder;

ForceInlineTemplate int Opt_Tree_Insert(Opt_Finder* const f, const Uint8* const source, const int idx, int searchCnt,
	const Uint8* const srcLastMatch, const Uint8* const dictEnd, const int dictSize, Opt_Cand* out);

static void Opt_Finder_Prime(Opt_Finder* const f, const Uint8* dict, int dictSize, const Uint8* const source,
	const Uint8* const srcLastMatch, int searchCnt, const int mask);
static void Opt_Prime_Tree(Opt_Finder* const f, const Uint8* dict, int dictSize, const Uint8* const source,
	const Uint8* const srcLastMatch, int searchCnt, const int part, const int parts);

/* allocates the indexes; primes those of `prime` with the dictionary (see Opt_Finder_Prime) */
static int Opt_Finder_Init(WZL_Sched* const S_, Opt_Finder* f, const Uint8* dict, int dictSize, const Uint8* const source,
	const Uint8* const srcLastMatch, int searchCnt, const int prime)
{
	f->maskA = BitMask[SrchWidth[4]]; f->maskB = BitMask[SrchWidth[6]]; f->maskC = BitMask[SrchWidth[8]];
	f->winA = WINDOW(SrchWidth[4]); f->winB = WINDOW(SrchWidth[6]); f->winC = WINDOW(SrchWidth[8]);
	/* The tree's nodes are those of the positions modulo 2^w(8): a position takes over the node of the one 2^w(8)
	   before it. The threads that split the tree (Opt_MT) insert up to OPT_TreeLead positions apart, so when the
	   positions span more than 2^w(8), the tree's window stops OPT_TreeLead short of it, with one thread as with many:
	   no walk then reaches a node that another thread may already have taken over. */
	if ((long long)min(dictSize, f->winC) + (srcLastMatch - source) > (long long)f->maskC + 1) f->winC -= OPT_TreeLead;
	f->hMaskA = BitMask[min(SrchWidth[4] + 2, 20)]; f->hMaskB = BitMask[SrchWidth[6]]; f->hMaskC = BitMask[SrchWidth[8]];
	f->headA = (int*)malloc(((size_t)f->hMaskA + 1) * sizeof(int));
	f->headB = (int*)malloc(((size_t)f->hMaskB + 1) * sizeof(int));
	f->headC = (int*)malloc(((size_t)f->hMaskC + 1) * sizeof(int));
	f->chainA = (Uint32*)malloc(((size_t)f->maskA + 1) * sizeof(Uint32));
	f->chainB = (Uint32*)malloc(((size_t)f->maskB + 1) * sizeof(Uint32));
	f->bt = (Uint32*)malloc(((size_t)f->maskC + 1) * 2 * sizeof(Uint32));
	if (!f->headA || !f->headB || !f->headC || !f->chainA || !f->chainB || !f->bt) return 0;
	memset(f->headA, 0x80, ((size_t)f->hMaskA + 1) * sizeof(int));  /* 0x80808080: before any history */
	memset(f->headB, 0x80, ((size_t)f->hMaskB + 1) * sizeof(int));
	memset(f->headC, 0x80, ((size_t)f->hMaskC + 1) * sizeof(int));
	f->nextA = f->nextB = f->nextC = 0;
	if (prime) Opt_Finder_Prime(f, dict, dictSize, source, srcLastMatch, searchCnt, prime);
	return 1;
}

/* Inserts the dictionary into the indexes of mask (1 chain A, 2 chain B, 4 tree C): its positions within each index's
   window (older ones cannot be reached), up to -16, as hashing and compares read 8 bytes, or, for a dictionary just
   before the input, to its end. An index is primed by the thread that searches it. */
static void Opt_Finder_Prime(Opt_Finder* const f, const Uint8* dict, int dictSize, const Uint8* const source,
	const Uint8* const srcLastMatch, int searchCnt, const int mask)
{
	if (!dict || dictSize <= 0) return;
	const Uint8* const dictEnd = dict + dictSize;
	const int lastDict = dictEnd == source ? -1 : -16;
	if (mask & 1) for (int i = -min(dictSize, f->winA); i <= lastDict; i++) {
		const Uint32 h = Hash_3B(dictEnd + i) & f->hMaskA;
		const int prev = f->headA[h];
		f->chainA[(Uint32)i & f->maskA] = (prev >= -dictSize && i - prev > 0 && i - prev <= (int)f->maskA) ? (Uint32)(i - prev) : f->maskA + 1;
		f->headA[h] = i;
	}
	if (mask & 2) for (int i = -min(dictSize, f->winB); i <= lastDict; i++) {
		const Uint32 h = Hash_5B(dictEnd + i) & f->hMaskB;
		const int prev = f->headB[h];
		f->chainB[(Uint32)i & f->maskB] = (prev >= -dictSize && i - prev > 0 && i - prev <= (int)f->maskB) ? (Uint32)(i - prev) : f->maskB + 1;
		f->headB[h] = i;
	}
	if (mask & 4) Opt_Prime_Tree(f, dict, dictSize, source, srcLastMatch, searchCnt, 0, 1);
}

/* Inserts into the tree the dictionary positions (those of Opt_Finder_Prime) of the hash buckets h with h % parts =
   part. Each bucket is a tree of its own, whose positions go in the same order whoever inserts them, so threads may
   prime the parts at once and build the tree that one would. */
static void Opt_Prime_Tree(Opt_Finder* const f, const Uint8* dict, int dictSize, const Uint8* const source,
	const Uint8* const srcLastMatch, int searchCnt, const int part, const int parts)
{
	if (!dict || dictSize <= 0) return;
	const Uint8* const dictEnd = dict + dictSize;
	const int lastDict = dictEnd == source ? -1 : -16;
	for (int i = -min(dictSize, f->winC); i <= lastDict; i++)
		if (parts == 1 || (Hash_7B(dictEnd + i) & f->hMaskC) % (Uint32)parts == (Uint32)part)
			Opt_Tree_Insert(f, source, i, searchCnt, srcLastMatch, dictEnd, dictSize, NULL);
}

#if WZIP_MULTITHREAD
/* empties the indexes (after a failed start of the threads that were priming them) */
static void Opt_Finder_Reset(Opt_Finder* const f)
{
	memset(f->headA, 0x80, ((size_t)f->hMaskA + 1) * sizeof(int));
	memset(f->headB, 0x80, ((size_t)f->hMaskB + 1) * sizeof(int));
	memset(f->headC, 0x80, ((size_t)f->hMaskC + 1) * sizeof(int));
	f->nextA = f->nextB = f->nextC = 0;
}
#endif

static void Opt_Finder_Free(Opt_Finder* f)
{
	free(f->headA); free(f->headB); free(f->headC); free(f->chainA); free(f->chainB); free(f->bt);
}

/* level C: inserts idx into the tree and, when out is given, records each match longer than all found before. Positions
   below 0 are the dictionary's (inserted before the input, their compares kept inside the dictionary); a match there
   may run on into the input. */
ForceInlineTemplate int Opt_Tree_Insert(Opt_Finder* const f, const Uint8* const source, const int idx, int searchCnt,
	const Uint8* const srcLastMatch, const Uint8* const dictEnd, const int dictSize, Opt_Cand* out)
{
	const int contig = dictEnd == source;                       /* a dictionary just before the input: one memory */
	const Uint8* const ip = idx >= 0 ? source + idx : dictEnd + idx;
	const Uint32 h = Hash_7B(ip) & f->hMaskC;
	int matchIdx = f->headC[h];
	f->headC[h] = idx;
	Uint32* smallerPtr = f->bt + 2 * ((Uint32)idx & f->maskC);
	Uint32* largerPtr = smallerPtr + 1;
	const int low = max(idx - f->winC, -dictSize - 1);
	int commonSmaller = 0, commonLarger = 0, bestLen = MinMatchLen - 1, n = 0;
	const Uint8* const limit = idx >= 0 || contig ? srcLastMatch : dictEnd - REG_SIZE * 2;
	const Uint8* const countEnd = limit - ip > MaxMatchLen ? ip + MaxMatchLen : limit;   /* counts stop where the walk does */
	while (searchCnt-- > 0 && matchIdx > low && matchIdx < idx) {
		Uint32* const nextPtr = f->bt + 2 * ((Uint32)matchIdx & f->maskC);
		int len = min(commonSmaller, commonLarger);
		const int m = matchIdx + len;                            /* where the compare resumes */
		len += (int)(m >= 0 || contig ? WLZ_Match_Count(ip + len, source + m, countEnd, NULL)
		                    : Hist_Match_Count(ip + len, dictEnd + m, countEnd, dictEnd, source));
		if (len > bestLen) {
			bestLen = len;
			if (out) { out[n].len = min(len, MaxMatchLen); out[n].off = (Uint32)(idx - matchIdx); n++; }
		}
		if (len >= MaxMatchLen || ip + len >= limit) break;     /* cannot be ordered: drop the rest */
		const int mb = matchIdx + len;
		if ((mb >= 0 ? source[mb] : dictEnd[mb]) < ip[len]) {
			*smallerPtr = (Uint32)matchIdx;
			commonSmaller = len;
			smallerPtr = nextPtr + 1;
			matchIdx = (int)nextPtr[1];
		}
		else {
			*largerPtr = (Uint32)matchIdx;
			commonLarger = len;
			largerPtr = nextPtr;
			matchIdx = (int)nextPtr[0];
		}
	}
	*smallerPtr = *largerPtr = OPT_Nil;
	return n;
}

/* chains A and B: insert the positions up to currIdx */
ForceInlineTemplate void Opt_Insert_A(Opt_Finder* const f, const Uint8* const source, const Uint32 currIdx, const int dictSize)
{
	for (; f->nextA <= currIdx; f->nextA++) {
		const Uint32 h = Hash_3B(source + f->nextA) & f->hMaskA;
		const int prev = f->headA[h], d = (int)f->nextA - prev;
		f->chainA[f->nextA & f->maskA] = (prev >= -dictSize && d > 0 && d <= (int)f->maskA) ? (Uint32)d : f->maskA + 1;
		f->headA[h] = (int)f->nextA;
	}
}

ForceInlineTemplate void Opt_Insert_B(Opt_Finder* const f, const Uint8* const source, const Uint32 currIdx, const int dictSize)
{
	for (; f->nextB <= currIdx; f->nextB++) {
		const Uint32 h = Hash_5B(source + f->nextB) & f->hMaskB;
		const int prev = f->headB[h], d = (int)f->nextB - prev;
		f->chainB[f->nextB & f->maskB] = (prev >= -dictSize && d > 0 && d <= (int)f->maskB) ? (Uint32)d : f->maskB + 1;
		f->headB[h] = (int)f->nextB;
	}
}

/* chain A (its positions inserted up to currIdx): the candidates of lengths 3-4, nearest first, until a match of
   length 5 */
ForceInlineTemplate int Opt_Search_A(Opt_Finder* const f, const Uint8* const source, const Uint32 currIdx, const int dictSize,
	const Uint8* const dictEnd, const int searchCnt, Opt_Cand* const out)
{
	const Uint8* const srcPtr = source + currIdx;
	const reg_t currPattern = MemReadARCH(srcPtr);
	Uint32 matchDist = f->chainA[currIdx & f->maskA];
	int bestLen = MinMatchLen - 1, cnt = min(4 * searchCnt, OPT_ChainMax), n = 0;
	while (matchDist < (Uint32)f->winA && cnt--) {
		const int matchIdx = (int)currIdx - (int)matchDist;
		if (matchIdx < -dictSize) break;
		const Uint8* const matchPtr = matchIdx < 0 ? dictEnd + matchIdx : srcPtr - matchDist;
		const reg_t diff = currPattern ^ MemReadARCH(matchPtr);
		const int len = diff ? (int)N_ZeroBytes(diff) : REG_SIZE;
		if (len > bestLen) {
			bestLen = len;
			out[n].len = len; out[n].off = matchDist; n++;
			if (len >= 5) break;
		}
		matchDist += f->chainA[(Uint32)matchIdx & f->maskA];
	}
	return n;
}

/* chain B (its positions inserted up to currIdx): the candidates of lengths 5-6, nearest first, until a match of
   length 7 */
ForceInlineTemplate int Opt_Search_B(Opt_Finder* const f, const Uint8* const source, const Uint32 currIdx, const int dictSize,
	const Uint8* const dictEnd, const Uint8* const srcLastMatch, const Uint8* const dictLastMatch, const int searchCnt,
	Opt_Cand* const out)
{
	const Uint8* const srcPtr = source + currIdx;
	const Uint32 currPattern = MemRead4(srcPtr);
	Uint32 matchDist = f->chainB[currIdx & f->maskB];
	int bestLen = 4, cnt = min(searchCnt, OPT_ChainMax), n = 0;
	while (matchDist < (Uint32)f->winB && cnt--) {
		const int matchIdx = (int)currIdx - (int)matchDist;
		if (matchIdx < -dictSize) break;
		const Uint8* const matchPtr = matchIdx < 0 ? dictEnd + matchIdx : srcPtr - matchDist;
		if (currPattern == MemRead4(matchPtr) && !CANNOT_REACH(bestLen + 1)) {
			const int len = 4 + (int)HIST_COUNT(srcPtr + 4, matchPtr + 4, srcLastMatch - srcPtr > MaxMatchLen ? srcPtr + MaxMatchLen : srcLastMatch);
			if (len > bestLen) {
				bestLen = len;
				out[n].len = min(len, MaxMatchLen); out[n].off = matchDist; n++;
				if (len >= 7) break;
			}
		}
		matchDist += f->chainB[(Uint32)matchIdx & f->maskB];
	}
	return n;
}

/* tree C: inserts the positions the parse skipped, then currIdx, listing its candidates of lengths 7+ */
ForceInlineTemplate int Opt_Search_C(Opt_Finder* const f, const Uint8* const source, const Uint32 currIdx, const int dictSize,
	const Uint8* const dictEnd, const Uint8* const srcLastMatch, const int searchCnt, Opt_Cand* const out)
{
	while (f->nextC < currIdx)
		Opt_Tree_Insert(f, source, (int)f->nextC++, searchCnt, srcLastMatch, dictEnd, dictSize, NULL);
	const int n = Opt_Tree_Insert(f, source, (int)currIdx, searchCnt, srcLastMatch, dictEnd, dictSize, out);
	f->nextC = currIdx + 1;
	return n;
}

/* the candidates of A, B and C (in that order in tmp), nearest first; keeps one only if it is longer than all nearer
   ones and its window admits its length */
ForceInlineTemplate int Opt_Filter(WZL_Sched* const S_, Opt_Cand* const tmp, const int nTmp, Opt_Cand* const cand)
{
	for (int i = 1; i < nTmp; i++) {
		const Opt_Cand c = tmp[i];
		int j = i - 1;
		while (j >= 0 && tmp[j].off > c.off) { tmp[j + 1] = tmp[j]; j--; }
		tmp[j + 1] = c;
	}
	int n = 0;
	Uint32 bestLen = MinMatchLen - 1;
	for (int i = 0; i < nTmp; i++)
		if (tmp[i].len > bestLen && tmp[i].off < (Uint32)WINDOW(OffWidth[min(8, tmp[i].len)])) {
			cand[n++] = tmp[i];
			bestLen = tmp[i].len;
		}
	return n;
}

/* Collects match candidates at currIdx: for each length, the nearest offset found for it (lengths and offsets both
   increase along the list), limited to offsets that the window of the length admits. */
ForceInlineTemplate int Opt_Candidates(WZL_Sched* const S_, Opt_Finder* const f, const Uint8* const source, Uint32 currIdx, const int dictSize,
	const Uint8* const dictEnd, const Uint8* const srcLastMatch, const Uint8* const dictLastMatch, int searchCnt,
	Opt_Cand* const cand, Opt_Cand* const tmp)
{
	Opt_Insert_A(f, source, currIdx, dictSize);
	Opt_Insert_B(f, source, currIdx, dictSize);
	int nTmp = Opt_Search_A(f, source, currIdx, dictSize, dictEnd, searchCnt, tmp);
	nTmp += Opt_Search_B(f, source, currIdx, dictSize, dictEnd, srcLastMatch, dictLastMatch, searchCnt, tmp + nTmp);
	nTmp += Opt_Search_C(f, source, currIdx, dictSize, dictEnd, srcLastMatch, searchCnt, tmp + nTmp);
	return Opt_Filter(S_, tmp, nTmp, cand);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Match finding in threads of its own, one per index (WZIP_MULTITHREAD) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* The finder's indexes (chain A, chain B, tree C) depend only on the positions inserted, not on the parse, so each can
   run in a thread of its own ahead of the parser: a producer searches every position in order, for its indexes, and
   writes the candidates into a ring of chunks; the parser takes those of each position from every producer, in the
   order A, B, C, and filters them as Opt_Candidates does. The parse, and so the output, is the same as with one
   thread (the single-threaded finder inserts the positions the parse skips with the same tree walk). The tree, the
   costliest index, is a tree per hash bucket, so it splits further: producers that each take the buckets of one
   residue modulo their count, and the dictionary's positions, inserted by every thread at once (Opt_Prime_Tree)
   before the tree is searched. Producers by worker count: 2: one (A, B, C); 3: two (A and B, C); 4: three (A, B, C);
   5 to 7: A, B and the tree in 2 to 4 parts. */
#if WZIP_MULTITHREAD
#include "wz_threads.h"

#define   MT_ChunkLog          12                  /* positions handed over at a time */
#define   MT_Ring              8                   /* chunks a producer may run ahead of the parser */
#define   MT_MaxProd           6                   /* producers: A, B, and the tree in up to 4 parts */
#if (MT_Ring - 1) << MT_ChunkLog >= OPT_TreeLead
#  error "producers may run further apart than the tree's window allows (OPT_TreeLead)"
#endif

typedef struct {
	Opt_Cand* pool;                                /* the candidates of the chunk's positions, in order */
	size_t cap;
	Uint32 start[(1 << MT_ChunkLog) + 1];          /* position i's candidates: pool[start[i] .. start[i + 1]) */
} Opt_MT_Chunk;

typedef struct Opt_MT_s Opt_MT;
typedef struct {
	Opt_MT* mt;
	int mask;                                      /* its indexes: 1 chain A, 2 chain B, 4 tree C */
	int part, parts;                               /* of the tree: the buckets h with h % parts = part */
	long produced;                                 /* positions done (atomic) */
	WZ_Thread thread;
	Opt_MT_Chunk ring[MT_Ring];
} Opt_MT_Producer;

struct Opt_MT_s {
	Opt_Finder* f;
	const Uint8* source, *dictEnd, *srcLastMatch, *dictLastMatch;
	int dictSize, searchCnt;
	Uint32 end;                                    /* positions 0 .. end - 1 are searched */
	long consumed;                                 /* atomic: the start of the parser's chunk */
	long abort;                                    /* atomic: stop (the parse ended, or memory ran out) */
	long primed;                                   /* atomic: threads done with their part of the tree's dictionary */
	int nProd, started;
	Uint32 readyEnd, chunk;                        /* the parser's: positions ready from every producer, its chunk */
	Opt_MT_Producer prod[MT_MaxProd];
};

/* waits until *p >= v or the run is aborted; returns 0 if aborted */
static int Opt_MT_Wait(long* const p, const long v, long* const abortFlag)
{
	for (int spin = 0; WZ_LOAD(p) < v; spin++) {
		if (WZ_LOAD(abortFlag)) return 0;
		if (spin > 256) WZ_YIELD();
	}
	return 1;
}

WZ_THREAD_FN(Opt_MT_Producer_Main, arg)
{
	Opt_MT_Producer* const p = (Opt_MT_Producer*)arg;
	Opt_MT* const mt = p->mt;
	Opt_Finder* const f = mt->f;
	const Uint32 K = 1u << MT_ChunkLog;
	const size_t perPos = 2 * (size_t)mt->searchCnt + 32;   /* the most one position adds (as tmp's size) */
	const Uint8* const dict = mt->dictSize ? mt->dictEnd - mt->dictSize : NULL;
	Opt_Prime_Tree(f, dict, mt->dictSize, mt->source, mt->srcLastMatch, mt->searchCnt, (int)(p - mt->prod), mt->nProd + 1);
	WZ_FETCH_ADD(&mt->primed, 1);                  /* its part of the tree's dictionary; then its chains' */
	Opt_Finder_Prime(f, dict, mt->dictSize, mt->source, mt->srcLastMatch, mt->searchCnt, p->mask & 3);
	if ((p->mask & 4) && !Opt_MT_Wait(&mt->primed, mt->nProd + 1, &mt->abort)) return 0;
	for (Uint32 base = 0; base < mt->end; base += K) {
		const Uint32 chunk = base >> MT_ChunkLog;
		if (chunk >= MT_Ring && !Opt_MT_Wait(&mt->consumed, (long)(chunk - MT_Ring + 1) << MT_ChunkLog, &mt->abort)) break;
		Opt_MT_Chunk* const c = &p->ring[chunk % MT_Ring];
		const Uint32 lim = min(K, mt->end - base);
		size_t used = 0;
		for (Uint32 i = 0; i < lim; i++) {
			if (c->cap < used + perPos) {
				const size_t cap = 2 * c->cap + perPos;
				Opt_Cand* const q = (Opt_Cand*)realloc(c->pool, cap * sizeof(Opt_Cand));
				if (!q) { WZ_STORE(&mt->abort, 1); return 0; }
				c->pool = q; c->cap = cap;
			}
			c->start[i] = (Uint32)used;
			const Uint32 idx = base + i;
			if (p->mask & 1) Opt_Insert_A(f, mt->source, idx, mt->dictSize);
			if (p->mask & 2) Opt_Insert_B(f, mt->source, idx, mt->dictSize);
			if (p->mask & 1) used += Opt_Search_A(f, mt->source, idx, mt->dictSize, mt->dictEnd, mt->searchCnt, c->pool + used);
			if (p->mask & 2) used += Opt_Search_B(f, mt->source, idx, mt->dictSize, mt->dictEnd, mt->srcLastMatch, mt->dictLastMatch,
			                                      mt->searchCnt, c->pool + used);
			if (p->mask & 4) {
				if (p->parts == 1)
					used += Opt_Search_C(f, mt->source, idx, mt->dictSize, mt->dictEnd, mt->srcLastMatch, mt->searchCnt, c->pool + used);
				else if ((Hash_7B(mt->source + idx) & f->hMaskC) % (Uint32)p->parts == (Uint32)p->part)
					used += Opt_Tree_Insert(f, mt->source, (int)idx, mt->searchCnt, mt->srcLastMatch, mt->dictEnd, mt->dictSize,
					                        c->pool + used);
			}
		}
		c->start[lim] = (Uint32)used;
		WZ_STORE(&p->produced, (long)(base + lim));
	}
	return 0;
}

static void Opt_MT_Stop(Opt_MT* const mt)
{
	WZ_STORE(&mt->abort, 1);
	for (int k = 0; k < mt->started; k++) WZ_THREAD_JOIN(mt->prod[k].thread);
	mt->started = 0;
	for (int k = 0; k < mt->nProd; k++)
		for (int r = 0; r < MT_Ring; r++) { free(mt->prod[k].ring[r].pool); mt->prod[k].ring[r].pool = NULL; mt->prod[k].ring[r].cap = 0; }
}

/* starts the producers for positions 0 .. end - 1, and primes this thread's part of the tree; returns 0 (and leaves
   no thread running, the indexes to be emptied) if a thread cannot be started */
static int Opt_MT_Start(Opt_MT* const mt, const int workers)
{
	static const int masks[3][3] = { { 7 }, { 3, 4 }, { 1, 2, 4 } };
	mt->nProd = min(workers, MT_MaxProd + 1) - 1;
	mt->consumed = mt->abort = mt->primed = 0;
	mt->readyEnd = 0;
	mt->chunk = 0;
	for (int k = 0; k < mt->nProd; k++) {
		Opt_MT_Producer* const p = &mt->prod[k];
		p->mt = mt;
		p->mask = mt->nProd <= 3 ? masks[mt->nProd - 1][k] : k < 2 ? 1 << k : 4;
		p->part = mt->nProd <= 3 ? 0 : max(k - 2, 0);
		p->parts = mt->nProd <= 3 ? 1 : mt->nProd - 2;
		p->produced = 0;
	}
	for (mt->started = 0; mt->started < mt->nProd; mt->started++)
		if (!WZ_THREAD_START(&mt->prod[mt->started].thread, Opt_MT_Producer_Main, &mt->prod[mt->started])) {
			Opt_MT_Stop(mt);
			return 0;
		}
	Opt_Prime_Tree(mt->f, mt->dictSize ? mt->dictEnd - mt->dictSize : NULL, mt->dictSize, mt->source, mt->srcLastMatch,
	               mt->searchCnt, mt->nProd, mt->nProd + 1);
	WZ_FETCH_ADD(&mt->primed, 1);
	return 1;
}

/* the candidates of idx (positions taken in increasing order), as Opt_Candidates gives them; -1 if the producers
   stopped (memory ran out) */
static int Opt_MT_Candidates(WZL_Sched* const S_, Opt_MT* const mt, const Uint32 idx, Opt_Cand* const cand, Opt_Cand* const tmp)
{
	if (idx >= mt->readyEnd) {
		const Uint32 chunk = idx >> MT_ChunkLog, chunkEnd = min((chunk + 1) << MT_ChunkLog, mt->end);
		if (chunk != mt->chunk) {                    /* earlier chunks are done with: their slots may be refilled */
			mt->chunk = chunk;
			WZ_STORE(&mt->consumed, (long)chunk << MT_ChunkLog);
		}
		for (int k = 0; k < mt->nProd; k++)
			if (!Opt_MT_Wait(&mt->prod[k].produced, (long)chunkEnd, &mt->abort)) return -1;
		mt->readyEnd = chunkEnd;
	}
	const Uint32 i = idx & ((1u << MT_ChunkLog) - 1), slot = (idx >> MT_ChunkLog) % MT_Ring;
	int nTmp = 0;
	for (int k = 0; k < mt->nProd; k++) {
		const Opt_MT_Chunk* const c = &mt->prod[k].ring[slot];
		const Uint32 n = c->start[i + 1] - c->start[i];
		memcpy(tmp + nTmp, c->pool + c->start[i], n * sizeof(Opt_Cand));
		nTmp += (int)n;
	}
	return Opt_Filter(S_, tmp, nTmp, cand);
}
#endif

ForceInlineTemplate void Opt_Relax(Opt_Node* const opt, int* const lastPos, int from, int fromC, Uint32 len, Uint32 off, int price)
{
	const int to = from + (int)len;
	while (*lastPos < to) {
		++*lastPos;
		opt[*lastPos * OPT_C].price = opt[*lastPos * OPT_C + 1].price = opt[*lastPos * OPT_C + 2].price = OPT_Inf;
	}
	if (price < opt[to * OPT_C].price) {        /* a match ends in class 0 */
		Opt_Node* const node = opt + to * OPT_C;
		node->price = price;
		node->litLen = 0;
		node->mLen = len;
		node->mOff = off;
		node->prevC = fromC;
		const Uint32* const rep = opt[from * OPT_C + fromC].rep;
		if (off == 1)                              /* runs leave the cache alone */
			memcpy(node->rep, rep, sizeof(node->rep));
		else {                                     /* the offset moves to the front; a new one pushes out the oldest */
			const int k = off == rep[0] ? 0 : off == rep[1] ? 1 : off == rep[2] ? 2 : off == rep[3] ? 3 : 4;
			node->rep[3] = k >= 3 ? rep[2] : rep[3];
			node->rep[2] = k >= 2 ? rep[1] : rep[2];
			node->rep[1] = k >= 1 ? rep[0] : rep[1];
			node->rep[0] = off;
		}
	}
}

static Uint32 WLZ2_Compress_Opt_Pass(
	WZIP_State_Str* const wzipStr,
	const Uint8* const source,
	const Uint32 srcSize,
	Uint8* wzipStream,
	int wzipCapSize,
	int maxSearchCnt,
	int sufficientLen,
	const int nStates,                     /* states per position: 1 or OPT_C */
	const Opt_Stats* const regionIn,       /* symbol counts per region from a previous pass, which price each region */
	Opt_Stats* const regionOut)            /* receives the symbol counts per region of this pass (zeroed by the caller) */
{
	SCHED(wzipStr);
	const Uint8* const srcEnd = source + srcSize;
	const Uint8* const srcLastMatch = srcEnd - REG_SIZE * 2;
	const Uint32 lastMatchIdx = srcSize > REG_SIZE * 2 ? srcSize - REG_SIZE * 2 : 0;
	const int dictSize = wzipStr->dictSize;
	const Uint8* dictEnd = wzipStr->dictEnd;
	const Uint8* const dictLastMatch = dictSize ? dictEnd - REG_SIZE * 2 : NULL;
	Uint32 nLzLits = 0;
	Huffman_Str litHuf[N_HufLits];
	Uint8* wzipLitPtr = wzipStream;
	const Uint8* const wzipLitEnd = wzipLitPtr + wzipCapSize;
	wzipLitPtr += (srcSize >> 16) ? 4 : 2;                 /* Reserved to record the number of literals */
	Uint32 zipLitBlkSize;
	Uint32 lastOffset[OffCasheSize];
	int litRunHufIdx, mchLenHufIdx, offsetHufIdx, extra;

	WLZ_Huffman_Set huffmanSet;
	Huffman_Prev litPrev;                              /* the literal code that later blocks may reuse */
	litPrev.valid = 0;
	Seq_Prev seqPrev;                                  /* the sequence codes that later blocks may reuse */
	Seq_Prev_Init(&seqPrev);
	memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));
	Uint8* const lzLitBuffer = (Uint8*)malloc(HUF_BlockSize);
	WLZ_Set* const wlzSeq = (WLZ_Set*)malloc(SEQ_BlockSize * sizeof(WLZ_Set));
	Uint8* wlzStream = (Uint8*)malloc(2 * SEQ_BlockBound + 64);         /* one coded block of sequences, and room for its stream B */
	Uint8* const litScratch = (Uint8*)malloc(HUF_BlockSize + LIT_BlockSlack + 64);
	Seq_Stack seqStack;
	const int stackOk = Seq_Stack_Init(&seqStack, wzipStream + wzipCapSize);
	Opt_Node* const opt = (Opt_Node*)malloc((size_t)(OPT_Num + MaxMatchLen + 2) * OPT_C * sizeof(Opt_Node));
	Opt_Cand* const cand = (Opt_Cand*)malloc((2 * maxSearchCnt + 32) * sizeof(Opt_Cand));
	Opt_Cand* const tmp = (Opt_Cand*)malloc((2 * maxSearchCnt + 32) * sizeof(Opt_Cand));
	Opt_Finder finder;
#if WZIP_MULTITHREAD
	Opt_MT* mt = NULL;
	const int useMT = wzipStr->nbWorkers > 1 && lastMatchIdx > 0;   /* the indexes are then primed by their threads */
#else
	const int useMT = 0;
#endif
	const int finderOk = Opt_Finder_Init(S_, &finder, dictSize ? dictEnd - dictSize : NULL, dictSize, source, srcLastMatch, maxSearchCnt,
	                                     useMT ? 0 : 7);
	Opt_Stats* const st = (Opt_Stats*)malloc(sizeof(Opt_Stats));
	Opt_LenTab* const lenTab = (Opt_LenTab*)malloc(sizeof(Opt_LenTab));
	Opt_Path* const path = (Opt_Path*)malloc(((OPT_Num + MaxMatchLen) / MinMatchLen + 2) * sizeof(Opt_Path));
	if (!finderOk || !lzLitBuffer || !wlzSeq || !wlzStream || !litScratch || !stackOk || !opt || !cand || !tmp || !st || !path || !lenTab)
		goto _lit_overflow;
#if WZIP_MULTITHREAD
	if (useMT) {                                       /* match finding in threads of its own; else, all here */
		if (NULL != (mt = (Opt_MT*)calloc(1, sizeof(Opt_MT)))) {
			mt->f = &finder; mt->source = source; mt->dictEnd = dictEnd; mt->dictSize = dictSize;
			mt->srcLastMatch = srcLastMatch; mt->dictLastMatch = dictLastMatch; mt->searchCnt = maxSearchCnt;
			mt->end = lastMatchIdx;
			if (!Opt_MT_Start(mt, wzipStr->nbWorkers)) { free(mt); mt = NULL; }
		}
		if (NULL == mt) {                              /* all here, from empty indexes (threads may have begun them) */
			Opt_Finder_Reset(&finder);
			Opt_Finder_Prime(&finder, dictSize ? dictEnd - dictSize : NULL, dictSize, source, srcLastMatch, maxSearchCnt, 7);
		}
	}
#endif
	Uint8* lzLitPtr = lzLitBuffer;
	const Uint8* const lzLitEnd = lzLitBuffer + HUF_BlockSize;
	WLZ_Set* wlzSeqPtr = wlzSeq;
	WLZ_Set* const wlzSeqEnd = wlzSeq + SEQ_BlockSize;

	memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
	memset(lastOffset, 0xFF, OffCasheSize * sizeof(int));     /* unset: above every offset, so never a hit */
	Opt_Init_Stats(S_, st, source, srcSize);
	Opt_Init_LenTab(lenTab);
	Uint32 nextUpdate = regionIn ? 0 : OPT_UpdateBytes, seqCount = 0;

	Uint32 anchor = 0, pos = 0;
	while (pos < lastMatchIdx) {
		if (pos >= nextUpdate) {
			if (regionIn) {                                          /* the counts of this region in the previous pass */
				const Opt_Stats* const r = regionIn + (pos >> OPT_RegionLog);
				memcpy(st->litFreq, r->litFreq, sizeof(st->litFreq));
				memcpy(st->litRunFreq, r->litRunFreq, sizeof(st->litRunFreq));
				memcpy(st->mchLenFreq, r->mchLenFreq, sizeof(st->mchLenFreq));
				memcpy(st->mchOffFreq, r->mchOffFreq, sizeof(st->mchOffFreq));
				Opt_Update_Prices(S_, st);
				nextUpdate = ((pos >> OPT_RegionLog) + 1) << OPT_RegionLog;
			}
			else {
				if (seqCount > (1u << 16)) { Opt_Age_Stats(S_, st); seqCount >>= 1; }
				Opt_Update_Prices(S_, st);
				nextUpdate = pos + OPT_UpdateBytes;
			}
		}

		/* ---- shortest path over the segment starting at pos, nStates states (literal-run classes) per position */
#define OPT_CLS(litLen)   (nStates > 1 ? OPT_Class(litLen) : 0)
		int lastPos = 0, endCur = -1;
		{
			const int litLen0 = (int)(pos - anchor), c0 = OPT_CLS(litLen0);
			opt[0].price = opt[1].price = opt[2].price = OPT_Inf;
			Opt_Node* const n0 = opt + c0;
			n0->litLen = litLen0;
			n0->price = LitRun_Price(S_, st, litLen0);
			n0->mLen = 0;
			n0->prevC = c0;
			memcpy(n0->rep, lastOffset, sizeof(lastOffset));
		}
		const int newRunPrice = LitRun_Price(S_, st, 0);

		for (int cur = 0; ; cur++) {
			if (cur > 0) {                                           /* literal steps into cur */
				if (cur > lastPos) {
					opt[cur * OPT_C].price = opt[cur * OPT_C + 1].price = opt[cur * OPT_C + 2].price = OPT_Inf;
					lastPos = cur;
				}
				const int litPrice = st->litPrice[source[pos + cur - 1]];
				for (int pc = 0; pc < nStates; pc++) {
					const Opt_Node* const prev = opt + (cur - 1) * OPT_C + pc;
					if (prev->price >= OPT_Inf) continue;
					const int litLen = prev->litLen + 1, c = OPT_CLS(litLen);
					const int price = prev->price + litPrice + LitRun_Price(S_, st, litLen) - LitRun_Price(S_, st, litLen - 1);
					Opt_Node* const node = opt + cur * OPT_C + c;
					if (price < node->price) {
						node->price = price;
						node->litLen = litLen;
						node->mLen = 0;
						node->prevC = pc;
						memcpy(node->rep, prev->rep, sizeof(prev->rep));
					}
				}
			}
			const Uint32 idx = pos + cur;
			const Uint8* const countEnd = srcLastMatch - (source + idx) > MaxMatchLen ? source + idx + MaxMatchLen : srcLastMatch;   /* lengths are capped there anyway */
			if ((cur > 0 && cur >= lastPos) || cur >= OPT_Num || idx >= lastMatchIdx) { endCur = cur; break; }

			/* searched matches, shared by the states */
#if WZIP_MULTITHREAD
			const int nCand = mt ? Opt_MT_Candidates(S_, mt, idx, cand, tmp)
			                     : Opt_Candidates(S_, &finder, source, idx, dictSize, dictEnd, srcLastMatch, dictLastMatch, maxSearchCnt, cand, tmp);
			if (nCand < 0) goto _lit_overflow;             /* a producer ran out of memory */
#else
			const int nCand = Opt_Candidates(S_, &finder, source, idx, dictSize, dictEnd, srcLastMatch, dictLastMatch, maxSearchCnt, cand, tmp);
#endif
			Uint32 longest = nCand ? cand[nCand - 1].len : 0, longestOff = nCand ? cand[nCand - 1].off : 0;
			int bestC = 0;
			for (int c = 1; c < nStates; c++)
				if (opt[cur * OPT_C + c].price < opt[cur * OPT_C + bestC].price) bestC = c;

			for (int c = 0; c < nStates; c++) {
				const Opt_Node* const node = opt + cur * OPT_C + c;
				if (node->price >= OPT_Inf) continue;
				const int base = node->price + newRunPrice;
				const int lc = LitClass(min(node->litLen, 2));
				/* repeat offsets */
				for (int k = 0; k < OffCasheSize; k++) {
					const Uint32 off = node->rep[k];
					if (k && (off == node->rep[0] || (k > 1 && off == node->rep[1]) || (k > 2 && off == node->rep[2]))) continue;
					Uint32 len = (Uint32)Repeat_Match_Len(source, idx, off, dictSize, dictEnd, countEnd, dictLastMatch);
					if (len < MinMatchLen) continue;
					len = min(len, MaxMatchLen);
					if (c == bestC && len > longest) { longest = len; longestOff = off; }
					if (len >= (Uint32)sufficientLen) continue;
					if (off == 1)                                /* a run (never cached; kept for exactness) */
						for (Uint32 l = MinMatchLen; l <= len; l++)
							Opt_Relax(opt, &lastPos, cur, c, l, off, base + Match_Price(S_, st, l, off, node->rep, node->litLen));
					else {                                       /* slot k: its joint symbol only, no window */
						const int* const lp = st->mchLenPrice + JointIdx(k, lc, 0);
						for (Uint32 l = MinMatchLen; l <= len; l++)
							Opt_Relax(opt, &lastPos, cur, c, l, off, base + lp[lenTab->sym[l]] + lenTab->extra[l]);
					}
				}
				if (longest >= (Uint32)sufficientLen) continue;
				if (idx >= 1 && source[idx] == source[idx - 1] && source[idx + 1] == source[idx - 1])    /* a run of two */
					Opt_Relax(opt, &lastPos, cur, c, 2, 1, base + Match_Price(S_, st, 2, 1, node->rep, node->litLen));
				Uint32 l = MinMatchLen;
				for (int ci = 0; ci < nCand; ci++) {
					const Uint32 off = cand[ci].off, clen = cand[ci].len;
					if (l > clen) continue;
					const int k = off == node->rep[0] ? 0 : off == node->rep[1] ? 1 : off == node->rep[2] ? 2 : off == node->rep[3] ? 3 : OffCasheSize;
					if (off == 1) {                              /* a run */
						for (; l <= clen; l++)
							Opt_Relax(opt, &lastPos, cur, c, l, off, base + Match_Price(S_, st, l, off, node->rep, node->litLen));
					}
					else if (k < OffCasheSize) {                 /* in the cache: its slot code only */
						const int* const lp = st->mchLenPrice + JointIdx(k, lc, 0);
						for (; l <= clen; l++)
							Opt_Relax(opt, &lastPos, cur, c, l, off, base + lp[lenTab->sym[l]] + lenTab->extra[l]);
					}
					else {                                       /* a new offset: its code in the offset group of each length */
						int offExtra;
						const int offSym = Offset_Symbol(off + OffCasheSize - 1, &offExtra);
						int offPrice[MaxMchOffGroup];
						for (int g = 0; g < MchOffGroup; g++) offPrice[g] = st->mchOffPrice[g][offSym] + offExtra * OPT_Unit;
						const int* const lp = st->mchLenPrice + JointIdx(OffCasheSize, lc, 0);
						while (l <= clen && off >= (Uint32)WINDOW(OffWidth[min(8, l)])) l++;   /* windows widen with the length */
						for (; l <= clen; l++) {
							const int sym = lenTab->sym[l];
							Opt_Relax(opt, &lastPos, cur, c, l, off, base + lp[sym] + lenTab->extra[l] + offPrice[OffGroupOf[sym]]);
						}
					}
				}
			}
			if (longest >= (Uint32)sufficientLen) {                  /* take it from the cheapest state and end the segment */
				while (lastPos > cur) {
					opt[lastPos * OPT_C].price = opt[lastPos * OPT_C + 1].price = opt[lastPos * OPT_C + 2].price = OPT_Inf;
					lastPos--;
				}
				Opt_Relax(opt, &lastPos, cur, bestC, longest, longestOff, 0);
				endCur = lastPos;
				break;
			}
		}
#undef OPT_CLS

		/* ---- back-trace the cheapest state at endCur; its matches are stored last to first */
		int nMatch = 0;
		{
			int c = 0;
			for (int k = 1; k < nStates; k++)
				if (opt[endCur * OPT_C + k].price < opt[endCur * OPT_C + c].price) c = k;
			for (int cur = endCur; cur > 0; ) {
				const Opt_Node* const node = opt + cur * OPT_C + c;
				if (node->mLen) {
					path[nMatch].start = pos + (Uint32)cur - node->mLen;
					path[nMatch].len = node->mLen;
					path[nMatch].off = node->mOff;
					nMatch++;
					cur -= (int)node->mLen;
				}
				else cur--;
				c = node->prevC;
			}
		}

		/* ---- emit the matches in order, exactly as the other parsers do */
		for (int i = nMatch - 1; i >= 0; i--) {
			const Uint32 start = path[i].start, len = path[i].len, rawOff = path[i].off;

			/* literals */
			Opt_Stats* const ro = regionOut ? regionOut + (start >> OPT_RegionLog) : NULL;
			for (Uint32 q = anchor; q < start; q++) {
				*lzLitPtr++ = source[q];
				litHuf[source[q]].freq++;
				st->litFreq[source[q]]++;
				if (ro) ro->litFreq[source[q]]++;
				if (lzLitPtr == lzLitEnd) {
					LIT_PUT_BLOCK(HUF_BlockSize);
					wzipLitPtr += zipLitBlkSize;
					lzLitPtr = lzLitBuffer;
					memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
					nLzLits += HUF_BlockSize;
				}
			}
			const Uint32 litRun = start - anchor;
			litRunHufIdx = LitRun_Symbol(S_, litRun, &extra);
			wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[extra]) << 8;
			if (litRunHufIdx >= 2) { st->litRunFreq[litRunHufIdx]++; if (ro) ro->litRunFreq[litRunHufIdx]++; }

			if (rawOff == 1) {                           /* a run */
				const Uint32 jc = JointIdx(OffCasheSize, LitClass(litRunHufIdx), RunSym), v = len + OffCasheSize - 1;
				const int os = Offset_Huffman_Index((int)v, High_Bit32(v)), group = OffGroupOf[RunSym];
				st->mchLenFreq[jc]++;
				st->mchOffFreq[group][os]++;
				if (ro) { ro->mchLenFreq[jc]++; ro->mchOffFreq[group][os]++; }
				wlzSeqPtr = Store_Run(S_, wlzSeqPtr, wlzSeq, &huffmanSet, litRun, len) - 1;
			}
			else {
			const Uint32 v = Offset_Cashe(lastOffset, rawOff);
			mchLenHufIdx = MchLen_Symbol(len, &extra);
			wlzSeqPtr->mchLen = mchLenHufIdx ^ (len & BitMask[extra]) << 8;
			const Uint32 jc = JointIdx(v < OffCasheSize ? v : OffCasheSize, LitClass(litRunHufIdx), mchLenHufIdx);
			st->mchLenFreq[jc]++;
			if (ro) ro->mchLenFreq[jc]++;
			const int group = OffGroupOf[mchLenHufIdx];
			offsetHufIdx = Offset_Symbol(v, &extra);
			wlzSeqPtr->mchOff = offsetHufIdx ^ (v & BitMask[extra]) << OFF_SymBits;
			if (v >= OffCasheSize) {                    /* a cache slot is coded in the joint symbol */
				huffmanSet.mchOffHuf[group][offsetHufIdx].freq++;
				st->mchOffFreq[group][offsetHufIdx]++;
				if (ro) ro->mchOffFreq[group][offsetHufIdx]++;
			}
			}
			seqCount++;

			anchor = start + len;
			if (++wlzSeqPtr == wlzSeqEnd) {
				SEQ_PUT_BLOCK();
				memset(&huffmanSet, 0, sizeof(WLZ_Huffman_Set));
				wlzSeqPtr = wlzSeq;
			}
		}
		pos += endCur > 0 ? (Uint32)endCur : 1;
	}

	/* ---- last literals, then the terminating record */
	const Uint8* srcPtr = source + anchor;
	const Uint32 litRun = srcSize - anchor;
	if (lzLitPtr + litRun > lzLitEnd) {
		while (lzLitPtr < lzLitEnd) {
			litHuf[*srcPtr].freq++;
			*lzLitPtr++ = *srcPtr++;
		}
		LIT_PUT_BLOCK(HUF_BlockSize);
		wzipLitPtr += zipLitBlkSize;
		lzLitPtr = lzLitBuffer;
		memset(litHuf, 0, N_HufLits * sizeof(Huffman_Str));
		nLzLits += HUF_BlockSize;
	}
	while (srcPtr < srcEnd) {
		litHuf[*srcPtr].freq++;
		*lzLitPtr++ = *srcPtr++;
	}
	const Uint32 lastBufLits = (Uint32)(lzLitPtr - lzLitBuffer);
	if (lastBufLits) {                                   /* none when the literals filled their last block exactly */
		LIT_PUT_BLOCK(lastBufLits);
		wzipLitPtr += zipLitBlkSize;
	}
	nLzLits += lastBufLits;
	if (srcSize >> 16) MemWriteLE4(wzipStream, nLzLits);
	else               MemWriteLE2(wzipStream, (Uint16)nLzLits);

	litRunHufIdx = LitRun_Symbol(S_, litRun, &extra);
	wlzSeqPtr->litRun = litRunHufIdx ^ (litRun & BitMask[extra]) << 8;
	huffmanSet.litRunHuf[litRunHufIdx].freq++;
	wlzSeqPtr->mchLen = 255;    /* protocal for ending */
	wlzSeqPtr++;
	SEQ_PUT_BLOCK();
	const int cmprSize = (int)((wzipLitPtr - wzipStream) + Seq_Stack_Place(&seqStack, wzipLitPtr));

#if WZIP_MULTITHREAD
	if (mt) { Opt_MT_Stop(mt); free(mt); mt = NULL; }
#endif
	Opt_Finder_Free(&finder);
	free(lzLitBuffer); free(wlzSeq); free(wlzStream); free(litScratch); free(seqStack.sizes); free(opt); free(cand); free(tmp); free(st); free(path); free(lenTab);
	return cmprSize;
_lit_overflow:                 /* the output buffer is full: the caller stores the input raw */
#if WZIP_MULTITHREAD
	if (mt) { Opt_MT_Stop(mt); free(mt); mt = NULL; }
#endif
	Opt_Finder_Free(&finder);
	free(lzLitBuffer); free(wlzSeq); free(wlzStream); free(litScratch); free(seqStack.sizes); free(opt); free(cand); free(tmp); free(st); free(path); free(lenTab);
	return 0;
}

/* Optimal parsing in one or more passes: after the first, each pass prices every region from the symbol counts the
   previous pass found in it (the parse and the codes it implies are refined in turn) */
static Uint32 WLZ2_Compress_Opt(WZIP_State_Str* const wzipStr, const Uint8* const source, const Uint32 srcSize,
	Uint8* wzipStream, int wzipCapSize, int maxSearchCnt, int sufficientLen, int passes, int nStates)
{
	const size_t nRegions = ((size_t)srcSize >> OPT_RegionLog) + 1;
	Opt_Stats* regionIn = NULL;
	Uint32 size = 0;
	for (int k = 0; k < passes; k++) {
		Opt_Stats* regionOut = NULL;
		if (k + 1 < passes && NULL == (regionOut = (Opt_Stats*)calloc(nRegions, sizeof(Opt_Stats)))) {
			free(regionIn);
			return 0;
		}
		const int last = k + 1 == passes;                       /* the others gather statistics */
		size = WLZ2_Compress_Opt_Pass(wzipStr, source, srcSize, wzipStream, wzipCapSize, last ? maxSearchCnt : OPT_StatsDepth,
		                              last ? sufficientLen : OPT_StatsSufficient, last ? nStates : 1, regionIn, regionOut);
		free(regionIn);
		regionIn = regionOut;
		if (!size) break;                                   /* the output buffer is full */
	}
	free(regionIn);
	return size;
}

/* The stream starts with the windows of lengths 3 to 7, each as its distance below the widest window (that of length 8,
   which the decoder derives from the size), 4 bits each: d3 | d4 << 4, d5 | d6 << 4, d7 | groups << 4, where groups is
   0 for the natural offset groups and 1 for the fine layout (other values are reserved) */
#define   WIN_HeaderSize       3
/* optimal parsing prices far short matches exactly: its windows of lengths 3, 4, 5 reach at least this close to the
   widest (on Silesia: +0.7% at level 11 over the default windows, which suit the greedy and lazy parsers better) */
static const int WideWinGap[3] = { 7, 3, 1 };

static void WZL_Insert_Dict(WZIP_State_Str* const wzipStr, const int from, const int to);
#define   WZL_PrimeBytes       (1 << 24)          /* the reach into a dictionary of a hash table without a chain */

int WZIP_Compress_L(WZIP_State_Str* wzipStr, const void* const source, int srcSize, void* const wzipStream, int wzipCapSize)
{
	SCHED(wzipStr);
	Uint8* const header = (Uint8*)wzipStream;
	header[0] = (Uint8)((OffWidth[8] - OffWidth[3]) | (OffWidth[8] - OffWidth[4]) << 4);
	header[1] = (Uint8)((OffWidth[8] - OffWidth[5]) | (OffWidth[8] - OffWidth[6]) << 4);
	header[2] = (Uint8)((OffWidth[8] - OffWidth[7]) | OffGroupsFine << 4);
	Uint8* const stream = header + WIN_HeaderSize;
	const int cap = wzipCapSize - WIN_HeaderSize;
	Uint32 size;
	LitRunTooLong = 0;
	/* a dictionary just before the input: its last 15 positions too, whose compares run on into the input */
	if (wzipStr->dictSize && wzipStr->dictEnd == (Uint8*)source && wzipStr->compressLevel <= 6)
		WZL_Insert_Dict(wzipStr, -min(wzipStr->dictSize, 15), -1);
	if (0 == wzipStr->compressLevel && 0 == wzipStr->dictSize)      /* the fast mode; with a dictionary, level 1's loop */
		size = WLZ2_Compress_Fast1(wzipStr, (Uint8*)source, srcSize, stream, cap);
	else if (wzipStr->compressLevel <= 1)
		size = WLZ2_Compress_Fast(wzipStr, (Uint8*)source, srcSize, stream, cap);
	else if (wzipStr->compressLevel >= 7)
		size = WLZ2_Compress_Opt(wzipStr, (Uint8*)source, srcSize, stream, cap, wzipStr->maxSearchCnt, OPT_LevelSufficient[wzipStr->compressLevel - 7],
		                         OPT_LevelPasses[wzipStr->compressLevel - 7], OPT_LevelStates[wzipStr->compressLevel - 7]);
	else
		size = WLZ2_Compress(wzipStr, (Uint8*)source, srcSize, stream, cap, wzipStr->maxSearchCnt);
	if (LitRunTooLong) size = 0;                         /* a literal run the format cannot code: the caller stores */
	return size ? (int)size + WIN_HeaderSize : 0;
}

/* Inserts the dictionary positions from..to (negative) into the hash tables and chains of levels 0-6, each table only
   as far back as it reaches: its window, and for a table without a chain, which keeps one position per hash, at most
   WZL_PrimeBytes (older positions are mostly overwritten anyway). */
static void WZL_Insert_Dict(WZIP_State_Str* const wzipStr, const int from, const int to)
{
	SCHED(wzipStr);
	int* hash0Table = (int*)wzipStr->hash0Table;
	int* hash1Table = (int*)wzipStr->hash1Table;
	int* hash2Table = (int*)wzipStr->hash2Table;
	Uint32* chain1Table = (Uint32*)wzipStr->chain1Table;
	Uint32* chain2Table = (Uint32*)wzipStr->chain2Table;
	const Uint32 chain1Mask = wzipStr->chain1Mask;
	const Uint32 chain2Mask = wzipStr->chain2Mask;
	const int hash1Len = wzipStr->hash1Len;
	const int hash2Len = wzipStr->hash2Len;
	const int reach0 = min(WINDOW(SrchWidth[3]), WZL_PrimeBytes);
	const int reach1 = chain1Mask ? WINDOW(SrchWidth[hash2Len - 1]) : min(WINDOW(SrchWidth[hash2Len - 1]), WZL_PrimeBytes);
	const int reach2 = chain2Mask ? WINDOW(SrchWidth[8]) : min(WINDOW(SrchWidth[8]), WZL_PrimeBytes);
	const int from0 = max(from, -reach0), from1 = max(from, -reach1), from2 = max(from, -reach2);

	int hashV, dist, matchIdx;
	for (int i = from0; i <= to; i++) {
		hashV = WLZ_Hash0(wzipStr->dictEnd + i) & wzipStr->hash0Mask;
		hash0Table[hashV] = i;
	}
	for (int i = from1; i <= to; i++) {
		hashV = WLZ_Hash1(wzipStr->dictEnd + i) & wzipStr->hash1Mask;
		matchIdx = hash1Table[hashV];
		hash1Table[hashV] = i;
		if (chain1Mask) {
			dist = i - matchIdx;
			chain1Table[(Uint32)i & chain1Mask] = (dist > 0 && dist < chain1Mask) ? dist : chain1Mask;
		}
	}
	for (int i = from2; i <= to; i++) {
		hashV = WLZ_Hash2(wzipStr->dictEnd + i) & wzipStr->hash2Mask;
		matchIdx = hash2Table[hashV];
		hash2Table[hashV] = i;
		if (chain2Mask) {
			dist = i - matchIdx;
			chain2Table[(Uint32)i & chain2Mask] = (dist > 0 && dist < chain2Mask) ? dist : chain2Mask;
		}
	}
}

/* the threads WZIP_Compress_L may use (WZIP.h) */
void WZIP_Set_Workers(WZIP_State_Str* const wzipStr, const int nbWorkers)
{
	if (wzipStr) wzipStr->nbWorkers = nbWorkers < 1 ? 1 : nbWorkers;
}

WZIP_State_Str* WZIP_New_State_L(int level, int srcSize, const void* dict, int dictSize)
{
	WZIP_State_Str* const wzipStr = (WZIP_State_Str*)calloc(1, sizeof(WZIP_State_Str));
	WZL_Sched* const S_ = (WZL_Sched*)calloc(1, sizeof(WZL_Sched));
	if (NULL == wzipStr || NULL == S_) { free(wzipStr); free(S_); return NULL; }
	wzipStr->sched = S_;
	if (NULL == dict || dictSize < 0) dictSize = 0;
	wzipStr->compressLevel = level;
	wzipStr->dictSize = dictSize;
	wzipStr->dictEnd = dict ? (Uint8*)dict + dictSize : NULL;

	WZIP_Set_OffWidth(WZL_History(srcSize, dictSize), OffWidth);
	/* wider short windows for optimal parsing, measured from the widest window of the short lengths (2^26 at most; on
	   enwik8, widening from the 2^27 window of lengths 8+ lost 0.06%); gaps fit in 4 bits */
	const int shortTop = min(OffWidth[8], WZIP_SHORT_OFF_WIDTH);
	for (int k = 3; k <= 7; k++) {
		if (level >= 7 && k <= 5) OffWidth[k] = max(OffWidth[k], shortTop - WideWinGap[k - 3]);
		OffWidth[k] = max(OffWidth[k], OffWidth[8] - 15);
	}
	/* windows widen with the length (the decoder checks it): from 16 MB on, the widening above can pass the next length's */
	for (int k = 4; k <= 8; k++)
		OffWidth[k] = max(OffWidth[k], OffWidth[k - 1]);

	int n = 8;
	while (n > 0 && OffWidth[n] == OffWidth[n - 1]) 
		n--;
	wzipStr->hash2Len = n;
	/* The encoder searches the input's windows under the level's cap (WZIP_LEVEL_WINDOW_LOG): the widest window is cut
	   to the cap, and each narrower one moves down just enough to stay below the next wider one, so that the windows
	   keep their order (lengths with equal windows keep them equal, and longer matches reach further). Its tables and
	   match finders follow SrchWidth; the stream keeps the input's windows, which set its offset codes, so decoders need
	   no level. With SrchWidth <= OffWidth for every length, every match found fits its length's window. */
	{
		int prev = WZIP_LEVEL_WINDOW_LOG(level) + 1;              /* the cut of the next wider window, plus one */
		for (int k = 8; k >= 3; k--) {
			if (k == 8 || OffWidth[k] != OffWidth[k + 1]) prev = min(OffWidth[k], prev - 1);
			SrchWidth[k] = prev;
		}
		SrchWidth[0] = SrchWidth[1] = SrchWidth[2] = 0;
	}
	

	if (wzipStr->hash2Len>=7) {
		wzipStr->hash1Len = 5;
	} else {
		wzipStr->hash1Len = 4;
	}

	OffGroupsFine = level == 13;
	Set_Offset_Groups(S_, wzipStr->hash2Len - MinMatchLen + 1, OffGroupsFine);
	

	wzipStr->hash0Mask = BitMask[SrchWidth[3] + 4];
	wzipStr->hash1Mask = BitMask[level > 1 ? SrchWidth[wzipStr->hash2Len-1] : min(SrchWidth[wzipStr->hash2Len-1], L0_HashLog)];
	wzipStr->hash2Mask = BitMask[level > 1 ? SrchWidth[8] : min(SrchWidth[8], L0_HashLog)];
	if (level <= 1 && level >= 0) {		
		wzipStr->chain1Mask = 0;
		wzipStr->chain2Mask = 0;
	}
	else if (level > 1 && level <= 13) {
		/* chain search depth: levels 2-3 greedy, 4-6 lazy, 7-13 optimal parsing (lazy searches deeper than 64 lose to the
		   optimal parser at equal speed) */
		static const int lazyDepth[7] = { 0, 0, 4, 16, 8, 16, 64 };
		wzipStr->maxSearchCnt = level >= 7 ? OPT_LevelDepth[level - 7] : lazyDepth[level];
		wzipStr->chain1Mask = level > 3 ? BitMask[SrchWidth[wzipStr->hash2Len - 1]] : 0;
		wzipStr->chain2Mask = BitMask[SrchWidth[8]];
	}
	else {
		fprintf(stderr, "compression level must be in [0, 13]\n");
		WZIP_Free_State(wzipStr);
		return NULL;
	}
	if (level >= 7) return wzipStr;                    /* the optimal levels index the input and dictionary themselves */
	/* zeroed: an entry not yet written then names position 0, which is always in the history. Left unset, a stale
	   entry from reused memory could name a position near the dictionary's end, whose compare reads past it. */
	wzipStr->hash0Table = calloc((size_t)wzipStr->hash0Mask + 1, sizeof(int));
	wzipStr->hash1Table = calloc((size_t)wzipStr->hash1Mask + 1, sizeof(int));
	wzipStr->hash2Table = calloc((size_t)wzipStr->hash2Mask + 1, sizeof(int));

	if (0 == wzipStr->chain1Mask) wzipStr->chain1Table = NULL;
	else wzipStr->chain1Table = calloc((size_t)wzipStr->chain1Mask + 1, sizeof(Uint32));

	if (0 == wzipStr->chain2Mask) wzipStr->chain2Table = NULL;
	else wzipStr->chain2Table = calloc((size_t)wzipStr->chain2Mask + 1, sizeof(Uint32));
	
	if (NULL == wzipStr->hash0Table || NULL == wzipStr->hash1Table || NULL == wzipStr->hash2Table
	    || (wzipStr->chain1Mask && NULL == wzipStr->chain1Table) || (wzipStr->chain2Mask && NULL == wzipStr->chain2Table)) {
		WZIP_Free_State(wzipStr);
		return NULL;
	}
	if (dictSize == 0 || dict == NULL)
		return wzipStr;

	WZL_Insert_Dict(wzipStr, -dictSize, -16);  /* to -16: 8-byte compares stay inside a dictionary in a buffer of its own */
	return wzipStr;
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ WZIP Decoompression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* decoding table of the joint symbol: its fields split out, so no division is needed */
typedef struct {
	Uint8 mlSym, litClass, slotSel, nbits;
} Joint_DemapX1;

/* canonical code as Build_Huffman_Table assigns it: shorter codes first, equal lengths in symbol order */
static void Build_Joint_DecTable(const Uint32 maxBits, const Uint8* wt, Joint_DemapX1* table)
{
	if (0 == maxBits) return;
	Uint32 start[MAX_HufWeight + 2] = { 0 }, pos = 0;
	Uint32 count[MAX_HufWeight + 2] = { 0 };
	for (Uint32 k = 0; k < N_HufJoint; k++) count[wt[k]]++;
	for (Uint32 b = 1; b <= maxBits; b++) { start[b] = pos; pos += count[b] << (maxBits - b); }
	for (Uint32 k = 0; k < N_HufJoint; k++) {
		const Uint32 b = wt[k];
		if (!b) continue;
		const Joint_DemapX1 e = { (Uint8)(k % N_HufMchLen), (Uint8)(k / N_HufMchLen % N_LitClass), (Uint8)(k / (N_HufMchLen * N_LitClass)), (Uint8)b };
		for (Uint32 r = 0; r < (1u << (maxBits - b)); r++) table[start[b] + r] = e;
		start[b] += 1u << (maxBits - b);
	}
}

/* The decoding tables of a stream's sequence codes, kept from block to block: a block builds only those of the codes
   it sends, and reuses the others (Seq_Prev). The offset table of each length group is built at the full
   CapHufMchOffBits width, at a fixed stride: a table is selected by arithmetic on the group, and all share one shift. */
typedef struct {
	Huffman_DemapX1 litRun[1 << CapHufLitRunBits];
	Joint_DemapX1 joint[1 << CapHufMchLenBits];        /* slot-joint codes */
	Huffman_DemapX1 mchLen[1 << CapHufMchLenBits];     /* classic codes */
	Huffman_DemapX1 off[MaxMchOffGroup << CapHufMchOffBits];
	Uint32 remLitRun, remMchLen;                       /* the shifts that look them up */
	int haveLitRun, haveMchLen, haveOff[MaxMchOffGroup];
	Uint32 mchLenSlotJoint;                            /* the coding of the joint code */
} Seq_DecTables;

/* Reads a sequence block's coding, which tables it reuses, and the lengths of the codes it sends, into w, after the
   block's first bit (slot-joint or classic, in w->slotJoint); builds their tables. Checked (unless trusted): a reused
   table must have been sent before, the joint table by a block of the same coding, and every code is validated, so
   that entries an empty or one-symbol code leaves unset decode as symbol 0 with no bits and a corrupt stream that
   reaches them stays in bounds. Returns 0 if the header is corrupt. */
static int Seq_Read_Tables(WZL_Sched* const S_, Bit_Stream* const bs, WLZ_HufWt_Set* const w, Seq_DecTables* const T, const int trusted)
{
	Bit_Stream bitStream = *bs;
	const Uint32 nTab = 2 + MchOffGroup;
	Uint32 reuse[2 + MaxMchOffGroup] = { 0 }, nSent = 0, t;     /* initialized for GCC's -m32 flow analysis only */
	for (t = 0; t < nTab; t++) {
		BITStream_Read(bitStream, 1, reuse[t]);
		nSent += !reuse[t];
	}
	BITStream_Read_Flush(bitStream);
	if (!trusted) {
		if ((reuse[0] && !T->haveLitRun) || (reuse[1] && (!T->haveMchLen || T->mchLenSlotJoint != w->slotJoint))) return 0;
		for (t = 2; t < nTab; t++)
			if (reuse[t] && !T->haveOff[t - 2]) return 0;
	}
	if (nSent) {
		Uint8 wtHufWt[MAX_HufWeight + 3];                /* the weight code, which codes the lengths */
		Huffman_DemapX1 wtHufDemapX1[1 << MAX_HufHufWt];
		int maxWt, maxBits;
		if (trusted) maxWt = (int)Read_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, wtHufWt);
		else if ((maxWt = Huffman_Read_Code(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, wtHufWt)) <= 0) return 0;
		Build_Huffman_DecTableX1(MAX_HufWeight + 3, (Uint32)maxWt, wtHufWt, wtHufDemapX1);
#define SEQ_READ_CODE(n_, cap_, lens_, max_) {                                                                       \
			if (trusted) max_ = Read_Huffman_Header_byHuffman(&bitStream, (Uint32)maxWt, wtHufDemapX1, n_, lens_);         \
			else if ((maxBits = Huffman_Read_Code_byHuffman(&bitStream, (Uint32)maxWt, wtHufDemapX1, n_, cap_, lens_)) < 0) return 0; \
			else max_ = (Uint32)maxBits; }
		if (!reuse[0]) SEQ_READ_CODE(N_HufLitRun, CapHufLitRunBits, w->litRunHufWt, w->maxLitRunHufWt);
		if (!reuse[1]) SEQ_READ_CODE(w->slotJoint ? N_HufJoint : N_HufJointClassic, CapHufMchLenBits, w->mchLenHufWt, w->maxMchLenHufWt);
		for (t = 2; t < nTab; t++)
			if (!reuse[t]) SEQ_READ_CODE(N_HufMchOff[t - 2], CapHufMchOffBits, w->mchOffHufWt[t - 2], w->maxMchOffHufWt[t - 2]);
#undef SEQ_READ_CODE
	}
	BITStream_Read_FlushEnd(bitStream);
	*bs = bitStream;

	if (!reuse[0]) {
		if (trusted) {
			T->remLitRun = BIT_CONTAINER_BITS - w->maxLitRunHufWt;
			Build_Huffman_DecTableX1(N_HufLitRun, w->maxLitRunHufWt, w->litRunHufWt, T->litRun);
		}
		else T->remLitRun = Huffman_Build_SafeX1(N_HufLitRun, (int)w->maxLitRunHufWt, w->litRunHufWt, T->litRun);
		T->haveLitRun = 1;
	}
	if (!reuse[1]) {
		if (w->slotJoint) {
			if (!trusted) memset(T->joint, 0, 2 * sizeof(Joint_DemapX1));
			Build_Joint_DecTable(w->maxMchLenHufWt, w->mchLenHufWt, T->joint);
			T->remMchLen = BIT_CONTAINER_BITS - (trusted ? w->maxMchLenHufWt : max(1u, w->maxMchLenHufWt));
		}
		else if (trusted) {
			T->remMchLen = BIT_CONTAINER_BITS - w->maxMchLenHufWt;
			Build_Huffman_DecTableX1(N_HufJointClassic, w->maxMchLenHufWt, w->mchLenHufWt, T->mchLen);
		}
		else T->remMchLen = Huffman_Build_SafeX1(N_HufJointClassic, (int)w->maxMchLenHufWt, w->mchLenHufWt, T->mchLen);
		T->haveMchLen = 1;
		T->mchLenSlotJoint = w->slotJoint;
	}
	for (t = 2; t < nTab; t++) {
		if (reuse[t]) continue;
		const Uint32 g = t - 2;
		Huffman_DemapX1* const offTable = T->off + (g << CapHufMchOffBits);
		if (trusted)                                     /* an empty table, never read, is left unbuilt */
			Build_Huffman_DecTableX1(N_HufMchOff[g], w->maxMchOffHufWt[g] ? CapHufMchOffBits : 0, w->mchOffHufWt[g], offTable);
		else {
			if (w->maxMchOffHufWt[g] <= 1) memset(offTable, 0, (1 << CapHufMchOffBits) * sizeof(Huffman_DemapX1));
			if (w->maxMchOffHufWt[g]) Build_Huffman_DecTableX1(N_HufMchOff[g], CapHufMchOffBits, w->mchOffHufWt[g], offTable);
		}
		T->haveOff[g] = 1;
	}
	return 1;
}

static Seq_DecTables* Seq_DecTables_New(void)
{
	Seq_DecTables* const T = (Seq_DecTables*)malloc(sizeof(Seq_DecTables));
	if (T) {
		T->haveLitRun = T->haveMchLen = 0;
		memset(T->haveOff, 0, sizeof(T->haveOff));
	}
	return T;
}

/* a match that starts in the dictionary, `produced` bytes into the output; a corrupt one may run on into the output */
static void Copy_Dict_Match(Uint8* destPtr, const Uint8* dest, const Uint32 produced, const Uint32 offset, const Uint32 len, const Uint8* dictEnd)
{
	const Uint32 inDict = min(len, offset - produced);
	memcpy(destPtr, dictEnd - (offset - produced), inDict);
	for (Uint32 k = inDict; k < len; k++) destPtr[k] = dest[k - inDict];
}

/* Matches that reach beyond the L2 cache stall the decoder on large inputs. A block where they are frequent is decoded
   as a pipeline, as Zstandard's long-offset decoder does: each step decodes one sequence, prefetches its match's
   source, and executes the sequence decoded SEQ_Lookahead steps before. The decoders count the matches at offsets of
   FAR_Offset or more, and pipeline a block when the previous one had at least one per FAR_BytesPerMatch bytes of
   output (16 per KB). On AMD EPYC 9334 this decodes enwik8 49% and enwik9 72% faster at level 11; on inputs whose
   matches stay in cache, where the rule keeps the plain loop, the pipeline would cost up to 8%. */
#define   SEQ_Lookahead        16
/* a sequence block: its tables, the u24 size of stream A, stream A (its sequences 0, 2, 4, ...), stream B (1, 3, 5, ...);
   each stream holds at most 8192 sequences of at most 105 bits, so A at most this */
#define   SEQ_MaxSizeA         ((SEQ_BlockSize / 2) * 14 + 16)
#define   FAR_Offset           (1u << 20)
#define   FAR_BytesPerMatch    64
#ifndef WZL_PIPELINE
#  define WZL_PIPELINE         0                   /* tests: 1 pipelines every block, -1 none; 0 chooses by the rule */
#endif
#define   PIPELINE_NEXT(nFar, bytes)  (WZL_PIPELINE ? WZL_PIPELINE > 0 : (Uint64)(nFar) * FAR_BytesPerMatch >= (Uint64)(bytes))

/* The literal stream precedes the sequence blocks. Up to LIT_EagerMax literals, which stay in the last-level cache, are
   decoded at once before the sequences. More are decoded only as the sequences need them, a block of HUF_BlockSize
   literals at a time, into a buffer that stays in cache, as Zstandard decodes each block's literals: decoding them
   all first would send them through memory and back (enwik9 at level 0, 230 MB of literals: 6% faster). Then the
   sequence decoders count the literals decoded and not yet claimed by a sequence (litLeft) and call Lit_Refill when
   a literal run needs more; the checked decoder counts them in either case, to reject runs past the literals. */
#ifndef LIT_EagerMax                               /* tests: 0 decodes every stream's literals as needed */
#  define LIT_EagerMax         (1u << 25)
#endif
#define   LIT_BufSize          (4 * HUF_BlockSize)
#define   LIT_Slack            256                 /* readable bytes past the buffer: wild copies, the literal decoders */
typedef struct {
	Uint8* buf;                                    /* the decoded literals: buf[0, cap), and LIT_Slack bytes more */
	size_t cap;
	const Uint8* src;                              /* the next block of the literal stream */
	const Uint8* srcEnd;                           /* the input's end (checked mode) */
	Uint32 left;                                   /* literals of the stream not yet decoded */
	int trusted;
	int lazy;                                      /* more than LIT_EagerMax literals: decoded as needed */
	Huffman_DecState* hst;                         /* the code of the stream's last block with code lengths */
	Uint8* exec;                                   /* between sequence blocks: the next literal to output */
	Uint8* avail;                                  /* the end of the decoded literals */
} Lit_Reader;

typedef struct { Uint8* exec; size_t left; } Lit_Refilled;

static int Lit_Reader_Init(Lit_Reader* const r, const Uint8* src, const Uint8* srcEnd, const Uint32 nLits, const int trusted)
{
	r->lazy = nLits > LIT_EagerMax;
	r->cap = r->lazy ? LIT_BufSize : nLits;
	r->buf = (Uint8*)malloc(r->cap + LIT_Slack);
	r->hst = (Huffman_DecState*)malloc(sizeof(Huffman_DecState));
	r->src = src; r->srcEnd = srcEnd; r->left = nLits; r->trusted = trusted;
	r->exec = r->avail = r->buf;
	if (NULL == r->buf || NULL == r->hst) return 0;
	r->hst->valid = 0; r->hst->table = 0;
	while (!r->lazy && r->left) {                    /* all at once */
		const Uint32 n = r->left < HUF_BlockSize ? r->left : HUF_BlockSize;
		const int used = trusted ? (int)Huffman_Decompress_Next_Trusted(r->src, r->avail, n, N_HufLits, r->hst)
		                         : Huffman_Decompress_Next(r->src, r->srcEnd, r->avail, n, N_HufLits, r->hst);
		if (used < 0) return 0;
		r->src += used;
		r->left -= n;
		r->avail += n;
	}
	return 1;
}

/* Makes need literals more available after the left ones a sequence decoder has not claimed yet, exec being the next
   one to output: moves the unexecuted ones to the buffer's start (a larger buffer for a long run) and decodes blocks
   after them. Returns the new exec and the unclaimed literals, or exec NULL if the stream has too few literals or a
   block is corrupt (or memory runs out). */
static Lit_Refilled Lit_Refill(Lit_Reader* const r, Uint8* const exec, const size_t left, const Uint32 need)
{
	Lit_Refilled f = { NULL, 0 };
	const size_t keep = (size_t)(r->avail - exec), claimed = keep - left;
	if ((size_t)need - left > r->left) return f;                    /* more literals than the stream holds */
	/* the room the decoded blocks need: at most a block more than claimed + need */
	const size_t room = claimed + need + (r->left < HUF_BlockSize ? r->left : HUF_BlockSize);
	if (room > r->cap) {
		size_t cap = 2 * r->cap;
		if (cap < room) cap = room;
		Uint8* const b = (Uint8*)malloc(cap + LIT_Slack);
		if (NULL == b) return f;
		memcpy(b, exec, keep);
		free(r->buf);
		r->buf = b; r->cap = cap;
	}
	else memmove(r->buf, exec, keep);
	Uint8* const claim = r->buf + claimed;
	Uint8* avail = r->buf + keep;
	while ((size_t)(avail - claim) < need) {
		const Uint32 n = r->left < HUF_BlockSize ? r->left : HUF_BlockSize;
		const int used = r->trusted ? (int)Huffman_Decompress_Next_Trusted(r->src, avail, n, N_HufLits, r->hst)
		                            : Huffman_Decompress_Next(r->src, r->srcEnd, avail, n, N_HufLits, r->hst);
		if (used < 0) return f;
		r->src += used;
		r->left -= n;
		avail += n;
	}
	r->avail = avail;
	f.exec = r->buf;
	f.left = (size_t)(avail - claim);
	return f;
}

/* claims a sequence's litRun literals, decoding more if needed; leaves through onFail if there are no more */
#define LIT_CLAIM(onFail)  {                                                                                       \
		if (unlikely(litRun > litLeft)) {                                                                          \
			const Lit_Refilled f_ = Lit_Refill(lits, lzLitBufPtr, litLeft, litRun);                                \
			if (NULL == f_.exec) onFail;                                                                           \
			lzLitBufPtr = f_.exec; litLeft = f_.left;                                                              \
		}                                                                                                          \
		litLeft -= litRun; }

/* Executes a sequence the decoder has checked: litRun literals, then matchLen bytes from matchOffset back (a run:
   offset 1), into *destRef, from the literals at *litRef; advances both */
ForceInlineTemplate void Execute_Sequence(Uint8** destRef, Uint8** litRef, const Uint32 litRun, const Uint32 matchLen,
	const Uint32 matchOffset, Uint8* const dest, Uint8* const destEnd, const Uint8* const dictEnd, const int dictSize)
{
	static const unsigned inc4table[8] = { 0, 0, 0,  1,  0,  4, 4, 4 };     /* 4 % matchOffset */
	static const unsigned inc8table[8] = { 0, 0, 0,  2,  0,  3, 2, 1 };     /* 8 % matchOffset */
	Uint8* destPtr = *destRef;
	Uint8* const lit = *litRef;
	Uint8* destPtrEnd = destPtr + litRun;
	*litRef = lit + litRun;
	*destRef = destPtrEnd + matchLen;
	if (unlikely(matchLen + 16 > (Uint32)(destEnd - destPtrEnd))) {        /* near the end: wild copies would write past it */
		memcpy(destPtr, lit, litRun);
		if (dictSize && (Uint32)(destPtrEnd - dest) < matchOffset)
			Copy_Dict_Match(destPtrEnd, dest, (Uint32)(destPtrEnd - dest), matchOffset, matchLen, dictEnd);
		else
			for (Uint8* q = destPtrEnd; q < destPtrEnd + matchLen; q++) *q = *(q - matchOffset);
		return;
	}
	MemWildCopy(destPtr, lit, destPtrEnd);
	destPtr = destPtrEnd;
	destPtrEnd += matchLen;
	if (dictSize && (Uint32)(destPtr - dest) < matchOffset) {
		/* the match starts in the dictionary, which the compressor never lets it run past: copy exactly, as the
		   dictionary may end at the end of its buffer */
		Copy_Dict_Match(destPtr, dest, (Uint32)(destPtr - dest), matchOffset, matchLen, dictEnd);
	}
	else if (likely(matchOffset >= 16))
		MemWildCopy(destPtr, destPtr - matchOffset, destPtrEnd);
	else if (matchOffset == 1) {                                             /* a run of the preceding byte */
		if (matchLen <= 16) {                                                /* short: two wild 8-byte stores */
			const Uint64 fill = destPtr[-1] * 0x0101010101010101ull;
			memcpy(destPtr, &fill, 8);
			memcpy(destPtr + 8, &fill, 8);
		}
		else memset(destPtr, destPtr[-1], matchLen);
	}
	else {
		const Uint8* matchPtr = destPtr - matchOffset;
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
}

/* Hands a checked sequence (litRun, matchLen, matchOffset) to execution, through the ring when pipelined; decEnd is
   the output's end once every sequence decoded so far is executed */
#define SEQ_DISPATCH()  {                                                                                          \
		if (pipelined) {                                                                                           \
			if (!dictSep || (Uint32)(decEnd - (Uint8*)dest) >= matchOffset) PREFETCH_L1(decEnd - matchOffset);     \
			const Uint32 t_ = (rHead + rCount) & (SEQ_Lookahead - 1);                                              \
			rLit[t_] = litRun; rLen[t_] = matchLen; rOff[t_] = matchOffset;                                        \
			decEnd += matchLen;                                                                                    \
			if (++rCount < SEQ_Lookahead) continue;                                                                \
			litRun = rLit[rHead]; matchLen = rLen[rHead]; matchOffset = rOff[rHead];                               \
			rHead = (rHead + 1) & (SEQ_Lookahead - 1); rCount--;                                                   \
		}                                                                                                          \
		else decEnd += matchLen;                                                                                   \
		Execute_Sequence(&destPtr, &lzLitBufPtr, litRun, matchLen, matchOffset, (Uint8*)dest, destEnd, dictEnd, dictSep); }

/* executes the sequences left in the ring, then the block's closing literal run, if it ends the output */
#define SEQ_DRAIN()  {                                                                                             \
		for (; rCount; rCount--, rHead = (rHead + 1) & (SEQ_Lookahead - 1))                                        \
			Execute_Sequence(&destPtr, &lzLitBufPtr, rLit[rHead], rLen[rHead], rOff[rHead], (Uint8*)dest, destEnd, dictEnd, dictSep); \
		if (ended) {                                                                                               \
			memcpy(destPtr, lzLitBufPtr, lastLit);             /* exact: the output buffer may end right here */   \
			destPtr += lastLit;                                                                                    \
			lzLitBufPtr += lastLit;                                                                                \
		}                                                                                                          \
		*farCount += nFar; }

/* Decodes one block of sequences; returns the decoded size so far, or -1 if the block is corrupt. The literal run and
   match length are checked against the output, the literal run against the literals (LIT_CLAIM), and the offset
   against the bytes decoded and the dictionary, for every sequence. The input needs no check here, as the caller
   guarantees that the block's longest possible read stays inside it. */
ForceInlineTemplate int Decompress_WLZ_Sequence_Body(WZL_Sched* const S_, Uint8** wzipSeqStart, Lit_Reader* const lits, Uint8* dest, int decPos, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize, WLZ_HufWt_Set* hufWtSet, Uint32 offsetLast[OffCasheSize],
	const Seq_DecTables* const T, Uint32* const farCount, const int pipelined,
	const int fineGroups, const int slotJoint)    /* compile-time constants in each instance: offset grouping, joint symbol */
{
	register Uint32 i, n, lsValue, mchLenHufIdx;
	register Uint32 litRun, matchLen, matchOffset;
	Uint8* destPtr = (Uint8*)dest+decPos;
	Uint8* decEnd = destPtr;                         /* the output's end once the sequences decoded so far are executed */
	Uint32 rLit[SEQ_Lookahead], rLen[SEQ_Lookahead], rOff[SEQ_Lookahead], rHead = 0, rCount = 0;   /* the pipeline's ring */
	Uint32 nFar = 0, lastLit = 0;
	int ended = 0;
	/* a dictionary in a buffer of its own is copied from through Copy_Dict_Match; one just before the output (a prefix
	   of the same buffer) is copied from as earlier output */
	const int dictSep = dictSize && dictEnd != (Uint8*)dest ? dictSize : 0;

	static const ExtHuffman_Lit extHuf[] = {
	{16, 1},  {17, 1}, {18, 1}, {19, 1},    {20, 1}, {21, 1}, {22, 1}, {23, 1},     {24, 1}, {25, 1}, {26, 1}, {27, 1},      {28, 1}, {29, 1}, {30, 1}, {31, 1},      /* 32-47:  32 - 63 */
	{8, 3},  {9, 3},  {10, 3},  {11, 3},    {12, 3},  {13, 3}, {14, 3}, {15, 3},             /* 48-55:  64 - 127 */
	{4, 5},  {5, 5}, {6, 5}, {7, 5},                                                         /* 56-59:  128 - 255 */
	{4, 6},  {5, 6}, {6, 6}, {7, 6},                                                         /* 60-63:  256 - 511 */
	{2, 8},  {3, 8},                                                                         /* 64-65:  512 - 1023 */
	{2, 9},  {3, 9}, 																		 /* 66-67:  1024 - 2047 */
	{2, 10}, {3, 10},																		 /* 68-69:  2048 - 4095 */
	{0, 24},                                                                                 /* 70:     2048 - 16M  */
	};

	register ExtHuffman_Lit extHufRes;
	/* the sequences alternate between streams A and B, so that two decode at once: bitStream reads the next
	   sequence's stream, otherStream the other, and the two swap after each sequence (register renaming, not copies) */
	Uint8* const seqA = *wzipSeqStart + 3;
	const Uint32 sizeA = MemReadLE2(*wzipSeqStart) | (Uint32)(*wzipSeqStart)[2] << 16;
	if (sizeA > SEQ_MaxSizeA) return -1;              /* corrupt: the caller made only so much readable */
	register Bit_Stream bitStream, otherStream;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(seqA);
	bitStream.streamPtr = seqA;
	otherStream.nUsedBits = 0;
	otherStream.container = MemReadBE8(seqA + sizeA);
	otherStream.streamPtr = seqA + sizeA;
	Uint8* lzLitBufPtr = lits->exec;                 /* the next literal to output */
	size_t litLeft = (size_t)(lits->avail - lzLitBufPtr);   /* decoded literals no sequence has claimed yet */
#ifdef WZIP_DEBUG
	FILE* fptr;
	char filename[100];
	snprintf(filename, sizeof(filename),
		"WZIP_Seq_Decompress_%d.txt", decPos);

	fptr = fopen(filename, "w");

#endif

	/* the tables, built by Seq_Read_Tables from validated codes */
	const Huffman_DemapX1* const litRunHufDemapX1 = T->litRun;
	const Uint32 remMaxLitRunHufWt = T->remLitRun;
	const Joint_DemapX1* const jointDemapX1 = T->joint;           /* slot-joint blocks */
	const Huffman_DemapX1* const mchLenHufDemapX1 = T->mchLen;    /* classic blocks */
	const Uint32 remMaxMchLenHufWt = T->remMchLen;
	const Huffman_DemapX1* const mchOff_HufDemapX1 = T->off;
	const Uint32 lastOffGroup = MchOffGroup - 1;
	(void)hufWtSet;

	int seqNo;
	for(seqNo=0; seqNo<SEQ_BlockSize; seqNo++) {

		Uint32 litClass, slotSel;
		if (slotJoint) {                                 /* joint symbol: cache slot (or new), literal-run class, length */
			const Joint_DemapX1 js = jointDemapX1[(Uint32)((bitStream.container << bitStream.nUsedBits) >> remMaxMchLenHufWt)];
			bitStream.nUsedBits += js.nbits;
			mchLenHufIdx = js.mlSym;
			litClass = js.litClass;
			slotSel = js.slotSel;
		}
		else {                                           /* joint symbol: literal-run class and length */
			BITStream_Read_ExtHufX1(bitStream, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdx);
			litClass = mchLenHufIdx >= 2 * N_HufMchLen ? 2 : mchLenHufIdx >= N_HufMchLen;
			mchLenHufIdx -= litClass * N_HufMchLen;
			slotSel = OffCasheSize;
		}
		{   /* the literal-run symbol is looked up for every sequence but consumed for class 2 only: no branch */
			const Huffman_DemapX1 lr = litRunHufDemapX1[(Uint32)((bitStream.container << bitStream.nUsedBits) >> remMaxLitRunHufWt)];
			const Uint32 isRun = litClass >> 1;
			bitStream.nUsedBits += lr.nbits & (0u - isRun);
			litRun = isRun ? lr.lit : litClass;
		}
		if (unlikely(litRun >= LitRunDirect)) { /* Read extra number of bytes */
			extHufRes = extHuf[litRun - LitRunDirect];
			BITStream_Read(bitStream, extHufRes.lsBits, lsValue);
			litRun = extHufRes.msValue << extHufRes.lsBits ^ lsValue;
			BITStream_Read_Flush(bitStream);
		}
		LIT_CLAIM(return -1);

#ifdef WZIP_DEBUG
		fprintf(fptr, "DecPos=%d, litRun=%d,  ", (Uint32)(destPtr - dest), litRun);
#endif

		if ( unlikely(litRun >= (Uint32)(destEnd - decEnd)) ) {   /* note the ending is checked right after literal run */
			if (litRun > (Uint32)(destEnd - decEnd)) return -1;
			lastLit = litRun;
			ended = 1;
			break;
		}
		decEnd += litRun;

		if (!slotJoint || slotSel == OffCasheSize) {     /* the offset symbol: a new offset, a run count, or (classic) a cache slot */
			const Uint32 offGroup = fineGroups ? min(5u, mchLenHufIdx) + (mchLenHufIdx >= 7) + (mchLenHufIdx >= 13) : min(lastOffGroup, mchLenHufIdx);
			const Huffman_DemapX1* const offTable = mchOff_HufDemapX1 + (offGroup << CapHufMchOffBits);
			BITStream_Read_ExtHufX1(bitStream, (sizeof(bitStream.container) * 8 - CapHufMchOffBits), offTable, n);
			if (slotJoint) n = n < OffCasheSize ? OffCasheSize : n;    /* only new offsets here: corrupt input harmless */
		}
		else n = slotSel;
		if (likely(n >= OffCasheSize)) {
			i = (n >> 1) - 1;
			//lsBits = ExtHufMchOff[n].lsBits;
			BITStream_Read(bitStream, i, lsValue);
			//matchOffset = (ExtHufMchOff[n].msValue << lsBits ^ lsValue) - (OffCasheSize - 1);
			matchOffset = ((2 ^ (n & 1)) << i ^ lsValue) - (OffCasheSize - 1);
			if (unlikely(mchLenHufIdx == RunSym)) {         /* a run: its count came in the offset field */
				matchLen = matchOffset;
				matchOffset = 1;
				BITStream_Read_Flush(bitStream);
				goto _check;
			}
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

		matchLen = mchLenHufIdx + MinMatchLen;
		if (unlikely(matchLen >= LitRunDirect)) {
			extHufRes = extHuf[matchLen - LitRunDirect];
			BITStream_Read_Flush(bitStream);
			BITStream_Read(bitStream, extHufRes.lsBits, lsValue);
			matchLen = extHufRes.msValue << extHufRes.lsBits ^ lsValue;
		}
		BITStream_Read_Flush(bitStream);

	_check:
		if (unlikely(matchOffset - 1 >= (Uint32)(decEnd - dest) + (Uint32)dictSize)) return -1;   /* corrupt: before the history */
		if (unlikely(matchLen > (Uint32)(destEnd - decEnd))) return -1;                          /* corrupt: past the output */
		nFar += matchOffset >= FAR_Offset;
#ifdef WZIP_DEBUG
		fprintf(fptr, "matchLen=%d,  matchOff=%d\n", matchLen, matchOffset);
		fflush(fptr);
#endif
		{ const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* the next sequence: the other stream */
		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Execute LZ Sequence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		SEQ_DISPATCH();
	}
	SEQ_DRAIN();
#ifdef WZIP_DEBUG
	fclose(fptr);
#endif

	lits->exec = lzLitBufPtr;
	if (seqNo & 1) { const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* bitStream: A */
	BITStream_Read_FlushEnd(bitStream);
	BITStream_Read_FlushEnd(otherStream);
	if (bitStream.streamPtr != seqA + sizeA) return -1;   /* corrupt: stream A is not of its stated size */
	*wzipSeqStart = otherStream.streamPtr;
	return (int)(destPtr - dest);
}

#define SEQ_BODY_CALL(body, fine)   (hufWtSet->slotJoint                                                                                                    \
    ? body(S_, wzipSeqStart, lits, dest, decPos, destEnd, dictEnd, dictSize, hufWtSet, offsetLast, T, farCount, pipelined, fine, 1)             \
    : body(S_, wzipSeqStart, lits, dest, decPos, destEnd, dictEnd, dictSize, hufWtSet, offsetLast, T, farCount, pipelined, fine, 0))

#define SEQ_DEC_PARAMS  WZL_Sched* const S_, Uint8** wzipSeqStart, Lit_Reader* const lits, Uint8* dest, int decPos, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize,  \
    WLZ_HufWt_Set* hufWtSet, Uint32 offsetLast[OffCasheSize], const Seq_DecTables* const T, Uint32* const farCount, const int pipelined

#define DECOMPRESS_SEQUENCE_GEN(fun)                                                                                                                  \
    static int fun(SEQ_DEC_PARAMS)                                                                                                                  \
    {                                                                                                                                               \
        return OffGroupsFine ? SEQ_BODY_CALL(fun##_Body, 1)                                               : SEQ_BODY_CALL(fun##_Body, 0);                 \
    }

DECOMPRESS_SEQUENCE_GEN(Decompress_WLZ_Sequence)

/* The same decoder compiled for BMI2, whose shifts by a register count take one micro-op and leave the flags alone;
   the bit reader shifts by a variable count several times per sequence. Chosen at run time, as zstd does. */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#  define WZIP_DYNAMIC_BMI2 1
static __attribute__((target("bmi,bmi2,lzcnt")))
int Decompress_WLZ_Sequence_Bmi2(SEQ_DEC_PARAMS)
{
	return OffGroupsFine ? SEQ_BODY_CALL(Decompress_WLZ_Sequence_Body, 1)
	                     : SEQ_BODY_CALL(Decompress_WLZ_Sequence_Body, 0);
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

/* a block of sequences reads at most this from its start: its code tables, the size of stream A, then streams A and B
   (A at most SEQ_MaxSizeA bytes, B at most 105 bits a sequence) */
#define   SEQ_ReadSpan         ((size_t)SEQ_BlockSize * 14 + 8192)

/* Returns destSize, or 0 if the stream is corrupt: every size, offset and code table is checked, and no read or write
   leaves source[0, srcSize), dest[0, destSize) or the dictionary. */
int WZIP_Decompress_L(
	const void* const source, int const srcSize,
	void* const dest, int const destSize,
	void* const dict, int const dictSize)
{
	WZL_Sched sched_;
	WZL_Sched* const S_ = &sched_;
	int i;
	Uint32 nLzLits;
	Uint8* srcPtr = (Uint8*)source;
	const Uint8* srcEnd = (const Uint8*)source + (srcSize > 0 ? srcSize : 0);
	const int histSize = dict && dictSize > 0 ? dictSize : 0;
	Uint8* const dictEnd = histSize ? (Uint8*)dict + dictSize : NULL;

	if (destSize <= 0) return 0;
	WZIP_Set_OffWidth(WZL_History(destSize, histSize), OffWidth);
	{   /* the windows of lengths 3-7, stored below the widest one; they must not narrow with length */
		if (srcSize < WIN_HeaderSize + 4 || (srcPtr[2] >> 4) > 1) return 0;
		OffGroupsFine = srcPtr[2] >> 4;
		const int gap[5] = { srcPtr[0] & 15, srcPtr[0] >> 4, srcPtr[1] & 15, srcPtr[1] >> 4, srcPtr[2] & 15 };
		for (i = 3; i <= 7; i++) OffWidth[i] = OffWidth[8] - gap[i - 3];
		if (OffWidth[3] < 4) return 0;
		for (i = 4; i <= 8; i++)
			if (OffWidth[i] < OffWidth[i - 1]) return 0;
		srcPtr += WIN_HeaderSize;
	}
	i = 8;
	while (i > 0 && OffWidth[i] == OffWidth[i - 1])
		i--;

	Set_Offset_Groups(S_, i - MinMatchLen + 1, OffGroupsFine);

	if ( destSize >> 16 ) {                            /* Read the length of LZ literal sequence */
		nLzLits = MemReadLE4(srcPtr);
		srcPtr += 4;
	}
	else {
		nLzLits = MemReadLE2(srcPtr);
		srcPtr += 2;
	}
	if (nLzLits > (Uint32)destSize) return 0;                                   /* corrupt */
	/* the literal stream is checked whole here and decoded as the sequences need it (Lit_Reader) */
	const int zipLitSize = Huffman_Skip(srcPtr, (int)(srcEnd - srcPtr), nLzLits, N_HufLits);
	if (zipLitSize < 0) return 0;
	Lit_Reader lits;
	const int litsOk = Lit_Reader_Init(&lits, srcPtr, srcEnd, nLzLits, 0);
	Seq_DecTables* const T = Seq_DecTables_New();
	if (!litsOk || NULL == T) { free(lits.buf); free(lits.hst); free(T); return 0; }
	srcPtr += zipLitSize;

	Bit_Stream bitStream;
	bitStream.streamPtr = srcPtr;

	WLZ_HufWt_Set hufWtSet;                          /* the lengths of the codes; their tables in T */

	Uint32 offsetLast[OffCasheSize];
	memset(offsetLast, 0x7F, OffCasheSize * sizeof(int));
	int decSize = 0, pipelined = WZL_PIPELINE > 0;
	Uint8* tail = NULL;
	Uint8* const destEnd = (Uint8*)dest + destSize;
	while (decSize < destSize) {
		/* a block reads at most SEQ_ReadSpan bytes: nearer the end of the input, continue from a zero-padded copy of
		   the rest, so the decoding loop needs no input check */
		if (NULL == tail && (size_t)(srcEnd - bitStream.streamPtr) < SEQ_ReadSpan) {
			const size_t rest = (size_t)(srcEnd - bitStream.streamPtr);
			if (NULL == (tail = (Uint8*)malloc(rest + SEQ_ReadSpan))) { decSize = -1; break; }
			memcpy(tail, bitStream.streamPtr, rest);
			memset(tail + rest, 0, SEQ_ReadSpan);
			bitStream.streamPtr = tail;
			srcEnd = tail + rest;
		}
		bitStream.nUsedBits = 0;
		bitStream.container = MemReadBE8(bitStream.streamPtr);
		BITStream_Read(bitStream, 1, hufWtSet.slotJoint);           /* the block's coding: slot-joint or classic */
		if (!Seq_Read_Tables(S_, &bitStream, &hufWtSet, T, 0)) { decSize = -1; break; }

		const int blockStart = decSize;
		Uint32 nFar = 0;
#if WZIP_DYNAMIC_BMI2
		if (CPU_Has_Bmi2())
			decSize = Decompress_WLZ_Sequence_Bmi2(S_, &(bitStream.streamPtr), &lits, (Uint8*)dest, decSize, destEnd, dictEnd, histSize, &hufWtSet, offsetLast, T, &nFar, pipelined);
		else
#endif
		decSize = Decompress_WLZ_Sequence(S_, &(bitStream.streamPtr), &lits, (Uint8*)dest, decSize, destEnd, dictEnd, histSize, &hufWtSet, offsetLast, T, &nFar, pipelined);
		if (decSize < 0 || bitStream.streamPtr > srcEnd) { decSize = -1; break; }    /* corrupt: read past the input */
		pipelined = PIPELINE_NEXT(nFar, decSize - blockStart);   /* for the next block */
	}

	free(tail);
	free(T);
	free(lits.buf);
	free(lits.hst);
	if (destSize == decSize) return decSize;
	else return 0;
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trusted mode (opt-in) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* The same decoder without checks, for input known to come unmodified from WZIP's encoder; the input must stay
   readable WZIP_TRUSTED_SRC_PAD bytes past its end. A damaged stream can make it read or write out of bounds. */

ForceInlineTemplate int Decompress_WLZ_Sequence_Trusted_Body(WZL_Sched* const S_, Uint8** wzipSeqStart, Lit_Reader* const lits, Uint8* dest, int decPos, Uint8* const destEnd, Uint8* const dictEnd, const int dictSize, WLZ_HufWt_Set* hufWtSet, Uint32 offsetLast[OffCasheSize],
	const Seq_DecTables* const T, Uint32* const farCount, const int pipelined,
	const int fineGroups, const int slotJoint, const int lazy)    /* compile-time constants in each instance: offset grouping, joint symbol, literals decoded as needed */
{
	register Uint32 i, n, lsValue, mchLenHufIdx;
	register Uint32 litRun, matchLen, matchOffset;
	Uint8* destPtr = (Uint8*)dest+decPos;
	Uint8* decEnd = destPtr;                         /* the output's end once the sequences decoded so far are executed */
	Uint32 rLit[SEQ_Lookahead], rLen[SEQ_Lookahead], rOff[SEQ_Lookahead], rHead = 0, rCount = 0;   /* the pipeline's ring */
	Uint32 nFar = 0, lastLit = 0;
	int ended = 0;
	/* a dictionary in a buffer of its own is copied from through Copy_Dict_Match; one just before the output (a prefix
	   of the same buffer) is copied from as earlier output */
	const int dictSep = dictSize && dictEnd != (Uint8*)dest ? dictSize : 0;

	static const ExtHuffman_Lit extHuf[] = {
	{16, 1},  {17, 1}, {18, 1}, {19, 1},    {20, 1}, {21, 1}, {22, 1}, {23, 1},     {24, 1}, {25, 1}, {26, 1}, {27, 1},      {28, 1}, {29, 1}, {30, 1}, {31, 1},      /* 32-47:  32 - 63 */
	{8, 3},  {9, 3},  {10, 3},  {11, 3},    {12, 3},  {13, 3}, {14, 3}, {15, 3},             /* 48-55:  64 - 127 */
	{4, 5},  {5, 5}, {6, 5}, {7, 5},                                                         /* 56-59:  128 - 255 */
	{4, 6},  {5, 6}, {6, 6}, {7, 6},                                                         /* 60-63:  256 - 511 */
	{2, 8},  {3, 8},                                                                         /* 64-65:  512 - 1023 */
	{2, 9},  {3, 9}, 																		 /* 66-67:  1024 - 2047 */
	{2, 10}, {3, 10},																		 /* 68-69:  2048 - 4095 */
	{0, 24},                                                                                 /* 70:     2048 - 16M  */
	};

	register ExtHuffman_Lit extHufRes;
	/* the sequences alternate between streams A and B, so that two decode at once: bitStream reads the next
	   sequence's stream, otherStream the other, and the two swap after each sequence (register renaming, not copies) */
	Uint8* const seqA = *wzipSeqStart + 3;
	const Uint32 sizeA = MemReadLE2(*wzipSeqStart) | (Uint32)(*wzipSeqStart)[2] << 16;
	register Bit_Stream bitStream, otherStream;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(seqA);
	bitStream.streamPtr = seqA;
	otherStream.nUsedBits = 0;
	otherStream.container = MemReadBE8(seqA + sizeA);
	otherStream.streamPtr = seqA + sizeA;
	Uint8* lzLitBufPtr = lits->exec;                 /* the next literal to output */
	size_t litLeft = (size_t)(lits->avail - lzLitBufPtr);   /* decoded literals no sequence has claimed yet */
#ifdef WZIP_DEBUG
	FILE* fptr;
	char filename[100];
	snprintf(filename, sizeof(filename),
		"WZIP_Seq_Decompress_%d.txt", decPos);

	fptr = fopen(filename, "w");

#endif

	const Huffman_DemapX1* const litRunHufDemapX1 = T->litRun;    /* the tables, built by Seq_Read_Tables */
	const Uint32 remMaxLitRunHufWt = T->remLitRun;
	const Joint_DemapX1* const jointDemapX1 = T->joint;           /* slot-joint blocks */
	const Huffman_DemapX1* const mchLenHufDemapX1 = T->mchLen;    /* classic blocks */
	const Uint32 remMaxMchLenHufWt = T->remMchLen;
	const Huffman_DemapX1* const mchOff_HufDemapX1 = T->off;
	(void)hufWtSet;
	const Uint32 lastOffGroup = MchOffGroup - 1;

	int seqNo;
	for(seqNo=0; seqNo<SEQ_BlockSize; seqNo++) {

		Uint32 litClass, slotSel;
		if (slotJoint) {                                 /* joint symbol: cache slot (or new), literal-run class, length */
			const Joint_DemapX1 js = jointDemapX1[(Uint32)((bitStream.container << bitStream.nUsedBits) >> remMaxMchLenHufWt)];
			bitStream.nUsedBits += js.nbits;
			mchLenHufIdx = js.mlSym;
			litClass = js.litClass;
			slotSel = js.slotSel;
		}
		else {                                           /* joint symbol: literal-run class and length */
			BITStream_Read_ExtHufX1(bitStream, remMaxMchLenHufWt, mchLenHufDemapX1, mchLenHufIdx);
			litClass = mchLenHufIdx >= 2 * N_HufMchLen ? 2 : mchLenHufIdx >= N_HufMchLen;
			mchLenHufIdx -= litClass * N_HufMchLen;
			slotSel = OffCasheSize;
		}
		{   /* the literal-run symbol is looked up for every sequence but consumed for class 2 only: no branch */
			const Huffman_DemapX1 lr = litRunHufDemapX1[(Uint32)((bitStream.container << bitStream.nUsedBits) >> remMaxLitRunHufWt)];
			const Uint32 isRun = litClass >> 1;
			bitStream.nUsedBits += lr.nbits & (0u - isRun);
			litRun = isRun ? lr.lit : litClass;
		}
		if (unlikely(litRun >= LitRunDirect)) { /* Read extra number of bytes */
			extHufRes = extHuf[litRun - LitRunDirect];
			BITStream_Read(bitStream, extHufRes.lsBits, lsValue);
			litRun = extHufRes.msValue << extHufRes.lsBits ^ lsValue;
			BITStream_Read_Flush(bitStream);
		}
		if (lazy) LIT_CLAIM(return -1);

#ifdef WZIP_DEBUG
		fprintf(fptr, "DecPos=%d, litRun=%d,  ", (Uint32)(destPtr - dest), litRun);
#endif

		if ( unlikely(litRun >= (Uint32)(destEnd - decEnd)) ) {   /* note the ending is checked right after literal run */
			lastLit = litRun;
			ended = 1;
			break;
		}
		decEnd += litRun;

		if (!slotJoint || slotSel == OffCasheSize) {     /* the offset symbol: a new offset, a run count, or (classic) a cache slot */
			const Uint32 offGroup = fineGroups ? min(5u, mchLenHufIdx) + (mchLenHufIdx >= 7) + (mchLenHufIdx >= 13) : min(lastOffGroup, mchLenHufIdx);
			const Huffman_DemapX1* const offTable = mchOff_HufDemapX1 + (offGroup << CapHufMchOffBits);
			BITStream_Read_ExtHufX1(bitStream, (sizeof(bitStream.container) * 8 - CapHufMchOffBits), offTable, n);
			if (slotJoint) n = n < OffCasheSize ? OffCasheSize : n;    /* only new offsets here: corrupt input harmless */
		}
		else n = slotSel;
		if (likely(n >= OffCasheSize)) {
			i = (n >> 1) - 1;
			//lsBits = ExtHufMchOff[n].lsBits;
			BITStream_Read(bitStream, i, lsValue);
			//matchOffset = (ExtHufMchOff[n].msValue << lsBits ^ lsValue) - (OffCasheSize - 1);
			matchOffset = ((2 ^ (n & 1)) << i ^ lsValue) - (OffCasheSize - 1);
			if (unlikely(mchLenHufIdx == RunSym)) {         /* a run: its count came in the offset field */
				matchLen = matchOffset;
				matchOffset = 1;
				BITStream_Read_Flush(bitStream);
				if (matchLen > (Uint32)(destEnd - decEnd)) break;         /* corrupt: the size check fails */
				goto _execute;
			}
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

		matchLen = mchLenHufIdx + MinMatchLen;
		if (unlikely(matchLen >= LitRunDirect)) {
			extHufRes = extHuf[matchLen - LitRunDirect];
			BITStream_Read_Flush(bitStream);
			BITStream_Read(bitStream, extHufRes.lsBits, lsValue);
			matchLen = extHufRes.msValue << extHufRes.lsBits ^ lsValue;
		}
		BITStream_Read_Flush(bitStream);

	_execute:
		nFar += matchOffset >= FAR_Offset;
#ifdef WZIP_DEBUG
		fprintf(fptr, "matchLen=%d,  matchOff=%d\n", matchLen, matchOffset);
		fflush(fptr);
#endif
		{ const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* the next sequence: the other stream */
		/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Execute LZ Sequence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		SEQ_DISPATCH();
	}
	SEQ_DRAIN();
#ifdef WZIP_DEBUG
	fclose(fptr);
#endif

	lits->exec = lzLitBufPtr;
	if (seqNo & 1) { const Bit_Stream t_ = bitStream; bitStream = otherStream; otherStream = t_; }   /* bitStream: A */
	BITStream_Read_FlushEnd(otherStream);
	*wzipSeqStart = otherStream.streamPtr;
	return (Uint32)(destPtr - dest);
}


#define SEQ_TRUSTED_CALL(body, fine, lazy)   (hufWtSet->slotJoint                                                                                      \
    ? body(S_, wzipSeqStart, lits, dest, decPos, destEnd, dictEnd, dictSize, hufWtSet, offsetLast, T, farCount, pipelined, fine, 1, lazy)          \
    : body(S_, wzipSeqStart, lits, dest, decPos, destEnd, dictEnd, dictSize, hufWtSet, offsetLast, T, farCount, pipelined, fine, 0, lazy))
#define SEQ_TRUSTED_CALLS(body)  (lits->lazy                                                                                                         \
    ? (OffGroupsFine ? SEQ_TRUSTED_CALL(body, 1, 1) : SEQ_TRUSTED_CALL(body, 0, 1))                                                              \
    : (OffGroupsFine ? SEQ_TRUSTED_CALL(body, 1, 0) : SEQ_TRUSTED_CALL(body, 0, 0)))
#define SEQ_TRUSTED_PARAMS  WZL_Sched* const S_, Uint8** wzipSeqStart, Lit_Reader* const lits, Uint8* dest, int decPos, Uint8* const destEnd, Uint8* const dictEnd, \
    const int dictSize, WLZ_HufWt_Set* hufWtSet, Uint32 offsetLast[OffCasheSize], const Seq_DecTables* const T, Uint32* const farCount, const int pipelined

static int Decompress_WLZ_Sequence_Trusted(SEQ_TRUSTED_PARAMS)
{
	return SEQ_TRUSTED_CALLS(Decompress_WLZ_Sequence_Trusted_Body);
}
#if WZIP_DYNAMIC_BMI2
static __attribute__((target("bmi,bmi2,lzcnt")))
int Decompress_WLZ_Sequence_Trusted_Bmi2(SEQ_TRUSTED_PARAMS)
{
	return SEQ_TRUSTED_CALLS(Decompress_WLZ_Sequence_Trusted_Body);
}
#endif

int WZIP_Decompress_L_Trusted(
	const void* const source, int const srcSize,
	void* const dest, int const destSize,
	void* const dict, int const dictSize)
{
	WZL_Sched sched_;
	WZL_Sched* const S_ = &sched_;
	int i;
	Uint32 nLzLits, zipLitSize;
	Uint8* srcPtr = (Uint8*)source;
	Uint8* const dictEnd = dict ? (Uint8*)dict + dictSize : NULL;

	WZIP_Set_OffWidth(WZL_History(destSize, dict ? dictSize : 0), OffWidth);
	{   /* the windows of lengths 3-7, stored below the widest one; they must not narrow with length */
		if (srcSize < WIN_HeaderSize || (srcPtr[2] >> 4) > 1) return 0;
		OffGroupsFine = srcPtr[2] >> 4;
		const int gap[5] = { srcPtr[0] & 15, srcPtr[0] >> 4, srcPtr[1] & 15, srcPtr[1] >> 4, srcPtr[2] & 15 };
		for (i = 3; i <= 7; i++) OffWidth[i] = OffWidth[8] - gap[i - 3];
		if (OffWidth[3] < 4) return 0;
		for (i = 4; i <= 8; i++)
			if (OffWidth[i] < OffWidth[i - 1]) return 0;
		srcPtr += WIN_HeaderSize;
	}
	i = 8;
	while (i > 0 && OffWidth[i] == OffWidth[i - 1])
		i--;

	Set_Offset_Groups(S_, i - MinMatchLen + 1, OffGroupsFine);

	if ( destSize >> 16 ) {                            /* Read the length of LZ literal sequence */
		nLzLits = MemReadLE4(srcPtr);
		srcPtr += 4;
	}
	else {
		nLzLits = MemReadLE2(srcPtr);
		srcPtr += 2;
	}
	if (nLzLits > (Uint32)destSize) return 0;                                   /* corrupt */
	Lit_Reader lits;                                 /* the literals, decoded as the sequences need them */
	Seq_DecTables* const T = Seq_DecTables_New();
	if (!Lit_Reader_Init(&lits, srcPtr, NULL, nLzLits, 1) || NULL == T) { free(lits.buf); free(lits.hst); free(T); return 0; }
	zipLitSize = Huffman_Skip_Trusted(srcPtr, nLzLits, N_HufLits);
	srcPtr += zipLitSize;

	Bit_Stream bitStream;
	bitStream.nUsedBits = 0;
	bitStream.container = MemReadBE8(srcPtr);
	bitStream.streamPtr = srcPtr;

	WLZ_HufWt_Set hufWtSet;                          /* the lengths of the codes; their tables in T */

	Uint32 offsetLast[OffCasheSize];
	memset(offsetLast, 0x7F, OffCasheSize * sizeof(int));
	int decSize = 0, pipelined = WZL_PIPELINE > 0;
	Uint8* const destEnd = (Uint8*)dest + destSize;
	while (decSize < destSize) {
		BITStream_Read(bitStream, 1, hufWtSet.slotJoint);           /* the block's coding: slot-joint or classic */
		Seq_Read_Tables(S_, &bitStream, &hufWtSet, T, 1);
		
		const int blockStart = decSize;
		Uint32 nFar = 0;
#if WZIP_DYNAMIC_BMI2
		if (CPU_Has_Bmi2())
			decSize = Decompress_WLZ_Sequence_Trusted_Bmi2(S_, &(bitStream.streamPtr), &lits, (Uint8*)dest, decSize, destEnd, dictEnd, dictSize, &hufWtSet, offsetLast, T, &nFar, pipelined);
		else
#endif
		decSize = Decompress_WLZ_Sequence_Trusted(S_, &(bitStream.streamPtr), &lits, (Uint8*)dest, decSize, destEnd, dictEnd, dictSize, &hufWtSet, offsetLast, T, &nFar, pipelined);
		if (decSize < 0) break;                      /* out of memory for a long literal run, or a damaged stream */
		pipelined = PIPELINE_NEXT(nFar, decSize - blockStart);   /* for the next block */
		bitStream.container = MemReadBE8(bitStream.streamPtr);
		bitStream.nUsedBits = 0;
	}

	free(lits.buf);
	free(lits.hst);
	free(T);
	if (destSize == decSize) return decSize;
	else return 0;
}

