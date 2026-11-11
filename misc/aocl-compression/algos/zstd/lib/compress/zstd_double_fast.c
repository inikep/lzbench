/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/**
 * Modifications Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

#include "utils/utils.h"
#include "zstd_compress_internal.h"
#include "zstd_double_fast.h"

#ifndef ZSTD_EXCLUDE_DFAST_BLOCK_COMPRESSOR

static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
void ZSTD_fillDoubleHashTableForCDict(ZSTD_MatchState_t* ms,
                              void const* end, ZSTD_dictTableLoadMethod_e dtlm)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashLarge = ms->hashTable;
    U32  const hBitsL = cParams->hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    U32  const mls = cParams->minMatch;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const BYTE* const base = ms->window.base;
    const BYTE* ip = base + ms->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const U32 fastHashFillStep = 3;

    /* Always insert every fastHashFillStep position into the hash tables.
     * Insert the other positions into the large hash table if their entry
     * is empty.
     */
    for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep) {
        U32 const curr = (U32)(ip - base);
        U32 i;
        for (i = 0; i < fastHashFillStep; ++i) {
            size_t const smHashAndTag = ZSTD_hashPtr(ip + i, hBitsS, mls);
            size_t const lgHashAndTag = ZSTD_hashPtr(ip + i, hBitsL, 8);
            if (i == 0) {
                ZSTD_writeTaggedIndex(hashSmall, smHashAndTag, curr + i);
            }
            if (i == 0 || hashLarge[lgHashAndTag >> ZSTD_SHORT_CACHE_TAG_BITS] == 0) {
                ZSTD_writeTaggedIndex(hashLarge, lgHashAndTag, curr + i);
            }
            /* Only load extra positions for ZSTD_dtlm_full */
            if (dtlm == ZSTD_dtlm_fast)
                break;
    }   }
}

static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
void ZSTD_fillDoubleHashTableForCCtx(ZSTD_MatchState_t* ms,
                              void const* end, ZSTD_dictTableLoadMethod_e dtlm)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashLarge = ms->hashTable;
    U32  const hBitsL = cParams->hashLog;
    U32  const mls = cParams->minMatch;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* ip = base + ms->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const U32 fastHashFillStep = 3;

    /* Always insert every fastHashFillStep position into the hash tables.
     * Insert the other positions into the large hash table if their entry
     * is empty.
     */
    for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep) {
        U32 const curr = (U32)(ip - base);
        U32 i;
        for (i = 0; i < fastHashFillStep; ++i) {
            size_t const smHash = ZSTD_hashPtr(ip + i, hBitsS, mls);
            size_t const lgHash = ZSTD_hashPtr(ip + i, hBitsL, 8);
            if (i == 0)
                hashSmall[smHash] = curr + i;
            if (i == 0 || hashLarge[lgHash] == 0)
                hashLarge[lgHash] = curr + i;
            /* Only load extra positions for ZSTD_dtlm_full */
            if (dtlm == ZSTD_dtlm_fast)
                break;
        }   }
}

void ZSTD_fillDoubleHashTable(ZSTD_MatchState_t* ms,
                        const void* const end,
                        ZSTD_dictTableLoadMethod_e dtlm,
                        ZSTD_tableFillPurpose_e tfp)
{
    if (tfp == ZSTD_tfp_forCDict) {
        ZSTD_fillDoubleHashTableForCDict(ms, end, dtlm);
    } else {
        ZSTD_fillDoubleHashTableForCCtx(ms, end, dtlm);
    }
}


FORCE_INLINE_TEMPLATE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_noDict_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize, U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    const U32 hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    const U32 hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];
    U32 offsetSaved1 = 0, offsetSaved2 = 0;

    size_t mLength;
    U32 offset;
    U32 curr;

    /* how many positions to search before increasing step size */
    const size_t kStepIncr = 1 << kSearchStrength;
    /* the position at which to increment the step size if no match is found */
    const BYTE* nextStep;
    size_t step; /* the current step size */

    size_t hl0; /* the long hash at ip */
    size_t hl1; /* the long hash at ip1 */

    U32 idxl0; /* the long match index for ip */
    U32 idxl1; /* the long match index for ip1 */

    const BYTE* matchl0; /* the long match for ip */
    const BYTE* matchs0; /* the short match for ip */
    const BYTE* matchl1; /* the long match for ip1 */
    const BYTE* matchs0_safe; /* matchs0 or safe address */

    const BYTE* ip = istart; /* the current position */
    const BYTE* ip1; /* the next position */
    /* Array of ~random data, should have low probability of matching data
     * we load from here instead of from tables, if matchl0/matchl1 are
     * invalid indices. Used to avoid unpredictable branches. */
    const BYTE dummy[] = {0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0xe2,0xb4};

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_noDict_generic");

    /* init */
    ip += ((ip - prefixLowest) == 0);
    {
        U32 const current = (U32)(ip - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, current, cParams->windowLog);
        U32 const maxRep = current - windowLow;
        if (offset_2 > maxRep) offsetSaved2 = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved1 = offset_1, offset_1 = 0;
    }

    /* Outer Loop: one iteration per match found and stored */
    while (1) {
        step = 1;
        nextStep = ip + kStepIncr;
        ip1 = ip + step;

        if (ip1 > ilimit) {
            goto _cleanup;
        }

        hl0 = ZSTD_hashPtr(ip, hBitsL, 8);
        idxl0 = hashLong[hl0];
        matchl0 = base + idxl0;

        /* Inner Loop: one iteration per search / position */
        do {
            const size_t hs0 = ZSTD_hashPtr(ip, hBitsS, mls);
            const U32 idxs0 = hashSmall[hs0];
            curr = (U32)(ip-base);
            matchs0 = base + idxs0;

            hashLong[hl0] = hashSmall[hs0] = curr;   /* update hash tables */

            /* check noDict repcode */
            if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
                mLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
                ip++;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
                goto _match_stored;
            }

            hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);

            /* idxl0 > prefixLowestIndex is a (somewhat) unpredictable branch.
             * However expression below complies into conditional move. Since
             * match is unlikely and we only *branch* on idxl0 > prefixLowestIndex
             * if there is a match, all branches become predictable. */
            {   const BYTE*  const matchl0_safe = ZSTD_selectAddr(idxl0, prefixLowestIndex, matchl0, &dummy[0]);

                /* check prefix long match */
                if (MEM_read64(matchl0_safe) == MEM_read64(ip) && matchl0_safe == matchl0) {
                    mLength = ZSTD_count(ip+8, matchl0+8, iend) + 8;
                    offset = (U32)(ip-matchl0);
                    while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; } /* catch up */
                    goto _match_found;
            }   }

            idxl1 = hashLong[hl1];
            matchl1 = base + idxl1;

            /* Same optimization as matchl0 above */
            matchs0_safe = ZSTD_selectAddr(idxs0, prefixLowestIndex, matchs0, &dummy[0]);

            /* check prefix short match */
            if(MEM_read32(matchs0_safe) == MEM_read32(ip) && matchs0_safe == matchs0) {
                  goto _search_next_long;
            }

            if (ip1 >= nextStep) {
                PREFETCH_L1(ip1 + 64);
                PREFETCH_L1(ip1 + 128);
                step++;
                nextStep += kStepIncr;
            }
            ip = ip1;
            ip1 += step;

            hl0 = hl1;
            idxl0 = idxl1;
            matchl0 = matchl1;
    #if defined(__aarch64__)
            PREFETCH_L1(ip+256);
    #endif
        } while (ip1 <= ilimit);

_cleanup:
        /* If offset_1 started invalid (offsetSaved1 != 0) and became valid (offset_1 != 0),
         * rotate saved offsets. See comment in ZSTD_compressBlock_fast_noDict for more context. */
        offsetSaved2 = ((offsetSaved1 != 0) && (offset_1 != 0)) ? offsetSaved1 : offsetSaved2;

        /* save reps for next block */
        rep[0] = offset_1 ? offset_1 : offsetSaved1;
        rep[1] = offset_2 ? offset_2 : offsetSaved2;

        /* Return the last literals size */
        return (size_t)(iend - anchor);

_search_next_long:

        /* short match found: let's check for a longer one */
        mLength = ZSTD_count(ip+4, matchs0+4, iend) + 4;
        offset = (U32)(ip - matchs0);

        /* check long match at +1 position */
        if ((idxl1 > prefixLowestIndex) && (MEM_read64(matchl1) == MEM_read64(ip1))) {
            size_t const l1len = ZSTD_count(ip1+8, matchl1+8, iend) + 8;
            if (l1len > mLength) {
                /* use the long match instead */
                ip = ip1;
                mLength = l1len;
                offset = (U32)(ip-matchl1);
                matchs0 = matchl1;
            }
        }

        while (((ip>anchor) & (matchs0>prefixLowest)) && (ip[-1] == matchs0[-1])) { ip--; matchs0--; mLength++; } /* complete backward */

        /* fall-through */

_match_found: /* requires ip, offset, mLength */
        offset_2 = offset_1;
        offset_1 = offset;

        if (step < 4) {
            /* It is unsafe to write this value back to the hashtable when ip1 is
             * greater than or equal to the new ip we will have after we're done
             * processing this match. Rather than perform that test directly
             * (ip1 >= ip + mLength), which costs speed in practice, we do a simpler
             * more predictable test. The minmatch even if we take a short match is
             * 4 bytes, so as long as step, the distance between ip and ip1
             * (initially) is less than 4, we know ip1 < new ip. */
            hashLong[hl1] = (U32)(ip1 - base);
        }

        ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

_match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
            }

            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                    & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
                U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff;  /* swap offset_2 <=> offset_1 */
                hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = (U32)(ip-base);
                hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = (U32)(ip-base);
                ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, rLength);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
            }
        }
    }
}


FORCE_INLINE_TEMPLATE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_dictMatchState_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    const U32 hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    const U32 hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];

    const ZSTD_MatchState_t* const dms = ms->dictMatchState;
    const ZSTD_compressionParameters* const dictCParams = &dms->cParams;
    const U32* const dictHashLong  = dms->hashTable;
    const U32* const dictHashSmall = dms->chainTable;
    const U32 dictStartIndex       = dms->window.dictLimit;
    const BYTE* const dictBase     = dms->window.base;
    const BYTE* const dictStart    = dictBase + dictStartIndex;
    const BYTE* const dictEnd      = dms->window.nextSrc;
    const U32 dictIndexDelta       = prefixLowestIndex - (U32)(dictEnd - dictBase);
    const U32 dictHBitsL           = dictCParams->hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const U32 dictHBitsS           = dictCParams->chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const U32 dictAndPrefixLength  = (U32)((ip - prefixLowest) + (dictEnd - dictStart));

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_dictMatchState_generic");

    /* if a dictionary is attached, it must be within window range */
    assert(ms->window.dictLimit + (1U << cParams->windowLog) >= endIndex);

    if (ms->prefetchCDictTables) {
        size_t const hashTableBytes = (((size_t)1) << dictCParams->hashLog) * sizeof(U32);
        size_t const chainTableBytes = (((size_t)1) << dictCParams->chainLog) * sizeof(U32);
        PREFETCH_AREA(dictHashLong, hashTableBytes);
        PREFETCH_AREA(dictHashSmall, chainTableBytes);
    }

    /* init */
    ip += (dictAndPrefixLength == 0);

    /* dictMatchState repCode checks don't currently handle repCode == 0
     * disabling. */
    assert(offset_1 <= dictAndPrefixLength);
    assert(offset_2 <= dictAndPrefixLength);

    /* Main Search Loop */
    while (ip < ilimit) {   /* < instead of <=, because repcode check at (ip+1) */
        size_t mLength;
        U32 offset;
        size_t const h2 = ZSTD_hashPtr(ip, hBitsL, 8);
        size_t const h = ZSTD_hashPtr(ip, hBitsS, mls);
        size_t const dictHashAndTagL = ZSTD_hashPtr(ip, dictHBitsL, 8);
        size_t const dictHashAndTagS = ZSTD_hashPtr(ip, dictHBitsS, mls);
        U32 const dictMatchIndexAndTagL = dictHashLong[dictHashAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS];
        U32 const dictMatchIndexAndTagS = dictHashSmall[dictHashAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS];
        int const dictTagsMatchL = ZSTD_comparePackedTags(dictMatchIndexAndTagL, dictHashAndTagL);
        int const dictTagsMatchS = ZSTD_comparePackedTags(dictMatchIndexAndTagS, dictHashAndTagS);
        U32 const curr = (U32)(ip-base);
        U32 const matchIndexL = hashLong[h2];
        U32 matchIndexS = hashSmall[h];
        const BYTE* matchLong = base + matchIndexL;
        const BYTE* match = base + matchIndexS;
        const U32 repIndex = curr + 1 - offset_1;
        const BYTE* repMatch = (repIndex < prefixLowestIndex) ?
                               dictBase + (repIndex - dictIndexDelta) :
                               base + repIndex;
        hashLong[h2] = hashSmall[h] = curr;   /* update hash tables */

        /* check repcode */
        if ((ZSTD_index_overlap_check(prefixLowestIndex, repIndex))
            && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < prefixLowestIndex ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, prefixLowest) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
            goto _match_stored;
        }

        if ((matchIndexL >= prefixLowestIndex) && (MEM_read64(matchLong) == MEM_read64(ip))) {
            /* check prefix long match */
            mLength = ZSTD_count(ip+8, matchLong+8, iend) + 8;
            offset = (U32)(ip-matchLong);
            while (((ip>anchor) & (matchLong>prefixLowest)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; } /* catch up */
            goto _match_found;
        } else if (dictTagsMatchL) {
            /* check dictMatchState long match */
            U32 const dictMatchIndexL = dictMatchIndexAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS;
            const BYTE* dictMatchL = dictBase + dictMatchIndexL;
            assert(dictMatchL < dictEnd);

            if (dictMatchL > dictStart && MEM_read64(dictMatchL) == MEM_read64(ip)) {
                mLength = ZSTD_count_2segments(ip+8, dictMatchL+8, iend, dictEnd, prefixLowest) + 8;
                offset = (U32)(curr - dictMatchIndexL - dictIndexDelta);
                while (((ip>anchor) & (dictMatchL>dictStart)) && (ip[-1] == dictMatchL[-1])) { ip--; dictMatchL--; mLength++; } /* catch up */
                goto _match_found;
        }   }

        if (matchIndexS > prefixLowestIndex) {
            /* short match  candidate */
            if (MEM_read32(match) == MEM_read32(ip)) {
                goto _search_next_long;
            }
        } else if (dictTagsMatchS) {
            /* check dictMatchState short match */
            U32 const dictMatchIndexS = dictMatchIndexAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS;
            match = dictBase + dictMatchIndexS;
            matchIndexS = dictMatchIndexS + dictIndexDelta;

            if (match > dictStart && MEM_read32(match) == MEM_read32(ip)) {
                goto _search_next_long;
        }   }

        ip += ((ip-anchor) >> kSearchStrength) + 1;
#if defined(__aarch64__)
        PREFETCH_L1(ip+256);
#endif
        continue;

_search_next_long:
        {   size_t const hl3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
            size_t const dictHashAndTagL3 = ZSTD_hashPtr(ip+1, dictHBitsL, 8);
            U32 const matchIndexL3 = hashLong[hl3];
            U32 const dictMatchIndexAndTagL3 = dictHashLong[dictHashAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS];
            int const dictTagsMatchL3 = ZSTD_comparePackedTags(dictMatchIndexAndTagL3, dictHashAndTagL3);
            const BYTE* matchL3 = base + matchIndexL3;
            hashLong[hl3] = curr + 1;

            /* check prefix long +1 match */
            if ((matchIndexL3 >= prefixLowestIndex) && (MEM_read64(matchL3) == MEM_read64(ip+1))) {
                mLength = ZSTD_count(ip+9, matchL3+8, iend) + 8;
                ip++;
                offset = (U32)(ip-matchL3);
                while (((ip>anchor) & (matchL3>prefixLowest)) && (ip[-1] == matchL3[-1])) { ip--; matchL3--; mLength++; } /* catch up */
                goto _match_found;
            } else if (dictTagsMatchL3) {
                /* check dict long +1 match */
                U32 const dictMatchIndexL3 = dictMatchIndexAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS;
                const BYTE* dictMatchL3 = dictBase + dictMatchIndexL3;
                assert(dictMatchL3 < dictEnd);
                if (dictMatchL3 > dictStart && MEM_read64(dictMatchL3) == MEM_read64(ip+1)) {
                    mLength = ZSTD_count_2segments(ip+1+8, dictMatchL3+8, iend, dictEnd, prefixLowest) + 8;
                    ip++;
                    offset = (U32)(curr + 1 - dictMatchIndexL3 - dictIndexDelta);
                    while (((ip>anchor) & (dictMatchL3>dictStart)) && (ip[-1] == dictMatchL3[-1])) { ip--; dictMatchL3--; mLength++; } /* catch up */
                    goto _match_found;
        }   }   }

        /* if no long +1 match, explore the short match we found */
        if (matchIndexS < prefixLowestIndex) {
            mLength = ZSTD_count_2segments(ip+4, match+4, iend, dictEnd, prefixLowest) + 4;
            offset = (U32)(curr - matchIndexS);
            while (((ip>anchor) & (match>dictStart)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
        } else {
            mLength = ZSTD_count(ip+4, match+4, iend) + 4;
            offset = (U32)(ip - match);
            while (((ip>anchor) & (match>prefixLowest)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
        }

_match_found:
        offset_2 = offset_1;
        offset_1 = offset;

        ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

_match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
            }

            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < prefixLowestIndex ?
                        dictBase + repIndex2 - dictIndexDelta :
                        base + repIndex2;
                if ( (ZSTD_index_overlap_check(prefixLowestIndex, repIndex2))
                   && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < prefixLowestIndex ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, prefixLowest) + 4;
                    U32 tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, repLength2);
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }   /* while (ip < ilimit) */

    /* save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
}

#define ZSTD_GEN_DFAST_FN(dictMode, mls)                                                                 \
    static size_t ZSTD_compressBlock_doubleFast_##dictMode##_##mls(                                      \
            ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],                          \
            void const* src, size_t srcSize)                                                             \
    {                                                                                                    \
        return ZSTD_compressBlock_doubleFast_##dictMode##_generic(ms, seqStore, rep, src, srcSize, mls); \
    }

ZSTD_GEN_DFAST_FN(noDict, 4)
ZSTD_GEN_DFAST_FN(noDict, 5)
ZSTD_GEN_DFAST_FN(noDict, 6)
ZSTD_GEN_DFAST_FN(noDict, 7)

ZSTD_GEN_DFAST_FN(dictMatchState, 4)
ZSTD_GEN_DFAST_FN(dictMatchState, 5)
ZSTD_GEN_DFAST_FN(dictMatchState, 6)
ZSTD_GEN_DFAST_FN(dictMatchState, 7)

size_t ZSTD_compressBlock_doubleFast(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    const U32 mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_noDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_noDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_noDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_noDict_7(ms, seqStore, rep, src, srcSize);
    }
}

size_t ZSTD_compressBlock_doubleFast_dictMatchState(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    const U32 mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_7(ms, seqStore, rep, src, srcSize);
    }
}


static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_extDict_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    U32  const hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ms->window.base;
    const U32   endIndex = (U32)((size_t)(istart - base) + srcSize);
    const U32   lowLimit = ZSTD_getLowestMatchIndex(ms, endIndex, cParams->windowLog);
    const U32   dictStartIndex = lowLimit;
    const U32   dictLimit = ms->window.dictLimit;
    const U32   prefixStartIndex = (dictLimit > lowLimit) ? dictLimit : lowLimit;
    const BYTE* const prefixStart = base + prefixStartIndex;
    const BYTE* const dictBase = ms->window.dictBase;
    const BYTE* const dictStart = dictBase + dictStartIndex;
    const BYTE* const dictEnd = dictBase + prefixStartIndex;
    U32 offset_1=rep[0], offset_2=rep[1];

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_extDict_generic (srcSize=%zu)", srcSize);

    /* if extDict is invalidated due to maxDistance, switch to "regular" variant */
    if (prefixStartIndex == dictStartIndex)
        return ZSTD_compressBlock_doubleFast(ms, seqStore, rep, src, srcSize);

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        const size_t hSmall = ZSTD_hashPtr(ip, hBitsS, mls);
        const U32 matchIndex = hashSmall[hSmall];
        const BYTE* const matchBase = matchIndex < prefixStartIndex ? dictBase : base;
        const BYTE* match = matchBase + matchIndex;

        const size_t hLong = ZSTD_hashPtr(ip, hBitsL, 8);
        const U32 matchLongIndex = hashLong[hLong];
        const BYTE* const matchLongBase = matchLongIndex < prefixStartIndex ? dictBase : base;
        const BYTE* matchLong = matchLongBase + matchLongIndex;

        const U32 curr = (U32)(ip-base);
        const U32 repIndex = curr + 1 - offset_1;   /* offset_1 expected <= curr +1 */
        const BYTE* const repBase = repIndex < prefixStartIndex ? dictBase : base;
        const BYTE* const repMatch = repBase + repIndex;
        size_t mLength;
        hashSmall[hSmall] = hashLong[hLong] = curr;   /* update hash table */

        if (((ZSTD_index_overlap_check(prefixStartIndex, repIndex))
            & (offset_1 <= curr+1 - dictStartIndex)) /* note: we are searching at curr+1 */
          && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < prefixStartIndex ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, prefixStart) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
        } else {
            if ((matchLongIndex > dictStartIndex) && (MEM_read64(matchLong) == MEM_read64(ip))) {
                const BYTE* const matchEnd = matchLongIndex < prefixStartIndex ? dictEnd : iend;
                const BYTE* const lowMatchPtr = matchLongIndex < prefixStartIndex ? dictStart : prefixStart;
                U32 offset;
                mLength = ZSTD_count_2segments(ip+8, matchLong+8, iend, matchEnd, prefixStart) + 8;
                offset = curr - matchLongIndex;
                while (((ip>anchor) & (matchLong>lowMatchPtr)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; }   /* catch up */
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

            } else if ((matchIndex > dictStartIndex) && (MEM_read32(match) == MEM_read32(ip))) {
                size_t const h3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
                U32 const matchIndex3 = hashLong[h3];
                const BYTE* const match3Base = matchIndex3 < prefixStartIndex ? dictBase : base;
                const BYTE* match3 = match3Base + matchIndex3;
                U32 offset;
                hashLong[h3] = curr + 1;
                if ( (matchIndex3 > dictStartIndex) && (MEM_read64(match3) == MEM_read64(ip+1)) ) {
                    const BYTE* const matchEnd = matchIndex3 < prefixStartIndex ? dictEnd : iend;
                    const BYTE* const lowMatchPtr = matchIndex3 < prefixStartIndex ? dictStart : prefixStart;
                    mLength = ZSTD_count_2segments(ip+9, match3+8, iend, matchEnd, prefixStart) + 8;
                    ip++;
                    offset = curr+1 - matchIndex3;
                    while (((ip>anchor) & (match3>lowMatchPtr)) && (ip[-1] == match3[-1])) { ip--; match3--; mLength++; } /* catch up */
                } else {
                    const BYTE* const matchEnd = matchIndex < prefixStartIndex ? dictEnd : iend;
                    const BYTE* const lowMatchPtr = matchIndex < prefixStartIndex ? dictStart : prefixStart;
                    mLength = ZSTD_count_2segments(ip+4, match+4, iend, matchEnd, prefixStart) + 4;
                    offset = curr - matchIndex;
                    while (((ip>anchor) & (match>lowMatchPtr)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; }   /* catch up */
                }
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

            } else {
                ip += ((ip-anchor) >> kSearchStrength) + 1;
                continue;
        }   }

        /* move to next sequence start */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
            }

            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < prefixStartIndex ? dictBase + repIndex2 : base + repIndex2;
                if ( ((ZSTD_index_overlap_check(prefixStartIndex, repIndex2))
                    & (offset_2 <= current2 - dictStartIndex))
                  && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < prefixStartIndex ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, prefixStart) + 4;
                    U32 const tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, repLength2);
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
    }   }   }

    /* save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
}

ZSTD_GEN_DFAST_FN(extDict, 4)
ZSTD_GEN_DFAST_FN(extDict, 5)
ZSTD_GEN_DFAST_FN(extDict, 6)
ZSTD_GEN_DFAST_FN(extDict, 7)

size_t ZSTD_compressBlock_doubleFast_extDict(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    U32 const mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_extDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_extDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_extDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_extDict_7(ms, seqStore, rep, src, srcSize);
    }
}

/* ***************************************************************
*  AOCL optimized functions
*****************************************************************/
#ifdef AOCL_ZSTD_OPT
#define PREFETCH_OFFSET 8
#define PREFETCH_SAFETY (PREFETCH_OFFSET + 11) // value at ip + PREFETCH_SAFETY can be read to prefetch match candidates. +11 as endmost prefetching happens with ip1 + 11.
/* Compute hash for search key at PREFETCH_OFFSET from ip
*  Prefetch match candidate stored at hashTable[hashIndex]
*  If hashTable[hashIndex] is empty, (base + 0) is still a valid address and can be accessed. No check added for this to avoid branching. */
#define PREFETCH_MATCH(ip, hBits, hSize, hashTable) { \
    size_t nexth = ZSTD_hashPtr((ip) + PREFETCH_OFFSET, hBits, hSize); \
    U32 idxn = hashTable[nexth]; \
    PREFETCH_L1(base + idxn); \
}
#define PREFETCH_MATCH_L(ip) PREFETCH_MATCH(ip, hBitsL, 8, hashLong) // prefetch match candidate stored in hashLong 
#define PREFETCH_MATCH_S(ip) PREFETCH_MATCH(ip, hBitsS, mls, hashSmall) // prefetch match candidate stored in hashSmall 

/* The following optimizations have been included in the optimized function:
    - code refactoring on the lines of AOCL_ZSTD_compressBlock_doubleFast_noDict_generic()
    - when the AOCL_ZSTD_SEARCH_SKIP_OPT flag is enabled,
        - the search tolerance is set to 2^5 (32)
        - for every 32 byte blocks that go without a single match, the step rate is increased by 3 instead of 1
    - in search_next_long, the condition is wrapped with an UNLIKELY() to improve branch prediction
    - More complementary insertions to improve ratio
    - Prefetch potential future match candidates
*/
size_t AOCL_ZSTD_compressBlock_doubleFast_extDict_generic(
    ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
    void const* src, size_t srcSize,
    U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    U32  const hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* anchor = istart;
    const U32   endIndex = (U32)((size_t)(istart - base) + srcSize);

    const U32   lowLimit = ZSTD_getLowestMatchIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8 - PREFETCH_SAFETY; //- HASH_READ_SIZE

    const U32   dictStartIndex = lowLimit;
    const U32   dictLimit = ms->window.dictLimit;
    const U32   prefixStartIndex = (dictLimit > lowLimit) ? dictLimit : lowLimit;
    const BYTE* const prefixStart = base + prefixStartIndex;
    const BYTE* const dictBase = ms->window.dictBase;
    const BYTE* const dictStart = dictBase + dictStartIndex;
    const BYTE* const dictEnd = dictBase + prefixStartIndex;

    U32 offset_1 = rep[0], offset_2 = rep[1];

    size_t mLength;
    U32 offset;
    U32 curr;

    /* how many positions to search before increasing step size */
#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
    const size_t kStepIncr = 1 << aocl_kSearchStrengthDoubleFast;
    const BYTE* nextStep;
    size_t step; /* the current step size */
#endif /* AOCL_ZSTD_SEARCH_SKIP_OPT */

    size_t hl0; /* the long hash at ip */
    size_t hl1; /* the long hash at ip1 */

    U32 idxl0; /* the long match index for ip */
    U32 idxl1; /* the long match index for ip1 */
    U32 idxs0;

    const BYTE* matchl0; /* the long match for ip */
    const BYTE* matchs0; /* the short match for ip */
    const BYTE* matchl1; /* the long match for ip1 */

    const BYTE* ip = istart; /* the current position */
    const BYTE* ip1; /* the next position */

    DEBUGLOG(5, "AOCL_ZSTD_compressBlock_doubleFast_extDict_generic (srcSize=%zu)", srcSize);

    /* if extDict is invalidated due to maxDistance, switch to "regular" variant */
    if (prefixStartIndex == dictStartIndex)
        return ZSTD_compressBlock_doubleFast(ms, seqStore, rep, src, srcSize);

    /* Outer Loop: one iteration per match found and stored */
    while (1) {  /* < instead of <=, because (ip+1) */
#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
        step = 1;
        nextStep = ip + kStepIncr;
        ip1 = ip + step;
#else
        ip1 = ip + 1;
#endif

        if (ip1 > ilimit) {
            goto _cleanup;
        }

        {
            hl0 = ZSTD_hashPtr(ip, hBitsL, 8);
            PREFETCH_MATCH_L(ip);
            idxl0 = hashLong[hl0];
            const BYTE* const matchLongBase = idxl0 < prefixStartIndex ? dictBase : base;
            matchl0 = matchLongBase + idxl0;
        }

        /* Inner Loop: one iteration per search / position */
        do {
#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
            // Alternate between values 1 and 2 for step while searching for a match. In case
            // the value of step exceeds 2 (this happens when the value of step is incremented
            // when ip1 exceeds nextStep), we let step retain its value.
            step = (step > 2) ? step : step ^ 3;
#endif
            const size_t hs0 = ZSTD_hashPtr(ip, hBitsS, mls);
            idxs0 = hashSmall[hs0];
            curr = (U32)(ip - base);
            const BYTE* const matchBase = idxs0 < prefixStartIndex ? dictBase : base;
            matchs0 = matchBase + idxs0;

            hashSmall[hs0] = hashLong[hl0] = curr;   /* update hash table */

            /* check extDict repcode */
            //DIFF: (offset_1 > 0) --> (offset_1 <= curr+1 - dictStartIndex)
            const U32 repIndex = curr + 1 - offset_1;   /* offset_1 expected <= curr +1 */
            const BYTE* const repBase = repIndex < prefixStartIndex ? dictBase : base;
            const BYTE* const repMatch = repBase + repIndex;
            if (((ZSTD_index_overlap_check(prefixStartIndex, repIndex))
                & (offset_1 <= curr + 1 - dictStartIndex)) /* note: we are searching at curr+1 */
                && (MEM_read32(repMatch) == MEM_read32(ip + 1)))
            {
                PREFETCH_MATCH_L(ip + 1 + 4);
                const BYTE* repMatchEnd = repIndex < prefixStartIndex ? dictEnd : iend;
                mLength = ZSTD_count_2segments(ip + 1 + 4, repMatch + 4, iend, repMatchEnd, prefixStart) + 4;
                ip++;
                ZSTD_storeSeq(seqStore, (size_t)(ip - anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
                goto _match_stored;
            }

            hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);

            if ((idxl0 > dictStartIndex) &&
                (MEM_read64(matchl0) == MEM_read64(ip))) {
                PREFETCH_MATCH_L(ip1 + 8);
                PREFETCH_MATCH_L(ip1 + 9);
                PREFETCH_MATCH_L(ip1 + 10);
                PREFETCH_MATCH_L(ip1 + 11);
                /* check prefix long match */
                const BYTE* const matchEnd = idxl0 < prefixStartIndex ? dictEnd : iend;
                const BYTE* const lowMatchPtr = idxl0 < prefixStartIndex ? dictStart : prefixStart;
                mLength = ZSTD_count_2segments(ip + 8, matchl0 + 8, iend, matchEnd, prefixStart) + 8;
                offset = curr - idxl0;
                while (((ip > anchor) & (matchl0 > lowMatchPtr)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; }   /* catch up */
                goto _match_found;

            }

            {
                idxl1 = hashLong[hl1];
                const BYTE* const match3Base = idxl1 < prefixStartIndex ? dictBase : base;
                matchl1 = match3Base + idxl1;
            }

            /* check prefix short match */
            if ((idxs0 > dictStartIndex) &&
                (MEM_read32(matchs0) == MEM_read32(ip))) {
                PREFETCH_MATCH_S(ip + 4);
                goto _search_next_long;
            }

#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
            //step factor logic extDict
            if (ip1 >= nextStep) {
                step += 3;
                nextStep += kStepIncr;
            }
            ip = ip1;
            ip1 += step;
#else
            ip = ip1;
            ip1 += ((ip1 - anchor) >> kSearchStrength) + 1;
#endif /* AOCL_ZSTD_SEARCH_SKIP_OPT */

            hl0 = hl1;
            idxl0 = idxl1;
            matchl0 = matchl1;
        } while (ip1 <= ilimit);

    _cleanup:
        /* save reps for next block */
        rep[0] = offset_1;
        rep[1] = offset_2;

        /* Return the last literals size */
        return (size_t)(iend - anchor);

    _search_next_long:
        hashLong[hl1] = curr + 1;
        if (idxl1 > dictStartIndex) {
            if (UNLIKELY(MEM_read64(matchl1) == MEM_read64(ip1))) {
                ip = ip1;
                const BYTE* const matchEnd = idxl1 < prefixStartIndex ? dictEnd : iend;
                const BYTE* const lowMatchPtr = idxl1 < prefixStartIndex ? dictStart : prefixStart;
                mLength = ZSTD_count_2segments(ip + 8, matchl1 + 8, iend, matchEnd, prefixStart) + 8;
                offset = (ip1 - base) - idxl1; //curr + 1 - idxl1;
                while (((ip > anchor) & (matchl1 > lowMatchPtr)) && (ip[-1] == matchl1[-1])) { ip--; matchl1--; mLength++; } /* catch up */
                goto _match_found;
            }
        }

        {
            /* if no long +1 match, explore the short match we found */
            const BYTE* const matchEnd = idxs0 < prefixStartIndex ? dictEnd : iend;
            const BYTE* const lowMatchPtr = idxs0 < prefixStartIndex ? dictStart : prefixStart;
            mLength = ZSTD_count_2segments(ip + 4, matchs0 + 4, iend, matchEnd, prefixStart) + 4;
            offset = curr - idxs0;
            while (((ip > anchor) & (matchs0 > lowMatchPtr)) && (ip[-1] == matchs0[-1])) { ip--; matchs0--; mLength++; }   /* catch up */
        }

    _match_found: /* requires ip, offset, mLength */
        offset_2 = offset_1;
        offset_1 = offset;
        ZSTD_storeSeq(seqStore, (size_t)(ip - anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
        if (step < 4) {
            /* It is unsafe to write this value back to the hashtable when ip1 is
             * greater than or equal to the new ip we will have after we're done
             * processing this match. Rather than perform that test directly
             * (ip1 >= ip + mLength), which costs speed in practice, we do a simpler
             * more predictable test. The minmatch even if we take a short match is
             * 4 bytes, so as long as step, the distance between ip and ip1
             * (initially) is less than 4, we know ip1 < new ip. */
            hashLong[hl1] = (U32)(ip1 - base);
        }
#endif

    _match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr + 2;
#if defined(AOCL_ZSTD_SEARCH_SKIP_OPT)
            /* More complementary insertions to improve ratio */
            hashLong[ZSTD_hashPtr(base + indexToInsert, hBitsL, 8)] = indexToInsert;
            hashLong[ZSTD_hashPtr(ip - 2, hBitsL, 8)] = (U32)(ip - 2 - base);
            hashLong[ZSTD_hashPtr(ip - 1, hBitsL, 8)] = (U32)(ip - 1 - base);
            hashSmall[ZSTD_hashPtr(base + indexToInsert, hBitsS, mls)] = indexToInsert;
            hashSmall[ZSTD_hashPtr(ip - 2, hBitsS, mls)] = (U32)(ip - 2 - base);
            hashSmall[ZSTD_hashPtr(ip - 1, hBitsS, mls)] = (U32)(ip - 1 - base);
#else
            hashLong[ZSTD_hashPtr(base + indexToInsert, hBitsL, 8)] = indexToInsert;
            hashLong[ZSTD_hashPtr(ip - 2, hBitsL, 8)] = (U32)(ip - 2 - base);
            hashSmall[ZSTD_hashPtr(base + indexToInsert, hBitsS, mls)] = indexToInsert;
            hashSmall[ZSTD_hashPtr(ip - 1, hBitsS, mls)] = (U32)(ip - 1 - base);
#endif
            }

            /* check immediate repcode */
            //DIFF: (offset_2 > 0) --> (offset_2 <= current2 - dictStartIndex)
            //if within loop and break;
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip - base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < prefixStartIndex ? dictBase + repIndex2 : base + repIndex2;
                if ( ((ZSTD_index_overlap_check(prefixStartIndex, repIndex2))
                    & (offset_2 <= current2 - dictStartIndex))
                    && (MEM_read32(repMatch2) == MEM_read32(ip))) {
                    const BYTE* const repEnd2 = repIndex2 < prefixStartIndex ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip + 4, repMatch2 + 4, iend, repEnd2, prefixStart) + 4;
                    U32 const tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, repLength2);
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }
}

#define AOCL_ZSTD_GEN_DFAST_EXTDICT_FN(mls)                                                              \
    static size_t AOCL_ZSTD_compressBlock_doubleFast_extDict_##mls(                                      \
        ZSTD_MatchState_t *ms, SeqStore_t *seqStore, U32 rep[ZSTD_REP_NUM],                              \
        void const *src, size_t srcSize)                                                                 \
    {                                                                                                    \
        return AOCL_ZSTD_compressBlock_doubleFast_extDict_generic(ms, seqStore, rep, src, srcSize, mls); \
    }

AOCL_ZSTD_GEN_DFAST_EXTDICT_FN(4)
AOCL_ZSTD_GEN_DFAST_EXTDICT_FN(5)
AOCL_ZSTD_GEN_DFAST_EXTDICT_FN(6)
AOCL_ZSTD_GEN_DFAST_EXTDICT_FN(7)

size_t AOCL_ZSTD_compressBlock_doubleFast_extDict(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    U32 const mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return AOCL_ZSTD_compressBlock_doubleFast_extDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return AOCL_ZSTD_compressBlock_doubleFast_extDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return AOCL_ZSTD_compressBlock_doubleFast_extDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return AOCL_ZSTD_compressBlock_doubleFast_extDict_7(ms, seqStore, rep, src, srcSize);
    }
}
#endif /* AOCL_ZSTD_OPT */

#ifdef AOCL_ZSTD_OPT
#if AOCL_DECOMPRESS_FAST > 2
#include "algos/zstd/lib/compress/aocl_zstd_compressBlock_doubleFast_noDict_generic_fds3_analyze.h"
#include "algos/zstd/lib/compress/aocl_zstd_compressBlock_doubleFast_noDict_generic_fds3_base.h"
#elif AOCL_DECOMPRESS_FAST == 2
#include "algos/zstd/lib/compress/aocl_zstd_compressBlock_doubleFast_noDict_generic_fds2_base.h"
#endif
#if AOCL_DECOMPRESS_FAST > 1
#define AOCL_ZSTD_SEARCH_SKIP_LESS_OPT 1 /* Use moderate skip strategy for better compression ratio */
#endif

#if AOCL_DECOMPRESS_FAST != 2  /* FDS 2 calls only *_fds2_base functions */
/* The following optimizations have been included in the optimized function:
    - when the AOCL_ZSTD_SEARCH_SKIP_OPT flag is enabled,
        - the search tolerance is reduced to 2^5 (32) instead of 2^8 (256)
        - for every 32 byte blocks that go without a single match, the step rate is increased by 3 instead of 1
        - unnecessary prefetching is avoided when increasing step size
    - in search_next_long, the condition is wrapped with an UNLIKELY() to improve branch prediction
    - AOCL_ZSTD_count is called in place of ZSTD_count
    - More complementary insertions to improve ratio 
    - Prefetch potential future match candidates
*/
FORCE_INLINE_TEMPLATE
size_t AOCL_ZSTD_compressBlock_doubleFast_noDict_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize, U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    const U32 hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    const U32 hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE - PREFETCH_SAFETY;
    U32 offset_1=rep[0], offset_2=rep[1];
    U32 offsetSaved1 = 0, offsetSaved2 = 0;

    size_t mLength;
    U32 offset;
    U32 curr;

    /* how many positions to search before increasing step size */
#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
    const size_t kStepIncr = 1 << aocl_kSearchStrengthDoubleFast;
#else
    const size_t kStepIncr = 1 << kSearchStrength;
#endif /* AOCL_ZSTD_SEARCH_SKIP_OPT */

    const BYTE* nextStep;
    size_t step; /* the current step size */

    size_t hl0; /* the long hash at ip */
    size_t hl1; /* the long hash at ip1 */

    U32 idxl0; /* the long match index for ip */
    U32 idxl1; /* the long match index for ip1 */

    const BYTE* matchl0; /* the long match for ip */
    const BYTE* matchs0; /* the short match for ip */
    const BYTE* matchl1; /* the long match for ip1 */

    /* the position at which to increment the step size if no match is found */
    const BYTE* ip = istart; /* the current position */
    const BYTE* ip1; /* the next position */

    DEBUGLOG(5, "AOCL_ZSTD_compressBlock_doubleFast_noDict_generic");

    /* init */
    ip += ((ip - prefixLowest) == 0);
    {
        U32 const current = (U32)(ip - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, current, cParams->windowLog);
        U32 const maxRep = current - windowLow;
        if (offset_2 > maxRep) offsetSaved2 = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved1 = offset_1, offset_1 = 0;
    }

    /* Outer Loop: one iteration per match found and stored */
    while (1) {
        step = 1;
        nextStep = ip + kStepIncr;
        ip1 = ip + step;

        if (ip1 > ilimit) {
            goto _cleanup;
        }

        hl0 = ZSTD_hashPtr(ip, hBitsL, 8);
        PREFETCH_MATCH_L(ip);
        idxl0 = hashLong[hl0];
        matchl0 = base + idxl0;

        /* Inner Loop: one iteration per search / position */
        do {
#if defined(AOCL_ZSTD_SEARCH_SKIP_OPT) && !defined(AOCL_ZSTD_SEARCH_SKIP_LESS_OPT)
            // Alternate between values 1 and 2 for step while searching for a match. In case
            // the value of step exceeds 2 (this happens when the value of step is incremented
            // when ip1 exceeds nextStep), we let step retain its value.
            step = (step > 2) ? step : step ^ 3;
#endif
            const size_t hs0 = ZSTD_hashPtr(ip, hBitsS, mls);
            const U32 idxs0 = hashSmall[hs0];
            curr = (U32)(ip-base);
            matchs0 = base + idxs0;

            hashLong[hl0] = hashSmall[hs0] = curr;   /* update hash tables */

            /* check noDict repcode */
            if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
                PREFETCH_MATCH_L(ip+1+4);
                mLength = AOCL_ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
                ip++;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
                goto _match_stored;
            }

            hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);

            if (idxl0 > prefixLowestIndex) {
                /* check prefix long match */
                if (MEM_read64(matchl0) == MEM_read64(ip)) {
                    PREFETCH_MATCH_L(ip1+8);
                    PREFETCH_MATCH_L(ip1+9);
                    PREFETCH_MATCH_L(ip1+10);
                    PREFETCH_MATCH_L(ip1+11);
                    mLength = AOCL_ZSTD_count(ip+8, matchl0+8, iend) + 8;
                    offset = (U32)(ip-matchl0);
                    while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; } /* catch up */
                    goto _match_found;
                }
            }

            idxl1 = hashLong[hl1];
            matchl1 = base + idxl1;

            if (idxs0 > prefixLowestIndex) {
                /* check prefix short match */
                if (MEM_read32(matchs0) == MEM_read32(ip)) {
                    PREFETCH_MATCH_S(ip+4);
                    goto _search_next_long;
                }
            }

            if (ip1 >= nextStep) {
#if defined(AOCL_ZSTD_SEARCH_SKIP_OPT) && !defined(AOCL_ZSTD_SEARCH_SKIP_LESS_OPT)
                step += 3;
                LOG_FORMATTED(DEBUG, logCtx, "step = %zu", step);
#else
                PREFETCH_L1(ip1 + 64);
                PREFETCH_L1(ip1 + 128);
                step++;
#endif /* AOCL_ZSTD_SEARCH_SKIP_OPT */
                nextStep += kStepIncr;
            }
            ip = ip1;
            ip1 += step;

            hl0 = hl1;
            idxl0 = idxl1;
            matchl0 = matchl1;
    #if defined(__aarch64__)
            PREFETCH_L1(ip+256);
    #endif
        } while (ip1 <= ilimit);

_cleanup:
        /* If offset_1 started invalid (offsetSaved1 != 0) and became valid (offset_1 != 0),
         * rotate saved offsets. See comment in ZSTD_compressBlock_fast_noDict for more context. */
        offsetSaved2 = ((offsetSaved1 != 0) && (offset_1 != 0)) ? offsetSaved1 : offsetSaved2;

        /* save reps for next block */
        rep[0] = offset_1 ? offset_1 : offsetSaved1;
        rep[1] = offset_2 ? offset_2 : offsetSaved2;

        /* Return the last literals size */
        return (size_t)(iend - anchor);

_search_next_long:

        /* check prefix long +1 match */
        if (idxl1 > prefixLowestIndex) {
            if (UNLIKELY(MEM_read64(matchl1) == MEM_read64(ip1))) {
                ip = ip1;
                mLength = AOCL_ZSTD_count(ip+8, matchl1+8, iend) + 8;
                offset = (U32)(ip-matchl1);
                while (((ip>anchor) & (matchl1>prefixLowest)) && (ip[-1] == matchl1[-1])) { ip--; matchl1--; mLength++; } /* catch up */
                goto _match_found;
            }
        }

        /* if no long +1 match, explore the short match we found */
        mLength = AOCL_ZSTD_count(ip+4, matchs0+4, iend) + 4;
        offset = (U32)(ip - matchs0);
        while (((ip>anchor) & (matchs0>prefixLowest)) && (ip[-1] == matchs0[-1])) { ip--; matchs0--; mLength++; } /* catch up */

        /* fall-through */

_match_found: /* requires ip, offset, mLength */
        offset_2 = offset_1;
        offset_1 = offset;

        if (step < 4) {
            /* It is unsafe to write this value back to the hashtable when ip1 is
             * greater than or equal to the new ip we will have after we're done
             * processing this match. Rather than perform that test directly
             * (ip1 >= ip + mLength), which costs speed in practice, we do a simpler
             * more predictable test. The minmatch even if we take a short match is
             * 4 bytes, so as long as step, the distance between ip and ip1
             * (initially) is less than 4, we know ip1 < new ip. */
            hashLong[hl1] = (U32)(ip1 - base);
        }

        ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

_match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
#ifdef AOCL_ZSTD_SEARCH_SKIP_OPT
                /* More complementary insertions to improve ratio */
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-3, hBitsL, 8)] = (U32)(ip-3-base);
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashLong[ZSTD_hashPtr(ip-1, hBitsL, 8)] = (U32)(ip-1-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-4, hBitsS, mls)] = (U32)(ip-4-base);
                hashSmall[ZSTD_hashPtr(ip-3, hBitsS, mls)] = (U32)(ip-3-base);
                hashSmall[ZSTD_hashPtr(ip-2, hBitsS, mls)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
#else
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
#endif
            }

            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                    & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = AOCL_ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
                U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff;  /* swap offset_2 <=> offset_1 */
                hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = (U32)(ip-base);
                hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = (U32)(ip-base);
                ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, rLength);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
            }
        }
    }
}
#endif /* AOCL_DECOMPRESS_FAST != 2 */

#if AOCL_DECOMPRESS_FAST == 2 /* FDS */
#define AOCL_ZSTD_GEN_DFAST_NODICT_FN(mls)                                                                              \
    static size_t AOCL_ZSTD_compressBlock_doubleFast_noDict_##mls(                                                      \
        ZSTD_MatchState_t *ms, SeqStore_t *seqStore, U32 rep[ZSTD_REP_NUM],                                             \
        void const *src, size_t srcSize)                                                                                \
    {                                                                                                                   \
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_generic_fds2_base(ms, seqStore, rep, src, srcSize, mls);       \
    }
#elif AOCL_DECOMPRESS_FAST > 2 /* dynamic FDS */
/* Runs FDS_ALL_CONF when FDS is supported. 
 * Occurance of long match at short offset triggers switch to FDS_NONE mode as skipping these can significantly worsen
 * decompression speed. Note: This switch is possible only in singlePass mode.
 * Compression ratio is analyzed every 'minBytesPerFrame' bytes. If ratio is < AOCL_RATIO_LOW, switch to FDS_NONE occurs
 * from next block/frame. 
 * < AOCL_RATIO_LOW : Ratio compromise from FDS is high for low ratio inputs. Hence avoided. */
#define AOCL_RATIO_LOW 10 /* Lower bound of ratio groups. Below this disable FDS. */
#define AOCL_ZSTD_GEN_DFAST_NODICT_FN(mls)                                                                              \
    static size_t AOCL_ZSTD_compressBlock_doubleFast_noDict_##mls(                                                      \
        ZSTD_MatchState_t *ms, SeqStore_t *seqStore, U32 rep[ZSTD_REP_NUM],                                             \
        void const *src, size_t srcSize)                                                                                \
    {                                                                                                                   \
        if (seqStore->fds_config.state == FDS_UNSUPPORTED) /* No analysis or transitions. Run in non-FDS mode. */       \
            return AOCL_ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, srcSize, mls);             \
        /* Data aware compression */                                                                                    \
        seqStore->fds_config.transition = AOCL_ZSTD_fds_trans_none;                                                     \
        size_t ret = 0;                                                                                                 \
        U64 cur_state = seqStore->fds_config.state;                                                                     \
        if (seqStore->singlePass) {                                                                                     \
            switch(cur_state)                                                                                           \
            {                                                                                                           \
            case FDS_ALL_CONF:                                                                                          \
            {                                                                                                           \
                ret = AOCL_ZSTD_compressBlock_doubleFast_noDict_generic_fds3_base_with_trans(ms, seqStore, rep, src, srcSize, mls);  \
                if (cur_state != seqStore->fds_config.state) /* Transition occurred in Current block */                 \
                    seqStore->fds_config.transition = AOCL_ZSTD_fds_trans_curr;                                         \
                break;                                                                                                  \
            }                                                                                                           \
            default:                                                                                                    \
                ret = AOCL_ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, srcSize, mls);          \
                break;                                                                                                  \
            }                                                                                                           \
        } else {                                                                                                        \
            switch(seqStore->fds_config.state)                                                                          \
            {                                                                                                           \
            case FDS_ALL_CONF:                                                                                          \
                ret = AOCL_ZSTD_compressBlock_doubleFast_noDict_generic_fds3_base(ms, seqStore, rep, src, srcSize, mls); \
                break;                                                                                                  \
            default:                                                                                                    \
                ret = AOCL_ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, srcSize, mls);          \
                break;                                                                                                  \
            }                                                                                                           \
        }                                                                                                               \
        /* Collect and analyze stats */                                                                                 \
        if (!ZSTD_isError(ret)) {                                                                                       \
            size_t ratio;                                                                                               \
            AOCL_DECOMPRESS_FAST_ANALYZE(seqStore, ratio) /* Analyze */                                                 \
            if (ratio != (size_t)(-1)) { /* Consolidate */                                                              \
                if (ratio < AOCL_RATIO_LOW)                                                                             \
                    seqStore->fds_config.state = FDS_NONE;                                                              \
                else                                                                                                    \
                    seqStore->fds_config.state = FDS_ALL_CONF;                                                          \
                if (cur_state != seqStore->fds_config.state)                                                            \
                    seqStore->fds_config.transition = AOCL_ZSTD_fds_trans_next; /* recommend state transition in next block */ \
            }                                                                                                           \
        }                                                                                                               \
        return ret;                                                                                                     \
    }
#else  /* no FDS */
#define AOCL_ZSTD_GEN_DFAST_NODICT_FN(mls)                                                                              \
    static size_t AOCL_ZSTD_compressBlock_doubleFast_noDict_##mls(                                                      \
        ZSTD_MatchState_t *ms, SeqStore_t *seqStore, U32 rep[ZSTD_REP_NUM],                                             \
        void const *src, size_t srcSize)                                                                                \
    {                                                                                                                   \
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, srcSize, mls);                 \
    }
#endif /* AOCL_DECOMPRESS_FAST > 1 */

AOCL_ZSTD_GEN_DFAST_NODICT_FN(4)
AOCL_ZSTD_GEN_DFAST_NODICT_FN(5)
AOCL_ZSTD_GEN_DFAST_NODICT_FN(6)
AOCL_ZSTD_GEN_DFAST_NODICT_FN(7)

size_t AOCL_ZSTD_compressBlock_doubleFast(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    const U32 mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return AOCL_ZSTD_compressBlock_doubleFast_noDict_7(ms, seqStore, rep, src, srcSize);
    }
}
#endif /* AOCL_ZSTD_OPT */

#endif /* ZSTD_EXCLUDE_DFAST_BLOCK_COMPRESSOR */
