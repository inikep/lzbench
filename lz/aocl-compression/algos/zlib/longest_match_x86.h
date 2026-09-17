/**
 * Copyright (C) 2023-2024, Advanced Micro Devices. All rights reserved.
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

// This header file is an template for avx and above function multiversion, application should not use it directly
/* LONGEST_MATCH_AVX_FAMILY optimized for smaller hash chain lengths */
TARGET_ISA_ATTRIBUTE
ZLIB_INTERNAL uint32_t LONGEST_MATCH_AVX_FAMILY(deflate_state* s, IPos cur_match)
{
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    unsigned chain_length = s->max_chain_length;
    register Bytef *scan = s->window + s->strstart;
    int len;
    int best_len = s->prev_length ? s->prev_length : MIN_MATCH-1;
    int nice_match = s->nice_match;
    int offset = 0;
    uInt lookahead = s->lookahead;
    uInt strstart = s->strstart;
    /* match candidates beyond MAX_DIST distance not allowed in deflate */
    IPos limit = s->strstart > (IPos)MAX_DIST(s) ?
        s->strstart - (IPos)MAX_DIST(s) : NIL;
    /* exit from match finding if match lengths are not meaningful */
    int exit = s->level < 5;       

    register Bytef *match_start = s->window;
    register Bytef *match_end;

    Posf *prev = s->prev;
    uInt wmask = s->w_mask;

    unsigned char scan_st[4];
    unsigned char scan_end[4];

    offset = best_len - 1;

    if(best_len >= sizeof(uint32_t))
        offset -= 2;

    memcpy(scan_st, scan, sizeof(uint32_t));
    memcpy(scan_end, scan + offset, sizeof(uint32_t));

    match_end = (match_start + offset);

    /* Do not waste too much time if we already have a good match: */
    if (best_len >= s->good_match) {
        chain_length >>= 2;
    }

    Assert((ulg)s->strstart <= s->window_size-MIN_LOOKAHEAD, "need lookahead");

#define NEXT_CANDIDATE \
    if ((--chain_length != 0) && ((cur_match = prev[cur_match & wmask]) > limit)) continue; \
    return best_len;

    for(;;) {
        /* break out when match candidate move out of search buffer */
        if(cur_match >= strstart)
            break;

        if (best_len < sizeof(uint32_t)) {
            /* compare 2 bytes */
            for (;;) {
                if (aocl_compare_2b(match_end + cur_match, scan_end) == 0 &&
                    aocl_compare_2b(match_start + cur_match, scan_st) == 0) break;
                NEXT_CANDIDATE;
            }
        } else {
            /* compare 4 bytes */
            for (;;) {
                if (aocl_compare_4b(match_end + cur_match, scan_end) == 0 &&
                    aocl_compare_4b(match_start + cur_match, scan_st) == 0) break;
                NEXT_CANDIDATE;
            }
        }
        /* scan is not updated in COMPARE256 , so no need to reset it at every iteration 
           no need to match first two bytes as they are already matched above */
        len = COMPARE256(scan + 2, match_start + cur_match + 2) + 2;

        Assert(scan + len <= s->window+(unsigned)(s->window_size-1), "wild scan");

        if (len > best_len) {
            /* new string is longer than previous - remember it */
            s->match_start = cur_match;
            best_len = len;
            if (best_len > lookahead) return lookahead;
            if (best_len >= nice_match) return best_len;

            offset = best_len - 1;

            if(best_len >= sizeof(uint32_t))
                offset -= 2;

            memcpy(scan_end, scan + offset, sizeof(uint32_t));
            match_end = (match_start + offset);
        } 
        else if (UNLIKELY(exit)) {
            /* We exit early for lower levels as finding matches further is unlikely */
            break;
        }
        NEXT_CANDIDATE;
    }
    return best_len;
}

/* LONGEST_MATCH_LAZY_AVX_FAMILY optimized for larger hash chain lengths */
TARGET_ISA_ATTRIBUTE
ZLIB_INTERNAL uint32_t LONGEST_MATCH_LAZY_AVX_FAMILY(deflate_state* s, IPos cur_match)
{
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    unsigned chain_length = s->max_chain_length;/* max hash chain length */
    register Bytef *scan = s->window + s->strstart; /* current string */
    register int len;                           /* length of current match */
    int best_len = s->prev_length ? s->prev_length : MIN_MATCH-1;   /* ignore strings, shorter or of the same length */
    int nice_match = s->nice_match;             /* stop if match long enough */
    int offset = 0;
    uInt lookahead = s->lookahead;
    uInt strstart = s->strstart;
    ush match_offset = 0;

    IPos limit_base = s->strstart > (IPos)MAX_DIST(s) ?
        s->strstart - (IPos)MAX_DIST(s) : NIL;
    /*?? are MAX_DIST matches allowed ?! */
    IPos limit = limit_base;                    /* limit will be limit_base+offset */
    /* Stop when cur_match becomes <= limit. To simplify the code,
     * we prevent matches with the string of window index 0.
     */
    register Bytef *match_base_start = s->window;
    register Bytef *match_base_end;

    Posf *prev = s->prev;                       /* lists of the hash chains */
    uInt wmask = s->w_mask;

    unsigned char scan_st[8];
    unsigned char scan_end[8];

    offset = best_len - 1;

    if(best_len >= sizeof(uint32_t)) {
        offset -= 2;
        if(best_len >= sizeof(uint64_t)) {
            offset -= 4;
        }
    }

    memcpy(scan_st, scan, sizeof(uint64_t));
    memcpy(scan_end, scan + offset, sizeof(uint64_t));

    match_base_end = (match_base_start + offset);

    /* Do not waste too much time if we already have a good match: */
    if (best_len >= s->good_match) {
        chain_length >>= 2;
    }

    Assert((ulg)s->strstart <= s->window_size-MIN_LOOKAHEAD, "need lookahead");

    if (best_len >= AOCL_MIN_MATCH) {
        /* We're continuing search (lazy evaluation).*/
        register int i;
        IPos pos;
        register uInt hash = 0;
        /* Find a most distant chain starting from scan with index=1 (index=0 corresponds
         * to cur_match). Note: we cannot use s->prev[strstart+1,...] immediately, because
         * these strings are not yet inserted into hash table yet.
         */
        for (i = 0; i <= best_len - AOCL_MIN_MATCH; i++) {
            UPDATE_HASH_MUL(s, hash, scan[i]);
            /* If we're starting with best_len >= AOCL_MIN_MATCH, we can use offset search. */
            pos = s->head[hash];
            if (pos < cur_match) {
                match_offset = (ush)i;
                cur_match = pos;
            }
        }
        /* update variables to correspond match_offset */
        limit = limit_base + match_offset;
        if (cur_match <= limit) goto break_matching;
        match_base_start -= match_offset;
        match_base_end -= match_offset;
    }

#define NEXT_CANDIDATE \
    if ((--chain_length != 0) && ((cur_match = prev[cur_match & wmask]) > limit)) continue; \
    return best_len;

    for(;;) {
        if(cur_match >= strstart)
            break;
        /* Find a candidate for matching using hash table. Jump over hash
         * table chain until we'll have a partial march. Doing "break" when
         * matched, and NEXT_CHAIN to try different place.
         */
        if (best_len < sizeof(uint32_t)) {
            /* compare 2 bytes */
            for (;;) {
                if (aocl_compare_2b(match_base_end + cur_match, scan_end) == 0 &&
                    aocl_compare_2b(match_base_start + cur_match, scan_st) == 0) break;
                NEXT_CANDIDATE;
            }
        } else if(best_len >= sizeof(uint64_t)) {
            /* compare 8 bytes */
            for (;;) {
                if (aocl_compare_8b(match_base_end + cur_match, scan_end) == 0 &&
                    aocl_compare_8b(match_base_start + cur_match, scan_st) == 0) break;
                NEXT_CANDIDATE;
            }
        } else {
            /* compare 4 bytes */
            for (;;) {
                if (aocl_compare_4b(match_base_end + cur_match, scan_end) == 0 &&
                    aocl_compare_4b(match_base_start + cur_match, scan_st) == 0) break;
                NEXT_CANDIDATE;
            }
        }
        /* scan is not updated in COMPARE256 , so no need to reset it at every iteration 
           no need to match first two bytes as they are already matched above */
        len = COMPARE256(scan + 2, match_base_start + cur_match + 2) + 2;

        Assert(scan + len <= s->window+(unsigned)(s->window_size-1), "wild scan");

        if (len > best_len) {
            /* new string is longer than previous - remember it */
            s->match_start = cur_match - match_offset;
            best_len = len;
            if (best_len > lookahead) return lookahead;
            if (best_len >= nice_match) return best_len;

            offset = best_len - 1;
            if(best_len >= sizeof(uint32_t)) {
                offset -= 2;
                if(best_len >= sizeof(uint64_t)) {
                    offset -= 4;
                }
            }
            memcpy(scan_end, scan + offset, sizeof(uint64_t));
            /* look for better string offset */
            if (UNLIKELY(len > MIN_MATCH && cur_match - match_offset + len < strstart)) {
                IPos pos, next_pos;
                register int i;
                register uInt hash;
                Bytef* scan_end0;

                /* go back to offset 0 */
                cur_match -= match_offset;
                match_offset = 0;
                next_pos = cur_match;
                for (i = 0; i <= len - MIN_MATCH; i++) {
                    pos = prev[(cur_match + i) & wmask];
                    if (pos < next_pos) {
                        /* this hash chain is more distant, use it */
                        if (pos <= limit_base + i) goto break_matching;
                        next_pos = pos;
                        match_offset = (ush)i;
                    }
                }
                /* Switch cur_match to next_pos chain */
                cur_match = next_pos;

                /* Try hash head at len-(MIN_MATCH) position to see if we could get
                 * a better cur_match at the end of string. Using (MIN_MATCH) lets
                 * us to include one more byte into hash - the byte which will be checked
                 * in main loop now, and which allows to grow match by 1.
                 */
                hash = 0;
                scan_end0 = scan + len - MIN_MATCH;
                UPDATE_HASH_MUL(s, hash, scan_end0[0]);
                pos = s->head[hash];
                if (pos < cur_match) {
                    match_offset = (ush)(len - MIN_MATCH);
                    if (pos <= limit_base + match_offset) goto break_matching;
                    cur_match = pos;
                }

                /* update offset-dependent vars */
                limit = limit_base + match_offset;
                match_base_start = s->window - match_offset;
                match_base_end = (match_base_start + offset);
                continue;
            } 
            match_base_end = (match_base_start + offset);
        }
        NEXT_CANDIDATE;
    }
    return best_len;

break_matching: /* sorry for goto's, but such code is smaller and easier to view ... */
    if ((uInt)best_len <= s->lookahead) return (uInt)best_len;
    return s->lookahead;
}

#undef LONGEST_MATCH_AVX_FAMILY
#undef LONGEST_MATCH_LAZY_AVX_FAMILY
#undef COMPARE256
#undef TARGET_ISA_ATTRIBUTE
