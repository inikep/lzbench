#include <csc_mf.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <csc_model.h>

#if defined(__amd64__) || defined(_M_AMD64) || defined(__i386__) || defined(_M_IX86)
#  include <emmintrin.h>
#  define PREFETCH_T0(addr) _mm_prefetch(((char *)(addr)),_MM_HINT_T0)
#elif defined(__GNUC__) && ((__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 2)))
#  define PREFETCH_T0(addr) __builtin_prefetch(addr)
#else
#  define PREFETCH_T0(addr)
#endif


/*
#define HASH6(a, bits) (((*(uint32_t*)(a)^(*(uint16_t*)((a)+4)<<13))*2654435761u)>>(32-(bits)))
#define HASH3(a) ((*(a)<<8)^(*((a)+1)<<5)^(*((a)+2)))
#define HASH2(a) ((*(uint16_t*)(a) * 2654435761u) & 0x3FFF)
*/

inline uint32_t HASH2(uint8_t *p) 
{
    uint16_t v = 0;
    memcpy(&v, p, 2);
    return (v * 65521u) & 0x3FFF; 
}

inline uint32_t HASH3(uint8_t *p) 
{
    return (*(p) << 8) ^ (*(p + 1) << 5) ^ *(p + 2);
}

inline uint32_t HASH6(uint8_t *p, uint32_t bits) 
{
    uint32_t v = 0;
    uint16_t v2 = 0;
    memcpy(&v, p, 4);
    memcpy(&v2, p + 4, 2);
    return (((v ^ ((uint32_t)v2 << 13)) * 2654435761u) >> (32 - bits));
}


int MatchFinder::Init(uint8_t *wnd, 
        uint32_t wnd_size,
        uint32_t bt_size, 
        uint32_t bt_bits, 
        uint32_t ht_width,
        uint32_t ht_bits,
        ISzAlloc *alloc
        )
{
    wnd_ = wnd;
    wnd_size_ = wnd_size;
    vld_rge_ = wnd_size_ - MinBlockSize - 4;
    pos_ = vld_rge_;
    bt_pos_ = 0;

    ht_bits_ = ht_bits;
    ht_width_ = ht_width;
    bt_bits_ = bt_bits;
    bt_size_ = bt_size;
    alloc_ = alloc;

    if (!bt_bits_ || !bt_size_) 
        bt_bits_ = bt_size_ = 0;
    if (!ht_bits_ || !ht_width_) 
        ht_bits_ = ht_width_ = 0;

    size_ = HT2_SIZE_ + HT3_SIZE_ + (1 << ht_bits_) * ht_width_;
    if (bt_bits_) {
        size_ += (1 << (uint64_t)bt_bits_);
        size_ += (uint64_t)bt_size_ * 2;
    }

    mfbuf_raw_ = (uint32_t *)alloc_->Alloc(alloc_, sizeof(uint32_t) * size_ + 128);
    if (!mfbuf_raw_)
        return -1;
    mfbuf_ = (uint32_t*)((uint8_t *)mfbuf_raw_ + (64 - ((uint64_t)(mfbuf_raw_) & 0x3F)));
    memset(mfbuf_, 0, size_ * sizeof(uint32_t));

    uint64_t cpos = 0;
    ht2_ = mfbuf_ + cpos;
    cpos += HT2_SIZE_;
    ht3_ = mfbuf_ + cpos;
    cpos += HT3_SIZE_;

    if (ht_bits_ && ht_width_) {
        ht6_ = mfbuf_ + cpos;
        cpos += ht_width_ * (1 << ht_bits_);
    } else {
        ht6_ = NULL;
    }

    if (bt_bits_) {
        bt_head_ = mfbuf_ + cpos;
        cpos += (1 << (uint64_t)bt_bits_);
        bt_nodes_ = mfbuf_ + cpos;
        cpos += bt_size_ * 2;
    } else {
        bt_head_ = NULL;
    }

    return 0;
}

void MatchFinder::normalize()
{
    uint32_t diff = pos_ - vld_rge_ + 1;
    for(uint32_t i = 0; i < size_; i++)
        mfbuf_[i] = mfbuf_[i] > diff? mfbuf_[i] - diff : 0;
    pos_ -= diff;
}

void MatchFinder::Destroy()
{
    alloc_->Free(alloc_, mfbuf_raw_);
}

void MatchFinder::SetArg(int bt_cyc, int ht_cyc, int ht_low, int good_len)
{
    bt_cyc_ = bt_cyc;
    ht_cyc_ = ht_cyc;
    ht_low_ = ht_low;
    good_len_ = good_len;
}

void bug()
{
    printf("!");
}

void MatchFinder::SlidePos(uint32_t wnd_pos, uint32_t len, uint32_t limit)
{
    uint32_t h2, h3, hbt, h6, lasth6 = 0;
    for(uint32_t i = 1; i < len; ) {
        uint32_t wpos = wnd_pos + i;
        if (pos_ >= 0xFFFFFFF0) normalize();
        h2 = HASH2(wnd_ + wpos);
        h3 = HASH3(wnd_ + wpos);
        ht2_[h2] = pos_;
        ht3_[h3] = pos_;

        if (i + 128 < len) {i += 4; pos_ += 4; bt_pos_ += 4; continue;}

        if (ht_width_) {
            h6 = HASH6(wnd_ + wpos, ht_bits_);
            uint32_t *ht6 = ht6_ + h6 * ht_width_;
            if (h6 != lasth6) {
                uint32_t cands = MIN(ht_width_, ht_cyc_);
                for(uint32_t j = cands - 1; j > 0; j--)
                    ht6[j] = ht6[j - 1];
            }
            ht6[0] = pos_;
            lasth6 = h6;
        }

        if (!bt_head_) { pos_++; i++; continue; }
        hbt = HASH6(wnd_ + wpos, bt_bits_);
        if (bt_pos_ >= bt_size_) bt_pos_ -= bt_size_;
        uint32_t dist = pos_ - bt_head_[hbt];
        uint32_t *l = &bt_nodes_[bt_pos_ * 2], *r = &bt_nodes_[bt_pos_ * 2 + 1];
        uint32_t lenl = 0, lenr = 0;

        for(uint32_t cyc = 0; ; cyc++) {
            if (cyc >= bt_cyc_ || dist >= bt_size_ || dist >= vld_rge_) { *l = *r = 0; break; }
            uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
            uint32_t clen = MIN(lenl, lenr);
            uint32_t climit = MIN(limit - i, wnd_size_ - cmp_pos);
            if (clen >= climit) { *l = *r = 0; break; }

            uint32_t bt_npos = bt_pos_ >= dist ? bt_pos_ - dist : bt_pos_ + bt_size_ - dist;
            uint32_t *tlast = &bt_nodes_[bt_npos * 2];
            PREFETCH_T0(tlast);
            uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos ;
            if (pcur[clen] == pmatch[clen]) {
                uint32_t climit2 = MIN(good_len_, climit);
                clen++;
                while(clen < climit2 && pcur[clen] == pmatch[clen])
                    clen++;

                if (clen >= good_len_) {
                    *l = tlast[0]; *r = tlast[1]; break;
                } else if (clen >= climit2) {
                    *l = *r = 0; break;
                }
            }

            if (pmatch[clen] < pcur[clen]) {
                *l = pos_ - dist;
                dist = pos_ - *(l = &tlast[1]);
                lenl = clen;
            } else {
                *r = pos_ - dist;
                dist = pos_ - *(r = &tlast[0]);
                lenr = clen;
            }
        }
        bt_head_[hbt] = pos_;

        bt_pos_++;
        pos_++;
        i++;
    }
}

void MatchFinder::SlidePosFast(uint32_t wnd_pos, uint32_t len)
{
    uint32_t h;
    for(uint32_t i = 0; i < len; ) {
        uint32_t wpos = wnd_pos + i;
        if (pos_ >= 0xFFFFFFF0) normalize();
        h = HASH2(wnd_ + wpos);
        if (h % 16) {
            // for 'BAD' data, only a small subset of data will be test by MF
            i++;
            pos_++;
            if (++bt_pos_ >= bt_size_) bt_pos_ -= bt_size_;
            continue;
        }

        if (ht_width_) {
            h = HASH6(wnd_ + wpos, ht_bits_);
            uint32_t *ht6 = ht6_ + h * ht_width_;
            for(uint32_t i = ht_width_ - 1; i > 0; i--)
                ht6[i] = ht6[i-1];
            ht6[0] = pos_;
        }

        if (bt_head_) { 
            h = HASH6(wnd_ + wpos, bt_bits_);
            bt_nodes_[bt_pos_ * 2] = bt_nodes_[bt_pos_ * 2 + 1] = 0;
            bt_head_[h] = pos_;
            if (++bt_pos_ >= bt_size_) bt_pos_ -= bt_size_;
        }

        i ++;
        pos_ ++;
    }
}

uint32_t MatchFinder::find_match(MFUnit *ret, uint32_t *rep_dist, uint32_t wpos, uint32_t limit)
{
    static const uint32_t bound[] = {0, 0, 64, 1024, 16 * KB, 256 * KB, 4 * MB};
    uint32_t h2 = HASH2(wnd_ + wpos);
    uint32_t h3 = HASH3(wnd_ + wpos);
    uint32_t h6 = 0; 
    uint32_t hbt = 0;
    uint32_t minlen = 1, cnt = 0, dist = 0;
    if (ht_width_) {
        h6 = HASH6(wnd_ + wpos, ht_bits_);
        PREFETCH_T0(ht6_ + h6 * ht_width_);
    }

    if (bt_head_) {
        hbt = HASH6(wnd_ + wpos, bt_bits_);
        PREFETCH_T0(bt_head_ + hbt);
    }

    if (ht_low_) {
        PREFETCH_T0(ht2_ + h2);
        PREFETCH_T0(ht3_ + h3);
    }

    for(uint32_t i = 0; i < 4; i++) {
        if (rep_dist[i] >= vld_rge_) continue;
        uint32_t cmp_pos = wpos >= rep_dist[i] ? wpos - rep_dist[i] : wpos + wnd_size_ - rep_dist[i];
        uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
        uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;
        if (minlen >= climit || pmatch[minlen] != pcur[minlen]) continue;
        /* avoid alignment warning)
        while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
            pmatch += 4; pcur += 4; }
        if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
            pmatch += 2; pcur += 2; }
        if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        */
        while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        uint32_t match_len = (pcur - wnd_) - wpos;
        if (match_len && i == 0) {
            // rep0len1
            ret[cnt].len = 1;
            ret[cnt].dist = 1;
            if (cnt + 2 < MF_CAND_LIMIT)
                cnt++;
        }
        if (match_len > minlen) {
            minlen = match_len;
            ret[cnt].len = match_len;
            ret[cnt].dist = 1 + i;
            if (cnt + 2 < MF_CAND_LIMIT)
                cnt++;
            if (match_len >= good_len_) {
                dist = 0xFFFFFFFF; //disable all further find
                break;
            }
        }
    }

    if (!ht_low_) goto MAIN_MF;

    if (pos_ - ht2_[h2] > dist) for(;;) {
        dist = pos_ - ht2_[h2];
        if (dist >= vld_rge_) break;
        uint32_t cmp_pos = wpos > dist ? wpos - dist : wpos + wnd_size_ - dist;
        uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
        uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;
        if (minlen >= climit || pmatch[minlen] != pcur[minlen]) break;
        /* avoid alignment warning)
        while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
            pmatch += 4; pcur += 4; }
        if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
            pmatch += 2; pcur += 2; }
        if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        */
        while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        uint32_t match_len = (pcur - wnd_) - wpos;
        if (match_len > minlen) {
            minlen = match_len;
            if (match_len <= 6 && dist >= bound[match_len]) break;
            ret[cnt].len = match_len;
            ret[cnt].dist = 4 + dist;
            if (cnt + 2 < MF_CAND_LIMIT)
                cnt++;
            if (match_len >= good_len_) {
                dist = 0xFFFFFFFF; //disable all further find
                break;
            }
        }
        break;
    }

    if (pos_ - ht3_[h3] > dist) for(;;) {
        dist = pos_ - ht3_[h3];
        if (dist >= vld_rge_) break;
        uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
        uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
        uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;
        if (minlen >= climit || pmatch[minlen] != pcur[minlen]) break;
        /* avoid alignment warning)
        while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
            pmatch += 4; pcur += 4; }
        if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
            pmatch += 2; pcur += 2; }
        if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        */
        while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }

        uint32_t match_len = (pcur - wnd_) - wpos;
        if (match_len > minlen) {
            minlen = match_len;
            if (match_len <= 6 && dist >= bound[match_len]) break;
            ret[cnt].len = match_len;
            ret[cnt].dist = 4 + dist;
            if (cnt + 2 < MF_CAND_LIMIT)
                cnt++;
            if (match_len >= good_len_) {
                dist = 0xFFFFFFFF; //disable all further find
                break;
            }
        }
        break;
    }
    ht2_[h2] = pos_;
    ht3_[h3] = pos_;

MAIN_MF:
    if (bt_head_) {
        dist = pos_ - bt_head_[hbt];
        uint32_t *l = &bt_nodes_[bt_pos_ * 2], *r = &bt_nodes_[bt_pos_ * 2 + 1];

        for(; dist >= bt_size_ && dist < vld_rge_; ) {
            // candidate in hash head of binary tree does not have match distance limit
            uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
            uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
            uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;
            if (minlen >= climit || pmatch[minlen] != pcur[minlen]) break;
            /* avoid alignment warning)
            while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
                pmatch += 4; pcur += 4; }
            if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
                pmatch += 2; pcur += 2; }
            if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
            */
            while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }

            uint32_t match_len = (pcur - wnd_) - wpos;
            if (match_len > minlen) {
                minlen = match_len;
                if (match_len <= 6 && dist >= bound[match_len]) break;
                ret[cnt].len = match_len;
                ret[cnt].dist = 4 + dist;
                if (cnt + 2 < MF_CAND_LIMIT)
                    cnt++;
                if (match_len >= good_len_) {
                    dist = 0xFFFFFFFF; //disable all further find
                    break;
                }
            }
            break;
        }

        uint32_t lenl = 0, lenr = 0;
        for(uint32_t cyc = 0; ; cyc++) {
            if (cyc >= bt_cyc_ || dist >= bt_size_ || dist >= vld_rge_) { *l = *r = 0; break; }
            uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
            uint32_t clen = MIN(lenl, lenr);
            uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
            if (clen >= climit) { *l = *r = 0; break; }

            uint32_t bt_npos = bt_pos_ >= dist ? bt_pos_ - dist : bt_pos_ + bt_size_ - dist;
            uint32_t *tlast = &bt_nodes_[bt_npos * 2];
            PREFETCH_T0(tlast);
            uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos ;
            if (pcur[clen] == pmatch[clen]) {
                //uint32_t climit2 = MIN(good_len_, climit);
                uint32_t climit2 = climit;
                clen++;
                while(clen < climit2 && pcur[clen] == pmatch[clen])
                    clen++;

                if (clen > minlen) {
                    minlen = clen;
                    if (clen > 6 || dist < bound[clen]) {
                        ret[cnt].len = clen;
                        ret[cnt].dist = 4 + dist;
                        if (cnt + 2 < MF_CAND_LIMIT) cnt++;
                    }
                }

                if (clen >= good_len_) {
                    *l = tlast[0]; *r = tlast[1]; dist = 0xFFFFFFFF; break;
                } else if (clen >= climit2) {
                    *l = *r = 0; break;
                }
            }

            if (pmatch[clen] < pcur[clen]) {
                *l = pos_ - dist;
                dist = pos_ - *(l = &tlast[1]);
                lenl = clen;
            } else {
                *r = pos_ - dist;
                dist = pos_ - *(r = &tlast[0]);
                lenr = clen;
            }
        }
        bt_head_[hbt] = pos_;
        if (++bt_pos_ >= bt_size_) bt_pos_ -= bt_size_;
    }

    uint32_t *ht6 = ht6_ + h6 * ht_width_;
    uint32_t cands = MIN(ht_width_, ht_cyc_);
    for(uint32_t i = 0; i < cands; i++) {
        if (pos_ - ht6[i] <= dist) continue;
        dist = pos_ - ht6[i];
        if (dist >= vld_rge_) continue;
        uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
        uint32_t climit = MIN(limit, wnd_size_ - cmp_pos);
        uint8_t *pcur = wnd_ + wpos, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;
        if (minlen >= climit || pmatch[minlen] != pcur[minlen]) continue;
        /* avoid alignment warning)
        while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
            pmatch += 4; pcur += 4; }
        if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
            pmatch += 2; pcur += 2; }
        if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        */
        while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }

        uint32_t match_len = (pcur - wnd_) - wpos;
        if (match_len > minlen) {
            minlen = match_len;
            if (match_len <= 6 && dist >= bound[match_len]) continue;
            ret[cnt].len = match_len;
            ret[cnt].dist = 4 + dist;
            if (cnt + 2 < MF_CAND_LIMIT)
                cnt++;
            if (match_len >= good_len_) {
                dist = 0xFFFFFFFF; //disable all further find
                break;
            }
        }
    }

    if (ht_width_) {
        for(uint32_t i = cands - 1; i > 0; i--)
            ht6[i] = ht6[i-1];
        ht6[0] = pos_;
    }

    if (++pos_ >= 0xFFFFFFF0) normalize();
    return cnt;
}

MFUnit MatchFinder::FindMatch(uint32_t *rep_dist, uint32_t wnd_pos, uint32_t limit)
{
    static const uint32_t cof[] = {0, 4, 8, 12};
    mfcand_[0].len = 1;
    mfcand_[0].dist = 0;
    uint32_t n = find_match(mfcand_ + 1, rep_dist, wnd_pos, limit);
    int bestidx = 0;
    for(uint32_t i = 1; i <= n; i++) {
        if (!bestidx) {
            bestidx = i;
            continue;
        }
        MFUnit &u1 = mfcand_[bestidx];
        MFUnit &u2 = mfcand_[i];
        if (u2.len > 1 && (
                (u2.len > u1.len + 3)
                || (u2.len > u1.len && u2.dist <= 4)
                || (u2.len + 2 > u1.len && u2.dist <= 4 && u1.dist > 4)
                || (u2.len >= u1.len //&& u1.dist > 4 
                    && (u2.dist >> cof[u2.len - u1.len]) <= u1.dist)
                || (u2.len < u1.len && u2.len + 2 >= u1.len && u1.dist > 4 
                    && (u1.dist >> cof[u1.len - u2.len]) > u2.dist)
                ))
            bestidx = i;

    }
    return mfcand_[bestidx];
}

bool MatchFinder::TestFind(uint32_t wpos, uint8_t *src, uint32_t limit)
{
    uint32_t dists[9] = {wnd_size_, wnd_size_};
    uint32_t depth = 0;
    uint32_t h = HASH2(src);
    if (h % 16) 
        // for 'BAD' data, only a small subset of data will be test by MF
        return false;

    if (ht_width_) {
        h = HASH6(src, ht_bits_);
        for(uint32_t i = 0; i < ht_width_ && i < 8; i++)
            dists[depth++] = pos_ - ht6_[h * ht_width_];
    } 

    if (bt_head_) {
        h = HASH6(src, bt_bits_);
        dists[depth++] = pos_ - bt_head_[h];
    }

    for(uint32_t i = 0; i < depth; i++) {
        uint32_t dist = dists[i];
        if (dist >= vld_rge_) continue;
        uint32_t cmp_pos = wpos >= dist ? wpos - dist : wpos + wnd_size_ - dist;
        uint32_t climit = MIN(limit, 24);
        climit = MIN(limit, wnd_size_ - cmp_pos);
        uint8_t *pcur = src, *pmatch = wnd_ + cmp_pos, *pend = pmatch + climit;

        /* avoid alignment warning)
        while(pmatch + 4 <= pend && *(uint32_t *)pcur == *(uint32_t *)pmatch) {
            pmatch += 4; pcur += 4; }
        if (pmatch + 2 <= pend && *(uint16_t *)pcur == *(uint16_t *)pmatch) {
            pmatch += 2; pcur += 2; }
        if (pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }
        */
        while(pmatch < pend && *pcur == *pmatch) { pmatch++; pcur++; }

        if (pcur - src > 18)
            // enough long match
            return true;
    }
    return false;
}

bool MatchFinder::SecondMatchBetter(MFUnit u1, MFUnit u2)
{
    static const uint32_t cof[] = {0, 4, 8, 12};
    return (u2.len > 1 && (
                (u2.len > u1.len + 3)
                || (u2.len > u1.len && u2.dist <= 4)
                || (u2.len + 2 > u1.len && u2.dist <= 4 && u1.dist > 4)
                || (u2.len >= u1.len //&& u1.dist > 4 
                    && (u2.dist >> cof[u2.len - u1.len]) <= u1.dist)
                || (u2.len < u1.len && u2.len + 2 >= u1.len && u1.dist > 4 
                    && (u1.dist >> cof[u1.len - u2.len]) > u2.dist)
                ));
}

void MatchFinder::FindMatchWithPrice(Model *model, uint32_t state, MFUnit *ret, uint32_t *rep_dist, uint32_t wnd_pos, uint32_t limit)
{
    static const uint32_t bound[] = {0, 0, 64, 1024, 16 * KB, 256 * KB, 4 * MB};
    mfcand_[0].len = 1;
    mfcand_[0].dist = 0;

    // ret[0] is the longest match
    // ret[1 .. n] are price tables by match length as index
    uint32_t n = find_match(mfcand_ + 1, rep_dist, wnd_pos, limit);
    ret[0] = mfcand_[n];

    if (ret[0].len >= good_len_)
        return;

    ret[1].dist = 0;
    uint32_t lpos = 1;
    for(uint32_t i = 1; i <= n; i++) {
        uint32_t distprice = 0;
        uint32_t rdist = 0;
        if (mfcand_[i].len == 1 && mfcand_[i].dist == 1) {
            ret[1].price = model->GetRep0Len1Price(state);
            ret[1].dist = 1;
            continue;
        } else if (mfcand_[i].dist <= 4) {
            distprice = model->GetRepDistPrice(state, mfcand_[i].dist - 1);
            rdist = 0;
        } else {
            distprice = model->GetMatchDistPrice(state, mfcand_[i].dist - 5);
            rdist = mfcand_[i].dist - 4;
        }

        while(lpos < mfcand_[i].len) {
            lpos++;
            if (lpos <= 6 && rdist >= bound[lpos]) {
                ret[lpos].dist = 0;
                continue;
            }
            ret[lpos].dist = mfcand_[i].dist;
            ret[lpos].price = distprice + model->GetMatchLenPrice(state, lpos - 2);
        }
    }
}


