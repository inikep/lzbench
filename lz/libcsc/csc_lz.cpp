#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <csc_common.h>
#include <csc_coder.h>
#include <csc_model.h>
#include <csc_lz.h>

int LZ::Init(const CSCProps *p, Model *model, ISzAlloc *alloc)
{
    model_ = model;
    alloc_ = alloc;

    wnd_size_ = p->dict_size;

    if(wnd_size_ < MinDictSize) wnd_size_ = MinDictSize;
    if(wnd_size_ > MaxDictSize) wnd_size_ = MaxDictSize;

    wnd_ = (uint8_t*)alloc_->Alloc(alloc_, wnd_size_ + 8);
    if (!wnd_)
        goto FREE_ON_ERROR;

    if (mf_.Init(wnd_, wnd_size_, p->bt_size, p->bt_hash_bits, 
                p->hash_width, p->hash_bits, alloc_))
        goto FREE_ON_ERROR;

    good_len_ = p->good_len;
    bt_cyc_ = p->bt_cyc;
    ht_cyc_ = p->hash_width;
    mf_.SetArg(bt_cyc_, ht_cyc_, 1, good_len_);
    appt_ = (MFUnit *)alloc_->Alloc(alloc_, sizeof(MFUnit) * good_len_ + 1);

    Reset();
    return 0;

FREE_ON_ERROR:
    alloc_->Free(alloc_, wnd_);
    return -1;
}


void LZ::Reset(void)
{
    wnd_curpos_=0;
    rep_dist_[0] =
        rep_dist_[1] =
        rep_dist_[2] =
        rep_dist_[3] = wnd_size_;
    memset(wnd_, 0, wnd_size_ + 8);
    model_->Reset();
}

void LZ::Destroy(void)
{
    mf_.Destroy();
    alloc_->Free(alloc_, wnd_);
    alloc_->Free(alloc_, appt_);
}

void LZ::EncodeNormal(uint8_t *src, uint32_t size, uint32_t lz_mode)
{
    for(uint32_t i = 0; i < size; ) {
        uint32_t cur_block_size;
        cur_block_size = MIN(wnd_size_ - wnd_curpos_, size - i);
        cur_block_size = MIN(cur_block_size, MinBlockSize);
        memcpy(wnd_ + wnd_curpos_, src + i, cur_block_size);
        if (lz_mode == 1) // fast, with no lazy parser
            compress_normal(cur_block_size, false);
        else if (lz_mode == 2)
            compress_normal(cur_block_size, true);
        else if (lz_mode == 3)
            compress_advanced(cur_block_size);
        /*
        else if (lz_mode == 4) {
            mf_.SetArg(1, 2, 0, good_len_);
            compress_normal(cur_block_size, false);
            mf_.SetArg(bt_cyc_, ht_cyc_, 1, good_len_);
            
        }*/ else if (lz_mode == 5) {
            // only copy the block data to window and put it into MF
            // No encoded output 
            mf_.SetArg(1, 1, 0, good_len_);
            compress_mf_skip(cur_block_size);
            mf_.SetArg(bt_cyc_, ht_cyc_, 1, good_len_);
        } else {
            printf("Error!");
            exit(0);
        }

        if (wnd_curpos_ >= wnd_size_) 
            wnd_curpos_ = 0;
        i += cur_block_size;
    }
    if (lz_mode != 5) {
        // for lz_mode == 5, Encode nothing, non terminator
        model_->EncodeMatch(64, 0);
    }
    return;
}

bool LZ::IsDuplicateBlock(uint8_t *src, uint32_t size)
{
    uint32_t mc = 0;
    for(uint32_t i = 0; i < size; i ++) 
        if (mf_.TestFind(wnd_curpos_, src + i, size - i)) {
            mc++;
            if (mc)
                return true;
        }
    return false;
}

void LZ::DuplicateInsert(uint8_t *src,uint32_t size)
{
    for(uint32_t i = 0; i < size; ) {
        uint32_t cur_block_size = MIN(wnd_size_ - wnd_curpos_, size - i);
        cur_block_size = MIN(cur_block_size, MinBlockSize);
        memcpy(wnd_ + wnd_curpos_, src + i, cur_block_size);
        wnd_curpos_ += cur_block_size;
        if (wnd_curpos_ >= wnd_size_) wnd_curpos_=0;
        i += cur_block_size;
    }
    return;
}

void LZ::encode_nonlit(MFUnit u)
{
    if (u.dist <= 4) {
        if (u.len == 1 && u.dist == 1)
            model_->EncodeRep0Len1();
        else  {
            model_->EncodeRepDistMatch(u.dist - 1, u.len - 2);
            uint32_t dist = rep_dist_[u.dist - 1];
            switch (u.dist) {
                case 4:
                    rep_dist_[3] = rep_dist_[2];
                case 3:
                    rep_dist_[2] = rep_dist_[1];
                case 2:
                    rep_dist_[1] = rep_dist_[0];
                case 1:
                    rep_dist_[0] = dist;
                break;
            }
        }
    } else {
        model_->EncodeMatch(u.dist - 5, u.len - 2);
        rep_dist_[3] = rep_dist_[2];
        rep_dist_[2] = rep_dist_[1];
        rep_dist_[1] = rep_dist_[0];
        rep_dist_[0] = u.dist - 4;
    }
}

void LZ::compress_normal(uint32_t size, bool lazy)
{
    MFUnit u1, u2;
    bool got_u1 = false;
    for(uint32_t i = 0; i < size; ) {
        if (!got_u1)
            u1 = mf_.FindMatch(rep_dist_, wnd_curpos_, size - i);

        if (u1.len == 1 || !lazy || u1.len >= good_len_) {
            if (u1.dist == 0)
                model_->EncodeLiteral(wnd_[wnd_curpos_]);//,wnd_curpos_-1
            else
                encode_nonlit(u1);
            mf_.SlidePos(wnd_curpos_, u1.len, size - i);
            i += u1.len; wnd_curpos_ += u1.len;
            if (u1.dist) {
                model_->SetLiteralCtx(wnd_[wnd_curpos_ - 1]);
                //if (u1.dist <= 4 && rep_dist_[u1.dist - 1] < 20)
                //    model_->SetLiteralCtx(wnd_[wnd_curpos_ - rep_dist_[u1.dist - 1]]);
            }
            got_u1 = false;
            continue;
        }

        u2 = mf_.FindMatch(rep_dist_, wnd_curpos_ + 1, size - i - 1);
        if (mf_.SecondMatchBetter(u1, u2)) {
            // choose literal output
            model_->EncodeLiteral(wnd_[wnd_curpos_]);//,wnd_curpos_-1
            mf_.SlidePos(wnd_curpos_, 1, size - i - 1);
            i++; wnd_curpos_++;
            u1 = u2;
            got_u1 = true;
        } else {
            encode_nonlit(u1);
            mf_.SlidePos(wnd_curpos_ + 1, u1.len - 1, size - i - 1);
            i += u1.len; wnd_curpos_ += u1.len;
            model_->SetLiteralCtx(wnd_[wnd_curpos_ - 1]);
                //if (u1.dist <= 4 && rep_dist_[u1.dist - 1] < 20)
                //    model_->SetLiteralCtx(wnd_[wnd_curpos_ - rep_dist_[u1.dist - 1]]);
            got_u1 = false;
        }
    }
    return;
}

void LZ::compress_mf_skip(uint32_t size)
{
    mf_.SlidePosFast(wnd_curpos_, size);
    wnd_curpos_ += size;
}

void LZ::compress_advanced(uint32_t size)
{
    uint32_t apend = 0, apcur = 0;

    for(uint32_t i = 0; i < size; ) {
        mf_.FindMatchWithPrice(model_, model_->state_, 
                appt_, rep_dist_, wnd_curpos_, size - i);
        if (appt_[0].dist == 0) {
            model_->EncodeLiteral(wnd_[wnd_curpos_]);
            mf_.SlidePos(wnd_curpos_, 1, size - i);
            i++; wnd_curpos_++;
        } else {
            apcur = 0;
            apend = 1;
            apunits_[0].price = 0;
            apunits_[0].back_pos = 0;
            apunits_[0].rep_dist[0] = rep_dist_[0];
            apunits_[0].rep_dist[1] = rep_dist_[1];
            apunits_[0].rep_dist[2] = rep_dist_[2];
            apunits_[0].rep_dist[3] = rep_dist_[3];
            apunits_[0].state = model_->state_;
            uint32_t aplimit = MIN(AP_LIMIT, size - i);
            for(;;) {
                apunits_[apcur].lit = wnd_[wnd_curpos_];
                // fix cur state
                if (apcur) {
                    int l = apunits_[apcur].back_pos;
                    apunits_[apcur].rep_dist[0] = apunits_[l].rep_dist[0];
                    apunits_[apcur].rep_dist[1] = apunits_[l].rep_dist[1];
                    apunits_[apcur].rep_dist[2] = apunits_[l].rep_dist[2];
                    apunits_[apcur].rep_dist[3] = apunits_[l].rep_dist[3];
                    if (apunits_[apcur].dist == 0) {
                        apunits_[apcur].state = (apunits_[l].state * 4) & 0x3F;
                    } else if (apunits_[apcur].dist <= 4) {
                        uint32_t len = apcur - l;
                        if (len == 1 && apunits_[apcur].dist == 1)
                            apunits_[apcur].state = (apunits_[l].state * 4 + 2) & 0x3F;
                        else {
                            apunits_[apcur].state = (apunits_[l].state * 4 + 3) & 0x3F;
                            uint32_t tmp = apunits_[apcur].rep_dist[apunits_[apcur].dist - 1];
                            switch (apunits_[apcur].dist) {
                                case 4:
                                    apunits_[apcur].rep_dist[3] = apunits_[apcur].rep_dist[2];
                                case 3:
                                    apunits_[apcur].rep_dist[2] = apunits_[apcur].rep_dist[1];
                                case 2:
                                    apunits_[apcur].rep_dist[1] = apunits_[apcur].rep_dist[0];
                                    apunits_[apcur].rep_dist[0] = tmp;
                                    break;
                            }
                       }
                    } else {
                        apunits_[apcur].state = (apunits_[l].state * 4 + 1) & 0x3F;
                        apunits_[apcur].rep_dist[0] = apunits_[apcur].dist- 4;
                        apunits_[apcur].rep_dist[1] = apunits_[l].rep_dist[0];
                        apunits_[apcur].rep_dist[2] = apunits_[l].rep_dist[1];
                        apunits_[apcur].rep_dist[3] = apunits_[l].rep_dist[2];
                    }

                    if (apcur < aplimit)
                        mf_.FindMatchWithPrice(model_, apunits_[apcur].state, 
                            appt_, apunits_[apcur].rep_dist, wnd_curpos_, size - i - apcur);
                }

                if (apcur == aplimit) {
                    ap_backward(apcur);
                    i += apcur;
                    break;
                }

                if (appt_[0].len == 1 && apcur + 1 == apend) {
                    ap_backward(apcur);
                    model_->EncodeLiteral(apunits_[apcur].lit);
                    i += apcur;
                    mf_.SlidePos(wnd_curpos_, 1, size - i);
                    wnd_curpos_ ++;
                    i++;
                    break;
                }

                if (apcur + 1 >= apend) 
                    apunits_[apend++].price = 0xFFFFFFFF;

                if (appt_[0].len >= good_len_ || (appt_[0].len > 1 && appt_[0].len + apcur >= aplimit)) {
                    ap_backward(apcur);
                    i += apcur;
                    encode_nonlit(appt_[0]);
                    mf_.SlidePos(wnd_curpos_, appt_[0].len, size - i);
                    i += appt_[0].len;
                    wnd_curpos_ += appt_[0].len;
                    model_->SetLiteralCtx(wnd_[wnd_curpos_ - 1]);
                    break;
                }
 
                uint32_t lit_ctx = wnd_curpos_? wnd_[wnd_curpos_ - 1] : 0;
                uint32_t cprice = model_->GetLiteralPrice(apunits_[apcur].state, lit_ctx, wnd_[wnd_curpos_]);
                if (cprice + apunits_[apcur].price < apunits_[apcur + 1].price) {
                    apunits_[apcur + 1].dist = 0;
                    apunits_[apcur + 1].back_pos = apcur;
                    apunits_[apcur + 1].price = cprice + apunits_[apcur].price;
                }

                if (appt_[1].dist && appt_[1].price + apunits_[apcur].price < apunits_[apcur + 1].price) {
                    apunits_[apcur + 1].dist = 1;
                    apunits_[apcur + 1].back_pos = apcur;
                    apunits_[apcur + 1].price = appt_[1].price + apunits_[apcur].price;
                }
                   
                uint32_t len = appt_[0].len;
                while (apcur + len >= apend) 
                    apunits_[apend++].price = 0xFFFFFFFF;

                while(len > 1) {
                    if (appt_[len].dist && appt_[len].price + apunits_[apcur].price < apunits_[apcur + len].price) {
                        apunits_[apcur + len].dist = appt_[len].dist;
                        apunits_[apcur + len].back_pos = apcur;
                        apunits_[apcur + len].price = appt_[len].price + apunits_[apcur].price;
                    }
                    len--;
                }
                apcur++;
                mf_.SlidePos(wnd_curpos_, 1, size - i - apcur);
                wnd_curpos_++;
            }
        }
    }
}

void LZ::ap_backward(int end)
{
    for(int i = end; i; ) {
        apunits_[apunits_[i].back_pos].next_pos = i;
        i = apunits_[i].back_pos;
    }

    for(int i = 0; i != end; ) {
        uint32_t next = apunits_[i].next_pos;
        if (apunits_[next].dist == 0) {
            model_->EncodeLiteral(apunits_[i].lit);
        } else if (apunits_[next].dist <= 4) {
            if (next - i == 1 && apunits_[next].dist == 1)
                model_->EncodeRep0Len1();
            else  
                model_->EncodeRepDistMatch(apunits_[next].dist - 1, next - i - 2);
            model_->SetLiteralCtx(apunits_[next - 1].lit);
        } else {
            model_->EncodeMatch(apunits_[next].dist - 5, next - i - 2);
            model_->SetLiteralCtx(apunits_[next - 1].lit);
        }
        i = next;
    }
    rep_dist_[0] = apunits_[end].rep_dist[0];
    rep_dist_[1] = apunits_[end].rep_dist[1];
    rep_dist_[2] = apunits_[end].rep_dist[2];
    rep_dist_[3] = apunits_[end].rep_dist[3];
}


