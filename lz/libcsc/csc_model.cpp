#include <csc_model.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <csc_profiler.h>

/* 

slot     extra_bits      dist_range
  0      0 bits         0 -         0
  1      0 bits         1 -         1
  2      0 bits         2 -         2
  3      1 bits         3 -         4
  4      2 bits         5 -         8
  5      3 bits         9 -        16
  6      4 bits        17 -        32
  7      5 bits        33 -        64
  8      6 bits        65 -       128
  9      7 bits       129 -       256
 10      8 bits       257 -       512
 11      9 bits       513 -      1024
 12     10 bits      1025 -      2048
 13     11 bits      2049 -      4096
 14     12 bits      4097 -      8192
 15     13 bits      8193 -     16384
 16     14 bits     16385 -     32768
 17     15 bits     32769 -     65536
 18     16 bits     65537 -    131072
 19     17 bits    131073 -    262144
 20     18 bits    262145 -    524288
 21     19 bits    524289 -   1048576
 22     20 bits   1048577 -   2097152
 23     21 bits   2097153 -   4194304
 24     22 bits   4194305 -   8388608
 25     23 bits   8388609 -  16777216
 26     24 bits  16777217 -  33554432
 27     25 bits  33554433 -  67108864
 28     26 bits  67108865 - 134217728
 29     27 bits 134217729 - 268435456
 30     28 bits 268435457 - 536870912
 31     29 bits 536870913 - 1073741824

*/

const uint32_t Model::dist_table_[33] = {
    0,      1,      2,      3,
    5,      9,      17,     33,
    65,     129,    257,    513,
    1025,   2049,   4097,   8193,
    16385,  32769,  65537,  131073,
    262145, 524289, 1048577,        2097153,
    4194305,        8388609,        16777217,       33554433,
    67108865,       134217729,      268435457,      536870913,
    1073741825,
};

const uint32_t Model::rev16_table_[16] = {
    0,      8,      4,      12,
    2,      10,     6,      14,
    1,      9,      5,      13,
    3,      11,     7,      15,
};

int Model::Init(Coder *coder, ISzAlloc *alloc)
{
    coder_ = coder;
    alloc_ = alloc;
    for (int i = 0; i < (4096 >> 3); i++)
        p_2_bits_[i]= (uint32_t)(128 * 
                log((float)(i * 8 + 4) / 4096) / log(0.5));

    p_delta_=NULL;
    p_lit_ = (uint32_t*)alloc_->Alloc(alloc_, 256 * 256 * sizeof(uint32_t));
    if (!p_lit_)
        return -1;

    return 0;
}

void Model::Destroy()
{
    alloc_->Free(alloc_, p_lit_);
    alloc_->Free(alloc_, p_delta_);
    PWriteLog();
}


void Model::Reset(void)
{
    alloc_->Free(alloc_, p_delta_);
    p_delta_ = NULL;
#define INIT_PROB(P, K) do{for(int i = 0; i < K; i++) P[i] = 2048;}while(0)
    INIT_PROB(p_state_, 64 * 3);
    INIT_PROB(p_lit_, 256 * 256);
    INIT_PROB(p_repdist_, 64 * 3);
    INIT_PROB(p_dist_, 8 + 16 * 2 + 32 * 4);
    INIT_PROB(p_rle_len_, 16);

    INIT_PROB(p_matchlen_slot_, 2);
    INIT_PROB(p_matchlen_extra1_, 8);
    INIT_PROB(p_matchlen_extra2_, 8);
    INIT_PROB(p_matchlen_extra3_, 128);
    INIT_PROB(p_matchdist_extra_, 29 * 16);
#undef INIT_PROB

    p_longlen_ = 2048;
    p_rle_flag_ = 2048;
    state_ = 0;    
    ctx_ = 0;
    lp_rebuild_int_ = 0;
}

void Model::encode_matchlen_1(uint32_t len)
{
    /* 0 - 7, 8 - 15, 16 - 143,  if 143 denoting more than 143 */
    uint32_t *p;
    if (len < 16) {
        if (len < 8) {
            EncodeBit(coder_, 0, p_matchlen_slot_[0]);
            p = p_matchlen_extra1_;
        } else {
            EncodeBit(coder_, 1, p_matchlen_slot_[0]);
            EncodeBit(coder_, 0, p_matchlen_slot_[1]);
            len -= 8;
            p = p_matchlen_extra2_;
        }

        uint32_t c = len | 0x08;
        do {
            EncodeBit(coder_, (c >> 2) & 1, p[c >> 3]);
            c <<= 1;
        } while (c < 0x40);
    } else {
        EncodeBit(coder_, 1, p_matchlen_slot_[0]);
        EncodeBit(coder_, 1, p_matchlen_slot_[1]);
        len -= 16;

        p = p_matchlen_extra3_;
        uint32_t c = len | 0x80;
        do {
            EncodeBit(coder_, (c >> 6) & 1, p[c >> 7]);
            c <<= 1;
        } while (c < 0x4000);
    }
}

void Model::encode_matchlen_2(uint32_t len) 
{
    if (len >= 143) {
        encode_matchlen_1(143);
        len -= 143;
        while(len >= 143) {
            len -= 143;
            EncodeBit(coder_, 0, p_longlen_);
        }
        EncodeBit(coder_, 1, p_longlen_);
    }
    encode_matchlen_1(len);
}

#define FEncodeBit(price,v,p) do\
{\
	if (v)\
		price += p_2_bits_[p>>3];\
	else\
		price += p_2_bits_[(4096-p)>>3];\
}while(0)

void Model::EncodeLiteral(uint32_t c)//,uint32_t pos
{
    PEncodeLiteral(c);
    //printf("[%u](%u)\n", c, ctx_);
    EncodeBit(coder_, 0, p_state_[state_ * 3 + 0]);

    state_ = (state_ * 4) & 0x3F;
    uint32_t *p = &p_lit_[ctx_ * 256];
    ctx_ = (c >> 0);
    c = c | 0x100;
    do {
        EncodeBit(coder_,(c >> 7) & 1,p[c>>8]);
        c <<= 1;
    } while (c < 0x10000);
}

uint32_t Model::GetLiteralPrice(uint32_t fstate, uint32_t fctx, uint32_t c)
{
	uint32_t ret = 0;
    FEncodeBit(ret, 0, p_state_[fstate * 3 + 0]);
    uint32_t *p = &p_lit_[fctx * 256];
    c = c | 0x100;
    do {
        FEncodeBit(ret, (c >> 7) & 1, p[c>>8]);
        c <<= 1;
    } while (c < 0x10000);
    return ret;
}

void Model::EncodeRep0Len1(void)
{
    PEncode1BMatch();
    //printf("Rep0Len1\n");
    EncodeBit(coder_, 1, p_state_[state_ *3 + 0]);
    EncodeBit(coder_, 0, p_state_[state_ *3 + 1]);
    EncodeBit(coder_, 0, p_state_[state_ *3 + 2]);
    ctx_ = 0;
    state_ = (state_ * 4 + 2) & 0x3F;
}

uint32_t Model::GetRep0Len1Price(uint32_t fstate)
{
	uint32_t ret = 0;
    FEncodeBit(ret, 1, p_state_[fstate *3 + 0]);
    FEncodeBit(ret, 0, p_state_[fstate *3 + 1]);
    FEncodeBit(ret, 0, p_state_[fstate *3 + 2]);
    return ret;
}

void Model::EncodeRepDistMatch(uint32_t rep_idx, uint32_t match_len)
{
    PEncodeRepMatch(match_len, rep_idx);
    //printf("Rep %u %u\n", rep_idx, match_len);
    EncodeBit(coder_, 1, p_state_[state_ *3 + 0]);
    EncodeBit(coder_, 0, p_state_[state_ *3 + 1]);
    EncodeBit(coder_, 1, p_state_[state_ *3 + 2]);

    uint32_t i = 1, j;
    j = (rep_idx >> 1) & 1; EncodeBit(coder_, j, p_repdist_[state_ * 3 + i - 1]); i += i + j;
    j = rep_idx & 1; EncodeBit(coder_, j, p_repdist_[state_ * 3 + i - 1]); 

    encode_matchlen_2(match_len);
    state_ = (state_ * 4 + 3) & 0x3F;
}

void Model::len_price_rebuild()
{
    for(int i = 0; i < 32; i++) {
        uint32_t ret = 0, len = i;
        uint32_t *p;
        if (len < 16) {
            if (len < 8) {
                FEncodeBit(ret , 0, p_matchlen_slot_[0]);
                p = p_matchlen_extra1_;
            } else {
                FEncodeBit(ret , 1, p_matchlen_slot_[0]);
                FEncodeBit(ret , 0, p_matchlen_slot_[1]);
                len -= 8;
                p = p_matchlen_extra2_;
            }

            uint32_t c = len | 0x08;
            do {
                FEncodeBit(ret, (c >> 2) & 1, p[c >> 3]);
                c <<= 1;
            } while (c < 0x40);
        } else {
            FEncodeBit(ret, 1, p_matchlen_slot_[0]);
            FEncodeBit(ret, 1, p_matchlen_slot_[1]);
            len -= 16;

            p = p_matchlen_extra3_;
            uint32_t c = len | 0x80;
            do {
                FEncodeBit(ret, (c >> 6) & 1, p[c >> 7]);
                c <<= 1;
            } while (c < 0x4000);
        }
        len_price_[i] = ret;
    }
    lp_rebuild_int_ = 4096;
}


uint32_t Model::GetRepDistPrice(uint32_t fstate,uint32_t rep_idx)
{
	uint32_t ret = 0;
    FEncodeBit(ret, 1, p_state_[fstate *3 + 0]);
    FEncodeBit(ret, 0, p_state_[fstate *3 + 1]);
    FEncodeBit(ret, 1, p_state_[fstate *3 + 2]);

    uint32_t i = 1, j;
    j = (rep_idx >> 1) & 1; FEncodeBit(ret, j, p_repdist_[fstate * 3 + i - 1]); i += i + j;
    j = rep_idx & 1; FEncodeBit(ret, j, p_repdist_[fstate * 3 + i - 1]); 
    return ret; //ret > 256  ? ret - 256 : 0;
}

uint32_t Model::GetMatchLenPrice(uint32_t fstate,uint32_t matchLen)
{
    uint32_t ret = 0;
    (void)fstate;
    if (matchLen >= 32)
        //long enough, some random reasonable price 
        ret += 128 * 6; 
    else {
        if (lp_rebuild_int_-- == 0)
            len_price_rebuild();
        ret += len_price_[matchLen];
    }
    return ret;
}

void Model::EncodeMatch(uint32_t dist, uint32_t len)
{
    //printf("%u %u\n", dist, len);
    PEncodeMatch(len, dist);
    EncodeBit(coder_, 1,p_state_[state_ * 3 + 0]);
    EncodeBit(coder_, 1,p_state_[state_ * 3 + 1]);
    encode_matchlen_2(len);
    uint32_t pdist_pos, sbits;
    switch(len) {
        case 0:
            pdist_pos = 0;
            sbits = 3;
            break;
        case 1:
        case 2:
            pdist_pos = 16 * (len - 1) + 8; 
            sbits = 4;
            break;
        case 3:
        case 4:
        case 5:
            pdist_pos = 32 * (len - 3) + 8 + 16 * 2; 
            sbits = 5;
            break;
        default:
            pdist_pos = 32 * 3 + 8 + 16 * 2; 
            sbits = 5;
            break;
    };

    uint32_t l = 0, r = 32;
    while(l + 1 < r) {
        uint32_t mid = (l + (r - l) / 2);
        if (dist_table_[mid] > dist) 
            r = mid;
        else if (dist_table_[mid] < dist) 
            l = mid;
        else 
            l = r = mid;
    }

    uint32_t slot = l, c = slot | (1 << sbits);
    uint32_t extra_len = 0, extra_bits = slot > 2? slot - 2 : 0;
    uint32_t *p = p_dist_ + pdist_pos;
    do {
        EncodeBit(coder_,(c >> (sbits - 1)) & 1, p[c >> sbits]);
        c <<= 1;
    } while (c < (1u << (sbits * 2)));

    if (extra_bits) {
        extra_len = dist - (1 << extra_bits) - 1;
        if (extra_bits > 4) {
            uint32_t dlen = (extra_len >> 4);
            uint32_t bits = (extra_bits - 4);
            EncodeDirect(coder_, dlen, bits);
        }
        c = rev16_table_[extra_len & 0x0F] | 0x10;
        p = &p_matchdist_extra_[(extra_bits - 1) * 16];
        do {
            EncodeBit(coder_,(c >> 3) & 1, p[c >> 4]);
            c <<= 1;
        } while (c < (1 << 8));
    }

    state_ = (state_ * 4 + 1) & 0x3F;
}

uint32_t Model::GetMatchDistPrice(uint32_t fstate, uint32_t dist)
{
    uint32_t ret = 0;
    FEncodeBit(ret, 1,p_state_[fstate * 3 + 0]);
    FEncodeBit(ret, 1,p_state_[fstate * 3 + 1]);

    // quick estimation, 4bit for slot + extrabits
    uint32_t l = 0, r = 32;
    while(l + 1 < r) {
        uint32_t mid = (l + (r - l) / 2);
        if (dist_table_[mid] > dist) 
            r = mid;
        else if (dist_table_[mid] < dist) 
            l = mid;
        else 
            l = r = mid;
    }
    ret += (l > 2? l + 2 : 2) * 128;
    return ret;
}

void Model::EncodeInt(uint32_t num)
{
    /* slot, extra_bits, range_end_close
       0 - 1  1
       1 - 1  3
       2 - 2  7
       3 - 3  15
       ...
       31 - 31 2^32 - 1
    */

    uint32_t tmp = num;
    uint32_t slot = 0;
    while(tmp) {
        tmp >>= 1;
        slot++;
    }
    if (slot) 
        slot--;

    EncodeDirect(coder_, slot, 5);
    if (slot == 0)
        EncodeDirect(coder_, num, 1);
    else
        EncodeDirect(coder_, num - (1 << slot), slot);
}

void Model::CompressDelta(uint8_t *src,uint32_t size)
{
    uint32_t i;
    uint32_t *p;
    uint32_t c;
    uint32_t sCtx=0;

    if (p_delta_ == NULL) {
        p_delta_ = (uint32_t*)alloc_->Alloc(alloc_, 256*256*4);
        for (i = 0; i < 256 * 256; i++) {
            p_delta_[i] = 2048;
        }
    }


    EncodeInt(size);
    for(i=0;i<size;i++)
    {
        p=&p_delta_[sCtx*256];
        c=src[i]|0x100;
        do
        {
            EncodeBit(coder_,(c >> 7) & 1,p[c>>8]);
            c <<= 1;
        }
        while (c < 0x10000);

        sCtx=src[i];
    }
    return;
}

void Model::CompressLiterals(uint8_t *src,uint32_t size)
{
    EncodeInt(size);
    for(uint32_t i = 0; i < size; i++) {
        uint32_t c  = src[i];
        uint32_t *p = &p_lit_[ctx_ * 256];
        ctx_ = (c >> 0);
        c = c | 0x100;
        do {
            EncodeBit(coder_,(c >> 7) & 1,p[c>>8]);
            c <<= 1;
        } while (c < 0x10000);
    }
}

void Model::CompressBad(uint8_t *src,uint32_t size)
{
    EncodeInt(size);
    for(uint32_t i = 0; i< size; i++)
        coder_->EncDirect16(src[i],8);
    return;
}

void Model::CompressRLE(uint8_t *src, uint32_t size)
{
    uint32_t i,j,c,len;
    uint32_t sCtx=0;
    uint32_t *p=NULL;

    EncodeInt(size);
    if (p_delta_==NULL) {
        p_delta_=(uint32_t*)alloc_->Alloc(alloc_, 256*256*4);
        for (i=0;i<256*256;i++)
            p_delta_[i]=2048;
    }

    for (i = 0; i < size;) {
        if (i > 0 && size - i > 3 
                && src[i - 1] == src[i] 
                && src[i] == src[i + 1]
                && src[i] == src[i + 2]) {
            j = i + 3;
            len = 3;
            while(j < size && src[j] == src[j-1]) { len++; j++; }

            if (len > 10) {
                sCtx = src[j-1];
                len -= 11;
                EncodeBit(coder_,1,p_rle_flag_);
                encode_matchlen_2(len);
                i=j;
                continue;
            }
        }
        EncodeBit(coder_,0,p_rle_flag_);
        p=&p_delta_[sCtx*256];
        c=src[i]|0x100;
        do {
            EncodeBit(coder_,(c >> 7) & 1,p[c>>8]);
            c <<= 1;
        } while (c < 0x10000);
        sCtx = src[i];
        i++;
    }
    return;
}


