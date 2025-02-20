#include <csc_dec.h>
#include <csc_memio.h>
#include <csc_filters.h>
#include <csc_typedef.h>
#include <csc_default_alloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define DecodeBit(coder, v, p) do{\
    if (coder->rc_range_<(1<<24)) {\
        coder->rc_range_<<=8;\
        coder->rc_code_=(coder->rc_code_<<8)+*coder->prc_++;\
        coder->rc_size_++;\
        if (coder->rc_size_>=coder->rc_bufsize_) {\
            coder->outsize_+=coder->rc_size_;\
            if (coder->io_->ReadRCData(coder->rc_buf_,coder->rc_bufsize_) < 0)\
                throw (int)READ_ERROR;\
            coder->rc_size_=0;\
            coder->prc_ = coder->rc_buf_;\
        }\
    }\
    \
    uint32_t bound = (coder->rc_range_ >> 12) * p;\
    if (coder->rc_code_ < bound) {\
        coder->rc_range_ = bound;\
        p += (0xFFF-p) >> 5;\
        v = v + v + 1;\
    } else {\
        coder->rc_range_ -= bound;\
        coder->rc_code_ -= bound;\
        p -= p >> 5;\
        v = v + v;\
    }\
} while(0)

#define DecodeDirect(x,v,l) do{if ((l) <= 16)\
        v = x->coder_decode_direct(l);\
    else {\
        v = (x->coder_decode_direct((l) - 16) << 16);\
        v = v | x->coder_decode_direct(16);\
    }}while (0)

static const uint32_t dist_table_[33] = {
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

static const uint32_t rev16_table_[16] = {
    0,      8,      4,      12,
    2,      10,     6,      14,
    1,      9,      5,      13,
    3,      11,     7,      15,
};

class CSCDecoder
{
    uint32_t coder_decode_direct(uint32_t len) {

#define BCRCheckBound() do{if (bc_size_ >= bc_bufsize_) \
        {\
            outsize_ += bc_size_;\
            if (io_->ReadBCData(bc_buf_, bc_bufsize_) < 0)\
                throw (int)READ_ERROR;\
            bc_size_ = 0;\
            pbc_ = bc_buf_;\
        }}while(0)

        uint32_t result;
        while(bc_curbits_ < len) {
            bc_curval_ = (bc_curval_ << 8) | *pbc_;
            pbc_++;
            bc_size_++;
            BCRCheckBound();
            bc_curbits_ += 8;
        }
        result = (bc_curval_ >> (bc_curbits_ - len)) & ((1 << len) - 1);
        bc_curbits_ -= len;
        return result;
    }

    uint32_t decode_int() {
        uint32_t slot, num;
        DecodeDirect(this, slot, 5);
        DecodeDirect(this, num, slot == 0? 1 : slot);
        if (slot)
            num += (1 << slot);
        return num;
    }

    int decode_bad(uint8_t *dst, uint32_t *size, uint32_t max_bsize) {
        *size = decode_int();
        if (*size > max_bsize) {
            return -1;
        }

        for(uint32_t i = 0; i < *size; i++) {
            dst[i] = coder_decode_direct(8);
        }
        return 0;
    }

    int decode_rle(uint8_t *dst, uint32_t *size, uint32_t max_bsize) {
        uint32_t c, flag, len, i;
        uint32_t sCtx = 0;
        uint32_t *p=NULL;

        if (p_delta_ == NULL) {
            p_delta_ = (uint32_t*)alloc_->Alloc(alloc_, 256*256*sizeof(uint32_t));
            for (i = 0;i< 256 * 256;i++)
                p_delta_[i]=2048;
        }

        *size = decode_int();
        if (*size > max_bsize) {
            return -1;
        }

        for (i = 0; i < *size; ) {
            flag=0;
            DecodeBit(this, flag, p_rle_flag_);
            c = 1;
            if (flag == 0) {
                p = &p_delta_[sCtx*256];
                do  { 
                    DecodeBit(this, c, p[c]);
                } while (c < 0x100);
                dst[i] = c & 0xFF;
                sCtx = dst[i];
                i++;
            } else {
                len = decode_matchlen_2() + 11;
                if (i == 0) {
                    // invalid compression stream
                    return -1;
                }

                while(len-- > 0 && i < *size) {
                    dst[i] = dst[i-1];
                    i++;
                }
                sCtx = dst[i-1];
            }
        }
        return 0;
    }

    uint32_t decode_literal() {
        uint32_t i = 1, *p;
        //uint32_t lctx = ctx_;
        p = &p_lit_[ctx_ * 256];
        do { 
            DecodeBit(this, i, p[i]);
        } while (i < 0x100);

        ctx_ = i & 0xFF;
        state_ = (state_ * 4 + 0) & 0x3F;
        //printf("[%u](%u)\n", ctx_, lctx);
        return ctx_;
    }

    int decode_literals(uint8_t *dst, uint32_t *size, uint32_t max_bsize) {
        *size = decode_int();
        if (*size > max_bsize) {
            return -1;
        }

        for(uint32_t i = 0; i < *size; i++) {
            uint32_t c = 1, *p;
            p = &p_lit_[ctx_ * 256];
            do { 
                DecodeBit(this, c, p[c]);
            } while (c < 0x100);
            ctx_ = c & 0xFF;
            dst[i] = ctx_;
        }
        return 0;
    }

    uint32_t decode_matchlen_1() {
        uint32_t v = 0, lenbase;
        uint32_t *p;
        DecodeBit(this, v, p_matchlen_slot_[0]);
        if (v == 0) {
            p = p_matchlen_extra1_;
            lenbase = 0;
        } else {
            v = 0;
            DecodeBit(this, v, p_matchlen_slot_[1]);
            if (v == 0) {
                p = p_matchlen_extra2_;
                lenbase = 8;
            } else {
                p = p_matchlen_extra3_;
                lenbase = 16;
            }
        }

        uint32_t i = 1;
        if (lenbase == 16) {  // decode 7 bits
            do { 
                DecodeBit(this, i, p[i]);
            } while (i < 0x80);
            return lenbase + (i & 0x7F);
        } else { // decode 3 bits
            do { 
                DecodeBit(this, i, p[i]);
            } while (i < 0x08);
            return lenbase + (i & 0x07);
        }
    }

    uint32_t decode_matchlen_2() {
        uint32_t len = decode_matchlen_1();
        if (len == 143) {
            uint32_t v = 0;
            for(;; v = 0, len += 143) {
                DecodeBit(this, v, p_longlen_);
                if (v) 
                    break;
            }
            return len + decode_matchlen_1();
        } else 
            return len;
    }

    void decode_match(uint32_t &dist, uint32_t &len) {
        len = decode_matchlen_2();
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
        uint32_t *p = p_dist_ + pdist_pos;
        uint32_t slot, i = 1;
        do { 
            DecodeBit(this, i, p[i]);
        } while (i < (1u << sbits));
        slot = i & ((1 << sbits) - 1);
        if (slot <= 2)
            dist = slot;
        else {
            uint32_t ebits = slot - 2;
            uint32_t elen = 0;
            if (ebits > 4) 
                DecodeDirect(this, elen, ebits - 4);
            i = 1;
            p = &p_matchdist_extra_[(ebits- 1) * 16];
            do { 
                DecodeBit(this, i, p[i]);
            } while (i < 0x10);
            dist = dist_table_[slot] + (elen << 4) + rev16_table_[i & 0x0F];
        }
        state_ = (state_ * 4 + 1) & 0x3F;
        //printf("%u %u\n", dist, len);
        return;
    }

    void set_lit_ctx(uint32_t c) {
        ctx_ = (c >> 0);
    }

    void decode_1byte_match(void) {
        state_ = (state_ * 4 + 2) & 0x3F;
        //printf("Rep0Len1\n");
        ctx_ = 0;
    }

    void decode_repdist_match(uint32_t &rep_idx, uint32_t &match_len) {
        uint32_t i = 1;
        do { 
            DecodeBit(this, i, p_repdist_[state_ * 3 + i - 1]);
        } while (i < 0x4);
        rep_idx = i & 0x3;
        match_len = decode_matchlen_2();
        state_ = (state_ * 4 + 3) & 0x3F;
        //printf("Rep %u %u\n", rep_idx, match_len);
    }

    int lz_decode(uint8_t *dst, uint32_t *size, uint32_t limit);
    void lz_copy2dict(uint8_t *src, uint32_t size);


public:
    int Init(MemIO *io, uint32_t dict_size, uint32_t csc_blocksize, ISzAlloc *alloc) {
        io_ = io;
        alloc_ = alloc;

        // entropy coder init
        rc_low_ = 0;
        rc_range_ = 0xFFFFFFFF;
        rc_cachesize_ = 1;
        rc_cache_ = 0;
        rc_code_=0;

        rc_size_ = bc_size_ = 0;
        bc_curbits_ = bc_curval_ =0;

        outsize_ = 0;

        rc_buf_ = bc_buf_ = NULL;
        p_delta_ = NULL;
        wnd_ = NULL;
        p_lit_ = NULL;
        filters_ = NULL;

        prc_ = rc_buf_ = (uint8_t *)alloc_->Alloc(alloc_, csc_blocksize);
        pbc_ = bc_buf_ = (uint8_t *)alloc_->Alloc(alloc_, csc_blocksize);
        if (!prc_ || !pbc_)
            goto FREE_ON_ERROR;

        if (io_->ReadRCData(rc_buf_, rc_bufsize_) < 0 ||
            io_->ReadBCData(bc_buf_, bc_bufsize_) < 0)
            goto FREE_ON_ERROR;

        rc_code_ = ((uint32_t)prc_[1] << 24) 
            | ((uint32_t)prc_[2] << 16) 
            | ((uint32_t)prc_[3] << 8) 
            | ((uint32_t)prc_[4]);
        prc_ += 5;
        rc_size_ += 5;

        // model
        p_lit_ = (uint32_t*)alloc_->Alloc(alloc_, 256 * 256 * sizeof(uint32_t));
        if (!p_lit_)
            goto FREE_ON_ERROR;

        filters_ = (Filters *)alloc_->Alloc(alloc_, sizeof(Filters));
        filters_->Init(alloc_);

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

        // LZ engine
        wnd_size_ = dict_size;
        wnd_ = (uint8_t *)alloc_->Alloc(alloc_, wnd_size_ + 8);
        if (!wnd_)
            goto FREE_ON_ERROR;

        wnd_curpos_=0;
        rep_dist_[0]=
            rep_dist_[1]=
            rep_dist_[2]=
            rep_dist_[3]=0;
        return 0;

FREE_ON_ERROR:
        Destroy();
        return -1;
    }

    void Destroy()
    {
        alloc_->Free(alloc_, p_lit_);
        alloc_->Free(alloc_, p_delta_);
        alloc_->Free(alloc_, wnd_);
        alloc_->Free(alloc_, rc_buf_);
        alloc_->Free(alloc_, bc_buf_);
        if (filters_) {
            filters_->Destroy();
            alloc_->Free(alloc_, filters_);
        }
        p_lit_ = p_delta_ = NULL;
        wnd_ = NULL;
        rc_buf_ = bc_buf_ = NULL;
        filters_ = NULL;
    }

    int Decompress(uint8_t *dst, uint32_t *size, uint32_t max_bsize);
    uint64_t GetCompressedSize() {
        return outsize_ + rc_size_ + bc_size_;
    }

private:
    ISzAlloc *alloc_;
    MemIO *io_;
    Filters *filters_;

    // For entropy coder
    uint8_t *rc_buf_;
    uint8_t *bc_buf_;        

    //indicates the full size of buffer range/bit coder
    uint32_t rc_bufsize_;
    uint32_t bc_bufsize_;                

    //identify it's a encoder/decoder
    uint32_t m_op;        

    uint64_t rc_low_,rc_cachesize_;
    uint32_t rc_range_,rc_code_;
    uint8_t rc_cache_;

    // for bit coder
    uint32_t bc_curbits_;
    uint32_t bc_curval_;    

    //the i/o pointer of range coder and bit coder
    uint8_t *prc_;
    uint8_t *pbc_;    

    //byte counts of output bytes by range coder and bit coder
    uint32_t bc_size_;
    uint32_t rc_size_;
    int64_t outsize_;

    // For model
    uint32_t p_rle_flag_;
    uint32_t p_rle_len_[16];

    // prob of literals
    uint32_t *p_lit_;
    uint32_t *p_delta_;

    uint32_t p_repdist_[64 * 4];
    uint32_t p_dist_[8 + 16 * 2 + 32 * 4];  //len 0 , < 64
    uint32_t p_matchdist_extra_[29 * 16];

    uint32_t p_matchlen_slot_[2];
    uint32_t p_matchlen_extra1_[8];
    uint32_t p_matchlen_extra2_[8];
    uint32_t p_matchlen_extra3_[128];

    uint32_t p_longlen_;
    uint32_t ctx_;
    uint32_t p_state_[4*4*4*3];//Original [64][3]
    uint32_t state_;

    // For LZ Engine
    uint32_t rep_dist_[4];
    uint32_t wnd_size_;
    uint8_t  *wnd_;
    uint32_t wnd_curpos_;
};

int CSCDecoder::lz_decode(uint8_t *dst, uint32_t *size, uint32_t limit)
{
    uint32_t copied_size = 0;
    uint32_t copied_wndpos = wnd_curpos_;
    uint32_t i;

    for(i = 0; i <= limit; ) {
        uint32_t v = 0;
        DecodeBit(this, v, p_state_[state_ *3 + 0]);
        if (v == 0) {
            wnd_[wnd_curpos_++] = decode_literal();
            i++;
        } else {
            v = 0;
            DecodeBit(this ,v, p_state_[state_ * 3 + 1]);

            uint32_t dist, len, cpy_pos;
            uint8_t *cpy_src ,*cpy_dst;
            if (v == 1) {
                decode_match(dist, len);
                if (len == 0 && dist == 64) {
                    // End of a block
                    break;
                }
                dist++;
                len += 2;
                rep_dist_[3] = rep_dist_[2];
                rep_dist_[2] = rep_dist_[1];
                rep_dist_[1] = rep_dist_[0];
                rep_dist_[0] = dist;
                cpy_pos = wnd_curpos_ >= dist? 
                    wnd_curpos_ - dist : wnd_curpos_ + wnd_size_ - dist;
                if (cpy_pos >= wnd_size_ || cpy_pos + len > wnd_size_ ||
                        len + i > limit || wnd_curpos_ + len > wnd_size_)
                    throw (int)DECODE_ERROR;

                cpy_dst = wnd_ + wnd_curpos_;
                cpy_src = wnd_ + cpy_pos;
                i += len;
                wnd_curpos_ += len;
                while(len--) {
                    *cpy_dst++ = *cpy_src++;
                }
                set_lit_ctx(wnd_[wnd_curpos_ - 1]);
            } else {
                v = 0;
                DecodeBit(this , v, p_state_[state_ * 3 + 2]);
                if (v == 0) {
                    decode_1byte_match();
                    cpy_pos = wnd_curpos_ > rep_dist_[0]?
                        wnd_curpos_ - rep_dist_[0] : wnd_curpos_ + wnd_size_ - rep_dist_[0];
                    wnd_[wnd_curpos_++] = wnd_[cpy_pos];
                    i++;
                    set_lit_ctx(wnd_[wnd_curpos_-1]);
                } else {
                    uint32_t repdist_idx;
                    decode_repdist_match(repdist_idx, len);
                    len += 2;
                    if (len + i > limit) {
                        throw (int)DECODE_ERROR;
                    }

                    dist = rep_dist_[repdist_idx];
                    for(int j = repdist_idx ; j > 0; j--) 
                        rep_dist_[j] = rep_dist_[j-1];
                    rep_dist_[0] = dist;

                    cpy_pos = wnd_curpos_ >= dist? 
                        wnd_curpos_ - dist : wnd_curpos_ + wnd_size_ - dist;
                    if (cpy_pos >= wnd_size_ || cpy_pos + len > wnd_size_ ||
                            len + i > limit || wnd_curpos_ + len > wnd_size_) 
                        throw (int)DECODE_ERROR;
                    cpy_dst = wnd_ + wnd_curpos_;
                    cpy_src = wnd_ + cpy_pos;
                    i += len;
                    wnd_curpos_ += len;
                    while(len--) 
                        *cpy_dst++ = *cpy_src++;
                    set_lit_ctx(wnd_[wnd_curpos_ - 1]);
                }
            }
        }

        if (wnd_curpos_ > wnd_size_) {
            throw (int)DECODE_ERROR;
        } else if (wnd_curpos_ == wnd_size_) {
            wnd_curpos_ = 0;
            memcpy(dst + copied_size ,wnd_ + copied_wndpos, i - copied_size);
            copied_wndpos = 0;
            copied_size = i;
        }
    }
    *size = i;
    memcpy(dst + copied_size ,wnd_ + copied_wndpos, *size - copied_size);
    return 0;
}

void CSCDecoder::lz_copy2dict(uint8_t *src, uint32_t size)
{
    for(uint32_t i = 0; i < size; ) {
        uint32_t cur_block;
        cur_block = MIN(wnd_size_ - wnd_curpos_, size - i);
        cur_block = MIN(cur_block , MinBlockSize);
        memcpy(wnd_ + wnd_curpos_, src + i, cur_block);
        wnd_curpos_ += cur_block;
        wnd_curpos_ = wnd_curpos_ >= wnd_size_? 0 : wnd_curpos_;
        i += cur_block;
    }
}

int CSCDecoder::Decompress(uint8_t *dst, uint32_t *size, uint32_t max_bsize)
{
    int ret = 0;
    uint32_t type = decode_int();
    switch (type) {
        case DT_NORMAL:
            ret = lz_decode(dst, size, max_bsize);
            if (ret < 0)
                return ret;
            break;
        case DT_EXE:
            ret = lz_decode(dst, size, max_bsize);
            if (ret < 0)
                return ret;
            filters_->Inverse_E89(dst, *size);
            break;
        case DT_ENGTXT:
            *size = decode_int();
            ret = lz_decode(dst, size, max_bsize);
            if (ret < 0) {
                return ret;
            }
            filters_->Inverse_Dict(dst, *size);
            break;
        case DT_BAD:
            ret = decode_bad(dst, size, max_bsize);
            if (ret < 0) {
                return ret;
            }
            lz_copy2dict(dst, *size);
            break;
        case DT_ENTROPY:
            ret = decode_literals(dst, size, max_bsize);
            if (ret < 0) {
                return ret;
            }
            lz_copy2dict(dst, *size);
            break;
        //case DT_HARD:
        //    m_model.DecompressHard(dst,size);
        //    m_lz.DuplicateInsert(dst,*size);
        //    break;
        /*case DT_AUDIO:
            m_model.DecompressHard(dst,size);
            m_filters.Inverse_Audio(dst,*size,typeArg1,typeArg2);
            m_lz.DuplicateInsert(dst,*size);
            break;
        case DT_RGB:
            typeArg1=m_model.DecodeInt(16);
            typeArg2=m_model.DecodeInt(6);
            m_model.DecompressHard(dst,size);
            m_filters.Inverse_RGB(dst,*size,typeArg1,typeArg2);
            m_lz.DuplicateInsert(dst,*size);
            break;*/
        case SIG_EOF:
            *size=0;
            break;
        default:
            if (type >= DT_DLT && type < DT_DLT + DLT_CHANNEL_MAX) {
                uint32_t chnNum = DltIndex[type - DT_DLT];
                ret = decode_rle(dst, size, max_bsize);
                if (ret<0) {
                    return ret;
                }
                filters_->Inverse_Delta(dst, *size, chnNum);
                lz_copy2dict(dst, *size);
            } else {
                throw (int)DECODE_ERROR;
            }
            break;
    }
    if (decode_int() == 1) {
        // entropy coder init
        rc_low_ = 0;
        rc_range_ = 0xFFFFFFFF;
        rc_cachesize_ = 1;
        rc_cache_ = 0;
        rc_code_=0;
        outsize_ += bc_size_ + rc_size_;
        rc_size_ = bc_size_ = 0;
        bc_curbits_ = bc_curval_ =0;
        prc_ = rc_buf_;
        pbc_ = bc_buf_;

        if (io_->ReadRCData(rc_buf_, rc_bufsize_) < 0 ||
            io_->ReadBCData(bc_buf_, bc_bufsize_) < 0)
            return -1;

        rc_code_ = ((uint32_t)prc_[1] << 24) 
            | ((uint32_t)prc_[2] << 16) 
            | ((uint32_t)prc_[3] << 8) 
            | ((uint32_t)prc_[4]);
        prc_ += 5;
        rc_size_ += 5;
    }
    return ret;
}

struct CSCDecInstance
{
    CSCDecoder *decoder;
    MemIO *io;
    ISzAlloc *alloc;
    uint32_t raw_blocksize;
};

CSCDecHandle CSCDec_Create(const CSCProps *props,
                           ISeqInStream *instream,
                           ISzAlloc *alloc)
{
    if (alloc == NULL) {
        alloc = default_alloc;
    }

    if (props->dict_size > 1024 * MB) {
        return NULL;
    }

    if (props->dict_size < 32 * KB) {
        return NULL;
    }

    CSCDecInstance *csc = (CSCDecInstance *)alloc->Alloc(alloc, sizeof(CSCDecInstance));
    csc->io = (MemIO *)alloc->Alloc(alloc, sizeof(MemIO));
    csc->io->Init(instream, props->csc_blocksize, alloc);
    csc->raw_blocksize = props->raw_blocksize;
    csc->decoder = (CSCDecoder *)alloc->Alloc(alloc, sizeof(CSCDecoder));
    csc->alloc = alloc;

    if (csc->decoder->Init(csc->io, props->dict_size, props->csc_blocksize, alloc) < 0) {
        CSCDec_Destroy((void *)csc);
        return NULL;
    }
    return (void*)csc;
}

void CSCDec_Destroy(CSCDecHandle p)
{
    CSCDecInstance *csc = (CSCDecInstance *)p;
    csc->decoder->Destroy();
    csc->io->Destroy();
    ISzAlloc *alloc = csc->alloc;
    alloc->Free(alloc, csc->decoder);
    alloc->Free(alloc, csc->io);
    alloc->Free(alloc, csc);
}

void CSCDec_ReadProperties(CSCProps *props, uint8_t *s)
{
    props->dict_size = ((uint32_t)s[0] << 24) + (s[1] << 16) + (s[2] << 8) + s[3];
    props->csc_blocksize = ((uint32_t)s[4] << 16) + (s[5] << 8) + s[6];
    props->raw_blocksize = ((uint32_t)s[7] << 16) + (s[8] << 8) + s[9];
}

int CSCDec_Decode(CSCDecHandle p, 
        ISeqOutStream *os,
        ICompressProgress *progress)
{
    int ret = 0;
    CSCDecInstance *csc = (CSCDecInstance *)p;
    uint8_t *buf = (uint8_t *)csc->alloc->Alloc(csc->alloc, csc->raw_blocksize);
    uint64_t outsize = 0;

    for(;;) {
        uint32_t size;
        try {
            // IOCallback error will not be passed by return value.
            // Instead, use exception
            ret = csc->decoder->Decompress(buf, &size, csc->raw_blocksize);
        } catch (int errcode) {
            ret = errcode;
        }
        if (ret == 0)
            outsize += size;

        if (progress)
            progress->Progress(progress, csc->decoder->GetCompressedSize(), outsize);

        if (size == 0 || ret < 0)
            break;

        size_t wrote = os->Write(os, buf, size);
        if (wrote == CSC_WRITE_ABORT)
            break;
        else if (wrote < size) {
            ret = WRITE_ERROR;
            break;
        }
    }
    csc->alloc->Free(csc->alloc, buf);
    return ret;
}


