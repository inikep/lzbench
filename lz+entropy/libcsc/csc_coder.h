#ifndef _CSC_CODER_H_
#define _CSC_CODER_H_
#include <csc_memio.h>
#include <csc_typedef.h>
#include <stdio.h>

#define BCWCheckBound() do{if (bc_size_ == bc_bufsize_) \
        {\
            outsize_ += bc_size_;\
            if (io_->WriteBCData(bc_buf_, bc_bufsize_) != (int)bc_bufsize_) {\
                throw (int)WRITE_ERROR; \
            } \
            bc_size_ = 0;\
            pbc_ = bc_buf_;\
        }}while(0)

class Coder
{
    MemIO *io_;
    ISzAlloc *alloc_;

public:

    int Init(MemIO *io, ISzAlloc *alloc);
    void Destroy();

    //flush the coder
    void Flush();    

    //Get coded length
    uint32_t GetCodedLength(void) {
        return bc_size_ + rc_size_;
    }

    void EncDirect16(uint32_t val,uint32_t len);

    void RC_ShiftLow(void);

    //the main buffer for encoding/decoding range/bit coder
    uint8_t *rc_buf_;
    uint8_t *bc_buf_;        

    //indicates the full size of buffer range/bit coder
    uint32_t rc_bufsize_;
    uint32_t bc_bufsize_;                

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
    uint32_t bnum_;
};


#define EncodeBit(coder,v,p) do {\
    uint32_t newBound = (coder->rc_range_ >> 12) * p;\
    if (v) {\
        coder->rc_range_ = newBound;\
        p += (0xFFF - p) >> 5;\
    } else {\
        coder->rc_low_ += newBound;\
        coder->rc_range_ -= newBound;\
        p -= p >> 5;\
    }\
    \
    if (coder->rc_range_ < (1 << 24)) {\
        coder->rc_range_ <<= 8;\
        coder->RC_ShiftLow();\
    }}while(0)

#define EncodeDirect(x,v,l) do{if ((l) <= 16)\
        x->EncDirect16(v,l);\
    else {\
        x->EncDirect16((v) >> 16,(l) - 16);\
        x->EncDirect16((v) & 0xFFFF,16);\
    }}while(0)

#endif

