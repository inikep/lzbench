#include <csc_coder.h>
#include <stdio.h>
#include <stdlib.h>


int Coder::Init(MemIO *io, ISzAlloc *alloc)
{
    alloc_ = alloc;
    rc_low_ = 0;
    rc_range_ = 0xFFFFFFFF;
    rc_cachesize_ = 1;
    rc_cache_ = 0;
    rc_code_=0;

    rc_size_ = bc_size_ = 0;
    bc_curbits_ = bc_curval_ =0;

    outsize_ = 0;

    io_ = io;
    rc_bufsize_ = bc_bufsize_ = io->GetBlockSize();
    prc_ = rc_buf_ = (uint8_t *)alloc_->Alloc(alloc_, rc_bufsize_);
    pbc_ = bc_buf_ = (uint8_t *)alloc_->Alloc(alloc_, bc_bufsize_);

    if (rc_buf_ && bc_buf_) {
        return 0;
    } else {
        alloc_->Free(alloc_, rc_buf_);
        alloc_->Free(alloc_, bc_buf_);
        return -1;
    }
}

void Coder::Destroy()
{
    alloc_->Free(alloc_, rc_buf_);
    alloc_->Free(alloc_, bc_buf_);
}

void Coder::Flush()
{
    for (int i=0;i<5;i++) // One more byte for EOF
    {
        RC_ShiftLow();
    }
    prc_++;
    rc_size_++;

    //one more byte for bitcoder is to prevent overflow while reading
    for(int i = 0; i < 2; i++) {
        if (i == 1)
            *pbc_++=0;
        else
            *pbc_++ = (bc_curval_ << (8 - bc_curbits_)) & 0xFF;
        bc_size_++;
        BCWCheckBound();
    }

    outsize_ += rc_size_ + bc_size_;
    if (io_->WriteRCData(rc_buf_, rc_size_) != (int)rc_size_
        || io_->WriteBCData(bc_buf_, bc_size_) != (int)bc_size_) {
        throw (int)WRITE_ERROR;
    }

    rc_low_ = 0;
    rc_range_ = 0xFFFFFFFF;
    rc_cachesize_ = 1;
    rc_cache_ = 0;
    rc_code_=0;
    rc_size_ = bc_size_ = 0;
    bc_curbits_ = bc_curval_ = 0;
    prc_ = rc_buf_;
    pbc_ = bc_buf_;
}

void Coder::EncDirect16(uint32_t val,uint32_t len)
{
    bc_curval_ = (bc_curval_ << len) | val;
    bc_curbits_ += len;
    while(bc_curbits_ >= 8)
    {
        *pbc_++ = (bc_curval_ >> (bc_curbits_ - 8)) & 0xFF;
        bc_size_++;
        BCWCheckBound();
        bc_curbits_ -= 8;
    }
}

void Coder::RC_ShiftLow(void)
{
    uint8_t temp;
    if ((uint32_t)rc_low_ < (uint32_t)0xFF000000 || (int32_t)(rc_low_ >> 32) != 0) {
        temp = rc_cache_;
        do {
            *prc_++ = (uint8_t)(temp + (uint8_t)(rc_low_ >> 32));
            rc_size_++;
            if (rc_size_ == rc_bufsize_) {
                outsize_+=rc_size_;
                if (io_->WriteRCData(rc_buf_,rc_bufsize_) != (int)rc_bufsize_) {
                    throw (int)WRITE_ERROR;
                }
                rc_size_=0;
                prc_=rc_buf_;
            }
            temp = 0xFF;
        }
        while (--rc_cachesize_ != 0);
        rc_cache_ = (uint8_t)((uint32_t)rc_low_ >> 24);
    }
    rc_cachesize_++;
    rc_low_ = (uint32_t)rc_low_ << 8;
}

