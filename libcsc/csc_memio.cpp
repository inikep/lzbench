#include <csc_memio.h>
#include <stdlib.h>
#include <string.h>

int MemIO::ReadBlock(uint8_t *buffer, uint32_t &size, int rc1bc0)
{
    DataBlock **blist;
    blist = rc1bc0 ? &rc_blocks_ : &bc_blocks_;

    if (*blist) {
        // already read into buffer
        size = (*blist)->size;
        memcpy(buffer, (*blist)->buf, size);
        DataBlock *tb = (*blist)->next;
        alloc_->Free(alloc_, *blist);
        (*blist) = tb;
    } else {
        // read fresh, if meet the other kind of block, keep it in its buffer
        for(;;) {
            uint8_t fb;
            size_t iosize;
            iosize = 1;
            is_->Read(is_, &fb, &iosize);
            if (iosize != 1)
                return -1;

            uint32_t cur_bsize;
            if ((fb >> 6) & 0x1) {
                // full block size
                cur_bsize = bsize_;
            } else {
                // extra 3 bytes denoting size
                uint8_t size_bytes[3];
                iosize = 3;
                is_->Read(is_, size_bytes, &iosize);
                if (iosize != 3)
                    return -1;
                cur_bsize = ((uint32_t)size_bytes[0] << 16)
                    + (size_bytes[1] << 8)
                    + size_bytes[2];
            }

            if (!cur_bsize || cur_bsize > bsize_) {
                // must be a abnormal stream
                return -1;
            }

            iosize = size = (int)cur_bsize;
            if (((fb >> 7) & 0x1) == rc1bc0) {
                // this is the block to read out
                is_->Read(is_, buffer, &iosize);
                if (iosize != cur_bsize)
                    return -1;
                break;
            } else {
                // other kind of block, append it to list
                // keep looping until meet the block it wants
                DataBlock *newblock = (DataBlock *)alloc_->Alloc(alloc_, sizeof(*newblock) + cur_bsize);
                newblock->size = cur_bsize;
                newblock->next = NULL;
                is_->Read(is_, newblock->buf, &iosize);
                if (iosize != cur_bsize) {
                    alloc_->Free(alloc_, newblock);
                    return -1;
                }

                DataBlock dummy;
                dummy.next = rc1bc0 ? bc_blocks_ : rc_blocks_;
                DataBlock *p = &dummy;
                while(p->next) p = p->next;
                p->next = newblock;
                if (rc1bc0) {
                    bc_blocks_ = dummy.next;
                } else {
                    rc_blocks_ = dummy.next;
                }
            }
        }
    }
    return (int)size;
}

int MemIO::WriteBlock(uint8_t *buffer, uint32_t size, int rc1bc0)
{
    uint8_t fb = 0;
    size_t iosize;
    fb |= ((uint8_t)rc1bc0 << 7);
    if (size == bsize_) 
        fb |= (1 << 6);

    iosize = 1;
    if (os_->Write(os_, &fb, iosize) != 1)
        return -1;

    if (size != bsize_) {
        uint8_t size_bytes[3];
        size_bytes[0] = ((size >> 16) & 0xff);
        size_bytes[1] = ((size >> 8) & 0xff);
        size_bytes[2] = (size & 0xff);
        iosize = 3;
        if (os_->Write(os_, size_bytes, iosize) != 3)
            return -1;
    }
    iosize = size;
    if (iosize && os_->Write(os_, buffer, iosize) != iosize)
        return -1;
    return size;
}

void MemIO::Init(void *iostream, uint32_t bsize, ISzAlloc *alloc)
{
    ios_ = iostream;
    bsize_ = bsize;
    rc_blocks_ = NULL;
    bc_blocks_ = NULL;
    alloc_ = alloc;
}


void MemIO::Destroy() {
    while (rc_blocks_) {
        DataBlock *next = rc_blocks_->next;
        alloc_->Free(alloc_, rc_blocks_);
        rc_blocks_ = next;
    }

    while (bc_blocks_) {
        DataBlock *next = bc_blocks_->next;
        alloc_->Free(alloc_, bc_blocks_);
        bc_blocks_ = next;
    }
}
