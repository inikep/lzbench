#ifndef _CSC_MEMIO_H_
#define _CSC_MEMIO_H_
#include <csc_common.h>

// rc/RC means for range coder
// bc/BC means for bit coder

class MemIO 
{
    uint32_t bsize_;
    ISzAlloc *alloc_;
    

    struct DataBlock {
        DataBlock *next;
        uint32_t size;
        char buf[1];
    };

    DataBlock *rc_blocks_;
    DataBlock *bc_blocks_;

    union {
        ISeqInStream *is_;
        ISeqOutStream *os_;
        void *ios_;
    };

    int ReadBlock(uint8_t *buffer, uint32_t &size, int rc1bc0);
    
    int WriteBlock(uint8_t *buffer, uint32_t size, int rc1bc0);

public:
    void Init(void *iostream, uint32_t bsize, ISzAlloc *alloc);
    void Destroy();

    uint32_t GetBlockSize() { return bsize_; }

    int ReadRCData(uint8_t *buffer,uint32_t& size)
    {
        return ReadBlock(buffer, size, 1);
    }

    int WriteRCData(uint8_t *buffer,uint32_t size)
    {
        return WriteBlock(buffer, size, 1);
    }

    int ReadBCData(uint8_t *buffer,uint32_t& size)
    {
        return ReadBlock(buffer, size, 0);
    }

    int WriteBCData(uint8_t *buffer,uint32_t size)
    {
        return WriteBlock(buffer, size, 0);
    }
};


#endif

