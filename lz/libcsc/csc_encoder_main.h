#ifndef _CCSC_H
#define _CCSC_H

#include <csc_model.h>
#include <csc_filters.h>
#include <csc_analyzer.h>
#include <csc_lz.h>
#include <csc_coder.h>

class MemIO;

class CSCEncoder
{
public:
    int Init(const CSCProps *p, MemIO *io, ISzAlloc *alloc);
    

    void WriteEOF();
    //Should be called when finished compression of one part.

    void Flush();
    //Should be called when finished the whole compression.

    void Destroy();

    void Compress(uint8_t *src, uint32_t size);

    int Decompress(uint8_t *src, uint32_t *size);
    //*size==0 means meets the EOF in raw stream.

    void CheckFileType(uint8_t *src, uint32_t size);
    //Should be called before compress a file.src points
    //to first several bytes of file.
    
    int64_t GetCompressedSize();
    //Get current compressed size.
    

private:
    ISzAlloc *alloc_;
    uint32_t fixed_datatype_;
    CSCProps p_;

    Filters filters_;
    Coder coder_;
    Model model_;
    LZ lz_;
    Analyzer analyzer_;

    uint32_t rawblock_limit_;
    //This determines how much maximumly the CSCEncoder:Decompress can decompress
    // in one time. 

    bool use_filters_;
    void compress_block(uint8_t *src,uint32_t size,uint32_t type);
    //compress the buffer and treat them in one type.It's called after analyze the data.

};

#endif
