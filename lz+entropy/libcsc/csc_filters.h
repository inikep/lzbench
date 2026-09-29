#ifndef _CSC_FILTERS_H_
#define _CSC_FILTERS_H_

#include <csc_typedef.h>
#define MAX_WORDTREE_NODE_NUM 300 //Enough now!

class Filters
{
public:
    void Forward_E89( uint8_t* src, uint32_t size);
    void Inverse_E89( uint8_t* src, uint32_t size);
    void Init(ISzAlloc *alloc);
    void Destroy();

    uint32_t Foward_Dict(uint8_t *src,uint32_t size);
    void Inverse_Dict(uint8_t *src,uint32_t size);

    void Forward_Delta(uint8_t *src,uint32_t size,uint32_t chnNum);
    void Inverse_Delta(uint8_t *src,uint32_t size,uint32_t chnNum);
    //void Forward_RGB(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits);
    //void Inverse_RGB(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits);
    //void Forward_Audio(uint8_t *src,uint32_t size,uint32_t wavChannels,uint32_t sampleBits);
    //void Inverse_Audio(uint8_t *src,uint32_t size,uint32_t wavChannels,uint32_t sampleBits);
    //void Forward_Audio4(uint8_t *src,uint32_t size);
    //void Inverse_Audio4(uint8_t *src,uint32_t size);


private:
    ISzAlloc *alloc_;

    typedef struct {
        uint32_t next[26];
        uint8_t symbol;
    } CTreeNode;
    CTreeNode wordTree[MAX_WORDTREE_NODE_NUM];
    uint32_t nodeMum;
    uint8_t maxSymbol;
    //Used for DICT transformer. Words are stored in trees.

    uint32_t wordIndex[256];
    //Used for DICT untransformer.choose words by symbols.
    void MakeWordTree();  //Init the DICT transformer

    //Swap buffer for all filters. 
    uint8_t *m_fltSwapBuf;
    uint32_t m_fltSwapSize;

    /*
    Below is Shelwien's E89 filter
    */
    void E89init(void);
    int32_t E89cache_byte(int32_t c);
    uint32_t E89xswap(uint32_t x);
    uint32_t E89yswap(uint32_t x);
    int32_t E89forward(int32_t c);
    int32_t E89inverse(int32_t c);
    int32_t E89flush(void);

    uint32_t x0,x1;
    uint32_t i,k;
    uint8_t cs; // cache size, F8 - 5 bytes
};


#endif
