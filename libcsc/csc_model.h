#ifndef _CSC_MODEL_H_
#define _CSC_MODEL_H_
#include <csc_typedef.h>
#include <csc_coder.h>


/*  Current specific 
The compression stream is made up by myriads of small data packs.
Each pack:

0-- 0 Rawbyte                    Literal byte
1-- 1 1  MatchDist MatchLen        Ordinary match
disabled-(2-- 1 0 0 0             Last match pair)
disabled-(3-- 1 0 0 1            1-Byte match with last repeat MatchDist)
3-- 1 0 0                        1-Byte match with last repeat MatchDist
4-- 1 0 1 0 0 MatchLen            A match with last repeat MatchDist(0)
5-- 1 0 1 0 1 MatchLen            A match with repeat MatchDist[1]
6-- 1 0 1 1 0 MatchLen            A match with repeat MatchDist[2]
7-- 1 0 1 1 1 MatchLen            A match with repeat MatchDist[3]

MatchDist
64 buckets with different num of extra direct bits.
probDists[64] is the statistical model.

MatchLen
16 buckets with different num of extra direct bits.
p_len_[16] is the statistical model.

//  0,  1 - 8
//  1 0, 9 - 16
//  1 1, 17 - 144

Rawbyte
Order-1 coder with only 3-MSBs as context
p_lit_[8][256] is the model

About state type:
pack 0            --- current type 0
pack 1            --- current type 1
pack 2,3        --- current type 2
pack 4,5,6,7    ---    current type 3

The state:
stores last 4 state type.
p_state_ is the model.
*/

class Model
{
    void encode_matchlen_1(uint32_t len);
    void encode_matchlen_2(uint32_t len);
    void encode_matchdist(uint32_t dist);

    uint32_t len_price_[32];
    void len_price_rebuild();
    uint32_t lp_rebuild_int_;

public:
    Coder *coder_;

    void Reset(void);
    int Init(Coder *coder, ISzAlloc *alloc);
    void Destroy();

    void EncodeLiteral(uint32_t c);
    void SetLiteralCtx(uint32_t c) {ctx_ = (c >> 0);}
    uint32_t GetLiteralCtx() {return ctx_;}


    void EncodeMatch(uint32_t matchDist,uint32_t matchLen);

    void EncodeRepDistMatch(uint32_t repIndex,uint32_t matchLen);


    void EncodeRep0Len1(void);
    void CompressDelta(uint8_t *src,uint32_t size);
    void CompressLiterals(uint8_t *src,uint32_t size);
    void CompressBad(uint8_t *src,uint32_t size);

    void CompressRLE(uint8_t *src,uint32_t size);

    void EncodeInt(uint32_t num);

    uint32_t p_state_[4*4*4*3];//Original [64][3]
    // public for advanced parser 
    uint32_t state_;

    //Fake Encode --- to get price
    uint32_t GetLiteralPrice(uint32_t fstate,uint32_t fctx,uint32_t c);
    uint32_t GetRep0Len1Price(uint32_t fstate);
    uint32_t GetRepDistPrice(uint32_t fstate,uint32_t repIndex);
    uint32_t GetMatchDistPrice(uint32_t fstate,uint32_t matchDist);
    uint32_t GetMatchLenPrice(uint32_t fstate,uint32_t matchLen);


private:
    ISzAlloc *alloc_;

    uint32_t p_rle_flag_;
    uint32_t p_rle_len_[16];

    // prob of literals
    uint32_t *p_lit_;//[256][256]
    uint32_t ctx_;
    uint32_t *p_delta_;

    uint32_t p_repdist_[64 * 4];
    // len 0 use 8 slots
    // len 1 - 2 use 16 slots
    // len 3, 4, 5, >= 6  use 32 slots
    uint32_t p_dist_[8 + 16 * 2 + 32 * 4];  //len 0 , < 64

    uint32_t p_longlen_;

    // prob to bits num
    uint32_t p_2_bits_[4096>>3];

    uint32_t p_matchlen_slot_[2];
    uint32_t p_matchlen_extra1_[8];
    uint32_t p_matchlen_extra2_[8];
    uint32_t p_matchlen_extra3_[128];
    uint32_t p_matchdist_extra_[29 * 16];

    static const uint32_t dist_table_[33];
    static const uint32_t rev16_table_[16];
};


#endif
