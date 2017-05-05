#ifndef _CSC_MF_H_
#define _CSC_MF_H_
#include <csc_typedef.h>


class Model;

struct MFUnit {
    union {
        uint32_t len;
        uint32_t price;
    };
    uint32_t dist;
};

class MatchFinder {
    static const uint32_t HT3_SIZE_ = 64 * KB;
    static const uint32_t HT2_SIZE_ = 16 * KB;

    ISzAlloc *alloc_;
    uint8_t *wnd_;
    uint32_t wnd_size_;
    uint32_t vld_rge_; // valid range, wnd_size - blocksize


    uint32_t *mfbuf_raw_;
    uint32_t *mfbuf_;
    uint32_t *ht2_;
    uint32_t *ht3_;
    uint32_t *ht6_;
    uint32_t ht_bits_;
    uint32_t ht_width_;
    uint32_t ht_low_;
    static const uint32_t MF_CAND_LIMIT = 32;
    MFUnit mfcand_[MF_CAND_LIMIT];

    uint32_t *bt_head_;
    uint32_t bt_bits_;
    uint32_t *bt_nodes_;
    uint32_t bt_size_;

    uint32_t bt_pos_;
    uint64_t size_;

    uint32_t ht_cyc_;
    uint32_t bt_cyc_;
    uint32_t good_len_;
    
    void normalize();

    uint32_t find_match(MFUnit *ret, uint32_t *rep_dist, uint32_t wnd_pos, uint32_t limit);

public:
    int Init(uint8_t *wnd, 
            uint32_t wnd_size,
            uint32_t bt_size, 
            uint32_t bt_bits, 
            uint32_t ht_width,
            uint32_t ht_bits,
            ISzAlloc *alloc
            );

    void SetArg(int bt_cyc, int ht_cyc, int ht_low, int good_len);
    void Destroy();

    // limit is useful only for binary tree mf
    void SlidePos(uint32_t wnd_pos, uint32_t len, uint32_t limit = 0xFFFFFFFF);
    void SlidePosFast(uint32_t wnd_pos, uint32_t len);
    MFUnit FindMatch(uint32_t *rep_dist, uint32_t wnd_pos, uint32_t limit);
    bool TestFind(uint32_t wnd_pos, uint8_t *src, uint32_t limit);
    bool SecondMatchBetter(MFUnit u1, MFUnit u2);
    void FindMatchWithPrice(Model *model, uint32_t state, MFUnit *ret, uint32_t *rep_dist, uint32_t wnd_pos, uint32_t limit);
    uint32_t pos_;
};

#endif

