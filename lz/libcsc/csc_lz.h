#ifndef _CSC_LZ_H_
#define _CSC_LZ_H_
#include <csc_typedef.h>
#include <csc_mf.h>

class Model;

class LZ
{
public:
    int Init(const CSCProps *p, Model *model, ISzAlloc *alloc);
    void EncodeNormal(uint8_t *src, uint32_t size, uint32_t lz_mode);
    bool IsDuplicateBlock(uint8_t *src,uint32_t size);
    void DuplicateInsert(uint8_t *src,uint32_t size);

    void Reset(void);
    void Destroy(void);
    uint32_t wnd_curpos_;

private:
    ISzAlloc *alloc_;
    uint8_t  *wnd_;
    uint32_t rep_dist_[4];
    uint32_t wnd_size_;
    uint32_t curblock_endpos;
    uint32_t good_len_;
    uint32_t bt_cyc_;
    uint32_t ht_cyc_;
    Model *model_;

    // New LZ77 Algorithm====================================
// ============== OPTIMAL ====
    struct APUnit {
        uint32_t dist;
        uint32_t state;
        int back_pos;
        int next_pos;
        uint32_t price;
        uint32_t lit;
        uint32_t rep_dist[4];
    };
// ===========================
    static const int AP_LIMIT = 2048;
    APUnit apunits_[AP_LIMIT + 1];

    void compress_normal(uint32_t size, bool lazy);
    void compress_mf_skip(uint32_t size);
    void encode_nonlit(MFUnit u);

    void compress_advanced(uint32_t size);
    void ap_backward(int idx);

    MatchFinder mf_;
    MFUnit *appt_;
};


#endif

