#include "aceapex.h"
#include "ax_align.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif
#endif
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#include <pthread.h>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <zstd.h>
#define XXH_STATIC_LINKING_ONLY
#ifndef ACEAPEX_NO_XXH
#define XXH_IMPLEMENTATION
#endif
#include "xxhash.h"
#define OUR_CHECKSUM(buf,sz) XXH3_64bits(buf,sz)
#include "lit_fse.cpp"
#include "ax_rans.h"
#include "ax_lit_open.h"

 
#define HASH_SIZE    0xFFFF
#define MAX_DIST     (128 * 1024 * 1024)
#define BLOCK_SIZE   (1 * 1024 * 1024)
// Per-call state is thread-local (30.09: two library calls at once - lzbench -T2 - shared the block
// size, the decode error flag and the DNA hint, and 2 of 4569 files came back as errors; 2.1.0 gave
// wrong bytes on 22). A thread started inside a call inherits its caller's state through ax_thread.
static thread_local size_t g_block_size = BLOCK_SIZE; // runtime adaptive, set in encode_file
extern thread_local int g_input_is_dna;
static thread_local std::atomic<int> t_dec_err_own{0};
static thread_local std::atomic<int>* t_dec_err = nullptr;
static inline std::atomic<int>& ax_dec_err(){ return t_dec_err ? *t_dec_err : t_dec_err_own; }
#define g_dec_err (ax_dec_err())
struct AxSpawn { void* (*fn)(void*); void* arg; std::atomic<int>* err; size_t bs; int dna; };
static void* ax_tramp(void* p){ AxSpawn s=*(AxSpawn*)p; delete (AxSpawn*)p;
    t_dec_err=s.err; g_block_size=s.bs; g_input_is_dna=s.dna; return s.fn(s.arg); }
static std::atomic<long> g_ax_spawned{0};             // threads started by the codec (claim head_enc_threads)
static int ax_thread(pthread_t* t, void* (*fn)(void*), void* arg){
    g_ax_spawned.fetch_add(1);
    return pthread_create(t,nullptr,ax_tramp,new AxSpawn{fn,arg,&ax_dec_err(),g_block_size,g_input_is_dna}); }
// Encoder thread budget of the current call (2.2.1): encode_file sets it from its `threads`; the entropy
// stage (literal lanes, the three token streams) keeps to it. Before 2.2.1 that stage started up to
// 3 + CPU-count threads whatever the budget (lzbench -I1: 146 % CPU). 0 = not set (no cap).
static thread_local int g_enc_threads = 0;
static inline int ax_enc_budget(int want){ return g_enc_threads > 0 ? std::max(1, std::min(want, g_enc_threads)) : std::max(1, want); }
// run a pool-style worker on n threads: n-1 started, one is the caller (n <= 1: no thread at all)
static void ax_run_pool(int n, void* (*fn)(void*), void* arg){
    if (n <= 1) { fn(arg); return; }
    std::vector<pthread_t> t(n-1);
    for (int i = 0; i < n-1; i++) ax_thread(&t[i], fn, arg);
    fn(arg);
    for (int i = 0; i < n-1; i++) pthread_join(t[i], nullptr);
}
#define MAX_THREADS  16
#define BLOCK_MARKER 0xFF
#define ZSTD_LEVEL   22
 
struct BlockResult {
    uint8_t* lit_buf; uint8_t* off_buf;
    uint8_t* len_buf; uint8_t* cmd_buf;
    size_t   lit_size, off_size, len_size, cmd_size;
    int      overflow;
};
 
struct PoolState {
    const uint8_t* src; size_t src_size;
    size_t num_blocks; BlockResult* results;
    std::atomic<size_t> next_block;
};
 
struct WorkerArgs {
    int thread_id;
    struct ThreadHashTable* htab;
    PoolState* pool;
};
 
// Positions in the tables are relative to the block start (matches never leave the
// block), stored as uint32: pos/epoch 4+4 bytes per slot, chain 4 bytes per block
// position instead of 8, and the chain is reset per block so a stale link can never
// look like a valid relative position. Halves the per-thread working set; the
// candidate sequence and the archive bytes are unchanged.
#define AX_NOPOS 0xFFFFFFFFu
struct ThreadHashTable {
    uint32_t* pos;
    uint32_t* epoch;
    uint32_t* chain;
    uint32_t  cur_epoch;
    uint32_t  hash_mask;
    uint32_t  chain_mask;
    int       max_attempts;
    int       l1;          // l1 encoder (ADR-020): 8K-slot head table (64 KB with epochs), no chain
    int       noflat;      // no offset flattening (l1)
    uint32_t  minl;        // shortest match taken (l1: 32; 0 = any)
    uint32_t  skip;        // literal-run skip shift (l1: 4; 0 = off)
};
 
struct BlockOffsets {
    uint64_t lit_off, off_off, len_off, cmd_off;
    uint64_t lit_sz,  off_sz,  len_sz,  cmd_sz;
};


 
static inline void wv(uint8_t* buf, size_t& ptr, uint32_t val,
                      size_t limit, int& ov, int sid) {
    while (val >= 0x80) {
        if (ptr >= limit) { ov=sid; return; }
        buf[ptr++] = (uint8_t)((val & 0x7F) | 0x80); val >>= 7;
    }
    if (ptr >= limit) { ov=sid; return; }
    buf[ptr++] = (uint8_t)val;
}
 
static inline uint32_t min_match_len(uint32_t dist) {
    if (dist < 128)     return 6;
    if (dist < 16384)   return 8;
    if (dist < 2097152) return 10;
    return 12;
}
 

// Match length from l on, capped at min(maxl, 65535): 8 bytes per step (XOR + ctz)
// instead of byte by byte. Same result as the byte loop it replaces.
static inline uint32_t ax_ext(const uint8_t* a, const uint8_t* b, uint32_t l, uint32_t maxl) {
    uint32_t cap = maxl < 65535 ? maxl : 65535;
    while (l + 8 <= cap) {
        uint64_t x = AX_read64(a + l) ^ AX_read64(b + l);
        if (x) return l + (uint32_t)(__builtin_ctzll(x) >> 3);
        l += 8;
    }
    while (l < cap && a[l] == b[l]) l++;
    return l;
}
struct Match { uint32_t len, off; int rep; };
static inline int find_matches(const uint8_t* src, size_t pos, size_t bstart, size_t bend,
                                ThreadHashTable* ht, uint32_t* rep, Match* out, int maxout) {
    int max_attempts = ht->max_attempts;
    int n = 0;
    uint32_t maxl = (uint32_t)(bend - pos);
    for (int i = 0; i < 4 && n < maxout; i++) {
        uint32_t d = rep[i]; if (pos < bstart+d) continue;
        if (AX_read32(src+pos)!=AX_read32(src+pos-d)) continue;
        uint32_t l=ax_ext(src+pos,src+pos-d,4,maxl);
        if (l>=6) out[n++]={l,d,i};
    }
    uint32_t h=((AX_read32(src+pos)*0x9E3779B1u)>>10)&ht->hash_mask;
    uint32_t rp=(uint32_t)(pos-bstart);
    uint32_t hr=(ht->epoch[h]==ht->cur_epoch)?ht->pos[h]:AX_NOPOS;
    ht->pos[h]=rp; ht->epoch[h]=ht->cur_epoch;
    if (hr!=AX_NOPOS && !ht->l1) ht->chain[rp & ht->chain_mask]=hr;
    int64_t cur=(hr==AX_NOPOS)?-1:(int64_t)bstart+hr; int attempts=max_attempts;
    while(cur>=(int64_t)bstart && attempts-->0 && n<maxout) {
        uint32_t dist=(uint32_t)(pos-cur); if(dist>=MAX_DIST) break;
        bool is_rep=false; for(int r=0;r<4;r++) if(dist==rep[r]){is_rep=true;break;}
        if(!is_rep){
            uint32_t mlen=min_match_len(dist);
            if(pos+8<=bend&&AX_read64(src+pos)==AX_read64(src+cur)){
                uint32_t l=ax_ext(src+pos,src+cur,8,maxl);
                if(l>=mlen) out[n++]={l,dist,-1};
            } else if(AX_read32(src+pos)==AX_read32(src+cur)){
                uint32_t l=ax_ext(src+pos,src+cur,4,maxl);
                if(l>=mlen) out[n++]={l,dist,-1};
            }
        }
        if (ht->l1) break;
        uint32_t nr=ht->chain[(uint32_t)(cur-bstart) & ht->chain_mask];
        if(nr==AX_NOPOS) break; int64_t nxt=(int64_t)bstart+nr;
        if(nxt>=cur) break; cur=nxt;
    }
    return n;
}
static void compress_block(const uint8_t* src, size_t src_size,
                            size_t bstart, size_t bend,
                            ThreadHashTable* ht, BlockResult* res) {
    size_t bsz = bend - bstart;
    size_t cap = bsz * 2 + 1024;
    res->lit_buf = (uint8_t*)malloc(cap);
    res->off_buf = (uint8_t*)malloc(cap * 6);
    res->len_buf = (uint8_t*)malloc(cap * 6);
    res->cmd_buf = (uint8_t*)malloc(cap + cap/4 + 4);
    res->overflow = 0;
    if (!res->lit_buf || !res->off_buf || !res->len_buf || !res->cmd_buf) {
        res->overflow = 99; return;
    }
    size_t lit_cap=cap, off_cap=cap*6, len_cap=cap*6, cmd_cap=cap+cap/4+4;
 
    ht->cur_epoch++;
    if (ht->cur_epoch == 0) {
        memset(ht->epoch, 0, (ht->hash_mask+1)*sizeof(uint32_t)); ht->cur_epoch = 1;
    }
    if (!ht->l1) { size_t n = bsz < (size_t)ht->chain_mask + 1 ? bsz : (size_t)ht->chain_mask + 1;
      memset(ht->chain, 0xFF, n * sizeof(uint32_t)); }        // no stale links from earlier blocks
 
    size_t lit_i=0, off_i=0, len_i=0, cmd_i=0, pos=bstart;
    uint32_t rep[4]={1,2,4,8}, lit_run=0, miss=0;
    int ov=0;
    res->cmd_buf[cmd_i++] = BLOCK_MARKER;

    // ULTRA: Chain flattening origin table
    // origin[local_pos] = original literal source position (local)
    static thread_local uint32_t origin[1048576];
    size_t flat_pos = 0; // track which positions are initialized
    auto init_origin = [&](size_t from, size_t to) {
        for (size_t i = from; i < to && i < 1048576; i++) origin[i] = (uint32_t)i;
    };
    // AX_ENC=l1 (enc-l1, not default; changes archive bytes, format unchanged): no offset
    // flattening, only matches >= 32 bytes (shorter repeats stay literals, zstd/rANS of the
    // literal stream takes them), literal runs skipped 1 + miss>>4 bytes at a time.
    // AX_NOFLAT / AX_MINL / AX_SKIP override each knob.
    const int noflat = ht->noflat;
    const uint32_t l1_minl = ht->minl, l1_skip = ht->skip;
    if (!noflat) init_origin(0, bsz); // init all as self-referential (literal)
 
    auto flush_lit = [&]() {
        while (lit_run > 0 && !ov) {
            uint32_t chunk = (lit_run > 128) ? 128 : lit_run;
            if (cmd_i >= cmd_cap) { ov=4; return; }
            res->cmd_buf[cmd_i++] = (uint8_t)(chunk-1);
            lit_run -= chunk;
        }
    };
 
    while (pos + 12 < bend && !ov) {
        uint32_t c_len=0, c_off=0; int c_rep=-1;
        Match matches[36]; int nm=find_matches(src,pos,bstart,bend,ht,rep,matches,36);
        for(int mi=0;mi<nm;mi++) if(matches[mi].len>c_len){c_len=matches[mi].len;c_off=matches[mi].off;c_rep=matches[mi].rep;}
        if (c_len >= 6 && c_len < 64 && pos+13 < bend) {
            uint32_t h1=((AX_read32(src+pos+1)*0x9E3779B1u)>>10)&ht->hash_mask;
            int64_t mp1=(ht->epoch[h1]==ht->cur_epoch)?(int64_t)bstart+ht->pos[h1]:-1;
            if (mp1>=0 && (size_t)mp1>=bstart && (size_t)mp1<pos+1) {
                uint32_t dist1=(uint32_t)(pos+1-mp1);
                if (dist1<MAX_DIST && dist1!=rep[0]) {
                    uint32_t mlen1=min_match_len(dist1);
                    uint32_t maxl1=(uint32_t)(bend-pos-1);
                    if (pos+9<=bend && AX_read64(src+pos+1)==AX_read64(src+mp1)) {
                        uint32_t l1=8;
                        l1=ax_ext(src+pos+1,src+mp1,l1,maxl1);
                        if (l1 >= mlen1 && l1 > c_len + 1) {
                            if (lit_i < lit_cap) {
                                res->lit_buf[lit_i++]=src[pos]; lit_run++; miss++;
                                pos++;
                                c_len=l1; c_off=dist1; c_rep=-1;
                            }
                        }
                    }
                }
            }
            // Lazy check pos+2
            if (c_len >= 6 && c_len < 64 && pos+14 < bend) {
                uint32_t h2=((AX_read32(src+pos+2)*0x9E3779B1u)>>10)&ht->hash_mask;
                int64_t mp2=(ht->epoch[h2]==ht->cur_epoch)?(int64_t)bstart+ht->pos[h2]:-1;
                if (mp2>=0 && (size_t)mp2>=bstart && (size_t)mp2<pos+2) {
                    uint32_t dist2=(uint32_t)(pos+2-mp2);
                    if (dist2<MAX_DIST && dist2!=rep[0]) {
                        uint32_t maxl2=(uint32_t)(bend-pos-2);
                        if (pos+10<=bend && AX_read64(src+pos+2)==AX_read64(src+mp2)) {
                            uint32_t l2=8;
                            l2=ax_ext(src+pos+2,src+mp2,l2,maxl2);
                            if (l2 >= 6 && l2 > c_len + 2 && lit_i+1 < lit_cap) {
                                res->lit_buf[lit_i++]=src[pos]; lit_run++; miss++;
                                res->lit_buf[lit_i++]=src[pos+1]; lit_run++;
                                pos+=2;
                                c_len=l2; c_off=dist2; c_rep=-1;
                            }
                        }
                    }
                }
            }
        }
        if (c_len >= 6 && c_len < l1_minl) c_len = 0;       // l1: short matches stay literals for zstd
        if (c_len >= 6) {
            flush_lit(); if (ov) break; miss=0;
            uint32_t lv=c_len-6;
            if (c_rep != -1) {
                if (cmd_i>=cmd_cap) { ov=4; break; }
                if (lv<15) { res->cmd_buf[cmd_i++]=(uint8_t)(0x80|(c_rep<<4)|lv); }
                else { res->cmd_buf[cmd_i++]=(uint8_t)(0x80|(c_rep<<4)|0x0F);
                       wv(res->len_buf,len_i,lv-15,len_cap,ov,3); if(ov) break; }
                uint32_t rd=rep[c_rep];
                for (int i=c_rep;i>0;i--) rep[i]=rep[i-1]; rep[0]=rd;
            } else {
                if (cmd_i>=cmd_cap) { ov=4; break; }
                if (lv<62) { res->cmd_buf[cmd_i++]=(uint8_t)(0xC0|lv); }
                else { res->cmd_buf[cmd_i++]=0xFE;
                       wv(res->len_buf,len_i,lv,len_cap,ov,3); if(ov) break; }
                // ULTRA: Chain flattening with validation
                size_t local_pos = pos - bstart;
                uint32_t flat_off = c_off;
                if (!noflat && c_off <= local_pos && local_pos < 1048576) {
                    size_t src_local = local_pos - c_off;
                    uint32_t orig_src = origin[src_local];
                    if (orig_src != src_local) {
                        uint32_t candidate = (uint32_t)(local_pos - orig_src);
                        // Validate ALL bytes
                        bool valid = (candidate <= local_pos && candidate < 8388608u);
                        for (size_t fi = 0; fi < c_len && valid; fi++) {
                            if (src[bstart + local_pos - candidate + fi] !=
                                src[bstart + local_pos - c_off + fi])
                                valid = false;
                        }
                        if (valid) flat_off = candidate;
                    }
                    for (size_t fi = 0; fi < c_len && local_pos+fi < 1048576 && src_local+fi < 1048576; fi++)
                        origin[local_pos + fi] = origin[src_local + fi];
                }
                wv(res->off_buf,off_i,flat_off,off_cap,ov,2); if(ov) break;
                rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=flat_off;
            }
            // Insert intermediate positions for short matches only
            // Insert intermediate positions for short matches only
            if (c_len < 32) {
              uint32_t step=1+(c_len>>3);
              for(size_t ii=1;ii<c_len&&pos+ii+4<bend;ii+=step){
                uint32_t hh=((AX_read32(src+pos+ii)*0x9E3779B1u)>>10)&ht->hash_mask;
                if (!ht->l1) ht->chain[(uint32_t)(pos+ii-bstart)&ht->chain_mask]=(ht->epoch[hh]==ht->cur_epoch)?ht->pos[hh]:AX_NOPOS;
                ht->pos[hh]=(uint32_t)(pos+ii-bstart); ht->epoch[hh]=ht->cur_epoch;
              }
            }
            pos+=c_len; continue;
        }
        if (lit_i>=lit_cap) { ov=1; break; }
        res->lit_buf[lit_i++]=src[pos++]; lit_run++; miss++;
        if (miss>=1 && pos+12<bend) {
            uint32_t hh=((AX_read32(src+pos)*0x9E3779B1u)>>10)&ht->hash_mask;
            if(hh<=ht->hash_mask) { ht->pos[hh]=(uint32_t)(pos-bstart); ht->epoch[hh]=ht->cur_epoch; }
            if (lit_i>=lit_cap) { ov=1; break; }
            res->lit_buf[lit_i++]=src[pos++]; lit_run++;
            // l1: skip faster through literal runs (LZ4-style): 1 + miss>>l1_skip extra bytes
            if (l1_skip) { size_t k = miss >> l1_skip; if (k > 64) k = 64;
                if (k && pos + k + 12 < bend && lit_i + k <= lit_cap) {
                    memcpy(res->lit_buf + lit_i, src + pos, k); lit_i += k; lit_run += (uint32_t)k; pos += k; } }
        }
    }
    if (!ov) {
        while (pos<bend) {
            if (lit_i>=lit_cap) { ov=1; break; }
            res->lit_buf[lit_i++]=src[pos++]; lit_run++;
        }
        flush_lit();
    }
    res->lit_size=lit_i; res->off_size=off_i;
    res->len_size=len_i; res->cmd_size=cmd_i; res->overflow=ov;
}
 
static void* worker_func(void* arg) {
    WorkerArgs* wa=(WorkerArgs*)arg;
    PoolState*  ps=wa->pool;
    while (true) {
        size_t bid=ps->next_block.fetch_add(1);
        if (bid>=ps->num_blocks) break;
        size_t bstart=bid*g_block_size, bend=bstart+g_block_size;
        if (bend>ps->src_size) bend=ps->src_size;
        compress_block(ps->src,ps->src_size,bstart,bend,wa->htab,&ps->results[bid]);
    }
    return nullptr;
}
 
static inline void copy_match(uint8_t* dst, size_t out_ptr, uint32_t dist, uint32_t len) {
    uint8_t* d = dst + out_ptr;
    const uint8_t* s = dst + out_ptr - dist;
    if (__builtin_expect(dist >= len, 1)) { memcpy(d, s, len); return; }
    if (dist == 1) { memset(d, s[0], len); return; }
    uint32_t done = 0;
    while (done + dist <= len) { memcpy(d + done, s, dist); done += dist; }
    if (done < len) memcpy(d + done, s, len - done);
}
 
static inline uint32_t read_varint(const uint8_t* buf, size_t& ptr, size_t limit) {
    if (__builtin_expect(ptr < limit, 1)) {
        uint8_t b0 = buf[ptr];
        if (__builtin_expect(!(b0 & 0x80), 1)) { ptr++; return b0; }
        if (__builtin_expect(ptr + 1 < limit, 1)) {
            uint8_t b1 = buf[ptr+1];
            if (__builtin_expect(!(b1 & 0x80), 1)) {
                ptr += 2;
                return (uint32_t)(b0 & 0x7F) | ((uint32_t)b1 << 7);
            }
        }
    }
    uint32_t val=0, shift=0;
    while (ptr<limit && shift<=28) {                 // a valid varint has <= 5 bytes; a corrupt one must not shift past 31 (UBSan)
        uint8_t b=buf[ptr++]; val|=(uint32_t)(b&0x7F)<<shift;
        if (!(b&0x80)) return val; shift+=7;
    }
    return val;
}
 

// ULTRA: D=1 depth analyzer
// Checks if ALL match sources are in literal-only zones (never in match-written zones)
// If true: two-phase decode is provably correct
static void analyze_depth(
    size_t dst_size,
    const uint8_t* off, size_t off_sz,
    const uint8_t* len, size_t len_sz,
    const uint8_t* cmd, size_t cmd_sz, size_t block_id)
{
    // First pass: collect all ops with positions
    struct Op { uint32_t dst; uint32_t src; uint32_t len; uint8_t is_lit; };
    static Op ops[131072];
    size_t n=0;
    size_t op=0,np=0,cp=0,out=0;
    uint32_t rep[4]={1,2,4,8};
    while(out<dst_size && cp<cmd_sz && n<131072) {
        uint8_t c=cmd[cp++];
        if(c==0xFF){rep[0]=1;rep[1]=2;rep[2]=4;rep[3]=8;continue;}
        if(c<0x80) {
            uint32_t l=c+1;
            ops[n++]={(uint32_t)out,(uint32_t)0,l,1};
            out+=l;
        } else if((c&0xC0)==0x80) {
            uint32_t ri=(c>>4)&3,lv=c&0x0F;
            if(lv==0x0F)lv+=read_varint(len,np,len_sz);
            uint32_t l=lv+6,dist=rep[ri];
            if(ri>0){for(int i=ri;i>0;i--)rep[i]=rep[i-1];rep[0]=dist;}
            if(!dist||out+l>dst_size)break;
            ops[n++]={(uint32_t)out,(uint32_t)(out-dist),l,0};
            out+=l;
        } else {
            uint32_t lv=(c==0xFE)?read_varint(len,np,len_sz):(uint32_t)(c&0x3F);
            uint32_t l=lv+6,dist=read_varint(off,op,off_sz);
            rep[3]=rep[2];rep[2]=rep[1];rep[1]=rep[0];rep[0]=dist;
            if(!dist||out+l>dst_size)break;
            ops[n++]={(uint32_t)out,(uint32_t)(out-dist),l,0};
            out+=l;
        }
    }

    // Second pass: for each match, check if src overlaps any MATCH dst
    // Build match dst ranges first
    size_t d1_safe=0, d2_dep=0;
    for(size_t i=0;i<n;i++) {
        if(ops[i].is_lit) continue;
        uint32_t src_start=ops[i].src;
        uint32_t src_end=ops[i].src+ops[i].len;
        bool has_match_dep=false;
        // Check all previous match destinations
        for(size_t j=0;j<i;j++) {
            if(ops[j].is_lit) continue; // skip literals
            uint32_t mdst_start=ops[j].dst;
            uint32_t mdst_end=ops[j].dst+ops[j].len;
            // Does our src overlap this match's dst?
            if(src_start < mdst_end && src_end > mdst_start) {
                has_match_dep=true;
                break;
            }
        }
        if(has_match_dep) d2_dep++;
        else d1_safe++;
    }
    fprintf(stderr, "block %zu: matches=%zu D1_safe=%.1f%% D2_dep=%.1f%%\n",
        block_id, d1_safe+d2_dep,
        (d1_safe+d2_dep)?100.0*d1_safe/(d1_safe+d2_dep):0,
        (d1_safe+d2_dep)?100.0*d2_dep/(d1_safe+d2_dep):0);
}

// 16-byte wild copies (the zstd/LZ4 way): a run or match is written in 16-byte steps
// that may run past its end, so the caller guarantees 16 bytes of slack inside the
// block and, for literals, inside the block's literal slice. A match closer than 16
// bytes is first expanded byte-wise to a period P = dist*ceil(16/dist) >= 16, after
// which 16-byte steps from d-P read only bytes already written (self-overlap as
// period; a known technique, see ZSTD_overlapCopy8 / LZ4).
static inline void ax_copy16(uint8_t* d, const uint8_t* s) { memcpy(d, s, 16); }
static inline void ax_wild_copy(uint8_t* d, const uint8_t* s, uint32_t len) {
    uint8_t* e = d + len;
    do { ax_copy16(d, s); d += 16; s += 16; } while (d < e);
}
static inline void ax_match_fast(uint8_t* d, uint32_t dist, uint32_t len) {
    if (dist >= 16) { ax_wild_copy(d, d - dist, len); return; }
    uint32_t P = dist * ((16 + dist - 1) / dist);          // smallest multiple of dist >= 16
    uint32_t head = P < len ? P : len;
    const uint8_t* s = d - dist;
    for (uint32_t i = 0; i < head; i++) d[i] = s[i];
    if (len > head) ax_wild_copy(d + head, d + head - P, len - head);
}

static void decompress_streams(
    uint8_t* dst, size_t dst_size,
    const uint8_t* lit, size_t lit_sz,
    const uint8_t* off, size_t off_sz,
    const uint8_t* len, size_t len_sz,
    const uint8_t* cmd, size_t cmd_sz)
{
    size_t lp=0, op=0, np=0, cp=0, out=0;
    uint32_t rep[4]={1,2,4,8};
    const size_t SL = 16;                                    // slack for wild copies
    while (out<dst_size && cp<cmd_sz) {
        uint8_t c=cmd[cp++];
        if (c==0xFF) { rep[0]=1;rep[1]=2;rep[2]=4;rep[3]=8; continue; }
        if (c<0x80) {
            uint32_t l=c+1;
            if (lp+l>lit_sz||out+l>dst_size) break;
            if (out+l+SL<=dst_size && lp+l+SL<=lit_sz) ax_wild_copy(dst+out,lit+lp,l);
            else memcpy(dst+out,lit+lp,l);
            out+=l; lp+=l;
        } else if ((c&0xC0)==0x80) {
            uint32_t ri=(c>>4)&3, lv=c&0x0F;
            if (lv==0x0F) lv+=read_varint(len,np,len_sz);
            uint32_t l=lv+6, dist=rep[ri];
            if (ri>0) { for(int i=ri;i>0;i--) rep[i]=rep[i-1]; rep[0]=dist; }
            // dist>out would read before the start of the block buffer.
            // A corrupted offset made this read arbitrary memory -> SIGSEGV.
            if (!dist||dist>out||out+l>dst_size) break;
            if (out+l+SL<=dst_size) ax_match_fast(dst+out,dist,l); else copy_match(dst,out,dist,l);
            out+=l;
        } else {
            uint32_t lv=(c==0xFE)?read_varint(len,np,len_sz):(uint32_t)(c&0x3F);
            uint32_t l=lv+6, dist=read_varint(off,op,off_sz);
            rep[3]=rep[2];rep[2]=rep[1];rep[1]=rep[0];rep[0]=dist;
            // Same guard: a corrupted offset varint yields a huge dist, and
            // copy_match would then read from dst+out-dist, far before the buffer.
            if (!dist||dist>out||out+l>dst_size) break;
            if (out+l+SL<=dst_size) ax_match_fast(dst+out,dist,l); else copy_match(dst,out,dist,l);
            out+=l;
        }
    }
}
 

// ULTRA: Adaptive parallel decoder
// Checks true source-readiness (not just self-overlap)
static void decompress_adaptive(
    uint8_t* dst, size_t dst_size,
    const uint8_t* lit, size_t lit_sz,
    const uint8_t* off, size_t off_sz,
    const uint8_t* len, size_t len_sz,
    const uint8_t* cmd, size_t cmd_sz)
{
    // Pass 1: decode all literals first into dst
    // This makes all literal bytes "source-ready"
    size_t lp=0, op=0, np=0, cp=0, out=0;
    uint32_t rep[4]={1,2,4,8};
    
    // First pass: literals only
    size_t cp1=0, lp1=0, out1=0;
    uint32_t rep1[4]={1,2,4,8};
    size_t np1=0, op1=0;
    while (out1<dst_size && cp1<cmd_sz) {
        uint8_t c=cmd[cp1++];
        if (c==0xFF){rep1[0]=1;rep1[1]=2;rep1[2]=4;rep1[3]=8;continue;}
        if (c<0x80) {
            uint32_t l=c+1;
            if (lp1+l>lit_sz||out1+l>dst_size) break;
            memcpy(dst+out1, lit+lp1, l);
            out1+=l; lp1+=l;
        } else if ((c&0xC0)==0x80) {
            uint32_t ri=(c>>4)&3,lv=c&0x0F;
            if(lv==0x0F) lv+=read_varint(len,np1,len_sz);
            uint32_t l=lv+6,dist=rep1[ri];
            if(ri>0){for(int i=ri;i>0;i--)rep1[i]=rep1[i-1];rep1[0]=dist;}
            out1+=l; // skip match for now
        } else {
            uint32_t lv=(c==0xFE)?read_varint(len,np1,len_sz):(uint32_t)(c&0x3F);
            uint32_t l=lv+6; read_varint(off,op1,off_sz);
            rep1[3]=rep1[2];rep1[2]=rep1[1];rep1[1]=rep1[0];
            out1+=l; // skip match for now
        }
    }
    size_t lit_ready_end = out1; // all literal positions are ready

    // Second pass: matches only, using lit_ready_end as source-readiness threshold
    size_t cp2=0, lp2=0, out2=0, op2=0, np2=0;
    uint32_t rep2[4]={1,2,4,8};
    while (out2<dst_size && cp2<cmd_sz) {
        uint8_t c=cmd[cp2++];
        if (c==0xFF){rep2[0]=1;rep2[1]=2;rep2[2]=4;rep2[3]=8;continue;}
        if (c<0x80) {
            uint32_t l=c+1; out2+=l; lp2+=l; // already done
        } else if ((c&0xC0)==0x80) {
            uint32_t ri=(c>>4)&3,lv=c&0x0F;
            if(lv==0x0F) lv+=read_varint(len,np2,len_sz);
            uint32_t l=lv+6,dist=rep2[ri];
            if(ri>0){for(int i=ri;i>0;i--)rep2[i]=rep2[i-1];rep2[0]=dist;}
            if(!dist||out2+l>dst_size) break;
            // Source-ready check: src fully before lit_ready_end
            if (out2-dist+l <= lit_ready_end) {
                memcpy(dst+out2, dst+out2-dist, l); // safe parallel copy
            } else {
                copy_match(dst,out2,dist,l); // fallback
            }
            out2+=l;
        } else {
            uint32_t lv=(c==0xFE)?read_varint(len,np2,len_sz):(uint32_t)(c&0x3F);
            uint32_t l=lv+6,dist=read_varint(off,op2,off_sz);
            rep2[3]=rep2[2];rep2[2]=rep2[1];rep2[1]=rep2[0];rep2[0]=dist;
            if(!dist||out2+l>dst_size) break;
            if (out2-dist+l <= lit_ready_end) {
                memcpy(dst+out2, dst+out2-dist, l);
            } else {
                copy_match(dst,out2,dist,l);
            }
            out2+=l;
        }
    }
}


struct DecOp {
    uint32_t src_off;
    uint32_t dst_off;
    uint32_t len;
    uint8_t  is_lit;
};

// ULTRA: True parallel decoder using Bernstein conditions
// Step 1: Sequential literals (creates ready zone)
// Step 2: Parallel matches where src is in ready zone
static void decompress_parallel(
    uint8_t* dst, size_t dst_size,
    const uint8_t* lit, size_t lit_sz,
    const uint8_t* off, size_t off_sz,
    const uint8_t* len, size_t len_sz,
    const uint8_t* cmd, size_t cmd_sz)
{
    // Build ops list first
    static thread_local DecOp ops_buf[131072];
    size_t ops_cnt = 0;
    size_t lp=0, op=0, np=0, cp=0, out=0;
    uint32_t rep[4]={1,2,4,8};
    while (out<dst_size && cp<cmd_sz) {
        uint8_t c=cmd[cp++];
        if (c==0xFF){rep[0]=1;rep[1]=2;rep[2]=4;rep[3]=8;continue;}
        if (c<0x80) {
            uint32_t l=c+1;
            if (lp+l>lit_sz||out+l>dst_size) break;
            ops_buf[ops_cnt++]={(uint32_t)lp,(uint32_t)out,l,1};
            out+=l; lp+=l;
        } else if ((c&0xC0)==0x80) {
            uint32_t ri=(c>>4)&3,lv=c&0x0F;
            if(lv==0x0F) lv+=read_varint(len,np,len_sz);
            uint32_t l=lv+6,dist=rep[ri];
            if(ri>0){for(int i=ri;i>0;i--)rep[i]=rep[i-1];rep[0]=dist;}
            if(!dist||out+l>dst_size) break;
            ops_buf[ops_cnt++]={(uint32_t)(out-dist),(uint32_t)out,l,0};
            out+=l;
        } else {
            uint32_t lv=(c==0xFE)?read_varint(len,np,len_sz):(uint32_t)(c&0x3F);
            uint32_t l=lv+6,dist=read_varint(off,op,off_sz);
            rep[3]=rep[2];rep[2]=rep[1];rep[1]=rep[0];rep[0]=dist;
            if(!dist||out+l>dst_size) break;
            ops_buf[ops_cnt++]={(uint32_t)(out-dist),(uint32_t)out,l,0};
            out+=l;
        }
    }

    // Step 1: Sequential literals — creates ready zone
    size_t ready_end = 0;
    for (size_t i = 0; i < ops_cnt; i++) {
        if (ops_buf[i].is_lit) {
            memcpy(dst+ops_buf[i].dst_off, lit+ops_buf[i].src_off, ops_buf[i].len);
            size_t end = ops_buf[i].dst_off + ops_buf[i].len;
            if (end > ready_end) ready_end = end;
        }
    }

    // Step 2: Parallel matches where src+len <= ready_end (Bernstein safe)
    // Split into parallel and sequential
    static thread_local size_t par_idx[131072];
    static thread_local size_t seq_idx[131072];
    size_t par_cnt=0, seq_cnt=0;
    for (size_t i = 0; i < ops_cnt; i++) {
        if (!ops_buf[i].is_lit) {
            const DecOp& o = ops_buf[i];
            if (o.src_off + o.len <= ready_end && o.len <= (o.dst_off - o.src_off)) {
                par_idx[par_cnt++] = i;
            } else {
                seq_idx[seq_cnt++] = i;
            }
        }
    }

    // Parallel copies — Bernstein conditions verified
    #pragma omp parallel for schedule(static) num_threads(2)
    for (size_t i = 0; i < par_cnt; i++) {
        const DecOp& o = ops_buf[par_idx[i]];
        memcpy(dst+o.dst_off, dst+o.src_off, o.len);
    }

    // Sequential fallback for dependent copies
    for (size_t i = 0; i < seq_cnt; i++) {
        const DecOp& o = ops_buf[seq_idx[i]];
        copy_match(dst, o.dst_off, o.dst_off-o.src_off, o.len);
    }
}

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
 
static void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto ror=[](uint32_t x,int n){ return (x>>n)|(x<<(32-n)); };
    size_t total=(len+9+63)&~63ULL;
    uint8_t* buf=(uint8_t*)calloc(total,1);
    if(!buf) return;
    memcpy(buf,data,len); buf[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) buf[total-1-i]=(uint8_t)(bits>>(i*8));
    for(size_t off=0;off<total;off+=64) {
        uint32_t w[64];
        for(int i=0;i<16;i++)
            w[i]=((uint32_t)buf[off+i*4]<<24)|((uint32_t)buf[off+i*4+1]<<16)|
                 ((uint32_t)buf[off+i*4+2]<<8)|(uint32_t)buf[off+i*4+3];
        for(int i=16;i<64;i++) {
            uint32_t s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++) {
            uint32_t S1=ror(e,6)^ror(e,11)^ror(e,25);
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K256[i]+w[i];
            uint32_t S0=ror(a,2)^ror(a,13)^ror(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    free(buf);
    for(int i=0;i<8;i++) {
        out[i*4+0]=(uint8_t)(h[i]>>24); out[i*4+1]=(uint8_t)(h[i]>>16);
        out[i*4+2]=(uint8_t)(h[i]>>8);  out[i*4+3]=(uint8_t)(h[i]);
    }
}
 
static void sha256_hex(const uint8_t* data, size_t len, char out[65]) {
    uint8_t d[32]; sha256(data,len,d);
    for(int i=0;i<32;i++) sprintf(out+i*2,"%02x",d[i]); out[64]=0;
}
 
static uint8_t* zstd_comp(const uint8_t* src, size_t sz, size_t& out_sz, int lv) {
    size_t b=ZSTD_compressBound(sz); uint8_t* buf=(uint8_t*)malloc(b);
    if(!buf){out_sz=0;return nullptr;}
    out_sz=ZSTD_compress(buf,b,src,sz,lv);
    if (ZSTD_isError(out_sz)) { free(buf); out_sz=0; return nullptr; }
    return buf;
}
 
#pragma pack(push,1)
struct AetHeader {
    char     magic[8];
    uint32_t version;
    uint64_t orig_size;
    uint32_t block_size;
    uint32_t num_blocks;
    uint8_t  xxhash[8];  // XXH3_64bits
    uint64_t zlit_sz, zoff_sz, zlen_sz, zcmd_sz;
};

// ---- Shared archive validation. Called from BOTH decode paths (CLI do_decompress
// and the aceapex_decompress API), so they cannot drift apart again.
// A single corrupted byte in the header or the BlockOffsets table used to send stream
// pointers into arbitrary memory (SIGSEGV). Absolute offsets make this cheap: every
// bound is a constant known before decoding starts, so all of this runs once per
// archive and costs nothing in the hot loop.
// Empty archive (28.09): an empty input is a valid archive of exactly one header,
// num_blocks == 0, orig_size == 0, all four stream sizes 0, block_size nonzero
// (encoders write 65536). Before this the API returned 0 bytes for an empty input,
// which is "no archive" - a caller could not round-trip it (lzbench CI, tiny inputs).
static inline bool ax_is_empty_archive(const AetHeader& h) {
    return h.num_blocks == 0 && h.orig_size == 0 && h.block_size != 0 &&
           h.zlit_sz == 0 && h.zoff_sz == 0 && h.zlen_sz == 0 && h.zcmd_sz == 0;
}
static inline void ax_empty_header(AetHeader& h) {
    memset(&h, 0, sizeof(h)); memcpy(h.magic, "ACEPX2\0\0", 8); h.version = 2; h.block_size = 65536;
    uint64_t hv = OUR_CHECKSUM(nullptr, 0); memcpy(h.xxhash, &hv, 8);
}
static bool ax_header_ok(const AetHeader& h, uint64_t src_size) {
    if (h.num_blocks == 0) return ax_is_empty_archive(h) && src_size >= sizeof(AetHeader);
    if (h.block_size == 0) return false;
    if ((uint64_t)h.num_blocks * (uint64_t)h.block_size < h.orig_size) return false;
    uint64_t need = (uint64_t)sizeof(AetHeader)
                  + (uint64_t)h.num_blocks * sizeof(BlockOffsets)
                  + h.zlit_sz + h.zoff_sz + h.zlen_sz + h.zcmd_sz;
    return need <= src_size;
}

static bool ax_boffs_ok(const BlockOffsets* b, uint64_t nb,
                        uint64_t ls, uint64_t os, uint64_t ns, uint64_t cs) {
    for (uint64_t i = 0; i < nb; i++) {
        const BlockOffsets& o = b[i];
        if (o.lit_off > ls || o.lit_sz > ls - o.lit_off) return false;
        if (o.off_off > os || o.off_sz > os - o.off_off) return false;
        if (o.len_off > ns || o.len_sz > ns - o.len_off) return false;
        if (o.cmd_off > cs || o.cmd_sz > cs - o.cmd_off) return false;
    }
    return true;
}
#pragma pack(pop)
 
static double now_sec() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
 
struct DecArgs {
    const uint8_t* lit; const uint8_t* off;
    const uint8_t* len; const uint8_t* cmd;
    const BlockOffsets* boffs;
    uint8_t* dst; size_t dst_size;
    size_t bid_start; size_t bid_end;
    size_t block_size;
};
 
static void* dec_worker(void* arg) {
    DecArgs* a = (DecArgs*)arg;
    for (size_t b = a->bid_start; b < a->bid_end; b++) {
        const BlockOffsets& bo = a->boffs[b];
        size_t bstart = b * a->block_size;
        size_t bsize  = a->dst_size > bstart ?
                        std::min<size_t>((size_t)a->block_size, a->dst_size - bstart) : 0;
        if (bsize > 0) {
#ifdef ANALYZE_DEPS
            analyze_depth(bsize,
                a->off + bo.off_off, bo.off_sz,
                a->len + bo.len_off, bo.len_sz,
                a->cmd + bo.cmd_off, bo.cmd_sz, b);
#endif
            decompress_streams(
                a->dst + bstart, bsize,
                a->lit + bo.lit_off, bo.lit_sz,
                a->off + bo.off_off, bo.off_sz,
                a->len + bo.len_off, bo.len_sz,
                a->cmd + bo.cmd_off, bo.cmd_sz);
        }
    }
    return nullptr;
}
 
static size_t compute_block_size(size_t src_size, int threads) {
    { const char* _e=getenv("ACEAPEX_BS"); if(_e){ size_t _v=strtoull(_e,0,10); if(_v>=4096) return _v; } }
    const size_t MIN_BS = 256*1024, MAX_BS = 1*1024*1024;
    if (threads < 1) threads = 1;
    size_t want_blocks = (size_t)threads * 4;
    size_t blocks_at_max = (src_size + MAX_BS - 1) / MAX_BS;
    if (blocks_at_max >= want_blocks) return MAX_BS; // 1MB enough -> best ratio, exact baseline
    size_t bs = (src_size + want_blocks - 1) / want_blocks;
    if (bs > MAX_BS) bs = MAX_BS;
    bs &= ~((size_t)65536 - 1); // round to 64KB
    if (bs < MIN_BS) bs = MIN_BS;
    return bs;
}

// Объявлены до encode_file: определения ниже (dna_worth ~910, флаг ~1040).
static bool dna_worth(const uint8_t* s, size_t n);

static bool encode_file(const uint8_t* src, size_t src_size, int threads, int level,
    std::vector<BlockOffsets>& boffs,
    uint8_t*& raw_lit, size_t& total_lit,
    uint8_t*& raw_off, size_t& total_off,
    uint8_t*& raw_len, size_t& total_len,
    uint8_t*& raw_cmd, size_t& total_cmd,
    size_t& num_blocks)
{
    g_enc_threads = threads > 0 ? threads : 1;              // entropy stage keeps to this (2.2.1)
    g_block_size = compute_block_size(src_size, threads);
    // Подсказка для lit_chunk_size(): проверяем сам вход, не литералы.
    if(!getenv("LIT_CHUNK"))
        g_input_is_dna = dna_worth(src, src_size < (1u<<22) ? src_size : (1u<<22)) ? 1 : -1;
    num_blocks = (src_size + g_block_size - 1) / g_block_size;
    boffs.resize(num_blocks);
 
    // Adaptive hash size
    uint32_t hash_log = (src_size < 16*1024*1024) ? 13 :
                        (src_size < 128*1024*1024) ? 15 : 17;
    // Encoder (ADR-020): l1 for DNA by default, the chain matcher for everything else.
    // AX_ENC=l1 / AX_ENC=chain force one; the probe is the same 4 MiB dna_worth sample.
    const char* ax_enc=getenv("AX_ENC");
    // level 3 asks for l1 on any input (the lzbench level of the l1 encoder).
    int l1 = ax_enc ? !strcmp(ax_enc,"l1")
                    : level == 3 || dna_worth(src, src_size < (1u<<22) ? src_size : (1u<<22));
    if (l1) hash_log = 13;
    { const char* e=getenv("AX_HLOG"); if(e&&atoi(e)>=8&&atoi(e)<=24) hash_log=(uint32_t)atoi(e); }
    uint32_t hash_mask = (1u << hash_log) - 1;
    size_t ht_sz = (hash_mask+1);
    uint32_t chain_mask = 1; while ((size_t)chain_mask + 1 < g_block_size) chain_mask = chain_mask * 2 + 1;   // one link per block position
    ThreadHashTable** htabs=(ThreadHashTable**)calloc(threads,sizeof(ThreadHashTable*));
    if(!htabs){return false;}
    for(int i=0;i<threads;i++) {
        htabs[i]=(ThreadHashTable*)calloc(1,sizeof(ThreadHashTable));
        if(!htabs[i]){return false;}
        htabs[i]->pos  =(uint32_t*)calloc(ht_sz,sizeof(uint32_t));
        htabs[i]->epoch=(uint32_t*)calloc(ht_sz,sizeof(uint32_t));
        htabs[i]->chain=(uint32_t*)malloc(((size_t)chain_mask+1)*sizeof(uint32_t));
        if(!htabs[i]->pos||!htabs[i]->epoch||!htabs[i]->chain){return false;}
        memset(htabs[i]->chain,0xFF,((size_t)chain_mask+1)*sizeof(uint32_t));
        htabs[i]->cur_epoch=0;
        htabs[i]->hash_mask=hash_mask;
        htabs[i]->chain_mask=chain_mask;
        htabs[i]->max_attempts=(level>=2)?32:4;
        htabs[i]->l1=l1; if (l1) htabs[i]->max_attempts=1;
        // l1 knobs; AX_NOFLAT / AX_MINL / AX_SKIP override (measurement only)
        htabs[i]->noflat = getenv("AX_NOFLAT") ? atoi(getenv("AX_NOFLAT")) : l1;
        htabs[i]->minl = !l1 ? 0 : getenv("AX_MINL") ? (uint32_t)atoi(getenv("AX_MINL")) : 32;
        htabs[i]->skip = !l1 ? 0 : getenv("AX_SKIP") ? (uint32_t)atoi(getenv("AX_SKIP")) : 4;
        { const char* e=getenv("AX_ATT"); if(e) htabs[i]->max_attempts=atoi(e); }
    }
    BlockResult* results=(BlockResult*)calloc(num_blocks,sizeof(BlockResult));
    if(!results){return false;}
    PoolState pool;
    pool.src=src; pool.src_size=src_size;
    pool.num_blocks=num_blocks; pool.results=results;
    pool.next_block.store(0);
    WorkerArgs* wargs=(WorkerArgs*)calloc(threads,sizeof(WorkerArgs));
    pthread_t* pts=(pthread_t*)calloc(threads,sizeof(pthread_t));
    if(!wargs||!pts){free(results);return false;}
    for(int i=0;i<threads;i++) { wargs[i].thread_id=i; wargs[i].htab=htabs[i]; wargs[i].pool=&pool; }
    for(int i=1;i<threads;i++) ax_thread(&pts[i],worker_func,&wargs[i]);   // worker 0 is the caller
    worker_func(&wargs[0]);
    for(int i=1;i<threads;i++) pthread_join(pts[i],nullptr);

    total_lit=0; total_off=0; total_len=0; total_cmd=0;
    for(size_t b=0;b<num_blocks;b++) {
        if(results[b].overflow==99) { free(results); return false; }
        boffs[b].lit_off=total_lit; boffs[b].lit_sz=results[b].lit_size;
        boffs[b].off_off=total_off; boffs[b].off_sz=results[b].off_size;
        boffs[b].len_off=total_len; boffs[b].len_sz=results[b].len_size;
        boffs[b].cmd_off=total_cmd; boffs[b].cmd_sz=results[b].cmd_size;
        total_lit+=results[b].lit_size; total_off+=results[b].off_size;
        total_len+=results[b].len_size; total_cmd+=results[b].cmd_size;
    }

    // at least 1 byte: an empty stream (zeros, tiny input) must not look like a failed malloc(0)
    raw_lit=(uint8_t*)malloc(total_lit?total_lit:1);
    raw_off=(uint8_t*)malloc(total_off?total_off:1);
    raw_len=(uint8_t*)malloc(total_len?total_len:1);
    raw_cmd=(uint8_t*)malloc(total_cmd?total_cmd:1);
    if(!raw_lit||!raw_off||!raw_len||!raw_cmd){free(results);return false;}

    size_t li=0,oi=0,ni=0,ci=0;
    for(size_t b=0;b<num_blocks;b++) {
        memcpy(raw_lit+li,results[b].lit_buf,results[b].lit_size); li+=results[b].lit_size;
        memcpy(raw_off+oi,results[b].off_buf,results[b].off_size); oi+=results[b].off_size;
        memcpy(raw_len+ni,results[b].len_buf,results[b].len_size); ni+=results[b].len_size;
        memcpy(raw_cmd+ci,results[b].cmd_buf,results[b].cmd_size); ci+=results[b].cmd_size;
        free(results[b].lit_buf); free(results[b].off_buf);
        free(results[b].len_buf); free(results[b].cmd_buf);
    }
    for(int i=0;i<threads;i++) {
        free(htabs[i]->pos); free(htabs[i]->epoch); free(htabs[i]->chain);
        free(htabs[i]);
    }
    free(htabs); free(results); free(wargs); free(pts);
    return true;
}
 
// Fail-closed decode: every zstd frame must decode without error and to exactly the
// expected size. Parallel paths raise g_dec_err, which entry points reset and check;
// the serial range paths return nullptr. Before this an error left malloc garbage in
// the output and the call reported success (region on a legacy archive, 21.09).
static inline bool zdec_ok(void* dst,size_t raw,const void* src,size_t csz){
    size_t r=ZSTD_decompress(dst,raw,src,csz); return !ZSTD_isError(r)&&r==raw; }

static void parallel_decode(
    const uint8_t* lit, const uint8_t* off,
    const uint8_t* len, const uint8_t* cmd,
    const BlockOffsets* boffs, size_t num_blocks,
    uint8_t* dst, size_t dst_size, size_t block_size,
    int nthreads = 0)
{
    if (nthreads <= 0) { nthreads = (int)std::thread::hardware_concurrency(); if (nthreads < 1) nthreads = 8; }
    size_t nt = std::min<size_t>((size_t)nthreads, num_blocks);
    std::vector<DecArgs> dargs(nt);
    size_t blocks_per_thread = (num_blocks + nt - 1) / nt;
    for(size_t t=0;t<nt;t++) {
        size_t bstart = t * blocks_per_thread;
        size_t bend   = std::min<size_t>(bstart + blocks_per_thread, num_blocks);
        dargs[t]={lit,off,len,cmd,boffs,dst,dst_size,bstart,bend,block_size};
    }
    if (nt == 1) { dec_worker(&dargs[0]); return; }   // threads=1: nothing is spawned
    std::vector<pthread_t> dpts(nt);
    for(size_t t=0;t<nt;t++) ax_thread(&dpts[t],dec_worker,&dargs[t]);
    for(size_t t=0;t<nt;t++) pthread_join(dpts[t],nullptr);
}
 
// Helper: chunked FSE decompress a stream
// Format: [8:orig_sz][nc*8:csizes][chunks...]
// Chunk entry: bits 0..47 stored size, bit 63 = stored raw, bit 62 = 32-lane rANS chunk
// (ax_rans.h, the zstd-free "rANS token profile", ADR-018). Bits 48..61 are reserved: a
// set bit, or 62 and 63 together, is a corrupt archive.
// Chunk tables live at arbitrary offsets inside the archive: read them byte-wise.
struct AxU64s { const uint8_t* p; uint64_t operator[](size_t i) const { uint64_t v; memcpy(&v, p + 8 * i, 8); return v; } };
static inline size_t ax_ce_size(uint64_t e){ return (size_t)(e & (((uint64_t)1<<48)-1)); }
static inline bool   ax_ce_raw (uint64_t e){ return (e >> 63) != 0; }
static inline bool   ax_ce_rans(uint64_t e){ return ((e >> 62) & 1) != 0; }
static inline bool   ax_ce_bad (uint64_t e){ return ((e >> 48) & 0x3FFF) != 0 || (ax_ce_raw(e) && ax_ce_rans(e)); }
static inline bool   ax_profile_open(){ const char* e=getenv("AX_PROFILE"); return e && !strcmp(e,"open"); }
static inline bool   ax_tok_rans(){ const char* e=getenv("AX_TOK"); return (e && !strcmp(e,"rans")) || ax_profile_open(); }
// zstd-free literal chunks (ax_lit_open.h, ADR-019): mode 2 open DNA pack, mode 3 open plain
static inline bool   ax_lit_open(){ const char* e=getenv("AX_LIT"); return (e && !strcmp(e,"open")) || ax_profile_open(); }
// decode one token chunk of any kind; false on any error
static inline bool ax_tok_chunk(uint64_t e, uint8_t* dst, size_t raw, const uint8_t* p, size_t z){
    if (ax_ce_bad(e)) return false;
    if (ax_ce_raw(e)) { memcpy(dst, p, raw); return true; }
    if (ax_ce_rans(e)) return axr_decode(p, z, dst, raw) == 0;
    return zdec_ok(dst, raw, p, z);
}
static size_t fse_chunk_size(){
    const char* e=getenv("FSE_CHUNK");
    if(e){ size_t v=strtoull(e,0,10); if(v>=4096){ v&=~(size_t)4095; if(v>((size_t)16<<20)) v=(size_t)16<<20; return v; } }
    return 512*1024;
}
// The first word of an FSE stream carries the stream's own size in bits 0..47 and
// the chunk size, as CHUNK/4096, in bits 48..62; bit 63 stays a flag. The chunk
// size therefore travels IN THE ARCHIVE: a reader no longer has to be told it
// through the environment, which is the same rule LIT_CHUNK already follows.
// Zero in the chunk field is an archive written before the field existed, and it
// decodes with the environment value exactly as it did.
// A 48-bit size caps one stream at 256 TB, far above any input the format takes.
static inline size_t fse_stream_size(const uint8_t* src){
    uint64_t h; memcpy(&h,src,8); return (size_t)(h & (((uint64_t)1<<48)-1));
}
static inline size_t fse_stream_chunk(const uint8_t* src){
    uint64_t h; memcpy(&h,src,8);
    size_t c=(size_t)((h>>48) & 0x7FFF);
    return c ? (c<<12) : fse_chunk_size();
}

// Validate a stored token stream against its stored length before anything reads it:
// the chunk table must fit, every entry must be well-formed, and the chunks must end
// inside the stream. A stream shorter than 8 bytes is the empty stream. Without this a
// corrupted (or future-format) chunk size sent the readers past the buffer.
static bool ax_fse_check(const uint8_t* z, size_t zsz, size_t* orig) {
    if (zsz < 8) { *orig = 0; return true; }
    size_t S = fse_stream_size(z), CH = fse_stream_chunk(z);
    uint64_t h; memcpy(&h, z, 8); if (h >> 63) return false;
    if (!CH) return false;
    size_t nc = (S + CH - 1) / CH;
    if (nc > (zsz - 8) / 8) return false;
    uint64_t sum = 8 + (uint64_t)nc * 8;
    for (size_t i = 0; i < nc; i++) {
        uint64_t e; memcpy(&e, z + 8 + 8 * i, 8);
        if (ax_ce_bad(e)) return false;
        size_t raw = std::min<size_t>(CH, S - i * CH);
        sum += ax_ce_raw(e) ? raw : ax_ce_size(e);
        if (sum > zsz) return false;
    }
    *orig = S; return true;
}

static void fse_chunked_decomp(const uint8_t* src, size_t orig_sz, uint8_t* dst) {
    const size_t CHUNK=fse_stream_chunk(src);
    const AxU64s cs{src + 8};
    size_t nc = (orig_sz + CHUNK - 1) / CHUNK;
    size_t p_off = 8 + nc * 8;
    for (size_t i = 0; i < nc; i++) {
        size_t raw = std::min<size_t>(CHUNK, orig_sz - i * CHUNK);
        const uint8_t* p = src + p_off;
        size_t z = ax_ce_raw(cs[i]) ? raw : ax_ce_size(cs[i]);
        if (!ax_tok_chunk(cs[i], dst + i * CHUNK, raw, p, z)) g_dec_err=1;
        p_off += z;
    }
}

// Several FSE streams through one pool of `threads` workers: every chunk of every
// stream is one job, taken dynamically, so the largest stream no longer sets the
// floor of the entropy phase. `#pragma omp` here was dead code (no -fopenmp) and
// with -fopenmp it oversubscribed the cores (one pool per stream): measured 42 ms
// against 32 ms sequential on 8 threads, silesia. Each job is a whole chunk
// (4 KiB..512 KiB decoded), so the atomic counter costs nothing next to zstd.
struct AxTokJob { const uint8_t* p; uint8_t* d; size_t raw; size_t z; uint64_t e; };
struct FseStream { const uint8_t* s; size_t orig; uint8_t* d; };
static void fse_multi_decomp(const FseStream* st, int n, int threads) {
    std::vector<AxTokJob> jobs;
    for (int k = 0; k < n; k++) {
        if (st[k].orig == 0) continue;
        const size_t CHUNK=fse_stream_chunk(st[k].s);
        const AxU64s cs{st[k].s + 8};
        size_t nc = (st[k].orig + CHUNK - 1) / CHUNK, p_off = 8 + nc * 8;
        for (size_t i = 0; i < nc; i++) {
            AxTokJob j; j.raw = std::min<size_t>(CHUNK, st[k].orig - i * CHUNK);
            j.p = st[k].s + p_off; j.d = st[k].d + i * CHUNK; j.e = cs[i];
            j.z = ax_ce_raw(cs[i]) ? j.raw : ax_ce_size(cs[i]);
            p_off += j.z; jobs.push_back(j);
        }
    }
    if (jobs.empty()) return;
    if (threads < 1) threads = 1;
    if ((size_t)threads > jobs.size()) threads = (int)jobs.size();
    struct Pool { std::vector<AxTokJob>* jobs; std::atomic<size_t> next; };
    Pool pool{&jobs, {0}};
    auto fn=[](void* a)->void* { Pool* q=(Pool*)a; size_t n=q->jobs->size();
        for (size_t i; (i = q->next.fetch_add(1)) < n; ) { const AxTokJob& j=(*q->jobs)[i];
            if (!ax_tok_chunk(j.e, j.d, j.raw, j.p, j.z)) g_dec_err=1; }
        return nullptr; };
    std::vector<pthread_t> pts(threads - 1);
    for (int t = 0; t < threads - 1; t++) ax_thread(&pts[t],fn, &pool);
    fn(&pool);
    for (int t = 0; t < threads - 1; t++) pthread_join(pts[t], nullptr);
}

// Split an entropy thread budget between the literal stream and the token streams
// by DECODED bytes (zstd decode time follows output, not input); each side gets at
// least one thread.
static size_t ax_lit_decoded_size(const uint8_t* zlit, size_t zlit_sz) {
    if (!zlit || zlit_sz < 8) return 0;
    uint64_t h; memcpy(&h, zlit, 8);
    return (size_t)(h & ~((uint64_t(1)<<62)|(uint64_t(1)<<61)|(uint64_t(1)<<60)));
}
static void ax_entropy_split(size_t zlit, size_t ztok, int budget, int& lit_t, int& tok_t) {
    if (budget < 2) { lit_t = tok_t = 1; return; }
    double f = (zlit + ztok) ? (double)ztok / (double)(zlit + ztok) : 0.5;
    tok_t = (int)(budget * f + 0.5); if (tok_t < 1) tok_t = 1; if (tok_t > budget - 1) tok_t = budget - 1;
    lit_t = budget - tok_t;
}

// Parallel entropy encode — 4 streams simultaneously
static int lit_lanes(){
    const char* e=getenv("LIT_LANES");
    if(e){ int v=atoi(e); if(v>0) return v; }
    unsigned n=std::thread::hardware_concurrency();     // online CPUs (sysconf is not on Windows)
    return n>0 ? (int)n : 8;
}

static int lit_level(){
    const char* e=getenv("LIT_LEVEL");
    if(e){ int v=atoi(e); if(v>=-5 && v<=19 && v!=0) return v; }
    return 3;
}

static bool dna_worth(const uint8_t* s, size_t n){
    if(n < 4096) return false;
    size_t step = n>65536 ? n/65536 : 1, bad=0, cnt=0;
    for(size_t i=0;i<n;i+=step){ uint8_t u=s[i]&0xDF;
        if(u!='A'&&u!='C'&&u!='G'&&u!='T') bad++;
        cnt++; }
    return cnt && (double)bad/(double)cnt < 0.10;
}

static uint8_t* dna_compress(const uint8_t* s, size_t n, size_t& out_sz){
    size_t np=(n+3)/4, nc=(n+7)/8, nexc=0;
    for(size_t i=0;i<n;i++){ uint8_t u=s[i]&0xDF;
        if(u!='A'&&u!='C'&&u!='G'&&u!='T') nexc++; }
    uint8_t* seq=(uint8_t*)calloc(np?np:1,1);
    uint8_t* cse=(uint8_t*)calloc(nc?nc:1,1);
    uint32_t* gap=(uint32_t*)malloc((nexc?nexc:1)*4);
    uint8_t*  val=(uint8_t*)malloc(nexc?nexc:1);
    if(!seq||!cse||!gap||!val){free(seq);free(cse);free(gap);free(val);out_sz=0;return nullptr;}
    size_t e=0, prev=0;
    for(size_t i=0;i<n;i++){
        uint8_t b=s[i], u=b&0xDF, code=0;
        if(u=='A') code=0; else if(u=='C') code=1;
        else if(u=='G') code=2; else if(u=='T') code=3;
        else { gap[e]=(uint32_t)(i-prev); prev=i; val[e]=b; e++; }
        seq[i>>2] |= (uint8_t)(code << (6-2*(i&3)));
        if(b>=0x61 && b<=0x7a) cse[i>>3] |= (uint8_t)(0x80>>(i&7));
    }
    size_t c1=ZSTD_compressBound(np)+8, c2=ZSTD_compressBound(nc)+8;
    size_t c3=ZSTD_compressBound(nexc*4)+8, c4=ZSTD_compressBound(nexc)+8;
    uint8_t* buf=(uint8_t*)malloc(20+c1+c2+c3+c4);
    if(!buf){free(seq);free(cse);free(gap);free(val);out_sz=0;return nullptr;}
    uint8_t* p=buf+20;
    int L=lit_level();
    size_t s1=ZSTD_compress(p,c1,seq,np,L); p+=s1;
    size_t s2=ZSTD_compress(p,c2,cse,nc,L); p+=s2;
    size_t s3=nexc?ZSTD_compress(p,c3,gap,nexc*4,L):0; p+=s3;
    size_t s4=nexc?ZSTD_compress(p,c4,val,nexc,L):0; p+=s4;
    uint32_t h[5]={(uint32_t)nexc,(uint32_t)s1,(uint32_t)s2,(uint32_t)s3,(uint32_t)s4};
    memcpy(buf,h,20);
    free(seq);free(cse);free(gap);free(val);
    out_sz=20+s1+s2+s3+s4;
    return buf;
}

static void dna_decompress(const uint8_t* src, size_t src_sz, uint8_t* dst, size_t n){
    // framing checked before any read: 20-byte header, sub-frames inside the chunk, nexc <= n
    if(src_sz<20){ g_dec_err=1; return; }
    uint32_t h[5]; memcpy(h,src,20);
    if((uint64_t)20+h[1]+h[2]+h[3]+h[4]>src_sz || h[0]>n){ g_dec_err=1; return; }
    size_t nexc=h[0], np=(n+3)/4, nc=(n+7)/8;
    const uint8_t* p=src+20;
    uint8_t* seq=(uint8_t*)malloc(np?np:1);
    uint8_t* cse=(uint8_t*)malloc(nc?nc:1);
    uint32_t* gap=(uint32_t*)malloc((nexc?nexc:1)*4);
    uint8_t*  val=(uint8_t*)malloc(nexc?nexc:1);
    if(!zdec_ok(seq,np,p,h[1])) g_dec_err=1; p+=h[1];
    if(!zdec_ok(cse,nc,p,h[2])) g_dec_err=1; p+=h[2];
    if(h[3]&&!zdec_ok(gap,nexc*4,p,h[3])) g_dec_err=1; p+=h[3];
    if(h[4]&&!zdec_ok(val,nexc,p,h[4])) g_dec_err=1;
    // Таблица на 256 входов: один упакованный байт разворачивается в четыре
    // основания одним 32-битным store вместо четырёх сдвигов с условием.
    static uint32_t T4[256]; static bool T4_ready=false;
    if(!T4_ready){
        static const uint8_t B[4]={'A','C','G','T'};
        for(int v=0;v<256;v++){
            uint8_t q[4]={B[(v>>6)&3],B[(v>>4)&3],B[(v>>2)&3],B[v&3]};
            memcpy(&T4[v],q,4);
        }
        T4_ready=true;
    }
    size_t full=n>>2;
    for(size_t k=0;k<full;k++) memcpy(dst+4*k,&T4[seq[k]],4);
    for(size_t i=full<<2;i<n;i++){
        static const uint8_t B2[4]={'A','C','G','T'};
        dst[i]=B2[(seq[i>>2] >> (6-2*(i&3))) & 3];
    }
    // Регистр: байты, где бит маски выставлен, получают 0x20. Байт маски покрывает
    // восемь позиций, поэтому нулевой байт пропускается целиком.
    size_t nb8=(n+7)>>3;
    for(size_t k=0;k<nb8;k++){
        uint8_t m=cse[k];
        if(!m) continue;
        size_t base=k<<3, lim=(base+8<=n)?8:(n-base);
        for(size_t j=0;j<lim;j++) if(m & (0x80>>j)) dst[base+j]|=0x20;
    }
    size_t pos=0;
    for(size_t k=0;k<nexc;k++){ pos+=gap[k]; if(pos<n) dst[pos]=val[k]; }
    free(seq);free(cse);free(gap);free(val);
}


static uint8_t* lit_compress_legacy(const uint8_t* src, size_t sz, size_t& out_sz) {
    const int NW=4; size_t csz=(sz+NW-1)/NW;
    struct ZW{const uint8_t*in;size_t isz;uint8_t*out;size_t osz;size_t cap;};
    ZW zws[NW];
    for(int t=0;t<NW;t++){
        // Unsigned-underflow guard: the last worker's offset 3*ceil(sz/4) exceeds
        // sz for sz in {1,2,5,...}, so sz-off wrapped to ~2^64 -> ZSTD_compressBound
        // huge -> malloc failed -> the literal stream was SILENTLY DROPPED and decode
        // produced garbage. (lzbench issue #2, reported by inikep.)
        size_t off=(size_t)t*csz; if(off>sz) off=sz;
        size_t isz=(t<NW-1)?((off+csz<=sz)?csz:(sz-off)):(sz-off);
        zws[t]={src+off,isz,nullptr,0,ZSTD_compressBound(isz)+8};
        zws[t].out=(uint8_t*)malloc(zws[t].cap);
        if(!zws[t].out){out_sz=0;return nullptr;}}
    struct LPool{ ZW* w; std::atomic<int> next; };
    LPool lp{zws,{0}};
    auto zfn=[](void*a)->void*{LPool*q=(LPool*)a;
        for(int i; (i=q->next.fetch_add(1))<NW;){ ZW*z=&q->w[i];
            ZSTD_CCtx*ctx=ZSTD_createCCtx();
            if(!ctx){z->osz=0; continue;}
            ZSTD_CCtx_setParameter(ctx,ZSTD_c_compressionLevel,3);
            z->osz=ZSTD_compress2(ctx,z->out,z->cap,z->in,z->isz);
            ZSTD_freeCCtx(ctx); }
        return nullptr;};
    ax_run_pool(ax_enc_budget(NW),zfn,&lp);
    size_t hdrsz=8+NW*8,totalsz=hdrsz;
    for(int t=0;t<NW;t++) totalsz+=zws[t].osz;
    uint8_t* res=(uint8_t*)malloc(totalsz);
    if(!res){out_sz=0;return nullptr;}
    AX_write64(res, sz|(uint64_t(1)<<62));
    uint64_t* zsz=(uint64_t*)(res+8); uint8_t* p=res+hdrsz;
    for(int t=0;t<NW;t++){zsz[t]=zws[t].osz;memcpy(p,zws[t].out,zws[t].osz);p+=zws[t].osz;free(zws[t].out);}
    out_sz=totalsz; return res;
}

// Literals are cut into fixed-size chunks instead of NW equal shares, which makes
// unpacking partial: a region needs one chunk rather than a quarter of the file.
// Opt-in through LIT_CHUNK; without it the previous layout is used unchanged.
// Bit 61 of the stream header marks the new scheme, the chunk count follows it.
// Выставляется один раз в encode_file по выборке из входа.
// 0 = не проверяли, 1 = входные данные проходят dna_worth, -1 = нет.
thread_local int g_input_is_dna = 0;

static size_t lit_chunk_size(){
    const char* e=getenv("LIT_CHUNK");
    if(e){ size_t v=strtoull(e,0,10); return v<(1u<<16) ? 0 : v; }
    // Дефолт по данным (11.09). Чанкование литералов стоит таблиц zstd на кусок,
    // но открывает DNA-трансформ, который на чистой ДНК с лихвой окупает:
    //   chr1 3.18065 -> 3.70807 (+16.6%), FASTQ 3.96476 -> 3.85914 (-2.7%),
    //   enwik8 2.63791 -> 2.42272 (-8.2%), silesia 3.00457 -> 2.80984 (-6.5%).
    // FASTQ геномный, но наполовину строки качества — трансформ не срабатывает,
    // а цена чанкования остаётся. Порог по доле ACGT разделяет верно.
    // Профили (--profile) задают LIT_CHUNK явно и эту ветку не используют:
    // для регионального доступа чанкование обязательно на любых данных
    // (без него seek 46 мс против 0.082).
    // the open literal profile has no 4-part zstd layout: it always writes tagged chunks
    return (g_input_is_dna == 1 || ax_lit_open()) ? 65536 : 0;
}
#define LIT_CHUNK lit_chunk_size()

static uint8_t* lit_compress(const uint8_t* src, size_t sz, size_t& out_sz) {
    const size_t CH = LIT_CHUNK;
    if (CH == 0) return lit_compress_legacy(src, sz, out_sz);
    const int NW = (int)((sz + CH - 1) / CH);
    if (NW < 1 || NW > 65535) return lit_compress_legacy(src, sz, out_sz);
    size_t csz = CH;
    struct ZW{const uint8_t*in;size_t isz;uint8_t*out;size_t osz;size_t cap;};
    std::vector<ZW> zws(NW);
    for(int t=0;t<NW;t++){
        size_t off=(size_t)t*csz; if(off>sz) off=sz;
        size_t isz=(off+csz<=sz)?csz:(sz-off);
        zws[t]={src+off,isz,nullptr,0,std::max(ZSTD_compressBound(isz),axo_piece_bound(isz))+8};  // a piece writes rANS before choosing raw
        zws[t].out=(uint8_t*)malloc(zws[t].cap);
        if(!zws[t].out){out_sz=0;return nullptr;}}
    struct CPool{ ZW* w; int n; std::atomic<int> next; };
    CPool cpool{zws.data(),NW,{0}};
    auto zfn=[](void*a)->void*{
        CPool* p=(CPool*)a;
        ZSTD_CCtx* ctx=ZSTD_createCCtx();
        if(!ctx) return nullptr;
        ZSTD_CCtx_setParameter(ctx,ZSTD_c_compressionLevel,lit_level());
        for(;;){ int i=p->next.fetch_add(1); if(i>=p->n) break;
            ZW& z=p->w[i];
            // Байт режима перед данными: 0 = обычный zstd, 1 = DNA-трансформ.
            // Считаем оба и берём меньший, поэтому режим не может ухудшить размер.
            if(ax_lit_open()){                              // zstd-free: mode 3 plain, or mode 2 when smaller
                std::vector<uint16_t> scr(z.isz+2*AXR_LANES);
                size_t ps=axo_piece_encode(z.in,z.isz,z.out+1,scr.data()); z.out[0]=3; z.osz=ps+1;
                if(dna_worth(z.in,z.isz)){
                    std::vector<uint8_t> db(axo_dna_bound(z.isz));
                    size_t ds=axo_dna_encode(z.in,z.isz,db.data());
                    if(ds && ds<ps){ z.out[0]=2; memcpy(z.out+1,db.data(),ds); z.osz=ds+1; }
                }
                continue;
            }
            size_t zs=ZSTD_compress2(ctx,z.out+1,z.cap-1,z.in,z.isz);
            uint8_t* db=nullptr; size_t dsz=0;
            if(dna_worth(z.in,z.isz)) db=dna_compress(z.in,z.isz,dsz);
            if(db && dsz && dsz<zs){
                z.out[0]=1; memcpy(z.out+1,db,dsz); z.osz=dsz+1;
            } else {
                z.out[0]=0; z.osz=zs+1;
            }
            free(db);
        }
        ZSTD_freeCCtx(ctx); return nullptr;};
    ax_run_pool(ax_enc_budget(std::min(lit_lanes(),NW)),zfn,&cpool);
    size_t hdrsz=8+8+(size_t)NW*8,totalsz=hdrsz;
    for(int t=0;t<NW;t++) totalsz+=zws[t].osz;
    uint8_t* res=(uint8_t*)malloc(totalsz);
    if(!res){out_sz=0;return nullptr;}
    AX_write64(res, sz|(uint64_t(1)<<62)|(uint64_t(1)<<61)|(uint64_t(1)<<60));
    AX_write64(res+8, (uint64_t)CH);   // размер чанка, а не их число
    uint64_t* zsz=(uint64_t*)(res+16); uint8_t* p=res+hdrsz;
    for(int t=0;t<NW;t++){zsz[t]=zws[t].osz;memcpy(p,zws[t].out,zws[t].osz);p+=zws[t].osz;free(zws[t].out);}
    out_sz=totalsz; return res;
}
// Literal chunk table (spec 3.2.1/3.2.2) checked against the stored stream before use: chunk
// size > 0, table inside the stream, every chunk inside the stream. Returns NW, or -1.
static long long lit_table_check(const uint8_t* src, size_t src_sz, bool chunked, uint64_t orig_sz, size_t& csz){
    const size_t hdr = chunked ? 16 : 8;
    if(src_sz < hdr) return -1;
    if(chunked){ uint64_t c; memcpy(&c,src+8,8); if(!c) return -1; csz=(size_t)c; }
    else csz=(size_t)((orig_sz+3)/4);
    const uint64_t NW = chunked ? (csz ? (orig_sz+csz-1)/csz : 0) : 4;
    if(NW > (src_sz-hdr)/8) return -1;
    uint64_t rest = src_sz-hdr-8*NW;
    for(uint64_t t=0;t<NW;t++){ uint64_t z; memcpy(&z,src+hdr+8*t,8); if(z>rest) return -1; rest-=z; }
    return (long long)NW;
}
static uint8_t* lit_decompress(const uint8_t* src, size_t src_sz, size_t& orig_sz, int lanes_req = 0) {
    // An empty or truncated literal stream must not be read as an 8-byte header.
    // Tiny inputs give zlit_sz==0; the out-of-bounds read corrupted heap metadata
    // and surfaced as a double-free thousands of calls later. (lzbench issue #2.)
    if (!src || src_sz < 8) { orig_sz = 0; return (uint8_t*)malloc(1); }
    uint64_t h; memcpy(&h, src, 8);   // alignment-safe (no AX_read64 in this tree)
    const int NW_LEGACY=4;
    const bool chunked=(h & (uint64_t(1)<<61))!=0;
    const bool tagged=(h & (uint64_t(1)<<60))!=0;
    // spec 3.2: bit 63 is reserved, bit 60 needs 61; such a word is an error, not a size
    if((h>>63) || (tagged && !chunked)){ orig_sz=0; g_dec_err=1; return nullptr; }
    orig_sz=h & ~((uint64_t(1)<<62)|(uint64_t(1)<<61)|(uint64_t(1)<<60));
    size_t csz=0; long long nwc=-1;
    if(h & (uint64_t(1)<<62)){ nwc=lit_table_check(src,src_sz,chunked,orig_sz,csz);
        if(nwc<0 || nwc>INT32_MAX){ orig_sz=0; g_dec_err=1; return nullptr; } }
    uint8_t* out=(uint8_t*)malloc(orig_sz?orig_sz:1);
    if(!out) return nullptr;
    if(!(h & (uint64_t(1)<<62))){fse_chunked_decomp(src,orig_sz,out);return out;}
    // Размер чанка читается ИЗ ФАЙЛА: архив не должен зависеть от окружения читателя.
    (void)NW_LEGACY;
    const int NW = (int)nwc;
    const AxU64s zsz{src+(chunked?16:8)};
    const uint8_t* p0=src+(chunked?16:8)+(size_t)NW*8;
    struct DW{uint8_t*out;size_t raw;const uint8_t*in;size_t isz;bool tg;};
    std::vector<DW> dws(NW); const uint8_t* p=p0;
    for(int t=0;t<NW;t++){
        // Same underflow guard as in lit_compress (decoder side).
        size_t off=(size_t)t*csz; if(off>orig_sz) off=orig_sz;
        // Правило совпадает с кодером: кусок равен csz, кроме последнего.
        // Прежняя формула с (t<NW-1) была под legacy и на чанках расходилась.
        size_t raw=(off+csz<=orig_sz)?csz:(orig_sz-off);
        dws[t]={out+off,raw,p,(size_t)zsz[t],tagged}; p+=(size_t)zsz[t];}
    struct Pool{ DW* w; int n; std::atomic<int> next; };
    Pool pool{dws.data(),NW,{0}};
    auto dfn=[](void*a)->void*{
        Pool* p=(Pool*)a;
        for(;;){ int i=p->next.fetch_add(1); if(i>=p->n) break;
            DW& d=p->w[i]; if(!d.isz){ if(d.raw) g_dec_err=1; continue; }   // a chunk with bytes needs a body
            if(!d.tg){ if(!zdec_ok(d.out,d.raw,d.in,d.isz)) g_dec_err=1; continue; }
            if(d.in[0]==1) dna_decompress(d.in+1,d.isz-1,d.out,d.raw);
            else if(d.in[0]==2){ if(axo_dna_decode(d.in+1,d.isz-1,d.out,d.raw)) g_dec_err=1; }
            else if(d.in[0]==3){ if(axo_piece_decode(d.in+1,d.isz-1,d.out,d.raw)) g_dec_err=1; }
            else if(!zdec_ok(d.out,d.raw,d.in+1,d.isz-1)) g_dec_err=1; }
        return nullptr;};
    // LANES был жёстко 8; на машинах с бо́льшим числом ядер половина простаивала.
    // Чанки независимы как кадры zstd, пул динамический — берём по числу ядер.
    int hw=(int)std::thread::hardware_concurrency(); if(hw<1) hw=8;
    if(lanes_req>0) hw=lanes_req;
    const char* le=getenv("LIT_LANES_DEC"); if(le) hw=atoi(le);
    if(hw<1) hw=1;
    const int LANES=std::min(hw,NW);
    if(LANES<=1){ dfn(&pool); return out; }          // lanes=1: decode inline, spawn nothing
    std::vector<pthread_t> pts(LANES-1);
    for(int t=0;t<LANES-1;t++) ax_thread(&pts[t],dfn,&pool);
    dfn(&pool);
    for(int t=0;t<LANES-1;t++) pthread_join(pts[t],nullptr);
    return out;
}
// Partial stream unpacking for region reads: touch only the compressed chunks
// covering [from,to). Buffers are allocated at full stream size; bytes outside
// the requested span are left zero and are never read by the block decoder.
static uint8_t* fse_range(const uint8_t* src, size_t orig_sz, size_t from, size_t to,
                          size_t* base_off=nullptr) {
    const size_t CHUNK=fse_stream_chunk(src);
    const AxU64s cs{src+8};
    size_t nc=(orig_sz+CHUNK-1)/CHUNK;
    // Пустой диапазон: поток len не нужен 29% регионов (7478 блоков chr1 из 15 499
    // имеют len_sz=0, медиана 1 байт при таблице 16 байт на блок). Вызов ради нуля
    // байт стоит ~12 us по закону амплификации.
    if(to<=from){ if(base_off) *base_off=0; return (uint8_t*)calloc(1,1); }
    size_t c0=from/CHUNK, c1=(to?to-1:0)/CHUNK;
    if(c1>=nc) c1=nc?nc-1:0;
    size_t win_lo=c0*CHUNK, win_hi=std::min(orig_sz,(c1+1)*CHUNK);
    if(base_off) *base_off=win_lo;
    // malloc, не calloc: распакованные чанки перезаписывают окно целиком, а вне
    // запрошенного диапазона декодер блоков не читает. Зануление 130 KB на вызов
    // было самой дорогой операцией профиля (memset_avx512, 9.75%).
    uint8_t* out=(uint8_t*)malloc(win_hi>win_lo?win_hi-win_lo:1);
    if(!out) return nullptr;
    size_t p_off=8+nc*8;
    for(size_t i=0;i<nc;i++){
        size_t d_off=i*CHUNK;
        size_t raw=std::min<size_t>(CHUNK,orig_sz-d_off);
        size_t csz=ax_ce_raw(cs[i])?raw:ax_ce_size(cs[i]);
        if(d_off<to && d_off+raw>from){
            uint8_t* dst=out+(d_off-win_lo);
            if(!ax_tok_chunk(cs[i],dst,raw,src+p_off,csz)){ free(out); return nullptr; }
        }
        p_off+=csz;
    }
    return out;
}

// Литеральный поток: новая схема — чанки LIT_CHUNK, старая — NW равных долей.
static uint8_t* lit_range(const uint8_t* src, size_t src_sz, size_t& orig_sz,
                          size_t from, size_t to, size_t* base_off=nullptr) {
    if(!src||src_sz<8){orig_sz=0;return (uint8_t*)malloc(1);}
    uint64_t h; memcpy(&h,src,8);
    const bool chunked=(h&(uint64_t(1)<<61))!=0;
    const bool tagged=(h&(uint64_t(1)<<60))!=0;
    if((h>>63) || (tagged && !chunked)){ orig_sz=0; return nullptr; }   // spec 3.2 reserved combinations
    orig_sz=h&~((uint64_t(1)<<62)|(uint64_t(1)<<61)|(uint64_t(1)<<60));
    if(!(h&(uint64_t(1)<<62))) return fse_range(src,orig_sz,from,to);
    size_t csz=0; const long long nwc=lit_table_check(src,src_sz,chunked,orig_sz,csz);
    if(nwc<0 || nwc>INT32_MAX){ orig_sz=0; return nullptr; }
    const int NW=(int)nwc;
    const AxU64s zsz{src+(chunked?16:8)};
    const uint8_t* p=src+(chunked?16:8)+(size_t)NW*8;
    size_t win_lo=(from/csz)*csz;
    size_t win_hi=std::min(orig_sz,((to?to-1:0)/csz+1)*csz);
    if(win_hi<win_lo) win_hi=win_lo;
    if(base_off) *base_off=win_lo;
    // malloc, не calloc: распакованные чанки перезаписывают окно целиком, а вне
    // запрошенного диапазона декодер блоков не читает. Зануление 130 KB на вызов
    // было самой дорогой операцией профиля (memset_avx512, 9.75%).
    uint8_t* out=(uint8_t*)malloc(win_hi>win_lo?win_hi-win_lo:1);
    if(!out) return nullptr;
    for(int t=0;t<NW;t++){
        size_t off=(size_t)t*csz; if(off>orig_sz) off=orig_sz;
        // Правило то же, что в кодере и в lit_decompress: кусок равен csz,
        // кроме последнего. Старая формула с (t<NW-1) на чанках расходилась.
        size_t raw=(off+csz<=orig_sz)?csz:(orig_sz-off);
        if(off<to && off+raw>from){
            uint8_t* d2=out+(off-win_lo);
            if(!zsz[t]){ free(out); return nullptr; }                      // a chunk with bytes needs a body
            if(!tagged)        { if(!zdec_ok(d2,raw,p,(size_t)zsz[t])){ free(out); return nullptr; } }
            else if(p[0]==1)   { dna_decompress(p+1,(size_t)zsz[t]-1,d2,raw); if(g_dec_err){ free(out); return nullptr; } }
            else if(p[0]==2)   { if(axo_dna_decode(p+1,(size_t)zsz[t]-1,d2,raw)){ free(out); return nullptr; } }
            else if(p[0]==3)   { if(axo_piece_decode(p+1,(size_t)zsz[t]-1,d2,raw)){ free(out); return nullptr; } }
            else               { if(!zdec_ok(d2,raw,p+1,(size_t)zsz[t]-1)){ free(out); return nullptr; } }
        }
        p+=(size_t)zsz[t];
    }
    return out;
}

static void entropy_encode(
    const uint8_t* raw_lit, size_t total_lit,
    const uint8_t* raw_off, size_t total_off,
    const uint8_t* raw_len, size_t total_len,
    const uint8_t* raw_cmd, size_t total_cmd,
    uint8_t*& zlit, size_t& zlit_sz,
    uint8_t*& zoff, size_t& zoff_sz,
    uint8_t*& zlen, size_t& zlen_sz,
    uint8_t*& zcmd, size_t& zcmd_sz)
{
    struct EA{const uint8_t*in;size_t isz;uint8_t**out;size_t*osz;};
    auto ew=[](void*a)->void*{
        EA*e=(EA*)a;
        const bool rans=ax_tok_rans();
        // rANS profile: 64 KiB chunks unless FSE_CHUNK says otherwise (4 KiB chunks
        // carry a 32-lane state block and a table each: +5-6 % on chr1 tokens)
        const size_t CHUNK=(rans && !getenv("FSE_CHUNK")) ? (size_t)65536 : fse_chunk_size();
        size_t nc=(e->isz+CHUNK-1)/CHUNK;
        size_t hdrsz=8+nc*8;
        size_t cap=hdrsz+e->isz+nc*(64+axr_bound(0));
        std::vector<uint16_t> rw(rans ? CHUNK+2*AXR_LANES : 0);
        std::vector<uint8_t>  rb(rans ? axr_bound(CHUNK) : 0);
        *e->out=(uint8_t*)malloc(cap);
        if(!*e->out) return nullptr;
        AX_write64(*e->out, (uint64_t)e->isz|((uint64_t)(CHUNK>>12)<<48));
        uint64_t* csizes=(uint64_t*)(*e->out+8);
        uint8_t* p=*e->out+hdrsz;
        size_t total=hdrsz;
        for(size_t i=0;i<nc;i++){
            size_t off=i*CHUNK;
            size_t isz=std::min<size_t>(CHUNK,e->isz-off);
            if(rans){
                size_t r=axr_encode(e->in+off,isz,rb.data(),rw.data());
                if(!r||r>=isz){memcpy(p,e->in+off,isz);csizes[i]=isz|(uint64_t(1)<<63);total+=isz;p+=isz;}
                else{memcpy(p,rb.data(),r);csizes[i]=r|(uint64_t(1)<<62);total+=r;p+=r;}
                continue;
            }
            size_t b=ZSTD_compressBound(isz)+4;
            size_t r=ZSTD_compress(p,b,e->in+off,isz,1);
            if(!r||ZSTD_isError(r)){memcpy(p,e->in+off,isz);csizes[i]=isz|(uint64_t(1)<<63);total+=isz;p+=isz;}
            else{csizes[i]=r;total+=r;p+=r;}
        }
        *e->osz=total;
        return nullptr;
    };
    EA ea[3]={
        {raw_off,total_off,&zoff,&zoff_sz},
        {raw_len,total_len,&zlen,&zlen_sz},
        {raw_cmd,total_cmd,&zcmd,&zcmd_sz}
    };
    struct EPool{ EA* a; std::atomic<int> next; void* (*fn)(void*); };
    EPool ep{ea,{0},ew};
    ax_run_pool(ax_enc_budget(3),[](void* x)->void*{ EPool* q=(EPool*)x;
        for(int i; (i=q->next.fetch_add(1))<3;) q->fn(&q->a[i]);
        return nullptr; },&ep);
}
 
static int do_compress(const char* in_path, const char* out_path, int threads, int level=2) {
    double t_fread=now_sec();
#ifdef _WIN32
    FILE* fin_w=fopen(in_path,"rb");
    if(!fin_w){fprintf(stderr,"Cannot open: %s\n",in_path);return 1;}
    fseek(fin_w,0,SEEK_END); size_t src_size=(size_t)ftell(fin_w); fseek(fin_w,0,SEEK_SET);
    uint8_t* src=(uint8_t*)malloc(src_size);
    if(!src){fprintf(stderr,"malloc failed\n");fclose(fin_w);return 1;}
    fread(src,1,src_size,fin_w); fclose(fin_w);
    bool src_is_mmap=false;
#else
    int fin_fd=open(in_path,O_RDONLY);
    if (fin_fd<0) { fprintf(stderr,"Cannot open: %s\n",in_path); return 1; }
    struct stat fin_st; fstat(fin_fd,&fin_st);
    size_t src_size=(size_t)fin_st.st_size;
    if (src_size == 0) {                               // empty input -> empty archive (one header)
        close(fin_fd); AetHeader eh; ax_empty_header(eh);
        FILE* fo=fopen(out_path,"wb"); if(!fo){ fprintf(stderr,"Cannot write: %s\n",out_path); return 1; }
        fwrite(&eh,sizeof(eh),1,fo); fclose(fo);
        fprintf(stderr,"[*] Compress: %s (0 bytes) -> empty archive, %zu bytes\n",in_path,sizeof(eh)); return 0; }
    uint8_t* src=(uint8_t*)mmap(nullptr,src_size,PROT_READ,MAP_SHARED|MAP_POPULATE,fin_fd,0);
    close(fin_fd);
    if (src==MAP_FAILED) { fprintf(stderr,"mmap failed\n"); return 1; }
    bool src_is_mmap=true;
#endif
    t_fread=now_sec()-t_fread;

    fprintf(stderr,"[*] Compress: %s (%.2f MB) threads=%d\n",in_path,src_size/1e6,threads);
    double t_total_c=now_sec();
 
    std::vector<BlockOffsets> boffs;
    uint8_t *raw_lit,*raw_off,*raw_len,*raw_cmd;
    size_t total_lit,total_off,total_len,total_cmd,num_blocks;
    double t0=now_sec();
    encode_file(src,src_size,threads,level,boffs,
                raw_lit,total_lit,raw_off,total_off,
                raw_len,total_len,raw_cmd,total_cmd,
                num_blocks);
    double enc_time=now_sec()-t0;
 
    size_t zlit_sz,zoff_sz,zlen_sz,zcmd_sz;
    uint8_t *zlit,*zoff,*zlen,*zcmd;
    double t_lz=enc_time;
    double t1=now_sec();
    zlit=lit_compress(raw_lit,total_lit,zlit_sz);
    double t_lit=now_sec()-t1;
    double t2=now_sec();
    entropy_encode(raw_lit,total_lit,raw_off,total_off,raw_len,total_len,raw_cmd,total_cmd,
                   zlit,zlit_sz,zoff,zoff_sz,zlen,zlen_sz,zcmd,zcmd_sz);
    double t_fse=now_sec()-t2;
 
    size_t total_z=zlit_sz+zoff_sz+zlen_sz+zcmd_sz;
 
    AetHeader hdr;
    memcpy(hdr.magic,"ACEPX2\0\0",8);
    hdr.version=2; hdr.orig_size=(uint64_t)src_size;
    hdr.block_size=(uint32_t)g_block_size; hdr.num_blocks=(uint32_t)num_blocks;
    double t_sha256=now_sec();
    uint64_t hv=OUR_CHECKSUM(src,src_size);
    memcpy(hdr.xxhash,&hv,8);
    t_sha256=now_sec()-t_sha256;
    char sha_hex[17];
    uint64_t hv2; memcpy(&hv2,hdr.xxhash,8);
    sprintf(sha_hex,"%016llx",(unsigned long long)hv2);
    hdr.zlit_sz=zlit_sz; hdr.zoff_sz=zoff_sz;
    hdr.zlen_sz=zlen_sz; hdr.zcmd_sz=zcmd_sz;
 
    FILE* fout=fopen(out_path,"wb");
    fwrite(&hdr,sizeof(hdr),1,fout);
    fwrite(boffs.data(),sizeof(BlockOffsets),num_blocks,fout);
    fwrite(zlit,1,zlit_sz,fout); fwrite(zoff,1,zoff_sz,fout);
    fwrite(zlen,1,zlen_sz,fout); fwrite(zcmd,1,zcmd_sz,fout);
    fclose(fout);
    fprintf(stderr,"  Original:   %14zu bytes\n",src_size);
    fprintf(stderr,"  Compressed: %14zu bytes\n",total_z);
    fprintf(stderr,"  Ratio:  %.5fx\n",(double)src_size/total_z);
    double t3=now_sec();
    (void)(t3-t_total_c-t_lz-t_lit-t_fse);
    double real_enc=now_sec()-t_total_c;
    fprintf(stderr,"  Phase LZ77:    %.3fs\n",t_lz);
    fprintf(stderr,"  Phase lit/zstd:%.3fs\n",t_lit);
    fprintf(stderr,"  Phase FSE:     %.3fs\n",t_fse);
    fprintf(stderr,"  Phase fread:   %.3fs\n",t_fread);
    fprintf(stderr,"  Phase sha256:  %.3fs\n",t_sha256);
    fprintf(stderr,"  Phase other:   %.3fs\n",real_enc-t_lz-t_lit-t_fse-t_fread-t_sha256);
    fprintf(stderr,"  Encode: %.2f MB/s  (%.3fs)\n",src_size/real_enc/1e6,real_enc);
    fprintf(stderr,"  XXH3:   %s\n",sha_hex);
 
    #ifdef _WIN32
    free((void*)src);
#else
    if(src_is_mmap) munmap((void*)src,src_size); else free((void*)src);
#endif
    free(raw_lit); free(raw_off); free(raw_len); free(raw_cmd);
    free(zlit); free(zoff); free(zlen); free(zcmd);
    return 0;
}
 
static int do_decompress(const char* in_path, const char* out_path, int threads=8) {
    g_dec_err=0;
    double t_wall=now_sec();
    FILE* fin=fopen(in_path,"rb");
    if (!fin) { fprintf(stderr,"Cannot open: %s\n",in_path); return 1; }
    AetHeader hdr;
    fread(&hdr,sizeof(hdr),1,fin);
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) { fprintf(stderr,"Bad magic\n"); return 1; }
    // Версия писалась с первого дня и не проверялась ни разу. Архив, созданный
    // более новым кодером, старый декодер читал как валидный и выдавал мусор:
    // при интерпретации новых битов orig_size получалось 3.4 эксабайта.
    if (hdr.version != 2) {
        fprintf(stderr,"Unsupported format version %u (this build reads 2)\n", hdr.version);
        return 1;
    }
    fprintf(stderr,"[*] Decompress: %s -> %s\n",in_path,out_path);
 
    { // archive-level sanity before we trust any offset from the file
      fseek(fin,0,SEEK_END); long fsz=ftell(fin); fseek(fin,sizeof(hdr),SEEK_SET);
      if (fsz < 0 || !ax_header_ok(hdr,(uint64_t)fsz)) {
          fprintf(stderr,"Corrupt archive (header)\n"); fclose(fin); return 1; } }
    if (hdr.num_blocks == 0) {                        // empty archive: one header, no blocks
        fclose(fin); FILE* fo=fopen(out_path,"wb"); if(!fo){ fprintf(stderr,"Cannot write: %s\n",out_path); return 1; }
        fclose(fo); uint64_t dv=OUR_CHECKSUM(nullptr,0), hv3; memcpy(&hv3,hdr.xxhash,8);
        fprintf(stderr,"  Empty archive: 0 bytes written, %s\n", dv==hv3?"hash OK":"HASH MISMATCH"); return dv==hv3?0:1; }

    uint32_t nb=hdr.num_blocks;
    std::vector<BlockOffsets> boffs(nb);
    fread(boffs.data(),sizeof(BlockOffsets),nb,fin);
 
    uint8_t* zlit=(uint8_t*)malloc(hdr.zlit_sz);
    uint8_t* zoff=(uint8_t*)malloc(hdr.zoff_sz);
    uint8_t* zlen=(uint8_t*)malloc(hdr.zlen_sz);
    uint8_t* zcmd=(uint8_t*)malloc(hdr.zcmd_sz);
    if(!zlit||!zoff||!zlen||!zcmd){free(zlit);free(zoff);free(zlen);free(zcmd);fclose(fin);return 1;}
    fread(zlit,1,hdr.zlit_sz,fin); fread(zoff,1,hdr.zoff_sz,fin);
    fread(zlen,1,hdr.zlen_sz,fin); fread(zcmd,1,hdr.zcmd_sz,fin);
    fclose(fin);
 
    // Stream headers come from the file: a corrupted length would malloc() garbage.
    // Bound them by orig_size, which the header check already validated.
    // A stream shorter than its 8-byte size header can only be an EMPTY stream
    // (tiny inputs legitimately produce these), not a corrupt one: treat it as 0.
    size_t off_sz=0, len_sz=0, cmd_sz=0;
    if (!ax_fse_check(zoff,hdr.zoff_sz,&off_sz) || !ax_fse_check(zlen,hdr.zlen_sz,&len_sz) ||
        !ax_fse_check(zcmd,hdr.zcmd_sz,&cmd_sz)) {
        fprintf(stderr,"Corrupt archive (token stream table)\n");
        free(zlit);free(zoff);free(zlen);free(zcmd); return 1; }
    // Bound decoded stream sizes. NOT by orig_size: on tiny inputs the command
    // stream legitimately exceeds the payload (format overhead > data). Bound by a
    // generous multiple of orig_size plus a floor, which still rejects the garbage
    // lengths a corrupted byte produces (those are astronomically large).
    {
        uint64_t cap = hdr.orig_size * 4 + ((uint64_t)1 << 20);
        if (off_sz > cap || len_sz > cap || cmd_sz > cap) {
            fprintf(stderr,"Corrupt archive (stream size)\n");
            free(zlit);free(zoff);free(zlen);free(zcmd); return 1; }
    }
 
    double dec_time=now_sec();
    double t_lit=now_sec();
    // Run lit + fse decompress in parallel
    size_t lit_sz=0; uint8_t* lit=nullptr;
    uint8_t* off=(uint8_t*)malloc(off_sz);
    uint8_t* len=(uint8_t*)malloc(len_sz);
    uint8_t* cmd=(uint8_t*)malloc(cmd_sz);
    if(!off||!len||!cmd){free(off);free(len);free(cmd);free(zlit);free(zoff);free(zlen);free(zcmd);return 1;}
    // One thread budget for the whole entropy phase: the literal stream gets lit_t
    // lanes, the three token streams share one pool of tok_t workers (ADR-014).
    int lit_t, tok_t; ax_entropy_split(ax_lit_decoded_size(zlit,hdr.zlit_sz), off_sz+len_sz+cmd_sz, threads>0?threads:8, lit_t, tok_t);
    struct LitArg{const uint8_t*s;size_t sz;uint8_t**out;size_t*osz;int lanes;};
    LitArg larg={zlit,(size_t)hdr.zlit_sz,&lit,&lit_sz,lit_t};
    auto litfn=[](void*a)->void*{LitArg*l=(LitArg*)a;
        *l->out=lit_decompress(l->s,l->sz,*l->osz,l->lanes); return nullptr;};
    FseStream fst[3]={{zoff,off_sz,off},{zlen,len_sz,len},{zcmd,cmd_sz,cmd}};
    if (threads == 1) { litfn(&larg); fse_multi_decomp(fst,3,1); }
    else { pthread_t lt; ax_thread(&lt,litfn,&larg);
           fse_multi_decomp(fst,3,tok_t); pthread_join(lt,nullptr); }
    if(!lit){free(off);free(len);free(cmd);free(zlit);free(zoff);free(zlen);free(zcmd);return 1;}
    // Четыре потока формата (lit, off, len, cmd) распаковываются одновременно
    // в pthread выше; раздельного времени у них нет, и печатать две одинаковые
    // строки — вводить в заблуждение. Одно число: время самого долгого.
    double t_fse=now_sec()-t_lit;
    free(zlit); free(zoff); free(zlen); free(zcmd);
    // Выходной буфер большой, и его первое касание внутри parallel_decode
    // даёт 156 000 page fault на 254 MB. Просим huge pages: 94 000 вместо
    // 156 000, и главное — разброс wall падает со 182 до 9 единиц.
    // THP на этой машине в режиме madvise, поэтому просить обязательно.
    uint8_t* dst=nullptr;
    {
        size_t align = 2u<<20, sz = (hdr.orig_size + align - 1) & ~(align - 1);
#ifndef _WIN32
        if (posix_memalign((void**)&dst, align, sz) != 0) dst = nullptr;
#endif
#ifdef MADV_HUGEPAGE
        if (dst) madvise(dst, sz, MADV_HUGEPAGE);
#endif
        if (!dst) dst = (uint8_t*)malloc(hdr.orig_size);
    }
    if(!dst){free(lit);free(off);free(len);free(cmd);return 1;}
    // Last barrier: every block's slice must lie inside its decoded stream.
    if (!ax_boffs_ok(boffs.data(), nb, lit_sz, off_sz, len_sz, cmd_sz)) {
        fprintf(stderr,"Corrupt archive (block offsets)\n");
        free(lit);free(off);free(len);free(cmd);free(dst); return 1; }
    // ACEAPEX_DUMP=1: write streams.bin (header, block table, decoded lit/off/len/cmd) for
    // the GPU harnesses (e2e_full.cu and kin); same layout the depth tool has written since v2.
    if(getenv("ACEAPEX_DUMP")){
        FILE* fs=fopen("streams.bin","wb");
        if(fs){ fwrite(&hdr,sizeof(hdr),1,fs); fwrite(boffs.data(),sizeof(BlockOffsets),nb,fs);
            fwrite(lit,1,lit_sz,fs); fwrite(off,1,off_sz,fs); fwrite(len,1,len_sz,fs); fwrite(cmd,1,cmd_sz,fs);
            fclose(fs); fprintf(stderr,"Dumped streams.bin (%u blocks)\n",(unsigned)nb); }
    }
    double t_lz=now_sec(); parallel_decode(lit,off,len,cmd,boffs.data(),nb,dst,hdr.orig_size,hdr.block_size,threads); t_lz=now_sec()-t_lz;
    dec_time=now_sec()-dec_time;
    fprintf(stderr,"  Phase entropy (4 streams in parallel): %.3fs\n  Phase lz77: %.3fs\n",t_fse,t_lz);
 
    uint64_t dv=OUR_CHECKSUM(dst,hdr.orig_size);
    uint64_t hv3; memcpy(&hv3,hdr.xxhash,8);
    bool ok=(dv==hv3);
    FILE* fout=fopen(out_path,"wb");
    if (fout) { fwrite(dst,1,hdr.orig_size,fout); fclose(fout); }
    double wall=now_sec()-t_wall;
    fprintf(stderr,"  Decode: %.2f MB/s  (%.3fs, algorithmic)\n",hdr.orig_size/dec_time/1e6,dec_time);
    fprintf(stderr,"  Decode wall: %.2f MB/s  (%.3fs, wall clock)\n",hdr.orig_size/wall/1e6,wall);
    if(g_dec_err){ ok=false; fprintf(stderr,"  Status: ❌ DECODE ERROR (zstd frame)\n"); }
    else if(!ok) fprintf(stderr,"  Status: ❌ HASH MISMATCH\n");
 
    free(lit); free(off); free(len); free(cmd); free(dst);
    return ok?0:1;
}
 
static int do_test(const char* in_path, int threads, int level=2) {
    g_dec_err=0;
    FILE* fin=fopen(in_path,"rb");
    if (!fin) { fprintf(stderr,"Cannot open: %s\n",in_path); return 1; }
    fseek(fin,0,SEEK_END); size_t src_size=(size_t)ftell(fin); fseek(fin,0,SEEK_SET);
    uint8_t* src=(uint8_t*)malloc(src_size);
    if(!src){fclose(fin);return 1;}
    fread(src,1,src_size,fin); fclose(fin);
    fprintf(stderr,"[*] Test: %s (%.2f MB) threads=%d\n",in_path,src_size/1e6,threads);
    double t_total_t=now_sec();
 
    std::vector<BlockOffsets> boffs;
    uint8_t *raw_lit,*raw_off,*raw_len,*raw_cmd;
    size_t total_lit,total_off,total_len,total_cmd,num_blocks;
    encode_file(src,src_size,threads,level,boffs,
                raw_lit,total_lit,raw_off,total_off,
                raw_len,total_len,raw_cmd,total_cmd,
                num_blocks);
 
    size_t zlit_sz,zoff_sz,zlen_sz,zcmd_sz;
    uint8_t *zlit,*zoff,*zlen,*zcmd;
    zlit=lit_compress(raw_lit,total_lit,zlit_sz);
    entropy_encode(raw_lit,total_lit,raw_off,total_off,raw_len,total_len,raw_cmd,total_cmd,
                   zlit,zlit_sz,zoff,zoff_sz,zlen,zlen_sz,zcmd,zcmd_sz);
 
    size_t total_z=zlit_sz+zoff_sz+zlen_sz+zcmd_sz;
 
    size_t off_sz=fse_stream_size(zoff);
    size_t len_sz=fse_stream_size(zlen);
    size_t cmd_sz=fse_stream_size(zcmd);
 
    size_t lit_sz=0; uint8_t* lit=lit_decompress(zlit,zlit_sz,lit_sz);
    if(!lit) return 1;
    uint8_t* off=(uint8_t*)malloc(off_sz);
    uint8_t* len=(uint8_t*)malloc(len_sz);
    uint8_t* cmd=(uint8_t*)malloc(cmd_sz);
    uint8_t* dst=(uint8_t*)malloc(src_size);
    if(!off||!len||!cmd||!dst){free(lit);free(off);free(len);free(cmd);free(dst);return ACEAPEX_ERR_MEMORY;}
    fse_chunked_decomp(zoff,off_sz,off);
    fse_chunked_decomp(zlen,len_sz,len);
    fse_chunked_decomp(zcmd,cmd_sz,cmd);
    parallel_decode(lit,off,len,cmd,boffs.data(),num_blocks,
                    dst,src_size,g_block_size);
 
    uint8_t digest_orig[32], digest_dec[32];
    sha256(src,src_size,digest_orig); sha256(dst,src_size,digest_dec);
    bool ok=(memcmp(digest_orig,digest_dec,32)==0);
    char sha_hex[65]; sha256_hex(src,src_size,sha_hex);
 
    fprintf(stderr,"\n  ====================================================\n");
    fprintf(stderr,"  ACEAPEX v3 FSE TEST REPORT\n");
    fprintf(stderr,"  ====================================================\n");
    fprintf(stderr,"  Original:   %14zu bytes\n",src_size);
    fprintf(stderr,"  Compressed: %14zu bytes\n",total_z);
    fprintf(stderr,"  Ratio:  %.5fx   BPB: %.4f\n",(double)src_size/total_z,total_z*8.0/src_size);
    double real_enc_t=now_sec()-t_total_t;
    fprintf(stderr,"  Encode: %.2f MB/s  (%.3fs)\n",src_size/real_enc_t/1e6,real_enc_t);
    fprintf(stderr,"  Decode: n/a (timing removed from library)\n");
    fprintf(stderr,"  SHA256: %.16s...\n",sha_hex);
    if(g_dec_err) ok=false;
    fprintf(stderr,"  Status: %s\n",ok?"✅ BIT-PERFECT":g_dec_err?"❌ DECODE ERROR (zstd frame)":"❌ HASH MISMATCH");
    fprintf(stderr,"  ====================================================\n");
 
    free(src); free(dst);
    free(raw_lit); free(raw_off); free(raw_len); free(raw_cmd);
    free(zlit); free(zoff); free(zlen); free(zcmd);
    free(lit); free(off); free(len); free(cmd);
    return ok?0:1;
}
 
#if !defined(ACEAPEX_NO_MAIN) || defined(ACEAPEX_CLI)
#ifdef ACEAPEX_CLI
// r: one region of the original through the public library entry point. The archive
// is mapped read-only and advised random: the kernel pages in only the size tables
// and the chunks the region touches, not the whole file.
static int do_region_cli(const char* in_path,const char* out_path,uint64_t off,uint64_t len) {
    int fd=open(in_path,O_RDONLY);
    if(fd<0){fprintf(stderr,"Cannot open: %s\n",in_path);return 1;}
    struct stat st;
    if(fstat(fd,&st)!=0||st.st_size<=0){fprintf(stderr,"Cannot stat: %s\n",in_path);close(fd);return 1;}
    void* map=mmap(nullptr,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(map==MAP_FAILED){fprintf(stderr,"mmap failed\n");close(fd);return 1;}
    madvise(map,(size_t)st.st_size,MADV_RANDOM);
    uint8_t* dst=(uint8_t*)malloc(len);
    if(!dst){fprintf(stderr,"out of memory\n");munmap(map,(size_t)st.st_size);close(fd);return 1;}
    int64_t r=aceapex_decompress_region(map,(size_t)st.st_size,dst,(size_t)len,off,len);
    munmap(map,(size_t)st.st_size); close(fd);
    if(r<0){fprintf(stderr,"region decode failed (%lld)\n",(long long)r);free(dst);return 1;}
    if((uint64_t)r!=len){fprintf(stderr,"region beyond end: %lld of %llu bytes\n",(long long)r,(unsigned long long)len);free(dst);return 1;}
    FILE* fo=fopen(out_path,"wb");
    if(!fo){fprintf(stderr,"Cannot create: %s\n",out_path);free(dst);return 1;}
    size_t w=fwrite(dst,1,(size_t)len,fo); fclose(fo); free(dst);
    if(w!=(size_t)len){fprintf(stderr,"short write: %s\n",out_path);return 1;}
    return 0;
}
#endif
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,"ACEAPEX v3 FSE — Global FSE + Parallel decode\n\n"
            "Usage:\n  %s c --in <f> --out <f.aet> [--threads N]\n"
            "  %s d --in <f.aet> --out <f>\n  %s t --in <f> [--threads N]\n"
            "  %s r --in <f.aet> --out <f> --region OFFSET LENGTH   (bytes of the original)\n",
            argv[0],argv[0],argv[0],argv[0]);
        return 1;
    }
    const char* cmd=argv[1]; const char* in=nullptr; const char* out=nullptr; int thr=8; int level=2;
    uint64_t reg_off=0,reg_len=0;
    for(int i=2;i<argc;i++) {
        if (!strcmp(argv[i],"--in")&&i+1<argc) in=argv[++i];
        else if (!strcmp(argv[i],"--out")&&i+1<argc) out=argv[++i];
        else if (!strcmp(argv[i],"--threads")&&i+1<argc) thr=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--level")&&i+1<argc) level=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--fast")) level=1;
        else if (!strcmp(argv[i],"--region")&&i+2<argc) { reg_off=strtoull(argv[++i],nullptr,10); reg_len=strtoull(argv[++i],nullptr,10); }
    }
    if (!in) { fprintf(stderr,"--in required\n"); return 1; }
    if (!strcmp(cmd,"c")) { if (!out) { fprintf(stderr,"--out required\n"); return 1; } return do_compress(in,out,thr,level); }
    if (!strcmp(cmd,"d")) { if (!out) { fprintf(stderr,"--out required\n"); return 1; } return do_decompress(in,out,thr); }
    if (!strcmp(cmd,"t")) return do_test(in,thr,level);
    if (!strcmp(cmd,"r")) {
        if (!out) { fprintf(stderr,"--out required\n"); return 1; }
        if (reg_len==0) { fprintf(stderr,"--region OFFSET LENGTH required\n"); return 1; }
#ifdef ACEAPEX_CLI
        return do_region_cli(in,out,reg_off,reg_len);
#else
        fprintf(stderr,"r needs the library build (make); this binary was built from aceapex_main.cpp alone\n"); return 1;
#endif
    }
    return 1;
}
#endif // ACEAPEX_NO_MAIN
