#define ACEAPEX_NO_MAIN
#include "aceapex_main.cpp"
#include "aceapex.h"
#include <vector>
#include <algorithm>
#include <atomic>

size_t aceapex_compress_bound(size_t src_size) {
    // Worst case: incompressible data + header overhead
    return src_size + src_size/8 + 1024;
}

int64_t aceapex_compress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    int         level,
    int         threads)
{
    // Empty input: nothing to encode. Returning 0 avoids a division by
    // num_blocks==0 further down (SIGFPE). Reported paths never hit this in
    // lzbench, but the public API must not crash on an empty buffer.
    if (src_size == 0) {                               // empty input -> empty archive (one header)
        if (!dst) return ACEAPEX_ERR_DATA;
        if (dst_capacity < sizeof(AetHeader)) return ACEAPEX_ERR_BUFFER;
        AetHeader eh; ax_empty_header(eh); memcpy(dst,&eh,sizeof(eh)); return (int64_t)sizeof(eh); }
    if (!src || !dst) return ACEAPEX_ERR_DATA;
    if (threads <= 0) threads = 8;
    if (level <= 0)   level   = 2;

    std::vector<BlockOffsets> boffs;
    uint8_t *rl,*ro,*rn,*rc;
    size_t tl,to,tn,tc,nb;
    if (!encode_file((const uint8_t*)src,src_size,threads,level,
                     boffs,rl,tl,ro,to,rn,tn,rc,tc,nb))
        return ACEAPEX_ERR_MEMORY;

    size_t zls,zos,zns,zcs;
    uint8_t *zl,*zo,*zn,*zc;
    zl=lit_compress(rl,tl,zls);
    entropy_encode(rl,tl,ro,to,rn,tn,rc,tc,
                   zl,zls,zo,zos,zn,zns,zc,zcs);
    free(rl);free(ro);free(rn);free(rc);

    AetHeader hdr;
    memcpy(hdr.magic,"ACEPX2\0\0",8);
    hdr.version=2; hdr.orig_size=src_size;
    // The block size is adaptive (compute_block_size in encode_file) and MUST be the one
    // the blocks were cut with; the constant here put every second block of an input
    // between 256 KiB and 4 MiB x threads at the wrong offset (lzbench t300k, 2026-09-29).
    hdr.block_size=(uint32_t)g_block_size; hdr.num_blocks=nb;
    uint64_t hv=OUR_CHECKSUM(src,src_size);
    memcpy(hdr.xxhash,&hv,8);
    hdr.zlit_sz=zls;hdr.zoff_sz=zos;
    hdr.zlen_sz=zns;hdr.zcmd_sz=zcs;

    size_t total=sizeof(hdr)+nb*sizeof(BlockOffsets)
                 +zls+zos+zns+zcs;
    if (total>dst_capacity) {
        free(zl);free(zo);free(zn);free(zc);
        return ACEAPEX_ERR_BUFFER;
    }

    uint8_t* p=(uint8_t*)dst;
    memcpy(p,&hdr,sizeof(hdr)); p+=sizeof(hdr);
    memcpy(p,boffs.data(),nb*sizeof(BlockOffsets));
    p+=nb*sizeof(BlockOffsets);
    memcpy(p,zl,zls);p+=zls;
    memcpy(p,zo,zos);p+=zos;
    memcpy(p,zn,zns);p+=zns;
    memcpy(p,zc,zcs);
    free(zl);free(zo);free(zn);free(zc);
    return (int64_t)total;
}

int64_t aceapex_decompress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity)
{
    return aceapex_decompress_mt(src, src_size, dst, dst_capacity, 0);
}

// Header validation + entropy phase shared by aceapex_decompress_mt and
// aceapex_decode_streams. On success the four decoded streams and the block table are
// owned by the caller; on failure nothing is allocated and a negative code is returned.
struct AxStreams { uint8_t *l,*o,*n,*c; size_t ls,os,ns,cs; std::vector<BlockOffsets> boffs; AetHeader hdr; };
static int64_t ax_entropy_decode(const void* src, size_t src_size, int threads, AxStreams& S)
{
    if (!src || src_size < sizeof(AetHeader)) return ACEAPEX_ERR_DATA;
    const uint8_t* p=(const uint8_t*)src;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return ACEAPEX_ERR_DATA;
    S.hdr = hdr;
    if (hdr.num_blocks == 0) return ax_is_empty_archive(hdr) ? 0 : ACEAPEX_ERR_DATA;

    // ---- Header validation. Runs once per archive, costs nothing in the hot loop.
    // A single corrupted byte in the header or in the BlockOffsets table used to send
    // stream pointers into arbitrary memory (SIGSEGV). Absolute offsets make this cheap
    // to check: every bound is a constant known before decoding starts.
    if (hdr.block_size == 0 || hdr.num_blocks == 0) return ACEAPEX_ERR_DATA;
    if ((uint64_t)hdr.num_blocks * (uint64_t)hdr.block_size < hdr.orig_size)
        return ACEAPEX_ERR_DATA;
    {
        uint64_t need = (uint64_t)sizeof(hdr)
                      + (uint64_t)hdr.num_blocks * sizeof(BlockOffsets)
                      + hdr.zlit_sz + hdr.zoff_sz + hdr.zlen_sz + hdr.zcmd_sz;
        if (need > src_size) return ACEAPEX_ERR_DATA;
    }
    p+=sizeof(hdr);
    S.boffs.assign(hdr.num_blocks, BlockOffsets());
    memcpy(S.boffs.data(),p,hdr.num_blocks*sizeof(BlockOffsets));
    p+=hdr.num_blocks*sizeof(BlockOffsets);
    // malloc(0) may legally return NULL; an empty stream is not an error.
    // The old !zl check turned zlit_sz==0 (tiny inputs) into ACEAPEX_ERR_MEMORY.
    uint8_t* zl=(uint8_t*)malloc(hdr.zlit_sz?hdr.zlit_sz:1);
    uint8_t* zo=(uint8_t*)malloc(hdr.zoff_sz?hdr.zoff_sz:1);
    uint8_t* zn=(uint8_t*)malloc(hdr.zlen_sz?hdr.zlen_sz:1);
    uint8_t* zc=(uint8_t*)malloc(hdr.zcmd_sz?hdr.zcmd_sz:1);
    if(!zl||!zo||!zn||!zc){free(zl);free(zo);free(zn);free(zc);return ACEAPEX_ERR_MEMORY;}
    memcpy(zl,p,hdr.zlit_sz); p+=hdr.zlit_sz;
    memcpy(zo,p,hdr.zoff_sz); p+=hdr.zoff_sz;
    memcpy(zn,p,hdr.zlen_sz); p+=hdr.zlen_sz;
    memcpy(zc,p,hdr.zcmd_sz);
    g_dec_err=0;
    size_t os=0,ns=0,cs=0;
    if (!ax_fse_check(zo,hdr.zoff_sz,&os) || !ax_fse_check(zn,hdr.zlen_sz,&ns) || !ax_fse_check(zc,hdr.zcmd_sz,&cs)) {
        free(zl);free(zo);free(zn);free(zc); return ACEAPEX_ERR_DATA; }
    { uint64_t cap = hdr.orig_size * 4 + ((uint64_t)1 << 20);
      if (os > cap || ns > cap || cs > cap) { free(zl);free(zo);free(zn);free(zc); return ACEAPEX_ERR_DATA; } }
    uint8_t* o=(uint8_t*)malloc(os?os:1);
    uint8_t* n=(uint8_t*)malloc(ns?ns:1);
    uint8_t* c=(uint8_t*)malloc(cs?cs:1);
    if(!o||!n||!c){free(o);free(n);free(c);free(zl);free(zo);free(zn);free(zc);return ACEAPEX_ERR_MEMORY;}
    // Entropy phase on one budget of hardware threads: literal lanes and one pool for
    // the token streams run concurrently (was: literals, then three streams serially).
    int budget=threads>0?threads:(int)std::thread::hardware_concurrency(); if(budget<1) budget=8;
    int lit_t, tok_t; ax_entropy_split(ax_lit_decoded_size(zl,hdr.zlit_sz), os+ns+cs, budget, lit_t, tok_t);
    struct LitArg{const uint8_t*s;size_t sz;uint8_t**out;size_t*osz;int lanes;};
    size_t ls=0; uint8_t* l=nullptr; LitArg larg={zl,(size_t)hdr.zlit_sz,&l,&ls,lit_t};
    auto litfn=[](void*a)->void*{LitArg*x=(LitArg*)a; *x->out=lit_decompress(x->s,x->sz,*x->osz,x->lanes); return nullptr;};
    FseStream fst[3]={{zo,os,o},{zn,ns,n},{zc,cs,c}};
    if (budget == 1) { litfn(&larg); fse_multi_decomp(fst,3,1); }
    else { pthread_t lt; ax_thread(&lt,litfn,&larg);
           fse_multi_decomp(fst,3,tok_t); pthread_join(lt,nullptr); }
    free(zl);free(zo);free(zn);free(zc);
    if(!l){free(o);free(n);free(c);return ACEAPEX_ERR_MEMORY;}
    if(g_dec_err){free(l);free(o);free(n);free(c);return ACEAPEX_ERR_DATA;}

    // Every block's stream slice must lie inside its decoded stream.
    for (size_t b = 0; b < hdr.num_blocks; b++) {
        const BlockOffsets& bo = S.boffs[b];
        if (bo.lit_off + bo.lit_sz > ls || bo.lit_off > ls ||
            bo.off_off + bo.off_sz > os || bo.off_off > os ||
            bo.len_off + bo.len_sz > ns || bo.len_off > ns ||
            bo.cmd_off + bo.cmd_sz > cs || bo.cmd_off > cs) {
            free(l);free(o);free(n);free(c);
            return ACEAPEX_ERR_DATA;
        }
    }
    S.l=l; S.o=o; S.n=n; S.c=c; S.ls=ls; S.os=os; S.ns=ns; S.cs=cs;
    return (int64_t)hdr.orig_size;
}

int64_t aceapex_decompress_mt(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity, int threads)
{
    AxStreams S; int64_t r = ax_entropy_decode(src, src_size, threads, S);
    if (r <= 0) return r;                                    // error, or the empty archive
    if (S.hdr.orig_size > dst_capacity) { free(S.l);free(S.o);free(S.n);free(S.c); return ACEAPEX_ERR_BUFFER; }
    int budget=threads>0?threads:(int)std::thread::hardware_concurrency(); if(budget<1) budget=8;
    parallel_decode(S.l,S.o,S.n,S.c,S.boffs.data(),S.hdr.num_blocks,
                    (uint8_t*)dst,S.hdr.orig_size,S.hdr.block_size,budget);
    free(S.l);free(S.o);free(S.n);free(S.c);
    return (int64_t)S.hdr.orig_size;
}

int aceapex_decode_streams(const void* src, size_t src_size, aceapex_streams_t* out)
{
    if (!out) return ACEAPEX_ERR_DATA;
    memset(out, 0, sizeof(*out));
    AxStreams S; int64_t r = ax_entropy_decode(src, src_size, 0, S);
    if (r < 0) return (int)r;
    if (r == 0) { out->block_size = S.hdr.block_size; return 0; }   // empty archive: no streams
    std::vector<BlockOffsets>* bv = new std::vector<BlockOffsets>(std::move(S.boffs));
    out->lit=S.l; out->off=S.o; out->len=S.n; out->cmd=S.c;
    out->lit_sz=S.ls; out->off_sz=S.os; out->len_sz=S.ns; out->cmd_sz=S.cs;
    out->boffs_vec=(void*)bv; out->boffs=(const void*)bv->data();
    out->num_blocks=S.hdr.num_blocks; out->block_size=S.hdr.block_size; out->orig_size=S.hdr.orig_size;
    return 0;
}

void aceapex_streams_free(aceapex_streams_t* s)
{
    if(!s) return;
    free(s->lit); free(s->off); free(s->len); free(s->cmd);
    delete (std::vector<BlockOffsets>*)s->boffs_vec;
    s->lit=s->off=s->len=s->cmd=nullptr; s->boffs_vec=nullptr; s->boffs=nullptr;
}

// The block table sits at offset 68 of the archive (4-byte aligned) and the region paths
// read it in place: load entries with a byte copy, never through a BlockOffsets*
// (UBSan: misaligned 8-byte member access; strict-alignment CPUs may trap).
static inline BlockOffsets ax_bo(const BlockOffsets* t, size_t i) {
    BlockOffsets b; memcpy(&b, (const uint8_t*)t + i * sizeof(BlockOffsets), sizeof b); return b; }

int64_t aceapex_decompress_region(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    uint64_t    offset, uint64_t length)
{
    if (!src || src_size < sizeof(AetHeader)) return ACEAPEX_ERR_DATA;
    const uint8_t* p = (const uint8_t*)src;
    AetHeader hdr; memcpy(&hdr, p, sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8) != 0) return ACEAPEX_ERR_DATA;
    if (hdr.num_blocks == 0) return (ax_is_empty_archive(hdr) && length == 0) ? 0 : ACEAPEX_ERR_DATA;
    if (hdr.block_size == 0) return ACEAPEX_ERR_DATA;
    if (length == 0) return 0;
    if (offset > hdr.orig_size || length > hdr.orig_size - offset) return ACEAPEX_ERR_DATA;
    if (length > dst_capacity) return ACEAPEX_ERR_BUFFER;

    uint64_t need = (uint64_t)sizeof(hdr)
                  + (uint64_t)hdr.num_blocks * sizeof(BlockOffsets)
                  + hdr.zlit_sz + hdr.zoff_sz + hdr.zlen_sz + hdr.zcmd_sz;
    if (need > src_size) return ACEAPEX_ERR_DATA;

    p += sizeof(hdr);
    // Таблица блоков читается ПРЯМО ИЗ АРХИВА. Копия в вектор стоила 969 KB на каждый
    // вызов ради 128 байт, что дало 485 page-faults на запрос — почти всю оставшуюся
    // латентность. Архив уже в памяти вызывающего, копировать нечего.
    const BlockOffsets* boffs = (const BlockOffsets*)p;
    p += (size_t)hdr.num_blocks * sizeof(BlockOffsets);

    const uint8_t* zlit = p;
    const uint8_t* zoff = zlit + hdr.zlit_sz;
    const uint8_t* zlen = zoff + hdr.zoff_sz;
    const uint8_t* zcmd = zlen + hdr.zlen_sz;
    { size_t a,b,c; if (!ax_fse_check(zoff,hdr.zoff_sz,&a) || !ax_fse_check(zlen,hdr.zlen_sz,&b) ||
                        !ax_fse_check(zcmd,hdr.zcmd_sz,&c)) return ACEAPEX_ERR_DATA; }

    size_t b0 = (size_t)(offset / hdr.block_size);
    size_t b1 = (size_t)((offset + length - 1) / hdr.block_size);
    if (b1 >= hdr.num_blocks) return ACEAPEX_ERR_DATA;

    size_t lf=ax_bo(boffs,b0).lit_off, lt=ax_bo(boffs,b1).lit_off+ax_bo(boffs,b1).lit_sz;
    size_t of=ax_bo(boffs,b0).off_off, ot=ax_bo(boffs,b1).off_off+ax_bo(boffs,b1).off_sz;
    size_t nf=ax_bo(boffs,b0).len_off, nt=ax_bo(boffs,b1).len_off+ax_bo(boffs,b1).len_sz;
    size_t cf=ax_bo(boffs,b0).cmd_off, ct=ax_bo(boffs,b1).cmd_off+ax_bo(boffs,b1).cmd_sz;

    size_t lit_sz = 0, wl=0, wo=0, wn=0, wc=0;
    g_dec_err=0;
    uint8_t* lit = lit_range(zlit, hdr.zlit_sz, lit_sz, lf, lt, &wl);
    uint8_t* off = fse_range(zoff, fse_stream_size(zoff), of, ot, &wo);
    uint8_t* len = fse_range(zlen, fse_stream_size(zlen), nf, nt, &wn);
    uint8_t* cmd = fse_range(zcmd, fse_stream_size(zcmd), cf, ct, &wc);
    // A range function returns nullptr when a zstd frame fails to decode (or on malloc
    // failure); either way the caller must not read the buffers: fail closed.
    if (!lit || !off || !len || !cmd || g_dec_err) {
        free(lit); free(off); free(len); free(cmd);
        return ACEAPEX_ERR_DATA;
    }

    size_t span_start = b0 * (size_t)hdr.block_size;
    size_t span_end   = (b1 + 1) * (size_t)hdr.block_size;
    if (span_end > hdr.orig_size) span_end = (size_t)hdr.orig_size;
    uint8_t* span = (uint8_t*)malloc(span_end - span_start + 64);
    if (!span) { free(lit); free(off); free(len); free(cmd); return ACEAPEX_ERR_MEMORY; }

    for (size_t b = b0; b <= b1; b++) {
        const BlockOffsets bo = ax_bo(boffs,b);
        size_t bstart = b * (size_t)hdr.block_size;
        size_t bsize  = (size_t)(hdr.orig_size - bstart);
        if (bsize > hdr.block_size) bsize = hdr.block_size;
        decompress_streams(span + (bstart - span_start), bsize,
            lit + (bo.lit_off-wl), bo.lit_sz, off + (bo.off_off-wo), bo.off_sz,
            len + (bo.len_off-wn), bo.len_sz, cmd + (bo.cmd_off-wc), bo.cmd_sz);
    }
    memcpy(dst, span + (offset - span_start), (size_t)length);

    free(lit); free(off); free(len); free(cmd); free(span);
    return (int64_t)length;
}

// ---------------------------------------------------------------------------
// BATCH. Стоимость одного региона определяется распаковкой чанков, покрывающих
// его блоки, а не размером ответа. При многих диапазонах те же блоки распаковыв-
// аются повторно: 10 000 случайных 16 KiB запросов трогают 11 342 различных блока
// из 15 499. Группировка по блокам превращает N_requests * T_decode в
// N_unique_blocks * T_decode + T_dispatch.
// ---------------------------------------------------------------------------
namespace {

struct RangeWork {
    size_t   idx;          // позиция в исходном массиве, чтобы вернуть порядок
    uint64_t offset, length;
    void*    dst;
    uint32_t b0, b1;       // покрываемые блоки
};

struct Group { size_t first, last; uint32_t b0, b1; };   // [first,last) в w[]

struct BatchTask {
    const uint8_t*      src;
    const AetHeader*    hdr;
    const BlockOffsets* boffs;
    const uint8_t      *zlit, *zoff, *zlen, *zcmd;
    RangeWork*          w;
    Group*              g;
    size_t              ng;
    std::atomic<size_t> next;
    std::atomic<int>    failed;
};

// Один рабочий берёт группы подряд идущих запросов. Группа — это набор запросов,
// чьи блоки перекрываются или соседствуют: для них выгодно распаковать один span.
void* batch_worker(void* arg) {
    BatchTask* t = (BatchTask*)arg;
    const AetHeader& h = *t->hdr;
    for (;;) {
        // Группы нарезаны ДО запуска потоков, рабочий берёт готовую целиком.
        // Прежняя схема с захватом соседей через compare_exchange давала гонку:
        // между load и обменом другой поток успевал взять запрос, и два рабочих
        // писали в один RangeWork. TSan это поймал, данные сходились случайно.
        size_t gi = t->next.fetch_add(1);
        if (gi >= t->ng) break;
        const Group& G = t->g[gi];
        size_t i = G.first, grp_end = G.last;
        RangeWork& r = t->w[i];

        size_t lf=ax_bo(t->boffs,G.b0).lit_off, lt=ax_bo(t->boffs,G.b1).lit_off+ax_bo(t->boffs,G.b1).lit_sz;
        size_t of=ax_bo(t->boffs,G.b0).off_off, ot=ax_bo(t->boffs,G.b1).off_off+ax_bo(t->boffs,G.b1).off_sz;
        size_t nf=ax_bo(t->boffs,G.b0).len_off, nt=ax_bo(t->boffs,G.b1).len_off+ax_bo(t->boffs,G.b1).len_sz;
        size_t cf=ax_bo(t->boffs,G.b0).cmd_off, ct=ax_bo(t->boffs,G.b1).cmd_off+ax_bo(t->boffs,G.b1).cmd_sz;

        size_t lit_sz=0, wl=0, wo=0, wn=0, wc=0;
        uint8_t* lit=lit_range(t->zlit,h.zlit_sz,lit_sz,lf,lt,&wl);
        uint8_t* off=fse_range(t->zoff,fse_stream_size(t->zoff),of,ot,&wo);
        uint8_t* len=fse_range(t->zlen,fse_stream_size(t->zlen),nf,nt,&wn);
        uint8_t* cmd=fse_range(t->zcmd,fse_stream_size(t->zcmd),cf,ct,&wc);
        if(!lit||!off||!len||!cmd||g_dec_err){
            free(lit);free(off);free(len);free(cmd);
            for(size_t k=G.first;k<G.last;k++) t->w[k].dst=nullptr;
            t->failed.store(1); continue;
        }

        size_t span_start=(size_t)G.b0*h.block_size;
        size_t span_end=(size_t)(G.b1+1)*h.block_size;
        if(span_end>h.orig_size) span_end=(size_t)h.orig_size;
        uint8_t* span=(uint8_t*)malloc(span_end-span_start+64);
        if(!span){ free(lit);free(off);free(len);free(cmd);
                   t->failed.store(1); continue; }

        for(uint32_t b=G.b0;b<=G.b1;b++){
            const BlockOffsets bo=ax_bo(t->boffs,b);
            size_t bs=(size_t)b*h.block_size;
            size_t bsz=(size_t)(h.orig_size-bs);
            if(bsz>h.block_size) bsz=h.block_size;
            decompress_streams(span+(bs-span_start),bsz,
                lit+(bo.lit_off-wl),bo.lit_sz, off+(bo.off_off-wo),bo.off_sz,
                len+(bo.len_off-wn),bo.len_sz, cmd+(bo.cmd_off-wc),bo.cmd_sz);
        }
        for(size_t k=i;k<grp_end;k++){
            RangeWork& q=t->w[k];
            if(q.offset<span_start || q.offset+q.length>span_end) continue;
            memcpy(q.dst, span+(q.offset-span_start), (size_t)q.length);
        }
        free(lit);free(off);free(len);free(cmd);free(span);
    }
    return nullptr;
}

} // namespace

int64_t aceapex_decompress_ranges(
    const void* src, size_t src_size,
    aceapex_range_t* ranges, size_t count, int threads)
{
    g_dec_err=0;
    if(!src||!ranges) return ACEAPEX_ERR_DATA;
    if(count==0) return 0;
    if(src_size<sizeof(AetHeader)) return ACEAPEX_ERR_DATA;

    const uint8_t* p=(const uint8_t*)src;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if(memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return ACEAPEX_ERR_DATA;
    if(hdr.num_blocks==0){ if(!ax_is_empty_archive(hdr)) return ACEAPEX_ERR_DATA;
        int64_t okn=0; for(size_t i=0;i<count;i++){ ranges[i].written = ranges[i].length==0 ? 0 : ACEAPEX_ERR_DATA; if(!ranges[i].length) okn++; }
        return okn; }
    if(hdr.block_size==0) return ACEAPEX_ERR_DATA;
    uint64_t need=(uint64_t)sizeof(hdr)+(uint64_t)hdr.num_blocks*sizeof(BlockOffsets)
                 +hdr.zlit_sz+hdr.zoff_sz+hdr.zlen_sz+hdr.zcmd_sz;
    if(need>src_size) return ACEAPEX_ERR_DATA;

    p+=sizeof(hdr);
    const BlockOffsets* boffs=(const BlockOffsets*)p;
    p+=(size_t)hdr.num_blocks*sizeof(BlockOffsets);
    const uint8_t* zlit=p;
    const uint8_t* zoff=zlit+hdr.zlit_sz;
    const uint8_t* zlen=zoff+hdr.zoff_sz;
    const uint8_t* zcmd=zlen+hdr.zlen_sz;
    { size_t a,b,c; if (!ax_fse_check(zoff,hdr.zoff_sz,&a) || !ax_fse_check(zlen,hdr.zlen_sz,&b) ||
                        !ax_fse_check(zcmd,hdr.zcmd_sz,&c)) return ACEAPEX_ERR_DATA; }

    // Проверяем каждый запрос отдельно: плохой диапазон не должен ронять батч.
    std::vector<RangeWork> w; w.reserve(count);
    for(size_t i=0;i<count;i++){
        aceapex_range_t& q=ranges[i];
        q.written=ACEAPEX_ERR_DATA;
        if(q.length==0){ q.written=0; continue; }
        if(!q.dst) continue;
        if(q.offset>hdr.orig_size||q.length>hdr.orig_size-q.offset) continue;
        uint32_t b0=(uint32_t)(q.offset/hdr.block_size);
        uint32_t b1=(uint32_t)((q.offset+q.length-1)/hdr.block_size);
        if(b1>=hdr.num_blocks) continue;
        w.push_back({i,q.offset,q.length,q.dst,b0,b1});
    }
    if(w.empty()) return 0;

    // Сортировка по первому блоку: соседние запросы попадают в один рабочий подряд,
    // а значит переиспользуют горячие страницы архива и кэш процессора.
    std::sort(w.begin(),w.end(),
              [](const RangeWork& a,const RangeWork& b){ return a.b0<b.b0; });

    // Нарезка на группы: подряд идущие запросы, чьи блоки соседствуют, обслуживаются
    // одной распаковкой span. Ограничение в 64 блока не даёт span раздуться.
    std::vector<Group> groups;
    for(size_t i=0;i<w.size();){
        uint32_t b0=w[i].b0, b1=w[i].b1;
        size_t j=i+1;
        while(j<w.size() && w[j].b0<=b1+1 && w[j].b1<=b0+63){
            if(w[j].b1>b1) b1=w[j].b1;
            j++;
        }
        groups.push_back({i,j,b0,b1});
        i=j;
    }

    BatchTask t{(const uint8_t*)src,&hdr,boffs,zlit,zoff,zlen,zcmd,
                w.data(),groups.data(),groups.size(),{0},{0}};
    // Порог: поднимать восемь потоков ради сотни запросов дороже, чем выполнить их
    // последовательно. Замер: при N=100 батч был вдвое медленнее цикла.
    int lanes = threads>0 ? threads : (int)std::thread::hardware_concurrency();
    if(lanes<1) lanes=1;
    if(w.size()<512) lanes=1;
    if((size_t)lanes>groups.size()) lanes=(int)groups.size();
    if(lanes<1) lanes=1;

    if(lanes==1){
        batch_worker(&t);
    } else {
        std::vector<pthread_t> th(lanes);
        for(int k=0;k<lanes;k++) ax_thread(&th[k],batch_worker,&t);
        for(int k=0;k<lanes;k++) pthread_join(th[k],nullptr);
    }

    int64_t ok=0;
    for(const RangeWork& r : w)
        if(r.dst){ ranges[r.idx].written=(int64_t)r.length; ok++; }
    return ok;
}
