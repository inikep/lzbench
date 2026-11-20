#define ACEAPEX_NO_MAIN
#include "aceapex_main.cpp"
#include "aceapex.h"
#ifdef ACEAPEX_ENV_TUNING
#include "ax_linemodel.h"                                   // AX_LINEMODEL experiment (tuning builds only)
#endif
#include <chrono>
#include <vector>
#include <algorithm>
#include <atomic>

size_t aceapex_compress_bound(size_t src_size) {
    // Worst case: incompressible data + header overhead; the fast profile's bound (ax_fast_bound) is the larger one
    const size_t d = src_size + src_size/8 + 1024, f = ax_fast_bound(src_size);
    return d > f ? d : f;
}

// 2.6.3: the public functions are thin wrappers (ax_api, end of this file) over these ax_*_impl: every exception and
// every allocation failure of the call (its t_oom flag, set on any of its threads) gives ACEAPEX_ERR_MEMORY.
static int64_t ax_compress_impl(
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
#ifdef ACEAPEX_ENV_TUNING
    if (axlm::wanted()) return axlm::compress((const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, level, threads);
#endif
    if (ax_fast_level(level)) {                        // levels 11..18: the fast profile (header version 3)
        if (threads == 1) { const int64_t r = ax_fast_compress_1t((const uint8_t*)src, src_size, ax_fast_level(level), (uint8_t*)dst, dst_capacity);
                            return r == -2 ? ACEAPEX_ERR_DATA : r < 0 ? ACEAPEX_ERR_BUFFER : r; }
        std::vector<std::vector<uint8_t>> blocks; axf::Params P; uint32_t bs;
        if (!ax_fast_encode((const uint8_t*)src, src_size, ax_fast_level(level), threads, blocks, P, bs)) return ACEAPEX_ERR_DATA;
        if (ax_oom()) return ACEAPEX_ERR_MEMORY;                // a block not encoded
        const int64_t r = ax_fast_assemble(blocks, P, bs, (const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity);
        return r < 0 ? ACEAPEX_ERR_BUFFER : r;
    }

    std::vector<BlockOffsets> boffs;
    struct Bufs { uint8_t* p[8] = {}; ~Bufs() { for (uint8_t* q : p) free(q); } } B;   // freed on every way out (2.6.3)
    uint8_t *&rl=B.p[0], *&ro=B.p[1], *&rn=B.p[2], *&rc=B.p[3], *&zl=B.p[4], *&zo=B.p[5], *&zn=B.p[6], *&zc=B.p[7];
    size_t tl,to,tn,tc,nb;
    if (!encode_file((const uint8_t*)src,src_size,threads,level,
                     boffs,rl,tl,ro,to,rn,tn,rc,tc,nb))
        return ACEAPEX_ERR_MEMORY;

    size_t zls=0,zos=0,zns=0,zcs=0;
    zl=lit_compress(rl,tl,zls);
    if (!zl || ax_oom()) return ACEAPEX_ERR_MEMORY;
    entropy_encode(rl,tl,ro,to,rn,tn,rc,tc,
                   zl,zls,zo,zos,zn,zns,zc,zcs);
    if (!zo || !zn || !zc || ax_oom()) return ACEAPEX_ERR_MEMORY;
    for (int i = 0; i < 4; i++) { free(B.p[i]); B.p[i] = nullptr; }

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
    if (total>dst_capacity) return ACEAPEX_ERR_BUFFER;

    uint8_t* p=(uint8_t*)dst;
    memcpy(p,&hdr,sizeof(hdr)); p+=sizeof(hdr);
    memcpy(p,boffs.data(),nb*sizeof(BlockOffsets));
    p+=nb*sizeof(BlockOffsets);
    memcpy(p,zl,zls);p+=zls;
    memcpy(p,zo,zos);p+=zos;
    memcpy(p,zn,zns);p+=zns;
    memcpy(p,zc,zcs);
    return (int64_t)total;
}

// Header validation + entropy phase shared by aceapex_decompress_mt and
// aceapex_decode_streams. On success the four decoded streams and the block table are
// held by S (freed with it, also when an exception unwinds: 2.6.3); on failure a negative code is returned.
struct AxStreams { uint8_t *l = nullptr, *o = nullptr, *n = nullptr, *c = nullptr; size_t ls = 0, os = 0, ns = 0, cs = 0;
                   std::vector<BlockOffsets> boffs; AetHeader hdr;
                   std::vector<AxLitChunk> lch; size_t lcsz = 0; bool tiled = false;   // tiled: literals not decoded yet (AX_LIT_TILE)
                   AxStreams() = default; AxStreams(const AxStreams&) = delete; AxStreams& operator=(const AxStreams&) = delete;
                   ~AxStreams() { free(l); free(o); free(n); free(c); } };
// AX_PHASE_TIMES=1: wall time of each decode phase on stderr (diagnostics; read once)
static bool ax_pt(){ static const bool on = [] { const char* e = ax_getenv("AX_PHASE_TIMES"); return e && atoi(e); }(); return on; }
static double ax_now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int64_t ax_entropy_decode(const void* src, size_t src_size, int threads, AxStreams& S, bool tile = false)
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
        // sum the four u64 stream sizes one by one against the remaining bytes so the total cannot wrap (2.6.2; affects <= 2.6.1)
        uint64_t need = (uint64_t)sizeof(hdr) + (uint64_t)hdr.num_blocks * sizeof(BlockOffsets);
        if (need > src_size) return ACEAPEX_ERR_DATA;
        for (uint64_t z : {hdr.zlit_sz, hdr.zoff_sz, hdr.zlen_sz, hdr.zcmd_sz}) {
            if (z > src_size - need) return ACEAPEX_ERR_DATA;
            need += z;
        }
    }
    p+=sizeof(hdr);
    const double t_a = ax_pt() ? ax_now() : 0;
    S.boffs.assign(hdr.num_blocks, BlockOffsets());
    memcpy(S.boffs.data(),p,hdr.num_blocks*sizeof(BlockOffsets));
    p+=hdr.num_blocks*sizeof(BlockOffsets);
    // the compressed streams are read in place (every reader takes const pointers and its own size; the archive
    // size was checked above): no copy of the archive (T2T: 853 MB copied and faulted in, serial, ~0.3 s before)
    const uint8_t* zl=p; p+=hdr.zlit_sz;
    const uint8_t* zo=p; p+=hdr.zoff_sz;
    const uint8_t* zn=p; p+=hdr.zlen_sz;
    const uint8_t* zc=p;
    g_dec_err=0;
    size_t os=0,ns=0,cs=0;
    if (!ax_fse_check(zo,hdr.zoff_sz,&os) || !ax_fse_check(zn,hdr.zlen_sz,&ns) || !ax_fse_check(zc,hdr.zcmd_sz,&cs)) {
        return ACEAPEX_ERR_DATA; }
    // the literal stream decodes to at most the original's bytes: its own 8-byte header is checked against orig_size
    // before anything is allocated for it (api_fuzz: a damaged header asked lit_decompress for 2^54 bytes)
    if (ax_lit_decoded_size(zl,hdr.zlit_sz) > hdr.orig_size) return ACEAPEX_ERR_DATA;
    { uint64_t cap = hdr.orig_size * 4 + ((uint64_t)1 << 20);
      if (os > cap || ns > cap || cs > cap) return ACEAPEX_ERR_DATA; }
    uint8_t *&o=S.o, *&n=S.n, *&c=S.c, *&l=S.l;               // S frees them on every way out
    o=ax_big_malloc(os); n=ax_big_malloc(ns); c=ax_big_malloc(cs);
    if(!o||!n||!c) return ACEAPEX_ERR_MEMORY;
    // Entropy phase on one budget of hardware threads: literal lanes and one pool for
    // the token streams run concurrently (was: literals, then three streams serially).
    int budget=ax_decode_budget(threads,hdr.orig_size);
    int lit_t, tok_t; ax_entropy_split(ax_lit_decoded_size(zl,hdr.zlit_sz), os+ns+cs, budget, lit_t, tok_t);
    struct LitArg{const uint8_t*s;size_t sz;uint8_t**out;size_t*osz;int lanes;};
    size_t ls=0; LitArg larg={zl,(size_t)hdr.zlit_sz,&l,&ls,lit_t};
    auto litfn=[](void*a)->void*{LitArg*x=(LitArg*)a; *x->out=lit_decompress(x->s,x->sz,*x->osz,x->lanes); return nullptr;};
    FseStream fst[3]={{zo,os,o},{zn,ns,n},{zc,cs,c}};
    const double t_b = ax_pt() ? ax_now() : 0;
    // AX_LIT_TILE: the literal stream is not decoded here; its chunk table is kept and the chunks are decoded tile by
    // tile next to the blocks that use them (ax_decode_tiled). Streams without the chunked layout: the usual way.
    if (tile) { const int k = ax_lit_chunks(zl, hdr.zlit_sz, ls, S.lcsz, S.lch); if (k < 0) return ACEAPEX_ERR_DATA; S.tiled = k == 1 && S.lcsz > 0 && S.lcsz <= ((size_t)256 << 10);   // a tile of G chunks must stay in L2 (silesia default: 4 chunks of 20 MB)
                if (ax_pt()) fprintf(stderr, "[phase] literal chunks: %zu of %zu B (%s)\n", S.lch.size(), S.lcsz, S.tiled ? "tiled" : "not tiled"); }
    if (S.tiled) { fse_multi_decomp(fst,3,budget); l=(uint8_t*)malloc(1); }
    else if (budget == 1) { litfn(&larg); fse_multi_decomp(fst,3,1); }
    else { AxThreads lt(1); lt.spawn(0,litfn,&larg);
           ax_guarded([&]{ fse_multi_decomp(fst,3,tok_t); }); lt.join(); }
    if (ax_pt()) fprintf(stderr, "[phase] checks+alloc %.3f s, entropy (lit %d + tok %d threads) %.3f s\n", t_b - t_a, lit_t, tok_t, ax_now() - t_b);
    if(!l || ax_oom()) return ACEAPEX_ERR_MEMORY;
    if(g_dec_err) return ACEAPEX_ERR_DATA;

    // Every block's stream slice must lie inside its decoded stream.
    for (size_t b = 0; b < hdr.num_blocks; b++) {
        const BlockOffsets& bo = S.boffs[b];
        if (bo.lit_off + bo.lit_sz > ls || bo.lit_off > ls ||
            bo.off_off + bo.off_sz > os || bo.off_off > os ||
            bo.len_off + bo.len_sz > ns || bo.len_off > ns ||
            bo.cmd_off + bo.cmd_sz > cs || bo.cmd_off > cs)
            return ACEAPEX_ERR_DATA;
    }
    S.ls=ls; S.os=os; S.ns=ns; S.cs=cs;
    return (int64_t)hdr.orig_size;
}

// AX_LIT_TILE (default 1; 0 = literal stream first, then the blocks): literals and blocks in one pass. The blocks are cut into groups whose literal slices span at most
// AX_TILE_CHUNKS chunks (+1 shared with the next group, decoded by both); a thread decodes a group's chunks into its
// tile (<= ~0.5 MiB: stays in L2), then decodes the group's blocks from the tile straight into the output. The 3 GB
// literal stream of T2T is never written to memory and read back. Blocks are independent (every match lies in its
// block), so groups run in any order.
static int64_t ax_decode_tiled(AxStreams& S, uint8_t* dst, int budget) {
    const size_t CH = S.lcsz, nb = S.hdr.num_blocks, bs = S.hdr.block_size, osz = S.hdr.orig_size, NC = S.lch.size();
    static const size_t G = [] { const char* e = ax_getenv("AX_TILE_CHUNKS"); size_t v = e ? strtoull(e, 0, 10) : 8; return v ? v : 1; }();
    struct Item { size_t b0, b1, k0, k1; };
    std::vector<Item> items;
    auto kof = [&](size_t off) { size_t k = off / CH; return k < NC ? k : (NC ? NC - 1 : 0); };
    for (size_t b = 0; b < nb; ) {
        Item it{b, b, (size_t)-1, 0};
        while (b < nb) {
            const BlockOffsets& bo = S.boffs[b];
            const size_t lo = kof(bo.lit_off), hi = bo.lit_sz ? kof(bo.lit_off + bo.lit_sz - 1) : lo;
            const size_t k0 = std::min(it.k0, lo), k1 = std::max(it.k1, hi);
            if (it.b1 > it.b0 && k1 - k0 + 1 > G + 1) break;
            it.k0 = k0; it.k1 = k1; it.b1 = ++b;
        }
        items.push_back(it);
    }
    // each thread a contiguous run of groups: the chunk a group shares with the next one is moved to the front of the
    // tile instead of being decoded again (decoded twice only where two threads meet)
    const int T = std::max(1, std::min<int>(budget, (int)items.size()));
    // AX_BLOCK_TIMES=<file> (diagnostics, tuning builds): per group its thread, start and end (ns from the call), the
    // literal chunk time and per block its decode time - results/reality-2026-10-02.log T2
    static const char* const times_path = ax_getenv("AX_BLOCK_TIMES");
    // AX_SCHED_COST (tuning builds; default 0 = equal numbers of groups): 1 = contiguous runs of equal estimated cost,
    // 2 = LPT (groups by cost, largest first, to the least loaded thread; each thread keeps its groups in stream order).
    // The cost of a group is known once the token streams are decoded: literal bytes + AX_COST_CMD x command bytes.
    static const int sched = [] { const char* e = ax_getenv("AX_SCHED_COST"); return e ? atoi(e) : 0; }();
    static const double ccmd = [] { const char* e = ax_getenv("AX_COST_CMD"); return e ? atof(e) : 4.0; }();
    std::vector<std::vector<size_t>> lists; std::vector<size_t> cut((size_t)T + 1, 0);
    for (int t = 0; t <= T; t++) cut[(size_t)t] = items.size() * (size_t)t / (size_t)T;
    if (sched && T > 1) {
        std::vector<double> cost(items.size()); double tot = 0;
        for (size_t i = 0; i < items.size(); i++) { double c = 0; for (size_t b = items[i].b0; b < items[i].b1; b++) c += (double)S.boffs[b].lit_sz + ccmd * (double)S.boffs[b].cmd_sz; cost[i] = c; tot += c; }
        if (sched == 1) { double acc = 0; int t = 1; for (size_t i = 0; i < items.size() && t < T; i++) { acc += cost[i]; while (t < T && acc >= tot * t / T) cut[(size_t)t++] = i + 1; } for (; t < T; t++) cut[(size_t)t] = items.size(); }
        else { std::vector<size_t> ix(items.size()); for (size_t i = 0; i < ix.size(); i++) ix[i] = i;
               std::stable_sort(ix.begin(), ix.end(), [&](size_t x, size_t y) { return cost[x] > cost[y]; });
               lists.assign((size_t)T, {}); std::vector<double> load((size_t)T, 0);
               for (size_t i : ix) { const size_t t = (size_t)(std::min_element(load.begin(), load.end()) - load.begin()); lists[t].push_back(i); load[t] += cost[i]; }
               for (auto& l : lists) std::sort(l.begin(), l.end()); }
    }
    struct GT { uint32_t t; double s, e, lit; };
    std::vector<GT> gt(times_path ? items.size() : 0); std::vector<float> bt(times_path ? nb : 0); const double tc0 = ax_now();
    struct Ctx { AxStreams* S; uint8_t* dst; const Item* items; size_t n; int T; size_t CH, bs, osz; bool nt; std::atomic<int> bad;
                 const size_t* cut; const std::vector<std::vector<size_t>>* lists; GT* gt; float* bt; double tc0; };
    Ctx cx{&S, dst, items.data(), items.size(), T, CH, bs, osz, ax_nt_for(budget, osz), {0}, cut.data(), &lists, gt.empty() ? nullptr : gt.data(), bt.empty() ? nullptr : bt.data(), tc0};
    struct Arg { Ctx* c; int t; };
    auto fn = [](void* v) -> void* {
        Arg* ar = (Arg*)v; Ctx* c = ar->c; std::vector<uint8_t> tile, nbuf;
        const bool lst = !c->lists->empty(); const std::vector<size_t>* L = lst ? &(*c->lists)[(size_t)ar->t] : nullptr;
        const size_t i0 = lst ? 0 : c->cut[ar->t], i1 = lst ? L->size() : c->cut[ar->t + 1];
        size_t have = (size_t)-1;                                   // chunk at the end of the tile from the previous group
        size_t have_t0 = 0, have_sz = 0;
        for (size_t q = i0; q < i1; q++) {
            const size_t i = lst ? (*L)[q] : q;
            const double gs = c->gt ? ax_now() : 0;
            const Item& it = c->items[i]; const size_t t0 = it.k0 * c->CH;
            const size_t tsz = std::min((it.k1 + 1) * c->CH, (size_t)c->S->ls) - t0;
            if (tile.size() < tsz + 64) { std::vector<uint8_t> nt(tsz + 64 + c->CH); if (have != (size_t)-1) memcpy(nt.data(), tile.data(), have_sz); tile.swap(nt); }
            size_t k = it.k0;
            if (have == it.k0) {                                    // the shared chunk: from its place in the last tile to the front
                const size_t at = it.k0 * c->CH - have_t0, len = std::min(c->CH, have_t0 + have_sz - it.k0 * c->CH);
                memmove(tile.data(), tile.data() + at, len); k++; }
            for (; k <= it.k1 && k < c->S->lch.size(); k++)
                if (!ax_lit_chunk_decode(c->S->lch[k], tile.data() + (c->S->lch[k].off - t0))) c->bad = 1;
            const double gl = c->gt ? ax_now() : 0; double bl = gl;
            for (size_t b = it.b0; b < it.b1; b++) {
                const BlockOffsets& bo = c->S->boffs[b]; const size_t bstart = b * c->bs;
                const size_t bsize = c->osz > bstart ? std::min(c->bs, c->osz - bstart) : 0;
                if (!bsize) continue;
                if (bo.lit_sz && (bo.lit_off < t0 || bo.lit_off + bo.lit_sz > t0 + tsz)) { c->bad = 1; continue; }
                ax_block_out(c->dst, bstart, bsize, bo, tile.data() + (bo.lit_sz ? bo.lit_off - t0 : 0), c->S->o, c->S->n, c->S->c, nbuf, c->nt);
                if (c->bt) { const double x = ax_now(); c->bt[b] = (float)((x - bl) * 1e9); bl = x; }
            }
            if (c->gt) c->gt[i] = {(uint32_t)ar->t, (gs - c->tc0) * 1e9, (ax_now() - c->tc0) * 1e9, (gl - gs) * 1e9};
            have = it.k1; have_t0 = t0; have_sz = tsz;
        }
        ax_nt_fence(); return nullptr;
    };
    std::vector<Arg> args((size_t)T); for (int t = 0; t < T; t++) args[(size_t)t] = {&cx, t};
    if (T == 1) fn(&args[0]);
    else { AxThreads th((size_t)T - 1); for (int t = 1; t < T; t++) th.spawn((size_t)t - 1, fn, &args[(size_t)t]);
           ax_guard(fn, &args[0]); th.join(); }
    if (times_path) { FILE* f = fopen(times_path, "w");
        if (f) { fprintf(f, "#threads %d sched %d wall_ns %.0f\n", T, sched, (ax_now() - tc0) * 1e9);
                 for (size_t i = 0; i < items.size(); i++) { fprintf(f, "G\t%zu\t%u\t%.0f\t%.0f\t%.0f\t%zu\t%zu\n", i, gt[i].t, gt[i].s, gt[i].e, gt[i].lit, items[i].b0, items[i].b1); }
                 for (size_t b = 0; b < nb; b++) fprintf(f, "B\t%zu\t%.0f\t%llu\t%llu\t%llu\n", b, (double)bt[b], (unsigned long long)S.boffs[b].lit_sz, (unsigned long long)S.boffs[b].cmd_sz, (unsigned long long)S.boffs[b].off_sz);
                 fclose(f); } }
    return cx.bad ? ACEAPEX_ERR_DATA : (int64_t)osz;
}

static int64_t ax_decompress_mt_impl(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity, int threads)
{
#ifdef ACEAPEX_ENV_TUNING
    if (axlm::is(src, src_size)) return dst ? axlm::decompress((const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, threads) : ACEAPEX_ERR_DATA;
#endif
    if (src && axf_is_v3((const uint8_t*)src, src_size)) return dst ? ax_fast_decompress((const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, threads) : ACEAPEX_ERR_DATA;
    const double t0 = ax_pt() ? ax_now() : 0;
    static const bool tile = [] { const char* e = ax_getenv("AX_LIT_TILE"); return e ? atoi(e) != 0 : true; }();   // default 1
    AxStreams S; int64_t r = ax_entropy_decode(src, src_size, threads, S, tile);
    if (r <= 0) return r;                                    // error, or the empty archive
    const double t1 = ax_pt() ? ax_now() : 0;
    if (S.hdr.orig_size > dst_capacity) return ACEAPEX_ERR_BUFFER;
    const int budget=ax_decode_budget(threads,S.hdr.orig_size);
    if (S.tiled) { const int64_t q = ax_decode_tiled(S, (uint8_t*)dst, budget); if (q < 0) return q; }
    else parallel_decode(S.l,S.o,S.n,S.c,S.boffs.data(),S.hdr.num_blocks,
                    (uint8_t*)dst,S.hdr.orig_size,S.hdr.block_size,budget);
    const double t2 = ax_pt() ? ax_now() : 0;
    free(S.l);free(S.o);free(S.n);free(S.c); S.l=S.o=S.n=S.c=nullptr;
    if (ax_pt()) fprintf(stderr, "[phase] entropy total %.3f s, match (%d threads) %.3f s, free %.3f s\n", t1 - t0, budget, t2 - t1, ax_now() - t2);
    // a literal chunk decoded inside the match phase (tiled path: DNA, zstd) reports through g_dec_err (lzbench #357)
    if (g_dec_err) return ACEAPEX_ERR_DATA;
    return (int64_t)S.hdr.orig_size;
}

static int ax_decode_streams_impl(const void* src, size_t src_size, aceapex_streams_t* out)
{
    if (!out) return ACEAPEX_ERR_DATA;
    memset(out, 0, sizeof(*out));
    if (src && axf_is_v3((const uint8_t*)src, src_size)) return ACEAPEX_ERR_DATA;   // the fast profile has no entropy streams
    AxStreams S; int64_t r = ax_entropy_decode(src, src_size, 0, S);
    if (r < 0) return (int)r;
    if (r == 0) { out->block_size = S.hdr.block_size; return 0; }   // empty archive: no streams
    std::vector<BlockOffsets>* bv = new std::vector<BlockOffsets>(std::move(S.boffs));
    out->lit=S.l; out->off=S.o; out->len=S.n; out->cmd=S.c; S.l=S.o=S.n=S.c=nullptr;   // the caller's now
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

static int64_t ax_decompress_region_impl(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    uint64_t    offset, uint64_t length)
{
#ifdef ACEAPEX_ENV_TUNING
    if (axlm::is(src, src_size)) return dst ? axlm::region((const uint8_t*)src, src_size, (uint8_t*)dst, dst_capacity, offset, length) : ACEAPEX_ERR_DATA;
#endif
    if (!src || src_size < sizeof(AetHeader)) return ACEAPEX_ERR_DATA;
    const uint8_t* p = (const uint8_t*)src;
    if (axf_is_v3(p, src_size)) {                      // fast profile: the covering blocks, one thread
        axf_arc_t a; if (axf_open(p, src_size, &a) != 0 || !dst) return ACEAPEX_ERR_DATA;
        return ax_fast_region(a, (uint8_t*)dst, dst_capacity, offset, length, 1);
    }
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
        if(!span){ free(lit);free(off);free(len);free(cmd); ax_nomem();   // the call reports ACEAPEX_ERR_MEMORY (2.6.3)
                   for(size_t k=G.first;k<G.last;k++) t->w[k].dst=nullptr;
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

static int64_t ax_decompress_ranges_impl(
    const void* src, size_t src_size,
    aceapex_range_t* ranges, size_t count, int threads)
{
    g_dec_err=0;
    if(!src||!ranges) return ACEAPEX_ERR_DATA;
    if(count==0) return 0;
    if(src_size<sizeof(AetHeader)) return ACEAPEX_ERR_DATA;

    const uint8_t* p=(const uint8_t*)src;
    if (axf_is_v3(p, src_size)) return ax_fast_ranges(p, src_size, ranges, count, threads);
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
        AxThreads th((size_t)lanes);
        for(int k=0;k<lanes;k++) th.spawn((size_t)k,batch_worker,&t);
        th.join();
    }

    int64_t ok=0;
    for(const RangeWork& r : w)
        if(r.dst){ ranges[r.idx].written=(int64_t)r.length; ok++; }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// Streaming decode (aceapex.h aceapex_decompress_stream): the tile path (AX_LIT_TILE) over a read callback.
// Main thread: header, chunk tables (literal: chunked FSE layout required; tokens: the chunk table of each stream),
// the block table in windows of 2048 entries, groups of blocks (literal span <= G+1 chunks, <= AX_STREAM_BLOCKS blocks
// so a group's output stays <= ~1 MiB), and the token chunks the group needs - decoded by the main thread in order into
// shared buffers the groups hold (a chunk decoded once, freed when its last group is written). Workers: a group's literal
// chunks read and decoded into the worker's tile, the group's blocks decoded from it into the worker's output buffer,
// which is handed to the write callback in group order (a sequence counter). Memory: threads x (tile + output + read
// buffer) + the chunk tables + the groups in flight; nothing proportional to the archive.
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "xxhash.h"
namespace {
struct StChunk { uint64_t off, raw, pos, csz; bool tg; };                   // a literal chunk: stream offset, size, file offset of its body
struct StTok { size_t S = 0, CH = 0, nc = 0; std::vector<uint64_t> cs, pos; };   // a token stream: size, chunk, entries and file offsets
struct StTokBuf { std::vector<uint8_t> d; uint64_t off0 = 0; };                // decoded chunk (off0: its offset in the literal stream)
struct StGroup { uint64_t seq, b0, b1, k0, k1; std::vector<BlockOffsets> bo;    // blocks, literal chunk span, block entries
                 uint64_t t0[4]; size_t tsz[4]; std::vector<std::shared_ptr<StTokBuf>> tok[4];   // per token stream: window start (stream offset), its chunks
                 uint64_t llo = 0, lhi = 0; std::vector<std::shared_ptr<StTokBuf>> lit; };   // 4-chunk layout: the group's literal span and the decoded chunks it touches
struct StCtx {
    aceapex_read_fn rd; void* rctx; aceapex_write_fn wr; void* wctx;
    uint64_t zoff[4]; uint64_t orig, bs, nb; size_t csz; std::vector<StChunk> lch; StTok tk[4];
    bool four = false;                                                          // the 4-chunk literal layout (bit 62 without 61): chunks decoded once by the main thread
    std::mutex m; std::condition_variable cv_q, cv_w; std::deque<std::shared_ptr<StGroup>> q; bool done = false; size_t qcap;
    uint64_t next_write = 0; std::atomic<int> err{0}; XXH3_state_t* xs = nullptr; unsigned flags;
};
// A worker's buffer: not zero-filled, grown only to what a group really needs (2.6.2: until 2.6.2 the worker sized its
// output and tile from the header's block_size, ~2 GiB per worker for a corrupt archive - api_fuzz under MemoryMax 8G).
// Growth: to the requested size, at most doubling and at most 16 MiB at a time; `keep` copies the bytes already there.
struct StBuf {
    std::unique_ptr<uint8_t[]> p; size_t n = 0;
    uint8_t* data() { return p.get(); }
    void need(size_t k, bool keep = false) {
        if (k <= n) return;
        const size_t m = std::max(k, std::min(n + std::min(n, (size_t)16 << 20), (size_t)SIZE_MAX - 64));
        std::unique_ptr<uint8_t[]> q(new uint8_t[m]); if (keep && n) memcpy(q.get(), p.get(), n); p.swap(q); n = m; }
};
static bool st_read(StCtx& c, uint64_t off, void* buf, size_t len) {
    while (len) { const int64_t r = c.rd(c.rctx, off, buf, len); if (r <= 0) return false; off += (uint64_t)r; buf = (uint8_t*)buf + r; len -= (size_t)r; }
    return true;
}
static void* st_work(StCtx& c) {
    StBuf tile, rbuf, out; std::vector<uint8_t> tw[4];
    for (;;) {
        std::shared_ptr<StGroup> g;
        { std::unique_lock<std::mutex> lk(c.m); c.cv_q.wait(lk, [&] { return !c.q.empty() || c.done || c.err; });
          if (c.err || (c.q.empty() && c.done)) return nullptr;
          g = c.q.front(); c.q.pop_front(); }
        c.cv_q.notify_all();
        uint64_t t0 = g->k0 * c.csz, tsz = std::min((g->k1 + 1) * c.csz, (uint64_t)c.lch.empty() ? 0 : c.lch.back().off + c.lch.back().raw) - t0;
        if (c.four) { t0 = g->llo; tsz = g->lhi - g->llo; }                     // the group's literals copied out of the shared decoded chunks
        tile.need(tsz + 64);
        bool bad = false;
        if (c.four) for (const auto& lb : g->lit) {
            const uint64_t a = std::max(t0, lb->off0), e = std::min(t0 + tsz, lb->off0 + lb->d.size());
            if (a < e) memcpy(tile.data() + (a - t0), lb->d.data() + (a - lb->off0), (size_t)(e - a)); }
        for (uint64_t k = g->k0; k <= g->k1 && !bad && !c.four; k++) {
            const StChunk& ch = c.lch[k];
            rbuf.need(ch.csz + 64);
            if (ch.csz && !st_read(c, ch.pos, rbuf.data(), ch.csz)) { bad = true; break; }
            const AxLitChunk d{(size_t)ch.off, (size_t)ch.raw, rbuf.data(), (size_t)ch.csz, ch.tg};
            if (!ax_lit_chunk_decode(d, tile.data() + (ch.off - t0))) bad = true;
        }
        const uint64_t osz = std::min((g->b1) * c.bs, c.orig) - g->b0 * c.bs;
        for (int st = 1; st < 4; st++) {                                   // the group's token chunks, contiguous (a block's slice may cross a chunk)
            size_t n = 0; for (auto& b : g->tok[st]) n += b->d.size();
            if (tw[st].size() < n + 64) tw[st].resize(n + 64);
            size_t o = 0; for (auto& b : g->tok[st]) { memcpy(tw[st].data() + o, b->d.data(), b->d.size()); o += b->d.size(); }
            g->tsz[st] = n; }
        for (uint64_t b = g->b0; b < g->b1 && !bad; b++) {
            const BlockOffsets& bo = g->bo[b - g->b0]; const uint64_t bstart = b * c.bs, bsize = std::min(c.bs, c.orig - bstart);
            if (bo.lit_sz && (bo.lit_off < t0 || bo.lit_off + bo.lit_sz > t0 + tsz)) { bad = true; break; }
            const uint8_t* tp[4] = {nullptr, nullptr, nullptr, nullptr}; const uint64_t so[4] = {0, bo.off_off, bo.len_off, bo.cmd_off}, ss[4] = {0, bo.off_sz, bo.len_sz, bo.cmd_sz};
            for (int st = 1; st < 4 && !bad; st++) {
                if (!ss[st]) { tp[st] = tile.data(); continue; }
                if (so[st] < g->t0[st] || so[st] - g->t0[st] + ss[st] > g->tsz[st]) { bad = true; break; }
                tp[st] = tw[st].data() + (so[st] - g->t0[st]);
            }
            if (bad) break;
            out.need((size_t)(bstart - g->b0 * c.bs + bsize) + 64, true);   // up to this block's end: the output as it is written
            decompress_streams(out.data() + (bstart - g->b0 * c.bs), (size_t)bsize, tile.data() + (bo.lit_sz ? bo.lit_off - t0 : 0), (size_t)bo.lit_sz,
                               tp[1], (size_t)ss[1], tp[2], (size_t)ss[2], tp[3], (size_t)ss[3]);
        }
        { std::unique_lock<std::mutex> lk(c.m); c.cv_w.wait(lk, [&] { return c.next_write == g->seq || c.err; });
          if (!c.err) {
              if (bad) c.err = ACEAPEX_ERR_DATA;
              else { if (c.xs) XXH3_64bits_update(c.xs, out.data(), (size_t)osz);
                     if (c.wr(c.wctx, out.data(), (size_t)osz)) c.err = ACEAPEX_ERR_BUFFER; }
          }
          c.next_write = g->seq + 1; }
        c.cv_w.notify_all(); c.cv_q.notify_all();
        if (c.err) return nullptr;
    }
}
// 2.6.3: a worker that runs out of memory ends the call (ACEAPEX_ERR_MEMORY) and wakes the others - its group would
// otherwise never be written and the workers after it would wait for it
static void* st_worker(void* v) {
    StCtx& c = *(StCtx*)v;
    try { return st_work(c); }
    catch (...) { ax_nomem(); { std::lock_guard<std::mutex> lk(c.m); if (!c.err) c.err = ACEAPEX_ERR_MEMORY; }
                  c.cv_q.notify_all(); c.cv_w.notify_all(); return nullptr; }
}
}  // namespace

static int64_t ax_decompress_stream_impl(aceapex_read_fn rd, void* rctx, aceapex_write_fn wr, void* wctx, int threads, unsigned flags)
{
    if (!rd || !wr || (flags & ~ACEAPEX_STREAM_VERIFY)) return ACEAPEX_ERR_BUFFER;
    StCtx c; c.rd = rd; c.rctx = rctx; c.wr = wr; c.wctx = wctx; c.flags = flags;
    AetHeader hdr; if (!st_read(c, 0, &hdr, sizeof hdr)) return ACEAPEX_ERR_DATA;
    if (memcmp(hdr.magic, "ACEPX2\0\0", 8) != 0) return ACEAPEX_ERR_DATA;
    if (axf_is_v3((const uint8_t*)&hdr, sizeof hdr)) return ax_fast_stream(rd, rctx, wr, wctx, threads, flags, (const uint8_t*)&hdr);
    if (hdr.num_blocks == 0) { if (!ax_is_empty_archive(hdr)) return ACEAPEX_ERR_DATA; return 0; }
    if (hdr.block_size == 0 || (uint64_t)hdr.num_blocks * hdr.block_size < hdr.orig_size || (uint64_t)(hdr.num_blocks - 1) * hdr.block_size >= hdr.orig_size) return ACEAPEX_ERR_DATA;
    c.orig = hdr.orig_size; c.bs = hdr.block_size; c.nb = hdr.num_blocks;
    uint64_t p = sizeof hdr + (uint64_t)hdr.num_blocks * sizeof(BlockOffsets);
    const uint64_t zsz[4] = {hdr.zlit_sz, hdr.zoff_sz, hdr.zlen_sz, hdr.zcmd_sz};
    for (int st = 0; st < 4; st++) { if (zsz[st] > ~0ull - p) return ACEAPEX_ERR_DATA; c.zoff[st] = p; p += zsz[st]; }
    { uint8_t last; if (!st_read(c, p - 1, &last, 1)) return ACEAPEX_ERR_DATA; }  // the archive reaches the end its header claims: stream sizes are bounded by real bytes
    // token streams: header + chunk table, file offset of every chunk
    for (int st = 1; st < 4; st++) {
        StTok& t = c.tk[st]; if (zsz[st] < 8) continue;
        uint8_t h8[8]; if (!st_read(c, c.zoff[st], h8, 8)) return ACEAPEX_ERR_DATA;
        uint64_t h; memcpy(&h, h8, 8); if (h >> 63) return ACEAPEX_ERR_DATA;
        t.S = fse_stream_size(h8); t.CH = fse_stream_chunk(h8); if (!t.CH) return ACEAPEX_ERR_DATA;
        t.nc = (t.S + t.CH - 1) / t.CH; if (t.nc > (zsz[st] - 8) / 8) return ACEAPEX_ERR_DATA;
        std::vector<uint8_t> tb(t.nc * 8); if (t.nc && !st_read(c, c.zoff[st] + 8, tb.data(), tb.size())) return ACEAPEX_ERR_DATA;
        t.cs.resize(t.nc); t.pos.resize(t.nc); uint64_t q = c.zoff[st] + 8 + 8 * t.nc, end = c.zoff[st] + zsz[st];
        for (size_t i = 0; i < t.nc; i++) { memcpy(&t.cs[i], tb.data() + 8 * i, 8); if (ax_ce_bad(t.cs[i])) return ACEAPEX_ERR_DATA;
            const uint64_t raw = std::min<uint64_t>(t.CH, t.S - i * t.CH), z = ax_ce_raw(t.cs[i]) ? raw : ax_ce_size(t.cs[i]);
            if (z > end - q) return ACEAPEX_ERR_DATA; t.pos[i] = q; q += z; }
    }
    // literal stream: the FSE layout (bit 62). Chunked (bit 61, what DNA archives carry): a chunk size and a table of
    // chunks of that size. Without bit 61 (the default profile on other data, 2.1+): 4 chunks of ceil(size / 4), the
    // table right after the 8-byte header (lit_table_check) - streamed after 2.6.1 (FREEZE-3), its 4 chunks decoded
    // once by the main thread and shared by the groups (memory: two decoded chunks, about half the literal stream).
    { if (zsz[0] < 8) { if (hdr.orig_size) return ACEAPEX_ERR_DATA; }
      uint8_t h16[16] = {0}; if (!st_read(c, c.zoff[0], h16, zsz[0] >= 16 ? 16 : 8)) return ACEAPEX_ERR_DATA;
      uint64_t h, cz; memcpy(&h, h16, 8); memcpy(&cz, h16 + 8, 8);
      const bool chunked = (h >> 61) & 1, tagged = (h >> 60) & 1;
      if (!((h >> 62) & 1) || (h >> 63) || (tagged && !chunked)) return ACEAPEX_ERR_DATA;   // the pre-FSE layout: not streamed
      const uint64_t lsz = h & ~((uint64_t(1) << 62) | (uint64_t(1) << 61) | (uint64_t(1) << 60));
      if (lsz > c.orig) return ACEAPEX_ERR_DATA;
      const uint64_t hb = chunked ? 16 : 8;
      if (chunked) { if (zsz[0] < 16 || !cz || cz % 4096) return ACEAPEX_ERR_DATA; }
      else { cz = (lsz + 3) / 4; c.four = true; }
      c.csz = (size_t)cz;
      const uint64_t NW = chunked ? (lsz + cz - 1) / cz : 4; if (NW > (zsz[0] - hb) / 8) return ACEAPEX_ERR_DATA;
      std::vector<uint8_t> tb(NW * 8); if (NW && !st_read(c, c.zoff[0] + hb, tb.data(), tb.size())) return ACEAPEX_ERR_DATA;
      uint64_t q = c.zoff[0] + hb + 8 * NW, end = c.zoff[0] + zsz[0]; c.lch.resize(NW);
      for (uint64_t t = 0; t < NW; t++) { uint64_t z; memcpy(&z, tb.data() + 8 * t, 8); if (z > end - q) return ACEAPEX_ERR_DATA;
          uint64_t off = t * cz; if (off > lsz) off = lsz;
          const uint64_t raw = off + cz <= lsz ? cz : lsz - off; c.lch[t] = {off, raw, q, z, tagged}; q += z; }
      if (NW && c.lch.back().off + c.lch.back().raw != lsz) return ACEAPEX_ERR_DATA; }
    const int T = std::max(1, ax_decode_budget(threads, c.orig)); c.qcap = (size_t)T * 2;
    static const uint64_t G = [] { const char* e = ax_getenv("AX_TILE_CHUNKS"); uint64_t v = e ? strtoull(e, 0, 10) : 8; return v ? v : 1; }();
    static const uint64_t MAXB = [] { const char* e = ax_getenv("AX_STREAM_BLOCKS"); uint64_t v = e ? strtoull(e, 0, 10) : 0; return v ? v : 0; }();
    const uint64_t maxb = MAXB ? MAXB : std::max<uint64_t>(1, ((uint64_t)1 << 20) / c.bs);
    if (flags & ACEAPEX_STREAM_VERIFY) { c.xs = XXH3_createState(); if (!c.xs) return ACEAPEX_ERR_MEMORY; XXH3_64bits_reset(c.xs); }
    struct XsFree { XXH3_state_t*& x; ~XsFree() { if (x) XXH3_freeState(x); x = nullptr; } } xs_free{c.xs};
    AxThreads th((size_t)T);
    for (int t = 0; t < T; t++) th.spawn((size_t)t, st_worker, &c);
    if (ax_oom()) c.err = ACEAPEX_ERR_MEMORY;                                     // a worker not started: the groups would wait for it
    // 2.6.3: an allocation that fails on the main thread (groups, token chunks) ends the call through c.err: the
    // workers are told and joined before it returns
    bool bad = false;
    ax_guarded([&] {
    // the main thread: block table in windows, groups, their token chunks (decoded once, shared)
    std::shared_ptr<StTokBuf> last[4]; uint64_t lastk[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    std::vector<BlockOffsets> win; const uint64_t WB = 2048; uint64_t wb0 = 0, wb1 = 0, seq = 0;
    auto tokchunk = [&](int st, uint64_t k) -> std::shared_ptr<StTokBuf> {
        if (lastk[st] == k) return last[st];
        const StTok& t = c.tk[st]; if (k >= t.nc) return nullptr;
        auto b = std::make_shared<StTokBuf>(); const uint64_t raw = std::min<uint64_t>(t.CH, t.S - k * t.CH), z = ax_ce_raw(t.cs[k]) ? raw : ax_ce_size(t.cs[k]);
        std::vector<uint8_t> zb(z + 64); if (z && !st_read(c, t.pos[k], zb.data(), z)) return nullptr;
        b->d.resize(raw + 64); if (!ax_tok_chunk(t.cs[k], b->d.data(), raw, zb.data(), z)) return nullptr;
        b->d.resize(raw); last[st] = b; lastk[st] = k; return b; };
    std::shared_ptr<StTokBuf> lcache[2]; uint64_t lck[2] = {~0ull, ~0ull};   // 4-chunk layout: the last two decoded literal chunks
    auto litchunk = [&](uint64_t k) -> std::shared_ptr<StTokBuf> {
        for (int i = 0; i < 2; i++) if (lck[i] == k) return lcache[i];
        if (k >= c.lch.size()) return nullptr;
        const StChunk& ch = c.lch[k]; auto b = std::make_shared<StTokBuf>(); b->off0 = ch.off;
        std::vector<uint8_t> zb(ch.csz + 64); if (ch.csz && !st_read(c, ch.pos, zb.data(), ch.csz)) return nullptr;
        b->d.resize(ch.raw + 64);
        const AxLitChunk d{(size_t)ch.off, (size_t)ch.raw, zb.data(), (size_t)ch.csz, ch.tg};
        if (!ax_lit_chunk_decode(d, b->d.data())) return nullptr;
        b->d.resize(ch.raw);
        lcache[0] = lcache[1]; lck[0] = lck[1]; lcache[1] = b; lck[1] = k; return b; };
    for (uint64_t b = 0; b < c.nb && !bad; ) {
        auto g = std::make_shared<StGroup>(); g->seq = seq++; g->b0 = g->b1 = b; g->k0 = ~0ull; g->k1 = 0;
        for (int st = 1; st < 4; st++) g->t0[st] = ~0ull;
        uint64_t tk0[4] = {0, ~0ull, ~0ull, ~0ull}, tk1[4] = {0, 0, 0, 0};
        while (b < c.nb && g->b1 - g->b0 < maxb) {
            if (b >= wb1) { wb0 = b; wb1 = std::min(c.nb, b + WB); win.resize((size_t)(wb1 - wb0));
                if (!st_read(c, sizeof hdr + wb0 * sizeof(BlockOffsets), win.data(), (size_t)(wb1 - wb0) * sizeof(BlockOffsets))) { bad = true; break; } }
            const BlockOffsets& bo = win[(size_t)(b - wb0)];
            const uint64_t ls = c.lch.empty() ? 0 : c.lch.back().off + c.lch.back().raw;
            if (bo.lit_off > ls || bo.lit_sz > ls - bo.lit_off) { bad = true; break; }
            const uint64_t lo = c.csz ? bo.lit_off / c.csz : 0, hi = c.csz ? (bo.lit_sz ? (bo.lit_off + bo.lit_sz - 1) / c.csz : lo) : 0;
            const uint64_t k0 = std::min(g->k0, lo), k1 = std::max(g->k1, hi);
            if (g->b1 > g->b0 && k1 - k0 + 1 > G + 1) break;
            const uint64_t so[4] = {0, bo.off_off, bo.len_off, bo.cmd_off}, ss[4] = {0, bo.off_sz, bo.len_sz, bo.cmd_sz}; bool ok = true;
            for (int st = 1; st < 4; st++) { if (!ss[st]) continue; const StTok& t = c.tk[st];
                if (!t.CH || so[st] > t.S || ss[st] > t.S - so[st]) { ok = false; break; }
                tk0[st] = std::min(tk0[st], so[st] / t.CH); tk1[st] = std::max(tk1[st], (so[st] + ss[st] - 1) / t.CH); }
            if (!ok) { bad = true; break; }
            if (c.four) { if (g->b1 == g->b0) g->llo = bo.lit_off; g->lhi = std::max(g->lhi, bo.lit_off + bo.lit_sz); if (bo.lit_off < g->llo) { bad = true; break; } }
            g->k0 = k0; g->k1 = k1; g->bo.push_back(bo); g->b1 = ++b;
        }
        if (bad) break;
        if (c.four && g->lhi > g->llo) for (uint64_t k = g->llo / c.csz; k <= (g->lhi - 1) / c.csz && !bad; k++) {
            auto lb = litchunk(k); if (!lb) { bad = true; break; } g->lit.push_back(lb); }
        if (bad) break;
        for (int st = 1; st < 4 && !bad; st++) { if (tk0[st] == ~0ull) continue; g->t0[st] = tk0[st] * c.tk[st].CH;
            for (uint64_t k = tk0[st]; k <= tk1[st]; k++) { auto tb = tokchunk(st, k); if (!tb) { bad = true; break; } g->tok[st].push_back(tb); } }
        if (bad) break;
        { std::unique_lock<std::mutex> lk(c.m); c.cv_q.wait(lk, [&] { return c.q.size() < c.qcap || c.err; }); if (c.err) break; c.q.push_back(g); }
        c.cv_q.notify_all();
    }
    });
    { std::lock_guard<std::mutex> lk(c.m); if (ax_oom() && !c.err) c.err = ACEAPEX_ERR_MEMORY; if (bad && !c.err) c.err = ACEAPEX_ERR_DATA; c.done = true; }
    c.cv_q.notify_all(); c.cv_w.notify_all();
    th.join();
    int64_t r = c.err ? (int64_t)c.err.load() : (int64_t)c.orig;
    if (c.xs) { const uint64_t h = XXH3_64bits_digest(c.xs); uint64_t want; memcpy(&want, hdr.xxhash, 8); if (!c.err && h != want) r = ACEAPEX_ERR_DATA; }
    return r;
}

// ---- public entry points (2.6.3) -------------------------------------------------------------------------------------
// One flag per call (t_oom, inherited by the threads the call starts; the previous one restored on return, so calls
// nest and run on many threads at once). An exception that reaches here, or a flag set anywhere in the call, gives
// ACEAPEX_ERR_MEMORY; every thread of the call is joined before its impl returns or unwinds (AxThreads).
namespace {
// The decode error flag (g_dec_err) is per call too: a flag left set by an earlier call on the same thread (ax_nomem, a
// corrupt archive) no longer fails the next one (aceapex_decompress_stream does not reset it).
struct AxCall {
    std::atomic<int> oom{0}, err{0}; std::atomic<int> *prev, *prev_err;
    AxCall() : prev(t_oom), prev_err(t_dec_err) { t_oom = &oom; t_dec_err = &err; }
    ~AxCall() { t_oom = prev; t_dec_err = prev_err; }
};
template <class R, class F> static R ax_api(F&& f) noexcept {
    AxCall call;
    try { const R r = f(); return call.oom.load() ? (R)ACEAPEX_ERR_MEMORY : r; }
    catch (...) { return (R)ACEAPEX_ERR_MEMORY; }
}
}  // namespace

int64_t aceapex_compress(const void* src, size_t src_size, void* dst, size_t dst_capacity, int level, int threads) {
    return ax_api<int64_t>([&] { return ax_compress_impl(src, src_size, dst, dst_capacity, level, threads); }); }
int64_t aceapex_decompress(const void* src, size_t src_size, void* dst, size_t dst_capacity) {
    return ax_api<int64_t>([&] { return ax_decompress_mt_impl(src, src_size, dst, dst_capacity, 0); }); }
int64_t aceapex_decompress_mt(const void* src, size_t src_size, void* dst, size_t dst_capacity, int threads) {
    return ax_api<int64_t>([&] { return ax_decompress_mt_impl(src, src_size, dst, dst_capacity, threads); }); }
int64_t aceapex_decompress_region(const void* src, size_t src_size, void* dst, size_t dst_capacity, uint64_t offset, uint64_t length) {
    return ax_api<int64_t>([&] { return ax_decompress_region_impl(src, src_size, dst, dst_capacity, offset, length); }); }
int64_t aceapex_decompress_ranges(const void* src, size_t src_size, aceapex_range_t* ranges, size_t count, int threads) {
    return ax_api<int64_t>([&] { return ax_decompress_ranges_impl(src, src_size, ranges, count, threads); }); }
int64_t aceapex_decompress_stream(aceapex_read_fn rd, void* rctx, aceapex_write_fn wr, void* wctx, int threads, unsigned flags) {
    return ax_api<int64_t>([&] { return ax_decompress_stream_impl(rd, rctx, wr, wctx, threads, flags); }); }
int aceapex_decode_streams(const void* src, size_t src_size, aceapex_streams_t* out) {
    const int r = ax_api<int>([&] { return ax_decode_streams_impl(src, src_size, out); });
    if (r < 0 && out && (out->lit || out->boffs_vec)) aceapex_streams_free(out);   // a failed call hands nothing out
    return r; }
