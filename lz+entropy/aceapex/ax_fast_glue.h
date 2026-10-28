// ax_fast_glue.h - the FAST profile (ACEPX2 header version 3) inside the C++ library and CLI: level mapping, the
// block pool for encode and decode, whole-archive and block-range decodes. Included by aceapex_main.cpp once the
// thread helpers (ax_run_pool, ax_decode_budget) exist. Format: ax_fast.h; encoder: ax_fast_enc.h.
#pragma once
#include "ax_fast.h"
#include "ax_fast_enc.h"

// Public level numbers 11..18 select the fast profile at its levels 1..8 (aceapex.h); 0 = not the fast profile.
static inline int ax_fast_level(int level) { return level >= 11 && level <= 18 ? level - 10 : 0; }

// src -> blocks (header + table are written by the caller from the vector). false: bad level.
static bool ax_fast_encode(const uint8_t* src, size_t n, int flevel, int threads,
                           std::vector<std::vector<uint8_t>>& blocks, axf::Params& P, uint32_t& bs) {
    if (!axf::level_params(flevel, P)) return false;
    bs = AXF_BLOCK_MAX;
    // tuning builds (the CLI, the repository's tools): ACEAPEX_BS sets the fast profile's block size too, 4096..2^20 bytes,
    // for matched-granularity comparisons (hw-apex-bench at g = 16 KiB); the library inside lzbench ignores it (ax_env.h)
    { const char* e = ax_getenv("ACEAPEX_BS"); if (e) { const unsigned long v = strtoul(e, 0, 10); if (v >= 4096 && v <= AXF_BLOCK_MAX) bs = (uint32_t)v; } }
    const uint32_t nb = (uint32_t)((n + bs - 1) / bs);
    blocks.assign(nb, std::vector<uint8_t>());
    axf::Job j; j.src = src; j.n = n; j.bs = bs; j.nb = nb; j.P = P; j.blocks = &blocks; j.next = 0; j.err = 0;
    const int T = std::max(1, std::min<int>(threads > 0 ? threads : 8, (int)nb));
    ax_run_pool(T, axf::enc_worker, &j);
    return true;
}
// whole archive -> dst (header + table + blocks), returns the archive size or ACEAPEX_ERR_BUFFER (-1) when cap is short
static int64_t ax_fast_assemble(const std::vector<std::vector<uint8_t>>& blocks, const axf::Params& P, uint32_t bs,
                                const uint8_t* src, size_t n, uint8_t* dst, size_t cap) {
    const size_t total = axf::archive_size(blocks);
    if (total > cap) return -1;
    const uint64_t hv = OUR_CHECKSUM(src, n);
    axf::write_header(dst, n, bs, (uint32_t)blocks.size(), hv, P.mm, total - 68 - 8 * (blocks.size() + 1));
    axf::write_body(dst + 68, blocks);
    return (int64_t)total;
}
// worst case for aceapex_compress_bound: header + table + every block at its bound (axf::enc_bound)
static inline size_t ax_fast_bound(size_t n) {
    const size_t nb = (n + AXF_BLOCK_MAX - 1) / AXF_BLOCK_MAX;
    return 68 + 8 * (nb + 1) + nb * axf::enc_bound(AXF_BLOCK_MAX, AXF_MM_MIN) + 64;
}

// ---- decode -------------------------------------------------------------------------------------------------
struct AxFastDec { axf_arc_t a; uint8_t* dst; uint32_t b0, b1; std::atomic<uint32_t> next; std::atomic<int> err; };
static void* ax_fast_dec_worker(void* arg) {
    AxFastDec* j = (AxFastDec*)arg;
    for (;;) {
        const uint32_t b = j->next.fetch_add(1); if (b >= j->b1 || j->err.load()) break;
        if (!axf_dec_block_of(&j->a, b, j->dst + (size_t)(b - j->b0) * j->a.bs)) j->err = 1;
    }
    return nullptr;
}
// blocks [b0, b1) of an opened archive into dst (contiguous, block b at (b - b0) * bs), on `threads` threads
static bool ax_fast_decode_blocks(const axf_arc_t& a, uint32_t b0, uint32_t b1, uint8_t* dst, int threads) {
    AxFastDec j; j.a = a; j.dst = dst; j.b0 = b0; j.b1 = b1; j.next = b0; j.err = 0;
    const int T = std::max(1, std::min<int>(threads, (int)(b1 - b0)));
    ax_run_pool(T, ax_fast_dec_worker, &j);
    return j.err.load() == 0;
}
// the whole archive: returns orig_size, ACEAPEX_ERR_DATA (-2) for a bad archive, ACEAPEX_ERR_BUFFER (-1) when dst is short
static int64_t ax_fast_decompress(const uint8_t* src, size_t ssz, uint8_t* dst, size_t cap, int threads) {
    axf_arc_t a; if (axf_open(src, ssz, &a) != 0) return -2;
    if (a.orig > cap) return -1;
    const int T = ax_decode_budget(threads, a.orig);
    return ax_fast_decode_blocks(a, 0, a.nb, dst, T) ? (int64_t)a.orig : -2;
}
// bytes [off, off + len) of the original: the covering blocks into a scratch buffer, the slice copied out
static int64_t ax_fast_region(const axf_arc_t& a, uint8_t* dst, size_t cap, uint64_t off, uint64_t len, int threads) {
    if (len == 0) return 0;
    if (off > a.orig || len > a.orig - off) return -2;
    if (len > cap) return -1;
    const uint32_t b0 = (uint32_t)(off / a.bs), b1 = (uint32_t)((off + len - 1) / a.bs) + 1;
    const size_t span = (size_t)(b1 - b0) * a.bs;
    uint8_t* tmp = (uint8_t*)malloc(span); if (!tmp) return -3;
    const bool ok = ax_fast_decode_blocks(a, b0, b1, tmp, threads);
    if (ok) memcpy(dst, tmp + (off - (uint64_t)b0 * a.bs), (size_t)len);
    free(tmp);
    return ok ? (int64_t)len : -2;
}

// ---- batch of ranges: sorted by offset, split among threads; each thread decodes a block once for its consecutive users
struct AxFastRg { const axf_arc_t* a; aceapex_range_t* rg; const size_t* order; size_t n; int T; std::atomic<int> next; std::atomic<int64_t> okn; };
static void* ax_fast_rg_worker(void* arg) {
    AxFastRg* j = (AxFastRg*)arg; const axf_arc_t& a = *j->a;
    std::vector<uint8_t> blk((size_t)a.bs); int64_t okn = 0;
    for (;;) {
        const int t = j->next.fetch_add(1); if (t >= j->T) break;
        const size_t lo = j->n * (size_t)t / (size_t)j->T, hi = j->n * (size_t)(t + 1) / (size_t)j->T; uint64_t cur = ~0ull;
        for (size_t oi = lo; oi < hi; oi++) {
            aceapex_range_t* r = &j->rg[j->order[oi]];
            if (r->length == 0) { r->written = 0; okn++; continue; }
            if (!r->dst || r->offset > a.orig || r->length > a.orig - r->offset) { r->written = -2; continue; }
            uint64_t pos = r->offset; const uint64_t end = r->offset + r->length; uint8_t* out = (uint8_t*)r->dst; int64_t st = 0;
            while (pos < end) {
                const uint64_t b = pos / a.bs;
                if (b != cur) { cur = ~0ull; if (!axf_dec_block_of(&a, (uint32_t)b, blk.data())) { st = -2; break; } cur = b; }
                const uint64_t bst = b * a.bs; uint64_t take = bst + a.bs - pos; if (take > end - pos) take = end - pos;
                memcpy(out, blk.data() + (pos - bst), (size_t)take); out += take; pos += take;
            }
            r->written = st ? st : (int64_t)r->length; if (!st) okn++;
        }
    }
    j->okn.fetch_add(okn);
    return nullptr;
}
static int64_t ax_fast_ranges(const uint8_t* p, size_t ssz, aceapex_range_t* rg, size_t count, int threads) {
    axf_arc_t a; if (axf_open(p, ssz, &a) != 0) return -2;
    std::vector<size_t> order(count); for (size_t i = 0; i < count; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return rg[x].offset < rg[y].offset; });
    AxFastRg j; j.a = &a; j.rg = rg; j.order = order.data(); j.n = count;
    const int T0 = ax_decode_budget(threads, a.orig);
    j.T = std::max(1, std::min<int>(T0, (int)std::min<size_t>(count, (size_t)a.nb))); j.next = 0; j.okn = 0;
    ax_run_pool(j.T, ax_fast_rg_worker, &j);
    return j.okn.load();
}

// ---- streaming decode: header + table through rd, then groups of T blocks: read, decode on the pool, write in order.
// Memory O(T x (block + its compressed bytes)); ACEAPEX_STREAM_VERIFY hashes the handed-out bytes.
static int64_t ax_fast_stream(aceapex_read_fn rd, void* rctx, aceapex_write_fn wr, void* wctx, int threads, unsigned flags, const AetHeader& hdr) {
    if (hdr.num_blocks == 0) return -2;
    const uint64_t need = 68 + 8ull * ((uint64_t)hdr.num_blocks + 1);
    std::vector<uint8_t> hd((size_t)need);
    for (uint64_t o = 0; o < need; ) { const int64_t r = rd(rctx, o, hd.data() + o, (size_t)(need - o)); if (r <= 0) return -2; o += (uint64_t)r; }
    uint64_t payload; memcpy(&payload, hd.data() + 44, 8);
    if (payload > ((uint64_t)1 << 62)) return -2;
    axf_arc_t a; if (axf_open(hd.data(), (size_t)(need + payload), &a) != 0) return -2;   // validates header + table; data is not touched
    const int T = std::max(1, ax_decode_budget(threads, a.orig));
    XXH3_state_t* xs = nullptr; if (flags & ACEAPEX_STREAM_VERIFY) { xs = XXH3_createState(); XXH3_64bits_reset(xs); }
    std::vector<uint8_t> zb, ob((size_t)T * a.bs); int64_t rc = 0;
    for (uint32_t b0 = 0; b0 < a.nb && rc == 0; ) {
        const uint32_t b1 = std::min<uint32_t>(a.nb, b0 + (uint32_t)T);
        const uint64_t z0 = axf_boff(&a, b0), z1 = axf_boff(&a, b1); zb.resize((size_t)(z1 - z0));
        for (uint64_t o = 0; o < z1 - z0; ) { const int64_t r = rd(rctx, need + z0 + o, zb.data() + o, (size_t)(z1 - z0 - o)); if (r <= 0) { rc = -2; break; } o += (uint64_t)r; }
        if (rc) break;
        axf_arc_t g = a; g.data = zb.data() - z0;   // block b of the group at zb + (boff[b] - z0)
        if (!ax_fast_decode_blocks(g, b0, b1, ob.data(), T)) { rc = -2; break; }
        const size_t n = (size_t)(std::min<uint64_t>((uint64_t)b1 * a.bs, a.orig) - (uint64_t)b0 * a.bs);
        if (xs) XXH3_64bits_update(xs, ob.data(), n);
        if (wr(wctx, ob.data(), n) != 0) { rc = -1; break; }
        b0 = b1;
    }
    if (xs) { const uint64_t h = XXH3_64bits_digest(xs); XXH3_freeState(xs); uint64_t want; memcpy(&want, hdr.xxhash, 8); if (rc == 0 && h != want) rc = -2; }
    return rc ? rc : (int64_t)a.orig;
}
