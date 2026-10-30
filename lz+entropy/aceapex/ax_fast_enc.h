// ax_fast_enc.h - encoder of the FAST profile (ACEPX2 header version 3; format and decoder in ax_fast.h,
// specification in docs/FORMAT_FAST.md). Written from the format, not from any other codec's code (#F1, 25-27.10.2026;
// prior art recorded in research/fastblk/PRIOR_ART_axf_vs_misa77.md). One independent block = one token array, one
// extension stream, one literal run; no entropy stage. The level sets the minimum match length and the parse:
//
//   level  min_match  parse                         (silesia, ace-core, research/fastblk/d1-lzbench-2026-10-26.md)
//     1       16      sparse probe (HW-ENC2)         61.2 %, the fastest decode; ~480 MB/s on one core (ace-core)
//     2       12      greedy + lazy                   54.1 %
//     3        8      greedy + lazy                   45.0 %
//     4        5      greedy + lazy                   40.7 %
//     5       16      optimal parse, depth 512        55.4 %  (encoder ~25x slower than 1-4)
//     6       12      optimal parse                   48.9 %
//     7        8      optimal parse                   41.8 %
//     8        5      optimal parse                   39.2 %
//
// Level 1 uses enc_sparse (below): probes every 3 bytes, table positions every 2nd (coprime strides), backward extension;
// results/hwenc2-2026-10-28-8201fb2.log: silesia 61.21 % at 482 MB/s against the greedy + lazy parse's 61.35 % at 25.8 MB/s,
// decode 11138 against 11011 MB/s. Levels 2-4 keep the greedy + lazy parse.
// A larger minimum match means fewer tokens per MiB and a faster decode at a lower ratio. The optimal parse prices a
// parse as literal bytes + 4 bytes per token + LAMBDA per token (the decode cost of a token in bytes) and so trades
// size for fewer tokens; it is a different point on the same axis, not a better encoder of the same stream.
#pragma once
#include "ax_fast.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <algorithm>

namespace axf {

enum : uint32_t { LL_EXT = 63, ML_EXT = 63, D_MAX = (1u << 20) - 1, WIN = AXF_BLOCK_MAX };
static const double LAMBDA = 2.0;        // optimal parse: decode cost of a token, in bytes
static const int    DEPTH_GREEDY = 64, DEPTH_OPT = 512;

struct Params { uint32_t mm; int depth; bool lazy, optimal; };
// fast-profile level 1..8 -> parameters; 0 = unknown level
static inline bool level_params(int level, Params& p) {
    static const uint32_t MM[8] = {16, 12, 8, 5, 16, 12, 8, 5};
    if (level < 1 || level > 8) return false;
    p.mm = MM[level - 1]; p.optimal = level >= 5; p.lazy = true; p.depth = p.optimal ? DEPTH_OPT : DEPTH_GREEDY;
    return true;
}
// bytes a block of n input bytes can take in the worst case. Every token carries a match of at least mm >= 4 bytes, so
// 4 T <= M (the matched bytes); the extensions are at most L/255 + L/63 (literal runs of 63 or more) + M/255 + M/65
// (escaped matches of 65 or more); with L + M = n the block is below 76 + 1.02 n. n/32 covers it for every mm.
static inline size_t enc_bound(size_t n, uint32_t mm) {
    (void)mm;
    return AXF_BHDR + AXF_PAD + n + n / 32 + 16;
}

struct Enc { std::vector<uint32_t> head, chain; std::vector<uint32_t> bl, bd; std::vector<double> price; std::vector<uint32_t> from, mlen, mdist;
             std::vector<uint32_t> stab, stok; std::vector<uint8_t> slit, sext; std::vector<uint8_t> out; };

static inline uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void wr32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
static inline void cpyn(uint8_t* d, const void* s, size_t k) { if (k) memcpy(d, s, k); }
static inline uint32_t ctz64(uint64_t x) {
#if defined(__GNUC__)
    return (uint32_t)__builtin_ctzll(x);
#else
    uint32_t n = 0; while (!(x & 1)) { x >>= 1; n++; } return n;
#endif
}
static inline uint32_t clz64(uint64_t x) {
#if defined(__GNUC__)
    return (uint32_t)__builtin_clzll(x);
#else
    uint32_t n = 0; while (!(x >> 63)) { x <<= 1; n++; } return n;
#endif
}
// length of the match between in+a and in+b, at least 4 (the hash matched), at most lim
static inline uint32_t match_len(const uint8_t* in, uint32_t a, uint32_t b, uint32_t lim) {
    uint32_t l = 4;
    while (l + 8 <= lim) { const uint64_t x = rd64(in + a + l) ^ rd64(in + b + l); if (x) { l += ctz64(x) >> 3; break; } l += 8; }
    return l > lim ? lim : l;
}

struct Out {
    std::vector<uint32_t> tok; std::vector<uint8_t> ext, lit;
    void leb(uint32_t r) { while (r >= 255) { ext.push_back(255); r -= 255; } ext.push_back((uint8_t)r); }
    // one token: ll literals then a match of ml bytes at distance d (>= AXF_MIN_DIST); matches above AXF_COPY use the escape
    void put(uint32_t ll, uint32_t ml, uint32_t d, uint32_t mm) {
        uint32_t llf = ll; if (ll >= LL_EXT) { llf = LL_EXT; leb(ll - LL_EXT); }
        uint32_t mlf; if (ml <= AXF_COPY) mlf = ml - mm; else { mlf = ML_EXT; leb(ml - AXF_COPY - 1); }
        tok.push_back(llf | mlf << 6 | (d - AXF_MIN_DIST) << 12);
    }
    size_t write(uint8_t* out, uint32_t tail) const {
        uint8_t* p = out; wr32(p, (uint32_t)tok.size()); wr32(p + 4, (uint32_t)ext.size()); wr32(p + 8, tail); p += AXF_BHDR;
        cpyn(p, tok.data(), tok.size() * 4); p += tok.size() * 4; cpyn(p, ext.data(), ext.size()); p += ext.size();
        cpyn(p, lit.data(), lit.size()); p += lit.size();
        memset(p, 0, AXF_PAD); p += AXF_PAD;
        return (size_t)(p - out);
    }
};

// greedy / lazy parse: hash head + chain, the longest match among `depth` candidates, a longer match one byte later wins
static size_t enc_greedy(const uint8_t* in, uint32_t n, uint8_t* out, const Params& P, Enc& e, Out& o) {
    const uint32_t HB = 15, HN = 1u << HB, mm = P.mm; e.head.assign(HN, 0xffffffffu); e.chain.assign(n + 1, 0xffffffffu);
    auto H = [&](uint32_t p) { return (rd32(in + p) * 0x9E3779B1u) >> (32 - HB); };
    auto longest = [&](uint32_t pos, uint32_t& bd) {
        uint32_t cand = e.head[H(pos)], best = 0; int dep = P.depth; const uint32_t lim = n - pos - 8;
        while (cand != 0xffffffffu && dep-- > 0) {
            const uint32_t d = pos - cand;
            if (d > D_MAX + AXF_MIN_DIST || d > WIN) break;
            if (d >= AXF_MIN_DIST && rd32(in + cand) == rd32(in + pos)) {
                const uint32_t l = match_len(in, pos, cand, lim);
                if (l > best) { best = l; bd = d; if (l >= 512) break; }
            }
            cand = e.chain[cand];
        }
        return best;
    };
    uint32_t pos = 0, anchor = 0;
    while (pos + 12 < n) {
        const uint32_t h = H(pos); uint32_t bd = 0, best = longest(pos, bd);
        if (best >= mm && P.lazy && pos + 13 < n) {
            uint32_t b2d = 0; const uint32_t b2 = longest(pos + 1, b2d);
            if (b2 > best + 1) { e.chain[pos] = e.head[h]; e.head[h] = pos; pos++; continue; }
        }
        if (best >= mm) {
            o.lit.insert(o.lit.end(), in + anchor, in + pos);
            // a match longer than AXF_COPY is one escaped token (the decoder copies it in 64-byte steps)
            o.put(pos - anchor, best, bd, mm);
            for (uint32_t p = pos; p < pos + best && p + 12 < n; p++) { const uint32_t hh = H(p); e.chain[p] = e.head[hh]; e.head[hh] = p; }
            pos += best; anchor = pos;
        } else { e.chain[pos] = e.head[h]; e.head[h] = pos; pos += 1 + ((pos - anchor) >> 6); }   // skip faster in incompressible runs
    }
    o.lit.insert(o.lit.end(), in + anchor, in + n);
    return o.write(out, n - anchor);
}


// sparse parse (level 1, min match 16; HW-ENC2, research in research/fastblk/enc2.cpp): an 8-byte hash key, one slot per
// hash. Positions == 0 mod SB enter the table, probes land every SA bytes (SA, SB coprime), so a match spanning enough probes
// is found at some probe; it is then extended backward to its true start. A candidate closer than AXF_MIN_DIST is lifted to
// the first legal multiple of its distance (same bytes on a run of that period) and verified. Matches of 8..15 bytes at the
// probe that reach the minimum after the backward extension are taken. A match of 65..128 bytes goes out as tokens of at
// most 64 bytes (same distance), which the decoder runs on its branch-free path. Inside a match every 8th table position
// and the one 2 bytes before its end are inserted. Misses step SA + (bytes since the last match >> 8). Output in place: tokens,
// extensions and literals into per-worker arrays (literals by 32-byte copies while 32 bytes of input remain).
static size_t enc_sparse(const uint8_t* in, uint32_t n, uint8_t* out, const Params& P, Enc& e) {
    enum : uint32_t { HB = 17, SA = 3, SB = 2, INS = 8, SKIP = 8, SPLIT = 128 };   // SPLIT: DEC2/DEC4, results/zen5-2026-10-28.log
    const uint32_t mm = P.mm;
    e.stab.assign((1u << HB) + 1, 0u);   // + 1: a slot that takes the inserts with nowhere to go
    if (e.stok.size() < n / 8 + 16) e.stok.resize(n / 8 + 16);
    if (e.slit.size() < (size_t)n + 64) e.slit.resize((size_t)n + 64); if (e.sext.size() < n / 16 + 64) e.sext.resize(n / 16 + 64);
    uint32_t* T = e.stab.data(); uint32_t* TK = e.stok.data(); uint8_t* E = e.sext.data(); uint8_t* L = e.slit.data();
    uint32_t nt = 0, ne = 0; size_t nl = 0;
    auto H = [](const uint8_t* p) { return (uint32_t)((rd64(p) * 0x9E3779B185EBCA87ull) >> (64 - HB)); };
    auto leb = [&](uint32_t r) { while (r >= 255) { E[ne++] = 255; r -= 255; } E[ne++] = (uint8_t)r; };
    uint32_t pos = 0, anchor = 0, ip = 0;
    if (n >= 80) {
        const uint32_t last = n - 16;
        while (pos < last) {
            {   // table positions behind the probe: 0, 1 or 2 per step of SA = 3, written without a branch (a position at or
                // past the probe goes to the spare slot DUMMY); a longer gap (skip in a literal run) takes the loop
                const uint32_t a = ip, b = ip + SB, DUMMY = 1u << HB;
                T[a < pos ? H(in + a) : DUMMY] = a; T[b < pos ? H(in + b) : DUMMY] = b;
                ip += a < pos ? (b < pos ? 2 * SB : SB) : 0;
                for (; ip < pos; ip += SB) T[H(in + ip)] = ip;
            }
            // every slot holds a position (0 until written): no empty test, the key compare decides
            uint32_t c = T[H(in + pos)];
            uint32_t len = 0, d = 0;
            {
                const uint64_t k = rd64(in + pos);
                if (((rd64(in + c) ^ k) | ((rd64(in + c + 8) ^ rd64(in + pos + 8)) & 0xffffull)) == 0 && c < pos) {
                    d = pos - c;
                    if (d < AXF_MIN_DIST) { d *= (AXF_MIN_DIST + d - 1) / d; if (d <= pos && rd64(in + pos - d) == k) c = pos - d; else d = 0; }
                    if (d) {
                        const uint32_t lim = n - pos; uint32_t l = 8;
                        for (;;) { if (l + 8 > lim) { while (l < lim && in[c + l] == in[pos + l]) l++; break; }
                                   const uint64_t x = rd64(in + c + l) ^ rd64(in + pos + l); if (x) { l += ctz64(x) >> 3; break; } l += 8; }
                        len = l;
                    }
                }
            }
            if (!len) { pos += SA + ((pos - anchor) >> SKIP); continue; }
            const uint32_t probe = pos;
            {   // backward extension inside [anchor, pos), 8 bytes per step
                uint32_t room = pos - anchor; if (room > pos - d) room = pos - d;
                while (room >= 8) { const uint64_t x = rd64(in + pos - 8) ^ rd64(in + pos - 8 - d);
                    if (x) { const uint32_t k = clz64(x) >> 3; pos -= k; len += k; room = 0; break; } pos -= 8; len += 8; room -= 8; }
                if (room && pos >= d + 8) {   // the last < 8 bytes in one compare, clamped to the room
                    const uint64_t x = rd64(in + pos - 8) ^ rd64(in + pos - 8 - d); uint32_t k = x ? clz64(x) >> 3 : 8; if (k > room) k = room;
                    pos -= k; len += k;
                } else while (room && in[pos - 1] == in[pos - 1 - d]) { pos--; len++; room--; }
            }
            if (len < mm) { pos = probe + SA + ((probe - anchor) >> SKIP); continue; }
            const uint32_t ll = pos - anchor;
            if (pos + 32 <= n) { for (uint32_t i = 0; i < ll; i += 32) memcpy(L + nl + i, in + anchor + i, 32); } else cpyn(L + nl, in + anchor, ll);
            nl += ll;
            uint32_t llf = ll; if (ll >= LL_EXT) { llf = LL_EXT; leb(ll - LL_EXT); }
            {   // a match up to SPLIT bytes goes out as pieces of <= AXF_COPY (each >= mm, same distance, no literals): the decoder
                // runs them on its branch-free path instead of the escape; longer matches keep one escaped token
                uint32_t rest = len, first = 1;
                if (rest > AXF_COPY && rest <= SPLIT)
                    while (rest > AXF_COPY) { const uint32_t piece = rest - AXF_COPY < mm ? rest - mm : AXF_COPY;
                        TK[nt++] = (first ? llf : 0u) | (piece - mm) << 6 | (d - AXF_MIN_DIST) << 12; first = 0; rest -= piece; }
                uint32_t mlf; if (rest <= AXF_COPY) mlf = rest - mm; else { mlf = ML_EXT; leb(rest - AXF_COPY - 1); }
                TK[nt++] = (first ? llf : 0u) | mlf << 6 | (d - AXF_MIN_DIST) << 12;
            }
            const uint32_t end = pos + len;
            if (ip < end) { for (; ip + 8 < end && ip < last; ip += SB * INS) T[H(in + ip)] = ip; ip = end + ((SB - end % SB) % SB); }
            if (end - 2 < last) T[H(in + end - 2)] = end - 2;
            pos = end; anchor = end;
        }
    }
    cpyn(L + nl, in + anchor, n - anchor); nl += n - anchor;
    uint8_t* p = out; wr32(p, nt); wr32(p + 4, ne); wr32(p + 8, n - anchor); p += AXF_BHDR;
    cpyn(p, TK, (size_t)nt * 4); p += (size_t)nt * 4; cpyn(p, E, ne); p += ne; cpyn(p, L, nl); p += nl;
    memset(p, 0, AXF_PAD); p += AXF_PAD;
    return (size_t)(p - out);
}

// optimal parse: the longest match at every position (depth DEPTH_OPT), then a forward DP over the block
static size_t enc_optimal(const uint8_t* in, uint32_t n, uint8_t* out, const Params& P, Enc& e, Out& o) {
    const uint32_t HB = 16, HN = 1u << HB, mm = P.mm; e.head.assign(HN, 0xffffffffu); e.chain.assign(n + 1, 0xffffffffu);
    auto H = [&](uint32_t p) { return (rd32(in + p) * 0x9E3779B1u) >> (32 - HB); };
    e.bl.assign(n + 1, 0); e.bd.assign(n + 1, 0);
    const uint32_t lastp = n > 12 ? n - 12 : 0, LONG = 4096;
    for (uint32_t pos = 0; pos < lastp; pos++) {
        const uint32_t h = H(pos); uint32_t cand = e.head[h], best = 0, bdist = 0; int dep = P.depth; const uint32_t lim = n - pos - 8;
        while (cand != 0xffffffffu && dep-- > 0) {
            const uint32_t d = pos - cand; if (d > D_MAX + AXF_MIN_DIST || d > WIN) break;
            if (d >= AXF_MIN_DIST && rd32(in + cand) == rd32(in + pos)) {
                const uint32_t l = match_len(in, pos, cand, lim);
                if (l > best) { best = l; bdist = d; if (l >= LONG) break; }
            }
            cand = e.chain[cand];
        }
        e.chain[pos] = e.head[h]; e.head[h] = pos;
        if (best >= mm) { e.bl[pos] = best; e.bd[pos] = bdist; }
        if (best >= LONG) {   // a long run: the positions inside it match the same source with the remaining length (no search, linear time)
            const uint32_t skip = best - LONG;
            for (uint32_t k = 1; k <= skip && pos + k < lastp; k++) { e.bl[pos + k] = best - k; e.bd[pos + k] = bdist; }
            pos += skip;
        }
    }
    const double TOKB = 4.0;
    e.price.assign(n + 1, 1e300); e.from.assign(n + 1, 0); e.mlen.assign(n + 1, 0); e.mdist.assign(n + 1, 0);
    e.price[0] = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double pi = e.price[i];
        if (pi + 1.0 < e.price[i + 1]) { e.price[i + 1] = pi + 1.0; e.from[i + 1] = i; e.mlen[i + 1] = 0; }
        const uint32_t L = e.bl[i]; if (!L) continue;
        const double c = pi + TOKB + LAMBDA;
        const uint32_t top = L < AXF_COPY ? L : AXF_COPY;
        for (uint32_t l = mm; l <= top; l++) { const uint32_t j = i + l; if (c < e.price[j]) { e.price[j] = c; e.from[j] = i; e.mlen[j] = l; e.mdist[j] = e.bd[i]; } }
        if (L > AXF_COPY) { const uint32_t j = i + L; if (c < e.price[j]) { e.price[j] = c; e.from[j] = i; e.mlen[j] = L; e.mdist[j] = e.bd[i]; } }
    }
    std::vector<uint32_t> ms, ml, md;
    for (uint32_t j = n; j > 0; ) { const uint32_t i = e.from[j]; if (e.mlen[j]) { ms.push_back(i); ml.push_back(e.mlen[j]); md.push_back(e.mdist[j]); } j = i; }
    o.tok.reserve(ms.size());
    uint32_t anchor = 0;
    for (size_t k = ms.size(); k-- > 0; ) {
        const uint32_t st = ms[k], l = ml[k], d = md[k];
        o.lit.insert(o.lit.end(), in + anchor, in + st);
        o.put(st - anchor, l, d, mm); anchor = st + l;
    }
    o.lit.insert(o.lit.end(), in + anchor, in + n);
    return o.write(out, n - anchor);
}

// one block -> out (at least enc_bound(n, P.mm) bytes); returns the block's size, pad included
static size_t enc_block(const uint8_t* in, uint32_t n, uint8_t* out, const Params& P, Enc& e) {
    if (!P.optimal && P.mm == 16) return enc_sparse(in, n, out, P, e);
    Out o; o.lit.reserve(n); o.tok.reserve(n / 8 + 1);
    return P.optimal ? enc_optimal(in, n, out, P, e, o) : enc_greedy(in, n, out, P, e, o);
}

// ---- whole archive --------------------------------------------------------------------------------------------
// Blocks are encoded independently on a pool; the archive is header + table + blocks. The block size is the profile's
// 1 MiB (the distance field covers it); smaller inputs get one block.
struct Job { const uint8_t* src; size_t n; uint32_t bs, nb; Params P; std::vector<std::vector<uint8_t>>* blocks; std::atomic<uint32_t> next; std::atomic<int> err; };
static void* enc_worker(void* arg) {
    Job* j = (Job*)arg; Enc e;
    for (;;) {
        const uint32_t b = j->next.fetch_add(1); if (b >= j->nb) break;
        const size_t off = (size_t)b * j->bs; const uint32_t k = (uint32_t)std::min<size_t>(j->bs, j->n - off);
        // encode into the worker's scratch (allocated once, never zeroed), keep only the block's bytes
        const size_t cap = enc_bound(k, j->P.mm); if (e.out.size() < cap) e.out.resize(cap);
        const size_t c = enc_block(j->src + off, k, e.out.data(), j->P, e);
        (*j->blocks)[b].assign(e.out.data(), e.out.data() + c);
    }
    return nullptr;
}
// header + table in front of `blocks`; total size
static inline size_t archive_size(const std::vector<std::vector<uint8_t>>& blocks) {
    size_t t = 68 + 8 * (blocks.size() + 1); for (const auto& b : blocks) t += b.size(); return t;
}
static inline void write_header(uint8_t* p, uint64_t orig, uint32_t bs, uint32_t nb, uint64_t xxh, uint32_t mm, uint64_t payload) {
    memset(p, 0, 68); memcpy(p, "ACEPX2\0\0", 8);
    const uint32_t v = AXF_VERSION, pr = AXF_PROFILE_FAST;
    memcpy(p + 8, &v, 4); memcpy(p + 12, &orig, 8); memcpy(p + 20, &bs, 4); memcpy(p + 24, &nb, 4); memcpy(p + 28, &xxh, 8);
    memcpy(p + 36, &pr, 4); memcpy(p + 40, &mm, 4); memcpy(p + 44, &payload, 8);
}
// table + blocks after the header: p points at byte 68 of the archive
static inline void write_body(uint8_t* p, const std::vector<std::vector<uint8_t>>& blocks) {
    uint64_t o = 0;
    for (size_t b = 0; b <= blocks.size(); b++) { memcpy(p + 8 * b, &o, 8); if (b < blocks.size()) o += blocks[b].size(); }
    p += 8 * (blocks.size() + 1);
    for (const auto& b : blocks) { memcpy(p, b.data(), b.size()); p += b.size(); }
}
} // namespace axf
