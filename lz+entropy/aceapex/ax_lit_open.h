/* ax_lit_open.h - zstd-free literal chunks (the "open" literal profile, ADR-019, spec 3.4).
 *
 * Two modes of a tagged literal chunk (spec 3.2.3), next to mode 0 (zstd) and 1 (DNA pack
 * in zstd frames):
 *   mode 2  open DNA pack: the DNA pack of spec 3.3 with the case mask as alternating run
 *           lengths and the exception gaps as LEB128, every part a "piece"
 *   mode 3  open plain: one piece holding the chunk's bytes
 * A piece is one mode byte and a payload: 0 = raw bytes, 1 = one rANS chunk of spec 3.1.1
 * (src/ax_rans.h). Its decoded length is always known from the context.
 *
 * Mode 2 layout (after the chunk's mode byte), all u32 little-endian:
 *   [0]  nexc   exception bytes (not ACGT/acgt)
 *   [4]  ncse   bytes of the case-run stream
 *   [8]  ngap   bytes of the gap stream
 *   [12] h1..h4 stored sizes of the pieces seq, cse, gap, val (h3 = h4 = 0 iff nexc = 0)
 *   [28] seq piece -> ceil(raw/4) bytes, 2 bits per base, MSB first, A=0 C=1 G=2 T=3
 *        cse piece -> ncse bytes: LEB128 run lengths, alternating upper, lower, upper ...
 *                     starting with upper; only the first run may be 0; the runs sum to raw
 *        gap piece -> ngap bytes: nexc LEB128 gaps, position = running sum (from 0); only the
 *                     first gap may be 0; every position < raw
 *        val piece -> nexc bytes: the exception byte itself
 * Reconstruction as in spec 3.3: bases, then case (|= 0x20 on lower runs), then exceptions.
 * A reader MUST reject: a size mismatch anywhere, an unknown piece mode, a LEB128 value over
 * 32 bits or not terminated inside its stream, unread bytes, runs not summing to raw, a zero
 * run or gap where forbidden, an exception position >= raw, a wrong count of gaps.
 * Plain C99, no allocation in the piece decoder; the pack decoder uses one scratch buffer.
 */
#ifndef AX_LIT_OPEN_H
#define AX_LIT_OPEN_H
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "ax_rans.h"

#define AXO_HDR 28u

static inline uint32_t axo_rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* one piece of `sz` stored bytes -> exactly n bytes; 0 ok, -1 malformed */
static inline int axo_piece_decode(const uint8_t* p, size_t sz, uint8_t* out, size_t n) {
    if (n == 0) return sz == 0 ? 0 : -1;
    if (sz < 1) return -1;
    if (p[0] == 0) { if (sz - 1 != n) return -1; memcpy(out, p + 1, n); return 0; }
    if (p[0] == 1) return axr_decode(p + 1, sz - 1, out, n);
    return -1;
}

/* LEB128 u32 from [*i, n); -1 on overflow or unterminated */
static inline int axo_leb(const uint8_t* b, size_t n, size_t* i, uint32_t* v) {
    uint64_t x = 0; int sh = 0;
    for (;;) {
        if (*i >= n || sh > 28) return -1;
        uint8_t c = b[(*i)++]; x |= (uint64_t)(c & 0x7F) << sh;
        if (!(c & 0x80)) break;
        sh += 7;
    }
    if (x > 0xFFFFFFFFull) return -1;
    *v = (uint32_t)x; return 0;
}

/* 2-bit bases -> ASCII by table: one packed byte gives four bases (first base in the top bits),
 * stored as one 4-byte word; the tail byte by byte. Same bytes as the per-base loop. */
#define AXO_B(c) ((uint32_t)((c) == 0 ? 'A' : (c) == 1 ? 'C' : (c) == 2 ? 'G' : 'T'))
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define AXO_W(b) (AXO_B((b) >> 6) << 24 | AXO_B(((b) >> 4) & 3) << 16 | AXO_B(((b) >> 2) & 3) << 8 | AXO_B((b) & 3))
#else
#define AXO_W(b) (AXO_B((b) >> 6) | AXO_B(((b) >> 4) & 3) << 8 | AXO_B(((b) >> 2) & 3) << 16 | AXO_B((b) & 3) << 24)
#endif
#define AXO_W4(b) AXO_W(b), AXO_W((b) + 1), AXO_W((b) + 2), AXO_W((b) + 3)
#define AXO_W16(b) AXO_W4(b), AXO_W4((b) + 4), AXO_W4((b) + 8), AXO_W4((b) + 12)
#define AXO_W64(b) AXO_W16(b), AXO_W16((b) + 16), AXO_W16((b) + 32), AXO_W16((b) + 48)
static const uint32_t axo_b4[256] = { AXO_W64(0), AXO_W64(64), AXO_W64(128), AXO_W64(192) };
static inline void axo_unpack_bases(const uint8_t* seq, uint8_t* dst, size_t raw) {
    size_t full = raw >> 2, i;
    for (i = 0; i < full; i++) memcpy(dst + 4 * i, &axo_b4[seq[i]], 4);
    for (i = 4 * full; i < raw; i++) dst[i] = (uint8_t)"ACGT"[(seq[i >> 2] >> (6 - 2 * (i & 3))) & 3];
}

/* mode 2 payload (without the chunk mode byte) -> raw bytes in dst; 0 ok, -1 malformed */
static inline int axo_dna_decode(const uint8_t* s, size_t sz, uint8_t* dst, size_t raw) {
    if (sz < AXO_HDR || raw == 0) return -1;
    uint32_t nexc = axo_rd32(s), ncse = axo_rd32(s + 4), ngap = axo_rd32(s + 8);
    uint32_t h[4] = {axo_rd32(s + 12), axo_rd32(s + 16), axo_rd32(s + 20), axo_rd32(s + 24)};
    if ((uint64_t)AXO_HDR + h[0] + h[1] + h[2] + h[3] != sz) return -1;
    if (nexc == 0 ? (ngap || h[2] || h[3]) : (ngap < nexc || !h[2] || !h[3])) return -1;
    if (ncse == 0 || nexc > raw) return -1;
    const size_t np = (raw + 3) / 4;
    uint8_t* buf = (uint8_t*)malloc(np + (size_t)ncse + ngap + nexc + 1);
    if (!buf) return -1;
    uint8_t *seq = buf, *cse = seq + np, *gap = cse + ncse, *val = gap + ngap;
    const uint8_t* p = s + AXO_HDR; int ok = 1;
    ok = ok && axo_piece_decode(p, h[0], seq, np) == 0; p += h[0];
    ok = ok && axo_piece_decode(p, h[1], cse, ncse) == 0; p += h[1];
    ok = ok && axo_piece_decode(p, h[2], gap, ngap) == 0; p += h[2];
    ok = ok && axo_piece_decode(p, h[3], val, nexc) == 0;
    if (ok) {
        axo_unpack_bases(seq, dst, raw);
        size_t i = 0; uint64_t pos = 0; uint32_t r; int lower = 0, first = 1;
        while (ok && i < ncse) {
            if (axo_leb(cse, ncse, &i, &r) || (r == 0 && !first) || pos + r > raw) { ok = 0; break; }
            if (lower) for (uint64_t k = pos; k < pos + r; k++) dst[k] |= 0x20;
            pos += r; lower = !lower; first = 0;
        }
        if (pos != raw) ok = 0;
        i = 0; pos = 0;
        for (uint32_t k = 0; ok && k < nexc; k++) {
            if (axo_leb(gap, ngap, &i, &r) || (r == 0 && k > 0)) { ok = 0; break; }
            pos += r;
            if (pos >= raw) { ok = 0; break; }
            dst[pos] = val[k];
        }
        if (i != ngap) ok = 0;
    }
    free(buf);
    return ok ? 0 : -1;
}

/* ---- encoder (used by the C++ writer; kept here so every copy of the format is one file) */

/* piece: rANS if smaller than raw, else raw. out needs axo_piece_bound(n); scratch n+64 u16 */
static inline size_t axo_piece_bound(size_t n) { return 1 + (axr_bound(n) > n ? axr_bound(n) : n); }
static inline size_t axo_piece_encode(const uint8_t* in, size_t n, uint8_t* out, uint16_t* scratch) {
    if (n == 0) return 0;
    size_t r = axr_encode(in, n, out + 1, scratch);
    if (r && r < n) { out[0] = 1; return 1 + r; }
    out[0] = 0; memcpy(out + 1, in, n); return 1 + n;
}
static inline size_t axo_put_leb(uint8_t* p, uint32_t v) {
    size_t i = 0; do { uint8_t b = v & 0x7F; v >>= 7; p[i++] = b | (v ? 0x80 : 0); } while (v); return i;
}
/* bound of a mode 2 payload for n input bytes */
static inline size_t axo_dna_bound(size_t n) {
    return AXO_HDR + axo_piece_bound((n + 3) / 4) + axo_piece_bound(5 * n + 5) + axo_piece_bound(5 * n) + axo_piece_bound(n);
}
/* mode 2 payload of n bytes (n >= 1) into out (axo_dna_bound(n)); returns the size, 0 on failure */
static inline size_t axo_dna_encode(const uint8_t* s, size_t n, uint8_t* out) {
    if (n == 0) return 0;
    size_t np = (n + 3) / 4, nexc = 0;
    for (size_t i = 0; i < n; i++) { uint8_t u = s[i] & 0xDF; if (u != 'A' && u != 'C' && u != 'G' && u != 'T') nexc++; }
    uint8_t* seq = (uint8_t*)calloc(np, 1);
    uint8_t* cse = (uint8_t*)malloc(5 * n + 5);
    uint8_t* gap = (uint8_t*)malloc(5 * nexc + 1);
    uint8_t* val = (uint8_t*)malloc(nexc + 1);
    uint16_t* scr = (uint16_t*)malloc((5 * n + 5 + 2 * AXR_LANES) * sizeof(uint16_t));
    size_t ncse = 0, ngap = 0, e = 0, prev = 0, total = 0;
    if (!seq || !cse || !gap || !val || !scr) goto out;
    {
        uint32_t run = 0; int lower = 0;
        for (size_t i = 0; i < n; i++) {
            uint8_t b = s[i], u = b & 0xDF, code = 0;
            if (u == 'C') code = 1; else if (u == 'G') code = 2; else if (u == 'T') code = 3;
            else if (u != 'A') { ngap += axo_put_leb(gap + ngap, (uint32_t)(i - prev)); prev = i; val[e++] = b; }
            seq[i >> 2] |= (uint8_t)(code << (6 - 2 * (i & 3)));
            int l = b >= 0x61 && b <= 0x7a;                     /* same case rule as mode 1 */
            if (l != lower) { ncse += axo_put_leb(cse + ncse, run); run = 0; lower = l; }
            run++;
        }
        ncse += axo_put_leb(cse + ncse, run);
    }
    {
        uint8_t* p = out + AXO_HDR; uint32_t h[4];
        h[0] = (uint32_t)axo_piece_encode(seq, np, p, scr); p += h[0];
        h[1] = (uint32_t)axo_piece_encode(cse, ncse, p, scr); p += h[1];
        h[2] = (uint32_t)axo_piece_encode(gap, ngap, p, scr); p += h[2];
        h[3] = (uint32_t)axo_piece_encode(val, nexc, p, scr); p += h[3];
        uint32_t w[7] = {(uint32_t)nexc, (uint32_t)ncse, (uint32_t)ngap, h[0], h[1], h[2], h[3]};
        memcpy(out, w, AXO_HDR);
        total = (size_t)(p - out);
    }
out:
    free(seq); free(cse); free(gap); free(val); free(scr);
    return total;
}
#endif
