/* ax_rans.h - 32-lane interleaved static rANS for one chunk (the "rANS" token profile).
 *
 * Designed so that one GPU warp decodes one chunk: symbol i belongs to lane i % 32, the
 * 32 lanes decode a group of 32 symbols in lock step, and the 16-bit renormalisation
 * words of a group are stored in lane order, so a lane finds its word by counting the
 * lanes below it that also renormalise (a ballot + popcount on a GPU). No zstd, no
 * nvCOMP: an open decoder on any device.
 *
 * Chunk layout (all little-endian):
 *   [32]      bitmap of the symbols present (bit s of byte s/8)
 *   [..]      frequency of each present symbol, ascending s, LEB128; they sum to 4096
 *   [128]     32 initial decoder states, u32, each in [2^16, 2^32)
 *   [4]       W = number of 16-bit words that follow
 *   [2W]      renormalisation words in decode order
 * A decoder MUST check: frequencies sum to 4096, every state in range, no read past W,
 * all W words consumed, and every lane back at state 2^16 at the end (the encoder's
 * start state) - a cheap integrity check of the whole chunk.
 */
#ifndef AX_RANS_H
#define AX_RANS_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define AXR_LANES 32
#define AXR_PBITS 12
#define AXR_M     (1u << AXR_PBITS)
#define AXR_L     (1u << 16)

/* Normalise a histogram to AXR_M keeping every present symbol >= 1. */
static inline void axr_normalize(const uint32_t cnt[256], size_t n, uint32_t f[256]) {
    uint32_t sum = 0; int best = -1; uint32_t bestc = 0;
    for (int s = 0; s < 256; s++) {
        f[s] = 0;
        if (!cnt[s]) continue;
        uint64_t v = ((uint64_t)cnt[s] * AXR_M) / n;
        f[s] = v ? (uint32_t)v : 1;
        sum += f[s];
        if (cnt[s] > bestc) { bestc = cnt[s]; best = s; }
    }
    /* move the rounding error onto the most frequent symbol; if that would push it
       below 1, take from others in turn */
    while (sum != AXR_M) {
        if (sum < AXR_M) { f[best] += AXR_M - sum; sum = AXR_M; break; }
        uint32_t over = sum - AXR_M;
        if (f[best] > over) { f[best] -= over; sum = AXR_M; break; }
        for (int s = 0; s < 256 && sum > AXR_M; s++) if (f[s] > 1) { f[s]--; sum--; }
    }
}

static inline size_t axr_put_leb(uint8_t* p, uint32_t v) {
    size_t i = 0; do { uint8_t b = v & 0x7F; v >>= 7; p[i++] = b | (v ? 0x80 : 0); } while (v); return i;
}

/* Bound of the encoded chunk for n input bytes. */
static inline size_t axr_bound(size_t n) { return 32 + 256 * 2 + 128 + 4 + 2 * (n + 2 * AXR_LANES) + 16; }

/* Encode n bytes (n >= 1). Returns the encoded size, 0 on failure. `words` is scratch of
   at least n + 2*AXR_LANES uint16 entries. */
static inline size_t axr_encode(const uint8_t* in, size_t n, uint8_t* out, uint16_t* words) {
    if (n == 0) return 0;
    uint32_t cnt[256] = {0}, f[256], c[256];
    for (size_t i = 0; i < n; i++) cnt[in[i]]++;
    axr_normalize(cnt, n, f);
    uint32_t acc = 0; for (int s = 0; s < 256; s++) { c[s] = acc; acc += f[s]; }
    uint32_t x[AXR_LANES]; for (int l = 0; l < AXR_LANES; l++) x[l] = AXR_L;
    size_t nw = 0;
    size_t groups = (n + AXR_LANES - 1) / AXR_LANES;
    for (size_t g = groups; g-- > 0;) {
        for (int l = AXR_LANES - 1; l >= 0; l--) {          /* lanes descending: reversed at the end */
            size_t i = g * AXR_LANES + (size_t)l; if (i >= n) continue;
            uint32_t s = in[i], fs = f[s];
            uint64_t xmax = (uint64_t)fs << (32 - AXR_PBITS);  /* ((L >> PBITS) << 16) * fs */
            if ((uint64_t)x[l] >= xmax) { words[nw++] = (uint16_t)(x[l] & 0xFFFF); x[l] >>= 16; }
            x[l] = ((x[l] / fs) << AXR_PBITS) + (x[l] % fs) + c[s];
        }
    }
    uint8_t* p = out;
    memset(p, 0, 32); for (int s = 0; s < 256; s++) if (f[s]) p[s >> 3] |= (uint8_t)(1u << (s & 7)); p += 32;
    for (int s = 0; s < 256; s++) if (f[s]) p += axr_put_leb(p, f[s]);
    for (int l = 0; l < AXR_LANES; l++) { memcpy(p, &x[l], 4); p += 4; }
    uint32_t W = (uint32_t)nw; memcpy(p, &W, 4); p += 4;
    for (size_t k = 0; k < nw; k++) { uint16_t w = words[nw - 1 - k]; memcpy(p, &w, 2); p += 2; }
    return (size_t)(p - out);
}

/* The SIMD decoders build in C and C++ (the C99 decoder c/ and the Python reader include this header too). */
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define AXR_SIMD 1
#include <immintrin.h>
#include "ax_env.h"     /* ax_getenv (AX_RANS_SIMD only with ACEAPEX_ENV_TUNING) */
#include <stdlib.h>
#endif

/* Decode one chunk of exactly n bytes from src[0..sz). Returns 0 on success, -1 on any
   malformed input (fail-closed). */
static inline int axr_decode_scalar(const uint8_t* src, size_t sz, uint8_t* out, size_t n) {
    if (sz < 32 + 128 + 4) return -1;
    const uint8_t* p = src; const uint8_t* e = src + sz;
    uint16_t f[256]; uint16_t c[256]; uint8_t sym[AXR_M];
    uint32_t acc = 0;
    const uint8_t* bm = p; p += 32;
    for (int s = 0; s < 256; s++) {
        f[s] = 0; c[s] = (uint16_t)acc;
        if (!(bm[s >> 3] & (1u << (s & 7)))) continue;
        uint32_t v = 0; int sh = 0;
        for (;;) { if (p >= e || sh > 14) return -1; uint8_t b = *p++; v |= (uint32_t)(b & 0x7F) << sh; if (!(b & 0x80)) break; sh += 7; }
        if (v == 0 || acc + v > AXR_M) return -1;
        f[s] = (uint16_t)v; memset(sym + acc, s, v); acc += v;
    }
    if (acc != AXR_M) return -1;
    if ((size_t)(e - p) < 128 + 4) return -1;
    uint32_t x[AXR_LANES];
    for (int l = 0; l < AXR_LANES; l++) { memcpy(&x[l], p, 4); p += 4; if (x[l] < AXR_L) return -1; }
    uint32_t W; memcpy(&W, p, 4); p += 4;
    if ((size_t)(e - p) != (size_t)W * 2) return -1;
    const uint8_t* wp = p; uint32_t wi = 0;
    size_t groups = (n + AXR_LANES - 1) / AXR_LANES;
    for (size_t g = 0; g < groups; g++) {
        for (int l = 0; l < AXR_LANES; l++) {
            size_t i = g * AXR_LANES + (size_t)l; if (i >= n) break;
            uint32_t slot = x[l] & (AXR_M - 1);
            uint32_t s = sym[slot];
            out[i] = (uint8_t)s;
            x[l] = (uint32_t)f[s] * (x[l] >> AXR_PBITS) + slot - c[s];
            if (x[l] < AXR_L) {
                if (wi >= W) return -1;
                uint16_t w; memcpy(&w, wp + 2 * (size_t)wi, 2); wi++;
                x[l] = (x[l] << 16) | w;
            }
        }
    }
    if (wi != W) return -1;
    for (int l = 0; l < AXR_LANES; l++) if (x[l] != AXR_L) return -1;
    return 0;
}

#ifdef AXR_SIMD
/* lane i -> rank of i among the set bits of an 8-bit mask (the words a vector takes, in lane order); a constant
 * table so that C and C++ share it without a static constructor */
static const uint32_t AXR_PERM[256][8] = {
    {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,1,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,1,0,0,0,0,0},{0,0,1,0,0,0,0,0},{0,1,2,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},{0,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,0},{0,1,0,2,0,0,0,0},{0,0,0,1,0,0,0,0},{0,0,1,2,0,0,0,0},{0,0,1,2,0,0,0,0},{0,1,2,3,0,0,0,0},
    {0,0,0,0,0,0,0,0},{0,0,0,0,1,0,0,0},{0,0,0,0,1,0,0,0},{0,1,0,0,2,0,0,0},{0,0,0,0,1,0,0,0},{0,0,1,0,2,0,0,0},{0,0,1,0,2,0,0,0},{0,1,2,0,3,0,0,0},
    {0,0,0,0,1,0,0,0},{0,0,0,1,2,0,0,0},{0,0,0,1,2,0,0,0},{0,1,0,2,3,0,0,0},{0,0,0,1,2,0,0,0},{0,0,1,2,3,0,0,0},{0,0,1,2,3,0,0,0},{0,1,2,3,4,0,0,0},
    {0,0,0,0,0,0,0,0},{0,0,0,0,0,1,0,0},{0,0,0,0,0,1,0,0},{0,1,0,0,0,2,0,0},{0,0,0,0,0,1,0,0},{0,0,1,0,0,2,0,0},{0,0,1,0,0,2,0,0},{0,1,2,0,0,3,0,0},
    {0,0,0,0,0,1,0,0},{0,0,0,1,0,2,0,0},{0,0,0,1,0,2,0,0},{0,1,0,2,0,3,0,0},{0,0,0,1,0,2,0,0},{0,0,1,2,0,3,0,0},{0,0,1,2,0,3,0,0},{0,1,2,3,0,4,0,0},
    {0,0,0,0,0,1,0,0},{0,0,0,0,1,2,0,0},{0,0,0,0,1,2,0,0},{0,1,0,0,2,3,0,0},{0,0,0,0,1,2,0,0},{0,0,1,0,2,3,0,0},{0,0,1,0,2,3,0,0},{0,1,2,0,3,4,0,0},
    {0,0,0,0,1,2,0,0},{0,0,0,1,2,3,0,0},{0,0,0,1,2,3,0,0},{0,1,0,2,3,4,0,0},{0,0,0,1,2,3,0,0},{0,0,1,2,3,4,0,0},{0,0,1,2,3,4,0,0},{0,1,2,3,4,5,0,0},
    {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,1,0},{0,0,0,0,0,0,1,0},{0,1,0,0,0,0,2,0},{0,0,0,0,0,0,1,0},{0,0,1,0,0,0,2,0},{0,0,1,0,0,0,2,0},{0,1,2,0,0,0,3,0},
    {0,0,0,0,0,0,1,0},{0,0,0,1,0,0,2,0},{0,0,0,1,0,0,2,0},{0,1,0,2,0,0,3,0},{0,0,0,1,0,0,2,0},{0,0,1,2,0,0,3,0},{0,0,1,2,0,0,3,0},{0,1,2,3,0,0,4,0},
    {0,0,0,0,0,0,1,0},{0,0,0,0,1,0,2,0},{0,0,0,0,1,0,2,0},{0,1,0,0,2,0,3,0},{0,0,0,0,1,0,2,0},{0,0,1,0,2,0,3,0},{0,0,1,0,2,0,3,0},{0,1,2,0,3,0,4,0},
    {0,0,0,0,1,0,2,0},{0,0,0,1,2,0,3,0},{0,0,0,1,2,0,3,0},{0,1,0,2,3,0,4,0},{0,0,0,1,2,0,3,0},{0,0,1,2,3,0,4,0},{0,0,1,2,3,0,4,0},{0,1,2,3,4,0,5,0},
    {0,0,0,0,0,0,1,0},{0,0,0,0,0,1,2,0},{0,0,0,0,0,1,2,0},{0,1,0,0,0,2,3,0},{0,0,0,0,0,1,2,0},{0,0,1,0,0,2,3,0},{0,0,1,0,0,2,3,0},{0,1,2,0,0,3,4,0},
    {0,0,0,0,0,1,2,0},{0,0,0,1,0,2,3,0},{0,0,0,1,0,2,3,0},{0,1,0,2,0,3,4,0},{0,0,0,1,0,2,3,0},{0,0,1,2,0,3,4,0},{0,0,1,2,0,3,4,0},{0,1,2,3,0,4,5,0},
    {0,0,0,0,0,1,2,0},{0,0,0,0,1,2,3,0},{0,0,0,0,1,2,3,0},{0,1,0,0,2,3,4,0},{0,0,0,0,1,2,3,0},{0,0,1,0,2,3,4,0},{0,0,1,0,2,3,4,0},{0,1,2,0,3,4,5,0},
    {0,0,0,0,1,2,3,0},{0,0,0,1,2,3,4,0},{0,0,0,1,2,3,4,0},{0,1,0,2,3,4,5,0},{0,0,0,1,2,3,4,0},{0,0,1,2,3,4,5,0},{0,0,1,2,3,4,5,0},{0,1,2,3,4,5,6,0},
    {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,1},{0,0,0,0,0,0,0,1},{0,1,0,0,0,0,0,2},{0,0,0,0,0,0,0,1},{0,0,1,0,0,0,0,2},{0,0,1,0,0,0,0,2},{0,1,2,0,0,0,0,3},
    {0,0,0,0,0,0,0,1},{0,0,0,1,0,0,0,2},{0,0,0,1,0,0,0,2},{0,1,0,2,0,0,0,3},{0,0,0,1,0,0,0,2},{0,0,1,2,0,0,0,3},{0,0,1,2,0,0,0,3},{0,1,2,3,0,0,0,4},
    {0,0,0,0,0,0,0,1},{0,0,0,0,1,0,0,2},{0,0,0,0,1,0,0,2},{0,1,0,0,2,0,0,3},{0,0,0,0,1,0,0,2},{0,0,1,0,2,0,0,3},{0,0,1,0,2,0,0,3},{0,1,2,0,3,0,0,4},
    {0,0,0,0,1,0,0,2},{0,0,0,1,2,0,0,3},{0,0,0,1,2,0,0,3},{0,1,0,2,3,0,0,4},{0,0,0,1,2,0,0,3},{0,0,1,2,3,0,0,4},{0,0,1,2,3,0,0,4},{0,1,2,3,4,0,0,5},
    {0,0,0,0,0,0,0,1},{0,0,0,0,0,1,0,2},{0,0,0,0,0,1,0,2},{0,1,0,0,0,2,0,3},{0,0,0,0,0,1,0,2},{0,0,1,0,0,2,0,3},{0,0,1,0,0,2,0,3},{0,1,2,0,0,3,0,4},
    {0,0,0,0,0,1,0,2},{0,0,0,1,0,2,0,3},{0,0,0,1,0,2,0,3},{0,1,0,2,0,3,0,4},{0,0,0,1,0,2,0,3},{0,0,1,2,0,3,0,4},{0,0,1,2,0,3,0,4},{0,1,2,3,0,4,0,5},
    {0,0,0,0,0,1,0,2},{0,0,0,0,1,2,0,3},{0,0,0,0,1,2,0,3},{0,1,0,0,2,3,0,4},{0,0,0,0,1,2,0,3},{0,0,1,0,2,3,0,4},{0,0,1,0,2,3,0,4},{0,1,2,0,3,4,0,5},
    {0,0,0,0,1,2,0,3},{0,0,0,1,2,3,0,4},{0,0,0,1,2,3,0,4},{0,1,0,2,3,4,0,5},{0,0,0,1,2,3,0,4},{0,0,1,2,3,4,0,5},{0,0,1,2,3,4,0,5},{0,1,2,3,4,5,0,6},
    {0,0,0,0,0,0,0,1},{0,0,0,0,0,0,1,2},{0,0,0,0,0,0,1,2},{0,1,0,0,0,0,2,3},{0,0,0,0,0,0,1,2},{0,0,1,0,0,0,2,3},{0,0,1,0,0,0,2,3},{0,1,2,0,0,0,3,4},
    {0,0,0,0,0,0,1,2},{0,0,0,1,0,0,2,3},{0,0,0,1,0,0,2,3},{0,1,0,2,0,0,3,4},{0,0,0,1,0,0,2,3},{0,0,1,2,0,0,3,4},{0,0,1,2,0,0,3,4},{0,1,2,3,0,0,4,5},
    {0,0,0,0,0,0,1,2},{0,0,0,0,1,0,2,3},{0,0,0,0,1,0,2,3},{0,1,0,0,2,0,3,4},{0,0,0,0,1,0,2,3},{0,0,1,0,2,0,3,4},{0,0,1,0,2,0,3,4},{0,1,2,0,3,0,4,5},
    {0,0,0,0,1,0,2,3},{0,0,0,1,2,0,3,4},{0,0,0,1,2,0,3,4},{0,1,0,2,3,0,4,5},{0,0,0,1,2,0,3,4},{0,0,1,2,3,0,4,5},{0,0,1,2,3,0,4,5},{0,1,2,3,4,0,5,6},
    {0,0,0,0,0,0,1,2},{0,0,0,0,0,1,2,3},{0,0,0,0,0,1,2,3},{0,1,0,0,0,2,3,4},{0,0,0,0,0,1,2,3},{0,0,1,0,0,2,3,4},{0,0,1,0,0,2,3,4},{0,1,2,0,0,3,4,5},
    {0,0,0,0,0,1,2,3},{0,0,0,1,0,2,3,4},{0,0,0,1,0,2,3,4},{0,1,0,2,0,3,4,5},{0,0,0,1,0,2,3,4},{0,0,1,2,0,3,4,5},{0,0,1,2,0,3,4,5},{0,1,2,3,0,4,5,6},
    {0,0,0,0,0,1,2,3},{0,0,0,0,1,2,3,4},{0,0,0,0,1,2,3,4},{0,1,0,0,2,3,4,5},{0,0,0,0,1,2,3,4},{0,0,1,0,2,3,4,5},{0,0,1,0,2,3,4,5},{0,1,2,0,3,4,5,6},
    {0,0,0,0,1,2,3,4},{0,0,0,1,2,3,4,5},{0,0,0,1,2,3,4,5},{0,1,0,2,3,4,5,6},{0,0,0,1,2,3,4,5},{0,0,1,2,3,4,5,6},{0,0,1,2,3,4,5,6},{0,1,2,3,4,5,6,7}
};
#endif

#ifdef AXR_SIMD
/* AVX2 decoder (AX_RANS_SIMD, default on when the CPU has AVX2; AX_RANS_SIMD=0: scalar). The same steps as
 * axr_decode_scalar, 8 lanes per vector, the 32 lanes of a group as 4 vectors in lane order (the order of the
 * group's words): slot = x & 4095; one gather from a table packed per slot as sym | (f-1) << 8 | (slot-c) << 20;
 * x = f * (x >> 12) + slot - c; the 8 symbols are bytes out[32g + 8v ..]; the lanes below 2^16 (movemask) take the
 * next popcount words, each lane its rank among them (a permutation picked by the mask from a 256-entry table).
 * Bounds once per vector: the words a vector needs must be < W (the scalar fails at the same word). The last,
 * partial group and the final checks run as in the scalar decoder. */
__attribute__((target("avx2,popcnt"))) static int axr_decode_avx2(const uint8_t* src, size_t sz, uint8_t* out, size_t n) {
    if (sz < 32 + 128 + 4) return -1;
    const uint8_t* p = src; const uint8_t* e = src + sz;
    uint32_t tab[AXR_M];
    uint32_t acc = 0;
    const uint8_t* bm = p; p += 32;
    for (int s = 0; s < 256; s++) {
        if (!(bm[s >> 3] & (1u << (s & 7)))) continue;
        uint32_t v = 0; int sh = 0;
        for (;;) { if (p >= e || sh > 14) return -1; uint8_t b = *p++; v |= (uint32_t)(b & 0x7F) << sh; if (!(b & 0x80)) break; sh += 7; }
        if (v == 0 || acc + v > AXR_M) return -1;
        const uint32_t base = (uint32_t)s | ((v - 1) << 8);
        for (uint32_t k = 0; k < v; k++) tab[acc + k] = base | (k << 20);
        acc += v;
    }
    if (acc != AXR_M) return -1;
    if ((size_t)(e - p) < 128 + 4) return -1;
    uint32_t x[AXR_LANES];
    for (int l = 0; l < AXR_LANES; l++) { memcpy(&x[l], p, 4); p += 4; if (x[l] < AXR_L) return -1; }
    uint32_t W; memcpy(&W, p, 4); p += 4;
    if ((size_t)(e - p) != (size_t)W * 2) return -1;
    const uint8_t* wp = p; uint32_t wi = 0;
    const size_t full = n / AXR_LANES;
    __m256i X[4];
    for (int v = 0; v < 4; v++) X[v] = _mm256_loadu_si256((const __m256i*)(x + 8 * v));
    const __m256i M12 = _mm256_set1_epi32(AXR_M - 1), MF = _mm256_set1_epi32(0xFFF), MB = _mm256_set1_epi32(0xFF), ONE = _mm256_set1_epi32(1), Z = _mm256_setzero_si256();
    const __m256i PK = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (size_t g = 0; g < full; g++) {
        uint8_t* o = out + g * AXR_LANES;
        for (int v = 0; v < 4; v++) {
            const __m256i slot = _mm256_and_si256(X[v], M12);
            const __m256i t = _mm256_i32gather_epi32((const int*)tab, slot, 4);
            const __m256i fr = _mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(t, 8), MF), ONE);
            __m256i nx = _mm256_add_epi32(_mm256_mullo_epi32(fr, _mm256_srli_epi32(X[v], AXR_PBITS)), _mm256_srli_epi32(t, 20));
            const __m256i sy = _mm256_shuffle_epi8(_mm256_and_si256(t, MB), PK);       /* bytes 0..3 of each 128-bit half */
            const uint32_t lo = (uint32_t)_mm256_cvtsi256_si32(sy), hi = (uint32_t)_mm256_extract_epi32(sy, 4);
            memcpy(o + 8 * v, &lo, 4); memcpy(o + 8 * v + 4, &hi, 4);
            const __m256i need = _mm256_cmpeq_epi32(_mm256_srli_epi32(nx, 16), Z);   /* x < 2^16 */
            const int m = _mm256_movemask_ps(_mm256_castsi256_ps(need));
            if (m) {
                const uint32_t k = (uint32_t)_mm_popcnt_u32((uint32_t)m);
                if (wi + k > W) return -1;
                __m128i w8;
                if (W - wi >= 8) w8 = _mm_loadu_si128((const __m128i*)(wp + 2 * (size_t)wi));
                else { uint16_t tmp[8] = {0}; memcpy(tmp, wp + 2 * (size_t)wi, 2 * (size_t)(W - wi)); w8 = _mm_loadu_si128((const __m128i*)tmp); }
                const __m256i wv = _mm256_permutevar8x32_epi32(_mm256_cvtepu16_epi32(w8), _mm256_loadu_si256((const __m256i*)AXR_PERM[m]));
                nx = _mm256_blendv_epi8(nx, _mm256_or_si256(_mm256_slli_epi32(nx, 16), wv), need);
                wi += k;
            }
            X[v] = nx;
        }
    }
    for (int v = 0; v < 4; v++) _mm256_storeu_si256((__m256i*)(x + 8 * v), X[v]);
    if (full * AXR_LANES < n) {                                      /* the partial group, scalar */
        for (int l = 0; l < AXR_LANES; l++) {
            const size_t i = full * AXR_LANES + (size_t)l; if (i >= n) break;
            const uint32_t slot = x[l] & (AXR_M - 1), t = tab[slot], s = t & 0xFF;
            out[i] = (uint8_t)s;
            x[l] = (((t >> 8) & 0xFFF) + 1) * (x[l] >> AXR_PBITS) + (t >> 20);
            if (x[l] < AXR_L) {
                if (wi >= W) return -1;
                uint16_t w; memcpy(&w, wp + 2 * (size_t)wi, 2); wi++;
                x[l] = (x[l] << 16) | w;
            }
        }
    }
    if (wi != W) return -1;
    for (int l = 0; l < AXR_LANES; l++) if (x[l] != AXR_L) return -1;
    return 0;
}
#endif

#ifdef AXR_SIMD
/* AVX-512 decoder (AX_RANS_SIMD=512; needs AVX-512 F/BW/VL/VBMI2): 16 lanes per vector, the 32 lanes as 2 vectors in
 * lane order; the same table and steps as axr_decode_avx2; renormalisation by an expand-load of 16-bit words (vpexpandw:
 * the lanes below 2^16 take the next popcount words in lane order, and only those words are read). */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi2,popcnt"))) static int axr_decode_avx512(const uint8_t* src, size_t sz, uint8_t* out, size_t n) {
    if (sz < 32 + 128 + 4) return -1;
    const uint8_t* p = src; const uint8_t* e = src + sz;
    uint32_t tab[AXR_M];
    uint32_t acc = 0;
    const uint8_t* bm = p; p += 32;
    for (int s = 0; s < 256; s++) {
        if (!(bm[s >> 3] & (1u << (s & 7)))) continue;
        uint32_t v = 0; int sh = 0;
        for (;;) { if (p >= e || sh > 14) return -1; uint8_t b = *p++; v |= (uint32_t)(b & 0x7F) << sh; if (!(b & 0x80)) break; sh += 7; }
        if (v == 0 || acc + v > AXR_M) return -1;
        const uint32_t base = (uint32_t)s | ((v - 1) << 8);
        for (uint32_t k = 0; k < v; k++) tab[acc + k] = base | (k << 20);
        acc += v;
    }
    if (acc != AXR_M) return -1;
    if ((size_t)(e - p) < 128 + 4) return -1;
    uint32_t x[AXR_LANES];
    for (int l = 0; l < AXR_LANES; l++) { memcpy(&x[l], p, 4); p += 4; if (x[l] < AXR_L) return -1; }
    uint32_t W; memcpy(&W, p, 4); p += 4;
    if ((size_t)(e - p) != (size_t)W * 2) return -1;
    const uint8_t* wp = p; uint32_t wi = 0;
    const size_t full = n / AXR_LANES;
    __m512i X[2];
    for (int v = 0; v < 2; v++) X[v] = _mm512_loadu_si512((const void*)(x + 16 * v));
    const __m512i M12 = _mm512_set1_epi32(AXR_M - 1), MF = _mm512_set1_epi32(0xFFF), ONE = _mm512_set1_epi32(1), LIM = _mm512_set1_epi32(AXR_L);
    for (size_t g = 0; g < full; g++) {
        uint8_t* o = out + g * AXR_LANES;
        for (int v = 0; v < 2; v++) {
            const __m512i slot = _mm512_and_si512(X[v], M12);
            const __m512i t = _mm512_i32gather_epi32(slot, (const void*)tab, 4);
            const __m512i fr = _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(t, 8), MF), ONE);
            __m512i nx = _mm512_add_epi32(_mm512_mullo_epi32(fr, _mm512_srli_epi32(X[v], AXR_PBITS)), _mm512_srli_epi32(t, 20));
            _mm_storeu_si128((__m128i*)(o + 16 * v), _mm512_cvtepi32_epi8(t));          /* low byte of each entry = symbol */
            const __mmask16 need = _mm512_cmplt_epu32_mask(nx, LIM);
            if (need) {
                const uint32_t k = (uint32_t)_mm_popcnt_u32((uint32_t)need);
                if (wi + k > W) return -1;
                const __m256i w16 = _mm256_maskz_expandloadu_epi16(need, (const void*)(wp + 2 * (size_t)wi));
                nx = _mm512_mask_or_epi32(nx, need, _mm512_slli_epi32(nx, 16), _mm512_cvtepu16_epi32(w16));
                wi += k;
            }
            X[v] = nx;
        }
    }
    for (int v = 0; v < 2; v++) _mm512_storeu_si512((void*)(x + 16 * v), X[v]);
    if (full * AXR_LANES < n) {
        for (int l = 0; l < AXR_LANES; l++) {
            const size_t i = full * AXR_LANES + (size_t)l; if (i >= n) break;
            const uint32_t slot = x[l] & (AXR_M - 1), t = tab[slot], s = t & 0xFF;
            out[i] = (uint8_t)s;
            x[l] = (((t >> 8) & 0xFFF) + 1) * (x[l] >> AXR_PBITS) + (t >> 20);
            if (x[l] < AXR_L) {
                if (wi >= W) return -1;
                uint16_t w; memcpy(&w, wp + 2 * (size_t)wi, 2); wi++;
                x[l] = (x[l] << 16) | w;
            }
        }
    }
    if (wi != W) return -1;
    for (int l = 0; l < AXR_LANES; l++) if (x[l] != AXR_L) return -1;
    return 0;
}

#endif

/* AX_RANS_SIMD: the AVX2 decoder when the CPU has it (default 1), 512 = the AVX-512 one, 0 = scalar; read once */
#ifdef AXR_SIMD
static int axr_simd_mode(void) {
    const char* ev = ax_getenv("AX_RANS_SIMD"); const int want = ev ? atoi(ev) : 1;
    if (want == 512 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512vbmi2")) return 512;
    return (want != 0 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt")) ? 1 : 0;
}
#endif
static inline int axr_decode(const uint8_t* src, size_t sz, uint8_t* out, size_t n) {
#ifdef AXR_SIMD
    /* computed once; concurrent first calls store the same value (as the other lazily built tables of the decoders) */
    static volatile int mode_s = -1; int mode = mode_s; if (mode < 0) { mode = axr_simd_mode(); mode_s = mode; }
    if (mode == 512) return axr_decode_avx512(src, sz, out, n);
    if (mode) return axr_decode_avx2(src, sz, out, n);
#endif
    return axr_decode_scalar(src, sz, out, n);
}
#endif
