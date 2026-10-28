/* ax_fast.h - ACEPX2 header version 3, profile FAST: the format constants, the archive reader and the block
 * decoder. C99, no dependency, no allocation. One identical copy in src/, c/ and python/csrc/ (scripts/cdec_test.sh
 * checks that they are byte for byte the same); the C++ library includes this file, so every reader runs the one
 * decoder. Specification: docs/FORMAT_FAST.md. Encoder: src/ax_fast_enc.h (C++).
 *
 * Archive (all little-endian, packed):
 *   [0]  AetHeader, 68 B: magic "ACEPX2\0\0", u32 version = 3, u64 orig_size, u32 block_size, u32 num_blocks,
 *        u8 xxh3[8] (XXH3-64 of the original), u32 profile = 1 (FAST), u32 min_match (4..32), u64 payload_size,
 *        u64 reserved = 0, u64 reserved = 0
 *   [68] u64 boff[num_blocks + 1]: start of every block inside the payload, boff[0] = 0, boff[num_blocks] = payload_size
 *   [..] payload: the blocks, in order
 * A version-2 reader that ignores the version field reads bytes 36..43 as a stream size of 2^32 x min_match + 1 bytes
 * and rejects the archive as truncated: the field placement is deliberate.
 *
 * Block (independent: every match source lies inside the block; decodable alone, so regions and threads work):
 *   u32 ntok, u32 next, u32 tail
 *   u32 tok[ntok]      ll (bits 0..5) | mc (bits 6..11) | d (bits 12..31)
 *   u8  ext[next]      length extensions: bytes of 255 then one byte < 255, in token order, ll first, then mc
 *   u8  lit[]          literals in token order, then the `tail` literals after the last token
 *   u8  pad[64]        64 bytes the decoder may read but never uses (zeros from the reference encoder)
 * Token: copy ll literals (ll = 63: 63 + extension), then ml bytes from dist = d + 64 back:
 *   mc < 63: ml = mc + min_match (the encoder keeps ml <= 64); mc = 63: ml = 65 + extension.
 * The minimum distance 64 makes every copy of up to 64 bytes a plain forward copy, and the padding lets the decoder
 * read literals 32 bytes at a time without a bound check: the reads land in the pad, never past the block.
 *
 * Decoder contract (axf_dec_block): src is exactly ssz bytes, dst exactly n bytes. Any input either decodes to n
 * bytes (return n) or returns 0, and never reads outside [src, src + ssz) or writes outside [dst, dst + n).
 * The unconditional 32-byte copies run in guarded runs: k tokens with k <= (output room) / 160 and k <= (literals left)
 * / 62 + 1, so no token of the run can write past dst or read past the pad; escapes and the block end go through exact
 * copies, one token at a time. */
#ifndef AX_FAST_H
#define AX_FAST_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define AXF_VERSION       3u
#define AXF_PROFILE_FAST  1u
#define AXF_MIN_DIST      64u
#define AXF_COPY          64u          /* longest match copied by a direct code; longer matches are split by the encoder */
#define AXF_PAD           64u          /* readable slack after the literals of a block */
#define AXF_BHDR          12u          /* ntok, next, tail */
#define AXF_BLOCK_MAX     (1u << 20)   /* 20-bit distance field + 64 covers a block of 2^20 bytes */
#define AXF_MM_MIN        4u
#define AXF_MM_MAX        32u
#define AXF_ROOM          160u         /* output room the fast path needs for one token */

#if defined(__GNUC__)
#define AXF_INLINE static inline __attribute__((always_inline))
#define AXF_LIKELY(x) __builtin_expect(!!(x), 1)
#define AXF_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define AXF_INLINE static inline
#define AXF_LIKELY(x) (x)
#define AXF_UNLIKELY(x) (x)
#endif

typedef struct {
    uint64_t orig, payload;
    uint32_t bs, nb, profile, mm;
    const uint8_t* tab;    /* u64 boff[nb + 1], unaligned: read through axf_boff */
    const uint8_t* data;   /* payload */
} axf_arc_t;

AXF_INLINE uint32_t axf_rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
AXF_INLINE uint64_t axf_rd64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
AXF_INLINE uint64_t axf_boff(const axf_arc_t* a, uint32_t i) { return axf_rd64(a->tab + 8ull * i); }

/* 1 if the 68 bytes at p are a version-3 header (magic + version only; axf_open validates the rest) */
AXF_INLINE int axf_is_v3(const uint8_t* p, size_t n) {
    return n >= 68 && memcmp(p, "ACEPX2\0\0", 8) == 0 && axf_rd32(p + 8) == AXF_VERSION;
}

/* Parse and validate a version-3 archive: header fields, the block table (monotonic, inside the payload, every block
 * at least a block header plus the pad), payload inside the buffer. Returns 0, or -2 (ACEAPEX_ERR_DATA). */
static int axf_open(const uint8_t* p, size_t n, axf_arc_t* a) {
    uint64_t need, prev, i;
    if (!axf_is_v3(p, n)) return -2;
    a->orig = axf_rd64(p + 12); a->bs = axf_rd32(p + 20); a->nb = axf_rd32(p + 24);
    a->profile = axf_rd32(p + 36); a->mm = axf_rd32(p + 40); a->payload = axf_rd64(p + 44);
    if (axf_rd64(p + 52) != 0 || axf_rd64(p + 60) != 0) return -2;                 /* reserved: must be zero */
    if (a->profile != AXF_PROFILE_FAST || a->mm < AXF_MM_MIN || a->mm > AXF_MM_MAX) return -2;
    if (a->bs == 0 || a->bs > AXF_BLOCK_MAX || a->nb == 0 || a->orig == 0) return -2;    /* the empty input is a version-2 archive */
    if ((uint64_t)a->nb * a->bs < a->orig || (uint64_t)(a->nb - 1) * a->bs >= a->orig) return -2;
    need = 68 + 8ull * ((uint64_t)a->nb + 1);
    if (need > n || a->payload > n - need) return -2;
    a->tab = p + 68; a->data = p + (size_t)need;
    if (axf_boff(a, 0) != 0 || axf_boff(a, a->nb) != a->payload) return -2;
    prev = 0;
    for (i = 1; i <= a->nb; i++) {
        const uint64_t o = axf_boff(a, (uint32_t)i);
        if (o < prev || o - prev < AXF_BHDR + AXF_PAD) return -2;
        prev = o;
    }
    return 0;
}

/* original bytes held by block b */
AXF_INLINE uint32_t axf_block_len(const axf_arc_t* a, uint32_t b) {
    const uint64_t st = (uint64_t)b * a->bs;
    return (uint32_t)(a->orig - st < a->bs ? a->orig - st : a->bs);
}

/* 32-byte copy; memcpy of a constant size compiles to one vector move (AVX2), two (SSE2/NEON) */
AXF_INLINE void axf_cp32(uint8_t* d, const uint8_t* s) { memcpy(d, s, 32); }
/* up to AXF_COPY + 30 bytes in 32-byte steps; exact when the source lies >= 64 bytes before the destination or is disjoint */
AXF_INLINE void axf_cpw(uint8_t* d, const uint8_t* s, uint32_t l) {
    uint32_t k;
    axf_cp32(d, s);
    for (k = 32; k < l; k += 32) axf_cp32(d + k, s + k);
}
/* a length extension: bytes of 255 then one byte below 255; bad = 1 when the stream ends or the value overflows */
AXF_INLINE uint32_t axf_ext(const uint8_t** ext, const uint8_t* eend, int* bad) {
    uint32_t r = 0;
    for (;;) {
        uint8_t b;
        if (*ext >= eend) { *bad = 1; return 0; }
        b = *(*ext)++; r += b;
        if (b < 255) return r;
        if (r > (1u << 30)) { *bad = 1; return 0; }
    }
}

/* One block. src: ssz bytes (the block as stored, pad included). dst: n bytes. mm: min_match from the header.
 * Returns n, or 0 for a malformed block. See the contract above. */
AXF_INLINE uint32_t axf_dec_block_mm(const uint8_t* src, size_t ssz, uint8_t* dst, uint32_t n, const uint32_t mm) {
    uint32_t ntok, next, tail, i;
    const uint8_t *tp, *ext, *lit, *lend, *eend;
    uint8_t *d, *dend;
    if (ssz < AXF_BHDR + AXF_PAD) return 0;
    ntok = axf_rd32(src); next = axf_rd32(src + 4); tail = axf_rd32(src + 8);
    if ((uint64_t)AXF_BHDR + 4ull * ntok + next + AXF_PAD > ssz) return 0;
    tp = src + AXF_BHDR; ext = tp + 4ull * ntok; lit = ext + next; eend = lit;
    lend = src + ssz - AXF_PAD;                     /* literals end here; the pad [lend, lend + 64) is readable */
    d = dst; dend = dst + n;
    for (i = 0; i < ntok; ) {
        /* Guarded run: k tokens with no bound check per token. A fast token reads at most 64 literal bytes from lit and
         * advances lit by at most 62, and writes at most 62 + 96 bytes from d. With lit <= lend at the start (kept by
         * every path) and k <= (lend - lit) / 62 + 1, every token of the run starts with lit <= lend, so its 64-byte
         * read ends inside the pad; with k <= (dend - d) / 160 every write stays in dst. Only an escape (ll or mc = 63)
         * or a source before the block leave the run. After the run lit may lie past lend: checked once. */
        uint32_t k = ntok - i;
        {   const size_t kd = (size_t)(dend - d) / AXF_ROOM, kl = (size_t)(lend - lit) / 62u + 1u;
            if (kd < k) k = (uint32_t)kd;
            if (kl < k) k = (uint32_t)kl; }
        for (; k; k--, i++) {
            const uint32_t t = axf_rd32(tp + 4ull * i);
            const uint32_t ll = t & 63u, mc = (t >> 6) & 63u, dist = (t >> 12) + AXF_MIN_DIST;
            if (AXF_UNLIKELY(ll == 63u || mc == 63u)) break;
            axf_cp32(d, lit); if (ll > 32u) axf_cp32(d + 32, lit + 32);
            d += ll; lit += ll;
            if (AXF_UNLIKELY((size_t)(d - dst) < dist)) return 0;    /* source before the block */
            axf_cpw(d, d - dist, mc + mm); d += mc + mm;             /* <= 94 bytes: at most 3 steps */
        }
        if (AXF_UNLIKELY(lit > lend)) return 0;                      /* the run consumed literals that are not there */
        if (i >= ntok) break;
        {   /* exact path, one token: block end, long literal run, long match */
            const uint32_t t = axf_rd32(tp + 4ull * i);
            uint32_t ll = t & 63u, mc = (t >> 6) & 63u, ml;
            const uint32_t dist = (t >> 12) + AXF_MIN_DIST;
            i++;
            int bad = 0; uint32_t c0;
            if (ll == 63u) { ll += axf_ext(&ext, eend, &bad); if (bad) return 0; }
            if ((size_t)(lend - lit) < ll || (size_t)(dend - d) < ll) return 0;
            /* lit + ll <= lend, so 32-byte steps read at most 31 bytes into the pad; with 32 bytes of room after the run
               the steps may write past it (the next token overwrites those bytes) - no call, no size-dependent branches */
            if ((size_t)(dend - d) >= (size_t)ll + 32u) axf_cpw(d, lit, ll); else memcpy(d, lit, ll);
            d += ll; lit += ll;
            if ((size_t)(d - dst) < dist) return 0;
            ml = mc + mm;
            if (mc == 63u) { ml = AXF_COPY + 1 + axf_ext(&ext, eend, &bad); if (bad) return 0; }
            if ((size_t)(dend - d) < ml) return 0;
            if ((size_t)(dend - d) >= (size_t)ml + 32u) axf_cpw(d, d - dist, ml);   /* dist >= 64 > 32: forward 32-byte steps are exact */
            else for (c0 = 0; c0 < ml; ) { uint32_t c = ml - c0 < dist ? ml - c0 : dist; memcpy(d + c0, d + c0 - dist, c); c0 += c; }
            d += ml;
        }
    }
    if ((size_t)(lend - lit) != tail || (size_t)(dend - d) != tail) return 0;
    memcpy(d, lit, tail);
    return n;
}
/* Variant with a bound check on every token (2.4.0 first cut, 9f8a4ab). Measured on ace-core (lzbench -t8,8, median of 7,
 * results/fast-bench-2026-10-27-0d3a065.md): with few tokens per KiB (min match 16 and 12, most tokens escape to the
 * exact path) it is 1.7-1.8 % faster than the guarded runs; with many tokens (min match 8 and 5) the guarded runs are
 * 4.5-15 % faster. Both keep the same contract; the dispatcher below picks per min_match. */
AXF_INLINE uint32_t axf_dec_block_tok_mm(const uint8_t* src, size_t ssz, uint8_t* dst, uint32_t n, const uint32_t mm) {
    uint32_t ntok, next, tail, i;
    const uint8_t *tp, *ext, *lit, *lend, *eend;
    uint8_t *d, *dend, *dfast;
    if (ssz < AXF_BHDR + AXF_PAD) return 0;
    ntok = axf_rd32(src); next = axf_rd32(src + 4); tail = axf_rd32(src + 8);
    if ((uint64_t)AXF_BHDR + 4ull * ntok + next + AXF_PAD > ssz) return 0;
    tp = src + AXF_BHDR; ext = tp + 4ull * ntok; lit = ext + next; eend = lit;
    lend = src + ssz - AXF_PAD;                     /* literals end here; the pad [lend, lend + 64) is readable */
    d = dst; dend = dst + n;
    dfast = n >= AXF_ROOM ? dend - (AXF_ROOM - 1) : dst;   /* fast path while d < dfast, i.e. >= 160 bytes of room */
    for (i = 0; i < ntok; i++) {
        const uint32_t t = axf_rd32(tp + 4ull * i);
        uint32_t ll = t & 63u, mc = (t >> 6) & 63u, ml;
        const uint32_t dist = (t >> 12) + AXF_MIN_DIST;
        if (AXF_LIKELY(d < dfast && ll != 63u && mc != 63u)) {
            /* literal copy reads lit..lit+63: lit <= lend holds here, so the read ends inside the pad */
            /* branch-free on lengths: always 64 literal bytes and 96 match bytes (ll <= 62, ml <= 94); the room check
               (>= 160 bytes before the copies, <= 62 literal bytes in front of the match: 98 left) covers both */
            axf_cp32(d, lit); axf_cp32(d + 32, lit + 32);
            d += ll; lit += ll;
            if (AXF_UNLIKELY((size_t)(d - dst) < dist)) return 0;    /* source before the block */
            ml = mc + mm;
            { const uint8_t* sm = d - dist; axf_cp32(d, sm); axf_cp32(d + 32, sm + 32); axf_cp32(d + 64, sm + 64); }
            d += ml;
            if (AXF_UNLIKELY(lit > lend)) return 0;                  /* the literals of this token were not all there */
            continue;
        }
        {   /* exact path: block end, long literal run, long match */
            int bad = 0; uint32_t k;
            if (ll == 63u) { ll += axf_ext(&ext, eend, &bad); if (bad) return 0; }
            if ((size_t)(lend - lit) < ll || (size_t)(dend - d) < ll) return 0;
            /* lit + ll <= lend, so 32-byte steps read at most 31 bytes into the pad; with 32 bytes of room after the run
               the steps may write past it (the next token overwrites those bytes) - no call, no size-dependent branches */
            if ((size_t)(dend - d) >= (size_t)ll + 32u) axf_cpw(d, lit, ll); else memcpy(d, lit, ll);
            d += ll; lit += ll;
            if ((size_t)(d - dst) < dist) return 0;
            ml = mc + mm;
            if (mc == 63u) { ml = AXF_COPY + 1 + axf_ext(&ext, eend, &bad); if (bad) return 0; }
            if ((size_t)(dend - d) < ml) return 0;
            if ((size_t)(dend - d) >= (size_t)ml + 32u) axf_cpw(d, d - dist, ml);   /* dist >= 64 > 32: forward 32-byte steps are exact */
            else for (k = 0; k < ml; ) { uint32_t c = ml - k < dist ? ml - k : dist; memcpy(d + k, d + k - dist, c); k += c; }
            d += ml;
        }
    }
    if ((size_t)(lend - lit) != tail || (size_t)(dend - d) != tail) return 0;
    memcpy(d, lit, tail);
    return n;
}
/* Built without -march (lzbench, distribution packages: baseline x86-64, SSE2) the 32-byte copies are two 16-byte moves.
 * On x86-64 Linux with GCC 6+ / Clang 14+ the block decoder is cloned for AVX2 and picked at load time (ifunc), so a
 * generic binary runs the 32-byte moves on CPUs that have them. Other targets build the one generic function. */
#if defined(__x86_64__) && defined(__linux__) && defined(__GLIBC__) && ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 6) || (defined(__clang__) && __clang_major__ >= 14)) && !defined(AXF_NO_CLONES)
#define AXF_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define AXF_CLONES
#endif
AXF_CLONES static uint32_t axf_dec_block(const uint8_t* src, size_t ssz, uint8_t* dst, uint32_t n, uint32_t mm) {
    switch (mm) {   /* the common min_match values as compile-time constants; per-token checks for 16 and 12, guarded runs below */
    case 16: return axf_dec_block_tok_mm(src, ssz, dst, n, 16);
    case 12: return axf_dec_block_tok_mm(src, ssz, dst, n, 12);
    case 8:  return axf_dec_block_mm(src, ssz, dst, n, 8);
    case 5:  return axf_dec_block_mm(src, ssz, dst, n, 5);
    default: return axf_dec_block_mm(src, ssz, dst, n, mm);
    }
}
/* block b of an opened archive into dst (axf_block_len(a, b) bytes); returns that length or 0 */
AXF_INLINE uint32_t axf_dec_block_of(const axf_arc_t* a, uint32_t b, uint8_t* dst) {
    const uint64_t o = axf_boff(a, b), e = axf_boff(a, b + 1);
    return axf_dec_block(a->data + o, (size_t)(e - o), dst, axf_block_len(a, b), a->mm);
}
#endif
