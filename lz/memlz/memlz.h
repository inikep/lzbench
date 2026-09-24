// #define MEMLZ_TEST

// SPDX-License-Identifier: MIT
//
// memlz 0.5 beta - extremely fast header-only compression library for C and C++ on x64/x86
//
// Copyright 2025 - 2026, Lasse Mikkel Reinhold
//
// Attributions: This is an 8-byte-word version of the Chameleon compression algorithmn
// by Guillaume Voirin. Thanks to Charles Bloom for his simple reference implementation and 
// for the original LZP-style of algorithms.

#ifndef memlz_h
#define memlz_h

#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h> 
#ifndef __cplusplus
#include <stdalign.h>
#endif

typedef struct memlz_state memlz_state;

#ifdef __cplusplus
extern "C" {
#endif

    /// Compress non-streaming data. The destination buffer must be at least
    /// memlz_max_compressed_len(len) large.
    /// 
    /// Returns 0 if internal memory allocation failed
    size_t memlz_compress(void* destination, const void* source, size_t len);

    /// Decompress non-streaming data. The destination buffer must be at least 
    /// memlz_decompressed_len(source) large.
    ///
    /// Returns 0 if compressed data was malformed or if internal memory allocation failed.
    size_t memlz_decompress(void* destination, const void* source);

    /// Compress stream: First call memlz_reset(state) and then call memlz_stream_compress()
    /// repeatedly. Each call will always compress and ouptput the full input data given.
    /// There is no flush function.
    ///
    /// The destination buffer must be at least memlz_max_compressed_len(len) large.
    size_t memlz_stream_compress(void* destination, const void* source, size_t len, memlz_state* state);

    /// Decompress streaming data: First call memlz_reset(state) and then call
    /// memlz_stream_decompress() repeatedly in the same order for the compressed data as when
    /// you called memlz_compress(). 
    ///
    /// The destination buffer must be at least memlz_decompressed_len(source) large.
    ///
    /// Returns 0 if compressed data was malformed
    size_t memlz_stream_decompress(void* destination, const void* source, memlz_state* state);

    /// Takes compressed data as input and returns the decompressed len. Only the first
    /// memlz_header_len() number of bytes need to present
    size_t memlz_compressed_len(const void* src);

    /// Takes compressed data as input and returns the compressed len. Only the first
    /// memlz_header_len() number of bytes need to present
    size_t memlz_decompressed_len(const void* src);

    /// Return the largest number of bytes that a given input can compress into. Note that certain
    /// kinds of data may grow beyond its original size.
    size_t memlz_max_compressed_len(size_t input);

    /// Returns the number of bytes of compressed data that need to be present in order to call 
    /// memlz_compressed_len() and memlz_decompressed_len()
    size_t memlz_header_len(void);

    ///  Call this before the first call to memlz_compress() or memlz_decompress()
    void memlz_reset(memlz_state* c);

#ifdef __cplusplus
} // extern C
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MEMLZ_SSE
#endif

#ifdef MEMLZ_TEST
#define MEMLZ_IMPLEMENTATION
#endif

#ifdef MEMLZ_IMPLEMENTATION

// Identify sequences of repeated bytes (such as sequences of zeroes) and special handle them
#define MEMLZ_DO_RLE

// Identify sequences of incompressible data (such as jpg or zip files, zip files) and special handle them
#define MEMLZ_DO_INCOMPRESSIBLE

// The rest of this header file is internals
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MEMLZ_SSE
#include <smmintrin.h>
#endif

#ifdef __cplusplus
#define MEMLZ_ALIGN_16 alignas(16)
#else
#define MEMLZ_ALIGN_16 _Alignas(16)
#endif

#ifndef _MSC_VER
#ifdef MEMLZ_SSE
#define MEMLZ_SSE42 __attribute__((target("sse4.2")))
#else
#define MEMLZ_SSE42
#endif
#define MEMLZ_UNUSED __attribute__((unused))
#else
#define MEMLZ_SSE42
#define MEMLZ_UNUSED
#endif

#define MEMLZ_INCOMPRESSIBLE_TRIGGER (4)
#define MEMLZ_INCOMPRESSIBLE_ADVANCE (16 * MEMLZ_INCOMPRESSIBLE_TRIGGER)
#define MEMLZ_PROBELEN (2 * 1024)
#define MEMLZ_BLOCKLEN (128 * 1024)
#define MEMLZ_RLE 'D'
#define MEMLZ_MIN_RLE (4 * sizeof(uint64_t))
#define MEMLZ_RESTRICT __restrict
#define MEMLZ_UNROLL4(op) op; op; op; op
#define MEMLZ_UNROLL16(op) op; op; op; op; op; op; op; op; op; op; op; op; op; op; op; op;
#define MEMLZ_NORMAL_4 'A'
#define MEMLZ_NORMAL_8 'B'
#define MEMLZ_UNCOMPRESSED 'C'
#define MEMLZ_MIN(X, Y) ((X) < (Y) ? (X) : (Y))

static const size_t memlz_fields = 2;

static inline uint16_t memlz_r16(const void* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t memlz_r32(const void* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t memlz_r64(const void* p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void memlz_w16(void* d, uint16_t v) { memcpy(d, &v, 2); }
static inline void memlz_w32(void* d, uint32_t v) { memcpy(d, &v, 4); }
static inline void memlz_w64(void* d, uint64_t v) { memcpy(d, &v, 8); }

static uint64_t memlz_read(const void* src) {
    const uint8_t* s = (const uint8_t*)src;
    size_t bytes = ((size_t)*s) >> 6;

    return bytes == 0 ? *s & 0x3f
        : bytes == 1 ? memlz_r16(s + 1)
        : bytes == 2 ? memlz_r32(s + 1)
        : memlz_r64(s + 1);
}

static size_t memlz_bytes(const void* src) {
    const uint8_t* s = (const uint8_t*)src;
    size_t bytes = ((size_t)*s) >> 6;

    return bytes == 0ULL ? 1ULL
        : bytes == 1ULL ? 3ULL
        : bytes == 2ULL ? 5ULL
        : 9ULL;
}

static void memlz_write(void* dst, uint64_t value, size_t bytes) {
    assert(bytes == 1 || bytes == 3 || bytes == 5 || bytes == 9);

    uint8_t* d = (uint8_t*)dst;

    if (bytes == 1) {
        assert(value < 64);
        *d = (uint8_t)value;
    }
    else if (bytes == 3) {
        assert(value <= 0xffff); // Rettet fra < til <= da 0xffff præcis kan være i 16-bit
        *d = 0x40;
        memlz_w16(d + 1, (uint16_t)value);
    }
    else if (bytes == 5) {
        assert(value <= 0xffffffff); // Rettet fra < til <= da 0xffffffff præcis kan være i 32-bit
        *d = 0x80;
        memlz_w32(d + 1, (uint32_t)value);
    }
    else if (bytes == 9) {
        *d = 0xc0;
        memlz_w64(d + 1, value);
    }
}


static uint64_t memlz_fit(uint64_t value) {
    return value < 64ULL ? 1ULL : value <= 0xffffULL ? 3ULL : value <= 0xffffffffULL ? 5ULL : 9ULL;
}

#ifdef _MSC_VER
#define MEMLZ_FORCE_INLINE __forceinline
#else
#define MEMLZ_FORCE_INLINE inline __attribute__((always_inline))
#endif

typedef struct memlz_state {
    uint64_t hash64[1 << 16];
    uint32_t hash32[1 << 16];
    uint64_t total_input;
    uint64_t total_output;
    size_t mod;
    size_t wordlen;
    size_t cs4;
    size_t cs8;
    size_t incompressible;
    char reset;
} memlz_state;

size_t memlz_max_compressed_len(size_t input) {
    return 68 * input / 64 + 100; // todo, find real bound
}

size_t memlz_header_len(void) {
    return 18;
}

void memlz_reset(memlz_state* c) {
    memset(c->hash32, 0, sizeof(c->hash32));
    memset(c->hash64, 0, sizeof(c->hash64));
    c->total_input = 0;
    c->total_output = 0;
    c->mod = 0;
    c->wordlen = 8;
    c->cs4 = 0;
    c->cs8 = 0;
    c->incompressible = 0;
    c->reset = 'Y';
}

static uint16_t memlz_hash32(uint32_t v) {
    return (uint16_t)(((v * 2654435761ull) >> 16));
}

static uint16_t memlz_hash64(uint64_t v) {
    return (uint16_t)(((v * 11400714819323198485ull) >> 48));
}


#ifdef MEMLZ_SSE

MEMLZ_ALIGN_16 static const uint8_t memlz_shuf32[16][16] = {
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f},
    {0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
};
static const uint8_t memlz_sse_len[16] = { 16, 14, 14, 12, 14, 12, 12, 10, 14, 12, 12, 10, 12, 10, 10, 8 };

MEMLZ_SSE42 static MEMLZ_FORCE_INLINE __m128i memlz_hash32x4_sse4(__m128i v) {
    return _mm_srli_epi32(_mm_mullo_epi32(v, _mm_set1_epi32((int)2654435761u)), 16);
}

static MEMLZ_FORCE_INLINE uint32_t memlz_idx32(uint32_t v) {
    return (v * 2654435761u) >> 16;
}

MEMLZ_SSE42 static MEMLZ_FORCE_INLINE __m128i memlz_gather4_direct(const uint32_t* tbl, uint32_t h0, uint32_t h1, uint32_t h2, uint32_t h3) {
    __m128i t = _mm_cvtsi32_si128((int)tbl[h0]);
    t = _mm_insert_epi32(t, (int)tbl[h1], 1);
    t = _mm_insert_epi32(t, (int)tbl[h2], 2);
    t = _mm_insert_epi32(t, (int)tbl[h3], 3);
    return t;
}

MEMLZ_SSE42 static MEMLZ_FORCE_INLINE unsigned memlz_block32x2_sse4(uint32_t* tbl, const uint8_t* src, uint8_t* dst, size_t* adv) {
    const __m128i va = _mm_loadu_si128((const __m128i*)src);
    const __m128i vb = _mm_loadu_si128((const __m128i*)(src + 16));

    const __m128i ha = memlz_hash32x4_sse4(va);
    const __m128i hb = memlz_hash32x4_sse4(vb);

    const uint32_t a0 = memlz_r32(src + 0), a1 = memlz_r32(src + 4);
    const uint32_t a2 = memlz_r32(src + 8), a3 = memlz_r32(src + 12);
    const uint32_t b0 = memlz_r32(src + 16), b1 = memlz_r32(src + 20);
    const uint32_t b2 = memlz_r32(src + 24), b3 = memlz_r32(src + 28);

    const uint32_t ia0 = memlz_idx32(a0), ia1 = memlz_idx32(a1), ia2 = memlz_idx32(a2), ia3 = memlz_idx32(a3);
    const uint32_t ib0 = memlz_idx32(b0), ib1 = memlz_idx32(b1), ib2 = memlz_idx32(b2), ib3 = memlz_idx32(b3);

    const __m128i ta = memlz_gather4_direct(tbl, ia0, ia1, ia2, ia3);
    tbl[ia0] = a0; tbl[ia1] = a1; tbl[ia2] = a2; tbl[ia3] = a3;

    const __m128i tb = memlz_gather4_direct(tbl, ib0, ib1, ib2, ib3);
    tbl[ib0] = b0; tbl[ib1] = b1; tbl[ib2] = b2; tbl[ib3] = b3;

    const __m128i eqa = _mm_cmpeq_epi32(ta, va);
    const __m128i eqb = _mm_cmpeq_epi32(tb, vb);
    const unsigned ma = (unsigned)_mm_movemask_ps(_mm_castsi128_ps(eqa));
    const unsigned mb = (unsigned)_mm_movemask_ps(_mm_castsi128_ps(eqb));
    const unsigned m = (ma << 4) | mb;

    const __m128i oa = _mm_shuffle_epi8(_mm_blendv_epi8(va, ha, eqa), _mm_load_si128((const __m128i*)memlz_shuf32[ma]));
    const __m128i ob = _mm_shuffle_epi8(_mm_blendv_epi8(vb, hb, eqb), _mm_load_si128((const __m128i*)memlz_shuf32[mb]));

    const size_t la = memlz_sse_len[ma];
    _mm_storeu_si128((__m128i*)dst, oa);
    _mm_storeu_si128((__m128i*)(dst + la), ob);
    *adv = la + memlz_sse_len[mb];
    return m;
}

#define memlz_block64_sse4() { \
                size_t adv; \
                flags = (flags << 8) | memlz_block32x2_sse4(state->hash32, src, dst, &adv); \
                src += 32; dst += adv; }

#define MEMLZ_DO_SSE_4() \
    memlz_block64_sse4(); \
    memlz_block64_sse4();

MEMLZ_ALIGN_16 static const uint8_t memlz_shuf64x2[16][32] = {
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
};

static const uint16_t memlz_lenadv64qq[16] = {
    0x2010, 0x1A0A, 0x1A0A, 0x1404, 0x1A10, 0x140A, 0x140A, 0x0E04, 0x1A10, 0x140A, 0x140A, 0x0E04, 0x1410, 0x0E0A, 0x0E0A, 0x0804,
};

MEMLZ_SSE42 static MEMLZ_FORCE_INLINE __m128i memlz_load2x64(const uint64_t* a, const uint64_t* b) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)a), _mm_loadl_epi64((const __m128i*)b));
}

MEMLZ_SSE42 static MEMLZ_FORCE_INLINE unsigned memlz_block64_sse8_single(uint64_t* tbl, const uint8_t* src, uint8_t* dst, size_t* adv) {
    const uint64_t v0 = memlz_r64(src + 0), v1 = memlz_r64(src + 8);
    const uint64_t v2 = memlz_r64(src + 16), v3 = memlz_r64(src + 24);
    const uint32_t h0 = memlz_hash64(v0), h1 = memlz_hash64(v1);
    const uint32_t h2 = memlz_hash64(v2), h3 = memlz_hash64(v3);

    const __m128i va = _mm_loadu_si128((const __m128i*)src);
    const __m128i vb = _mm_loadu_si128((const __m128i*)(src + 16));

    const __m128i t0 = memlz_load2x64(&tbl[h0], &tbl[h1]);
    const __m128i t1 = memlz_load2x64(&tbl[h2], &tbl[h3]);

    const __m128i eqa = _mm_cmpeq_epi64(t0, va);
    const __m128i eqb = _mm_cmpeq_epi64(t1, vb);

    const unsigned m = (unsigned)_mm_movemask_ps(_mm_shuffle_ps(_mm_castsi128_ps(eqa), _mm_castsi128_ps(eqb), _MM_SHUFFLE(3, 1, 3, 1)));

    const __m128i ha = _mm_set_epi64x((long long)h1, (long long)h0);
    const __m128i hb = _mm_set_epi64x((long long)h3, (long long)h2);

    const __m128i oa = _mm_shuffle_epi8(_mm_blendv_epi8(va, ha, eqa), _mm_loadu_si128((const __m128i*) & memlz_shuf64x2[m][0]));
    const __m128i ob = _mm_shuffle_epi8(_mm_blendv_epi8(vb, hb, eqb), _mm_loadu_si128((const __m128i*) & memlz_shuf64x2[m][16]));

    const unsigned e = memlz_lenadv64qq[m];
    const size_t   la = e & 0xff;
    _mm_storeu_si128((__m128i*)dst, oa);
    _mm_storeu_si128((__m128i*)(dst + la), ob);
    *adv = e >> 8;

    tbl[h0] = v0; tbl[h1] = v1; tbl[h2] = v2; tbl[h3] = v3;
    return m;
}


#define memlz_block64_sse8() { \
                size_t adv; \
                flags = (flags << 4) | memlz_block64_sse8_single(state->hash64, src, dst, &adv); \
                src += 32; dst += adv; }

#define MEMLZ_DO_SSE_8() \
                memlz_block64_sse8(); \
                memlz_block64_sse8(); \
                memlz_block64_sse8(); \
                memlz_block64_sse8();

#endif // SSE

#define MEMLZ_STEP(tbl, typ, h_func, idx, shift, flg) \
            uint##typ##_t val_##idx = memlz_r##typ(src + idx * sizeof(uint##typ##_t)); \
            uint##typ##_t hash_##idx = h_func(val_##idx); \
            uint64_t hit_##idx = (tbl[hash_##idx] == val_##idx); \
            flg |= (hit_##idx << shift);

#define MEMLZ_COMMIT(tbl, typ, idx) \
            memlz_w##typ(dst, hit_##idx ? hash_##idx : val_##idx); \
            tbl[hash_##idx] = val_##idx; \
            dst += sizeof(uint##typ##_t) - (hit_##idx * (sizeof(uint##typ##_t) - 2));

#define MEMLZ_BLOCK(tbl, typ, h_func, flg) {\
            flg <<= 4; \
            MEMLZ_STEP(tbl, typ, h_func, 0, 0, flg) \
            MEMLZ_STEP(tbl, typ, h_func, 1, 1, flg) \
            MEMLZ_STEP(tbl, typ, h_func, 2, 2, flg) \
            MEMLZ_STEP(tbl, typ, h_func, 3, 3, flg) \
            MEMLZ_COMMIT(tbl, typ, 0) \
            MEMLZ_COMMIT(tbl, typ, 1) \
            MEMLZ_COMMIT(tbl, typ, 2) \
            MEMLZ_COMMIT(tbl, typ, 3) \
            src += 4 * sizeof(uint##typ##_t);}


MEMLZ_SSE42 size_t memlz_stream_compress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source, size_t len, memlz_state* state) {
    if (state->reset != 'Y') {
        return 0;
    }

    const size_t max = memlz_max_compressed_len(len) > len ? memlz_max_compressed_len(len) : len;
    const size_t header_len = memlz_fields * memlz_fit(max);
    size_t missing = len;
    const uint8_t* src = (const uint8_t*)source;
    uint8_t* dst = (uint8_t*)destination;
    uint64_t flags = 0;
    dst += header_len;
    uint16_t* flags_ptr = (uint16_t*)dst;

    for (;;) {
        state->mod++;
        if (state->mod == MEMLZ_PROBELEN / 128) {
            state->cs8 = (state->total_output + (dst - (uint8_t*)destination)) - state->cs8;
            state->cs4 = (state->total_output + (dst - (uint8_t*)destination));
            state->wordlen = 4;
        }
        else if (state->mod == 3 * MEMLZ_PROBELEN / 128) {
            state->cs4 = state->total_output + (dst - (uint8_t*)destination) - state->cs4;
            if (state->cs8 < state->cs4) {
                state->wordlen = 8;
            }
        }
        else if (state->mod == (MEMLZ_BLOCKLEN + MEMLZ_PROBELEN) / 128) {
            state->wordlen = 8;
            state->mod = 0;
            state->cs8 = state->total_output + (dst - (uint8_t*)destination);
            state->cs4 = 0;
        }

#ifdef MEMLZ_DO_RLE
        if (missing >= sizeof(uint64_t)) {
            size_t e = 1;
            uint64_t first_val = memlz_r64(src);
            while (e < missing / sizeof(uint64_t) && memlz_r64(src + e * sizeof(uint64_t)) == first_val) {
                e++;
            }
            e *= sizeof(uint64_t);
            if (e >= MEMLZ_MIN_RLE) {
                *dst++ = MEMLZ_RLE;
                size_t length = memlz_fit(e);
                memlz_write(dst, e, length);
                memlz_w64(dst + length, first_val);
                dst += sizeof(uint64_t) + length;
                missing -= e;
                src += e;
                continue;
            }
        }

#endif
        {
            *dst++ = state->wordlen == 8 ? MEMLZ_NORMAL_8 : MEMLZ_NORMAL_4;
            if (missing < 16 * state->wordlen) {
                break;
            }

            flags_ptr = (uint16_t*)dst;
            dst += 2;

            if (state->wordlen == 8) {
#ifdef MEMLZ_SSE
                MEMLZ_DO_SSE_8();
#else
                MEMLZ_UNROLL4(MEMLZ_BLOCK(state->hash64, 64, memlz_hash64, flags);)
#endif
            }
            else {
#ifdef MEMLZ_SSE
                MEMLZ_DO_SSE_4();
#else
                MEMLZ_UNROLL4(MEMLZ_BLOCK(state->hash32, 32, memlz_hash32, flags);)
#endif
            }

            memlz_w16(flags_ptr, (uint16_t)flags);
            missing -= 16 * state->wordlen;
        }


#ifdef MEMLZ_DO_INCOMPRESSIBLE
        {
            state->incompressible = flags ? 0 : state->incompressible + 1;
            if (state->incompressible > 0 && missing >= MEMLZ_INCOMPRESSIBLE_ADVANCE && state->incompressible % MEMLZ_INCOMPRESSIBLE_TRIGGER == 0) {
                size_t u = MEMLZ_INCOMPRESSIBLE_ADVANCE * state->incompressible;
                u = u > missing ? missing : u;
                u = u > 1024 ? 1024 : u;
                u = u & ~(sizeof(uint64_t) - 1);
                *dst++ = MEMLZ_UNCOMPRESSED;
                memlz_w16(dst, (uint16_t)u);
                memlz_write(dst, u, memlz_fit(u));
                dst += memlz_fit(u);
                for (size_t n = 0; n < u / sizeof(uint64_t); n++) {
                    memlz_w64(dst + n * sizeof(uint64_t), memlz_r64(src + n * sizeof(uint64_t)));
                }
                dst += u;
                src += u;
                missing -= u;
            }
        }
#endif
    }

    if (missing >= 4 * state->wordlen) {
        flags_ptr = (uint16_t*)dst;
        dst += 2;

        flags = 0;

        if (state->wordlen == 8) {
            while (missing >= 4 * 8) {
                MEMLZ_BLOCK(state->hash64, 64, memlz_hash64, flags);
                missing -= 4 * 8;
            }
        }
        else {
            while (missing >= 4 * 4) {
                MEMLZ_BLOCK(state->hash32, 32, memlz_hash32, flags);
                missing -= 4 * 4;
            }
        }

        memlz_w16(flags_ptr, (uint16_t)flags);
    }

    size_t tail_count = missing;
    memcpy(dst, src, tail_count);
    dst += tail_count;

    size_t compressed_len = (size_t)(dst - (uint8_t*)destination);
    if (compressed_len < memlz_header_len()) {
        memset(dst, 'M', memlz_header_len() - compressed_len);
        compressed_len = memlz_header_len();
    }

    memlz_write(destination, len, header_len / memlz_fields);
    memlz_write((uint8_t*)destination + header_len / memlz_fields, compressed_len, header_len / memlz_fields);

    state->total_input += len;
    state->total_output += compressed_len;

    return compressed_len;
}

size_t memlz_decompressed_len(const void* src) {
    return memlz_read(src);
}

size_t memlz_compressed_len(const void* src) {
    size_t header_field_len = memlz_bytes(src);
    return memlz_read((uint8_t*)src + header_field_len);
}

#define MEMLZ_R(p, l) do { if ((p) < r1 || (l) > (size_t)((r2) - (p))) return 0; } while (0)
#define MEMLZ_W(p, l) do { if ((p) < w1 || (l) > (size_t)((w2) - (p))) return 0; } while (0)

#ifdef MEMLZ_SSE

static const uint8_t memlz_len32[16] = { 16,14,14,12,14,12,12,10,14,12,12,10,12,10,10,8 };

MEMLZ_ALIGN_16 static const uint8_t memlz_dexp32[16][16] = {
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b, 0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x04,0x05, 0x06,0x07,0x08,0x09, 0x0a,0x0b,0x0c,0x0d},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x80,0x80, 0x06,0x07,0x08,0x09, 0x0a,0x0b,0x0c,0x0d},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x80,0x80, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x80,0x80, 0x0a,0x0b,0x0c,0x0d},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x04,0x05, 0x06,0x07,0x80,0x80, 0x08,0x09,0x0a,0x0b},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x80,0x80, 0x06,0x07,0x80,0x80, 0x08,0x09,0x0a,0x0b},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x80,0x80, 0x04,0x05,0x80,0x80, 0x06,0x07,0x08,0x09},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b, 0x0c,0x0d,0x80,0x80},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x04,0x05, 0x06,0x07,0x08,0x09, 0x0a,0x0b,0x80,0x80},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x80,0x80, 0x06,0x07,0x08,0x09, 0x0a,0x0b,0x80,0x80},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x80,0x80, 0x04,0x05,0x06,0x07, 0x08,0x09,0x80,0x80},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x80,0x80, 0x0a,0x0b,0x80,0x80},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x04,0x05, 0x06,0x07,0x80,0x80, 0x08,0x09,0x80,0x80},
    {0x00,0x01,0x02,0x03, 0x04,0x05,0x80,0x80, 0x06,0x07,0x80,0x80, 0x08,0x09,0x80,0x80},
    {0x00,0x01,0x80,0x80, 0x02,0x03,0x80,0x80, 0x04,0x05,0x80,0x80, 0x06,0x07,0x80,0x80},
};

static const uint8_t memlz_len64[4] = { 16, 10, 10, 4 };

MEMLZ_ALIGN_16 static const uint8_t memlz_dexp64[4][16] = {
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f},
    {0x00,0x01,0x80,0x80,0x80,0x80,0x80,0x80, 0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x80,0x80,0x80,0x80,0x80,0x80, 0x02,0x03,0x80,0x80,0x80,0x80,0x80,0x80},
};

MEMLZ_SSE42 static unsigned int memlz_decode_4_sse(uint32_t* tbl, const uint8_t* src, uint8_t* dst, unsigned m) {
    const __m128i raw = _mm_loadu_si128((const __m128i*)src);
    const __m128i field = _mm_shuffle_epi8(raw, _mm_load_si128((const __m128i*)memlz_dexp32[m]));

    const uint32_t f0 = (uint32_t)_mm_extract_epi32(field, 0);
    const uint32_t f1 = (uint32_t)_mm_extract_epi32(field, 1);
    const uint32_t f2 = (uint32_t)_mm_extract_epi32(field, 2);
    const uint32_t f3 = (uint32_t)_mm_extract_epi32(field, 3);

    const uint32_t g0 = tbl[f0 & 0xffffu];
    const uint32_t g1 = tbl[f1 & 0xffffu];
    const uint32_t g2 = tbl[f2 & 0xffffu];
    const uint32_t g3 = tbl[f3 & 0xffffu];

    const uint32_t w0 = (m & 1) ? g0 : f0;
    const uint32_t w1 = (m & 2) ? g1 : f1;
    const uint32_t w2 = (m & 4) ? g2 : f2;
    const uint32_t w3 = (m & 8) ? g3 : f3;

    memlz_w32(dst + 0, w0);
    memlz_w32(dst + 4, w1);
    memlz_w32(dst + 8, w2);
    memlz_w32(dst + 12, w3);

    tbl[memlz_hash32(w0)] = w0;
    tbl[memlz_hash32(w1)] = w1;
    tbl[memlz_hash32(w2)] = w2;
    tbl[memlz_hash32(w3)] = w3;

    return memlz_len32[m];
}

MEMLZ_SSE42 static unsigned memlz_decode_8_sse(uint64_t* tbl, const uint8_t* src, uint8_t* dst, unsigned m) {
    const unsigned ma = m & 3, mb = (m >> 2) & 3;

    const __m128i rawA = _mm_loadu_si128((const __m128i*)src);
    const __m128i fieldA = _mm_shuffle_epi8(rawA, _mm_load_si128((const __m128i*)memlz_dexp64[ma]));
    const size_t la = memlz_len64[ma];

    const __m128i rawB = _mm_loadu_si128((const __m128i*)(src + la));
    const __m128i fieldB = _mm_shuffle_epi8(rawB, _mm_load_si128((const __m128i*)memlz_dexp64[mb]));

    const uint64_t f0 = (uint64_t)_mm_cvtsi128_si64(fieldA);
    const uint64_t f1 = (uint64_t)_mm_extract_epi64(fieldA, 1);
    const uint64_t f2 = (uint64_t)_mm_cvtsi128_si64(fieldB);
    const uint64_t f3 = (uint64_t)_mm_extract_epi64(fieldB, 1);

    const uint64_t g0 = tbl[f0 & 0xffffu];
    const uint64_t g1 = tbl[f1 & 0xffffu];
    const uint64_t g2 = tbl[f2 & 0xffffu];
    const uint64_t g3 = tbl[f3 & 0xffffu];

    const uint64_t w0 = (ma & 1) ? g0 : f0;
    const uint64_t w1 = (ma & 2) ? g1 : f1;
    const uint64_t w2 = (mb & 1) ? g2 : f2;
    const uint64_t w3 = (mb & 2) ? g3 : f3;

    memlz_w64(dst + 0, w0);
    memlz_w64(dst + 8, w1);
    memlz_w64(dst + 16, w2);
    memlz_w64(dst + 24, w3);

    tbl[memlz_hash64(w0)] = w0;
    tbl[memlz_hash64(w1)] = w1;
    tbl[memlz_hash64(w2)] = w2;
    tbl[memlz_hash64(w3)] = w3;

    return (unsigned)(la + memlz_len64[mb]);
}

#define MEMLZ_DECODE_8_SSE() { \
                const unsigned m = (unsigned)((flags >> 12) & 0xF); \
                src += memlz_decode_8_sse(state->hash64, src, dst, m); \
                dst += 32; flags <<= 4; }

#define MEMLZ_DECODE_4_SSE4() { \
                const unsigned m = (unsigned)((flags >> 12) & 0xF); \
                src += memlz_decode_4_sse(state->hash32, src, dst, m); \
                dst += 16; flags <<= 4; }

#endif

#define MEMLZ_DECODE_STEP(safe, tbl, typ, idx) \
        const uint8_t* src_##idx = curr_src; \
        uintptr_t hit_##idx = ((flags >> (12 + idx)) & 1); \
        uint##typ##_t word_##idx; \
        if (safe) { \
            MEMLZ_R(src_##idx, hit_##idx ? 2 : sizeof(uint##typ##_t)); \
            if (hit_##idx) { \
                word_##idx = tbl[memlz_r16(src_##idx)]; \
            } else { \
                word_##idx = memlz_r##typ(src_##idx); \
            } \
        } else { \
            uint16_t hash_##idx = memlz_r16(src_##idx); \
            uint##typ##_t raw_##idx = memlz_r##typ(src_##idx); \
            word_##idx = hit_##idx ? tbl[hash_##idx] : raw_##idx; \
        } \
        curr_src += sizeof(uint##typ##_t) - (hit_##idx * (sizeof(uint##typ##_t) - 2));

#define MEMLZ_DECODE_COMMIT(tbl, typ, h_func, idx) \
        memlz_w##typ(dst + (idx * sizeof(uint##typ##_t)), word_##idx); \
        tbl[h_func(word_##idx)] = word_##idx;

#define MEMLZ_DECODE(safe, tbl, typ, h_func) \
        const uint8_t* curr_src = src; \
        MEMLZ_DECODE_STEP(safe, tbl, typ, 0) \
        MEMLZ_DECODE_STEP(safe, tbl, typ, 1) \
        MEMLZ_DECODE_STEP(safe, tbl, typ, 2) \
        MEMLZ_DECODE_STEP(safe, tbl, typ, 3) \
        MEMLZ_DECODE_COMMIT(tbl, typ, h_func, 0) \
        MEMLZ_DECODE_COMMIT(tbl, typ, h_func, 1) \
        MEMLZ_DECODE_COMMIT(tbl, typ, h_func, 2) \
        MEMLZ_DECODE_COMMIT(tbl, typ, h_func, 3) \
        src = curr_src; \
        dst += 4 * sizeof(uint##typ##_t); \
        flags <<= 4; 

size_t memlz_stream_decompress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source, memlz_state* MEMLZ_RESTRICT state) {
    if (state->reset != 'Y') {
        return 0;
    }

    const size_t decompressed_len = memlz_decompressed_len(source);
    const size_t compressed_len = memlz_compressed_len(source);

    if (compressed_len > memlz_max_compressed_len(decompressed_len)) {
        return 0;
    }

    // For memory safe decompression
    const uint8_t* r1 = (uint8_t*)source;
    const uint8_t* r2 = (uint8_t*)source + compressed_len;
    const uint8_t* w1 = (uint8_t*)destination;
    const uint8_t* w2 = (uint8_t*)destination + decompressed_len;

    size_t header_length = memlz_bytes(source) * memlz_fields;
    const uint8_t* MEMLZ_RESTRICT src = (const uint8_t*)source + header_length;
    uint8_t* MEMLZ_RESTRICT dst = (uint8_t*)destination;
    size_t missing = decompressed_len;
    size_t last_missing = 0;
    uint8_t blocktype = 0;
    size_t memlz_wordlen = 0;
    uint64_t flags = 0;

    for (;;) {
        // Prevent infinite loops or slow advance 
        const size_t min_advance = MEMLZ_MIN(64, MEMLZ_MIN(MEMLZ_INCOMPRESSIBLE_ADVANCE, MEMLZ_MIN_RLE));
        if (last_missing != 0 && missing > last_missing + min_advance) {
            return 0;
        }
        last_missing = missing;

        MEMLZ_R(src, 1);
        blocktype = *(uint8_t*)src++;

#ifdef MEMLZ_DO_INCOMPRESSIBLE
        if (blocktype == MEMLZ_UNCOMPRESSED) {
            MEMLZ_R(src, 1);
            size_t len = memlz_bytes(src);
            MEMLZ_R(src, len);
            size_t unc = memlz_read(src);
            src += len;
            MEMLZ_R(src, unc);
            MEMLZ_W(dst, unc);
            for (size_t n = 0; n < unc / sizeof(uint64_t); n++) {
                memlz_w64(dst + n * sizeof(uint64_t), memlz_r64(src + n * sizeof(uint64_t)));
            }
            src += unc;
            dst += unc;
            missing -= unc;
            continue;
        }
#endif

#ifdef MEMLZ_DO_RLE
        if (blocktype == MEMLZ_RLE) {
            MEMLZ_R(src, 1);
            size_t len = memlz_bytes(src);
            MEMLZ_R(src, len);
            uint64_t z = memlz_read(src);
            src += len;
            MEMLZ_R(src, sizeof(uint64_t));
            uint64_t v = memlz_r64(src);
            src += sizeof(uint64_t);
            MEMLZ_W(dst, z);
            for (uint64_t n = 0; n < z / sizeof(uint64_t); n++) {
                memlz_w64(dst + n * sizeof(uint64_t), v);
            }
            dst += z;
            missing -= z;
            continue;
        }
#endif

        if (blocktype == MEMLZ_NORMAL_8) {
            memlz_wordlen = 8;
        }
        else if (blocktype == MEMLZ_NORMAL_4) {
            memlz_wordlen = 4;
        }
        else {
            return 0;
        }

        if (missing < memlz_wordlen * 16) {
            break;
        }

        MEMLZ_R(src, 2);
        flags = memlz_r16(src);
        src += 2;

        if (src + 16 * sizeof(uint64_t) < r2) {
            if (blocktype == MEMLZ_NORMAL_8) {
                MEMLZ_W(dst, 16 * sizeof(uint64_t));
#ifdef MEMLZ_SSE
                MEMLZ_UNROLL4({ MEMLZ_DECODE_8_SSE(); })
#else
                MEMLZ_UNROLL4({ MEMLZ_DECODE(0, state->hash64, 64, memlz_hash64); })
#endif
                    missing -= 16 * sizeof(uint64_t);
            }
            else {
                MEMLZ_W(dst, 16 * sizeof(uint32_t));
#ifdef MEMLZ_SSE
                MEMLZ_UNROLL4({ MEMLZ_DECODE_4_SSE4(); })
#else
                MEMLZ_UNROLL4({ MEMLZ_DECODE(0, state->hash32, 32, memlz_hash32); })
#endif
                    missing -= 16 * sizeof(uint32_t);
            }
        }
        else {
            if (blocktype == MEMLZ_NORMAL_8) {
                MEMLZ_W(dst, 16 * sizeof(uint64_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE(1, state->hash64, 64, memlz_hash64); })
                    missing -= 16 * sizeof(uint64_t);
            }
            else {
                MEMLZ_W(dst, 16 * sizeof(uint32_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE(1, state->hash32, 32, memlz_hash32); })
                    missing -= 16 * sizeof(uint32_t);
            }
        }
    }

    if (missing >= 4U * (blocktype == MEMLZ_NORMAL_8 ? 8U : 4U)) {
        MEMLZ_R(src, 2);
        uint64_t raw_flags = memlz_r16(src);
        src += 2;

        if (blocktype == MEMLZ_NORMAL_8) {
            size_t tail_blocks = (missing / 8) / 4;
            flags = raw_flags << (16 - (tail_blocks * 4));

            while (missing >= 4 * 8) {
                MEMLZ_W(dst, 4 * sizeof(uint64_t));
                MEMLZ_DECODE(1, state->hash64, 64, memlz_hash64);
                missing -= 4 * 8;
            }
        }
        else {
            size_t tail_blocks = (missing / 4) / 4;
            flags = raw_flags << (16 - (tail_blocks * 4));

            while (missing >= 4 * 4) {
                MEMLZ_W(dst, 4 * sizeof(uint32_t));
                MEMLZ_DECODE(1, state->hash32, 32, memlz_hash32);
                missing -= 4 * 4;
            }
        }
    }

    size_t tail_count = missing;

    while (tail_count) {
        MEMLZ_R(src, 1);
        MEMLZ_W(dst, 1);
        *dst++ = *src++;
        tail_count--;
    }

    state->total_input += compressed_len;
    state->total_output += decompressed_len;
    return decompressed_len;
}

MEMLZ_UNUSED size_t memlz_decompress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source) {
    memlz_state* s = (memlz_state*)malloc(sizeof(memlz_state));
    if (!s) {
        return 0;
    }
    memlz_reset(s);
    size_t r = memlz_stream_decompress(destination, source, s);
    free(s);
    return r;
}

MEMLZ_UNUSED size_t memlz_compress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source, size_t len) {
    memlz_state* s = (memlz_state*)malloc(sizeof(memlz_state));
    if (!s) {
        return 0;
    }
    memlz_reset(s);
    size_t r = memlz_stream_compress(destination, source, len, s);
    free(s);
    return r;
}

#undef MEMLZ_UNROLL4
#undef MEMLZ_UNROLL16
#undef MEMLZ_ENCODE_WORD
#undef MEMLZ_DECODE_WORD
#undef MEMLZ_VOID
#undef MEMLZ_NORMAL_4
#undef MEMLZ_NORMAL_8
#undef MEMLZ_UNCOMPRESSED
#undef MEMLZ_RLE
#undef MEMLZ_WORDPROBE4096
#undef MEMLZ_BLOCKLEN
#undef MEMLZ_MIN
#undef MEMLZ_DO_RLE
#undef MEMLZ_DO_INCOMPRESSIBLE
#undef MEMLZ_INCOMPRESSIBLE
#undef MEMLZ_PROBELEN
#undef MEMLZ_MIN_RLE
#undef MEMLZ_RESTRICT
#undef MEMLZ_UNUSED

#endif // MEMLZ_IMPLEMENTATION

#endif // memlz_h

#ifdef MEMLZ_TEST

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    char in[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
        "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
        "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat";

    char* out = (char*)malloc(memlz_max_compressed_len(sizeof(in)));
    memlz_compress(out, in, sizeof(in));
    char* de = (char*)malloc(memlz_decompressed_len(out));
    size_t d = memlz_decompress(de, out);
    free(out);
    free(de);
    return d == sizeof(in) ? 0 : 1;
}

#endif
