// SPDX-License-Identifier: MIT
//
// memlz 0.3 beta - extremely fast header-only compression library for C and C++ on x64/x86
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
size_t memlz_header_len();

///  Call this before the first call to memlz_compress() or memlz_decompress()
void memlz_reset(memlz_state* c);

#ifdef __cplusplus
} // extern C
#endif

// The rest of this header file is internals
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MEMLZ_IMPLEMENTATION

#define MEMLZ_DO_RLE
#define MEMLZ_DO_INCOMPRESSIBLE
#define MEMLZ_INCOMPRESSIBLE_TRIGGER (4)
#define MEMLZ_INCOMPRESSIBLE_ADVANCE (16 * MEMLZ_INCOMPRESSIBLE_TRIGGER)
#define MEMLZ_PROBELEN (2 * 1024)
#define MEMLZ_BLOCKLEN (128 * 1024)
#define MEMLZ_RLE 'D'
#define MEMLZ_MIN_RLE (4 * sizeof(uint64_t))
#define MEMLZ_RESTRICT __restrict
#define MEMLZ_UNROLL4(op) op; op; op; op
#define MEMLZ_UNROLL16(op) op; op; op; op; op; op; op; op; op; op; op; op; op; op; op; op;
#define MEMLZ_NORMAL32 'A'
#define MEMLZ_NORMAL64 'B'
#define MEMLZ_UNCOMPRESSED 'C'
#define MEMLZ_MIN(X, Y) ((X) < (Y) ? (X) : (Y))

#ifdef _WIN32
#define MEMLZ_UNUSED
#else
#define MEMLZ_UNUSED __attribute__((unused))
#endif

static const size_t memlz_fields = 2;
static const size_t memlz_words_per_round = 16;

static uint16_t memlz_hash32(uint32_t v) {
    return (uint16_t)(((v * 2654435761ull) >> 16));
}

static uint16_t memlz_hash64(uint64_t v) {
    return (uint16_t)(((v * 11400714819323198485ull) >> 48));
}


static uint64_t memlz_read(const void* src) {
    uint8_t* s = (uint8_t*)src;
    size_t bytes = ((size_t)*s) >> 6;
    return bytes == 0 ? *s & 0b00111111
        : bytes == 1 ? *(uint16_t*)(s + 1)
        : bytes == 2 ? *(uint32_t*)(s + 1)
        : *(uint64_t*)(s + 1);
}

static size_t memlz_bytes(const void* src) {
    uint8_t* s = (uint8_t*)src;
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
        assert(value < 0xffff);
        *d = 0b01000000;
        *(uint16_t*)(d + 1) = (uint16_t)value;
    }
    else if (bytes == 5) {
        assert(value < 0xffffffff);
        *d = 0b10000000;
        *(uint32_t*)(d + 1) = (uint32_t)value;
    }
    else if (bytes == 9) {
        *d = 0b11000000;
        *(uint64_t*)(d + 1) = (uint64_t)value;
    }
}

static uint64_t memlz_fit(uint64_t value) {
    return value < 64ULL ? 1ULL : value <= 0xffffULL ? 3ULL : value <= 0xffffffffULL ? 5ULL : 9ULL;
}

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

size_t memlz_header_len() {
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

size_t memlz_stream_compress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source, size_t len, memlz_state* state) {
    if (state->reset != 'Y') {
        return 0;
    }

    const size_t max = memlz_max_compressed_len(len) > len ? memlz_max_compressed_len(len) : len;
    const size_t header_len = memlz_fields * memlz_fit(max);
    size_t missing = len;
    const uint8_t* src = (const uint8_t*)source;
    uint8_t* dst = (uint8_t*)destination;
    uint16_t flags = 0;
    dst += header_len;

    for (;;) {
        // Compress 8-byte words, then 4-byte words and compare ratios and select best.
        // TODO: Occurences of RLE or incompressible blocks wil disturb the result.
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
        {
            size_t e = 1;
            while (e < missing / sizeof(uint64_t) && ((uint64_t*)src)[e] == *(uint64_t*)src) {
                e++;
            }
            e *= sizeof(uint64_t);
            if (e >= MEMLZ_MIN_RLE) {
                *dst++ = MEMLZ_RLE;
                size_t length = memlz_fit(e);
                memlz_write(dst, e, length);
                *(uint64_t*)(dst + length) = *(uint64_t*)src;
                dst += sizeof(uint64_t) + length;
                missing -= e;
                src += e;
                continue;
            }
        }
#endif
        {
            *dst++ = state->wordlen == 8 ? MEMLZ_NORMAL64 : MEMLZ_NORMAL32;
            if (missing < 16 * state->wordlen) {
                break;
            }

            uint16_t* flags_ptr = (uint16_t*)dst;
            dst += 2;

#define MEMLZ_STEP(tbl, typ, h_func, idx, shift) \
            uint##typ##_t val_##idx = *(const uint##typ##_t*)(src + idx * sizeof(uint##typ##_t)); \
            uint##typ##_t hash_##idx = h_func(val_##idx); \
            uint64_t hit_##idx = (tbl[hash_##idx] == val_##idx); \
            local_flags |= (hit_##idx << shift);

#define MEMLZ_COMMIT(tbl, typ, idx) \
            *(uint##typ##_t*)dst = hit_##idx ? hash_##idx : val_##idx; \
            tbl[hash_##idx] = val_##idx; \
            dst += sizeof(uint##typ##_t) - (hit_##idx * (sizeof(uint##typ##_t) - 2));

#define MEMLZ_BLOCK(tbl, typ, h_func) \
            local_flags <<= 4; \
            MEMLZ_STEP(tbl, typ, h_func, 0, 0) \
            MEMLZ_STEP(tbl, typ, h_func, 1, 1) \
            MEMLZ_STEP(tbl, typ, h_func, 2, 2) \
            MEMLZ_STEP(tbl, typ, h_func, 3, 3) \
            MEMLZ_COMMIT(tbl, typ, 0) \
            MEMLZ_COMMIT(tbl, typ, 1) \
            MEMLZ_COMMIT(tbl, typ, 2) \
            MEMLZ_COMMIT(tbl, typ, 3) \
            src += 4 * sizeof(uint##typ##_t); 

            uint16_t local_flags = flags;
            if (state->wordlen == 8) {
                MEMLZ_UNROLL4({
                    MEMLZ_BLOCK(state->hash64, 64, memlz_hash64);
                })
            }
            else {
                MEMLZ_UNROLL4({
                    MEMLZ_BLOCK(state->hash32, 32, memlz_hash32);
                })
            }
            flags = local_flags;
            *flags_ptr = flags;
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
                *(uint16_t*)dst = (uint16_t)u;
                memlz_write(dst, u, memlz_fit(u));
                dst += memlz_fit(u);
                for (size_t n = 0; n < u / sizeof(uint64_t); n++) {
                    ((uint64_t*)dst)[n] = ((uint64_t*)src)[n];
                }
                dst += u;
                src += u;
                missing -= u;
            }
        }
#endif
    }

    if (missing >= 4 * state->wordlen) {
        uint16_t local_flags = flags;
        uint16_t* tail_flags_ptr = (uint16_t*)dst;
        dst += 2;
        local_flags = 0;

        if (state->wordlen == 8) {
            while (missing >= 4 * 8) {
                MEMLZ_BLOCK(state->hash64, 64, memlz_hash64);
                missing -= 4 * 8;
            }
        }
        else {
            while (missing >= 4 * 4) {
                MEMLZ_BLOCK(state->hash32, 32, memlz_hash32);
                missing -= 4 * 4;
            }
        }

        *tail_flags_ptr = local_flags;
        flags = local_flags;
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

size_t memlz_stream_decompress(void* MEMLZ_RESTRICT destination, const void* MEMLZ_RESTRICT source, memlz_state* state) {
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
    const uint8_t* src = (const uint8_t*)source + header_length;
    uint8_t* dst = (uint8_t*)destination;
    size_t missing = decompressed_len;
    size_t last_missing = 0;
    uint8_t blocktype = 0;
    size_t memlz_wordlen = 0;
    uint16_t flags = 0;

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
                ((uint64_t*)dst)[n] = ((uint64_t*)src)[n];
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
            uint64_t v = *((uint64_t*)src);
            src += sizeof(uint64_t);
            MEMLZ_W(dst, z);
            for (uint64_t n = 0; n < z / sizeof(uint64_t); n++) {
                ((uint64_t*)dst)[n] = v;
            }
            dst += z;
            missing -= z;
            continue;
        }
#endif

        if (blocktype == MEMLZ_NORMAL64) {
            memlz_wordlen = 8;
        }
        else if (blocktype == MEMLZ_NORMAL32) {
            memlz_wordlen = 4;
        }
        else {
            return 0;
        }

        if (missing < memlz_wordlen * 16) {
            break;
        }

        MEMLZ_R(src, 2);
        flags = *(uint16_t*)src;
        src += 2;

#define MEMLZ_DECODE_STEP(safe, tbl, typ, idx) \
        const uint8_t* src_##idx = curr_src; \
        uintptr_t hit_##idx = ((flags >> (12 + idx)) & 1); \
        typ word_##idx; \
        if (safe) { \
            MEMLZ_R(src_##idx, hit_##idx ? 2 : sizeof(typ)); \
            if (hit_##idx) { \
                word_##idx = tbl[*(const uint16_t*)src_##idx]; \
            } else { \
                word_##idx = *(const typ*)src_##idx; \
            } \
        } else { \
            uint16_t hash_##idx = *(const uint16_t*)src_##idx; \
            typ raw_##idx = *(const typ*)src_##idx; \
            word_##idx = hit_##idx ? tbl[hash_##idx] : raw_##idx; \
        } \
        curr_src += sizeof(typ) - (hit_##idx * (sizeof(typ) - 2));

#define MEMLZ_DECODE_COMMIT(tbl, typ, h_func, idx) \
        *(typ*)(dst + (idx * sizeof(typ))) = word_##idx; \
        tbl[h_func(word_##idx)] = word_##idx;

#define MEMLZ_DECODE_BLOCK(safe, tbl, typ, h_func) \
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
        dst += 4 * sizeof(typ); \
        flags <<= 4; 

        if (src + 16 * sizeof(uint64_t) < r2) {
            if (blocktype == MEMLZ_NORMAL64) {
                MEMLZ_W(dst, 16 * sizeof(uint64_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE_BLOCK(0, state->hash64, uint64_t, memlz_hash64); })
                    missing -= 16 * sizeof(uint64_t);
            }
            else {
                MEMLZ_W(dst, 16 * sizeof(uint32_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE_BLOCK(0, state->hash32, uint32_t, memlz_hash32); })
                    missing -= 16 * sizeof(uint32_t);
            }
        }
        else {
            if (blocktype == MEMLZ_NORMAL64) {
                MEMLZ_W(dst, 16 * sizeof(uint64_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE_BLOCK(1, state->hash64, uint64_t, memlz_hash64); })
                    missing -= 16 * sizeof(uint64_t);
            }
            else {
                MEMLZ_W(dst, 16 * sizeof(uint32_t));
                MEMLZ_UNROLL4({ MEMLZ_DECODE_BLOCK(1, state->hash32, uint32_t, memlz_hash32); })
                    missing -= 16 * sizeof(uint32_t);
            }
        }
    }

    if (missing >= 4 * (blocktype == MEMLZ_NORMAL64 ? 8 : 4)) {
        MEMLZ_R(src, 2);
        uint16_t raw_flags = *(const uint16_t*)src;
        src += 2;

        if (blocktype == MEMLZ_NORMAL64) {
            size_t tail_blocks = (missing / 8) / 4;
            flags = raw_flags << (16 - (tail_blocks * 4));

            while (missing >= 4 * 8) {
                MEMLZ_W(dst, 4 * sizeof(uint64_t));
                MEMLZ_DECODE_BLOCK(1, state->hash64, uint64_t, memlz_hash64);
                missing -= 4 * 8;
            }
        }
        else {
            size_t tail_blocks = (missing / 4) / 4;
            flags = raw_flags << (16 - (tail_blocks * 4));

            while (missing >= 4 * 4) {
                MEMLZ_W(dst, 4 * sizeof(uint32_t));
                MEMLZ_DECODE_BLOCK(1, state->hash32, uint32_t, memlz_hash32);
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
#undef MEMLZ_NORMAL32
#undef MEMLZ_NORMAL64
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
