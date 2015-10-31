/*
 * Centaurean SpookyHash
 *
 * Copyright (c) 2015, Guillaume Voirin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     1. Redistributions of source code must retain the above copyright notice, this
 *        list of conditions and the following disclaimer.
 *
 *     2. Redistributions in binary form must reproduce the above copyright notice,
 *        this list of conditions and the following disclaimer in the documentation
 *        and/or other materials provided with the distribution.
 *
 *     3. Neither the name of the copyright holder nor the names of its
 *        contributors may be used to endorse or promote products derived from
 *        this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * 25/01/15 12:21
 *
 * ----------
 * SpookyHash
 * ----------
 *
 * Author(s)
 * Bob Jenkins (http://burtleburtle.net/bob/hash/spooky.html)
 *
 * Description
 * Very fast non cryptographic hash
 */

#include "spookyhash.h"

SPOOKYHASH_FORCE_INLINE void spookyhash_short_end(uint64_t *SPOOKYHASH_RESTRICT h0, uint64_t *SPOOKYHASH_RESTRICT h1, uint64_t *SPOOKYHASH_RESTRICT h2, uint64_t *SPOOKYHASH_RESTRICT h3) {
    *h3 ^= *h2;
    *h2 = SPOOKYHASH_ROTATE(*h2, 15);
    *h3 += *h2;
    *h0 ^= *h3;
    *h3 = SPOOKYHASH_ROTATE(*h3, 52);
    *h0 += *h3;
    *h1 ^= *h0;
    *h0 = SPOOKYHASH_ROTATE(*h0, 26);
    *h1 += *h0;
    *h2 ^= *h1;
    *h1 = SPOOKYHASH_ROTATE(*h1, 51);
    *h2 += *h1;
    *h3 ^= *h2;
    *h2 = SPOOKYHASH_ROTATE(*h2, 28);
    *h3 += *h2;
    *h0 ^= *h3;
    *h3 = SPOOKYHASH_ROTATE(*h3, 9);
    *h0 += *h3;
    *h1 ^= *h0;
    *h0 = SPOOKYHASH_ROTATE(*h0, 47);
    *h1 += *h0;
    *h2 ^= *h1;
    *h1 = SPOOKYHASH_ROTATE(*h1, 54);
    *h2 += *h1;
    *h3 ^= *h2;
    *h2 = SPOOKYHASH_ROTATE(*h2, 32);
    *h3 += *h2;
    *h0 ^= *h3;
    *h3 = SPOOKYHASH_ROTATE(*h3, 25);
    *h0 += *h3;
    *h1 ^= *h0;
    *h0 = SPOOKYHASH_ROTATE(*h0, 63);
    *h1 += *h0;
}

SPOOKYHASH_FORCE_INLINE void spookyhash_short_mix(uint64_t *SPOOKYHASH_RESTRICT h0, uint64_t *SPOOKYHASH_RESTRICT h1, uint64_t *SPOOKYHASH_RESTRICT h2, uint64_t *SPOOKYHASH_RESTRICT h3) {
    *h2 = SPOOKYHASH_ROTATE(*h2, 50);
    *h2 += *h3;
    *h0 ^= *h2;
    *h3 = SPOOKYHASH_ROTATE(*h3, 52);
    *h3 += *h0;
    *h1 ^= *h3;
    *h0 = SPOOKYHASH_ROTATE(*h0, 30);
    *h0 += *h1;
    *h2 ^= *h0;
    *h1 = SPOOKYHASH_ROTATE(*h1, 41);
    *h1 += *h2;
    *h3 ^= *h1;
    *h2 = SPOOKYHASH_ROTATE(*h2, 54);
    *h2 += *h3;
    *h0 ^= *h2;
    *h3 = SPOOKYHASH_ROTATE(*h3, 48);
    *h3 += *h0;
    *h1 ^= *h3;
    *h0 = SPOOKYHASH_ROTATE(*h0, 38);
    *h0 += *h1;
    *h2 ^= *h0;
    *h1 = SPOOKYHASH_ROTATE(*h1, 37);
    *h1 += *h2;
    *h3 ^= *h1;
    *h2 = SPOOKYHASH_ROTATE(*h2, 62);
    *h2 += *h3;
    *h0 ^= *h2;
    *h3 = SPOOKYHASH_ROTATE(*h3, 34);
    *h3 += *h0;
    *h1 ^= *h3;
    *h0 = SPOOKYHASH_ROTATE(*h0, 5);
    *h0 += *h1;
    *h2 ^= *h0;
    *h1 = SPOOKYHASH_ROTATE(*h1, 36);
    *h1 += *h2;
    *h3 ^= *h1;
}

SPOOKYHASH_FORCE_INLINE void spookyhash_short(const void *SPOOKYHASH_RESTRICT message, size_t length, uint64_t *SPOOKYHASH_RESTRICT hash1, uint64_t *SPOOKYHASH_RESTRICT hash2) {
#if !SPOOKYHASH_ALLOW_UNALIGNED_READS
    uint64_t buffer[2 * SPOOKYHASH_VARIABLES];
#endif

    union {
        const uint8_t *p8;
        uint32_t *p32;
        uint64_t *p64;
        size_t i;
    } u;
    u.p8 = (const uint8_t *) message;

#if !SPOOKYHASH_ALLOW_UNALIGNED_READS
    if (u.i & 0x7) {
        SPOOKYHASH_MEMCPY(buffer, message, length);
        u.p64 = buffer;
    }
#endif

    size_t remainder = length % 32;
    uint64_t a = *hash1;
    uint64_t b = *hash2;
    uint64_t c = SPOOKYHASH_CONSTANT;
    uint64_t d = SPOOKYHASH_CONSTANT;

    if (length > 15) {
        const uint64_t *end = u.p64 + (length / 32) * 4;
        for (; u.p64 < end; u.p64 += 4) {
            c += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[0]);
            d += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[1]);
            spookyhash_short_mix(&a, &b, &c, &d);
            a += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[2]);
            b += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[3]);
        }

        if (remainder >= 16) {
            c += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[0]);
            d += SPOOKYHASH_LITTLE_ENDIAN_64(u.p64[1]);
            spookyhash_short_mix(&a, &b, &c, &d);
            u.p64 += 2;
            remainder -= 16;
        }
    }

    d += ((uint64_t) length) << 56;
    switch (remainder) {
        default:
            break;
        case 15:
            d += ((uint64_t) u.p8[14]) << 48;
        case 14:
            d += ((uint64_t) u.p8[13]) << 40;
        case 13:
            d += ((uint64_t) u.p8[12]) << 32;
        case 12:
            d += u.p32[2];
            c += u.p64[0];
            break;
        case 11:
            d += ((uint64_t) u.p8[10]) << 16;
        case 10:
            d += ((uint64_t) u.p8[9]) << 8;
        case 9:
            d += (uint64_t) u.p8[8];
        case 8:
            c += u.p64[0];
            break;
        case 7:
            c += ((uint64_t) u.p8[6]) << 48;
        case 6:
            c += ((uint64_t) u.p8[5]) << 40;
        case 5:
            c += ((uint64_t) u.p8[4]) << 32;
        case 4:
            c += u.p32[0];
            break;
        case 3:
            c += ((uint64_t) u.p8[2]) << 16;
        case 2:
            c += ((uint64_t) u.p8[1]) << 8;
        case 1:
            c += (uint64_t) u.p8[0];
            break;
        case 0:
            c += SPOOKYHASH_CONSTANT;
            d += SPOOKYHASH_CONSTANT;
    }
    spookyhash_short_end(&a, &b, &c, &d);
    *hash1 = a;
    *hash2 = b;
}

SPOOKYHASH_FORCE_INLINE void spookyhash_mix(const uint64_t *SPOOKYHASH_RESTRICT data, uint64_t *SPOOKYHASH_RESTRICT s0, uint64_t *SPOOKYHASH_RESTRICT s1, uint64_t *SPOOKYHASH_RESTRICT s2, uint64_t *SPOOKYHASH_RESTRICT s3, uint64_t *SPOOKYHASH_RESTRICT s4, uint64_t *SPOOKYHASH_RESTRICT s5, uint64_t *SPOOKYHASH_RESTRICT s6, uint64_t *SPOOKYHASH_RESTRICT s7, uint64_t *SPOOKYHASH_RESTRICT s8, uint64_t *SPOOKYHASH_RESTRICT s9, uint64_t *SPOOKYHASH_RESTRICT s10, uint64_t *SPOOKYHASH_RESTRICT s11) {
    *s0 += SPOOKYHASH_LITTLE_ENDIAN_64(data[0]);
    *s2 ^= *s10;
    *s11 ^= *s0;
    *s0 = SPOOKYHASH_ROTATE(*s0, 11);
    *s11 += *s1;
    *s1 += SPOOKYHASH_LITTLE_ENDIAN_64(data[1]);
    *s3 ^= *s11;
    *s0 ^= *s1;
    *s1 = SPOOKYHASH_ROTATE(*s1, 32);
    *s0 += *s2;
    *s2 += SPOOKYHASH_LITTLE_ENDIAN_64(data[2]);
    *s4 ^= *s0;
    *s1 ^= *s2;
    *s2 = SPOOKYHASH_ROTATE(*s2, 43);
    *s1 += *s3;
    *s3 += SPOOKYHASH_LITTLE_ENDIAN_64(data[3]);
    *s5 ^= *s1;
    *s2 ^= *s3;
    *s3 = SPOOKYHASH_ROTATE(*s3, 31);
    *s2 += *s4;
    *s4 += SPOOKYHASH_LITTLE_ENDIAN_64(data[4]);
    *s6 ^= *s2;
    *s3 ^= *s4;
    *s4 = SPOOKYHASH_ROTATE(*s4, 17);
    *s3 += *s5;
    *s5 += SPOOKYHASH_LITTLE_ENDIAN_64(data[5]);
    *s7 ^= *s3;
    *s4 ^= *s5;
    *s5 = SPOOKYHASH_ROTATE(*s5, 28);
    *s4 += *s6;
    *s6 += SPOOKYHASH_LITTLE_ENDIAN_64(data[6]);
    *s8 ^= *s4;
    *s5 ^= *s6;
    *s6 = SPOOKYHASH_ROTATE(*s6, 39);
    *s5 += *s7;
    *s7 += SPOOKYHASH_LITTLE_ENDIAN_64(data[7]);
    *s9 ^= *s5;
    *s6 ^= *s7;
    *s7 = SPOOKYHASH_ROTATE(*s7, 57);
    *s6 += *s8;
    *s8 += SPOOKYHASH_LITTLE_ENDIAN_64(data[8]);
    *s10 ^= *s6;
    *s7 ^= *s8;
    *s8 = SPOOKYHASH_ROTATE(*s8, 55);
    *s7 += *s9;
    *s9 += SPOOKYHASH_LITTLE_ENDIAN_64(data[9]);
    *s11 ^= *s7;
    *s8 ^= *s9;
    *s9 = SPOOKYHASH_ROTATE(*s9, 54);
    *s8 += *s10;
    *s10 += SPOOKYHASH_LITTLE_ENDIAN_64(data[10]);
    *s0 ^= *s8;
    *s9 ^= *s10;
    *s10 = SPOOKYHASH_ROTATE(*s10, 22);
    *s9 += *s11;
    *s11 += SPOOKYHASH_LITTLE_ENDIAN_64(data[11]);
    *s1 ^= *s9;
    *s10 ^= *s11;
    *s11 = SPOOKYHASH_ROTATE(*s11, 46);
    *s10 += *s0;
}

SPOOKYHASH_FORCE_INLINE void spookyhash_end_partial(uint64_t *SPOOKYHASH_RESTRICT h0, uint64_t *SPOOKYHASH_RESTRICT h1, uint64_t *SPOOKYHASH_RESTRICT h2, uint64_t *SPOOKYHASH_RESTRICT h3, uint64_t *SPOOKYHASH_RESTRICT h4, uint64_t *SPOOKYHASH_RESTRICT h5, uint64_t *SPOOKYHASH_RESTRICT h6, uint64_t *SPOOKYHASH_RESTRICT h7, uint64_t *SPOOKYHASH_RESTRICT h8, uint64_t *SPOOKYHASH_RESTRICT h9, uint64_t *SPOOKYHASH_RESTRICT h10, uint64_t *SPOOKYHASH_RESTRICT h11) {
    *h11 += *h1;
    *h2 ^= *h11;
    *h1 = SPOOKYHASH_ROTATE(*h1, 44);
    *h0 += *h2;
    *h3 ^= *h0;
    *h2 = SPOOKYHASH_ROTATE(*h2, 15);
    *h1 += *h3;
    *h4 ^= *h1;
    *h3 = SPOOKYHASH_ROTATE(*h3, 34);
    *h2 += *h4;
    *h5 ^= *h2;
    *h4 = SPOOKYHASH_ROTATE(*h4, 21);
    *h3 += *h5;
    *h6 ^= *h3;
    *h5 = SPOOKYHASH_ROTATE(*h5, 38);
    *h4 += *h6;
    *h7 ^= *h4;
    *h6 = SPOOKYHASH_ROTATE(*h6, 33);
    *h5 += *h7;
    *h8 ^= *h5;
    *h7 = SPOOKYHASH_ROTATE(*h7, 10);
    *h6 += *h8;
    *h9 ^= *h6;
    *h8 = SPOOKYHASH_ROTATE(*h8, 13);
    *h7 += *h9;
    *h10 ^= *h7;
    *h9 = SPOOKYHASH_ROTATE(*h9, 38);
    *h8 += *h10;
    *h11 ^= *h8;
    *h10 = SPOOKYHASH_ROTATE(*h10, 53);
    *h9 += *h11;
    *h0 ^= *h9;
    *h11 = SPOOKYHASH_ROTATE(*h11, 42);
    *h10 += *h0;
    *h1 ^= *h10;
    *h0 = SPOOKYHASH_ROTATE(*h0, 54);
}

SPOOKYHASH_FORCE_INLINE void spookyhash_end(const uint64_t *SPOOKYHASH_RESTRICT data, uint64_t *SPOOKYHASH_RESTRICT h0, uint64_t *SPOOKYHASH_RESTRICT h1, uint64_t *SPOOKYHASH_RESTRICT h2, uint64_t *SPOOKYHASH_RESTRICT h3, uint64_t *SPOOKYHASH_RESTRICT h4, uint64_t *SPOOKYHASH_RESTRICT h5, uint64_t *SPOOKYHASH_RESTRICT h6, uint64_t *SPOOKYHASH_RESTRICT h7, uint64_t *SPOOKYHASH_RESTRICT h8, uint64_t *SPOOKYHASH_RESTRICT h9, uint64_t *SPOOKYHASH_RESTRICT h10, uint64_t *SPOOKYHASH_RESTRICT h11) {
    *h0 += SPOOKYHASH_LITTLE_ENDIAN_64(data[0]);
    *h1 += SPOOKYHASH_LITTLE_ENDIAN_64(data[1]);
    *h2 += SPOOKYHASH_LITTLE_ENDIAN_64(data[2]);
    *h3 += SPOOKYHASH_LITTLE_ENDIAN_64(data[3]);
    *h4 += SPOOKYHASH_LITTLE_ENDIAN_64(data[4]);
    *h5 += SPOOKYHASH_LITTLE_ENDIAN_64(data[5]);
    *h6 += SPOOKYHASH_LITTLE_ENDIAN_64(data[6]);
    *h7 += SPOOKYHASH_LITTLE_ENDIAN_64(data[7]);
    *h8 += SPOOKYHASH_LITTLE_ENDIAN_64(data[8]);
    *h9 += SPOOKYHASH_LITTLE_ENDIAN_64(data[9]);
    *h10 += SPOOKYHASH_LITTLE_ENDIAN_64(data[10]);
    *h11 += SPOOKYHASH_LITTLE_ENDIAN_64(data[11]);
    spookyhash_end_partial(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11);
    spookyhash_end_partial(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11);
    spookyhash_end_partial(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11);
}

SPOOKYHASH_WINDOWS_EXPORT SPOOKYHASH_FORCE_INLINE void spookyhash_128(const void *SPOOKYHASH_RESTRICT message, size_t length, uint64_t *SPOOKYHASH_RESTRICT hash1, uint64_t *SPOOKYHASH_RESTRICT hash2) {
    if (length < SPOOKYHASH_BUFFER_SIZE) {
        spookyhash_short(message, length, hash1, hash2);
        return;
    }

    uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;
    uint64_t buf[SPOOKYHASH_VARIABLES];
    uint64_t *end;
    union {
        const uint8_t *p8;
        uint64_t *p64;
        size_t i;
    } u;
    size_t remainder;

    h0 = h3 = h6 = h9 = *hash1;
    h1 = h4 = h7 = h10 = *hash2;
    h2 = h5 = h8 = h11 = SPOOKYHASH_CONSTANT;

    u.p8 = (const uint8_t *) message;
    end = u.p64 + (length / SPOOKYHASH_BLOCK_SIZE) * SPOOKYHASH_VARIABLES;

    if (SPOOKYHASH_ALLOW_UNALIGNED_READS || ((u.i & 0x7) == 0)) {
        while (u.p64 < end) {
            spookyhash_mix(u.p64, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
            u.p64 += SPOOKYHASH_VARIABLES;
        }
    } else {
        while (u.p64 < end) {
            SPOOKYHASH_MEMCPY(buf, u.p64, SPOOKYHASH_BLOCK_SIZE);
            spookyhash_mix(buf, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
            u.p64 += SPOOKYHASH_VARIABLES;
        }
    }

    remainder = (length - ((const uint8_t *) end - (const uint8_t *) message));
    SPOOKYHASH_MEMCPY(buf, end, remainder);
    SPOOKYHASH_MEMSET(((uint8_t *) buf) + remainder, 0, SPOOKYHASH_BLOCK_SIZE - remainder);
    ((uint8_t *) buf)[SPOOKYHASH_BLOCK_SIZE - 1] = (uint8_t) remainder;

    spookyhash_end(buf, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
    *hash1 = h0;
    *hash2 = h1;
}

SPOOKYHASH_WINDOWS_EXPORT SPOOKYHASH_FORCE_INLINE uint64_t spookyhash_64(const void *message, size_t length, uint64_t seed) {
    uint64_t hash1 = seed;
    spookyhash_128(message, length, &hash1, &seed);
    return hash1;
}

SPOOKYHASH_WINDOWS_EXPORT SPOOKYHASH_FORCE_INLINE uint32_t spookyhash_32(const void *message, size_t length, uint32_t seed) {
    uint64_t hash1 = seed, hash2 = seed;
    spookyhash_128(message, length, &hash1, &hash2);
    return (uint32_t) hash1;
}

SPOOKYHASH_WINDOWS_EXPORT SPOOKYHASH_FORCE_INLINE void spookyhash_update(spookyhash_context *SPOOKYHASH_RESTRICT context, const void *SPOOKYHASH_RESTRICT message, size_t length) {
    uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;
    size_t newLength = length + context->m_remainder;
    uint8_t remainder;
    union {
        const uint8_t *p8;
        uint64_t *p64;
        size_t i;
    } u;
    const uint64_t *end;

    if (newLength < SPOOKYHASH_BUFFER_SIZE) {
        SPOOKYHASH_MEMCPY(&((uint8_t *) context->m_data)[context->m_remainder], message, length);
        context->m_length = length + context->m_length;
        context->m_remainder = (uint8_t) newLength;
        return;
    }

    if (context->m_length < SPOOKYHASH_BUFFER_SIZE) {
        h0 = h3 = h6 = h9 = context->m_state[0];
        h1 = h4 = h7 = h10 = context->m_state[1];
        h2 = h5 = h8 = h11 = SPOOKYHASH_CONSTANT;
    }
    else {
        h0 = context->m_state[0];
        h1 = context->m_state[1];
        h2 = context->m_state[2];
        h3 = context->m_state[3];
        h4 = context->m_state[4];
        h5 = context->m_state[5];
        h6 = context->m_state[6];
        h7 = context->m_state[7];
        h8 = context->m_state[8];
        h9 = context->m_state[9];
        h10 = context->m_state[10];
        h11 = context->m_state[11];
    }
    context->m_length = length + context->m_length;

    if (context->m_remainder) {
        uint8_t prefix = (uint8_t) (SPOOKYHASH_BUFFER_SIZE - context->m_remainder);
        SPOOKYHASH_MEMCPY(&(((uint8_t *) context->m_data)[context->m_remainder]), message, prefix);
        u.p64 = context->m_data;
        spookyhash_mix(u.p64, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
        spookyhash_mix(&u.p64[SPOOKYHASH_VARIABLES], &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
        u.p8 = ((const uint8_t *) message) + prefix;
        length -= prefix;
    } else {
        u.p8 = (const uint8_t *) message;
    }

    end = u.p64 + (length / SPOOKYHASH_BLOCK_SIZE) * SPOOKYHASH_VARIABLES;
    remainder = (uint8_t) (length - ((const uint8_t *) end - u.p8));
    if (SPOOKYHASH_ALLOW_UNALIGNED_READS || (u.i & 0x7) == 0) {
        while (u.p64 < end) {
            spookyhash_mix(u.p64, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
            u.p64 += SPOOKYHASH_VARIABLES;
        }
    }
    else {
        while (u.p64 < end) {
            SPOOKYHASH_MEMCPY(context->m_data, u.p8, SPOOKYHASH_BLOCK_SIZE);
            spookyhash_mix(context->m_data, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
            u.p64 += SPOOKYHASH_VARIABLES;
        }
    }

    context->m_remainder = remainder;
    SPOOKYHASH_MEMCPY(context->m_data, end, remainder);

    context->m_state[0] = h0;
    context->m_state[1] = h1;
    context->m_state[2] = h2;
    context->m_state[3] = h3;
    context->m_state[4] = h4;
    context->m_state[5] = h5;
    context->m_state[6] = h6;
    context->m_state[7] = h7;
    context->m_state[8] = h8;
    context->m_state[9] = h9;
    context->m_state[10] = h10;
    context->m_state[11] = h11;
}

SPOOKYHASH_WINDOWS_EXPORT SPOOKYHASH_FORCE_INLINE void spookyhash_final(spookyhash_context *SPOOKYHASH_RESTRICT context, uint64_t *SPOOKYHASH_RESTRICT hash1, uint64_t *SPOOKYHASH_RESTRICT hash2) {
    if (context->m_length < SPOOKYHASH_BUFFER_SIZE) {
        *hash1 = context->m_state[0];
        *hash2 = context->m_state[1];
        spookyhash_short(context->m_data, context->m_length, hash1, hash2);
        return;
    }

    const uint64_t *data = (const uint64_t *) context->m_data;
    uint8_t remainder = context->m_remainder;

    uint64_t h0 = context->m_state[0];
    uint64_t h1 = context->m_state[1];
    uint64_t h2 = context->m_state[2];
    uint64_t h3 = context->m_state[3];
    uint64_t h4 = context->m_state[4];
    uint64_t h5 = context->m_state[5];
    uint64_t h6 = context->m_state[6];
    uint64_t h7 = context->m_state[7];
    uint64_t h8 = context->m_state[8];
    uint64_t h9 = context->m_state[9];
    uint64_t h10 = context->m_state[10];
    uint64_t h11 = context->m_state[11];

    if (remainder >= SPOOKYHASH_BLOCK_SIZE) {
        spookyhash_mix(data, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);
        data += SPOOKYHASH_VARIABLES;
        remainder -= SPOOKYHASH_BLOCK_SIZE;
    }

    SPOOKYHASH_MEMSET(&((uint8_t *) data)[remainder], 0, (SPOOKYHASH_BLOCK_SIZE - remainder));

    ((uint8_t *) data)[SPOOKYHASH_BLOCK_SIZE - 1] = remainder;

    spookyhash_end(data, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9, &h10, &h11);

    *hash1 = h0;
    *hash2 = h1;
}
