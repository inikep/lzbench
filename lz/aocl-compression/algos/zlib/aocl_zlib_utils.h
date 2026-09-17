/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef AOCL_ZLIB_UTILS_H
#define AOCL_ZLIB_UTILS_H

#define DQUICK_LIT_MAX_BITS 9
#define DQUICK_OVERHEAD(x) ((x * (DQUICK_LIT_MAX_BITS - 8) + 7) >> 3)
/* deflate_quick worst-case overhead: 9 bits per literal, round up to next byte (+7) */

#define FIXED_HUFFFMAN_COMPRESSED_SIZE(x) (x + DQUICK_OVERHEAD(x))
#define STORED_ZLIB_COMPRESSED_SIZE(x) (x + (x >> 12) + (x >> 14) + (x >> 25) + 13)

extern void aocl_zlib_set_enable_dquick(int val);
extern int aocl_zlib_get_enable_dquick(void);

#ifdef AOCL_ENABLE_THREADS

#define CALCULATE_CHECKSUM(source, len, wrap) \
    (wrap == 1) ? adler32_x86(1L, (const Bytef *)source, len) : \
    (wrap == 2) ? crc32(0L, (const Bytef *)source, len) : \
    0

#define UPDATE_CHECKSUM(checksum1, checksum2, len2, wrap) \
    (wrap == 1) ? adler32_combine(checksum1, checksum2, len2) : \
    (wrap == 2) ? crc32_combine(checksum1, checksum2, len2) : \
    0

extern int insert_Header_generic(Bytef *dest, int level, const int wrap);
extern int insert_Trailer_generic(Bytef *dest, AOCL_UINT32 checksum, uLong sourceLen, const int wrap);
#endif /* AOCL_ENABLE_THREADS */

#endif /* AOCL_ZLIB_UTILS_H */
