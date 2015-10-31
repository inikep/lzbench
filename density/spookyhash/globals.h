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
 * 25/01/15 12:23
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

#ifndef SPOOKYHASH_GLOBALS_H
#define SPOOKYHASH_GLOBALS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#if defined(_WIN64) || defined(_WIN32)
#define SPOOKYHASH_WINDOWS_EXPORT __declspec(dllexport)
#define SPOOKYHASH_RESTRICT     __restrict
#else
#define SPOOKYHASH_WINDOWS_EXPORT
#define SPOOKYHASH_RESTRICT     restrict
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SPOOKYHASH_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define SPOOKYHASH_FORCE_INLINE __forceinline
#else
#warning Impossible to force functions inlining. Expect performance issues.
#define SPOOKYHASH_FORCE_INLINE
#define SPOOKYHASH_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SPOOKYHASH_MEMCPY   __builtin_memcpy
#define SPOOKYHASH_MEMSET   __builtin_memset
#else
#include <string.h>
#define SPOOKYHASH_MEMCPY   memcpy
#define SPOOKYHASH_MEMSET   memset
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define SPOOKYHASH_LITTLE_ENDIAN_64(b)   ((uint64_t)b)
#define SPOOKYHASH_LITTLE_ENDIAN_32(b)   ((uint32_t)b)
#define SPOOKYHASH_LITTLE_ENDIAN_16(b)   ((uint16_t)b)
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#if __GNUC__ * 100 + __GNUC_MINOR__ >= 403 || defined(__clang__)
#define SPOOKYHASH_LITTLE_ENDIAN_64(b)   __builtin_bswap64(b)
#define SPOOKYHASH_LITTLE_ENDIAN_32(b)   __builtin_bswap32(b)
#define SPOOKYHASH_LITTLE_ENDIAN_16(b)   __builtin_bswap16(b)
#else
#warning Using bulk byte swap routines. Expect performance issues.
#define SPOOKYHASH_LITTLE_ENDIAN_64(b)   ((((b) & 0xFF00000000000000ull) >> 56) | (((b) & 0x00FF000000000000ull) >> 40) | (((b) & 0x0000FF0000000000ull) >> 24) | (((b) & 0x000000FF00000000ull) >> 8) | (((b) & 0x00000000FF000000ull) << 8) | (((b) & 0x0000000000FF0000ull) << 24ull) | (((b) & 0x000000000000FF00ull) << 40) | (((b) & 0x00000000000000FFull) << 56))
#define SPOOKYHASH_LITTLE_ENDIAN_32(b)   ((((b) & 0xFF000000) >> 24) | (((b) & 0x00FF0000) >> 8) | (((b) & 0x0000FF00) << 8) | (((b) & 0x000000FF) << 24))
#define SPOOKYHASH_LITTLE_ENDIAN_16(b)   ((((b) & 0xFF00) >> 8) | (((b) & 0x00FF) << 8))
#endif
#else
#error Unknow endianness
#endif

#define SPOOKYHASH_MAJOR_VERSION   1
#define SPOOKYHASH_MINOR_VERSION   0
#define SPOOKYHASH_REVISION        5

#endif
