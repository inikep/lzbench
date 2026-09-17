/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Modifications Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_DEC_BLOCK_H
#define ZSTD_DEC_BLOCK_H

/*-*******************************************************
 *  Dependencies
 *********************************************************/
#include "../common/zstd_deps.h"   /* size_t */
#include "../zstd.h"    /* DCtx, and some public functions */
#include "../common/zstd_internal.h"  /* blockProperties_t, and some public functions */
#include "zstd_decompress_internal.h"  /* ZSTD_seqSymbol */


/* ===   Prototypes   === */

/* note: prototypes already published within `zstd.h` :
 * ZSTD_decompressBlock()
 */

/* note: prototypes already published within `zstd_internal.h` :
 * ZSTD_getcBlockSize()
 * ZSTD_decodeSeqHeaders()
 */


 /* Streaming state is used to inform allocation of the literal buffer */
typedef enum {
    not_streaming = 0,
    is_streaming = 1
} streaming_operation;

/* ZSTD_decompressBlock_internal() :
 * decompress block, starting at `src`,
 * into destination buffer `dst`.
 * @return : decompressed block size,
 *           or an error code (which can be tested using ZSTD_isError())
 */
size_t ZSTD_decompressBlock_internal(ZSTD_DCtx* dctx,
                               void* dst, size_t dstCapacity,
                               const void* src, size_t srcSize, const streaming_operation streaming);

/* ZSTD_buildFSETable() :
 * generate FSE decoding table for one symbol (ll, ml or off)
 * this function must be called with valid parameters only
 * (dt is large enough, normalizedCounter distribution total is a power of 2, max is within range, etc.)
 * in which case it cannot fail.
 * The workspace must be 4-byte aligned and at least ZSTD_BUILD_FSE_TABLE_WKSP_SIZE bytes, which is
 * defined in zstd_decompress_internal.h.
 * Internal use only.
 */
void ZSTD_buildFSETable(ZSTD_seqSymbol* dt,
             const short* normalizedCounter, unsigned maxSymbolValue,
             const U32* baseValue, const U8* nbAdditionalBits,
                   unsigned tableLog, void* wksp, size_t wkspSize,
                   int bmi2);

/* Internal definition of ZSTD_decompressBlock() to avoid deprecation warnings. */
size_t ZSTD_decompressBlock_deprecated(ZSTD_DCtx* dctx,
                            void* dst, size_t dstCapacity,
                      const void* src, size_t srcSize);

void aocl_setup_zstd_decompress_block(int optOff, uint64_t cpuFeatures);

void aocl_setup_zstd_decompress_block_native(void);

void aocl_destroy_zstd_decompress_block(void);

#endif /* ZSTD_DEC_BLOCK_H */

#ifdef AOCL_UNIT_TEST
#if defined (__cplusplus)
extern "C" {
#endif

/*
* src and dst have to be valid. length > 0. src < dst.
* src+length, dst+length cannot overflow.
* ovtype must indicate nature of src and dst positions.
* ovtype 0 : no overlap between src and dst. (dst-src) >= WILDCOPY_VECLEN.
* ovtype 1 : dst and src may overlap, but must be at least 8 bytes apart.
*/
ZSTDLIB_API void TEST_AOCL_ZSTD_wildcopy_long(void* dst, const void* src, size_t length, int ovtype);

#if defined (__cplusplus)
}
#endif
#endif
