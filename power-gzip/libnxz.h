/*
 * NX-GZIP compression accelerator user library
 * implementing zlib library interfaces
 *
 * Copyright (C) IBM Corporation, 2020
 *
 * Licenses for GPLv2 and Apache v2.0:
 *
 * GPLv2:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * Apache v2.0:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/** @file libnxz.h
 *  @brief Provides a public API
 */


#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef _LIBNXZ_H
#define _LIBNXZ_H

/* This must be kept in sync with zlib.h */
typedef unsigned int uInt;

/* zlib crc32.c and adler32.c */
extern "C" ulong nx_adler32(ulong adler, const char *buf, uint len);
extern "C" ulong nx_adler32_z(ulong adler, const char *buf, size_t len);
extern "C" ulong nx_adler32_combine(ulong adler1, ulong adler2, off_t len2);
extern "C" ulong nx_crc32(ulong crc, const unsigned char *buf, uint64_t len);
extern "C" ulong nx_crc32_combine(ulong crc1, ulong crc2, uint64_t len2);
extern "C" ulong nx_crc32_combine64(ulong crc1, ulong crc2, uint64_t len2);

/* nx_deflate.c */
extern "C" int nx_deflateInit_(void *strm, int level, const char *version,
			   int stream_size);
extern "C" int nx_deflateInit2_(void *strm, int level, int method, int windowBits,
			    int memLevel, int strategy, const char *version,
			    int stream_size);
#define nx_deflateInit(strm, level) nx_deflateInit_((strm), (level), \
					ZLIB_VERSION, (int)sizeof(z_stream))
#define nx_deflateInit2(strm, level, method, windowBits, memLevel, strategy) \
	nx_deflateInit2_((strm), (level), (method), (windowBits), (memLevel), \
		(strategy), ZLIB_VERSION, (int)sizeof(z_stream))
extern "C" int nx_deflate(void *strm, int flush);
extern "C" int nx_deflateEnd(void *strm);
extern "C" ulong nx_deflateBound(void *strm, ulong sourceLen);
extern "C" int nx_deflateSetHeader(void *strm, void *head);
extern "C" int nx_deflateCopy(void *dest, void *source);
extern "C" int nx_deflateReset(void *strm);
extern "C" int nx_deflateResetKeep(void *strm);
extern "C" int nx_deflateSetDictionary(void *strm, const unsigned char *dictionary,
				   uint dictLength);

/* nx_inflate.c */
extern "C" int nx_inflateInit_(void *strm, const char *version, int stream_size);
extern "C" int nx_inflateInit2_(void *strm, int windowBits, const char *version,
			    int stream_size);
#define nx_inflateInit(strm) nx_inflateInit_((strm), ZLIB_VERSION, \
					(int)sizeof(z_stream))
#define nx_inflateInit2(strm, windowBits) \
	nx_inflateInit2_((strm), (windowBits), ZLIB_VERSION, (int)sizeof(z_stream))
extern "C" int nx_inflate(void *strm, int flush);
extern "C" int nx_inflateEnd(void *strm);
extern "C" int nx_inflateCopy(void *dest, void *source);
extern "C" int nx_inflateGetHeader(void *strm, void *head);
extern "C" int nx_inflateSyncPoint(void *strm);
extern "C" int nx_inflateResetKeep(void *strm);
extern "C" int nx_inflateSetDictionary(void *strm, const unsigned char *dictionary,
				   uint dictLength);
extern "C" int nx_inflateReset(void *strm);

/* nx_compress.c */
extern "C" int nx_compress2(unsigned char *dest, ulong *destLen,
			const unsigned char *source, ulong sourceLen,
			int level);
extern "C" int nx_compress(unsigned char *dest, ulong *destLen,
		       const unsigned char *source, ulong sourceLen);
extern "C" ulong nx_compressBound(ulong sourceLen);

/* nx_uncompr.c */
extern "C" int nx_uncompress2(unsigned char *dest, ulong *destLen,
			  const unsigned char *source, ulong *sourceLen);
extern "C" int nx_uncompress(unsigned char *dest, ulong *destLen,
			 const unsigned char *source, ulong sourceLen);


// extern "C" int deflateInit_(void *strm, int level, const char* version,
// 			int stream_size);
// extern "C" int deflateInit2_(void *strm, int level, int method, int windowBits,
// 			 int memLevel, int strategy, const char *version,
// 			 int stream_size);
// extern "C" int deflateReset(void *strm);
// extern "C" int deflateEnd(void *strm);
// extern "C" int deflate(void *strm, int flush);
// extern "C" ulong deflateBound(void *strm, ulong sourceLen);
// extern "C" int deflateSetHeader(void *strm, void *head);
// extern "C" int deflateSetDictionary(void *strm, const unsigned char *dictionary,
// 				uint  dictLength);
// extern "C" int deflateCopy(void *dest, void *source);
// extern "C" int inflateInit_(void *strm, const char *version, int stream_size);
// extern "C" int inflateInit2_(void *strm, int windowBits, const char *version,
// 			 int stream_size);
// extern "C" int inflateReset(void *strm);
// extern "C" int inflateEnd(void *strm);
// 
// /** @brief Attempt to decompress data
//  *
//  *  @param strm A stream structure initialized by a call to inflateInit2().
//  *  @param flush Determines when uncompressed bytes are added to the output
//  *         buffer.  Possible values are:
//  *         - Z_NO_FLUSH: May return with some data pending output.
//  *         - Z_SYNC_FLUSH: Flush as much as possible to the output buffer.
//  *         - Z_FINISH: Performs decompression in a single step.  The output
//  *                     buffer must be large enough to fit all the decompressed
//  *                     data.  Otherwise, behaves as Z_NO_FLUSH.
//  *         - Z_BLOCK: Stop when it gets to the next block boundary.
//  *         - Z_TREES: Like Z_BLOCK, but also returns at the end of each deflate
//  *                    block header.
//  *  @return Possible values are:
//  *          - Z_OK: Decompression progress has been made.
//  *          - Z_STREAM_END: All the input has been decompressed and there was
//  *                          enough space in the output buffer to store the
//  *                          uncompressed result.
//  *          - Z_BUF_ERROR: No progress is possible.
//  *          - Z_MEM_ERROR: Insufficient memory.
//  *          - Z_STREAM_ERROR: The state (as represented in \c stream) is
//  *                            inconsistent, or stream was \c NULL.
//  *          - Z_NEED_DICT: A preset dictionary is required. Set the \c adler
//  *                         field to the Adler-32 checksum of the dictionary.
//  *          - Z_DATA_ERROR: The input data was corrupted.
//  */
// extern "C" int inflate(void *strm, int flush);
// extern "C" int inflateSetDictionary(void *strm, const unsigned char *dictionary,
// 				uint dictLength);
// extern "C" int inflateCopy(void *dest, void *source);
// extern "C" int inflateGetHeader(void *strm, void *head);
// extern "C" int inflateSyncPoint(void *strm);
// extern "C" ulong adler32_combine(ulong adler1, ulong adler2, off_t len2);
// extern "C" ulong adler32_combine64(ulong adler1, ulong adler2, off_t len2);
// extern "C" ulong adler32(ulong adler, const char *buf, uint len);
// extern "C" ulong adler32_z(ulong adler, const char *buf, size_t len);
// extern "C" ulong crc32(ulong crc, const unsigned char *buf, uInt len);
// extern "C" ulong crc32_combine(ulong crc1, ulong crc2, uint64_t len2);
// extern "C" ulong crc32_combine64(ulong crc1, ulong crc2, uint64_t len2);
// extern "C" int compress(unsigned char *dest, ulong *destLen,
// 		    const unsigned char *source, ulong sourceLen);
// extern "C" int compress2(unsigned char *dest, ulong *destLen,
// 		     const unsigned char *source, ulong sourceLen, int level);
// extern "C" ulong compressBound(ulong sourceLen);
// extern "C" int uncompress(unsigned char *dest, ulong *destLen,
// 		      const unsigned char *source, ulong sourceLen);
// extern "C" int uncompress2(unsigned char *dest, ulong *destLen,
// 		       const unsigned char *source, ulong *sourceLen);

#endif /* _LIBNXZ_H */
