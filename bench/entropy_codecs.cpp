/*
 * Copyleft (c) 2021 @tansy
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * entropy_codecs.cpp: pure entropy coders, like Huffman, FSE/ANS,
 *     or range coder
 */

#include "codecs.h"

#include <stdint.h>
#include <stdio.h> // printf
#include <string.h> // memcpy
#include <algorithm> // std::max



#ifndef BENCH_REMOVE_HUFFMAN_ZLIB
#include "zlib/zlib.h"

int64_t lzbench_huffman_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
	int err;
	long complen;
	z_stream strm; strm.zalloc = Z_NULL; strm.zfree = Z_NULL; strm.opaque = Z_NULL;

	err = deflateInit2(&strm, codec_options->level, Z_DEFLATED, -MAX_WBITS, 8, Z_HUFFMAN_ONLY);

	if (err != Z_OK)
		return -1;

	strm.next_in = (Bytef *)inbuf;
	strm.avail_in = (uInt)insize;
	strm.next_out = (Bytef *)outbuf;
	strm.avail_out = (uInt)outsize;

	do {
		err = deflate(&strm, Z_FINISH);
	} while (err == Z_OK);
	if (err != Z_STREAM_END)
	{
		deflateEnd(&strm);
		return -1;
	}
	complen = (long)strm.total_out;
	deflateEnd(&strm);

	return complen;
}

int64_t lzbench_huffman_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
	int err;
	long result;
	z_stream strm;
	memset(&strm, 0, sizeof(strm));

	if (inflateInit2(&strm, -15) != Z_OK)
		return -1;

	strm.next_in = (Bytef *)inbuf;
	strm.avail_in = (uInt)insize;
	strm.next_out = (Bytef *)outbuf;
	strm.avail_out = (uInt)outsize;

	do {
		err = inflate(&strm, Z_NO_FLUSH);
		} while (err == Z_OK && strm.avail_out > 0 && strm.avail_in > 0);

	if (err == Z_STREAM_END)
		result = (long)strm.total_out;
	else
		result = -2;
		inflateEnd(&strm);

	return result;
}

#endif // BENCH_REMOVE_HUFFMAN_ZLIB

