/*
 * Copyright (c) 2021 @tansy
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * entropy_codecs.cpp: pure entropy coders (Huffman, FSE/ANS or range coding) without a modelling stage
 */

#include "codecs.h"

#include <string.h> // memset



#ifndef BENCH_REMOVE_HUFFMAN_ZLIB
#include "zlib/zlib.h"

// zlib's avail_in/avail_out are uInt, so buffers larger than 4 GB are passed in pieces, as compress2() does
static void huffman_zlib_refill(uInt* avail, size_t* left)
{
    if (*avail == 0) {
        const uInt max = (uInt)-1;
        *avail = *left > max ? max : (uInt)*left;
        *left -= *avail;
    }
}

// zlib with Z_HUFFMAN_ONLY: deflate blocks of literals only (no LZ77 matches), as raw deflate
int64_t lzbench_huffman_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (deflateInit2(&strm, codec_options->level, Z_DEFLATED, -MAX_WBITS, 8, Z_HUFFMAN_ONLY) != Z_OK)
        return 0;

    size_t in_left = insize, out_left = outsize;
    strm.next_in = (Bytef*)inbuf;
    strm.next_out = (Bytef*)outbuf;
    int err;
    do {
        huffman_zlib_refill(&strm.avail_in, &in_left);
        huffman_zlib_refill(&strm.avail_out, &out_left);
        err = deflate(&strm, in_left ? Z_NO_FLUSH : Z_FINISH);
    } while (err == Z_OK);

    int64_t complen = (err == Z_STREAM_END) ? (int64_t)(strm.next_out - (Bytef*)outbuf) : 0;
    deflateEnd(&strm);
    return complen;
}

int64_t lzbench_huffman_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, -MAX_WBITS) != Z_OK)
        return 0;

    size_t in_left = insize, out_left = outsize;
    strm.next_in = (Bytef*)inbuf;
    strm.next_out = (Bytef*)outbuf;
    int err;
    do {
        huffman_zlib_refill(&strm.avail_in, &in_left);
        huffman_zlib_refill(&strm.avail_out, &out_left);
        err = inflate(&strm, Z_NO_FLUSH);
    } while (err == Z_OK);

    int64_t decomplen = (err == Z_STREAM_END) ? (int64_t)(strm.next_out - (Bytef*)outbuf) : 0;
    inflateEnd(&strm);
    return decomplen;
}

#endif // BENCH_REMOVE_HUFFMAN_ZLIB
