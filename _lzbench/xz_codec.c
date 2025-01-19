/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 */

#ifndef BENCH_REMOVE_XZ

#include "xz/src/liblzma/common/common.h"

int64_t lzbench_xz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t ignore1, char* ignore2)
{
    lzma_options_lzma opt_lzma;
    lzma_stream strm = LZMA_STREAM_INIT;
    uint32_t preset = level; // preset |= LZMA_PRESET_EXTREME;

    if (lzma_lzma_preset(&opt_lzma, preset))
        return 0;

    lzma_ret ret = lzma_alone_encoder(&strm, &opt_lzma);
    if (ret != LZMA_OK)
        return 0;

    strm.next_in = inbuf;
    strm.avail_in = insize;
    strm.next_out = outbuf;
    strm.avail_out = outsize;

    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END)
        return 0;

    lzma_end(&strm);

    return (char*)strm.next_out - outbuf;
}

int64_t lzbench_xz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t ignore1, size_t ignore2, char* ignore3)
{
    lzma_stream strm = LZMA_STREAM_INIT;

    lzma_ret ret = lzma_alone_decoder(&strm, UINT64_MAX);
    if (ret != LZMA_OK)
        return 0;

    strm.next_in = inbuf;
    strm.avail_in = insize;
    strm.next_out = outbuf;
    strm.avail_out = outsize;

    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END)
        return 0;

    lzma_end(&strm);

    return (char*)strm.next_out - outbuf;
}

#endif // BENCH_REMOVE_XZ
