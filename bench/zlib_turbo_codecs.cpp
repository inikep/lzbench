/* zlib-turbo wrappers: in a file of their own, as zlib-turbo's zlib.h is built
 * with Z_PREFIX and cannot be included with lz+entropy/zlib/zlib.h. */

#include "codecs.h"
#include <string.h>
#include "zlib.h"
#include "zturbo.h"

int64_t lzbench_zlib_turbo_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (deflateInit(&strm, codec_options->level) != Z_OK)
        return 0;
    // levels 4-9 search on worker threads, with the output of one thread; without
    // threads in the build this fails, and the stream stays on one thread
    if (codec_options->threads > 1)
        deflateSetThreads(&strm, codec_options->threads);

    // as compress2(), which cannot take deflateSetThreads()
    const uInt max = (uInt)-1;
    size_t in_left = insize, out_left = outsize;
    strm.next_in = (Bytef*)inbuf;
    strm.next_out = (Bytef*)outbuf;
    int err;
    do {
        if (strm.avail_out == 0) {
            strm.avail_out = out_left > max ? max : (uInt)out_left;
            out_left -= strm.avail_out;
        }
        if (strm.avail_in == 0) {
            strm.avail_in = in_left > max ? max : (uInt)in_left;
            in_left -= strm.avail_in;
        }
        err = deflate(&strm, in_left ? Z_NO_FLUSH : Z_FINISH);
    } while (err == Z_OK);
    int64_t complen = (char*)strm.next_out - outbuf;
    deflateEnd(&strm);
    return err == Z_STREAM_END ? complen : 0;
}

int64_t lzbench_zlib_turbo_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    uLongf zdecomplen = outsize;
    int err = uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize);
    if (err != Z_OK)
        return 0;
    return zdecomplen;
}
