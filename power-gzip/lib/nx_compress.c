/* nx_compress.c -- compress a memory buffer
 * This is a modified version of compress.c from the zlib library.
 *
 * Copyright (C) 1995-2011, 2016 Mark Adler
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include <zlib.h>
#include "nx_zlib.h"

int nx_compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
{
    z_stream stream;
    int rc;
    const uInt max = 1<<30;
    uLong remaining;

    remaining = *destLen;
    *destLen = 0;

    memset(&stream, 0, sizeof(stream));

    prt_info("nx_compress2 begin: sourceLen %ld\n", sourceLen);

    rc = nx_deflateInit(&stream, level);
    if (rc != Z_OK) return rc;

    stream.next_out = dest;
    stream.avail_out = 0;
    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;
    do {
        if (stream.avail_out == 0) {
            stream.avail_out = remaining > (uLong)max ? max : (uInt)remaining;
            remaining -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = sourceLen > (uLong)max ? max : (uInt)sourceLen;
            sourceLen -= stream.avail_in;
        }
        rc = nx_deflate(&stream, sourceLen ? Z_NO_FLUSH : Z_FINISH);
	prt_info("     err %d\n", rc);
    } while (rc == Z_OK);

    *destLen = stream.total_out;
    nx_deflateEnd(&stream);

    prt_info("nx_compress2 end: destLen %ld\n", *destLen);
    return rc == Z_STREAM_END ? Z_OK : rc;
}

int nx_compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
{
    return nx_compress2(dest, destLen, source, sourceLen, Z_DEFAULT_COMPRESSION);
}

uLong nx_compressBound(uLong sourceLen)
{
    return nx_deflateBound(NULL, sourceLen);
}

#ifdef ZLIB_API

int compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
{
	return compress2(dest, destLen, source, sourceLen, Z_DEFAULT_COMPRESSION);
}

int compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
{
	int rc=0;

	if(nx_config.mode.deflate == GZIP_AUTO){
		if(sourceLen <= COMPRESS_THRESHOLD)
			rc = sw_compress2(dest, destLen, source, sourceLen, level);
		else
			rc = nx_compress2(dest, destLen, source, sourceLen, level);
	}else if(nx_config.mode.deflate == GZIP_NX){
		rc = nx_compress2(dest, destLen, source, sourceLen, level);
	}else{
		rc = sw_compress2(dest, destLen, source, sourceLen, level);
	}

	/* statistic */
	zlib_stats_inc(&zlib_stats.compress);

	return rc;
}

uLong compressBound(uLong sourceLen)
{
	return	NX_MAX(nx_deflateBound(NULL, sourceLen),
                   sw_deflateBound(NULL, sourceLen));
}

#endif
