#ifndef BENCH_REMOVE_ZLIB_NG
#include "../zlib-ng/zlib-ng.h"

int64_t lzbench_zlib_ng_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	uLongf zcomplen = insize;
	int err = zng_compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, level);
	if (err != Z_OK)
		return 0;
	return zcomplen;
}

int64_t lzbench_zlib_ng_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	uLongf zdecomplen = outsize;
	int err = zng_uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize); 
	if (err != Z_OK)
		return 0;
	return outsize;
}

#endif


