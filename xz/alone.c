#include "common.h"
#include "alone.h"

int64_t xz_alone_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t x, size_t y)
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
 //   printf("%d %d %d %d\n", strm.next_in, strm.avail_in, strm.next_out, strm.avail_out);
/*
	ret = lzma_code(&strm, LZMA_RUN);
	if (ret != LZMA_OK)
		return 0;
*/
	ret = lzma_code(&strm, LZMA_FINISH);
	if (ret != LZMA_STREAM_END)
        return 0;

 //   printf("%d after %d %d %d %d\n", (char*)strm.next_out - outbuf, strm.next_in, strm.avail_in, strm.next_out, strm.avail_out);

    lzma_end(&strm);
    
    return (char*)strm.next_out - outbuf;
}


int64_t xz_alone_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t x, size_t y)
{
    lzma_stream strm = LZMA_STREAM_INIT;

	lzma_ret ret = lzma_alone_decoder(&strm, UINT64_MAX);
	if (ret != LZMA_OK)
		return 0;

	strm.next_in = inbuf;
	strm.avail_in = insize;
	strm.next_out = outbuf;
	strm.avail_out = outsize;
 //   printf("%d %d %d %d\n", strm.next_in, strm.avail_in, strm.next_out, strm.avail_out);

	ret = lzma_code(&strm, LZMA_FINISH);
	if (ret != LZMA_STREAM_END)
        return 0;

 //   printf("%d after %d %d %d %d\n", (char*)strm.next_out - outbuf, strm.next_in, strm.avail_in, strm.next_out, strm.avail_out);

    lzma_end(&strm);
    
    return (char*)strm.next_out - outbuf;
}
