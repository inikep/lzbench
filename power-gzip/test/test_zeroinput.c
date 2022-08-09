/* Test calling deflate with zero available input and different flush modes. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#include "test.h"
#include "test_utils.h"

const char* flush2str[] = {
	"Z_NO_FLUSH",
	"Z_PARTIAL_FLUSH",
	"Z_SYNC_FLUSH",
	"Z_FULL_FLUSH",
	"Z_FINISH",
};

int main()
{
	Byte *src, *compr;
	unsigned int src_len = 2048;
	unsigned int compr_len = src_len*3;
	z_stream strm;
	int failed = 0;

	generate_random_data(src_len);
	src = (Byte*)&ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	if (compr == NULL || src == NULL) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	printf("-> empty data stream\n");
	/* Test with Z_{NO,PARTIAL,SYNC,FULL}_FLUSH, and Z_FINISH */
	for(int flush = Z_NO_FLUSH; flush <= Z_FINISH; flush++) {
		memset(&strm, 0, sizeof(strm));
		assert(deflateInit(&strm, Z_DEFAULT_COMPRESSION) == Z_OK);

		strm.next_in  = src;
		strm.next_out = compr;

		strm.avail_in = 0;
		strm.avail_out = compr_len;
		int ret = deflate(&strm, flush);
		int exp = flush == Z_FINISH ? Z_STREAM_END : Z_OK;
		if(ret != exp){
			printf("%s:%d when using %s expected: %d, but got %d\n", __FILE__, __LINE__, flush2str[flush], exp, ret);
			failed = 1;
		}

		if(flush != Z_FINISH)
			assert(deflate(&strm, Z_FINISH) == Z_STREAM_END);

		printf("Output for %s:", flush2str[flush]);
		for(int j = 0; j < strm.total_out; j++)
			printf(" 0x%02x", compr[j]);
		printf("\n");

		assert(deflateEnd(&strm) == Z_OK);
	}

	printf("-> mid-decompression zero input\n");
	/* Test with Z_{NO,PARTIAL,SYNC,FULL}_FLUSH, and Z_FINISH */
	for(int flush = Z_NO_FLUSH; flush <= Z_FINISH; flush++) {
		memset(&strm, 0, sizeof(strm));
		assert(deflateInit(&strm, Z_DEFAULT_COMPRESSION) == Z_OK);

		strm.next_in  = src;
		strm.next_out = compr;

		strm.avail_in = src_len/2;
		strm.avail_out = compr_len;
		assert(deflate(&strm, Z_NO_FLUSH) == Z_OK);

		strm.avail_in = 0;
		int ret = deflate(&strm, flush);
		int exp = flush == Z_NO_FLUSH ? Z_BUF_ERROR :
			  flush == Z_FINISH ? Z_STREAM_END : Z_OK;
		if(ret != exp){
			printf("%s:%d when using %s expected %d, but got %d\n", __FILE__, __LINE__, flush2str[flush], exp, ret);
			failed = 1;
		}

		if(flush != Z_FINISH)
			assert(deflate(&strm, Z_FINISH) == Z_STREAM_END);

		assert(deflateEnd(&strm) == Z_OK);
	}

	printf("*** %s %s\n", __FILE__, failed ? "failed" : "passed");
	free(compr);

	return failed ? TEST_ERROR : TEST_OK;
}
