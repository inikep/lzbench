#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#include "test.h"
#include "test_utils.h"

#define NCHUNKS 50
#define CHUNKSIZE 256

int main()
{
	Byte *src, *compr, *uncompr;
	unsigned int src_len = CHUNKSIZE*NCHUNKS;
	unsigned int compr_len = src_len*2;
	int sync_point_pos[NCHUNKS];
	unsigned int uncompr_len = src_len*2;
	z_stream strm;

	generate_random_data(src_len);
	src = (Byte*)&ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	/* ** Compression ** */

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = (voidpf)0;

	assert(nx_deflateInit(&strm, Z_DEFAULT_COMPRESSION) == Z_OK);

	strm.next_in  = src;
	strm.next_out = compr;

	/* Compress input data into NCHUNKS chunks, each ending with a sync
	   block */
	for(int i = 0; i < NCHUNKS; i++) {
		strm.avail_in = CHUNKSIZE;
		strm.avail_out = compr_len - strm.total_out;

		assert(nx_deflate(&strm, Z_SYNC_FLUSH) == Z_OK);

		/* Remember where each sync point is. Sync point is 4 bytes
		   before the end of 0-length literal block. */
		sync_point_pos[i] = strm.total_out - 4;
	}

	assert(nx_deflate(&strm, Z_FINISH) == Z_STREAM_END);

	assert(strm.total_in == src_len);

	assert(nx_deflateEnd(&strm) == Z_OK);

	/* ** Decompression ** */

	memset(&strm, 0, sizeof(strm));

	assert(nx_inflateInit(&strm) == Z_OK);

	/* Assert if value was correctly initialized */
	assert(inflateSyncPoint(&strm) == 0);

	strm.next_in  = compr;
	strm.next_out = uncompr;

	/* Let's give inflate just enough data to stop 1 byte before the first
	 * sync point, and then walk the boundary 1 byte at a time.  */

	strm.avail_in = sync_point_pos[0] - 1;
	strm.avail_out = uncompr_len;
	assert(nx_inflate(&strm, Z_SYNC_FLUSH) == Z_OK);

	/* 1 byte before the sync point */
	assert(strm.avail_in == 0);
	assert(nx_inflateSyncPoint(&strm) == 0);

	strm.avail_in = 1;
	strm.avail_out = uncompr_len - strm.total_out;
	assert(nx_inflate(&strm, Z_SYNC_FLUSH) == Z_OK);

	/* At the sync point (i.e. after BTYPE and before LEN of a 0-length
	 * literal block) */
	assert(nx_inflateSyncPoint(&strm) == 1);

	/* Process 1 more byte and assert */
	strm.avail_in = 1;
	strm.avail_out = uncompr_len - strm.total_out;
	assert(nx_inflate(&strm, Z_SYNC_FLUSH) == Z_OK);

	/* 1 byte after the sync point */
	assert(nx_inflateSyncPoint(&strm) == 0);

	/* Align to the next sync point  */
	strm.avail_in = sync_point_pos[1] - sync_point_pos[0] - 1;
	strm.avail_out = uncompr_len - strm.total_out;
	assert(nx_inflate(&strm, Z_SYNC_FLUSH) == Z_OK);
	assert(nx_inflateSyncPoint(&strm) == 1);

	/* Check all other sync points */
	for (int i = 2; i < NCHUNKS; i++) {
		strm.avail_in = sync_point_pos[i] - sync_point_pos[i-1];
		strm.avail_out = uncompr_len - strm.total_out;
		assert(nx_inflate(&strm, Z_SYNC_FLUSH) == Z_OK);
		assert(nx_inflateSyncPoint(&strm) == 1);
	}

	/* Finish what's left */
	strm.avail_in = compr_len - strm.total_in;
	strm.avail_out = uncompr_len - strm.total_out;
	assert(nx_inflate(&strm, Z_FINISH) == Z_STREAM_END);

	assert(nx_inflateEnd(&strm) == Z_OK);

	if (compare_data(uncompr, src, src_len))
		return TEST_ERROR;

	printf("*** %s passed\n", __FILE__);
	free(compr);
	free(uncompr);
	return TEST_OK;
}
