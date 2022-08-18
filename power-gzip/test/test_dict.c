#include <assert.h>
#include <endian.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#include "test.h"
#include "test_utils.h"

#define DEF_MAX_DICT_LEN   (1L<<15)
#define DATALEN 1024*1024 // 1MB

Byte *src, *compr, *uncompr;
const unsigned int src_len = DATALEN;
const unsigned int compr_len = DATALEN*2;
const unsigned int uncompr_len = DATALEN*2;

/* Creates a dictionary suitable to be used with *SetDictionary family of
 * functions. It consists of the concatenation of strings of sizes between 3 and
 * 258 (maximum match size) randomly taken from ran_data. */
unsigned long fill_dict(unsigned char* dict, int dict_len, int data_len)
{
	int bytes_filled = 0;
	while (bytes_filled < dict_len) {
		int pos = rand() % (data_len-258);
		int len = 3 + rand()%256; // in range [3,258]
		len = bytes_filled + len > dict_len ?
			dict_len - bytes_filled : len;

		memcpy(&dict[bytes_filled], &ran_data[pos], len);
		bytes_filled += len;
	}

	return adler32(1, dict, dict_len);
}

void test_setDictionary(unsigned char* dict, int dict_len,
		unsigned long dict_adler32, bool raw_mode)
{
	unsigned long total_exp;
	z_stream strm;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = (voidpf)0;

	if (raw_mode)
		zcheck(deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
					-15, 8, Z_DEFAULT_STRATEGY), Z_OK);
	else
		zcheck(deflateInit(&strm, Z_DEFAULT_COMPRESSION), Z_OK);

	zcheck(deflateSetDictionary(&strm, dict, dict_len), Z_OK);
	total_exp = MIN(dict_len, DEF_MAX_DICT_LEN);
	if (strm.total_in != total_exp) // zlib behavior
		test_error("Wrong strm.total_in! Expected %ld but got %ld\n",
			total_exp, strm.total_in);

	/* On ZLIB mode, strm.adler must contain the dictionary's adler32 */
	if (!raw_mode && strm.adler != dict_adler32)
		test_error("deflateSetDictionary set wrong strm.adler"
			"Expected 0x%08lx but got 0x%08lx\n", dict_adler32,
			strm.adler);

	strm.next_in  = src;
	strm.avail_in = src_len;
	strm.next_out = compr;
	strm.avail_out = compr_len;

	zcheck(deflate(&strm, Z_FINISH), Z_STREAM_END);
	zcheck(deflateEnd(&strm), Z_OK);

	/* On ZLIB mode, check FDICT, DICTID and adler32 fields in the
	   generated zlib stream. */
	if (!raw_mode) {

		/* FLG is the second byte in the stream, and FLG.FDICT is bit
		   6. The dictionary's adler32 (DICTID) comes right after
		   FLG. FDICT must be set */
		if((*(compr + 1) & (Byte)0x20) >> 5 != (Byte) 1)
			test_error("FLG.FDICT not set in zlib header!\n");

		/* DICTID must match the adler32 calculated by fill_dict */
		unsigned long dictid = (unsigned long) be32toh(*((uint32_t *)(compr + 2)));
		if (dictid != dict_adler32)
			test_error("Wrong dictid! Expected 0x%08lx but got 0x%08lx\n",
				dict_adler32, dictid);

		/* adler32 written to compressed stream (last 4 bytes) should be
		   the same as adler32 for data only, ignoring the
		   dictionary. */
		unsigned long out_adler32 = (unsigned long) be32toh(
			*((uint32_t *)(compr + strm.total_out - 4)));
		unsigned long data_adler32 = adler32(0L, Z_NULL, 0);
		data_adler32 = adler32(data_adler32, ran_data, src_len);

		if (out_adler32 != data_adler32)
			test_error("Wrong adler32! Expected 0x%08lx but got 0x%08lx\n",
				data_adler32, out_adler32);
	}

	memset(&strm, 0, sizeof(strm));

	if (raw_mode)
		zcheck(nx_inflateInit2(&strm, -15), Z_OK);
	else
		zcheck(inflateInit(&strm), Z_OK);

	strm.next_in  = compr;
	strm.avail_in = compr_len;
	strm.next_out = uncompr;
	strm.avail_out = uncompr_len;

	if (raw_mode) {
		/* nx_* functions are used just for raw mode to guarantee we test
		 * compressing with zlib and decompressing with NX, which will be
		 * handled by this case when NX_GZIP_TYPE_SELECTOR=1 */
		zcheck(nx_inflateSetDictionary(&strm, dict, dict_len), Z_OK);
		zcheck(nx_inflate(&strm, Z_NO_FLUSH), Z_STREAM_END);
		zcheck(nx_inflateEnd(&strm), Z_OK);
	} else {
		/* Check that inflateSetDictionary fails if called before inflate */
		zcheck(inflateSetDictionary(&strm, dict, dict_len), Z_STREAM_ERROR);
		zcheck(inflate(&strm, Z_NO_FLUSH), Z_NEED_DICT);

		/* At this point strm.adler must contain the adler32 of the
		   dictionary used by the compressor */
		if (strm.adler != dict_adler32)
			test_error("inflate set the wrong dictionary id! "
				"Expected 0x%08lx but got 0x%08lx\n", dict_adler32,
				strm.adler);

		/* Using a different dictionary (e.g. other size) should fail */
		zcheck(inflateSetDictionary(&strm, dict, dict_len-1), Z_DATA_ERROR);

		zcheck(inflateSetDictionary(&strm, dict, dict_len), Z_OK);
		zcheck(inflate(&strm, Z_NO_FLUSH), Z_STREAM_END);
		zcheck(inflateEnd(&strm), Z_OK);
	}

	if (compare_data(uncompr, src, src_len))
		exit(TEST_ERROR);
}

int main()
{
	unsigned long dict_adler32;
	unsigned char dict[DEF_MAX_DICT_LEN + 1024];

	/* lengths chosen to test the following cases:
	   1. small, medium, maximum and over-maximum lengths
	   2. lengths multiple of 16
	   3. lengths not multiple of 16 */
	int dict_lengths[] = {100, 783, 2520, 15341, 21769, 28800,
		DEF_MAX_DICT_LEN, DEF_MAX_DICT_LEN + 500 };

	generate_random_data(src_len);
	src = (Byte*)&ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL )
		test_error("*** alloc buffer failed\n");

	for (int i = 0 ; i < sizeof(dict_lengths)/sizeof(int) ; i++) {
		printf("Testing dictionary size %d\n", dict_lengths[i]);

		dict_adler32 = fill_dict(dict, dict_lengths[i], src_len);

		/* Test raw mode */
		printf("- RAW mode\n");
		test_setDictionary(dict, dict_lengths[i], dict_adler32, true);

		/* Test zlib mode */
		printf("- ZLIB mode\n");
		test_setDictionary(dict, dict_lengths[i], dict_adler32, false);
	}

	printf("*** %s passed\n", __FILE__);
	free(compr);
	free(uncompr);
	return TEST_OK;
}
