#include "../test_deflate.h"
#include "../test_utils.h"

/* use nx to compress */
static int _test_nx_compress(Byte* compr, unsigned long *compr_len, Byte* src, unsigned long src_len)
{
	int rc;
	rc = nx_compress(compr, compr_len, src, src_len);
	dbg_printf("rc %d src_len %ld compr_len %ld\n", rc, src_len, *compr_len);
	if (rc < 0) {
		printf("*** failed: nx_compress returned %d\n", rc);
		return TEST_ERROR;
	}
	return TEST_OK;
}

/* use zlib to uncomress */
static int _test_uncompress(Byte* uncompr, unsigned long *uncomprLen, Byte* compr, unsigned long comprLen, Byte* src, unsigned long src_len)
{
	int rc;
	memset(uncompr, 0, *uncomprLen);
	rc = uncompress(uncompr, uncomprLen, compr, comprLen);
	dbg_printf("rc %d compr_len %ld uncompr_len %ld\n", rc, comprLen, *uncomprLen);
	if (rc < 0) {
		printf("*** failed: uncompress returned %d\n", rc);
		return TEST_ERROR;
	}
	if (compare_data(uncompr, src, src_len)) {
		return TEST_ERROR;
	}
	return TEST_OK;
}

/* use nx to uncomress */
static int _test_nx_uncompress(Byte* uncompr, unsigned long *uncomprLen, Byte* compr, unsigned long comprLen, Byte* src, unsigned long src_len)
{
	int rc;
	memset(uncompr, 0, *uncomprLen);
	rc = nx_uncompress(uncompr, uncomprLen, compr, comprLen);
	dbg_printf("rc %d compr_len %ld uncompr_len %ld\n", rc, comprLen, *uncomprLen);
	if (rc < 0) {
		printf("*** failed: nx_uncompress returned %d\n", rc);
		return TEST_ERROR;
	}
	if (compare_data(uncompr, src, src_len)) {
		return TEST_ERROR;
	}
	return TEST_OK;
}

static int run(unsigned int len, int all, char digit, const char* test)
{
	Byte *src, *compr, *uncompr;
	unsigned long src_len = len;
	unsigned long compr_len = src_len*2;
	unsigned long uncompr_len = src_len*2;
	if (all)
		generate_all_data(src_len, digit);
	else
		generate_random_data(src_len);
	src = &ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	if (_test_nx_compress(compr, &compr_len, src, src_len)) goto err;
	if (_test_uncompress(uncompr, &uncompr_len, compr, compr_len, src, src_len)) goto err;
	if (_test_nx_uncompress(uncompr, &uncompr_len, compr, compr_len, src,
				src_len)) goto err;

	printf("*** %s %s passed\n", __FILE__, test);
	free(compr);
	free(uncompr);
	return TEST_OK;
err:
	printf("*** %s %s failed\n", __FILE__, test);
	free(compr);
	free(uncompr);
	return TEST_ERROR;
}

/* case prefix is 10-19 */
int run_case10()
{
	/* The total src buffer < default cache_threshold. */
	return run(5*1024, 0, 0, __func__);
}

int run_case11()
{
	/* The total src buffer > cache_threshold. */
	return run(20*1024, 0, 0, __func__);
}

/* The total src buffer = 2M */
int run_case12()
{
	return run(2*1024*1024, 0, 0, __func__);
}

/* The total src buffer > 8M */
int run_case13()
{
	return run(20*1024*1024, 0, 0, __func__);
}

/* The total src buffer > 64M */
int run_case14()
{
	return run(64*1024*1024, 0, 0, __func__);
}

/* The total buffer is 0 */
int run_case15()
{
	return run(2*1024*1024, 1, 0, __func__);
}
