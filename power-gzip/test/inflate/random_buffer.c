#include "../test_deflate.h"
#include "../test_utils.h"

static int run(unsigned int len, int step, const char* test, int flush)
{
	Byte *src, *compr, *uncompr;
	unsigned int src_len = len;
	unsigned int compr_len = src_len*2;
	/* When compressing a short array, we may need a buffer larger than
	   2*src_len. Round this up to 1KiB. */
	compr_len = compr_len > 1024 ? compr_len : 1024;
	unsigned int uncompr_len = src_len*2;
	generate_random_data(src_len);
	src = &ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	if( (flush != Z_NO_FLUSH) && (flush != Z_PARTIAL_FLUSH) )
		goto err;

	if (_test_deflate(src, src_len, compr, &compr_len, src_len, NULL))
		goto err;
	if (_test_inflate(compr, compr_len, uncompr, uncompr_len, src, src_len,
			  step, NULL))
		goto err;
	if (_test_nx_inflate(compr, compr_len, uncompr, uncompr_len, src,
			     src_len, step, flush, NULL))
		goto err;

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

/* Test with short arrays.  */
int run_case1()
{
	int rc = TEST_OK;
	for (int i = 1; i <= 100; i++) {
		for (int j = 1; j <= i; j++) {
			rc = run(i, j, __func__, Z_NO_FLUSH);
			if (rc != TEST_OK) {
				printf("** %s %s with i = %d, j = %d failed\n",
				       __FILE__, __func__, i, j);
				return rc;
			}
			printf("** %s %s with i = %d, j = %d passed\n",
			       __FILE__, __func__, i, j);
		}
	}
	return TEST_OK;
}

/* The total src buffer < 64K and avail_in is 1 */
int run_case2()
{
	return run(5*1024, 1,  __func__,Z_NO_FLUSH);
}

/* The total src buffer < 64K and 1 < avail_in < total */
int run_case3()
{
	return run(5*1000, 100, __func__,Z_NO_FLUSH);
}

/* The total src buffer < 64K and avail_in is total */
int run_case4()
{
	return run(5*1024, 5*1024, __func__,Z_NO_FLUSH);
}

/* The total src buffer > 64K and avail_in is 1 */
int run_case5()
{
	return run(25*1024, 1, __func__,Z_NO_FLUSH);
}

/* The total src buffer > 64K and 1 < avail_in < total */
int run_case6()
{
	return run(128*1024, 10000, __func__,Z_NO_FLUSH);
}

/* The total src buffer > 64K and avail_in is total */
int run_case7()
{
	return run(128*1024, 128*1024, __func__,Z_NO_FLUSH);
}

/* A large buffer and 1 < avail_in < total */
int run_case8()
{
	return run(1024*1024*64, 4096, __func__,Z_NO_FLUSH);
}

/* A large buffer and avail_in > total */
int run_case9()
{
	return run(1024*1024*64, 1024*1024*64*2, __func__,Z_NO_FLUSH);
}

/* A large buffer and avail_in > total */
int run_case9_1()
{
	return run(4194304, 4194304, __func__,Z_NO_FLUSH);
}


/* The total src buffer < 64K and avail_in is 1 */
int run_case12()
{
	return run(5*1024, 1,  __func__,Z_PARTIAL_FLUSH);
}

/* The total src buffer < 64K and 1 < avail_in < total */
int run_case13()
{
	return run(5*1000, 100, __func__,Z_PARTIAL_FLUSH);
}

/* The total src buffer < 64K and avail_in is total */
int run_case14()
{
	return run(5*1024, 5*1024, __func__,Z_PARTIAL_FLUSH);
}

/* The total src buffer > 64K and avail_in is 1 */
int run_case15()
{
	// return run(128*1024, 1, __func__);
	return run(25*1024, 1, __func__,Z_PARTIAL_FLUSH);
}

/* The total src buffer > 64K and 1 < avail_in < total */
int run_case16()
{
	return run(128*1024, 10000, __func__,Z_PARTIAL_FLUSH);
}

/* The total src buffer > 64K and avail_in is total */
int run_case17()
{
	return run(128*1024, 128*1024, __func__,Z_PARTIAL_FLUSH);
}

/* A large buffer and 1 < avail_in < total */
int run_case18()
{
	return run(1024*1024*64, 4096, __func__,Z_PARTIAL_FLUSH);
}

/* A large buffer and avail_in > total */
int run_case19()
{
	return run(1024*1024*64, 1024*1024*64*2, __func__,Z_PARTIAL_FLUSH);
}

/* A large buffer and avail_in > total */
int run_case19_1()
{
	return run(4194304, 4194304, __func__,Z_PARTIAL_FLUSH);
}
