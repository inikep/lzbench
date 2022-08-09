#include "../test_deflate.h"
#include "../test_utils.h"

static int run(unsigned int len, int step, const char* test)
{
	Byte *src, *compr, *uncompr;
	unsigned int src_len = len;
	unsigned int compr_len = src_len*2;
	unsigned int uncompr_len = src_len*2;
	generate_random_data(src_len);
	src = &ran_data[0];

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	if (_test_nx_deflate(src, src_len, compr, &compr_len, step, NULL))
		goto err;
	if (_test_inflate(compr, compr_len, uncompr, uncompr_len, src, src_len,
			  step, NULL))
		goto err;
	if (_test_nx_inflate(compr, compr_len, uncompr, uncompr_len, src,
			     src_len, step, Z_NO_FLUSH, NULL))
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

/* case prefix is 2 ~ 9 */

int run_case2()
{
	/* The total src buffer < default cache_threshold and avail_in is 1. */
	return run(5*1024, 1,  __func__);
}

int run_case3()
{
	/* The total src buffer < default cache_threshold and
	   1 < avail_in < total. */
	return run(5*1000, 100, __func__);
}

/* The total src buffer < default cache_threshold and 1 < avail_in < total
 * but avail_in is not aligned with src_len
 * TODO: this is an error case
 */
int run_case3_1()
{
	// return run(5*1024, 100, __func__);
	return 0;
}

int run_case4()
{
	/* The total src buffer < default cache_threshold and avail_in is
	   total. */
	return run(5*1024, 5*1024, __func__);
}

int run_case5()
{
	/* The total src buffer > default cache_threshold and avail_in is 1. */
	return run(64*1024, 1, __func__);
}

int run_case6()
{
	/* The total src buffer > default cache_threshold and
	   1 < avail_in < 10*1024. */
	return run(64*10000, 10000, __func__);
}

int run_case7()
{
	/* The total src buffer > default cache_threshold and
	   avail_in > 10*1024. */
	return run(64*20000, 20000, __func__);
}

/* A large buffer > fifo_in len and and 1 < avail_in < 10*1024 */
int run_case8()
{
	return run(1024*1024*8, 4096, __func__);
}

/* A large buffer > fifo_in len and and avail_in > 10K */
int run_case8_1()
{
	return run(1024*1024*8, 1024*128, __func__);
}

/* A large buffer > fifo_in len and and avail_in == total */
int run_case8_2()
{
	return run(1024*1024*8, 1024*1024*8, __func__);
}

/* A large buffer > fifo_in len and and avail_in == total */
int run_case8_3()
{
	return run(1024*1024*20, 1024*1024*20, __func__);
}

/* A large buffer > fifo_in len and and avail_in > 10*1024 */
int run_case9()
{
	return run(1024*1024*8, 1024*32, __func__);
}

