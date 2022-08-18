#include "../test_deflate.h"
#include "../test_utils.h"

extern alloc_func zalloc;
extern free_func zfree;

/* use nx to deflate */
static int _test_nx_deflatef(Byte* src, unsigned int src_len, Byte* compr,
			     unsigned int compr_len)
{
	int err;
	z_stream c_stream;
	
	c_stream.zalloc = zalloc;
	c_stream.zfree = zfree;
	c_stream.opaque = (voidpf)0;
	
	err = nx_deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
	if (err != 0) {
		printf("nx_deflateInit err %d\n", err);
		return TEST_ERROR;
	}
	
	c_stream.next_in  = (z_const unsigned char *)src;
	c_stream.next_out = compr;

	c_stream.avail_in = src_len;
	c_stream.avail_out = compr_len;
	err = nx_deflate(&c_stream, Z_FINISH);

	assert(c_stream.total_in == src_len);
	assert(err == Z_STREAM_END);

	err = nx_deflateEnd(&c_stream);
	if (err != 0) {
		return TEST_ERROR;
	}

	return TEST_OK;
}

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

	if (_test_nx_deflatef(src, src_len, compr, compr_len)) goto err;
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

/* case prefix is 30 ~ 39 */
int run_case30()
{
	/* The total src buffer < default cache_threshold. */
	return run(5*1024, 1,  __func__);
}

int run_case31()
{
	/* The total src buffer > default cache_threshold. */
	return run(64*1024, 1, __func__);
}

/* A large buffer > fifo_in len and and avail_in == total */
int run_case32()
{
	return run(1024*1024*8, 1024*1024*8, __func__);
}

/* A large buffer > fifo_in len and and avail_in > 10*1024 */
int run_case33()
{
	return run(1024*33, 1024*32, __func__);
}

int run_case33_1()
{
	return run(1024*33, 1024*32, __func__);
}
/* A large buffer > fifo_in len and and avail_in == total */
int run_case34()
{
	return run(1024*1024*20, 1024*1024*20, __func__);
}

