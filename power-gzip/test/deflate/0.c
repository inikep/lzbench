#include "../test_deflate.h"
#include "../test_utils.h"

static int run(unsigned int len, int digit, const char* test)
{
	Byte *src, *compr, *uncompr;
	unsigned int src_len = len;
	unsigned int compr_len = src_len*2;
	unsigned int uncompr_len = src_len*2;

	src =(Byte*)calloc((uInt)len, 1);
	memset(src, 0, len);

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (src == NULL || compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	printf("*** src_len %d compr_len %d uncompr_len %d\n", src_len, compr_len, uncompr_len);
	if (_test_nx_deflate(src, src_len, compr, &compr_len, src_len, NULL))
		goto err;
	if (_test_inflate(compr, compr_len, uncompr, uncompr_len, src, src_len,
			  compr_len, NULL))
		goto err;
	if (_test_nx_inflate(compr, compr_len, uncompr, uncompr_len, src,
			     src_len, compr_len, Z_NO_FLUSH, NULL))
		goto err;

	printf("*** %s %s passed\n", __FILE__, test);
	free(compr);
	free(src);
	free(uncompr);
	return TEST_OK;
err:
	printf("*** %s %s failed\n", __FILE__, test);
	free(compr);
	free(src);
	free(uncompr);
	return TEST_ERROR;
}

/* case prefix is 21 - 29 */

int run_case21()
{
	return run(4*1024, 0,  __func__);
}
