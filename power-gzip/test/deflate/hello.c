#include "../test_deflate.h"
#include "../test_utils.h"

static z_const char hello[] = "hello, hello!";

/* a simple hello, hello test */
int run_case1()
{
	Byte *compr, *uncompr;
	unsigned int comprLen = 10000*sizeof(int);
	uLong uncomprLen = comprLen;
	uLong len = (uLong) strlen(hello) + 1;
	compr = (Byte*)calloc((uInt)comprLen, 1);
	uncompr = (Byte*)calloc((uInt)uncomprLen, 1);

	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	/* Reset comprLen. */
	comprLen = 10000*sizeof(int);
	if (_test_nx_deflate((Byte *)hello, len, compr, &comprLen, 1, NULL))
		return TEST_ERROR;
	if (_test_inflate(compr, comprLen, uncompr, uncomprLen,
				(Byte *) hello, len, 1, NULL))
		return TEST_ERROR;
	if (_test_nx_inflate(compr, comprLen, uncompr, uncomprLen,
				(Byte *) hello, len, 1, Z_NO_FLUSH, NULL))
		return TEST_ERROR;

	free(compr);
	free(uncompr);
	printf("*** deflate %s passed\n", __func__);
	return TEST_OK;
}
