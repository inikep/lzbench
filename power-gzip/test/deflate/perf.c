#include "../test_deflate.h"
#include "../test_utils.h"
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

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

	if (_test_deflate(src, src_len, compr, &compr_len, step, NULL))
		goto err;
	/* Reset compr_len to its original value. */
	compr_len = src_len*2;
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

/* case prefix is 40 ~ 49 */

int run_case40()
{
	return run(64*1024, 1024*32, __func__);
}

int run_case41()
{
	#define THREAD_NUM 1
	pthread_t tid[THREAD_NUM];
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_create(&(tid[i]), NULL, (void*) run_case40, NULL) != 0) {
        		printf ("Create pthread1 error!\n");
		}
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(tid[i], NULL);
	}

	return 0;
}
