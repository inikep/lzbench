#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include "test.h"
#include "test_utils.h"

#define THREAD_MAX 60

static int ready_count;

pthread_t tid[THREAD_MAX];
pthread_mutex_t mutex;
pthread_cond_t cond;

struct use_time {
		struct f_interval nx_deflate;
		struct f_interval nx_inflate;
		struct f_interval deflate;
		struct f_interval inflate;
};
static struct use_time count_info[THREAD_MAX];
struct time_duration {
	float nx_deflate_init;
	float nx_deflate;
	float nx_inflate_init;
	float nx_inflate;
	float deflate_init;
	float deflate;
	float inflate_init;
	float inflate;
} duration[THREAD_MAX+3]; /* last is average, min, max */


static struct use_time* get_use_time_by_tid(pthread_t id)
{
	for (int i = 0; i < THREAD_MAX; i++){
		if (tid[i] == id)
			return count_info + i;
	}
	/* This should never happen. */
	assert(0);
	return NULL;
}

static int run(unsigned int len, int step, const char* test)
{
	Byte *src, *compr, *uncompr;
	unsigned int src_len = len;
	unsigned int compr_len = src_len*2;
	unsigned int uncompr_len = src_len*2;

	src = generate_allocated_random_data(src_len);
	if (src == NULL) return TEST_ERROR;

	compr = (Byte*)calloc((uInt)compr_len, 1);
	uncompr = (Byte*)calloc((uInt)uncompr_len, 1);
	if (compr == NULL || uncompr == NULL ) {
		printf("*** alloc buffer failed\n");
		return TEST_ERROR;
	}

	/* make sure all thread goto the deflate and inflate at the same time */
	pthread_mutex_lock(&mutex);  
	ready_count--;
	printf("thread %lu is ready\n", pthread_self());  
	if (ready_count == 0) {
		pthread_cond_broadcast(&cond);
	}
	else {
		pthread_cond_wait(&cond, &mutex);
	}
	pthread_mutex_unlock(&mutex);

	if (_test_deflate(src, src_len, compr, &compr_len, step,
			  &get_use_time_by_tid(pthread_self())->deflate))
		goto err;
	/* Reset compr_len. */
	compr_len = src_len*2;
	if (_test_nx_deflate(src, src_len, compr, &compr_len, step,
			     &get_use_time_by_tid(pthread_self())->nx_deflate))
		goto err;
	if (_test_inflate(compr, compr_len, uncompr, uncompr_len, src,
			  src_len, step,
			  &get_use_time_by_tid(pthread_self())->inflate))
		goto err;
	if (_test_nx_inflate(compr, compr_len, uncompr, uncompr_len, src,
			     src_len, step, Z_NO_FLUSH,
			     &get_use_time_by_tid(pthread_self())->nx_inflate))
		goto err;

	free(src);
	free(compr);
	free(uncompr);
	return TEST_OK;
err:
	free(src);
	free(compr);
	free(uncompr);
	return TEST_ERROR;
}

static int run_case()
{
	run(64*1024, 1024*32, __func__);
	return 0;
}

static float get_time_duration(struct timeval e, struct timeval s)
{
	return ((e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec)/1000.0);
}

int main()
{

	int thread_num = THREAD_MAX;
	ready_count = thread_num;

	pthread_mutex_init(&mutex, NULL);  
	pthread_cond_init(&cond, NULL);

	for (int i = 0; i < thread_num; i++) {
		if (pthread_create(&(tid[i]), NULL, (void*) run_case, NULL) != 0) {
        		printf ("Create pthread1 error!\n");
		}
	}

	for (int i = 0; i < thread_num; i++) {
		pthread_join(tid[i], NULL);
	}

	for (int i = 0; i < thread_num; i++) {
		duration[i].deflate_init =
			get_time_duration(count_info[i].deflate.init_end,
					  count_info[i].deflate.init_start);
		duration[i].deflate =
			get_time_duration(count_info[i].deflate.end,
					  count_info[i].deflate.start);
		duration[i].inflate_init =
			get_time_duration(count_info[i].inflate.init_end,
					  count_info[i].inflate.init_start);
		duration[i].inflate =
			get_time_duration(count_info[i].inflate.end,
					  count_info[i].inflate.start);
		duration[i].nx_deflate_init =
			get_time_duration(count_info[i].nx_deflate.init_end,
					  count_info[i].nx_deflate.init_start);
		duration[i].nx_deflate =
			get_time_duration(count_info[i].nx_deflate.end,
					  count_info[i].nx_deflate.start);
		duration[i].nx_inflate_init =
			get_time_duration(count_info[i].nx_inflate.init_end,
					  count_info[i].nx_inflate.init_start);
		duration[i].nx_inflate =
			get_time_duration(count_info[i].nx_inflate.end,
					  count_info[i].nx_inflate.start);
	}

	for (int i = 0; i < thread_num; i++) {
		duration[thread_num+0].deflate_init = MIN(duration[thread_num+0].deflate_init, duration[i].deflate_init);
		duration[thread_num+1].deflate_init = MAX(duration[thread_num+1].deflate_init, duration[i].deflate_init);
		duration[thread_num+2].deflate_init += duration[i].deflate_init;
		duration[thread_num+0].inflate_init = MIN(duration[thread_num+0].inflate_init, duration[i].inflate_init);
		duration[thread_num+1].inflate_init = MAX(duration[thread_num+1].inflate_init, duration[i].inflate_init);
		duration[thread_num+2].inflate_init += duration[i].inflate_init;

		duration[thread_num+0].deflate = MIN(duration[thread_num+0].deflate, duration[i].deflate);
		duration[thread_num+1].deflate = MAX(duration[thread_num+1].deflate, duration[i].deflate);
		duration[thread_num+2].deflate += duration[i].deflate;
		duration[thread_num+0].inflate = MIN(duration[thread_num+0].inflate, duration[i].inflate);
		duration[thread_num+1].inflate = MAX(duration[thread_num+1].inflate, duration[i].inflate);
		duration[thread_num+2].inflate += duration[i].inflate;

		duration[thread_num+0].nx_deflate_init = MIN(duration[thread_num+0].nx_deflate_init, duration[i].nx_deflate_init);
		duration[thread_num+1].nx_deflate_init = MAX(duration[thread_num+1].nx_deflate_init, duration[i].nx_deflate_init);
		duration[thread_num+2].nx_deflate_init += duration[i].nx_deflate_init;
		duration[thread_num+0].nx_inflate_init = MIN(duration[thread_num+0].nx_inflate_init, duration[i].nx_inflate_init);
		duration[thread_num+1].nx_inflate_init = MAX(duration[thread_num+1].nx_inflate_init, duration[i].nx_inflate_init);
		duration[thread_num+2].nx_inflate_init += duration[i].nx_inflate_init;

		duration[thread_num+0].nx_deflate = MIN(duration[thread_num+0].nx_deflate, duration[i].nx_deflate);
		duration[thread_num+1].nx_deflate = MAX(duration[thread_num+1].nx_deflate, duration[i].nx_deflate);
		duration[thread_num+2].nx_deflate += duration[i].nx_deflate;
		duration[thread_num+0].nx_inflate = MIN(duration[thread_num+0].nx_inflate, duration[i].nx_inflate);
		duration[thread_num+1].nx_inflate = MAX(duration[thread_num+1].nx_inflate, duration[i].nx_inflate);
		duration[thread_num+2].nx_inflate += duration[i].nx_inflate;
	}
/*
	for (int i = 0; i < thread_num; i++) {
		printf("--------------------- thread %d ---------------------\n", i);
		printf("deflate_init %f ms nx_deflate_init %f ms\n", duration[i].deflate_init, duration[i].nx_deflate_init);
		printf("deflate      %f ms nx_deflate      %f ms\n", duration[i].deflate, duration[i].nx_deflate); 
		printf("inflate_init %f ms nx_inflate_init %f ms\n", duration[i].inflate_init, duration[i].nx_inflate_init);
		printf("inflate      %f ms nx_inflate      %f ms\n", duration[i].inflate, duration[i].nx_inflate);
		printf("\n");
	}
*/
	printf("Thread number %d\n", thread_num);
	printf("------------------------ min ------------------------\n");
	printf("deflate_init %f ms nx_deflate_init %f ms\n", duration[thread_num+0].deflate_init, duration[thread_num+0].nx_deflate_init);
	printf("deflate      %f ms nx_deflate      %f ms\n", duration[thread_num+0].deflate, duration[thread_num+0].nx_deflate); 
	printf("inflate_init %f ms nx_inflate_init %f ms\n", duration[thread_num+0].inflate_init, duration[thread_num+0].nx_inflate_init);
	printf("inflate      %f ms nx_inflate      %f ms\n", duration[thread_num+0].inflate, duration[thread_num+0].nx_inflate);
	printf("\n");

	printf("------------------------ max ------------------------\n");
	printf("deflate_init %f ms nx_deflate_init %f ms\n", duration[thread_num+1].deflate_init, duration[thread_num+1].nx_deflate_init);
	printf("deflate      %f ms nx_deflate      %f ms\n", duration[thread_num+1].deflate, duration[thread_num+1].nx_deflate); 
	printf("inflate_init %f ms nx_inflate_init %f ms\n", duration[thread_num+1].inflate_init, duration[thread_num+1].nx_inflate_init);
	printf("inflate      %f ms nx_inflate      %f ms\n", duration[thread_num+1].inflate, duration[thread_num+1].nx_inflate);
	printf("\n");

	printf("------------------------ avg ------------------------\n");
	printf("deflate_init %f ms nx_deflate_init %f ms\n", duration[thread_num+2].deflate_init/thread_num, duration[thread_num+2].nx_deflate_init/thread_num);
	printf("deflate      %f ms nx_deflate      %f ms\n", duration[thread_num+2].deflate/thread_num, duration[thread_num+2].nx_deflate/thread_num); 
	printf("inflate_init %f ms nx_inflate_init %f ms\n", duration[thread_num+2].inflate_init/thread_num, duration[thread_num+2].nx_inflate_init/thread_num);
	printf("inflate      %f ms nx_inflate      %f ms\n", duration[thread_num+2].inflate/thread_num, duration[thread_num+2].nx_inflate/thread_num);
}

