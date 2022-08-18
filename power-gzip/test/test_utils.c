#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <endian.h>
#include "test.h"
#include "test_utils.h"

Byte ran_data[DATA_MAX_LEN];

static char dict[] = {
	'a', 'b', 'c', 'd', 'e', 'f', 'g',
	'h', 'i', 'j', 'k', 'l', 'm', 'n',
	'o', 'p', 'q', 'r', 's', 't', 'u',
	'v', 'w', 'x', 'y', 'z',
	',', '.', '!', '?', '.', '{', '}'
};

/* Map to convert between a zlib return error code to its string
   equivalent. Should always be used with an index offset of 6.
   E.g. zret2str[ret+6] */
static const char *zret2str[] = {
	"Z_VERSION_ERROR",
	"Z_BUF_ERROR",
	"Z_MEM_ERROR",
	"Z_DATA_ERROR",
	"Z_STREAM_ERROR",
	"Z_ERRNO",
	"Z_OK",
	"Z_STREAM_END",
	"Z_NEED_DICT",
};

/* Check the return code of a zlib/libnxz API call with pretty printing */
void zcheck_internal(int retval, int expected, char* file, int line) {
	if (retval < -6 || retval > 2) {
		printf("%s:%d ERROR: Unknown return value: %d\n", file, line,
			retval);
		exit(TEST_ERROR);
	}
	if (retval != expected) {
		printf("%s:%d ERROR: Expected %s but got %s\n", file, line,
			zret2str[expected+6], zret2str[retval+6]);
			exit(TEST_ERROR);
	}
}

void generate_all_data(int len, char digit)
{
	assert(len > 0);

	srand(time(NULL));

	for (int i = 0; i < len; i++) {
		ran_data[i] = digit;
	}
}

void generate_random_data(int len)
{
	assert(len > 0);

	srand(time(NULL));

	for (int i = 0; i < len; i++) {
		ran_data[i] = dict[rand() % sizeof(dict)];
	}
}

Byte* generate_allocated_random_data(unsigned int len)
{
	assert(len > 0);

	Byte *data = malloc(len);
	if (data == NULL) return NULL;

	srand(time(NULL));

	for (int i = 0; i < len; i++) {
		data[i] = dict[rand() % sizeof(dict)];
	}
	return data;
}

int compare_data(Byte* src, Byte* dest, int len)
{
	for (int i = 0; i < len; i++) {
		if (src[i] != dest[i]) {
			printf(" src[%d] %02x != dest[%d] %02x \n", i, src[i], i, dest[i]);
			return TEST_ERROR;
		}
	}
	return TEST_OK;
}

/* TODO: mark these as static later. */
alloc_func zalloc = (alloc_func) 0;
free_func zfree = (free_func) 0;

/* Use nx to deflate. */
int _test_nx_deflate(Byte* src, unsigned int src_len, Byte* compr,
		     unsigned int *compr_len, int step,
		     struct f_interval * time)
{
	int err;
	z_stream c_stream;

	c_stream.zalloc = zalloc;
	c_stream.zfree = zfree;
	c_stream.opaque = (voidpf)0;

	if (time != NULL)
		gettimeofday(&time->init_start, NULL);

	err = nx_deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);

	if (time != NULL)
		gettimeofday(&time->init_end, NULL);

	if (err != 0) {
		printf("nx_deflateInit err %d\n", err);
		return TEST_ERROR;
	}

	c_stream.next_in  = src;
	c_stream.next_out = compr;

	if (time != NULL)
		gettimeofday(&time->start, NULL);

	while (c_stream.total_in != src_len
	       && c_stream.total_out < *compr_len) {
		step = (step < (src_len - c_stream.total_in))
		       ? (step) : (src_len - c_stream.total_in);
		c_stream.avail_in = c_stream.avail_out = step;
		err = nx_deflate(&c_stream, Z_NO_FLUSH);
		if (c_stream.total_in > src_len) break;
		if (err < 0) {
			printf("*** failed: nx_deflate returned %d\n", err);
			return TEST_ERROR;
		}
	}
	assert(c_stream.total_in == src_len);

	for (;;) {
		c_stream.avail_out = 1;
		err = nx_deflate(&c_stream, Z_FINISH);
		if (err == Z_STREAM_END) break;
		if (err < 0) {
			printf("*** failed: nx_deflate returned %d\n", err);
			return TEST_ERROR;
		}
	}

	if (time != NULL)
		gettimeofday(&time->end, NULL);

	err = nx_deflateEnd(&c_stream);
	if (err != 0)
		return TEST_ERROR;

	*compr_len = c_stream.total_out;
	return TEST_OK;
}

/* Use zlib to deflate. */
int _test_deflate(Byte* src, unsigned int src_len, Byte* compr,
		  unsigned int* compr_len, int step,
		  struct f_interval * time)
{
	int err;
	z_stream c_stream;

	c_stream.zalloc = zalloc;
	c_stream.zfree = zfree;
	c_stream.opaque = (voidpf)0;

	if (time != NULL)
		gettimeofday(&time->init_start, NULL);

	err = deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);

	if (time != NULL)
		gettimeofday(&time->init_end, NULL);

	if (err != 0) {
		printf("deflateInit err %d\n", err);
		return TEST_ERROR;
	}

	c_stream.next_in  = src;
	c_stream.next_out = compr;

	if (time != NULL)
		gettimeofday(&time->start, NULL);

	while (c_stream.total_in != src_len
	       && c_stream.total_out < *compr_len) {
		c_stream.avail_in = c_stream.avail_out = step;
		err = deflate(&c_stream, Z_NO_FLUSH);
		if (c_stream.total_in > src_len) break;
		if (err < 0) {
			printf("*** failed: deflate returned %d\n", err);
			return TEST_ERROR;
		}
	}
	assert(c_stream.total_in == src_len);

	for (;;) {
		c_stream.avail_out = 1;
		err = deflate(&c_stream, Z_FINISH);
		if (err == Z_STREAM_END) break;
		if (err < 0) {
			printf("*** failed: deflate returned %d\n", err);
			return TEST_ERROR;
		}
	}

	if (time != NULL)
		gettimeofday(&time->end, NULL);

	printf("\n*** c_stream.total_out %ld\n",
	       (unsigned long) c_stream.total_out);

	err = deflateEnd(&c_stream);
	if (err != 0)
		return TEST_ERROR;

	*compr_len = c_stream.total_out;
	return TEST_OK;
}

/* Use zlib to inflate. */
int _test_inflate(Byte* compr, unsigned int comprLen, Byte* uncompr,
		  unsigned int uncomprLen, Byte* src, unsigned int src_len,
		  int step, struct f_interval * time)
{
	int err;
	z_stream d_stream;

	memset(uncompr, 0, uncomprLen);

	d_stream.zalloc = zalloc;
	d_stream.zfree = zfree;
	d_stream.opaque = (voidpf)0;

	d_stream.next_in  = compr;
	d_stream.avail_in = 0;
	d_stream.next_out = uncompr;

	if (time != NULL)
		gettimeofday(&time->init_start, NULL);

	err = inflateInit(&d_stream);

	if (time != NULL)
		gettimeofday(&time->init_end, NULL);

	if (time != NULL)
		gettimeofday(&time->start, NULL);

	while (d_stream.total_out < uncomprLen) {
		if (d_stream.total_in < comprLen)
			d_stream.avail_in = step;
		d_stream.avail_out = step;
		err = inflate(&d_stream, Z_NO_FLUSH);
		if (err == Z_STREAM_END) break;
		if (err < 0) {
			printf("*** failed: inflate returned %d\n", err);
			return TEST_ERROR;
		}
	}

	if (time != NULL)
		gettimeofday(&time->end, NULL);

	printf("*** d_stream.total_in %ld d_stream.total_out %ld src_len %d\n",
	       (unsigned long) d_stream.total_in,
	       (unsigned long) d_stream.total_out, src_len);
	assert(d_stream.total_out == src_len);

	err = inflateEnd(&d_stream);

	if (compare_data(uncompr, src, src_len))
		return TEST_ERROR;

	return TEST_OK;
}

/* Use nx to inflate. */
int _test_nx_inflate(Byte* compr, unsigned int comprLen, Byte* uncompr,
		     unsigned int uncomprLen, Byte* src, unsigned int src_len,
		     int step, int flush, struct f_interval * time)
{
	int err;
	z_stream d_stream;

	memset(uncompr, 0, uncomprLen);

	d_stream.zalloc = zalloc;
	d_stream.zfree = zfree;
	d_stream.opaque = (voidpf)0;

	d_stream.next_in  = compr;
	d_stream.avail_in = 0;
	d_stream.next_out = uncompr;

	if (time != NULL)
		gettimeofday(&time->init_start, NULL);

	err = nx_inflateInit(&d_stream);

	if (time != NULL)
		gettimeofday(&time->init_end, NULL);

	if (time != NULL)
		gettimeofday(&time->start, NULL);

	while (d_stream.total_out < uncomprLen) {
		if (d_stream.total_in < comprLen)
			d_stream.avail_in = step;
		d_stream.avail_out = step;
		err = nx_inflate(&d_stream, flush);
		if (err == Z_STREAM_END) break;
		if (err < 0) {
			printf("*** failed: nx_inflate returned %d\n", err);
			return TEST_ERROR;
		}
	}

	if (time != NULL)
		gettimeofday(&time->end, NULL);

	printf("*** d_stream.total_in %ld d_stream.total_out %ld src_len %d\n",
	       (unsigned long) d_stream.total_in,
	       (unsigned long) d_stream.total_out, src_len);
	assert(d_stream.total_out == src_len);

	err = nx_inflateEnd(&d_stream);

	if (compare_data(uncompr, src, src_len))
		return TEST_ERROR;

	return TEST_OK;
}
