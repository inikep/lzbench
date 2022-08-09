#define _GNU_SOURCE
#include "gzip_simple.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>


pthread_barrier_t compress;
pthread_barrier_t decompress;
int options = 0;
char *_bufferSize;
char *_threads;
int bufferSize = 1;
int threads = 1;
char *file = NULL;
int opt = 0;
int nx_dbg = 0;
FILE *nx_gzip_log = NULL;

void *run(void *arg)
{
	void *handle = NULL;
	long bufferSizeinBytes;
	char *inBuffer;
	char *outBuffer;
	char *outBuffer2;
	int cpuid = 0;
	int threadNum = (int)arg;
	pid_t tid = syscall(SYS_gettid);

	syscall(SYS_getcpu, &cpuid, NULL, NULL);
	printf("thread no : %d with Thread ID : %d  Running on %d\n", threadNum,
	       (int)tid, cpuid);

	if (file != NULL) {
		// read from file
		FILE *f = fopen(file, "r");
		fseek(f, 0, SEEK_END);
		bufferSizeinBytes = ftell(f);
		printf("file name:%s  file size:%lu\n", file,
		       bufferSizeinBytes);
		fseek(f, 0, SEEK_SET);
		inBuffer = malloc(bufferSizeinBytes + 1);
		fread(inBuffer, bufferSizeinBytes, 1, f);
		fclose(f);
	} else if (bufferSize) {
		bufferSizeinBytes = bufferSize * 1024 * 1024;
		inBuffer = malloc(bufferSizeinBytes + 1);
		// create our own buffer
		int chunk = 4;
		time_t t;
		srand((unsigned)time(&t));
		char *pool = (char *)malloc(27);
		strcpy(pool, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
		size_t poolsize = strlen(pool);
		int i = 0;
		while (i <= bufferSizeinBytes) {
			int index = rand() % (poolsize - chunk);
			strncpy(inBuffer + i, pool + index, chunk);
			i = i + chunk;
		}
		free(pool);
	}
	outBuffer = malloc(bufferSizeinBytes + 20);
	outBuffer2 = malloc(bufferSizeinBytes + 20);
	memset(outBuffer, 0, bufferSizeinBytes + 20);
	memset(outBuffer2, 0, bufferSizeinBytes + 20);
	pthread_barrier_wait(&compress);

	/*open the compressor*/
	handle = p9open();
	if (!handle) {
		printf("open () retval = %d\n", errno);
		exit(-1);
	}

	struct timeval start, end;
	int elapsed, retval;
	gettimeofday(&start, NULL);
	retval = p9deflate(handle, inBuffer, outBuffer, bufferSizeinBytes,
			   bufferSizeinBytes, NULL, GZIP_WRAPPER);
	gettimeofday(&end, NULL);
	elapsed = (((end.tv_sec - start.tv_sec) * 1000000)
		   + (end.tv_usec - start.tv_usec));
	float sec, mb;
	sec = ((float)elapsed) / ((float)(1000000));
	mb = ((float)bufferSizeinBytes) / (1024 * 1024);
	printf("thread id %d: compression time %.4f ms throughput %.4f mb/sec\n",
	       threadNum, ((float)elapsed) / (float)(1000), mb / sec);

	if (retval == -1) {
		fprintf(stderr, "nx deflate error: %d\n", retval);
		exit(-1);
	}

        pthread_barrier_wait(&decompress);
	int compressed = retval;
	gettimeofday(&start, NULL);
	retval = p9inflate(handle, outBuffer, outBuffer2, compressed,
			   bufferSizeinBytes, GZIP_WRAPPER);
	gettimeofday(&end, NULL);
	elapsed = (((end.tv_sec - start.tv_sec) * 1000000)
		   + (end.tv_usec - start.tv_usec));
	sec = ((float)elapsed) / ((float)(1000000));
	mb = ((float)bufferSizeinBytes) / (1024 * 1024);
	printf("thread id %d: decompression time %.4f ms throughput %.4f mb/sec\n",
	       threadNum, ((float)elapsed) / (float)(1000), mb / sec);
	if (retval == -1) {
		fprintf(stderr, "nx inflate error: %d\n", retval);
		exit(-1);
	}

	int uncompressed = retval;
	printf("thread id %d: compression ratio:%.3f\n", threadNum,
	       (((float)uncompressed) / (float)compressed));
	retval = p9close(handle);
	if (retval < 0) {
		printf("close() retval = %d\n", retval);
	}
}

int main(int argc, char **argv)
{
	while ((options = getopt(argc, argv, "f:s:t:")) != -1) {
		switch (options) {
		case 'f':
			file = optarg;
			break;
		case 's':
			_bufferSize = optarg;
			break;
		case 't':
			_threads = optarg;
			break;
		default:
			fprintf(stderr,
				"Usage: %s [-f file] [-s buffer size in mb] [-t threads]...\n",
				argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if (_bufferSize != NULL) {
		bufferSize = atoi(_bufferSize);
	}
	if (_threads != NULL) {
		threads = atoi(_threads);
	}
	pthread_t *thread;
	int retval = 0;
	thread = (pthread_t *)malloc(threads * sizeof(pthread_t));
	pthread_barrier_init(&compress, NULL, threads);
	pthread_barrier_init(&decompress, NULL, threads);
	int i, j = 0;
	for (i = 1; i <= threads; i++) {
		pthread_attr_t attr;
		cpu_set_t cpus;
		pthread_attr_init(&attr);
		CPU_ZERO(&cpus);
		CPU_SET((i * 80) % 160 + j, &cpus);
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
		retval = pthread_create(&thread[i - 1], &attr, run, (void *)i);
		if (retval != 0) {
			printf("pthread_create failed\n", i);
			exit(-1);
		}
		if (i != 0 && i % 2 == 0) {
			j += 4;
		}
	}

	for (i = 0; i < threads; i++) {
		retval = pthread_join(thread[i], NULL);
		if (retval != 0) {
			printf("pthread_join failed in %d_th pass\n", i);
			exit(-1);
		}
	}
}
