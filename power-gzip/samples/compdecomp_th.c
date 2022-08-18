/*
 * P9 gunzip sample code for demonstrating the P9 NX hardware
 * interface.  Not intended for productive uses or for performance or
 * compression ratio measurements. For simplicity of demonstration,
 * this sample code compresses in to fixed Huffman blocks only
 * (Deflate btype=1) and has very simple memory management.  Dynamic
 * Huffman blocks (Deflate btype=2) are more involved as detailed in
 * the user guide.  Note also that /dev/crypto/gzip, VAS and skiboot
 * support are required (version numbers TBD)
 *
 * Changelog:
 *   2018-04-02 Initial version
 *
 *
 * Copyright (C) IBM Corporation, 2011-2020
 *
 * Licenses for GPLv2 and Apache v2.0:
 *
 * GPLv2:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Apache v2.0:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Bulent Abali <abali@us.ibm.com>
 *
 */

/* how to compile run:
   make
   cd samples
   make compdecomp_th
   sudo ./compdecomp_th <filename> <thread_count>
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <endian.h>
#include <bits/endian.h>
#include <sys/ioctl.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include "zlib.h"

/* Caller must free the allocated buffer 
   return nonzero on error */
int read_alloc_input_file(char *fname, char **buf, size_t *bufsize)
{
	struct stat statbuf;
	FILE *fp;
	char *p;
	size_t num_bytes;
	if (stat(fname, &statbuf)) {
		perror(fname);
		return(-1);
	}
	if (NULL == (fp = fopen(fname, "r"))) {
		perror(fname);
		return(-1);
	}
	assert(NULL != (p = (char *)malloc(statbuf.st_size)));
	num_bytes = fread(p, 1, statbuf.st_size, fp);
	if (ferror(fp) || (num_bytes != statbuf.st_size)) {
		perror(fname);
		return(-1);
	}
	*buf = p;
	*bufsize = num_bytes;
	return 0;
}

/* Returns nonzero on error */
int write_output_file(char *fname, char *buf, size_t bufsize)
{
	FILE *fp;
	size_t num_bytes;
	if (NULL == (fp = fopen(fname, "w"))) {
		perror(fname);
		return(-1);
	}
	num_bytes = fwrite(buf, 1, bufsize, fp);
	if (ferror(fp) || (num_bytes != bufsize)) {
		perror(fname);
		return(-1);
	}
	fclose(fp);
	return 0;
}

typedef struct thread_args_t {
	int my_id;
	char *inbuf;
	size_t inlen;
        long iterations;
	double elapsed_time;
	uint64_t checksum;
} thread_args_t;

#define TPRT fprintf(stderr,"tid %d: ", argsp->my_id)
#define DBGPRT(X) do {;}while(0)

pthread_barrier_t barr;


void *comp_file_multith(void *argsv)
{
	char *inbuf;
	size_t inlen;
	char *compbuf, *decompbuf;
	size_t compbuf_len, decompbuf_len;
	uLongf compdata_len, decompdata_len;
	long iterations;
	long i;
	struct timeval ts, te;
	double elapsed;
	thread_args_t *argsp = (thread_args_t *) argsv;
	int tid;

	inbuf = argsp->inbuf;
	inlen = argsp->inlen;	
	iterations = argsp->iterations;
	tid = argsp->my_id;
	
	compbuf_len = 2*inlen;
	decompbuf_len = 2*inlen;

	/* alloc some scratch memory */
	assert(NULL != (compbuf = (char *)malloc(compbuf_len)));
	assert(NULL != (decompbuf = (char *)malloc(decompbuf_len)));

	/* compress */
	compdata_len = compbuf_len;
	/* compdata_len is the buffer length before the call; returned
	   as actual compressed data len on return */
	if (Z_OK != compress((Bytef *)compbuf, &compdata_len, (Bytef *)inbuf, (uLong)inlen) ) {
		fprintf(stderr, "tid %d: compress error\n", tid);
		return (void *) -1;
	}	

	/* wait all threads to finish their first runs; want this for pretty printing */	
	pthread_barrier_wait(&barr);
	
	/* uncompress */
	decompdata_len = decompbuf_len;
	/* decompdata_len is the buffer length before the call;
	   returned as actual uncompressed data len on return */	
	if (Z_OK != uncompress((Bytef *)decompbuf, &decompdata_len, (Bytef *)compbuf, (uLong)compdata_len) ) {
		fprintf(stderr, "tid %d: uncompress error\n", tid);
		return (void *) -1;		
	}

	/* wait all threads to finish their first runs; */		
	pthread_barrier_wait(&barr);
	
	/* TIMING RUNS start here; when we report bandwidth it's the
	   larger of the input and output; for compress it is input
	   size divided by time for decompress it is output size
	   divided by time */

	gettimeofday(&ts, NULL);

	for (i = 0; i < iterations; i++) {
		compdata_len = compbuf_len;
		if (Z_OK != compress((Bytef *)compbuf, &compdata_len, (Bytef *)inbuf, (uLong)inlen) ) {
			fprintf(stderr, "tid %d: compress error\n", tid);
			return (void *) -1;					
		}
	}

	/* wait all threads to finish; min max not useful anymore since timer is after this barrier */	
	pthread_barrier_wait(&barr);	

	gettimeofday(&te, NULL);

	elapsed = ((double) te.tv_sec + (double)te.tv_usec/1.0e6) 
		- ((double) ts.tv_sec + (double)ts.tv_usec/1.0e6);
	argsp->elapsed_time = elapsed;

	if (tid == 0)
		fprintf(stderr, "tid %d: compressed %ld bytes %ld times in %7.4g seconds\n",
			tid, (long)inlen, iterations, elapsed);

	free(decompbuf);
	free(compbuf);		
	
	return (void *) -1;	
}


void *decomp_file_multith(void *argsv)
{
	char *inbuf;
	size_t inlen;
	char *compbuf, *decompbuf;
	size_t compbuf_len, decompbuf_len;
	uLongf compdata_len, decompdata_len;
	long iterations ;
	long i;
	struct timeval ts, te;
	double elapsed;
	thread_args_t *argsp = (thread_args_t *) argsv;
	int tid;
#ifdef SIMPLE_CHECKSUM
	unsigned long cksum = 1;
#endif

	inbuf = argsp->inbuf;
	inlen = argsp->inlen;	
	iterations = argsp->iterations;
	tid = argsp->my_id;
	
	compbuf_len = 2*inlen;
	decompbuf_len = 2*inlen;

	/* alloc some scratch memory */
	assert(NULL != (compbuf = (char *)malloc(compbuf_len)));
	assert(NULL != (decompbuf = (char *)malloc(decompbuf_len)));

	/* compress */
	compdata_len = compbuf_len;
	/* compdata_len is the buffer length before the call; returned
	   as actual compressed data len on return */
	if (Z_OK != compress((Bytef *)compbuf, &compdata_len, (Bytef *)inbuf, (uLong)inlen) ) {
		fprintf(stderr, "tid %d: compress error\n", tid);
		return (void *) -1;
	}	

	/* wait all threads to finish their first runs; want this for pretty printing */
	pthread_barrier_wait(&barr);
	
	/* uncompress */
	decompdata_len = decompbuf_len;
	/* decompdata_len is the buffer length before the call;
	   returned as actual uncompressed data len on return */	
	if (Z_OK != uncompress((Bytef *)decompbuf, &decompdata_len, (Bytef *)compbuf, (uLong)compdata_len) ) {
		fprintf(stderr, "uncompress error\n");
		return (void *) -1;		
	}

	/* wait all threads to finish their first runs; */	
	pthread_barrier_wait(&barr);
	
	/* TIMING RUNS start here; when we report bandwidth it's the
	   larger of the input and output; for compress it is input
	   size divided by time for decompress it is output size
	   divided by time */
	gettimeofday(&ts, NULL);

	for (i = 0; i < iterations; i++) {

		decompdata_len = decompbuf_len;
		/* decompdata_len is the buffer length before the call;
		   returned as actual uncompressed data len on return */	
		if (Z_OK != uncompress((Bytef *)decompbuf, &decompdata_len, (Bytef *)compbuf, (uLong)compdata_len) ) {
			fprintf(stderr, "tid %d: uncompress error\n", tid);
			return (void *) -1;
		}

#ifdef SIMPLE_CHECKSUM

		cksum = 1;		
		cksum = crc32(cksum, decompbuf, decompdata_len);
		assert( cksum == argsp->checksum );
#endif
		
	}

	/* wait all threads to finish; min max not useful anymore since timer is after this barrier */
	pthread_barrier_wait(&barr);
	
	gettimeofday(&te, NULL);

	elapsed = ((double) te.tv_sec + (double)te.tv_usec/1.0e6)
		- ((double) ts.tv_sec + (double)ts.tv_usec/1.0e6);
	argsp->elapsed_time = elapsed;	

	if (tid == 0)
		fprintf(stderr, "tid %d: uncompressed to %ld bytes %ld times in %7.4g seconds\n",
			tid, (long)decompdata_len, iterations, elapsed);

#ifdef SIMPLE_CHECKSUM
	cksum = 1;
	cksum = crc32(cksum, decompbuf, decompdata_len);
	assert( cksum == argsp->checksum );
#endif
	
	free(decompbuf);
	free(compbuf);		

	return (void *) -1;	
}

#define MAX_THREADS 1024

int main(int argc, char **argv)
{
	int rc;
	char *inbuf;
	size_t inlen;
	pthread_t threads[MAX_THREADS];
	thread_args_t th_args[MAX_THREADS];	
	int num_threads, i;
	void *ret;
	long iterations;
	double sum;
	
	if (argc != 3 && argc != 4) {
		fprintf(stderr, "usage: %s <fname> <thread_count> [<iterations>]\n", argv[0]);
		exit(-1);
	}
	assert( (num_threads = atoi(argv[2])) <= MAX_THREADS);
	
	if (read_alloc_input_file(argv[1], &inbuf, &inlen))
		exit(-1);
	fprintf(stderr, "file %s read, %ld bytes\n", argv[1], inlen);

	/* need this for pretty print */
	pthread_barrier_init(&barr, NULL, num_threads);

	if (argc == 4)
		iterations = atoi(argv[3]);
	else
		iterations = 100;

	unsigned long cksum = 1;	
#ifdef SIMPLE_CHECKSUM
	cksum = crc32(cksum, inbuf, inlen);	
	fprintf(stderr, "source checksum %08lx; note: checksum verif will reduce throughput; assert thrown on mismatch\n", cksum);
#endif
	
	fprintf(stderr, "starting %d compress threads %ld iterations\n", num_threads, iterations);
	for (i = 0; i < num_threads; i++) {
		th_args[i].inbuf = inbuf;
		th_args[i].inlen = inlen;
		th_args[i].checksum = cksum;				
		th_args[i].my_id = i;
		th_args[i].iterations = iterations;

		rc = pthread_create(&threads[i], NULL, comp_file_multith, (void *)&th_args[i]);
		if (rc != 0) {
			fprintf(stderr, "error: pthread_create %d\n", rc);
			return rc;
		}
	}

	/* wait for the threads to finish */
	for (i = 0; i < num_threads; i++) {
		rc = pthread_join(threads[i], &ret);
		if (rc != 0) {
			fprintf(stderr, "error: pthread %d cannot be joined %p\n", i, ret);
			return rc;
		}
	}

	/* report results */
	sum = 0;
	double maxbw = 0;
	double minbw = 1.0e20;
	for (i=0; i < num_threads; i++) {
		double gbps = (double)th_args[i].inlen * (double)th_args[i].iterations /
			(double)th_args[i].elapsed_time / 1.0e9;
		sum += gbps;
		if (gbps < minbw) minbw = gbps;
		if (gbps > maxbw) maxbw = gbps;
	}
	fprintf(stderr, "\nTotal compress throughput GB/s %7.4g, bytes %ld, iterations %ld, threads %d, per thread maxbw %7.4g, minbw %7.4g\n\n",
		sum, th_args[0].inlen, th_args[0].iterations, num_threads, maxbw, minbw);	


	
	fprintf(stderr, "starting %d uncompress threads\n", num_threads);
	for (i = 0; i < num_threads; i++) {
		th_args[i].inbuf = inbuf;
		th_args[i].inlen = inlen;
		th_args[i].checksum = cksum;						
		th_args[i].my_id = i;
		th_args[i].iterations = iterations;

		rc = pthread_create(&threads[i], NULL, decomp_file_multith, (void *)&th_args[i]);
		if (rc != 0) {
			fprintf(stderr, "error: pthread_create %d\n", rc);
			return rc;
		}
	}

	/* wait for the threads to finish */
	for (i = 0; i < num_threads; i++) {
		rc = pthread_join(threads[i], &ret);
		if (rc != 0) {
			fprintf(stderr, "error: pthread %d cannot be joined %p\n", i, ret);
			return rc;
		}

	}

	/* report results */
	sum = 0;
	maxbw = 0;
	minbw = 1.0e20;	
	for (i=0; i < num_threads; i++) {
		double gbps = (double)th_args[i].inlen * (double)th_args[i].iterations /
			(double)th_args[i].elapsed_time / 1.0e9;
		sum += gbps;
		if (gbps < minbw) minbw = gbps;
		if (gbps > maxbw) maxbw = gbps;
	}
	fprintf(stderr, "\nTotal uncompress throughput GB/s %7.4g, bytes %ld, iterations %ld, threads %d, per thread maxbw %7.4g, minbw %7.4g\n\n",
		sum, th_args[0].inlen, th_args[0].iterations, num_threads, maxbw, minbw);	
	
	return rc;
}



