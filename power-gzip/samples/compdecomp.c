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
 * Copyright (C) IBM Corporation, 2011-2017
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

int compress_file(int argc, char **argv)
{
	char *inbuf, *compbuf, *decompbuf;
	size_t inlen;
	size_t compbuf_len, decompbuf_len;
	uLongf compdata_len, decompdata_len;
	int iterations = 100;
	int i;
	struct timeval ts, te;
	double elapsed;
	
	if (argc != 2) {
		fprintf(stderr, "usage: %s <fname>\n", argv[0]);
		exit(-1);
	}

	if (read_alloc_input_file(argv[1], &inbuf, &inlen))
		exit(-1);
	fprintf(stderr, "file %s read, %ld bytes\n", argv[1], inlen);

	compbuf_len = 2*inlen;
	decompbuf_len = 2*inlen;

	/* alloc some scratch memory */
	assert(NULL != (compbuf = (char *)malloc(compbuf_len)));
	assert(NULL != (decompbuf = (char *)malloc(decompbuf_len)));

	/* **************** */
	
	/* compress */
	compdata_len = compbuf_len;
	/* compdata_len is the buffer length before the call; returned
	   as actual compressed data len on return */
	if (Z_OK != compress((Bytef *)compbuf, &compdata_len, (Bytef *)inbuf, (uLong)inlen) ) {
		fprintf(stderr, "compress error\n");
		return -1;
	}	
	fprintf(stderr, "compressed %ld to %ld bytes\n", (long)inlen, (long)compdata_len);		

	/* **************** */	
	
	/* uncompress */
	decompdata_len = decompbuf_len;
	/* decompdata_len is the buffer length before the call;
	   returned as actual uncompressed data len on return */	
	if (Z_OK != uncompress((Bytef *)decompbuf, &decompdata_len, (Bytef *)compbuf, (uLong)compdata_len) ) {
		fprintf(stderr, "uncompress error\n");
		return -1;
	}
	fprintf(stderr, "uncompressed %ld to %ld bytes\n", (long)compdata_len, (long)decompdata_len);		
	fflush(stderr);

	/* **************** */
	
	/* now do some timing runs; when we report bandwidth it's the
	   larger of the input and output; for compress it is input
	   size divided by time for decompress it is output size
	   divided by time */
	
	fprintf(stderr, "begin compressing %ld bytes %d times\n", (long)inlen, iterations);

	gettimeofday(&ts, NULL);

	for (i = 0; i < iterations; i++) {
		compdata_len = compbuf_len;
		if (Z_OK != compress((Bytef *)compbuf, &compdata_len, (Bytef *)inbuf, (uLong)inlen) ) {
			fprintf(stderr, "compress error\n");
			return -1;
		}
	}

	gettimeofday(&te, NULL);
	elapsed = (double) te.tv_sec + (double)te.tv_usec/1.0e6
		- (double) ts.tv_sec + (double)ts.tv_usec/1.0e6;
	fprintf(stderr, "compressed %ld bytes to %ld bytes %d times in %g seconds, %g GB/s\n",
		(long)inlen, (long)compdata_len, iterations, elapsed,
		(double)inlen * (double)iterations / elapsed / 1.0e9);	

	/* **************** */

	fprintf(stderr, "begin uncompressing %ld bytes %d times\n", (long)compdata_len, iterations);

	gettimeofday(&ts, NULL);

	for (i = 0; i < iterations; i++) {

		decompdata_len = decompbuf_len;
		/* decompdata_len is the buffer length before the call;
		   returned as actual uncompressed data len on return */	
		if (Z_OK != uncompress((Bytef *)decompbuf, &decompdata_len, (Bytef *)compbuf, (uLong)compdata_len) ) {
			fprintf(stderr, "uncompress error\n");
			return -1;
		}
	}

	gettimeofday(&te, NULL);
	elapsed = (double) te.tv_sec + (double)te.tv_usec/1.0e6
		- (double) ts.tv_sec + (double)ts.tv_usec/1.0e6;
	fprintf(stderr, "uncompressed %ld bytes to %ld bytes %d times in %g seconds, %g GB/s\n",
		(long) compdata_len, (long)decompdata_len, iterations, elapsed,
		(double)decompdata_len * (double)iterations / elapsed / 1.0e9);	
	
	return 0;
}



int main(int argc, char **argv)
{
	int rc;
	rc = compress_file(argc, argv);
	return rc;
}



