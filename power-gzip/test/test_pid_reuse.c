/* Check if there is a bug with the kernel where it reuses a PID of a idle
 * thread with a VAS window open.
 * This test opens a VAS window and then fork other process to see if one of
 * the children can use the same kernel PID of the first one with the window
 * still open.
 *
 * Copyright (C) 2020 IBM Corporation
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
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include "zlib.h"

/* Globals shared by all threads. */
z_stream c_stream;

void* comp_file_multith (void *iterations)
{
	int err;
	long *sleep_sec;

	pid_t my_pid = syscall(SYS_gettid);
	fprintf(stderr, "thread started: pid: %d\n", (int) my_pid);

	/* Open the window. */
	err = deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
	if (err != 0) {
		printf("deflateInit err %d\n", err);
		return (void *) 1;
	}

	sleep_sec = (long *) iterations;
	sleep(sleep_sec[0]);

	fprintf(stderr, "thread %d exits\n", (int) my_pid);
	return (void *) 0;
}

void fork_children (int parent_pid) {
	int child_id, i = 0;
	int seconds = 10;
	int children = 2048;

	for (i=0; i < children; i++) {
		child_id = fork();

		if (child_id > 0) {
			/* PID is being reused. */
			if (child_id == parent_pid) {
				fprintf(stderr, "!!! pid %d reused (see ps)\n",
					child_id);
				/* Keep this child alive enough time to hit the
				 * csb error signal. */
				sleep(300);
			}
			continue;
		} else if (child_id == 0) {
			pid_t pid = getpid();
			fprintf(stderr, "%d: child(%d)\n", i, (int) pid);
			/* Wait enough time to fork all children. */
			sleep(seconds);
			/* Finish the deflate structures copied to each fork. */
			int err = deflateEnd(&c_stream);
			if (err != 0) {
				fprintf(stderr, "%d: child(%d) deflateEnd err %d\n", i, (int)pid, err);
				exit(-1);
			}
			fprintf(stderr, "%d: child(%d) exits\n", i, (int) pid);
			exit(0);
		} else if (child_id == -1) {
			fprintf(stderr, "fork failed: %s\n", strerror(errno));
			exit(-1);
		}
	}

}

int main(int argc, char **argv)
{
	int rc, compLen = 1024;
	Byte *compr;
	pthread_t thread;
	void *ret;
	long iterations;
	z_const char hello[] = "hello, hello!";

	if (argc == 2)
		iterations = atoi(argv[1]);
	else
		iterations = 2;

	/* Get parent thread kernel PID (userspace sees as Thread ID).  */
	pid_t my_pid = syscall(SYS_gettid);
	fprintf(stderr, "parent thread started: pid: %d\n", (int) my_pid);

	memset(&c_stream, 0, sizeof(c_stream));
	c_stream.zalloc = Z_NULL;
	c_stream.zfree = Z_NULL;
	c_stream.opaque = Z_NULL;

	rc = pthread_create(&thread, NULL, comp_file_multith,
			    (void *) &iterations);
	if (rc != 0) {
		fprintf(stderr, "error: pthread_create %d\n", rc);
		return rc;
	}

	/* Run the compression and leave the window open. */
	rc = pthread_join(thread, &ret);
	if (rc != 0) {
		fprintf(stderr, "error: pthread cannot be joined %p"
			"\n", ret);
		return rc;
	}

	/* Fork many children to see if any use the same PID as the parent. */
	fork_children((int) my_pid);
	/* Wait for all children to run. */
	sleep(60);

	compr = malloc(compLen);
	assert(compr != NULL);

	/* Try to trigger csb error signal. */
	c_stream.next_in  = (z_const unsigned char *) hello;
	c_stream.next_out = compr;

	c_stream.avail_in = c_stream.avail_out = compLen;
	rc = deflate(&c_stream, Z_NO_FLUSH);
	assert(rc == Z_OK || rc == Z_STREAM_END);
	fprintf(stderr, "rc = %d\n", rc);
	for (;;) {
		rc = deflate(&c_stream, Z_FINISH);
		if (rc == Z_STREAM_END)
			break;
	}

	/* Finish the deflate internal structures of the main thread.*/
	rc = deflateEnd(&c_stream);
	if (rc != 0) {
		fprintf(stderr, "deflateEnd failed %d\n", rc);
		return -1;
	}

	free(compr);
	/* Wait for all children to terminate. */
	do {
		rc = wait(NULL);
	} while(rc != -1 && errno != ECHILD);
	/* If we didn't hit a segmentation fault the test passed. */
	fprintf(stderr, "Success, parent thread exits.\n");
	return 0;
}
