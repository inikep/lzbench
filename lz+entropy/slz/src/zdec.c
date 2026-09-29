/*
 * Copyright (C) 2026 HAProxy Technologies
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "slz.h"

#ifndef SLZ_VERSION
#define SLZ_VERSION "unknown"
#endif

/* the ring must be at least 32 kB, see uslz_init() */
#define MIN_RING 32768

static void usage(const char *name, int ret)
{
	printf("zdec version %s\n"
	       "Usage: %s [option]* [file]\n"
	       "\n"
	       "Decompresses <file> or stdin to stdout. The stream format is\n"
	       "detected automatically unless one of -D/-G/-Z is given.\n"
	       "\n"
	       "The following arguments are supported :\n"
	       "  -b <size>  use a <size> bytes output ring (default %d, min %d)\n"
	       "  -c         send output to stdout [default]\n"
	       "  -h         display this help\n"
	       "  -i <size>  read the input by chunks of <size> bytes (default %d)\n"
	       "  -l <loops> loop <loops> times over the same file (needs a seekable input)\n"
	       "  -t         test mode: decompress but do not emit anything\n"
	       "  -v         increase verbosity\n"
	       "\n"
	       "  -D         input is raw Deflate (RFC1951), which cannot be detected\n"
	       "  -G         input is Gzip (RFC1952)\n"
	       "  -Z         input is Zlib (RFC1950)\n"
	       "\n", SLZ_VERSION, name, MIN_RING, MIN_RING, 8192);
	exit(ret);
}

int main(int argc, char **argv)
{
	unsigned char *obuf, *inbuf;
	const char *file = NULL;
	struct uslz_stream state;
	long ring_size = MIN_RING;
	long in_size = 8192;
	long decoded_total;
	int test_only = 0;
	int verbose = 0;
	int format = -1;       /* -1 = detect */
	int loops = 1;
	int fd = 0;
	int ret;

	while (argc > 1 && argv[1][0] == '-') {
		if (strcmp(argv[1], "-b") == 0 && argc > 2) {
			ring_size = atol(argv[2]);
			argv++; argc--;
		}
		else if (strcmp(argv[1], "-i") == 0 && argc > 2) {
			in_size = atol(argv[2]);
			argv++; argc--;
		}
		else if (strcmp(argv[1], "-l") == 0 && argc > 2) {
			loops = atoi(argv[2]);
			argv++; argc--;
		}
		else if (strcmp(argv[1], "-t") == 0)
			test_only = 1;
		else if (strcmp(argv[1], "-c") == 0)
			test_only = 0;
		else if (strcmp(argv[1], "-v") == 0)
			verbose++;
		else if (strcmp(argv[1], "-D") == 0)
			format = SLZ_FMT_DEFLATE;
		else if (strcmp(argv[1], "-G") == 0)
			format = SLZ_FMT_GZIP;
		else if (strcmp(argv[1], "-Z") == 0)
			format = SLZ_FMT_ZLIB;
		else if (strcmp(argv[1], "-h") == 0)
			usage(argv[0], 0);
		else {
			fprintf(stderr, "unknown option '%s'\n", argv[1]);
			usage(argv[0], 1);
		}
		argv++; argc--;
	}

	if (argc > 1)
		file = argv[1];

	if (ring_size < MIN_RING || in_size < 1) {
		fprintf(stderr, "ring size must be at least %d and input chunk at least 1\n", MIN_RING);
		exit(1);
	}

	obuf = malloc(ring_size);
	inbuf = malloc(in_size);
	if (!obuf || !inbuf) {
		fprintf(stderr, "out of memory\n");
		exit(1);
	}

	if (file) {
		fd = open(file, O_RDONLY);
		if (fd < 0) {
			perror("open");
			exit(1);
		}
	}

	if (loops > 1 && lseek(fd, 0, SEEK_SET) == (off_t)-1) {
		fprintf(stderr, "input is not seekable, cannot loop\n");
		exit(1);
	}

	while (loops--) {
		int done = 0;

		if (lseek(fd, 0, SEEK_SET) == (off_t)-1 && loops) {
			fprintf(stderr, "input is not seekable, cannot loop\n");
			exit(1);
		}

		decoded_total = 0;

		if (format < 0)
			ret = uslz_init(&state, obuf, ring_size);
		else
			ret = uslz_init_fmt(&state, obuf, ring_size, format);

		if (!ret) {
			fprintf(stderr, "failed to init uslz state\n");
			exit(1);
		}

		while (!done) {
			unsigned char *input;
			unsigned char *decoded;
			long decoded_len;
			long consumed;
			enum uslz_decode_ret decode_ret;
			long rret;

			rret = read(fd, inbuf, in_size);
			if (rret <= 0) {
				if (rret < 0)
					perror("read");
				else
					fprintf(stderr, "truncated stream\n");
				exit(1);
			}
			input = inbuf;
 redo:
			decode_ret = uslz_decode(&state, input, rret, &decoded,
			                         &decoded_len, &consumed, NULL);
			switch (decode_ret) {
			case USLZ_DECODE_SUCCESS:
			case USLZ_DECODE_OUT_OF_SPACE:
			case USLZ_DECODE_OUT_OF_DATA:
				decoded_total += decoded_len;

				while (!test_only && decoded_len) {
					long wret = write(1, decoded, decoded_len);

					if (wret < 0) {
						perror("write");
						exit(1);
					}
					decoded += wret;
					decoded_len -= wret;
				}

				input += consumed;
				rret -= consumed;

				/* A gzip stream may be a series of members, so
				 * keep feeding while there is input left rather
				 * than stopping on the first success.
				 */
				if (decode_ret == USLZ_DECODE_SUCCESS && !rret)
					done = 1;
				else if (decode_ret != USLZ_DECODE_OUT_OF_DATA)
					goto redo;
				break;
			default:
				fprintf(stderr, "error (%d)\n", decode_ret);
				exit(1);
			}
		}

		if (verbose)
			fprintf(stderr, "totout=%ld\n", decoded_total);
	}
	return 0;
}
