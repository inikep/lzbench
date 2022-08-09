/*
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

/*
 * zpipe.c: example of proper use of zlib's inflate() and deflate()
 * Not copyrighted -- provided to the public domain
 * Version 1.4  11 December 2005  Mark Adler
 */

/*
 * Most of the code comes from GenWQE Library:
 * https://github.com/ibm-genwqe/genwqe-user/blob/master/tools/genwqe_gzip.c
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <errno.h>
#include <limits.h>
#include <endian.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <asm/byteorder.h>

#include <sched.h>

#include <zlib.h>

#define SET_BINARY_MODE(file)

/** common error printf */
#define pr_err(fmt, ...) do {					\
		fprintf(stderr, "gzip: " fmt, ## __VA_ARGS__);	\
	} while (0)

static const char *version = "1.0";

#define EX_ERRNO	79 /* libc problem */

static int verbose = 0;

/* Default Buffer Size */
static unsigned int CHUNK_i = 128 * 1024; /* 128 KiB; */
static unsigned int CHUNK_o = 128 * 1024; /* 128 KiB; */


/* Compress from file source to file dest until EOF on source.
   def() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_STREAM_ERROR if an invalid compression
   level is supplied, Z_VERSION_ERROR if the version of zlib.h and the
   version of the library linked do not match, or Z_ERRNO if there is
   an error reading or writing the files. */
static int def(FILE *source, FILE *dest, z_stream *strm,
	       unsigned char *in, unsigned char *out)
{
	int ret, flush;
	unsigned have;

	/* compress until end of file */
	do {
		strm->avail_in = fread(in, 1, CHUNK_i, source);
		if (ferror(source)) {
			return Z_ERRNO;
		}
		flush = feof(source) ? Z_FINISH : Z_NO_FLUSH;
		strm->next_in = in;

		/* run deflate() on input until output buffer not full, finish
		   compression if all of source has been read in */
		do {
			if (verbose)
				fprintf(stderr, "CHUNK_o=%d\n", CHUNK_o);

			strm->avail_out = CHUNK_o;
			strm->next_out = out;
			ret = deflate(strm, flush);	/* no bad ret value */
			assert(ret != Z_STREAM_ERROR);	/* not clobbered */
			have = CHUNK_o - strm->avail_out;
			if (fwrite(out, 1, have, dest) != have ||
				ferror(dest)) {
				return Z_ERRNO;
			}
		} while (strm->avail_out == 0);
		assert(strm->avail_in == 0);	/* all input will be used */

		/* done when last data in file processed */
	} while (flush != Z_FINISH);
	assert(ret == Z_STREAM_END);	    /* stream will be complete */

	return Z_OK;
}

/* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
static int inf(FILE *source, FILE *dest, z_stream *strm,
	       unsigned char *in, unsigned char *out)
{
	int ret = Z_OK;
	int rc;
	long start_offs;
	long read_offs = 0;
	long have;

	strm->avail_in = 0;
	strm->next_in = Z_NULL;

	start_offs = ftell(source);
	read_offs = 0;

	/* decompress until deflate stream ends or end of file */
	do {
		strm->avail_in = fread(in, 1, CHUNK_i, source);
		if (ferror(source)) {
			fprintf(stderr, "fread error\n");
			return Z_ERRNO;
		}
		if (0 == strm->avail_in)
			break;
		strm->next_in = in;
__more_inf:
		/* run inflate() on input until output buffer not full */
		do {
			strm->avail_out = CHUNK_o;
			strm->next_out = out;
			ret = inflate(strm, Z_NO_FLUSH /* Z_SYNC_FLUSH */);
			assert(ret != Z_STREAM_ERROR);	/* not clobbered */

			switch (ret) {
			case Z_OK:
				/* Need to continue with Read more data */
				break;
			case Z_STREAM_END:
				read_offs += strm->total_in;
				break;
			case Z_NEED_DICT:
				fprintf(stderr, "NEED Dict........\n");
				return Z_DATA_ERROR;
			case Z_DATA_ERROR:
			case Z_MEM_ERROR:
				fprintf(stderr, "Fault..... %d\n", ret);
				return ret;
			}
			have = CHUNK_o - strm->avail_out;
			if (fwrite(out, 1, have, dest) != (size_t)have ||
				ferror(dest)) {
				fprintf(stderr, "fwrite fault\n");
				return Z_ERRNO;
			}
		} while (strm->avail_out == 0);
		/* done when inflate() says it's done */
	} while (ret != Z_STREAM_END);

	/* FIXME: this goto and the limit check is not nice. */
	if (strm->avail_in > (16 * 1024)) {
		inflateReset(strm);	/* reset and continue */
		goto __more_inf;
	}

	/* Set the file position right after the absorbed input */
	start_offs += read_offs;	/* Add to seek offset */
	rc = fseek(source, start_offs, SEEK_SET);
	if (rc == -1)
		fprintf(stderr, "err: fseek rc=%d\n", rc);

	inflateReset(strm);
	return ret;
}

/* report a zlib or i/o error */
static void zerr(int ret)
{
	switch (ret) {
	case Z_ERRNO:
		if (ferror(stdin))
			pr_err("error reading stdin\n");
		if (ferror(stdout))
			pr_err("error writing stdout\n");
		break;
	case Z_STREAM_ERROR:
		pr_err("invalid compression level\n");
		break;
	case Z_DATA_ERROR:
		pr_err("invalid or incomplete deflate data\n");
		break;
	case Z_MEM_ERROR:
		pr_err("out of memory\n");
		break;
	case Z_VERSION_ERROR:
		pr_err("zlib version mismatch!\n");
		break;
	}
}

/**
 * str_to_num() - Convert string into number and cope with endings like
 *              KiB for kilobyte
 *              MiB for megabyte
 *              GiB for gigabyte
 */
static inline uint64_t str_to_num(char *str)
{
	char *s = str;
	uint64_t num = strtoull(s, &s, 0);

	if (*s == '\0')
		return num;

	if (strcmp(s, "KiB") == 0)
		num *= 1024;
	else if (strcmp(s, "MiB") == 0)
		num *= 1024 * 1024;
	else if (strcmp(s, "GiB") == 0)
		num *= 1024 * 1024 * 1024;

	return num;
}

static void userinfo(FILE *fp, char *prog, const char *version)
{
	fprintf(fp, "%s %s\n(c) Copyright IBM Corp. 2011, 2017\n",
		basename(prog), version);
}

static void print_args(FILE *fp, int argc, char **argv)
{
	int i;

	fprintf(fp, "Called with:\n");
	for (i = 0; i < argc; i++)
		fprintf(fp, "  ARGV[%d]: \"%s\"\n", i, argv[i]);
	fprintf(fp, "\n");
}

static void print_version(FILE *fp)
{
	fprintf(fp, "Code: zlibVersion()=%s Header: ZLIB_VERSION=%s %s\n\n",
		zlibVersion(), ZLIB_VERSION,
		strcmp(zlibVersion(), ZLIB_VERSION) == 0 ?
		"consistent" : "inconsistent");
}

static void usage(FILE *fp, char *prog, int argc, char *argv[])
{
	fprintf(fp, "Usage: %s [OPTION]... [FILE]...\n"
		"Compress or uncompress FILEs (by default, compress FILES in-place).\n"
		"\n"
		"Mandatory arguments to long options are mandatory for short options too.\n"
		"\n"
		"  -c, --stdout      write on standard output, keep original files unchanged\n"
		"  -d, --decompress  decompress\n"
		"  -f, --force       force overwrite of output file and compress links\n"
		"  -h, --help        give this help\n"
		"  -l, --list        list compressed file contents\n"
		"  -L, --license     display software license\n"
		"  -N, --name        save or restore the original name and time stamp\n"
		"  -q, --quiet       suppress all warnings\n"
		"  -S, --suffix=SUF  use suffix SUF on compressed files\n"
		"  -v, --verbose     verbose mode\n"
		"  -V, --version     display version number\n"
		"  -1, --fast        compress faster\n"
		"  -9, --best        compress better\n"
		"\n"
		"Special options for testing and debugging:\n"
		"  -i, --i_bufsize   input buffer size (%d KiB)\n"
		"  -o, --o_bufsize   output buffer size (%d KiB)\n"
		"  -N, --name=NAME   write NAME into gzip header\n"
		"  -C, --comment=CM  write CM into gzip header\n"
		"  -E, --extra=EXTRA write EXTRA (file) into gzip header\n"
		"\n"
		"With no FILE, or when FILE is -, read standard input.\n"
		"\n"
		"NOTE: Not all options are supported in this limited version!\n"
		"Suggestions or patches are welcome!\n"
		"\n"
		"\n", prog, CHUNK_i/1024, CHUNK_o/1024);

	print_version(fp);
	print_args(fp, argc, argv);
}

static inline void hexdump(FILE *fp, const void *buff, unsigned int size)
{
	unsigned int i, j = 0;
	const uint8_t *b = (uint8_t *)buff;
	char ascii[17];

	if (size == 0)
		return;

	for (i = 0; i < size; i++) {
		if ((i & 0x0f) == 0x00) {
			fprintf(fp, " %08x:", i);
			memset(ascii, '\0', sizeof(ascii));
			j = 0;
		}
		fprintf(fp, " %02x", b[i]);
		ascii[j++] = isalnum(b[i]) ? b[i] : '.';

		if ((i & 0x0f) == 0x0f)
			fprintf(fp, " | %s\n", ascii);
	}

	/* print trailing up to a 16 byte boundary. */
	for (; i < ((size + 0xf) & ~0xf); i++) {
		fprintf(fp, "   ");
		ascii[j++] = ' ';

		if ((i & 0x0f) == 0x0f)
			fprintf(fp, " | %s\n", ascii);
	}

	fprintf(fp, "\n");
}

static void do_print_gzip_hdr(gz_headerp head, FILE *fp)
{
	fprintf(fp, "GZIP Header\n"
		" Text:        %01X\n", head->text);
	fprintf(fp, " Time:        %s", ctime((time_t*) &head->time));
	fprintf(fp, " xflags:      %08X\n", head->xflags);
	fprintf(fp, " OS:          %01X (0x03 Linux per RFC1952)\n", head->os);
	fprintf(fp, " Extra Len:   %d\n", head->extra_len);
	fprintf(fp, " Extra Max:   %d\n", head->extra_max);
	hexdump(fp, head->extra, head->extra_len);
	fprintf(fp, " Name:        %s\n",
		head->name ? (char *)head->name : "");
	fprintf(fp, " Name Max:    %d\n", head->name_max);
	fprintf(fp, " Comment:     %s\n",
		head->comment ? (char *)head->comment : "");
	fprintf(fp, " Comment Max: %d\n", head->comm_max);
	fprintf(fp, " Header CRC : %X\n", head->hcrc);
	fprintf(fp, " Done:        %01X\n", head->done);
}

static inline
ssize_t file_size(const char *fname)
{
	int rc;
	struct stat s;

	rc = lstat(fname, &s);
	if (rc != 0) {
		fprintf(stderr, "err: Cannot find %s!\n", fname);
		return rc;
	}

	return s.st_size;
}

static inline ssize_t
file_read(const char *fname, uint8_t *buff, size_t len)
{
	int rc;
	FILE *fp;

	if ((fname == NULL) || (buff == NULL) || (len == 0))
		return -EINVAL;

	fp = fopen(fname, "r");
	if (!fp) {
		fprintf(stderr, "err: Cannot open file %s: %s\n",
			fname, strerror(errno));
		return -ENODEV;
	}
	rc = fread(buff, len, 1, fp);
	if (rc == -1) {
		fprintf(stderr, "err: Cannot read from %s: %s\n",
			fname, strerror(errno));
		fclose(fp);
		return -EIO;
	}

	fclose(fp);
	return rc;
}

static inline ssize_t
file_write(const char *fname, const uint8_t *buff, size_t len)
{
	int rc;
	FILE *fp;

	if ((fname == NULL) || (buff == NULL) || (len == 0))
		return -EINVAL;

	fp = fopen(fname, "w+");
	if (!fp) {
		fprintf(stderr, "err: Cannot open file %s: %s\n",
			fname, strerror(errno));
		return -ENODEV;
	}
	rc = fwrite(buff, len, 1, fp);
	if (rc == -1) {
		fprintf(stderr, "err: Cannot write to %s: %s\n",
			fname, strerror(errno));
		fclose(fp);
		return -EIO;
	}

	fclose(fp);
	return rc;
}

/**
 * FIXME Verbose mode missing yet.
 */
static int do_list_contents(FILE *fp, char *out_f, int list_contents)
{
	int rc;
	struct stat st;
	uint32_t d, crc32, size, compressed_size;
	float ratio = 0.0;
	z_stream strm;
	uint8_t in[4096];
	uint8_t out[4096];
	gz_header head;
	uint8_t extra[64 * 1024];
	uint8_t comment[1024];
	uint8_t name[1024];
	int window_bits = 31;	/* GZIP */
	const char *mon[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun",
			      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

	rc = fstat(fileno(fp), &st);
	if (rc != 0)
		return rc;

	memset(&strm, 0, sizeof(strm));
	strm.avail_in = 0;
	strm.next_in = Z_NULL;
	rc = inflateInit2(&strm, window_bits);
	if (Z_OK != rc)
		return rc;

	strm.next_out = out;
	strm.avail_out = sizeof(out);
	strm.next_in = in;
	strm.avail_in = fread(in, 1, sizeof(in), fp);
	if (ferror(fp))
		return Z_ERRNO;

	head.extra = extra;
	head.extra_len = 0;
	head.extra_max = sizeof(extra);

	head.comment = comment;
	head.comm_max = sizeof(comment);

	head.name = name;
	head.name_max = sizeof(name);

	rc = inflateGetHeader(&strm, &head);
	if (Z_OK != rc) {
		fprintf(stderr, "err: Cannot read gz header! rc=%d\n", rc);
		return rc;
	}

	rc = inflate(&strm, Z_SYNC_FLUSH); /*Z_BLOCK not supported in libnxz*/
	if (Z_OK != rc) {
		fprintf(stderr, "err: inflate(Z_BLOCK) failed rc=%d\n", rc);
		return rc;
	}

	if (head.done == 0) {
		fprintf(stderr, "err: gzip header not entirely decoded! "
			"total_in=%ld total_out=%ld head.done=%d\n",
			strm.total_in, strm.total_out, head.done);
		return Z_DATA_ERROR;
	}

	rc = fseek(fp, st.st_size - 2 * sizeof(uint32_t), SEEK_SET);
	if (rc != 0)
		return rc;

	rc = fread(&d, sizeof(d), 1, fp);
	if (rc != 1)
		return -1;
	crc32 = __le32_to_cpu(d);

	rc = fread(&d, sizeof(d), 1, fp);
	if (rc != 1)
		return -1;
	size = __le32_to_cpu(d);

	/* Compressed size is total file size reduced by gzip header
	   size and 8 bytes for the gzip trailer. */
	compressed_size = st.st_size - strm.total_in - 8;
	if (size)
		ratio = 100 - (float)compressed_size * 100 / size;

	if (!verbose) {
		fprintf(stderr,
			"         compressed        uncompressed  ratio "
			"uncompressed_name\n"
			"%19lld %19lld  %2.2f%% %s\n",
			(long long)st.st_size, (long long)size, ratio,
			out_f);
	} else {
		time_t t = time(NULL);
		struct tm *tm = localtime(&t);
		/* (const time_t *)&head.time */

		fprintf(stderr, "method  crc     date  time           "
			"compressed        uncompressed  ratio "
			"uncompressed_name\n"
			"%s %x %s %2d %d:%d %19lld %19lld  %2.2f%% %s\n",
			"defla", crc32,
			mon[tm->tm_mon], tm->tm_mday, tm->tm_hour, tm->tm_min,
			(long long)st.st_size, (long long)size, ratio,
			out_f);
	}

	if (list_contents > 1)
		do_print_gzip_hdr(&head, stderr);

	return 0;
}

static int strip_ending(char *oname, const char *iname, size_t n,
			const char *suffix)
{
	char *ending;

	snprintf(oname, n, "%s", iname);  /* create a copy */

	ending = strstr(oname, suffix);	 /* find suffix ... */
	if (ending == NULL)
		return -1;	/* hey, suffix not found! */

	ending--;
	*ending = 0;		/* ... and strip suffix */
	return 0;
}


/* compress or decompress from stdin to stdout */
int main(int argc, char **argv)
{
	int rc = Z_OK;
	bool compress = true;
	int list_contents = 0;
	bool force = false;
	bool quiet __attribute__((unused)) = false;
	int window_bits = 31;	/* GZIP */
	int level = Z_DEFAULT_COMPRESSION;
	char *prog = basename(argv[0]);
	const char *in_f = NULL;
	char out_f[PATH_MAX];
	FILE *i_fp = stdin;
	FILE *o_fp = NULL;
	const char *suffix = "gz";
	unsigned char *in = NULL;
	unsigned char *out = NULL;
	z_stream strm;
	const char *name = NULL;
	char *comment = NULL;
	const char *extra_fname = NULL;
	uint8_t *extra = NULL;
	int extra_len = 0;
	struct stat s;


	/* avoid end-of-line conversions */
	SET_BINARY_MODE(stdin);
	SET_BINARY_MODE(stdout);

	if (strstr(prog, "gunzip") != 0) {
		compress = false;
		CHUNK_o *= 4; /* adjust default output buffer size to avoid memcpy */
	}

	while (1) {
		int ch;
		int option_index = 0;
		static struct option long_options[] = {
			{ "stdout",	 no_argument,       NULL, 'c' },
			{ "decompress",  no_argument,       NULL, 'd' },
			{ "force",       no_argument,       NULL, 'f' },
			{ "help",	 no_argument,       NULL, 'h' },

			/* list */
			{ "list",	 no_argument,	    NULL, 'l' },
			{ "license",     no_argument,       NULL, 'L' },
			{ "suffix",      required_argument, NULL, 'S' },
			{ "verbose",	 no_argument,       NULL, 'v' },
			{ "version",	 no_argument,       NULL, 'V' },
			{ "fast",	 no_argument,       NULL, '1' },
			{ "best",	 no_argument,       NULL, '9' },

			/* our own options */
			{ "extra",	 required_argument, NULL, 'E' },
			{ "name",	 required_argument, NULL, 'N' },
			{ "comment",	 required_argument, NULL, 'C' },
			{ "i_bufsize",   required_argument, NULL, 'i' },
			{ "o_bufsize",   required_argument, NULL, 'o' },
			{ 0,		 no_argument,       NULL, 0   },
		};

		ch = getopt_long(argc, argv,
				 "E:N:C:cdfqhlLsS:vV123456789?i:o:X:A:B:",
				 long_options, &option_index);
		if (ch == -1)    /* all params processed ? */
			break;

		switch (ch) {
		case 'E':
			extra_fname = optarg;
			break;
		case 'N':
			name = optarg;
			break;
		case 'C':
			comment = optarg;
			break;
		case 'd':
			compress = false;
			break;
		case 'f':
			force = true;
			break;
		case 'q':
			/* Currently does nothing, zless needs it */
			quiet = true;
			break;
		case 'c':
			o_fp = stdout;
			break;
		case 'S':
			suffix = optarg;
			break;
		case 'l':
			list_contents++;
			break;
		case '1':
			level = Z_BEST_SPEED;
			break;
		case '2':
			level = 2;
			break;
		case '3':
			level = 3;
			break;
		case '4':
			level = 4;
			break;
		case '5':
			level = 5;
			break;
		case '6':
			level = 6;
			break;
		case '7':
			level = 7;
			break;
		case '8':
			level = 8;
			break;
		case '9':
			level = Z_BEST_COMPRESSION;
			break;
		case 'v':
			verbose++;
			break;
		case 'V':
			fprintf(stdout, "%s\n", version);
			exit(EXIT_SUCCESS);
			break;
		case 'i':
			CHUNK_i = str_to_num(optarg);
			break;
		case 'o':
			CHUNK_o = str_to_num(optarg);
			break;
		case 'L':
			userinfo(stdout, prog, version);
			exit(EXIT_SUCCESS);
			break;
		case 'h':
		case '?':
			usage(stdout, prog, argc, argv);
			exit(EXIT_SUCCESS);
			break;
		}
	}

	/* FIXME loop over this ... */
	if (optind < argc) {      /* input file */
		in_f = argv[optind++];

		i_fp = fopen(in_f, "r");
		if (!i_fp) {
			pr_err("%s\n", strerror(errno));
			print_args(stderr, argc, argv);
			exit(EX_ERRNO);
		}

		rc = lstat(in_f, &s);
		if ((rc == 0) && S_ISLNK(s.st_mode)) {
			pr_err("%s: Too many levels of symbolic links\n",
			       in_f);
			exit(EXIT_FAILURE);
		}

		if (list_contents) {
			rc = strip_ending(out_f, in_f, PATH_MAX, suffix);
			if (rc < 0) {
				pr_err("No .%s file!\n", suffix);
				print_args(stderr, argc, argv);
				exit(EXIT_FAILURE);
			}

			rc = do_list_contents(i_fp, out_f, list_contents);
			if (rc != 0) {
				pr_err("Unable to list contents.\n");
				print_args(stderr, argc, argv);
				exit(EXIT_FAILURE);
			}
			fclose(i_fp);
			exit(EXIT_SUCCESS);
		}
	}

	if (in_f == NULL)
		o_fp = stdout;	/* should not be a terminal! */

	if (o_fp == NULL) {
		if (compress)
			snprintf(out_f, PATH_MAX, "%s.%s", in_f, suffix);
		else {
			rc = strip_ending(out_f, in_f, PATH_MAX, suffix);
			if (rc < 0) {
				pr_err("No .%s file!\n", suffix);
				print_args(stderr, argc, argv);
				exit(EXIT_FAILURE);
			}
		}

		rc = stat(out_f, &s);
		if (!force && (rc == 0)) {
			pr_err("File %s already exists!\n", out_f);
			print_args(stderr, argc, argv);
			exit(EX_ERRNO);
		}

		o_fp = fopen(out_f, "w+");
		if (!o_fp) {
			pr_err("Cannot open output file %s: %s\n", out_f,
			       strerror(errno));
			print_args(stderr, argc, argv);
			exit(EX_ERRNO);
		}

		/* get mode settings for existing file and ... */
		rc = fstat(fileno(i_fp), &s);
		if (rc == 0) {
			rc = fchmod(fileno(o_fp), s.st_mode);
			if (rc != 0) {
				pr_err("Cannot set mode %s: %s\n", out_f,
				       strerror(errno));
				exit(EX_ERRNO);
			}
		} else /* else ignore ... */
			pr_err("Cannot set mode %s: %s\n", out_f,
			       strerror(errno));


		/* If output does not go to stdout and a filename is
		   given, set it */
		if (name == NULL)
			name = in_f;
	}

	if (isatty(fileno(o_fp))) {
		pr_err("Output must not be a terminal!\n");
		print_args(stderr, argc, argv);
		exit(EXIT_FAILURE);
	}

	if (optind != argc) {   /* now it must fit */
		usage(stderr, prog, argc, argv);
		exit(EXIT_FAILURE);
	}

	in = malloc(CHUNK_i);	/* This is the bigger Buffer by default */
	if (NULL == in) {
		pr_err("%s\n", strerror(errno));
		print_args(stderr, argc, argv);
		exit(EXIT_FAILURE);
	}

	out = malloc(CHUNK_o);	/* This is the smaller Buffer by default */
	if (NULL == out) {
		pr_err("%s\n", strerror(errno));
		print_args(stderr, argc, argv);
		exit(EXIT_FAILURE);
	}

	/* allocate inflate state */
	memset(&strm, 0, sizeof(strm));
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;

	if (compress) {
		gz_header head;
		struct timeval tv;

		if (extra_fname) {
			extra_len = file_size(extra_fname);
			if (extra_len <= 0) {
				rc = extra_len;
				goto err_out;
			}

			extra = malloc(extra_len);
			if (extra == NULL) {
				rc = -ENOMEM;
				goto err_out;
			}

			rc = file_read(extra_fname, extra, extra_len);
			if (rc != 1) {
				fprintf(stderr, "err: Unable to read extra "
					"data rc=%d\n", rc);
				free(extra);
				goto err_out;
			}

			//hexdump(stderr, extra, extra_len);
		}

		/* --------------- DEFALTE ----------------- */
		rc = deflateInit2(&strm, level, Z_DEFLATED, window_bits, 8,
				  Z_DEFAULT_STRATEGY);
		if (Z_OK != rc)
			goto err_out;

		memset(&head, 0, sizeof(head));

		gettimeofday(&tv, NULL);
		head.time = tv.tv_sec;
		head.os = 0x03;
		head.hcrc=1;

		if (extra != NULL) {
			head.extra = extra;
			head.extra_len = extra_len;
			head.extra_max = extra_len;
		}
		if (comment != NULL) {
			head.comment = (Bytef *)comment;
			head.comm_max = strlen(comment) + 1;
		}
		if (name != NULL) {
			head.name = (Bytef *)name;
			head.name_max = strlen(name) + 1;
		}

		rc = deflateSetHeader(&strm, &head);
		if (Z_OK != rc) {
			fprintf(stderr, "err: Cannot set gz header! rc=%d\n",
				rc);
			deflateEnd(&strm);
			goto err_out;
		}
		if (verbose) {
			fprintf(stderr,
				"deflateBound() %lld bytes for %lld bytes input\n",
				(long long)deflateBound(&strm, CHUNK_i),
				(long long)CHUNK_i);
			fprintf(stderr,
				"compressBound() %lld bytes for %lld bytes input\n",
				(long long)compressBound(CHUNK_i),
				(long long)CHUNK_i);
		}

		/* do compression if no arguments */
		rc = def(i_fp, o_fp, &strm, in, out);
		if (Z_OK != rc)
			zerr(rc);

		if (extra != NULL)
			free(extra);

		deflateEnd(&strm);
	} else {
		/* --------------- INFALTE ----------------- */
		strm.avail_in = 0;
		strm.next_in = Z_NULL;
		rc = inflateInit2(&strm, window_bits);
		if (Z_OK != rc)
			goto err_out;

		do {
			rc = inf(i_fp, o_fp, &strm, in, out);
			if (Z_STREAM_END != rc) {
				zerr(rc);
				break;
			}
		} while (!feof(i_fp) && !ferror(i_fp));

		inflateEnd(&strm);
	}

 err_out:
	/* Delete the input file, only if input is not stdin and if
	   output is not stdout */
	if ((rc == EXIT_SUCCESS) && (i_fp != stdin) && (o_fp != stdout)) {
		rc = unlink(in_f);
		if (rc != 0) {
			pr_err("%s\n", strerror(errno));
			print_args(stderr, argc, argv);
			exit(EXIT_FAILURE);
		}
	}

	fclose(i_fp);
	fclose(o_fp);
	free(in);
	free(out);

	exit(rc);
}
