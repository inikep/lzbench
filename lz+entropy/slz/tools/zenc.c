/*
 * Copyright (C) 2013-2015 Willy Tarreau <w@1wt.eu>
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

#define _GNU_SOURCE /* for F_SETPIPE_SZ */
#include <errno.h>
#include <stdarg.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/user.h>
#include <fcntl.h>
#include <signal.h>
#include "../src/slz.h"
#include "../src/slz-prv.h"
#if defined(__linux__)
#include <linux/fs.h>
#endif


/* some platforms do not provide PAGE_SIZE */
#ifndef PAGE_SIZE
#define PAGE_SIZE sysconf(_SC_PAGESIZE)
#endif

/* display the message and exit with the code */
__attribute__((noreturn)) void die(int code, const char *format, ...)
{
        va_list args;

        va_start(args, format);
        vfprintf(stderr, format, args);
        va_end(args);
        exit(code);
}

__attribute__((noreturn)) void usage(const char *name, int code)
{
	die(code,
	    "slz"
#ifdef SLZ_VERSION
	    " version " SLZ_VERSION
#endif
#if defined(SLZ_DEV_BUILD) && defined(SLZ_BUILD)
	    " (" SLZ_BUILD " build)"
#endif
	    "\nUsage: %s [option]* [file]*\n"
	    "\n"
	    "Compresses each <file> in place: the output goes to <file> followed by the\n"
	    "format's extension (.gz, .zz or .dfl), which receives <file>'s mode and\n"
	    "times, and <file> is then removed. With -d, decompresses each <file> to\n"
	    "<file> without its extension, whichever it is.\n"
	    "Without <file>, or with -c, the output goes to stdout. A <file> named \"-\"\n"
	    "is stdin.\n"
	    "\n"
	    "The following arguments are supported :\n"
	    "  -0         disable compression, only uses output format\n"
	    "  -1..9      compression level (1=fastest, 4 and above=best); default=2\n"
	    "  -K <size>  block size, in kB or with a b/k/m/g suffix (default: 4096); with\n"
	    "             -d, output ring size (default and minimum: 32 kB)\n"
	    "  -b <size>  only use <size> bytes from the input file (needs -c or -t,\n"
	    "             not with -d)\n"
	    "  -m         memory mode: read the input into a buffer of the block size\n"
	    "             instead of mapping it (default when it cannot be mapped; not\n"
	    "             with -d)\n"
	    "  -c         send output to stdout, keep the files [default without file]\n"
	    "  -d         decompress\n"
	    "  -f         force sending output to a terminal, overwrite existing output\n"
	    "             files, and compress symbolic links and files which already\n"
	    "             have the extension\n"
	    "  -h         display this help\n"
#ifdef SLZ_DEV_BUILD
	    "  -i <size>  with -d, read the input by chunks of <size>, in kB or with a\n"
	    "             b/k/m suffix (default: 8 kB); for debugging only\n"
#endif
	    "  -k         keep the input files\n"
	    "  -l <loops> loop <loops> times over the same file (test mode only)\n"
	    "  -n         does nothing, just for gzip compatibility\n"
	    "  -p         pre-heat: fault the input (mmap mode) and output buffers in\n"
	    "             before compressing, and keep the input mapped between loops\n"
	    "             (test mode only, not with -d)\n"
	    "  -t         test mode: do not emit anything; with -d, the exit status says\n"
	    "             whether the inputs could be decompressed\n"
	    "  -v         increase verbosity\n"
	    "\n"
	    "  -D         use raw Deflate format (RFC1951) -> .dfl\n"
	    "  -G         use Gzip format (RFC1952) [default] -> .gz\n"
	    "  -Z         use Zlib format (RFC1950) -> .zz\n"
	    "\n"
	    "When decompressing, the format is detected from the stream's header unless one\n"
	    "of -D/-G/-Z is set, or the file's extension is .dfl (raw Deflate cannot be\n"
	    "detected).\n"
	    "\n"
	    ,name);
}


/* options, shared by all inputs */
static size_t block_size = 4194304;
static off_t max_read = -1;     /* -b, or -1 for the whole input */
static int buffer_opt = 0;      /* -m */
static int preheat = 0;
static int loops = 1;
static int level   = 2;
static int verbose = 0;
static int test    = 0;
static int format  = SLZ_FMT_GZIP;

static int fmt_forced = 0;     /* -D, -G or -Z was set */
static int decompress = 0;     /* -d */
static int force   = 0;
static int keep    = 0;
static int to_stdout = 0;
static const char *progname;

/* the output file being created, removed if we are interrupted */
static const char *volatile cur_output;

/* the output buffer, shared by all inputs */
static unsigned char *outbuf;
static size_t outbsize;

/* decompression: the output ring (at least 32 kB, see uslz_init()), and
 * the input buffer, read by chunks of <in_size> bytes (-i in development
 * builds).
 */
#define DEC_MIN_RING 32768
#define DEC_IN_SIZE  8192
static unsigned char *ring;
static long ring_size = DEC_MIN_RING;
static unsigned char *inbuf;
static long in_size = DEC_IN_SIZE;

/* the options only offered by development builds (see BUILD in the
 * Makefile), appended to getopt()'s list.
 */
#ifdef SLZ_DEV_BUILD
#define DEV_OPTS "i:"
#else
#define DEV_OPTS ""
#endif

/* the extensions of the formats. Raw deflate cannot be detected, so it is
 * the only one that an extension imposes when decompressing, the other
 * ones are detected from the stream's header.
 */
static const struct {
	const char *ext;
	int fmt;
} exts[] = {
	{ ".gz",   SLZ_FMT_GZIP    },
	{ ".zz",   SLZ_FMT_ZLIB    },
	{ ".dfl",  SLZ_FMT_DEFLATE },
};

/* Parses size <str>, in kB by default, or in the unit of its suffix: "b"
 * for bytes, "k" for kB, "m" for MB, in lower or upper case. Returns the
 * size in bytes, or 0 if <str> is not a valid size.
 */
static unsigned long long parse_size(const char *str)
{
	unsigned long long size;
	char *end;

	size = strtoull(str, &end, 0);
	if (end == str)
		return 0;

	switch (*end) {
	case 'g': case 'G': size *= 1024; __fallthrough;
	case 'm': case 'M': size *= 1024; __fallthrough;
	case 'k': case 'K': size *= 1024; __fallthrough;
	case 'b': case 'B': end++; break;
	case 0: size *= 1024; break;
	}

	return *end ? 0 : size;
}

/* reports an error about <name> (or stdin if NULL) caused by errno */
static void report(const char *name, const char *what)
{
	fprintf(stderr, "%s: %s: %s: %s\n", progname, name ? name : "stdin", what, strerror(errno));
}

/* writes all <len> bytes of <buf> to <fd>, returns 0 on success or -1 on
 * error with errno set.
 */
static int write_all(int fd, const unsigned char *buf, size_t len)
{
	while (len) {
		ssize_t ret = write(fd, buf, len);

		if (ret < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		buf += ret;
		len -= ret;
	}
	return 0;
}

/* returns the index in exts[] of the extension <name> ends with, or -1 */
static int find_ext(const char *name)
{
	size_t nlen = strlen(name);
	int i;

	for (i = 0; i < sizeof(exts) / sizeof(exts[0]); i++) {
		size_t elen = strlen(exts[i].ext);

		if (nlen > elen && strcmp(name + nlen - elen, exts[i].ext) == 0)
			return i;
	}
	return -1;
}

/* returns the length of the extension of <name>, its dot included, or 0 if
 * it has none. The extension is what follows the last dot of the last path
 * component, which must neither be its first character (hidden files) nor
 * its last one.
 */
static size_t ext_len(const char *name)
{
	const char *base = strrchr(name, '/');
	const char *dot;

	base = base ? base + 1 : name;
	dot = strrchr(base, '.');
	if (!dot || dot == base || !dot[1])
		return 0;
	return strlen(dot);
}

/* returns the format to decompress <name> (NULL for stdin) with, or -1 to
 * detect it from the stream. -D, -G and -Z come first, then the .dfl
 * extension since raw deflate cannot be detected, and the stream must not
 * be taken for gzip or zlib when its name says it is raw deflate.
 */
static int dec_format(const char *name)
{
	int i;

	if (fmt_forced)
		return format;
	i = name ? find_ext(name) : -1;
	if (i >= 0 && exts[i].fmt == SLZ_FMT_DEFLATE)
		return SLZ_FMT_DEFLATE;
	return -1;
}

/* removes the partial output file on SIGINT/SIGTERM/SIGHUP, then dies from
 * the same signal.
 */
static void interrupted(int sig)
{
	if (cur_output)
		unlink(cur_output);
	signal(sig, SIG_DFL);
	raise(sig);
}

/* Compresses the input read from <fd> to <ofd> following the options above.
 * <name> is the input's name for the messages, NULL for stdin. Returns 0 on
 * success, non-zero on error.
 */
static int compress_fd(int fd, int ofd, const char *name)
{
	size_t released;
	struct stat statbuf;
	struct slz_stream strm;
	unsigned char *buffer;
	off_t toread = max_read;
	off_t tocompress = 0;
	off_t ofs;
	size_t outblen;
	size_t mapsize = 0;
	unsigned long long totin = 0;
	unsigned long long totout = 0;
	int buffer_mode = buffer_opt;
	int passes = loops;
	int round = 0;          /* index of the pass being run */
	int timed_round;        /* index of the pass the measurement starts on */
	struct timeval tv_beg, tv_end;
	unsigned long long timed_in = 0;
	int error = 0;

	/* principle : we'll determine the input file size and try to map the
	 * file at once. If it works we have a single input buffer of the whole
	 * file size. If it fails we'll bound the input buffer to the buffer size
	 * and read the input in smaller blocks.
	 */
	if (toread < 0) {
		if (fstat(fd, &statbuf) == -1) {
			toread = 0;
		}
		else {
			toread = statbuf.st_size;

#if defined(BLKGETSIZE64)
			/* block devices are reported as size zero */
			if (S_ISBLK(statbuf.st_mode)) {
				if (ioctl(fd, BLKGETSIZE64, &toread) == -1)
					toread = 0;
			}
#endif

#if defined(F_GETPIPE_SZ) && defined(F_SETPIPE_SZ)
			/* attempt to optimize the pipe size if needed and possible */
			if (S_ISFIFO(statbuf.st_mode)) {
				int size = fcntl(fd, F_GETPIPE_SZ);
				if (size > 0 && size < block_size)
					fcntl(fd, F_SETPIPE_SZ, block_size);
			}
#endif
		}
	}

	if (toread && !buffer_mode) {
		/* we know the size to map, let's do it */
		mapsize = (toread + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
		buffer = mmap(NULL, mapsize, PROT_READ, MAP_PRIVATE, fd, 0);
		if (buffer == MAP_FAILED)
			mapsize = 0;
		else {
#if defined(MADV_DONTDUMP)
			madvise((void *)buffer, mapsize, MADV_DONTDUMP);
#endif
#if defined(MADV_SEQUENTIAL)
			madvise((void *)buffer, mapsize, MADV_SEQUENTIAL);
#endif
			if (preheat) {
				/* touch every page now so that no page fault is
				 * left for the measured passes. The sum is only
				 * there to keep the reads.
				 */
				volatile unsigned char sink;
				unsigned char sum = 0;
				off_t pos;

				for (pos = 0; pos < toread; pos += PAGE_SIZE)
					sum += buffer[pos];
				sink = sum;
				(void)sink;
			}
		}
	}

	if (!mapsize) {
		/* no mmap() done, read the size of a default block */
		buffer_mode = 1;
		mapsize = block_size;
		if (toread && toread < mapsize)
			mapsize = toread;

		buffer = calloc(1, mapsize);
		if (!buffer) {
			perror("calloc");
			exit(1);
		}

		if (toread && toread <= mapsize) {
			/* file-at-once, read it now */
			toread = read(fd, buffer, toread);
			if (toread < 0) {
				perror("read");
				exit(2);
			}
		}

		if (passes > 1 && !toread) {
			passes = 1;
			fprintf(stderr, "Warning: disabling loops on non-regular file\n");
		}
	}

	/* with -p, also fault the output buffer in now: with large blocks it
	 * is large enough to be mapped on demand by calloc(), and the first
	 * pass would otherwise pay for its page faults (~22k for -k 262144).
	 */
	if (preheat) {
		size_t pos;

		for (pos = 0; pos < outbsize + 4096; pos += PAGE_SIZE)
			outbuf[pos] = 0;
	}

	/* When looping, the first pass is not measured: it is the one that
	 * faults the input in and warms the caches, and counting it would
	 * report the cost of the I/O rather than that of the compression.
	 * A single pass is measured as-is, there is nothing else to report.
	 */
	timed_round = (passes > 1);
	gettimeofday(&tv_beg, NULL); /* in case no pass runs at all, eg -l 0 */

	while (passes-- && !error) {
		if (round == timed_round) {
			timed_in = totin;
			gettimeofday(&tv_beg, NULL);
		}

		slz_init(&strm, level, format);

		outblen = ofs = 0;
		released = 0;
		do {
			int more = !toread || (toread - ofs) > block_size;
			unsigned char *start;

			if (toread && toread <= mapsize) {
				/* We use the memory-mapped file so the next
				 * block starts at the buffer + file offset. We
				 * read by blocks of <block_size> bytes at one
				 * except the last one.
				 */
				tocompress = more ? block_size : toread - ofs;
				start = buffer + ofs;
			}
			else {
				/* fill the whole block before compressing it,
				 * even from a pipe which delivers small reads:
				 * compressing partial blocks loses ratio and
				 * speed for no benefit, nobody waits for each
				 * block here.
				 */
				ssize_t ret;
				tocompress = 0;

				do {
					ret = read(fd, buffer + tocompress, more ? block_size - tocompress : toread - ofs - tocompress);
					if (ret <= 0) {
						if (ret < 0) {
							perror("read");
							exit(2);
						}
						break;
					}
					tocompress += ret;
				} while (tocompress < (more ? block_size : toread - ofs));

				if (!tocompress) // done
					break;

				start = buffer;
			}
			outblen += slz_encode(&strm, outbuf + outblen, start, tocompress, more);

#if defined(MADV_DONTNEED)
			if (!buffer_mode && !preheat && !passes && ((ofs + tocompress) & -2097152) > released) {
				/* release all complete 2 MB areas consumed so far, unless
				 * pre-heating, or before the last pass, which want the
				 * input to stay mapped.
				 */
				size_t upto = (ofs + tocompress) & -2097152;

				madvise((void *)(buffer + released), upto - released, MADV_DONTNEED);
				released = upto;
			}
#endif
			if (outblen + block_size > outbsize) {
				/* not enough space left, need to flush */
				if (!test && !error)
					if (write_all(ofd, outbuf, outblen) < 0) {
						report(name, "write");
						error = 1;
					}
				totout += outblen;
				outblen = 0;
			}
			ofs += tocompress;
		} while ((!toread || ofs < toread) && !error);

		outblen += slz_finish(&strm, outbuf + outblen);
		totin += ofs;
		totout += outblen;
		if (!test && !error)
			if (write_all(ofd, outbuf, outblen) < 0) {
				report(name, "write");
				error = 1;
			}

		if (passes && (!toread || toread > mapsize)) {
			/* this is a seeked file, let's rewind it now */
			lseek(fd, 0, SEEK_SET);
		}
		round++;
	}
	gettimeofday(&tv_end, NULL);

	if (verbose) {
		/* <timed_in> holds the input already compressed when the clock
		 * was started, so the difference is what the measured passes
		 * went through. MB/s are decimal megabytes per second.
		 */
		double sec = (tv_end.tv_sec  - tv_beg.tv_sec) +
		             (tv_end.tv_usec - tv_beg.tv_usec) / 1000000.0;
		unsigned long long done = totin - timed_in;

		if (name)
			fprintf(stderr, "%s: ", name);
		fprintf(stderr, "totin=%llu totout=%llu ratio=%.2f%% crc32=%08x time=%.3fs %.1fMB/s saving %.1fMB/s\n",
		        totin, totout, totout * 100.0 / totin, strm.crc32,
		        sec, sec > 0.0 ? done / sec / 1000000.0 : 0.0,
		        (sec > 0.0 ? done / sec / 1000000.0 : 0.0) * (1.0 - (double)totout / totin));
	}

	if (buffer_mode)
		free(buffer);
	else
		munmap(buffer, mapsize);
	return error;
}

/* Decompresses the input read from <fd> to <ofd> following the options
 * above, in format <fmt> (SLZ_FMT_*, or -1 to detect it). <name> is the
 * input's name for the messages, NULL for stdin. Returns 0 on success,
 * non-zero on error, including a truncated or corrupted input.
 */
static int decompress_fd(int fd, int ofd, const char *name, int fmt)
{
	struct uslz_stream state;
	struct timeval tv_beg, tv_end;
	unsigned long long timed_out = 0;
	unsigned long long decoded_total = 0;
	unsigned long long tot_in = 0;
	int timed_round;        /* index of the pass the measurement starts on */
	int round = 0;          /* index of the pass being run */
	int passes = loops;
	int ret;

	if (passes > 1 && lseek(fd, 0, SEEK_SET) == (off_t)-1) {
		fprintf(stderr, "%s: %s: input is not seekable, cannot loop\n", progname, name ? name : "stdin");
		return 1;
	}

	/* When looping, the first pass is not measured: it is the one that
	 * faults the input in and warms the caches, and counting it would
	 * report the cost of the I/O rather than that of the decompression.
	 * A single pass is measured as-is, there is nothing else to report.
	 */
	timed_round = (passes > 1);
	gettimeofday(&tv_beg, NULL); /* in case no pass runs at all, eg -l 0 */

	while (passes--) {
		int complete = 0;
		int done = 0;

		if (round == timed_round) {
			timed_out = decoded_total;
			gettimeofday(&tv_beg, NULL);
		}

		if (round)
			lseek(fd, 0, SEEK_SET);

		if (fmt < 0)
			ret = uslz_init(&state, ring, ring_size);
		else
			ret = uslz_init_fmt(&state, ring, ring_size, fmt);

		if (!ret) {
			fprintf(stderr, "%s: failed to init uslz state\n", progname);
			return 1;
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
				if (rret < 0) {
					if (errno == EINTR)
						continue;
					report(name, "read");
					return 1;
				}
				if (!complete) {
					fprintf(stderr, "%s: %s: truncated stream\n", progname, name ? name : "stdin");
					return 1;
				}
				/* a complete stream followed by end of file:
				 * there simply was no other member.
				 */
				break;
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
				tot_in += consumed;

				if (!test && decoded_len && write_all(ofd, decoded, decoded_len) < 0) {
					report(name, "write");
					return 1;
				}

				input += consumed;
				rret -= consumed;

				/* A gzip stream may be a series of members, so
				 * keep feeding rather than stopping on the
				 * first success. Only end of file tells us
				 * there is no other member; stopping here on
				 * an exhausted buffer would silently ignore
				 * every remaining member whenever one ends on
				 * a read boundary.
				 */
				complete = (decode_ret == USLZ_DECODE_SUCCESS);
				if (complete && !rret)
					continue;
				else if (decode_ret != USLZ_DECODE_OUT_OF_DATA)
					goto redo;
				break;
			default:
				fprintf(stderr, "%s: %s: invalid stream (error %d)\n", progname, name ? name : "stdin", decode_ret);
				return 1;
			}
		}
		round++;
	}
	gettimeofday(&tv_end, NULL);

	if (verbose) {
		/* <timed_out> holds the output already produced when the clock
		 * was started, so the difference is what the measured passes
		 * went through. MB/s are decimal megabytes per second.
		 */
		double sec = (tv_end.tv_sec  - tv_beg.tv_sec) +
		             (tv_end.tv_usec - tv_beg.tv_usec) / 1000000.0;
		unsigned long long done = decoded_total - timed_out;

		if (name)
			fprintf(stderr, "%s: ", name);
		fprintf(stderr, "totin=%llu totout=%llu ratio=%.2f%% time=%.3fs %.1fMB/s\n",
		        tot_in, decoded_total, decoded_total * 100.0 / tot_in,
		        sec, sec > 0.0 ? done / sec / 1000000.0 : 0.0);
	}
	return 0;
}

/* Compresses or decompresses file <name> in place. When compressing, the
 * output is <name> followed by the extension of the format. When
 * decompressing, the output is <name> without its extension, whichever it
 * is, the format being chosen by dec_format(); a name without extension is
 * refused since there is no name for the output. The output receives <name>'s mode, owner and times,
 * then <name> is removed unless -k is set. The output is only created once
 * the input could be opened, and an existing output is only removed with
 * -f. On error, the output is removed and the input kept. Returns 0 on
 * success, non-zero on error.
 */
static int process_file(const char *name)
{
	const char *verb = decompress ? "decompress" : "compress";
	const char *ext = format == SLZ_FMT_GZIP ? ".gz" : format == SLZ_FMT_ZLIB ? ".zz" : ".dfl";
	size_t nlen = strlen(name), elen = strlen(ext);
	struct timespec times[2];
	struct stat st;
	char *out;
	int fd, ofd;
	int err = 1;

	if (lstat(name, &st) == 0 && S_ISLNK(st.st_mode) && !force) {
		fprintf(stderr, "%s: %s is a symbolic link, use -f to %s it\n", progname, name, verb);
		return 1;
	}

	if (decompress) {
		elen = ext_len(name);
		if (!elen) {
			fprintf(stderr, "%s: %s has no extension, not decompressing it in place, use -c\n", progname, name);
			return 1;
		}
	}
	else if (nlen > elen && strcmp(name + nlen - elen, ext) == 0 && !force) {
		fprintf(stderr, "%s: %s already has the %s extension, use -f to compress it\n", progname, name, ext);
		return 1;
	}

	fd = open(name, O_RDONLY);
	if (fd < 0) {
		report(name, "open");
		return 1;
	}

	if (fstat(fd, &st) < 0) {
		report(name, "fstat");
		goto close_in;
	}

	if (!S_ISREG(st.st_mode)) {
		fprintf(stderr, "%s: %s is not a regular file, ignored\n", progname, name);
		goto close_in;
	}

	out = malloc(nlen + elen + 1);
	if (!out) {
		report(name, "malloc");
		goto close_in;
	}
	if (decompress) {
		memcpy(out, name, nlen - elen);
		out[nlen - elen] = 0;
	}
	else {
		memcpy(out, name, nlen);
		memcpy(out + nlen, ext, elen + 1);
	}

	/* only readable by us until it is complete */
	cur_output = out;
	ofd = open(out, O_WRONLY | O_CREAT | O_EXCL, 0600);
	if (ofd < 0 && errno == EEXIST && force) {
		if (unlink(out) < 0) {
			cur_output = NULL;
			report(out, "unlink");
			goto free_out;
		}
		ofd = open(out, O_WRONLY | O_CREAT | O_EXCL, 0600);
	}

	if (ofd < 0) {
		cur_output = NULL;
		if (errno == EEXIST)
			fprintf(stderr, "%s: %s already exists, use -f to overwrite it\n", progname, out);
		else
			report(out, "open");
		goto free_out;
	}

	if ((decompress ? decompress_fd(fd, ofd, name, dec_format(name)) :
	     compress_fd(fd, ofd, name)) != 0)
		goto remove_out;

	/* the owner first, since changing it may clear the mode's setuid
	 * bits. It may fail when we are not root, that's fine.
	 */
	if (fchown(ofd, st.st_uid, st.st_gid) < 0 && fchown(ofd, -1, st.st_gid) < 0) {
		/* ignored */
	}
	times[0] = st.st_atim;
	times[1] = st.st_mtim;
	if (fchmod(ofd, st.st_mode & 07777) < 0 || futimens(ofd, times) < 0) {
		report(out, "setting the mode and times");
		goto remove_out;
	}

	if (close(ofd) < 0) {
		ofd = -1;
		report(out, "close");
		goto remove_out;
	}
	ofd = -1;
	cur_output = NULL;

	if (!keep && unlink(name) < 0) {
		report(name, "unlink");
		goto free_out;
	}
	err = 0;
	goto free_out;

 remove_out:
	if (ofd >= 0)
		close(ofd);
	unlink(out);
	cur_output = NULL;
 free_out:
	free(out);
 close_in:
	close(fd);
	return err;
}

/* Sets the default mode from the name the program was called by, so that it
 * can be installed under the names of the tools it replaces. The options
 * given on the command line apply on top of it. Any other name compresses.
 */
static void mode_from_name(const char *name)
{
	static const struct {
		const char *name;
		int decompress;
		int to_stdout;
	} names[] = {
		{ "gunzip", 1, 0 },     /* gunzip: decompress in place */
		{ "unslz",  1, 0 },
		{ "zcat",   1, 1 },     /* zcat: decompress to stdout */
		{ "gzcat",  1, 1 },     /* zcat's name on macOS and the BSDs */
		{ "slzcat", 1, 1 },
		{ "zdec",   1, 1 },     /* the former decompression tool */
	};
	const char *slash = strrchr(name, '/');
	int i;

	if (slash)
		name = slash + 1;

	for (i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
		if (strcmp(name, names[i].name) == 0) {
			decompress = names[i].decompress;
			to_stdout = names[i].to_stdout;
			return;
		}
	}
}

int main(int argc, char **argv)
{
	const char *name = argv[0];
	struct stat statbuf;
	int uses_stdout = 0;
	int error = 0;
	int opt;

	mode_from_name(name);

	/* digits are options too, like with gzip: "-c1" is "-c -1". A level
	 * is thus a single digit, which is enough since the levels above 4
	 * work as level 4.
	 */
	while ((opt = getopt(argc, argv, "0123456789b:cdfhkK:l:mnptvDGZ" DEV_OPTS)) != -1) {
		switch (opt) {
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			level = opt - '0';
			break;
		case 'b':
			max_read = strtoll(optarg, NULL, 0);
			break;
		case 'K':
			block_size = parse_size(optarg);
			if (!block_size)
				usage(name, 1);
			ring_size = block_size;
			if (ring_size < DEC_MIN_RING)
				ring_size = DEC_MIN_RING;
			break;
		case 'c':
			to_stdout = 1;
			break;
		case 'd':
			decompress = 1;
			break;
		case 'f':
			force = 1;
			break;
		case 'h':
			usage(name, 0);
#ifdef SLZ_DEV_BUILD
		case 'i':
			in_size = parse_size(optarg);
			if (in_size <= 0)
				usage(name, 1);
			break;
#endif
		case 'k':
			keep = 1;
			break;
		case 'l':
			loops = atoi(optarg);
			break;
		case 'n':
			/* just for gzip compatibility */
			break;
		case 'm':
			buffer_opt = 1;
			break;
		case 'p':
			preheat = 1;
			break;
		case 't':
			test = 1;
			break;
		case 'v':
			verbose++;
			break;
		case 'D':
			format = SLZ_FMT_DEFLATE;
			fmt_forced = 1;
			break;
		case 'G':
			format = SLZ_FMT_GZIP;
			fmt_forced = 1;
			break;
		case 'Z':
			format = SLZ_FMT_ZLIB;
			fmt_forced = 1;
			break;
		default:
			usage(name, 1);
		}
	}
	argv += optind;
	argc -= optind;

	progname = name;
	if (!argc)
		to_stdout = 1;

	/* stdin given as "-" is always sent to stdout */
	for (opt = 0; opt < argc; opt++)
		if (strcmp(argv[opt], "-") == 0)
			uses_stdout = 1;

	/* loops are a benchmarking tool: repeating the output would produce a
	 * concatenation of identical streams that nobody wants.
	 */
	if (loops != 1 && !test)
		die(1, "-l is only supported in test mode (-t).\n");

	/* pre-heating only makes measurements more stable, it is useless and
	 * wastes memory otherwise.
	 */
	if (preheat && !test)
		die(1, "-p is only supported in test mode (-t).\n");

	if (in_size != DEC_IN_SIZE && !decompress)
		die(1, "-i is only supported with -d.\n");

	/* these ones only act on the compression's input */
	if (decompress && (preheat || buffer_opt || max_read >= 0))
		die(1, "-b, -m and -p are not supported with -d.\n");

	/* -b would remove the end of the input files */
	if (max_read >= 0 && !to_stdout && !test)
		die(1, "-b is only supported with -c or -t.\n");

	if (!decompress && (to_stdout || uses_stdout) && isatty(1) && !test && !force)
		die(1, "Use -f if you really want to send compressed data to a terminal, or -h for help.\n");

	if (decompress && !argc && isatty(0) && !force)
		die(1, "Use -f if you really want to read compressed data from a terminal, or -h for help.\n");

	slz_make_crc_table();
	slz_prepare_dist_table();

	if (decompress) {
		ring = malloc(ring_size);
		inbuf = malloc(in_size);
		if (!ring || !inbuf) {
			perror("malloc");
			exit(1);
		}
	}
	else {
		outbsize = 2 * block_size; // allows to pack more than one full output at each round
		outbuf = calloc(1, outbsize + 4096);
		if (!outbuf) {
			perror("calloc");
			exit(1);
		}
	}

	/* try to increase output pipe size to avoid blocking on writes */
#if defined(F_GETPIPE_SZ) && defined(F_SETPIPE_SZ)
	if (fstat(1, &statbuf) != -1) {
		/* attempt to optimize the pipe size if needed and possible */
		if (S_ISFIFO(statbuf.st_mode)) {
			int size = fcntl(1, F_GETPIPE_SZ);
			if (size > 0 && size < block_size)
				fcntl(1, F_SETPIPE_SZ, block_size);
		}
	}
#endif

	if (!argc)
		return (decompress ? decompress_fd(0, 1, NULL, dec_format(NULL)) :
		        compress_fd(0, 1, NULL)) != 0;

	signal(SIGINT, interrupted);
	signal(SIGTERM, interrupted);
	signal(SIGHUP, interrupted);

	for (; argc; argc--, argv++) {
		int fd;

		if (strcmp(*argv, "-") == 0)
			error |= (decompress ? decompress_fd(0, 1, NULL, dec_format(NULL)) :
			          compress_fd(0, 1, NULL)) != 0;
		else if (to_stdout || test) {
			fd = open(*argv, O_RDONLY);
			if (fd < 0) {
				report(*argv, "open");
				error = 1;
				continue;
			}
			error |= (decompress ? decompress_fd(fd, 1, *argv, dec_format(*argv)) :
			          compress_fd(fd, 1, *argv)) != 0;
			close(fd);
		}
		else
			error |= process_file(*argv);
	}
	return error;
}
