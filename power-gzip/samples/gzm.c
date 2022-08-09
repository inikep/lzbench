/*
   gzm.c compresses files with various zlib options.

   ./gzm -h
   usage: ./gzm [-d] < source > dest
   usage: ./gzm [-s s] [-t t] [-f f] < source > dest
   -s followed by one of ints: default(DH+LZ): 0, fixed(FH+LZ): 1, huffman_only: 2, RLE: 3
   -t followed by one of ints: gz file format: gzip: 0, zlib: 1, raw: 2, autodetect: 47(inflate only)
   for Deflate window sizes 2^9 to 2^15, other than the default 32K use:
   -t followed by one of ints: gzip 25 to 31, zlib 9 to 15, raw -8 to -15
   -f followed by one of ints: no_flush: 0, sync_flush: 1, partial_flush: 2, full_flush: 3, block: 5
   To force a certain Deflate block size, change the CHUNK constant and use one of
   the flush options

   gzm is based on:
   zpipe.c: example of proper use of zlib's inflate() and deflate()
   Not copyrighted -- provided to the public domain
   Version 1.4  11 December 2005  Mark Adler */

/* Version history:
   1.0  30 Oct 2004  First version
   1.1   8 Nov 2004  Add void casting for unused return values
		     Use switch statement for inflate() return values
   1.2   9 Nov 2004  Add assertions to document zlib guarantees
   1.3   6 Apr 2005  Remove incorrect assertion in inf()
   1.4  11 Dec 2005  Add hack to avoid MSDOS end-of-line conversions
		     Avoid some compiler warnings for input and output buffers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "zlib.h"

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

typedef struct {
    int z_hist_sz;    /* file format use -15 for raw deflate; use 15 for zlib; 31 for gzip */
    int z_strategy;   /* Z_DEFAULT_STRATEGY Z_FIXED Z_HUFFMAN_ONLY Z_RLE */
    int z_flush_type; /* Z_SYNC_FLUSH Z_NO_FLUSH Z_PARTIAL_FLUSH Z_FULL_FLUSH Z_BLOCK */
    long compressed_bytes_total;
} gzcfg_t;

#ifndef COMPRESS_LEVEL
#define COMPRESS_LEVEL Z_DEFAULT_COMPRESSION
#endif
#ifndef CHUNK
#define CHUNK          (1024*1024)
#endif
#ifndef DEF_MEM_LEVEL
#define DEF_MEM_LEVEL  8  /* hash memory size */
#endif
#define DEF_HIST_SZ    31

#define DEBG( X ) do{ ; } while(0);

/* map the command line arguments to zlib constants */
static int get_file_format(int n)
{
    if      (n == 0) return 31;
    else if (n == 1) return 15;
    else if (n == 2) return -15;
    else if (n == 47) return n;
    else if (n >= 9  && n <= 15) return n;  /* zlib with window size 2^9 to 2^15 */
    else if (n >= 25 && n <= 31) return n;  /* gzip with window size 2^9 to 2^15 */
    else if (n >= -15 && n <= -9) return n; /* raw with window size 2^9 to 2^15 */
    else             return Z_ERRNO;
}
/* usage() helpers */
static const char file_format_str[] = "gz file format: gzip: 0, zlib: 1, raw: 2, autodetect format: 47(inflate only)";
static const char window_sz_str[] = "gzip 25 to 31, zlib 9 to 15, raw -8 to -15 for window sizes 2^9 to 2^15";

static int get_strategy(int n)
{
    if      (n == 0) return Z_DEFAULT_STRATEGY;
    else if (n == 1) return Z_FIXED;
    else if (n == 2) return Z_HUFFMAN_ONLY;
    else if (n == 3) return Z_RLE;
    else             return Z_ERRNO;
}
static const char strategy_str[] = "default(DH+LZ): 0, fixed(FH+LZ): 1, huffman_only: 2, RLE: 3";

static int get_flush_type(int n)
{
    if      (n == 0) return Z_NO_FLUSH;
    else if (n == 1) return Z_SYNC_FLUSH;
    else if (n == 2) return Z_PARTIAL_FLUSH;
    else if (n == 3) return Z_FULL_FLUSH;
    else if (n == 5) return Z_BLOCK;
    else             return Z_ERRNO;
}
static const char flush_type_str[] = "no_flush: 0, sync_flush: 1, partial_flush: 2, full_flush: 3, z_block: 5";

unsigned char in[CHUNK];
unsigned char out[CHUNK];


/* Compress from file source to file dest until EOF on source.
   def() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_STREAM_ERROR if an invalid compression
   level is supplied, Z_VERSION_ERROR if the version of zlib.h and the
   version of the library linked do not match, or Z_ERRNO if there is
   an error reading or writing the files. */
int def(FILE *source, FILE *dest, int level, gzcfg_t *cf)
{
    int ret, flush;
    unsigned have;
    z_stream strm;

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.next_in = Z_NULL;

    /* ret = deflateInit(&strm, level); */

    ret = deflateInit2(&strm,
		       COMPRESS_LEVEL,
		       Z_DEFLATED,
		       cf->z_hist_sz,
		       DEF_MEM_LEVEL,
		       cf->z_strategy );
    if (ret != Z_OK)
	return ret;

    /* compress until end of file */
    do {
	strm.avail_in = fread(in, 1, CHUNK, source);
	if (ferror(source)) {
	    (void)deflateEnd(&strm);
	    return Z_ERRNO;
	}
	flush = feof(source) ? Z_FINISH : cf->z_flush_type;
	strm.next_in = in;

	/* run deflate() on input until output buffer not full, finish
	   compression if all of source has been read in */
	do {
	    strm.avail_out = CHUNK;
	    strm.next_out = out;
	    DEBG( fprintf(stderr, "Before deflate call: input size: %d. ", strm.avail_in) );
	    ret = deflate(&strm, flush);    /* no bad return value */
	    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
	    have = CHUNK - strm.avail_out;
	    DEBG( fprintf(stderr, "After deflate call: input data remaining: %d; output data to write: %d, rc: %d\n", strm.avail_in, have, ret) );
	    cf->compressed_bytes_total += have;
	    if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
		(void)deflateEnd(&strm);
		return Z_ERRNO;
	    }
	} while (strm.avail_out == 0);
	assert(strm.avail_in == 0);     /* all input will be used */

	/* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END);        /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
}

/* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
int inf(FILE *source, FILE *dest, gzcfg_t *cf)
{
    int ret;
    unsigned have;
    z_stream strm;

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    ret = inflateInit2( &strm, cf->z_hist_sz );
    if (ret != Z_OK)
	return ret;

    /* decompress until deflate stream ends or end of file */
    do {
	strm.avail_in = fread(in, 1, CHUNK, source);
	if (ferror(source)) {
	    (void)inflateEnd(&strm);
	    return Z_ERRNO;
	}
	if (strm.avail_in == 0)
	    break;
	strm.next_in = in;

	/* run inflate() on input until output buffer not full */
	do {
	    strm.avail_out = CHUNK;
	    strm.next_out = out;
	    DEBG( fprintf(stderr, "Before inflate call: input size: %d. ", strm.avail_in) );
	    ret = inflate(&strm, Z_NO_FLUSH);
	    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
	    switch (ret) {
	    case Z_NEED_DICT:
		ret = Z_DATA_ERROR;     /* and fall through */
	    case Z_DATA_ERROR:
	    case Z_MEM_ERROR:
		(void)inflateEnd(&strm);
		return ret;
	    }
	    have = CHUNK - strm.avail_out;
	    DEBG( fprintf(stderr, "After inflate call: input data remaining: %d; output data to write: %d, rc: %d\n", strm.avail_in, have, ret) );
	    if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
		(void)inflateEnd(&strm);
		return Z_ERRNO;
	    }
	} while (strm.avail_out == 0);

	/* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);

    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

/* report a zlib or i/o error */
void zerr(int ret)
{
    fputs("zpipe: ", stderr);
    switch (ret) {
    case Z_ERRNO:
	if (ferror(stdin))
	    fputs("error reading stdin\n", stderr);
	if (ferror(stdout))
	    fputs("error writing stdout\n", stderr);
	break;
    case Z_STREAM_ERROR:
	fputs("invalid compression level\n", stderr);
	break;
    case Z_DATA_ERROR:
	fputs("invalid or incomplete deflate data\n", stderr);
	break;
    case Z_MEM_ERROR:
	fputs("out of memory\n", stderr);
	break;
    case Z_VERSION_ERROR:
	fputs("zlib version mismatch!\n", stderr);
    }
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "usage: %s [-d] [-t t] < source > dest\n", argv[0] );
    fprintf(stderr, "usage: %s [-s s] [-t t] [-f f] < source > dest\n", argv[0] );
    fprintf(stderr, "   -s followed by one of ints: %s\n", strategy_str);
    fprintf(stderr, "   -t followed by one of ints: %s\n", file_format_str);
    fprintf(stderr, "or -t followed by one of ints: %s\n", window_sz_str);
    fprintf(stderr, "   -f followed by one of ints: %s\n", flush_type_str);
}

/* compress or decompress from stdin to stdout */
int main(int argc, char **argv)
{
    int ret;
    gzcfg_t cf;

    /* avoid end-of-line conversions */
    SET_BINARY_MODE(stdin);
    SET_BINARY_MODE(stdout);

    /* default deflate parameters */
    cf.z_hist_sz = DEF_HIST_SZ;
    cf.z_strategy = Z_DEFAULT_STRATEGY;
    cf.z_flush_type = Z_NO_FLUSH;
    cf.compressed_bytes_total = 0;

    /* do compression if no arguments */
    if (argc == 1) {
	ret = def(stdin, stdout, COMPRESS_LEVEL, &cf);
	if (ret != Z_OK)
	    zerr(ret);
	DEBG( fprintf(stderr, "compressed bytes: %ld\n", cf.compressed_bytes_total ) );
	return ret;
    }

    /* do decompression if -d specified */
    else if (argc == 2 && strcmp(argv[1], "-d") == 0) {
	ret = inf(stdin, stdout, &cf);
	if (ret != Z_OK)
	    zerr(ret);
	return ret;
    }

    /* extended arguments  */
    else {
	int i;
	char fn='c';
	i = 1;
	if (strcmp(argv[i], "-d") == 0) {
	    fn = 'd';
	    i = i + 1;
	}
	while (i<argc) {
	    int e=0;
	    if (strcmp(argv[i], "-s") == 0) {
		if (Z_ERRNO == (cf.z_strategy = get_strategy( atoi(argv[i+1]) )))  ++ e;
		if (fn == 'd') ++e; /* cannot combine -d and -s */
		i = i + 2;
	    }
	    else if (strcmp(argv[i], "-t") == 0) {
		if (Z_ERRNO == (cf.z_hist_sz = get_file_format( atoi(argv[i+1]) ))) ++ e;
		if (cf.z_hist_sz == 47 && fn == 'c') ++e; /* autodetect is inflate only */
		i = i + 2;
	    }
	    else if (strcmp(argv[i], "-f") == 0) {
		if (Z_ERRNO == (cf.z_flush_type = get_flush_type( atoi(argv[i+1]) ))) ++ e;
		i = i + 2;
	    }
	    else if (strcmp(argv[i], "-d") == 0) {
		++ e; /* -d must be first */
	    }
	    else {
		++ e;
	    }
	    if (e>0) {
		usage(argc, argv);
		return -1;
	    }
	}

	if( fn == 'c' ) {
	    ret = def(stdin, stdout, COMPRESS_LEVEL, &cf);
	    /* fprintf(stderr, "compressed bytes: %ld\n", cf.compressed_bytes_total ); */
	}
	else {
	    ret = inf(stdin, stdout, &cf);
	}
	if (ret != Z_OK)
	    zerr(ret);
	return ret;
    }
    return 1;
}


/* sample def code that doesn't use FILE I/O but memory buffers
   Note:  size_of_data is the source bytes when called, and the
   dest bytes in the dest buffer on return */
int def_mb(char *source, char *dest, int *size_of_data, gzcfg_t *cf)
{
    int ret, flush, remainder;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.next_in = Z_NULL;

    ret = deflateInit2(&strm,
		       COMPRESS_LEVEL,
		       Z_DEFLATED,
		       cf->z_hist_sz,
		       DEF_MEM_LEVEL,
		       cf->z_strategy );
    if (ret != Z_OK)
	return ret;

    remainder = *size_of_data;
    /* compress until end of file */
    do {
	int nbyte;
	/* strm.avail_in = fread(in, 1, CHUNK, source); */
	strm.avail_in = nbyte = (remainder > CHUNK) ? CHUNK : remainder;
	memcpy( in, source, nbyte );
	remainder = remainder - nbyte;
	source = source + nbyte;

	/* flush = feof(source) ? Z_FINISH : cf->z_flush_type; */
	flush = ( remainder == 0 ) ? Z_FINISH : cf->z_flush_type;
	strm.next_in = in;

	/* run deflate() on input until output buffer not full, finish
	   compression if all of source has been read in */
	do {
	    strm.avail_out = CHUNK;
	    strm.next_out = out;
	    DEBG( fprintf(stderr, "Before deflate call: input size: %d. ", strm.avail_in) );
	    ret = deflate(&strm, flush);    /* no bad return value */
	    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
	    have = CHUNK - strm.avail_out;
	    DEBG( fprintf(stderr, "After deflate call: input data remaining: %d; output data to write: %d, rc: %d\n", strm.avail_in, have, ret) );
	    cf->compressed_bytes_total += have;
	    /* if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
		(void)deflateEnd(&strm);
		return Z_ERRNO;
	    } */
	    memcpy( dest, out, have );
	    dest = dest + have;

	} while (strm.avail_out == 0);
	assert(strm.avail_in == 0);     /* all input will be used */

	/* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END);        /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);

    *size_of_data = cf->compressed_bytes_total;

    return Z_OK;
}
