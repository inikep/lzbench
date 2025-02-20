/*
 * libxpack.h - public header for libxpack
 */

#ifndef LIBXPACK_H
#define LIBXPACK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Microsoft C / Visual Studio garbage.  If you want to link to the DLL version
 * of libxpack, then #define LIBXPACK_DLL. */
#ifdef _MSC_VER
#  ifdef BUILDING_LIBXPACK
#    define LIBXPACKAPI __declspec(dllexport)
#  elif defined(LIBXPACK_DLL)
#    define LIBXPACKAPI __declspec(dllimport)
#  endif
#endif
#ifndef LIBXPACKAPI
#  define LIBXPACKAPI
#endif

/* ========================================================================== */
/*                               Compression                                  */
/* ========================================================================== */

struct xpack_compressor;

/*
 * xpack_alloc_compressor() allocates a new compressor.
 *
 * 'max_buffer_size' is the maximum size of any buffer which will be compressed
 * by the compressor.  This specifies the maximum allowed value for the
 * 'uncompressed_size' parameter of xpack_compress() when called using this
 * compressor.
 *
 * 'compression_level' is the compression level on a zlib-like scale (1 =
 * fastest, 6 = medium/default, 9 = slowest).
 *
 * Returns a pointer to the new compressor, or NULL if out of memory or the
 * maximum buffer size or compression level is not supported.
 */
LIBXPACKAPI struct xpack_compressor *
xpack_alloc_compressor(size_t max_buffer_size, int compression_level);

/*
 * xpack_compress() compresses a buffer of data.  The function attempts to
 * compress 'in_nbytes' bytes of data located at 'in' and write the results to
 * 'out', which has space for 'out_nbytes_avail' bytes.  The return value is the
 * compressed size in bytes, or 0 if the data could not be compressed to
 * 'out_nbytes_avail' bytes or fewer.
 */
LIBXPACKAPI size_t
xpack_compress(struct xpack_compressor *compressor,
	       const void *in, size_t in_nbytes,
	       void *out, size_t out_nbytes_avail);

/*
 * xpack_free_compressor() frees a compressor allocated with
 * xpack_alloc_compressor().  If NULL is passed, then no action is taken.
 */
LIBXPACKAPI void
xpack_free_compressor(struct xpack_compressor *compressor);

/* ========================================================================== */
/*                               Decompression                                */
/* ========================================================================== */

struct xpack_decompressor;

/*
 * xpack_alloc_decompressor() allocates a new decompressor.
 *
 * Returns a pointer to the new decompressor, or NULL if out of memory.
 */
LIBXPACKAPI struct xpack_decompressor *
xpack_alloc_decompressor(void);

/* Result of a call to xpack_decompress() */
enum decompress_result {

	/* Decompression was successful */
	DECOMPRESS_SUCCESS = 0,

	/* Decompressed failed because the compressed data was invalid, corrupt,
	 * or otherwise unsupported */
	DECOMPRESS_BAD_DATA = 1,

	/* A NULL 'actual_out_nbytes_ret' was provided, but the data would have
	 * decompressed to fewer than 'out_nbytes_avail' bytes */
	DECOMPRESS_SHORT_OUTPUT = 2,

	/* The data would have decompressed to more than 'out_nbytes_avail'
	 * bytes */
	DECOMPRESS_INSUFFICIENT_SPACE = 3,
};

/*
 * xpack_decompress() decompresses 'in_nbytes' bytes of compressed data at 'in'
 * and writes the uncompressed data to 'out', which is a buffer of at least
 * 'out_nbytes_avail' bytes.  If decompression was successful, then 0
 * (DECOMPRESS_SUCCESS) is returned; otherwise, a nonzero result code such as
 * DECOMPRESS_BAD_DATA is returned.  If a nonzero result code is returned, then
 * the contents of the output buffer are undefined.
 *
 * xpack_decompress() can be used in cases where the actual uncompressed size is
 * known (recommended) or unknown (not recommended):
 *
 *   - If the actual uncompressed size is known, then pass the actual
 *     uncompressed size as 'out_nbytes_avail' and pass NULL for
 *     'actual_out_nbytes_ret'.  This makes xpack_decompress() fail with
 *     DECOMPRESS_SHORT_OUTPUT if the data decompressed to fewer than the
 *     specified number of bytes.
 *
 *   - If the actual uncompressed size is unknown, then provide a non-NULL
 *     'actual_out_nbytes_ret' and provide a buffer with some size
 *     'out_nbytes_avail' that you think is large enough to hold all the
 *     uncompressed data.  In this case, if the data decompresses to less than
 *     or equal to 'out_nbytes_avail' bytes, then xpack_decompress() will write
 *     the actual uncompressed size to *actual_out_nbytes_ret and return 0
 *     (DECOMPRESS_SUCCESS).  Otherwise, it will return
 *     DECOMPRESS_INSUFFICIENT_SPACE if the provided buffer was not large enough
 *     but no other problems were encountered, or another nonzero result code if
 *     decompression failed for another reason.
 */
LIBXPACKAPI enum decompress_result
xpack_decompress(struct xpack_decompressor *decompressor,
		 const void *in, size_t in_nbytes,
		 void *out, size_t out_nbytes_avail,
		 size_t *actual_out_nbytes_ret);

/*
 * xpack_free_decompressor() frees a decompressor allocated with
 * xpack_alloc_decompressor().  If NULL is passed, no action is taken.
 */
LIBXPACKAPI void
xpack_free_decompressor(struct xpack_decompressor *decompressor);


#ifdef __cplusplus
}
#endif

#endif /* LIBXPACK_H */
