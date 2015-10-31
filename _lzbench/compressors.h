#include <stdlib.h> 

typedef size_t (*compress_func)(char *in, size_t insize, char *out, size_t outsize, size_t, size_t, size_t);


#ifndef BENCH_REMOVE_BROTLI
	size_t bench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_brotli_compress NULL
	#define bench_brotli_decompress NULL
#endif


#ifndef BENCH_REMOVE_CRUSH
	size_t bench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_crush_compress NULL
	#define bench_crush_decompress NULL
#endif


#ifndef BENCH_REMOVE_CSC
	size_t bench_csc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_csc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_csc_compress NULL
	#define bench_csc_decompress NULL
#endif


#ifndef BENCH_REMOVE_DENSITY
	size_t bench_density_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_density_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_density_compress NULL
	#define bench_density_decompress NULL
#endif


#ifndef BENCH_REMOVE_FASTLZ
	size_t bench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_fastlz_compress NULL
	#define bench_fastlz_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZ4
	size_t bench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize,  size_t level, size_t, size_t);
	size_t bench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize,  size_t level, size_t, size_t);
	size_t bench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lz4_compress NULL
	#define bench_lz4fast_compress NULL
	#define bench_lz4hc_compress NULL
	#define bench_lz4_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZ5
	size_t bench_lz5_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lz5fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize,  size_t level, size_t, size_t);
	size_t bench_lz5hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize,  size_t level, size_t, size_t);
	size_t bench_lz5_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lz5_compress NULL
	#define bench_lz5fast_compress NULL
	#define bench_lz5hc_compress NULL
	#define bench_lz5_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZF
	size_t bench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzf_compress NULL
	#define bench_lzf_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZHAM
	size_t bench_lzham_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzham_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzham_compress NULL
	#define bench_lzham_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZJB
	size_t bench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzjb_compress NULL
	#define bench_lzjb_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZMA
	size_t bench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzma_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzma_compress NULL
	#define bench_lzma_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZMAT
	size_t bench_lzmat_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzmat_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzmat_compress NULL
	#define bench_lzmat_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZO
	size_t bench_lzo_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzo_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzo_compress NULL
	#define bench_lzo_decompress NULL
#endif


#ifndef BENCH_REMOVE_LZRW
	size_t bench_lzrw_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_lzrw_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_lzrw_compress NULL
	#define bench_lzrw_decompress NULL
#endif



#ifndef BENCH_REMOVE_PITHY
	size_t bench_pithy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_pithy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_pithy_compress NULL
	#define bench_pithy_decompress NULL
#endif


#ifndef BENCH_REMOVE_QUICKLZ
	size_t bench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_quicklz_compress NULL
	#define bench_quicklz_decompress NULL
#endif


#ifndef BENCH_REMOVE_SHRINKER
	size_t bench_shrinker_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_shrinker_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_shrinker_compress NULL
	#define bench_shrinker_decompress NULL
#endif


#ifndef BENCH_REMOVE_SNAPPY
	size_t bench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_snappy_compress NULL
	#define bench_snappy_decompress NULL
#endif


#ifndef BENCH_REMOVE_TORNADO
	size_t bench_tornado_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_tornado_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_tornado_compress NULL
	#define bench_tornado_decompress NULL
#endif


#ifndef BENCH_REMOVE_UCL
	size_t bench_ucl_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_ucl_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_ucl_compress NULL
	#define bench_ucl_decompress NULL
#endif


#ifndef BENCH_REMOVE_WFLZ
	size_t bench_wflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_wflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_wflz_compress NULL
	#define bench_wflz_decompress NULL
#endif


#ifndef BENCH_REMOVE_YAPPY
	size_t bench_yappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
	size_t bench_yappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_yappy_compress NULL
	#define bench_yappy_decompress NULL
#endif


#ifndef BENCH_REMOVE_ZLIB
	size_t bench_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
	size_t bench_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_zlib_compress NULL
	#define bench_zlib_decompress NULL
#endif


#ifndef BENCH_REMOVE_ZLING
	size_t bench_zling_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
	size_t bench_zling_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_zling_compress NULL
	#define bench_zling_decompress NULL
#endif


#ifndef BENCH_REMOVE_ZSTD
	size_t bench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
	size_t bench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_zstd_compress NULL
	#define bench_zstd_decompress NULL
#endif


#ifndef BENCH_REMOVE_ZSTDHC
	size_t bench_zstdhc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
	size_t bench_zstdhc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t);
#else
	#define bench_zstdhc_compress NULL
	#define bench_zstdhc_decompress NULL
#endif

