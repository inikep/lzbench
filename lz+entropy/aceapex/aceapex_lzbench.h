#pragma once
#include "../../bench/codecs.h"

char* lzbench_aceapex_init(size_t insize, size_t level, size_t threads);
void lzbench_aceapex_deinit(char* workmem);
int64_t lzbench_aceapex_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
int64_t lzbench_aceapex_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
char* lzbench_aceapex_stream_init(size_t insize, size_t level, size_t threads);
int64_t lzbench_aceapex_stream_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
int64_t lzbench_aceapex_stream_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
char* lzbench_aceapex3_init(size_t insize, size_t level, size_t threads);
int64_t lzbench_aceapex3_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
int64_t lzbench_aceapex3_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
