#ifndef LZMA_ALONE_ENCODER_H
#define LZMA_ALONE_ENCODER_H

#if defined (__cplusplus)
extern "C"
{
#endif
    int64_t xz_alone_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t);
    int64_t xz_alone_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t x, size_t y);
#if defined (__cplusplus) 
}
#endif

#endif
