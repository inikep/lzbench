#ifndef LZJB_H

#define LZJB_H 

#ifdef __cplusplus
extern "C" {
#endif

#define uchar_t unsigned char

size_t lzjb_compress2010(uchar_t *s_start, uchar_t *d_start, size_t s_len, size_t d_len, int n);
size_t lzjb_decompress2010(uchar_t *s_start, uchar_t *d_start, size_t s_len, size_t d_len, int n);

#ifdef __cplusplus
}
#endif


#endif // #ifndef LZJB_H

