/*
  Buffer-to-buffer entry points for lbzip2, for lzbench.  See
  lbzip2_lzbench.c.  Both return 0 on error, otherwise the number of bytes
  written to outbuf.
*/
#ifndef LBZIP2_LZBENCH_H
#define LBZIP2_LZBENCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t lbzip2_buf_compress(const void *inbuf, size_t insize, void *outbuf,
                           size_t outsize, int level);
size_t lbzip2_buf_decompress(const void *inbuf, size_t insize, void *outbuf,
                             size_t outsize);

#ifdef __cplusplus
}
#endif

#endif
