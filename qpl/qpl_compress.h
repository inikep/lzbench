#ifndef QPL_COMPRESS_H
#define QPL_COMPRESS_H

int64_t qpl_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param, char* workmem);
int64_t qpl_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param, char* workmem);
char* qpl_init(size_t insize, size_t level, size_t path);
void qpl_deinit(char* workmem);

int64_t qpl_hl_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t path, char* workmem);
int64_t qpl_hl_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t path, char* workmem);
char* qpl_hl_init(size_t insize, size_t level, size_t path);
void qpl_hl_deinit(char* workmem);

#endif
