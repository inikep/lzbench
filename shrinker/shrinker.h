#ifndef _DATA_SHRINKER_H_
#define _DATA_SHRINKER_H_

#if defined (__cplusplus)
extern "C" {
#endif

int shrinker_compress(void *in, void *out, int size);
/*
in:     inbuf --- source data
out:    outbuf --- compressed data to place in
size:   inbuf size
        

 ******* IMPORTAT *******:
        the outbuf's size MUST equal or greater than that of inbuf
        if size < 32 or size >= 128 MB the function will refuse to run and returns -1

return value:
    positive integer means compress success and it's the size of compressed data,
    or -1 means compress failed which mostly means the data is uncompressable
*/

int shrinker_decompress(void *in, void *out, int size);
/*
in:     inbuf --- compressed data
out:    outbuf --- decompressed data to place in
size:   decompressed(original) data size should be

return value:
    positive integer means decompress success and it's the sizeof decompressed data,
    which should be equal to size.
    or -1 means decompress failed
*/

#if defined (__cplusplus)
}
#endif

#endif
