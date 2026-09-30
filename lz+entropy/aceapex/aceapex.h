#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ACEAPEX_VERSION_MAJOR 2
#define ACEAPEX_VERSION_MINOR 2
#define ACEAPEX_VERSION_PATCH 0
#define ACEAPEX_VERSION_STRING "2.2.0"

/* One-shot compression */
int64_t aceapex_compress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    int         level,    /* 1=fast, 2=default, 3=l1 encoder on any input (DNA uses l1 at every level, ADR-020) */
    int         threads   /* 0=auto */
);

/* One-shot decompression; uses every hardware thread. */
int64_t aceapex_decompress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity
);

/* Same with an explicit thread budget for the whole decode (entropy phase and match
   phase). threads = 1 decodes on the calling thread only and creates none, which is
   what a benchmark harness with its own thread pool (lzbench -T) needs; 0 = auto. */
int64_t aceapex_decompress_mt(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    int         threads
);

/* Decompress only the bytes [offset, offset+length) of the original input,
   without decoding the rest of the archive. dst must hold at least length bytes.
   Returns length on success, or a negative error code.

   Cost is set by the block size and, for archives written with chunked literals,
   by the chunk size: the decoder touches the blocks covering the range and, in
   each of the four streams, only the compressed chunks those blocks fall into.
   Both are recorded in the archive; nothing is read from the environment. */
int64_t aceapex_decompress_region(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    uint64_t    offset, uint64_t length
);

/* Batch region read.

   One region read costs about the same whether it returns 16 KB or 1 KB, because the
   cost is dominated by unpacking the chunks that cover it. When many ranges are wanted
   from one archive, the same block is often decoded repeatedly: 10,000 random 16 KiB
   ranges over a 254 MB genome touch 11,342 distinct blocks, and 100,000 ranges touch
   all 15,499. Requesting them together lets the decoder group by block, unpack each
   one once and hand out slices, so the cost follows the number of distinct blocks
   rather than the number of requests.

   Outputs are written in the order the requests were given, regardless of the order
   in which they were decoded. Each entry's 'written' field receives the byte count,
   or a negative error code for that entry alone; one bad range does not fail the batch.

   threads = 0 lets the implementation choose. */
typedef struct {
    uint64_t offset;      /* byte offset into the original input */
    uint64_t length;      /* bytes wanted */
    void*    dst;         /* caller's buffer, at least length bytes */
    int64_t  written;     /* out: bytes written, or a negative error code */
} aceapex_range_t;

int64_t aceapex_decompress_ranges(
    const void*       src, size_t src_size,
    aceapex_range_t*  ranges, size_t count,
    int               threads
);

/* Bound for output buffer */
size_t aceapex_compress_bound(size_t src_size);

/* Streaming context (v2.0) */
typedef struct aceapex_stream_s aceapex_stream_t;

aceapex_stream_t* aceapex_stream_new(int level, int threads);
int  aceapex_stream_update(aceapex_stream_t* s,
                           const void* in,  size_t in_size,
                           void*       out, size_t out_capacity,
                           size_t*     out_written);
int  aceapex_stream_finish(aceapex_stream_t* s,
                           void*   out, size_t out_capacity,
                           size_t* out_written);
void aceapex_stream_free(aceapex_stream_t* s);

/* Error codes */
#define ACEAPEX_OK           0
#define ACEAPEX_ERR_BUFFER  -1
#define ACEAPEX_ERR_DATA    -2
#define ACEAPEX_ERR_MEMORY  -3

/* The entropy layer only: the four decoded streams and the block table of an archive,
   for a caller that runs the match phase itself (the lzbench CUDA decoder). Returns 0
   or a negative error code; free with aceapex_streams_free. An empty archive yields
   zero streams and num_blocks 0. */
typedef struct {
    uint8_t *lit, *off, *len, *cmd;     /* decoded streams (malloc'd) */
    uint64_t lit_sz, off_sz, len_sz, cmd_sz;
    void    *boffs_vec;                 /* owner (opaque), freed by aceapex_streams_free */
    const void *boffs;                  /* BlockOffsets[num_blocks], 64 bytes each */
    size_t   num_blocks;
    uint32_t block_size;
    uint64_t orig_size;
} aceapex_streams_t;
int  aceapex_decode_streams(const void* src, size_t src_size, aceapex_streams_t* out);
void aceapex_streams_free(aceapex_streams_t* s);

#ifdef __cplusplus
}
#endif
