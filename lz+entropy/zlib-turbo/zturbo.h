/* zturbo.h -- zlib-turbo extensions to the zlib interface
 * Copyright (C) 2026 Yingquan (Cody) Wu
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * zlib.h is unchanged in zlib-turbo, and so is everything it declares. This
 * header declares what zlib-turbo adds. Programs that use these functions
 * need zlib-turbo; all others work with either library.
 */

#ifndef ZTURBO_H
#define ZTURBO_H

#include "zlib.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZLIB_TURBO 1
#define ZLIB_TURBO_VERSION "1.1.0"
#define ZLIB_TURBO_VERNUM 0x1100   /* 0xMmp0: major, minor, patch, as ZLIB_VERNUM */

ZEXTERN const char * ZEXPORT zlibTurboVersion(void);
/*
     Returns ZLIB_TURBO_VERSION of the library: zlibVersion() returns the version
   of zlib that zlib-turbo is based on, as zlib does, so that programs that check
   it see no difference. zlibTurboVersion() tells that the library is zlib-turbo.
*/

/*
     Levels 10 to 12

     deflateInit(), deflateInit2(), deflateParams() and compress2() also take
   levels 10, 11 and 12, which compress smaller than level 9 (by 4% to 6% on
   typical data) and more slowly. They are a compressor of their own, with an
   optimal parse: their output is not zlib's (no zlib level has it), but it is
   valid deflate data that every inflater decompresses, the same for the same
   input, flushes and parameters, however the input is fed. They take about
   24 MB per stream, and keep up to 256 KB of input before they compress it;
   flushes work as at other levels. Z_FIXED gives fixed codes only, and
   Z_HUFFMAN_ONLY and Z_RLE work as at other levels. A stream may change
   among levels 10 to 12 at any time, but between them and levels 0 to 9 only
   before its first deflate() call: deflateParams() returns Z_STREAM_ERROR
   otherwise. deflateSetThreads() does not speed them up. Z_BEST_COMPRESSION is
   still 9. A library built with FASTEST, or for 16-bit int, refuses levels
   above 9.
*/

/*
     Multithreaded compression

     With deflateSetThreads(), deflate() searches for matches on worker threads
   at levels 4 to 9, while the calling thread chooses the matches as before.
   The workers also write the blocks, while the calling thread goes on, when
   the output space left for that call of deflate() can take the whole block:
   give deflate() 256 KB of output space or more per call for full speed. The
   compressed output is the same, byte for byte, as with one thread, and so the
   same as zlib's: for any number of threads, any sizes of input and output per
   call, flushes, deflateParams() and dictionaries. Levels 1 to 3,
   Z_HUFFMAN_ONLY and Z_RLE do not use the workers.
*/

ZEXTERN int ZEXPORT deflateSetThreads(z_streamp strm, int threads);
/*
     Lets strm use up to threads threads: the calling thread and threads - 1
   worker threads. A single worker cannot search far enough ahead of the parse
   to gain, so with 2, as with 0 or 1, the stream stays on one thread. The
   workers start at the first call of deflate() that can use them (levels 4 to
   9, not Z_HUFFMAN_ONLY or Z_RLE), so a stream at other levels takes none.
   deflateSetThreads() may be called after deflateInit() or deflateInit2(),
   and before or between calls of deflate(); 2 or less stops the workers, as
   deflateEnd() does. deflateReset() keeps them, and deflateCopy() makes a
   copy that runs without them. Each worker takes about 4 MB.

     The workers read ahead in the input. So when deflate() returns with input
   left (avail_in not zero), the next call must be given the same bytes that
   were left, though they may be at another address: once deflate() returns,
   the workers no longer read next_in.

     Returns Z_OK, or Z_STREAM_ERROR if strm is not valid, threads is negative
   or more than 256, or threads is more than 1 and the library was built
   without threads (configure --no-threads, or CMake with ZLIB_TURBO_THREADS
   off). If memory or a thread cannot be had when the workers start, deflate()
   goes on with the calling thread alone.

     In zlib-turbo 1.0.0, threads was the number of workers, started at once at
   every level, besides the calling thread.
*/

/*
     Prepared dictionaries

     deflateSetDictionary() indexes its dictionary for every stream: up to 32K
   hash insertions before the first byte of input. For many short streams
   compressed with the same dictionary, that is most of the work. A prepared
   dictionary is indexed once and then used by any number of streams, also at
   the same time from different threads: it is read-only once made.

     deflateUseDictionary(strm, dict) has the same effect as
   deflateSetDictionary(strm, dictionary, dictLength) with the dictionary dict
   was made from: the same return value and the same compressed output
   afterwards, byte for byte. On a stream that has the windowBits and memLevel
   of the stream the dictionary was prepared on, right after deflateInit2() or
   deflateReset(), it copies the prepared tables instead of indexing; on any
   other stream it calls deflateSetDictionary().
*/

typedef struct z_dict_s z_dict;

ZEXTERN int ZEXPORT deflatePrepareDictionary(z_streamp strm,
                                             const Bytef *dictionary,
                                             uInt dictLength, z_dict **dict);
/*
     Calls deflateSetDictionary(strm, dictionary, dictLength), and if that
   succeeds, also makes a prepared dictionary from the result and returns it in
   *dict. strm must have been just initialized or reset, before any other call
   on it (deflateParams() and deflatePrime() are fine). The dictionary is
   allocated with strm's zalloc and freed with its zfree, and strm can be ended
   before the dictionary is freed.

     Returns Z_OK, or Z_STREAM_ERROR if a parameter is invalid or strm is not
   in that state, or Z_MEM_ERROR if there was not enough memory, in which case
   the dictionary is still set on strm. *dict is Z_NULL unless Z_OK is
   returned.
*/

ZEXTERN int ZEXPORT deflateUseDictionary(z_streamp strm, const z_dict *dict);
/*
     Same as deflateSetDictionary() with the dictionary that dict was made
   from, see above. Returns what deflateSetDictionary() would return.
*/

ZEXTERN void ZEXPORT deflateFreeDictionary(z_dict *dict);
/*
     Frees a prepared dictionary. dict may be Z_NULL.
*/

#ifdef __cplusplus
}
#endif

#endif /* ZTURBO_H */
