/*
 * WLZ4 - multi-window, LZ4-class compression: public interface
 * Copyright (c) 2019-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 * Portions derived from LZ4, Copyright (c) 2011-present Yann Collet (BSD 2-Clause); see NOTICE.
 */
#if defined (__cplusplus)
extern "C" {
#endif

#ifndef WLZ_H_1983827168213
#define WLZ_H_1983827168213

/* --- Dependency --- */
#include <stddef.h>   /* size_t */


/**
  Introduction

  The WLZ compression library provides in-memory compression and decompression functions.
  It gives full buffer control to user.
  Compression can be done in:
    - a single step (described as Simple Functions)
    - a single step, reusing a context (described in Advanced Functions)

  WLZ4.h generates and decodes WLZ4 blocks, specified in doc/WLZ4_format.md.
  Decompressing a block requires its compressed size, which the application stores as it wants.

  Block format: the decoded size (2 bytes below 32 KB, else 4), then sequences of a token (literal run in the high
  nibble, match code c in the low one), the literals, and the match, of length 3 + c (code 15: 18 + extension).
  Code 0 takes a one-byte offset (window 256). Every other code takes a flagged offset of 1 + (c > 2) bytes, one more
  when its low bit, the flag, is set; the offset is the other bits: lengths 4-5 one byte (7 bits, window 128) or two
  (15 bits, 32K), lengths 6 and up two bytes or three (23 bits, 8M). Literal runs from 15 and lengths from 18 take
  an extension (one byte below 252, else 251+n and n bytes). A zero offset ends the block.

  A block holds one whole input, with no checksum and no version field; the WZ frame (wzframe.h,
  doc/frame_format.md) adds them, and splits larger content into blocks.
*/

/*^***************************************************************
*  Export parameters
*****************************************************************/
/*
*  WLZ_DLL_EXPORT :
*  Enable exporting of functions when building a Windows DLL
*  WLZLIB_VISIBILITY :
*  Control library symbols visibility.
*/
#ifndef WLZLIB_VISIBILITY
#  if defined(__GNUC__) && (__GNUC__ >= 4)
#    define WLZLIB_VISIBILITY __attribute__ ((visibility ("default")))
#  else
#    define WLZLIB_VISIBILITY
#  endif
#endif
#if defined(WLZ_DLL_EXPORT) && (WLZ_DLL_EXPORT==1)
#  define WLZLIB_API __declspec(dllexport) WLZLIB_VISIBILITY
#elif defined(WLZ_DLL_IMPORT) && (WLZ_DLL_IMPORT==1)
#  define WLZLIB_API __declspec(dllimport) WLZLIB_VISIBILITY /* It isn't required but allows to generate better code, saving a function pointer load from the IAT and an indirect jump.*/
#else
#  define WLZLIB_API WLZLIB_VISIBILITY
#endif

/*------   Version   ------*/
#define WLZ_VERSION_MAJOR    1    /* the library's version, as WZIP_VERSION_* in WZIP.h */
#define WLZ_VERSION_MINOR    1
#define WLZ_VERSION_RELEASE  0

#define WLZ_VERSION_NUMBER (WLZ_VERSION_MAJOR *100*100 + WLZ_VERSION_MINOR *100 + WLZ_VERSION_RELEASE)

#define WLZ_LIB_VERSION WLZ_VERSION_MAJOR.WLZ_VERSION_MINOR.WLZ_VERSION_RELEASE
#define WLZ_QUOTE(str) #str
#define WLZ_EXPAND_AND_QUOTE(str) WLZ_QUOTE(str)
#define WLZ_VERSION_STRING WLZ_EXPAND_AND_QUOTE(WLZ_LIB_VERSION)

WLZLIB_API int WLZ_versionNumber (void);  /**< library version number; useful to check dll version */
WLZLIB_API const char* WLZ_versionString (void);   /**< library version string; useful to check dll version */
 



/*-************************************
*  Advanced Functions
**************************************/
#define WLZ_MAX_INPUT_SIZE       0x7FFFFF00   
#define WLZ_COMP_BOUND           16
#define WLZ_MEM_OVERHEAD          32      /* output room the trusted decoder needs past the decoded size */
#define WLZ_COMPRESSBOUND(n)     ((n) + WLZ_COMP_BOUND + 8)   /* output capacity the compressors require: the output
                                                              is at most n + 15 bytes, plus room for wild copies */


/*-************************************************************
 *  PRIVATE DEFINITIONS
 **************************************************************
 * Do not use these definitions directly.
 * They are only exposed to allow static allocation of `WLZ_stream_Str`.
 * Accessing members will expose code to API and/or ABI break in future versions of the library.
 **************************************************************/

#define WLZ_HASH1BITS   12
#define WLZ_HASH2BITS   15

#define WLZhc_HASH1BITS   13
#define WLZhc_HASH2BITS   16
#define WLZhc_FAR8BITS    20        /* at most: the table takes one entry per 4 input bytes */
/* WLZhc_Compress's window at each level: the top level of each parser (7 and 12) reaches the format's 2^23 bytes, each
   level below it one bit less (levels 0-7: 2^16-2^23, levels 8-12: 2^19-2^23). It caps the far matches (lengths 6+
   beyond 64K) and the tables that find them; inputs no larger than a level's window compress as before. */
#define WLZhc_LEVEL_WINDOW_LOG(level) ((level) >= 8 ? (level) + 11 : (level) + 16)

#if defined(__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#include <stdint.h>

typedef struct WLZ_State_Str WLZ_State_Str;
struct WLZ_State_Str {
	int32_t hash1Table[1 << WLZ_HASH1BITS];
	int32_t *hash2Table;
	uint32_t dictSize;
	const uint8_t* dictEnd;
};

typedef struct WLZhc_State_Str WLZhc_State_Str;
struct WLZhc_State_Str {
	int32_t currIdx;
	int32_t curr1Idx;
	int32_t hash1Table[1 << WLZhc_HASH1BITS];
	int32_t *hash2Table;
	uint16_t *chain2Table;
	uint32_t *farLink;           /* per position (a 64K ring): uncapped distance to the previous one of its 5-byte hash */
	uint32_t *far8Prev;          /* per position (a 64K ring): the previous position of its 8-byte hash */
	uint32_t *far8Head;          /* the latest position of each 8-byte hash */
	uint32_t far8Bits;           /* the size of far8Head in use (log2), by the input size */
	uint32_t farWindow;          /* the far matches' reach, by the level (WLZhc_LEVEL_WINDOW_LOG) */
	uint32_t dictSize;
	const uint8_t* dictEnd;
};

#else

typedef struct WLZ_State_Str WLZ_State_Str;
struct WLZ_State_Str {
	int hash1Table[1 << WLZ_HASH1BITS];
	int *hash2Table;
	unsigned int dictSize;
	const unsigned char* dictEnd;
};

typedef struct WLZhc_State_Str WLZhc_State_Str;
struct WLZhc_State_Str {
	int currIdx;
	int curr1Idx;
	int hash1Table[1 << WLZhc_HASH1BITS];
	int *hash2Table;
	unsigned short *chain2Table;
	unsigned int *farLink;
	unsigned int *far8Prev;
	unsigned int *far8Head;
	unsigned int far8Bits;
	unsigned int farWindow;
	unsigned int dictSize;
	const unsigned char* dictEnd;
};
#endif

/*-*********************************************
*  Streaming Compression Functions
***********************************************/

/*! WLZ_Init_State() :
 *  An WLZ_stream_t structure must be initialized at least once.
 *  This is automatically done when invoking WLZ_createStream(),
 *  but it's not when the structure is simply declared on stack (for example).
 *
 *  Use WLZ_initStream() to properly initialize a newly declared WLZ_stream_t.
 *  It can also initialize any arbitrary buffer of sufficient size,
 *  and will @return a pointer of proper type upon initialization.
 */

WLZLIB_API WLZ_State_Str *WLZ_New_State();
WLZLIB_API void WLZ_Init_State(WLZ_State_Str *lzbStr);
WLZLIB_API void WLZ_Free_State (WLZ_State_Str *lzbStr);

WLZLIB_API WLZhc_State_Str *WLZhc_New_State();
WLZLIB_API void WLZhc_Init_State(WLZhc_State_Str *lzbStr);
WLZLIB_API void WLZhc_Free_State(WLZhc_State_Str *lzbStr);

/*! WLZ_Load_Dictionary() :
 *  Use this function to reference a static dictionary into WLZ_State_Str.
 *  The dictionary must remain available during compression.
 *  WLZ_Load_Dictionary() triggers a reset, so any previous data will be forgotten.
 *  The same dictionary will have to be loaded on decompression side for successful decoding.
 *  Dictionary are useful for better compression of small data (KB range).
 *  While WLZ accept any input as dictionary,
 *  results are generally better when using Zstandard's Dictionary Builder.
 *  Loading a size of 0 is allowed, and is the same as reset.
 * @return : loaded dictionary size, in bytes 
 */
WLZLIB_API unsigned WLZ_Load_Dictionary (WLZ_State_Str* streamPtr, const char* dictionary, unsigned dictSize);

/*! WLZ_Save_Dictionary() :
 *  save it into a safer place (char* safeBuffer).
 *  This is schematically equivalent to a memcpy() followed by WLZ_loadDict(),
 *  but is much faster, because WLZ_Save_Dictionary() doesn't need to rebuild tables.
 * @return : saved dictionary size in bytes (necessarily <= maxDictSize), or 0 if error.
 */
WLZLIB_API unsigned WLZ_Save_Dictionary(WLZ_State_Str* dictStr, WLZ_State_Str* workStr);


/*! WLZ_Read_DecSize() : the decoded size, from the block header: 2 bytes below 32 KB, else 4 (the first two carry
    the flag in their top bit and the low 15 bits, the next two the high bits). 'srcSize' (the compressed size,
    or just the bytes available) only guards the read; 4 bytes always suffice. Returns 0 if too few bytes. */
WLZLIB_API unsigned WLZ_Read_DecSize(const char *source, unsigned srcSize);

/*-************************************
*  Simple Functions
**************************************/
/*! WLZ_compress() :
	Compresses 'srcSize' bytes from buffer 'src'
	into already allocated 'dst' buffer of size 'dstCapacity'.
	The compressors require 'dstCapacity' >= WLZ_COMPRESSBOUND(srcSize) and return 0 otherwise.
	It also runs faster, so it's a recommended setting.
	If the function cannot compress 'src' into a more limited 'dst' budget,
	compression stops *immediately*, and the function result is zero.
	In which case, 'dst' content is undefined (invalid).
		srcSize : max supported value is WLZ_MAX_INPUT_SIZE.
		dstCapacity : size of buffer 'dst' (which must be already allocated)
	   @return  : the number of bytes written into buffer 'dst' (necessarily <= dstCapacity)
				  or 0 if compression fails
	Note : This function is protected against buffer overflow scenarios (never writes outside 'dst' buffer, nor read outside 'source' buffer).
*/
WLZLIB_API unsigned WLZ_Compress(WLZ_State_Str *lzbStr, const char* src, char* dst, unsigned srcSize, unsigned dstCapSize);

/*! WLZhc_Compress() : level 0-7: lazy hash-chain parsing (64K chains, plus probes of the far window); 8-12: optimal
    parsing (the smallest output for the matches found, with a hash chain over the whole 8M window; 12 searches
    deepest, and the far search dominates the time). Levels above 12 act as 12. The loaded dictionary is not used
    (the state is reset). */
WLZLIB_API unsigned WLZhc_Compress(WLZhc_State_Str *lzbStr, const char* src, char* dst, unsigned srcSize, unsigned dstCapSize, int level);


/*! WLZ_Compress_Fast() :
	Same as WLZ_compress(), but allows selection of "acceleration" factor.
	The larger the acceleration value, the faster the algorithm, but also the lesser the compression.
	It's a trade-off. It can be fine tuned, with each successive value providing roughly +~3% to speed.
	An acceleration value of "1" is the same as regular WLZ_compress_default()
	Values <= 0 will be replaced by ACCELERATION_DEFAULT (currently == 1, see WLZ.c).
*/
WLZLIB_API unsigned WLZ_Compress_Fast(WLZ_State_Str *lzbStr, const char* src, char* dst, unsigned srcSize, unsigned dstCapSize, int acceleration);


/*! WLZ_decompress() :
	compressedSize : is the exact complete size of the compressed block.
	dstCapacity : is the size of destination buffer, which must be already allocated.
   @return : the decoded size, or 0 if dstCapacity is too small or the input is malformed.
	Note : the input is validated as LZ4's safe decoder validates its own: whatever the input, no read leaves
	       src[0, compressedSize) or the dictionary, and no write leaves dst[0, dstCapacity); a malformed or truncated
	       input returns 0 (so does a decoded size that differs from the header's).
	'dstCapacity' must hold the decoded size (WLZ_Read_DecSize); nothing is written past it.
*/
WLZLIB_API unsigned WLZ_Decompress(const char* src, char* dst, unsigned compressedSize, unsigned dstCapSize);


WLZLIB_API unsigned WLZ_Compress_wDict(const char *dictionary, unsigned dictSize, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int acceleration);
WLZLIB_API unsigned WLZ_Compress_wDictStr(WLZ_State_Str dictStr, const char* source, char* destiny, unsigned srcSize, unsigned destCapSize, int acceleration);


WLZLIB_API unsigned WLZ_Decompress_wDict(const char* src, char* destiny, unsigned srcSize, unsigned dstCapSize, const char* dictionary, unsigned dictSize);

/*! Trusted mode, opt-in: the same decoding without any check, for input known to come unmodified from a WLZ4 encoder
	(e.g. data this program compressed, or verified by a cryptographic MAC). The stored size still sizes the output
	(dstCapacity >= decoded size + WLZ_MEM_OVERHEAD); the input must stay readable WLZ_TRUSTED_SRC_PAD bytes past
	compressedSize. A damaged stream can make it read or write out of bounds. On Silesia it decodes 9-15% faster than WLZ_Decompress; the paper's figures are this
	mode's. */
#define WLZ_TRUSTED_SRC_PAD       32
WLZLIB_API unsigned WLZ_Decompress_Trusted(const char* src, char* dst, unsigned compressedSize, unsigned dstCapSize);
WLZLIB_API unsigned WLZ_Decompress_wDict_Trusted(const char* src, char* destiny, unsigned srcSize, unsigned dstCapSize, const char* dictionary, unsigned dictSize);


/*^*************************************
 * !!!!!!   STATIC LINKING ONLY   !!!!!!
 ***************************************/

/*-****************************************************************************
 * Experimental section
 *
 * Symbols declared in this section must be considered unstable. Their
 * signatures or semantics may change, or they may be removed altogether in the
 * future. They are therefore only safe to depend on when the caller is
 * statically linked against the library.
 *
 * To protect against unsafe usage, not only are the declarations guarded,
 * the definitions are hidden by default
 * when building WLZ as a shared/dynamic library.
 *
 * In order to access these declarations,
 * define WLZ_STATIC_LINKING_ONLY in your application
 * before including WLZ's headers.
 *
 * In order to make their implementations accessible dynamically, you must
 * define WLZ_PUBLISH_STATIC_FUNCTIONS when building the WLZ library.
 ******************************************************************************/

#ifdef WLZ_PUBLISH_STATIC_FUNCTIONS
#define WLZLIB_STATIC_API WLZLIB_API
#else
#define WLZLIB_STATIC_API
#endif

#ifdef WLZ_STATIC_LINKING_ONLY

WLZLIB_STATIC_API void WLZ_Attach_Dictionary(WLZ_State_Str *workStr, const WLZ_State_Str *dictStr);

#endif

#endif /* WLZ_H_1983827168213 */


#if defined (__cplusplus)
}
#endif
