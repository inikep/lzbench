/* config.h for compiling liblzma (*not* the whole XZ Utils) with MSVC 2013 */

/* Prefix for symbols exported by tuklib_*.c files */
#define TUKLIB_SYMBOL_PREFIX lzma_

/* How many MiB of RAM to assume if the real amount cannot be determined. */
#define ASSUME_RAM 128

/* Define to 1 if lzma1 decoder is enabled. */
#define HAVE_DECODER_LZMA1 1

/* Define to 1 if lzma2 decoder is enabled. */
#define HAVE_DECODER_LZMA2 1

/* Define to 1 if lzma1 encoder is enabled. */
#define HAVE_ENCODER_LZMA1 1

/* Define to 1 if lzma2 encoder is enabled. */
#define HAVE_ENCODER_LZMA2 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 to enable bt2 match finder. */
#define HAVE_MF_BT2 1

/* Define to 1 to enable bt3 match finder. */
#define HAVE_MF_BT3 1

/* Define to 1 to enable bt4 match finder. */
#define HAVE_MF_BT4 1

/* Define to 1 to enable hc3 match finder. */
#define HAVE_MF_HC3 1

/* Define to 1 to enable hc4 match finder. */
#define HAVE_MF_HC4 1

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 or 0, depending whether the compiler supports simple visibility
   declarations. */
#define HAVE_VISIBILITY 0

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Define to 1 to disable debugging code. */
#define NDEBUG 1

/* The size of `size_t', as computed by sizeof. */
#ifdef _WIN64
#define SIZEOF_SIZE_T 8
#else
#define SIZEOF_SIZE_T 4
#endif

/* Define to 1 if the system supports fast unaligned access to 16-bit and
   32-bit integers. */
#define TUKLIB_FAST_UNALIGNED_ACCESS 1
