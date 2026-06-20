/*
 * Copyright (c) Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/* Note : this module is expected to remain private, do not expose it */

#ifndef ERROR_H_MODULE
#define ERROR_H_MODULE

#include "rename.h"

#if defined (__cplusplus)
extern "C" {
#endif


/* ****************************************
*  Dependencies
******************************************/
#include "compiler.h"
#include "debug.h"
#include "zstd_deps.h"       /* size_t */


/* ****************************************
*  Compiler-specific
******************************************/
#if defined(__GNUC__)
#  define ERR_STATIC static __attribute__((unused))
#elif defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#  define ERR_STATIC static inline
#elif defined(_MSC_VER)
#  define ERR_STATIC static __inline
#else
#  define ERR_STATIC static  /* this version may generate warnings for unused static functions; disable the relevant warning */
#endif


/*-****************************************
*  Customization (error_public.h)
******************************************/
typedef enum {
  ZSTD_error_no_error = 0,
  ZSTD_error_GENERIC  = 1,
  ZSTD_error_prefix_unknown                = 10,
  ZSTD_error_version_unsupported           = 12,
  ZSTD_error_frameParameter_unsupported    = 14,
  ZSTD_error_frameParameter_windowTooLarge = 16,
  ZSTD_error_corruption_detected = 20,
  ZSTD_error_checksum_wrong      = 22,
  ZSTD_error_dictionary_corrupted      = 30,
  ZSTD_error_dictionary_wrong          = 32,
  ZSTD_error_dictionaryCreation_failed = 34,
  ZSTD_error_parameter_unsupported   = 40,
  ZSTD_error_parameter_outOfBound    = 42,
  ZSTD_error_tableLog_tooLarge       = 44,
  ZSTD_error_maxSymbolValue_tooLarge = 46,
  ZSTD_error_maxSymbolValue_tooSmall = 48,
  ZSTD_error_stabilityCondition_notRespected = 50,
  ZSTD_error_stage_wrong       = 60,
  ZSTD_error_init_missing      = 62,
  ZSTD_error_memory_allocation = 64,
  ZSTD_error_workSpace_tooSmall= 66,
  ZSTD_error_dstSize_tooSmall = 70,
  ZSTD_error_srcSize_wrong    = 72,
  ZSTD_error_dstBuffer_null   = 74,
  /* following error codes are __NOT STABLE__, they can be removed or changed in future versions */
  ZSTD_error_frameIndex_tooLarge = 100,
  ZSTD_error_seekableIO          = 102,
  ZSTD_error_dstBuffer_wrong     = 104,
  ZSTD_error_srcBuffer_wrong     = 105,
  ZSTD_error_maxCode = 120  /* never EVER use this value directly, it can change in future versions! Use ZSTD_isError() instead */
} ZSTD_ErrorCode;

typedef ZSTD_ErrorCode ERR_enum;
#define PREFIX(name) ZSTD_error_##name


/*-****************************************
*  Error codes handling
******************************************/
#undef ERROR   /* already defined on Visual Studio */
#define ERROR(name) ZSTD_ERROR(name)
#define ZSTD_ERROR(name) ((size_t)-PREFIX(name))

ERR_STATIC unsigned ERR_isError(size_t code) { return (code > ERROR(maxCode)); }

ERR_STATIC ERR_enum ERR_getErrorCode(size_t code) { if (!ERR_isError(code)) return (ERR_enum)0; return (ERR_enum) (0-code); }

/* check and forward error code */
#define CHECK_V_F(e, f) size_t const e = f; if (ERR_isError(e)) return e
#define CHECK_F(f)   { CHECK_V_F(_var_err__, f); }


/*-****************************************
*  Error Strings
******************************************/

const char* ERR_getErrorString(ERR_enum code);   /* error_private.c */

ERR_STATIC const char* ERR_getErrorName(size_t code)
{
    return ERR_getErrorString(ERR_getErrorCode(code));
}

/**
 * Ignore: this is an internal helper.
 *
 * This is a helper function to help force C99-correctness during compilation.
 * Under strict compilation modes, variadic macro arguments can't be empty.
 * However, variadic function arguments can be. Using a function therefore lets
 * us statically check that at least one (string) argument was passed,
 * independent of the compilation flags.
 */
static INLINE_KEYWORD UNUSED_ATTR
void _force_has_format_string(const char *format, ...) {
  (void)format;
}

/**
 * Ignore: this is an internal helper.
 *
 * We want to force this function invocation to be syntactically correct, but
 * we don't want to force runtime evaluation of its arguments.
 */
#define _FORCE_HAS_FORMAT_STRING(...) \
  if (0) { \
    _force_has_format_string(__VA_ARGS__); \
  }

#define ERR_QUOTE(str) #str

/**
 * Return the specified error if the condition evaluates to true.
 *
 * In debug modes, prints additional information.
 * In order to do that (particularly, printing the conditional that failed),
 * this can't just wrap RETURN_ERROR().
 */
#define RETURN_ERROR_IF(cond, err, ...) \
  if (cond) { \
    RAWLOG(3, "%s:%d: ERROR!: check %s failed, returning %s", \
           __FILE__, __LINE__, ERR_QUOTE(cond), ERR_QUOTE(ERROR(err))); \
    _FORCE_HAS_FORMAT_STRING(__VA_ARGS__); \
    RAWLOG(3, ": " __VA_ARGS__); \
    RAWLOG(3, "\n"); \
    return ERROR(err); \
  }

/**
 * Unconditionally return the specified error.
 *
 * In debug modes, prints additional information.
 */
#define RETURN_ERROR(err, ...) \
  do { \
    RAWLOG(3, "%s:%d: ERROR!: unconditional check failed, returning %s", \
           __FILE__, __LINE__, ERR_QUOTE(ERROR(err))); \
    _FORCE_HAS_FORMAT_STRING(__VA_ARGS__); \
    RAWLOG(3, ": " __VA_ARGS__); \
    RAWLOG(3, "\n"); \
    return ERROR(err); \
  } while(0);

/**
 * If the provided expression evaluates to an error code, returns that error code.
 *
 * In debug modes, prints additional information.
 */
#define FORWARD_IF_ERROR(err, ...) \
  do { \
    size_t const err_code = (err); \
    if (ERR_isError(err_code)) { \
      RAWLOG(3, "%s:%d: ERROR!: forwarding error in %s: %s", \
             __FILE__, __LINE__, ERR_QUOTE(err), ERR_getErrorName(err_code)); \
      _FORCE_HAS_FORMAT_STRING(__VA_ARGS__); \
      RAWLOG(3, ": " __VA_ARGS__); \
      RAWLOG(3, "\n"); \
      return err_code; \
    } \
  } while(0);

#if defined (__cplusplus)
}
#endif

#endif /* ERROR_H_MODULE */
