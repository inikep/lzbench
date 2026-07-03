/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/**
 * Modifications Copyright (C) 2025-2026, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* Note : this module is expected to remain private, do not expose it */

#ifndef ERROR_H_MODULE
#define ERROR_H_MODULE

/* ****************************************
*  Dependencies
******************************************/
#ifdef AOCL_LLC_PREFIX
#include "../../../common/aoclPrefix.h"
#endif
#include "../zstd_errors.h"  /* enum list */
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
#define CHECK_V_F(e, f)     \
    size_t const e = f;     \
    do {                    \
        if (ERR_isError(e)) \
            return e;       \
    } while (0)
#define CHECK_F(f)   do { CHECK_V_F(_var_err__, f); } while (0)


/*-****************************************
*  Error Strings
******************************************/
#ifdef AOCL_UNIT_TEST /* called by FORWARD_IF_ERROR, etc that are used in unit tests */
#if defined (__cplusplus)
extern "C" {
#endif
const char* ERR_getErrorString(ERR_enum code);   /* error_private.c */
#if defined (__cplusplus)
}
#endif
#else
const char* ERR_getErrorString(ERR_enum code);   /* error_private.c */
#endif

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
#define _FORCE_HAS_FORMAT_STRING(...)              \
    do {                                           \
        if (0) {                                   \
            _force_has_format_string(__VA_ARGS__); \
        }                                          \
    } while (0)

#define ERR_QUOTE(str) #str

/**
 * Return the specified error if the condition evaluates to true.
 *
 * In debug modes, prints additional information.
 * In order to do that (particularly, printing the conditional that failed),
 * this can't just wrap RETURN_ERROR().
 */
#define RETURN_ERROR_IF(cond, err, ...)                                        \
    do {                                                                       \
        if (cond) {                                                            \
            RAWLOG(3, "%s:%d: ERROR!: check %s failed, returning %s",          \
                  __FILE__, __LINE__, ERR_QUOTE(cond), ERR_QUOTE(ERROR(err))); \
            _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                             \
            RAWLOG(3, ": " __VA_ARGS__);                                       \
            RAWLOG(3, "\n");                                                   \
            LOG_FORMATTED(ERR, logCtx, "check %s failed, returning %s",        \
                ERR_QUOTE(cond), ERR_QUOTE(ERROR(err)));                       \
            return ERROR(err);                                                 \
        }                                                                      \
    } while (0)

/**
 * Unconditionally return the specified error.
 *
 * In debug modes, prints additional information.
 */
#define RETURN_ERROR(err, ...)                                                 \
    do {                                                                       \
        RAWLOG(3, "%s:%d: ERROR!: unconditional check failed, returning %s",   \
              __FILE__, __LINE__, ERR_QUOTE(ERROR(err)));                      \
        _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                                 \
        RAWLOG(3, ": " __VA_ARGS__);                                           \
        RAWLOG(3, "\n");                                                       \
        LOG_FORMATTED(ERR, logCtx, "unconditional check failed, returning %s", \
            ERR_QUOTE(ERROR(err)));                                            \
        return ERROR(err);                                                     \
    } while(0)

/**
 * If the provided expression evaluates to an error code, returns that error code.
 *
 * In debug modes, prints additional information.
 */
#define FORWARD_IF_ERROR(err, ...)                                                 \
    do {                                                                           \
        size_t const err_code = (err);                                             \
        if (ERR_isError(err_code)) {                                               \
            RAWLOG(3, "%s:%d: ERROR!: forwarding error in %s: %s",                 \
                  __FILE__, __LINE__, ERR_QUOTE(err), ERR_getErrorName(err_code)); \
            _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                                 \
            RAWLOG(3, ": " __VA_ARGS__);                                           \
            RAWLOG(3, "\n");                                                       \
            LOG_FORMATTED(ERR, logCtx, "forwarding error in %s: %s",               \
                ERR_QUOTE(err), ERR_getErrorName(err_code));                       \
            return err_code;                                                       \
        }                                                                          \
    } while(0)

#endif /* ERROR_H_MODULE */
