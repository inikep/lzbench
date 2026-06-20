// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/common/logging.h"

#include <stdarg.h> // va_list, va_start, va_end

int ZL_g_logLevel = ZL_LOG_LVL;

// Note: requires -Wno-format-nonliteral to build without a warning.

void ZL_LOG_func(
        const char* file,
        const char* func,
        int line,
        const char* fmt,
        ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VLOG_func(file, func, line, fmt, args);
    va_end(args);
}

void ZL_LOG_func_if_nonempty(
        const char* file,
        const char* func,
        int line,
        const char* prefix,
        const char* fmt,
        ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VLOG_func_if_nonempty(file, func, line, prefix, fmt, args);
    va_end(args);
}

void ZL_VLOG_func(
        const char* file,
        const char* func,
        int line,
        const char* fmt,
        va_list args)
{
    (void)func;
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

void ZL_VLOG_func_if_nonempty(
        const char* file,
        const char* func,
        int line,
        const char* prefix,
        const char* fmt,
        va_list args)
{
    (void)func;
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);
    if (len > 0) {
        fprintf(stderr, "%s:%d: %s", file, line, prefix);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
    }
}

// Note: requires -Wno-format-nonliteral to build without a warning.
void ZL_RLOG_func(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VRLOG_func(fmt, args);
    va_end(args);
}

void ZL_VRLOG_func(const char* fmt, va_list args)
{
    vfprintf(stderr, fmt, args);
}
