/**
 * Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
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

/** @file utils.h
 *  
 *  @brief Utility functions related to logger, timer and others.
 *
 *  This file contains the utility functions related to logging mechanism,
 *  timer and others along with the associated data structures and macros.
 *
 *  @author S. Biplab Raut
 */

#ifndef UTILS_H
#define UTILS_H

#define AOCL_CL_STATS

#define AOCL_CL_CPUID_SIMD_DETECTION

#if defined(__GNUC__) && __GNUC__ >= 3
#define FUNC_NAME __func__
#else
#define FUNC_NAME __FUNCTION__
#endif

#ifdef AOCL_ENABLE_LOG_FEATURE
#define ERR      1
#define INFO     2
#define DEBUG    3
#define TRACE    4
#else
#define ERR      0
#define INFO     0
#define DEBUG    0
#define TRACE    0

#endif /* AOCL_ENABLE_LOG_FEATURE */

#ifdef AOCL_CL_STATS
#ifdef _WINDOWS
#define NOMINMAX
#include <windows.h>
typedef LARGE_INTEGER timer;
typedef LARGE_INTEGER timeVal;
#else
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
typedef struct timespec timer;
typedef struct timespec timeVal;
#endif
#endif
#include <stddef.h>

/* Place content that needs to run within a critical section
 * between AOCL_ENTER_CRITICAL() and AOCL_EXIT_CRITICAL() calls */
#ifdef AOCL_ENABLE_THREADS
#define AOCL_BUILD_PRAGMA(x) _Pragma (#x)
#define AOCL_ENTER_CRITICAL(func) AOCL_BUILD_PRAGMA(omp critical (func)) {
#define AOCL_EXIT_CRITICAL(func) }
#else
#ifdef __cplusplus
#include <atomic>
#ifdef AOCL_ENABLE_LOG_FEATURE
static std::atomic_flag setup_logs __attribute__((unused)) = ATOMIC_FLAG_INIT;
#endif
#else
#include <stdatomic.h>
#ifdef AOCL_ENABLE_LOG_FEATURE
static atomic_flag setup_logs __attribute__((unused)) = ATOMIC_FLAG_INIT;
#endif
#endif
#include <immintrin.h> // For _mm_pause()
// Atomic operations for critical section management
#define AOCL_ENTER_CRITICAL(flag) while (atomic_flag_test_and_set(&flag)) {  \
        _mm_pause();                                                                    \
    }
#define AOCL_EXIT_CRITICAL(flag) atomic_flag_clear(&flag);
#endif /* AOCL_ENABLE_THREADS */

#include <string.h>
#include <stdio.h>
#include "types.h"

//Logger - DTL
#ifdef AOCL_ENABLE_LOG_FEATURE

typedef struct {
int maxLevel; // set via AOCL_ENABLE_LOG
//LOG* filePtr; in the future, we can add a pointer to a log file here and pass it instead of logging to stdout
} aocl_log_ctx;

#ifdef __cplusplus
extern "C" {
#endif

    extern aocl_log_ctx logCtx;

#ifdef __cplusplus
}
#endif

#ifdef AOCL_ENABLE_LOG_FEATURE
#ifdef _WINDOWS
#include <stdlib.h>
#define SET_MAX_LOG_LEVEL(logCtx)\
{\
                            AOCL_ENTER_CRITICAL(setup_logs)\
                                char AOCL_Compression_logs[8] = {0};\
                                size_t env_var_size = 8;\
                                getenv_s(&env_var_size, AOCL_Compression_logs, env_var_size, "AOCL_ENABLE_LOG");\
                                if(env_var_size == 0)\
                                    logCtx.maxLevel = 0;\
                                else if(!strcmp(AOCL_Compression_logs, "ERR"))\
                                    logCtx.maxLevel = 1;\
                                else if(!strcmp(AOCL_Compression_logs, "INFO"))\
                                    logCtx.maxLevel = 2;\
                                else if(!strcmp(AOCL_Compression_logs, "DEBUG"))\
                                    logCtx.maxLevel = 3;\
                                else if(!strcmp(AOCL_Compression_logs, "TRACE"))\
                                    logCtx.maxLevel = 4;\
                                else\
                                    logCtx.maxLevel = 0;\
                            AOCL_EXIT_CRITICAL(setup_logs)\
}
#else
#define SET_MAX_LOG_LEVEL(logCtx)\
{\
                            AOCL_ENTER_CRITICAL(setup_logs)\
                                const char * AOCL_Compression_logs = getenv("AOCL_ENABLE_LOG");\
                                if(AOCL_Compression_logs == NULL)\
                                    logCtx.maxLevel = 0;\
                                else if(!strcmp(AOCL_Compression_logs, "ERR"))\
                                    logCtx.maxLevel = 1;\
                                else if(!strcmp(AOCL_Compression_logs, "INFO"))\
                                    logCtx.maxLevel = 2;\
                                else if(!strcmp(AOCL_Compression_logs, "DEBUG"))\
                                    logCtx.maxLevel = 3;\
                                else if(!strcmp(AOCL_Compression_logs, "TRACE"))\
                                    logCtx.maxLevel = 4;\
                                else\
                                    logCtx.maxLevel = 0;\
                            AOCL_EXIT_CRITICAL(setup_logs)\
}
#endif /* _WINDOWS */
#endif

#define LOG_UNFORMATTED(logType, logCtx, str)     do {\
                            SET_MAX_LOG_LEVEL(logCtx)\
                            if (logType <= logCtx.maxLevel)\
                            {\
                                const char *type=NULL;\
                                if (logType == ERR)\
                                    type = "ERR";\
                                else if (logType == INFO)\
                                    type = "INFO";\
                                else if (logType == DEBUG)\
                                    type = "DEBUG";\
                                else if (logType == TRACE)\
                                    type = "TRACE";\
                                if(logType == ERR)\
                                {\
                                    fprintf(stderr, "[%s] : %s : %s : %d : ",\
                                    type, __FILE__, FUNC_NAME, __LINE__);\
                                    fprintf(stderr, str);\
                                    fprintf(stderr, "\n");\
                                }\
                                else\
                                {\
                                    fprintf(stdout, "[%s] : %s : %s : %d : ",\
                                    type, __FILE__, FUNC_NAME, __LINE__);\
                                    fprintf(stdout, str);\
                                    fprintf(stdout, "\n");\
                                }\
                            }\
                        } while (0)
#define LOG_FORMATTED(logType, logCtx, str, ...)     do {\
                            SET_MAX_LOG_LEVEL(logCtx)\
                            if (logType <= logCtx.maxLevel)\
                            {\
                                const char *type=NULL;\
                                if (logType == ERR)\
                                    type = "ERR";\
                                else if (logType == INFO)\
                                    type = "INFO";\
                                else if (logType == DEBUG)\
                                    type = "DEBUG";\
                                else if (logType == TRACE)\
                                    type = "TRACE";\
                                if(logType == ERR)\
                                {\
                                    fprintf(stderr, "[%s] : %s : %s : %d : ",\
                                    type, __FILE__, FUNC_NAME, __LINE__);\
                                    fprintf(stderr, str, __VA_ARGS__);\
                                    fprintf(stderr, "\n");\
                                }\
                                else\
                                {\
                                    fprintf(stdout, "[%s] : %s : %s : %d : ",\
                                    type, __FILE__, FUNC_NAME, __LINE__);\
                                    fprintf(stdout, str, __VA_ARGS__);\
                                    fprintf(stdout, "\n");\
                                }\
                            }\
                        } while (0)
#else /* !AOCL_ENABLE_LOG_FEATURE */
#define LOG_UNFORMATTED(logType, logCtx, str)
#define LOG_FORMATTED(logType, logCtx, str, ...)
#endif

//Timer and stats keeping
#ifdef AOCL_CL_STATS
#ifdef _WINDOWS
#ifdef __cplusplus
#define initTimer(timerClk) if(!QueryPerformanceFrequency(&timerClk))\
                         {\
                            fprintf(stderr, "[ERR] : %s : %s : %d : ",\
                                __FILE__, FUNC_NAME, __LINE__);\
                            fprintf(stderr, "QueryPerformanceFrequency based Timer failed.\n");\
                         }
#else
#define initTimer(timerClk) if(!QueryPerformanceFrequency(&timerClk))\
                         {\
                            printf ("[ERR] : %s : %s : %d : QueryPerformanceFrequency based Timer failed.\n",\
                                __FILE__, FUNC_NAME, __LINE__);\
                         }
#endif /* __cplusplus */
#define getTime(timeVal) QueryPerformanceCounter(&timeVal)
#define diffTime(timerClk, startTime, endTime) ((1000000000ULL * \
                (endTime.QuadPart - startTime.QuadPart))/timerClk.QuadPart)
#else
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#define initTimer(timer)
#define getTime(timeVal) clock_gettime(CLOCK_REALTIME, &timeVal)
#define diffTime(timer, startTime, endTime) (1000000000ULL * \
                (endTime.tv_sec - startTime.tv_sec) + \
                endTime.tv_nsec - startTime.tv_nsec)
#endif
#endif

#if defined(WIN32) && defined(AOCL_UNIT_TEST)
#define EXPORT_UTILS_DYN_TEST __declspec(dllexport)
#else
/**
 * For Linux EXPORT_UTILS_DYN_TEST is NULL, by default the symbols are publicly exposed.
 */
#define EXPORT_UTILS_DYN_TEST
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /* Similar to get_cpu_opt_flags(), but stores the selected
     * optimization level in handle->optLevel. */
    EXPORT_UTILS_DYN_TEST AOCL_VOID set_cpu_opt_flags(AOCL_VOID* handle);

    /* Returns requested optimization level derived from
     * AOCL_ENABLE_INSTRUCTIONS.
     *
     * Mapping:
     *   OPTLEVEL_AUTO   (-1) : variable not set
     *                          (or OPTLEVEL_AVX512 when
     *                           AOCL_DYNAMIC_DISPATCHER is enabled)
     *   OPTLEVEL_SCALAR (0)  : unknown/unsupported value
     *   OPTLEVEL_SSE2   (1)  : SSE2
     *   OPTLEVEL_AVX    (2)  : AVX
     *   OPTLEVEL_AVX2   (3)  : AVX2
     *   OPTLEVEL_AVX512 (4)  : AVX512
     *
     * printDebugLogs is currently unused. */
    EXPORT_UTILS_DYN_TEST ptrdiff_t get_cpu_opt_flags(int printDebugLogs);

    /* Reads the value of AOCL_DISABLE_OPT environment variable
     * return
     *        0 if AOCL_DISABLE_OPT=OFF
     *        1 if AOCL_DISABLE_OPT=ON
     *        0 if not set */
    EXPORT_UTILS_DYN_TEST ptrdiff_t get_disable_opt_flags(int printDebugLogs);

#ifdef __cplusplus
}
#endif


#ifdef AOCL_UNIT_TEST
#ifndef AOCL_ENABLE_THREADS
#define AOCL_SIMD_UNIT_TEST(logType, logCtx, str)     do {\
                            update_test_log_counter(FUNC_NAME); \
                        } while (0)
#else
#define AOCL_SIMD_UNIT_TEST(logType, logCtx, str)
#endif

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct {
        const char* name;
        int optLevel;
    } aocl_func_info;

    /* Keeps count of number of LOG_UNFORMATTED() calls from any function.
    * This can be used to check hits for aocl optimized functions by
    * placing LOG_UNFORMATTED() calls in them. */
    void update_test_log_counter(const char* name);

    EXPORT_UTILS_DYN_TEST void clear_test_log_counter(void);

    /* Call this after running desired compression/decompression apis.
    * aocl_func_info* is the list of aocl optimized functions using simd.
    * maxOptLevel is the max optLevel allowed. compression/decompression apis
    *   are expected to not take any code paths that use simd above this level.
    * This function validates if that is the case based on hits counted by
    *   update_test_log_counter().
    */
    EXPORT_UTILS_DYN_TEST int validate_simd_func_access(const aocl_func_info* aocl_simd_funcs, size_t cnt, int maxOptLevel);

    EXPORT_UTILS_DYN_TEST int set_env_var(const char* name, const char* value);

    EXPORT_UTILS_DYN_TEST int unset_env_var(const char* name);

    EXPORT_UTILS_DYN_TEST int set_opt_off(int optOff);

    EXPORT_UTILS_DYN_TEST int set_opt_level(int optLevel);

    /* Wrapper function to test reading AOCL_ENABLE_INSTRUCTIONS
    * environment variable */
    EXPORT_UTILS_DYN_TEST int test_get_enabled_inst(void);

#ifdef __cplusplus
}
#endif

#else
#define AOCL_SIMD_UNIT_TEST(logType, logCtx, str)
#endif /* AOCL_UNIT_TEST */
#endif
