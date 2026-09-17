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

/** @file utils.c
 *  
 *  @brief Utility functions that are used by library framework and methods.
 *
 *  This file contains the utility functions to provide functionalities like
 *  CPU feature detection and others.
 *
 *  @author S. Biplab Raut
 */

#include <string.h>
#include "api/types.h"
#include "api/aocl_compression.h"
#include "utils/utils.h"
#include "dispatcher.h"

#ifdef AOCL_ENABLE_LOG_FEATURE

#ifdef __cplusplus
extern "C" {
#endif

aocl_log_ctx logCtx = {-1};

#ifdef __cplusplus
}
#endif

#endif /* AOCL_ENABLE_LOG_FEATURE */

AOCL_VOID set_cpu_opt_flags(AOCL_VOID* handle)
{
    aocl_compression_desc* aocl_codec_handle = (aocl_compression_desc*)handle;
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    aocl_codec_handle->optLevel = (AOCL_INTP) Dispatcher_GetEnvRequestedLevel();

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
}

AOCL_INTP get_cpu_opt_flags(int printDebugLogs)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");

    return (AOCL_INTP) Dispatcher_GetEnvRequestedLevel();
}

AOCL_INTP get_disable_opt_flags(int printDebugLogs)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    const char* AOCL_Enable_Opt = getenv("AOCL_DISABLE_OPT");
    if (AOCL_Enable_Opt != NULL && (strcmp(AOCL_Enable_Opt, "ON") == 0)) {
        LOG_UNFORMATTED(TRACE, logCtx, "Exit");
        return 1;
    }
    
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return 0;
}

#ifdef AOCL_UNIT_TEST
#include <string>
#include <cstring>
#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

unordered_map<string, size_t> unit_test_log_counter;
#ifndef AOCL_ENABLE_THREADS
static atomic_flag unit_test_log_counter_update = ATOMIC_FLAG_INIT;
#endif

#ifdef WIN32
#define SET_ENV_VAR(var) 
#else
#define SET_ENV_VAR(var) putenv(var)
#endif

void update_test_log_counter(const char* name) {
    AOCL_ENTER_CRITICAL(unit_test_log_counter_update);
    if (unit_test_log_counter.find(name) == unit_test_log_counter.end())
        unit_test_log_counter[name] = 0;
    unit_test_log_counter[name]++; /* keep count of function hits */
    AOCL_EXIT_CRITICAL(unit_test_log_counter_update);
}

void clear_test_log_counter(void) {
    AOCL_ENTER_CRITICAL(unit_test_log_counter_update);
    unit_test_log_counter.clear();
    AOCL_EXIT_CRITICAL(unit_test_log_counter_update);
}

int validate_simd_func_access(const aocl_func_info* aocl_simd_funcs, size_t cnt, int maxOptLevel) {
    int access_ok = 1;
    AOCL_ENTER_CRITICAL(unit_test_log_counter_update);
    for (size_t i = 0; i < cnt; ++i) {
        if (aocl_simd_funcs[i].optLevel <= maxOptLevel)
            continue; // ok to have hits for functions with optLevel <= maxOptLevel
        if (unit_test_log_counter.find(aocl_simd_funcs[i].name) != unit_test_log_counter.end()) {
            printf("%s called with maxOptLevel = %d\n",
                aocl_simd_funcs[i].name, maxOptLevel);
            access_ok = 0; // function with optLevel > maxOptLevel had hits
            break;
        }
    }
    AOCL_EXIT_CRITICAL(unit_test_log_counter_update);
    return access_ok;
}

int set_env_var(const char* name, const char* value) {
#ifdef WIN32
    std::string env_var = std::string(name) + "=" + std::string(value);
    return _putenv((char*)(env_var.c_str()));
#else
    return setenv(name, value, 1);
#endif

}

int unset_env_var(const char* name) {
#ifdef WIN32
    std::string env_var = std::string(name) + "=";
    return _putenv((char*)(env_var.c_str()));
#else
    return unsetenv(name);
#endif
}

int set_opt_off(int optOff) {
    switch (optOff) {
    case 0:
        return set_env_var("AOCL_DISABLE_OPT", "OFF");
    case 1:
        return set_env_var("AOCL_DISABLE_OPT", "ON");
    default:
        return unset_env_var("AOCL_DISABLE_OPT");
    }
}

int set_opt_level(int optLevel) {
    switch (optLevel) {
    case 0:
        return set_env_var("AOCL_ENABLE_INSTRUCTIONS", "C");
    case 1:
        return set_env_var("AOCL_ENABLE_INSTRUCTIONS", "SSE2");
    case 2:
        return set_env_var("AOCL_ENABLE_INSTRUCTIONS", "AVX");
    case 3:
        return set_env_var("AOCL_ENABLE_INSTRUCTIONS", "AVX2");
    case 4:
        return set_env_var("AOCL_ENABLE_INSTRUCTIONS", "AVX512");
    default:
        return unset_env_var("AOCL_ENABLE_INSTRUCTIONS");
    }
}

int test_get_enabled_inst(void) {
    return (int) Dispatcher_GetEnvRequestedLevel();
}
#endif
