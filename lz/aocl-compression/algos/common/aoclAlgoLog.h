/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

 /** @file aoclAlgoLog.h
 *
 *  @brief Modules to log statistics from AOCL compression algorithms.
 *
 */

#ifndef __COMMON_LOG_H
#define __COMMON_LOG_H

#ifdef AOCL_ENABLE_LOG_FEATURE
#define AOCL_LOG_INIT_STATS() \
  size_t longestSkip = 0; \
  size_t minMatch = (size_t)(-1); \
  size_t maxMatch = 0, avgMatch = 0, cntMatch = 0; \

#define AOCL_LOG_UPDATE_SKIP(skip) { \
  longestSkip = longestSkip > (size_t)(skip) ? longestSkip : (size_t)(skip); \
}

#define AOCL_LOG_UPDATE_MATCH(match) { \
  minMatch = minMatch < (size_t)(match) ? minMatch : (size_t)(match); \
  maxMatch = maxMatch > (size_t)(match) ? maxMatch : (size_t)(match); \
  avgMatch += (size_t)(match); \
  cntMatch++; \
}

#define AOCL_LOG_CLEAR_STATS(input_size) { \
  LOG_FORMATTED(DEBUG, logCtx, "Stats: blockSize=%zu, longestSkip=%zu, minMatch=%zu, maxMatch=%zu, avgMatch=%zu", \
                (size_t)input_size, longestSkip, minMatch == (size_t)(-1) ? 0 : minMatch, \
                maxMatch, cntMatch ? (avgMatch / cntMatch) : 0); \
}

#define AOCL_LOG_API_SUMMARY(level, inSize, outSize) { \
  LOG_FORMATTED(INFO, logCtx, "API: level=%d, inSize=%zu, outSize=%zu, ratio=%.2f", \
                (int)(level), (size_t)(inSize), (size_t)(outSize), \
                (inSize > 0) ? ((double)(outSize) / inSize) : 0.0); \
}
#else
#define AOCL_LOG_INIT_STATS()
#define AOCL_LOG_UPDATE_SKIP(skip)
#define AOCL_LOG_UPDATE_MATCH(match)
#define AOCL_LOG_CLEAR_STATS(input_size)
#define AOCL_LOG_API_SUMMARY(level, inSize, outSize)
#endif

#endif // __COMMON_LOG_H
