/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
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

#ifndef AOCL_LZMA_DISPATCH_VARIANTS_H
#define AOCL_LZMA_DISPATCH_VARIANTS_H

#include "utils/dispatcher.h"

/*
 * Shared variant entry layouts used by lzma FMV registration tables.
 *
 * Notes:
 *  - required_features is kept as the first field in each variant entry.
 *  - Tables are expected to be ordered from highest-priority to fallback.
 */

typedef enum {
    AOCL_LZMA_ENCODE_PROFILE_BASELINE = 0,
    AOCL_LZMA_ENCODE_PROFILE_OPT
} AoclLzmaEncodeDispatchProfile;

typedef struct {
    CpuFeatures required_features;
    AoclLzmaEncodeDispatchProfile profile;
} AoclLzmaEncodeDispatchVariant;

typedef enum {
    AOCL_LZMA_DECODE_PROFILE_BASELINE = 0,
    AOCL_LZMA_DECODE_PROFILE_OPT
} AoclLzmaDecodeDispatchProfile;

typedef struct {
    CpuFeatures required_features;
    AoclLzmaDecodeDispatchProfile profile;
} AoclLzmaDecodeDispatchVariant;

#endif /* AOCL_LZMA_DISPATCH_VARIANTS_H */
