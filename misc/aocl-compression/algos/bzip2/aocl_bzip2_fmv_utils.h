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

#ifndef AOCL_BZIP2_FMV_UTILS_H
#define AOCL_BZIP2_FMV_UTILS_H

#include <stddef.h>

#include "utils/dispatcher.h"

/* Compile-time array length helper for static variant tables. */
#define AOCL_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/*
 * Select the first compatible FMV variant from a priority-ordered table.
 *
 * Parameters:
 *  - variants:         Pointer to first table entry.
 *  - variant_count:    Number of table entries.
 *  - variant_size:     Size of each table entry in bytes.
 *  - required_offset:  Byte offset of 'required_features' field in entry.
 *  - cpu_features:     Runtime CPU feature mask.
 *
 * Compatibility rule:
 *  - required_features == 0           -> unconditional fallback match
 *  - otherwise all required bits must be present in cpu_features
 *
 * Return value:
 *  - Index of first matching variant.
 *  - variant_count if no entry matches.
 */
static __inline__
size_t aocl_select_fmv_variant(const void* variants,
                               size_t variant_count,
                               size_t variant_size,
                               size_t required_offset,
                               CpuFeatures cpu_features)
{
    size_t i = 0;
    const unsigned char* bytes = (const unsigned char*)variants;

    for (i = 0; i < variant_count; ++i) {
        const CpuFeatures* required =
            (const CpuFeatures*)(bytes + (i * variant_size) + required_offset);

        if (*required == 0 || (cpu_features & *required) == *required) {
            return i;
        }
    }

    return variant_count;
}

#endif /* AOCL_BZIP2_FMV_UTILS_H */
