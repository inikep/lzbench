/**
 * Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
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
#ifndef AOCL_ZLIB_SETUP_H
#define AOCL_ZLIB_SETUP_H

#ifdef AOCL_ZLIB_OPT

#include "utils/dispatcher.h"

extern void aocl_register_slide_hash(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_slide_hash(void);

extern void aocl_register_longest_match(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_longest_match(void);

extern void aocl_setup_adler32(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_adler32(void);

extern void aocl_setup_deflate(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_deflate(void);

extern void aocl_setup_tree(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_tree(void);

extern void aocl_setup_inflate(int optOff, CpuFeatures cpuFeatures);
extern void aocl_destroy_inflate(void);

extern void aocl_setup_count_lengths(int optOff, CpuFeatures cpuFeatures);

#endif /* AOCL_ZLIB_OPT */

#endif /* AOCL_ZLIB_SETUP_H */
