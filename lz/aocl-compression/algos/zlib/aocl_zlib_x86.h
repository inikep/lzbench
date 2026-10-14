/**
 * Copyright (C) 2023-2024, Advanced Micro Devices. All rights reserved.
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

#ifndef AOCL_X86_H
#define AOCL_X86_H
#include "deflate.h"

#ifdef AOCL_ZLIB_OPT
ZEXTERN void slide_hash_x86(deflate_state *s);
ZEXTERN uInt longest_match_x86 (deflate_state *s, IPos cur_match);
ZEXTERN uInt longest_match_lazy_x86 (deflate_state *s, IPos cur_match);

/* Equivalent functions for adler32_x86
 * that do not call AOCL_SETUP_NATIVE(). When these functions are called
 * from other APIs, dynamic dispatcher setup is already done, and overhead
 * from calling AOCL_SETUP_NATIVE() can be avoided. 
 * This function also take care of copying data to sliding window when it 
 * does not have enough data for further processing. It prevents copying
 * twice, once during checksum calculation and other when data is copied
 * to sliding window.
 */
ZEXTERN uint32_t adler32_x86_internal_with_copy(uint32_t adler, Bytef *dst, const Bytef *buf, z_size_t len, const short copy);
#endif

#endif
