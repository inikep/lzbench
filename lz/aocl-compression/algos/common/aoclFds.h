/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

 /** @file aoclFds.h
 *
 *  @brief Definitions for Fast Decompression Settings (FDS) mode.
 *
 *  This file contains definitions for Fast Decompression Settings (FDS) mode
 *  that can be used across different algos in AOCL compression library.
 *
 *  @author Ashish Sriram
 */

#ifndef __COMMON_FDS_H
#define __COMMON_FDS_H

#if AOCL_DECOMPRESS_FAST > 1
/*
* AOCL fast decompress settings (FDS) frame format:
*
* | <-- FDS Magic word (8 bytes) --> | <-- Settings (8 bytes) --> |
*
* Settings are specific to each compression method
*/

// AOCL Fast Decompress Settings
#define FDS_MAGIC_WORD 0x5344465F4C434F41 // ASCII encoding of AOCL_FDS
#define FDS_MAGIC_WORD_BYTES 8
#define FDS_METADATA_BYTES 8
#define FDS_FRAME_LENGTH (FDS_MAGIC_WORD_BYTES + FDS_METADATA_BYTES)
#endif /* AOCL_DECOMPRESS_FAST > 1 */

#endif /* __COMMON_FDS_H */
