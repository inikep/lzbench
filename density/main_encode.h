/*
 * Centaurean Density
 *
 * Copyright (c) 2013, Guillaume Voirin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     1. Redistributions of source code must retain the above copyright notice, this
 *        list of conditions and the following disclaimer.
 *
 *     2. Redistributions in binary form must reproduce the above copyright notice,
 *        this list of conditions and the following disclaimer in the documentation
 *        and/or other materials provided with the distribution.
 *
 *     3. Neither the name of the copyright holder nor the names of its
 *        contributors may be used to endorse or promote products derived from
 *        this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * 18/10/13 23:30
 */

#ifndef DENSITY_MAIN_ENCODE_H
#define DENSITY_MAIN_ENCODE_H

#include <string.h>

#include "memory_location.h"
#include "block_footer.h"
#include "block_header.h"
#include "main_header.h"
#include "main_footer.h"
#include "block_mode_marker.h"
#include "block_encode.h"
#include "kernel_chameleon_encode.h"
#include "kernel_cheetah_encode.h"
#include "kernel_lion_encode.h"
#include "density_api.h"

typedef enum {
    DENSITY_ENCODE_STATE_READY = 0,
    DENSITY_ENCODE_STATE_STALL_ON_INPUT,
    DENSITY_ENCODE_STATE_STALL_ON_OUTPUT,
    DENSITY_ENCODE_STATE_ERROR
} DENSITY_ENCODE_STATE;

typedef enum {
    DENSITY_ENCODE_PROCESS_WRITE_HEADER,
    DENSITY_ENCODE_PROCESS_WRITE_BLOCKS,
    DENSITY_ENCODE_PROCESS_WRITE_FOOTER,
} DENSITY_ENCODE_PROCESS;

#pragma pack(push)
#pragma pack(4)
typedef struct {
    DENSITY_ENCODE_PROCESS process;
    DENSITY_COMPRESSION_MODE compressionMode;
    DENSITY_BLOCK_TYPE blockType;
    const struct stat *fileAttributes;

    uint_fast64_t totalRead;
    uint_fast64_t totalWritten;

    density_block_encode_state blockEncodeState;
} density_encode_state;
#pragma pack(pop)

DENSITY_WINDOWS_EXPORT DENSITY_ENCODE_STATE density_encode_init(density_memory_location *, density_encode_state *, const DENSITY_COMPRESSION_MODE, const DENSITY_BLOCK_TYPE, void *(*mem_alloc)(size_t));

DENSITY_WINDOWS_EXPORT DENSITY_ENCODE_STATE density_encode_continue(density_memory_teleport *, density_memory_location *, density_encode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_ENCODE_STATE density_encode_finish(density_memory_teleport *, density_memory_location *, density_encode_state *, void (*mem_free)(void *));

#endif