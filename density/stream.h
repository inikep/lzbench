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
 * 18/10/13 22:34
 */

#ifndef DENSITY_STREAM_H
#define DENSITY_STREAM_H

#include <stdint.h>

#include "block.h"
#include "globals.h"
#include "main_encode.h"
#include "main_decode.h"
#include "density_api.h"
#include "memory_location.h"

#define DENSITY_STREAM_MEMORY_TELEPORT_BUFFER_SIZE     (1 << 12)

typedef enum {
    DENSITY_STREAM_PROCESS_PREPARED,
    DENSITY_STREAM_PROCESS_COMPRESSION_INITED,
    DENSITY_STREAM_PROCESS_COMPRESSION_STARTED,
    DENSITY_STREAM_PROCESS_COMPRESSION_FINISHED,
    DENSITY_STREAM_PROCESS_DECOMPRESSION_INITED,
    DENSITY_STREAM_PROCESS_DECOMPRESSION_STARTED,
    DENSITY_STREAM_PROCESS_DECOMPRESSION_FINISHED,
} DENSITY_STREAM_PROCESS;

typedef struct {
    DENSITY_STREAM_PROCESS process;

    void *(*mem_alloc)(size_t);
    void (*mem_free)(void *);

    density_encode_state internal_encode_state;
    density_decode_state internal_decode_state;
} density_stream_state;

DENSITY_WINDOWS_EXPORT density_stream *density_stream_create(void *(*)(size_t), void (*)(void *));

DENSITY_WINDOWS_EXPORT void density_stream_destroy(density_stream *);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_prepare(density_stream *, const uint8_t*, const uint_fast64_t, uint8_t*, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_update_input(density_stream *, const uint8_t*, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_update_output(density_stream *, uint8_t *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT uint_fast64_t density_stream_output_available_for_use(density_stream* );

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_compress_init(density_stream *, const DENSITY_COMPRESSION_MODE, const DENSITY_BLOCK_TYPE);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_compress_continue(density_stream *);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_compress_finish(density_stream *);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_decompress_init(density_stream *, density_stream_header_information *);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_decompress_continue(density_stream *);

DENSITY_WINDOWS_EXPORT DENSITY_STREAM_STATE density_stream_decompress_finish(density_stream *);

#endif