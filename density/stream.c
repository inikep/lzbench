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
 * 18/10/13 22:36
 */

#include "stream.h"

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_stream *density_stream_create(void *(*mem_alloc)(size_t), void (*mem_free)(void *)) {
    density_stream *stream;
    void *(*memory_alloc)(size_t) = mem_alloc == NULL ? malloc : mem_alloc;
    void (*memory_free)(void *) = mem_free == NULL ? free : mem_free;

    stream = memory_alloc(sizeof(density_stream));
    stream->in = density_memory_teleport_allocate(DENSITY_STREAM_MEMORY_TELEPORT_BUFFER_SIZE, memory_alloc);
    stream->out = density_memory_location_allocate(memory_alloc);
    stream->internal_state = memory_alloc(sizeof(density_stream_state));
    ((density_stream_state *) stream->internal_state)->mem_alloc = memory_alloc;
    ((density_stream_state *) stream->internal_state)->mem_free = memory_free;

    return stream;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_stream_destroy(density_stream *stream) {
    void (*memory_free)(void *) = ((density_stream_state *) stream->internal_state)->mem_free;
    memory_free(stream->internal_state);
    density_memory_location_free(stream->out, memory_free);
    density_memory_teleport_free(stream->in, memory_free);
    memory_free(stream);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_prepare(density_stream *restrict stream, const uint8_t *restrict in, const uint_fast64_t availableIn, uint8_t *restrict out, const uint_fast64_t availableOut) {
    density_memory_teleport_reset_staging_buffer(stream->in);
    density_stream_update_input(stream, in, availableIn);
    density_stream_update_output(stream, out, availableOut);

    ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_PREPARED;

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_update_input(density_stream *restrict stream, const uint8_t *restrict in, const uint_fast64_t availableIn) {
    density_memory_teleport_change_input_buffer(stream->in, in, availableIn);

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_update_output(density_stream *restrict stream, uint8_t *out, const uint_fast64_t availableOut) {
    density_memory_location_encapsulate(stream->out, out, availableOut);

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE uint_fast64_t density_stream_output_available_for_use(density_stream *stream) {
    return density_memory_location_used((density_memory_location *) stream->out);
}

DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_check_conformity(density_stream *stream) {
    if (((density_memory_location *) stream->out)->initial_available_bytes < DENSITY_MINIMUM_OUTPUT_BUFFER_SIZE)
        return DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL;

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_compress_init(density_stream *restrict stream, const DENSITY_COMPRESSION_MODE compressionMode, const DENSITY_BLOCK_TYPE blockType) {
    if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_PREPARED)
        return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;

    DENSITY_STREAM_STATE streamState = density_stream_check_conformity(stream);
    if (streamState)
        return streamState;

    DENSITY_ENCODE_STATE encodeState = density_encode_init(stream->out, &((density_stream_state *) stream->internal_state)->internal_encode_state, compressionMode, blockType, ((density_stream_state *) stream->internal_state)->mem_alloc);
    switch (encodeState) {
        case DENSITY_ENCODE_STATE_READY:
            break;

        case DENSITY_ENCODE_STATE_STALL_ON_INPUT:
            return DENSITY_STREAM_STATE_STALL_ON_INPUT;

        case DENSITY_ENCODE_STATE_STALL_ON_OUTPUT:
            return DENSITY_STREAM_STATE_STALL_ON_OUTPUT;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }

    stream->totalBytesRead = &((density_stream_state *) stream->internal_state)->internal_encode_state.totalRead;
    stream->totalBytesWritten = &((density_stream_state *) stream->internal_state)->internal_encode_state.totalWritten;

    ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_COMPRESSION_INITED;

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_compress_continue(density_stream *stream) {
    if(((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_COMPRESSION_STARTED) {
        if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_COMPRESSION_INITED)
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
        else
            ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_COMPRESSION_STARTED;
    }

    DENSITY_STREAM_STATE streamState = density_stream_check_conformity(stream);
    if (streamState)
        return streamState;

    DENSITY_ENCODE_STATE encodeState = density_encode_continue(stream->in, stream->out, &((density_stream_state *) stream->internal_state)->internal_encode_state);
    switch (encodeState) {
        case DENSITY_ENCODE_STATE_READY:
            return DENSITY_STREAM_STATE_READY;

        case DENSITY_ENCODE_STATE_STALL_ON_INPUT:
            return DENSITY_STREAM_STATE_STALL_ON_INPUT;

        case DENSITY_ENCODE_STATE_STALL_ON_OUTPUT:
            return DENSITY_STREAM_STATE_STALL_ON_OUTPUT;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_compress_finish(density_stream *stream) {
    if(((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_COMPRESSION_STARTED) {
        if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_COMPRESSION_INITED)
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
        else
            ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_COMPRESSION_STARTED;
    }

    DENSITY_STREAM_STATE streamState = density_stream_check_conformity(stream);
    if (streamState)
        return streamState;

    DENSITY_ENCODE_STATE encodeState = density_encode_finish(stream->in, stream->out, &((density_stream_state *) stream->internal_state)->internal_encode_state, ((density_stream_state *) stream->internal_state)->mem_free);
    switch (encodeState) {
        case DENSITY_ENCODE_STATE_READY:
            break;

        case DENSITY_ENCODE_STATE_STALL_ON_OUTPUT:
            return DENSITY_STREAM_STATE_STALL_ON_OUTPUT;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }

    ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_COMPRESSION_FINISHED;

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_decompress_init(density_stream *restrict stream, density_stream_header_information *restrict headerInformation) {
    if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_PREPARED)
        return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;

    DENSITY_STREAM_STATE streamState = density_stream_check_conformity(stream);
    if (streamState)
        return streamState;

    DENSITY_DECODE_STATE decodeState = density_decode_init(stream->in, &((density_stream_state *) stream->internal_state)->internal_decode_state, ((density_stream_state *) stream->internal_state)->mem_alloc);
    switch (decodeState) {
        case DENSITY_DECODE_STATE_READY:
            break;

        case DENSITY_DECODE_STATE_STALL_ON_INPUT:
            return DENSITY_STREAM_STATE_STALL_ON_INPUT;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }

    stream->totalBytesRead = &((density_stream_state *) stream->internal_state)->internal_decode_state.totalRead;
    stream->totalBytesWritten = &((density_stream_state *) stream->internal_state)->internal_decode_state.totalWritten;

    ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_DECOMPRESSION_INITED;

    if(headerInformation != NULL) {
        density_main_header header = ((density_stream_state *) stream->internal_state)->internal_decode_state.header;
        headerInformation->majorVersion = header.version[0];
        headerInformation->minorVersion = header.version[1];
        headerInformation->revision = header.version[2];
        headerInformation->compressionMode = (DENSITY_COMPRESSION_MODE)header.compressionMode;
        headerInformation->blockType = (DENSITY_BLOCK_TYPE)header.blockType;
    }

    return DENSITY_STREAM_STATE_READY;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_decompress_continue(density_stream *stream) {
    if(((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_DECOMPRESSION_STARTED) {
        if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_DECOMPRESSION_INITED)
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
        else
            ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_DECOMPRESSION_STARTED;
    }

    DENSITY_STREAM_STATE streamState = density_stream_check_conformity(stream);
    if (streamState)
        return streamState;

    DENSITY_DECODE_STATE decodeState = density_decode_continue(stream->in, stream->out, &((density_stream_state *) stream->internal_state)->internal_decode_state);
    switch (decodeState) {
        case DENSITY_DECODE_STATE_READY:
            return DENSITY_STREAM_STATE_READY;

        case DENSITY_DECODE_STATE_STALL_ON_INPUT:
            return DENSITY_STREAM_STATE_STALL_ON_INPUT;

        case DENSITY_DECODE_STATE_STALL_ON_OUTPUT:
            return DENSITY_STREAM_STATE_STALL_ON_OUTPUT;

        case DENSITY_DECODE_STATE_INTEGRITY_CHECK_FAIL:
            return DENSITY_STREAM_STATE_ERROR_INTEGRITY_CHECK_FAIL;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_STREAM_STATE density_stream_decompress_finish(density_stream *stream) {
    if(((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_DECOMPRESSION_STARTED) {
        if (((density_stream_state *) stream->internal_state)->process ^ DENSITY_STREAM_PROCESS_DECOMPRESSION_INITED)
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
        else
            ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_DECOMPRESSION_STARTED;
    }

    DENSITY_DECODE_STATE decodeState = density_decode_finish(stream->in, stream->out, &((density_stream_state *) stream->internal_state)->internal_decode_state, ((density_stream_state *) stream->internal_state)->mem_free);
    switch (decodeState) {
        case DENSITY_DECODE_STATE_READY:
            break;

        case DENSITY_DECODE_STATE_STALL_ON_OUTPUT:
            return DENSITY_STREAM_STATE_STALL_ON_OUTPUT;

        case DENSITY_DECODE_STATE_INTEGRITY_CHECK_FAIL:
            return DENSITY_STREAM_STATE_ERROR_INTEGRITY_CHECK_FAIL;

        default:
            return DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE;
    }

    ((density_stream_state *) stream->internal_state)->process = DENSITY_STREAM_PROCESS_DECOMPRESSION_FINISHED;

    return DENSITY_STREAM_STATE_READY;
}