/*
 * Centaurean Density
 *
 * Copyright (c) 2015, Guillaume Voirin
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
 * 3/02/15 19:53
 */

#include "buffer.h"

DENSITY_FORCE_INLINE density_buffer_processing_result density_buffer_return_processing_result(density_stream* stream, DENSITY_BUFFER_STATE state) {
    density_buffer_processing_result result;
    result.state = state;
    DENSITY_MEMCPY(&result.bytesRead, stream->totalBytesRead, sizeof(uint64_t));
    DENSITY_MEMCPY(&result.bytesWritten, stream->totalBytesWritten, sizeof(uint64_t));
    density_stream_destroy(stream);

    return result;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_buffer_processing_result density_buffer_compress(const uint8_t *restrict input_buffer, const uint_fast64_t input_size, uint8_t *restrict output_buffer, const uint_fast64_t output_size, const DENSITY_COMPRESSION_MODE compression_mode, const DENSITY_BLOCK_TYPE block_type, void *(*mem_alloc)(size_t), void (*mem_free)(void *)) {
    DENSITY_STREAM_STATE streamState;
    density_stream* stream = density_stream_create(mem_alloc, mem_free);

    streamState = density_stream_prepare(stream, input_buffer, input_size, output_buffer, output_size);
    if(streamState)
        return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);

    streamState = density_stream_compress_init(stream, compression_mode, block_type);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
            break;
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    }

    streamState = density_stream_compress_continue(stream);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
        case DENSITY_STREAM_STATE_STALL_ON_INPUT:
            break;
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    };


    streamState = density_stream_compress_finish(stream);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_OK);
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    }
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_buffer_processing_result density_buffer_decompress(const uint8_t *restrict input_buffer, const uint_fast64_t input_size, uint8_t *restrict output_buffer, const uint_fast64_t output_size, void *(*mem_alloc)(size_t), void (*mem_free)(void *)) {
    DENSITY_STREAM_STATE streamState;
    density_stream* stream = density_stream_create(mem_alloc, mem_free);

    streamState = density_stream_prepare(stream, input_buffer, input_size, output_buffer, output_size);
    if(streamState)
        return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);

    streamState = density_stream_decompress_init(stream, NULL);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
            break;
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    }

    streamState = density_stream_decompress_continue(stream);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
        case DENSITY_STREAM_STATE_STALL_ON_INPUT:
        case DENSITY_STREAM_STATE_STALL_ON_OUTPUT:
            break;
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        case DENSITY_STREAM_STATE_ERROR_INTEGRITY_CHECK_FAIL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_INTEGRITY_CHECK_FAIL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    }


    streamState = density_stream_decompress_finish(stream);
    switch(streamState) {
        case DENSITY_STREAM_STATE_READY:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_OK);
        case DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL);
        case DENSITY_STREAM_STATE_ERROR_INTEGRITY_CHECK_FAIL:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_INTEGRITY_CHECK_FAIL);
        default:
            return density_buffer_return_processing_result(stream, DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING);
    }
}
