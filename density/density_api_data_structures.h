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
 * 25/12/14 17:08
 */

#ifndef DENSITY_API_DATA_STRUCTURES_H
#define DENSITY_API_DATA_STRUCTURES_H

#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN64) || defined(_WIN32)
#define DENSITY_WINDOWS_EXPORT __declspec(dllexport)
#else
#define DENSITY_WINDOWS_EXPORT
#endif

/***********************************************************************************************************************
 *                                                                                                                     *
 * API data structures                                                                                                 *
 *                                                                                                                     *
 ***********************************************************************************************************************/


typedef uint8_t density_byte;
typedef bool density_bool;

typedef enum {
    DENSITY_COMPRESSION_MODE_COPY = 0,
    DENSITY_COMPRESSION_MODE_CHAMELEON_ALGORITHM = 1,
    DENSITY_COMPRESSION_MODE_CHEETAH_ALGORITHM = 2,
    DENSITY_COMPRESSION_MODE_LION_ALGORITHM = 3,
} DENSITY_COMPRESSION_MODE;

typedef enum {
    DENSITY_BLOCK_TYPE_DEFAULT = 0,                                     // Standard, no integrity check
    DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK = 1                 // Add data integrity check to the stream
} DENSITY_BLOCK_TYPE;

typedef enum {
    DENSITY_BUFFER_STATE_OK = 0,                                        // Everything went alright
    DENSITY_BUFFER_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL,                 // Output buffer size is too small
    DENSITY_BUFFER_STATE_ERROR_DURING_PROCESSING,                       // Error during processing
    DENSITY_BUFFER_STATE_ERROR_INTEGRITY_CHECK_FAIL                     // Integrity check has failed
} DENSITY_BUFFER_STATE;

typedef struct {
    DENSITY_BUFFER_STATE state;
    uint_fast64_t bytesRead;
    uint_fast64_t bytesWritten;
} density_buffer_processing_result;

typedef enum {
    DENSITY_STREAM_STATE_READY = 0,                                     // Awaiting further instructions (new action or adding data to the input buffer)
    DENSITY_STREAM_STATE_STALL_ON_INPUT,                                // There is not enough space left in the input buffer to continue
    DENSITY_STREAM_STATE_STALL_ON_OUTPUT,                               // There is not enough space left in the output buffer to continue
    DENSITY_STREAM_STATE_ERROR_OUTPUT_BUFFER_TOO_SMALL,                 // Output buffer size is too small
    DENSITY_STREAM_STATE_ERROR_INVALID_INTERNAL_STATE,                  // Error during processing
    DENSITY_STREAM_STATE_ERROR_INTEGRITY_CHECK_FAIL                     // Integrity check has failed
} DENSITY_STREAM_STATE;

typedef struct {
    density_byte majorVersion;
    density_byte minorVersion;
    density_byte revision;
    DENSITY_COMPRESSION_MODE compressionMode;
    DENSITY_BLOCK_TYPE blockType;
} density_stream_header_information;

typedef struct {
    void *in;
    uint_fast64_t *totalBytesRead;

    void *out;
    uint_fast64_t *totalBytesWritten;

    void *internal_state;
} density_stream;

#endif

