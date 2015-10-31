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
 * 8/03/15 11:59
 *
 * --------------
 * Lion algorithm
 * --------------
 *
 * Author(s)
 * Guillaume Voirin (https://github.com/gpnuma)
 *
 * Description
 * Multiform compression algorithm
 */

#ifndef DENSITY_LION_DECODE_H
#define DENSITY_LION_DECODE_H

#include "kernel_lion_dictionary.h"
#include "kernel_lion.h"
#include "kernel_lion_form_model.h"
#include "block.h"
#include "kernel_decode.h"
#include "density_api.h"
#include "memory_teleport.h"
#include "block_footer.h"
#include "main_footer.h"
#include "block_mode_marker.h"
#include "kernel_lion_encode.h"
#include "main_header.h"

#define DENSITY_LION_DECODE_MAX_BITS_TO_READ_FOR_CHUNK              (density_bitsizeof(density_lion_signature) + 3 + 2 * (3 + density_bitsizeof(uint16_t)))  // 8 bytes (new signature) + 3 bits (lowest rank form) + 2 * (3 bit flags (DENSITY_LION_FORM_SECONDARY_ACCESS + DENSITY_LION_BIGRAM_PRIMARY_SIGNATURE_FLAG_SECONDARY_ACCESS + DENSITY_LION_BIGRAM_SECONDARY_SIGNATURE_FLAG_PLAIN) + 2 bytes)
#define DENSITY_LION_DECODE_MAX_BYTES_TO_READ_FOR_PROCESS_UNIT      (1 + ((DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG * DENSITY_LION_DECODE_MAX_BITS_TO_READ_FOR_CHUNK) >> 3))

typedef enum {
    DENSITY_LION_DECODE_PROCESS_CHECK_BLOCK_STATE,
    DENSITY_LION_DECODE_PROCESS_CHECK_OUTPUT_SIZE,
    DENSITY_LION_DECODE_PROCESS_UNIT,
} DENSITY_LION_DECODE_PROCESS;

typedef enum {
    DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_PROCEED = 0,
    DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_STALL_ON_OUTPUT,
    DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_END_MARKER
} DENSITY_LION_DECODE_STEP_BY_STEP_STATUS;

#define DENSITY_LION_DECODE_BITMASK_VALUES {\
    DENSITY_BINARY_TO_UINT(0),\
    DENSITY_BINARY_TO_UINT(1),\
    DENSITY_BINARY_TO_UINT(11),\
    DENSITY_BINARY_TO_UINT(111),\
    DENSITY_BINARY_TO_UINT(1111),\
    DENSITY_BINARY_TO_UINT(11111),\
    DENSITY_BINARY_TO_UINT(111111),\
    DENSITY_BINARY_TO_UINT(1111111),\
}

#define DENSITY_LION_DECODE_NUMBER_OF_BITMASK_VALUES                                    8

#pragma pack(push)
#pragma pack(4)
typedef struct {
    density_lion_signature signature;
    density_lion_form_data formData;
    uint_fast16_t lastHash;
    uint32_t lastChunk;
    uint_fast8_t shift;
    uint_fast64_t chunksCount;
    bool efficiencyChecked;

    DENSITY_LION_DECODE_PROCESS process;
    uint_fast8_t endDataOverhead;
    density_main_header_parameters parameters;

    density_lion_dictionary dictionary;
    uint_fast64_t resetCycle;
} density_lion_decode_state;
#pragma pack(pop)

DENSITY_WINDOWS_EXPORT void density_lion_decode_prediction_a(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_prediction_b(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_prediction_c(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_dictionary_a(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_dictionary_b(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_dictionary_c(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_dictionary_d(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT void density_lion_decode_plain(density_memory_location *, density_memory_location *, density_lion_decode_state *, uint16_t *const, uint32_t *const);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_lion_decode_init(density_lion_decode_state *, const density_main_header_parameters parameters, const uint_fast8_t);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_lion_decode_continue(density_memory_teleport *, density_memory_location *, density_lion_decode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_lion_decode_finish(density_memory_teleport *, density_memory_location *, density_lion_decode_state *);

#endif