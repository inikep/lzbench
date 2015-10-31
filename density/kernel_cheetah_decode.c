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
 * 07/12/13 15:49
 *
 * -----------------
 * Cheetah algorithm
 * -----------------
 *
 * Author(s)
 * Guillaume Voirin (https://github.com/gpnuma)
 * Piotr Tarsa (https://github.com/tarsa)
 *
 * Description
 * Very fast two level dictionary hash algorithm derived from Chameleon, with predictions lookup
 */

#include "kernel_cheetah_decode.h"

DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_cheetah_decode_exit_process(density_cheetah_decode_state *state, DENSITY_CHEETAH_DECODE_PROCESS process, DENSITY_KERNEL_DECODE_STATE kernelDecodeState) {
    state->process = process;
    return kernelDecodeState;
}

DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_cheetah_decode_check_state(density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    if (out->available_bytes < DENSITY_CHEETAH_DECOMPRESSED_UNIT_SIZE)
        return DENSITY_KERNEL_DECODE_STATE_STALL_ON_OUTPUT;

    switch (state->signaturesCount) {
        case DENSITY_CHEETAH_PREFERRED_EFFICIENCY_CHECK_SIGNATURES:
            if (state->efficiencyChecked ^ 0x1) {
                state->efficiencyChecked = 1;
                return DENSITY_KERNEL_DECODE_STATE_INFO_EFFICIENCY_CHECK;
            }
            break;
        case DENSITY_CHEETAH_PREFERRED_BLOCK_SIGNATURES:
            state->signaturesCount = 0;
            state->efficiencyChecked = 0;

            if (state->resetCycle)
                state->resetCycle--;
            else {
                density_byte resetDictionaryCycleShift = state->parameters.as_bytes[0];
                if (resetDictionaryCycleShift) {
                    density_cheetah_dictionary_reset(&state->dictionary);
                    state->resetCycle = (uint_fast64_t) (1 << resetDictionaryCycleShift) - 1;
                }
            }

            return DENSITY_KERNEL_DECODE_STATE_INFO_NEW_BLOCK;
        default:
            break;
    }
    return DENSITY_KERNEL_DECODE_STATE_READY;
}

DENSITY_FORCE_INLINE void density_cheetah_decode_read_signature(density_memory_location *restrict in, density_cheetah_decode_state *restrict state) {
    DENSITY_MEMCPY(&state->signature, in->pointer, sizeof(density_cheetah_signature));
    in->pointer += sizeof(density_cheetah_signature);
    state->shift = 0;
    state->signaturesCount++;
}

DENSITY_FORCE_INLINE void density_cheetah_decode_process_predicted(density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    const uint32_t chunk = state->dictionary.prediction_entries[state->lastHash].next_chunk_prediction;
    DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
    state->lastHash = DENSITY_CHEETAH_HASH_ALGORITHM(chunk);
}

DENSITY_FORCE_INLINE void density_cheetah_decode_process_compressed_a(const uint16_t hash, density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    __builtin_prefetch(&state->dictionary.prediction_entries[hash]);
    const uint32_t chunk = state->dictionary.entries[hash].chunk_a;
    DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
    state->dictionary.prediction_entries[state->lastHash].next_chunk_prediction = chunk;
    state->lastHash = hash;
}

DENSITY_FORCE_INLINE void density_cheetah_decode_process_compressed_b(const uint16_t hash, density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    __builtin_prefetch(&state->dictionary.prediction_entries[hash]);
    density_cheetah_dictionary_entry *const entry = &state->dictionary.entries[hash];
    const uint32_t chunk = entry->chunk_b;
    entry->chunk_b = entry->chunk_a;
    entry->chunk_a = chunk;
    DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
    state->dictionary.prediction_entries[state->lastHash].next_chunk_prediction = chunk;
    state->lastHash = hash;
}

DENSITY_FORCE_INLINE void density_cheetah_decode_process_uncompressed(const uint32_t chunk, density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    const uint16_t hash = DENSITY_CHEETAH_HASH_ALGORITHM(chunk);
    __builtin_prefetch(&state->dictionary.prediction_entries[hash]);
    density_cheetah_dictionary_entry *const entry = &state->dictionary.entries[hash];
    entry->chunk_b = entry->chunk_a;
    entry->chunk_a = chunk;
    DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
    state->dictionary.prediction_entries[state->lastHash].next_chunk_prediction = chunk;
    state->lastHash = hash;
}

DENSITY_FORCE_INLINE void density_cheetah_decode_kernel(density_memory_location *restrict in, density_memory_location *restrict out, const uint8_t mode, density_cheetah_decode_state *restrict state) {
    uint16_t hash;
    uint32_t chunk;

    switch (mode) {
        DENSITY_CASE_GENERATOR_4_4_COMBINED(\
            density_cheetah_decode_process_predicted(out, state);, \
            DENSITY_CHEETAH_SIGNATURE_FLAG_PREDICTED, \
            DENSITY_MEMCPY(&hash, in->pointer, sizeof(uint16_t)); \
            density_cheetah_decode_process_compressed_a(hash, out, state);\
            in->pointer += sizeof(uint16_t);, \
            DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_A, \
            DENSITY_MEMCPY(&hash, in->pointer, sizeof(uint16_t)); \
            density_cheetah_decode_process_compressed_b(hash, out, state);\
            in->pointer += sizeof(uint16_t);, \
            DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_B, \
            DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t)); \
            density_cheetah_decode_process_uncompressed(chunk, out, state);\
            in->pointer += sizeof(uint32_t);, \
            DENSITY_CHEETAH_SIGNATURE_FLAG_CHUNK, \
            out->pointer += sizeof(uint32_t);, \
            2\
        );
        default:
            break;
    }

    out->pointer += sizeof(uint32_t);
}

DENSITY_FORCE_INLINE void density_cheetah_decode_process_data(density_memory_location *restrict in, density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    uint_fast8_t count = 0;

#ifdef __clang__
    for (uint_fast8_t count_b = 0; count_b < 8; count_b ++) {
        density_cheetah_decode_kernel(in, out, (uint8_t const) ((state->signature >> count) & 0xFF), state);
        count += 8;
    }
#else
    for (uint_fast8_t count_b = 0; count_b < density_bitsizeof(density_cheetah_signature); count_b += 8)
        density_cheetah_decode_kernel(in, out, (uint8_t const) ((state->signature >> count_b) & 0xFF), state);
#endif

    state->shift = density_bitsizeof(density_cheetah_signature);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_cheetah_decode_init(density_cheetah_decode_state *state, const density_main_header_parameters parameters, const uint_fast8_t endDataOverhead) {
    state->signaturesCount = 0;
    state->efficiencyChecked = 0;
    density_cheetah_dictionary_reset(&state->dictionary);

    state->parameters = parameters;
    density_byte resetDictionaryCycleShift = state->parameters.as_bytes[0];
    if (resetDictionaryCycleShift)
        state->resetCycle = (uint_fast64_t) (1 << resetDictionaryCycleShift) - 1;

    state->endDataOverhead = endDataOverhead;

    state->lastHash = 0;

    return density_cheetah_decode_exit_process(state, DENSITY_CHEETAH_DECODE_PROCESS_CHECK_SIGNATURE_STATE, DENSITY_KERNEL_DECODE_STATE_READY);
}

#include "kernel_cheetah_decode_template.h"

#define DENSITY_CHEETAH_DECODE_FINISH

#include "kernel_cheetah_decode_template.h"
