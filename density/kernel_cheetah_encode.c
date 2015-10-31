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
 * 06/12/13 20:28
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

#include "kernel_cheetah_encode.h"

DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_exit_process(density_cheetah_encode_state *state, DENSITY_CHEETAH_ENCODE_PROCESS process, DENSITY_KERNEL_ENCODE_STATE kernelEncodeState) {
    state->process = process;
    return kernelEncodeState;
}

DENSITY_FORCE_INLINE void density_cheetah_encode_prepare_new_signature(density_memory_location *restrict out, density_cheetah_encode_state *restrict state) {
    state->signaturesCount++;
    state->shift = 0;
    state->signature = (density_cheetah_signature *) (out->pointer);
    state->proximitySignature = 0;

    out->pointer += sizeof(density_cheetah_signature);
    out->available_bytes -= sizeof(density_cheetah_signature);
}

DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_prepare_new_block(density_memory_location *restrict out, density_cheetah_encode_state *restrict state) {
    if (DENSITY_CHEETAH_MAXIMUM_COMPRESSED_UNIT_SIZE > out->available_bytes)
        return DENSITY_KERNEL_ENCODE_STATE_STALL_ON_OUTPUT;

    switch (state->signaturesCount) {
        case DENSITY_CHEETAH_PREFERRED_EFFICIENCY_CHECK_SIGNATURES:
            if (state->efficiencyChecked ^ 0x1) {
                state->efficiencyChecked = 1;
                return DENSITY_KERNEL_ENCODE_STATE_INFO_EFFICIENCY_CHECK;
            }
            break;
        case DENSITY_CHEETAH_PREFERRED_BLOCK_SIGNATURES:
            state->signaturesCount = 0;
            state->efficiencyChecked = 0;

#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
            if (state->resetCycle)
                state->resetCycle--;
            else {
                density_cheetah_dictionary_reset(&state->dictionary);
                state->resetCycle = DENSITY_DICTIONARY_PREFERRED_RESET_CYCLE - 1;
            }
#endif

            return DENSITY_KERNEL_ENCODE_STATE_INFO_NEW_BLOCK;
        default:
            break;
    }
    density_cheetah_encode_prepare_new_signature(out, state);

    return DENSITY_KERNEL_ENCODE_STATE_READY;
}

DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_check_state(density_memory_location *restrict out, density_cheetah_encode_state *restrict state) {
    DENSITY_KERNEL_ENCODE_STATE returnState;

    switch (state->shift) {
        case density_bitsizeof(density_cheetah_signature):
            DENSITY_MEMCPY(state->signature, &state->proximitySignature, sizeof(density_cheetah_signature));
            if ((returnState = density_cheetah_encode_prepare_new_block(out, state)))
                return returnState;
            break;
        default:
            break;
    }

    return DENSITY_KERNEL_ENCODE_STATE_READY;
}

DENSITY_FORCE_INLINE void density_cheetah_encode_kernel(density_memory_location *restrict out, const uint16_t hash, const uint32_t chunk, const uint_fast8_t shift, density_cheetah_encode_state *restrict state) {
    uint32_t *predictedChunk = (uint32_t*)&state->dictionary.prediction_entries[state->lastHash];

    if (*predictedChunk ^ chunk) {
        density_cheetah_dictionary_entry *found = &state->dictionary.entries[hash];
        uint32_t *found_a = &found->chunk_a;
        if (*found_a ^ chunk) {
            uint32_t *found_b = &found->chunk_b;
            if (*found_b ^ chunk) {
                state->proximitySignature |= ((uint64_t)DENSITY_CHEETAH_SIGNATURE_FLAG_CHUNK << shift);
                DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
                out->pointer += sizeof(uint32_t);
            } else {
                state->proximitySignature |= ((uint64_t)DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_B << shift);
                DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
                out->pointer += sizeof(uint16_t);
            }
            *found_b = *found_a;
            *found_a = chunk;
        } else {
            state->proximitySignature |= ((uint64_t)DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_A << shift);
            DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
            out->pointer += sizeof(uint16_t);
        }
        *predictedChunk = chunk;
    }
    state->lastHash = hash;
}

DENSITY_FORCE_INLINE void density_cheetah_encode_process_unit(density_memory_location *restrict in, density_memory_location *restrict out, density_cheetah_encode_state *restrict state) {
    uint32_t chunk;
    uint_fast8_t count = 0;

#ifdef __clang__
    for(; count < density_bitsizeof(density_cheetah_signature); count += 2) {
        DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t));
        density_cheetah_encode_kernel(out, DENSITY_CHEETAH_HASH_ALGORITHM(chunk), chunk, count, state);
        in->pointer += sizeof(uint32_t);
    }
#else
    for (uint_fast8_t count_b = 0; count_b < 16; count_b++) {
        DENSITY_UNROLL_2(\
            DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t));\
            density_cheetah_encode_kernel(out, DENSITY_CHEETAH_HASH_ALGORITHM(chunk), chunk, count, state);\
            in->pointer += sizeof(uint32_t);\
            count += 2);
    }
#endif

    state->shift = density_bitsizeof(density_cheetah_signature);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_init(density_cheetah_encode_state *state) {
    state->signaturesCount = 0;
    state->efficiencyChecked = 0;
    density_cheetah_dictionary_reset(&state->dictionary);

#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
    state->resetCycle = DENSITY_DICTIONARY_PREFERRED_RESET_CYCLE - 1;
#endif

    state->lastHash = 0;

    return density_cheetah_encode_exit_process(state, DENSITY_CHEETAH_ENCODE_PROCESS_PREPARE_NEW_BLOCK, DENSITY_KERNEL_ENCODE_STATE_READY);
}

#include "kernel_cheetah_encode_template.h"

#define DENSITY_CHEETAH_ENCODE_FINISH

#include "kernel_cheetah_encode_template.h"
