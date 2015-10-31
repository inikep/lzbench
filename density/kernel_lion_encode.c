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

#include "kernel_lion_encode.h"

DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_lion_encode_exit_process(density_lion_encode_state *state, DENSITY_LION_ENCODE_PROCESS process, DENSITY_KERNEL_ENCODE_STATE kernelEncodeState) {
    state->process = process;
    return kernelEncodeState;
}

DENSITY_FORCE_INLINE void density_lion_encode_prepare_new_signature(density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    state->signature = (density_lion_signature *) (out->pointer);
    state->proximitySignature = 0;

    out->pointer += sizeof(density_lion_signature);
}

DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_lion_encode_check_block_state(density_lion_encode_state *restrict state) {
    if(density_unlikely(!(state->chunksCount & (DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG - 1)))) {
        if (density_unlikely((state->chunksCount >= DENSITY_LION_PREFERRED_EFFICIENCY_CHECK_CHUNKS) && (!state->efficiencyChecked))) {
            state->efficiencyChecked = true;
            return DENSITY_KERNEL_ENCODE_STATE_INFO_EFFICIENCY_CHECK;
        } else if (density_unlikely(state->chunksCount >= DENSITY_LION_PREFERRED_BLOCK_CHUNKS)) {
            state->chunksCount = 0;
            state->efficiencyChecked = false;

#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
            if (state->resetCycle)
                state->resetCycle--;
            else {
                density_lion_dictionary_reset(&state->dictionary);
                state->resetCycle = DENSITY_DICTIONARY_PREFERRED_RESET_CYCLE - 1;
            }
#endif

            return DENSITY_KERNEL_ENCODE_STATE_INFO_NEW_BLOCK;
        }
    }

    return DENSITY_KERNEL_ENCODE_STATE_READY;
}

DENSITY_FORCE_INLINE void density_lion_encode_push_to_proximity_signature(density_lion_encode_state *restrict state, const uint64_t content, const uint_fast8_t bits) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    state->proximitySignature |= (content << state->shift);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    state->proximitySignature |= (content << ((56 - (state->shift & ~0x7)) + (state->shift & 0x7)));
#else
#error Unknow endianness
#endif

    state->shift += bits;
}

DENSITY_FORCE_INLINE void density_lion_encode_push_to_signature(density_memory_location *restrict out, density_lion_encode_state *restrict state, const uint64_t content, const uint_fast8_t bits) {
    if (density_likely(state->shift)) {
        density_lion_encode_push_to_proximity_signature(state, content, bits);

        if (density_unlikely(state->shift >= density_bitsizeof(density_lion_signature))) {
            DENSITY_MEMCPY(state->signature, &state->proximitySignature, sizeof(density_lion_signature));

            const uint_fast8_t remainder = (uint_fast8_t)(state->shift & 0x3f);
            state->shift = 0;
            if (remainder) {
                density_lion_encode_prepare_new_signature(out, state);
                density_lion_encode_push_to_proximity_signature(state, content >> (bits - remainder), remainder);
            }
        }
    } else {
        density_lion_encode_prepare_new_signature(out, state);
        density_lion_encode_push_to_proximity_signature(state, content, bits);
    }
}

DENSITY_FORCE_INLINE void density_lion_encode_push_zero_to_signature(density_memory_location *restrict out, density_lion_encode_state *restrict state, const uint_fast8_t bits) {
    if (density_likely(state->shift)) {
        state->shift += bits;

        if (density_unlikely(state->shift >= density_bitsizeof(density_lion_signature))) {
            DENSITY_MEMCPY(state->signature, &state->proximitySignature, sizeof(density_lion_signature));

            const uint_fast8_t remainder = (uint_fast8_t)(state->shift & 0x3f);
            if (remainder) {
                density_lion_encode_prepare_new_signature(out, state);
                state->shift = remainder;
            } else
                state->shift = 0;
        }
    } else {
        density_lion_encode_prepare_new_signature(out, state);
        state->shift = bits;
    }
}

DENSITY_FORCE_INLINE void density_lion_encode_push_code_to_signature(density_memory_location *restrict out, density_lion_encode_state *restrict state, const density_lion_entropy_code code) {
    density_lion_encode_push_to_signature(out, state, code.value, code.bitLength);
}

DENSITY_FORCE_INLINE void density_lion_encode_kernel(density_memory_location *restrict out, const uint16_t hash, const uint32_t chunk, density_lion_encode_state *restrict state) {
    density_lion_dictionary *const dictionary = &state->dictionary;
    density_lion_dictionary_chunk_prediction_entry *const predictions = &dictionary->predictions[state->lastHash];
    __builtin_prefetch(&dictionary->predictions[hash]);

    if (*(uint32_t *) predictions ^ chunk) {
        if (*((uint32_t *) predictions + 1) ^ chunk) {
            if (*((uint32_t *) predictions + 2) ^ chunk) {
                density_lion_dictionary_chunk_entry *const in_dictionary = &dictionary->chunks[hash];
                if (*(uint32_t *) in_dictionary ^ chunk) {
                    if (*((uint32_t *) in_dictionary + 1) ^ chunk) {
                        if (*((uint32_t *) in_dictionary + 2) ^ chunk) {
                            if (*((uint32_t *) in_dictionary + 3) ^ chunk) {
                                density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_PLAIN));
                                DENSITY_MEMCPY(out->pointer, &chunk, sizeof(uint32_t));
                                out->pointer += sizeof(uint32_t);
                            } else {
                                density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_DICTIONARY_D));
                                DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
                                out->pointer += sizeof(uint16_t);
                            }
                        } else {
                            density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_DICTIONARY_C));
                            DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
                            out->pointer += sizeof(uint16_t);
                        }
                    } else {
                        density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_DICTIONARY_B));
                        DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
                        out->pointer += sizeof(uint16_t);
                    }
                    DENSITY_MEMMOVE((uint32_t*)in_dictionary + 1, in_dictionary, 3 * sizeof(uint32_t));
                    *(uint32_t *) in_dictionary = chunk;
                } else {
                    density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_DICTIONARY_A));
                    DENSITY_MEMCPY(out->pointer, &hash, sizeof(uint16_t));
                    out->pointer += sizeof(uint16_t);
                }
            } else {
                density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_PREDICTIONS_C));
            }
        } else {
            density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_PREDICTIONS_B));
        }
        DENSITY_MEMMOVE((uint32_t*)predictions + 1, predictions, 2 * sizeof(uint32_t));
        *(uint32_t *) predictions = chunk;
    } else
        density_lion_encode_push_code_to_signature(out, state, density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_PREDICTIONS_A));
    state->lastHash = hash;
}

DENSITY_FORCE_INLINE void density_lion_encode_process_unit_generic(const uint_fast8_t chunks_per_process_unit, const uint_fast16_t process_unit_size, density_memory_location *restrict in, density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    uint32_t chunk;

#ifdef __clang__
    for (uint_fast8_t count = 0; count < (chunks_per_process_unit >> 2); count++) {
        DENSITY_UNROLL_4(\
            DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t));\
            density_lion_encode_kernel(out, DENSITY_LION_HASH_ALGORITHM(chunk), chunk, state);\
            in->pointer += sizeof(uint32_t));
    }
#else
    for (uint_fast8_t count = 0; count < (chunks_per_process_unit >> 1); count++) {
        DENSITY_UNROLL_2(\
            DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t));\
            density_lion_encode_kernel(out, DENSITY_LION_HASH_ALGORITHM(chunk), chunk, state);\
            in->pointer += sizeof(uint32_t));
    }
#endif

    state->chunksCount += chunks_per_process_unit;
    in->available_bytes -= process_unit_size;
}

DENSITY_FORCE_INLINE void density_lion_encode_process_unit_small(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    density_lion_encode_process_unit_generic(DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_SMALL, DENSITY_LION_PROCESS_UNIT_SIZE_SMALL, in, out, state);
}

DENSITY_FORCE_INLINE void density_lion_encode_process_unit_big(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    density_lion_encode_process_unit_generic(DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG, DENSITY_LION_PROCESS_UNIT_SIZE_BIG, in, out, state);
}

DENSITY_FORCE_INLINE void density_lion_encode_process_step_unit(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    uint32_t chunk;
    DENSITY_MEMCPY(&chunk, in->pointer, sizeof(uint32_t));
    density_lion_encode_kernel(out, DENSITY_LION_HASH_ALGORITHM(DENSITY_LITTLE_ENDIAN_32(chunk)), chunk, state);
    state->chunksCount++;

    in->pointer += sizeof(uint32_t);
    in->available_bytes -= sizeof(uint32_t);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE density_lion_encode_init(density_lion_encode_state *state) {
    state->chunksCount = 0;
    state->efficiencyChecked = false;
    state->signature = NULL;
    state->shift = 0;
    density_lion_dictionary_reset(&state->dictionary);

#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
    state->resetCycle = DENSITY_DICTIONARY_PREFERRED_RESET_CYCLE - 1;
#endif

    density_lion_form_model_init(&state->formData);

    state->lastHash = 0;
    state->lastChunk = 0;

    state->signatureInterceptMode = false;
    state->endMarker = false;

    return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE, DENSITY_KERNEL_ENCODE_STATE_READY);
}

#include "kernel_lion_encode_template.h"

#define DENSITY_LION_ENCODE_FINISH

#include "kernel_lion_encode_template.h"
