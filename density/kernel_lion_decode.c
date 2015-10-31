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

#include "kernel_lion_decode.h"

const uint8_t density_lion_decode_bitmasks[DENSITY_LION_DECODE_NUMBER_OF_BITMASK_VALUES] = DENSITY_LION_DECODE_BITMASK_VALUES;

DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_lion_decode_exit_process(density_lion_decode_state *state, DENSITY_LION_DECODE_PROCESS process, DENSITY_KERNEL_DECODE_STATE kernelDecodeState) {
    state->process = process;
    return kernelDecodeState;
}

DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_lion_decode_check_block_state(density_lion_decode_state *restrict state) {
    if (density_unlikely((state->chunksCount >= DENSITY_LION_PREFERRED_EFFICIENCY_CHECK_CHUNKS) && (!state->efficiencyChecked))) {
        state->efficiencyChecked = true;
        return DENSITY_KERNEL_DECODE_STATE_INFO_EFFICIENCY_CHECK;
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

        return DENSITY_KERNEL_DECODE_STATE_INFO_NEW_BLOCK;
    }

    return DENSITY_KERNEL_DECODE_STATE_READY;
}

DENSITY_FORCE_INLINE void density_lion_decode_read_signature_from_memory(density_memory_location *restrict in, density_lion_decode_state *restrict state) {
    DENSITY_MEMCPY(&state->signature, in->pointer, sizeof(density_lion_signature));
    in->pointer += sizeof(density_lion_signature);
}

DENSITY_FORCE_INLINE void density_lion_decode_update_predictions_model(density_lion_dictionary_chunk_prediction_entry *const restrict predictions, const uint32_t chunk) {
    DENSITY_MEMMOVE((uint32_t *) predictions + 1, predictions, 2 * sizeof(uint32_t));
    *(uint32_t *) predictions = chunk;     // Move chunk to the top of the predictions list
}

DENSITY_FORCE_INLINE void density_lion_decode_update_dictionary_model(density_lion_dictionary_chunk_entry *const restrict entry, const uint32_t chunk) {
    DENSITY_MEMMOVE((uint32_t*)entry + 1, entry, 3 * sizeof(uint32_t));
    *(uint32_t *) entry = chunk;
}

DENSITY_FORCE_INLINE void density_lion_decode_read_hash(density_memory_location *restrict in, uint16_t *restrict const hash) {
    DENSITY_MEMCPY(hash, in->pointer, sizeof(uint16_t));
    in->pointer += sizeof(uint16_t);
}

DENSITY_FORCE_INLINE void density_lion_decode_prediction_generic(density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    *hash = DENSITY_LION_HASH_ALGORITHM(*chunk);
    DENSITY_MEMCPY(out->pointer, chunk, sizeof(uint32_t));
    out->pointer += sizeof(uint32_t);
}

DENSITY_FORCE_INLINE void density_lion_decode_dictionary_generic(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    DENSITY_MEMCPY(out->pointer, chunk, sizeof(uint32_t));
    out->pointer += sizeof(uint32_t);
    density_lion_dictionary_chunk_prediction_entry *p = &(state->dictionary.predictions[state->lastHash]);
    density_lion_decode_update_predictions_model(p, *chunk);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_prediction_a(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    *chunk = state->dictionary.predictions[state->lastHash].next_chunk_a;
    density_lion_decode_prediction_generic(out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_prediction_b(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_dictionary_chunk_prediction_entry *const p = &state->dictionary.predictions[state->lastHash];
    *chunk = p->next_chunk_b;
    density_lion_decode_update_predictions_model(p, *chunk);
    density_lion_decode_prediction_generic(out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_prediction_c(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_dictionary_chunk_prediction_entry *const p = &state->dictionary.predictions[state->lastHash];
    *chunk = p->next_chunk_c;
    density_lion_decode_update_predictions_model(p, *chunk);
    density_lion_decode_prediction_generic(out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_dictionary_a(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_decode_read_hash(in, hash);
    __builtin_prefetch(&state->dictionary.predictions[*hash]);
    *chunk = state->dictionary.chunks[*hash].chunk_a;
    density_lion_decode_dictionary_generic(in, out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_dictionary_b(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_decode_read_hash(in, hash);
    __builtin_prefetch(&state->dictionary.predictions[*hash]);
    density_lion_dictionary_chunk_entry *entry = &state->dictionary.chunks[*hash];
    *chunk = entry->chunk_b;
    density_lion_decode_update_dictionary_model(entry, *chunk);
    density_lion_decode_dictionary_generic(in, out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_dictionary_c(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_decode_read_hash(in, hash);
    __builtin_prefetch(&state->dictionary.predictions[*hash]);
    density_lion_dictionary_chunk_entry *entry = &state->dictionary.chunks[*hash];
    *chunk = entry->chunk_c;
    density_lion_decode_update_dictionary_model(entry, *chunk);
    density_lion_decode_dictionary_generic(in, out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_dictionary_d(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    density_lion_decode_read_hash(in, hash);
    __builtin_prefetch(&state->dictionary.predictions[*hash]);
    density_lion_dictionary_chunk_entry *entry = &state->dictionary.chunks[*hash];
    *chunk = entry->chunk_d;
    density_lion_decode_update_dictionary_model(entry, *chunk);
    density_lion_decode_dictionary_generic(in, out, state, hash, chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_lion_decode_plain(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, uint16_t *restrict const hash, uint32_t *restrict const chunk) {
    DENSITY_MEMCPY(chunk, in->pointer, sizeof(uint32_t));
    in->pointer += sizeof(uint32_t);
    *hash = DENSITY_LION_HASH_ALGORITHM(*chunk);
    density_lion_dictionary_chunk_entry *entry = &state->dictionary.chunks[*hash];
    density_lion_decode_update_dictionary_model(entry, *chunk);
    DENSITY_MEMCPY(out->pointer, chunk, sizeof(uint32_t));
    out->pointer += sizeof(uint32_t);
    density_lion_dictionary_chunk_prediction_entry *p = &(state->dictionary.predictions[state->lastHash]);
    density_lion_decode_update_predictions_model(p, *chunk);

    state->lastChunk = *chunk;
    state->lastHash = *hash;
}

DENSITY_FORCE_INLINE void density_lion_decode_chunk(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state, const DENSITY_LION_FORM form) {
    uint16_t hash;
    uint32_t chunk;

    state->formData.attachments[form](in, out, state, &hash, &chunk);
}

DENSITY_FORCE_INLINE const DENSITY_LION_FORM density_lion_decode_read_form(density_memory_location *restrict in, density_lion_decode_state *restrict state) {
    density_lion_form_data *const form_data = &state->formData;
    const uint_fast8_t shift = state->shift;
    const uint_fast8_t trailing_zeroes = __builtin_ctz(0x80 | (state->signature >> shift));
    if (density_likely(!trailing_zeroes)) {
        state->shift = (shift + 1) & 0x3f;
        return density_lion_form_model_increment_usage(form_data, (density_lion_form_node *) form_data->formsPool);
    } else if (density_likely(trailing_zeroes <= 6)) {
        state->shift = (shift + (trailing_zeroes + 1)) & 0x3f;
        return density_lion_form_model_increment_usage(form_data, (density_lion_form_node *) form_data->formsPool + trailing_zeroes);
    } else {
        if (density_likely(shift <= (density_bitsizeof(density_lion_signature) - 7))) {
            state->shift = (shift + 7) & 0x3f;
            return density_lion_form_model_increment_usage(form_data, (density_lion_form_node *) form_data->formsPool + 7);
        } else {
            density_lion_decode_read_signature_from_memory(in, state);
            const uint_fast8_t primary_trailing_zeroes = density_bitsizeof(density_lion_signature) - shift;
            const uint_fast8_t ctz_barrier_shift = 7 - primary_trailing_zeroes;
            const uint_fast8_t secondary_trailing_zeroes = __builtin_ctz((1 << ctz_barrier_shift) | state->signature);
            if (density_likely(secondary_trailing_zeroes != ctz_barrier_shift))
                state->shift = secondary_trailing_zeroes + 1;
            else
                state->shift = secondary_trailing_zeroes;
            return density_lion_form_model_increment_usage(form_data, (density_lion_form_node *) form_data->formsPool + primary_trailing_zeroes + secondary_trailing_zeroes);
        }
    }
}

DENSITY_FORCE_INLINE void density_lion_decode_process_form(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state) {
    const uint_fast8_t shift = state->shift;
    density_lion_form_data *const form_data = &state->formData;

    if (density_unlikely(!shift))
        density_lion_decode_read_signature_from_memory(in, state);

    switch ((state->signature >> shift) & 0x1) {
        case 0:
            density_lion_decode_chunk(in, out, state, density_lion_decode_read_form(in, state));
            break;
        default:
            density_lion_decode_chunk(in, out, state, density_lion_form_model_increment_usage(form_data, (density_lion_form_node *) form_data->formsPool));
            state->shift = (shift + 1) & 0x3f;
            break;
    }
}

DENSITY_FORCE_INLINE void density_lion_decode_process_unit(density_memory_location *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state) {
#ifdef __clang__
    for (uint_fast8_t count = 0; count < (DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG >> 2); count++) {
        DENSITY_UNROLL_4(density_lion_decode_process_form(in, out, state));
    }
#else
    for (uint_fast8_t count = 0; count < (DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG >> 2); count++) {
        DENSITY_UNROLL_4(density_lion_decode_process_form(in, out, state));
    }
#endif

    state->chunksCount += DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG;
}

DENSITY_FORCE_INLINE DENSITY_LION_DECODE_STEP_BY_STEP_STATUS density_lion_decode_chunk_step_by_step(density_memory_location *restrict readMemoryLocation, density_memory_teleport *restrict in, density_memory_location *restrict out, density_lion_decode_state *restrict state) {
    density_byte *startPointer = readMemoryLocation->pointer;
    if (density_unlikely(!state->shift))
        density_lion_decode_read_signature_from_memory(readMemoryLocation, state);
    DENSITY_LION_FORM form = density_lion_decode_read_form(readMemoryLocation, state);
    readMemoryLocation->available_bytes -= (readMemoryLocation->pointer - startPointer);
    switch (form) {
        case DENSITY_LION_FORM_PLAIN:  // Potential end marker, we need 2 bytes for a chunk dictionary hash, if remaining bytes < 2 + 2 bytes then this form is the last one
            if (density_unlikely(density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead) <= sizeof(uint32_t)))
                return DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_END_MARKER;
            break;
        default:
            break;
    }
    if (out->available_bytes < sizeof(uint32_t))
        return DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_STALL_ON_OUTPUT;
    startPointer = readMemoryLocation->pointer;
    density_lion_decode_chunk(readMemoryLocation, out, state, form);
    state->chunksCount ++;
    readMemoryLocation->available_bytes -= (readMemoryLocation->pointer - startPointer);
    return DENSITY_LION_DECODE_STEP_BY_STEP_STATUS_PROCEED;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE density_lion_decode_init(density_lion_decode_state *state, const density_main_header_parameters parameters, const uint_fast8_t endDataOverhead) {
    state->chunksCount = 0;
    state->efficiencyChecked = false;
    state->shift = 0;
    density_lion_dictionary_reset(&state->dictionary);

    state->parameters = parameters;
    density_byte resetDictionaryCycleShift = state->parameters.as_bytes[0];
    if (resetDictionaryCycleShift)
        state->resetCycle = (uint_fast64_t) (1 << resetDictionaryCycleShift) - 1;

    state->endDataOverhead = endDataOverhead;

    density_lion_form_data *const form_data = &state->formData;
    density_lion_form_model_init(form_data);
    void (*attachments[DENSITY_LION_NUMBER_OF_FORMS])(void *, void *, void *, void *const, void *const) = {(void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_prediction_a, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_prediction_b, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_prediction_c, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_dictionary_a, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_dictionary_b, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_dictionary_c, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_dictionary_d, (void (*)(void *, void *, void *, void *const, void *const)) &density_lion_decode_plain};
    density_lion_form_model_attach(form_data, attachments);

    state->lastHash = 0;
    state->lastChunk = 0;

    return density_lion_decode_exit_process(state, DENSITY_LION_DECODE_PROCESS_CHECK_BLOCK_STATE, DENSITY_KERNEL_DECODE_STATE_READY);
}

#include "kernel_lion_decode_template.h"

#define DENSITY_LION_DECODE_FINISH

#include "kernel_lion_decode_template.h"
