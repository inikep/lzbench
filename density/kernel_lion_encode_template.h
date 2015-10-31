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
 * 6/03/15 01:08
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

#undef DENSITY_LION_ENCODE_FUNCTION_NAME

#ifndef DENSITY_LION_ENCODE_FINISH
#define DENSITY_LION_ENCODE_FUNCTION_NAME(name) name ## continue
#else
#define DENSITY_LION_ENCODE_FUNCTION_NAME(name) name ## finish
#endif

#ifndef DENSITY_LION_ENCODE_MANAGE_INTERCEPT
#define DENSITY_LION_ENCODE_MANAGE_INTERCEPT\
    if(start_shift > state->shift) {\
        if(density_likely(state->shift)) {\
            const density_byte *content_start = (density_byte *) state->signature + sizeof(density_lion_signature);\
            state->transientContent.size = (uint8_t) (out->pointer - content_start);\
            DENSITY_MEMCPY(state->transientContent.content, content_start, state->transientContent.size);\
            out->pointer = (density_byte *) state->signature;\
            out->available_bytes -= (out->pointer - pointerOutBefore);\
            return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_CHECK_OUTPUT_SIZE, DENSITY_KERNEL_ENCODE_STATE_STALL_ON_OUTPUT);\
        } else {\
            out->available_bytes -= (out->pointer - pointerOutBefore);\
            return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE, DENSITY_KERNEL_ENCODE_STATE_STALL_ON_OUTPUT);\
        }\
    }
#endif

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE DENSITY_LION_ENCODE_FUNCTION_NAME(density_lion_encode_)(density_memory_teleport *restrict in, density_memory_location *restrict out, density_lion_encode_state *restrict state) {
    DENSITY_KERNEL_ENCODE_STATE returnState;
    density_byte *pointerOutBefore;
    density_memory_location *readMemoryLocation;

    // Dispatch
    switch (state->process) {
        case DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE:
            goto check_block_state;
        case DENSITY_LION_ENCODE_PROCESS_CHECK_OUTPUT_SIZE:
            goto check_output_size;
        case DENSITY_LION_ENCODE_PROCESS_UNIT:
            goto process_unit;
        default:
            return DENSITY_KERNEL_ENCODE_STATE_ERROR;
    }

    // Check block metadata
    check_block_state:
    if (density_unlikely(!state->shift)) {
        if(density_unlikely(out->available_bytes < DENSITY_LION_ENCODE_MINIMUM_LOOKAHEAD))
            return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE, DENSITY_KERNEL_ENCODE_STATE_STALL_ON_OUTPUT);  // Direct exit possible, if coming from copy mode
        if (density_unlikely(returnState = density_lion_encode_check_block_state(state)))
            return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE, returnState);
    }

    // Check output size
    check_output_size:
    if(density_unlikely(state->signatureInterceptMode)) {
        if (out->available_bytes >= DENSITY_LION_ENCODE_MINIMUM_LOOKAHEAD) {     // New buffer
            if(density_likely(state->shift)) {
                state->signature = (density_lion_signature *) (out->pointer);
                out->pointer += sizeof(density_lion_signature);
                DENSITY_MEMCPY(out->pointer, state->transientContent.content, state->transientContent.size);
                out->pointer += state->transientContent.size;
                out->available_bytes -= (sizeof(density_lion_signature) + state->transientContent.size);
            }
            state->signatureInterceptMode = false;
        }
    } else {
        if (density_unlikely(out->available_bytes < DENSITY_LION_ENCODE_MINIMUM_LOOKAHEAD))
            state->signatureInterceptMode = true;
    }

    // Try to read a complete process unit
    process_unit:
    pointerOutBefore = out->pointer;
    if (!(readMemoryLocation = density_memory_teleport_read(in, DENSITY_LION_PROCESS_UNIT_SIZE_BIG)))
#ifndef DENSITY_LION_ENCODE_FINISH
        return density_lion_encode_exit_process(state, DENSITY_LION_ENCODE_PROCESS_UNIT, DENSITY_KERNEL_ENCODE_STATE_STALL_ON_INPUT);
#else
        goto step_by_step;
#endif

    // Chunk was read properly, process
    if(density_unlikely(state->signatureInterceptMode)) {
        const uint_fast32_t start_shift = state->shift;

        density_lion_encode_process_unit_small(readMemoryLocation, out, state);

        DENSITY_LION_ENCODE_MANAGE_INTERCEPT;
    } else {
        if(density_unlikely(state->chunksCount & (DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG - 1)))
            density_lion_encode_process_unit_small(readMemoryLocation, out, state);  // Attempt to resync the chunks count with a multiple of DENSITY_LION_CHUNKS_PER_PROCESS_UNIT_BIG
        else
            density_lion_encode_process_unit_big(readMemoryLocation, out, state);
    }

#ifdef DENSITY_LION_ENCODE_FINISH
    goto exit;

    // Read step by step
    step_by_step:
    while ((readMemoryLocation = density_memory_teleport_read(in, sizeof(uint32_t)))) {
        if(density_unlikely(state->signatureInterceptMode)) {
            const uint_fast32_t start_shift = state->shift;

            density_lion_encode_process_step_unit(readMemoryLocation, out, state);

            DENSITY_LION_ENCODE_MANAGE_INTERCEPT;
        } else
            density_lion_encode_process_step_unit(readMemoryLocation, out, state);

        if(!state->shift)
            goto exit;
    }
    exit:
#endif
    out->available_bytes -= (out->pointer - pointerOutBefore);

    // New loop
#ifndef DENSITY_LION_ENCODE_FINISH
    goto check_block_state;
#else
    if (density_memory_teleport_available_bytes(in) > sizeof(uint32_t))
        goto check_block_state;

    // Marker for decode loop exit
    if(!state->endMarker) {
        state->endMarker = true;
        goto check_block_state;
    }

    pointerOutBefore = out->pointer;
    const density_lion_entropy_code code = density_lion_form_model_get_encoding(&state->formData, DENSITY_LION_FORM_PLAIN);
    if(density_unlikely(state->signatureInterceptMode)) {
        const uint_fast32_t start_shift = state->shift;

        density_lion_encode_push_to_signature(out, state, code.value, code.bitLength);

        DENSITY_LION_ENCODE_MANAGE_INTERCEPT;
    } else
        density_lion_encode_push_to_signature(out, state, code.value, code.bitLength);

    // Copy the remaining bytes
    DENSITY_MEMCPY(state->signature, &state->proximitySignature, sizeof(density_lion_signature));
    out->available_bytes -= (out->pointer - pointerOutBefore);
    density_memory_teleport_copy_remaining(in, out);

    return DENSITY_KERNEL_ENCODE_STATE_READY;
#endif
}