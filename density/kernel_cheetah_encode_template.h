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
 * 6/03/15 01:27
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

#undef DENSITY_CHEETAH_ENCODE_FUNCTION_NAME

#ifndef DENSITY_CHEETAH_ENCODE_FINISH
#define DENSITY_CHEETAH_ENCODE_FUNCTION_NAME(name) name ## continue
#else
#define DENSITY_CHEETAH_ENCODE_FUNCTION_NAME(name) name ## finish
#endif

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_ENCODE_STATE DENSITY_CHEETAH_ENCODE_FUNCTION_NAME(density_cheetah_encode_) (density_memory_teleport *restrict in, density_memory_location *restrict out, density_cheetah_encode_state *restrict state) {
    DENSITY_KERNEL_ENCODE_STATE returnState;
    density_byte *pointerOutBefore;
    density_memory_location *readMemoryLocation;

    // Dispatch
    switch (state->process) {
        case DENSITY_CHEETAH_ENCODE_PROCESS_PREPARE_NEW_BLOCK:
            goto prepare_new_block;
        case DENSITY_CHEETAH_ENCODE_PROCESS_CHECK_SIGNATURE_STATE:
            goto check_signature_state;
        case DENSITY_CHEETAH_ENCODE_PROCESS_READ_CHUNK:
            goto read_chunk;
        default:
            return DENSITY_KERNEL_ENCODE_STATE_ERROR;
    }

    // Prepare new block
    prepare_new_block:
    if ((returnState = density_cheetah_encode_prepare_new_block(out, state)))
        return density_cheetah_encode_exit_process(state, DENSITY_CHEETAH_ENCODE_PROCESS_PREPARE_NEW_BLOCK, returnState);

    // Check signature state
    check_signature_state:
    if ((returnState = density_cheetah_encode_check_state(out, state)))
        return density_cheetah_encode_exit_process(state, DENSITY_CHEETAH_ENCODE_PROCESS_CHECK_SIGNATURE_STATE, returnState);

    // Try to read a complete chunk unit
    read_chunk:
    pointerOutBefore = out->pointer;
    if (!(readMemoryLocation = density_memory_teleport_read(in, DENSITY_CHEETAH_ENCODE_PROCESS_UNIT_SIZE)))
#ifndef DENSITY_CHEETAH_ENCODE_FINISH
        return density_cheetah_encode_exit_process(state, DENSITY_CHEETAH_ENCODE_PROCESS_READ_CHUNK, DENSITY_KERNEL_ENCODE_STATE_STALL_ON_INPUT);
#else
        goto step_by_step;
#endif

    // Chunk was read properly, process
    density_cheetah_encode_process_unit(readMemoryLocation, out, state);
    readMemoryLocation->available_bytes -= DENSITY_CHEETAH_ENCODE_PROCESS_UNIT_SIZE;
#ifdef DENSITY_CHEETAH_ENCODE_FINISH
    goto exit;

    // Read step by step
    step_by_step:
    while (state->shift != density_bitsizeof(density_cheetah_signature) && (readMemoryLocation = density_memory_teleport_read(in, sizeof(uint32_t)))) {
        uint32_t chunk;
        DENSITY_MEMCPY(&chunk, readMemoryLocation->pointer, sizeof(uint32_t));
        density_cheetah_encode_kernel(out, DENSITY_CHEETAH_HASH_ALGORITHM(DENSITY_LITTLE_ENDIAN_32(chunk)), chunk, state->shift, state);
        state->shift += 2;
        readMemoryLocation->pointer += sizeof(uint32_t);
        readMemoryLocation->available_bytes -= sizeof(uint32_t);
    }
    exit:
#endif
    out->available_bytes -= (out->pointer - pointerOutBefore);

    // New loop
#ifndef DENSITY_CHEETAH_ENCODE_FINISH
    goto check_signature_state;
#else
    if (density_memory_teleport_available_bytes(in) >= sizeof(uint32_t))
        goto check_signature_state;

    // Marker for decode loop exit
    if ((returnState = density_cheetah_encode_check_state(out, state)))
        return density_cheetah_encode_exit_process(state, DENSITY_CHEETAH_ENCODE_PROCESS_CHECK_SIGNATURE_STATE, returnState);
    state->proximitySignature |= ((uint64_t)DENSITY_CHEETAH_SIGNATURE_FLAG_CHUNK << state->shift);

    // Copy the remaining bytes
    DENSITY_MEMCPY(state->signature, &state->proximitySignature, sizeof(density_cheetah_signature));
    density_memory_teleport_copy_remaining(in, out);

    return DENSITY_KERNEL_ENCODE_STATE_READY;
#endif
}