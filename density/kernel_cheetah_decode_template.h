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
 * 8/03/15 02:44
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

#undef DENSITY_CHEETAH_DECODE_FUNCTION_NAME

#ifndef DENSITY_CHEETAH_DECODE_FINISH
#define DENSITY_CHEETAH_DECODE_FUNCTION_NAME(name) name ## continue
#else
#define DENSITY_CHEETAH_DECODE_FUNCTION_NAME(name) name ## finish
#endif

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_KERNEL_DECODE_STATE DENSITY_CHEETAH_DECODE_FUNCTION_NAME(density_cheetah_decode_)(density_memory_teleport *restrict in, density_memory_location *restrict out, density_cheetah_decode_state *restrict state) {
    DENSITY_KERNEL_DECODE_STATE returnState;
    density_memory_location *readMemoryLocation;
    uint_fast64_t availableBytesReserved;

    // Dispatch
    switch (state->process) {
        case DENSITY_CHEETAH_DECODE_PROCESS_CHECK_SIGNATURE_STATE:
            goto check_signature_state;
        case DENSITY_CHEETAH_DECODE_PROCESS_READ_PROCESSING_UNIT:
            goto read_processing_unit;
        default:
            return DENSITY_KERNEL_DECODE_STATE_ERROR;
    }

    check_signature_state:
    if (density_unlikely((returnState = density_cheetah_decode_check_state(out, state)))) {
#ifdef DENSITY_CHEETAH_DECODE_FINISH
        if(returnState == DENSITY_KERNEL_DECODE_STATE_STALL_ON_OUTPUT)
            if(density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead) < DENSITY_CHEETAH_MAXIMUM_COMPRESSED_UNIT_SIZE)
                goto step_by_step;
#endif
        return density_cheetah_decode_exit_process(state, DENSITY_CHEETAH_DECODE_PROCESS_CHECK_SIGNATURE_STATE, returnState);
    }

    // Try to read the next processing unit
    read_processing_unit:
    if (density_unlikely(!(readMemoryLocation = density_memory_teleport_read_reserved(in, DENSITY_CHEETAH_MAXIMUM_COMPRESSED_UNIT_SIZE, state->endDataOverhead))))
#ifndef DENSITY_CHEETAH_DECODE_FINISH
        return density_cheetah_decode_exit_process(state, DENSITY_CHEETAH_DECODE_PROCESS_READ_PROCESSING_UNIT, DENSITY_KERNEL_DECODE_STATE_STALL_ON_INPUT);
#else
        goto step_by_step;
#endif

    density_byte *readMemoryLocationPointerBefore = readMemoryLocation->pointer;

    // Decode the signature (endian processing)
    density_cheetah_decode_read_signature(readMemoryLocation, state);

    // Process body
    density_cheetah_decode_process_data(readMemoryLocation, out, state);

    readMemoryLocation->available_bytes -= (readMemoryLocation->pointer - readMemoryLocationPointerBefore);
    out->available_bytes -= DENSITY_CHEETAH_DECOMPRESSED_UNIT_SIZE;

    // New loop
    goto check_signature_state;

    // Try to read and process signature and body, step by step
    step_by_step:
    if (!(readMemoryLocation = density_memory_teleport_read_reserved(in, sizeof(density_cheetah_signature), state->endDataOverhead)))
        goto finish;
    density_cheetah_decode_read_signature(readMemoryLocation, state);
    readMemoryLocation->available_bytes -= sizeof(density_cheetah_signature);
    uint16_t hash;
    uint32_t chunk;

    while (state->shift != density_bitsizeof(density_cheetah_signature)) {
        switch ((uint8_t const) ((state->signature >> state->shift) & 0x3)) {
            case DENSITY_CHEETAH_SIGNATURE_FLAG_PREDICTED:
                if (out->available_bytes < sizeof(uint32_t))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                density_cheetah_decode_process_predicted(out, state);
                break;
            case DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_A:
                if (!(readMemoryLocation = density_memory_teleport_read_reserved(in, sizeof(uint16_t), state->endDataOverhead)))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                if (out->available_bytes < sizeof(uint32_t))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                DENSITY_MEMCPY(&hash, readMemoryLocation->pointer, sizeof(uint16_t));
                density_cheetah_decode_process_compressed_a(hash, out, state);
                readMemoryLocation->pointer += sizeof(uint16_t);
                readMemoryLocation->available_bytes -= sizeof(uint16_t);
                break;
            case DENSITY_CHEETAH_SIGNATURE_FLAG_MAP_B:
                if (!(readMemoryLocation = density_memory_teleport_read_reserved(in, sizeof(uint16_t), state->endDataOverhead)))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                if (out->available_bytes < sizeof(uint32_t))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                DENSITY_MEMCPY(&hash, readMemoryLocation->pointer, sizeof(uint16_t));
                density_cheetah_decode_process_compressed_b(hash, out, state);
                readMemoryLocation->pointer += sizeof(uint16_t);
                readMemoryLocation->available_bytes -= sizeof(uint16_t);
                break;
            case DENSITY_CHEETAH_SIGNATURE_FLAG_CHUNK:
                if (!(readMemoryLocation = density_memory_teleport_read_reserved(in, sizeof(uint32_t), state->endDataOverhead)))
                    goto finish;
                if (out->available_bytes < sizeof(uint32_t))
                    return DENSITY_KERNEL_DECODE_STATE_ERROR;
                DENSITY_MEMCPY(&chunk, readMemoryLocation->pointer, sizeof(uint32_t));
                density_cheetah_decode_process_uncompressed(chunk, out, state);
                readMemoryLocation->pointer += sizeof(uint32_t);
                readMemoryLocation->available_bytes -= sizeof(uint32_t);
                break;
        }
        out->pointer += sizeof(uint32_t);
        out->available_bytes -= sizeof(uint32_t);
        state->shift += 2;
    }

    // New loop
    goto check_signature_state;

    finish:
    availableBytesReserved = density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead);
    if (out->available_bytes < availableBytesReserved)
        return DENSITY_KERNEL_DECODE_STATE_ERROR;
    density_memory_teleport_copy(in, out, availableBytesReserved);

    return DENSITY_KERNEL_DECODE_STATE_READY;
}
