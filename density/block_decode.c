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
 * 19/10/13 00:20
 */

#include "block_decode.h"
#include "main_header.h"

DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE exitProcess(density_block_decode_state *state, DENSITY_BLOCK_DECODE_PROCESS process, DENSITY_BLOCK_DECODE_STATE blockDecodeState) {
    state->process = process;
    return blockDecodeState;
}

DENSITY_FORCE_INLINE void density_block_decode_update_integrity_data(density_memory_location *restrict out, density_block_decode_state *restrict state) {
    state->integrityData.available = out->available_bytes;
    state->integrityData.outputPointer = out->pointer;

    state->integrityData.update = false;
}

DENSITY_FORCE_INLINE void density_block_decode_update_integrity_hash(density_memory_location *restrict out, density_block_decode_state *restrict state, bool pendingExit) {
    uint_fast64_t availableBefore = state->integrityData.available;
    uint_fast64_t available = out->available_bytes;
    uint_fast64_t used = availableBefore - available;

    spookyhash_update(state->integrityData.context, state->integrityData.outputPointer, used);

    if (pendingExit)
        state->integrityData.update = true;
    else
        density_block_decode_update_integrity_data(out, state);
}

DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_read_block_header(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_decode_state *restrict state) {
    density_memory_location *readLocation;
    if (!(readLocation = density_memory_teleport_read_reserved(in, sizeof(density_block_header), state->endDataOverhead)))
        return DENSITY_BLOCK_DECODE_STATE_STALL_ON_INPUT;

    state->currentBlockData.inStart = state->totalRead;
    state->currentBlockData.outStart = state->totalWritten;

    if (state->readBlockHeaderContent)
        state->totalRead += density_block_header_read(readLocation, &state->lastBlockHeader);

    state->currentMode = state->targetMode;

    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK) {
        spookyhash_context_init(state->integrityData.context, DENSITY_SPOOKYHASH_SEED_1, DENSITY_SPOOKYHASH_SEED_2);
        density_block_decode_update_integrity_data(out, state);
    }

    return DENSITY_BLOCK_DECODE_STATE_READY;
}

DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_read_block_footer(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_decode_state *restrict state) {
    density_memory_location *readLocation;
    if (!(readLocation = density_memory_teleport_read(in, sizeof(density_block_footer))))
        return DENSITY_BLOCK_DECODE_STATE_STALL_ON_INPUT;

    density_block_decode_update_integrity_hash(out, state, false);

    uint64_t hashsum1, hashsum2;
    spookyhash_final(state->integrityData.context, &hashsum1, &hashsum2);

    state->totalRead += density_block_footer_read(readLocation, &state->lastBlockFooter);

    if (state->lastBlockFooter.hashsum1 != hashsum1 || state->lastBlockFooter.hashsum2 != hashsum2)
        return DENSITY_BLOCK_DECODE_STATE_INTEGRITY_CHECK_FAIL;

    return DENSITY_BLOCK_DECODE_STATE_READY;
}

DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_read_block_mode_marker(density_memory_teleport *restrict in, density_block_decode_state *restrict state) {
    density_memory_location *readLocation;
    if (!(readLocation = density_memory_teleport_read_reserved(in, sizeof(density_mode_marker), state->endDataOverhead)))
        return DENSITY_BLOCK_DECODE_STATE_STALL_ON_INPUT;

    state->totalRead += density_block_mode_marker_read(readLocation, &state->lastModeMarker);

    state->currentMode = (DENSITY_COMPRESSION_MODE) state->lastModeMarker.activeBlockMode;

    return DENSITY_BLOCK_DECODE_STATE_READY;
}

DENSITY_FORCE_INLINE void density_block_decode_update_totals(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_decode_state *restrict state, const uint_fast64_t inAvailableBefore, const uint_fast64_t outAvailableBefore) {
    state->totalRead += inAvailableBefore - density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead);
    state->totalWritten += outAvailableBefore - out->available_bytes;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_init(density_block_decode_state *restrict state, const DENSITY_COMPRESSION_MODE mode, const DENSITY_BLOCK_TYPE blockType, const density_main_header_parameters parameters, const uint_fast8_t endDataOverhead, void *kernelState, DENSITY_KERNEL_DECODE_STATE (*kernelInit)(void *, const density_main_header_parameters, const uint_fast8_t), DENSITY_KERNEL_DECODE_STATE (*kernelProcess)(density_memory_teleport *, density_memory_location *, void *), DENSITY_KERNEL_DECODE_STATE (*kernelFinish)(density_memory_teleport *, density_memory_location *, void *), void *(*mem_alloc)(size_t)) {
    state->targetMode = mode;
    state->currentMode = mode;
    state->blockType = blockType;
    state->readBlockHeaderContent = (parameters.as_bytes[0] ? true : false);

    state->totalRead = 0;
    state->totalWritten = 0;
    state->endDataOverhead = endDataOverhead;

    // Add to the integrity check hashsum
    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK) {
        state->integrityData.context = spookyhash_context_allocate(mem_alloc);
        state->integrityData.update = true;
        state->endDataOverhead += sizeof(density_block_footer);
    }

    switch (state->currentMode) {
        case DENSITY_COMPRESSION_MODE_COPY:
            break;

        default:
            state->kernelDecodeState = kernelState;
            state->kernelDecodeInit = kernelInit;
            state->kernelDecodeProcess = kernelProcess;
            state->kernelDecodeFinish = kernelFinish;

            state->kernelDecodeInit(state->kernelDecodeState, parameters, state->endDataOverhead);
            break;
    }

    return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_HEADER, DENSITY_BLOCK_DECODE_STATE_READY);
}

#include "block_decode_template.h"

#define DENSITY_BLOCK_DECODE_FINISH

#include "block_decode_template.h"