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
 * 3/04/15 16:54
 */

#ifndef DENSITY_BLOCK_ENCODE_FINISH
DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_BLOCK_ENCODE_STATE density_block_encode_continue(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_encode_state *restrict state) {
#else
DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_BLOCK_ENCODE_STATE density_block_encode_finish(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_encode_state *restrict state, void (*mem_free)(void *)) {
#endif
    DENSITY_BLOCK_ENCODE_STATE blockEncodeState;
    DENSITY_KERNEL_ENCODE_STATE kernelEncodeState;
    uint_fast64_t availableInBefore;
    uint_fast64_t availableOutBefore;
    uint_fast64_t blockRemaining;
    uint_fast64_t inRemaining;
    uint_fast64_t outRemaining;

    // Add to the integrity check hashsum
    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK && state->integrityData.update)
        density_block_encode_update_integrity_data(in, state);

    // Dispatch
    switch (state->process) {
        case DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_HEADER:
            goto write_block_header;
        case DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_MODE_MARKER:
            goto write_mode_marker;
        case DENSITY_BLOCK_ENCODE_PROCESS_WRITE_DATA:
            goto write_data;
        case DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_FOOTER:
            goto write_block_footer;
        default:
            return DENSITY_BLOCK_ENCODE_STATE_ERROR;
    }

    write_mode_marker:
    if ((blockEncodeState = density_block_encode_write_mode_marker(out, state)))
        return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_MODE_MARKER, blockEncodeState);
    goto write_data;

    write_block_header:
    if ((blockEncodeState = density_block_encode_write_block_header(in, out, state)))
        return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_HEADER, blockEncodeState);

    write_data:
    availableInBefore = density_memory_teleport_available_bytes(in);
    availableOutBefore = out->available_bytes;

    switch (state->currentMode) {
        case DENSITY_COMPRESSION_MODE_COPY:
            blockRemaining = (uint_fast64_t) DENSITY_PREFERRED_COPY_BLOCK_SIZE - (state->totalRead - state->currentBlockData.inStart);
            inRemaining = density_memory_teleport_available_bytes(in);
            outRemaining = out->available_bytes;

            if (inRemaining <= outRemaining) {
                if (blockRemaining <= inRemaining)
                    goto copy_until_end_of_block;
                else {
                    density_memory_teleport_copy(in, out, inRemaining);
                    density_block_encode_update_totals(in, out, state, availableInBefore, availableOutBefore);
#ifndef DENSITY_BLOCK_ENCODE_FINISH
                    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
                        density_block_encode_update_integrity_hash(in, state, true);
                    density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(in);
                    return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_DATA, DENSITY_BLOCK_ENCODE_STATE_STALL_ON_INPUT);
#else
                    goto write_block_footer;
#endif
                }
            } else {
                if (blockRemaining <= outRemaining)
                    goto copy_until_end_of_block;
                else {
                    density_memory_teleport_copy(in, out, outRemaining);
                    density_block_encode_update_totals(in, out, state, availableInBefore, availableOutBefore);
                    return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_DATA, DENSITY_BLOCK_ENCODE_STATE_STALL_ON_OUTPUT);
                }
            }

        copy_until_end_of_block:
            density_memory_teleport_copy(in, out, blockRemaining);
            density_block_encode_update_totals(in, out, state, availableInBefore, availableOutBefore);
            goto write_block_footer;

        default:
#ifndef DENSITY_BLOCK_ENCODE_FINISH
            kernelEncodeState = state->kernelEncodeProcess(in, out, state->kernelEncodeState);
#else
            kernelEncodeState = state->kernelEncodeFinish(in, out, state->kernelEncodeState);
#endif
            density_block_encode_update_totals(in, out, state, availableInBefore, availableOutBefore);

            switch (kernelEncodeState) {
                case DENSITY_KERNEL_ENCODE_STATE_STALL_ON_INPUT:
#ifndef DENSITY_BLOCK_ENCODE_FINISH
                    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
                        density_block_encode_update_integrity_hash(in, state, true);
                    return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_DATA, DENSITY_BLOCK_ENCODE_STATE_STALL_ON_INPUT);
#else
                    return DENSITY_BLOCK_ENCODE_STATE_ERROR;
#endif
                case DENSITY_KERNEL_ENCODE_STATE_STALL_ON_OUTPUT:
                    return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_DATA, DENSITY_BLOCK_ENCODE_STATE_STALL_ON_OUTPUT);
#ifdef DENSITY_BLOCK_ENCODE_FINISH
                case DENSITY_KERNEL_ENCODE_STATE_READY:
#endif
                case DENSITY_KERNEL_ENCODE_STATE_INFO_NEW_BLOCK:
                    goto write_block_footer;
                case DENSITY_KERNEL_ENCODE_STATE_INFO_EFFICIENCY_CHECK:
                    goto write_mode_marker;
                default:
                    return DENSITY_BLOCK_ENCODE_STATE_ERROR;
            }
    }

    write_block_footer:
    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK) if ((blockEncodeState = density_block_encode_write_block_footer(in, out, state)))
        return density_block_encode_exit_process(state, DENSITY_BLOCK_ENCODE_PROCESS_WRITE_BLOCK_FOOTER, blockEncodeState);
#ifndef DENSITY_BLOCK_ENCODE_FINISH
    goto write_block_header;
#else
    if (density_memory_teleport_available_bytes(in))
        goto write_block_header;

    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
        spookyhash_context_free(state->integrityData.context, mem_free);

    return DENSITY_BLOCK_ENCODE_STATE_READY;
#endif
}