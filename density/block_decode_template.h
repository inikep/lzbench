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
 * 3/04/15 16:22
 */

#ifndef DENSITY_BLOCK_DECODE_FINISH
DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_continue(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_decode_state *restrict state) {
#else
DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE DENSITY_BLOCK_DECODE_STATE density_block_decode_finish(density_memory_teleport *restrict in, density_memory_location *restrict out, density_block_decode_state *restrict state, void (*mem_free)(void *)) {
#endif
    DENSITY_KERNEL_DECODE_STATE kernelDecodeState;
    DENSITY_BLOCK_DECODE_STATE blockDecodeState;
    uint_fast64_t inAvailableBefore;
    uint_fast64_t outAvailableBefore;
    uint_fast64_t blockRemaining;
    uint_fast64_t inRemaining;
    uint_fast64_t outRemaining;

    // Update integrity pointers
    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK && state->integrityData.update)
        density_block_decode_update_integrity_data(out, state);

    // Dispatch
    switch (state->process) {
        case DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_HEADER:
            goto read_block_header;
        case DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_MODE_MARKER:
            goto read_mode_marker;
        case DENSITY_BLOCK_DECODE_PROCESS_READ_DATA:
            goto read_data;
        case DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_FOOTER:
            goto read_block_footer;
        default:
            return DENSITY_BLOCK_DECODE_STATE_ERROR;
    }

    read_mode_marker:
    if ((blockDecodeState = density_block_decode_read_block_mode_marker(in, state)))
        return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_MODE_MARKER, blockDecodeState);
    goto read_data;

    read_block_header:
    if ((blockDecodeState = density_block_decode_read_block_header(in, out, state)))
        return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_HEADER, blockDecodeState);

    read_data:
    inAvailableBefore = density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead);
    outAvailableBefore = out->available_bytes;

    switch (state->currentMode) {
        case DENSITY_COMPRESSION_MODE_COPY:
            blockRemaining = (uint_fast64_t) DENSITY_PREFERRED_COPY_BLOCK_SIZE - (state->totalWritten - state->currentBlockData.outStart);
            inRemaining = density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead);
            outRemaining = out->available_bytes;

            if (inRemaining <= outRemaining) {
                if (blockRemaining <= inRemaining)
                    goto copy_until_end_of_block;
                else {
                    density_memory_teleport_copy(in, out, inRemaining);
                    density_block_decode_update_totals(in, out, state, inAvailableBefore, outAvailableBefore);
#ifndef DENSITY_BLOCK_DECODE_FINISH
                    density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(in);
                    return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_DATA, DENSITY_BLOCK_DECODE_STATE_STALL_ON_INPUT);
#else
                    goto read_block_footer;
#endif
                }
            } else {
                if (blockRemaining <= outRemaining)
                    goto copy_until_end_of_block;
                else {
                    density_memory_teleport_copy(in, out, outRemaining);
                    density_block_decode_update_totals(in, out, state, inAvailableBefore, outAvailableBefore);
                    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
                        density_block_decode_update_integrity_hash(out, state, true);
                    return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_DATA, DENSITY_BLOCK_DECODE_STATE_STALL_ON_OUTPUT);
                }
            }

        copy_until_end_of_block:
            density_memory_teleport_copy(in, out, blockRemaining);
            density_block_decode_update_totals(in, out, state, inAvailableBefore, outAvailableBefore);
            goto read_block_footer;

        default:
#ifndef DENSITY_BLOCK_DECODE_FINISH
            kernelDecodeState = state->kernelDecodeProcess(in, out, state->kernelDecodeState);
#else
            kernelDecodeState = state->kernelDecodeFinish(in, out, state->kernelDecodeState);
#endif
            density_block_decode_update_totals(in, out, state, inAvailableBefore, outAvailableBefore);

            switch (kernelDecodeState) {
                case DENSITY_KERNEL_DECODE_STATE_STALL_ON_INPUT:
#ifndef DENSITY_BLOCK_DECODE_FINISH
                    return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_DATA, DENSITY_BLOCK_DECODE_STATE_STALL_ON_INPUT);
#else
                    return DENSITY_BLOCK_DECODE_STATE_ERROR;
#endif
                case DENSITY_KERNEL_DECODE_STATE_STALL_ON_OUTPUT:
                    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
                        density_block_decode_update_integrity_hash(out, state, true);
                    return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_DATA, DENSITY_BLOCK_DECODE_STATE_STALL_ON_OUTPUT);
#ifdef DENSITY_BLOCK_DECODE_FINISH
                case DENSITY_KERNEL_DECODE_STATE_READY:
#endif
                case DENSITY_KERNEL_DECODE_STATE_INFO_NEW_BLOCK:
                    goto read_block_footer;
                case DENSITY_KERNEL_DECODE_STATE_INFO_EFFICIENCY_CHECK:
                    goto read_mode_marker;
                default:
                    return DENSITY_BLOCK_DECODE_STATE_ERROR;
            }
    }

    read_block_footer:
    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK) if ((blockDecodeState = density_block_decode_read_block_footer(in, out, state)))
        return exitProcess(state, DENSITY_BLOCK_DECODE_PROCESS_READ_BLOCK_FOOTER, blockDecodeState);
#ifndef DENSITY_BLOCK_DECODE_FINISH
    goto read_block_header;
#else
    if (density_memory_teleport_available_bytes_reserved(in, state->endDataOverhead))
        goto read_block_header;

    if (state->blockType == DENSITY_BLOCK_TYPE_WITH_HASHSUM_INTEGRITY_CHECK)
        spookyhash_context_free(state->integrityData.context, mem_free);

    return DENSITY_BLOCK_DECODE_STATE_READY;
#endif
}