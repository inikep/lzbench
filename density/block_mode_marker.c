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
 * 19/10/13 00:01
 */

#include "block_mode_marker.h"
#include "density_api_data_structures.h"

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE uint_fast32_t density_block_mode_marker_read(density_memory_location *restrict in, density_mode_marker *restrict modeMarker) {
    modeMarker->activeBlockMode = *in->pointer;

    in->pointer += sizeof(density_mode_marker);
    in->available_bytes -= sizeof(density_mode_marker);

    return sizeof(density_mode_marker);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE uint_fast32_t density_block_mode_marker_write(density_memory_location *out, const DENSITY_COMPRESSION_MODE mode) {
    *(out->pointer) = (density_byte) mode;
    *(out->pointer + 1) = 0x0;

    out->pointer += sizeof(density_mode_marker);
    out->available_bytes -= sizeof(density_mode_marker);

    return sizeof(density_mode_marker);
}