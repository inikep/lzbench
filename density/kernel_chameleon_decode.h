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
 * 24/10/13 12:27
 *
 * -------------------
 * Chameleon algorithm
 * -------------------
 *
 * Author(s)
 * Guillaume Voirin (https://github.com/gpnuma)
 *
 * Description
 * Hash based superfast kernel
 */

#ifndef DENSITY_CHAMELEON_DECODE_H
#define DENSITY_CHAMELEON_DECODE_H

#include "kernel_chameleon_dictionary.h"
#include "kernel_chameleon.h"
#include "block.h"
#include "kernel_decode.h"
#include "density_api.h"
#include "block_mode_marker.h"
#include "block_header.h"
#include "kernel_chameleon_encode.h"
#include "density_api_data_structures.h"
#include "block_footer.h"
#include "main_footer.h"
#include "main_header.h"

typedef enum {
    DENSITY_CHAMELEON_DECODE_PROCESS_CHECK_SIGNATURE_STATE,
    DENSITY_CHAMELEON_DECODE_PROCESS_READ_PROCESSING_UNIT,
} DENSITY_CHAMELEON_DECODE_PROCESS;

#pragma pack(push)
#pragma pack(4)
typedef struct {
    density_chameleon_signature signature;
    uint_fast8_t shift;
    uint_fast32_t bodyLength;
    uint_fast32_t signaturesCount;
    uint_fast8_t efficiencyChecked;

    DENSITY_CHAMELEON_DECODE_PROCESS process;
    uint_fast8_t endDataOverhead;
    density_main_header_parameters parameters;

    density_chameleon_dictionary dictionary;
    uint_fast64_t resetCycle;
} density_chameleon_decode_state;
#pragma pack(pop)

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_chameleon_decode_init(density_chameleon_decode_state *, const density_main_header_parameters, const uint_fast8_t);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_chameleon_decode_continue(density_memory_teleport *, density_memory_location *, density_chameleon_decode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_DECODE_STATE density_chameleon_decode_finish(density_memory_teleport *, density_memory_location *, density_chameleon_decode_state *);

#endif