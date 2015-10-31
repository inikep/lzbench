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
 * 06/12/13 20:20
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

#ifndef DENSITY_LION_ENCODE_H
#define DENSITY_LION_ENCODE_H

#include "kernel_lion_dictionary.h"
#include "kernel_lion.h"
#include "kernel_lion_form_model.h"
#include "block.h"
#include "kernel_encode.h"
#include "density_api.h"
#include "memory_location.h"
#include "memory_teleport.h"
#include "kernel_lion_decode.h"
#include "block_footer.h"
#include "block_header.h"
#include "block_mode_marker.h"

typedef enum {
    DENSITY_LION_ENCODE_PROCESS_CHECK_BLOCK_STATE,
    DENSITY_LION_ENCODE_PROCESS_CHECK_OUTPUT_SIZE,
    DENSITY_LION_ENCODE_PROCESS_UNIT,
} DENSITY_LION_ENCODE_PROCESS;

typedef struct {
    density_byte content[DENSITY_LION_MAXIMUM_COMPRESSED_BODY_SIZE_PER_SIGNATURE];
    uint_fast8_t size;
} density_lion_encode_content;

#define DENSITY_LION_ENCODE_MINIMUM_LOOKAHEAD   (sizeof(density_block_footer) + sizeof(density_block_header) + sizeof(density_mode_marker) + (DENSITY_LION_MAXIMUM_COMPRESSED_UNIT_SIZE << 1))  // On a normal cycle, DENSITY_LION_CHUNKS_PER_PROCESS_UNIT = 64 chunks = 256 bytes can be compressed at once, before being in intercept mode where another 256 input bytes could be processed before ending the signature

#pragma pack(push)
#pragma pack(4)
typedef struct {
    density_lion_signature proximitySignature;
    density_lion_form_data formData;
    uint_fast16_t lastHash;
    uint32_t lastChunk;
    uint_fast8_t shift;
    density_lion_signature * signature;
    uint_fast64_t chunksCount;
    bool efficiencyChecked;

    DENSITY_LION_ENCODE_PROCESS process;

    density_lion_encode_content transientContent;
    bool signatureInterceptMode;
    bool endMarker;

    density_lion_dictionary dictionary;
#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
    uint_fast64_t resetCycle;
#endif
} density_lion_encode_state;
#pragma pack(pop)

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_lion_encode_init(density_lion_encode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_lion_encode_continue(density_memory_teleport *, density_memory_location *, density_lion_encode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_lion_encode_finish(density_memory_teleport *, density_memory_location *, density_lion_encode_state *);
#endif