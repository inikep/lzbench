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

#ifndef DENSITY_CHEETAH_ENCODE_H
#define DENSITY_CHEETAH_ENCODE_H

#include "kernel_cheetah_dictionary.h"
#include "kernel_cheetah.h"
#include "block.h"
#include "kernel_encode.h"
#include "density_api.h"
#include "memory_location.h"
#include "memory_teleport.h"

#define DENSITY_CHEETAH_ENCODE_PROCESS_UNIT_SIZE                    ((density_bitsizeof(density_cheetah_signature) >> 1) * sizeof(uint32_t))

typedef enum {
    DENSITY_CHEETAH_ENCODE_PROCESS_PREPARE_NEW_BLOCK,
    DENSITY_CHEETAH_ENCODE_PROCESS_CHECK_SIGNATURE_STATE,
    DENSITY_CHEETAH_ENCODE_PROCESS_READ_CHUNK,
} DENSITY_CHEETAH_ENCODE_PROCESS;

#pragma pack(push)
#pragma pack(4)
typedef struct {
    density_cheetah_signature proximitySignature;
    uint_fast16_t lastHash;
    uint_fast8_t shift;
    density_cheetah_signature * signature;
    uint_fast32_t signaturesCount;
    uint_fast8_t efficiencyChecked;

    DENSITY_CHEETAH_ENCODE_PROCESS process;

    density_cheetah_dictionary dictionary;
#if DENSITY_ENABLE_PARALLELIZABLE_DECOMPRESSIBLE_OUTPUT == DENSITY_YES
    uint_fast64_t resetCycle;
#endif
} density_cheetah_encode_state;
#pragma pack(pop)

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_init(density_cheetah_encode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_continue(density_memory_teleport *, density_memory_location *, density_cheetah_encode_state *);

DENSITY_WINDOWS_EXPORT DENSITY_KERNEL_ENCODE_STATE density_cheetah_encode_finish(density_memory_teleport *, density_memory_location *, density_cheetah_encode_state *);
#endif