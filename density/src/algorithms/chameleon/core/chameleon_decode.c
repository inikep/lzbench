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
 * 23/06/15 22:11
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

#include "chameleon_decode.h"

DENSITY_FORCE_INLINE void density_chameleon_decode_process_compressed(const uint16_t hash, uint8_t **DENSITY_RESTRICT out, density_chameleon_dictionary *const DENSITY_RESTRICT dictionary) {
    DENSITY_MEMCPY(*out, &dictionary->entries[hash].as_uint32_t, sizeof(uint32_t));
}

DENSITY_FORCE_INLINE void density_chameleon_decode_process_uncompressed(const uint32_t chunk, uint8_t **DENSITY_RESTRICT out, density_chameleon_dictionary *const DENSITY_RESTRICT dictionary) {
    const uint16_t hash = DENSITY_CHAMELEON_HASH_ALGORITHM(chunk);
    (&dictionary->entries[hash])->as_uint32_t = chunk;
    DENSITY_MEMCPY(*out, &chunk, sizeof(uint32_t));
}

DENSITY_FORCE_INLINE void density_chameleon_decode_kernel(const uint8_t **DENSITY_RESTRICT in, uint8_t **DENSITY_RESTRICT out, const density_bool compressed, density_chameleon_dictionary *const DENSITY_RESTRICT dictionary) {
    if (compressed) {
        uint16_t hash;
        DENSITY_MEMCPY(&hash, *in, sizeof(uint16_t));
        density_chameleon_decode_process_compressed(hash, out, dictionary);
        *in += sizeof(uint16_t);
    } else {
        uint32_t unit;
        DENSITY_MEMCPY(&unit, *in, sizeof(uint32_t));
        density_chameleon_decode_process_uncompressed(unit, out, dictionary);
        *in += sizeof(uint32_t);
    }
    *out += sizeof(uint32_t);
}

DENSITY_FORCE_INLINE bool density_chameleon_decode_test_compressed(const uint_fast64_t signature, const uint_fast8_t shift) {
    return (density_bool const) ((signature >> shift) & DENSITY_CHAMELEON_SIGNATURE_FLAG_MAP);
}

DENSITY_FORCE_INLINE void density_chameleon_decode_4(const uint8_t **DENSITY_RESTRICT in, uint8_t **DENSITY_RESTRICT out, const uint_fast64_t signature, const uint_fast8_t shift, density_chameleon_dictionary *const DENSITY_RESTRICT dictionary) {
    density_chameleon_decode_kernel(in, out, density_chameleon_decode_test_compressed(signature, shift), dictionary);
}

DENSITY_FORCE_INLINE void density_chameleon_decode_256(const uint8_t **DENSITY_RESTRICT in, uint8_t **DENSITY_RESTRICT out, const uint_fast64_t signature, density_chameleon_dictionary *const DENSITY_RESTRICT dictionary) {
    uint_fast8_t count = 0;

#ifdef __clang__
    for(uint_fast8_t count_b = 0; count_b < 8; count_b ++) {
        DENSITY_UNROLL_8(density_chameleon_decode_kernel(in, out, density_chameleon_decode_test_compressed(signature, count++), dictionary));
    }
#else
    for (uint_fast8_t count_b = 0; count_b < 16; count_b++) {
        DENSITY_UNROLL_4(density_chameleon_decode_kernel(in, out, density_chameleon_decode_test_compressed(signature, count++), dictionary));
    }
#endif
}

DENSITY_FORCE_INLINE void density_chameleon_decode_read_signature(const uint8_t **DENSITY_RESTRICT in, density_chameleon_signature *DENSITY_RESTRICT signature) {
    DENSITY_MEMCPY(signature, *in, sizeof(density_chameleon_signature));
    *in += sizeof(density_chameleon_signature);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_algorithm_exit_status density_chameleon_decode(density_algorithm_state *const DENSITY_RESTRICT state, const uint8_t **DENSITY_RESTRICT in, const uint_fast64_t in_size, uint8_t **DENSITY_RESTRICT out, const uint_fast64_t out_size) {
    if (out_size < DENSITY_CHAMELEON_DECOMPRESSED_UNIT_SIZE)
        return DENSITY_ALGORITHMS_EXIT_STATUS_OUTPUT_STALL;

    density_chameleon_signature signature;
    uint_fast8_t shift;
    uint_fast64_t remaining;

    const uint8_t *start = *in;

    if (in_size < DENSITY_CHAMELEON_MAXIMUM_COMPRESSED_UNIT_SIZE) {
        goto read_signature;
    }

    const uint8_t *in_limit = *in + in_size - DENSITY_CHAMELEON_MAXIMUM_COMPRESSED_UNIT_SIZE;
    uint8_t *out_limit = *out + out_size - DENSITY_CHAMELEON_DECOMPRESSED_UNIT_SIZE;

    while (DENSITY_LIKELY(*in <= in_limit && *out <= out_limit)) {
        if (DENSITY_UNLIKELY(!(state->counter & 0xf))) {
            DENSITY_ALGORITHM_REDUCE_COPY_PENALTY_START;
        }
        state->counter++;
        if (DENSITY_UNLIKELY(state->copy_penalty)) {
            DENSITY_ALGORITHM_COPY(DENSITY_CHAMELEON_WORK_BLOCK_SIZE);
            DENSITY_ALGORITHM_INCREASE_COPY_PENALTY_START;
        } else {
            const uint8_t *in_start = *in;
            density_chameleon_decode_read_signature(in, &signature);
            density_chameleon_decode_256(in, out, signature, (density_chameleon_dictionary *const) state->dictionary);
            DENSITY_ALGORITHM_TEST_INCOMPRESSIBILITY((*in - in_start), DENSITY_CHAMELEON_WORK_BLOCK_SIZE);
        }
    }

    if (*out > out_limit)
        return DENSITY_ALGORITHMS_EXIT_STATUS_OUTPUT_STALL;

    read_signature:
    if (in_size - (*in - start) < sizeof(density_chameleon_signature))
        return DENSITY_ALGORITHMS_EXIT_STATUS_INPUT_STALL;
    shift = 0;
    density_chameleon_decode_read_signature(in, &signature);
    read_and_decode_4:
    switch (in_size - (*in - start)) {
        case 0:
        case 1:
            if (density_chameleon_decode_test_compressed(signature, shift))
                return DENSITY_ALGORITHMS_EXIT_STATUS_ERROR_DURING_PROCESSING;
            else    // End marker
                goto process_remaining_bytes;
        case 2:
        case 3:
            if (density_chameleon_decode_test_compressed(signature, shift++))
                density_chameleon_decode_kernel(in, out, true, (density_chameleon_dictionary *const) state->dictionary);
            else    // End marker
                goto process_remaining_bytes;
            break;
        default:
            density_chameleon_decode_4(in, out, signature, shift++, (density_chameleon_dictionary *const) state->dictionary);
            break;
    }

    if (DENSITY_UNLIKELY(shift == density_bitsizeof(density_chameleon_signature)))
        goto read_signature;
    else
        goto read_and_decode_4;

    process_remaining_bytes:
    remaining = in_size - (*in - start);
    DENSITY_ALGORITHM_COPY(remaining);

    return DENSITY_ALGORITHMS_EXIT_STATUS_FINISHED;
}
