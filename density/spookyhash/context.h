/*
 * Centaurean SpookyHash
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
 * 24/01/15 22:32
 *
 * ----------
 * SpookyHash
 * ----------
 *
 * Author(s)
 * Bob Jenkins (http://burtleburtle.net/bob/hash/spooky.html)
 *
 * Description
 * Very fast non cryptographic hash
 */

#ifndef SPOOKYHASH_CONTEXT_H
#define SPOOKYHASH_CONTEXT_H

#include "globals.h"

#define SPOOKYHASH_VARIABLES (12)
#define SPOOKYHASH_BLOCK_SIZE (SPOOKYHASH_VARIABLES * 8)
#define SPOOKYHASH_BUFFER_SIZE (2 * SPOOKYHASH_BLOCK_SIZE)
#define SPOOKYHASH_CONSTANT (0xdeadbeefdeadbeefLL)

typedef struct {
    uint64_t m_data[2 * SPOOKYHASH_VARIABLES];
    uint64_t m_state[SPOOKYHASH_VARIABLES];
    size_t m_length;
    uint8_t m_remainder;
} spookyhash_context;

SPOOKYHASH_WINDOWS_EXPORT spookyhash_context *spookyhash_context_allocate(void *(*)(size_t));

SPOOKYHASH_WINDOWS_EXPORT void spookyhash_context_free(spookyhash_context *, void (*)(void *));

SPOOKYHASH_WINDOWS_EXPORT void spookyhash_context_init(spookyhash_context *, uint64_t, uint64_t);

#endif