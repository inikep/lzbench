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
 * 16/12/14 17:12
 */

#ifndef DENSITY_MEMORY_TELEPORT_H
#define DENSITY_MEMORY_TELEPORT_H

#include <string.h>

#include "globals.h"
#include "density_api.h"
#include "memory_location.h"
#include "density_api_data_structures.h"

typedef struct {
    density_byte *originalPointer;
    density_byte *writePointer;
    density_memory_location *memoryLocation;
} density_staging_memory_location;

typedef struct {
    density_staging_memory_location *stagingMemoryLocation;
    density_memory_location *directMemoryLocation;
} density_memory_teleport;

DENSITY_WINDOWS_EXPORT density_memory_teleport *density_memory_teleport_allocate(uint_fast64_t, void *(*)(size_t));

DENSITY_WINDOWS_EXPORT void density_memory_teleport_free(density_memory_teleport *, void (*)(void *));

DENSITY_WINDOWS_EXPORT void density_memory_teleport_change_input_buffer(density_memory_teleport *, const density_byte *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT void density_memory_teleport_reset_staging_buffer(density_memory_teleport *);

DENSITY_WINDOWS_EXPORT void density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(density_memory_teleport *);

DENSITY_WINDOWS_EXPORT density_memory_location *density_memory_teleport_read(density_memory_teleport *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT density_memory_location *density_memory_teleport_read_reserved(density_memory_teleport *, const uint_fast64_t, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT density_memory_location *density_memory_teleport_read_remaining_reserved(density_memory_teleport *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT uint_fast64_t density_memory_teleport_available_bytes(density_memory_teleport *);

DENSITY_WINDOWS_EXPORT uint_fast64_t density_memory_teleport_available_bytes_reserved(density_memory_teleport *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT void density_memory_teleport_copy(density_memory_teleport *, density_memory_location *, const uint_fast64_t);

DENSITY_WINDOWS_EXPORT void density_memory_teleport_copy_remaining(density_memory_teleport *, density_memory_location *);

#endif
