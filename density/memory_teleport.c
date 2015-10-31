/*
 * Centaurean Density
 *
 * Copyright (c) 2014, Guillaume Voirin
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
 * 23/12/14 17:01
 */

#include "memory_teleport.h"
#include "memory_location.h"

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_memory_teleport *density_memory_teleport_allocate(const uint_fast64_t size, void *(*mem_alloc)(size_t)) {
    density_memory_teleport *teleport = mem_alloc(sizeof(density_memory_teleport));

    teleport->stagingMemoryLocation = mem_alloc(sizeof(density_staging_memory_location));
    teleport->stagingMemoryLocation->memoryLocation = density_memory_location_allocate(mem_alloc);
    teleport->stagingMemoryLocation->memoryLocation->pointer = mem_alloc(size * sizeof(density_byte));
    teleport->stagingMemoryLocation->memoryLocation->available_bytes = 0;

    teleport->stagingMemoryLocation->originalPointer = teleport->stagingMemoryLocation->memoryLocation->pointer;
    teleport->stagingMemoryLocation->writePointer = teleport->stagingMemoryLocation->originalPointer;

    teleport->directMemoryLocation = density_memory_location_allocate(mem_alloc);
    teleport->directMemoryLocation->available_bytes = 0;

    return teleport;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_free(density_memory_teleport *teleport, void (*mem_free)(void *)) {
    density_memory_location_free(teleport->directMemoryLocation, mem_free);

    mem_free(teleport->stagingMemoryLocation->originalPointer);
    density_memory_location_free(teleport->stagingMemoryLocation->memoryLocation, mem_free);
    mem_free(teleport->stagingMemoryLocation);

    mem_free(teleport);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_change_input_buffer(density_memory_teleport *restrict teleport, const density_byte *restrict in, const uint_fast64_t availableIn) {
    density_memory_location_encapsulate(teleport->directMemoryLocation, (density_byte*)in, availableIn);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_reset_staging_buffer(density_memory_teleport *restrict teleport) {
    teleport->stagingMemoryLocation->memoryLocation->pointer = teleport->stagingMemoryLocation->originalPointer;
    teleport->stagingMemoryLocation->memoryLocation->available_bytes = 0;
    teleport->stagingMemoryLocation->writePointer = teleport->stagingMemoryLocation->originalPointer;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(density_memory_teleport *restrict teleport) {
    const uint_fast64_t addonBytes = teleport->directMemoryLocation->available_bytes;

    DENSITY_MEMCPY(teleport->stagingMemoryLocation->writePointer, teleport->directMemoryLocation->pointer, addonBytes);
    teleport->stagingMemoryLocation->writePointer += addonBytes;
    teleport->stagingMemoryLocation->memoryLocation->available_bytes += addonBytes;

    teleport->directMemoryLocation->pointer += addonBytes;
    teleport->directMemoryLocation->available_bytes = 0;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_memory_location *density_memory_teleport_read(density_memory_teleport *restrict teleport, const uint_fast64_t bytes) {
    const uint_fast64_t directAvailableBytes = teleport->directMemoryLocation->available_bytes;
    const uint_fast64_t stagingAvailableBytes = teleport->stagingMemoryLocation->memoryLocation->available_bytes;
    uint_fast64_t addonBytes;

    if (density_unlikely(stagingAvailableBytes)) {
        if (stagingAvailableBytes >= bytes) {
            return teleport->stagingMemoryLocation->memoryLocation;
        } else if (density_unlikely((addonBytes = (bytes - stagingAvailableBytes)) <= directAvailableBytes)) {
            if (stagingAvailableBytes <= (teleport->directMemoryLocation->initial_available_bytes - directAvailableBytes)) {    // Revert to direct buffer reading
                density_memory_teleport_reset_staging_buffer(teleport);

                teleport->directMemoryLocation->pointer -= stagingAvailableBytes;
                teleport->directMemoryLocation->available_bytes += stagingAvailableBytes;

                return teleport->directMemoryLocation;
            } else { // Copy missing bytes from direct input buffer
                DENSITY_MEMCPY(teleport->stagingMemoryLocation->writePointer, teleport->directMemoryLocation->pointer, addonBytes);
                teleport->stagingMemoryLocation->writePointer += addonBytes;
                teleport->stagingMemoryLocation->memoryLocation->available_bytes += addonBytes;

                teleport->directMemoryLocation->pointer += addonBytes;
                teleport->directMemoryLocation->available_bytes -= addonBytes;

                return teleport->stagingMemoryLocation->memoryLocation;
            }
        } else {    // Copy as much as we can from direct input buffer
            density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(teleport);
            return NULL;
        }
    } else {
        if (density_likely(directAvailableBytes >= bytes)) {
            return teleport->directMemoryLocation;
        } else {  // Copy what we have in our staging buffer
            density_memory_teleport_copy_from_direct_buffer_to_staging_buffer(teleport);
            return NULL;
        }
    }
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_memory_location *density_memory_teleport_read_reserved(density_memory_teleport *restrict teleport, const uint_fast64_t bytes, const uint_fast64_t reserved) {
    return density_memory_teleport_read(teleport, bytes + reserved);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE density_memory_location *density_memory_teleport_read_remaining_reserved(density_memory_teleport *restrict teleport, const uint_fast64_t reserved) {
    return density_memory_teleport_read_reserved(teleport, density_memory_teleport_available_bytes_reserved(teleport, reserved), reserved);
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE uint_fast64_t density_memory_teleport_available_bytes(density_memory_teleport *teleport) {
    return teleport->stagingMemoryLocation->memoryLocation->available_bytes + teleport->directMemoryLocation->available_bytes;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE uint_fast64_t density_memory_teleport_available_bytes_reserved(density_memory_teleport *teleport, const uint_fast64_t reserved) {
    uint_fast64_t contained = teleport->stagingMemoryLocation->memoryLocation->available_bytes + teleport->directMemoryLocation->available_bytes;
    if (density_unlikely(reserved >= contained))
        return 0;
    else
        return contained - reserved;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_copy(density_memory_teleport *restrict teleport, density_memory_location *restrict out, const uint_fast64_t bytes) {
    uint_fast64_t fromStaging = 0;
    uint_fast64_t fromDirect = 0;
    uint_fast64_t stagingAvailableBytes = teleport->stagingMemoryLocation->memoryLocation->available_bytes;
    uint_fast64_t directAvailableBytes = teleport->directMemoryLocation->available_bytes;

    if (stagingAvailableBytes) {
        if (bytes <= stagingAvailableBytes) {
            fromStaging = bytes;
        } else if (bytes <= stagingAvailableBytes + directAvailableBytes) {
            fromStaging = stagingAvailableBytes;
            fromDirect = bytes - stagingAvailableBytes;
        } else {
            fromStaging = stagingAvailableBytes;
            fromDirect = directAvailableBytes;
        }
    } else {
        if (bytes <= directAvailableBytes) {
            fromDirect = bytes;
        } else {
            fromDirect = directAvailableBytes;
        }
    }

    DENSITY_MEMCPY(out->pointer, teleport->stagingMemoryLocation->memoryLocation->pointer, fromStaging);
    if (fromStaging == stagingAvailableBytes)
        density_memory_teleport_reset_staging_buffer(teleport);
    else {
        teleport->stagingMemoryLocation->memoryLocation->pointer += fromStaging;
        teleport->stagingMemoryLocation->memoryLocation->available_bytes -= fromStaging;
    }
    out->pointer += fromStaging;
    out->available_bytes -= fromStaging;

    DENSITY_MEMCPY(out->pointer, teleport->directMemoryLocation->pointer, fromDirect);
    teleport->directMemoryLocation->pointer += fromDirect;
    teleport->directMemoryLocation->available_bytes -= fromDirect;
    out->pointer += fromDirect;
    out->available_bytes -= fromDirect;
}

DENSITY_WINDOWS_EXPORT DENSITY_FORCE_INLINE void density_memory_teleport_copy_remaining(density_memory_teleport *restrict teleport, density_memory_location *restrict out) {
    return density_memory_teleport_copy(teleport, out, density_memory_teleport_available_bytes(teleport));
}