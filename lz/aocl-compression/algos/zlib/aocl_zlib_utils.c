/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "utils/utils.h"

static int enable_dquick = 0; // flag to enable/disable deflate quick compression
#ifndef AOCL_ENABLE_THREADS
static atomic_flag sync_enable_dquick = ATOMIC_FLAG_INIT;
#endif

void aocl_zlib_set_enable_dquick(int val)
{
    AOCL_ENTER_CRITICAL(sync_enable_dquick)
    enable_dquick = val;
    AOCL_EXIT_CRITICAL(sync_enable_dquick)
}

int aocl_zlib_get_enable_dquick(void)
{
    int ret = 0;
    AOCL_ENTER_CRITICAL(sync_enable_dquick)
    ret = enable_dquick;
    AOCL_EXIT_CRITICAL(sync_enable_dquick)
    return ret;
}

#ifdef AOCL_ENABLE_THREADS
#include "zconf.h"
#include "zutil.h"
int insert_Header_generic(Bytef *dest, int level, const int wrap) {
    int header_size = 0; // no header
    if(wrap == 1)
    {
        // zlib
        uInt header = (Z_DEFLATED + (7 << 4)) << 8;
        uInt level_flags;
        if(level < 2)
            level_flags = 0; // compressor used fastest algorithm
        else if (level < 6)
            level_flags = 1; // compressor used fast algorithm
        else if (level == 6)
            level_flags = 2; // compressor used default algorithm
        else
            level_flags = 3; // compressor used maximum compression, slowest algorithm

        header |= (level_flags << 6);
        header += 31 - (header % 31);
        dest[0] = (Byte)(header >> 8);
        dest[1] = (Byte)(header & 0xff);
        header_size = 2;
    }
#ifdef GZIP
    else if(wrap == 2)
    {
        // gzip
        dest[0] = 0x1F; // IDentification 1
        dest[1] = 0x8B; // IDentification 2
        dest[2] = 0x08; // CM
        dest[3] = 0x00; // FLG
        dest[4] = 0x00; // Modification TIME
        dest[5] = 0x00; // Modification TIME
        dest[6] = 0x00; // Modification TIME
        dest[7] = 0x00; // Modification TIME
        dest[8] = (level == 9 ? 0x02 : (level == 1 ? 0x04 : 0x00)); // XFL
        dest[9] = OS_CODE; // OS_CODE
        header_size = 10;
    }
#endif
    return header_size;
}

int insert_Trailer_generic(Bytef *dest, AOCL_UINT32 checksum, uLong sourceLen, const int wrap) {
    int trailer_size = 0; // no trailer
    if(wrap == 1) {
        // zlib
        checksum = ((((checksum) >> 24) & 0xff) + (((checksum) >> 8) & 0xff00) + (((checksum) & 0xff00) << 8) + (((checksum) & 0xff) << 24));
        memcpy(dest, &checksum, 4);
        trailer_size = 4;
    }
#ifdef GZIP
    else if (wrap == 2) {
        // gzip
        memcpy(dest, &checksum, 4); // CRC32
        sourceLen = sourceLen % 4294967296L;
        memcpy(dest + 4, &((AOCL_UINT32)sourceLen), 4); // ISIZE
        trailer_size = 8;
    }
#endif
    return trailer_size;
}
#endif /* AOCL_ENABLE_THREADS */
