/**
* Modifications Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/*
 * Author: Lasse Collin
 *
 * This file has been put into the public domain.
 * You can do whatever you want with this file.
 */

//xz lzma
#include "algos/lzma/lzma.h"

//aocl lzma
#include "algos/lzma/Alloc.h"
#include "algos/lzma/LzmaDec.h"
#include "algos/lzma/LzmaEnc.h"

/*******************************
 * Compression/decompression *
 ******************************/

/**
* \brief       Mask for preset level
*
* This is useful only if you need to extract the level from the preset
* variable. That should be rare.
*/
#define LZMA_PRESET_LEVEL_MASK  UINT32_C(0x1F)

/*
 * Preset flags
 *
 * Currently only one flag is defined.
 */

/**
 * \brief       Extreme compression preset
 *
 * This flag modifies the preset to make the encoding significantly slower
 * while improving the compression ratio only marginally. This is useful
 * when you don't mind wasting time to get as small result as possible.
 *
 * This flag doesn't affect the memory usage requirements of the decoder (at
 * least not significantly). The memory usage of the encoder may be increased
 * a little but only at the lowest preset levels (0-3).
 */
#define LZMA_PRESET_EXTREME       (UINT32_C(1) << 31)

/************
* Decoding *
************/

/**
* This flag makes lzma_code() return LZMA_NO_CHECK if the input stream
* being decoded has no integrity check. Note that when used with
* lzma_auto_decoder(), all .lzma files will trigger LZMA_NO_CHECK
* if LZMA_TELL_NO_CHECK is used.
*/
#define LZMA_TELL_NO_CHECK              UINT32_C(0x01)


/**
* This flag makes lzma_code() return LZMA_UNSUPPORTED_CHECK if the input
* stream has an integrity check, but the type of the integrity check is not
* supported by this liblzma version or build. Such files can still be
* decoded, but the integrity check cannot be verified.
*/
#define LZMA_TELL_UNSUPPORTED_CHECK     UINT32_C(0x02)


/**
* This flag makes lzma_code() return LZMA_GET_CHECK as soon as the type
* of the integrity check is known. The type can then be got with
* lzma_get_check().
*/
#define LZMA_TELL_ANY_CHECK             UINT32_C(0x04)


/**
* This flag makes lzma_code() not calculate and verify the integrity check
* of the compressed data in .xz files. This means that invalid integrity
* check values won't be detected and LZMA_DATA_ERROR won't be returned in
* such cases.
*
* This flag only affects the checks of the compressed data itself; the CRC32
* values in the .xz headers will still be verified normally.
*
* Don't use this flag unless you know what you are doing. Possible reasons
* to use this flag:
*
*   - Trying to recover data from a corrupt .xz file.
*
*   - Speeding up decompression, which matters mostly with SHA-256
*     or with files that have compressed extremely well. It's recommended
*     to not use this flag for this purpose unless the file integrity is
*     verified externally in some other way.
*
* Support for this flag was added in liblzma 5.1.4beta.
*/
#define LZMA_IGNORE_CHECK               UINT32_C(0x10)


/**
* This flag enables decoding of concatenated files with file formats that
* allow concatenating compressed files as is. From the formats currently
* supported by liblzma, only the .xz and .lz formats allow concatenated
* files. Concatenated files are not allowed with the legacy .lzma format.
*
* This flag also affects the usage of the `action' argument for lzma_code().
* When LZMA_CONCATENATED is used, lzma_code() won't return LZMA_STREAM_END
* unless LZMA_FINISH is used as `action'. Thus, the application has to set
* LZMA_FINISH in the same way as it does when encoding.
*
* If LZMA_CONCATENATED is not used, the decoders still accept LZMA_FINISH
* as `action' for lzma_code(), but the usage of LZMA_FINISH isn't required.
*/
#define LZMA_CONCATENATED               UINT32_C(0x08)


/**
* This flag makes the threaded decoder report errors (like LZMA_DATA_ERROR)
* as soon as they are detected. This saves time when the application has no
* interest in a partially decompressed truncated or corrupt file. Note that
* due to timing randomness, if the same truncated or corrupt input is
* decompressed multiple times with this flag, a different amount of output
* may be produced by different runs, and even the error code might vary.
*
* When using LZMA_FAIL_FAST, it is recommended to use LZMA_FINISH to tell
* the decoder when no more input will be coming because it can help fast
* detection and reporting of truncated files. Note that in this situation
* truncated files might be diagnosed with LZMA_DATA_ERROR instead of
* LZMA_OK or LZMA_BUF_ERROR!
*
* Without this flag the threaded decoder will provide as much output as
* possible at first and then report the pending error. This default behavior
* matches the single-threaded decoder and provides repeatable behavior
* with truncated or corrupt input. There are a few special cases where the
* behavior can still differ like memory allocation failures (LZMA_MEM_ERROR).
*
* Single-threaded decoders currently ignore this flag.
*
* Support for this flag was added in liblzma 5.3.3alpha. Note that in older
* versions this flag isn't supported (LZMA_OPTIONS_ERROR) even by functions
* that ignore this flag in newer liblzma versions.
*/
#define LZMA_FAIL_FAST                  UINT32_C(0x20)


/// Supported flags that can be passed to lzma_stream_decoder(),
/// lzma_auto_decoder(), or lzma_stream_decoder_mt().
#define LZMA_SUPPORTED_FLAGS \
	( LZMA_TELL_NO_CHECK \
	| LZMA_TELL_UNSUPPORTED_CHECK \
	| LZMA_TELL_ANY_CHECK \
	| LZMA_IGNORE_CHECK \
	| LZMA_CONCATENATED \
	| LZMA_FAIL_FAST )

lzma_ret
get_xz_ret(SRes res) {
    switch (res) {
    case SZ_OK:
        return LZMA_OK;

    case SZ_ERROR_DATA:
        return LZMA_DATA_ERROR;
    case SZ_ERROR_MEM:
        return LZMA_MEM_ERROR;
    case SZ_ERROR_UNSUPPORTED:
        return LZMA_FORMAT_ERROR;
    case SZ_ERROR_PARAM:
        return LZMA_OPTIONS_ERROR;
    case SZ_ERROR_INPUT_EOF:
        return LZMA_STREAM_END;
    case SZ_ERROR_PROGRESS:
        return LZMA_PROG_ERROR;

    //unsure about matching error codes for these. Return LZMA_DATA_ERROR
    case SZ_ERROR_CRC:
    case SZ_ERROR_OUTPUT_EOF:
    case SZ_ERROR_READ:
    case SZ_ERROR_WRITE:
    case SZ_ERROR_FAIL:
    case SZ_ERROR_THREAD:
    case SZ_ERROR_ARCHIVE:
    case SZ_ERROR_NO_ARCHIVE:
    default:
        return LZMA_DATA_ERROR;
    }
}

lzma_ret
lzma_easy_buffer_encode(uint32_t preset, lzma_check check,
    const lzma_allocator* allocator, const uint8_t* in,
    size_t in_size, uint8_t* out, size_t* out_pos, size_t out_size) 
{
    // Is preset valid?
    const uint32_t level = preset & LZMA_PRESET_LEVEL_MASK;
    const uint32_t flags = preset & ~LZMA_PRESET_LEVEL_MASK;
    const uint32_t supported_flags = LZMA_PRESET_EXTREME;
    if (level > 9 || (flags & ~supported_flags))
        return LZMA_OPTIONS_ERROR; // invalid preset
    if (flags & LZMA_PRESET_EXTREME)
        return LZMA_OPTIONS_ERROR; // mode not supported in aocl lzma

    // Is check valid?
    if (check != LZMA_CHECK_NONE)
        return LZMA_OPTIONS_ERROR; // other check modes are not supported in aocl lzma

    // Io buffers valid?
    if (in == NULL || in_size == 0 || out == NULL || out_pos == NULL || out_size < (*out_pos))
        return LZMA_OPTIONS_ERROR;

    // Encode
    Byte* inbuf = (Byte*)in;
    SizeT insize = in_size;
    Byte* outbuf = out + (*out_pos);
    SizeT outsize = out_size - (*out_pos);

    CLzmaEncProps encProps;
    SizeT headerSize = LZMA_PROPS_SIZE;
    SizeT outLen = outsize - LZMA_PROPS_SIZE;

    LzmaEncProps_Init(&encProps);
    encProps.level = level;

    SRes res = LzmaEncode(outbuf + LZMA_PROPS_SIZE, &outLen, inbuf,
        insize, &encProps, outbuf, &headerSize, 0, NULL,
        &g_Alloc, &g_Alloc);

    // Map aocl to xz
    lzma_ret ret = get_xz_ret(res);
    if (ret != LZMA_OK)
        return ret;

    *out_pos += (LZMA_PROPS_SIZE + outLen);
    return LZMA_OK;
}

lzma_ret
lzma_stream_buffer_decode(uint64_t* memlimit, uint32_t flags,
    const lzma_allocator* allocator,
    const uint8_t* in, size_t* in_pos, size_t in_size,
    uint8_t* out, size_t* out_pos, size_t out_size)
{
    // Sanity checks
    if (in_pos == NULL || in == NULL || *in_pos > in_size || 
        out_pos == NULL || out == NULL || *out_pos > out_size)
        return LZMA_OPTIONS_ERROR;
    //(in == NULL && *in_pos == in_size), (out == NULL && *out_pos == out_size) modes not supported in aocl lzma

    // Catch flags that are not allowed in buffer-to-buffer decoding.
    if (flags & LZMA_TELL_ANY_CHECK)
        return LZMA_OPTIONS_ERROR;
    if (flags & ~LZMA_SUPPORTED_FLAGS)
        return LZMA_OPTIONS_ERROR;

    // Set aocl variables
    Byte* inbuf = (Byte*)(in + (*in_pos));
    SizeT insize = in_size - (*in_pos);
    Byte* outbuf = (Byte*)(out + (*out_pos));
    SizeT outsize = out_size - (*out_pos);

    // Calling aocl functions
    SizeT outLen = outsize;
    SizeT srcLen = insize - LZMA_PROPS_SIZE;
    ELzmaStatus status;

    SRes res = LzmaDecode(outbuf, &outLen, inbuf + LZMA_PROPS_SIZE,
        &srcLen, inbuf, LZMA_PROPS_SIZE, LZMA_FINISH_END,
        &status, &g_Alloc);

    // Map aocl to xz
    lzma_ret ret = get_xz_ret(res);
    if (ret != LZMA_OK)
        return ret;

    *in_pos += srcLen;
    *out_pos += outLen;

    return LZMA_OK;
}

lzma_ret
lzma_auto_decoder(lzma_stream* strm, uint64_t memlimit, uint32_t flags)
{
    return LZMA_PROG_ERROR;
}

/************
 * Version *
 ************/

uint32_t
lzma_version_number(void)
{
    return LZMA_VERSION;
}


const char*
lzma_version_string(void)
{
    return LZMA_VERSION_STRING;
}
