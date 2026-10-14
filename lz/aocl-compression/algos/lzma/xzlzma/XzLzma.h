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

#include "base.h"
#include "check.h"

#ifndef __XZ_LZMA_H
#define __XZ_LZMA_H

#ifndef LZMA_H_INTERNAL
#	error Never include this file directly. Use <lzma.h> instead.
#endif

#ifndef LZMALIB_VISIBILITY
#  if defined(__GNUC__) && (__GNUC__ >= 4)
#    define LZMALIB_VISIBILITY __attribute__ ((visibility ("default")))
#  else
#    define LZMALIB_VISIBILITY
#  endif
#endif
#if defined(LZMA_DLL_EXPORT) && (LZMA_DLL_EXPORT==1)
#  define LZMALIB_API __declspec(dllexport) LZMALIB_VISIBILITY
#elif defined(LZMA_DLL_IMPORT) && (LZMA_DLL_IMPORT==1)
#  define LZMALIB_API __declspec(dllimport) LZMALIB_VISIBILITY /* It isn't required but allows to generate better code, saving a function pointer load from the IAT and an indirect jump.*/
#else
#  define LZMALIB_API LZMALIB_VISIBILITY
#endif

/**
 * \brief       Single-call .xz Stream encoding using a preset number
 *
 * The maximum required output buffer size can be calculated with
 * lzma_stream_buffer_bound().
 *
 * \param       preset      Compression preset to use. See the description
 *                          in lzma_easy_encoder().
 * \param       check       Type of the integrity check to calculate from
 *                          uncompressed data.
 * \param       allocator   lzma_allocator for custom allocator functions.
 *                          Set to NULL to use malloc() and free().
 * \param       in          Beginning of the input buffer
 * \param       in_size     Size of the input buffer
 * \param       out         Beginning of the output buffer
 * \param       out_pos     The next byte will be written to out[*out_pos].
 *                          *out_pos is updated only if encoding succeeds.
 * \param       out_size    Size of the out buffer; the first byte into
 *                          which no data is written to is out[out_size].
 *
 * \return      - LZMA_OK: Encoding was successful.
 *              - LZMA_BUF_ERROR: Not enough output buffer space.
 *              - LZMA_UNSUPPORTED_CHECK
 *              - LZMA_OPTIONS_ERROR
 *              - LZMA_MEM_ERROR
 *              - LZMA_DATA_ERROR
 *              - LZMA_PROG_ERROR
 * 
 * Current status:
 *  preset    : Used to read level only. Other flags are not supported.
 *  check     : Only LZMA_CHECK_NONE is supported.
 *	allocator : Not supported. If non-NULL, it gets ignored.
 */
LZMALIB_API lzma_ret
lzma_easy_buffer_encode(uint32_t preset, lzma_check check,
	const lzma_allocator* allocator, const uint8_t* in,
	size_t in_size, uint8_t* out, size_t* out_pos, size_t out_size);

/**
 * \brief       Single-call .xz Stream decoder
 *
 * \param       memlimit    Pointer to how much memory the decoder is allowed
 *                          to allocate. The value pointed by this pointer is
 *                          modified if and only if LZMA_MEMLIMIT_ERROR is
 *                          returned.
 * \param       flags       Bitwise-or of zero or more of the decoder flags:
 *                          LZMA_TELL_NO_CHECK, LZMA_TELL_UNSUPPORTED_CHECK,
 *                          LZMA_CONCATENATED. Note that LZMA_TELL_ANY_CHECK
 *                          is not allowed and will return LZMA_PROG_ERROR.
 * \param       allocator   lzma_allocator for custom allocator functions.
 *                          Set to NULL to use malloc() and free().
 * \param       in          Beginning of the input buffer
 * \param       in_pos      The next byte will be read from in[*in_pos].
 *                          *in_pos is updated only if decoding succeeds.
 * \param       in_size     Size of the input buffer; the first byte that
 *                          won't be read is in[in_size].
 * \param       out         Beginning of the output buffer
 * \param       out_pos     The next byte will be written to out[*out_pos].
 *                          *out_pos is updated only if decoding succeeds.
 * \param       out_size    Size of the out buffer; the first byte into
 *                          which no data is written to is out[out_size].
 *
 * \return      - LZMA_OK: Decoding was successful.
 *              - LZMA_FORMAT_ERROR
 *              - LZMA_OPTIONS_ERROR
 *              - LZMA_DATA_ERROR
 *              - LZMA_NO_CHECK: This can be returned only if using
 *                the LZMA_TELL_NO_CHECK flag.
 *              - LZMA_UNSUPPORTED_CHECK: This can be returned only if using
 *                the LZMA_TELL_UNSUPPORTED_CHECK flag.
 *              - LZMA_MEM_ERROR
 *              - LZMA_MEMLIMIT_ERROR: Memory usage limit was reached.
 *                The minimum required memlimit value was stored to *memlimit.
 *              - LZMA_BUF_ERROR: Output buffer was too small.
 *              - LZMA_PROG_ERROR
 * 
 * Current status:
 *  memlimit  : Not supported. If non-NULL, it gets ignored.
 *  flags     : Not supported.
 *	allocator : Not supported. If non-NULL, it gets ignored.
 */
LZMALIB_API lzma_ret
lzma_stream_buffer_decode(uint64_t* memlimit, uint32_t flags,
	const lzma_allocator* allocator,
	const uint8_t* in, size_t* in_pos, size_t in_size,
	uint8_t* out, size_t* out_pos, size_t out_size);

/* 
* Current status:
*  Some applications that need xz utils library, look for this function
*  to confirm if the said .so is liblzma.so.
*  Dummy API provided here to facilitate this. 
*  It just returns LZMA_PROG_ERROR. */
LZMALIB_API lzma_ret
lzma_auto_decoder(
	lzma_stream* strm, uint64_t memlimit, uint32_t flags)
	lzma_nothrow lzma_attr_warn_unused_result;
#endif
