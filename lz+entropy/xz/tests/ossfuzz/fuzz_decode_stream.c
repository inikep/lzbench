// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       fuzz_decode_stream.c
/// \brief      Fuzz test program for single threaded .xz decoding
//
//  Authors:    Lasse Collin
//              Maksym Vatsyk
//
///////////////////////////////////////////////////////////////////////////////

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "lzma.h"
#include "fuzz_common.h"


extern int
LLVMFuzzerTestOneInput(const uint8_t *inbuf, size_t inbuf_size)
{
	lzma_stream strm = LZMA_STREAM_INIT;
	prepare_stream(&strm, inbuf, inbuf_size);

	lzma_ret ret;

	for (int i = 0; i < 3; ++i) {
		// Initialize a .xz decoder using the memory usage limit
		// defined in fuzz_common.h
		//
		// After the first two iterations, enable support for
		// concatenated .xz files which is used when decompressing
		// regular .xz files (instead of data embedded inside some
		// other file format). This way the first iteration won't
		// consume all the input until the input is invalid.
		//
		// Integrity checks on the uncompressed data are ignored to
		// make fuzzing more effective (incorrect check values won't
		// prevent the decoder from processing more input).
		//
		// The flag LZMA_IGNORE_CHECK doesn't disable verification
		// of header CRC32 values. Those checks are disabled when
		// liblzma is built with the
		// #define FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION.
		ret = lzma_stream_decoder(&strm, MEM_LIMIT, LZMA_IGNORE_CHECK
				| (i >= 2 ? LZMA_CONCATENATED : 0));

		if (ret == LZMA_MEM_ERROR)
			continue;

		if (ret != LZMA_OK) {
			// This should never happen unless the system has
			// no free memory or address space to allow the small
			// allocations that the initialization requires.
			fprintf(stderr, "lzma_stream_decoder() failed (%d)\n",
					ret);
			abort();
		}

		fuzz_code(&strm, inbuf, inbuf_size);
	}

	// Free the allocated memory.
	lzma_end(&strm);

	return 0;
}
