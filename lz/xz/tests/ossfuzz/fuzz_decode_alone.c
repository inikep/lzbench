// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       fuzz_decode_alone.c
/// \brief      Fuzz test program for liblzma .lzma decoding
//
//  Authors:    Maksym Vatsyk
//              Lasse Collin
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
		ret = lzma_alone_decoder(&strm, MEM_LIMIT);

		if (ret == LZMA_MEM_ERROR)
			continue;

		if (ret != LZMA_OK) {
			// This should never happen unless the system has
			// no free memory or address space to allow the small
			// allocations that the initialization requires.
			fprintf(stderr, "lzma_alone_decoder() failed (%d)\n",
					ret);
			abort();
		}

		fuzz_code(&strm, strm.next_in, strm.avail_in);
	}

	// Free the allocated memory.
	lzma_end(&strm);
	return 0;
}
