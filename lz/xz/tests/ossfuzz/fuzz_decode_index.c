// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       fuzz_decode_alone.c
/// \brief      Fuzz test program for .xz Index decoding
//
//  Author:     Lasse Collin
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
		// The initial pointer value must be ignored.
		lzma_index *idx = (void *)1;
		ret = lzma_index_decoder(&strm, &idx, MEM_LIMIT);

		if (ret == LZMA_MEM_ERROR)
			continue;

		if (ret != LZMA_OK) {
			// This should never happen unless the system has
			// no free memory or address space to allow the small
			// allocations that the initialization requires.
			fprintf(stderr, "lzma_index_decoder() failed (%d)\n",
					ret);
			abort();
		}

		fuzz_code(&strm, strm.next_in, strm.avail_in);
		lzma_index_end(idx, NULL);
	}

	{
		// Fuzz the single-call API too.
		lzma_index *idx = (void *)1;
		uint64_t memlimit = MEM_LIMIT;
		size_t inbuf_pos = 0;
		if (lzma_index_buffer_decode(&idx, &memlimit, NULL,
				inbuf, &inbuf_pos, inbuf_size)
				== LZMA_PROG_ERROR) {
			fprintf(stderr, "lzma_index_buffer_decode() failed "
					"with LZMA_PROG_ERROR\n");
			abort();
		}

		lzma_index_end(idx, NULL);
	}

	lzma_end(&strm);
	return 0;
}
