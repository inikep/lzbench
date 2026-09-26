// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_alone_decoder.c
/// \brief      Test lzma_alone_decoder()
//
//  Author:     Lasse Collin
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


#define MEMLIMIT (16U << 20)
#define TOO_BIG_ALLOC (1U << 20)


#ifdef HAVE_DECODERS
static void *
my_alloc(void *opaque, size_t nmemb, size_t size)
{
	(void)opaque;
	(void)nmemb;

	if (size >= TOO_BIG_ALLOC)
		return NULL;

	return malloc(size);
}


static lzma_allocator my_allocator = { &my_alloc, NULL, NULL };


static void
test_reuse_after_failure(void)
{
	// 4 KiB dictionary
	size_t small_dict_size;
	uint8_t *small_dict = tuktest_file_from_srcdir(
			"files/good-unknown_size-with_eopm.lzma",
			&small_dict_size);

	// 8 MiB dictionary
	size_t big_dict_size;
	uint8_t *big_dict = tuktest_file_from_srcdir(
			"files/good-unknown_size-with_eopm.lzma",
			&big_dict_size);
	big_dict[2] = 0x00;
	big_dict[3] = 0x80;

	uint8_t out[256];

	lzma_stream strm = LZMA_STREAM_INIT;
	strm.allocator = &my_allocator;

	assert_lzma_ret(lzma_alone_decoder(&strm, MEMLIMIT), LZMA_OK);
	strm.next_in = small_dict;
	strm.avail_in = small_dict_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	assert_lzma_ret(lzma_alone_decoder(&strm, MEMLIMIT), LZMA_OK);
	strm.next_in = big_dict;
	strm.avail_in = big_dict_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEM_ERROR);

	assert_lzma_ret(lzma_alone_decoder(&strm, MEMLIMIT), LZMA_OK);
	strm.next_in = small_dict;
	strm.avail_in = small_dict_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	lzma_end(&strm);
}
#endif


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

#ifndef HAVE_DECODERS
	tuktest_early_skip("Decoder support is disabled");
#else
	tuktest_run(test_reuse_after_failure);
#endif

	return tuktest_end();
}
