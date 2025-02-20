/*
 * BriefLZ - small fast Lempel-Ziv
 *
 * C safe depacker
 *
 * Copyright (c) 2002-2018 Joergen Ibsen
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 *   1. The origin of this software must not be misrepresented; you must
 *      not claim that you wrote the original software. If you use this
 *      software in a product, an acknowledgment in the product
 *      documentation would be appreciated but is not required.
 *
 *   2. Altered source versions must be plainly marked as such, and must
 *      not be misrepresented as being the original software.
 *
 *   3. This notice may not be removed or altered from any source
 *      distribution.
 */

#include "brieflz.h"

/* Internal data structure */
struct blz_state {
	const unsigned char *src;
	unsigned char *dst;
	unsigned long src_avail;
	unsigned long dst_avail;
	unsigned int tag;
	int bits_left;
};

#if !defined(BLZ_NO_LUT)
static const unsigned char blz_gamma_lookup[256][2] = {
	/* 00xxxxxx = 2 */
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
	{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},

	/* 0100xxxx = 4 */
	{4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4},
	{4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4},

	/* 010100xx = 8 */
	{8, 6}, {8, 6}, {8, 6}, {8, 6},

	/* 01010100 = 16  01010101 = 16+ 01010110 = 17  01010111 = 17+ */
	{16, 8}, {16, 0}, {17, 8}, {17, 0},

	/* 010110xx = 9 */
	{9, 6}, {9, 6}, {9, 6}, {9, 6},

	/* 01011100 = 18  01011101 = 18+ 01011110 = 19  01011111 = 19+ */
	{18, 8}, {18, 0}, {19, 8}, {19, 0},

	/* 0110xxxx = 5 */
	{5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4},
	{5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4}, {5, 4},

	/* 011100xx = 10 */
	{10, 6}, {10, 6}, {10, 6}, {10, 6},

	/* 01110100 = 20  01110101 = 20+ 01110110 = 21  01110111 = 21+ */
	{20, 8}, {20, 0}, {21, 8}, {21, 0},

	/* 011110xx = 11 */
	{11, 6}, {11, 6}, {11, 6}, {11, 6},

	/* 01111100 = 22  01111101 = 22+ 01111110 = 23  01111111 = 23+ */
	{22, 8}, {22, 0}, {23, 8}, {23, 0},

	/* 10xxxxxx = 3 */
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},
	{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2},

	/* 1100xxxx = 6 */
	{6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4},
	{6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4}, {6, 4},

	/* 110100xx = 12 */
	{12, 6}, {12, 6}, {12, 6}, {12, 6},

	/* 11010100 = 24  11010101 = 24+ 11010110 = 25  11010111 = 25+ */
	{24, 8}, {24, 0}, {25, 8}, {25, 0},

	/* 110110xx = 13 */
	{13, 6}, {13, 6}, {13, 6}, {13, 6},

	/* 11011100 = 26  11011101 = 26+ 11011110 = 27  11011111 = 27+ */
	{26, 8}, {26, 0}, {27, 8}, {27, 0},

	/* 1110xxxx = 7 */
	{7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4},
	{7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4}, {7, 4},

	/* 111100xx = 14 */
	{14, 6}, {14, 6}, {14, 6}, {14, 6},

	/* 11110100 = 28  11110101 = 28+ 11110110 = 29  11110111 = 29+ */
	{28, 8}, {28, 0}, {29, 8}, {29, 0},

	/* 111110xx = 15 */
	{15, 6}, {15, 6}, {15, 6}, {15, 6},

	/* 11111100 = 30  11111101 = 30+ 11111110 = 31  11111111 = 31+ */
	{30, 8}, {30, 0}, {31, 8}, {31, 0}
};
#endif

static int
blz_getbit_safe(struct blz_state *bs, unsigned int *result)
{
	unsigned int bit;

	/* Check if tag is empty */
	if (!bs->bits_left--) {
		if (bs->src_avail < 2) {
			return 0;
		}
		bs->src_avail -= 2;

		/* Load next tag */
		bs->tag = (unsigned int) bs->src[0]
		       | ((unsigned int) bs->src[1] << 8);
		bs->src += 2;
		bs->bits_left = 15;
	}

	/* Shift bit out of tag */
	bit = (bs->tag & 0x8000) ? 1 : 0;
	bs->tag <<= 1;

	*result = bit;

	return 1;
}

static int
blz_getgamma_safe(struct blz_state *bs, unsigned long *result)
{
	unsigned int bit;
	unsigned long v = 1;

#if !defined(BLZ_NO_LUT)
	/* Decode up to 8 bits of gamma2 code using lookup if possible */
	if (bs->bits_left >= 8) {
		unsigned int top8 = (bs->tag >> 8) & 0x00FF;
		int shift;

		v = blz_gamma_lookup[top8][0];
		shift = (int) blz_gamma_lookup[top8][1];

		if (shift) {
			bs->tag <<= shift;
			bs->bits_left -= shift;

			*result = v;

			return 1;
		}

		bs->tag <<= 8;
		bs->bits_left -= 8;
	}
#endif

	/* Input gamma2-encoded bits */
	do {
		if (!blz_getbit_safe(bs, &bit)) {
			return 0;
		}

		if (v & 0x80000000UL) {
			return 0;
		}

		v = (v << 1) + bit;

		if (!blz_getbit_safe(bs, &bit)) {
			return 0;
		}
	} while (bit);

	*result = v;

	return 1;
}

unsigned long
blz_depack_safe(const void *src, unsigned long src_size,
                void *dst, unsigned long depacked_size)
{
	struct blz_state bs;
	unsigned long dst_size = 0;
	unsigned int bit;

	bs.src = (const unsigned char *) src;
	bs.src_avail = src_size;
	bs.dst = (unsigned char *) dst;
	bs.dst_avail = depacked_size;

	/* Initialise to one bit left in tag; that bit is zero (a literal) */
	bs.bits_left = 1;
	bs.tag = 0;

	/* Main decompression loop */
	while (dst_size < depacked_size) {
		if (!blz_getbit_safe(&bs, &bit)) {
			return BLZ_ERROR;
		}

		if (bit) {
			unsigned long len;
			unsigned long off;

			/* Input match length and offset */
			if (!blz_getgamma_safe(&bs, &len)) {
				return BLZ_ERROR;
			}
			if (!blz_getgamma_safe(&bs, &off)) {
				return BLZ_ERROR;
			}

			len += 2;
			off -= 2;

			if (off >= 0x00FFFFFFUL) {
				return BLZ_ERROR;
			}

			if (!bs.src_avail--) {
				return BLZ_ERROR;
			}

			off = (off << 8) + (unsigned long) *bs.src++ + 1;

			if (off > depacked_size - bs.dst_avail) {
				return BLZ_ERROR;
			}

			if (len > bs.dst_avail) {
				return BLZ_ERROR;
			}

			bs.dst_avail -= len;

			/* Copy match */
			{
				const unsigned char *p = bs.dst - off;
				unsigned long i;

				for (i = len; i > 0; --i) {
					*bs.dst++ = *p++;
				}
			}

			dst_size += len;
		}
		else {
			/* Copy literal */
			if (!bs.src_avail-- || !bs.dst_avail--) {
				return BLZ_ERROR;
			}
			*bs.dst++ = *bs.src++;

			dst_size++;
		}
	}

	/* Return decompressed size */
	return dst_size;
}

/* clang -g -O1 -fsanitize=fuzzer,address -DBLZ_FUZZING depacks.c */
#if defined(BLZ_FUZZING)
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
	if (size > ULONG_MAX / 2) { return 0; }
	void *depacked = malloc(4096);
	if (!depacked) { abort(); }
	blz_depack_safe(data, size, depacked, 4096);
	free(depacked);
	return 0;
}
#endif
