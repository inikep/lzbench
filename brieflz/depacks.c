/*
 * BriefLZ - small fast Lempel-Ziv
 *
 * C safe depacker
 *
 * Copyright (c) 2002-2015 Joergen Ibsen
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
	unsigned int bits_left;
};

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
	unsigned long dst_size = 1;
	unsigned int bit;

	/* Check for empty input */
	if (depacked_size == 0) {
		return 0;
	}

	bs.src = (const unsigned char *) src;
	bs.src_avail = src_size;
	bs.dst = (unsigned char *) dst;
	bs.dst_avail = depacked_size;
	bs.bits_left = 0;

	/* First byte verbatim */
	if (!bs.src_avail-- || !bs.dst_avail--) {
		return BLZ_ERROR;
	}
	*bs.dst++ = *bs.src++;

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
