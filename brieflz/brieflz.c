/*
 * BriefLZ - small fast Lempel-Ziv
 *
 * C packer
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

/*
 * Number of bits of hash to use for lookup.
 *
 * The size of the lookup table (and thus workmem) is computed from this.
 *
 * Values between 10 and 18 work well. The default value 17 corresponds
 * to a workmem size of 1mb on 64-bit systems.
 */
#ifndef BLZ_HASH_BITS
#  define BLZ_HASH_BITS 17
#endif

#define LOOKUP_SIZE (1UL << BLZ_HASH_BITS)

#define WORKMEM_SIZE (LOOKUP_SIZE * sizeof(const unsigned char *))

/* Internal data structure */
struct blz_state {
	const unsigned char *src;
	unsigned char *dst;
	unsigned char *tagpos;
	unsigned int tag;
	unsigned int bits_left;
};

static void
blz_putbit(struct blz_state *bs, unsigned int bit)
{
	/* Check if tag is full */
	if (!bs->bits_left--) {
		/* store tag */
		bs->tagpos[0] = bs->tag & 0x00FF;
		bs->tagpos[1] = (bs->tag >> 8) & 0x00FF;

		/* init next tag */
		bs->tagpos = bs->dst;
		bs->dst += 2;
		bs->bits_left = 15;
	}

	/* Shift bit into tag */
	bs->tag = (bs->tag << 1) + bit;
}

static void
blz_putgamma(struct blz_state *bs, unsigned long val)
{
	unsigned long mask = val >> 1;

	/* mask = highest_bit(val >> 1) */
	while (mask & (mask - 1)) {
		mask &= mask - 1;
	}

	/* Output gamma2-encoded bits */
	blz_putbit(bs, (val & mask) ? 1 : 0);

	while (mask >>= 1) {
		blz_putbit(bs, 1);
		blz_putbit(bs, (val & mask) ? 1 : 0);
	}

	blz_putbit(bs, 0);
}

static unsigned long
blz_hash4(const unsigned char *s)
{
	unsigned long val = (unsigned long) s[0]
	                 | ((unsigned long) s[1] << 8)
	                 | ((unsigned long) s[2] << 16)
	                 | ((unsigned long) s[3] << 24);

	return ((val * 2654435761UL) & 0xFFFFFFFFUL) >> (32 - BLZ_HASH_BITS);
}

unsigned long
blz_workmem_size(unsigned long src_size)
{
	(void) src_size;

	return WORKMEM_SIZE;
}

unsigned long
blz_max_packed_size(unsigned long src_size)
{
	return src_size + src_size / 8 + 64;
}

unsigned long
blz_pack(const void *src, void *dst, unsigned long src_size, void *workmem)
{
	struct blz_state bs;
	const unsigned char **lookup = (const unsigned char **) workmem;
	const unsigned char *prevsrc = (const unsigned char *) src;
	unsigned long src_avail = src_size;

	/* Check for empty input */
	if (src_avail == 0) {
		return 0;
	}

	/* Initialize lookup[] */
	{
		unsigned long i;

		for (i = 0; i < LOOKUP_SIZE; ++i) {
			lookup[i] = 0;
		}
	}

	bs.src = (const unsigned char *) src;
	bs.dst = (unsigned char *) dst;

	/* First byte verbatim */
	*bs.dst++ = *bs.src++;

	/* Check for 1 byte input */
	if (--src_avail == 0) {
		return 1;
	}

	/* Initialize first tag */
	bs.tagpos = bs.dst;
	bs.dst += 2;
	bs.tag = 0;
	bs.bits_left = 16;

	/* Main compression loop */
	while (src_avail > 4) {
		const unsigned char *p;
		unsigned long len = 0;

		/* Update lookup[] up to current position */
		while (prevsrc < bs.src) {
			lookup[blz_hash4(prevsrc)] = prevsrc;
			prevsrc++;
		}

		/* Look up current position */
		p = lookup[blz_hash4(bs.src)];

		/* Check match */
		if (p) {
			while (len < src_avail && p[len] == bs.src[len]) {
				++len;
			}
		}

		/* Output match or literal */
		if (len > 3) {
			unsigned long off = (unsigned long) (bs.src - p - 1);

			/* Output match tag */
			blz_putbit(&bs, 1);

			/* Output match length */
			blz_putgamma(&bs, len - 2);

			/* Output match offset */
			blz_putgamma(&bs, (off >> 8) + 2);
			*bs.dst++ = off & 0x00FF;

			bs.src += len;
			src_avail -= len;
		}
		else {
			/* Output literal tag */
			blz_putbit(&bs, 0);

			/* Copy literal */
			*bs.dst++ = *bs.src++;
			src_avail--;
		}
	}

	/* Output any remaining literals */
	while (src_avail > 0) {
		/* Output literal tag */
		blz_putbit(&bs, 0);

		/* Copy literal */
		*bs.dst++ = *bs.src++;
		src_avail--;
	}

	/* Shift last tag into position and store */
	bs.tag <<= bs.bits_left;
	bs.tagpos[0] = bs.tag & 0x00FF;
	bs.tagpos[1] = (bs.tag >> 8) & 0x00FF;

	/* Return compressed size */
	return (unsigned long) (bs.dst - (unsigned char *) dst);
}
