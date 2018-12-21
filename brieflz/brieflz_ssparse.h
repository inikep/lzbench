//
// BriefLZ - small fast Lempel-Ziv
//
// Backwards dynamic programming parse
//
// Copyright (c) 2016-2018 Joergen Ibsen
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must
//      not claim that you wrote the original software. If you use this
//      software in a product, an acknowledgment in the product
//      documentation would be appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must
//      not be misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//      distribution.
//

#ifndef BRIEFLZ_SSPARSE_H_INCLUDED
#define BRIEFLZ_SSPARSE_H_INCLUDED

static unsigned long
blz_ssparse_workmem_size(unsigned long src_size)
{
	return (LOOKUP_SIZE < 2 * src_size ? 3 * src_size : src_size + LOOKUP_SIZE)
	     * sizeof(unsigned long);
}

// Backwards dynamic programming parse like Storer–Szymanski, but checking all
// possible matches.
//
// In BriefLZ, the length and offset of matches are encoded using a variable
// length code, this means for instance choosing a literal and a match close
// by might be better than one match further away. To get a bit optimal parse
// for BriefLZ, we consider all possible matches for a given position, not
// just the longest.
//
// So if max_depth and accept_len are ULONG_MAX, this version is bit-optimal
// (up to possibly the final tag). However, this is worst-case O(n^3) for long
// repeated patterns, which means in practice you usually need to limit the
// search.
//
static unsigned long
blz_pack_ssparse(const void *src, void *dst, unsigned long src_size, void *workmem,
                 const unsigned long max_depth, const unsigned long accept_len)
{
	struct blz_state bs;
	const unsigned char *const in = (const unsigned char *) src;
	const unsigned long last_match_pos = src_size > 4 ? src_size - 4 : 0;

	// Check for empty input
	if (src_size == 0) {
		return 0;
	}

	bs.next_out = (unsigned char *) dst;

	// First byte verbatim
	*bs.next_out++ = in[0];

	// Check for 1 byte input
	if (src_size == 1) {
		return 1;
	}

	// Initialize first tag
	bs.tag_out = bs.next_out;
	bs.next_out += 2;
	bs.tag = 0;
	bs.bits_left = 16;

	if (src_size < 4) {
		for (unsigned long i = 1; i < src_size; ++i) {
			// Output literal tag
			blz_putbit(&bs, 0);

			// Copy literal
			*bs.next_out++ = in[i];
		}
		goto finalize;
	}

	// With a bit of careful ordering we can fit in 3 * src_size words.
	//
	// The idea is that the lookup is only used in the first phase to
	// build the hash chains, so we overlap it with mpos and mlen.
	// Also, since we are using prev from right to left in phase two,
	// and that is the order we fill in cost, we can overlap these.
	//
	// One detail is that we actually use src_size + 1 elements of cost,
	// but we put mpos after it, where we do not need the first element.
	//
	unsigned long *const prev = (unsigned long *) workmem;
	unsigned long *const mpos = prev + src_size;
	unsigned long *const mlen = mpos + src_size;
	unsigned long *const cost = prev;
	unsigned long *const lookup = mpos;

	// Phase 1: Build hash chains
	const int bits = 2 * src_size < LOOKUP_SIZE ? BLZ_HASH_BITS : blz_log2(src_size);

	// Initialize lookup
	for (unsigned long i = 0; i < (1UL << bits); ++i) {
		lookup[i] = NO_MATCH_POS;
	}

	// Build hash chains in prev
	if (last_match_pos > 0) {
		for (unsigned long i = 0; i <= last_match_pos; ++i) {
			const unsigned long hash = blz_hash4_bits(&in[i], bits);
			prev[i] = lookup[hash];
			lookup[hash] = i;
		}
	}

	// Initialize last three positions as literals
	mlen[src_size - 3] = 1;
	mlen[src_size - 2] = 1;
	mlen[src_size - 1] = 1;

	cost[src_size - 3] = 27;
	cost[src_size - 2] = 18;
	cost[src_size - 1] = 9;
	cost[src_size] = 0;

	// Phase 2: Find lowest cost path from each position to end
	for (unsigned long cur = last_match_pos; cur > 0; --cur) {
		// Since we updated prev to the end in the first phase, we
		// do not need to hash, but can simply look up the previous
		// position directly.
		unsigned long pos = prev[cur];

		assert(pos == NO_MATCH_POS || pos < cur);

		// Start with a literal
		cost[cur] = cost[cur + 1] + 9;
		mlen[cur] = 1;

		unsigned long max_len = 3;

		const unsigned long len_limit = src_size - cur;
		unsigned long num_chain = max_depth;

		// Go through the chain of prev matches
		for (; pos != NO_MATCH_POS && num_chain--; pos = prev[pos]) {
			unsigned long len = 0;

			// If next byte matches, so this has a chance to be a longer match
			if (max_len < len_limit && in[pos + max_len] == in[cur + max_len]) {
				// Find match len
				while (len < len_limit && in[pos + len] == in[cur + len]) {
					++len;
				}
			}

			// Extend current match if possible
			//
			// Note that we are checking matches in order from the
			// closest and back. This means for a match further
			// away, the encoding of all lengths up to the current
			// max length will always be longer or equal, so we need
			// only consider the extension.
			if (len > max_len) {
				unsigned long min_cost = ULONG_MAX;
				unsigned long min_cost_len = 3;

				// Find lowest cost match length
				for (unsigned long i = max_len + 1; i <= len; ++i) {
					unsigned long match_cost = blz_match_cost(cur - pos - 1, i);
					assert(match_cost < ULONG_MAX - cost[cur + i]);
					unsigned long cost_here = match_cost + cost[cur + i];

					if (cost_here < min_cost) {
						min_cost = cost_here;
						min_cost_len = i;
					}
				}

				max_len = len;

				// Update cost if cheaper
				if (min_cost < cost[cur]) {
					cost[cur] = min_cost;
					mpos[cur] = pos;
					mlen[cur] = min_cost_len;
				}
			}

			if (len >= accept_len) {
				break;
			}
		}
	}

	mpos[0] = 0;
	mlen[0] = 1;

	// Phase 3: Output compressed data, following lowest cost path
	for (unsigned long i = 1; i < src_size; i += mlen[i]) {
		if (mlen[i] == 1) {
			// Output literal tag
			blz_putbit(&bs, 0);

			// Copy literal
			*bs.next_out++ = in[i];
		}
		else {
			const unsigned long offs = i - mpos[i] - 1;

			// Output match tag
			blz_putbit(&bs, 1);

			// Output match length
			blz_putgamma(&bs, mlen[i] - 2);

			// Output match offset
			blz_putgamma(&bs, (offs >> 8) + 2);
			*bs.next_out++ = offs & 0x00FF;
		}
	}

finalize:
	// Trailing one bit to delimit any literal tags
	blz_putbit(&bs, 1);

	// Shift last tag into position and store
	bs.tag <<= bs.bits_left;
	bs.tag_out[0] = bs.tag & 0x00FF;
	bs.tag_out[1] = (bs.tag >> 8) & 0x00FF;

	// Return compressed size
	return (unsigned long) (bs.next_out - (unsigned char *) dst);
}

#endif /* BRIEFLZ_SSPARSE_H_INCLUDED */
