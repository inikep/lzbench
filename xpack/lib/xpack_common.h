#ifndef LIB_XPACK_COMMON_H
#define LIB_XPACK_COMMON_H

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "common_defs.h"
#include "unaligned.h"
#include "xpack_constants.h"

#include "libxpack.h"

#ifdef ENABLE_PREPROCESSING
extern void preprocess(void *data, u32 size);
extern void postprocess(void *data, u32 size);
#endif

/*
 * Given the number of states, return the corresponding state generator, which
 * is the amount by which we will step through the states when assigning symbols
 * to states.  We require a value such that every state will be visited exactly
 * once after num_states steps.  Mathematically, we require a generator of the
 * cyclic group consisting of the set of integers {0...num_states - 1} and the
 * group operation of addition modulo num_states.  By a well-known theorem, the
 * generators are the set of integers relatively prime to num_states.  In this
 * case, since num_states is a power of 2, its prime factors are all 2's;
 * therefore, the generators are all numbers that do not have 2 as a prime
 * factor, i.e. all odd numbers.
 *
 * The number '1' is always a valid choice, but a poor one because it is
 * advantageous to distribute each symbol's states more evenly.  The value we
 * actually use that works well in practice is five-eighths the number of states
 * plus 3.  But use | instead of + to guarantee an odd number if num_states <=
 * 8.  Also, it is okay to use a value greater than num_states because we have
 * to mod with num_states after each addition anyway.
 *
 * Note: it is essential that the encoder and decoder always choose the same
 * generator as each other for a given num_states!  If you were to change this
 * formula, then you would change the on-disk compression format.
 */
static forceinline unsigned
get_state_generator(unsigned num_states)
{
	return (num_states >> 1) | (num_states >> 3) | 3;
}

/* Initialize the recent offsets queue. */
static forceinline void
init_recent_offsets(u32 recent_offsets[NUM_REPS])
{
	unsigned i;

	for (i = 0; i < NUM_REPS; i++)
		recent_offsets[i] = 1 + i;
}

#endif /* LIB_XPACK_COMMON_H */
