#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include "../src/tables.h"

void main()
{
	int mlen;
	int word;
	int code;
	int bits1;
	int bits2;

	for (mlen = 0; mlen <= 258; mlen++) {
		word = len_sym[mlen];
		code = (word & 31) + 257;
		if (mlen >= 3)
			code = fixed_huff[code];
		else
			code = 0;

		bits1 = code & 15;
		code >>= 4;
		word >>= 5; bits2 = word & 7;

		if (bits2) {
			code |= (word >> 3) << bits1;
			bits1 += bits2;
		}

		printf(" 0x%06x, ", code + (bits1 << 16));
		if ((mlen & 3) == 3 || mlen == 258)
			printf("  /* %d-%d */\n", mlen - 3, mlen);
	}
	exit(0);
}
