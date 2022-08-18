#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* change this to the output file size you desire */
#ifndef LOGFILESZ
#define LOGFILESZ 20
#endif
#ifndef OUTFILESZ
#define OUTFILESZ (1<<LOGFILESZ)
#endif

typedef unsigned long ulong;

int main(int argc, char **argv)
{
	size_t bufsz = OUTFILESZ;
	size_t readsz;
	char *buf;
	size_t idx;
	int seed=0;

	if (argc == 5 && strcmp(argv[1], "-s") == 0 && strcmp(argv[3], "-b") == 0) {
		seed = atoi(argv[2]);
		bufsz = 1UL << atoi(argv[4]);
	}	
	if ((argc == 1) || (argc == 2 && strcmp(argv[1], "-h") == 0)) {
		fprintf(stderr,"randomly produces a data file using a seed number and any input file\n");
		fprintf(stderr,"usage: %s -s rngseed -b log2bufsize < seedfile > outputfile\n", argv[0]);
		return -1;
	}

	srand48(seed);

	/* randomize buffer size too; some power of 2; some arbitrary */
	bufsz = bufsz + (lrand48() % 2) * (lrand48() % (bufsz/10));

	fprintf(stderr, "seed %d bufsz %ld\n", seed, bufsz);
	
	assert(NULL != (buf = malloc(bufsz)));

	/* read up to half the buffer size */
	readsz = fread(buf, 1, bufsz/2, stdin);
	fprintf(stderr, "read bytes %ld\n", readsz);

	/* next free location */
	idx = readsz;
	
	ulong len_max = (lrand48() % 240UL) + 10;
	ulong dist_max = (lrand48() % (1UL<<16)) + 1;
	  
	while(idx < bufsz) {

		/* pick random point in the buffer and copy */
		ulong dist = lrand48() % ( (idx > dist_max) ? dist_max: idx );
		ulong len = (lrand48() % len_max) + 16;

		/* fprintf(stderr, "dist_max %ld len_max %ld dist %ld len %ld\n", dist_max, len_max, dist, len); */

		if (dist > idx) 
			dist = idx; /* out of bounds */

		/* copy */
		while (len-- > 0 && idx < bufsz) {
			buf[idx] = buf[idx-dist];
			++idx;
		}
	}

	fwrite(buf, 1, idx, stdout);

	return 0;
}

