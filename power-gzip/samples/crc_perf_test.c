#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <zlib.h>

typedef unsigned long ulong;

int main(int argc, char **argv)
{
	size_t bufsz = 1<<20; /* test up to 1MB */
	size_t readsz;
	char *buf;
	ulong crc;
	long i;

	assert(NULL != (buf = malloc(bufsz)));	

	readsz = fread(buf, 1, bufsz, stdin);

	for(i=0; i<100000; i++)
		crc = crc32(0, buf, readsz);
	printf("read %ld bytes %ld times, crc32 %08lx\n", readsz, i, crc );
		
	return 0;
}

