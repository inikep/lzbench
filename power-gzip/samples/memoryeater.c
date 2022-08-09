#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
	printf("usage %s <GB>\n", argv[0]);
	return -1;
    }
    long gb = atoi(argv[1]);
    gb = gb * (1L<<30);
    volatile char *p = (char *) malloc(gb);
    if (p == NULL) {
	printf("cannot malloc\n");
	return -1;
    }
    long i;
    while(1) {
	for(i = 0; i < gb; i = i+(1L<<16)) {
	    // make page faults
	    volatile char x = 1;
	    *(p+i) = x;
	}
    }
    return 0;
}

