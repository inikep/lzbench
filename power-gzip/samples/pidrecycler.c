#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    long i,j,scan;
    char *p;
    
    if (argc != 3) {
	printf("usage %s <num> <GB>\n", argv[0]);
	printf(" produce <num> processes with <GB> memory\n");
	return -1;
    }
    long gb = atoi(argv[2]);
    long num = atoi(argv[1]); 

    gb = gb * (1L<<30);
    
 rpt:

    p = (char *) malloc(gb);
    if (p == NULL) {
	printf("cannot malloc\n");
	return -1;
    }

    /* page faulter */

    scan = 1; /* n */
    while (scan-- > 0) {
	for (i = 0; i < gb; i = i+(1L<<16)) {
	    // make page faults
	    volatile char x = 1;
	    *(p+i) = x;
	}
    }

    num -= 1;
    if (num > 0) {
	pid_t pid = fork();
	if (pid == 0) { /* child continues; parent dies */
	    if (p != NULL)
		free(p);
	    goto rpt;
	}
	else if (pid < 0) {
	    perror("forke failed\n");
	    return pid;
	}
	else printf("%ld forked %d\n", num, pid);
    }
    fflush(stdout);

    return 0;
}

