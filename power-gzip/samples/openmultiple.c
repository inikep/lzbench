#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <endian.h>
#include <pthread.h>
#include <signal.h>
#include <dirent.h>

#include <sys/stat.h>

FILE *nx_gzip_log = NULL;

int main()
{
    nx_gzip_log = fopen("/tmp/nx.log", "a+");
    if (NULL == nx_gzip_log) {
	perror("cannot open\n");
	return -1;
    }

    /* read the definition of the sticky bit S_ISVTX; I don't
     * understand how it's used but I put it here anyway */
    if ( chmod("/tmp/nx.log", (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH | S_ISVTX)) ) {
	perror("cannot chmod but will continue\n");
    }
    
    fprintf(nx_gzip_log, "this is a test, pid=%d uid=%d\n", getpid(), getuid());
    fflush(nx_gzip_log);
    
    sleep(100);

    if ( fclose(nx_gzip_log) ) {
	perror("file close error\n");
	return -1;
    }

    return 0;
}
