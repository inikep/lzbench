#ifndef GZIPSIMPLEAPI_H
#define GZIPSIMPLEAPI_H
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <syscall.h>
#include <pthread.h>
#include "nx.h"
#include "nxu.h"

#define OVERFLOW_BUFFER_SIZE 16
#define NX_MAX_DEVICES 2
#define GZIP_WRAPPER 0x01 //[GZIP header] [Deflate Stream] [CRC32] [ISIZE]
#define ZLIB_WRAPPER 0x02 //[ZLIB header] [Deflate Stream] [Adler32]
#define NO_WRAPPER 0x04   //[Deflate Stream]
#define NX_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

static long pagesize = 65536;
typedef struct p9_simple_handle_t {
	void *vas_handle; // device handle
	int chipId;
	int open_count;
} p9_simple_handle_t;

extern void *nx_function_begin(int function, int pri);
extern int nx_function_end(void *handle);

struct sigaction sigact;
void *nx_fault_storage_address;

void *nx_overflow_buffer;
p9_simple_handle_t *nx_devices[NX_MAX_DEVICES];

void sigsegv_handler(int sig, siginfo_t *info, void *ctx);

/*open the device*/
p9_simple_handle_t *p9open();

/*compress*/
int p9deflate(p9_simple_handle_t *handle, void *src, void *dst, int srclen,
	      int dstlen, char *fname, int flag);
/*decompress*/
int p9inflate(p9_simple_handle_t *handle, void *src, void *dst, int srclen,
	      int dstlen, int flag);
/*close the device*/
int p9close(p9_simple_handle_t *handle);

#endif
