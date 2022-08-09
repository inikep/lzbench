/*
 * P9 gunzip sample code for demonstrating the P9 NX hardware
 * interface.  Not intended for productive uses or for performance or
 * compression ratio measurements.  Note also that /dev/crypto/gzip,
 * VAS and skiboot support are required (version numbers TBD)
 *
 * Changelog:
 *   2018-04-02 Initial version
 *
 *
 * Copyright (C) IBM Corporation, 2011-2017
 *
 * Licenses for GPLv2 and Apache v2.0:
 *
 * GPLv2:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Apache v2.0:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Bulent Abali <abali@us.ibm.com>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <endian.h>
#include <bits/endian.h>
#include <sys/ioctl.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include "nxu.h"
#include "nx.h"
#include "nx_dbg.h"

#define ASSERT(X) assert(X)

struct _nx_time_dbg {
	uint64_t freq;
	uint64_t sub1, sub2, subc;
	uint64_t touch1, touch2;
	uint64_t faultc, targetlenc, datalenc;
} td;

#define NX_MIN(X,Y) (((X)<(Y))?(X):(Y))
#define NX_MAX(X,Y) (((X)>(Y))?(X):(Y))

#define mb()     asm volatile("sync" ::: "memory")

const int fifo_in_len = 1<<24;
const int fifo_out_len = 1<<24;	
const int page_sz = 1<<16;
const int line_sz = 1<<7;
const int window_max = 1<<15;
const int retry_max = 50;

extern void *nx_fault_storage_address;
extern void *nx_function_begin(int function, int pri);
extern int nx_function_end(void *handle);

/*
  Fault in pages prior to NX job submission.  wr=1 may be required to
  touch writeable pages. System zero pages do not fault-in the page as
  intended.  Typically set wr=1 for NX target pages and set wr=0 for
  NX source pages.
*/
static int nx_touch_pages(void *buf, long buf_len, long page_len, int wr)
{
	char *begin = buf;
	char *end = (char *)buf + buf_len - 1;
	volatile char t;

	ASSERT(buf_len >= 0 && !!buf);

	NXPRT( fprintf(stderr, "touch %p %p len 0x%lx wr=%d\n", buf, buf + buf_len, buf_len, wr) );
	
	if (buf_len <= 0 || buf == NULL)
		return -1;
	
	do {
		t = *begin;
		if (wr) *begin = t;
		begin = begin + page_len;
	} while (begin < end);

	/* when buf_sz is small or buf tail is in another page */
	t = *end;
	if (wr) *end = t;	

	return 0;
}

void sigsegv_handler(int sig, siginfo_t *info, void *ctx)
{
	fprintf(stderr, "%d: Got signal %d si_code %d, si_addr %p\n", getpid(),
	       sig, info->si_code, info->si_addr);

	nx_fault_storage_address = info->si_addr;
	/* exit(0); */
}

__attribute__ ((unused))
static void nx_print_dde(nx_dde_t *ddep, const char *msg)
{
	uint32_t indirect_count;
	uint32_t buf_len __attribute__ ((unused));
	uint64_t buf_addr __attribute__ ((unused));
	nx_dde_t *dde_list;
	int i;

	ASSERT(!!ddep);
	
	indirect_count = getpnn(ddep, dde_count);
	buf_len = getp32(ddep, ddebc);

	NXPRT( fprintf(stderr, "%s dde %p dde_count %d, ddebc 0x%x\n", msg, ddep, indirect_count, buf_len) );
	
	if (indirect_count == 0) {
		/* direct dde */
		buf_len = getp32(ddep, ddebc);
		buf_addr = getp64(ddep, ddead);
		NXPRT( fprintf(stderr, "  direct dde: ddebc 0x%x ddead %p %p\n", buf_len, (void *)buf_addr, (void *)buf_addr + buf_len) );	
		return;
	}

	/* indirect dde */
	if (indirect_count > MAX_DDE_COUNT) {
		NXPRT( fprintf(stderr, "  error MAX_DDE_COUNT\n") );
		return;
	}

	/* first address of the list */
	dde_list = (nx_dde_t *) getp64(ddep, ddead);

	for (i=0; i < indirect_count; i++) {
		buf_len = get32(dde_list[i], ddebc);
		buf_addr = get64(dde_list[i], ddead);
		NXPRT( fprintf(stderr, " indirect dde: ddebc 0x%x ddead %p %p\n", buf_len, (void *)buf_addr, (void *)buf_addr + buf_len) );	
	}
	return;
}

/* 
   Adds an (address, len) pair to the list of ddes (ddl) and updates
   the base dde.  ddl[0] is the only dde in a direct dde which
   contains a single (addr,len) pair.  For more pairs, ddl[0] becomes
   the indirect (base) dde that points to a list of direct ddes.
   See Section 6.4 of the NX-gzip user manual for DDE description.
   Addr=NULL, len=0 clears the ddl[0].  Returns the total number of
   bytes in ddl.  Caller is responsible for allocting the array of
   nx_dde_t *ddl.  If N addresses are required in the scatter-gather
   list, the ddl array must have N+1 entries minimum.
*/
static inline uint32_t nx_append_dde(nx_dde_t *ddl, void *addr, uint32_t len)
{
	uint32_t ddecnt;
	uint32_t bytes;

	if (addr == NULL && len == 0) {
		clearp_dde(ddl);
		return 0;
	}

	NXPRT( fprintf(stderr, "%d: nx_append_dde addr %p len %x\n", __LINE__, addr, len ) );
	
	/* number of ddes in the dde list ; == 0 when it is a direct dde */
	ddecnt = getpnn(ddl, dde_count); 
	bytes = getp32(ddl, ddebc);

	/* NXPRT( fprintf(stderr, "%d: get dde_count %d ddebc %d\n", __LINE__, ddecnt, bytes ) ); */
	
	if (ddecnt == 0 && bytes == 0) {
		/* first dde is unused; make it a direct dde */
		bytes = len;
		putp32(ddl, ddebc, bytes); 
		putp64(ddl, ddead, (uint64_t) addr);

		/* NXPRT( fprintf(stderr, "%d: put ddebc %d ddead %p\n", __LINE__, bytes, (void *)addr ) ); */
	}
	else if (ddecnt == 0) {
		/* converting direct to indirect dde */
		/* ddl[0] becomes head dde of ddl */
		/* copy direct to indirect first */	  
		ddl[1]= ddl[0]; 

		/* add the new dde next */
		clear_dde(ddl[2]);			
		put32(ddl[2], ddebc, len);
		put64(ddl[2], ddead, (uint64_t) addr);

		/* ddl head points to 2 direct ddes */
		ddecnt = 2;
		putpnn(ddl, dde_count, ddecnt);
		bytes = bytes + len;
		putp32(ddl, ddebc, bytes);
		/* pointer to the first direct dde */			
		putp64(ddl, ddead, (uint64_t) &ddl[1]); 
	}
	else {
		/* append a dde to an existing indirect ddl */
		++ddecnt;
		clear_dde(ddl[ddecnt]);
		put64(ddl[ddecnt], ddead, (uint64_t) addr);
		put32(ddl[ddecnt], ddebc, len);
		
		putpnn(ddl, dde_count, ddecnt);
		bytes = bytes + len;
		putp32(ddl, ddebc, bytes); /* byte sum of all dde */
	}
	return bytes;
}

/* 
   Touch specified number of pages represented in number bytes
   beginning from the first buffer in a dde list. 
   Do not touch the pages past buf_sz-th byte's page.
   
   Set buf_sz = 0 to touch all pages described by the ddep.
*/
static int nx_touch_pages_dde(nx_dde_t *ddep, long buf_sz, long page_sz, int wr)
{
	uint32_t indirect_count;
	uint32_t buf_len;
	long total;
	uint64_t buf_addr;
	nx_dde_t *dde_list;
	int i;

	ASSERT(!!ddep);
	
	indirect_count = getpnn(ddep, dde_count);

	NXPRT( fprintf(stderr, "nx_touch_pages_dde dde_count %d request len 0x%lx\n", indirect_count, buf_sz) ); 
	
	if (indirect_count == 0) {
		/* direct dde */
		buf_len = getp32(ddep, ddebc);
		buf_addr = getp64(ddep, ddead);

		NXPRT( fprintf(stderr, "touch direct ddebc 0x%x ddead %p\n", buf_len, (void *)buf_addr) ); 
		
		if (buf_sz == 0)
			nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);
		else 
			nx_touch_pages((void *)buf_addr, NX_MIN(buf_len, buf_sz), page_sz, wr);		

		return ERR_NX_OK;
	}

	/* indirect dde */
	if (indirect_count > MAX_DDE_COUNT)
		return ERR_NX_EXCESSIVE_DDE;

	/* first address of the list */
	dde_list = (nx_dde_t *) getp64(ddep, ddead);

	if( buf_sz == 0 )
		buf_sz = getp32(ddep, ddebc);
	
	total = 0;
	for (i=0; i < indirect_count; i++) {
		buf_len = get32(dde_list[i], ddebc);
		buf_addr = get64(dde_list[i], ddead);
		total += buf_len;

		NXPRT( fprintf(stderr, "touch loop len 0x%x ddead %p total 0x%lx\n", buf_len, (void *)buf_addr, total) ); 

		/* touching fewer pages than encoded in the ddebc */
		if ( total > buf_sz) {
			buf_len = NX_MIN(buf_len, total - buf_sz);
			nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);
			NXPRT( fprintf(stderr, "touch loop break len 0x%x ddead %p\n", buf_len, (void *)buf_addr) ); 			
			break;
		}
		nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);		
	}
	return ERR_NX_OK;
}

/* 
   Src and dst buffers are supplied in scatter gather lists. 
   NX function code and other parameters supplied in cmdp 
*/
static int nx_submit_job(nx_dde_t *src, nx_dde_t *dst, nx_gzip_crb_cpb_t *cmdp, void *handle)
{
	int cc;
	uint64_t csbaddr;

	memset( (void *)&cmdp->crb.csb, 0, sizeof(cmdp->crb.csb) );
	
	cmdp->crb.source_dde = *src;
	cmdp->crb.target_dde = *dst;

	/* status, output byte count in tpbc */
	csbaddr = ((uint64_t) &cmdp->crb.csb) & csb_address_mask;
	put64(cmdp->crb, csb_address, csbaddr);

	/* nx reports input bytes in spbc; cleared */	
	cmdp->cpb.out_spbc_comp_wrap = 0;
	cmdp->cpb.out_spbc_comp_with_count = 0;
	cmdp->cpb.out_spbc_decomp = 0;

	/* clear output */
	put32(cmdp->cpb, out_crc, INIT_CRC );
	put32(cmdp->cpb, out_adler, INIT_ADLER);
	
	NXPRT( nx_print_dde(src, "source") );
	NXPRT( nx_print_dde(dst, "target") );
	
	cc = nxu_run_job(cmdp, handle);

	if( !cc ) 
		cc = getnn( cmdp->crb.csb, csb_cc ); 	/* CC Table 6-8 */
	
	return cc;
}


/* fifo queue management */
#define fifo_used_bytes(used) (used)
#define fifo_free_bytes(used, len) ((len)-(used))
/* amount of free bytes in the first and last parts */
#define fifo_free_first_bytes(cur, used, len)  ((((cur)+(used))<=(len))? (len)-((cur)+(used)): 0)
#define fifo_free_last_bytes(cur, used, len)   ((((cur)+(used))<=(len))? (cur): (len)-(used))
/* amount of used bytes in the first and last parts */
#define fifo_used_first_bytes(cur, used, len)  ((((cur)+(used))<=(len))? (used) : (len)-(cur))
#define fifo_used_last_bytes(cur, used, len)   ((((cur)+(used))<=(len))? 0: ((used)+(cur))-(len))
/* first and last free parts start here */
#define fifo_free_first_offset(cur, used)      ((cur)+(used))
#define fifo_free_last_offset(cur, used, len)  fifo_used_last_bytes(cur, used, len)
/* first and last used parts start here */
#define fifo_used_first_offset(cur)            (cur)
#define fifo_used_last_offset(cur)             (0)	


int decompress_file(int argc, char **argv, void *devhandle)
{
#ifdef NX_MMAP
	int inpf = 0;
	int outf = 0;
#else
	FILE *inpf = NULL;
	FILE *outf = NULL;
#endif
	int c, expect, i, cc, rc = 0;
	char gzfname[1024];

	/* queuing, file ops, byte counting */
	char *fifo_in, *fifo_out;
	int used_in, cur_in, used_out, cur_out, read_sz, n;
	int first_free, last_free, first_used, last_used;
	int first_offset, last_offset;
	int write_sz, free_space, source_sz;
	int source_sz_estimate, target_sz_estimate;
	uint64_t last_comp_ratio = 1; /* 1000 max */
	uint64_t total_out = 0;
	int is_final, is_eof;
	
	/* nx hardware */
	int sfbt, subc, spbc, tpbc, nx_ce, fc, resuming = 0;
	int history_len=0;
	nx_gzip_crb_cpb_t cmd, *cmdp;
        nx_dde_t *ddl_in;        
        nx_dde_t dde_in[6]  __attribute__ ((aligned (128)));
        nx_dde_t *ddl_out;
        nx_dde_t dde_out[6] __attribute__ ((aligned (128)));
	int pgfault_retries;

	/* when using mmap'ed files */
	off_t input_file_offset;
#ifdef NX_MMAP
	off_t input_file_size;
	off_t fifo_in_mmap_offset;
	off_t fifo_out_mmap_offset;
	size_t fifo_in_mmap_size;
	size_t fifo_out_mmap_size;
#endif

	NX_CLK( memset(&td, 0, sizeof(td)) );
	NX_CLK( (td.freq = nx_get_freq())  );
	
	if (argc > 2) {
		fprintf(stderr, "usage: %s <fname> or stdin\n", argv[0]);
		fprintf(stderr, "    writes to stdout or <fname>.nx.gunzip\n");
		return -1;
	}

	if (argc == 1) {
#ifndef NX_MMAP		
		inpf = stdin;
		outf = stdout;
#else
		fprintf(stderr, "mmap needs a file name");
		return -1;
#endif		
	}
	else if (argc == 2) {
		char w[1024];
		char *wp;
#ifdef NX_MMAP
		struct stat statbuf;
		inpf = open(argv[1], O_RDONLY);
		if (inpf < 0) {
			perror(argv[1]);
			return -1;
		}
		if (fstat(inpf, &statbuf)) {
			perror("cannot stat file");
			return -1;
		}
		input_file_size = statbuf.st_size;
		input_file_offset = 0;

		fifo_in_mmap_offset = 0;
		fifo_in_mmap_size = fifo_in_len + page_sz;
		fifo_in = mmap(0, fifo_in_mmap_size, PROT_READ, MAP_PRIVATE, inpf, fifo_in_mmap_offset);
		if (fifo_in == MAP_FAILED) {
			perror("cannot mmap input file");
			return -1;
		}
		NXPRT( fprintf(stderr, "mmap fifo_in %p %p %lx\n", (void *)fifo_in, (void *)fifo_in + fifo_in_mmap_size, fifo_in_mmap_offset));
#else	/* !NX_MMAP */		
		inpf = fopen(argv[1], "r");
		if (inpf == NULL) {
			perror(argv[1]);
			return -1;
		}			
#endif

		/* make a new file name to write to; ignoring .gz stored name */
		wp = strrchr(argv[1], '/');
		if (NULL != wp) ++wp;
		else wp = argv[1];
		strcpy(w, wp);
		strcat(w, ".nx.gunzip");

#ifdef NX_MMAP
		outf = open(w, O_RDWR|O_CREAT|O_APPEND, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP );
		if (outf < 0) {
			perror(argv[1]);
			return -1;
		}
		total_out = 0;

		fifo_out_mmap_offset = 0;
		fifo_out_mmap_size = fifo_out_len + page_sz;		
		
		/* since output doesn't exist yet we pick an mmap size
		   that can be truncated at the end */
		if (ftruncate(outf, fifo_out_mmap_size)) {
			perror("cannot resize output");
			return -1;
		}

		/* and get a memory address for it */
		fifo_out = mmap(0, fifo_out_mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, outf, fifo_out_mmap_offset);
		if (fifo_out == MAP_FAILED) {
			perror("cannot mmap output file");
			return -1;
		}
		NXPRT( fprintf(stderr, "mmap fifo_out %p %p %lx\n", (void *)fifo_out, (void *)fifo_out + fifo_out_mmap_size, fifo_out_mmap_offset) );		
#else	/* !NX_MMAP */
		outf = fopen(w, "w");
		if (outf == NULL) {
			perror(w);
			return -1;
		}
#endif
	}

#ifdef NX_MMAP
#define GETINPC(X) ((input_file_offset < input_file_size) ? ((int)((char)fifo_in[input_file_offset++])) : EOF)
#else
#define GETINPC(X) fgetc(X)
#endif
	
	/* Decode the gzip header */
	c = GETINPC(inpf); expect = 0x1f; /* ID1 */
	if (c != expect) goto err1;

	c = GETINPC(inpf); expect = 0x8b; /* ID2 */
	if (c != expect) goto err1;

	c = GETINPC(inpf); expect = 0x08; /* CM */
	if (c != expect) goto err1;

	int flg = GETINPC(inpf); /* FLG */
	if (flg & 0b11100000 || flg & 0b100) goto err2;

	fprintf(stderr, "gzHeader FLG %x\n", flg);

	/* Read 6 bytes; ignoring the MTIME, XFL, OS fields in this
	   sample code */
	for (i=0; i<6; i++) {
		char tmp[10];
		if (EOF == (tmp[i] = GETINPC(inpf)))
			goto err3;
		fprintf(stderr, "%02x ", tmp[i]);
		if (i == 5) fprintf(stderr, "\n");
	}
	fprintf(stderr, "gzHeader MTIME, XFL, OS ignored\n");	

	/* FNAME */
	if (flg & 0b1000) {
		int k=0;
		do {
			if (EOF == (c = GETINPC(inpf)))
				goto err3;
			gzfname[k++] = c;
		} while (c);
		fprintf(stderr, "gzHeader FNAME: %s \n", gzfname);
	}

	/* FHCRC */
	if (flg & 0b10) {
		c = GETINPC(inpf); c = GETINPC(inpf);
		fprintf(stderr, "gzHeader FHCRC: ignored\n");
	}	

	used_in = cur_in = used_out = cur_out = 0;
	is_final = is_eof = 0;
#ifdef NX_MMAP
	cur_in = input_file_offset;
#endif	

#ifndef NX_MMAP	
	/* allocate one page larger to prevent page faults due to NX overfetching */
	/* either do this (char*)(uintptr_t)aligned_alloc or use
	   -std=c11 flag to make the int-to-pointer warning go away */
	assert( NULL != (fifo_in  = (char *)(uintptr_t)aligned_alloc(line_sz, fifo_in_len + page_sz) ) );
	assert( NULL != (fifo_out = (char *)(uintptr_t)aligned_alloc(line_sz, fifo_out_len + page_sz + line_sz) ) );
	fifo_out = fifo_out + line_sz; /* leave unused space due to history rounding rules */
	nx_touch_pages(fifo_out, fifo_out_len, page_sz, 1);		
#endif
	
	ddl_in  = &dde_in[0];
	ddl_out = &dde_out[0];	
	cmdp = &cmd;
	memset( &cmdp->crb, 0, sizeof(cmdp->crb) );
	
read_state:

	/* Read from .gz file */
	
	NXPRT( fprintf(stderr, "read_state:\n") );
	
	if (is_eof != 0) goto write_state;

#ifndef NX_MMAP	

	/* we read in to fifo_in in two steps: first: read in to from
	   cur_in to the end of the buffer.  last: if free space wrapped
	   around, read from fifo_in offset 0 to offset cur_in  */

	/* reset fifo head to reduce unnecessary wrap arounds */
	cur_in = (used_in == 0) ? 0 : cur_in; 
	
	/* free space total is reduced by a gap */
	free_space = NX_MAX( 0, fifo_free_bytes(used_in, fifo_in_len) - line_sz);

	/* free space may wrap around as first and last */
	first_free = fifo_free_first_bytes(cur_in, used_in, fifo_in_len);
	last_free  = fifo_free_last_bytes(cur_in, used_in, fifo_in_len);

	/* start offsets of the free memory */
	first_offset = fifo_free_first_offset(cur_in, used_in);
	last_offset  = fifo_free_last_offset(cur_in, used_in, fifo_in_len);

	/* reduce read_sz because of the line_sz gap */
	read_sz = NX_MIN(free_space, first_free);
	n = 0;
	if (read_sz > 0) {
		/* read in to offset cur_in + used_in */
		n = fread(fifo_in + first_offset, 1, read_sz, inpf);
		used_in = used_in + n;
		free_space = free_space - n; 
		ASSERT(n <= read_sz);
		if (n != read_sz) {
			/* either EOF or error; exit the read loop */
			is_eof = 1;
			goto write_state;
		}
	}
	
	/* if free space wrapped around */
	if (last_free > 0) {
		/* reduce read_sz because of the line_sz gap */
		read_sz = NX_MIN(free_space, last_free); 
		n = 0;		
		if (read_sz > 0) {
			n = fread(fifo_in + last_offset, 1, read_sz, inpf);
			used_in = used_in + n;       /* increase used space */
			free_space = free_space - n; /* decrease free space */
			ASSERT(n <= read_sz);
			if (n != read_sz) {
				/* either EOF or error; exit the read loop */
				is_eof = 1;
				goto write_state;
			}			
		}
	}

	/* At this point we have used_in bytes in fifo_in with the
	   data head starting at cur_in and possibly wrapping
	   around */

#else /* NX_MMAP */

	if (input_file_size == input_file_offset) {
		is_eof = 1;
		goto write_state;
	}
	
	cur_in = input_file_offset - fifo_in_mmap_offset;
	/* valid bytes from cur_in to the end of the mmap window */
	used_in = NX_MIN(fifo_in_len - cur_in, input_file_size - input_file_offset);
	ASSERT( fifo_in_len > 2*page_sz );
	if (cur_in > (fifo_in_len - 2*page_sz) ) {
		/* when near the tail of the mmap'ed region move the mmap window */
		if (munmap(fifo_in, fifo_in_mmap_size)) {
			perror("munmap");
			return -1;
		}
		NXPRT( fprintf(stderr, "munmap %p %p\n", (void *)fifo_in, (void *)fifo_in + fifo_in_mmap_size) ); 

		/* round down to page boundary */
		fifo_in_mmap_offset = (input_file_offset / page_sz) * page_sz;
		fifo_in_mmap_size = fifo_in_len + page_sz;

		fifo_in = mmap(0, fifo_in_mmap_size, PROT_READ, MAP_PRIVATE, inpf, fifo_in_mmap_offset);
		if (fifo_in == MAP_FAILED) {
			perror("cannot mmap input file");
			return -1;
		}
		NXPRT( fprintf(stderr, "mmap fifo_in %p %p %lx\n", (void *)fifo_in, (void *)fifo_in + fifo_in_mmap_size, fifo_in_mmap_offset) );	

		cur_in = input_file_offset - fifo_in_mmap_offset;
		used_in = NX_MIN(fifo_in_len - cur_in, input_file_size - input_file_offset);
	}

#endif /* NX_MMAP */
	

write_state:

	/* write decompressed data to output file */
	
	NXPRT( fprintf(stderr, "write_state:\n") );
	
	if (used_out == 0) goto decomp_state;

#ifndef NX_MMAP	
	/* If fifo_out has data waiting, write it out to the file to
	   make free target space for the accelerator used bytes in
	   the first and last parts of fifo_out */

	first_used = fifo_used_first_bytes(cur_out, used_out, fifo_out_len);
	last_used  = fifo_used_last_bytes(cur_out, used_out, fifo_out_len);

	write_sz = first_used;

	n = 0;
	if (write_sz > 0) {
		n = fwrite(fifo_out + cur_out, 1, write_sz, outf);
		used_out = used_out - n; 
		cur_out = (cur_out + n) % fifo_out_len; /* move head of the fifo */
		ASSERT(n <= write_sz);
		if (n != write_sz) {
			fprintf(stderr, "error: write\n");
			rc = -1;
			goto err5;
		}
	}
	
	if (last_used > 0) { /* if more data available in the last part */
		write_sz = last_used; /* keep it here for later */
		n = 0;		
		if (write_sz > 0) {
			n = fwrite(fifo_out, 1, write_sz, outf);
			used_out = used_out - n; 
			cur_out = (cur_out + n) % fifo_out_len;		
			ASSERT(n <= write_sz);
			if (n != write_sz) {
				fprintf(stderr, "error: write\n");
				rc = -1;
				goto err5;				
			}
		}
	}
	
	/* reset the fifo tail to reduce unnecessary wrap arounds
	   cur_out = (used_out == 0) ? 0 : cur_out; */

#else  /* NX_MMAP */

	cur_out = (int)(total_out - fifo_out_mmap_offset);	
	/* valid bytes from beginning of the mmap window to cur_out */
	used_out = 0;

	ASSERT( fifo_out_len > 2*page_sz );
	if (cur_out > (fifo_out_len - 2*page_sz)) {
		/* when near the tail of the mmap'ed region move the mmap window */
		if (munmap(fifo_out, fifo_out_mmap_size)) {
			perror("munmap");
			return -1;
		}
		NXPRT( fprintf(stderr, "munmap %p %p\n", (void *)fifo_out, (void *)fifo_out + fifo_out_mmap_size) ); 
		/* round down to page boundary; keep a page from behind for the LZ history */
		if (total_out > page_sz)
			fifo_out_mmap_offset = ( ((off_t)total_out - page_sz) / page_sz) * page_sz;
		else
			fifo_out_mmap_offset = 0;

		fifo_out_mmap_size = fifo_out_len + page_sz;

		/* resize output file */
		if (ftruncate(outf, fifo_out_mmap_offset + fifo_out_mmap_size)) {
			perror("cannot resize output");
			return -1;
		}

		fifo_out = mmap(0, fifo_out_mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, outf, fifo_out_mmap_offset);
		if (fifo_out == MAP_FAILED) {
			perror("cannot mmap input file");
			return -1;
		}
		NXPRT( fprintf(stderr,"mmap fifo_out %p %p %lx\n", (void *)fifo_out, (void *)fifo_out + fifo_out_mmap_size,fifo_out_mmap_offset));

		cur_out = (int)(total_out - fifo_out_mmap_offset);
		used_out = 0; 
	}
	
#endif /* NX_MMAP */	

decomp_state:

	/* NX decompresses input data */
	
	NXPRT( fprintf(stderr, "decomp_state:\n") );
	
	if (is_final) goto finish_state;
	
	/* address/len lists */
	clearp_dde(ddl_in);
	clearp_dde(ddl_out);	
	
	/* FC, CRC, HistLen, Table 6-6 */
	if (resuming) {
		/* Resuming a partially decompressed input.
		   The key to resume is supplying the 32KB
		   dictionary (history) to NX, which is basically
		   the last 32KB of output produced. 
		*/
		fc = GZIP_FC_DECOMPRESS_RESUME; 

		cmdp->cpb.in_crc   = cmdp->cpb.out_crc;
		cmdp->cpb.in_adler = cmdp->cpb.out_adler;

		/* Round up the history size to quadword. Section 2.10 */
		history_len = (history_len + 15) / 16;
		putnn(cmdp->cpb, in_histlen, history_len);
		history_len = history_len * 16; /* bytes */

		if (history_len > 0) {
			/* Chain in the history buffer to the DDE list */
			if ( cur_out >= history_len ) {
				nx_append_dde(ddl_in, fifo_out + (cur_out - history_len),
					      history_len);
			}
			else {
				nx_append_dde(ddl_in, fifo_out + ((fifo_out_len + cur_out) - history_len),
					      history_len - cur_out);
#ifndef NX_MMAP			
				/* Up to 32KB history wraps around fifo_out */
				nx_append_dde(ddl_in, fifo_out, cur_out);
#endif				
			}

		}
	}
	else {
		/* first decompress job */
		fc = GZIP_FC_DECOMPRESS; 

		history_len = 0;
		/* writing 0 clears out subc as well */
		cmdp->cpb.in_histlen = 0;
		total_out = 0;
		
		put32(cmdp->cpb, in_crc, INIT_CRC );
		put32(cmdp->cpb, in_adler, INIT_ADLER);
		put32(cmdp->cpb, out_crc, INIT_CRC );
		put32(cmdp->cpb, out_adler, INIT_ADLER);

		/* Assuming 10% compression ratio initially; I use the
		   most recently measured compression ratio as a
		   heuristic to estimate the input and output
		   sizes. If we give too much input, the target buffer
		   overflows and NX cycles are wasted, and then we
		   must retry with smaller input size. 1000 is 100%  */
		last_comp_ratio = 100UL;
	}
	cmdp->crb.gzip_fc = 0;   
	putnn(cmdp->crb, gzip_fc, fc);

	/*
	 * NX source buffers
	 */
	first_used = fifo_used_first_bytes(cur_in, used_in, fifo_in_len);
#ifndef NX_MMAP	
	last_used = fifo_used_last_bytes(cur_in, used_in, fifo_in_len);
#else
	last_used = 0;
#endif
	
	if (first_used > 0)
		nx_append_dde(ddl_in, fifo_in + cur_in, first_used);
		
	if (last_used > 0)
		nx_append_dde(ddl_in, fifo_in, last_used);

	/*
	 * NX target buffers
	 */
	first_free = fifo_free_first_bytes(cur_out, used_out, fifo_out_len);
#ifndef NX_MMAP	
	last_free = fifo_free_last_bytes(cur_out, used_out, fifo_out_len);
#else
	last_free = 0;
#endif

#ifndef NX_MMAP	
	/* reduce output free space amount not to overwrite the history */
	int target_max = NX_MAX(0, fifo_free_bytes(used_out, fifo_out_len) - (1<<16));
#else
	int target_max = first_free; /* no wrap-around in the mmap case */
#endif
	NXPRT( fprintf(stderr, "target_max %d (0x%x)\n", target_max, target_max) );
	
	first_free = NX_MIN(target_max, first_free);
	if (first_free > 0) {
		first_offset = fifo_free_first_offset(cur_out, used_out);		
		nx_append_dde(ddl_out, fifo_out + first_offset, first_free);
	}

	if (last_free > 0) {
		last_free = NX_MIN(target_max - first_free, last_free); 
		if (last_free > 0) {
			last_offset = fifo_free_last_offset(cur_out, used_out, fifo_out_len);
			nx_append_dde(ddl_out, fifo_out + last_offset, last_free);
		}
	}

	/* Target buffer size is used to limit the source data size
	   based on previous measurements of compression ratio. */

	/* source_sz includes history */
	source_sz = getp32(ddl_in, ddebc);
	ASSERT( source_sz > history_len );
	source_sz = source_sz - history_len;

	/* Estimating how much source is needed to 3/4 fill a
	   target_max size target buffer. If we overshoot, then NX
	   must repeat the job with smaller input and we waste
	   bandwidth. If we undershoot then we use more NX calls than
	   necessary. */

	source_sz_estimate = ((uint64_t)target_max * last_comp_ratio * 3UL)/4000;

	if ( source_sz_estimate < source_sz ) {
		/* target might be small, therefore limiting the
		   source data */
		source_sz = source_sz_estimate;
		target_sz_estimate = target_max;
	}
	else {
		/* Source file might be small, therefore limiting target
		   touch pages to a smaller value to save processor cycles.
		*/
		target_sz_estimate = ((uint64_t)source_sz * 1000UL) / (last_comp_ratio + 1);
		target_sz_estimate = NX_MIN( 2 * target_sz_estimate, target_max );
	}

	source_sz = source_sz + history_len;

	/* Some NX condition codes require submitting the NX job again */
	/* Kernel doesn't handle NX page faults. Expects user code to
	   touch pages */
	pgfault_retries = retry_max;

#ifndef REPCNT	
#define REPCNT 1L
#endif
#ifdef REPEATING
	/* this is for bandwidth measurement of multiple jobs only */
	int repeat_count = 0;
#endif
	
restart_nx:

 	putp32(ddl_in, ddebc, source_sz);  

	NX_CLK( (td.touch1 = nx_get_time()) );	
	
	/* fault in pages */
	nx_touch_pages_dde(ddl_in, 0, page_sz, 0);
	nx_touch_pages_dde(ddl_out, target_sz_estimate, page_sz, 1);

	NX_CLK( (td.touch2 += (nx_get_time() - td.touch1)) );	

	NX_CLK( (td.sub1 = nx_get_time()) );
	NX_CLK( (td.subc += 1) );
	
	/* send job to NX */
	cc = nx_submit_job(ddl_in, ddl_out, cmdp, devhandle);

	NX_CLK( (td.sub2 += (nx_get_time() - td.sub1)) );	
	
	switch (cc) {

	case ERR_NX_AT_FAULT:

		/* We touched the pages ahead of time. In the most common case we shouldn't
		   be here. But may be some pages were paged out. Kernel should have 
		   placed the faulting address to fsaddr */
		NXPRT( fprintf(stderr, "ERR_NX_AT_FAULT %p\n", (void *)cmdp->crb.csb.fsaddr) );
		NX_CLK( (td.faultc += 1) );

		/* Touch 1 byte, read-only  */
		nx_touch_pages( (void *)cmdp->crb.csb.fsaddr, 1, page_sz, 0);

		if (pgfault_retries == retry_max) {
			/* try once with exact number of pages */
			--pgfault_retries;
			goto restart_nx;
		}
		else if (pgfault_retries > 0) {
			/* if still faulting try fewer input pages
			 * assuming memory outage */
			if (source_sz > page_sz)
				source_sz = NX_MAX(source_sz / 2, page_sz);
			--pgfault_retries;
			goto restart_nx;
		}
		else {
			/* TODO what to do when page faults are too many?
			   Kernel MM would have killed the process. */
			fprintf(stderr, "cannot make progress; too many page fault retries cc= %d\n", cc);
			rc = -1;
			goto err5;
		}

	case ERR_NX_DATA_LENGTH:

		NXPRT( fprintf(stderr, "ERR_NX_DATA_LENGTH; not an error usually; stream may have trailing data\n") );
		NX_CLK( (td.datalenc += 1) );
		
		/* Not an error in the most common case; it just says 
		   there is trailing data that we must examine */

		/* CC=3 CE(1)=0 CE(0)=1 indicates partial completion
		   Fig.6-7 and Table 6-8 */
		nx_ce = get_csb_ce_ms3b(cmdp->crb.csb);

		if (!csb_ce_termination(nx_ce) &&
		    csb_ce_partial_completion(nx_ce)) {
			/* check CPB for more information
			   spbc and tpbc are valid */
			sfbt = getnn(cmdp->cpb, out_sfbt); /* Table 6-4 */
			subc = getnn(cmdp->cpb, out_subc); /* Table 6-4 */
			spbc = get32(cmdp->cpb, out_spbc_decomp);
			tpbc = get32(cmdp->crb.csb, tpbc);
			ASSERT(target_max >= tpbc);
#ifndef REPEATING			
			goto ok_cc3; /* not an error */
#else
			if (++repeat_count < REPCNT) goto restart_nx;
			else goto ok_cc3;
#endif
		}
		else {
			/* History length error when CE(1)=1 CE(0)=0. 
			   We have a bug */
			rc = -1;
			fprintf(stderr, "history length error cc= %d\n", cc);
			goto err5;
		}
		
	case ERR_NX_TARGET_SPACE:

		NX_CLK( (td.targetlenc += 1) );		
		/* Target buffer not large enough; retry smaller input
		   data; give at least 1 byte. SPBC/TPBC are not valid */
		ASSERT( source_sz > history_len );
		source_sz = ((source_sz - history_len + 2) / 2) + history_len;
		NXPRT( fprintf(stderr, "ERR_NX_TARGET_SPACE; retry with smaller input data src %d hist %d\n", source_sz, history_len) );
		goto restart_nx;

	case ERR_NX_OK:

		/* This should not happen for gzip formatted data;
		 * we need trailing crc and isize */
		fprintf(stderr, "ERR_NX_OK\n");
		spbc = get32(cmdp->cpb, out_spbc_decomp);
		tpbc = get32(cmdp->crb.csb, tpbc);
		ASSERT(target_max >= tpbc);			
		ASSERT(spbc >= history_len);
		source_sz = spbc - history_len;		
		goto offsets_state;

	default:
		fprintf(stderr, "error: cc= %d\n", cc);
		rc = -1;
		goto err5;
	}

ok_cc3:

	NXPRT( fprintf(stderr, "cc3: sfbt: %x\n", sfbt) );

	ASSERT(spbc > history_len);
	source_sz = spbc - history_len;

	/* Table 6-4: Source Final Block Type (SFBT) describes the
	   last processed deflate block and clues the software how to
	   resume the next job.  SUBC indicates how many input bits NX
	   consumed but did not process.  SPBC indicates how many
	   bytes of source were given to the accelerator including
	   history bytes.
	*/

	switch (sfbt) { 
		int dhtlen;
		
	case 0b0000: /* Deflate final EOB received */

		/* Calculating the checksum start position. */

		source_sz = source_sz - subc / 8;
		is_final = 1;
		break;

		/* Resume decompression cases are below. Basically
		   indicates where NX has suspended and how to resume
		   the input stream */
		
	case 0b1000: /* Within a literal block; use rembytecount */
	case 0b1001: /* Within a literal block; use rembytecount; bfinal=1 */

		/* Supply the partially processed source byte again */
		source_sz = source_sz - ((subc + 7) / 8);

		/* SUBC LS 3bits: number of bits in the first source byte need to be processed. */
		/* 000 means all 8 bits;  Table 6-3 */
		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen  */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);
		putnn(cmdp->cpb, in_rembytecnt, getnn( cmdp->cpb, out_rembytecnt));
		break;
		
	case 0b1010: /* Within a FH block; */
	case 0b1011: /* Within a FH block; bfinal=1 */

		source_sz = source_sz - ((subc + 7) / 8);

		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;		
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);		
		break;
		
	case 0b1100: /* Within a DH block; */
	case 0b1101: /* Within a DH block; bfinal=1 */

		source_sz = source_sz - ((subc + 7) / 8);		

		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;				
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);
		
		dhtlen = getnn(cmdp->cpb, out_dhtlen);
		putnn(cmdp->cpb, in_dhtlen, dhtlen);
		ASSERT(dhtlen >= 42);
		
		/* Round up to a qword */
		dhtlen = (dhtlen + 127) / 128;

		/* Clear any unused bits in the last qword */
		/* cmdp->cpb.in_dht[dhtlen-1].dword[0] = 0; */
		/* cmdp->cpb.in_dht[dhtlen-1].dword[1] = 0; */

		while (dhtlen > 0) { /* Copy dht from cpb.out to cpb.in */
			--dhtlen;
			cmdp->cpb.in_dht[dhtlen] = cmdp->cpb.out_dht[dhtlen];
		}
		break;		
		
	case 0b1110: /* Within a block header; bfinal=0; */
		     /* Also given if source data exactly ends (SUBC=0) with EOB code */
		     /* with BFINAL=0. Means the next byte will contain a block header. */
	case 0b1111: /* within a block header with BFINAL=1. */

		source_sz = source_sz - ((subc + 7) / 8);
		
		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);		
	}

offsets_state:	

	/* Adjust the source and target buffer offsets and lengths  */

	NXPRT( fprintf(stderr, "offsets_state:\n") );

	/* delete input data from fifo_in */
	used_in = used_in - source_sz;
	cur_in = (cur_in + source_sz) % fifo_in_len;
	input_file_offset = input_file_offset + source_sz;

	/* add output data to fifo_out */
	used_out = used_out + tpbc;

	ASSERT(used_out <= fifo_out_len);

	total_out = total_out + tpbc;
	
	/* Deflate history is 32KB max. No need to supply more
	   than 32KB on a resume */
	history_len = (total_out > window_max) ? window_max : total_out;

	/* To estimate expected expansion in the next NX job; 500 means 50%.
	   Deflate best case is around 1 to 1000 */
	last_comp_ratio = (1000UL * ((uint64_t)source_sz + 1)) / ((uint64_t)tpbc + 1);
	last_comp_ratio = NX_MAX( NX_MIN(1000UL, last_comp_ratio), 1 );
	NXPRT( fprintf(stderr, "comp_ratio %ld source_sz %d spbc %d tpbc %d\n", last_comp_ratio, source_sz, spbc, tpbc ) );
	
	resuming = 1;

finish_state:	

	NXPRT( fprintf(stderr, "finish_state:\n") );

	if (is_final) {
		if (used_out) goto write_state; /* more data to write out */
		else if(used_in < 8) {
			/* need at least 8 more bytes containing gzip crc and isize */
			rc = -1;
			goto err4;
		}
		else {
			/* compare checksums and exit */
			int i;
			char tail[8];
			uint32_t cksum, isize;
			for(i=0; i<8; i++) tail[i] = fifo_in[(cur_in + i) % fifo_in_len];
			fprintf(stderr, "computed checksum %08x isize %08x\n", cmdp->cpb.out_crc, (uint32_t)(total_out % (1ULL<<32)));
			cksum = (tail[0] | tail[1]<<8 | tail[2]<<16 | tail[3]<<24);
			isize = (tail[4] | tail[5]<<8 | tail[6]<<16 | tail[7]<<24);
			fprintf(stderr, "stored   checksum %08x isize %08x\n", cksum, isize);

			NX_CLK( fprintf(stderr, "DECOMP %s ", argv[1]) );			
			NX_CLK( fprintf(stderr, "obytes %ld ", total_out*REPCNT) );
			NX_CLK( fprintf(stderr, "freq   %ld ticks/sec ", td.freq)    );	
			NX_CLK( fprintf(stderr, "submit %ld ticks %ld count ", td.sub2, td.subc) );
			NX_CLK( fprintf(stderr, "touch  %ld ticks ", td.touch2)     );
			NX_CLK( fprintf(stderr, "%g byte/s ", ((double)total_out*REPCNT)/((double)td.sub2/(double)td.freq)) );
			NX_CLK( fprintf(stderr, "fault %ld target %ld datalen %ld\n", td.faultc, td.targetlenc, td.datalenc) );

			if (cksum == cmdp->cpb.out_crc && isize == (uint32_t)(total_out % (1ULL<<32))) {
				rc = 0;	goto ok1;
			}
			else {
				rc = -1; goto err4;
			}
		}
	}
	else goto read_state;
	
	return -1;

err1:
	fprintf(stderr, "error: not a gzip file, expect %x, read %x\n", expect, c);
	return -1;

err2:
	fprintf(stderr, "error: the FLG byte is wrong or not handled by this code sample\n");
	return -1;

err3:
	fprintf(stderr, "error: gzip header\n");
	return -1;

err4:
	fprintf(stderr, "error: checksum\n");

err5:
ok1:
	fprintf(stderr, "decomp is complete: fclose\n");
#ifndef NX_MMAP	
	fclose(outf);
#else
	NXPRT( fprintf(stderr, "ftruncate %ld\n", total_out) );
	if (ftruncate(outf, (off_t)total_out)) {
		perror("cannot resize file");
		rc = -1;
	}
	close(outf);
#endif
	return rc;
}


int main(int argc, char **argv)
{
    int rc;
    struct sigaction act;
    void *handle;

    act.sa_handler = 0;
    act.sa_sigaction = sigsegv_handler;
    act.sa_flags = SA_SIGINFO;
    act.sa_restorer = 0;
    sigemptyset(&act.sa_mask);
    sigaction(SIGSEGV, &act, NULL);

    handle = nx_function_begin( NX_FUNC_COMP_GZIP, 0);
    if (!handle) {
	fprintf( stderr, "Unable to init NX, errno %d\n", errno);
	exit(-1);
    }

    rc = decompress_file(argc, argv, handle);

    nx_function_end(handle);
    
    return rc;
}



