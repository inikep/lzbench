/*
 * P9 gunzip sample code for demonstrating the P9 NX hardware
 * interface.  Not intended for productive uses or for performance or
 * compression ratio measurements. For simplicity of demonstration,
 * this sample code compresses in to fixed Huffman blocks only
 * (Deflate btype=1) and has very simple memory management.  Dynamic
 * Huffman blocks (Deflate btype=2) are more involved as detailed in
 * the user guide.  Note also that /dev/crypto/gzip, VAS and skiboot
 * support are required (version numbers TBD)
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
#include <stdarg.h>
#include "nxu.h"
#include "nx_dht.h"
#include "nx.h"
#include "nx_dbg.h"
#include <zlib.h>

typedef struct nxlite_handle_t {
	void *device_handle;
	void *dht_handle;
} nxlite_handle_t;

#define NX_MIN(X,Y) (((X)<(Y))?(X):(Y))
#define NX_MAX(X,Y) (((X)>(Y))?(X):(Y))

/* maximum of 5% of input and the nbyte value is used to sample symbols towards dht statistics */
#define NX_LZ_SAMPLE_PERCENT  5
#ifndef NX_LZ_SAMPLE_NBYTE
#define NX_LZ_SAMPLE_NBYTE    (1UL<<15)
#endif
#define NX_CHUNK_SZ  (1<<18)

extern void *nx_fault_storage_address;
extern void *nx_function_begin(int function, int pri);
extern int nx_function_end(void *handle);
static void sigsegv_handler(int sig, siginfo_t *info, void *ctx);


static int compress_dht_sample(char *src, uint32_t srclen, char *dst, uint32_t dstlen,
			       int with_count, nx_gzip_crb_cpb_t *cmdp, void *handle)
{
	int i,cc;
	uint32_t fc;

	assert(!!cmdp);

	/* memset(&cmdp->crb, 0, sizeof(cmdp->crb)); */ /* cc=21 error; revisit clearing below */
	put32(cmdp->crb, gzip_fc, 0);   /* clear */

	/* The reason we use a RESUME function code from get go is
	   because we can; resume is equivalent to a non-resume
	   function code when in_histlen=0 */
	if (with_count) 
		fc = GZIP_FC_COMPRESS_RESUME_DHT_COUNT;
	else 
		fc = GZIP_FC_COMPRESS_RESUME_DHT;

	putnn(cmdp->crb, gzip_fc, fc);
	/* resuming with no history; not optimal but good enough for the sample code */
	putnn(cmdp->cpb, in_histlen, 0);
	memset((void *)&cmdp->crb.csb, 0, sizeof(cmdp->crb.csb));
    
	/* Section 6.6 programming notes; spbc may be in two different places depending on FC */
	if (!with_count) 
		put32(cmdp->cpb, out_spbc_comp, 0);
	else 
		put32(cmdp->cpb, out_spbc_comp_with_count, 0);
    
	/* Figure 6-3 6-4; CSB location */
	put64(cmdp->crb, csb_address, 0);    
	put64(cmdp->crb, csb_address, (uint64_t) &cmdp->crb.csb & csb_address_mask);
    
	/* source direct dde */
	clear_dde(cmdp->crb.source_dde);    
	putnn(cmdp->crb.source_dde, dde_count, 0); 
	put32(cmdp->crb.source_dde, ddebc, srclen); 
	put64(cmdp->crb.source_dde, ddead, (uint64_t) src);

	/* target direct dde */
	clear_dde(cmdp->crb.target_dde);        
	putnn(cmdp->crb.target_dde, dde_count, 0);
	put32(cmdp->crb.target_dde, ddebc, dstlen);
	put64(cmdp->crb.target_dde, ddead, (uint64_t) dst);   

	/* fprintf(stderr, "in_dhtlen %x\n", getnn(cmdp->cpb, in_dhtlen) );
	   fprintf(stderr, "in_dht %02x %02x\n", cmdp->cpb.in_dht_char[0],cmdp->cpb.in_dht_char[16]); */

	/* submit the crb */
	nxu_run_job(cmdp, handle);

	/* poll for the csb.v bit; you should also consider expiration */        
	do {;} while (getnn(cmdp->crb.csb, csb_v) == 0);

	/* CC Table 6-8 */        
	cc = getnn(cmdp->crb.csb, csb_cc);

	return cc;
}

/* Prepares a blank no filename no timestamp gzip header and returns
   the number of bytes written to buf;
   https://tools.ietf.org/html/rfc1952 */
static int gzip_header_blank(char *buf)
{
	int i=0;
	buf[i++] = 0x1f; /* ID1 */
	buf[i++] = 0x8b; /* ID2 */
	buf[i++] = 0x08; /* CM  */
	buf[i++] = 0x00; /* FLG */
	buf[i++] = 0x00; /* MTIME */
	buf[i++] = 0x00; /* MTIME */
	buf[i++] = 0x00; /* MTIME */
	buf[i++] = 0x00; /* MTIME */            
	buf[i++] = 0x04; /* XFL 4=fastest */
	buf[i++] = 0x03; /* OS UNIX */
	return i;
}

/* Returns number of appended bytes */
static int append_sync_flush(char *buf, int tebc, int final)
{
	uint64_t flush;
	int shift = (tebc & 0x7);
	if (tebc > 0) {
		/* last byte is partially full */	
		buf = buf - 1; 
		*buf = *buf & (unsigned char)((1<<tebc)-1);
	}
	else *buf = 0;
	flush = ((0x1ULL & final) << shift) | *buf;
	shift = shift + 3; /* BFINAL and BTYPE written */
	shift = (shift <= 8) ? 8 : 16;
	flush |= (0xFFFF0000ULL) << shift; /* Zero length block */
	shift = shift + 32;
	while (shift > 0) {
		*buf++ = (unsigned char)(flush & 0xffULL);
		flush = flush >> 8;
		shift = shift - 8;
	}
	return(((tebc > 5) || (tebc == 0)) ? 5 : 4);
}

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

	assert(buf_len >= 0 && !!buf);

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


static void nx_print_dde(nx_dde_t *ddep, const char *msg)
{
	uint32_t indirect_count;
	uint32_t buf_len;
	uint64_t buf_addr;
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
	volatile char t;
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

		nx_touch_pages((void *)&(dde_list[i]), sizeof(nx_dde_t), page_sz, 0);
		
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
  Final deflate block bit. This call assumes the block 
  beginning is byte aligned.
*/
static void set_bfinal(void *buf, int bfinal)
{
	char *b = buf;
	if (bfinal)
		*b = *b | (unsigned char) 0x01;
	else
		*b = *b & (unsigned char) 0xfe;
}


/* Memory pointed to by nxhandle is also free'ed */
void nxlite_end(void *nxhandle)
{
	nxlite_handle_t *h;

	if (nxhandle == NULL)
		return;

	h = (nxlite_handle_t *) nxhandle;

	/* close the device */
	if (h->device_handle != NULL)
		nx_function_end(h->device_handle);

	/* if a dht struct was allocated, free it */
	if (h->dht_handle != NULL)
		dht_end(h->dht_handle);

	free(nxhandle);
}

/* Returns a handle which may be used on subsequent calls to
 * *compress() and *uncompress(). Caller must return the handle to
 * nxlite_end().  A NULL return value indicates an error.
 */
void *nxlite_begin()
{
	int rc;
	struct sigaction act;
	nxlite_handle_t *h;

	if (NULL == (h = malloc(sizeof(nxlite_handle_t)))) {
		fprintf(stderr, "error: cannot malloc\n");
		return NULL;
	}
	memset(h, 0, sizeof(nxlite_handle_t));
	
	/* open the device and save the device pointer in h */
	if (NULL == (h->device_handle = nx_function_begin(NX_FUNC_COMP_GZIP, 0))) {
		fprintf(stderr, "Unable to init NX, errno %d\n", errno);
		nxlite_end(h);
		return NULL;
	}

	/* One time init of the dht tables; */
	if (NULL == (h->dht_handle = dht_begin(NULL, NULL))) {
		fprintf(stderr, "Unable to init dht tables\n");
		nxlite_end(h);
		return NULL;
	}

	/* use signals for catching CSB page faults */
	act.sa_handler = 0;
	act.sa_sigaction = sigsegv_handler;
	act.sa_flags = SA_SIGINFO;
	act.sa_restorer = 0;
	sigemptyset(&act.sa_mask);
	sigaction(SIGSEGV, &act, NULL);
	
	return h;
}

static int nxlite_compress(char *dest, uint64_t *destLen,
			   char *source, uint64_t sourceLen,
			   void **handlep)
{
	char *inbuf, *outbuf, *srcbuf, *dstbuf;
	uint32_t srclen, dstlen;
	uint32_t flushlen, chunk;
	size_t inlen, outlen, dsttotlen, srctotlen;	
	uint32_t adler, crc, spbc, tpbc, tebc;
	int lzcounts=1; /* always collect lzcounts */
	int initial_pass;
	int cc,fc;
	int num_hdr_bytes;
	nx_gzip_crb_cpb_t nxcmd, *cmdp;
	uint32_t pagelen = 65536; /* should get page size with a syscall */
	int fault_tries = 50;
	nxlite_handle_t *myhandle=NULL;

	if (!source || sourceLen == 0 || !dest || *destLen == 0) {
		fprintf(stderr, "error: a buffer address or length is 0 (%s %d)\n", __FILE__, __LINE__);
		return Z_BUF_ERROR;
	}
	inbuf = source;
	inlen = sourceLen;
	outbuf = dest;
	outlen = *destLen;

	/* open device if not opened */
	if ( !*handlep) {
		myhandle = nxlite_begin();
		if (!myhandle)
			return Z_STREAM_ERROR;
		/* return to the caller for subsequent calls */
		*handlep = myhandle;
	}
	/* reuse the supplied handle */
	else myhandle = *handlep;
	
	/* compress piecemeal in small chunks */    
	chunk = NX_CHUNK_SZ;

	/* write the gzip header */    
	num_hdr_bytes = gzip_header_blank(outbuf); 
	dstbuf    = outbuf + num_hdr_bytes;
	outlen    = outlen - num_hdr_bytes;	
	dsttotlen = num_hdr_bytes;
	
	srcbuf    = inbuf;
	srctotlen = 0;

	/* prep the CRB */
	cmdp = &nxcmd;
	memset(&cmdp->crb, 0, sizeof(cmdp->crb));

	/* prep the CPB */
	/* memset(&cmdp->cpb.out_lzcount, 0, sizeof(uint32_t) * (LLSZ+DSZ) ); */
	put32(cmdp->cpb, in_crc, 0); /* initial gzip crc */

	/* Fill in with the default dht here; instead we could also do
	   fixed huffman with counts for sampling the LZcounts; fixed
	   huffman doesn't need a dht_lookup */
	dht_lookup(cmdp, dht_default_req, myhandle->dht_handle); 
	initial_pass = 1;

	fault_tries = 50;

	while (inlen > 0) {

	initial_pass_done:
		/* will submit a chunk size source per job */
		srclen = NX_MIN(chunk, inlen);
		/* supply large target in case data expands; 288
		   is for very small src plus the dht headroom */				
		dstlen = NX_MIN(2 * srclen + 288, outlen); 

		if (initial_pass == 1) {
			/* If requested a first pass to collect
			   lzcounts; first pass can be short; no need
			   to run the entire data through typically */
			/* If srclen is very large, use 5% of it. If
			   srclen is smaller than 32KB, then use
			   srclen itself as the sample */
			srclen = NX_MIN( NX_MAX(((uint64_t)srclen * NX_LZ_SAMPLE_PERCENT)/100, NX_LZ_SAMPLE_NBYTE), srclen);
			NXPRT( fprintf(stderr, "sample size %d\n", srclen) );
		}
		else {
			/* Here I will use the lzcounts collected from
			   the previous second pass to lookup a cached
			   or computed DHT; I don't need to sample the
			   data anymore; previous run's lzcount
			   is a good enough as an lzcount of this run */
			dht_lookup(cmdp, dht_search_req, myhandle->dht_handle); 
		}

		/* Page faults are handled by the user code */		

		/* Fault-in pages; an improved code wouldn't touch so
		   many pages but would try to estimate the
		   compression ratio and adjust both the src and dst
		   touch amounts */
		nx_touch_pages (cmdp, sizeof(*cmdp), pagelen, 0);
		nx_touch_pages (srcbuf, srclen, pagelen, 0);
		nx_touch_pages (dstbuf, dstlen, pagelen, 1);	    

		cc = compress_dht_sample(
			srcbuf, srclen,
			dstbuf, dstlen,
			lzcounts, cmdp,
			myhandle->device_handle);

		if (cc != ERR_NX_OK && cc != ERR_NX_TPBC_GT_SPBC && cc != ERR_NX_AT_FAULT && cc != ERR_NX_TARGET_SPACE) {
			fprintf(stderr, "nx error: cc= %d\n", cc);
			return Z_STREAM_ERROR; //exit(-1);
		}
		
		/* Page faults are handled by the user code */
		if (cc == ERR_NX_AT_FAULT) {
			volatile char touch = *(char *)cmdp->crb.csb.fsaddr;
			NXPRT( fprintf(stderr, "page fault: cc= %d, try= %d, fsa= %08llx\n", cc, fault_tries, (long long unsigned) cmdp->crb.csb.fsaddr) );
			fault_tries --;
			if (fault_tries > 0) {
				continue;
			}
			else {
				fprintf(stderr, "error: cannot progress; too many faults\n");
				return Z_STREAM_ERROR; //exit(-1);				
			}			    
		}
		else if (cc == ERR_NX_TARGET_SPACE) {
			fprintf(stderr, "target buffer size too small: cc= %d\n", cc );
			return Z_BUF_ERROR;
		}

		fault_tries = 50; /* reset for the next chunk */

		if (initial_pass == 1) {
			/* we got our lzcount sample from the 1st pass */
			NXPRT( fprintf(stderr, "first pass done\n") );
			initial_pass = 0;
			goto initial_pass_done;
		}
	    
		inlen     = inlen - srclen;
		srcbuf    = srcbuf + srclen;
		srctotlen = srctotlen + srclen;

		/* two possible locations for spbc depending on the function code */
		spbc = (!lzcounts) ? get32(cmdp->cpb, out_spbc_comp) :
			get32(cmdp->cpb, out_spbc_comp_with_count);
		assert(spbc == srclen);

		tpbc = get32(cmdp->crb.csb, tpbc);  /* target byte count */
		tebc = getnn(cmdp->cpb, out_tebc);  /* target ending bit count */
		NXPRT( fprintf(stderr, "compressed chunk %d to %d bytes, tebc= %d\n", spbc, tpbc, tebc) );
	    
		if (inlen > 0) { /* more chunks to go */
			/* This sample code does not use compression
			   history.  It will hurt the compression
			   ratio for small size chunks */
			set_bfinal(dstbuf, 0);
			dstbuf    = dstbuf + tpbc;
			dsttotlen = dsttotlen + tpbc;
			outlen    = outlen - tpbc;
			/* round up to the next byte with a flush
			 * block; do not set the BFINAL bit */		    
			flushlen  = append_sync_flush(dstbuf, tebc, 0);
			dsttotlen = dsttotlen + flushlen;
			outlen    = outlen - flushlen;			
			dstbuf    = dstbuf + flushlen;
			NXPRT( fprintf(stderr, "added deflate sync_flush %d bytes\n", flushlen) );
		}
		else {  /* done */ 
			/* set the BFINAL bit of the last block per deflate std */
			set_bfinal(dstbuf, 1);		    
			/* *dstbuf   = *dstbuf | (unsigned char) 0x01;  */
			dstbuf    = dstbuf + tpbc;
			dsttotlen = dsttotlen + tpbc;
			outlen    = outlen - tpbc;
		}

		/* resuming crc for the next chunk */
		crc = get32(cmdp->cpb, out_crc);
		put32(cmdp->cpb, in_crc, crc); 
		crc = be32toh(crc);
	}

	/* append CRC32 and ISIZE to the end */
	memcpy(dstbuf, &crc, 4);
	memcpy(dstbuf+4, &srctotlen, 4);
	dsttotlen = dsttotlen + 8;
	outlen    = outlen - 8;

	/* write out how many bytes are in the dest buffer */
	*destLen = dsttotlen;

	NXPRT( fprintf(stderr, "compressed %ld to %ld bytes total, crc32=%08x\n", srctotlen, dsttotlen, crc) );

	dht_end(myhandle->dht_handle);

	return Z_OK;
}

static void sigsegv_handler(int sig, siginfo_t *info, void *ctx)
{
	fprintf(stderr, "%d: Got signal %d si_code %d, si_addr %p\n", getpid(),
		sig, info->si_code, info->si_addr);

	/* exit(-1); */
	nx_fault_storage_address = info->si_addr; 
}


/*
  Compresses the source buffer into the destination buffer.  sourceLen
  is the byte length of the source buffer.  Upon entry, destLen is the
  total size of the destination buffer.

  compress returns Z_OK if success, Z_MEM_ERROR if there was not
  enough memory, Z_BUF_ERROR if there was not enough room in the
  output buffer, Z_STREAM_ERROR for other device or memory specific
  errors.
*/
int compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
{
	void *handle = NULL;
	int rc;

	rc = nxlite_compress((char *)dest, (uint64_t *)destLen,
			     (char *)source, (uint64_t)sourceLen,
			     &handle);

	nxlite_end(handle);

	return rc;
}	


/*
     Decompresses the source buffer into the destination buffer.  sourceLen is
   the byte length of the source buffer.  Upon entry, destLen is the total size
   of the destination buffer, which must be large enough to hold the entire
   uncompressed data.  (The size of the uncompressed data must have been saved
   previously by the compressor and transmitted to the decompressor by some
   mechanism outside the scope of this compression library.) Upon exit, destLen
   is the actual size of the uncompressed data.

     uncompress returns Z_OK if success, Z_MEM_ERROR if there was not
   enough memory, Z_BUF_ERROR if there was not enough room in the output
   buffer, or Z_DATA_ERROR if the input data was corrupted or incomplete.  In
   the case where there is not enough room, uncompress() will fill the output
   buffer with the uncompressed data up to that point.
*/
int uncompress (Bytef *dest, uLongf *destLen,
		const Bytef *source, uLong sourceLen)
{
	return 0;
}

int main()
{
	return 0;
}


  




/* ************************************ */

const int fifo_in_len = 1<<24;
const int fifo_out_len = 1<<24;	
const int page_sz = 1<<16;
const int line_sz = 1<<7;
const int window_max = 1<<15;
const int retry_max = 50;


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


/* fifo queue management */
#define buffer_used_bytes(cur, used, len) (used) /* amount */
#define buffer_free_bytes(cur, used, len) ((len)-((cur)+(used)))
/* amount of free bytes in the first and last parts */
#define buffer_free_first_bytes(cur, used, len)  ((len)-((cur)+(used)))
#define buffer_free_last_bytes(cur, used, len)   0 
/* amount of used bytes in the first and last parts */
#define buffer_used_first_bytes(cur, used, len)  (used)
#define buffer_used_last_bytes(cur, used, len)   0
/* first and last free parts start here */
#define buffer_free_first_offset(cur, used)      ((cur)+(used))
#define buffer_free_last_offset(cur, used, len)  buffer_used_last_bytes(cur, used, len)
/* first and last used parts start here */
#define buffer_used_first_offset(cur)            (cur)
#define buffer_used_last_offset(cur)             (0)	



static int nxlite_uncompress(char *dest, uint64_t *destLen,
			     char *source, uint64_t sourceLen,
			     void **handlep)
{
	FILE *inpf;
	FILE *outf;

	int c, expect, i, cc, rc = 0;
	char gzfname[1024];

	/* queuing, file ops, byte counting */
	char *fifo_in, *fifo_out;
	int used_in, cur_in, used_out, cur_out, free_in, read_sz, n;
	int first_free, last_free, first_used, last_used;
	int first_offset, last_offset;
	int write_sz, free_space, copy_sz, source_sz;
	int source_sz_estimate, target_sz_estimate;
	uint64_t last_comp_ratio; /* 1000 max */
	uint64_t total_out;
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

	used_in = cur_in = used_out = cur_out = 0;
	is_final = is_eof = 0;

#define GETINPC(X) *(source + cur_in++)
	
	/* Decode the gzip header */
	c = GETINPC(); expect = 0x1f; /* ID1 */
	if (c != expect) goto err1;

	c = GETINPC(); expect = 0x8b; /* ID2 */
	if (c != expect) goto err1;

	c = GETINPC(); expect = 0x08; /* CM */
	if (c != expect) goto err1;

	int flg = GETINPC(); /* FLG */
	if (flg & 0b11100000 || flg & 0b100) goto err2;

	fprintf(stderr, "gzHeader FLG %x\n", flg);

	/* Read 6 bytes; ignoring the MTIME, XFL, OS fields in this
	   sample code */
	for (i=0; i<6; i++) {
		char tmp[10];
		if (EOF == (tmp[i] = GETINPC()))
			goto err3;
		fprintf(stderr, "%02x ", tmp[i]);
		if (i == 5) fprintf(stderr, "\n");
	}
	fprintf(stderr, "gzHeader MTIME, XFL, OS ignored\n");	

	/* FNAME */
	if (flg & 0b1000) {
		int k=0;
		do {
			if (EOF == (c = GETINPC()))
				goto err3;
			gzfname[k++] = c;
		} while (c);
		fprintf(stderr, "gzHeader FNAME: %s \n", gzfname);
	}

	/* FHCRC */
	if (flg & 0b10) {
		c = GETINPC(); c = GETINPC();
		fprintf(stderr, "gzHeader FHCRC: ignored\n");
	}	

	cmdp = &cmd;
	memset( &cmdp->crb, 0, sizeof(cmdp->crb) );
	
	ddl_in  = &dde_in[0];
	ddl_out = &dde_out[0];	
	
read_state:

	/* Read from .gz file */
	
	NXPRT( fprintf(stderr, "read_state:\n") );
	
	if (is_eof != 0) goto write_state;

	cur_in = (used_in == 0) ? 0 : cur_in; 
	
	free_space = buffer_free_bytes(cur_in, used_in, fifo_in_len);

	/* free space may wrap around as first and last */
	first_free = free_space;
	last_free  = 0; //buffer_free_last_bytes(cur_in, used_in, fifo_in_len);

	/* start offsets of the free memory */
	first_offset = buffer_free_first_offset(cur_in, used_in);
	last_offset  = 0; //buffer_free_last_offset(cur_in, used_in, fifo_in_len);

	read_sz = free_space;
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

	used_in = sourceLen;
	

	/* At this point we have used_in bytes in fifo_in with the
	   data head starting at cur_in and possibly wrapping
	   around */


write_state:

	/* write decompressed data to output file */
	
	NXPRT( fprintf(stderr, "write_state:\n") );
	
	if (used_out == 0) goto decomp_state;


	/* If fifo_out has data waiting, write it out to the file to
	   make free target space for the accelerator used bytes in
	   the first and last parts of fifo_out */

	first_used = buffer_used_first_bytes(cur_out, used_out, fifo_out_len);
	last_used  = 0; //buffer_used_last_bytes(cur_out, used_out, fifo_out_len);

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

				/* Up to 32KB history wraps around fifo_out */
				nx_append_dde(ddl_in, fifo_out, cur_out);

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
	first_used = buffer_used_first_bytes(cur_in, used_in, fifo_in_len);
	last_used = 0; //buffer_used_last_bytes(cur_in, used_in, fifo_in_len);

	
	if (first_used > 0)
		nx_append_dde(ddl_in, fifo_in + cur_in, first_used);
		
	if (last_used > 0)
		nx_append_dde(ddl_in, fifo_in, last_used);

	/*
	 * NX target buffers
	 */
	first_free = buffer_free_bytes(cur_out, used_out, fifo_out_len);

	last_free = 0; //buffer_free_last_bytes(cur_out, used_out, fifo_out_len);

	/* reduce output free space amount not to overwrite the history */
	int target_max = NX_MAX(0, buffer_free_bytes(used_out, fifo_out_len) - (1<<16));

	NXPRT( fprintf(stderr, "target_max %d (0x%x)\n", target_max, target_max) );
	
	first_free = NX_MIN(target_max, first_free);
	if (first_free > 0) {
		first_offset = buffer_free_first_offset(cur_out, used_out);		
		nx_append_dde(ddl_out, fifo_out + first_offset, first_free);
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



	
restart_nx:

 	putp32(ddl_in, ddebc, source_sz);  

	/* fault in pages */
	nx_touch_pages_dde(ddl_in, 0, page_sz, 0);
	nx_touch_pages_dde(ddl_out, target_sz_estimate, page_sz, 1);

	/* send job to NX */
	cc = nx_submit_job(ddl_in, ddl_out, cmdp, devhandle);

	switch (cc) {

	case ERR_NX_AT_FAULT:

		/* We touched the pages ahead of time. In the most common case we shouldn't
		   be here. But may be some pages were paged out. Kernel should have 
		   placed the faulting address to fsaddr */
		NXPRT( fprintf(stderr, "ERR_NX_AT_FAULT %p\n", (void *)cmdp->crb.csb.fsaddr) );

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

			goto ok_cc3; /* not an error */

		}
		else {
			/* History length error when CE(1)=1 CE(0)=0. 
			   We have a bug */
			rc = -1;
			fprintf(stderr, "history length error cc= %d\n", cc);
			goto err5;
		}
		
	case ERR_NX_TARGET_SPACE:

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

	fclose(outf);

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



