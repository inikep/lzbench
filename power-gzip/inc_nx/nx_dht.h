/*
 * Deflate Huffman Tables
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

#ifndef _NX_DHT_H
#define _NX_DHT_H

#include <nxu.h>

#define DHT_TOPSYM_MAX   4     /* number of most frequent symbols tracked */
#define DHT_NUM_MAX      128   /* max number of dht table entries */
#define DHT_SZ_MAX       (DHT_MAXSZ+1)   /* number of dht bytes per entry */
#define DHT_NUM_BUILTIN  35    /* number of built-in entries */
/* use the last dht if accumulated source data sizes is less than this
   value to amortize dht_lookup overheads over many */
#define DHT_NUM_SRC_BYTES    (512*1024) 

typedef struct dht_entry_t {
	/* 32bit XOR of the entire struct, inclusive of cksum, must
	   equal 0. May use the cksum if this struct is read/write to
	   a file; note that XOR is endian agnostic */
	uint32_t cksum;
	volatile int valid;
	/* reference count for atomic reads; 1 or more is the number
	   of threads reading */
	int ref_count;
	/* for the clock algorithm; since the last clock sweep
	   0 is not accessed 
	   1 is accessed once 
	   2 is accessed two or more time */
	volatile int64_t accessed;
	/* for alignment */
	/* uint32_t cpb_reserved[3]; */
	/* last 32b contains the 12 bit length; use
	   the getnn/putnn macros to access
	   endian-safe */
	uint32_t in_dhtlen; 
	/* actual dht here */
	char in_dht_char[DHT_MAXSZ];
	/* most freq symbols and their code lengths; use them to
	   lookup the dht cache; */
	int litlen[DHT_TOPSYM_MAX];  
	int dist[DHT_TOPSYM_MAX];
} dht_entry_t;

typedef struct dht_tab_t {
	/* put any locks here */
	int last_used_builtin_idx;
	int last_cache_idx;
	int clock;
	int reused_count;
	long nbytes_accumulated;
	dht_entry_t *last_used_entry;
	dht_entry_t *builtin;
	dht_entry_t cache[DHT_NUM_MAX+1];
} dht_tab_t;


#define dht_default_req    0  /* use this if no lzcounts available */
#define dht_search_req     1  /* search the cache and generate if not found */
#define dht_gen_req        2  /* unconditionally generate; do not cache */
#define dht_invalidate_req 3  /* erase cache contents except builtin ones */

/* call in deflateInit; returns a handle for dht_lookup.
   ifile and ofile are unused in this implementation */
void *dht_begin(char *ifile, char *ofile);             

/* call in deflateEnd  */
void dht_end(void *handle);                            

/* call in deflate */
int dht_lookup(nx_gzip_crb_cpb_t *cmdp, int request, void *handle);

void *dht_copy(void *handle);

/* use this utility to make built-in dht data structures */
int dht_print(void *handle);

/* given lzcounts produce a dynamic huffman table */
int dhtgen(uint32_t  *lhist,        /* supply the P9 LZ counts here */
	   int num_lhist,
	   uint32_t *dhist,
	   int num_dhist,
	   char *dht,               /* dht returned here; caller is responsible for alloc/free of min 320 bytes */    
	   int  *dht_num_bytes,     /* number of valid bytes in *dht */
	   int  *dht_num_valid_bits,/* valid bits in the LAST byte; note the value of 0 is encoded as 8 bits */    
	   int  cpb_header          /* set nonzero if prepending the 16 byte P9 compliant cpbin header with the bit length of dht */
	); 

void fill_zero_lzcounts(uint32_t *llhist, uint32_t *dhist, uint32_t val);
void fill_zero_len_dist(uint32_t *llhist, uint32_t *dhist, uint32_t val);

#endif /* _NX_DHT_H */
