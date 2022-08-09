/*
 * NX-GZIP compression accelerator user library
 * implementing zlib compression library interfaces
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
 */

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
#include "nxu.h"
#include "nx_dht.h"
#include "nx_zlib.h"

/* If cache keys are too many then dhtgen overhead increases; if cache
   keys are too few then compression ratio suffers.
   #undef this to compare with two keys instead of one */
/* #define DHT_ONE_KEY */

/* Approximately greater. If the counts (probabilities) are similar
   then the code lengths will probably end up being equal do not make
   unnecessary dhtgen calls */
//#define DHT_GT(X,Y) ((X) > (((Y)*7)/4))
#define DHT_GT(X,Y) ((X) > (Y))

/* util */
#ifdef NXDBG
#define DHTPRT(X) do{ X;}while(0)
#else
#define DHTPRT(X) do{  ;}while(0)
#endif

/* use top symbol or top two symbols as cache lookup keys */
#if !defined(DHT_ONE_KEY)
#define SECOND_KEY(X) (X)
#else
#define SECOND_KEY(X) (!!1) /* always TRUE */
#endif

#define NUMLIT 256  /* literals count in deflate */
#define EOB 256     /* end of block symbol */

typedef struct top_sym_t {
	struct {
		uint32_t lzcnt;
		int sym;
	} sorted[3];
} top_sym_t;

#define llns 0

extern dht_entry_t *get_builtin_table();

/* One time setup of the tables. Returns a handle.  ifile ofile
   unused */
void *dht_begin5(char *ifile, char *ofile)
{
	int i;
	dht_tab_t *dht_tab;

	if (NULL == (dht_tab = malloc(sizeof(dht_tab_t))))
		return NULL;

	for (i=0; i<DHT_NUM_MAX; i++) {
		/* set all invalid */
		dht_tab->cache[i].valid = 0;
		dht_tab->cache[i].ref_count = 0;
		dht_tab->cache[i].accessed = 0;
	}
	dht_tab->builtin = get_builtin_table();
	dht_tab->last_used_builtin_idx = -1;
	dht_tab->last_cache_idx = -1;
	dht_tab->last_used_entry = NULL;
	dht_tab->nbytes_accumulated = 0;
	dht_tab->clock = 0;

	return (void *)dht_tab;
}

void dht_end(void *handle)
{
	if (!!handle) free(handle);
}

/* One time setup of the tables. Returns a handle */
void *dht_begin(char *ifile, char *ofile)
{
	return dht_begin5(ifile, ofile);
}

void *dht_copy(void *handle)
{
	dht_tab_t *new_tab;
	dht_tab_t *old_tab = handle;

	if (!old_tab)
		return NULL;

	if (NULL == (new_tab = malloc(sizeof(dht_tab_t))))
		return NULL;

	memcpy((char *)new_tab, (const char *)old_tab, sizeof(dht_tab_t));

	if (old_tab->last_used_entry != NULL) {
		uint64_t offset;
		/* last_used_entry points to a dht_tab->cache or
		   dht_tab->builtin entry; issue 123 first identify
		   which table it's pointing to; then compute offset
		   relative to that table */
		if (((char *)old_tab->last_used_entry >= (char *)&old_tab->builtin[0]) &&
		    ((char *)old_tab->last_used_entry <= (char *)&old_tab->builtin[DHT_NUM_BUILTIN-1])) {
			/* points to a builtin entry; find byte offset relative to the table base */
			offset = (char *)(old_tab->last_used_entry) - (char *)&old_tab->builtin[0];
			new_tab->last_used_entry = (dht_entry_t *)((char *)&new_tab->builtin[0] + offset);
		}
		else if (((char *)old_tab->last_used_entry >= (char *)&old_tab->cache[0]) &&
			 ((char *)old_tab->last_used_entry <= (char *)&old_tab->cache[DHT_NUM_MAX-1])) {
			offset = (char *)(old_tab->last_used_entry) - (char *)&old_tab->cache[0];
			new_tab->last_used_entry = (dht_entry_t *)((char *)&new_tab->cache[0] + offset);
		}
		else {
			/* dht table out of bounds */
			assert(0);
		}
	}

	return (void *)new_tab;
}

static int dht_sort4(nx_gzip_crb_cpb_t *cmdp, top_sym_t *t)
{
	int i;
	int llscan;
	uint32_t *lzcount;
	top_sym_t top[1];

	/* where to look for the top search keys */
	if ( (nx_config.dht & 0x1) == 0x1 )
		llscan = LLSZ;   /* scan literals and lengths */
	else
		llscan = NUMLIT; /* scan literals only */

	/* init */
	top[llns].sorted[0].lzcnt = 0;
	top[llns].sorted[0].sym = -1;
	top[llns].sorted[2] = top[llns].sorted[1] = top[llns].sorted[0];

	lzcount = (uint32_t *)cmdp->cpb.out_lzcount;

	/* EOB symbol decimal 256 comes out with a count of 1 which we
	   use as an endian detector */
	if (1 != lzcount[EOB]) {
		for (i = 0; i < LLSZ+DSZ; i++)
			lzcount[i] = be32toh(lzcount[i]);
		lzcount[EOB] = 1;
		DHTPRT( fprintf(stderr, "dht_sort: lzcounts endian corrected\n") );
	}
	else {
		DHTPRT( fprintf(stderr, "dht_sort: lzcounts endian ok\n") );
	}

	for (i = 0; i < llscan; i++) { /* Look for the top keys */
		uint32_t c = lzcount[i];

		DHTPRT( fprintf(stderr, "%d %d, ", i, lzcount[i] ) );

		if ( DHT_GT(c, top[llns].sorted[0].lzcnt) ) {
			/* count greater than the top count */
#if !defined(DHT_ONE_KEY)
			top[llns].sorted[1] = top[llns].sorted[0];
#endif
			top[llns].sorted[0].lzcnt = c;
			top[llns].sorted[0].sym = i;
		}
#if !defined(DHT_ONE_KEY)
		else if ( DHT_GT(c, top[llns].sorted[1].lzcnt) ) {
			/* count greater than the second most count */
			top[llns].sorted[2] = top[llns].sorted[1];
			top[llns].sorted[1].lzcnt = c;
			top[llns].sorted[1].sym = i;
		}
		else if ( DHT_GT(c, top[llns].sorted[2].lzcnt) ) {
			/* count greater than the second most count */
			top[llns].sorted[2].lzcnt = c;
			top[llns].sorted[2].sym = i;
		}
#endif
	}

	/* copy back */
	t[llns] = top[llns];

	/* Will not use distances as cache keys */

	DHTPRT( fprintf(stderr, "top litlens %d %d %d\n", top[llns].sorted[0].sym, top[llns].sorted[1].sym, top[llns].sorted[2].sym) );

	return 0;
}

/*
   Finds the top symbols in lit, len and dist ranges.
   cmdp->cpb.out_lzcount array will be endian reversed first.  (To
   protect cmdp from double reversals I will test and set the Literal 256
   count to 1 after the first endian reversal)

   Results returned in the top[3] struct.  We will use the top symbols
   as cache keys to locate a matching dht.
*/
static int dht_sort(nx_gzip_crb_cpb_t *cmdp, top_sym_t *top)
{
	return dht_sort4(cmdp, top);
}

static inline int copy_dht_to_cpb(nx_gzip_crb_cpb_t *cmdp, dht_entry_t *d)
{
	int dhtlen, dht_num_bytes;
	dhtlen = d->in_dhtlen;
	dht_num_bytes = (dhtlen + 7)/8; /* bits to bytes */
	putnn(cmdp->cpb, in_dhtlen, (uint32_t)dhtlen);
	memcpy(cmdp->cpb.in_dht_char, d->in_dht_char, dht_num_bytes);
	return 0;
}

#define DHT_WRITER 0x8FFF  /* flag to indicate the exclusive writer */
#define DHT_LOCK_RETRY 384

#if defined(DHT_ATOMICS)
/* libnxz is single threaded; doesn't need this; only special apps
   that share the dht structures here may need atomic ops */

#define dht_atomic_load(P)         __atomic_load_n((P), __ATOMIC_RELAXED)
#define dht_atomic_store(P,V)      __atomic_store_n((P), (V), __ATOMIC_RELAXED)
#define dht_atomic_fetch_add(P,V)  __atomic_fetch_add((P), (V), __ATOMIC_RELAXED)
#define dht_atomic_fetch_sub(P,V)  __atomic_fetch_sub((P), (V), __ATOMIC_RELAXED)

#else /* defined(DHT_ATOMICS) */

#define dht_atomic_load(P)  (*(P))
#define dht_atomic_store(P,V)  do { *(P) = (V); } while(0)
#define dht_atomic_fetch_add(P,V)  ({typeof(*(P)) tmp = *(P); *(P) = tmp + (V); tmp;})
#define dht_atomic_fetch_sub(P,V)  ({typeof(*(P)) tmp = *(P); *(P) = tmp - (V); tmp;})

#endif /* defined(DHT_ATOMICS) */

#if !defined(DHT_ATOMICS)  /* non-atomic case */

#define read_lock(P)     1
#define read_unlock(P)   1
#define write_lock(P)    1
#define write_unlock(P)  1

#else /* !defined(DHT_ATOMICS) */

static int inline read_lock(int *ref_count)
{
	/*
	   bool __atomic_compare_exchange_n (type *ptr, type *expected, type desired,
	   bool weak, int success_memorder, int failure_memorder)

	   Compares the contents of *ptr with
	   the contents of *expected. If equal, the operation is a
	   read-modify-write operation that writes desired into
	   *ptr.

	   If they are not equal, the operation is a read and the
	   current contents of *ptr are written into *expected. weak
	   is true for weak compare_exchange, which may fail
	   spuriously, and false for the strong variation, which never
	   fails spuriously. Many targets only offer the strong
	   variation and ignore the parameter. When in doubt, use the
	   strong variation.

	   If desired is written into *ptr then true is returned and
	   memory is affected according to the memory order specified
	   by success_memorder. There are no restrictions on what
	   memory order can be used here.  Otherwise, false is
	   returned and memory is affected according to
	   failure_memorder. This memory order cannot be
	   __ATOMIC_RELEASE nor __ATOMIC_ACQ_REL. It also cannot be a
	   stronger order than that specified by success_memorder.
	*/

	int retry = DHT_LOCK_RETRY;
	while( retry-- > 0)  {
		int readers = dht_atomic_load(ref_count);
		int new_readers = readers + 1;
		if (readers != DHT_WRITER) {
			if (__atomic_compare_exchange_n(ref_count,  /* actual */
							&readers,   /* expected */
							new_readers,/* new; +1 reader */
							0,
							__ATOMIC_RELAXED,
							__ATOMIC_RELAXED))
				return 1; /* success */
			/* retry few times */
		}
		else return 0; /* writer will take long time  */
	}
	return 0;
}

static int inline read_unlock(int *ref_count)
{
	int retry = DHT_LOCK_RETRY;
	while( retry-- > 0)  {
		int readers = dht_atomic_load(ref_count);
		int new_readers = readers - 1;
		/* read_unlock error; needs a matching lock */
		assert( readers != DHT_WRITER && readers > 0 );
		if (__atomic_compare_exchange_n(ref_count,  /* actual */
						&readers,   /* expected */
						new_readers,/* new; readers-1 */
						0,
						__ATOMIC_RELAXED,
						__ATOMIC_RELAXED))
			return 1; /* success */
		/* retry few times */
		/* ?? retry = dht_atomic_fetch_add(ref_count, 5); */
		retry += dht_atomic_load(ref_count);
	}
	return 0;
}

static int inline write_lock(int *ref_count)
{
	int retry = 1;
	while( retry-- > 0)  {
		int readers = dht_atomic_load(ref_count);
		if (readers == 0) { /* unlocked */
			if (__atomic_compare_exchange_n(ref_count,  /* actual */
							&readers,   /* expected */
							DHT_WRITER, /* new; write locked */
							0,
							__ATOMIC_RELAXED,
							__ATOMIC_RELAXED))
				return 1; /* success */
		}
	}
	return 0;
}

static int inline write_unlock(int *ref_count)
{
	int retry = 1;
	while( retry-- > 0)  {
		int readers = dht_atomic_load(ref_count);
		assert(readers == DHT_WRITER);
		if (__atomic_compare_exchange_n(ref_count,  /* actual */
						&readers,   /* expected */
						0,          /* new; unlocked*/
						0,
						__ATOMIC_RELAXED,
						__ATOMIC_RELAXED))
			return 1; /* success */
	}
	return 0;
}

#endif	/* !defined(DHT_ATOMICS) */

/* search nx_dht_builtin.c */
static int dht_search_builtin(nx_gzip_crb_cpb_t *cmdp, dht_tab_t *dht_tab, top_sym_t *top)
{
	int i, sidx;
	dht_entry_t *builtin = dht_tab->builtin;

	/* speed up the search */
	sidx = dht_atomic_load( &dht_tab->last_used_builtin_idx );
	sidx = (sidx < 0) ? 0 : sidx;
	sidx = sidx % DHT_NUM_BUILTIN;

	/* search the builtin dht cache */
	for (i = 0; i < DHT_NUM_BUILTIN; i++, sidx = (sidx+1) % DHT_NUM_BUILTIN) {

		if ( dht_atomic_load( &builtin[sidx].valid ) == 0)
			continue; /* skip unused entries */

		if (builtin[sidx].litlen[0] == top[llns].sorted[0].sym && /* top litlen */
		    SECOND_KEY((builtin[sidx].litlen[1] == top[llns].sorted[1].sym)) ) { /* second top litlen */

			DHTPRT( fprintf(stderr, "dht_search_builtin: hit idx %d (litlen %d %d)\n", sidx, builtin[sidx].litlen[0], builtin[sidx].litlen[1] ) );
			copy_dht_to_cpb(cmdp, &(builtin[sidx]));

			dht_atomic_store( &dht_tab->last_used_builtin_idx, sidx );

			dht_atomic_store( &dht_tab->last_used_entry, &(builtin[sidx]) );

			return 0;
		}
	}
	return -1; /* not found in the builtin table */
}

/* search the user generated cached dhts */
static int dht_search_cache(nx_gzip_crb_cpb_t *cmdp, dht_tab_t *dht_tab, top_sym_t *top)
{
	int i, sidx;
	dht_entry_t *dht_cache = dht_tab->cache;

	/* speed up the search starting from the last */
	sidx = dht_atomic_load( &dht_tab->last_cache_idx );
	sidx = (sidx < 0) ? 0 : sidx;
	sidx = sidx % DHT_NUM_MAX;

	/* search the dht cache */
	for (i = 0; i < DHT_NUM_MAX; i++, sidx = (sidx+1) % DHT_NUM_MAX) {

		if ( dht_atomic_load( &dht_cache[sidx].valid ) == 0)
			continue; /* skip unused entries */

		if (dht_cache[sidx].litlen[0] == top[llns].sorted[0].sym && /* top litlen */
		    SECOND_KEY((dht_cache[sidx].litlen[1] == top[llns].sorted[1].sym)) ) {

			if (read_lock( &dht_cache[sidx].ref_count)) {

				DHTPRT( fprintf(stderr, "dht_search_cache: hit idx %d, accessed %ld, (litlen %d %d)\n", sidx, dht_cache[sidx].accessed, dht_cache[sidx].litlen[0], dht_cache[sidx].litlen[1]) );

				/* copy the cached dht back to cpb */
				copy_dht_to_cpb(cmdp, &(dht_cache[sidx]));

				/* for lru */
				dht_atomic_store( &dht_cache[sidx].accessed, 1);

				dht_atomic_store( &dht_tab->last_cache_idx, sidx );

				dht_atomic_store( &dht_tab->last_used_entry, &(dht_cache[sidx]) );

				if (!read_unlock( &dht_cache[sidx].ref_count )){
					DHTPRT( fprintf(stderr, "dht_cache unlock failed\n") );
					return -1;
				}

				return 0;
			}
		}
	}
	/* search did not find anything */
	return -1;
}

static int dht_use_last(nx_gzip_crb_cpb_t *cmdp, dht_tab_t *dht_tab)
{
	long source_bytes;
	uint32_t fc, histlen;
	dht_entry_t *dht_entry = dht_atomic_load( &dht_tab->last_used_entry );

	if (dht_entry == NULL)
		return -1;

	DHTPRT( fprintf(stderr, "dht_use_last: entry %p\n", dht_entry) );

	if (read_lock( &dht_entry->ref_count)) {

		if (dht_atomic_load( &dht_entry->valid) == 0) {
			if (!read_unlock( &dht_entry->ref_count ))
				DHTPRT( fprintf(stderr, "dht_entry unlock failed\n") );
			return -1;
		}

		/* extract the source data amount this crb has processed */
		fc = getnn(cmdp->crb, gzip_fc);

		/* exclude history bytes read */
		if (fc == GZIP_FC_COMPRESS_RESUME_FHT ||
		    fc == GZIP_FC_COMPRESS_RESUME_DHT ||
		    fc == GZIP_FC_COMPRESS_RESUME_FHT_COUNT ||
		    fc == GZIP_FC_COMPRESS_RESUME_DHT_COUNT) {
			histlen = getnn(cmdp->cpb, in_histlen) * 16;
			DHTPRT( fprintf(stderr, "dht_use_last: resume fc 0x%x\n", fc) );
		}
		else {
			histlen = 0;
		}

		source_bytes = 0;

		if (fc == GZIP_FC_COMPRESS_FHT_COUNT ||
		    fc == GZIP_FC_COMPRESS_DHT_COUNT ||
		    fc == GZIP_FC_COMPRESS_RESUME_FHT_COUNT ||
		    fc == GZIP_FC_COMPRESS_RESUME_DHT_COUNT) {
			source_bytes = get32(cmdp->cpb, out_spbc_comp_with_count) - histlen;
			DHTPRT( fprintf(stderr, "dht_use_last: fc 0x%x source_bytes %ld\n", fc, source_bytes) );
		}
		else if (fc == GZIP_FC_COMPRESS_FHT ||
			 fc == GZIP_FC_COMPRESS_DHT ||
			 fc == GZIP_FC_COMPRESS_RESUME_FHT ||
			 fc == GZIP_FC_COMPRESS_RESUME_DHT) {
			/* this might be an error producing a dht with no lzcounts */
			source_bytes = get32(cmdp->cpb, out_spbc_comp) - histlen;
			DHTPRT( fprintf(stderr, "dht_use_last: producing a dht with no lzcounts???\n") );
			assert(0);
		}

		if (source_bytes < 0 ) source_bytes = 0;

		dht_atomic_fetch_add( &dht_tab->nbytes_accumulated, source_bytes);

		DHTPRT( fprintf(stderr, "dht_use_last: bytes accumulated so far %ld\n", dht_tab->nbytes_accumulated) );

		/* if last dht has been reused many times, for greater or equal to
		 * DHT_NUM_SRC_BYTES, then return early to refresh the dht */
		if (source_bytes == 0 || (dht_atomic_load( &dht_tab->nbytes_accumulated ) >= DHT_NUM_SRC_BYTES)) {
			dht_atomic_store( &dht_tab->last_used_entry, NULL );
			dht_atomic_store( &dht_tab->nbytes_accumulated, source_bytes);
			if (!read_unlock( &dht_entry->ref_count ))
				DHTPRT( fprintf(stderr, "dht_entry unlock failed\n") );
			DHTPRT( fprintf(stderr, "dht_use_last: quit reusing, search caches or dhtgen\n") );
			return -1;
		}

		DHTPRT( fprintf(stderr, "dht_use_last: reusing last (litlen %d %d)\n", dht_entry->litlen[0], dht_entry->litlen[1]));

		/* copy the cached dht back to cpb */
		copy_dht_to_cpb(cmdp, dht_entry);

		/* for lru */
		dht_atomic_store( &dht_entry->accessed, 1);

		if (!read_unlock( &dht_entry->ref_count )){
			DHTPRT( fprintf(stderr, "dht_entry unlock failed\n") );
			return -1;
		}

		return 0;
	}
	return -1;
}

static int dht_lookup5(nx_gzip_crb_cpb_t *cmdp, int request, void *handle)
{
	int clock=0;
	int dht_num_bytes, dht_num_valid_bits, dhtlen;
	top_sym_t top[1];
	dht_tab_t *dht_tab = (dht_tab_t *) handle;
	dht_entry_t *dht_cache = dht_tab->cache;

	__builtin_bzero(top, sizeof(top_sym_t));

	if (request == dht_default_req) {
		/* first builtin entry is the default */
		copy_dht_to_cpb(cmdp, &dht_tab->builtin[0]);
		dht_atomic_store( &dht_tab->last_used_entry, &dht_tab->builtin[0] );
		return 0;
	}
	else if (request == dht_gen_req)
		goto force_dhtgen;
	else if (request == dht_search_req)
		goto search_cache;
	else if (request == dht_invalidate_req) {
		/* erases all non-builtin entries TODO */
		assert(0);
	}
	else assert(0);

search_cache:
	/* reuse the last dht to eliminate sort and dhtgen overheads */
	if (!dht_use_last(cmdp, dht_tab))
		return 0;

	/* find most frequent symbols */
	dht_sort(cmdp, top);

	if (!dht_search_cache(cmdp, dht_tab, top))
		return 0; /* found */

	if (!dht_search_builtin(cmdp, dht_tab, top))
		return 0; /* found */

	/* Did not find the DHT. Throw away LRU cache entry*/
	while (1) {
		/* advance the clock hand */
		clock = dht_atomic_load( &dht_tab->clock ); /* old value */
		dht_atomic_store( &dht_tab->clock, (clock + 1) % DHT_NUM_MAX );
		/* check for an unused entry since the last sweep */
		if (dht_atomic_load( &dht_cache[clock].accessed) == 0) {
			/* unused found; now try to lock it to write dht in to it */
			if (write_lock( &dht_cache[clock].ref_count ))
				break;
		}
		else {
			/* clear the access bit to indicate lru */
			dht_atomic_store( &dht_cache[clock].accessed, 0 );
		}
	}

force_dhtgen:
	/* makes a universal dht with no missing codes */
	fill_zero_lzcounts((uint32_t *)cmdp->cpb.out_lzcount,        /* LitLen */
			   (uint32_t *)cmdp->cpb.out_lzcount + LLSZ, /* Dist */
			   1);

	/* dhtgen writes directly to cpb; 286 LitLen counts followed by 30 Dist counts */
	dhtgen( (uint32_t *)cmdp->cpb.out_lzcount,
		LLSZ,
	        (uint32_t *)cmdp->cpb.out_lzcount + LLSZ,
		DSZ,
		(char *)(cmdp->cpb.in_dht_char),
		&dht_num_bytes,
		&dht_num_valid_bits, /* last byte bits, 0 is encoded as 8 bits */
		0
		);
	dhtlen = 8 * dht_num_bytes - ((dht_num_valid_bits) ? 8 - dht_num_valid_bits : 0 );
	putnn(cmdp->cpb, in_dhtlen, dhtlen); /* write to cpb */

	DHTPRT( fprintf(stderr, "dhtgen: bytes %d last byte bits %d\n", dht_num_bytes, dht_num_valid_bits) );

	if (request == dht_gen_req) /* without updating cache */
		return 0;

	/* make a copy in the cache at the least used position */
	memcpy(dht_cache[clock].in_dht_char, cmdp->cpb.in_dht_char, dht_num_bytes);
	dht_cache[clock].in_dhtlen = dhtlen;

	/* save the dht identifying key */
	dht_cache[clock].litlen[0] = top[llns].sorted[0].sym;
	dht_cache[clock].litlen[1] = top[llns].sorted[1].sym;
	dht_cache[clock].litlen[2] = top[llns].sorted[2].sym;

	dht_atomic_store( &dht_cache[clock].valid, 1 );

	DHTPRT( fprintf(stderr, "dht_lookup: insert idx %d (litlen %d %d)\n", clock, dht_cache[clock].litlen[0],dht_cache[clock].litlen[1]));

	/* for lru */
	dht_atomic_store( &dht_cache[clock].accessed, 1);
	dht_atomic_store( &dht_tab->last_cache_idx, clock );
	dht_atomic_store( &dht_tab->last_used_entry, &(dht_cache[clock]) );

	assert( write_unlock( &dht_cache[clock].ref_count ) );

	return 0;
}


int dht_lookup(nx_gzip_crb_cpb_t *cmdp, int request, void *handle)
{
	return dht_lookup5(cmdp, request, handle);
}

/* use this utility to make built-in dht data structures */
int dht_print(void *handle)
{
	int i, j, dht_num_bytes, dhtlen;
	dht_entry_t *dht_cache = ((dht_tab_t *) handle)->cache;

	/* search the dht cache */
	for (j = 0; j < DHT_NUM_MAX; j++) {
		int64_t ref_count = dht_cache[j].ref_count;

		/* skip unused and builtin ones */
		if (ref_count <= 0)
			continue;

		dhtlen = dht_cache[j].in_dhtlen;
		dht_num_bytes = (dhtlen + 7)/8;

		fprintf(stderr, "{\n");

		/* unused at the moment */
		dht_cache[j].cksum = 0;
		fprintf(stderr, "\t%d, /* cksum */\n", dht_cache[j].cksum);
		fprintf(stderr, "\t%d, /* valid */\n", dht_cache[j].valid);
		fprintf(stderr, "\t%d, /* ref_count */\n", dht_cache[j].ref_count);
		fprintf(stderr, "\t%ld, /* accessed */\n", dht_cache[j].accessed);
		fprintf(stderr, "\t%d, /* in_dhtlen */\n", dht_cache[j].in_dhtlen);

		fprintf(stderr, "\t{ /* dht bytes start */\n");
		for (i=0; i<dht_num_bytes; i++) {
			if (i % 16 == 0)
				fprintf(stderr, "\n\t\t");
			fprintf(stderr, "0x%02x, ", (unsigned char)dht_cache[j].in_dht_char[i]);
		}
		fprintf(stderr, "\n\t}, /* dht bytes end */\n");

		fprintf(stderr, "\t{%d, %d, %d}, /* top litlens */\n",
			dht_cache[j].litlen[0], dht_cache[j].litlen[1], dht_cache[j].litlen[2] );

		fprintf(stderr, "\t{%d, %d, %d}, /* top dists */\n",
			dht_cache[j].dist[0], dht_cache[j].dist[1], dht_cache[j].dist[2] );

		fprintf(stderr, "},\n\n");
	}

	return 0;
}
